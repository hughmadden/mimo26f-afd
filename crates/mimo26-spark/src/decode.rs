//! FFN dispatch — the expert FFN over the device, in-process.
//!
//! `decode_ffn` runs the R18c mixed-M kernel (`m26x_expert_ffn_mixed_v2`, M ≤ 8);
//! `prefill_ffn` runs the frozen per-M kernel (`m26x_expert_ffn_v2`, M up to the
//! capacity class). Both upload a resident grouped image plus the `[T,4096]`
//! hidden rows and a `GroupedPlan`, run gate → up → SiLU → down, check the fault
//! word, and return the `[T,4096]` rank partial as f32. Route weighting and rank
//! pre-sum live above this (the wire return path).

use std::time::Instant;

use mimo26_expert::grouped::{Group, GroupedPlan};
use mimo26_repack::geom::QUARTER_SLICE_BYTES;
#[cfg(feature = "cuda")]
use mimo26_repack::geom::EXPERTS_PER_LAYER;

use crate::cuda;
use crate::device::{DeviceBuffer, PinnedBuffer};
use crate::ffi::{self, M26xPlan};
#[cfg(feature = "cuda")]
use crate::route::RoutePlan;

/// A pinned (zero-copy) grouped image for the smoke tests: `dptr()` is the
/// device pointer, and the host memory is unregistered on drop so a later test
/// that reuses the freed heap address does not hit `resource already mapped`.
/// (The daemon uses [`upload_grouped`] instead.)
pub struct RegisteredGrouped {
    dptr: *const u8,
    host: *mut core::ffi::c_void,
}

impl RegisteredGrouped {
    /// The device pointer to the pinned host image.
    pub fn dptr(&self) -> *const u8 {
        self.dptr
    }
}

impl Drop for RegisteredGrouped {
    fn drop(&mut self) {
        // SAFETY: `host` was registered by `register_grouped` and is owned here.
        unsafe { cuda::cudaHostUnregister(self.host) };
    }
}

/// Pin a resident grouped image in host memory (zero-copy on GB10 unified
/// memory) and return its device pointer plus an unregister-on-drop guard. Test
/// helper only — the daemon uses [`upload_grouped`].
pub fn register_grouped(grouped: &[u8]) -> Result<RegisteredGrouped, String> {
    if grouped.len() % QUARTER_SLICE_BYTES != 0 {
        return Err("grouped image is not whole quarter slices".to_string());
    }
    let host = grouped.as_ptr() as *mut core::ffi::c_void;
    let rc = unsafe { cuda::cudaHostRegister(host, grouped.len(), cuda::HOST_REGISTER_MAPPED) };
    if rc != cuda::SUCCESS {
        return Err(cuda::error_string(rc));
    }
    let mut dptr: *mut core::ffi::c_void = core::ptr::null_mut();
    let rc = unsafe { cuda::cudaHostGetDevicePointer(&mut dptr, host, 0) };
    if rc != cuda::SUCCESS {
        // SAFETY: roll back the successful register before returning the error.
        unsafe { cuda::cudaHostUnregister(host) };
        return Err(cuda::error_string(rc));
    }
    Ok(RegisteredGrouped { dptr: dptr as *const u8, host })
}

/// Upload a resident grouped image into a device allocation (device-resident;
/// the kernel reads cudaMalloc memory at full bandwidth, unlike the pinned
/// zero-copy mapping which is ~3x slower). The host bytes may be dropped after
/// this returns — the daemon frees its per-layer host copy to keep the footprint
/// near 41 GB.
pub fn upload_grouped(grouped: &[u8]) -> Result<DeviceBuffer, String> {
    if grouped.len() % QUARTER_SLICE_BYTES != 0 {
        return Err("grouped image is not whole quarter slices".to_string());
    }
    let d = DeviceBuffer::alloc(grouped.len()).map_err(cuda::error_string)?;
    d.upload(grouped).map_err(cuda::error_string)?;
    Ok(d)
}

/// Reusable device scratch buffers (x / intermediate scratch / out) at the max
/// capacity-class size, plus the per-request gather staging (pinned host hidden +
/// device hidden/src/dst) allocated once at the launch cap so the decode path
/// pays no per-request `cudaHostRegister`/`cudaMalloc`.
pub struct Scratch {
    x: Option<DeviceBuffer>,
    scratch: Option<DeviceBuffer>,
    out: Option<DeviceBuffer>,
    d_hidden: Option<DeviceBuffer>,
    d_src: Option<DeviceBuffer>,
    d_dst: Option<DeviceBuffer>,
    d_ids: Option<DeviceBuffer>,
    d_offsets: Option<DeviceBuffer>,
    d_fault: Option<DeviceBuffer>,
    pinned: Option<PinnedBuffer>,
}

impl Scratch {
    pub fn new() -> Self {
        Self {
            x: None,
            scratch: None,
            out: None,
            d_hidden: None,
            d_src: None,
            d_dst: None,
            d_ids: None,
            d_offsets: None,
            d_fault: None,
            pinned: None,
        }
    }

    /// The pooled output buffer (valid after an `out_alloc` that sized it).
    pub fn out_ref(&self) -> &DeviceBuffer {
        self.out.as_ref().expect("scratch out not allocated")
    }

    /// Grow (or reuse) the x buffer to at least `bytes`: the full-plan gather
    /// target in the device path, or the per-chunk upload buffer in the fallback.
    fn grow_x(&mut self, bytes: usize) -> Result<&DeviceBuffer, String> {
        if self.x.as_ref().map_or(true, |b| b.bytes() < bytes) {
            self.x = Some(DeviceBuffer::alloc(bytes).map_err(cuda::error_string)?);
        }
        Ok(self.x.as_ref().unwrap())
    }

    /// Grow (or reuse) the intermediate scratch buffer to at least `bytes`.
    fn grow_scratch(&mut self, bytes: usize) -> Result<&DeviceBuffer, String> {
        if self.scratch.as_ref().map_or(true, |b| b.bytes() < bytes) {
            self.scratch = Some(DeviceBuffer::alloc(bytes).map_err(cuda::error_string)?);
        }
        Ok(self.scratch.as_ref().unwrap())
    }

    /// Grow (or reuse) the out buffer to at least `bytes`, returning it.
    fn out_alloc(&mut self, bytes: usize) -> Result<&DeviceBuffer, String> {
        if self.out.as_ref().map_or(true, |b| b.bytes() < bytes) {
            self.out = Some(DeviceBuffer::alloc(bytes).map_err(cuda::error_string)?);
        }
        Ok(self.out.as_ref().unwrap())
    }

    /// Grow (or reuse) the persistent pinned host staging buffer to at least
    /// `bytes`. Allocated + registered once at the launch cap, then reused by the
    /// gather and the plan uploads (no per-copy page-locking).
    fn pinned_alloc(&mut self, bytes: usize) -> Result<&mut [u8], String> {
        if self.pinned.as_ref().map_or(true, |b| b.len() < bytes) {
            self.pinned = Some(PinnedBuffer::new(bytes).map_err(cuda::error_string)?);
        }
        Ok(self.pinned.as_mut().unwrap().as_mut_slice())
    }

    /// Grow (or reuse) the device hidden buffer to at least `bytes`.
    #[cfg(feature = "cuda")]
    fn d_hidden_alloc(&mut self, bytes: usize) -> Result<&DeviceBuffer, String> {
        if self.d_hidden.as_ref().map_or(true, |b| b.bytes() < bytes) {
            self.d_hidden = Some(DeviceBuffer::alloc(bytes).map_err(cuda::error_string)?);
        }
        Ok(self.d_hidden.as_ref().unwrap())
    }

    /// Grow (or reuse) the device gather-src buffer to at least `bytes`.
    #[cfg(feature = "cuda")]
    fn d_src_alloc(&mut self, bytes: usize) -> Result<&DeviceBuffer, String> {
        if self.d_src.as_ref().map_or(true, |b| b.bytes() < bytes) {
            self.d_src = Some(DeviceBuffer::alloc(bytes).map_err(cuda::error_string)?);
        }
        Ok(self.d_src.as_ref().unwrap())
    }

    /// Grow (or reuse) the device gather-dst buffer to at least `bytes`.
    #[cfg(feature = "cuda")]
    fn d_dst_alloc(&mut self, bytes: usize) -> Result<&DeviceBuffer, String> {
        if self.d_dst.as_ref().map_or(true, |b| b.bytes() < bytes) {
            self.d_dst = Some(DeviceBuffer::alloc(bytes).map_err(cuda::error_string)?);
        }
        Ok(self.d_dst.as_ref().unwrap())
    }

    /// Grow (or reuse) the device expert-id buffer (≤ 256 groups x 4 B).
    fn d_ids_alloc(&mut self, bytes: usize) -> Result<&DeviceBuffer, String> {
        if self.d_ids.as_ref().map_or(true, |b| b.bytes() < bytes) {
            self.d_ids = Some(DeviceBuffer::alloc(bytes).map_err(cuda::error_string)?);
        }
        Ok(self.d_ids.as_ref().unwrap())
    }

    /// Grow (or reuse) the device group-offsets buffer (≤ 257 x 4 B).
    fn d_offsets_alloc(&mut self, bytes: usize) -> Result<&DeviceBuffer, String> {
        if self.d_offsets.as_ref().map_or(true, |b| b.bytes() < bytes) {
            self.d_offsets = Some(DeviceBuffer::alloc(bytes).map_err(cuda::error_string)?);
        }
        Ok(self.d_offsets.as_ref().unwrap())
    }

    /// Grow (or reuse) the device fault-word buffer (4 B).
    fn d_fault_alloc(&mut self) -> Result<&DeviceBuffer, String> {
        if self.d_fault.is_none() {
            self.d_fault = Some(DeviceBuffer::alloc(4).map_err(cuda::error_string)?);
        }
        Ok(self.d_fault.as_ref().unwrap())
    }
}

/// Decode: the R18c mixed-M dispatch (M ≤ 8). `resident` is the number of expert
/// quarter slices present in `d_grouped` (256 in the daemon; `plan.expert_count()`
/// in the tiny smoke tests).
pub fn decode_ffn(
    d_grouped: *const u8,
    x: &[f32],
    plan: &mimo26_expert::grouped::GroupedPlan,
    resident: usize,
    out_dtype: i32,
    scratch: &mut Scratch,
) -> Result<Vec<f32>, String> {
    ffn_roundtrip(d_grouped, x, plan, resident, out_dtype, true, scratch)
}

/// Prefill: the frozen per-M kernel (M up to the capacity class, e.g. 16/64).
pub fn prefill_ffn(
    d_grouped: *const u8,
    x: &[f32],
    plan: &mimo26_expert::grouped::GroupedPlan,
    resident: usize,
    out_dtype: i32,
    scratch: &mut Scratch,
) -> Result<Vec<f32>, String> {
    ffn_roundtrip(d_grouped, x, plan, resident, out_dtype, false, scratch)
}

/// One FFN round-trip over a zero-copy (pinned) grouped image: FFN → fault check
/// → download. The x/scratch/out buffers are pooled in `scratch`.
fn ffn_roundtrip(
    d_grouped: *const u8,
    x: &[f32],
    plan: &mimo26_expert::grouped::GroupedPlan,
    resident: usize,
    out_dtype: i32,
    mixed: bool,
    scratch: &mut Scratch,
) -> Result<Vec<f32>, String> {
    let kind = if mixed { "decode" } else { "prefill" };
    plan.validate().map_err(|e| e.to_string())?;
    let total = plan.total_tokens();
    if total == 0 {
        return Err(format!("{kind}: empty plan"));
    }
    let max_m = if mixed { 8 } else { bake_capacity() as usize };
    if plan.max_tokens() > max_m {
        return Err(format!("{kind}: max_m {} > {max_m}", plan.max_tokens()));
    }
    if x.len() != total * mimo26_expert::slice::HIDDEN {
        return Err(format!(
            "{kind}: x has {} elements, expected {} tokens x {}",
            x.len(),
            total,
            mimo26_expert::slice::HIDDEN
        ));
    }
    // `d_grouped` holds `resident` expert quarter slices (256 in the daemon).

    let dtype_bytes = if out_dtype == ffi::OUT_F32 { 4 } else { 2 };
    let out_bytes = total * 4096 * dtype_bytes;

    // One kernel launch caps at 8*capacity_class rows and 256 groups; the X1a
    // prefill (4,027 tokens x 8 routes) exceeds that, so chunk the plan and
    // accumulate each chunk's rows into the single output.
    let capacity = bake_capacity() as usize;
    let max_total = 8 * capacity;
    let max_groups = 256usize;

    // Plan-structure telemetry (I5-R16): the real skewed plan's tile-M histogram
    // — the synthetic uniform plan (all M=64) hid the real cost.
    {
        use std::collections::BTreeMap;
        let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
        for g in &plan.groups {
            *hist.entry(g.tokens).or_default() += 1;
        }
        let hist_s: Vec<String> = hist.iter().map(|(m, c)| format!("M{m}:{c}")).collect();
        eprintln!(
            "plan {kind} groups={} rows={} hist=[{}]",
            plan.groups.len(), total, hist_s.join(" ")
        );
    }

    let mut out = vec![0u8; out_bytes];
    let mut chunk_start = 0usize;
    let mut gi = 0usize;
    let mut launch_idx = 0usize;
    while gi < plan.groups.len() {
        let mut chunk_groups = Vec::new();
        let mut chunk_total = 0usize;
        while gi < plan.groups.len()
            && chunk_groups.len() < max_groups
            && chunk_total + plan.groups[gi].tokens <= max_total
        {
            let g = &plan.groups[gi];
            chunk_total += g.tokens;
            chunk_groups.push(g);
            gi += 1;
        }
        if chunk_groups.is_empty() {
            return Err(format!(
                "{kind}: a single group exceeds the launch cap ({max_total} rows)"
            ));
        }
        let sub_plan = GroupedPlan {
            groups: chunk_groups
                .iter()
                .map(|g| Group {
                    expert: g.expert,
                    tokens: g.tokens,
                    token_offset: g.token_offset - chunk_start,
                })
                .collect(),
            padded: 0,
        };
        let sub_x = &x[chunk_start * mimo26_expert::slice::HIDDEN
            ..(chunk_start + chunk_total) * mimo26_expert::slice::HIDDEN];
        // Upload this chunk's x rows into the pooled buffer (fallback path; the
        // hot device path gathers x on-device instead).
        let d_x = {
            let buf = scratch.grow_x(max_total * mimo26_expert::slice::HIDDEN * 4)?;
            buf.upload_prefix(f32_bytes(sub_x)).map_err(cuda::error_string)?;
            buf.as_ptr() as *const f32
        };
        let out_ptr = scratch.out_alloc(chunk_total * 4096 * dtype_bytes)?.as_ptr();
        let sub_out = launch_chunk(d_grouped, d_x, &sub_plan, resident, out_dtype, mixed, scratch, out_ptr, true)?;
        eprintln!("launch {launch_idx} groups={} rows={}", chunk_groups.len(), chunk_total);
        launch_idx += 1;
        let nbytes = chunk_total * 4096 * dtype_bytes;
        out[chunk_start * 4096 * dtype_bytes
            ..(chunk_start + chunk_total) * 4096 * dtype_bytes]
            .copy_from_slice(&sub_out[..nbytes]);
        chunk_start += chunk_total;
    }

    if out_dtype == ffi::OUT_F32 {
        Ok(bytes_f32(&out))
    } else {
        Err(format!("{kind}: BF16 output download is not yet wired"))
    }
}

/// One kernel launch over a sub-plan whose `total_tokens` and `n_groups` fit the
/// kernel's `valid_plan` caps. Returns the chunk's `[chunk_total, 4096]` bytes.
fn launch_chunk(
    d_grouped: *const u8,
    d_x: *const f32,
    plan: &GroupedPlan,
    resident: usize,
    out_dtype: i32,
    mixed: bool,
    scratch: &mut Scratch,
    d_out: *mut core::ffi::c_void,
    download: bool,
) -> Result<Vec<u8>, String> {
    let total = plan.total_tokens();
    let dtype_bytes = if out_dtype == ffi::OUT_F32 { 4 } else { 2 };
    let out_bytes = total * 4096 * dtype_bytes;

    let ids: Vec<i32> = plan.groups.iter().map(|g| g.expert as i32).collect();
    let mut offsets: Vec<i32> = plan.groups.iter().map(|g| g.token_offset as i32).collect();
    offsets.push(total as i32);

    let scratch_bytes = 2 * total * 512 * 4;
    // Size the intermediate scratch at the full launch cap (8×capacity rows) so
    // the device pages are faulted once and reused across chunks. `d_x` is the
    // already-staged x (device-gathered in the hot path, uploaded in fallback).
    let max_total = 8 * bake_capacity() as usize;
    let d_scratch_ptr = scratch.grow_scratch(2 * max_total * 512 * 4)?.as_ptr();

    let t_start = Instant::now();
    // Pooled plan buffers (ids/offsets/fault): no per-launch cudaMalloc, and the
    // ids/offsets are staged in the pinned buffer so their H2D upload reads
    // page-locked memory (no per-copy page-locking of the small Vec<i32>).
    let ids_bytes = ids.len() * 4;
    let offsets_bytes = offsets.len() * 4;
    let (ids_p, offsets_p) = {
        let pinned = scratch.pinned_alloc(ids_bytes + offsets_bytes)?;
        let base = pinned.as_ptr();
        pinned[..ids_bytes].copy_from_slice(i32_bytes(&ids));
        let ip = base as *const u8;
        pinned[ids_bytes..ids_bytes + offsets_bytes].copy_from_slice(i32_bytes(&offsets));
        let op = unsafe { base.add(ids_bytes) };
        (ip, op)
    };
    let d_ids_ptr = {
        let d_ids = scratch.d_ids_alloc(ids_bytes)?;
        let s = unsafe { std::slice::from_raw_parts(ids_p, ids_bytes) };
        d_ids.upload_prefix_async(s, core::ptr::null_mut()).map_err(cuda::error_string)?;
        d_ids.as_ptr()
    };
    let d_offsets_ptr = {
        let d_offsets = scratch.d_offsets_alloc(offsets_bytes)?;
        let s = unsafe { std::slice::from_raw_parts(offsets_p, offsets_bytes) };
        d_offsets.upload_prefix_async(s, core::ptr::null_mut()).map_err(cuda::error_string)?;
        d_offsets.as_ptr()
    };
    let d_fault_ptr = {
        let d_fault = scratch.d_fault_alloc()?;
        d_fault.zero().map_err(cuda::error_string)?;
        d_fault.as_ptr()
    };
    let t_upload = Instant::now();

    let p = M26xPlan {
        layout_version: mimo26_expert::slice::LAYOUT_VERSION,
        manifest_arch: bake_arch(),
        manifest_sms: bake_sms(),
        capacity_class: bake_capacity(),
        resident_experts: resident as i32,
        n_groups: plan.groups.len() as i32,
        padded_groups: plan.padded as i32,
        total_tokens: total as i32,
        max_m: plan.max_tokens() as i32,
        grouped_bytes: (resident as u64) * (QUARTER_SLICE_BYTES as u64),
        x_bytes: (total * 4096 * 4) as u64,
        out_bytes: out_bytes as u64,
        scratch_bytes: scratch_bytes as u64,
        expert_ids: d_ids_ptr as *const i32,
        group_offsets: d_offsets_ptr as *const i32,
        fault: d_fault_ptr as *mut u32,
    };

    // SAFETY: every device pointer is a valid cudaMalloc allocation of the size
    // recorded in `p`; the kernel validates the plan before touching weights.
    crate::timeline::tl("ffn_launch", None);
    let rc = if mixed {
        unsafe {
            ffi::m26x_expert_ffn_mixed_v2(
                &p,
                d_grouped,
                d_x,
                out_dtype,
                0,
                d_scratch_ptr as *mut f32,
                d_out,
                core::ptr::null_mut(),
            )
        }
    } else {
        unsafe {
            ffi::m26x_expert_ffn_v2(
                &p,
                d_grouped,
                d_x,
                out_dtype,
                0,
                d_scratch_ptr as *mut f32,
                d_out,
                core::ptr::null_mut(),
            )
        }
    };
    if rc != cuda::SUCCESS {
        return Err(cuda::error_string(rc));
    }
    let rc = unsafe { cuda::cudaDeviceSynchronize() };
    if rc != cuda::SUCCESS {
        return Err(cuda::error_string(rc));
    }
    crate::timeline::tl("ffn_sync", None);
    let t_kernel = Instant::now();

    let mut fault = [0u8; 4];
    scratch
        .d_fault_alloc()?
        .download(&mut fault)
        .map_err(cuda::error_string)?;
    if fault != [0, 0, 0, 0] {
        return Err(format!("device fault word {fault:?}"));
    }

    eprintln!(
        "ffn-sub upload={:.3} kernel+sync={:.3} download={:.3} ms (total={} rows)",
        (t_upload - t_start).as_secs_f64() * 1e3,
        (t_kernel - t_upload).as_secs_f64() * 1e3,
        t_kernel.elapsed().as_secs_f64() * 1e3,
        total
    );
    if !download {
        // Leave the output on the device (for the route-reduce path).
        return Ok(Vec::new());
    }
    let mut raw = vec![0u8; out_bytes];
    let rc = unsafe { cuda::cudaMemcpy(raw.as_mut_ptr() as *mut core::ffi::c_void, d_out, out_bytes, cuda::D2H) };
    if rc != cuda::SUCCESS {
        return Err(cuda::error_string(rc));
    }
    Ok(raw)
}

/// The baked arch/SM/capacity are compile-time constants emitted by build.rs
/// (they match exactly what nvcc baked) — no runtime env fallback, so a daemon
/// launched without the env still carries the correct manifest identity.
fn bake_arch() -> i32 {
    env!("MIMO26F_BAKED_ARCH").parse().expect("MIMO26F_BAKED_ARCH must be an integer")
}
fn bake_sms() -> i32 {
    env!("MIMO26F_BAKED_SMS").parse().expect("MIMO26F_BAKED_SMS must be an integer")
}
fn bake_capacity() -> i32 {
    env!("MIMO26F_CAPACITY_CLASS").parse().expect("MIMO26F_CAPACITY_CLASS must be an integer")
}

/// The startup AOT self-check: the manifest identity (baked arch/SM/capacity)
/// must match the live device. Runs once at boot, before "listening"; refuses
/// to serve on any mismatch and prints the three pairs.
#[cfg(feature = "cuda")]
pub fn check_aot() -> Result<(), String> {
    let manifest_arch = bake_arch();
    let manifest_sms = bake_sms();
    let capacity = bake_capacity();
    let rc = unsafe { ffi::m26x_check_aot(manifest_arch, manifest_sms, capacity, 0) };
    if rc != cuda::SUCCESS {
        let mut live_arch = 0i32;
        let mut live_sms = 0i32;
        let _ = unsafe { ffi::m26x_device_identity(&mut live_arch, &mut live_sms) };
        return Err(format!(
            "AOT gate failed (rc {rc}, {}): baked=({},{}) manifest=({manifest_arch},{manifest_sms}) live=({live_arch},{live_sms})",
            cuda::error_string(rc),
            env!("MIMO26F_BAKED_ARCH"),
            env!("MIMO26F_BAKED_SMS"),
        ));
    }
    Ok(())
}

/// `&[f32]` as raw little-endian bytes (matches the device's f32 layout).
fn f32_bytes(x: &[f32]) -> &[u8] {
    // SAFETY: f32 is 4 bytes and x is contiguous, so the byte view is x.len()*4.
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) }
}

fn i32_bytes(x: &[i32]) -> &[u8] {
    // SAFETY: i32 is 4 bytes and x is contiguous, so the byte view is x.len()*4.
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) }
}

fn bytes_f32(b: &[u8]) -> Vec<f32> {
    assert_eq!(b.len() % 4, 0, "f32 buffer length");
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `&mut [u16]` as raw little-endian bytes (BF16 return rows).
fn u16_bytes_mut(x: &mut [u16]) -> &mut [u8] {
    // SAFETY: u16 is 2 bytes and x is contiguous, so the byte view is x.len()*2.
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut u8, x.len() * 2) }
}

/// Device route reduce: collapse the padded FFN output (already on device) into
/// per-token BF16 return rows. `padded`/`weight` are the token-major route arrays
/// from `RoutePlan::token_major_routes`; token `t` owns `[t*8, t*8+8)`.
#[cfg(feature = "cuda")]
pub fn route_reduce(
    d_ffn_out: &DeviceBuffer,
    padded: &[i32],
    weight: &[f32],
    tokens: usize,
    hidden: usize,
) -> Result<Vec<u16>, String> {
    use core::ffi::c_int;
    let routes = padded.len();
    let token_off: Vec<i32> = (0..=tokens).map(|t| (t * 8) as i32).collect();

    let d_padded = DeviceBuffer::alloc(padded.len() * 4).map_err(cuda::error_string)?;
    d_padded.upload(i32_bytes(padded)).map_err(cuda::error_string)?;
    let d_weight = DeviceBuffer::alloc(weight.len() * 4).map_err(cuda::error_string)?;
    d_weight.upload(f32_bytes(weight)).map_err(cuda::error_string)?;
    let d_off = DeviceBuffer::alloc(token_off.len() * 4).map_err(cuda::error_string)?;
    d_off.upload(i32_bytes(&token_off)).map_err(cuda::error_string)?;
    let d_bf16 = DeviceBuffer::alloc(tokens * hidden * 2).map_err(cuda::error_string)?;

    let rc = unsafe {
        ffi::m26x_route_reduce(
            d_ffn_out.as_ptr() as *const f32,
            d_padded.as_ptr() as *const c_int,
            d_weight.as_ptr() as *const f32,
            d_off.as_ptr() as *const c_int,
            tokens as c_int,
            hidden as c_int,
            routes as c_int,
            d_bf16.as_ptr() as *mut u16,
            core::ptr::null_mut(),
        )
    };
    if rc != cuda::SUCCESS {
        return Err(cuda::error_string(rc));
    }
    let rc = unsafe { cuda::cudaDeviceSynchronize() };
    if rc != cuda::SUCCESS {
        return Err(cuda::error_string(rc));
    }
    let mut bf16 = vec![0u16; tokens * hidden];
    d_bf16.download(u16_bytes_mut(&mut bf16)).map_err(cuda::error_string)?;
    Ok(bf16)
}

/// Device gather: upload the per-token hidden rows (via a persistent pinned
/// staging buffer) and build the padded routed x on the device (zero-fill then
/// gather), removing the pageable host replicate and its pageable H2D copy.
/// All staging (pinned host hidden + device hidden/src/dst/x) is preallocated at
/// the launch cap and reused, so a 1-token decode pays no per-request
/// `cudaHostRegister`/`cudaMalloc`/`cudaDeviceSynchronize`. Returns the device
/// pointer to the full `[padded_rows, HIDDEN]` x.
#[cfg(feature = "cuda")]
fn upload_and_gather(
    tok_hidden: &[f32],
    rp: &RoutePlan,
    scratch: &mut Scratch,
) -> Result<*const f32, String> {
    let hidden = mimo26_expert::slice::HIDDEN;
    let routed = rp.routed_rows();
    let nbytes = tok_hidden.len() * 4;

    // Gather indices: group-order row_token (src) and row_padded (dst).
    let src: Vec<i32> = rp.row_token.iter().map(|&t| t as i32).collect();
    let dst: Vec<i32> = rp.row_padded.iter().map(|&p| p as i32).collect();

    // Stage hidden + src + dst into the persistent pinned buffer (registered
    // once, one allocation) so every H2D upload reads page-locked memory — no
    // per-request pin and no per-copy page-locking of the small arrays.
    let src_bytes = src.len() * 4;
    let dst_bytes = dst.len() * 4;
    let (hidden_p, src_p, dst_p) = {
        let pinned = scratch.pinned_alloc(nbytes + src_bytes + dst_bytes)?;
        let base = pinned.as_ptr();
        pinned[..nbytes].copy_from_slice(f32_bytes(tok_hidden));
        let hp = base as *const u8;
        pinned[nbytes..nbytes + src_bytes].copy_from_slice(i32_bytes(&src));
        let sp = unsafe { base.add(nbytes) };
        pinned[nbytes + src_bytes..nbytes + src_bytes + dst_bytes].copy_from_slice(i32_bytes(&dst));
        let dp = unsafe { base.add(nbytes + src_bytes) };
        (hp, sp, dp)
    };

    let d_hidden_ptr = {
        let d_hidden = scratch.d_hidden_alloc(nbytes)?;
        // SAFETY: hidden_p points into scratch.pinned (page-locked, alive).
        let s = unsafe { std::slice::from_raw_parts(hidden_p, nbytes) };
        d_hidden.upload_prefix_async(s, core::ptr::null_mut()).map_err(cuda::error_string)?;
        d_hidden.as_ptr()
    };
    let d_src_ptr = {
        let d_src = scratch.d_src_alloc(src_bytes)?;
        let s = unsafe { std::slice::from_raw_parts(src_p, src_bytes) };
        d_src.upload_prefix_async(s, core::ptr::null_mut()).map_err(cuda::error_string)?;
        d_src.as_ptr()
    };
    let d_dst_ptr = {
        let d_dst = scratch.d_dst_alloc(dst_bytes)?;
        let s = unsafe { std::slice::from_raw_parts(dst_p, dst_bytes) };
        d_dst.upload_prefix_async(s, core::ptr::null_mut()).map_err(cuda::error_string)?;
        d_dst.as_ptr()
    };

    // Full-plan padded x, zeroed then gathered (padding rows read as 0).
    let d_x_ptr = {
        let d_x = scratch.grow_x(rp.padded_rows() * hidden * 4)?;
        d_x.zero().map_err(cuda::error_string)?;
        d_x.as_ptr()
    };

    let t_g = Instant::now();
    let rc = unsafe {
        ffi::m26x_gather_x(
            d_hidden_ptr as *const f32,
            d_src_ptr as *const core::ffi::c_int,
            d_dst_ptr as *const core::ffi::c_int,
            routed as core::ffi::c_int,
            hidden as core::ffi::c_int,
            d_x_ptr as *mut f32,
            core::ptr::null_mut(),
        )
    };
    if rc != cuda::SUCCESS {
        return Err(cuda::error_string(rc));
    }
    // No cudaDeviceSynchronize here: the gather and the FFN launches run on the
    // same (null) stream and serialize; the FFN's own sync covers both.
    eprintln!(
        "gather rows={routed} hidden_bytes={nbytes} gather={:.3} ms",
        t_g.elapsed().as_secs_f64() * 1e3
    );
    Ok(d_x_ptr as *const f32)
}

/// FFN + device route reduce in one pass: gather the padded x on the device from
/// the per-token hidden, run the FFN leaving its output on the device, then
/// collapse the padded rows into per-token BF16 return rows. Handles both single-
/// and multi-chunk plans (the full output buffer is allocated once and each launch
/// writes its own offset), so the real skewed plan (which exceeds one launch's
/// 8×capacity rows) no longer falls back to the CPU path.
#[cfg(feature = "cuda")]
pub fn ffn_route_reduce(
    d_grouped: *const u8,
    tok_hidden: &[f32],
    rp: &RoutePlan,
    out_dtype: i32,
    mixed: bool,
    scratch: &mut Scratch,
) -> Result<Vec<u16>, String> {
    let plan = &rp.plan;
    let total = plan.total_tokens();
    let hidden = mimo26_expert::slice::HIDDEN;
    let dtype_bytes = if out_dtype == ffi::OUT_F32 { 4 } else { 2 };
    let capacity = bake_capacity() as usize;
    let max_total = 8 * capacity;
    let max_groups = 256usize;

    // Gather the padded x on the device (no pageable host replicate).
    let d_x_full = upload_and_gather(tok_hidden, rp, scratch)?;
    crate::timeline::tl("gather_done", None);

    // Full output buffer, allocated once; each launch writes its chunk's offset.
    let full_out = scratch.out_alloc(total * 4096 * dtype_bytes)?;
    let full_out_ptr = full_out.as_ptr();

    let mut chunk_start = 0usize;
    let mut gi = 0usize;
    let mut launch_idx = 0usize;
    while gi < plan.groups.len() {
        let mut chunk_groups = Vec::new();
        let mut chunk_total = 0usize;
        while gi < plan.groups.len()
            && chunk_groups.len() < max_groups
            && chunk_total + plan.groups[gi].tokens <= max_total
        {
            chunk_total += plan.groups[gi].tokens;
            chunk_groups.push(plan.groups[gi]);
            gi += 1;
        }
        if chunk_groups.is_empty() {
            return Err(format!("ffn: a single group exceeds the launch cap ({max_total} rows)"));
        }
        let sub_plan = GroupedPlan {
            groups: chunk_groups
                .iter()
                .map(|g| Group {
                    expert: g.expert,
                    tokens: g.tokens,
                    token_offset: g.token_offset - chunk_start,
                })
                .collect(),
            padded: 0,
        };
        let d_x = unsafe { d_x_full.add(chunk_start * hidden) };
        let out_ptr = unsafe {
            (full_out_ptr as *mut u8).add(chunk_start * 4096 * dtype_bytes) as *mut core::ffi::c_void
        };
        let _ = launch_chunk(
            d_grouped, d_x, &sub_plan, EXPERTS_PER_LAYER, out_dtype, mixed, scratch,
            out_ptr, false,
        )?;
        eprintln!("launch {launch_idx} groups={} rows={}", chunk_groups.len(), chunk_total);
        launch_idx += 1;
        chunk_start += chunk_total;
    }

    // Reduce the full device output into BF16.
    let d_out = scratch.out_ref();
    let (padded, weight) = rp.token_major_routes();
    let bf16 = route_reduce(d_out, &padded, &weight, rp.tokens, hidden)?;
    crate::timeline::tl("reduce_done", None);
    Ok(bf16)
}

/// CPU route reduce + BF16, bitwise-matching the device reduce: accumulate each
/// token's 8 routes in the token-major order (the same order `token_major_routes`
/// emits), then RNE BF16. The CPU merge gate and the multi-chunk fallback use this.
pub fn cpu_reduce_bf16(
    ffn_out: &[f32],
    padded: &[i32],
    weight: &[f32],
    tokens: usize,
    hidden: usize,
) -> Vec<u16> {
    let mut out = vec![0.0f32; tokens * hidden];
    for (t, (&p, &w)) in padded.iter().zip(weight.iter()).enumerate() {
        let token = t / 8; // token-major: token `token` owns [token*8, token*8+8)
        for h in 0..hidden {
            out[token * hidden + h] += ffn_out[p as usize * hidden + h] * w;
        }
    }
    out.iter().map(|&v| mimo26_wire::bf16::f32_to_bf16_rne(v)).collect()
}

/// A synthetic MXFP4 quarter slice: varied E2M1 nibbles in the payload regions,
/// scale 127 (1.0) everywhere, so both the CPU oracle and the kernel decode the
/// same bytes.
#[cfg(all(test, feature = "cuda"))]
fn synthetic_slice(seed: u8) -> Vec<u8> {
    use mimo26_repack::geom::Proj;
    let mut bytes = vec![0u8; QUARTER_SLICE_BYTES];
    for p in [Proj::Gate, Proj::Up, Proj::Down] {
        let poff = p.slice_payload_off();
        let plen = p.slice_payload_bytes();
        for (j, b) in bytes[poff..poff + plen].iter_mut().enumerate() {
            let lo = ((j as u32 * 7 + seed as u32 + 1) % 16) as u8;
            let hi = ((j as u32 * 11 + seed as u32 + 3) % 16) as u8;
            *b = (hi << 4) | lo;
        }
        let soff = p.slice_scale_off();
        let slen = p.slice_scale_bytes();
        bytes[soff..soff + slen].fill(127);
    }
    bytes
}

/// Device smoke — only under the `cuda` feature. Runs the FFN on tiny synthetic
/// plans and checks the output against the CPU oracle (the kernel accumulates in
/// f32 FMA + warp reduction, the oracle in f32 adds, so this is a tolerance
/// check, not a bitwise proof — the bitwise proof is the frozen per-M comparison
/// in `step_b2_mixed.cu`).
#[cfg(all(test, feature = "cuda"))]
mod smoke {
    use super::*;
    use mimo26_expert::grouped::GroupedPlan;
    use mimo26_expert::NaiveBits;

    fn check_against_oracle(plan: &GroupedPlan, grouped: &[u8], x: &[f32], kind: &str) {
        let want = mimo26_expert::grouped::expert_ffn_self_contained(
            grouped,
            x,
            plan,
            NaiveBits::NONE,
        )
        .expect("cpu oracle");
        let reg = register_grouped(grouped).expect("register grouped");
        let d_grouped = reg.dptr();
        let mut scratch = Scratch::new();
        let got = if kind == "decode" {
            decode_ffn(d_grouped, x, plan, plan.expert_count(), ffi::OUT_F32, &mut scratch)
        } else {
            prefill_ffn(d_grouped, x, plan, plan.expert_count(), ffi::OUT_F32, &mut scratch)
        }
        .unwrap_or_else(|e| panic!("{kind} ffn failed: {e}"));
        assert_eq!(got.len(), want.data.len(), "output extent");

        let (atol, rtol) = (1e-2f32, 1e-3f32);
        let mut max_ratio = 0.0f32;
        for (g, w) in got.iter().zip(want.data.iter()) {
            let bound = atol + rtol * w.abs();
            max_ratio = max_ratio.max((g - w).abs() / bound);
        }
        assert!(
            max_ratio <= 1.0,
            "{kind} diverged from oracle: max_ratio={max_ratio}"
        );
    }

    fn build(n_experts: usize, m: usize, padded: usize) -> (GroupedPlan, Vec<u8>, Vec<f32>) {
        let plan = GroupedPlan::uniform(n_experts, m, padded);
        let total = plan.total_tokens();
        let mut grouped = Vec::with_capacity(n_experts * QUARTER_SLICE_BYTES);
        for e in 0..n_experts {
            grouped.extend_from_slice(&synthetic_slice(e as u8));
        }
        let x: Vec<f32> = (0..total * 4096)
            .map(|i| ((i as f32 * 0.6180339887).fract() - 0.5) * 2.0)
            .collect();
        (plan, grouped, x)
    }

    #[test]
    fn decode_matches_cpu_oracle() {
        let (plan, grouped, x) = build(3, 4, 1); // 12 tokens, 1 padded slot
        check_against_oracle(&plan, &grouped, &x, "decode");
    }

    #[test]
    fn prefill_matches_cpu_oracle_m16() {
        let (plan, grouped, x) = build(2, 16, 0); // 32 tokens, M16 per expert
        check_against_oracle(&plan, &grouped, &x, "prefill");
    }

    #[test]
    fn prefill_matches_cpu_oracle_m64() {
        let (plan, grouped, x) = build(1, 64, 0); // 64 tokens, single M64 expert
        check_against_oracle(&plan, &grouped, &x, "prefill");
    }

    /// I5-R16: zero-copy (cudaHostRegister) grouped-image reads vs a
    /// device-allocated copy — the builder's question whether the pinned host
    /// mapping gets the same cached path as cudaMalloc memory on GB10. Outputs
    /// must be bit-identical; the wall-clock is printed, not asserted.
    #[test]
    fn zero_copy_vs_device_grouped_timing() {
        let (plan, grouped, x) = build(4, 64, 0); // 256 rows, M64 per expert
        let mut scratch = Scratch::new();

        let reg_zc = register_grouped(&grouped).expect("register");
        let d_zc = reg_zc.dptr();
        let t = Instant::now();
        let out_zc = prefill_ffn(d_zc, &x, &plan, plan.expert_count(), ffi::OUT_F32, &mut scratch).expect("zc ffn");
        let zc_ms = t.elapsed().as_secs_f64() * 1e3;

        let d_dev = DeviceBuffer::alloc(grouped.len()).expect("alloc");
        d_dev.upload(&grouped).expect("upload");
        let t = Instant::now();
        let out_dev = prefill_ffn(d_dev.as_ptr() as *const u8, &x, &plan, plan.expert_count(), ffi::OUT_F32, &mut scratch).expect("dev ffn");
        let dev_ms = t.elapsed().as_secs_f64() * 1e3;

        assert_eq!(out_zc, out_dev, "zero-copy and device FFN outputs must match bitwise");
        eprintln!("zero-copy ffn={zc_ms:.3}ms device ffn={dev_ms:.3}ms (grouped={} B)", grouped.len());
    }

    /// I5-R16: isolate the large H2D copy speed with and without a registered
    /// (pinned) grouped image — the real plan's upload is ~0.83 GB/s for ~125 MB.
    #[test]
    fn h2d_copy_speed_with_and_without_registered_grouped() {
        use std::time::Instant;
        let n = 125 * 1024 * 1024 / 4; // 125 MB of f32
        let host: Vec<f32> = vec![0.5f32; n];
        let d = DeviceBuffer::alloc(host.len() * 4).expect("alloc");
        d.upload_prefix(f32_bytes(&host)).expect("warm");

        let t = Instant::now();
        for _ in 0..4 {
            d.upload_prefix(f32_bytes(&host)).expect("copy");
            unsafe { cuda::cudaDeviceSynchronize() };
        }
        let no_zc = t.elapsed().as_secs_f64() / 4.0;

        // Register a moderately large grouped image (64 experts ≈ 214 MB).
        let grouped = vec![0x22u8; 64 * QUARTER_SLICE_BYTES];
        let _dg = register_grouped(&grouped).expect("register");

        let t = Instant::now();
        for _ in 0..4 {
            d.upload_prefix(f32_bytes(&host)).expect("copy");
            unsafe { cuda::cudaDeviceSynchronize() };
        }
        let with_zc = t.elapsed().as_secs_f64() / 4.0;

        eprintln!(
            "H2D 125MB: no-register={no_zc:.3}ms ({:.1} GB/s) with-register={with_zc:.3}ms ({:.1} GB/s)",
            125.0 / no_zc,
            125.0 / with_zc
        );
    }

    /// The X1a prefill (4,027 tokens x 8 routes) exceeds one launch's 8*class
    /// row cap, so `ffn_roundtrip` chunks it. 65 experts x M256 = 16,640 rows >
    /// 16,384 forces a 2-launch chunk. Correctness is already pinned by the
    /// m16/m64 oracle tests; this checks the chunked path runs (no fault, no
    /// crash) and returns the full padded extent.
    #[test]
    fn prefill_chunked_runs_and_has_right_extent() {
        let (plan, grouped, x) = build(65, 256, 0);
        assert!(plan.total_tokens() > 16_384, "plan should exceed the launch cap");
        let reg = register_grouped(&grouped).expect("register grouped");
        let d_grouped = reg.dptr();
        let mut scratch = Scratch::new();
        let got = prefill_ffn(d_grouped, &x, &plan, plan.expert_count(), ffi::OUT_F32, &mut scratch).expect("chunked prefill");
        assert_eq!(got.len(), plan.total_tokens() * 4096, "output extent");
    }
}

#[cfg(all(test, feature = "cuda"))]
mod gpu_reduce_tests {
    use super::*;
    use crate::route::RoutePlan;
    use mimo26_wire::bf16::f32_to_bf16_rne;
    use mimo26_wire::frame::RouteEntry;

    fn entry(token: u32, expert: u32, w: f32) -> RouteEntry {
        RouteEntry { row_index: token, expert_id: expert, gate_weight: w }
    }

    /// The device route reduce must match the CPU `RoutePlan::reduce` +
    /// `f32_to_bf16_rne` bitwise on the BF16 return rows.
    #[test]
    fn gpu_route_reduce_matches_cpu_bitwise() {
        // 3 tokens, 8 routes each, skewed across experts (forces padding).
        let routes = vec![
            entry(0, 0, 0.5), entry(0, 1, 0.25), entry(0, 2, 0.125), entry(0, 0, 0.0625),
            entry(0, 3, 0.03125), entry(0, 1, 0.015625), entry(0, 0, 0.0078125), entry(0, 4, 0.0078125),
            entry(1, 1, 0.5), entry(1, 0, 0.25), entry(1, 4, 0.125), entry(1, 2, 0.0625),
            entry(1, 3, 0.03125), entry(1, 0, 0.015625), entry(1, 2, 0.0078125), entry(1, 1, 0.00390625),
            entry(2, 0, 0.4), entry(2, 2, 0.3), entry(2, 1, 0.2), entry(2, 3, 0.05),
            entry(2, 4, 0.025), entry(2, 1, 0.0125), entry(2, 0, 0.00625), entry(2, 2, 0.00625),
        ];
        let rp = RoutePlan::from_routes(&routes, 3).expect("plan");
        let tokens = 3usize;
        let hidden = mimo26_expert::slice::HIDDEN; // 4096

        // Deterministic synthetic FFN output [padded_rows, 4096].
        let n = rp.padded_rows() * hidden;
        let ffn_out: Vec<f32> = (0..n).map(|i| ((i % 251) as f32) * 1e-3).collect();

        // CPU reference.
        let cpu_partial = rp.reduce(&ffn_out).expect("cpu reduce");
        let cpu_bf16: Vec<u16> = cpu_partial.iter().map(|&v| f32_to_bf16_rne(v)).collect();

        // GPU.
        let d_ffn_out = DeviceBuffer::alloc(ffn_out.len() * 4).expect("alloc");
        d_ffn_out.upload(f32_bytes(&ffn_out)).expect("upload");
        let (padded, weight) = rp.token_major_routes();
        let gpu_bf16 = route_reduce(&d_ffn_out, &padded, &weight, tokens, hidden).expect("gpu reduce");

        assert_eq!(gpu_bf16.len(), cpu_bf16.len());
        let diffs: Vec<usize> = gpu_bf16
            .iter()
            .zip(cpu_bf16.iter())
            .enumerate()
            .filter_map(|(i, (g, c))| if g != c { Some(i) } else { None })
            .collect();
        assert!(diffs.is_empty(), "{} BF16 diffs (first few {:?})", diffs.len(), &diffs[..diffs.len().min(8)]);
    }

    /// The device gather must reproduce `RoutePlan::replicate_x` bitwise: zero-fill
    /// the padded x then copy each token's hidden row to its routed padded rows.
    #[test]
    fn gather_x_matches_cpu_replicate_bitwise() {
        let routes = vec![
            entry(0, 0, 0.5), entry(0, 1, 0.25), entry(0, 0, 0.125), entry(0, 2, 0.0625),
            entry(0, 3, 0.03125), entry(0, 1, 0.015625), entry(0, 0, 0.0078125), entry(0, 4, 0.0078125),
            entry(1, 1, 0.5), entry(1, 0, 0.25), entry(1, 4, 0.125), entry(1, 2, 0.0625),
            entry(1, 3, 0.03125), entry(1, 0, 0.015625), entry(1, 2, 0.0078125), entry(1, 1, 0.00390625),
            entry(2, 0, 0.4), entry(2, 2, 0.3), entry(2, 1, 0.2), entry(2, 3, 0.05),
            entry(2, 4, 0.025), entry(2, 1, 0.0125), entry(2, 0, 0.00625), entry(2, 2, 0.00625),
        ];
        let rp = RoutePlan::from_routes(&routes, 3).expect("plan");
        let hidden = mimo26_expert::slice::HIDDEN;
        let tok_hidden: Vec<f32> = (0..3 * hidden).map(|i| ((i % 251) as f32) * 1e-3).collect();
        let want = rp.replicate_x(&tok_hidden).expect("cpu replicate");

        let mut scratch = Scratch::new();
        let _d_x = upload_and_gather(&tok_hidden, &rp, &mut scratch).expect("gather");
        // The gathered x lives in the scratch x buffer; download it for comparison.
        let buf = scratch.grow_x(rp.padded_rows() * hidden * 4).expect("grow");
        let mut raw = vec![0u8; want.len() * 4];
        buf.download(&mut raw).expect("download");
        let got: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, want, "device gather must match cpu replicate_x bitwise");
    }
}
