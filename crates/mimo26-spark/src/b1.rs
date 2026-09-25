//! Spark B1 serving path (perf reset R3): the Rust side of `kernels/b1_serve.cu`.
//!
//! A [`Layer`] owns one MoE layer's B1P1 prepared pool (prepared once from the
//! resident canonical-v2 image); a [`Scratch`] owns the per-request device
//! buffers and stream. [`ffn`] runs one rank's E-W4A8-v1 expert FFN on the wire
//! payload as received (E4M3 rows + K32 scales, no decode) and returns the
//! rank's BF16 partial rows.

use core::ffi::{c_char, c_int};

#[repr(C)]
struct RawLayer {
    _p: [u8; 0],
}

#[repr(C)]
struct RawScratch {
    _p: [u8; 0],
}

unsafe extern "C" {
    fn m26s_b1_layer_new(
        canonical_dev: *const u8,
        canonical_bytes: u64,
        rank: u32,
        out: *mut *mut RawLayer,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn m26s_b1_layer_free(l: *mut RawLayer);
    fn m26s_b1_layer_all_normal(l: *const RawLayer) -> c_int;
    fn m26s_b1_scratch_new(out: *mut *mut RawScratch, err: *mut c_char, errlen: usize) -> c_int;
    fn m26s_b1_scratch_free(s: *mut RawScratch);
    fn m26s_b1_ffn(
        l: *const RawLayer,
        s: *mut RawScratch,
        payload: *const u8,
        payload_pitch: usize,
        scales: *const u8,
        scales_pitch: usize,
        ids: *const i32,
        weights: *const f32,
        rows: u32,
        bf16_out: *mut u16,
        ms: *mut f32,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
}

fn err_string(buf: &[u8]) -> String {
    let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

/// One MoE layer's prepared B1 pool (256 experts, this rank's quarter slices).
pub struct Layer(*mut RawLayer);

// SAFETY: the handle is only used from the serving thread; the device memory it
// owns has no thread affinity.
unsafe impl Send for Layer {}

impl Layer {
    /// Whether every weight scale of the layer lies in [2, 252] (prepare-time
    /// scan), so the FC1/FC2 MMAs run without the per-MMA exceptional-scale vote.
    pub fn all_scales_normal(&self) -> bool {
        // SAFETY: a live handle.
        unsafe { m26s_b1_layer_all_normal(self.0) != 0 }
    }

    /// Prepare from the layer's resident canonical image (`256 * 3,342,336`
    /// bytes of device memory). The caller may free the canonical buffer after.
    pub fn new(canonical_dev: *const u8, bytes: usize, rank: u32) -> Result<Self, String> {
        let mut out = core::ptr::null_mut();
        let mut err = [0u8; 256];
        // SAFETY: canonical_dev is a live device allocation of `bytes`; out/err are valid.
        let rc = unsafe {
            m26s_b1_layer_new(canonical_dev, bytes as u64, rank, &mut out, err.as_mut_ptr() as *mut c_char, err.len())
        };
        if rc != 0 {
            return Err(err_string(&err));
        }
        Ok(Self(out))
    }
}

impl Drop for Layer {
    fn drop(&mut self) {
        // SAFETY: created by m26s_b1_layer_new and freed exactly once.
        unsafe { m26s_b1_layer_free(self.0) }
    }
}

/// Per-request device scratch and the B1 stream.
pub struct Scratch(*mut RawScratch);

// SAFETY: as for Layer.
unsafe impl Send for Scratch {}

impl Scratch {
    pub fn new() -> Result<Self, String> {
        let mut out = core::ptr::null_mut();
        let mut err = [0u8; 256];
        // SAFETY: out/err are valid for writes.
        let rc = unsafe { m26s_b1_scratch_new(&mut out, err.as_mut_ptr() as *mut c_char, err.len()) };
        if rc != 0 {
            return Err(err_string(&err));
        }
        Ok(Self(out))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // SAFETY: created by m26s_b1_scratch_new and freed exactly once.
        unsafe { m26s_b1_scratch_free(self.0) }
    }
}

/// Stage timings of one [`ffn`] call.
#[derive(Debug, Clone, Copy, Default)]
pub struct FfnStats {
    pub upload_plan_ms: f32,
    pub gpu_ms: f32,
    pub tail_ms: f32,
    pub groups: u32,
    /// GPU phases (events): FC1, intermediate quantizer, FC2, route reduce.
    pub phase_ms: [f32; 4],
}

/// One rank's FFN for `rows` tokens: `payload` `[rows*4096]` E4M3, `scales`
/// `[rows*128]` UE8M0, `ids`/`weights` `[rows*8]` token-major top-8. Writes the
/// rank's BF16 partial `[rows*4096]` into `bf16_out`.
#[allow(clippy::too_many_arguments)]
pub fn ffn(
    layer: &Layer,
    scratch: &mut Scratch,
    payload: &[u8],
    scales: &[u8],
    ids: &[i32],
    weights: &[f32],
    rows: usize,
    bf16_out: &mut [u16],
) -> Result<FfnStats, String> {
    const H: usize = 4096;
    if rows == 0 || payload.len() != rows * H || scales.len() != rows * H / 32 {
        return Err(format!("b1 ffn: extents do not match {rows} rows"));
    }
    // SAFETY: extents checked above.
    unsafe { ffn_raw(layer, scratch, payload.as_ptr(), H, scales.as_ptr(), H / 32, ids, weights, rows, bf16_out) }
}

/// [`ffn`] on the request frame's interleaved hidden rows, read in place (perf
/// reset: Spark zero-copy receive): `hidden` holds `rows` rows of `row_stride`
/// bytes, each the 4,096 E4M3 payload then its 128 scales.
#[allow(clippy::too_many_arguments)]
pub fn ffn_strided(
    layer: &Layer,
    scratch: &mut Scratch,
    hidden: &[u8],
    row_stride: usize,
    ids: &[i32],
    weights: &[f32],
    rows: usize,
    bf16_out: &mut [u16],
) -> Result<FfnStats, String> {
    const H: usize = 4096;
    if rows == 0 || row_stride < H + H / 32 || hidden.len() < (rows - 1) * row_stride + H + H / 32 {
        return Err(format!("b1 ffn: hidden rows do not match {rows} rows of {row_stride} B"));
    }
    // SAFETY: rows of `row_stride` bytes checked above; scales start 4,096 in.
    unsafe {
        ffn_raw(layer, scratch, hidden.as_ptr(), row_stride, hidden.as_ptr().add(H), row_stride, ids, weights, rows,
            bf16_out)
    }
}

/// # Safety
/// `payload`/`scales` point at `rows` readable rows of 4,096 / 128 bytes at the
/// given pitches.
#[allow(clippy::too_many_arguments)]
unsafe fn ffn_raw(
    layer: &Layer,
    scratch: &mut Scratch,
    payload: *const u8,
    payload_pitch: usize,
    scales: *const u8,
    scales_pitch: usize,
    ids: &[i32],
    weights: &[f32],
    rows: usize,
    bf16_out: &mut [u16],
) -> Result<FfnStats, String> {
    const H: usize = 4096;
    if rows == 0 || ids.len() != rows * 8 || weights.len() != rows * 8 || bf16_out.len() != rows * H {
        return Err(format!("b1 ffn: extents do not match {rows} rows"));
    }
    let mut ms = [0f32; 8];
    let mut err = [0u8; 256];
    // SAFETY: the caller guarantees the input rows; the slices are sized above.
    let rc = unsafe {
        m26s_b1_ffn(
            layer.0,
            scratch.0,
            payload,
            payload_pitch,
            scales,
            scales_pitch,
            ids.as_ptr(),
            weights.as_ptr(),
            rows as u32,
            bf16_out.as_mut_ptr(),
            ms.as_mut_ptr(),
            err.as_mut_ptr() as *mut c_char,
            err.len(),
        )
    };
    if rc != 0 {
        return Err(err_string(&err));
    }
    Ok(FfnStats {
        upload_plan_ms: ms[0],
        gpu_ms: ms[1],
        tail_ms: ms[2],
        groups: ms[3] as u32,
        phase_ms: [ms[4], ms[5], ms[6], ms[7]],
    })
}
