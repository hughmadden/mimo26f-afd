//! The attention serving path over the device — the safe wrappers that upload
//! host activations, invoke the C3/P8 decode (split-KV + reduce) or the P1
//! chunked prefill, and download the result. Mirrors `mimo26-spark::decode`.
//!
//! Only compiled + run under the `cuda` feature (the FFI resolves against
//! `libcudart`); the CPU merge gate never references these symbols. The safe
//! wrapper enforces the header's contract: buffers are caller-owned and live
//! through the launch, and the sink enters ONCE at the reduce.
//!
//! # Tensor-core swap (I5-R17, `14880dc`)
//!
//! On the engine's real shape (`use_tc`: n_q 64, d_qk 192, d_v 128, n_kv 4/8) the
//! serve path runs the tensor-core kernels — P1 `m26_attn_prefill_fp8_tc_split`
//! (prefill) and C3/P8 `m26_attn_decode_splitkv_fp8_tc` + `m26_attn_reduce_tc`
//! (decode) — at the A-f32q default (f32 Q, FP8 unit-scale KV, f32 accumulate).
//! The naive f64 kernels remain the reference/bisect path and serve the tiny
//! golden shape. A-f32q parity bars vs the f64 reference (I4 audits,
//! `docs/design/attn-perf.md`): **original contract 1e-5 absolute**, measured
//! maxima **5.811e-7**; R7 high-V stress bounds **0.00448 / 0.00896**, measured
//! maxima **0.001740**.

use crate::cuda;
use crate::device::DeviceBuffer;
use crate::ffi::{self, CudaError, M26Geom};

fn f32_bytes(x: &[f32]) -> &[u8] {
    // SAFETY: a &[f32] is a valid &[u8] of 4x the length (little-endian host).
    unsafe { core::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) }
}

fn i64_bytes(x: &[i64]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 8) }
}

/// Per-stage serve-path timer (host wall), enabled by `MIMO26_PROFILE=1`.
fn serve_profile(stage: &str, ms: f32) {
    if std::env::var_os("MIMO26_PROFILE").is_some() {
        eprintln!("PROFILE {stage} {ms:.3}");
    }
}

/// Run `launch` (the kernel launches) and time it on the GPU via cudaEvent, so
/// the async kernel cost is separated from the host-side upload/download. Returns
/// the elapsed GPU milliseconds.
fn time_kernel_ms<F: FnOnce() -> Result<(), String>>(launch: F) -> Result<f32, String> {
    let mut start: cuda::CudaEvent = std::ptr::null_mut();
    let mut stop: cuda::CudaEvent = std::ptr::null_mut();
    unsafe {
        check(cuda::cudaEventCreate(&mut start))?;
        check(cuda::cudaEventCreate(&mut stop))?;
        check(cuda::cudaEventRecord(start, std::ptr::null_mut()))?;
    }
    launch()?;
    unsafe {
        check(cuda::cudaEventRecord(stop, std::ptr::null_mut()))?;
        check(cuda::cudaEventSynchronize(stop))?;
    }
    let mut ms = 0.0f32;
    unsafe { check(cuda::cudaEventElapsedTime(&mut ms, start, stop))?; }
    unsafe {
        let _ = cuda::cudaEventDestroy(start);
        let _ = cuda::cudaEventDestroy(stop);
    }
    Ok(ms)
}

/// The tensor-core kernels are compiled for the engine's real shape only
/// (n_q 64, d_qk 192, d_v 128, n_kv 4 or 8). The tiny golden shape falls back to
/// the naive f64 kernels (the reference/bisect path).
fn use_tc(g: &M26Geom) -> bool {
    g.n_q == 64 && g.d_qk == 192 && g.d_v == 128 && (g.n_kv == 4 || g.n_kv == 8)
}

/// One-time P1 (prefill tensor-core) config: set the dynamic shared-memory size
/// + carveout before the first launch. Idempotent; the result is cached.
fn prefill_tc_config() -> Result<(), String> {
    use std::sync::OnceLock;
    static DONE: OnceLock<Result<(), String>> = OnceLock::new();
    DONE
        .get_or_init(|| {
            let mut reg = 0i32;
            let mut ctas = 0i32;
            let rc = unsafe { ffi::m26_attn_prefill_tc_config_split(&mut reg, &mut ctas) };
            if rc == cuda::SUCCESS {
                Ok(())
            } else {
                Err(cuda::error_string(rc))
            }
        })
        .clone()
}

/// Split the fused QKV device plane `[T, q_rows + k_rows + v_rows]` into separate
/// `q`/`k`/`v` device planes (strided 2-D copies). Returns `(d_q, d_k, d_v)`.
pub fn split_qkv_device(
    d_qkv: &DeviceBuffer,
    t: usize,
    q_rows: usize,
    k_rows: usize,
    v_rows: usize,
) -> Result<(DeviceBuffer, DeviceBuffer, DeviceBuffer), String> {
    let total = q_rows + k_rows + v_rows;
    let d_q = DeviceBuffer::alloc(t * q_rows * 4).map_err(cuda::error_string)?;
    let d_k = DeviceBuffer::alloc(t * k_rows * 4).map_err(cuda::error_string)?;
    let d_v = DeviceBuffer::alloc(t * v_rows * 4).map_err(cuda::error_string)?;
    let copy_plane = |dst: &DeviceBuffer, src_off: usize, rows: usize| -> Result<(), String> {
        let rc = unsafe {
            cuda::cudaMemcpy2D(
                dst.as_ptr(),
                rows * 4,
                (d_qkv.as_ptr() as *const u8).add(src_off * 4) as *const core::ffi::c_void,
                total * 4,
                rows * 4,
                t,
                cuda::D2D,
            )
        };
        if rc == cuda::SUCCESS {
            Ok(())
        } else {
            Err(cuda::error_string(rc))
        }
    };
    copy_plane(&d_q, 0, q_rows)?;
    copy_plane(&d_k, q_rows, k_rows)?;
    copy_plane(&d_v, q_rows + k_rows, v_rows)?;
    Ok((d_q, d_k, d_v))
}

/// Apply RoPE on the device (`m26_rope_apply`): `d_x` `[T, h, d]` f32 → `d_y`.
pub fn rope_apply_device(
    theta: f64,
    partial_rotary_factor: f64,
    d_x: &DeviceBuffer,
    d_pos: &DeviceBuffer,
    t: i32,
    h: i32,
    d: i32,
) -> Result<DeviceBuffer, String> {
    let d_y = DeviceBuffer::alloc(t as usize * h as usize * d as usize * 4)
        .map_err(cuda::error_string)?;
    let rc = unsafe {
        ffi::m26_rope_apply(
            theta,
            partial_rotary_factor,
            d_x.as_ptr() as *const f32,
            d_y.as_ptr() as *mut f32,
            d_pos.as_ptr() as *const i64,
            t,
            h,
            d,
            0, // naive: correct path
            core::ptr::null_mut(),
        )
    };
    check(rc)?;
    Ok(d_y)
}

/// Encode K/V to FP8 on the device (`m26_kv_store_fp8`, unit scale, T18 V prescale
/// applied inside). `d_k_raw`/`d_v_raw` are `[n_tok, n_kv, d_qk/d_v]` f32; returns
/// the FP8 `d_k_codes`/`d_v_codes`.
pub fn kv_store_fp8_device(
    g: &M26Geom,
    d_k_raw: &DeviceBuffer,
    d_v_raw: &DeviceBuffer,
    n_tok: i32,
) -> Result<(DeviceBuffer, DeviceBuffer), String> {
    let n_kv = g.n_kv as usize;
    let d_k_codes = DeviceBuffer::alloc(n_tok as usize * n_kv * g.d_qk as usize)
        .map_err(cuda::error_string)?;
    let d_v_codes = DeviceBuffer::alloc(n_tok as usize * n_kv * g.d_v as usize)
        .map_err(cuda::error_string)?;
    let d_clip = DeviceBuffer::alloc(8).map_err(cuda::error_string)?;
    let rc = unsafe {
        ffi::m26_kv_store_fp8(
            g,
            d_k_raw.as_ptr() as *const f32,
            d_v_raw.as_ptr() as *const f32,
            n_tok,
            1, // unit scale
            0, // naive: correct path
            d_k_codes.as_ptr() as *mut u8,
            core::ptr::null_mut(), // k_scales: unit scale
            d_v_codes.as_ptr() as *mut u8,
            core::ptr::null_mut(), // v_scales: unit scale
            d_clip.as_ptr() as *mut u64,
            core::ptr::null_mut(),
        )
    };
    check(rc)?;
    Ok((d_k_codes, d_v_codes))
}

/// Upload the i64 position vector to the device (shared by RoPE + attention).
pub fn pos_to_device(pos: &[i64]) -> Result<DeviceBuffer, String> {
    let d = DeviceBuffer::alloc(pos.len() * 8).map_err(cuda::error_string)?;
    d.upload(i64_bytes(pos)).map_err(cuda::error_string)?;
    Ok(d)
}

/// Encode the hidden rows to E4M3 on the device (`m26_quantize_hidden_fp8`):
/// `payload[i] = e4m3_encode(x[i] * scale_inv[i >> 5])`, where `scale_inv` is the
/// per-K32-block power-of-two inverse scale (2^(127-s), exact). Bit-identical to
/// the CPU `quantize_hidden`. Returns the host payload bytes.
pub fn quantize_hidden_device(hidden: &[f32], scale_inv: &[f32]) -> Result<Vec<u8>, String> {
    let n = hidden.len();
    if scale_inv.len() != n / 32 {
        return Err(format!(
            "quantize: {} scale_inv for {} hidden elems (want {})",
            scale_inv.len(),
            n,
            n / 32
        ));
    }
    let d_x = DeviceBuffer::alloc(n * 4).map_err(cuda::error_string)?;
    let d_scale = DeviceBuffer::alloc(scale_inv.len() * 4).map_err(cuda::error_string)?;
    let d_payload = DeviceBuffer::alloc(n).map_err(cuda::error_string)?;
    d_x.upload(f32_bytes(hidden)).map_err(cuda::error_string)?;
    d_scale.upload(f32_bytes(scale_inv)).map_err(cuda::error_string)?;
    let rc = unsafe {
        ffi::m26_quantize_hidden_fp8(
            d_x.as_ptr() as *const f32,
            d_scale.as_ptr() as *const f32,
            d_payload.as_ptr() as *mut u8,
            n as i64,
            core::ptr::null_mut(),
        )
    };
    check(rc)?;
    let mut payload = vec![0u8; n];
    d_payload.download(&mut payload).map_err(cuda::error_string)?;
    Ok(payload)
}

/// C3/P8 decode: split-KV over FP8 unit-scale KV, then reduce. Returns
/// `[T, n_q, d_v]` f32.
///
/// `k_codes`/`v_codes` are the cached FP8 codes `[S, n_kv*d_qk]` / `[S, n_kv*d_v]`
/// (V already carries `v_scale`, T18). `n_splits` is the flash-decoding split
/// count. No page table (flat rows) — the paged variant is a follow-up.
#[allow(clippy::too_many_arguments)]
pub fn decode_attention(
    g: &M26Geom,
    q: &[f32],
    k_codes: &[u8],
    v_codes: &[u8],
    q_pos: &[i64],
    k_pos: &[i64],
    n_splits: i32,
    sink: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    let (d_out, out_elems) = decode_attention_core(g, q, k_codes, v_codes, q_pos, k_pos, n_splits, sink)?;
    let mut out = vec![0f32; out_elems];
    d_out.download(f32_bytes_mut(&mut out)).map_err(cuda::error_string)?;
    Ok(out)
}

/// [`decode_attention`] without the output download — the caller keeps the
/// result on the device (Q/O-on-device, I5-R17 step 2).
pub fn decode_attention_device(
    g: &M26Geom,
    q: &[f32],
    k_codes: &[u8],
    v_codes: &[u8],
    q_pos: &[i64],
    k_pos: &[i64],
    n_splits: i32,
    sink: Option<&[f32]>,
) -> Result<DeviceBuffer, String> {
    decode_attention_core(g, q, k_codes, v_codes, q_pos, k_pos, n_splits, sink).map(|(d, _)| d)
}

fn decode_attention_core(
    g: &M26Geom,
    q: &[f32],
    k_codes: &[u8],
    v_codes: &[u8],
    q_pos: &[i64],
    k_pos: &[i64],
    n_splits: i32,
    sink: Option<&[f32]>,
) -> Result<(DeviceBuffer, usize), String> {
    let t = q_pos.len() as i32;
    let s = k_pos.len() as i32;
    if q.len() != t as usize * g.n_q as usize * g.d_qk as usize {
        return Err(format!("decode: q has {} elems, expected {t} x {} x {}", q.len(), g.n_q, g.d_qk));
    }
    if k_codes.len() != s as usize * g.n_kv as usize * g.d_qk as usize {
        return Err(format!("decode: k_codes has {} bytes, expected {s} x {} x {}", k_codes.len(), g.n_kv, g.d_qk));
    }
    if v_codes.len() != s as usize * g.n_kv as usize * g.d_v as usize {
        return Err(format!("decode: v_codes has {} bytes, expected {s} x {} x {}", v_codes.len(), g.n_kv, g.d_v));
    }
    if let Some(sink) = sink {
        if sink.len() != g.n_q as usize {
            return Err(format!("decode: sink has {} elems, expected {}", sink.len(), g.n_q));
        }
    }

    let out_elems = t as usize * g.n_q as usize * g.d_v as usize;
    let tc = use_tc(g);

    let d_q = DeviceBuffer::alloc(q.len() * 4).map_err(cuda::error_string)?;
    let d_k = DeviceBuffer::alloc(k_codes.len()).map_err(cuda::error_string)?;
    let d_v = DeviceBuffer::alloc(v_codes.len()).map_err(cuda::error_string)?;
    let d_qpos = DeviceBuffer::alloc(q_pos.len() * 8).map_err(cuda::error_string)?;
    let d_kpos = DeviceBuffer::alloc(k_pos.len() * 8).map_err(cuda::error_string)?;
    // Partial layout differs by class: f64 `(t,h,split)(2+d_v)` (naive) vs f32
    // `m[l]o` planes of `count*130` (tensor-core).
    let partials_elems = if tc {
        let count = t as usize * 64usize * n_splits as usize;
        count * 130
    } else {
        t as usize * g.n_q as usize * n_splits as usize * (2 + g.d_v as usize)
    };
    let partial_bytes = if tc { partials_elems * 4 } else { partials_elems * 8 };
    let d_partials = DeviceBuffer::alloc(partial_bytes).map_err(cuda::error_string)?;
    let d_sink = DeviceBuffer::alloc(g.n_q as usize * 4).map_err(cuda::error_string)?;
    let d_out = DeviceBuffer::alloc(out_elems * 4).map_err(cuda::error_string)?;

    let t_up = std::time::Instant::now();
    d_q.upload(f32_bytes(q)).map_err(cuda::error_string)?;
    d_k.upload(k_codes).map_err(cuda::error_string)?;
    d_v.upload(v_codes).map_err(cuda::error_string)?;
    d_qpos.upload(i64_bytes(q_pos)).map_err(cuda::error_string)?;
    d_kpos.upload(i64_bytes(k_pos)).map_err(cuda::error_string)?;
    if let Some(sink) = sink {
        d_sink.upload(f32_bytes(sink)).map_err(cuda::error_string)?;
    }
    serve_profile("attn_upload", t_up.elapsed().as_secs_f32() * 1e3);

    // SAFETY: every device pointer is a valid cudaMalloc allocation of the size
    // derived above; the buffers stay live through the launch (same thread).
    let sink_ptr = if sink.is_some() { d_sink.as_ptr() as *const f32 } else { core::ptr::null() };
    let kernel_ms = time_kernel_ms(|| {
        if tc {
            // Tensor-core decode (C3/P8): f32 partials, unit-scale KV, then reduce_tc.
            let rc = unsafe {
                ffi::m26_attn_decode_splitkv_fp8_tc(
                    g,
                    d_q.as_ptr() as *const f32,
                    d_k.as_ptr() as *const u8,
                    d_v.as_ptr() as *const u8,
                    core::ptr::null(), // page_table: flat rows
                    0,                 // page_tokens: unused when page_table is null
                    d_qpos.as_ptr() as *const i64,
                    d_kpos.as_ptr() as *const i64,
                    t,
                    s,
                    n_splits,
                    0, // naive: correct path
                    d_partials.as_ptr() as *mut f32,
                    core::ptr::null_mut(),
                )
            };
            check(rc)?;
            let rc = unsafe {
                ffi::m26_attn_reduce_tc(
                    g,
                    d_partials.as_ptr() as *const f32,
                    sink_ptr,
                    t,
                    n_splits,
                    0,
                    d_out.as_ptr() as *mut f32,
                    core::ptr::null_mut(),
                )
            };
            check(rc)
        } else {
            let rc = unsafe {
                ffi::m26_attn_decode_splitkv_fp8(
                    g,
                    d_q.as_ptr() as *const f32,
                    d_k.as_ptr() as *const u8,
                    core::ptr::null(), // k_scales: unit scale
                    d_v.as_ptr() as *const u8,
                    core::ptr::null(), // v_scales: unit scale
                    core::ptr::null(), // page_table: flat rows
                    0,                 // page_tokens: unused when page_table is null
                    d_qpos.as_ptr() as *const i64,
                    d_kpos.as_ptr() as *const i64,
                    t,
                    s,
                    n_splits,
                    0, // naive: correct path
                    d_partials.as_ptr() as *mut f64,
                    core::ptr::null_mut(), // default stream
                )
            };
            check(rc)?;
            let rc = unsafe {
                ffi::m26_attn_reduce(
                    g,
                    d_partials.as_ptr() as *const f64,
                    sink_ptr,
                    t,
                    n_splits,
                    0,
                    d_out.as_ptr() as *mut f32,
                    core::ptr::null_mut(),
                )
            };
            check(rc)
        }
    })?;
    serve_profile("attn_kernel", kernel_ms);

    Ok((d_out, out_elems))
}

/// P1 chunked prefill over FP8 unit-scale KV. Returns `[T, n_q, d_v]` f32.
/// `chunk_rows` is the per-chunk fold width (online softmax with rescale).
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention(
    g: &M26Geom,
    q: &[f32],
    k_codes: &[u8],
    v_codes: &[u8],
    q_pos: &[i64],
    k_pos: &[i64],
    chunk_rows: i32,
    sink: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    let (d_out, out_elems) =
        prefill_attention_core(g, q, k_codes, v_codes, q_pos, k_pos, chunk_rows, sink)?;
    let mut out = vec![0f32; out_elems];
    d_out.download(f32_bytes_mut(&mut out)).map_err(cuda::error_string)?;
    Ok(out)
}

/// [`prefill_attention`] without the output download (Q/O-on-device).
pub fn prefill_attention_device(
    g: &M26Geom,
    q: &[f32],
    k_codes: &[u8],
    v_codes: &[u8],
    q_pos: &[i64],
    k_pos: &[i64],
    chunk_rows: i32,
    sink: Option<&[f32]>,
) -> Result<DeviceBuffer, String> {
    prefill_attention_core(g, q, k_codes, v_codes, q_pos, k_pos, chunk_rows, sink).map(|(d, _)| d)
}

fn prefill_attention_core(
    g: &M26Geom,
    q: &[f32],
    k_codes: &[u8],
    v_codes: &[u8],
    q_pos: &[i64],
    k_pos: &[i64],
    chunk_rows: i32,
    sink: Option<&[f32]>,
) -> Result<(DeviceBuffer, usize), String> {
    let t = q_pos.len() as i32;
    let s = k_pos.len() as i32;
    if q.len() != t as usize * g.n_q as usize * g.d_qk as usize {
        return Err(format!("prefill: q has {} elems, expected {t} x {} x {}", q.len(), g.n_q, g.d_qk));
    }
    if k_codes.len() != s as usize * g.n_kv as usize * g.d_qk as usize {
        return Err(format!("prefill: k_codes has {} bytes, expected {s} x {} x {}", k_codes.len(), g.n_kv, g.d_qk));
    }
    if v_codes.len() != s as usize * g.n_kv as usize * g.d_v as usize {
        return Err(format!("prefill: v_codes has {} bytes, expected {s} x {} x {}", v_codes.len(), g.n_kv, g.d_v));
    }
    if let Some(sink) = sink {
        if sink.len() != g.n_q as usize {
            return Err(format!("prefill: sink has {} elems, expected {}", sink.len(), g.n_q));
        }
    }

    let out_elems = t as usize * g.n_q as usize * g.d_v as usize;
    let d_q = DeviceBuffer::alloc(q.len() * 4).map_err(cuda::error_string)?;
    let d_k = DeviceBuffer::alloc(k_codes.len()).map_err(cuda::error_string)?;
    let d_v = DeviceBuffer::alloc(v_codes.len()).map_err(cuda::error_string)?;
    let d_qpos = DeviceBuffer::alloc(q_pos.len() * 8).map_err(cuda::error_string)?;
    let d_kpos = DeviceBuffer::alloc(k_pos.len() * 8).map_err(cuda::error_string)?;
    let d_sink = DeviceBuffer::alloc(g.n_q as usize * 4).map_err(cuda::error_string)?;
    let d_out = DeviceBuffer::alloc(out_elems * 4).map_err(cuda::error_string)?;

    let t_up = std::time::Instant::now();
    d_q.upload(f32_bytes(q)).map_err(cuda::error_string)?;
    d_k.upload(k_codes).map_err(cuda::error_string)?;
    d_v.upload(v_codes).map_err(cuda::error_string)?;
    d_qpos.upload(i64_bytes(q_pos)).map_err(cuda::error_string)?;
    d_kpos.upload(i64_bytes(k_pos)).map_err(cuda::error_string)?;
    if let Some(sink) = sink {
        d_sink.upload(f32_bytes(sink)).map_err(cuda::error_string)?;
    }
    serve_profile("attn_upload", t_up.elapsed().as_secs_f32() * 1e3);

    let sink_ptr = if sink.is_some() { d_sink.as_ptr() as *const f32 } else { core::ptr::null() };
    // SAFETY: every device pointer is a valid cudaMalloc allocation of the size
    // derived above; the buffers stay live through the launch.
    let kernel_ms = time_kernel_ms(|| {
        if use_tc(g) {
            // Tensor-core prefill (P1): unit-scale KV, direct out (no reduce).
            prefill_tc_config()?;
            let rc = unsafe {
                ffi::m26_attn_prefill_fp8_tc_split(
                    g,
                    d_q.as_ptr() as *const f32,
                    d_k.as_ptr() as *const u8,
                    d_v.as_ptr() as *const u8,
                    core::ptr::null(), // page_table: flat rows
                    0,                 // page_tokens: unused when page_table is null
                    d_qpos.as_ptr() as *const i64,
                    d_kpos.as_ptr() as *const i64,
                    t,
                    s,
                    0, // naive: correct path
                    sink_ptr,
                    d_out.as_ptr() as *mut f32,
                    core::ptr::null_mut(),
                )
            };
            check(rc)
        } else {
            let rc = unsafe {
                ffi::m26_attn_prefill_fp8(
                    g,
                    d_q.as_ptr() as *const f32,
                    d_k.as_ptr() as *const u8,
                    core::ptr::null(), // k_scales: unit scale
                    d_v.as_ptr() as *const u8,
                    core::ptr::null(), // v_scales: unit scale
                    core::ptr::null(), // page_table: flat rows
                    0,                 // page_tokens: unused when page_table is null
                    d_qpos.as_ptr() as *const i64,
                    d_kpos.as_ptr() as *const i64,
                    t,
                    s,
                    chunk_rows,
                    0,
                    sink_ptr,
                    d_out.as_ptr() as *mut f32,
                    core::ptr::null_mut(),
                )
            };
            check(rc)
        }
    })?;
    serve_profile("attn_kernel", kernel_ms);

    Ok((d_out, out_elems))
}

/// Prefill attention over device-resident Q/K/V (the Q-on-device + device-KV
/// path): `d_q` `[T, n_q, d_qk]` f32, `d_k_codes`/`d_v_codes` FP8, positions on
/// device. Only the sink is uploaded; returns the device output.
pub fn prefill_attention_dev(
    g: &M26Geom,
    d_q: &DeviceBuffer,
    d_k_codes: &DeviceBuffer,
    d_v_codes: &DeviceBuffer,
    d_qpos: &DeviceBuffer,
    d_kpos: &DeviceBuffer,
    t: i32,
    s: i32,
    sink: Option<&[f32]>,
) -> Result<DeviceBuffer, String> {
    let out_elems = t as usize * g.n_q as usize * g.d_v as usize;
    let d_sink = DeviceBuffer::alloc(g.n_q as usize * 4).map_err(cuda::error_string)?;
    let d_out = DeviceBuffer::alloc(out_elems * 4).map_err(cuda::error_string)?;
    if let Some(sink) = sink {
        d_sink.upload(f32_bytes(sink)).map_err(cuda::error_string)?;
    }
    let sink_ptr = if sink.is_some() { d_sink.as_ptr() as *const f32 } else { core::ptr::null() };
    if use_tc(g) {
        prefill_tc_config()?;
        let rc = unsafe {
            ffi::m26_attn_prefill_fp8_tc_split(
                g,
                d_q.as_ptr() as *const f32,
                d_k_codes.as_ptr() as *const u8,
                d_v_codes.as_ptr() as *const u8,
                core::ptr::null(),
                0,
                d_qpos.as_ptr() as *const i64,
                d_kpos.as_ptr() as *const i64,
                t,
                s,
                0,
                sink_ptr,
                d_out.as_ptr() as *mut f32,
                core::ptr::null_mut(),
            )
        };
        check(rc)?;
    } else {
        let rc = unsafe {
            ffi::m26_attn_prefill_fp8(
                g,
                d_q.as_ptr() as *const f32,
                d_k_codes.as_ptr() as *const u8,
                core::ptr::null(),
                d_v_codes.as_ptr() as *const u8,
                core::ptr::null(),
                core::ptr::null(),
                0,
                d_qpos.as_ptr() as *const i64,
                d_kpos.as_ptr() as *const i64,
                t,
                s,
                64,
                0,
                sink_ptr,
                d_out.as_ptr() as *mut f32,
                core::ptr::null_mut(),
            )
        };
        check(rc)?;
    }
    Ok(d_out)
}

fn f32_bytes_mut(x: &mut [f32]) -> &mut [u8] {
    unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut u8, x.len() * 4) }
}

fn check(rc: CudaError) -> Result<(), String> {
    if rc == cuda::SUCCESS {
        Ok(())
    } else {
        Err(cuda::error_string(rc))
    }
}
