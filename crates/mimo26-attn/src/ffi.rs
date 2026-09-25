//! Rust FFI surface for the attention CUDA ABI.
//!
//! Mirrors `crates/mimo26-attn/kernels/include/mimo26_attn_kernels.h`. No CUDA
//! header is imported: `cudaError_t` is an `i32` (`cudaSuccess == 0`) and
//! `cudaStream_t` is an opaque pointer, so these declarations type-check and
//! link on the CPU merge gate with **no nvcc**. The CUDA symbols resolve only
//! when the serving binary links the compiled kernel object (a separate
//! feature-gated build, like `mimo26-spark`'s `ffi.rs`).
//!
//! Declarations, not safe wrappers — the safe wrapper that enforces the header's
//! safety contract (caller-owned buffers live through the stream, the sink
//! enters once at reduce, the AOT gate before any launch) lands with the
//! serving daemon.

use core::ffi::{c_double, c_int, c_void};

/// `cudaError_t` — an enum; `cudaSuccess == 0`.
pub type CudaError = c_int;

/// `cudaStream_t` (`CUstream`) — an opaque pointer on every supported build.
pub type CudaStream = *mut c_void;

/// `m26_geom` — the geometry block the kernels read. Field order and types are
/// the C ABI; the layout test below pins `size_of == 32` and every offset.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct M26Geom {
    pub n_q: i32,
    pub n_kv: i32,
    pub d_qk: i32,
    pub d_v: i32,
    /// `<= 0` => none (GA); otherwise the SWA window.
    pub window: i64,
    /// 1.0 on cached V (the STORE applies 0.707, T18).
    pub value_scale: c_double,
}

unsafe extern "C" {
    // ---- split-KV decode (flash decoding): per (t, h, split) partials --------
    pub fn m26_attn_decode_splitkv_f32(
        g: *const M26Geom,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        q_pos: *const i64,
        k_pos: *const i64,
        t: i32,
        s: i32,
        n_splits: i32,
        naive: u32,
        partials: *mut f64,
        stream: CudaStream,
    ) -> CudaError;

    pub fn m26_attn_decode_splitkv_fp8(
        g: *const M26Geom,
        q: *const f32,
        k_codes: *const u8,
        k_scales: *const f32,
        v_codes: *const u8,
        v_scales: *const f32,
        page_table: *const i32,
        page_tokens: i32,
        q_pos: *const i64,
        k_pos: *const i64,
        t: i32,
        s: i32,
        n_splits: i32,
        naive: u32,
        partials: *mut f64,
        stream: CudaStream,
    ) -> CudaError;

    pub fn m26_attn_decode_splitkv_fp8_tc(
        g: *const M26Geom,
        q: *const f32,
        k_codes: *const u8,
        v_codes: *const u8,
        page_table: *const i32,
        page_tokens: i32,
        q_pos: *const i64,
        k_pos: *const i64,
        t: i32,
        s: i32,
        n_splits: i32,
        naive: u32,
        partials: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    pub fn m26_attn_decode_pipe_config(warps: i32, registers: *mut i32, active_ctas: *mut i32) -> CudaError;
    pub fn m26_attn_decode_pipe_config_bf16q(warps: i32, registers: *mut i32, active_ctas: *mut i32) -> CudaError;

    pub fn m26_attn_decode_splitkv_fp8_pipe(
        g: *const M26Geom,
        q: *const f32,
        k_codes: *const u8,
        v_codes: *const u8,
        page_table: *const i32,
        page_tokens: i32,
        q_pos: *const i64,
        k_pos: *const i64,
        t: i32,
        s: i32,
        n_splits: i32,
        naive: u32,
        warps: i32,
        partials: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    pub fn m26_attn_decode_splitkv_fp8_pipe_bf16q(
        g: *const M26Geom,
        q: *const f32,
        k_codes: *const u8,
        v_codes: *const u8,
        page_table: *const i32,
        page_tokens: i32,
        q_pos: *const i64,
        k_pos: *const i64,
        t: i32,
        s: i32,
        n_splits: i32,
        naive: u32,
        warps: i32,
        partials: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    pub fn m26_attn_decode_c1_config(warps: i32, registers: *mut i32, active_ctas: *mut i32) -> CudaError;
    pub fn m26_attn_decode_c1_config_bf16q(warps: i32, registers: *mut i32, active_ctas: *mut i32) -> CudaError;

    pub fn m26_attn_decode_splitkv_fp8_c1(
        g: *const M26Geom,
        q: *const f32,
        k_codes: *const u8,
        v_codes: *const u8,
        page_table: *const i32,
        page_tokens: i32,
        q_pos: *const i64,
        k_pos: *const i64,
        t: i32,
        s: i32,
        n_splits: i32,
        naive: u32,
        warps: i32,
        partials: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    pub fn m26_attn_decode_splitkv_fp8_c1_bf16q(
        g: *const M26Geom,
        q: *const f32,
        k_codes: *const u8,
        v_codes: *const u8,
        page_table: *const i32,
        page_tokens: i32,
        q_pos: *const i64,
        k_pos: *const i64,
        t: i32,
        s: i32,
        n_splits: i32,
        naive: u32,
        warps: i32,
        partials: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    // ---- reduce (P8): combine splits; the sink enters ONCE per query --------
    pub fn m26_attn_reduce(
        g: *const M26Geom,
        partials: *const f64,
        sink: *const f32,
        t: i32,
        n_splits: i32,
        naive: u32,
        out: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    pub fn m26_attn_reduce_tc(
        g: *const M26Geom,
        partials: *const f32,
        sink: *const f32,
        t: i32,
        n_splits: i32,
        naive: u32,
        out: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    // ---- chunked prefill over pages (P1, online softmax with rescale) -------
    pub fn m26_attn_prefill_f32(
        g: *const M26Geom,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        q_pos: *const i64,
        k_pos: *const i64,
        t: i32,
        s: i32,
        chunk_rows: i32,
        naive: u32,
        sink: *const f32,
        out: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    pub fn m26_attn_prefill_fp8(
        g: *const M26Geom,
        q: *const f32,
        k_codes: *const u8,
        k_scales: *const f32,
        v_codes: *const u8,
        v_scales: *const f32,
        page_table: *const i32,
        page_tokens: i32,
        q_pos: *const i64,
        k_pos: *const i64,
        t: i32,
        s: i32,
        chunk_rows: i32,
        naive: u32,
        sink: *const f32,
        out: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    pub fn m26_attn_prefill_tc_config(registers: *mut i32, active_ctas: *mut i32) -> CudaError;
    pub fn m26_attn_prefill_tc_config_bf16q(registers: *mut i32, active_ctas: *mut i32) -> CudaError;
    pub fn m26_attn_prefill_tc_config_split(registers: *mut i32, active_ctas: *mut i32) -> CudaError;

    pub fn m26_attn_prefill_fp8_tc(
        g: *const M26Geom,
        q: *const f32,
        kc: *const u8,
        vc: *const u8,
        pages: *const i32,
        page_tokens: i32,
        qpos: *const i64,
        kpos: *const i64,
        t: i32,
        s: i32,
        naive: u32,
        sink: *const f32,
        out: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    pub fn m26_attn_prefill_fp8_tc_bf16q(
        g: *const M26Geom,
        q: *const f32,
        kc: *const u8,
        vc: *const u8,
        pages: *const i32,
        page_tokens: i32,
        qpos: *const i64,
        kpos: *const i64,
        t: i32,
        s: i32,
        naive: u32,
        sink: *const f32,
        out: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    pub fn m26_attn_prefill_fp8_tc_split(
        g: *const M26Geom,
        q: *const f32,
        kc: *const u8,
        vc: *const u8,
        pages: *const i32,
        page_tokens: i32,
        qpos: *const i64,
        kpos: *const i64,
        t: i32,
        s: i32,
        naive: u32,
        sink: *const f32,
        out: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    /// P2 serving prefill (`attn_prefill_fa.cu`): queries are KV rows
    /// `q_row0..q_row0+t`, KV positions row-contiguous; `naive` must be 0.
    pub fn m26_attn_prefill_fp8_fa(
        g: *const M26Geom,
        q: *const f32,
        kc: *const u8,
        vc: *const u8,
        t: i32,
        s: i32,
        q_row0: i32,
        naive: u32,
        sink: *const f32,
        out: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    // ---- RoPE (T19/T7): FP32 on-the-fly, partial 64/192, dual theta ---------
    pub fn m26_rope_apply(
        theta: c_double,
        partial_rotary_factor: c_double,
        x: *const f32,
        y: *mut f32,
        pos: *const i64,
        t: i32,
        h: i32,
        d: i32,
        naive: u32,
        stream: CudaStream,
    ) -> CudaError;

    // ---- FP8 KV store/decode (T18/T20 + amax clip gate) --------------------
    pub fn m26_kv_store_fp8(
        g: *const M26Geom,
        k_raw: *const f32,
        v_raw: *const f32,
        n_tok: i32,
        unit_scale: i32,
        naive: u32,
        k_codes: *mut u8,
        k_scales: *mut f32,
        v_codes: *mut u8,
        v_scales: *mut f32,
        clip_count: *mut u64,
        stream: CudaStream,
    ) -> CudaError;

    pub fn m26_kv_decode_fp8(
        g: *const M26Geom,
        k_codes: *const u8,
        k_scales: *const f32,
        v_codes: *const u8,
        v_scales: *const f32,
        n_tok: i32,
        naive: u32,
        k_out: *mut f32,
        v_out: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    // ---- hidden quantize (MoE wire-out Fp8E4m3Ue8m0K32) ----------------------
    pub fn m26_quantize_hidden_fp8(
        x: *const f32,
        scale_inv: *const f32,
        payload: *mut u8,
        n_elem: i64,
        stream: CudaStream,
    ) -> CudaError;
}
