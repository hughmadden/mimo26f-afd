/* mimo26-attn kernel ABI — handwritten CUDA (AGENTS.md §3 "Write").
 *
 * Correctness-first: one thread per (query, Q head[, split]) partial with a
 * sequential key loop over f64 (m, l, o) accumulators. At the parity sizes
 * (decode T=1: n_q × n_splits partials; prefill T<=512) this is well inside
 * budget on the dev host's RTX 4090 (sm_89) and the mapping to the CPU twin
 * (`mimo26_attn::attn`) is line-for-line obvious — the point of this pass.
 * A tiled flash-style kernel is follow-up work AFTER every trap is pinned.
 *
 * NOT compiled or run by the attn-writer (LAW: no nvcc/cargo by hand) — the
 * captain fires `tests/gpu/run_gpu_parity.sh` (a `scripts/dev.sh test attn`
 * cell) on the dev host only, never the 5090 / the Sparks.
 *
 * Invariants the CPU suite pins (and these kernels must keep):
 *   T18  cached V carries `v_scale` BEFORE quantization; attention reads V as
 *        stored (no rescale on read).
 *   T19  RoPE angles are FP32 `float(pos) × inv_freq` computed on the fly
 *        (no cos/sin table), partial 64/192, dual θ (1e7 GA / 1e4 SWA).
 *   T20  FP8 scale planes are per token × head with K and V SEPARATE
 *        (block-128 cannot divide the 192-dim K).
 *   T3/T9 causal `k <= q` + window `q − k < window` on absolute positions; GA
 *        is NEVER windowed.
 *   T6   sink bias is per Q head [n_q], SWA-only, ZERO value, and enters the
 *        denominator ONCE (at the reduce), never once per split (T-split-sink).
 *   c2   logit scale is 1/sqrt(d_qk) — never d_v (QK 192 / V 128).
 *   GA   KV lives in 256-token pages through a logical→slot page table.
 */

#pragma once

#include <stdint.h>
#if defined(__CUDACC__) || defined(M26_NO_CUDA_RUNTIME)
typedef void* m26_stream_t;
#else
#include <cuda_runtime.h>
typedef cudaStream_t m26_stream_t;
typedef int cudaError_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define M26_MAX_D_QK 192
#define M26_MAX_D_V 128

/* geometry (real dims: n_q 64, n_kv 4 GA / 8 SWA, d_qk 192, d_v 128) */
typedef struct {
  int32_t n_q;
  int32_t n_kv;
  int32_t d_qk;
  int32_t d_v;
  int64_t window;      /* <= 0 => none (GA) */
  double value_scale;  /* 1.0 on cached V (T18); the STORE applies 0.707 */
} m26_geom;

/* naive-path flags (MIMO26_SPIKE_NAIVE mirror — negative-test switches). */
enum {
  M26_NAIVE_SCALE_BY_DV = 1u << 0,
  M26_NAIVE_SINK_ON_GA = 1u << 1,
  M26_NAIVE_GA_WINDOWED = 1u << 2,
  M26_NAIVE_SINK_PER_KV = 1u << 3,
  M26_NAIVE_POS_ZEROED = 1u << 4,
  M26_NAIVE_VSCALE_AFTER_STORE = 1u << 5,
  M26_NAIVE_VSCALE_ON_READ = 1u << 6,
  M26_NAIVE_SINK_PER_SPLIT = 1u << 7,
  M26_NAIVE_NO_RUNNING_RESCALE = 1u << 8,
  M26_NAIVE_BLOCK128_SHARED_SCALES = 1u << 9,
  M26_NAIVE_SILENT_CLAMP = 1u << 10,
  M26_NAIVE_ROPE_TRUNC_TABLE = 1u << 11,
  M26_NAIVE_ROPE_FULL_WIDTH = 1u << 12,
  M26_NAIVE_ROPE_SINGLE_THETA = 1u << 13,
  /* CUDA-only experimental traps; not Rust NaiveBits bit positions. */
  M26_NAIVE_TC_DROP_Q_LOW = 1u << 14,
  M26_NAIVE_TC_DROP_P_LOW = 1u << 15,
  M26_NAIVE_TC_IGNORE_PAGES = 1u << 16,
  M26_NAIVE_TC_DROP_Q_TAIL = 1u << 17,
  /* Parity harness only: select Q3 instead of the requested BF16-Q lattice. */
  M26_NAIVE_TC_IGNORE_Q_ROUND = 1u << 18
};

/* ---- split-KV decode (flash decoding): per (t, h, split) partials ---------
 * partials layout: [t][h][split][0]=m, [1]=l, [2..2+d_v)=o  (f64).
 * f32 KV rows (CPU-twin cache form): k [S, n_kv, d_qk], v [S, n_kv, d_v].
 */
cudaError_t m26_attn_decode_splitkv_f32(
    const m26_geom* g, const float* q, const float* k, const float* v,
    const int64_t* q_pos, const int64_t* k_pos,
    int32_t T, int32_t S, int32_t n_splits, uint32_t naive,
    double* partials, m26_stream_t stream);

/* FP8 KV variant. Codes are row-major [S, n_kv*width] u8 (flat) or paged:
 * when page_table != NULL, logical row j lives at physical
 * (page_table[j / page_tokens]) * page_tokens + (j % page_tokens).
 * scales: [S, n_kv] f32, K and V SEPARATE planes (T20); NULL => unit scale.
 * v_codes carry CACHED V (v_scale applied before quantization — T18). */
cudaError_t m26_attn_decode_splitkv_fp8(
    const m26_geom* g, const float* q,
    const uint8_t* k_codes, const float* k_scales,
    const uint8_t* v_codes, const float* v_scales,
    const int32_t* page_table, int32_t page_tokens,
    const int64_t* q_pos, const int64_t* k_pos,
    int32_t T, int32_t S, int32_t n_splits, uint32_t naive,
    double* partials, m26_stream_t stream);

/* Experimental opt-in TC decode, not a replacement/default dispatch.
 * Only n_q=64, n_kv=4/8, QK192/V128 and UNIT FP8 KV. Q stays FP32.
 * Separate FP32 planes in one allocation: M[T*64*splits], L[same],
 * O[T*64*splits*128]. Each O row is aligned (no interleaved m/l header).
 * Required allocation: T*64*splits*130 floats. The baseline double buffer
 * must NEVER be passed to this ABI or its matching reduce by reinterpretation.
 * All device pointers belong to the caller and remain live through the stream.
 * Launch status only; callers must check asynchronous errors/readback too. */
cudaError_t m26_attn_decode_splitkv_fp8_tc(
    const m26_geom* g, const float* q, const uint8_t* k_codes,
    const uint8_t* v_codes, const int32_t* page_table, int32_t page_tokens,
    const int64_t* q_pos, const int64_t* k_pos,
    int32_t T, int32_t S, int32_t n_splits, uint32_t naive,
    float* partials, m26_stream_t stream);
/* C3 opt-in: configure once on the current device, outside timed regions.
 * 4/8 warps, 49536 dynamic shared bytes, same Q3/P2 partial/reduce ABI.
 * N4 amendment: one raw stage and phase-disjoint K / score-P-alpha storage.
 * KV base pointers must be 16-byte aligned. No implicit baseline fallback. */
cudaError_t m26_attn_decode_pipe_config(int32_t warps, int32_t* registers, int32_t* active_ctas);
cudaError_t m26_attn_decode_splitkv_fp8_pipe(
    const m26_geom* g, const float* q, const uint8_t* k_codes,
    const uint8_t* v_codes, const int32_t* page_table, int32_t page_tokens,
    const int64_t* q_pos, const int64_t* k_pos,
    int32_t T, int32_t S, int32_t n_splits, uint32_t naive, int32_t warps,
    float* partials, m26_stream_t stream);
/* Explicit opt-in native BF16-Q: q is post-RoPE FP32, rounded once RNE in
 * the CTA prologue. Q1/P2, FP32 m/l/O. Separate lattice-local reference only;
 * not an FP32-Q equivalence promise and not the serving default. */
cudaError_t m26_attn_decode_pipe_config_bf16q(int32_t warps, int32_t* registers, int32_t* active_ctas);
cudaError_t m26_attn_decode_splitkv_fp8_pipe_bf16q(
    const m26_geom* g, const float* q, const uint8_t* k_codes,
    const uint8_t* v_codes, const int32_t* page_table, int32_t page_tokens,
    const int64_t* q_pos, const int64_t* k_pos,
    int32_t T, int32_t S, int32_t n_splits, uint32_t naive, int32_t warps,
    float* partials, m26_stream_t stream);
/* Experimental C1, standalone opt-in (qualification scope in attn-lead packet):
 * eight warps only, M16/N16,
 * 44416 shared bytes, 2 producer / 2 QK / 4 PV warps. Same partial/reduce ABI.
 * Configuration requests shared carveout preference; actual residency needs
 * measurement. Explicit standalone opt-in only, never a silent C3 replacement. */
cudaError_t m26_attn_decode_c1_config(int32_t warps, int32_t* registers, int32_t* active_ctas);
cudaError_t m26_attn_decode_c1_config_bf16q(int32_t warps, int32_t* registers, int32_t* active_ctas);
cudaError_t m26_attn_decode_splitkv_fp8_c1(
    const m26_geom* g, const float* q, const uint8_t* k_codes,
    const uint8_t* v_codes, const int32_t* page_table, int32_t page_tokens,
    const int64_t* q_pos, const int64_t* k_pos,
    int32_t T, int32_t S, int32_t n_splits, uint32_t naive, int32_t warps,
    float* partials, m26_stream_t stream);
cudaError_t m26_attn_decode_splitkv_fp8_c1_bf16q(
    const m26_geom* g, const float* q, const uint8_t* k_codes,
    const uint8_t* v_codes, const int32_t* page_table, int32_t page_tokens,
    const int64_t* q_pos, const int64_t* k_pos,
    int32_t T, int32_t S, int32_t n_splits, uint32_t naive, int32_t warps,
    float* partials, m26_stream_t stream);
cudaError_t m26_attn_reduce_tc(
    const m26_geom* g, const float* partials, const float* sink,
    int32_t T, int32_t n_splits, uint32_t naive, float* out,
    m26_stream_t stream);

/* ---- reduce: combine splits; the sink enters ONCE per query -------------- */
cudaError_t m26_attn_reduce(
    const m26_geom* g, const double* partials,
    const float* sink /* [n_q] or NULL — SWA-only, per Q head (T6/c1) */,
    int32_t T, int32_t n_splits, uint32_t naive, float* out,
    m26_stream_t stream);

/* ---- chunked prefill over pages (online softmax with running rescale) ----
 * out [T, n_q, d_v]; q [T, n_q, d_qk]. Same KV conventions as above.
 * `chunk` is in logical rows (the kernel folds chunk by chunk with
 * (m, l, o) rescale — `M26_NAIVE_NO_RUNNING_RESCALE` is the trap). */
cudaError_t m26_attn_prefill_f32(
    const m26_geom* g, const float* q, const float* k, const float* v,
    const int64_t* q_pos, const int64_t* k_pos,
    int32_t T, int32_t S, int32_t chunk_rows, uint32_t naive,
    const float* sink, float* out, m26_stream_t stream);

cudaError_t m26_attn_prefill_fp8(
    const m26_geom* g, const float* q,
    const uint8_t* k_codes, const float* k_scales,
    const uint8_t* v_codes, const float* v_scales,
    const int32_t* page_table, int32_t page_tokens,
    const int64_t* q_pos, const int64_t* k_pos,
    int32_t T, int32_t S, int32_t chunk_rows, uint32_t naive,
    const float* sink, float* out, m26_stream_t stream);

/* Experimental P1 prefill (bounded qualification: attn-lead packet).
 * M64/N16, 256 threads, unit
 * E4M3 KV with already-prescaled V; configure before launch/capture. Separate
 * one-RNE post-RoPE BF16-Q lattice, never an implicit FP32-Q replacement. */
cudaError_t m26_attn_prefill_tc_config(int32_t* registers,int32_t* active_ctas);
cudaError_t m26_attn_prefill_tc_config_bf16q(int32_t* registers,int32_t* active_ctas);
cudaError_t m26_attn_prefill_tc_config_split(int32_t* registers,int32_t* active_ctas);
cudaError_t m26_attn_prefill_fp8_tc(
    const m26_geom* g,const float* q,const uint8_t* kc,const uint8_t* vc,
    const int32_t* pages,int32_t page_tokens,const int64_t* qpos,const int64_t* kpos,
    int32_t T,int32_t S,uint32_t naive,const float* sink,float* out,m26_stream_t stream);
cudaError_t m26_attn_prefill_fp8_tc_bf16q(
    const m26_geom* g,const float* q,const uint8_t* kc,const uint8_t* vc,
    const int32_t* pages,int32_t page_tokens,const int64_t* qpos,const int64_t* kpos,
    int32_t T,int32_t S,uint32_t naive,const float* sink,float* out,m26_stream_t stream);
/* P1 trial-1 occupancy split (P1M=32, f32q only). Bitwise-identical to the
 * f32q baseline: the T-split only doubles grid.x; per-row softmax stays in one
 * CTA. Configure with m26_attn_prefill_tc_config_split before launch. */
cudaError_t m26_attn_prefill_fp8_tc_split(
    const m26_geom* g,const float* q,const uint8_t* kc,const uint8_t* vc,
    const int32_t* pages,int32_t page_tokens,const int64_t* qpos,const int64_t* kpos,
    int32_t T,int32_t S,uint32_t naive,const float* sink,float* out,m26_stream_t stream);

/* P2 serving prefill (attn_prefill_fa.cu): FlashAttention-2-style, FP16 Q/P
 * with FP32 accumulate over the unit-scale E4M3 cache. Contract: the T query
 * tokens are KV rows q_row0..q_row0+T-1 and KV positions are row-contiguous
 * (visibility in row space); naive must be 0. */
cudaError_t m26_attn_prefill_fp8_fa(
    const m26_geom* g,const float* q,const uint8_t* kc,const uint8_t* vc,
    int32_t T,int32_t S,int32_t q_row0,uint32_t naive,const float* sink,float* out,m26_stream_t stream);

/* ---- RoPE (T19/T7): FP32 on-the-fly, partial 64/192, dual θ -------------- */
cudaError_t m26_rope_apply(
    double theta, double partial_rotary_factor,
    const float* x, float* y /* may alias x */, const int64_t* pos,
    int32_t T, int32_t H, int32_t d, uint32_t naive, m26_stream_t stream);

/* ---- FP8 KV store (T18/T20 + amax clip gate) ----------------------------
 * k_raw [n_tok, n_kv, d_qk], v_raw [n_tok, n_kv, d_v] (RAW V — the kernel
 * applies `value_scale` BEFORE quantization). scales out: [n_tok, n_kv] each,
 * K and V separate (unit mode passes scale 1.0 and may pass NULL buffers).
 * clip_count: elements clamped at encode (must be 0 on real activations,
 * ADVISOR-I3 §10.4.2 — `M26_NAIVE_SILENT_CLAMP` reports 0 silently). */
cudaError_t m26_kv_store_fp8(
    const m26_geom* g,
    const float* k_raw, const float* v_raw,
    int32_t n_tok, int32_t unit_scale /* 1 = unit, 0 = per-token×head */,
    uint32_t naive,
    uint8_t* k_codes, float* k_scales,
    uint8_t* v_codes, float* v_scales,
    uint64_t* clip_count, m26_stream_t stream);

/* decode FP8 KV rows back to f32 (parity of the kv_store case). */
cudaError_t m26_kv_decode_fp8(
    const m26_geom* g,
    const uint8_t* k_codes, const float* k_scales,
    const uint8_t* v_codes, const float* v_scales,
    int32_t n_tok, uint32_t naive,
    float* k_out, float* v_out, m26_stream_t stream);

#ifdef __cplusplus
} /* extern "C" */
#endif
