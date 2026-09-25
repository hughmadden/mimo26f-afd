/* Split-KV decode partials (flash decoding): one thread per (t, h, split),
 * sequential key loop folding into f64 (m, l, o[d_v]) — the CPU twin's
 * `decode_split_kv`/`fold_range_kvh` mapping, line for line. The sink column
 * is NOT folded here (T-split-sink is the trap): `attn_reduce.cu` adds it ONCE
 * per query. Correctness-first — see the header's rationale. */

#include "include/mimo26_attn_device.cuh"

namespace {

template <typename Kv>
__global__ void splitkv_kernel(
    m26_geom g, const float* q, Kv kv, const int64_t* q_pos, const int64_t* k_pos,
    int T, int S, int n_splits, uint32_t naive, double* partials) {
  const int total = T * g.n_q * n_splits;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const int sp = idx % n_splits;
    /* partials are [t][h][split] — slot (t,h,sp) = (t*n_q + h)*n_splits + sp
     * (mimo26_attn_kernels.h:76, read back at attn_reduce.cu:33 via idx =
     * t*n_q + h). Invert EXACTLY that; the old `(idx/n_splits) % T` and
     * `idx / (n_splits * g.n_q)` swapped the roles and collapsed every thread
     * onto (t=0, h=0) whenever T == 1. */
    const int h = (idx / n_splits) % g.n_q;
    const int t = idx / (n_splits * g.n_q);
    const int kvh = h / (g.n_q / g.n_kv); /* GQA: kv head h / n_rep (T6 layout) */
    const int lo = (int)(((int64_t)S * sp) / n_splits);
    const int hi = (int)(((int64_t)S * (sp + 1)) / n_splits);
    const int64_t qp = q_pos[t];
    int64_t window = g.window;
    if ((naive & M26_NAIVE_GA_WINDOWED) && window <= 0) {
      window = 128; /* bug: GA inherits the SWA window (T3) */
    }
    const double scale = m26::attn_scale(g.d_qk, g.d_v, naive);
    const float* q_row = q + ((int64_t)t * g.n_q + h) * g.d_qk;
    float kbuf[M26_MAX_D_QK];
    float vbuf[M26_MAX_D_V];
    double o[M26_MAX_D_V];
    for (int i = 0; i < g.d_v; ++i) o[i] = 0.0;
    double m = -INFINITY;
    double l = 0.0;
    for (int j = lo; j < hi; ++j) {
      if (!m26::is_visible(qp, k_pos[j], window, naive)) continue;
      kv.load(j, kvh, kbuf, vbuf);
      double s = 0.0;
      for (int i = 0; i < g.d_qk; ++i) s += (double)q_row[i] * (double)kbuf[i];
      s *= scale;
      m26::fold_key(s, vbuf, g.d_v, &m, &l, o, naive);
    }
    double* dst = partials + (int64_t)idx * (2 + g.d_v);
    dst[0] = m;
    dst[1] = l;
    for (int i = 0; i < g.d_v; ++i) dst[2 + i] = o[i];
  }
}

inline int launch_blocks(int total) { return total < 4096 ? total : 4096; }

} /* namespace */

extern "C" cudaError_t m26_attn_decode_splitkv_f32(
    const m26_geom* g, const float* q, const float* k, const float* v,
    const int64_t* q_pos, const int64_t* k_pos,
    int32_t T, int32_t S, int32_t n_splits, uint32_t naive,
    double* partials, m26_stream_t stream) {
  m26::KvF32 kv = {k, v, g->n_kv, g->d_qk, g->d_v};
  const int total = T * g->n_q * n_splits;
  splitkv_kernel<m26::KvF32><<<launch_blocks(total), 256, 0, (cudaStream_t)stream>>>(
      *g, q, kv, q_pos, k_pos, T, S, n_splits, naive, partials);
  return cudaGetLastError();
}

extern "C" cudaError_t m26_attn_decode_splitkv_fp8(
    const m26_geom* g, const float* q,
    const uint8_t* k_codes, const float* k_scales,
    const uint8_t* v_codes, const float* v_scales,
    const int32_t* page_table, int32_t page_tokens,
    const int64_t* q_pos, const int64_t* k_pos,
    int32_t T, int32_t S, int32_t n_splits, uint32_t naive,
    double* partials, m26_stream_t stream) {
  m26::KvFp8 kv = {k_codes, k_scales, v_codes, v_scales,
                   page_table, page_tokens, g->n_kv, g->d_qk, g->d_v};
  const int total = T * g->n_q * n_splits;
  splitkv_kernel<m26::KvFp8><<<launch_blocks(total), 256, 0, (cudaStream_t)stream>>>(
      *g, q, kv, q_pos, k_pos, T, S, n_splits, naive, partials);
  return cudaGetLastError();
}
