/* Chunked prefill (online softmax over KV chunks/pages): one thread per
 * (query row t, Q head h), folding `chunk_rows` at a time with running
 * (m, l, o) and the `exp(old_m − m_new)` rescale. Mirrors the CPU twin's
 * `attention_chunked` + `Partial::fold`. `M26_NAIVE_NO_RUNNING_RESCALE` is the
 * trap (wrong whenever a later chunk holds a larger logit) and
 * `M26_NAIVE_SINK_PER_SPLIT` adds the sink once per chunk — both flip the CPU
 * negative tests in `tests/splitkv_prefill_equiv.rs` and, through the parity
 * harness, the GPU cell as well. */

#include "include/mimo26_attn_device.cuh"

namespace {

template <typename Kv>
__global__ void prefill_kernel(
    m26_geom g, const float* q, Kv kv, const int64_t* q_pos, const int64_t* k_pos,
    int T, int S, int chunk_rows, uint32_t naive, const float* sink, float* out) {
  const int total = T * g.n_q;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const int t = idx / g.n_q;
    const int h = idx % g.n_q;
    const int kvh = h / (g.n_q / g.n_kv);
    const int64_t qp = q_pos[t];
    int64_t window = g.window;
    if ((naive & M26_NAIVE_GA_WINDOWED) && window <= 0) {
      window = 128; /* bug: GA inherits the SWA window (T3) */
    }
    const bool sink_on = sink != NULL &&
                         (g.window > 0 || (naive & M26_NAIVE_SINK_ON_GA));
    const double b = sink_on
        ? (double)sink[(naive & M26_NAIVE_SINK_PER_KV) ? (int64_t)kvh : (int64_t)h]
        : 0.0;
    const double scale = m26::attn_scale(g.d_qk, g.d_v, naive);
    const float* q_row = q + ((int64_t)t * g.n_q + h) * g.d_qk;
    float kbuf[M26_MAX_D_QK];
    float vbuf[M26_MAX_D_V];
    double o[M26_MAX_D_V];
    for (int i = 0; i < g.d_v; ++i) o[i] = 0.0;
    double m = -INFINITY;
    double l = 0.0;
    for (int lo = 0; lo < S; lo += chunk_rows) {
      const int hi = lo + chunk_rows < S ? lo + chunk_rows : S;
      for (int j = lo; j < hi; ++j) {
        if (!m26::is_visible(qp, k_pos[j], window, naive)) continue;
        kv.load(j, kvh, kbuf, vbuf);
        double s = 0.0;
        for (int i = 0; i < g.d_qk; ++i) s += (double)q_row[i] * (double)kbuf[i];
        s *= scale;
        m26::fold_key(s, vbuf, g.d_v, &m, &l, o, naive);
      }
      if (sink_on && (naive & M26_NAIVE_SINK_PER_SPLIT)) {
        /* bug: the sink column added once per CHUNK (T-split-sink on the
         * prefill path) instead of once per query */
        m26::fold_sink_column(b, g.d_v, &m, &l, o);
      }
    }
    if (sink_on && !(naive & M26_NAIVE_SINK_PER_SPLIT)) {
      /* correct: the sink is ONE column with ZERO value (T6), added once */
      m26::fold_sink_column(b, g.d_v, &m, &l, o);
    }
    float* row = out + (int64_t)idx * g.d_v;
    for (int i = 0; i < g.d_v; ++i) {
      row[i] = (l > 0.0) ? (float)(o[i] / l) : 0.0f;
    }
  }
}

} /* namespace */

extern "C" cudaError_t m26_attn_prefill_f32(
    const m26_geom* g, const float* q, const float* k, const float* v,
    const int64_t* q_pos, const int64_t* k_pos,
    int32_t T, int32_t S, int32_t chunk_rows, uint32_t naive,
    const float* sink, float* out, m26_stream_t stream) {
  m26::KvF32 kv = {k, v, g->n_kv, g->d_qk, g->d_v};
  const int total = T * g->n_q;
  const int blocks = total < 4096 ? total : 4096;
  prefill_kernel<m26::KvF32><<<blocks, 256, 0, (cudaStream_t)stream>>>(
      *g, q, kv, q_pos, k_pos, T, S, chunk_rows, naive, sink, out);
  return cudaGetLastError();
}

extern "C" cudaError_t m26_attn_prefill_fp8(
    const m26_geom* g, const float* q,
    const uint8_t* k_codes, const float* k_scales,
    const uint8_t* v_codes, const float* v_scales,
    const int32_t* page_table, int32_t page_tokens,
    const int64_t* q_pos, const int64_t* k_pos,
    int32_t T, int32_t S, int32_t chunk_rows, uint32_t naive,
    const float* sink, float* out, m26_stream_t stream) {
  m26::KvFp8 kv = {k_codes, k_scales, v_codes, v_scales,
                   page_table, page_tokens, g->n_kv, g->d_qk, g->d_v};
  const int total = T * g->n_q;
  const int blocks = total < 4096 ? total : 4096;
  prefill_kernel<m26::KvFp8><<<blocks, 256, 0, (cudaStream_t)stream>>>(
      *g, q, kv, q_pos, k_pos, T, S, chunk_rows, naive, sink, out);
  return cudaGetLastError();
}
