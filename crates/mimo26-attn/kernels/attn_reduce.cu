/* Split-KV reduce: combine per-split (m, l, o) f64 partials into the output
 * row. The sink column (SWA-only, per Q head [64], ZERO value — T6/c1) enters
 * the denominator EXACTLY ONCE per query here. `M26_NAIVE_SINK_PER_SPLIT`
 * reproduces the trap (a sink folded into every split counts it n_splits
 * times) for the negative test.
 *
 * `partials` layout (from attn_decode_splitkv.cu):
 *   [t][h][split][0]=m, [1]=l, [2..2+d_v)=o
 */

#include "include/mimo26_attn_device.cuh"

namespace {

__global__ void reduce_kernel(m26_geom g, const double* partials, const float* sink,
                              int T, int n_splits, uint32_t naive, float* out) {
  const int total = T * g.n_q;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const int h = idx % g.n_q; /* idx = t*n_q + h carries the query row t */
    const int kvh = h / (g.n_q / g.n_kv);
    /* c1: GA ignores the sink bitwise (sink_active is the CPU twin's gate) */
    const bool sink_on = sink != NULL &&
                         (g.window > 0 || (naive & M26_NAIVE_SINK_ON_GA));
    const double b = sink_on
        ? (double)sink[(naive & M26_NAIVE_SINK_PER_KV) ? (int64_t)kvh : (int64_t)h]
        : 0.0;
    const bool per_split_sink = sink_on && (naive & M26_NAIVE_SINK_PER_SPLIT);

    double m = -INFINITY;
    for (int sp = 0; sp < n_splits; ++sp) {
      const double* p = partials + ((int64_t)idx * n_splits + sp) * (2 + g.d_v);
      double pm = p[0];
      if (per_split_sink) pm = fmax(pm, b);
      m = fmax(m, pm);
    }
    if (sink_on && !per_split_sink) m = fmax(m, b);

    double l = 0.0;
    double o[M26_MAX_D_V];
    for (int i = 0; i < g.d_v; ++i) o[i] = 0.0;
    for (int sp = 0; sp < n_splits; ++sp) {
      const double* p = partials + ((int64_t)idx * n_splits + sp) * (2 + g.d_v);
      double pm = p[0], pl = p[1];
      if (per_split_sink) {
        /* bug: this split carried its own sink column */
        const double pm2 = fmax(pm, b);
        pl = (isfinite(pm) ? pl * exp(pm - pm2) : 0.0) + exp(b - pm2);
        pm = pm2;
      }
      if (!isfinite(pm) && pl == 0.0) continue;
      const double w = exp(pm - m);
      l += w * pl;
      for (int i = 0; i < g.d_v; ++i) o[i] += w * p[2 + i];
    }
    if (sink_on && !per_split_sink) l += exp(b - m); /* ONCE (T6 reduce) */

    float* row = out + (int64_t)idx * g.d_v;
    for (int i = 0; i < g.d_v; ++i) {
      row[i] = (l > 0.0) ? (float)(o[i] / l) : 0.0f; /* empty row => zero */
    }
  }
}

} /* namespace */

extern "C" cudaError_t m26_attn_reduce(
    const m26_geom* g, const double* partials,
    const float* sink, int32_t T, int32_t n_splits, uint32_t naive, float* out,
    m26_stream_t stream) {
  const int total = T * g->n_q;
  const int blocks = total < 4096 ? total : 4096;
  reduce_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
      *g, partials, sink, T, n_splits, naive, out);
  return cudaGetLastError();
}
