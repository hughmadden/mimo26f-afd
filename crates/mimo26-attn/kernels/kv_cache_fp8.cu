/* FP8 KV store + decode — T18/T20 + the amax clip gate.
 *
 *   T18  `value_scale` (0.707) is applied to V BEFORE quantization — both the
 *        codes and the plane amax see scaled V. `M26_NAIVE_VSCALE_AFTER_STORE`
 *        stores raw V (and `M26_NAIVE_VSCALE_ON_READ` rescales on read) — the
 *        T18 negative flips on the resulting code mismatch.
 *   T20  scale planes are PER TOKEN × HEAD with K and V SEPARATE
 *        (`M26_NAIVE_BLOCK128_SHARED_SCALES` is the shared-grid bug oracle —
 *        block-128 cannot divide the 192-dim K).
 *   Gate every clamped value into `clip_count` (must be 0 on real activations,
 *   ADVISOR-I3 §10.4.2); `M26_NAIVE_SILENT_CLAMP` hides the count (P-106).
 */

#include "include/mimo26_attn_device.cuh"

namespace {

__device__ __forceinline__ float plane_amax(const float* row, int d) {
  float amax = 0.0f;
  for (int i = 0; i < d; ++i) amax = fmaxf(amax, fabsf(row[i]));
  return amax;
}

/* plane amax of (scale · V) — T18: the codec sees scaled V */
__device__ __forceinline__ float plane_amax_scaled(const float* row, int d, float scale) {
  float amax = 0.0f;
  for (int i = 0; i < d; ++i) amax = fmaxf(amax, fabsf(row[i] * scale));
  return amax;
}

__device__ __forceinline__ float plane_scale_of(float amax) {
  return amax > 0.0f ? (float)((double)amax / 448.0) : 1.0f;
}

/* one thread per (t, kv head): quantizes both planes (K then V) */
__global__ void store_kernel(m26_geom g, const float* k_raw, const float* v_raw,
                             int n_tok, int unit_scale, uint32_t naive,
                             uint8_t* k_codes, float* k_scales,
                             uint8_t* v_codes, float* v_scales,
                             unsigned long long* clips) {
  const int total = n_tok * g.n_kv;
  const int row_elems = g.d_qk + g.d_v;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const int t = idx / g.n_kv;
    const int h = idx % g.n_kv;
    const float* kp = k_raw + (int64_t)idx * g.d_qk;
    const float* vp = v_raw + (int64_t)idx * g.d_v;
    unsigned long long* clip = (naive & M26_NAIVE_SILENT_CLAMP) ? NULL : clips;

    /* T20 shared-grid bug oracle: one scale per 128 flattened K‖V elements of
     * the whole token (K tail and V head share a block — the trap) */
    float shared[24];
    const bool block128 = (naive & M26_NAIVE_BLOCK128_SHARED_SCALES) != 0;
    if (block128) {
      const int n_blocks = (g.n_kv * row_elems + 127) / 128;
      for (int b = 0; b < n_blocks && b < 24; ++b) {
        float amax = 0.0f;
        for (int flat = b * 128; flat < (b + 1) * 128 && flat < g.n_kv * row_elems; ++flat) {
          const int hh = flat / row_elems;
          const int off = flat % row_elems;
          const float val = (off < g.d_qk)
              ? k_raw[((int64_t)t * g.n_kv + hh) * g.d_qk + off]
              : v_raw[((int64_t)t * g.n_kv + hh) * g.d_v + (off - g.d_qk)];
          amax = fmaxf(amax, fabsf(val));
        }
        shared[b] = plane_scale_of(amax);
      }
    }

    /* per token × head planes, K and V SEPARATE (T20) */
    float s_k = unit_scale ? 1.0f : plane_scale_of(plane_amax(kp, g.d_qk));
    float s_v;
    if (naive & M26_NAIVE_VSCALE_AFTER_STORE) {
      /* bug: quantize RAW V (T18) — the scale would land at read */
      s_v = unit_scale ? 1.0f : plane_scale_of(plane_amax(vp, g.d_v));
    } else {
      /* correct: the plane amax sees v_scale·V */
      s_v = unit_scale
          ? 1.0f
          : plane_scale_of(plane_amax_scaled(vp, g.d_v, (float)g.value_scale));
    }

    uint8_t* kdst = k_codes + (int64_t)idx * g.d_qk;
    uint8_t* vdst = v_codes + (int64_t)idx * g.d_v;
    for (int i = 0; i < g.d_qk; ++i) {
      const int flat = h * row_elems + i;
      const float s = block128 ? shared[flat / 128] : s_k;
      kdst[i] = m26::e4m3_encode(kp[i] / s, clip);
    }
    for (int i = 0; i < g.d_v; ++i) {
      const int flat = h * row_elems + g.d_qk + i;
      const float s = block128 ? shared[flat / 128] : s_v;
      /* T18: v_scale BEFORE the codec (unless the bug is on) */
      const float val = (naive & M26_NAIVE_VSCALE_AFTER_STORE)
          ? vp[i]
          : vp[i] * (float)g.value_scale;
      vdst[i] = m26::e4m3_encode(val / s, clip);
    }
    if (k_scales) k_scales[idx] = s_k;
    if (v_scales) v_scales[idx] = s_v;
  }
}

/* decode planes back to f32 (parity of the kv_store case) */
__global__ void decode_kernel(m26_geom g,
                              const uint8_t* k_codes, const float* k_scales,
                              const uint8_t* v_codes, const float* v_scales,
                              int n_tok, uint32_t naive, float* k_out, float* v_out) {
  const int per_head = g.d_qk + g.d_v;
  const int64_t total = (int64_t)n_tok * g.n_kv * per_head;
  for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * (int64_t)blockDim.x) {
    const int64_t head_base = idx / per_head; /* (t, h) plane index */
    const int off = (int)(idx % per_head);
    if (off < g.d_qk) {
      const float s = k_scales ? k_scales[head_base] : 1.0f;
      const int64_t o = head_base * g.d_qk + off;
      k_out[o] = m26::e4m3_decode(k_codes[o]) * s;
    } else {
      float s = v_scales ? v_scales[head_base] : 1.0f;
      if (naive & M26_NAIVE_VSCALE_ON_READ) s *= (float)g.value_scale;
      const int64_t o = head_base * g.d_v + (off - g.d_qk);
      v_out[o] = m26::e4m3_decode(v_codes[o]) * s;
    }
  }
}

} /* namespace */

extern "C" cudaError_t m26_kv_store_fp8(
    const m26_geom* g, const float* k_raw, const float* v_raw,
    int32_t n_tok, int32_t unit_scale, uint32_t naive,
    uint8_t* k_codes, float* k_scales, uint8_t* v_codes, float* v_scales,
    uint64_t* clip_count, m26_stream_t stream) {
  unsigned long long* d_clips = (unsigned long long*)clip_count;
  cudaError_t err = cudaMemsetAsync(d_clips, 0, sizeof(unsigned long long),
                                    (cudaStream_t)stream);
  if (err != cudaSuccess) return err;
  const int total = n_tok * g->n_kv;
  const int blocks = total < 4096 ? (total ? total : 1) : 4096;
  store_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
      *g, k_raw, v_raw, n_tok, unit_scale, naive,
      k_codes, k_scales, v_codes, v_scales, d_clips);
  return cudaGetLastError();
}

extern "C" cudaError_t m26_kv_decode_fp8(
    const m26_geom* g, const uint8_t* k_codes, const float* k_scales,
    const uint8_t* v_codes, const float* v_scales,
    int32_t n_tok, uint32_t naive,
    float* k_out, float* v_out, m26_stream_t stream) {
  const int64_t total = (int64_t)n_tok * g->n_kv * (g->d_qk + g->d_v);
  const int blocks = total < (4096ll * 256) ? (int)((total + 255) / 256) : 4096;
  decode_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
      *g, k_codes, k_scales, v_codes, v_scales, n_tok, naive, k_out, v_out);
  return cudaGetLastError();
}

/* Hidden quantize encode (the MoE wire-out Fp8E4m3Ue8m0K32): one thread per
 * element, `payload[i] = e4m3_encode(x[i] * scale_inv[i>>5])`. The scale_inv is a
 * power of two (2^(127-s)), so the f32 multiply is exact and bit-identical to the
 * CPU's f64 `v * scale_inv` (same mantissa, shifted exponent). */
__global__ void hidden_quantize_encode_kernel(
    const float* x, const float* scale_inv, uint8_t* payload, int64_t n_elem) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_elem) return;
  payload[i] = m26::e4m3_encode(x[i] * scale_inv[i >> 5], nullptr);
}

extern "C" cudaError_t m26_quantize_hidden_fp8(
    const float* x, const float* scale_inv, uint8_t* payload, int64_t n_elem,
    m26_stream_t stream) {
  const int64_t blocks = (n_elem + 255) / 256;
  hidden_quantize_encode_kernel<<<(int)blocks, 256, 0, (cudaStream_t)stream>>>(
      x, scale_inv, payload, n_elem);
  return cudaGetLastError();
}
