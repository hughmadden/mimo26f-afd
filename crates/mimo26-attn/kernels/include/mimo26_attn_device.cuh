/* Device-side shared pieces for the mimo26-attn kernels: E4M3 codec (nearest,
 * ties-to-LARGER magnitude, clamps to ±448 — mirrors `mimo26_load::e4m3` and
 * `oracle/mimo26/quant/fp8_block.py`, all three golden-pinned to the same
 * table), the KV row accessors (f32 flat / FP8 flat-or-paged), and the
 * visibility predicate.
 *
 * Included only by the kernel TUs in this directory. */

#pragma once

#include "mimo26_attn_kernels.h"

#include <stdint.h>

namespace m26 {

/* ---- E4M3 (e4m3fn: bias 7, no infinities, 0x7F/0xFF = NaN) -------------- */

__device__ __forceinline__ float e4m3_decode(uint8_t b) {
  const float s = (b & 0x80u) ? -1.0f : 1.0f;
  const int e = (b >> 3) & 0x0Fu;
  const int m = b & 0x07u;
  if (e == 0) {
    return s * ((float)m * 0x1p-9f); /* subnormal: (m/8) * 2^-6 */
  }
  return s * (1.0f + (float)m * 0.125f) * exp2f((float)(e - 7));
}

/* Nearest representable, ties to the LARGER magnitude, clamp ±448.
 * `clip` (optional, device atomic) counts inputs outside ±448 (the amax clip
 * gate — ADVISOR-I3 §10.4.2). Magnitude grid: (1 + m/8)·2^(e−7) normals,
 * m·2^-9 subnormals — the same monotone grid the oracle searchsorted walks. */
__device__ __forceinline__ uint8_t e4m3_encode(float x, unsigned long long* clip) {
  const uint8_t sign = (x < 0.0f) ? 0x80u : 0x00u;
  float mag = fabsf(x);
  if (mag > 448.0f) {
    if (clip) atomicAdd(clip, 1ULL);
    mag = 448.0f;
  }
  if (mag == 0.0f) return sign;
  int code;
  const int E = ilogbf(mag);
  if (E <= -7) {
    /* subnormal band: magnitudes q·2^-9, q in 0..8 (q == 8 == smallest normal) */
    code = (int)floorf(mag * 512.0f + 0.5f); /* half-up: ties to larger */
  } else {
    const float step = ldexpf(1.0f, E - 3);
    int q = (int)floorf(mag / step + 0.5f); /* half-up: ties to larger */
    int ee = E + 7;
    if (q >= 16) { q = 8; ee += 1; }
    code = (ee << 3) | (q - 8);
    if (code > 0x7E) code = 0x7E;
  }
  return (uint8_t)(sign | (uint8_t)code);
}

/* ---- predicate: causal k <= q, window q − k < window (T3/T9) ------------- */

__device__ __forceinline__ bool is_visible(int64_t qp, int64_t kp, int64_t window, uint32_t naive) {
  if (naive & M26_NAIVE_POS_ZEROED) { qp = 0; kp = 0; }
  if (kp > qp) return false;
  return (window <= 0) || ((qp - kp) < window);
}

__device__ __forceinline__ double attn_scale(int d_qk, int d_v, uint32_t naive) {
  const int d = (naive & M26_NAIVE_SCALE_BY_DV) ? d_v : d_qk;
  return 1.0 / sqrt((double)d);
}

/* ---- KV row accessors ---------------------------------------------------
 * load(): decoded f32 rows for logical row j, kv head kvh into kbuf/vbuf. */

struct KvF32 {
  const float* k;
  const float* v;
  int n_kv, d_qk, d_v;
  __device__ __forceinline__ void load(int j, int kvh, float* kbuf, float* vbuf) const {
    const float* kr = k + ((int64_t)j * n_kv + kvh) * d_qk;
    const float* vr = v + ((int64_t)j * n_kv + kvh) * d_v;
    for (int i = 0; i < d_qk; ++i) kbuf[i] = kr[i];
    for (int i = 0; i < d_v; ++i) vbuf[i] = vr[i];
  }
};

struct KvFp8 {
  const uint8_t* k_codes;
  const float* k_scales; /* NULL => unit scale */
  const uint8_t* v_codes;
  const float* v_scales;
  const int32_t* page_table; /* NULL => flat */
  int page_tokens;
  int n_kv, d_qk, d_v;
  __device__ __forceinline__ int phys(int j) const {
    if (!page_table) return j;
    const int page = j / page_tokens;
    return page_table[page] * page_tokens + (j % page_tokens);
  }
  __device__ __forceinline__ void load(int j, int kvh, float* kbuf, float* vbuf) const {
    const int r = phys(j);
    const uint8_t* kc = k_codes + ((int64_t)r * n_kv + kvh) * d_qk;
    const uint8_t* vc = v_codes + ((int64_t)r * n_kv + kvh) * d_v;
    /* T20: K and V scales are separate planes, per token × head. The index is
     * the LOGICAL row j — scales follow the token, not the page slot. */
    const float ks = k_scales ? k_scales[(int64_t)j * n_kv + kvh] : 1.0f;
    const float vs = v_scales ? v_scales[(int64_t)j * n_kv + kvh] : 1.0f;
    for (int i = 0; i < d_qk; ++i) kbuf[i] = e4m3_decode(kc[i]) * ks;
    for (int i = 0; i < d_v; ++i) vbuf[i] = e4m3_decode(vc[i]) * vs;
  }
};

/* one key fold into (m, l, o) with running-max rescale (the online softmax) */
__device__ __forceinline__ void fold_key(double s, const float* vbuf, int d_v,
                                        double* m, double* l, double* o,
                                        uint32_t naive) {
  const double m_new = fmax(*m, s);
  double rescale = 1.0;
  if (!(naive & M26_NAIVE_NO_RUNNING_RESCALE)) {
    rescale = isfinite(*m) ? exp(*m - m_new) : 0.0;
  }
  const double p = exp(s - m_new);
  *l = *l * rescale + p;
  for (int i = 0; i < d_v; ++i) o[i] = o[i] * rescale + p * (double)vbuf[i];
  *m = m_new;
}

/* the sink column: ONE extra softmax column with ZERO value (T6) — it moves
 * the denominator only. */
__device__ __forceinline__ void fold_sink_column(double b, int d_v, double* m,
                                                double* l, double* o) {
  const double m_new = fmax(*m, b);
  const double rescale = isfinite(*m) ? exp(*m - m_new) : 0.0;
  *l = *l * rescale + exp(b - m_new);
  for (int i = 0; i < d_v; ++i) o[i] *= rescale;
  *m = m_new;
}

} /* namespace m26 */
