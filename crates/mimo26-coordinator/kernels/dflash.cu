/* DFlash drafter kernels (perf reset S1) and the device argmax.
 *
 * The drafter is the checkpoint's `dflash/` model (5 Qwen3-style layers, 64 Q /
 * 8 KV heads of 128, partial rotary 64 of 128, SWA window 1024, non-causal,
 * attention sink per Q head, V scaled by 0.612 on both paths). Semantics are
 * vLLM's as run by the D7 reference (`runs/20260923-d7-tp4-baseline/stage/
 * qwen3_dflash.py` + `v1/spec_decode/dflash.py` from its image):
 *
 *   - the draft KV holds context rows (target aux features projected per layer)
 *     and the current block's rows (bonus + 7 mask slots), BF16 here;
 *   - a query at position p sees every cached key with position >= p - 1023
 *     (FlashInfer non-causal, window_left = 1023) up to the block's end;
 *   - softmax includes the sink logit (no value), scores scaled by 1/sqrt(128).
 *
 * The draft KV is one ring per request and layer, [R][8 * 128] BF16 at row pos % R;
 * the kernels take per-sequence ring pointer tables.
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kHeadDim = 128;
constexpr int kQPerKv = 8;     /* 64 Q heads / 8 KV heads */
constexpr int kBlockRows = 8;  /* bonus + 7 mask slots */
constexpr int kTile = 32;      /* keys per shared-memory tile */

/* Row r goes to sequence row_seq[r]'s ring (skipped when negative) at pos % ring. */
__global__ void store_kv_kernel(const float* __restrict__ k, const float* __restrict__ v, int ld,
                                const int* __restrict__ row_seq, const int64_t* __restrict__ pos, int n, int hd, int ring,
                                float v_scale, const uint64_t* __restrict__ pk, const uint64_t* __restrict__ pv) {
  const int64_t total = (int64_t)n * hd;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int r = (int)(i / hd);
    const int e = (int)(i % hd);
    const int sq = row_seq[r];
    if (sq < 0) continue;
    const int64_t row = pos[r] % ring;
    ((__nv_bfloat16*)pk[sq])[row * hd + e] = __float2bfloat16_rn(k[(int64_t)r * ld + e]);
    ((__nv_bfloat16*)pv[sq])[row * hd + e] = __float2bfloat16_rn(v[(int64_t)r * ld + e] * v_scale);
  }
}

/* One CTA per (sequence, KV head, key split); warp w is Q head kvh * 8 + w and
 * carries the block's 8 rows. Partials: [seq][split][row][qhead][2 + 128] f32
 * holding (max, sum, unnormalized acc). */
__global__ void __launch_bounds__(256) attn_split_kernel(
    const float* __restrict__ q, const int64_t* __restrict__ qpos, const int64_t* __restrict__ seq_klo,
    const int64_t* __restrict__ seq_khi, const uint64_t* __restrict__ pk, const uint64_t* __restrict__ pv, int ring,
    int n_kv, int window, float scale, int split_keys, int splits, float* __restrict__ part) {
  __shared__ float ks[kTile][kHeadDim];
  __shared__ float vs[kTile][kHeadDim];
  const int seq = blockIdx.x / n_kv;
  const int kvh = blockIdx.x % n_kv;
  const int split = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n_q = n_kv * kQPerKv;
  const int qh = kvh * kQPerKv + warp;
  const int64_t klo = seq_klo[seq];
  const int64_t khi = seq_khi[seq];
  const int64_t k0 = klo + (int64_t)split * split_keys;
  const int64_t k1 = min(khi, k0 + split_keys);
  const __nv_bfloat16* rk = (const __nv_bfloat16*)pk[seq];
  const __nv_bfloat16* rv = (const __nv_bfloat16*)pv[seq];
  const int kv_row = n_kv * kHeadDim;

  float qr[kBlockRows][4];
  int64_t lo[kBlockRows];
  float m[kBlockRows], l[kBlockRows], acc[kBlockRows][4];
#pragma unroll
  for (int r = 0; r < kBlockRows; ++r) {
    const int row = seq * kBlockRows + r;
    const float* qp = q + ((int64_t)row * n_q + qh) * kHeadDim + lane * 4;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      qr[r][d] = qp[d] * scale;
      acc[r][d] = 0.f;
    }
    lo[r] = qpos[row] - (window - 1);
    m[r] = -INFINITY;
    l[r] = 0.f;
  }
  for (int64_t t0 = k0; t0 < k1; t0 += kTile) {
    const int nk = (int)min((int64_t)kTile, k1 - t0);
    __syncthreads();
    for (int i = threadIdx.x; i < kTile * kHeadDim; i += blockDim.x) {
      const int j = i / kHeadDim;
      const int d = i % kHeadDim;
      float kv = 0.f, vv = 0.f;
      if (j < nk) {
        const int64_t row = (t0 + j) % ring;
        const int64_t off = row * kv_row + kvh * kHeadDim + d;
        kv = __bfloat162float(rk[off]);
        vv = __bfloat162float(rv[off]);
      }
      ks[j][d] = kv;
      vs[j][d] = vv;
    }
    __syncthreads();
    for (int j = 0; j < nk; ++j) {
      const int64_t kp = t0 + j;
      const float4 kk = *reinterpret_cast<const float4*>(&ks[j][lane * 4]);
      const float4 vv = *reinterpret_cast<const float4*>(&vs[j][lane * 4]);
#pragma unroll
      for (int r = 0; r < kBlockRows; ++r) {
        float s = qr[r][0] * kk.x + qr[r][1] * kk.y + qr[r][2] * kk.z + qr[r][3] * kk.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (kp < lo[r]) continue; /* outside this query's window */
        const float mn = fmaxf(m[r], s);
        const float c = __expf(m[r] - mn);
        const float p = __expf(s - mn);
        l[r] = l[r] * c + p;
        acc[r][0] = acc[r][0] * c + p * vv.x;
        acc[r][1] = acc[r][1] * c + p * vv.y;
        acc[r][2] = acc[r][2] * c + p * vv.z;
        acc[r][3] = acc[r][3] * c + p * vv.w;
        m[r] = mn;
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kBlockRows; ++r) {
    float* out = part + ((((int64_t)seq * splits + split) * kBlockRows + r) * n_q + qh) * (2 + kHeadDim);
    if (lane == 0) {
      out[0] = m[r];
      out[1] = l[r];
    }
#pragma unroll
    for (int d = 0; d < 4; ++d) out[2 + lane * 4 + d] = acc[r][d];
  }
}

/* One CTA per (row, Q head), one thread per value dim: merge the splits and the
 * sink logit. */
__global__ void attn_reduce_kernel(const float* __restrict__ part, const float* __restrict__ sink, int splits, int n_q,
                                   float* __restrict__ out) {
  const int row = blockIdx.x / n_q;
  const int qh = blockIdx.x % n_q;
  const int seq = row / kBlockRows;
  const int r = row % kBlockRows;
  const int d = threadIdx.x;
  float mx = sink ? sink[qh] : -INFINITY;
  for (int s = 0; s < splits; ++s) {
    const float* p = part + ((((int64_t)seq * splits + s) * kBlockRows + r) * n_q + qh) * (2 + kHeadDim);
    mx = fmaxf(mx, p[0]);
  }
  float den = sink ? __expf(sink[qh] - mx) : 0.f;
  float num = 0.f;
  for (int s = 0; s < splits; ++s) {
    const float* p = part + ((((int64_t)seq * splits + s) * kBlockRows + r) * n_q + qh) * (2 + kHeadDim);
    if (p[1] == 0.f) continue; /* no visible key in this split */
    const float w = __expf(p[0] - mx);
    den += w * p[1];
    num += w * p[2 + d];
  }
  out[((int64_t)row * n_q + qh) * kHeadDim + d] = num / den;
}

/* First index of the row maximum (the host `greedy` rule; NaN never wins).
 * With `prob`, also the softmax probability of that maximum (1 / sum exp(x - max)). */
__global__ void __launch_bounds__(1024) argmax_kernel(const float* __restrict__ x, int64_t ld, int vocab,
                                                      int* __restrict__ out, float* __restrict__ prob) {
  __shared__ float bv[32];
  __shared__ int bi[32];
  const float* row = x + (int64_t)blockIdx.x * ld;
  float best = -INFINITY;
  int besti = 0x7fffffff;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float v = row[i];
    if (v > best || (v == best && i < besti)) {
      best = v;
      besti = i;
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
    if (ov > best || (ov == best && oi < besti)) {
      best = ov;
      besti = oi;
    }
  }
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) {
    bv[warp] = best;
    bi[warp] = besti;
  }
  __syncthreads();
  if (warp == 0) {
    const int nw = blockDim.x >> 5;
    best = lane < nw ? bv[lane] : -INFINITY;
    besti = lane < nw ? bi[lane] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, besti, o);
      if (ov > best || (ov == best && oi < besti)) {
        best = ov;
        besti = oi;
      }
    }
    if (lane == 0) {
      out[blockIdx.x] = besti == 0x7fffffff ? 0 : besti;
      bv[0] = best;
    }
  }
  if (!prob) return;
  __syncthreads();
  const float mx = bv[0];
  __syncthreads();
  float sum = 0.f;
  for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
    const float v = row[i];
    if (v == v) sum += __expf(v - mx);
  }
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (lane == 0) bv[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    const int nw = blockDim.x >> 5;
    sum = lane < nw ? bv[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) prob[blockIdx.x] = sum > 0.f ? 1.f / sum : 0.f;
  }
}

} /* namespace */

extern "C" cudaError_t m26c_dflash_store_kv(const float* k, const float* v, int ld, const int* row_seq,
                                           const int64_t* pos, int n, int hd, int ring, float v_scale,
                                           const uint64_t* pk, const uint64_t* pv, cudaStream_t s) {
  if (n <= 0) return cudaSuccess;
  const int64_t total = (int64_t)n * hd;
  const int blocks = (int)((total + 255) / 256 < 4096 ? (total + 255) / 256 : 4096);
  store_kv_kernel<<<blocks, 256, 0, s>>>(k, v, ld, row_seq, pos, n, hd, ring, v_scale, pk, pv);
  return cudaGetLastError();
}

extern "C" cudaError_t m26c_dflash_attn(const float* q, const int64_t* qpos, const int64_t* seq_klo,
                                       const int64_t* seq_khi, int nseq, const uint64_t* pk, const uint64_t* pv,
                                       int ring, int n_kv, int window, float scale, int split_keys, int splits,
                                       const float* sink, float* part, float* out, cudaStream_t s) {
  if (nseq <= 0) return cudaSuccess;
  attn_split_kernel<<<dim3(nseq * n_kv, splits), 256, 0, s>>>(q, qpos, seq_klo, seq_khi, pk, pv, ring, n_kv, window,
                                                             scale, split_keys, splits, part);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return e;
  const int n_q = n_kv * kQPerKv;
  attn_reduce_kernel<<<nseq * kBlockRows * n_q, kHeadDim, 0, s>>>(part, sink, splits, n_q, out);
  return cudaGetLastError();
}

extern "C" cudaError_t m26c_argmax_rows(const float* x, int64_t ld, int rows, int vocab, int* out, cudaStream_t s) {
  if (rows <= 0) return cudaSuccess;
  argmax_kernel<<<rows, 1024, 0, s>>>(x, ld, vocab, out, nullptr);
  return cudaGetLastError();
}

extern "C" cudaError_t m26c_argmax_prob_rows(const float* x, int64_t ld, int rows, int vocab, int* out, float* prob,
                                            cudaStream_t s) {
  if (rows <= 0) return cudaSuccess;
  argmax_kernel<<<rows, 1024, 0, s>>>(x, ld, vocab, out, prob);
  return cudaGetLastError();
}
