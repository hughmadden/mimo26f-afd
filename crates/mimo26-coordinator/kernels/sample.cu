// Served sampling (perf reset V3): DS41RT v15's target-sampling contract on the coordinator GPU.
//
// One CTA of 1024 threads per sampled logit row; the row's greedy id (from `m26c_argmax_rows`) is
// overwritten with the draw. Filters in vLLM's order: temperature, min_p, top_k, top_p, then an
// exact draw from what is left (`sampling.rs` has the contract and the CPU reference):
//   s_i = logit_i * (1 / T), m = max s_i;
//   min_p keeps s_i >= m + ln(min_p) (ln taken on the host, as DS41RT);
//   top_k keeps every token at or above the k-th largest s_i (ties at the boundary kept, as vLLM);
//   top_p keeps the smallest descending prefix of what is left whose weight reaches
//     ceil(top_p * its total) (ties at the boundary kept);
//   the draw is the kept token with the largest s_i + g_i (Gumbel-max, ties to the lower id), g_i
//     Gumbel noise keyed by the row's position key `rnd` and the token id (TensorFold's keyed
//     sampling, ashhart/TensorFold, MIT): an exact draw from the kept tokens' softmax.
// Weights for top_p are fixed point, q_i = floor(exp(s_i - m) * 2^40), so every sum is an integer and
// a draw is the same on every run and at every batch position. A token below m - 28 has q_i = 0
// (under 2^-40 of the top token's weight) and is never drawn, so the threshold searches start there.
// Thresholds come from a 16-way search over order-preserving float keys: no atomics.
//
// Coupled drafts (`m26c_draft_rows`, TensorFold import): the same kernel over a drafter's logit row
// picks the kept token with the largest s_i + w * g_i, with the target's own noise for that position,
// so a draft lands on the target's draw wherever the two distributions agree; `prob` gets the pick's
// share of the softmax of those scores over the kept set (the drafter's confidence for the chain cut).

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
constexpr int kSplits = 15;                 // thresholds per search pass
constexpr float kQScale = 1099511627776.f;  // 2^40
constexpr float kFloor = 28.f;              // exp(-28) * 2^40 < 1

// One sampled row; `sampling::DeviceRow` on the host (32 bytes).
struct Row {
  int32_t row;
  float inv_t;
  float top_p;     // >= 1: off
  float ln_min_p;  // -inf: off
  int32_t top_k;   // 0: off
  int32_t pad;
  uint64_t rnd;
};
static_assert(sizeof(Row) == 32, "Row layout");

// Order-preserving key: a larger float has a larger key; -0 == +0; NaN is 0 (never kept).
__device__ __forceinline__ uint32_t okey(float s) {
  if (s != s) return 0u;
  const uint32_t b = __float_as_uint(s == 0.f ? 0.f : s);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ unsigned long long weight(float s, float m) {
  return __float2ull_rz(expf(s - m) * kQScale);
}

// SplitMix64's finalizer.
__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// Gumbel noise for token `id` at the position keyed `rnd`: -ln(-ln u), u a 23-bit uniform in
// [2^-24, 1 - 2^-24], both ends exact in f32. (A 24-bit one, (k + 0.5) * 2^-24, rounds its top value
// to 1.0, and g = +inf would win a row for whichever kept token drew it: 1 id in 2^24.)
__device__ __forceinline__ float gumbel(unsigned long long rnd, int id) {
  const unsigned long long x = mix64(rnd ^ (unsigned long long)(unsigned)id);
  const float u = ((float)(uint32_t)(x >> 41) + 0.5f) * (1.f / 8388608.f);
  return -logf(-logf(u));
}

__device__ float block_max(float v, float* sh) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  __syncthreads();
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  v = sh[lane];
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ unsigned long long block_sum(unsigned long long v, unsigned long long* sh) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  v = sh[lane];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// The largest key t in [lo, hi) with W(t) >= target, where W(t) sums the weight (1, or q_i when
// `kMass`) of every token with key >= t. Requires W(lo) >= target > W(hi).
template <bool kMass>
__device__ uint32_t search(const float* row, int vocab, float inv_t, float m, uint32_t lo, uint32_t hi,
                           unsigned long long target, unsigned long long (*sh)[kSplits + 1],
                           unsigned long long* tot, uint32_t* ts) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  while (hi - lo > 1) {
    __syncthreads();
    if (threadIdx.x < kSplits) {
      const uint64_t w = hi - lo;
      ts[threadIdx.x] = lo + (uint32_t)((w * (uint64_t)(threadIdx.x + 1) + kSplits) / (kSplits + 1));
    }
    __syncthreads();
    uint32_t t[kSplits];
#pragma unroll
    for (int j = 0; j < kSplits; ++j) t[j] = ts[j];
    unsigned long long acc[kSplits + 1] = {};
    for (int i = threadIdx.x; i < vocab; i += kThreads) {
      const float s = row[i] * inv_t;
      const uint32_t k = okey(s);
      if (k < lo) continue;
      const unsigned long long w = kMass ? weight(s, m) : 1ull;
      if (k >= hi) {
        acc[kSplits] += w;
        continue;
      }
#pragma unroll
      for (int j = 0; j < kSplits; ++j) acc[j] += k >= t[j] ? w : 0ull;
    }
#pragma unroll
    for (int j = 0; j <= kSplits; ++j)
      for (int o = 16; o > 0; o >>= 1) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], o);
    if (lane == 0)
      for (int j = 0; j <= kSplits; ++j) sh[warp][j] = acc[j];
    __syncthreads();
    if (threadIdx.x <= kSplits) {
      unsigned long long v = 0;
      for (int w = 0; w < kWarps; ++w) v += sh[w][threadIdx.x];
      tot[threadIdx.x] = v;
    }
    __syncthreads();
    // W(t_j) = tot[j] + tot[kSplits] is non-increasing in j: lo moves to the last split that still
    // reaches the target, hi to the first that does not.
    uint32_t nlo = lo, nhi = hi;
    for (int j = 0; j < kSplits; ++j) {
      if (tot[j] + tot[kSplits] >= target) {
        nlo = t[j] > nlo ? t[j] : nlo;
      } else {
        nhi = t[j] < nhi ? t[j] : nhi;
        break;
      }
    }
    lo = nlo;
    hi = nhi;
  }
  return lo;
}

__global__ void __launch_bounds__(kThreads) sample_kernel(const float* __restrict__ x, int64_t ld, int vocab,
                                                         const Row* __restrict__ rows, float w,
                                                         int* __restrict__ out, float* __restrict__ prob) {
  __shared__ float shf[kWarps];
  __shared__ int shi[kWarps];
  __shared__ unsigned long long shu[kWarps];
  __shared__ unsigned long long shv[kWarps][kSplits + 1];
  __shared__ unsigned long long tot[kSplits + 1];
  __shared__ uint32_t ts[kSplits];
  const Row p = rows[blockIdx.x];
  const float* row = x + (int64_t)p.row * ld;
  const float inv_t = p.inv_t;

  float m = -INFINITY;
  for (int i = threadIdx.x; i < vocab; i += kThreads) m = fmaxf(m, row[i] * inv_t);
  m = block_max(m, shf);
  if (!(m > -INFINITY && m < INFINITY)) return;  // a non-finite row keeps its argmax
  const float smin = p.ln_min_p > -INFINITY ? m + p.ln_min_p : -INFINITY;
  const uint32_t kfloor = okey(fmaxf(smin, m - kFloor));
  const uint32_t hi = okey(m) + 1u;

  uint32_t kk = kfloor;
  if (p.top_k > 0) {
    unsigned long long n = 0;
    for (int i = threadIdx.x; i < vocab; i += kThreads) n += okey(row[i] * inv_t) >= kfloor;
    n = block_sum(n, shu);
    if (n > (unsigned long long)p.top_k)
      kk = search<false>(row, vocab, inv_t, m, kfloor, hi, (unsigned long long)p.top_k, shv, tot, ts);
  }
  uint32_t kp = kk;
  if (p.top_p < 1.f) {
    unsigned long long q = 0;
    for (int i = threadIdx.x; i < vocab; i += kThreads) {
      const float s = row[i] * inv_t;
      if (okey(s) >= kk) q += weight(s, m);
    }
    q = block_sum(q, shu);
    unsigned long long target = (unsigned long long)ceil((double)p.top_p * (double)q);
    target = target < 1ull ? 1ull : (target > q ? q : target);
    kp = search<true>(row, vocab, inv_t, m, kk, hi, target, shv, tot, ts);
  }

  // The draw: the kept token with the largest s_i + w * g_i. Each thread walks its ids in increasing
  // order and keeps the first maximum, and the reductions prefer the lower id, so ties go to it.
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float best = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = threadIdx.x; i < vocab; i += kThreads) {
    const float s = row[i] * inv_t;
    if (okey(s) < kp) continue;
    const float v = s + w * gumbel(p.rnd, i);
    if (v > best) {
      best = v;
      bi = i;
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > best || (ov == best && oi < bi)) {
      best = ov;
      bi = oi;
    }
  }
  __syncthreads();
  if (lane == 0) {
    shf[warp] = best;
    shi[warp] = bi;
  }
  __syncthreads();
  if (warp == 0) {
    best = shf[lane];
    bi = shi[lane];
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > best || (ov == best && oi < bi)) {
        best = ov;
        bi = oi;
      }
    }
    if (lane == 0) {
      if (bi != 0x7fffffff) out[p.row] = bi;
      shf[0] = best;
      shi[0] = bi;
    }
  }
  if (!prob) return;
  __syncthreads();
  const float top = shf[0];
  if (shi[0] == 0x7fffffff) return;
  // The pick's share of softmax(s_i + w * g_i) over the kept set.
  float z = 0.f;
  for (int i = threadIdx.x; i < vocab; i += kThreads) {
    const float s = row[i] * inv_t;
    if (okey(s) < kp) continue;
    z += expf(s + w * gumbel(p.rnd, i) - top);
  }
  for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
  __syncthreads();
  if (lane == 0) shf[warp] = z;
  __syncthreads();
  if (warp == 0) {
    z = shf[lane];
    for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    if (lane == 0) prob[p.row] = z > 0.f ? 1.f / z : 0.f;
  }
}

}  // namespace

// Overwrite `out[rows[i].row]` with row i's draw over ids [0, vocab) of `x` (row stride `ld`).
extern "C" cudaError_t m26c_sample_rows(const float* x, int64_t ld, int vocab, const void* rows, int n, int* out,
                                       cudaStream_t s) {
  if (n <= 0) return cudaSuccess;
  sample_kernel<<<n, kThreads, 0, s>>>(x, ld, vocab, static_cast<const Row*>(rows), 1.f, out, nullptr);
  return cudaGetLastError();
}

// Coupled drafts: overwrite `out[rows[i].row]` with the kept token of the largest s + w * g (the
// target's noise for row i's position) and `prob[rows[i].row]` with its share of those scores' softmax.
extern "C" cudaError_t m26c_draft_rows(const float* x, int64_t ld, int vocab, const void* rows, int n, float w,
                                      int* out, float* prob, cudaStream_t s) {
  if (n <= 0) return cudaSuccess;
  sample_kernel<<<n, kThreads, 0, s>>>(x, ld, vocab, static_cast<const Row*>(rows), w, out, prob);
  return cudaGetLastError();
}
