// P2: serving prefill attention (perf reset, docs/design/perf-reset-vs-ds41rt.md W8).
// FlashAttention-2-style GQA prefill for MiMo-V2.6-Flash (n_q 64, n_kv 4|8,
// d_qk 192, d_v 128) over the device FP8 KV cache. Handwritten; no upstream
// kernel copied.
//
// Contract (prefill only): the T query tokens are KV rows q_row0 .. q_row0+T-1
// (just appended) and KV row positions are contiguous (kpos[j] = kpos[0] + j),
// so visibility is evaluated in row space: key j is visible to query row r iff
// j <= r and (window <= 0 or r - j < window). The device forward's GA and SWA
// caches satisfy this (append in position order; SWA compaction keeps order).
//
// Numerics (native formats, FP32 accumulate; W8):
//   Q: FP32 post-RoPE -> FP16 (RNE), one term. |q| beyond FP16 range becomes
//      inf and surfaces as NaN output (loud).
//   K, V: E4M3 codes -> FP16, exact. NaN codes (0x7F/0xFF) are scrubbed to zero
//      and poison every output of the CTA (NaN), loud.
//   S = Q K^T with m16n8k16 FP16 MMA into FP32, scaled by log2(e)/sqrt(192).
//   Online softmax in FP32 (exp2); P -> FP16 (RNE); O += P V in FP32.
//   Sink (a sink vector on a windowed layer): folded into the running max and
//   denominator exactly as P1 does, in the log2 domain.
// Differences from P1 (A-f32q: 3-term BF16 Q, 2-term BF16 P, expf) are rounding
// only; qualified by kernels/bench/p2_check.cu and X1a.
//
// Tiling: 256 threads = 8 warps x 16 GQA-packed query rows (row = token x head
// of one KV head), 64-key tiles. Raw FP8 K/V tiles are double-buffered with
// cp.async and converted once per tile into padded FP16 smem (conflict-free
// ldmatrix). Softmax state lives in registers (four lanes per row); two block
// barriers per tile. Each CTA visits only its causal/window key range, and the
// longest CTAs (latest tokens) launch first. GA layers (window 0) run the same
// per-row math as two 64-row CTAs per SM (prefill_fa_ga, perf reset P3).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "include/mimo26_attn_kernels.h"

namespace {

constexpr int BM = 128, BN = 64, WARPS = 8, THREADS = WARPS * 32;
constexpr int DQK = 192, DV = 128;
constexpr int KPAD = DQK + 8;  // halves per FP16 K row: 400 B, 8 ldmatrix rows hit disjoint banks
constexpr int VPAD = DV + 8;   // halves per FP16 V row: 272 B
constexpr int RAW_K = BN * DQK, RAW_V = BN * DV;
constexpr int K_CHUNKS = BN * DQK / 16, V_CHUNKS = BN * DV / 16;  // 16-byte copies per tile
constexpr float LOG2E = 1.4426950408889634f;

struct __align__(16) Smem {
  uint8_t raw[2][RAW_K + RAW_V];  // FP8 K then V, row-major, double buffer
  __half k16[BN * KPAD];
  __half v16[BN * VPAD];
};

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
  const uint32_t d = uint32_t(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t codes) {
  uint32_t r;
  const uint16_t x = uint16_t(codes);
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(x));
  return r;
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);  // .x (low 16 bits) = lo
  return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const __half* p) {
  const uint32_t a = uint32_t(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const __half* p) {
  const uint32_t a = uint32_t(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(THREADS, 1)
prefill_fa(int n_kv, int window, const float* __restrict__ q, const uint8_t* __restrict__ kc,
           const uint8_t* __restrict__ vc, int T, int q_row0, const float* __restrict__ sink,
           float* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char smem_bytes[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_bytes);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;
  const int rep = 64 / n_kv, tpc = BM / rep, kvh = blockIdx.y;
  const int t0 = (int(gridDim.x) - 1 - int(blockIdx.x)) * tpc;  // longest CTAs first
  // This thread's two rows of the warp's 16: A = g, B = g + 8.
  const int mA = warp * 16 + g, mB = mA + 8;
  const int tA = t0 + mA / rep, tB = t0 + mB / rep;
  const int hA = kvh * rep + mA % rep, hB = kvh * rep + mB % rep;
  const int rA = q_row0 + tA, rB = q_row0 + tB;  // query rows in KV space
  const bool okA = tA < T, okB = tB < T;

  // Q fragments (FP32 -> FP16 RNE), 12 k-steps of 16 over d_qk.
  uint32_t qa[12][4];
  {
    const float* qA = q + ((int64_t)(okA ? tA : 0) * 64 + hA) * DQK;
    const float* qB = q + ((int64_t)(okB ? tB : 0) * 64 + hB) * DQK;
#pragma unroll
    for (int kk = 0; kk < 12; ++kk) {
      const int d = kk * 16 + 2 * c;
      const float2 z = make_float2(0.f, 0.f);
      const float2 a0 = okA ? *reinterpret_cast<const float2*>(qA + d) : z;
      const float2 a1 = okB ? *reinterpret_cast<const float2*>(qB + d) : z;
      const float2 a2 = okA ? *reinterpret_cast<const float2*>(qA + d + 8) : z;
      const float2 a3 = okB ? *reinterpret_cast<const float2*>(qB + d + 8) : z;
      qa[kk][0] = pack_half2(a0.x, a0.y);
      qa[kk][1] = pack_half2(a1.x, a1.y);
      qa[kk][2] = pack_half2(a2.x, a2.y);
      qa[kk][3] = pack_half2(a3.x, a3.y);
    }
  }

  // Key rows [lo, hi) visible to any row of this CTA.
  const int t_last = min(t0 + tpc, T) - 1;
  const int hi = q_row0 + t_last + 1;
  const int lo = window > 0 ? max(0, q_row0 + t0 - window + 1) : 0;
  const int n_tiles = (hi - lo + BN - 1) / BN;

  auto load_tile = [&](int it, int buf) {
    const int j0 = lo + it * BN;
    uint8_t* rk = sm.raw[buf];
    uint8_t* rv = sm.raw[buf] + RAW_K;
    for (int ch = tid; ch < K_CHUNKS + V_CHUNKS; ch += THREADS) {
      const bool isk = ch < K_CHUNKS;
      const int local = isk ? ch : ch - K_CHUNKS, per = isk ? DQK / 16 : DV / 16;
      const int key = local / per, part = local % per, j = j0 + key;
      const bool valid = j < hi;
      const uint8_t* src = isk ? kc + ((int64_t)(valid ? j : 0) * n_kv + kvh) * DQK + part * 16
                               : vc + ((int64_t)(valid ? j : 0) * n_kv + kvh) * DV + part * 16;
      cp_async16((isk ? rk + key * DQK : rv + key * DV) + part * 16, src, valid);  // invalid: zero fill
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };

  float o[16][4] = {};  // 16 n8 tiles over d_v; regs 0,1 row A, regs 2,3 row B
  float mxA = -INFINITY, mxB = -INFINITY, lA = 0.f, lB = 0.f;  // log2-domain max, per-lane partial sums
  uint32_t nan_codes = 0;
  const float scale = float(1.0 / sqrt(192.0)) * LOG2E;

  if (n_tiles > 0) load_tile(0, 0);
  for (int it = 0; it < n_tiles; ++it) {
    const int buf = it & 1;
    if (it + 1 < n_tiles) {
      load_tile(it + 1, buf ^ 1);  // that buffer's readers passed the previous tile's barriers
      asm volatile("cp.async.wait_group 1;" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();  // raw[buf] landed; every reader of the FP16 tiles is past the previous tile
    {
      const uint8_t* rk = sm.raw[buf];
      const uint8_t* rv = sm.raw[buf] + RAW_K;
      for (int ch = tid; ch < K_CHUNKS + V_CHUNKS; ch += THREADS) {
        const bool isk = ch < K_CHUNKS;
        const int local = isk ? ch : ch - K_CHUNKS, per = isk ? DQK / 16 : DV / 16;
        const int key = local / per, part = local % per;
        const uint4 w = *reinterpret_cast<const uint4*>((isk ? rk + key * DQK : rv + key * DV) + part * 16);
        const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
        uint32_t h[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          uint32_t x = ws[i];
          const uint32_t nb = ((x & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u;  // E4M3 NaN bytes
          nan_codes |= nb;
          x &= ~((nb >> 7) * 0xFFu);
          h[2 * i] = e4m3x2_to_f16x2(x & 0xFFFFu);
          h[2 * i + 1] = e4m3x2_to_f16x2(x >> 16);
        }
        __half* dst = isk ? sm.k16 + key * KPAD + part * 16 : sm.v16 + key * VPAD + part * 16;
        reinterpret_cast<uint4*>(dst)[0] = make_uint4(h[0], h[1], h[2], h[3]);
        reinterpret_cast<uint4*>(dst)[1] = make_uint4(h[4], h[5], h[6], h[7]);
      }
    }
    __syncthreads();  // FP16 K/V published

    const int j0 = lo + it * BN;
    const int mi = lane >> 3, ri = lane & 7;  // ldmatrix: matrix index / row within it
    float s[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 12; ++kk) {
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t b[4];
        ldsm_x4(b, sm.k16 + ((2 * np + (mi >> 1)) * 8 + ri) * KPAD + kk * 16 + (mi & 1) * 8);
        mma16816(s[2 * np], qa[kk], b[0], b[1]);
        mma16816(s[2 * np + 1], qa[kk], b[2], b[3]);
      }
    }
    // Scale, mask, running max (four lanes share a row).
    float rmA = -INFINITY, rmB = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = j0 + nt * 8 + 2 * c + (e & 1);
        const int r = e < 2 ? rA : rB;
        const bool vis = j < hi && j <= r && (window <= 0 || r - j < window);
        const float v = vis ? s[nt][e] * scale : -INFINITY;
        s[nt][e] = v;
        if (e < 2) rmA = fmaxf(rmA, v); else rmB = fmaxf(rmB, v);
      }
    }
    rmA = fmaxf(rmA, __shfl_xor_sync(0xffffffffu, rmA, 1));
    rmA = fmaxf(rmA, __shfl_xor_sync(0xffffffffu, rmA, 2));
    rmB = fmaxf(rmB, __shfl_xor_sync(0xffffffffu, rmB, 1));
    rmB = fmaxf(rmB, __shfl_xor_sync(0xffffffffu, rmB, 2));
    const float nA = fmaxf(mxA, rmA), nB = fmaxf(mxB, rmB);
    const float bA = nA == -INFINITY ? 0.f : nA, bB = nB == -INFINITY ? 0.f : nB;  // fully masked so far
    const float alA = exp2f(mxA - bA), alB = exp2f(mxB - bB);
    mxA = nA;
    mxB = nB;
    // P = exp2(S - max) as FP16 A fragments: n-tile 2k -> a0/a1, 2k+1 -> a2/a3.
    uint32_t pa[4][4];
    float sumA = 0.f, sumB = 0.f;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const float p0 = exp2f(s[nt][0] - bA), p1 = exp2f(s[nt][1] - bA);
      const float p2 = exp2f(s[nt][2] - bB), p3 = exp2f(s[nt][3] - bB);
      sumA += p0 + p1;
      sumB += p2 + p3;
      pa[nt >> 1][(nt & 1) * 2] = pack_half2(p0, p1);
      pa[nt >> 1][(nt & 1) * 2 + 1] = pack_half2(p2, p3);
    }
    lA = lA * alA + sumA;
    lB = lB * alB + sumB;
#pragma unroll
    for (int dn = 0; dn < 16; ++dn) {
      o[dn][0] *= alA;
      o[dn][1] *= alA;
      o[dn][2] *= alB;
      o[dn][3] *= alB;
    }
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
      for (int dp = 0; dp < 8; ++dp) {
        uint32_t b[4];
        ldsm_x4_t(b, sm.v16 + (ks * 16 + (mi & 1) * 8 + ri) * VPAD + (2 * dp + (mi >> 1)) * 8);
        mma16816(o[2 * dp], pa[ks], b[0], b[1]);
        mma16816(o[2 * dp + 1], pa[ks], b[2], b[3]);
      }
    }
  }

  // Complete each row's denominator across its four lanes.
  lA += __shfl_xor_sync(0xffffffffu, lA, 1);
  lA += __shfl_xor_sync(0xffffffffu, lA, 2);
  lB += __shfl_xor_sync(0xffffffffu, lB, 1);
  lB += __shfl_xor_sync(0xffffffffu, lB, 2);
  bool badA = false, badB = false;
  if (sink && window > 0) {  // P1: on = sink && window > 0 (default flags); bias per Q head
    const float sA = sink[hA] * LOG2E, sB = sink[hB] * LOG2E;
    badA = !isfinite(sA);
    badB = !isfinite(sB);
    const float nA = fmaxf(mxA, sA), nB = fmaxf(mxB, sB);
    const float alA = mxA == -INFINITY ? 0.f : exp2f(mxA - nA), alB = mxB == -INFINITY ? 0.f : exp2f(mxB - nB);
    lA = lA * alA + exp2f(sA - nA);
    lB = lB * alB + exp2f(sB - nB);
#pragma unroll
    for (int dn = 0; dn < 16; ++dn) {
      o[dn][0] *= alA;
      o[dn][1] *= alA;
      o[dn][2] *= alB;
      o[dn][3] *= alB;
    }
  }
  const bool poisoned = __syncthreads_or(nan_codes != 0);
  const float iA = lA > 0.f ? 1.f / lA : 0.f, iB = lB > 0.f ? 1.f / lB : 0.f;
  float* oA = out + ((int64_t)(okA ? tA : 0) * 64 + hA) * DV;
  float* oB = out + ((int64_t)(okB ? tB : 0) * 64 + hB) * DV;
#pragma unroll
  for (int dn = 0; dn < 16; ++dn) {
    const int col = dn * 8 + 2 * c;
    if (okA)
      *reinterpret_cast<float2*>(oA + col) = (poisoned || badA) ? make_float2(NAN, NAN)
                                             : make_float2(o[dn][0] * iA, o[dn][1] * iA);
    if (okB)
      *reinterpret_cast<float2*>(oB + col) = (poisoned || badB) ? make_float2(NAN, NAN)
                                             : make_float2(o[dn][2] * iB, o[dn][3] * iB);
  }
}

// GA layers (window 0): two 64-row CTAs per SM (4 warps each), so one CTA's
// softmax and K/V conversion overlap the other's MMAs; in the 128-row kernel all
// eight warps meet at the same two barriers and the tensor pipe idles through
// every conversion and softmax. The FP8 tile is staged in registers (issued
// before PV, converted after it), so the CTA's shared memory is just the FP16
// K/V tile. Per-row math is the 128-row kernel's, tile for tile: GA tiles start
// at key 0, masked keys add exact zeros, and fully masked tiles are no-ops, so
// the output is bit-identical. One exact shortcut: a tile every row of the CTA
// sees skips the mask. NaN codes still poison the CTA's outputs; they are no
// longer scrubbed first, which only changes values that are overwritten by NaN.
//
// Measured on the 5090 (kernels/bench/p2_long.cu, T 4096 lanes): 163 -> 212
// TFLOPS at 8K..1M keys (1.30x), output bit-identical; the FP32-accumulate MMA
// peak at the sustained 2.91 GHz clock is ~254 TFLOPS.
constexpr int GA_BM = 64, GA_THREADS = 4 * 32;
constexpr int GA_K_ITERS = BN * (DQK / 16) / GA_THREADS;  // 6 16-byte K chunks per thread per tile
constexpr int GA_V_ITERS = BN * (DV / 16) / GA_THREADS;   // 4 V chunks
struct __align__(16) GaSmem {
  __half k16[BN * KPAD];
  __half v16[BN * VPAD];
};

__device__ __forceinline__ uint4 ldg_stream(const uint8_t* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

// Sixteen E4M3 codes -> sixteen FP16 at dst; NaN codes are recorded (they poison
// the CTA's outputs) and converted as is.
__device__ __forceinline__ void cvt16(const uint4 w, __half* dst, uint32_t& nan_codes) {
  const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
  uint32_t h[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t x = ws[i];
    nan_codes |= ((x & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u;
    h[2 * i] = e4m3x2_to_f16x2(x & 0xFFFFu);
    h[2 * i + 1] = e4m3x2_to_f16x2(x >> 16);
  }
  reinterpret_cast<uint4*>(dst)[0] = make_uint4(h[0], h[1], h[2], h[3]);
  reinterpret_cast<uint4*>(dst)[1] = make_uint4(h[4], h[5], h[6], h[7]);
}

__global__ void __launch_bounds__(GA_THREADS, 2)
prefill_fa_ga(int n_kv, const float* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
              int T, int q_row0, float* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char smem_bytes[];
  GaSmem& sm = *reinterpret_cast<GaSmem*>(smem_bytes);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;
  const int rep = 64 / n_kv, tpc = GA_BM / rep, kvh = blockIdx.y;
  const int t0 = (int(gridDim.x) - 1 - int(blockIdx.x)) * tpc;  // longest CTAs first
  const int mA = warp * 16 + g, mB = mA + 8;
  const int tA = t0 + mA / rep, tB = t0 + mB / rep;
  const int hA = kvh * rep + mA % rep, hB = kvh * rep + mB % rep;
  const int rA = q_row0 + tA, rB = q_row0 + tB;
  const bool okA = tA < T, okB = tB < T;

  uint32_t qa[12][4];
  {
    const float* qA = q + ((int64_t)(okA ? tA : 0) * 64 + hA) * DQK;
    const float* qB = q + ((int64_t)(okB ? tB : 0) * 64 + hB) * DQK;
#pragma unroll
    for (int kk = 0; kk < 12; ++kk) {
      const int d = kk * 16 + 2 * c;
      const float2 z = make_float2(0.f, 0.f);
      const float2 a0 = okA ? *reinterpret_cast<const float2*>(qA + d) : z;
      const float2 a1 = okB ? *reinterpret_cast<const float2*>(qB + d) : z;
      const float2 a2 = okA ? *reinterpret_cast<const float2*>(qA + d + 8) : z;
      const float2 a3 = okB ? *reinterpret_cast<const float2*>(qB + d + 8) : z;
      qa[kk][0] = pack_half2(a0.x, a0.y);
      qa[kk][1] = pack_half2(a1.x, a1.y);
      qa[kk][2] = pack_half2(a2.x, a2.y);
      qa[kk][3] = pack_half2(a3.x, a3.y);
    }
  }

  const int t_last = min(t0 + tpc, T) - 1;
  const int hi = q_row0 + t_last + 1;
  const int n_tiles = (hi + BN - 1) / BN;
  const int r_min = q_row0 + t0;  // every key <= r_min is visible to every row of the CTA

  // This thread's chunks of a tile: K chunk tid + i * THREADS (12 per key), V chunk likewise (8 per key).
  uint4 st[GA_K_ITERS + GA_V_ITERS];
  auto load_tile = [&](int it) {
    const int j0 = it * BN;
#pragma unroll
    for (int i = 0; i < GA_K_ITERS; ++i) {
      const int ch = tid + i * GA_THREADS, key = ch / 12, part = ch % 12, j = j0 + key;
      st[i] = j < hi ? ldg_stream(kc + ((int64_t)j * n_kv + kvh) * DQK + part * 16) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int i = 0; i < GA_V_ITERS; ++i) {
      const int ch = tid + i * GA_THREADS, key = ch >> 3, part = ch & 7, j = j0 + key;
      st[GA_K_ITERS + i] = j < hi ? ldg_stream(vc + ((int64_t)j * n_kv + kvh) * DV + part * 16) : make_uint4(0, 0, 0, 0);
    }
  };

  float o[16][4] = {};
  float mxA = -INFINITY, mxB = -INFINITY, lA = 0.f, lB = 0.f;
  uint32_t nan_codes = 0;
  const float scale = float(1.0 / sqrt(192.0)) * LOG2E;

  if (n_tiles > 0) load_tile(0);
  for (int it = 0; it < n_tiles; ++it) {
    __syncthreads();  // every reader of the previous FP16 tile is done
#pragma unroll
    for (int i = 0; i < GA_K_ITERS; ++i) {
      const int ch = tid + i * GA_THREADS;
      cvt16(st[i], sm.k16 + (ch / 12) * KPAD + (ch % 12) * 16, nan_codes);
    }
#pragma unroll
    for (int i = 0; i < GA_V_ITERS; ++i) {
      const int ch = tid + i * GA_THREADS;
      cvt16(st[GA_K_ITERS + i], sm.v16 + (ch >> 3) * VPAD + (ch & 7) * 16, nan_codes);
    }
    __syncthreads();  // FP16 K/V published

    const int j0 = it * BN;
    const int mi = lane >> 3, ri = lane & 7;
    float s[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 12; ++kk) {
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t b[4];
        ldsm_x4(b, sm.k16 + ((2 * np + (mi >> 1)) * 8 + ri) * KPAD + kk * 16 + (mi & 1) * 8);
        mma16816(s[2 * np], qa[kk], b[0], b[1]);
        mma16816(s[2 * np + 1], qa[kk], b[2], b[3]);
      }
    }
    float rmA = -INFINITY, rmB = -INFINITY;
    if (j0 + BN - 1 <= r_min) {  // the whole tile is visible to every row
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float v = s[nt][e] * scale;
          s[nt][e] = v;
          if (e < 2) rmA = fmaxf(rmA, v); else rmB = fmaxf(rmB, v);
        }
      }
    } else {
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int j = j0 + nt * 8 + 2 * c + (e & 1);
          const int r = e < 2 ? rA : rB;
          const bool vis = j < hi && j <= r;
          const float v = vis ? s[nt][e] * scale : -INFINITY;
          s[nt][e] = v;
          if (e < 2) rmA = fmaxf(rmA, v); else rmB = fmaxf(rmB, v);
        }
      }
    }
    rmA = fmaxf(rmA, __shfl_xor_sync(0xffffffffu, rmA, 1));
    rmA = fmaxf(rmA, __shfl_xor_sync(0xffffffffu, rmA, 2));
    rmB = fmaxf(rmB, __shfl_xor_sync(0xffffffffu, rmB, 1));
    rmB = fmaxf(rmB, __shfl_xor_sync(0xffffffffu, rmB, 2));
    const float nA = fmaxf(mxA, rmA), nB = fmaxf(mxB, rmB);
    const float bA = nA == -INFINITY ? 0.f : nA, bB = nB == -INFINITY ? 0.f : nB;
    const float alA = exp2f(mxA - bA), alB = exp2f(mxB - bB);
    mxA = nA;
    mxB = nB;
    uint32_t pa[4][4];
    float sumA = 0.f, sumB = 0.f;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const float p0 = exp2f(s[nt][0] - bA), p1 = exp2f(s[nt][1] - bA);
      const float p2 = exp2f(s[nt][2] - bB), p3 = exp2f(s[nt][3] - bB);
      sumA += p0 + p1;
      sumB += p2 + p3;
      pa[nt >> 1][(nt & 1) * 2] = pack_half2(p0, p1);
      pa[nt >> 1][(nt & 1) * 2 + 1] = pack_half2(p2, p3);
    }
    lA = lA * alA + sumA;
    lB = lB * alB + sumB;
#pragma unroll
    for (int dn = 0; dn < 16; ++dn) {
      o[dn][0] *= alA;
      o[dn][1] *= alA;
      o[dn][2] *= alB;
      o[dn][3] *= alB;
    }
    if (it + 1 < n_tiles) load_tile(it + 1);  // in flight during PV
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
      for (int dp = 0; dp < 8; ++dp) {
        uint32_t b[4];
        ldsm_x4_t(b, sm.v16 + (ks * 16 + (mi & 1) * 8 + ri) * VPAD + (2 * dp + (mi >> 1)) * 8);
        mma16816(o[2 * dp], pa[ks], b[0], b[1]);
        mma16816(o[2 * dp + 1], pa[ks], b[2], b[3]);
      }
    }
  }

  lA += __shfl_xor_sync(0xffffffffu, lA, 1);
  lA += __shfl_xor_sync(0xffffffffu, lA, 2);
  lB += __shfl_xor_sync(0xffffffffu, lB, 1);
  lB += __shfl_xor_sync(0xffffffffu, lB, 2);
  const bool poisoned = __syncthreads_or(nan_codes != 0);
  const float iA = lA > 0.f ? 1.f / lA : 0.f, iB = lB > 0.f ? 1.f / lB : 0.f;
  float* oA = out + ((int64_t)(okA ? tA : 0) * 64 + hA) * DV;
  float* oB = out + ((int64_t)(okB ? tB : 0) * 64 + hB) * DV;
#pragma unroll
  for (int dn = 0; dn < 16; ++dn) {
    const int col = dn * 8 + 2 * c;
    if (okA) *reinterpret_cast<float2*>(oA + col) = poisoned ? make_float2(NAN, NAN) : make_float2(o[dn][0] * iA, o[dn][1] * iA);
    if (okB) *reinterpret_cast<float2*>(oB + col) = poisoned ? make_float2(NAN, NAN) : make_float2(o[dn][2] * iB, o[dn][3] * iB);
  }
}

}  // namespace

extern "C" cudaError_t m26_attn_prefill_fp8_fa(const m26_geom* g, const float* q, const uint8_t* kc,
                                              const uint8_t* vc, int32_t T, int32_t S, int32_t q_row0,
                                              uint32_t naive, const float* sink, float* out,
                                              m26_stream_t stream) {
  if (!g || g->n_q != 64 || (g->n_kv != 4 && g->n_kv != 8) || g->d_qk != DQK || g->d_v != DV || T <= 0 ||
      q_row0 < 0 || int64_t(q_row0) + T > S || g->window < 0 || g->window > (1 << 30) || naive != 0 || !q ||
      !kc || !vc || !out || ((uintptr_t(kc) | uintptr_t(vc)) & 15) || (uintptr_t(q) & 7) || (uintptr_t(out) & 7))
    return cudaErrorInvalidValue;
  // One-time dynamic shared memory opt-in. A constant-initialized flag, not a
  // guarded static: the Rust link carries no C++ runtime (__cxa_guard_*).
  if (g->window == 0) {
    static int ga_ready = 0;
    if (!ga_ready) {
      const cudaError_t e =
          cudaFuncSetAttribute(prefill_fa_ga, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(GaSmem)));
      if (e != cudaSuccess) return e;
      ga_ready = 1;
    }
    const int tpc = GA_BM / (64 / g->n_kv);
    prefill_fa_ga<<<dim3((T + tpc - 1) / tpc, g->n_kv), GA_THREADS, sizeof(GaSmem), (cudaStream_t)stream>>>(
        g->n_kv, q, kc, vc, T, q_row0, out);
    return cudaGetLastError();
  }
  static int smem_ready = 0;
  if (!smem_ready) {
    const cudaError_t e =
        cudaFuncSetAttribute(prefill_fa, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem)));
    if (e != cudaSuccess) return e;
    smem_ready = 1;
  }
  const int tpc = BM / (64 / g->n_kv);
  prefill_fa<<<dim3((T + tpc - 1) / tpc, g->n_kv), THREADS, sizeof(Smem), (cudaStream_t)stream>>>(
      g->n_kv, int(g->window), q, kc, vc, T, q_row0, sink, out);
  return cudaGetLastError();
}
