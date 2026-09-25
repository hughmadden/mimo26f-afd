// Vision tower kernels (perf reset V2): MiMo-V2.6-Flash's image encoder
// (`visual.*`, `MiMoVisionTransformer` in the checkpoint's modeling code) on the
// coordinator GPU. Handwritten; the arithmetic follows the reference module:
//
//   patch embed  Conv3d(3 -> 1280, [2,16,16], no bias) == a GEMM over 1536-float patches
//   28 blocks    x += proj(attn(rope(qkv(rmsnorm1(x)))));  x += down(silu(gate(n)) * up(n)), n = rmsnorm2(x)
//                RMSNorm eps 1e-6 (weight); every Linear has a bias.
//   attention    32 query heads, 8 KV heads (4:1), head dim 64, scale 1/8; 2-D rotary (NeoX
//                rotate_half over 64 dims, angle = h or w position x inv_freq[16], theta 1e4).
//                Blocks 0/9/18/27 attend over the whole image; the rest over a band |i - j| <= 64
//                in the block's token order (row-major merge units, or column-major for window
//                type 1), with a learned per-head bias on the logit of the image's first key.
//   merger       LayerNorm(1280, eps 1e-6, weight, zero bias) -> 4 patches per row (5120) ->
//                Linear(5120, 5120) -> GELU(erf) -> Linear(5120, 4096); the checkpoint has no
//                merger biases (transformers initialises the missing ones to zero).
//
// Precision: BF16 weights, BF16 GEMM inputs, FP32 accumulate and FP32 residual stream (the
// reference runs in BF16 end to end); attention on FP16 Q/K/V with FP32 softmax/accumulate.
// Every entry point is asynchronous on `stream` and returns the launch status.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int VD = 64;                        // head dim
constexpr int VH = 32;                        // query heads
constexpr int VKV = 8;                        // key/value heads
constexpr int VQD = VH * VD;                  // 2048
constexpr int VQKV = (VH + 2 * VKV) * VD;     // 3072: [q 2048 | k 512 | v 512]
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ uint16_t bf16_bits(float x) {
  const __nv_bfloat16 b = __float2bfloat16_rn(x);
  return *reinterpret_cast<const uint16_t*>(&b);
}

__device__ __forceinline__ float block_sum(float v, float* smem) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  __syncthreads();
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  float t = 0.f;
  for (int i = 0; i < nw; ++i) t += smem[i];
  return t;
}

// RMSNorm (nn.RMSNorm in FP32: x * rsqrt(mean(x^2) + eps) * w) -> BF16. One block per row.
__global__ void rmsnorm_bf16_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                    uint16_t* __restrict__ y, int dim, float eps) {
  __shared__ float smem[32];
  const float* row = x + (size_t)blockIdx.x * dim;
  float s = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) s += row[i] * row[i];
  const float inv = rsqrtf(block_sum(s, smem) / (float)dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x) y[(size_t)blockIdx.x * dim + i] = bf16_bits(row[i] * inv * w[i]);
}

// LayerNorm (FP32, biased variance, zero bias) -> BF16. One block per row.
__global__ void layernorm_bf16_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                      uint16_t* __restrict__ y, int dim, float eps) {
  __shared__ float smem[32];
  const float* row = x + (size_t)blockIdx.x * dim;
  float s = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) s += row[i];
  const float mean = block_sum(s, smem) / (float)dim;
  float v = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float d = row[i] - mean;
    v += d * d;
  }
  const float inv = rsqrtf(block_sum(v, smem) / (float)dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x) y[(size_t)blockIdx.x * dim + i] = bf16_bits((row[i] - mean) * inv * w[i]);
}

// x[r, c] += y[r, c] + b[c]
__global__ void add_bias_residual_kernel(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ b,
                                         long n, int cols) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
    x[i] += y[i] + b[i % cols];
}

// gu = [gate | up] (FP32, [rows, 2*inter], no bias yet) -> BF16 silu(gate + bg) * (up + bu).
__global__ void swiglu_bias_bf16_kernel(const float* __restrict__ gu, const float* __restrict__ b,
                                        uint16_t* __restrict__ y, long rows, int inter) {
  const long n = rows * inter;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const long r = i / inter;
    const int c = (int)(i - r * inter);
    const float g = gu[r * 2 * inter + c] + b[c];
    const float u = gu[r * 2 * inter + inter + c] + b[inter + c];
    y[i] = bf16_bits(g / (1.f + expf(-g)) * u);
  }
}

// y = BF16(GELU_erf(x)) (nn.GELU default).
__global__ void gelu_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, long n) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const float v = x[i];
    y[i] = bf16_bits(0.5f * v * (1.f + erff(v * 0.70710678118654752f)));
  }
}

// Reorder whole merge units (4 patch rows): y[u*4 + r] = x[idx[u]*4 + r].
__global__ void gather_units_kernel(const float* __restrict__ x, float* __restrict__ y, const int32_t* __restrict__ idx,
                                    long n, int cols) {
  const long unit = 4L * cols;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const long u = i / unit;
    y[i] = x[(long)idx[u] * unit + (i - u * unit)];
  }
}

// qkv bias + 2-D rotary on q and k, FP16 out: q [n, 32, 64], k [n, 8, 64], v [n, 8, 64].
// Reference: q * cos + rotate_half(q) * sin in FP32, angle(d) = pos(d) * inv_freq[d % 16] with
// pos = h for d % 32 < 16 and w otherwise (the 32 frequencies are repeated for d >= 32).
__global__ void rope_qkv_kernel(const float* __restrict__ qkv, const float* __restrict__ bias,
                                const int32_t* __restrict__ hw, const float* __restrict__ inv_freq, int t0,
                                __half* __restrict__ q, __half* __restrict__ k, __half* __restrict__ v) {
  const int r = blockIdx.x, t = t0 + r;
  const float* row = qkv + (size_t)r * VQKV;
  const float hp = (float)hw[2 * t], wp = (float)hw[2 * t + 1];
  for (int i = threadIdx.x; i < (VH + VKV) * 32; i += blockDim.x) {
    const int head = i >> 5, d = i & 31;
    const float ang = (d < 16 ? hp : wp) * inv_freq[d & 15];
    const float c = cosf(ang), s = sinf(ang);
    const int base = head * VD;
    const float x0 = row[base + d] + bias[base + d];
    const float x1 = row[base + d + 32] + bias[base + d + 32];
    const float o0 = __fadd_rn(__fmul_rn(x0, c), __fmul_rn(-x1, s));
    const float o1 = __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s));
    __half* dst = head < VH ? q + ((size_t)t * VH + head) * VD : k + ((size_t)t * VKV + (head - VH)) * VD;
    dst[d] = __float2half_rn(o0);
    dst[d + 32] = __float2half_rn(o1);
  }
  for (int i = threadIdx.x; i < VKV * VD; i += blockDim.x) {
    const int j = VQD + VKV * VD + i;
    v[(size_t)t * VKV * VD + i] = __float2half_rn(row[j] + bias[j]);
  }
}

// ---------------------------------------------------------------------------
// Attention: FlashAttention-2 style, FP16 m16n8k16 MMA with FP32 accumulate.
// CTA = 16 tokens x the 4 query heads of one KV head (warp w -> head 4*kvh + w),
// 64-key tiles double-buffered with cp.async, one barrier per tile.
// ---------------------------------------------------------------------------
constexpr int AT = 16, AN = 64, AWARPS = 4, ATHREADS = AWARPS * 32;
constexpr int KP = VD + 8;  // padded smem row (halves): 144 B, conflict-free ldmatrix

struct __align__(16) AttnSmem {
  __half k[2][AN * KP];
  __half v[2][AN * KP];
};

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
  const uint32_t d = uint32_t(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const __half* p) {
  const uint32_t a = uint32_t(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const __half* p) {
  const uint32_t a = uint32_t(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

__global__ void __launch_bounds__(ATHREADS)
vit_attn_kernel(const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v, int n,
                int window, const float* __restrict__ sinks, uint16_t* __restrict__ out) {
  extern __shared__ __align__(16) unsigned char smem_bytes[];
  AttnSmem& sm = *reinterpret_cast<AttnSmem*>(smem_bytes);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;
  const int kvh = blockIdx.y, head = kvh * 4 + warp;
  const int t0 = blockIdx.x * AT;
  const int tA = t0 + g, tB = t0 + g + 8;
  const bool okA = tA < n, okB = tB < n;
  const int t_last = min(t0 + AT, n) - 1;
  const int lo = window > 0 ? max(0, t0 - window) : 0;
  const int hi = window > 0 ? min(n, t_last + window + 1) : n;
  const int n_tiles = (hi - lo + AN - 1) / AN;

  uint32_t qa[4][4];
  {
    const __half* qA = q + ((size_t)(okA ? tA : 0) * VH + head) * VD;
    const __half* qB = q + ((size_t)(okB ? tB : 0) * VH + head) * VD;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      const int d = kk * 16 + 2 * c;
      qa[kk][0] = okA ? *reinterpret_cast<const uint32_t*>(qA + d) : 0u;
      qa[kk][1] = okB ? *reinterpret_cast<const uint32_t*>(qB + d) : 0u;
      qa[kk][2] = okA ? *reinterpret_cast<const uint32_t*>(qA + d + 8) : 0u;
      qa[kk][3] = okB ? *reinterpret_cast<const uint32_t*>(qB + d + 8) : 0u;
    }
  }

  auto load_tile = [&](int it, int buf) {
    const int j0 = lo + it * AN;
#pragma unroll
    for (int i = 0; i < 8; ++i) {  // 64 rows x 8 16-byte chunks, K then V: 1024 chunks / 128 threads
      const int ch = tid + i * ATHREADS;
      const bool isk = ch < 512;
      const int local = ch & 511, key = local >> 3, part = local & 7, j = j0 + key;
      const bool valid = j < hi;
      const __half* src = (isk ? k : v) + ((size_t)(valid ? j : 0) * VKV + kvh) * VD + part * 8;
      cp_async16((isk ? sm.k[buf] : sm.v[buf]) + key * KP + part * 8, src, valid);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };

  float o[8][4] = {};
  float mxA = -INFINITY, mxB = -INFINITY, lA = 0.f, lB = 0.f;
  const float scale = 0.125f;  // head_dim ** -0.5
  const float sink = sinks ? sinks[head] : 0.f;

  if (n_tiles > 0) load_tile(0, 0);
  for (int it = 0; it < n_tiles; ++it) {
    const int buf = it & 1;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();  // tile `it` landed; every warp is done with tile it-1 (the other buffer)
    if (it + 1 < n_tiles) load_tile(it + 1, buf ^ 1);

    const int j0 = lo + it * AN;
    const int mi = lane >> 3, ri = lane & 7;
    float s[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t b[4];
        ldsm_x4(b, sm.k[buf] + ((2 * np + (mi >> 1)) * 8 + ri) * KP + kk * 16 + (mi & 1) * 8);
        mma16816(s[2 * np], qa[kk], b[0], b[1]);
        mma16816(s[2 * np + 1], qa[kk], b[2], b[3]);
      }
    }
    float rmA = -INFINITY, rmB = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = j0 + nt * 8 + 2 * c + (e & 1);
        const int r = e < 2 ? tA : tB;
        const int dist = r > j ? r - j : j - r;
        const bool vis = j < hi && (window <= 0 || dist <= window);
        float val = s[nt][e] * scale;
        if (j == 0) val += sink;
        val = vis ? val * LOG2E : -INFINITY;
        s[nt][e] = val;
        if (e < 2) rmA = fmaxf(rmA, val); else rmB = fmaxf(rmB, val);
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
    for (int dn = 0; dn < 8; ++dn) {
      o[dn][0] *= alA;
      o[dn][1] *= alA;
      o[dn][2] *= alB;
      o[dn][3] *= alB;
    }
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
      for (int dp = 0; dp < 4; ++dp) {
        uint32_t b[4];
        ldsm_x4_t(b, sm.v[buf] + (ks * 16 + (mi & 1) * 8 + ri) * KP + (2 * dp + (mi >> 1)) * 8);
        mma16816(o[2 * dp], pa[ks], b[0], b[1]);
        mma16816(o[2 * dp + 1], pa[ks], b[2], b[3]);
      }
    }
  }
  lA += __shfl_xor_sync(0xffffffffu, lA, 1);
  lA += __shfl_xor_sync(0xffffffffu, lA, 2);
  lB += __shfl_xor_sync(0xffffffffu, lB, 1);
  lB += __shfl_xor_sync(0xffffffffu, lB, 2);
  const float iA = lA > 0.f ? 1.f / lA : 0.f, iB = lB > 0.f ? 1.f / lB : 0.f;
#pragma unroll
  for (int dn = 0; dn < 8; ++dn) {
    const int col = dn * 8 + 2 * c;
    if (okA) {
      uint16_t* p = out + ((size_t)tA * VH + head) * VD + col;
      p[0] = bf16_bits(o[dn][0] * iA);
      p[1] = bf16_bits(o[dn][1] * iA);
    }
    if (okB) {
      uint16_t* p = out + ((size_t)tB * VH + head) * VD + col;
      p[0] = bf16_bits(o[dn][2] * iB);
      p[1] = bf16_bits(o[dn][3] * iB);
    }
  }
}

inline int grid_for(long n) {
  const long b = (n + 255) / 256;
  return (int)(b < 4096 ? (b > 0 ? b : 1) : 4096);
}

}  // namespace

extern "C" {

cudaError_t m26v_rmsnorm_bf16(const float* x, const float* w, uint16_t* y, int rows, int dim, float eps, cudaStream_t s) {
  if (rows <= 0) return cudaSuccess;
  rmsnorm_bf16_kernel<<<rows, 256, 0, s>>>(x, w, y, dim, eps);
  return cudaGetLastError();
}

cudaError_t m26v_layernorm_bf16(const float* x, const float* w, uint16_t* y, int rows, int dim, float eps, cudaStream_t s) {
  if (rows <= 0) return cudaSuccess;
  layernorm_bf16_kernel<<<rows, 256, 0, s>>>(x, w, y, dim, eps);
  return cudaGetLastError();
}

cudaError_t m26v_add_bias_residual(float* x, const float* y, const float* b, long rows, int cols, cudaStream_t s) {
  const long n = rows * cols;
  if (n <= 0) return cudaSuccess;
  add_bias_residual_kernel<<<grid_for(n), 256, 0, s>>>(x, y, b, n, cols);
  return cudaGetLastError();
}

cudaError_t m26v_swiglu_bias_bf16(const float* gu, const float* b, uint16_t* y, long rows, int inter, cudaStream_t s) {
  const long n = rows * inter;
  if (n <= 0) return cudaSuccess;
  swiglu_bias_bf16_kernel<<<grid_for(n), 256, 0, s>>>(gu, b, y, rows, inter);
  return cudaGetLastError();
}

cudaError_t m26v_gelu_bf16(const float* x, uint16_t* y, long n, cudaStream_t s) {
  if (n <= 0) return cudaSuccess;
  gelu_bf16_kernel<<<grid_for(n), 256, 0, s>>>(x, y, n);
  return cudaGetLastError();
}

cudaError_t m26v_gather_units(const float* x, float* y, const int32_t* idx, long units, int cols, cudaStream_t s) {
  const long n = units * 4 * cols;
  if (n <= 0) return cudaSuccess;
  gather_units_kernel<<<grid_for(n), 256, 0, s>>>(x, y, idx, n, cols);
  return cudaGetLastError();
}

// rows [t0, t0 + rows) of the image; `qkv` holds just those rows.
cudaError_t m26v_rope_qkv(const float* qkv, const float* bias, const int32_t* hw, const float* inv_freq, int t0, int rows,
                          __half* q, __half* k, __half* v, cudaStream_t s) {
  if (rows <= 0) return cudaSuccess;
  rope_qkv_kernel<<<rows, 256, 0, s>>>(qkv, bias, hw, inv_freq, t0, q, k, v);
  return cudaGetLastError();
}

// One image of `n` patches; window 0 = full attention; sinks null = none.
cudaError_t m26v_attn(const __half* q, const __half* k, const __half* v, int n, int window, const float* sinks,
                      uint16_t* out, cudaStream_t s) {
  if (n <= 0) return cudaSuccess;
  static int ready = 0;  // constant-initialised flag (no C++ runtime guard in the Rust link)
  if (!ready) {
    const cudaError_t e = cudaFuncSetAttribute(vit_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               (int)sizeof(AttnSmem));
    if (e != cudaSuccess) return e;
    ready = 1;
  }
  vit_attn_kernel<<<dim3((n + AT - 1) / AT, VKV), ATHREADS, sizeof(AttnSmem), s>>>(q, k, v, n, window, sinks, out);
  return cudaGetLastError();
}

}  // extern "C"
