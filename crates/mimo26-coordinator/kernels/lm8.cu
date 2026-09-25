// 8-bit drafter LM head (perf reset KN4, after knapcio's "FP8 twins": the draft
// model streams the 152,576 x 4,096 lm_head every step, 1.25 GB in BF16).
//
// The drafter's logits only choose which tokens the target verifies, so the
// drafter may read a quantized copy: INT8 with one FP32 scale per vocabulary row
// (max |w| / 127, round to nearest even), 625 MB. The target's verify keeps the
// BF16 lm_head, so outputs are unchanged; only the drafts (acceptance) can move.
//
// m26c_lm8_quantize: BF16 [V, K] -> INT8 [V, K] + scales [V] (once, at load).
// m26c_lm8_gemv: out[r, v] = scale[v] * sum_k bf16(x[r, k]) * w[v, k] for up to
// LM8_MAX_ROWS rows. x is cast to BF16 in shared memory (the BF16 GEMM path casts
// it the same way), each warp streams two vocabulary rows of weights per step
// with 16-byte loads, and blocks stride over the vocabulary so x is staged once
// per block.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

constexpr int LM8_MAX_ROWS = 8;
constexpr int LM8_THREADS = 256;

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) { return __uint_as_float(uint32_t(b) << 16); }

// One block per vocabulary row: max |w|, then the quantized row.
__global__ void __launch_bounds__(256) lm8_quantize_kernel(const uint16_t* __restrict__ w, int K,
                                                           int8_t* __restrict__ q, float* __restrict__ scale) {
    const long v = blockIdx.x;
    const uint16_t* row = w + v * K;
    float m = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) m = fmaxf(m, fabsf(bf16_bits_to_f32(row[k])));
    __shared__ float red[256];
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float amax = red[0];
    const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
    if (threadIdx.x == 0) scale[v] = sc;
    const float inv = 1.0f / sc;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const float x = rintf(bf16_bits_to_f32(row[k]) * inv);
        q[v * K + k] = int8_t(fminf(fmaxf(x, -127.0f), 127.0f));
    }
}

// Sixteen INT8 weights as floats (converted once, used for every row).
__device__ __forceinline__ void widen16(const int4 q, float (&f)[16]) {
    const int8_t* w = reinterpret_cast<const int8_t*>(&q);
#pragma unroll
    for (int i = 0; i < 16; ++i) f[i] = float(w[i]);
}

// Sixteen widened weights times sixteen BF16 activations, in k order.
__device__ __forceinline__ float dot16(const float (&f)[16], const uint16_t* __restrict__ xs) {
    const uint4 a = *reinterpret_cast<const uint4*>(xs);
    const uint4 b = *reinterpret_cast<const uint4*>(xs + 8);
    const uint32_t xa[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        s = fmaf(f[2 * i], __uint_as_float(xa[i] << 16), s);
        s = fmaf(f[2 * i + 1], __uint_as_float(xa[i] & 0xffff0000u), s);
    }
    return s;
}

template <int R>
__global__ void __launch_bounds__(LM8_THREADS) lm8_gemv_kernel(const float* __restrict__ x, const int8_t* __restrict__ w,
                                                              const float* __restrict__ scale, float* __restrict__ out,
                                                              long ld, int V, int K) {
    extern __shared__ uint16_t xs[];  // [R, K] BF16
    for (int i = threadIdx.x; i < R * K; i += blockDim.x) {
        const __nv_bfloat16 b = __float2bfloat16_rn(x[i]);
        xs[i] = *reinterpret_cast<const uint16_t*>(&b);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warps = LM8_THREADS / 32;
    for (long v0 = (long(blockIdx.x) * warps + warp) * 2; v0 < V; v0 += long(gridDim.x) * warps * 2) {
        const bool two = v0 + 1 < V;
        float acc0[R], acc1[R];
#pragma unroll
        for (int r = 0; r < R; ++r) acc0[r] = acc1[r] = 0.0f;
        const int4* w0 = reinterpret_cast<const int4*>(w + v0 * K);
        const int4* w1 = reinterpret_cast<const int4*>(w + (two ? v0 + 1 : v0) * K);
        for (int c = lane; c < K / 16; c += 32) {
            int4 q0, q1;
            asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(q0.x), "=r"(q0.y), "=r"(q0.z), "=r"(q0.w) : "l"(w0 + c));
            asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(q1.x), "=r"(q1.y), "=r"(q1.z), "=r"(q1.w) : "l"(w1 + c));
            float f0[16], f1[16];
            widen16(q0, f0);
            widen16(q1, f1);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint16_t* xr = xs + r * K + c * 16;
                acc0[r] += dot16(f0, xr);
                acc1[r] += dot16(f1, xr);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                acc0[r] += __shfl_xor_sync(0xffffffffu, acc0[r], o);
                acc1[r] += __shfl_xor_sync(0xffffffffu, acc1[r], o);
            }
        }
        if (lane == 0) {
            const float s0 = scale[v0];
#pragma unroll
            for (int r = 0; r < R; ++r) out[r * ld + v0] = acc0[r] * s0;
            if (two) {
                const float s1 = scale[v0 + 1];
#pragma unroll
                for (int r = 0; r < R; ++r) out[r * ld + v0 + 1] = acc1[r] * s1;
            }
        }
    }
}

template <int R>
cudaError_t launch_gemv(const float* x, const int8_t* w, const float* scale, float* out, long ld, int V, int K,
                        cudaStream_t s) {
    const size_t smem = size_t(R) * K * 2;
    static bool attr = false;
    if (!attr) {
        const cudaError_t e = cudaFuncSetAttribute(lm8_gemv_kernel<R>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   int(smem));
        if (e != cudaSuccess) return e;
        attr = true;
    }
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const int per_sm = R <= 4 ? 3 : 1;  // shared memory bounds residency: R x 8 KB of x per block
    lm8_gemv_kernel<R><<<sms * per_sm, LM8_THREADS, smem, s>>>(x, w, scale, out, ld, V, K);
    return cudaGetLastError();
}

}  // namespace

extern "C" {

cudaError_t m26c_lm8_quantize(const uint16_t* w, long V, int K, int8_t* q, float* scale, cudaStream_t s) {
    if (V <= 0 || K <= 0) return cudaSuccess;
    lm8_quantize_kernel<<<unsigned(V), 256, 0, s>>>(w, K, q, scale);
    return cudaGetLastError();
}

// Rows 1..LM8_MAX_ROWS; K a multiple of 16. Returns cudaErrorInvalidValue otherwise.
cudaError_t m26c_lm8_gemv(const float* x, int rows, const int8_t* w, const float* scale, float* out, long ld, int V,
                          int K, cudaStream_t s) {
    if (rows < 1 || rows > LM8_MAX_ROWS || K % 16 != 0) return cudaErrorInvalidValue;
    switch (rows) {
        case 1: return launch_gemv<1>(x, w, scale, out, ld, V, K, s);
        case 2: return launch_gemv<2>(x, w, scale, out, ld, V, K, s);
        case 3: return launch_gemv<3>(x, w, scale, out, ld, V, K, s);
        case 4: return launch_gemv<4>(x, w, scale, out, ld, V, K, s);
        case 5: return launch_gemv<5>(x, w, scale, out, ld, V, K, s);
        case 6: return launch_gemv<6>(x, w, scale, out, ld, V, K, s);
        case 7: return launch_gemv<7>(x, w, scale, out, ld, V, K, s);
        default: return launch_gemv<8>(x, w, scale, out, ld, V, K, s);
    }
}

}  // extern "C"
