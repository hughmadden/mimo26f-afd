// Coordinator glue kernels for the device-resident forward (perf reset R1,
// docs/design/perf-reset-vs-ds41rt.md). Each kernel mirrors the arithmetic of
// its CPU twin in `crates/mimo26-coordinator/src` (norm.rs, router.rs,
// wire.rs), so the device forward keeps the CPU path's numerics: FP64
// accumulation where the twin uses FP64, the same tie-break, the same scale rule.
//
// Every entry point is asynchronous on `stream` and returns the launch status.

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <cuda_bf16.h>

#include "mimo26_attn_device.cuh"  // m26::e4m3_encode, the KV store's codec

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ double block_sum_f64(double v, double* smem) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    const int nw = blockDim.x >> 5;
    v = (threadIdx.x < nw) ? smem[threadIdx.x] : 0.0;
    if (warp == 0) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    }
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    const double r = smem[0];
    __syncthreads();
    return r;
}

// norm.rs::rmsnorm: mean of squares in FP64; out = (f64(x) * inv * f64(w)) as f32.
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                               float* __restrict__ out, int dim, double eps) {
    __shared__ double smem[32];
    const float* row = x + (size_t)blockIdx.x * dim;
    float* orow = out + (size_t)blockIdx.x * dim;
    double s = 0.0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const double v = (double)row[i];
        s += v * v;
    }
    s = block_sum_f64(s, smem);
    const double inv = 1.0 / sqrt(s / (double)dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        orow[i] = (float)((double)row[i] * inv * (double)w[i]);
    }
}

__global__ void add_inplace_kernel(float* __restrict__ h, const float* __restrict__ a, long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        h[i] += a[i];
    }
}

// norm.rs::silu (FP64 denominator, clip to [-60, 60]) then an f32 multiply by up.
__global__ void silu_mul_kernel(float* __restrict__ gate, const float* __restrict__ up, long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const float x = gate[i];
        const float xc = fminf(fmaxf(x, -60.0f), 60.0f);
        const double denom = 1.0 + exp(-(double)xc);
        const float s = (float)((double)x / denom);
        gate[i] = s * up[i];
    }
}

// router.rs::router_from_logits: FP64 sigmoid scores, select top-k by
// (score + bias) descending with ties to the lower expert id; weights are the
// unbiased scores over the selected set, summed in selection order, in FP64.
// One block per row; blockDim.x == n_experts (<= 1024).
__global__ void router_topk_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                   int n_experts, int top_k, int* __restrict__ idx_out,
                                   float* __restrict__ w_out) {
    extern __shared__ unsigned char sm_raw[];
    double* score = reinterpret_cast<double*>(sm_raw);          // [n_experts]
    double* sel = score + n_experts;                           // [n_experts]
    double* red_v = sel + n_experts;                           // [32]
    int* red_i = reinterpret_cast<int*>(red_v + 32);           // [32]
    int* chosen = red_i + 32;                                  // [top_k]
    const int e = threadIdx.x;
    const float* lrow = logits + (size_t)blockIdx.x * n_experts;
    if (e < n_experts) {
        const double s = 1.0 / (1.0 + exp(-(double)lrow[e]));
        score[e] = s;
        sel[e] = s + (double)bias[e];
    }
    __syncthreads();
    const int lane = e & 31, warp = e >> 5, nw = (blockDim.x + 31) >> 5;
    for (int k = 0; k < top_k; ++k) {
        double v = (e < n_experts) ? sel[e] : -INFINITY;
        int i = (e < n_experts) ? e : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            const double ov = __shfl_down_sync(0xffffffffu, v, o);
            const int oi = __shfl_down_sync(0xffffffffu, i, o);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if (lane == 0) { red_v[warp] = v; red_i[warp] = i; }
        __syncthreads();
        if (e == 0) {
            double bv = red_v[0];
            int bi = red_i[0];
            for (int wv = 1; wv < nw; ++wv) {
                if (red_v[wv] > bv || (red_v[wv] == bv && red_i[wv] < bi)) { bv = red_v[wv]; bi = red_i[wv]; }
            }
            chosen[k] = bi;
            sel[bi] = -INFINITY;
        }
        __syncthreads();
    }
    if (e == 0) {
        double wsum = 0.0;
        for (int k = 0; k < top_k; ++k) wsum += score[chosen[k]];
        for (int k = 0; k < top_k; ++k) {
            idx_out[(size_t)blockIdx.x * top_k + k] = chosen[k];
            w_out[(size_t)blockIdx.x * top_k + k] = (float)(score[chosen[k]] / wsum);
        }
    }
}

// wire.rs::quantize_hidden_scales: per K32 block, amax in FP64,
// s = amax == 0 ? 0 : clamp(ceil(log2(amax / 448)) + 127, 0, 254),
// scale_inv = 2^(127 - s) (exact in f32 for s in 0..=254).
__global__ void quant_scales_kernel(const float* __restrict__ x, long n_blocks,
                                    unsigned char* __restrict__ scales, float* __restrict__ scale_inv) {
    for (long b = blockIdx.x * (long)blockDim.x + threadIdx.x; b < n_blocks; b += (long)gridDim.x * blockDim.x) {
        const float* blk = x + b * 32;
        double amax = 0.0;
        for (int k = 0; k < 32; ++k) amax = fmax(amax, fabs((double)blk[k]));
        int s = 0;
        if (amax != 0.0) {
            double e = ceil(log2(amax / 448.0)) + 127.0;
            e = fmin(fmax(e, 0.0), 254.0);
            s = (int)e;
        }
        scales[b] = (unsigned char)s;
        scale_inv[b] = ldexpf(1.0f, 127 - s);
    }
}

// Scatter the positions of one appended batch into a layer's KV position column.
__global__ void copy_i64_kernel(const int64_t* __restrict__ src, int64_t* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// Fused attention prep (perf reset D3): the qkv split, the partial RoPE of q and
// k, the unit-scale E4M3 KV store (V times value_scale first) and the cache
// positions, in one pass. The unfused path is 3 strided copies, 2 x (copy then
// rotate) m26_rope_apply, m26_kv_store_fp8 (one thread per token x head) and a
// position copy. Every value is computed by the same expressions as rope.cu and
// kv_cache_fp8.cu, in the same order, so the outputs are bit-identical.
// One element per thread; inv_freq[j] (rope.cu's double expression) is built
// once per block in shared memory.
__global__ void attn_prep_kernel(const float* __restrict__ qkv, const int64_t* __restrict__ pos, int T, int nq,
                                 int nkv, int dqk, int dv, int rot_dim, double theta, float value_scale,
                                 float* __restrict__ q_rot, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                 int64_t* __restrict__ kpos, unsigned long long* clip) {
    __shared__ float inv_freq[128];
    const int half = rot_dim / 2;
    for (int j = threadIdx.x; j < half; j += blockDim.x)
        inv_freq[j] = (float)exp(-2.0 * (double)j / (double)rot_dim * log(theta));
    __syncthreads();
    const int q_rows = nq * dqk, k_rows = nkv * dqk, v_rows = nkv * dv, total = q_rows + k_rows + v_rows;
    const long n = (long)T * total;
    for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < n; idx += (long)gridDim.x * blockDim.x) {
        const int t = (int)(idx / total), c = (int)(idx % total);
        const float* row = qkv + (long)t * total;
        const int64_t p = pos[t];
        if (c == 0) kpos[t] = p;
        if (c < q_rows + k_rows) {
            const bool isq = c < q_rows;
            const int cc = isq ? c : c - q_rows;
            const int h = cc / dqk, d = cc % dqk;
            const float* x = row + (isq ? 0 : q_rows) + h * dqk;
            float out;
            if (d < rot_dim) {  // rope.cu: pair j joins channels (j, j + half)
                const int j = d < half ? d : d - half;
                const float ang = (float)p * inv_freq[j];
                const float cs = cosf(ang), sn = sinf(ang);
                const float a = x[j], b = x[j + half];
                out = d < half ? a * cs - b * sn : b * cs + a * sn;
            } else {
                out = x[d];  // channels beyond rot_dim pass through
            }
            if (isq) q_rot[((long)t * nq + h) * dqk + d] = out;
            else kc[((long)t * nkv + h) * dqk + d] = m26::e4m3_encode(out / 1.0f, clip);  // unit scale
        } else {
            const int cc = c - q_rows - k_rows, h = cc / dv, d = cc % dv;
            const float val = row[q_rows + k_rows + h * dv + d] * value_scale;  // T18: scale before the codec
            vc[((long)t * nkv + h) * dv + d] = m26::e4m3_encode(val / 1.0f, clip);
        }
    }
}

// l4.rs CoordinatorSum (R8): s = 0; then s += f32(bf16) for ranks 0,1,2,3, FP32,
// no fused ops. The same order and start value as the host sum, so bit-identical.
__global__ void rank_sum_bf16_kernel(const uint16_t* __restrict__ p0, const uint16_t* __restrict__ p1,
                                     const uint16_t* __restrict__ p2, const uint16_t* __restrict__ p3,
                                     float* __restrict__ out, long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        float s = 0.0f;
        s = __fadd_rn(s, __uint_as_float((uint32_t)p0[i] << 16));
        s = __fadd_rn(s, __uint_as_float((uint32_t)p1[i] << 16));
        s = __fadd_rn(s, __uint_as_float((uint32_t)p2[i] << 16));
        s = __fadd_rn(s, __uint_as_float((uint32_t)p3[i] << 16));
        out[i] = s;
    }
}

// FP32 -> BF16 round-to-nearest-even (the GEMM input cast, perf reset R1b).
__global__ void f32_to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        y[i] = __bfloat16_as_ushort(__float2bfloat16_rn(x[i]));
    }
}

inline int grid_for(long n) {
    long g = (n + kThreads - 1) / kThreads;
    if (g > 65535L * 8) g = 65535L * 8;
    if (g < 1) g = 1;
    return (int)g;
}

}  // namespace

extern "C" {

cudaError_t m26c_rmsnorm(const float* x, const float* w, float* out, int rows, int dim, double eps,
                         cudaStream_t stream) {
    if (rows <= 0) return cudaSuccess;
    rmsnorm_kernel<<<rows, kThreads, 0, stream>>>(x, w, out, dim, eps);
    return cudaGetLastError();
}

cudaError_t m26c_add_inplace(float* h, const float* a, long n, cudaStream_t stream) {
    if (n <= 0) return cudaSuccess;
    add_inplace_kernel<<<grid_for(n), kThreads, 0, stream>>>(h, a, n);
    return cudaGetLastError();
}

cudaError_t m26c_silu_mul(float* gate, const float* up, long n, cudaStream_t stream) {
    if (n <= 0) return cudaSuccess;
    silu_mul_kernel<<<grid_for(n), kThreads, 0, stream>>>(gate, up, n);
    return cudaGetLastError();
}

cudaError_t m26c_router_topk(const float* logits, const float* bias, int rows, int n_experts, int top_k,
                             int* idx_out, float* w_out, cudaStream_t stream) {
    if (rows <= 0) return cudaSuccess;
    if (n_experts <= 0 || n_experts > 1024 || top_k <= 0 || top_k > n_experts || (n_experts % 32) != 0) {
        return cudaErrorInvalidValue;
    }
    const size_t smem = sizeof(double) * (2 * (size_t)n_experts + 32) + sizeof(int) * (32 + (size_t)top_k);
    router_topk_kernel<<<rows, n_experts, smem, stream>>>(logits, bias, n_experts, top_k, idx_out, w_out);
    return cudaGetLastError();
}

cudaError_t m26c_quant_scales(const float* x, long n_blocks, unsigned char* scales, float* scale_inv,
                              cudaStream_t stream) {
    if (n_blocks <= 0) return cudaSuccess;
    quant_scales_kernel<<<grid_for(n_blocks), kThreads, 0, stream>>>(x, n_blocks, scales, scale_inv);
    return cudaGetLastError();
}

cudaError_t m26c_rank_sum_bf16(const uint16_t* p0, const uint16_t* p1, const uint16_t* p2, const uint16_t* p3,
                               float* out, long n, cudaStream_t stream) {
    if (n <= 0) return cudaSuccess;
    rank_sum_bf16_kernel<<<grid_for(n), kThreads, 0, stream>>>(p0, p1, p2, p3, out, n);
    return cudaGetLastError();
}

cudaError_t m26c_f32_to_bf16(const float* x, uint16_t* y, long n, cudaStream_t stream) {
    if (n <= 0) return cudaSuccess;
    f32_to_bf16_kernel<<<grid_for(n), kThreads, 0, stream>>>(x, y, n);
    return cudaGetLastError();
}

cudaError_t m26c_attn_prep(const float* qkv, const int64_t* pos, int T, int nq, int nkv, int dqk, int dv,
                           int rot_dim, double theta, float value_scale, float* q_rot, uint8_t* kc, uint8_t* vc,
                           int64_t* kpos, unsigned long long* clip, cudaStream_t stream) {
    if (T <= 0) return cudaSuccess;
    if (rot_dim < 0 || rot_dim > dqk || rot_dim % 2 || rot_dim / 2 > 128) return cudaErrorInvalidValue;
    const long n = (long)T * (nq * dqk + nkv * dqk + nkv * dv);
    attn_prep_kernel<<<grid_for(n), kThreads, 0, stream>>>(qkv, pos, T, nq, nkv, dqk, dv, rot_dim, theta,
                                                           value_scale, q_rot, kc, vc, kpos, clip);
    return cudaGetLastError();
}

cudaError_t m26c_copy_i64(const int64_t* src, int64_t* dst, int n, cudaStream_t stream) {
    if (n <= 0) return cudaSuccess;
    copy_i64_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(src, dst, n);
    return cudaGetLastError();
}

// KV host tier (perf reset K2): positions of restored GA rows, dst[i] = start + i.
__global__ void iota_i64_kernel(int64_t* __restrict__ dst, int64_t start, long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) dst[i] = start + i;
}

cudaError_t m26c_iota_i64(int64_t* dst, int64_t start, long n, cudaStream_t stream) {
    if (n <= 0) return cudaSuccess;
    iota_i64_kernel<<<grid_for(n), kThreads, 0, stream>>>(dst, start, n);
    return cudaGetLastError();
}

// Perf reset P9: write one MoE request's route entries and hidden rows straight
// into the page-locked, device-mapped RDMA frame body (DS41RTE3 v3 layout: route
// entry = row u32, expert u32, weight f32 bits; hidden row = hid E4M3 payload
// then hid/32 UE8M0 scales at `pitch`). One kernel instead of two D2H route
// downloads and two strided copies; the bytes equal the host encoder's.
__global__ void frame_fill_kernel(const int* __restrict__ idx, const float* __restrict__ wts,
                                  const uint8_t* __restrict__ payload, const uint8_t* __restrict__ scales, int t,
                                  int topk, int hid, uint8_t* __restrict__ routes, uint8_t* __restrict__ hidden,
                                  int pitch) {
    // 8-byte stores: the hidden rows start at 136 * t bytes into the body (40 B
    // descriptor + 8 x 12 B routes per row), 8- but not always 16-byte aligned.
    const long nr = (long)t * topk;
    const int row8 = hid / 8, sc8 = hid / 32 / 8;
    const long nh = (long)t * (row8 + sc8);
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < nr + nh; i += (long)gridDim.x * blockDim.x) {
        if (i < nr) {
            uint32_t* e = reinterpret_cast<uint32_t*>(routes + i * 12);
            e[0] = uint32_t(i / topk);
            e[1] = uint32_t(idx[i]);
            e[2] = __float_as_uint(wts[i]);
        } else {
            const long j = i - nr;
            const long r = j / (row8 + sc8), c = j % (row8 + sc8);
            const uint2* src = c < row8 ? reinterpret_cast<const uint2*>(payload + r * hid) + c
                                        : reinterpret_cast<const uint2*>(scales + r * (hid / 32)) + (c - row8);
            *reinterpret_cast<uint2*>(hidden + r * pitch + c * 8) = *src;
        }
    }
}

cudaError_t m26c_frame_fill(const int* idx, const float* wts, const uint8_t* payload, const uint8_t* scales, int t,
                            int topk, int hid, uint8_t* routes, uint8_t* hidden, int pitch, cudaStream_t stream) {
    if (t <= 0) return cudaSuccess;
    const long n = (long)t * topk + (long)t * (hid / 8 + hid / 32 / 8);
    frame_fill_kernel<<<grid_for(n), kThreads, 0, stream>>>(idx, wts, payload, scales, t, topk, hid, routes, hidden,
                                                           pitch);
    return cudaGetLastError();
}

}  // extern "C"
