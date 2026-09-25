// Route reduce → compact BF16 return rows, on the device.
//
// The FFN emits one FP32 row per padded routed row (expert-grouped,
// `[total_tokens, 4096]`, ~268 MB for a 2K chunk). This kernel collapses those
// padded rows into the per-token pre-sum and converts to BF16 (8,192 B/token)
// BEFORE anything leaves the device — removing the 268 MB download, the ~155 ms
// output copy, and the ~82 ms CPU reduce in one change.
//
// Numerics (bitwise-pinned against the CPU reduce in `route::RoutePlan::reduce`
// + `mimo26_wire::bf16::f32_to_bf16_rne`):
//   * per token, the routes are accumulated in the exact route order the CPU
//     uses (expert-grouped order preserved by a stable token-major sort on the
//     host) — `add(acc, multiply(w, z))` with round-to-nearest;
//   * the FP32 pre-sum converts to BF16 with round-to-nearest-even, the same
//     bit-level `(bits + 0x7fff + ((bits>>16)&1)) >> 16` the wire codec uses.

#include <cuda_bf16.h>
#include <cstdint>

namespace m26x {

__device__ __forceinline__ float rr_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float rr_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ uint16_t rr_bf16_rne(float v) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

// One token's 4096 output columns, in a single warp-strided block. Block = token.
__global__ void route_reduce_kernel(
    const float* __restrict__ ffn_out,     // [padded_rows, 4096] FP32
    const int32_t* __restrict__ padded,    // [routes] sorted by token (stable)
    const float* __restrict__ weight,      // [routes] sorted by token (stable)
    const int32_t* __restrict__ token_off, // [tokens + 1] route range per token
    int tokens, int hidden, uint16_t* __restrict__ bf16) // [tokens, hidden] out
{
    const int t = blockIdx.x;
    if (t >= tokens) return;
    const int begin = token_off[t];
    const int end = token_off[t + 1];
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float acc = 0.0f;
        for (int j = begin; j < end; ++j) {
            acc = rr_add(acc, rr_mul(ffn_out[padded[j] * hidden + h], weight[j]));
        }
        bf16[t * hidden + h] = rr_bf16_rne(acc);
    }
}

// Device gather: build the padded routed x rows from the token-major hidden rows.
// For each routed row i, copy d_hidden[src[i]] -> x[dst[i]]; the caller zeroes x
// first so the padding rows read as 0 (bitwise-matching the host replicate_x,
// which zero-fills then copies). Element-wise copy; the row mapping carries the
// expert-tile gaps so no per-tile logic is needed here.
__global__ void gather_x_kernel(
    const float* __restrict__ d_hidden, // [tokens, hidden] token-major
    const int32_t* __restrict__ src,    // [rows] source token index per routed row
    const int32_t* __restrict__ dst,    // [rows] padded row index per routed row
    int rows, int hidden, float* __restrict__ x) // [padded_rows, hidden] out
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * hidden) return;
    const int row = i / hidden;
    const int col = i - row * hidden;
    x[(size_t)dst[row] * hidden + col] = d_hidden[(size_t)src[row] * hidden + col];
}

} // namespace m26x

extern "C" cudaError_t m26x_gather_x(
    const float* hidden,
    const int32_t* src,
    const int32_t* dst,
    int rows,
    int hidden_dim,
    float* x,
    cudaStream_t stream)
{
    if (rows <= 0 || hidden_dim <= 0) return cudaErrorInvalidValue;
    const int total = rows * hidden_dim;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    m26x::gather_x_kernel<<<blocks, threads, 0, stream>>>(hidden, src, dst, rows, hidden_dim, x);
    return cudaGetLastError();
}

extern "C" cudaError_t m26x_route_reduce(
    const float* ffn_out,
    const int32_t* padded,
    const float* weight,
    const int32_t* token_off,
    int tokens,
    int hidden,
    int routes,
    uint16_t* bf16,
    cudaStream_t stream)
{
    if (tokens <= 0 || hidden <= 0 || routes < tokens) return cudaErrorInvalidValue;
    const uint64_t max_rows = 8ull * 4096ull; // 8 x capacity_class (class 4096)
    if (uint64_t(tokens) > max_rows) return cudaErrorInvalidValue;
    // One block per token; 256 threads stride the hidden dim.
    m26x::route_reduce_kernel<<<tokens, 256, 0, stream>>>(
        ffn_out, padded, weight, token_off, tokens, hidden, bf16);
    return cudaGetLastError();
}
