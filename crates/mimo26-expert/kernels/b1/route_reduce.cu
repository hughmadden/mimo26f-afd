// Compact-return scaffold adapted from ds41rt
// native/cuda/kernels/v41_route_reduce.cu:36-55,73-97,
// e2a6f2ca5af56b8b567fb5086f82597427f28477; SHA256
// 49fcb3270485505bfe57e91bc55e3eba005f68efb1c0b8b5d014f487095b3466.
// Copyright (c) 2026 T.J. Purtell. MIT; see LICENSE.ds41rt.
// Changes: H 4096 / top-8, explicit once-only weights after FC2, FP32 pre-sum
// output, no shared expert/final coordinator cast, checked extents/faults.
// Numerical mode: E-W4A8-v1. This scaffolds the adopted wire seam only.
#include "route_reduce.cuh"
#include <cstring>
#ifdef __CUDACC__
#include <cuda_bf16.h>
#endif
namespace m26b1 {
static Status validate_reduce(const RouteInput& x, const RankOutput& y) {
    if (x.rows > 4096) return Status::geometry;
    const uint64_t n = uint64_t(x.rows) * hidden;
    if (x.values != n * 8 || x.weight_elements != uint64_t(x.rows) * 8 ||
        y.partial_elements != n || y.bf16_elements != n) return Status::size;
    const void* inputs[] = {x.unweighted, x.weights};
    const uint64_t ib[] = {x.values * 4, x.weight_elements * 4};
    const void* outputs[] = {y.partial, y.bf16, y.fault};
    const uint64_t ob[] = {n * 4, n * 2, 4};
    for (int i = 0; i < 2; ++i)
        if (ib[i] && !span_ok(inputs[i], ib[i], 4)) return Status::pointer;
    for (int i = 0; i < 3; ++i) {
        if (ob[i] && !span_ok(outputs[i], ob[i], i == 1 ? 2 : 4)) return Status::pointer;
        for (int j = 0; j < i; ++j)
            if (overlaps(outputs[i], ob[i], outputs[j], ob[j])) return Status::overlap;
        for (int j = 0; j < 2; ++j)
            if (overlaps(outputs[i], ob[i], inputs[j], ib[j])) return Status::overlap;
    }
    return Status::ok;
}
M26B1_HD static bool finite(float v) { return v >= -0x1.fffffep127f && v <= 0x1.fffffep127f; }
M26B1_HD static float multiply(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fmul_rn(a, b);
#else
    volatile float value = a * b;
    return value;
#endif
}
M26B1_HD static float add(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fadd_rn(a, b);
#else
    volatile float value = a + b;
    return value;
#endif
}
M26B1_HD static uint16_t bf16_rne(float value) {
#ifdef __CUDA_ARCH__
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
#else
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
#endif
}
M26B1_HD static bool reduce_one(const RouteInput& x, uint64_t i, float& value, uint16_t& code) {
    const uint64_t row = i / hidden, column = i % hidden;
    value = 0.0f;
    for (uint32_t route = 0; route < 8; ++route) {
        const float w = x.weights[row * 8 + route];
        const float z = x.unweighted[(row * 8 + route) * hidden + column];
        if (!finite(w) || w < 0 || !finite(z)) return false;
        value = add(value, multiply(w, z));
        if (!finite(value)) return false;
    }
    code = bf16_rne(value);
    // BF16 rounding can overflow even when its FP32 input is finite.
    return (code & 0x7f80u) != 0x7f80u;
}
Status reduce_host(const RouteInput& x, const RankOutput& y) {
    const auto status = validate_reduce(x, y);
    if (status != Status::ok) return status;
    *y.fault = 0;
    for (uint64_t i = 0; i < y.partial_elements; ++i) {
        float value;
        uint16_t code;
        if (!reduce_one(x, i, value, code)) {
            *y.fault = static_cast<uint32_t>(Status::nonfinite);
            return Status::nonfinite;
        }
        y.partial[i] = value;
        y.bf16[i] = code;
    }
    return Status::ok;
}
#ifdef __CUDACC__
__global__ void reduce_kernel(RouteInput x, RankOutput y) {
    bool failed = false;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < y.partial_elements; i += uint64_t(gridDim.x) * blockDim.x) {
        float value;
        uint16_t code;
        if (!reduce_one(x, i, value, code)) {
            // Outputs are invalid on fault; the element keeps its NaN marker.
            y.partial[i] = __int_as_float(0x7fc00000);
            y.bf16[i] = 0;
            failed = true;
        } else {
            y.partial[i] = value;
            y.bf16[i] = code;
        }
    }
    // Fault status in the same pass, still without atomics: each block ORs its
    // own threads' faults and a failing block's thread 0 stores the single fault
    // value. Every writer stores the same value, so the result does not depend on
    // order. (Perf reset R3b: the former one-block sweep of all FP32 outputs ran
    // at about 2.4 GB/s, about 13 ms of a 2K-row rank FFN.)
    if (__syncthreads_or(failed) && threadIdx.x == 0)
        *y.fault = static_cast<uint32_t>(Status::nonfinite);
}
cudaError_t reduce_async(const RouteInput& x, const RankOutput& y, cudaStream_t stream) {
    if (validate_reduce(x, y) != Status::ok) return cudaErrorInvalidValue;
    auto status = cudaMemsetAsync(y.fault, 0, 4, stream);
    if (status != cudaSuccess || !x.rows) return status;
    const uint64_t required = (y.partial_elements + 255) / 256;
    const uint32_t blocks = static_cast<uint32_t>(required < 4096 ? required : 4096);
    reduce_kernel<<<blocks, 256, 0, stream>>>(x, y);
    return cudaGetLastError();
}
#endif
} // namespace m26b1
