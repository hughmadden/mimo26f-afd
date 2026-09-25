// B1 ordered compact-return scaffold. Numerical mode: E-W4A8-v1.
#pragma once
#include "prepared.cuh"
namespace m26b1 {
struct RouteInput {
    uint32_t rows;
    const float* unweighted; // Original token/route order [rows,8,4096].
    const float* weights;    // [rows,8], applied exactly once by this unit.
    uint64_t values, weight_elements;
};
struct RankOutput {
    float* partial;          // Full hidden FP32 rank pre-sum [rows,4096].
    uint16_t* bf16;          // Exactly 8192 payload bytes per row.
    uint64_t partial_elements, bf16_elements;
    uint32_t* fault;
};
// This does not reduce slice planes: restore each complete, UNWEIGHTED route
// output before calling. Caller must check status/fault before consuming either
// output. Outputs are not valid on a numerical fault; partial writes may occur.
Status reduce_host(const RouteInput&, const RankOutput&);
#ifdef __CUDACC__
cudaError_t reduce_async(const RouteInput&, const RankOutput&, cudaStream_t);
#endif
} // namespace m26b1
