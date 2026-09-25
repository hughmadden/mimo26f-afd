// Numerical mode: E-W4A8-v1. Representation-only preparation, no GEMM.
// Adapted from ds41rt native/cuda/kernels/v41_expert_pack.cu:8-100,
// e2a6f2ca5af56b8b567fb5086f82597427f28477; SHA256
// c72c6cee143fba9522b7485573eeb2889d49eef848a3b7b809bb7434edfbe23e.
// Copyright (c) 2026 T.J. Purtell. MIT; see LICENSE.ds41rt.
// Approved docs/REUSE.md row: Lossless N256/K128 prepare transform.
// Changes: MiMo v2 contiguous regions, exact I = 512, tagged destination, bytewise
// loads (no aliased uint32_t accesses), checked sizes and nonoverlapping spans.
#include "prepared.cuh"
namespace m26b1 {
static Status validate_prepare(const CanonicalInfo& m, const uint8_t* src,
        uint64_t src_bytes, uint8_t* dst, uint64_t dst_bytes, PreparedInfo* out) {
    auto status = check_canonical(m);
    if (status != Status::ok) return status;
    if (src_bytes != image_bytes || dst_bytes != image_bytes) return Status::size;
    if (!span_ok(src, src_bytes) || !span_ok(dst, dst_bytes, 16) ||
        !span_ok(out, sizeof(*out), alignof(PreparedInfo))) return Status::pointer;
    if (overlaps(src, src_bytes, dst, dst_bytes) ||
        overlaps(out, sizeof(*out), src, src_bytes) || overlaps(out, sizeof(*out), dst, dst_bytes))
        return Status::overlap;
    return Status::ok;
}
static PreparedInfo prepared_info(const CanonicalInfo& m) {
    return {prepared_tag, m.version, m.hidden_size, m.intermediate_size, m.rank, image_bytes};
}
Status prepare_host(const CanonicalInfo& m, const uint8_t* src, uint64_t src_bytes,
        uint8_t* dst, uint64_t dst_bytes, PreparedInfo* out) {
    auto status = validate_prepare(m, src, src_bytes, dst, dst_bytes, out);
    if (status != Status::ok) return status; // Fail before any input read/output write.
    for (uint32_t r = 0; r < 4; ++r) {
        const auto region = static_cast<Region>(r);
        for (uint64_t i = 0; i < region_bytes(region); ++i)
            dst[region_offset(region) + i] = src[canonical_byte(region, i)];
    }
    *out = prepared_info(m);
    return Status::ok;
}
#ifdef __CUDACC__
__global__ void prepare_region(const uint8_t* src, uint8_t* dst, Region region) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < region_bytes(region); i += uint64_t(gridDim.x) * blockDim.x)
        dst[region_offset(region) + i] = src[canonical_byte(region, i)];
}
cudaError_t prepare_async(const CanonicalInfo& m, const uint8_t* src, uint64_t src_bytes,
        uint8_t* dst, uint64_t dst_bytes, PreparedInfo* out, cudaStream_t stream) {
    if (validate_prepare(m, src, src_bytes, dst, dst_bytes, out) != Status::ok)
        return cudaErrorInvalidValue;
    // An enqueue failure invalidates the result; do not publish a stale tag.
    out->tag = 0;
    for (uint32_t r = 0; r < 4; ++r) {
        prepare_region<<<256, 256, 0, stream>>>(src, dst, static_cast<Region>(r));
        auto status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    *out = prepared_info(m);
    return cudaSuccess;
}
#endif
} // namespace m26b1
