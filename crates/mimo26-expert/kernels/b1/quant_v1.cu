// Independently derived from lattice-v1.md, not the Python reference codec.
// E-W4A8-v1 / e4m3fn-k32-v1. No atomics, clamp, or BF16 pre-round.
#include "quant_v1.cuh"
namespace m26b1 {
static Status validate_quant(const QuantInput& x, const QuantOutput& y) {
    if (x.elements % 32 || x.elements > UINT64_MAX / sizeof(float)) return Status::geometry;
    const uint64_t blocks = x.elements / 32;
    if (y.payload_bytes != x.elements || y.scale_bytes != blocks || y.fault_elements != blocks)
        return Status::size;
    if (x.elements && !span_ok(x.values, x.elements * sizeof(float), alignof(float))) return Status::pointer;
    const void* outputs[] = {y.payload, y.scales, y.block_faults};
    const uint64_t bytes[] = {x.elements, blocks, blocks * sizeof(uint32_t)};
    for (int i = 0; i < 3; ++i) {
        if (bytes[i] && !span_ok(outputs[i], bytes[i], i == 2 ? alignof(uint32_t) : 1))
            return Status::pointer;
        if (overlaps(outputs[i], bytes[i], x.values, x.elements * sizeof(float))) return Status::overlap;
        for (int j = 0; j < i; ++j)
            if (overlaps(outputs[i], bytes[i], outputs[j], bytes[j])) return Status::overlap;
    }
    return Status::ok;
}
Status quantize_host(const QuantInput& x, const QuantOutput& y) {
    const auto status = validate_quant(x, y);
    if (status != Status::ok) return status;
    bool failed = false;
    for (uint64_t block = 0; block < x.elements / 32; ++block) {
        const auto fault = quantize_block(x.values + block * 32, y.payload + block * 32, y.scales + block);
        y.block_faults[block] = static_cast<uint32_t>(fault);
        failed = failed || fault != QuantFault::none;
    }
    return failed ? Status::nonfinite : Status::ok;
}
#ifdef __CUDACC__
__global__ void quantize_kernel(QuantInput x, QuantOutput y) {
    constexpr uint32_t mask = 0xffffffffu;
    const uint32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const uint64_t warp_start = uint64_t(blockIdx.x) * 4 + warp;
    const uint64_t warp_stride = uint64_t(gridDim.x) * 4;
    for (uint64_t block = warp_start; block < x.elements / 32; block += warp_stride) {
        const uint32_t bits = quant_float_bits(x.values[block * 32 + lane]);
        uint32_t amax = bits & 0x7fffffffu;
        for (uint32_t delta = 16; delta; delta >>= 1) {
            const uint32_t other = __shfl_xor_sync(mask, amax, delta);
            if (other > amax) amax = other;
        }
        if (amax >= 0x7f800000u) {
            if (lane == 0) y.block_faults[block] = static_cast<uint32_t>(QuantFault::nonfinite_input);
            continue; // Uniform warp branch; never publish invalid block bytes.
        }
        const int k = quant_scale_exponent(amax);
        const uint8_t code = quant_payload(bits, k);
        const int overflow = __any_sync(mask, !quant_reconstruction_finite(code, k));
        if (lane == 0) y.block_faults[block] = overflow ?
            static_cast<uint32_t>(QuantFault::reconstruction_overflow) : 0;
        if (overflow) continue;
        y.payload[block * 32 + lane] = code;
        if (lane == 0) y.scales[block] = static_cast<uint8_t>(k + 127);
    }
}
cudaError_t quantize_async(const QuantInput& x, const QuantOutput& y, cudaStream_t stream) {
    if (validate_quant(x, y) != Status::ok) return cudaErrorInvalidValue;
    const uint64_t blocks = x.elements / 32;
    if (!blocks) return cudaSuccess;
    const uint64_t needed = (blocks + 3) / 4;
    const uint32_t grid = static_cast<uint32_t>(needed < 65535 ? needed : 65535);
    quantize_kernel<<<grid, 128, 0, stream>>>(x, y);
    return cudaGetLastError();
}
#endif
} // namespace m26b1
