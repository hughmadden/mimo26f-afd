// Independent implementation of docs/design/lattice-v1.md section 4.
// Written from the specification BEFORE reading the Python reference codec.
// Numerical mode: E-W4A8-v1. Codec: e4m3fn-k32-v1. No borrowed quantizer body.
#pragma once
#include "prepared.cuh"
#include <cstring>
namespace m26b1 {
constexpr const char* quantizer_version = "e4m3fn-k32-v1";
constexpr uint32_t quant_floor_bits = 0x38d1b717u; // IEEE FP32(1e-4), scale floor only.
enum class QuantFault : uint32_t { none = 0, nonfinite_input = 1, reconstruction_overflow = 2 };
struct QuantInput {
    const float* values;      // Consecutive independent K32 blocks, immutable.
    uint64_t elements;
};
struct QuantOutput {
    uint8_t* payload;         // Same element count as input, E4M3FN bytes.
    uint8_t* scales;          // One UE8M0 scale per block, bytes 105..247.
    uint32_t* block_faults;   // One uniquely owned status per block; no atomics.
    uint64_t payload_bytes, scale_bytes, fault_elements;
};
M26B1_HD inline uint32_t quant_float_bits(float value) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(value);
#else
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#endif
}
// The floor makes amax normal. frexp(a) has m <= .875 iff the normalized
// significand is <= 1.75, i.e. fraction <= 0x600000. e = unbiased_exp + 1.
M26B1_HD inline int quant_scale_exponent(uint32_t amax_bits) {
    if (amax_bits < quant_floor_bits) amax_bits = quant_floor_bits;
    const int exponent = int(amax_bits >> 23) - 127;
    return exponent - ((amax_bits & 0x7fffffu) <= 0x600000u ? 8 : 7);
}
M26B1_HD inline uint32_t quant_round_right(uint32_t value, unsigned shift) {
    // All callers supply a <=24-bit significand and a positive shift.
    if (shift >= 32) return 0;
    const uint32_t integer = value >> shift;
    const uint32_t remainder = value & ((uint32_t(1) << shift) - 1);
    const uint32_t half = uint32_t(1) << (shift - 1);
    return integer + (remainder > half || (remainder == half && (integer & 1)));
}
// Exact integer-grid RNE. No scaling multiply, BF16 pre-round, FTZ/DAZ,
// approximate logarithm, floating conversion instruction or activation clamp.
// Precondition: finite input bits and scale exponent from the same K32 block.
M26B1_HD inline uint8_t quant_payload(uint32_t bits, int scale_exponent) {
    const uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (!magnitude) return sign; // Preserve the sign of exact zero.
    const uint32_t exponent_field = magnitude >> 23;
    uint32_t significand = magnitude & 0x7fffffu;
    int exponent;
    if (exponent_field) {
        significand |= 0x800000u;
        exponent = int(exponent_field) - 127;
    } else {
        exponent = -126;
        while (significand < 0x800000u) { significand <<= 1; --exponent; }
    }
    exponent -= scale_exponent;
    uint32_t code;
    if (exponent < -6) {
        // E4M3 subnormal quantum is 2^-9, including the code-8 carry.
        code = quant_round_right(significand, static_cast<unsigned>(14 - exponent));
    } else {
        uint32_t rounded = quant_round_right(significand, 20);
        if (rounded == 16) { rounded = 8; ++exponent; }
        code = uint32_t(exponent + 7) * 8 + rounded - 8;
        if (code > 126) code = 126; // Defensive satfinite; valid scale prevents it.
    }
    return static_cast<uint8_t>(sign | code);
}
M26B1_HD inline bool quant_reconstruction_finite(uint8_t payload, int scale_exponent) {
    const uint32_t magnitude = payload & 0x7fu;
    if (!magnitude) return true;
    const uint32_t exponent_field = magnitude >> 3;
    // Subnormal E4M3 values times any encoder scale are representable in FP32.
    return exponent_field == 0 || int(exponent_field) - 7 + scale_exponent <= 127;
}
// Scalar CPU/shared logic for a single complete block. On fault no payload or
// scale byte is written. Caller provides 32 input/output elements and 1 scale.
M26B1_HD inline QuantFault quantize_block(const float* input, uint8_t* payload, uint8_t* scale) {
    uint32_t amax = quant_floor_bits;
    for (uint32_t i = 0; i < 32; ++i) {
        const uint32_t a = quant_float_bits(input[i]) & 0x7fffffffu;
        if (a >= 0x7f800000u) return QuantFault::nonfinite_input;
        if (a > amax) amax = a;
    }
    const int k = quant_scale_exponent(amax);
    uint8_t codes[32];
    for (uint32_t i = 0; i < 32; ++i) {
        codes[i] = quant_payload(quant_float_bits(input[i]), k);
        if (!quant_reconstruction_finite(codes[i], k)) return QuantFault::reconstruction_overflow;
    }
    for (uint32_t i = 0; i < 32; ++i) payload[i] = codes[i];
    *scale = static_cast<uint8_t>(k + 127);
    return QuantFault::none;
}
Status quantize_host(const QuantInput&, const QuantOutput&);
#ifdef __CUDACC__
// Four warps per CTA, one K32 block per warp. No GPU execution is claimed by
// the host test: warp collective behavior requires a later target-SM receipt.
cudaError_t quantize_async(const QuantInput&, const QuantOutput&, cudaStream_t);
#endif
} // namespace m26b1
