/* MXFP4 arithmetic decoder and aligned async staging, CUDA12.8 / CUDA13.0. */
#pragma once
#include <cuda_runtime.h>
#include "mimo26_expert_bits.h"
namespace m26x {
__device__ __forceinline__ float decode(uint32_t nibble, uint32_t scale, uint32_t naive) {
    return __uint_as_float(m26x_decode_bits(nibble, scale, naive));
}
__device__ __forceinline__ uint8_t scale_byte(const uint8_t* scales, int block, int count, uint32_t naive) {
    if (naive & M26X_NAIVE_SCALE_OFF_BY_ONE) ++block;
    return block < count ? scales[block] : uint8_t(0);
}
__device__ __forceinline__ float unpack(const uint8_t* w, const uint8_t* s, int k, int cols, uint32_t naive) {
    const uint32_t byte = w[k / 2];
    const int parity = (k & 1) ^ !!(naive & M26X_NAIVE_NIBBLE_SWAP);
    return decode((byte >> (4 * parity)) & 15u, scale_byte(s, k / 32, cols / 32, naive), naive);
}
__device__ __forceinline__ float round_acc(float value, uint32_t naive) {
    return naive & M26X_NAIVE_BF16_ACCUM
        ? __uint_as_float(uint32_t(m26x_bf16_bits(__float_as_uint(value))) << 16) : value;
}
__device__ __forceinline__ void store(void* out, uint64_t index, float value, int dtype) {
    if (dtype == 0) static_cast<float*>(out)[index] = value;
    else static_cast<uint16_t*>(out)[index] = m26x_bf16_bits(__float_as_uint(value));
}
/* Source is always a valid aligned pointer, even for a zero-filled token tail. */
__device__ __forceinline__ void copy16(float* dst, const float* src, bool valid) {
    const uint32_t shared = uint32_t(__cvta_generic_to_shared(dst));
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::
                 "r"(shared), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void commit_copies() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void wait_one() {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
}
__device__ __forceinline__ void wait_all() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}
} // namespace m26x
