/* First-party, host/device arithmetic MXFP4 decoder; no constant-memory LUT.
 * Host exhaustive tests use an independent FP64/LUT reference.
 */
#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define M26X_HD __host__ __device__ __forceinline__
#else
#define M26X_HD inline
#endif
#define M26X_NAIVE_NIBBLE_SWAP 1u
#define M26X_NAIVE_E8M0_NO_CLAMP 2u
#define M26X_NAIVE_SCALE_OFF_BY_ONE 4u
#define M26X_NAIVE_PAD_ROW_READ 8u
#define M26X_NAIVE_AOT_MIXED_GATE 16u
#define M26X_NAIVE_AOT_CAPACITY_IGNORED 32u
#define M26X_NAIVE_BF16_ACCUM 64u
#define M26X_NAIVE_SCALE_ONE 128u
#define M26X_THREADS 256
#define M26X_ROWS_PER_BLOCK 32
#define M26X_LANES_PER_ROW 8
#define M26X_K_TILE 256
#define M26X_BLOCK 32

/* Returns IEEE754 bits, including signed zero, subnormals and finite saturation.
 * The lowest possible biased exponent is -1, so the subnormal shift is <=2
 * and exact (E2M1 has at most two significant bits). No FP64/SFU on device.
 */
M26X_HD uint32_t m26x_decode_bits(uint32_t nibble, uint32_t scale, uint32_t naive) {
    const uint32_t sign = (nibble & 8u) << 28;
    const uint32_t mag = nibble & 7u;
    if (!mag) return sign;
    if (naive & M26X_NAIVE_SCALE_ONE) scale = 127;
    else if (!(naive & M26X_NAIVE_E8M0_NO_CLAMP) && scale == 255) scale = 254;
    const int exponent = int(scale) + int(mag >> 1) - 1;
    const uint32_t fraction = mag >= 2 ? (mag & 1u) << 22 : 0u;
    if (exponent >= 255) return sign | 0x7f7fffffu;
    if (exponent <= 0) return sign | ((0x800000u | fraction) >> (1 - exponent));
    return sign | (uint32_t(exponent) << 23) | fraction;
}
M26X_HD int m26x_x_swizzle(int k) {
    return ((k % 32) / 4) * 32 + (k / 32) * 4 + k % 4;
}
M26X_HD int m26x_owned_row(int block, int thread) {
    return block * M26X_ROWS_PER_BLOCK + thread / M26X_LANES_PER_ROW;
}
M26X_HD uint16_t m26x_bf16_bits(uint32_t bits) {
    const uint32_t mag = bits & 0x7fffffffu;
    if (mag > 0x7f800000u) return uint16_t((bits >> 16) | 0x40u);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
