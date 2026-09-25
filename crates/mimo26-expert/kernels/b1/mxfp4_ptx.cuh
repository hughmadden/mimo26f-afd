/* Copyright (c) 2025 by the b12x authors.
 * Licensed under the Apache License, Version 2.0; see LICENSE.b1-compute.
 * http://www.apache.org/licenses/LICENSE-2.0
 * Distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
 *
 * Approved REUSE rows 116-117: b12x/_lib/intrinsics.py:4847-4881,4059-4120
 * at 3882b935ede761d6c73a5d6fd68e690f1e3f5380, SHA256
 * 9357389a24880c8a17d2b8718df9268f55cd79b58ad7802c910f0dc77dc1ee28.
 * Changed: extracted PTX into handwritten CUDA wrappers; no Python/CuTe body.
 * E-W4A8-v1 only. Caller owns operand mapping, scale eligibility and MiMo's
 * exceptional-scale/saturation fallback. This fragment is not an FFN dispatch.
 */
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
namespace m26b1 {
// Nibble i becomes byte i, occupying bits 5:2; original order/sign preserved.
__device__ __forceinline__ uint2 e2m1_containers(uint32_t packed) {
    uint2 result;
    asm("{ .reg .b32 s,t;\n"
        "shl.b32 s, %2, 2;\n"
        "shr.u32 t, %2, 2;\n"
        "prmt.b32 %0, s, t, 0x5140;\n"
        "prmt.b32 %1, s, t, 0x7362;\n"
        "and.b32 %0, %0, 0x3C3C3C3C;\n"
        "and.b32 %1, %1, 0x3C3C3C3C; }"
        : "=r"(result.x), "=r"(result.y) : "r"(packed));
    return result;
}
// Warp-collective, all 32 lanes active. A: four E4M3 registers; B: two
// E2M1-container registers. Scale thread selectors retain upstream {byte,0}.
template<int ByteA, int ByteB>
__device__ __forceinline__ void mma_e4m3_e2m1(float (&d)[4],
        uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
        uint32_t b0,uint32_t b1,uint32_t sa,uint32_t sb) {
    static_assert(ByteA>=0 && ByteA<4 && ByteB>=0 && ByteB<4,"scale byte selector");
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%12,0}, {%11}, {%13,0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb),"n"(ByteA),"n"(ByteB));
}
} // namespace m26b1
