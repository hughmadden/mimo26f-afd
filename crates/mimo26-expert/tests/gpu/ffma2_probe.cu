// First-party compile-only ISA probe. No host main, launch, or timing code.
#include <cuda_runtime.h>
#include <stdint.h>
#ifndef PROBE_MODE
#error PROBE_MODE must be 0 (scalar), 1 (PTX), or 2 (intrinsic)
#endif
static_assert(PROBE_MODE >= 0 && PROBE_MODE <= 2);
__device__ __forceinline__ uint64_t pack(float2 x) {
    return uint64_t(__float_as_uint(x.x)) | (uint64_t(__float_as_uint(x.y)) << 32);
}
extern "C" __global__ void probe(const float2* a, const float2* b,
                                const float2* c, float2* out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const float2 x=a[i], y=b[i], z=c[i];
#if PROBE_MODE == 0
    out[i] = make_float2(__fmaf_rn(x.x,y.x,z.x), __fmaf_rn(x.y,y.y,z.y));
#elif PROBE_MODE == 1
    uint64_t r;
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;"
                 : "=l"(r) : "l"(pack(x)), "l"(pack(y)), "l"(pack(z)));
    out[i] = make_float2(__uint_as_float(uint32_t(r)), __uint_as_float(uint32_t(r >> 32)));
#else
    out[i] = __ffma2_rn(x,y,z);
#endif
}
