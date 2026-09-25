// Track P (i) arity-3 tensor-core prototype — the 3-term split-MMA GEMM core.
//
// I5-R3 ruled arity 3 (3-term exact, a67ade5). This is the correctness kernel of
// the arity-switchable TC prefill: the FP32 activation is split into three BF16
// terms (hi/mid/lo, exact), the E2M1 weight is dequantised to BF16 (the E8M0
// scale folded as a power-of-2 shift outside the MMA — here scale 127 = 1.0 so
// the MMA computes the codebook-value dot), then three m16n8k16 BF16xBF16->FP32
// MMAs per tile accumulate in FP32. Verified against the frozen FP64 dot.
// Not a timing/occupancy qualification yet.

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",#x,cudaGetErrorString(e_));exit(4);} } while(0)

namespace {
__device__ __forceinline__ float e2m1_value(uint8_t code) {
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = mag[code & 7];
    return (code & 8) ? -v : v;
}
__device__ __forceinline__ uint16_t f32_to_bf16(float a) {
    const uint32_t bits = __float_as_uint(a);
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    const uint32_t mag = bits & 0x7fffffffu;
    if (mag > 0x7f800000u) return (uint16_t)(sign | 0x7fffu);
    uint32_t r = mag + 0x7fffu + ((mag >> 16) & 1u);
    if (r >= 0x7f800000u) r = 0x7f800000u;
    return (uint16_t)(sign | (r >> 16));
}
__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    return __uint_as_float((uint32_t)b << 16);
}
__device__ __forceinline__ void split3(float a, uint16_t& hi, uint16_t& mid, uint16_t& lo) {
    hi = f32_to_bf16(a);
    mid = f32_to_bf16(a - bf16_to_f32(hi));
    lo = f32_to_bf16(a - bf16_to_f32(hi) - bf16_to_f32(mid));
}
__device__ __forceinline__ uint32_t pack2(uint16_t lo, uint16_t hi) {
    return (uint32_t(hi) << 16) | lo;
}

// m16n8k16 BF16 fragment layout (matching decode_tc_layout.h A/B/C, as driven by
// attn_decode_tc.cu load_a/load_b — the B operand of row.col is the transposed
// "col-major" K tile): g = lane/4, c = lane&3.
//   A reg r (0..3): row = g + (r&1)*8, col = 2c + (r/2)*8 (2 consecutive cols).
//   B reg r (0..1): row = g, col = 2c + r*8 (2 consecutive cols).
//   C reg r (0..3): row = g + (r/2)*8, col = 2c + (r&1).
__global__ void tc_gemm_3term(const float* __restrict__ A, const uint8_t* __restrict__ Wc,
    int K, float* __restrict__ C) {
    const int lane = threadIdx.x;
    const int g = lane >> 2, c = lane & 3;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 16) {
        // A tile: 4 registers x 2 cols, split into 3 BF16 terms.
        uint16_t hi[4][2], mid[4][2], lo[4][2];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = g + (r & 1) * 8;
            const int col = k0 + 2 * c + (r >> 1) * 8;
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                split3(A[size_t(row) * K + col + j], hi[r][j], mid[r][j], lo[r][j]);
        }
        // B tile: 2 registers (row g; cols 2c and 2c+8, 2 consecutive each), dequant.
        auto wcode = [&](int n, int k) -> uint8_t {
            return (Wc[size_t(n) * (K / 2) + k / 2] >> ((k & 1) * 4)) & 15;
        };
        const uint32_t b0 = pack2(f32_to_bf16(e2m1_value(wcode(g, k0 + 2*c))),
                                  f32_to_bf16(e2m1_value(wcode(g, k0 + 2*c + 1))));
        const uint32_t b1 = pack2(f32_to_bf16(e2m1_value(wcode(g, k0 + 2*c + 8))),
                                  f32_to_bf16(e2m1_value(wcode(g, k0 + 2*c + 9))));
        // 3 MMAs (hi/mid/lo) accumulate into the same C.
        #pragma unroll
        for (int term = 0; term < 3; ++term) {
            const uint16_t (*t)[2] = (term == 0) ? hi : (term == 1) ? mid : lo;
            const uint32_t a0 = pack2(t[0][0], t[0][1]);
            const uint32_t a1 = pack2(t[1][0], t[1][1]);
            const uint32_t a2 = pack2(t[2][0], t[2][1]);
            const uint32_t a3 = pack2(t[3][0], t[3][1]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(acc0), "+f"(acc1), "+f"(acc2), "+f"(acc3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    C[size_t(g) * 8 + 2 * c] = acc0;
    C[size_t(g) * 8 + 2 * c + 1] = acc1;
    C[size_t(g + 8) * 8 + 2 * c] = acc2;
    C[size_t(g + 8) * 8 + 2 * c + 1] = acc3;
}
} // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    const char* permit = getenv("MIMO26_BUILDER_GPU");
    if (!permit || strcmp(permit, "1") != 0) { fprintf(stderr, "GPU permission required\n"); return 2; }
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 12 || prop.minor != 1 || prop.multiProcessorCount != 48) { fprintf(stderr, "require GB10 sm121/48SM\n"); return 2; }
    const int K = 4096;
    std::vector<float> A(16 * K);
    for (int i = 0; i < 16 * K; ++i) A[i] = std::sin(float(i) * 0.01f) * 0.5f;
    std::vector<uint8_t> Wc(8 * (K / 2));
    for (int n = 0; n < 8; ++n)
        for (int k = 0; k < K; ++k) {
            const uint8_t code = uint8_t((n * 3 + k * 5 + 2) % 16);
            if (k & 1) Wc[n * (K / 2) + k / 2] |= code << 4; else Wc[n * (K / 2) + k / 2] = code;
        }
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    std::vector<float> Cref(16 * 8);
    for (int m = 0; m < 16; ++m) for (int n = 0; n < 8; ++n) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
            const uint8_t code = (Wc[n * (K / 2) + k / 2] >> ((k & 1) * 4)) & 15;
            acc += (code & 8 ? -mag[code & 7] : mag[code & 7]) * A[m * K + k];
        }
        Cref[m * 8 + n] = float(acc);
    }
    float *dA, *dC; uint8_t* dWc;
    CK(cudaMalloc(&dA, A.size() * 4)); CK(cudaMalloc(&dC, 16 * 8 * 4)); CK(cudaMalloc(&dWc, Wc.size()));
    CK(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWc, Wc.data(), Wc.size(), cudaMemcpyHostToDevice));
    tc_gemm_3term<<<1, 32>>>(dA, dWc, K, dC);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> Cgpu(16 * 8);
    CK(cudaMemcpy(Cgpu.data(), dC, 16 * 8 * 4, cudaMemcpyDeviceToHost));
    double max_rel = 0.0;
    for (int i = 0; i < 16 * 8; ++i)
        max_rel = std::max(max_rel, std::abs(double(Cgpu[i]) - Cref[i]) / (1e-5 + 1e-5 * std::abs(double(Cref[i]))));
    printf("ARITY3 CORRECT coordinates=%d max_ratio=%.6e (metric |err| <= 1e-5 + 1e-5|want|; pass <= 1.0)\n", 16 * 8, max_rel);
    if (max_rel > 1.0) { fprintf(stderr, "ARITY3 FAIL max_ratio=%g\n", max_rel); return 5; }
    puts("RESULT: PASS arity-3 TC GEMM core (correctness vs FP64; not a timing qualification)");
    return 0;
}
