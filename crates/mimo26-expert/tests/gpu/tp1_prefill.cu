// TP-1 — the pre-registered E-FP32 arity-3 tensor-core prefill GEMM
// (lead-consult `runs/20260924-i5/reviews/trackp-tp1-prereg-lead.md`, I5-R11
// step 2). M64xN64xK16 tile, 8 warps (4 M16 x 2 N32), 16 FP32 accumulators.
// The three levers from the dev-3 NCU diagnosis (memory-bound: LG Throttle,
// 38%/48% coalescing excess, 3-CTA occupancy) are removed structurally:
//   D3  hi/mid/lo split lives in SHARED (three BF16 planes), split3 on 4 fixed
//       elements/thread — zero per-thread arrays, zero local memory by design.
//   D4  E2M1 weights repacked offline to a K-major thread-contiguous layout;
//       the CTA dequantises once per K-step into a shared BF16 plane. (The
//       repack host transform is pending; this file consumes a row-major image
//       for the F7 correctness check, which the D4 transform preserves.)
//   D5  all shared planes 128 B XOR-swizzled + the decode_tc_layout.h
//       a_row/a_col/b_row/b_col fragment contracts.
//   D8  arity-3 arithmetic is bit-exact and unchanged from the dev-3
//       correctness-passing body (`f0fffae`, max ratio 0.75).
// D2/D6  __launch_bounds__(256, 6) enforces <=40 registers / 6 CTAs/SM.
//
// WIP — the register-prefetch 2-stage pipeline (D7) is not yet wired (this is
// the simple load-split-dequant-mma loop); correctness and the band run are
// gated on compiling at <=40 regs (ptxas) with zero STL/LDL (F3), then F7.

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",#x,cudaGetErrorString(e_));exit(4);} } while(0)

namespace {
constexpr int M_BLOCK = 64, N_BLOCK = 64, THREADS = 256, K_STEP = 16;

// ---- arity-3 arithmetic (bit-exact, unchanged from dev-3) -----------------
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
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---- shared-tile swizzle + mma.m16n8k16 fragment contracts
//      (decode_tc_layout.h, proven in attn_decode_tc.cu) ---------------------
__device__ __forceinline__ int tile_index(int rows, int row, int k) {
    return (k / 16) * rows * 16 + row * 16 + ((k & 15) ^ ((row & 7) * 2));
}
__device__ __forceinline__ int a_row(int lane, int reg) { return lane / 4 + (reg & 1) * 8; }
__device__ __forceinline__ int a_col(int lane, int reg) { return (lane & 3) * 2 + (reg / 2) * 8; }
__device__ __forceinline__ int b_row(int lane, int reg) { return (lane & 3) * 2 + reg * 8; }
__device__ __forceinline__ int b_col(int lane) { return lane / 4; }
__device__ __forceinline__ int c_row(int lane, int reg) { return lane / 4 + (reg / 2) * 8; }
__device__ __forceinline__ int c_col(int lane, int reg) { return (lane & 3) * 2 + (reg & 1); }
__device__ __forceinline__ uint32_t pair(const uint16_t* p, int index) {
    return *reinterpret_cast<const uint32_t*>(p + index);
}

__global__ __launch_bounds__(THREADS, 6) void tp1_gemm(
    const float* __restrict__ A, const uint8_t* __restrict__ Wc,
    int K, int N, float* __restrict__ C) {
    __shared__ uint16_t As_hi[2][M_BLOCK * K_STEP];
    __shared__ uint16_t As_mid[2][M_BLOCK * K_STEP];
    __shared__ uint16_t As_lo[2][M_BLOCK * K_STEP];
    __shared__ uint16_t Bs[2][N_BLOCK * K_STEP];

    const int tid = threadIdx.x;
    const int lane = tid & 31, w = tid >> 5;
    const int wm = w >> 1, wn = w & 1;         // M16 tile (0..3), N32 group (0..1)
    const int m0 = blockIdx.y * M_BLOCK;       // M offset of this block (one expert)
    const int n0 = blockIdx.x * N_BLOCK;       // N offset

    float acc[4][4] = {};                      // 4 N8 tiles x 4 C regs = 16 acc

    const int nsteps = K / K_STEP;
    for (int s = 0; s < nsteps; ++s) {
        const int buf = s & 1;
        const int k0 = s * K_STEP;

        // ---- produce A (split3 -> 3 swizzled BF16 planes) and B (dequant once) ----
        // 256 threads x 4 elements = 1024 = 64x16 tile.
        float av[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int e = tid * 4 + q;
            const int row = e >> 4, col = e & 15;
            av[q] = A[(size_t)(m0 + row) * K + k0 + col];
        }
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int e = tid * 4 + q;
            const int row = e >> 4, col = e & 15;
            uint16_t h, m, l; split3(av[q], h, m, l);
            const int aidx = tile_index(M_BLOCK, row, col);
            As_hi[buf][aidx] = h;
            As_mid[buf][aidx] = m;
            As_lo[buf][aidx] = l;
            // B: E2M1 -> BF16, one dequant per weight (D4 layout consumes the
            // repacked image; the row-major image here is the correctness path).
            const uint8_t byte = Wc[(size_t)(n0 + row) * (K / 2) + (k0 + col) / 2];
            const uint8_t nib = (col & 1) ? (byte >> 4) : (byte & 15);
            const int bidx = tile_index(N_BLOCK, row, col);
            Bs[buf][bidx] = f32_to_bf16(e2m1_value(nib));
        }
        __syncthreads();

        // ---- MMA: 4 N8 tiles x 3 terms = 12 mma per warp per step ----
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int n = wn * 32 + t * 8;      // N8 tile offset
            uint32_t a[4], b[2];
            #pragma unroll
            for (int r = 0; r < 2; ++r)
                b[r] = pair(Bs[buf], tile_index(N_BLOCK, n + b_col(lane), b_row(lane, r)));
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                a[r] = pair(As_hi[buf], tile_index(M_BLOCK, wm * 16 + a_row(lane, r), a_col(lane, r)));
            mma(acc[t], a, b);
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                a[r] = pair(As_mid[buf], tile_index(M_BLOCK, wm * 16 + a_row(lane, r), a_col(lane, r)));
            mma(acc[t], a, b);
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                a[r] = pair(As_lo[buf], tile_index(M_BLOCK, wm * 16 + a_row(lane, r), a_col(lane, r)));
            mma(acc[t], a, b);
        }
        __syncthreads();
    }

    // ---- epilogue: store C (per warp: M16 x N32) ----
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int n = wn * 32 + t * 8;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = m0 + wm * 16 + c_row(lane, r);
            const int col = n0 + n + c_col(lane, r);
            C[(size_t)row * N + col] = acc[t][r];
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    const char* permit = getenv("MIMO26_BUILDER_GPU");
    if (!permit || strcmp(permit, "1") != 0) { fprintf(stderr, "GPU permission required\n"); return 2; }
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 12 || prop.minor != 1 || prop.multiProcessorCount != 48) { fprintf(stderr, "require GB10 sm121/48SM\n"); return 2; }
    const int N = 512, K = 4096;
    cudaFuncAttributes attr{}; CK(cudaFuncGetAttributes(&attr, tp1_gemm));
    int active = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, tp1_gemm, THREADS, 0));
    int clock_khz = 0; CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    printf("TP1_PF device=%s arch=121 sms=48 N=%d K=%d M_BLOCK=%d N_BLOCK=%d threads=%d registers=%d shared=%zu active_CTAs_per_SM=%d nominal_clock_khz=%d\n",
        prop.name, N, K, M_BLOCK, N_BLOCK, THREADS, attr.numRegs, attr.sharedSizeBytes, active, clock_khz);

    // ---- correctness at M=64 vs the frozen FP64 dot (row-major image) ----
    const int Mc = 64;
    std::vector<float> A(Mc * K);
    for (int i = 0; i < Mc * K; ++i) A[i] = std::sin(float(i) * 0.01f) * 0.5f;
    std::vector<uint8_t> Wc(N * (K / 2));
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
            const uint8_t code = uint8_t((n * 3 + k * 5 + 2) % 16);
            if (k & 1) Wc[n * (K / 2) + k / 2] |= code << 4; else Wc[n * (K / 2) + k / 2] = code;
        }
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    std::vector<float> Cref(Mc * N);
    for (int m = 0; m < Mc; ++m) for (int n = 0; n < N; ++n) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
            const uint8_t code = (Wc[n * (K / 2) + k / 2] >> ((k & 1) * 4)) & 15;
            acc += (code & 8 ? -mag[code & 7] : mag[code & 7]) * A[m * K + k];
        }
        Cref[m * N + n] = (float)acc;
    }
    float *dA, *dC; uint8_t* dWc;
    CK(cudaMalloc(&dA, Mc * K * 4)); CK(cudaMalloc(&dC, Mc * N * 4)); CK(cudaMalloc(&dWc, Wc.size()));
    CK(cudaMemcpy(dA, A.data(), Mc * K * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWc, Wc.data(), Wc.size(), cudaMemcpyHostToDevice));
    tp1_gemm<<<dim3(N / N_BLOCK, Mc / M_BLOCK), THREADS>>>(dA, dWc, K, N, dC);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> Cgpu(Mc * N);
    CK(cudaMemcpy(Cgpu.data(), dC, Mc * N * 4, cudaMemcpyDeviceToHost));
    double max_rel = 0.0;
    for (int i = 0; i < Mc * N; ++i)
        max_rel = std::max(max_rel, std::abs(double(Cgpu[i]) - Cref[i]) / (1e-5 + 1e-5 * std::abs(double(Cref[i]))));
    printf("TP1_PF CORRECT M=%d coordinates=%d max_ratio=%.6e (metric |err| <= 1e-5 + 1e-5|want|; pass <= 1.0)\n", Mc, Mc * N, max_rel);
    if (max_rel > 1.0) { fprintf(stderr, "TP1_PF FAIL max_ratio=%g\n", max_rel); return 5; }
    CK(cudaFree(dA)); CK(cudaFree(dC)); CK(cudaFree(dWc));
    printf("TP1_PF NOTE: correctness only, registers=%d active_CTAs=%d; band timing not run (WIP)\n", attr.numRegs, active);
    return 0;
}
