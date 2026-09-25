// Track P (i) arity-3 prefill prototype — M64-class up/gate projection GEMM.
//
// I5-R3 ruled arity 3 (3-term exact, 6469de4 restated band 4,300–6,000 tok/s
// = ε 0.5–0.7 of the 40.6 TFLOPS 3-term-adjusted roof, central 5,100). This is
// the compute-bound core of the arity-switchable TC prefill: the FP32
// activation is split into three BF16 terms (hi/mid/lo, exact), the E2M1
// weight is dequantised to BF16 (the E8M0 scale folds as a power-of-2
// post-accumulate at 32-k granularity — here scale 127 = 1.0 so the MMA
// computes the codebook-value dot), and three m16n8k16 BF16xBF16->FP32 MMAs
// per tile accumulate in FP32.
//
// Geometry (up/gate, per EP rank): C[M x 512] = A[M x 4096] x W[512 x 4096]^T,
// M64-class per-expert batch. Block tile M_BLOCK=64 x N_BLOCK=128; 8 warps,
// each warp owns one M16 tile and eight N8 tiles (32 accumulators). cp.async
// double-buffers A (FP32) and B (raw E2M1 nibbles) so the stream overlaps the
// MMA; A is split and B is dequantised in registers. Correctness is verified
// against the frozen FP64 dot at M=64; sustained TFLOPS (real 2*M*N*K, not the
// 3x term overhead) is measured at M=2048 (32 M64 blocks) against the band.

#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",#x,cudaGetErrorString(e_));exit(4);} } while(0)

namespace {
constexpr int M_BLOCK = 64, N_BLOCK = 128, THREADS = 256;

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
__device__ __forceinline__ void mma16(float (&d)[4], uint32_t a0, uint32_t a1,
    uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// m16n8k16 row.col fragment layout (matching attn_decode_tc.cu load_a/load_b):
//   g = lane/4, c = lane&3.
//   A reg r: row = g + (r&1)*8, col = 2c + (r/2)*8 (2 consecutive cols).
//   B reg r: row = g, col = 2c + r*8 (2 consecutive cols).
//   C reg r: row = g + (r/2)*8, col = 2c + (r&1).
__global__ void tc_gemm_3term(const float* __restrict__ A, const uint8_t* __restrict__ Wc,
    int K, int N, float* __restrict__ C) {
    __shared__ float As[2][M_BLOCK][16];
    __shared__ uint8_t Br[2][N_BLOCK][8];
    const int tid = threadIdx.x;
    const int lane = tid & 31, w = tid >> 5;
    const int wm = w >> 1, wn = w & 1;         // M16 tile index (0..3), N64 group (0..1)
    const int n0 = blockIdx.x * N_BLOCK;       // N offset of this block
    const int m0 = blockIdx.y * M_BLOCK;       // M offset of this block
    const int g = lane >> 2, c = lane & 3;
    float acc[8][4] = {};

    // Prologue: issue cp.async for tile 0 (A: 16-byte chunks, B: 8-byte rows).
    {
        const int r = tid >> 2, kc = (tid & 3) * 4;      // 256 threads x 16 B = M_BLOCK*16 floats
        __pipeline_memcpy_async(&As[0][r][kc], &A[(size_t)(m0 + r) * K + kc], 16);
    }
    if (tid < N_BLOCK)
        __pipeline_memcpy_async(&Br[0][tid][0], &Wc[(size_t)(n0 + tid) * (K / 2)], 8);
    __pipeline_commit();

    const int nsteps = K / 16;
    for (int s = 0; s < nsteps; ++s) {
        const int buf = s & 1, nbuf = buf ^ 1;
        const int k0 = s * 16;
        if (s + 1 < nsteps) {
            {
                const int r = tid >> 2, kc = (tid & 3) * 4;
                __pipeline_memcpy_async(&As[nbuf][r][kc], &A[(size_t)(m0 + r) * K + (k0 + 16) + kc], 16);
            }
            if (tid < N_BLOCK)
                __pipeline_memcpy_async(&Br[nbuf][tid][0], &Wc[(size_t)(n0 + tid) * (K / 2) + ((k0 + 16) >> 1)], 8);
            __pipeline_commit();
        }
        __pipeline_wait_prior(s + 1 < nsteps ? 1 : 0);
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int n = wn * 64 + t * 8;     // N8 tile offset within the block
            uint16_t hi[4][2], mi[4][2], lo[4][2];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = g + (r & 1) * 8;
                const int col = 2 * c + (r >> 1) * 8;
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const float a = As[buf][wm * 16 + row][col + j];
                    uint16_t h, m, l; split3(a, h, m, l);
                    hi[r][j] = h; mi[r][j] = m; lo[r][j] = l;
                }
            }
            const int brow = n + g;
            const uint8_t b0b = Br[buf][brow][c];
            const uint8_t b1b = Br[buf][brow][c + 4];
            const uint32_t b0 = pack2(f32_to_bf16(e2m1_value(b0b & 15)), f32_to_bf16(e2m1_value(b0b >> 4)));
            const uint32_t b1 = pack2(f32_to_bf16(e2m1_value(b1b & 15)), f32_to_bf16(e2m1_value(b1b >> 4)));
            // 3 MMAs (hi/mid/lo) accumulate — manual unroll so the split arrays
            // stay in registers (a pointer ternary over them forces a local-memory
            // spill, which was the 0.89 TFLOPS regression).
            mma16(acc[t], pack2(hi[0][0], hi[0][1]), pack2(hi[1][0], hi[1][1]),
                  pack2(hi[2][0], hi[2][1]), pack2(hi[3][0], hi[3][1]), b0, b1);
            mma16(acc[t], pack2(mi[0][0], mi[0][1]), pack2(mi[1][0], mi[1][1]),
                  pack2(mi[2][0], mi[2][1]), pack2(mi[3][0], mi[3][1]), b0, b1);
            mma16(acc[t], pack2(lo[0][0], lo[0][1]), pack2(lo[1][0], lo[1][1]),
                  pack2(lo[2][0], lo[2][1]), pack2(lo[3][0], lo[3][1]), b0, b1);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        const int n = wn * 64 + t * 8;
        const int mrow = m0 + wm * 16 + g;
        C[(size_t)mrow * N + n0 + n + 2 * c] = acc[t][0];
        C[(size_t)mrow * N + n0 + n + 2 * c + 1] = acc[t][1];
        C[(size_t)(mrow + 8) * N + n0 + n + 2 * c] = acc[t][2];
        C[(size_t)(mrow + 8) * N + n0 + n + 2 * c + 1] = acc[t][3];
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
    const int clock_khz = [&]{ int v = 0; CK(cudaDeviceGetAttribute(&v, cudaDevAttrClockRate, 0)); return v; }();

    cudaFuncAttributes attr{}; CK(cudaFuncGetAttributes(&attr, tc_gemm_3term));
    int active = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, tc_gemm_3term, THREADS, 0));
    printf("ARITY3_PF device=%s arch=121 sms=48 N=%d K=%d M_BLOCK=%d N_BLOCK=%d threads=%d registers=%d shared=%zu active_CTAs_per_SM=%d nominal_clock_khz=%d\n",
        prop.name, N, K, M_BLOCK, N_BLOCK, THREADS, attr.numRegs, attr.sharedSizeBytes, active, clock_khz);

    // ---- Correctness at M=64 (one block row) vs the frozen FP64 dot ----
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
    tc_gemm_3term<<<dim3(N / N_BLOCK, Mc / M_BLOCK), THREADS>>>(dA, dWc, K, N, dC);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> Cgpu(Mc * N);
    CK(cudaMemcpy(Cgpu.data(), dC, Mc * N * 4, cudaMemcpyDeviceToHost));
    double max_rel = 0.0;
    for (int i = 0; i < Mc * N; ++i)
        max_rel = std::max(max_rel, std::abs(double(Cgpu[i]) - Cref[i]) / (1e-5 + 1e-5 * std::abs(double(Cref[i]))));
    printf("ARITY3_PF CORRECT M=%d coordinates=%d max_ratio=%.6e (metric |err| <= 1e-5 + 1e-5|want|; pass <= 1.0)\n", Mc, Mc * N, max_rel);
    if (max_rel > 1.0) { fprintf(stderr, "ARITY3_PF FAIL max_ratio=%g\n", max_rel); return 5; }
    CK(cudaFree(dA)); CK(cudaFree(dC)); CK(cudaFree(dWc));

    // ---- Throughput at M=2048 (32 M64 block rows) against the band ----
    const int Mt = 2048;
    std::vector<float> At(Mt * K);
    for (int i = 0; i < Mt * K; ++i) At[i] = std::sin(float(i) * 0.01f) * 0.5f;
    CK(cudaMalloc(&dA, Mt * K * 4)); CK(cudaMalloc(&dC, Mt * N * 4)); CK(cudaMalloc(&dWc, Wc.size()));
    CK(cudaMemcpy(dA, At.data(), Mt * K * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWc, Wc.data(), Wc.size(), cudaMemcpyHostToDevice));
    const dim3 grid(N / N_BLOCK, Mt / M_BLOCK);
    for (int w = 0; w < 3; ++w) { tc_gemm_3term<<<grid, THREADS>>>(dA, dWc, K, N, dC); CK(cudaGetLastError()); }
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<float> times;
    for (int i = 0; i < 7; ++i) {
        CK(cudaEventRecord(e0)); tc_gemm_3term<<<grid, THREADS>>>(dA, dWc, K, N, dC); CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    const double ms = times[3];
    const double real_flop = 2.0 * Mt * N * K;
    const double sustained = real_flop / (ms * 1e-3) / 1e12;
    const double roof3 = 40.6;   // 121.7 / 3 (3-term-adjusted dense peak, 04b6367)
    const double eff = sustained / roof3;
    printf("ARITY3_PF DEV SAMPLE M=%d grid=(%d,%d) real_flops=%.0f median_ms=%.6f min_ms=%.6f max_ms=%.6f sustained_tflops=%.3f eff_vs_40p6=%.3f band_tflops=20.3-28.4 band=4300-6000tok/s nominal_clock_khz=%d note=development-timing-not-a-band-test\n",
        Mt, grid.x, grid.y, real_flop, ms, times[0], times[6], sustained, eff, clock_khz);
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1)); CK(cudaFree(dA)); CK(cudaFree(dC)); CK(cudaFree(dWc));
    if (sustained < 20.3)
        puts("RESULT: MISS arity-3 TC prefill prototype (correctness PASS vs FP64; sustained timing below the 20.3 TFLOPS band floor — development row, no promotion)");
    else
        puts("RESULT: PASS arity-3 TC prefill prototype (correctness vs FP64 + sustained TFLOPS in band; not a promotion)");
    return 0;
}
