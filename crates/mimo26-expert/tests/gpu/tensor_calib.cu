// GB10 tensor-core MMA calibration (MiMo F4, builder 6073d3a). Four opcode arms,
// independent-chain K sweep, exact known-answer + mutation checks, actual clock
// logging, per-SM FLOP/cycle and sustained TFLOPS. Adapted from lead M0
// (attn-replan-lead §4) to sm_121/48 SMs.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include "mxfp4_ptx.cuh"

namespace {
#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",#x,cudaGetErrorString(e_));exit(4);} } while(0)
constexpr int ARMS = 4;
const char* ARM_NAMES[ARMS] = {"bf16f32","f16f32","f16f16","fp8x4"};
// FLOP per warp-MMA = 2*M*N*K.
constexpr int MMA_K[ARMS]  = {16, 16, 16, 32};
constexpr int MMA_FLOP[ARMS] = {4096, 4096, 4096, 8192};

__device__ __forceinline__ uint32_t pack_half(float lo, float hi) {
    return (uint32_t)__half_as_ushort(__float2half_rn(lo)) | ((uint32_t)__half_as_ushort(__float2half_rn(hi)) << 16);
}
// A/B "one" bit patterns (packed), per arm. Arm3 B uses e2m1_containers.
__host__ __device__ __forceinline__ uint32_t one_u(int arm) {
    return arm == 0 ? 0x3F803F80u : arm == 3 ? 0x38383838u : 0x3C003C00u;
}
__host__ __device__ __forceinline__ uint32_t mone_u(int arm) {
    return arm == 0 ? 0xBF80BF80u : arm == 3 ? 0xB8B8B8B8u : 0xBC00BC00u;
}

template<int ARM>
__device__ __forceinline__ void mma_apply(uint32_t c[4], const uint32_t a[4], const uint32_t b[2]) {
    if (ARM == 0) {
        float* f = reinterpret_cast<float*>(c);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if (ARM == 1) {
        float* f = reinterpret_cast<float*>(c);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else if (ARM == 2) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
            : "+r"(c[0]), "+r"(c[1])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
        float (&d)[4] = *reinterpret_cast<float(*)[4]>(c);
        m26b1::mma_e4m3_e2m1<0,0>(d, a[0], a[1], a[2], a[3], b[0], b[1], 127u, 0x7F7F7F7Fu);
    }
}

// MODE 0 = positive-only, 1 = negative-only, 2 = alternating (timed).
// When `cycles` is non-null, thread 0 of each block records the SM-cycle delta
// of the timed loop (per-block) for FLOP/SM/measured-cycle reconciliation.
template<int ARM, int K, int MODE>
__global__ void m0_run(int N, uint32_t* out, unsigned long long* cycles = nullptr) {
    uint32_t ap[4], an[4], b[2];
    const uint32_t ONE = one_u(ARM), MONE = mone_u(ARM);
#pragma unroll
    for (int i = 0; i < 4; ++i) { ap[i] = ONE; an[i] = MONE; }
    if (ARM == 3) { uint2 w = m26b1::e2m1_containers(0x22222222u); b[0] = w.x; b[1] = w.y; }
    else { b[0] = b[1] = ONE; }
    uint32_t c[K][4];
    for (int k = 0; k < K; ++k) {
        if (ARM == 2) {
            c[k][0] = pack_half((float)(k * 4), (float)(k * 4 + 1));
            c[k][1] = pack_half((float)(k * 4 + 2), (float)(k * 4 + 3));
            c[k][2] = c[k][3] = 0;
        } else {
            for (int j = 0; j < 4; ++j) c[k][j] = __float_as_uint((float)(k * 4 + j));
        }
    }
    unsigned long long t0 = 0;
    if (cycles && threadIdx.x == 0) t0 = clock64();
    for (int i = 0; i < N; ++i) {
        if (MODE == 0 || MODE == 2) {
#pragma unroll
            for (int k = 0; k < K; ++k) mma_apply<ARM>(c[k], ap, b);
        }
        if (MODE == 1 || MODE == 2) {
#pragma unroll
            for (int k = 0; k < K; ++k) mma_apply<ARM>(c[k], an, b);
        }
    }
    if (cycles && threadIdx.x == 0) cycles[blockIdx.x] = clock64() - t0;
    int warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    uint32_t* dst = out + size_t(warp) * K * 4;
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < 4; ++j) dst[k * 4 + j] = c[k][j];
}

template<int ARM>
__global__ void m0_latency(int N, uint32_t* out) {
    if (blockIdx.x || threadIdx.x) return;
    uint32_t a[4], b[2]; const uint32_t ONE = one_u(ARM);
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = ONE;
    if (ARM == 3) { uint2 w = m26b1::e2m1_containers(0x22222222u); b[0] = w.x; b[1] = w.y; }
    else { b[0] = b[1] = ONE; }
    uint32_t c[4];
    if (ARM == 2) { c[0] = pack_half(1.0f, 2.0f); c[1] = pack_half(3.0f, 4.0f); c[2] = c[3] = 0; }
    else { for (int j = 0; j < 4; ++j) c[j] = __float_as_uint((float)(j + 1)); }
    unsigned long long t0 = clock64();
    for (int i = 0; i < N; ++i) mma_apply<ARM>(c, a, b);
    unsigned long long t1 = clock64();
    out[0] = (uint32_t)(t1 - t0);
    out[1] = c[0] ^ c[1];
}

double f16_to_float(uint16_t h) { return __half2float(__ushort_as_half(h)); }

template<int ARM, int K>
void config_run(int grid, int threads, int N, double clock_mhz) {
    size_t warps = size_t(grid) * (threads / 32), nout = warps * K * 4;
    uint32_t* out; CK(cudaMalloc(&out, nout * 4));
    const double inc = double(MMA_K[ARM]); // each all-ones MMA adds K to every output
    auto component = [&](int k, int j, double delta) -> double { return k * 4 + j + delta; };
    auto validate = [&](int mode, int n, const char* what) {
        CK(cudaMemset(out, 0xff, nout * 4));
        if (mode == 0) m0_run<ARM, K, 0><<<grid, threads>>>(n, out);
        else if (mode == 1) m0_run<ARM, K, 1><<<grid, threads>>>(n, out);
        else m0_run<ARM, K, 2><<<grid, threads>>>(n, out);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<uint32_t> got(nout); CK(cudaMemcpy(got.data(), out, nout * 4, cudaMemcpyDeviceToHost));
        double delta = mode == 0 ? n * inc : mode == 1 ? -n * inc : 0.0;
        for (size_t w = 0; w < warps; ++w) for (int k = 0; k < K; ++k) {
            int nj = ARM == 2 ? 2 : 4;
            for (int j = 0; j < nj; ++j) {
                uint32_t v = got[w * K * 4 + k * 4 + j];
                double gotv, wantv;
                if (ARM == 2) { gotv = f16_to_float((uint16_t)(v & 0xFFFF)); wantv = component(k, 2 * j, delta); }
                else { float f; memcpy(&f, &v, 4); gotv = f; wantv = component(k, j, delta); }
                if (gotv != wantv) { fprintf(stderr, "RESULT: FAIL M0 known-answer %s arm=%s K=%d warp=%zu k=%d j=%d got=%.6g want=%.6g\n", what, ARM_NAMES[ARM], K, w, k, j, gotv, wantv); exit(5); }
            }
        }
    };
    validate(0, 16, "positive-only");
    validate(1, 16, "negative-mutation");
    validate(2, N, "alternating");
    double exec = double(grid) * (threads / 32) * N * (2 * K) * MMA_FLOP[ARM];
    unsigned long long* cycles; CK(cudaMalloc(&cycles, grid * 8));
    for (int w = 0; w < 3; ++w) { m0_run<ARM, K, 2><<<grid, threads>>>(N, out, cycles); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<float> times;
    for (int i = 0; i < 7; ++i) {
        CK(cudaEventRecord(a)); m0_run<ARM, K, 2><<<grid, threads>>>(N, out, cycles); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b)); times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    double ms = times[3];
    double tflops = exec / (ms * 1e-3) / 1e12;
    std::vector<unsigned long long> hcyc(grid); CK(cudaMemcpy(hcyc.data(), cycles, grid * 8, cudaMemcpyDeviceToHost));
    unsigned long long maxcyc = *std::max_element(hcyc.begin(), hcyc.end());
    double per_sm_flop_cycle = exec / (48.0 * (double)maxcyc);
    printf("M0_SAMPLE arm=%s K=%d threads=%d grid=%d N=%d executed_flops=%.0f median_ms=%.6f min_ms=%.6f max_ms=%.6f tflops=%.6f max_sm_cycles=%llu flop_per_sm_cycle=%.3f clock_mhz=%.1f\n",
        ARM_NAMES[ARM], K, threads, grid, N, exec, ms, times[0], times[6], tflops, maxcyc, per_sm_flop_cycle, clock_mhz);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b)); CK(cudaFree(out)); CK(cudaFree(cycles));
}

template<int ARM>
void arm_run(double clock_mhz) {
    cudaFuncAttributes attr; CK(cudaFuncGetAttributes(&attr, m0_run<ARM, 1, 2>));
    int active = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, m0_run<ARM, 1, 2>, 128, 0));
    printf("M0_ARM arm=%s registers=%d active_CTAs_per_SM_128=%d\n", ARM_NAMES[ARM], attr.numRegs, active);
    config_run<ARM, 1>(192, 128, 8192, clock_mhz);
    config_run<ARM, 2>(192, 128, 8192, clock_mhz);
    config_run<ARM, 4>(192, 128, 8192, clock_mhz);
    config_run<ARM, 8>(192, 128, 8192, clock_mhz);
    config_run<ARM, 8>(192, 256, 8192, clock_mhz);
    config_run<ARM, 8>(384, 128, 8192, clock_mhz);
    config_run<ARM, 8>(384, 256, 8192, clock_mhz);
    config_run<ARM, 8>(384, 256, 16384, clock_mhz);
    uint32_t* lo; CK(cudaMalloc(&lo, 16));
    std::vector<int> lens{64, 128, 256, 512};
    for (int n : lens) {
        CK(cudaMemset(lo, 0, 16)); m0_latency<ARM><<<1, 32>>>(n, lo); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        uint32_t h[4]; CK(cudaMemcpy(h, lo, 16, cudaMemcpyDeviceToHost));
        printf("M0_LATENCY arm=%s N=%d cycles=%u\n", ARM_NAMES[ARM], n, h[0]);
    }
    CK(cudaFree(lo));
}
} // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    const char* permit = getenv("MIMO26_BUILDER_GPU");
    if (!permit || strcmp(permit, "1") != 0) { fprintf(stderr, "GPU permission required\n"); return 2; }
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 12 || prop.minor != 1 || prop.multiProcessorCount != 48) { fprintf(stderr, "require GB10 sm121/48SM\n"); return 2; }
    size_t free_b = 0, total = 0; CK(cudaMemGetInfo(&free_b, &total));
    if (free_b < size_t(2) * 1024 * 1024 * 1024) { fprintf(stderr, "need2GiB reserve\n"); return 2; }
    int clock_khz = 0; CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    double clock_mhz = clock_khz / 1000.0;
    printf("M0_BEGIN device=%s arch=121 sms=48 arms=4 K=1,2,4,8 threads=128,256 grids=192,384 N=8192/16384 scope=gb10-tensor-core-calibration nominal_clock_khz=%d\n", prop.name, clock_khz);
    arm_run<0>(clock_mhz);
    arm_run<1>(clock_mhz);
    arm_run<2>(clock_mhz);
    arm_run<3>(clock_mhz);
    puts("RESULT: PASS GB10 tensor-core calibration (calibration, not a ceiling certification)");
    return 0;
}
