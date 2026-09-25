// M0 — exact dense MMA throughput/latency calibration (lead §4, R19).
// Three opcode arms, identical m16n8k16 geometry: BF16->FP32, FP16->FP32,
// FP16->FP16. Independent-chain K sweep, exact-integer known-answer and sign
// mutations, exact FLOP count, actual clocks, per-arm residency, and a separate
// one-warp dependent-latency arm. No memory access in the timed MMA body.
#include <cuda_fp16.h>
#include "op1_cell_generated.h"

constexpr const char* ARM_NAMES[3] = {"bf16f32", "f16f32", "f16f16"};

__device__ __forceinline__ uint32_t pack_half(float lo, float hi) {
  return (uint32_t)__half_as_ushort(__float2half_rn(lo))
       | ((uint32_t)__half_as_ushort(__float2half_rn(hi)) << 16);
}
__host__ __device__ __forceinline__ uint32_t one_u(int arm) { return arm == 0 ? 0x3F803F80u : 0x3C003C00u; }
__host__ __device__ __forceinline__ uint32_t mone_u(int arm) { return arm == 0 ? 0xBF80BF80u : 0xBC00BC00u; }

template<int ARM>
__device__ __forceinline__ void mma_apply(uint32_t c[4], const uint32_t a[4], const uint32_t b[2]) {
  if constexpr (ARM == 0) {
    float* f = reinterpret_cast<float*>(c);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else if constexpr (ARM == 1) {
    float* f = reinterpret_cast<float*>(c);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
      : "+r"(c[0]), "+r"(c[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
}

// MODE 0 = positive-only, 1 = negative-only, 2 = alternating (timed).
// When `cycles` is non-null, thread 0 of each block records the SM-cycle delta
// of the timed loop (per-block), for FLOP/SM/measured-cycle reconciliation.
template<int ARM, int K, int MODE>
__global__ void m0_run(int N, uint32_t* out, unsigned long long* cycles = nullptr) {
  uint32_t ap[4], an[4], b[2];
  const uint32_t ONE = one_u(ARM), MONE = mone_u(ARM);
#pragma unroll
  for (int i = 0; i < 4; ++i) { ap[i] = ONE; an[i] = MONE; }
  b[0] = b[1] = ONE;
  uint32_t c[K][4];
  for (int k = 0; k < K; ++k) {
    if constexpr (ARM == 2) {
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
    if constexpr (MODE == 0 || MODE == 2) {
#pragma unroll
      for (int k = 0; k < K; ++k) mma_apply<ARM>(c[k], ap, b);
    }
    if constexpr (MODE == 1 || MODE == 2) {
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
  b[0] = b[1] = ONE;
  uint32_t c[4];
  if constexpr (ARM == 2) { c[0] = pack_half(1.0f, 2.0f); c[1] = pack_half(3.0f, 4.0f); c[2] = c[3] = 0; }
  else { for (int j = 0; j < 4; ++j) c[j] = __float_as_uint((float)(j + 1)); }
  unsigned long long t0 = clock64();
  for (int i = 0; i < N; ++i) mma_apply<ARM>(c, a, b);
  unsigned long long t1 = clock64();
  out[0] = (uint32_t)(t1 - t0);
  out[1] = c[0] ^ c[1];  // dependent final consumer
}

double f16_to_float(uint16_t h) { __half x = __ushort_as_half(h); return __half2float(x); }

template<int ARM, int K>
void config_run(int grid, int threads, int N) {
  size_t warps = size_t(grid) * (threads / 32), nout = warps * K * 4;
  uint32_t* out = alloc<uint32_t>(nout);
  auto component = [&](int k, int j, int delta) -> double { return k * 4 + j + delta; };
  auto validate = [&](int mode, int n, int delta, const char* what) {
    CK(cudaMemset(out, 0xff, nout * 4));
    if (mode == 0) m0_run<ARM, K, 0><<<grid, threads>>>(n, out);
    else if (mode == 1) m0_run<ARM, K, 1><<<grid, threads>>>(n, out);
    else m0_run<ARM, K, 2><<<grid, threads>>>(n, out);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<uint32_t> got(nout); CK(cudaMemcpy(got.data(), out, nout * 4, cudaMemcpyDeviceToHost));
    for (size_t w = 0; w < warps; ++w) for (int k = 0; k < K; ++k) {
      for (int j = 0; j < (ARM == 2 ? 2 : 4); ++j) {
        uint32_t v = got[w * K * 4 + k * 4 + j];
        if (ARM == 2) {
          double lo = f16_to_float((uint16_t)(v & 0xFFFF)), hi = f16_to_float((uint16_t)(v >> 16));
          if (lo != component(k, 2 * j, delta) || hi != component(k, 2 * j + 1, delta)) {
            fprintf(stderr, "RESULT: FAIL M0 known-answer %s arm=%s K=%d warp=%zu\n", what, ARM_NAMES[ARM], K, w); std::exit(5);
          }
        } else {
          float f; memcpy(&f, &v, 4);
          if ((double)f != component(k, j, delta)) {
            fprintf(stderr, "RESULT: FAIL M0 known-answer %s arm=%s K=%d warp=%zu\n", what, ARM_NAMES[ARM], K, w); std::exit(5);
          }
        }
      }
    }
  };
  validate(0, 16, 256, "positive-only");   // seed + 16*16
  validate(1, 16, -256, "negative-mutation");
  validate(2, N, 0, "alternating");
  double exec = double(grid) * (threads / 32) * N * (2 * K) * 4096;
  unsigned long long* cycles = alloc<unsigned long long>(grid);
  for (int w = 0; w < 3; ++w) { CK(cudaMemset(out, 0xff, nout * 4)); m0_run<ARM, K, 2><<<grid, threads>>>(N, out, cycles); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  std::vector<float> times;
  for (int i = 0; i < 7; ++i) {
    CK(cudaEventRecord(a)); m0_run<ARM, K, 2><<<grid, threads>>>(N, out, cycles); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b)); times.push_back(ms);
  }
  double ms = bench::median(times);
  std::vector<unsigned long long> cyc(grid); CK(cudaMemcpy(cyc.data(), cycles, grid * 8, cudaMemcpyDeviceToHost));
  unsigned long long maxcyc = *std::max_element(cyc.begin(), cyc.end());
  double per_sm = exec / (170.0 * (double)maxcyc);
  printf("M0_SAMPLE arm=%s K=%d threads=%d grid=%d N=%d executed_flops=%.0f median_ms=%.6f min_ms=%.6f max_ms=%.6f tflops=%.6f max_sm_cycles=%llu flop_per_sm_cycle=%.3f\n",
         ARM_NAMES[ARM], K, threads, grid, N, exec, ms, *std::min_element(times.begin(), times.end()), *std::max_element(times.begin(), times.end()), bench::tflops(exec, ms), maxcyc, per_sm);
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b)); CK(cudaFree(out)); CK(cudaFree(cycles));
}

template<int ARM>
void arm_run() {
  cudaFuncAttributes attr; CK(cudaFuncGetAttributes(&attr, m0_run<ARM, 1, 2>));
  int active = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, m0_run<ARM, 1, 2>, 128, 0));
  printf("M0_ARM arm=%s registers=%d active_CTAs_per_SM_128=%d\n", ARM_NAMES[ARM], attr.numRegs, active);
  config_run<ARM, 1>(680, 128, 8192);
  config_run<ARM, 2>(680, 128, 8192);
  config_run<ARM, 4>(680, 128, 8192);
  config_run<ARM, 8>(680, 128, 8192);
  config_run<ARM, 8>(680, 256, 8192);
  config_run<ARM, 8>(1360, 128, 8192);
  config_run<ARM, 8>(1360, 256, 8192);
  // N-scaling for the largest config.
  config_run<ARM, 8>(1360, 256, 16384);
  // One-warp dependent latency, several lengths.
  uint32_t* lo = alloc<uint32_t>(4);
  std::vector<unsigned long long> cycles; std::vector<int> lens{64, 128, 256, 512};
  for (int n : lens) {
    CK(cudaMemset(lo, 0, 16)); m0_latency<ARM><<<1, 32>>>(n, lo); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    uint32_t h[4]; CK(cudaMemcpy(h, lo, 16, cudaMemcpyDeviceToHost)); cycles.push_back(h[0]);
    printf("M0_LATENCY arm=%s N=%d cycles=%llu\n", ARM_NAMES[ARM], n, (unsigned long long)h[0]);
  }
  double slope = double(cycles[3] - cycles[0]) / (lens[3] - lens[0]);
  printf("M0_LATENCY_SLOPE arm=%s cycles_per_mma=%.6f\n", ARM_NAMES[ARM], slope);
  CK(cudaFree(lo));
}

void run_m0(bool proxy) {
  if (proxy) { fprintf(stderr, "RESULT: REFUSE M0 coordinator only\n"); std::exit(3); }
  reserve_check(uint64_t(2) << 30);
  int clock = 0;
  CK(cudaDeviceGetAttribute(&clock, cudaDevAttrClockRate, 0));
  printf("M0_BEGIN source=%s arms=3 K=1,2,4,8 threads=128,256 grids=680,1360 N=8192 scope=dense-mma-calibration gate=UNSET clock_khz=%d\n",
         op1cell::d7_sha, clock);
  arm_run<0>(); arm_run<1>(); arm_run<2>();
  reserve_check(0);
  puts("RESULT: PASS M0 calibration harness (calibration, not a ceiling certification)");
}
