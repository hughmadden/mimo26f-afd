// P2 at serving lane shapes over long KV (perf reset P3): median kernel time and
// TFLOPS over the visible (query, key, head) pairs; with -DCAND, a candidate
// kernel (entry m26_attn_prefill_fp8_fa_cand, e.g. a renamed copy of
// attn_prefill_fa.cu) is timed too and compared bit for bit.
//
//   nvcc -O3 -std=c++17 --ftz=false -arch=sm_120 -I../include -I.. p2_long.cu \
//        ../attn_prefill_fa.cu -o p2_long && ./p2_long <n_kv> <window> <T> <S> [reps]
//
// Q, K and V are seeded on the device (Irwin-Hall normals; E4M3 with SATFINITE,
// so no NaN codes); the T queries are rows S-T..S-1. GA: n_kv 4, window 0;
// SWA: n_kv 8, window 128 (S = T + 127 is a lane's compacted SWA cache).
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "mimo26_attn_kernels.h"
extern "C" cudaError_t m26_attn_prefill_fp8_fa(const m26_geom*, const float*, const uint8_t*, const uint8_t*, int32_t,
                                              int32_t, int32_t, uint32_t, const float*, float*, m26_stream_t);
#ifdef CAND
extern "C" cudaError_t m26_attn_prefill_fp8_fa_cand(const m26_geom*, const float*, const uint8_t*, const uint8_t*,
                                                   int32_t, int32_t, int32_t, uint32_t, const float*, float*,
                                                   m26_stream_t);
#endif
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2);} } while (0)
__device__ unsigned hash(unsigned long long x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return unsigned(x);
}
__device__ float gauss(unsigned long long i, unsigned seed) {  // Irwin-Hall(4), unit variance
  float s = 0;
  for (int k = 0; k < 4; ++k) s += (hash(i * 4 + k + (unsigned long long)seed * 0x9E3779B97F4A7C15ULL) & 0xFFFFFF) / 16777216.f;
  return (s - 2.f) * 1.7320508f;
}
__global__ void fill_f32(float* p, size_t n, float sd, unsigned seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) p[i] = sd * gauss(i, seed);
}
__global__ void fill_e4m3(uint8_t* p, size_t n, float sd, unsigned seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = __nv_cvt_float_to_fp8(sd * gauss(i, seed), __NV_SATFINITE, __NV_E4M3);
}
int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: p2_long n_kv window T S [reps]\n"); return 2; }
  const int nkv = atoi(argv[1]), window = atoi(argv[2]), T = atoi(argv[3]), S = atoi(argv[4]);
  const int reps = argc > 5 ? atoi(argv[5]) : 5;
  const int q_row0 = S - T;
  float *q, *sink, *o1, *o2; uint8_t *k, *v;
  CK(cudaMalloc(&q, size_t(T) * 64 * 192 * 4));
  CK(cudaMalloc(&k, size_t(S) * nkv * 192));
  CK(cudaMalloc(&v, size_t(S) * nkv * 128));
  CK(cudaMalloc(&sink, 64 * 4));
  CK(cudaMalloc(&o1, size_t(T) * 64 * 128 * 4));
  CK(cudaMalloc(&o2, size_t(T) * 64 * 128 * 4));
  fill_f32<<<1024, 256>>>(q, size_t(T) * 64 * 192, 2.5f, 1);
  fill_e4m3<<<1024, 256>>>(k, size_t(S) * nkv * 192, 1.5f, 2);
  fill_e4m3<<<1024, 256>>>(v, size_t(S) * nkv * 128, 0.707f, 3);
  fill_f32<<<1, 64>>>(sink, 64, 1.f, 4);
  CK(cudaDeviceSynchronize());
  m26_geom g{}; g.n_q = 64; g.n_kv = nkv; g.d_qk = 192; g.d_v = 128; g.window = window; g.value_scale = 1.0;
  const float* sk = window > 0 ? sink : nullptr;
  double pairs = 0;
  for (int t = 0; t < T; ++t) { const long r = q_row0 + t; pairs += window > 0 ? std::min<long>(r + 1, window) : r + 1; }
  const double flops = pairs * 64 * 640;
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  auto time = [&](auto f) {
    CK(f()); CK(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < reps; ++i) {
      CK(cudaEventRecord(a)); CK(f()); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
      float x; CK(cudaEventElapsedTime(&x, a, b)); ms.push_back(x);
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
  };
  auto base = [&] { return m26_attn_prefill_fp8_fa(&g, q, k, v, T, S, q_row0, 0, sk, o1, nullptr); };
  const float t1 = time(base);
  printf("n_kv %d window %d T %d S %d: base %.3f ms %.1f TFLOPS", nkv, window, T, S, t1, flops / (t1 * 1e-3) / 1e12);
#ifdef CAND
  auto cand = [&] { return m26_attn_prefill_fp8_fa_cand(&g, q, k, v, T, S, q_row0, 0, sk, o2, nullptr); };
  const float t2 = time(cand);
  std::vector<uint32_t> h1(size_t(T) * 64 * 128), h2(h1.size());
  CK(cudaMemcpy(h1.data(), o1, h1.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h2.data(), o2, h2.size() * 4, cudaMemcpyDeviceToHost));
  size_t diff = 0;
  for (size_t i = 0; i < h1.size(); ++i) diff += h1[i] != h2[i];
  printf(" | cand %.3f ms %.1f TFLOPS (%.3fx) | bitwise differing %zu of %zu", t2, flops / (t2 * 1e-3) / 1e12, t1 / t2,
         diff, h1.size());
#endif
  printf("\n");
  return 0;
}
