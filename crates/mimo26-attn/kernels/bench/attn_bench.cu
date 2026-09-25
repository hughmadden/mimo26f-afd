// Synthetic one-layer benchmark. No fleet/service operations.
#include <cuda_runtime.h>
#include "../include/mimo26_attn_kernels.h"
#include "../include/decode_pipe_storage.h"
#include "../include/decode_c1_model_storage.h"
#include "metrics.h"
#include "p1_check.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef M26_BENCH_ARCH
#error AOT architecture must be explicit
#endif
#ifndef M26_BENCH_SMS
#error AOT SM count must be explicit
#endif
#ifndef M26_BENCH_COMMIT
#define M26_BENCH_COMMIT "unknown"
#endif
#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "RESULT: FAIL CUDA %s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
  exit(2); } } while (0)

int m26_attn_eta_micro(int sms);

namespace {
__host__ __device__ uint32_t mix(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16);
}
__host__ __device__ float query(uint32_t i) {
  return (int(mix(i + 11) % 33) - 16) * 0.03125f;
}
__host__ __device__ uint8_t code(uint32_t i, uint32_t seed) {
  uint32_t x = mix(i + seed);
  return uint8_t(0x20 + (x % 32)) | uint8_t((x >> 8) & 0x80);
}
// Independent host reference for the generated *normal*, finite E4M3 subset.
double decode(uint8_t c) {
  return ((c & 128) ? -1.0 : 1.0) * std::ldexp(1.0 + (c & 7) / 8.0, ((c >> 3) & 15) - 7);
}
__global__ void init_codes(uint8_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x)*blockDim.x)
    p[i] = code(uint32_t(i), seed);
}
__global__ void init_queries(float* p, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x)*blockDim.x)
    p[i] = query(uint32_t(i));
}
__global__ void init_positions(int64_t* p, int n, int start) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x) p[i] = start+i;
}
__global__ void init_pages(int32_t* p, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x) p[i] = n-1-i;
}
std::vector<void*> allocations;
template<class T> T* alloc(size_t n) {
  T* p = nullptr; CK(cudaMalloc(&p, n*sizeof(T))); allocations.push_back(p); return p;
}
void reserve_check(uint64_t needed) {
  size_t free, total; CK(cudaMemGetInfo(&free, &total));
  printf("MEMORY free_bytes=%zu total_bytes=%zu requested_bytes=%llu reserve_bytes=%llu\n",
         free, total, (unsigned long long)needed, (unsigned long long)bench::reserve_bytes);
  if (!bench::memory_fits(free, needed)) {
    fprintf(stderr, "RESULT: REFUSE insufficient memory to leave 4 GiB free\n"); exit(3);
  }
}
int env_int(const char* key, int def, int lo, int hi) {
  const char* p = getenv(key); if (!p) return def;
  char* end; long n = strtol(p, &end, 10);
  if (!*p || *end || n < lo || n > hi) { fprintf(stderr, "invalid %s\n", key); exit(2); }
  return int(n);
}
// Only the QK+PV work actually visible under the causal/window mask is credited.
void run_cell(const std::string& cell, bool proxy) {
  bool prefill = cell.rfind("prefill-", 0) == 0;
  bool swa = cell == "prefill-swa";
  int s = 0;
  if (cell == "decode-4k" || cell == "prefill-4k") s = 4096;
  else if (cell == "decode-32k" || cell == "prefill-32k" || swa) s = 32768;
  else if (cell == "decode-128k") s = 131072;
  else if (cell == "decode-1m") s = 1048576;
  else if (cell == "prefill-2k") s = 2048;
  else { fprintf(stderr, "unknown cell: %s\n", cell.c_str()); exit(2); }
  const char* impl=getenv("MIMO26_ATTN_DECODE_IMPL");
  bool c1=impl&&(!std::strcmp(impl,"c1")||!std::strcmp(impl,"c1-bf16q"));
  bool bf16q=impl&&(!std::strcmp(impl,"pipe4-bf16q")||!std::strcmp(impl,"pipe8-bf16q")||!std::strcmp(impl,"c1-bf16q"));
  int pipe_warps=c1?8:impl&&(!std::strcmp(impl,"pipe4")||!std::strcmp(impl,"pipe4-bf16q"))?4:
                 impl&&(!std::strcmp(impl,"pipe8")||!std::strcmp(impl,"pipe8-bf16q"))?8:0;
  auto pipe_config=c1?(bf16q?m26_attn_decode_c1_config_bf16q:m26_attn_decode_c1_config):
    (bf16q?m26_attn_decode_pipe_config_bf16q:m26_attn_decode_pipe_config);
  auto pipe_launch=c1?(bf16q?m26_attn_decode_splitkv_fp8_c1_bf16q:m26_attn_decode_splitkv_fp8_c1):
    (bf16q?m26_attn_decode_splitkv_fp8_pipe_bf16q:m26_attn_decode_splitkv_fp8_pipe);
  int mma_tile_n=c1?16:32;
  bool tc=pipe_warps || (impl && !std::strcmp(impl,"tc"));
  if ((impl && std::strcmp(impl,"baseline") && !tc) || (tc && (prefill || swa))) {
    fprintf(stderr,"RESULT: REFUSE unsupported benchmark implementation/cell\n"); exit(3);
  }
  int t = prefill ? 2048 : 1;
  int splits = env_int("MIMO26_ATTN_BENCH_SPLITS", 0, 0, 4096);
  if (!splits) splits = tc ? bench::tc_default_splits(s) : 256;
  int warmup = env_int("MIMO26_ATTN_BENCH_WARMUP", 2, 1, 20);
  int samples = env_int("MIMO26_ATTN_BENCH_SAMPLES", 5, 3, 101);
  m26_geom g{64, swa ? 8 : 4, 192, 128, swa ? 128 : 0, 1.0};
  int pipe_registers=0,pipe_ctas=0;
  if(pipe_warps) {
    CK(pipe_config(pipe_warps,&pipe_registers,&pipe_ctas));
    if(pipe_ctas<2){std::fprintf(stderr,"RESULT: REFUSE pipeline requires at least two theoretical CTAs per SM\n");std::exit(3);}
  }
  uint64_t bytes = bench::kv_bytes(s, g.n_kv);
  // Conservative allowance for context/module/local-memory allocations of the
  // correctness-first baseline; recheck after warmup and every timed sample.
  // Baseline scratch remains distinct and is used for an untimed full-output
  // cross-check when TC is selected. Never reinterpret its AoS doubles as TC SoA.
  uint64_t scratch = prefill ? 0 : uint64_t(64)*splits*130*(sizeof(double)+(tc?sizeof(float):0));
  if (tc) scratch += uint64_t(64)*128*sizeof(float);
  reserve_check(bytes + uint64_t(t)*64*320*sizeof(float) + uint64_t(s+t)*8 +
                uint64_t(s/256)*4 + scratch + (uint64_t(2)<<30));
  uint8_t* k = alloc<uint8_t>(size_t(s)*g.n_kv*192);
  uint8_t* v = alloc<uint8_t>(size_t(s)*g.n_kv*128);
  float* q = alloc<float>(size_t(t)*64*192);
  float* out = alloc<float>(size_t(t)*64*128);
  int64_t* qp = alloc<int64_t>(t); int64_t* kp = alloc<int64_t>(s);
  int32_t* pages = alloc<int32_t>(s/256);
  double* part = prefill ? nullptr : alloc<double>(size_t(64)*splits*130);
  float* tc_part = tc ? alloc<float>(size_t(64)*splits*130) : nullptr;
  float* baseline_out = tc ? alloc<float>(size_t(64)*128) : nullptr;
  if (tc) CK(cudaMemset(tc_part,0xff,size_t(64)*splits*130*sizeof(float)));
  float* sink = swa ? alloc<float>(64) : nullptr;
  std::vector<float> sinks(64);
  for (int h=0; h<64; ++h) sinks[h] = (h-32)*0.0625f;
  if (sink) CK(cudaMemcpy(sink, sinks.data(), 64*sizeof(float), cudaMemcpyHostToDevice));
  init_codes<<<256,256>>>(k, size_t(s)*g.n_kv*192, 23); CK(cudaGetLastError());
  init_codes<<<256,256>>>(v, size_t(s)*g.n_kv*128, 47); CK(cudaGetLastError());
  init_queries<<<256,256>>>(q, size_t(t)*64*192); CK(cudaGetLastError());
  init_positions<<<256,256>>>(qp, t, s-t); CK(cudaGetLastError());
  init_positions<<<256,256>>>(kp, s, 0); CK(cudaGetLastError());
  init_pages<<<256,256>>>(pages, s/256); CK(cudaGetLastError());
  CK(cudaMemset(out, 0xff, size_t(t)*64*128*sizeof(float)));
  CK(cudaDeviceSynchronize());
  auto launch = [&] {
    if (prefill) CK(m26_attn_prefill_fp8(&g, q, k, nullptr, v, nullptr, pages, 256,
        qp, kp, t, s, 2048, 0, sink, out, nullptr));
    else if (tc) {
      if(pipe_warps) CK(pipe_launch(&g,q,k,v,pages,256,qp,kp,t,s,splits,0,pipe_warps,tc_part,nullptr));
      else CK(m26_attn_decode_splitkv_fp8_tc(&g,q,k,v,pages,256,qp,kp,t,s,splits,0,tc_part,nullptr));
      CK(m26_attn_reduce_tc(&g,tc_part,nullptr,t,splits,0,out,nullptr));
    } else {
      CK(m26_attn_decode_splitkv_fp8(&g, q, k, nullptr, v, nullptr, pages, 256,
          qp, kp, t, s, splits, 0, part, nullptr));
      CK(m26_attn_reduce(&g, part, nullptr, t, splits, 0, out, nullptr));
    }
  };
  printf("CELL %s label=%s Q=%s Q_storage=f32 reference=%s K_dtype=E4M3-unit V_dtype=E4M3-unit KV=E4M3-unit cached_V=prescaled n_q=64 n_kv=%d QK=192 V=128 T=%d S=%d window=%lld sink=%s splits=%d warmup=%d samples=%d paged=reverse-256 kernel=%s Q_values=BF16-exact KV_abs_max=1.875 precision_scope=bounded-synthetic pipe_warps=%d pipe_registers=%d pipe_shared_bytes=%d pipe_active_ctas=%d mma_tile_n=%d\n",
      cell.c_str(), proxy ? "PROXY" : "TARGET", bf16q?"bf16-RNE-post-RoPE":"f32",
       bf16q?"bf16q-lattice-local":"f32q", g.n_kv,t,s,(long long)g.window,
      swa ? "per-Q-head" : "absent", prefill ? 0 : splits, warmup,samples,
      c1?(bf16q?"tc-bf16q-p2-c1-w8":"tc-q3-p2-c1-w8"):
       bf16q?(pipe_warps==4?"tc-bf16q-p2-c3-w4":"tc-bf16q-p2-c3-w8"):
       pipe_warps==4?"tc-q3-p2-c3-w4":pipe_warps==8?"tc-q3-p2-c3-w8":tc?"tc-q3-p2-d01":"scalar-f64-baseline",
      pipe_warps,pipe_registers,c1?int(sizeof(m26tc::WarpPipeModel<uint16_t>)):pipe_warps?int(sizeof(m26tc::CompactPipe<uint16_t>)):0,pipe_ctas,mma_tile_n);
  auto start = std::chrono::steady_clock::now();
  auto budget = [&] {
    if (std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() > 120) {
      fprintf(stderr,"RESULT: INCOMPLETE cell exceeded 120s; no median claimed\n"); exit(4);
    }
    reserve_check(0);
  };
  for (int i=0;i<warmup;++i) { launch(); CK(cudaDeviceSynchronize()); budget(); }
  // Full finite-output scan plus independent scalar softmax at three coordinates.
  // Not a replacement for the external-oracle, two-run parity ladder.
  std::vector<float> result(size_t(t)*64*128);
  CK(cudaMemcpy(result.data(),out,result.size()*sizeof(float),cudaMemcpyDeviceToHost));
  for (float x : result) if (!std::isfinite(x)) { fprintf(stderr,"RESULT: FAIL nonfinite output\n"); exit(5); }
  double worst = 0;
  for (int c=0;c<3;++c) {
    int row = c*(t-1)/2, h = c*31, d = c*53, kh = h/(64/g.n_kv);
    int end = s-t+row+1, begin = swa ? std::max(0,end-128) : 0;
    double m = -INFINITY, l = 0, o = 0;
    auto fold = [&](double score,double value) {
      double next = std::max(m,score), a = std::isfinite(m) ? std::exp(m-next) : 0;
      double b = std::exp(score-next); o=o*a+b*value; l=l*a+b; m=next;
    };
    for (int j=begin;j<end;++j) {
      int phys=(s/256-1-j/256)*256+j%256;
      uint32_t ki=(uint32_t(phys)*g.n_kv+kh)*192;
      double dot=0;
      for (int z=0;z<192;++z) dot+=query((row*64+h)*192+z)*decode(code(ki+z,23));
      double val=decode(code((uint32_t(phys)*g.n_kv+kh)*128+d,47));
      fold(dot/std::sqrt(192.0),val);
    }
    if (swa) fold(sinks[h],0);
    double err=std::abs(result[(size_t(row)*64+h)*128+d]-o/l);
    worst=std::max(worst,err);
    if (err > 2e-5) { fprintf(stderr,"RESULT: FAIL sampled oracle diff=%.9g\n",err); exit(5); }
  }
  double baseline_worst=0;
  if (tc) {
    CK(cudaMemset(baseline_out,0xff,result.size()*sizeof(float)));
    CK(m26_attn_decode_splitkv_fp8(&g,q,k,nullptr,v,nullptr,pages,256,
        qp,kp,t,s,splits,0,part,nullptr));
    CK(m26_attn_reduce(&g,part,nullptr,t,splits,0,baseline_out,nullptr));
    std::vector<float> reference(result.size());
    CK(cudaMemcpy(reference.data(),baseline_out,reference.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for (size_t i=0;i<result.size();++i) {
      double error=std::abs(double(result[i])-reference[i]);
      if (!std::isfinite(reference[i]) || error>2e-5) {
        fprintf(stderr,"RESULT: FAIL full baseline comparison index=%zu diff=%.9g\n",i,error); exit(5);
      }
      baseline_worst=std::max(baseline_worst,error);
    }
    // Restore the candidate execution/cache state after the untimed reference.
    for (int i=0;i<warmup;++i) { launch(); CK(cudaDeviceSynchronize()); budget(); }
  }
  printf("CORRECT sampled_oracle=3/3 full_finite_scan=PASS max_abs=%.9g full_baseline=%s baseline_max_abs=%.9g\n",
      worst,tc?"8192/8192":"not-applicable",baseline_worst);
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  std::vector<float> times;
  for (int i=0;i<samples;++i) {
    budget(); CK(cudaEventRecord(a)); launch(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms,a,b)); times.push_back(ms);
    printf("SAMPLE cell=%s index=%d ms=%.6f\n",cell.c_str(),i,ms);
  }
  budget();
  double ms=bench::median(times), gb=bench::gbs(bytes,ms);
  double tf=bench::tflops(bench::flops(t,s,int(g.window)),ms);
  bool pass=prefill ? tf>=bench::target_tflops : gb>=bench::target_gbs;
  // 10 ms refers to all NINE GA layers, not the one-layer microbenchmark.
  if (!prefill && s==1048576) pass=pass && 9*ms<=10;
  printf("METRIC cell=%s label=%s median_ms=%.6f min_ms=%.6f max_ms=%.6f unique_KV_bytes=%llu useful_flops=%.0f effective_KV_GBs=%.3f pct_5090_peak=%.3f BF16_equiv_TFLOPS=%.6f GA9_extrapolated_ms=%.6f verdict=%s target_GBs=1253 target_TFLOPS=100 executed_mma_flops=%.0f executed_mma_TFLOPS=%.6f mma_work_factor=%.8f\n",
      cell.c_str(),proxy?"PROXY":"TARGET",ms,*std::min_element(times.begin(),times.end()),
      *std::max_element(times.begin(),times.end()),(unsigned long long)bytes,
      bench::flops(t,s,int(g.window)),gb,gb/1790*100,tf,prefill?0:9*ms,pass?"PASS":"MISS",
      tc?bench::tc_decode_mma_flops(s,splits,!bf16q,mma_tile_n):0,
      tc?bench::tflops(bench::tc_decode_mma_flops(s,splits,!bf16q,mma_tile_n),ms):0,
      tc?bench::tc_decode_mma_flops(s,splits,!bf16q,mma_tile_n)/bench::flops(t,s):0);
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  for (void* p:allocations) CK(cudaFree(p)); allocations.clear();
}
#include "p1_bench.cuh"
#include "op1_select.cuh"
#include "op1_cell.cuh"
#include "op1_cold.cuh"
#include "m0_calib.cuh"
} // namespace
int main(int argc,char** argv) {
  setvbuf(stdout,nullptr,_IOLBF,0);
  if (argc!=2) { fprintf(stderr,"expected one cell\n"); return 2; }
  CK(cudaSetDevice(0)); cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  int arch=p.major*10+p.minor;
  bool proxy=arch==89 && std::strstr(p.name,"4090");
  bool target=arch==120 && std::strstr(p.name,"5090");
  printf("IDENTITY gpu=%s arch=sm_%d sms=%d baked_arch=sm_%d baked_sms=%d source=%s label=%s\n",
      p.name,arch,p.multiProcessorCount,M26_BENCH_ARCH,M26_BENCH_SMS,M26_BENCH_COMMIT,proxy?"PROXY":"TARGET");
  if ((!proxy&&!target) || !bench::aot_matches(arch,p.multiProcessorCount,M26_BENCH_ARCH,M26_BENCH_SMS)) {
    fprintf(stderr,"RESULT: REFUSE AOT architecture/SM-count/device mismatch\n"); return 3;
  }
  if (target && (!getenv("MIMO26_ATTN_BUILDER_WINDOW") || std::strcmp(getenv("MIMO26_ATTN_BUILDER_WINDOW"),"1"))) {
    fprintf(stderr,"RESULT: REFUSE 5090 requires builder window acknowledgement\n"); return 3;
  }
  reserve_check(0);
  // Launch a baked device function as well as reading identity: a successful
  // property query alone does not demonstrate that the AOT image is loadable.
  float* probe = alloc<float>(1);
  init_queries<<<1,32>>>(probe,1); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  float got; CK(cudaMemcpy(&got,probe,sizeof(float),cudaMemcpyDeviceToHost));
  if (got != query(0)) { fprintf(stderr,"RESULT: FAIL AOT launch readback\n"); return 5; }
  CK(cudaFree(probe)); allocations.clear();
  printf("AOT: PASS architecture, SM-count, and baked kernel launch/readback\n");
  std::string cell=argv[1];
  if (cell=="aot") { puts("RESULT: PASS positive hardware AOT gate"); return 0; }
  if (cell=="eta") return m26_attn_eta_micro(p.multiProcessorCount);
  if (cell=="op1-select") { run_op1_select(proxy); return 0; }
  if (cell=="op1-cell") { run_op1_cell(proxy); return 0; }
  if (cell=="op1-cold") { run_op1_cold(proxy); return 0; }
  if (cell=="m0") { run_m0(proxy); return 0; }
  if (cell.rfind("p1-",0)==0) { run_p1_pair(cell,proxy); return 0; }
  if (cell=="all" || cell=="decode")
    for (auto c:{"decode-4k","decode-32k","decode-128k","decode-1m"}) run_cell(c,proxy);
  if (cell=="decode-long")
    for (auto c:{"decode-128k","decode-1m"}) run_cell(c,proxy);
  if (cell=="all" || cell=="prefill")
    for (auto c:{"prefill-2k","prefill-4k","prefill-32k","prefill-swa"}) run_cell(c,proxy);
  if (cell!="all" && cell!="decode" && cell!="decode-long" && cell!="prefill") run_cell(cell,proxy);
  puts("RESULT: PASS timing harness (performance verdicts are per-cell PASS/MISS)");
}
