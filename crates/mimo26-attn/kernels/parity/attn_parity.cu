/* attn_parity — the GPU parity harness for `crates/mimo26-attn`.
 *
 * Reads one manifest directory (produced by `tests/oracle_driver.py gen`, or by
 * a Rust test through `eval`) and runs the REAL kernels against the oracle's
 * expected tensors, printing one PASS/FAIL per case and a final
 * `RESULT: PASS` / `RESULT: FAIL` line. Built and fired ONLY by
 * `tests/gpu/run_gpu_parity.sh` (a proposed `scripts/dev.sh test attn` cell) on
 * The dev host's RTX 4090 (sm_89) or the guarded, owner-run coordinator RTX 5090 (sm_120).
 * Never the Sparks. The baked architecture/SM probe must match.
 *
 * Case-type -> kernel mapping (manifest `case` line):
 *   attn | decode            -> split-KV partials (f32 KV) + reduce, 8 splits
 *   prefill                  -> chunked prefill (f32 KV), 2048 rows/chunk
 *   decode_fp8_{unit,pth}    -> split-KV FP8 (flat AND shuffled-page-table run)
 *   prefill_fp8_{unit,pth}   -> chunked prefill FP8 (paged run)
 *   rope                     -> m26_rope_apply
 *   kv_store_{unit,pth}      -> m26_kv_store_fp8 + m26_kv_decode_fp8 (+ clip pin)
 *
 * Naive mirror for the two-run convention: `M26_NAIVE_FLAGS=<int>` selects
 * kernel bug bits (mimo26_attn_kernels.h); `MIMO26_SPIKE_NAIVE=1` sets them
 * ALL — the NEGATIVE expectations (CPU suite) then fail on this run.
 */

#include "../include/mimo26_attn_kernels.h"
#include "../include/decode_tc_layout.h"
#include "../include/decode_pipe_storage.h"
#include "../include/decode_c1_model_storage.h"
#ifndef M26_PARITY_ARCH
#define M26_PARITY_ARCH 89
#endif
__global__ void parity_arch_probe(int* p) {
#if defined(__CUDA_ARCH__)
  *p=__CUDA_ARCH__;
#endif
}
__global__ void packed_codec_probe(uint32_t* p) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<65536)p[i]=m26tc::e4m3x2_bf16(uint16_t(i));
}

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "compare.h"

namespace {

struct TensorRef {
  std::string name, dtype, file;
  std::vector<int> shape;
  bool is_expect = false;
};

struct CaseHeader {
  std::string name, ty, family;
  long long window = 0;
  int sink = 0;
  double theta = 0, partial = 0, vscale = 1, tol_abs = 0, tol_rel = 0;
  std::vector<TensorRef> tensors;
};

const TensorRef* find_tensor(const CaseHeader& c, const std::string& name) {
  for (const auto& t : c.tensors)
    if (t.name == name) return &t;
  return nullptr;
}

std::vector<std::string> split_ws(const std::string& line) {
  std::istringstream is(line);
  std::vector<std::string> f;
  std::string w;
  while (is >> w) f.push_back(w);
  return f;
}

std::vector<int> parse_shape(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(atoi(item.c_str()));
  return out;
}

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "attn_parity: cannot open %s\n", path.c_str()); exit(2); }
  std::vector<uint8_t> buf((size_t)f.tellg());
  f.seekg(0);
  f.read((char*)buf.data(), (std::streamsize)buf.size());
  return buf;
}

bool parse_manifest(const std::string& dir, std::vector<CaseHeader>* out) {
  std::ifstream f(dir + "/manifest.txt");
  if (!f) { fprintf(stderr, "attn_parity: no manifest.txt in %s\n", dir.c_str()); return false; }
  std::string line;
  bool header = false;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto fl = split_ws(line);
    if (fl.empty()) continue;
    if (fl[0] == "mimo26-attn-parity-manifest") {
      header = fl.size() == 2 && fl[1] == "1";
    } else if (fl[0] == "case") {
      if (fl.size() != 11) return false;
      CaseHeader c;
      c.name = fl[1]; c.ty = fl[2]; c.family = fl[3];
      c.window = atoll(fl[4].c_str()); c.sink = atoi(fl[5].c_str());
      c.theta = atof(fl[6].c_str()); c.partial = atof(fl[7].c_str());
      c.vscale = atof(fl[8].c_str());
      c.tol_abs = atof(fl[9].c_str()); c.tol_rel = atof(fl[10].c_str());
      out->push_back(c);
    } else if (fl[0] == "tensor" || fl[0] == "expect") {
      if (fl.size() != 5 || out->empty()) return false;
      TensorRef t;
      t.name = fl[1]; t.dtype = fl[2]; t.shape = parse_shape(fl[3]);
      t.file = fl[4]; t.is_expect = (fl[0] == "expect");
      out->back().tensors.push_back(t);
    } else {
      return false;
    }
  }
  return header;
}

/* ---------------- device helpers ---------------- */

#define CUDA_OK(call) do { cudaError_t error = (call); if (error != cudaSuccess) { \
  fprintf(stderr, "RESULT: FAIL CUDA %s at line %d: %s\n", #call, __LINE__, cudaGetErrorString(error)); \
  exit(2); } } while (0)
#define M26_CALL(fn, ...) CUDA_OK(fn(__VA_ARGS__))

// One case owns all of its device allocations. Explicit early frees remove an
// entry; remaining input allocations are reclaimed at the case boundary.
std::vector<void*> device_allocations;
void memory_guard(size_t bytes, size_t overhead = size_t(2) << 30) {
  size_t free, total; CUDA_OK(cudaMemGetInfo(&free, &total));
  const size_t reserve = (size_t(4) << 30) + overhead;
  if (free < reserve || bytes > free - reserve) {
    fprintf(stderr, "RESULT: REFUSE parity memory: free=%zu requested=%zu reserve=%zu\n", free, bytes, reserve);
    exit(3);
  }
}
template <class T> cudaError_t checked_malloc(T** p, size_t bytes) {
  if (bytes == 0) { fprintf(stderr, "RESULT: FAIL empty device allocation\n"); exit(2); }
  memory_guard(bytes);
  CUDA_OK(cudaMalloc(p, bytes));
  device_allocations.push_back(*p);
  return cudaSuccess;
}
void checked_copy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
  CUDA_OK(cudaMemcpy(dst, src, bytes, kind));
}
void checked_free(void* p) {
  if (!p) return;
  auto entry = std::find(device_allocations.begin(), device_allocations.end(), p);
  if (entry == device_allocations.end()) { fprintf(stderr, "RESULT: FAIL unowned device allocation\n"); exit(2); }
  CUDA_OK(cudaFree(p));
  device_allocations.erase(entry);
}
float* dev_output(size_t elements) {
  float* p = nullptr;
  checked_malloc(&p, elements * sizeof(float));
  // 0xffffffff is a quiet NaN. A missing/skipped/partial write must fail, even
  // if the expected tensor happens to be zero or was used as an input earlier.
  CUDA_OK(cudaMemset(p, 0xff, elements * sizeof(float)));
  return p;
}

void* dev_upload(const void* host, size_t bytes) {
  void* p = nullptr;
  if (checked_malloc(&p, bytes) != cudaSuccess) { fprintf(stderr, "cudaMalloc %zu\n", bytes); exit(2); }
  checked_copy(p, host, bytes, cudaMemcpyHostToDevice);
  return p;
}

template <typename T>
std::vector<T> as_vec(const std::vector<uint8_t>& raw) {
  std::vector<T> v(raw.size() / sizeof(T));
  memcpy(v.data(), raw.data(), v.size() * sizeof(T));
  return v;
}

struct F32Tensor {
  std::vector<float> host;
  std::vector<int> shape;
  float* dev = nullptr;
};

F32Tensor load_f32(const std::string& dir, const TensorRef& t) {
  F32Tensor out;
  out.shape = t.shape;
  out.host = as_vec<float>(read_file(dir + "/" + t.file));
  out.dev = (float*)dev_upload(out.host.data(), out.host.size() * sizeof(float));
  return out;
}

using m26_parity::within_tol;

/* ---------------- case runners ---------------- */

int g_fails = 0;
bool g_use_tc = false;
bool g_bf16q = false;
bool g_c1 = false;
bool g_p1 = false;
const char* g_impl = "baseline";
int g_pipe_warps = 0;
int g_tc_launches = 0;

void report(const CaseHeader& c, const char* what, bool ok, double worst) {
  printf("case %s [%s]: %s (max diff %.3e) %s\n", c.name.c_str(), c.ty.c_str(),
         ok ? "PASS" : "FAIL", worst, what);
  if (!ok) g_fails++;
}

m26_geom geom_of(const F32Tensor& q, const F32Tensor& k, const CaseHeader& c) {
  m26_geom g;
  g.n_q = q.shape[1];
  g.n_kv = k.shape[1];
  g.d_qk = q.shape[2];
  g.d_v = 128;
  g.window = (c.family == "ga") ? 0 : c.window;
  g.value_scale = 1.0; /* tensors are CACHED V (T18) */
  if (g.n_q % g.n_kv != 0 || g.d_qk != 192 || g.d_v != 128) {
    fprintf(stderr, "attn_parity %s: unexpected dims (n_q %d n_kv %d d_qk %d d_v %d)\n",
            c.name.c_str(), g.n_q, g.n_kv, g.d_qk, g.d_v);
    exit(2);
  }
  return g;
}

uint32_t naive_of(int argc, char** argv) {
  if (const char* f = getenv("M26_NAIVE_FLAGS")) return (uint32_t)strtoul(f, nullptr, 0);
  if (const char* n = getenv("MIMO26_SPIKE_NAIVE")) {
    if (atoi(n) == 1) {
      return 0x3FFFu; /* every kernel bug bit */
    }
  }
  return 0;
}

void run_attn_f32(const std::string& dir, const CaseHeader& c, uint32_t naive) {
  F32Tensor q = load_f32(dir, *find_tensor(c, "q"));
  F32Tensor k = load_f32(dir, *find_tensor(c, "k"));
  F32Tensor v = load_f32(dir, *find_tensor(c, "v"));
  auto q_pos = as_vec<int64_t>(read_file(dir + "/" + find_tensor(c, "q_pos")->file));
  auto k_pos = as_vec<int64_t>(read_file(dir + "/" + find_tensor(c, "k_pos")->file));
  F32Tensor sink{};
  bool has_sink = c.sink && find_tensor(c, "sink");
  if (has_sink) sink = load_f32(dir, *find_tensor(c, "sink"));
  auto exp = as_vec<float>(read_file(dir + "/" + find_tensor(c, "o")->file));

  m26_geom g = geom_of(q, k, c);
  int T = q.shape[0];
  int S = k.shape[0];
  int64_t* d_qpos = (int64_t*)dev_upload(q_pos.data(), q_pos.size() * 8);
  int64_t* d_kpos = (int64_t*)dev_upload(k_pos.data(), k_pos.size() * 8);

  std::vector<float> got(exp.size(), 0.0f);
  bool ok = true;
  double worst = 0.0;
  if (c.ty == "prefill") {
    float* d_out = dev_output(got.size());
    M26_CALL(m26_attn_prefill_f32, &g, q.dev, k.dev, v.dev, d_qpos, d_kpos, T, S, 2048,
                         naive, has_sink ? sink.dev : nullptr, d_out, 0);
    checked_copy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost);
    ok = within_tol(got, exp, c.tol_abs, c.tol_rel, &worst);
    checked_free(d_out);
  } else {
    const int n_splits = 8;
    double* d_part = nullptr;
    size_t part_elems = (size_t)T * g.n_q * n_splits * (2 + g.d_v);
    checked_malloc(&d_part, part_elems * sizeof(double));
    float* d_out = dev_output(got.size());
    M26_CALL(m26_attn_decode_splitkv_f32, &g, q.dev, k.dev, v.dev, d_qpos, d_kpos, T, S,
                                n_splits, naive, d_part, 0);
    M26_CALL(m26_attn_reduce, &g, d_part, has_sink ? sink.dev : nullptr, T, n_splits,
                    naive, d_out, 0);
    checked_copy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost);
    ok = within_tol(got, exp, c.tol_abs, c.tol_rel, &worst);
    checked_free(d_out);
    checked_free(d_part);
  }
  report(c, "f32", ok, worst);
}

void run_attn_fp8(const std::string& dir, const CaseHeader& c, uint32_t naive) {
  bool lattice=c.name.rfind("tc_bf16q_",0)==0;
  if(g_use_tc && lattice!=g_bf16q) { fprintf(stderr,"RESULT: REFUSE Q/reference lattice mismatch\n");exit(2); }
  bool native=g_bf16q&&!(naive&M26_NAIVE_TC_IGNORE_Q_ROUND);
  auto pipe_launch=g_c1?(native?m26_attn_decode_splitkv_fp8_c1_bf16q:m26_attn_decode_splitkv_fp8_c1):
    (native?m26_attn_decode_splitkv_fp8_pipe_bf16q:m26_attn_decode_splitkv_fp8_pipe);
  F32Tensor q = load_f32(dir, *find_tensor(c, "q"));
  if(lattice) {
    printf("REFERENCE case=%s Q=bf16-RNE-post-RoPE Q_storage=f32 K=E4M3-unit V=E4M3-unit cached_V=prescaled scope=bf16q-lattice-local\n",c.name.c_str());
    if(!g_use_tc) {
      for(float& x:q.host) { uint32_t u;std::memcpy(&u,&x,4);u=(u+0x7fff+((u>>16)&1))&0xffff0000u;std::memcpy(&x,&u,4); }
      checked_copy(q.dev,q.host.data(),q.host.size()*4,cudaMemcpyHostToDevice);
    }
  }
  auto q_pos = as_vec<int64_t>(read_file(dir + "/" + find_tensor(c, "q_pos")->file));
  auto k_pos = as_vec<int64_t>(read_file(dir + "/" + find_tensor(c, "k_pos")->file));
  auto k_codes = read_file(dir + "/" + find_tensor(c, "k_codes")->file);
  auto v_codes = read_file(dir + "/" + find_tensor(c, "v_codes")->file);
  bool pth = c.ty.size() > 4 && c.ty.compare(c.ty.size() - 4, 4, "_pth") == 0;
  F32Tensor k_scales{}, v_scales{}, sink{};
  if (pth) k_scales = load_f32(dir, *find_tensor(c, "k_scales"));
  if (pth) v_scales = load_f32(dir, *find_tensor(c, "v_scales"));
  bool has_sink = c.sink && find_tensor(c, "sink");
  if (has_sink) sink = load_f32(dir, *find_tensor(c, "sink"));
  auto exp = as_vec<float>(read_file(dir + "/" + find_tensor(c, "o")->file));

  int T = q.shape[0];
  int S = (int)k_pos.size();
  const TensorRef* kc_t = find_tensor(c, "k_codes");
  if (kc_t->shape.size() != 2 || kc_t->shape[0] != S || kc_t->shape[1] % 192 != 0) {
    fprintf(stderr, "attn_parity %s: k_codes must be [S, n_kv*192]\n", c.name.c_str());
    exit(2);
  }
  F32Tensor k_shaped{};
  k_shaped.shape = {S, kc_t->shape[1] / 192, 192};
  m26_geom g = geom_of(q, k_shaped, c);
  if (pth) {
    if (k_scales.shape.size() != 2 || k_scales.shape[0] != S ||
        k_scales.shape[1] != g.n_kv) {
      fprintf(stderr, "attn_parity %s: k_scales must be [S, n_kv]\n", c.name.c_str());
      exit(2);
    }
  }
  int64_t* d_qpos = (int64_t*)dev_upload(q_pos.data(), q_pos.size() * 8);
  int64_t* d_kpos = (int64_t*)dev_upload(k_pos.data(), k_pos.size() * 8);
  uint8_t* d_kc = (uint8_t*)dev_upload(k_codes.data(), k_codes.size());
  uint8_t* d_vc = (uint8_t*)dev_upload(v_codes.data(), v_codes.size());

  /* physical scatter for the paged run: logical row j lives at slot
   * (page_table[j / 256] * 256 + j % 256) — a real engine page table. The
   * physical buffer is page-GRANULAR (n_pages * 256 rows: the last page may be
   * partial and may land on a far slot). */
  const int page_tokens = 256;
  const int n_pages = (S + page_tokens - 1) / page_tokens;
  std::vector<int32_t> page_table(n_pages);
  for (int i = 0; i < n_pages; ++i) page_table[i] = i;
  uint32_t seed = 0x9E3779B9u ^ (uint32_t)n_pages;
  for (int i = n_pages - 1; i > 0; --i) { /* Fisher-Yates, fixed seed */
    seed = seed * 1664525u + 1013904223u;
    int j = (int)(seed % (uint32_t)(i + 1));
    int tmp = page_table[i]; page_table[i] = page_table[j]; page_table[j] = tmp;
  }
  const size_t k_row = (size_t)g.n_kv * g.d_qk;
  const size_t v_row = (size_t)g.n_kv * g.d_v;
  std::vector<uint8_t> phys_k((size_t)n_pages * page_tokens * k_row);
  std::vector<uint8_t> phys_v((size_t)n_pages * page_tokens * v_row);
  for (int j = 0; j < S; ++j) {
    int phys = page_table[j / page_tokens] * page_tokens + (j % page_tokens);
    memcpy(&phys_k[(size_t)phys * k_row], &k_codes[(size_t)j * k_row], k_row);
    memcpy(&phys_v[(size_t)phys * v_row], &v_codes[(size_t)j * v_row], v_row);
  }
  uint8_t* d_pkt = (uint8_t*)dev_upload(phys_k.data(), phys_k.size());
  uint8_t* d_pvt = (uint8_t*)dev_upload(phys_v.data(), phys_v.size());
  int32_t* d_pt = (int32_t*)dev_upload(page_table.data(), page_table.size() * 4);

  std::vector<float> got(exp.size(), 0.0f);
  float* d_out = dev_output(got.size());
  const char* split_env=getenv("MIMO26_ATTN_PARITY_SPLITS");
  const int n_splits=split_env?atoi(split_env):8;
  if(n_splits<1||n_splits>1024){fprintf(stderr,"RESULT: REFUSE parity split count\n");exit(2);}
  double* d_part = nullptr;
  size_t part_elems = (size_t)T * g.n_q * n_splits * (2 + g.d_v);
  checked_malloc(&d_part, part_elems * sizeof(double));
  const bool use_tc = g_use_tc && c.ty == "decode_fp8_unit";
  float* d_tc_part = nullptr;
  if (use_tc) d_tc_part = dev_output(part_elems);
  printf("PATH case=%s decode_impl=%s unit_fp8=%d T=%d S=%d splits=%d\n",
         c.name.c_str(), use_tc ? g_impl : "baseline", !pth, T, S, n_splits);

  bool ok = true, ok_paged = true;
  double worst = 0.0, worst_paged = 0.0;
  if (g_p1 && use_tc) {
    auto launch=native?m26_attn_prefill_fp8_tc_bf16q:m26_attn_prefill_fp8_tc;
    M26_CALL(launch,&g,q.dev,d_kc,d_vc,nullptr,0,d_qpos,d_kpos,T,S,naive,has_sink?sink.dev:nullptr,d_out,0);
    ++g_tc_launches;
    checked_copy(got.data(),d_out,got.size()*4,cudaMemcpyDeviceToHost);
    ok=within_tol(got,exp,c.tol_abs,c.tol_rel,&worst);
    CUDA_OK(cudaMemset(d_out,0xff,got.size()*sizeof(float)));
    M26_CALL(launch,&g,q.dev,d_pkt,d_pvt,d_pt,page_tokens,d_qpos,d_kpos,T,S,naive,has_sink?sink.dev:nullptr,d_out,0);
    ++g_tc_launches;
    checked_copy(got.data(),d_out,got.size()*4,cudaMemcpyDeviceToHost);
    ok_paged=within_tol(got,exp,c.tol_abs,c.tol_rel,&worst_paged);
  } else if (c.ty.rfind("prefill", 0) == 0) {
    M26_CALL(m26_attn_prefill_fp8, &g, q.dev, d_kc, pth ? k_scales.dev : nullptr, d_vc,
                         pth ? v_scales.dev : nullptr, nullptr, 0, d_qpos,
                         d_kpos, T, S, 2048, naive, has_sink ? sink.dev : nullptr,
                         d_out, 0);
    checked_copy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost);
    ok = within_tol(got, exp, c.tol_abs, c.tol_rel, &worst);
  } else if (use_tc) {
    if(g_pipe_warps) {
      M26_CALL(pipe_launch, &g, q.dev, d_kc, d_vc, nullptr, 0,
               d_qpos, d_kpos, T, S, n_splits, naive, g_pipe_warps, d_tc_part, 0);
    } else {
      M26_CALL(m26_attn_decode_splitkv_fp8_tc, &g, q.dev, d_kc, d_vc, nullptr, 0,
               d_qpos, d_kpos, T, S, n_splits, naive, d_tc_part, 0);
    }
    M26_CALL(m26_attn_reduce_tc, &g, d_tc_part, has_sink ? sink.dev : nullptr,
             T, n_splits, naive, d_out, 0);
    ++g_tc_launches;
    checked_copy(got.data(), d_out, got.size()*4, cudaMemcpyDeviceToHost);
    ok = within_tol(got, exp, c.tol_abs, c.tol_rel, &worst);
    CUDA_OK(cudaMemset(d_out, 0xff, got.size()*sizeof(float)));
    CUDA_OK(cudaMemset(d_tc_part, 0xff, part_elems*sizeof(float)));
    if(g_pipe_warps) {
      M26_CALL(pipe_launch, &g, q.dev, d_pkt, d_pvt, d_pt, page_tokens,
               d_qpos, d_kpos, T, S, n_splits, naive, g_pipe_warps, d_tc_part, 0);
    } else {
      M26_CALL(m26_attn_decode_splitkv_fp8_tc, &g, q.dev, d_pkt, d_pvt, d_pt, page_tokens,
               d_qpos, d_kpos, T, S, n_splits, naive, d_tc_part, 0);
    }
    M26_CALL(m26_attn_reduce_tc, &g, d_tc_part, has_sink ? sink.dev : nullptr,
             T, n_splits, naive, d_out, 0);
    ++g_tc_launches;
    checked_copy(got.data(), d_out, got.size()*4, cudaMemcpyDeviceToHost);
    ok_paged = within_tol(got, exp, c.tol_abs, c.tol_rel, &worst_paged);
  } else {
    M26_CALL(m26_attn_decode_splitkv_fp8, &g, q.dev, d_kc, pth ? k_scales.dev : nullptr,
                                d_vc, pth ? v_scales.dev : nullptr, nullptr, 0,
                                d_qpos, d_kpos, T, S, n_splits, naive, d_part, 0);
    M26_CALL(m26_attn_reduce, &g, d_part, has_sink ? sink.dev : nullptr, T, n_splits,
                    naive, d_out, 0);
    checked_copy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost);
    ok = within_tol(got, exp, c.tol_abs, c.tol_rel, &worst);
    /* second run through the shuffled page table (A2 paged-read pin) */
    CUDA_OK(cudaMemset(d_out, 0xff, got.size() * sizeof(float)));
    CUDA_OK(cudaMemset(d_part, 0xff, part_elems * sizeof(double)));
    M26_CALL(m26_attn_decode_splitkv_fp8, &g, q.dev, d_pkt, pth ? k_scales.dev : nullptr,
                                d_pvt, pth ? v_scales.dev : nullptr, d_pt,
                                page_tokens, d_qpos, d_kpos, T, S, n_splits,
                                naive, d_part, 0);
    M26_CALL(m26_attn_reduce, &g, d_part, has_sink ? sink.dev : nullptr, T, n_splits,
                    naive, d_out, 0);
    checked_copy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost);
    ok_paged = within_tol(got, exp, c.tol_abs, c.tol_rel, &worst_paged);
  }
  report(c, use_tc ? "tc fp8 flat" : "baseline fp8 flat", ok, worst);
  if (c.ty.rfind("decode", 0) == 0)
    report(c, use_tc ? "tc fp8 paged" : "baseline fp8 paged", ok_paged, worst_paged);

  if (d_tc_part) checked_free(d_tc_part);
  checked_free(d_out);
  checked_free(d_part);
  checked_free(d_kc);
  checked_free(d_vc);
  checked_free(d_pkt);
  checked_free(d_pvt);
  checked_free(d_pt);
}

void run_rope(const std::string& dir, const CaseHeader& c, uint32_t naive) {
  F32Tensor x = load_f32(dir, *find_tensor(c, "x"));
  auto pos = as_vec<int64_t>(read_file(dir + "/" + find_tensor(c, "pos")->file));
  auto exp = as_vec<float>(read_file(dir + "/" + find_tensor(c, "y")->file));
  int T = x.shape[0], H = x.shape[1], d = x.shape[2];
  int64_t* d_pos = (int64_t*)dev_upload(pos.data(), pos.size() * 8);
  float* d_y = dev_output(exp.size());
  M26_CALL(m26_rope_apply, c.theta, c.partial, x.dev, d_y, d_pos, T, H, d, naive, 0);
  std::vector<float> got(exp.size(), 0.0f);
  checked_copy(got.data(), d_y, got.size() * 4, cudaMemcpyDeviceToHost);
  double worst = 0.0;
  bool ok = within_tol(got, exp, c.tol_abs, c.tol_rel, &worst);
  report(c, "rope", ok, worst);
  checked_free(d_y);
  checked_free(d_pos);
}

void run_kv_store(const std::string& dir, const CaseHeader& c, uint32_t naive) {
  F32Tensor k = load_f32(dir, *find_tensor(c, "k"));
  F32Tensor v_raw = load_f32(dir, *find_tensor(c, "v_raw"));
  auto exp_k = as_vec<float>(read_file(dir + "/" + find_tensor(c, "k_dec")->file));
  auto exp_v = as_vec<float>(read_file(dir + "/" + find_tensor(c, "v_dec")->file));
  auto exp_clip = as_vec<int64_t>(read_file(dir + "/" + find_tensor(c, "clip")->file));
  m26_geom g{};
  g.n_kv = k.shape[1];
  g.d_qk = k.shape[2];
  g.d_v = v_raw.shape[2];
  g.n_q = 1;
  g.window = 0;
  g.value_scale = c.vscale; /* the STORE applies it before quantizing (T18) */
  int n_tok = k.shape[0];
  int unit = (c.ty == "kv_store_unit") ? 1 : 0;

  size_t k_codes_n = (size_t)n_tok * g.n_kv * g.d_qk;
  size_t v_codes_n = (size_t)n_tok * g.n_kv * g.d_v;
  size_t scales_n = (size_t)n_tok * g.n_kv;
  uint8_t* d_kc = nullptr; uint8_t* d_vc = nullptr;
  float* d_ks = nullptr; float* d_vs = nullptr;
  uint64_t* d_clip = nullptr;
  checked_malloc(&d_kc, k_codes_n);
  checked_malloc(&d_vc, v_codes_n);
  if (!unit) { checked_malloc(&d_ks, scales_n * 4); checked_malloc(&d_vs, scales_n * 4); }
  checked_malloc(&d_clip, 8);
  M26_CALL(m26_kv_store_fp8, &g, k.dev, v_raw.dev, n_tok, unit, naive, d_kc, d_ks, d_vc,
                   d_vs, d_clip, 0);
  float* d_ko = dev_output(exp_k.size());
  float* d_vo = dev_output(exp_v.size());
  M26_CALL(m26_kv_decode_fp8, &g, d_kc, d_ks, d_vc, d_vs, n_tok, naive, d_ko, d_vo, 0);
  std::vector<float> got_k(exp_k.size(), 0.0f), got_v(exp_v.size(), 0.0f);
  checked_copy(got_k.data(), d_ko, got_k.size() * 4, cudaMemcpyDeviceToHost);
  checked_copy(got_v.data(), d_vo, got_v.size() * 4, cudaMemcpyDeviceToHost);
  uint64_t clip = 0;
  checked_copy(&clip, d_clip, 8, cudaMemcpyDeviceToHost);

  double w1 = 0.0, w2 = 0.0;
  bool ok1 = within_tol(got_k, exp_k, c.tol_abs, c.tol_rel, &w1);
  bool ok2 = within_tol(got_v, exp_v, c.tol_abs, c.tol_rel, &w2);
  bool ok3 = (int64_t)clip == exp_clip[0]; /* amax clip gate: exact count */
  report(c, "kv k_dec", ok1, w1);
  report(c, "kv v_dec (T18 codes)", ok2, w2);
  printf("case %s [kv clip]: %s (got %llu want %lld)\n", c.name.c_str(),
         ok3 ? "PASS" : "FAIL", (unsigned long long)clip, (long long)exp_clip[0]);
  if (!ok3) g_fails++;
  checked_free(d_kc); checked_free(d_vc); checked_free(d_ks); checked_free(d_vs);
  checked_free(d_clip); checked_free(d_ko); checked_free(d_vo);
}

} /* namespace */

// C1 hardware handshake/drain probes, separate from the external-oracle corpus.
// Constant cached V has an analytic answer; invalid-score cases must fully drain
// and poison every output. No timing or numerical-envelope promotion here.
void c1_protocol_probe() {
  const char* names[]={"empty","empty-splits","tail-wrap","invisible-nan","alternating-nan","poison-first","poison-last","query-nan"};
  auto launch=g_bf16q?m26_attn_decode_splitkv_fp8_c1_bf16q:m26_attn_decode_splitkv_fp8_c1;
  m26_geom g{64,4,192,128,0,1.0};
  for(int test=0;test<8;++test) {
    int S=test==0?0:test==1?7:145,splits=test==0?3:test==1?11:1,rows=std::max(S,1);
    bool poison_case=test>=5;
    std::vector<float> q(64*192,test==7?NAN:0.25f),out(64*128);
    std::vector<uint8_t> k(rows*4*192,0x38),v(rows*4*128,0x30); // 1.0 and already-prescaled 0.5.
    std::vector<int64_t> kp(rows,0);int64_t qp=1;
    for(int row=0;row<S;++row) {
      bool invisible=test==3||(test==4&&row%2==0);kp[row]=invisible?2:0;
      if(invisible) {
        std::fill(k.begin()+row*4*192,k.begin()+(row+1)*4*192,0x7f);
        std::fill(v.begin()+row*4*128,v.begin()+(row+1)*4*128,0x7f);
      }
      if((test==5&&row==0)||(test==6&&row==S-1))for(int h=0;h<4;++h)k[(row*4+h)*192]=0x7f;
    }
    int64_t count=int64_t(64)*splits;std::vector<float> partial(count*130);
    float *dq=nullptr,*dp=nullptr,*dout=nullptr;uint8_t *dk=nullptr,*dv=nullptr;int64_t *dqp=nullptr,*dkp=nullptr;
    checked_malloc(&dq,q.size()*4);checked_malloc(&dp,partial.size()*4);checked_malloc(&dout,out.size()*4);
    checked_malloc(&dk,k.size());checked_malloc(&dv,v.size());checked_malloc(&dqp,8);checked_malloc(&dkp,kp.size()*8);
    checked_copy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice);checked_copy(dk,k.data(),k.size(),cudaMemcpyHostToDevice);
    checked_copy(dv,v.data(),v.size(),cudaMemcpyHostToDevice);checked_copy(dqp,&qp,8,cudaMemcpyHostToDevice);
    checked_copy(dkp,kp.data(),kp.size()*8,cudaMemcpyHostToDevice);CUDA_OK(cudaMemset(dp,0xff,partial.size()*4));
    CUDA_OK(launch(&g,dq,dk,dv,nullptr,0,dqp,dkp,1,S,splits,0,8,dp,0));
    CUDA_OK(m26_attn_reduce_tc(&g,dp,nullptr,1,splits,0,dout,0));
    checked_copy(out.data(),dout,out.size()*4,cudaMemcpyDeviceToHost);
    checked_copy(partial.data(),dp,partial.size()*4,cudaMemcpyDeviceToHost);
    float expected=test==0||test==3?0.f:0.5f;
    for(float x:out)if(poison_case?!std::isnan(x):(!std::isfinite(x)||std::abs(x-expected)>1e-6f)) {
      fprintf(stderr,"RESULT: FAIL C1 protocol output case=%s value=%g\n",names[test],x);exit(2);
    }
    for(size_t i=0;i<partial.size();++i) {
      float x=partial[i];bool valid=poison_case?std::isnan(x):(std::isfinite(x)||(i<size_t(count)&&x==-INFINITY));
      if(!valid){fprintf(stderr,"RESULT: FAIL C1 protocol partial case=%s index=%zu\n",names[test],i);exit(2);}
    }
    checked_free(dq);checked_free(dp);checked_free(dout);checked_free(dk);checked_free(dv);checked_free(dqp);checked_free(dkp);
    printf("C1_PROTOCOL_CASE PASS name=%s outputs=8192 partials=%zu\n",names[test],partial.size());
  }
  printf("C1_PROTOCOL PASS cases=8 outputs=65536 query=%s partial_scans=full\n",g_bf16q?"bf16q":"f32q");
}

#include "p1_protocol_probe.cuh"

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: attn_parity <manifest_dir>\n");
    return 2;
  }
  const char* impl = getenv("MIMO26_ATTN_DECODE_IMPL");
  if (impl && std::string(impl) != "tc" && std::string(impl) != "baseline" &&
      std::string(impl) != "pipe4" && std::string(impl) != "pipe8" &&
      std::string(impl) != "pipe4-bf16q" && std::string(impl) != "pipe8-bf16q" &&
      std::string(impl) != "c1" && std::string(impl) != "c1-bf16q" &&
      std::string(impl) != "p1" && std::string(impl) != "p1-bf16q") {
    fprintf(stderr,"unknown MIMO26_ATTN_DECODE_IMPL\n"); return 2;
  }
  g_impl=impl?impl:"baseline";
  g_c1=std::string(g_impl)=="c1"||std::string(g_impl)=="c1-bf16q";
  g_p1=std::string(g_impl)=="p1"||std::string(g_impl)=="p1-bf16q";
  g_bf16q=std::string(g_impl)=="pipe4-bf16q"||std::string(g_impl)=="pipe8-bf16q"||std::string(g_impl)=="c1-bf16q"||std::string(g_impl)=="p1-bf16q";
  g_pipe_warps=g_c1?8:std::string(g_impl).rfind("pipe4",0)==0?4:std::string(g_impl).rfind("pipe8",0)==0?8:0;
  g_use_tc = g_p1 || g_pipe_warps || (impl && std::string(impl) == "tc");
  cudaDeviceProp props{};CUDA_OK(cudaGetDeviceProperties(&props,0));
  int arch=props.major*10+props.minor,expected_sms=M26_PARITY_ARCH==120?170:128;
  if(arch!=M26_PARITY_ARCH||props.multiProcessorCount!=expected_sms){fprintf(stderr,"RESULT: REFUSE parity architecture/SM mismatch\n");return 3;}
  int* probe=nullptr;checked_malloc(&probe,sizeof(int));
  parity_arch_probe<<<1,1>>>(probe);CUDA_OK(cudaGetLastError());int baked=0;
  checked_copy(&baked,probe,sizeof(int),cudaMemcpyDeviceToHost);checked_free(probe);
  if(baked!=M26_PARITY_ARCH*10){fprintf(stderr,"RESULT: REFUSE parity baked arch mismatch\n");return 3;}
  printf("PARITY_AOT PASS arch=sm_%d sms=%d baked=%d impl=%s\n",arch,props.multiProcessorCount,baked,impl?impl:"baseline");
  if(g_pipe_warps||g_p1) {
    bool native=g_bf16q&&!(naive_of(argc,argv)&M26_NAIVE_TC_IGNORE_Q_ROUND);
    auto config=g_c1?(native?m26_attn_decode_c1_config_bf16q:m26_attn_decode_c1_config):
      (native?m26_attn_decode_pipe_config_bf16q:m26_attn_decode_pipe_config);
    int regs=0,ctas=0;
    if(g_p1){auto pconfig=native?m26_attn_prefill_tc_config_bf16q:m26_attn_prefill_tc_config;CUDA_OK(pconfig(&regs,&ctas));}
    else {CUDA_OK(config(g_pipe_warps,&regs,&ctas));}
    size_t shared=g_p1?(native?38976:88128):g_c1?sizeof(m26tc::WarpPipeModel<uint16_t>):sizeof(m26tc::CompactPipe<uint16_t>);
    printf("PIPE_RESOURCES warps=%d registers=%d shared_bytes=%zu active_CTAs_per_SM=%d\n",g_p1?8:g_pipe_warps,regs,shared,ctas);
    if(g_p1&&(regs>128||ctas<1)){fprintf(stderr,"RESULT: REFUSE P1 resource budget\n");return 3;}
    if(g_c1&&ctas<2){fprintf(stderr,"RESULT: REFUSE C1 capacity below two CTAs/SM\n");return 3;}
    uint32_t* dp=nullptr;checked_malloc(&dp,65536*sizeof(uint32_t));
    packed_codec_probe<<<256,256>>>(dp);CUDA_OK(cudaGetLastError());
    std::vector<uint32_t> result(65536);checked_copy(result.data(),dp,result.size()*4,cudaMemcpyDeviceToHost);checked_free(dp);
    for(unsigned i=0;i<65536;++i) {
      uint32_t expected=(m26tc::e4m3_bits(uint8_t(i))>>16)|(m26tc::e4m3_bits(uint8_t(i>>8))&0xffff0000u);
      if(result[i]!=expected){fprintf(stderr,"RESULT: FAIL packed GPU codec index=%u\n",i);return 2;}
    }
    puts("GPU_CODEC PASS pairs=65536/65536");
  }
  uint32_t naive = naive_of(argc, argv);
  if(g_c1&&!naive)c1_protocol_probe();
  if(g_p1&&!naive)p1_protocol_probe();
  if (naive) printf("naive flags: 0x%x (MIMO26_SPIKE_NAIVE run)\n", naive);
  std::vector<CaseHeader> cases;
  if (!parse_manifest(argv[1], &cases) || cases.empty()) {
    fprintf(stderr, "attn_parity: bad or empty manifest in %s\n", argv[1]);
    return 2;
  }
  for (const auto& c : cases) {
    if (c.ty == "attn" || c.ty == "decode" || c.ty == "prefill") {
      run_attn_f32(argv[1], c, naive);
    } else if (c.ty.rfind("decode_fp8", 0) == 0 || c.ty.rfind("prefill_fp8", 0) == 0) {
      run_attn_fp8(argv[1], c, naive);
    } else if (c.ty == "rope") {
      run_rope(argv[1], c, naive);
    } else if (c.ty == "kv_store_unit" || c.ty == "kv_store_pth") {
      run_kv_store(argv[1], c, naive);
    } else {
      fprintf(stderr, "attn_parity: unknown case type %s\n", c.ty.c_str());
      g_fails++;
    }
    CUDA_OK(cudaDeviceSynchronize());
    memory_guard(0, 0); // actual post-launch reserve remains at least 4 GiB
    while (!device_allocations.empty()) checked_free(device_allocations.back());
  }
  printf("TC_COVERAGE launches=%d requested=%d\n", g_tc_launches, int(g_use_tc));
  if (g_use_tc && g_tc_launches == 0) {
    fprintf(stderr,"RESULT: REFUSE no tensor decode coverage\n"); return 2;
  }
  printf("RESULT: %s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
