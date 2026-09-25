// OP1 full 48-layer cell: verification replay + short single-chunk prefill.
// Retained attention kernels unchanged. Reuses op1_init_q/op1_init_kv from
// op1_select.cuh and query/code/decode from attn_bench.cu. Generated schedule
// comes from op1_cell_generated.h (namespace op1cell). Selection paths frozen
// from attn-bench.8qKHcw: verification GA C3/P8, SWA C3/P4; short prefill GA
// C3/P1, SWA P1. Cold 2K-64K multi-chunk arm is a separate cell (op1_cold).
#include "op1_cell_generated.h"

constexpr int NV = 48, VER_T = 8, PRE_T = 309, FULL_SLOTS = 512, CMP_SLOTS = 256;

struct CellLayer {
  bool swa; m26_geom g; int id;
  float *q8, *qpre, *out8, *ref8, *outpre, *refpre, *sink;
  uint8_t *kfull, *vfull, *kcompact, *vcompact;
  std::vector<float> sinks;
};

unsigned qseed(int l) { return unsigned(l) * 104729u; }
unsigned kseed(int l) { return 23u + unsigned(l) * 104729u; }
unsigned vseed(int l) { return 47u + unsigned(l) * 104729u; }

double host_coordinate(const CellLayer& L, const op1cell::CellCase& c, int row, int head, int dim) {
  int nk = L.g.n_kv, kh = head * nk / 64;
  int qpos = c.phase ? row : c.prefix + row;
  int end = qpos;
  int begin = L.swa ? std::max(c.phase ? 0 : c.swa_start, qpos - 127) : 0;
  double m = -INFINITY, l = 0, o = 0;
  auto fold = [&](double score, double value) {
    double next = std::max(m, score);
    double a = std::isfinite(m) ? std::exp(m - next) : 0;
    double b = std::exp(score - next);
    o = o * a + b * value; l = l * a + b; m = next;
  };
  for (int j = begin; j <= end; ++j) {
    double dot = 0;
    for (int z = 0; z < 192; ++z)
      dot += query(uint32_t((qpos * 64 + head) * 192 + z) + qseed(L.id))
           * decode(code(uint32_t((j * nk + kh) * 192 + z), kseed(L.id)));
    fold(dot / std::sqrt(192.0), decode(code(uint32_t((j * nk + kh) * 128 + dim), vseed(L.id))));
  }
  if (L.swa) fold((head - 32) * .0625 + L.id * .00390625, 0);
  return o / l;
}

struct CellRun {
  CellLayer layer[NV];
  float *tc_partial;      // T*64*splits*130 (max 309*64*1*130)
  double *ref_partial;    // T*64*ref_splits*130 (max 309*64*2*130)
  int64_t *qpos;          // query positions (prefix..prefix+T-1), regenerated
  int64_t *kpfull, *kpc;  // full key positions 0..511; compact window positions
  int32_t *pages;         // reverse 2 pages
  unsigned *bad;

  CellRun() {
    for (int i = 0; i < NV; ++i) {
      bool swa = op1cell::layers[i] == 1;
      CellLayer& L = layer[i];
      L.swa = swa; L.id = i;
      L.g = m26_geom{64, swa ? 8 : 4, 192, 128, swa ? 128 : 0, 1.0};
      L.q8 = alloc<float>(size_t(VER_T) * 64 * 192);
      L.qpre = alloc<float>(size_t(PRE_T) * 64 * 192);
      L.out8 = alloc<float>(size_t(VER_T) * 64 * 128);
      L.ref8 = alloc<float>(size_t(VER_T) * 64 * 128);
      L.outpre = alloc<float>(size_t(PRE_T) * 64 * 128);
      L.refpre = alloc<float>(size_t(PRE_T) * 64 * 128);
      L.kfull = alloc<uint8_t>(size_t(FULL_SLOTS) * L.g.n_kv * 192);
      L.vfull = alloc<uint8_t>(size_t(FULL_SLOTS) * L.g.n_kv * 128);
      L.kcompact = alloc<uint8_t>(size_t(CMP_SLOTS) * (swa ? 8 : 1) * 192);
      L.vcompact = alloc<uint8_t>(size_t(CMP_SLOTS) * (swa ? 8 : 1) * 128);
      L.sink = swa ? alloc<float>(64) : nullptr;
      L.sinks.resize(64);
      for (int h = 0; h < 64; ++h) L.sinks[h] = (h - 32) * .0625f + i * .00390625f;
      if (L.sink) CK(cudaMemcpy(L.sink, L.sinks.data(), 64 * 4, cudaMemcpyHostToDevice));
      // kvfull: absolute-indexed, all 512 slots, first=0 (deterministic by position).
      op1_init_kv<<<64, 256>>>(L.kfull, size_t(FULL_SLOTS) * L.g.n_kv * 192, 192, L.g.n_kv, 2, 0, FULL_SLOTS, kseed(i));
      op1_init_kv<<<64, 256>>>(L.vfull, size_t(FULL_SLOTS) * L.g.n_kv * 128, 128, L.g.n_kv, 2, 0, FULL_SLOTS, vseed(i));
      CK(cudaGetLastError());
      // qpre: positions 0..PRE_T-1, prefix 0 (fixed across short-prefill cases).
      op1_init_q<<<64, 256>>>(L.qpre, size_t(PRE_T) * 64 * 192, 0, qseed(i));
      CK(cudaGetLastError());
    }
    tc_partial = alloc<float>(size_t(PRE_T) * 64 * 1 * 130);
    ref_partial = alloc<double>(size_t(PRE_T) * 64 * 2 * 130);
    qpos = alloc<int64_t>(PRE_T);
    kpfull = alloc<int64_t>(FULL_SLOTS); kpc = alloc<int64_t>(CMP_SLOTS);
    pages = alloc<int32_t>(2); bad = alloc<unsigned>(1);
    init_positions<<<4, 256>>>(kpfull, FULL_SLOTS, 0);
    init_pages<<<1, 256>>>(pages, 2);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
  }

  size_t t_of(const op1cell::CellCase& c) { return c.phase ? size_t(c.t) : VER_T; }
  float* q_of(const op1cell::CellCase& c, int l) { return c.phase ? layer[l].qpre : layer[l].q8; }
  float* out_of(const op1cell::CellCase& c, int l) { return c.phase ? layer[l].outpre : layer[l].out8; }
  float* ref_of(const op1cell::CellCase& c, int l) { return c.phase ? layer[l].refpre : layer[l].ref8; }

  void prep(const op1cell::CellCase& c) {
    init_positions<<<4, 256>>>(qpos, t_of(c), c.prefix);
    if (!c.phase) {
      for (int i = 0; i < NV; ++i) {
        CellLayer& L = layer[i];
        op1_init_q<<<64, 256>>>(L.q8, size_t(VER_T) * 64 * 192, c.prefix, qseed(i));
        if (L.swa) {
          op1_init_kv<<<64, 256>>>(L.kcompact, size_t(CMP_SLOTS) * 8 * 192, 192, 8, 1, c.swa_start, c.swa_s, kseed(i));
          op1_init_kv<<<64, 256>>>(L.vcompact, size_t(CMP_SLOTS) * 8 * 128, 128, 8, 1, c.swa_start, c.swa_s, vseed(i));
        }
      }
      init_positions<<<4, 256>>>(kpc, c.swa_s, c.swa_start);
    }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  }

  // Launch candidate for one layer; writes out (out8/outpre). mode 0=f32q, 1=bf16q.
  void launch_candidate(int l, const op1cell::CellCase& c, bool native, cudaStream_t s) {
    CellLayer& L = layer[l];
    float* q = q_of(c, l); float* out = out_of(c, l);
    if (c.phase) {
      int T = c.t, S = c.t;
      if (!L.swa) {
        auto fn = native ? m26_attn_decode_splitkv_fp8_pipe_bf16q : m26_attn_decode_splitkv_fp8_pipe;
        CK(fn(&L.g, q, L.kfull, L.vfull, pages, 256, qpos, kpfull, T, S, 1, 0, 8, tc_partial, s));
        CK(m26_attn_reduce_tc(&L.g, tc_partial, nullptr, T, 1, 0, out, s));
      } else {
        auto fn = native ? m26_attn_prefill_fp8_tc_bf16q : m26_attn_prefill_fp8_tc;
        CK(fn(&L.g, q, L.kfull, L.vfull, pages, 256, qpos, kpfull, T, S, 0, L.sink, out, s));
      }
    } else {
      int T = VER_T, splits = L.swa ? 4 : 8;
      auto fn = native ? m26_attn_decode_splitkv_fp8_pipe_bf16q : m26_attn_decode_splitkv_fp8_pipe;
      if (L.swa) {
        CK(fn(&L.g, q, L.kcompact, L.vcompact, nullptr, 0, qpos, kpc, T, c.swa_s, splits, 0, 8, tc_partial, s));
      } else {
        CK(fn(&L.g, q, L.kfull, L.vfull, pages, 256, qpos, kpfull, T, c.s, splits, 0, 8, tc_partial, s));
      }
      CK(m26_attn_reduce_tc(&L.g, tc_partial, L.sink, T, splits, 0, out, s));
    }
  }

  // FP64 scalar reference for one layer -> ref (ref8/refpre).
  void launch_reference(int l, const op1cell::CellCase& c, cudaStream_t s) {
    CellLayer& L = layer[l];
    float* q = q_of(c, l); float* ref = ref_of(c, l);
    int T = c.phase ? c.t : VER_T;
    int splits = c.phase ? 2 : 16;
    if (c.phase) {
      CK(m26_attn_decode_splitkv_fp8(&L.g, q, L.kfull, nullptr, L.vfull, nullptr, pages, 256, qpos, kpfull, T, T, splits, 0, ref_partial, s));
    } else if (L.swa) {
      CK(m26_attn_decode_splitkv_fp8(&L.g, q, L.kcompact, nullptr, L.vcompact, nullptr, nullptr, 0, qpos, kpc, T, c.swa_s, splits, 0, ref_partial, s));
    } else {
      CK(m26_attn_decode_splitkv_fp8(&L.g, q, L.kfull, nullptr, L.vfull, nullptr, pages, 256, qpos, kpfull, T, c.s, splits, 0, ref_partial, s));
    }
    CK(m26_attn_reduce(&L.g, ref_partial, L.sink, T, splits, 0, ref, s));
  }

  // Full host check of one layer output vs its reference.
  double check_layer(int l, const op1cell::CellCase& c) {
    size_t n = t_of(c) * 64 * 128;
    std::vector<float> got(n), ref(n);
    CK(cudaMemcpy(got.data(), out_of(c, l), n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ref.data(), ref_of(c, l), n * 4, cudaMemcpyDeviceToHost));
    auto checked = bench::p1_full_check(got, ref, n);
    if (!checked.ok) { fprintf(stderr, "RESULT: FAIL OP1 cell output layer=%d index=%zu\n", l, checked.index); std::exit(5); }
    return checked.max_error;
  }

  double check_all(const op1cell::CellCase& c) {
    double worst = 0;
    for (int l = 0; l < NV; ++l) worst = std::max(worst, check_layer(l, c));
    return worst;
  }

  // Independent host coordinates + finite scan of every reference output.
  double verify_reference(const op1cell::CellCase& c, size_t& checked) {
    double worst = 0;
    int T = c.phase ? c.t : VER_T;
    for (int l = 0; l < NV; ++l) {
      size_t n = size_t(T) * 64 * 128;
      std::vector<float> ref(n);
      CK(cudaMemcpy(ref.data(), ref_of(c, l), n * 4, cudaMemcpyDeviceToHost));
      for (float x : ref) if (!std::isfinite(x)) { fprintf(stderr, "RESULT: FAIL OP1 reference finite\n"); std::exit(5); }
      int row = (c.id + l * 3) % T, head = (c.id * 5 + l * 7) % 64, dim = (c.id * 11 + l * 13) % 128;
      double value = host_coordinate(layer[l], c, row, head, dim);
      if (!std::isfinite(value)) { fprintf(stderr, "RESULT: FAIL OP1 coordinate finite\n"); std::exit(5); }
      worst = std::max(worst, std::abs(double(ref[(size_t(row) * 64 + head) * 128 + dim]) - value));
    }
    checked = size_t(NV) * size_t(T) * 64 * 128;
    return worst;
  }
};

void run_op1_cell(bool proxy) {
  if (proxy) { fprintf(stderr, "RESULT: REFUSE OP1 cell coordinator only\n"); std::exit(3); }
  auto start = std::chrono::steady_clock::now();
  auto budget = [&]{ if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > 480) {
    fprintf(stderr, "RESULT: INCOMPLETE OP1 cell 480s budget\n"); std::exit(4); } };
  reserve_check(uint64_t(2) << 30);
  op1_configure();  // prints OP1_RESOURCE x4; validates P1/C3 register/shared/capacity.
  int verification = 0, short_prefill = 0;
  for (const auto& c : op1cell::cases) { if (c.phase) ++short_prefill; else ++verification; }
  constexpr int case_count = (int)(sizeof(op1cell::cases) / sizeof(op1cell::CellCase));
  printf("OP1_BEGIN d7_sha=%s cases=%d verification=%d short_prefill=%d layers=48 ga=9 swa=39 modes=2 paths=ga-ver-c3p8,swa-ver-c3p4,ga-pre-c3p1,swa-pre-p1 timing=direct-48-layer-span scope=attention-critical-path gate=UNSET boundary=core-post-rope-prescaled-kv Q_values=BF16-exact\n",
         op1cell::d7_sha, case_count, verification, short_prefill);
  CellRun R;
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  int done = 0;
  for (const auto& c : op1cell::cases) {
    budget(); reserve_check(uint64_t(1) << 30);
    R.prep(c);
    size_t query_checked = size_t(NV) * R.t_of(c) * 64 * 192;
    CK(cudaMemset(R.bad, 0, 4));
    for (int l = 0; l < NV; ++l) p1_query_lattice_probe<<<64, 256>>>(R.q_of(c, l), R.t_of(c) * 64 * 192, R.bad);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    unsigned invalid = 0; CK(cudaMemcpy(&invalid, R.bad, 4, cudaMemcpyDeviceToHost));
    if (invalid) { fprintf(stderr, "RESULT: FAIL OP1 cell query lattice\n"); std::exit(5); }
    // Reference: FP64 scalar over all 48 layers; one per case, shared across modes.
    for (int l = 0; l < NV; ++l) R.launch_reference(l, c, nullptr);
    CK(cudaDeviceSynchronize());
    size_t ref_checked = 0; double coord = R.verify_reference(c, ref_checked);
    printf("OP1_REFERENCE case=%d category=%s request=%d step=%d phase=%d layers=48 checked=%zu coordinates=48 query_checked=%zu query_exact=PASS reference_finite=PASS max_coordinate_diff=%.9g\n",
           c.id, op1cell::categories[c.category], c.request, c.step, c.phase, ref_checked, query_checked, coord);
    // Pre-timing full checks for both modes.
    for (int native = 0; native < 2; ++native) {
      for (int l = 0; l < NV; ++l) R.launch_candidate(l, c, native, nullptr);
      CK(cudaDeviceSynchronize());
      double error = R.check_all(c);
      printf("OP1_PRECHECK case=%d category=%s request=%d step=%d phase=%d mode=%s layers=48 checked=%zu max_error=%.9g finite=PASS\n",
             c.id, op1cell::categories[c.category], c.request, c.step, c.phase, native ? "bf16q" : "f32q", ref_checked, error);
    }
    // Timed 48-layer spans: 7 samples per mode.
    for (int native = 0; native < 2; ++native) {
      for (int sample = 0; sample < 7; ++sample) {
        budget();
        CK(cudaEventRecord(a, nullptr));
        for (int l = 0; l < NV; ++l) R.launch_candidate(l, c, native, nullptr);
        CK(cudaEventRecord(b, nullptr)); CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
        int probe_layer = (done * 7 + sample) % NV;
        double error = R.check_layer(probe_layer, c);
        printf("OP1_SAMPLE case=%d category=%s request=%d step=%d phase=%d mode=%s timing=core index=%d ms=%.9f checked_layer=%d max_error=%.9g\n",
               c.id, op1cell::categories[c.category], c.request, c.step, c.phase, native ? "bf16q" : "f32q", sample, ms, probe_layer, error);
      }
      double error = R.check_all(c);
      printf("OP1_CHECK case=%d category=%s request=%d step=%d phase=%d mode=%s layers=48 checked=%zu max_error=%.9g finite=PASS\n",
             c.id, op1cell::categories[c.category], c.request, c.step, c.phase, native ? "bf16q" : "f32q", ref_checked, error);
    }
    ++done;
  }
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  reserve_check(0);
  printf("OP1_COMPLETE verification_cases=%d short_prefill_cases=%d cases=%d samples=%d all_layer_step_measured=1\n",
         verification, short_prefill, done, done * 2 * 7);
  puts("RESULT: PASS OP1 cell harness (no gate, no promotion)");
}
