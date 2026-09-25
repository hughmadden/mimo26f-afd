// OP1 cold-prefill arm (F-1): full-request multi-chunk P1 at 2K/8K/32K/64K.
// Retained P1 kernel unchanged (qualified at 3139ee4, full-checked at T=2048 over
// S=2K/128K/1M). This cell verifies the 2048-query chunking, not the kernel.
// GA (Hkv4, causal) and SWA (Hkv8, window 128) per point, both lattices.
#include "op1_cell_generated.h"

double cold_coordinate(int nkv, int window, int qbase, int layer, int row, int head, int dim) {
  int kh = head * nkv / 64;
  int qpos = qbase + row;
  int end = qpos;
  int begin = window ? std::max(0, qpos - 127) : 0;
  double m = -INFINITY, l = 0, o = 0;
  auto fold = [&](double score, double value) {
    double next = std::max(m, score);
    double a = std::isfinite(m) ? std::exp(m - next) : 0;
    double b = std::exp(score - next);
    o = o * a + b * value; l = l * a + b; m = next;
  };
  unsigned qs = unsigned(layer) * 104729u, ks = 23u + unsigned(layer) * 104729u, vs = 47u + unsigned(layer) * 104729u;
  for (int j = begin; j <= end; ++j) {
    double dot = 0;
    for (int z = 0; z < 192; ++z)
      dot += query(uint32_t((qpos * 64 + head) * 192 + z) + qs)
           * decode(code(uint32_t((j * nkv + kh) * 192 + z), ks));
    fold(dot / std::sqrt(192.0), decode(code(uint32_t((j * nkv + kh) * 128 + dim), vs)));
  }
  if (window) fold((head - 32) * .0625 + layer * .00390625, 0);
  return o / l;
}

void run_op1_cold(bool proxy) {
  if (proxy) { fprintf(stderr, "RESULT: REFUSE OP1 cold coordinator only\n"); std::exit(3); }
  auto start = std::chrono::steady_clock::now();
  auto budget = [&]{ if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > 480) {
    fprintf(stderr, "RESULT: INCOMPLETE OP1 cold 480s budget\n"); std::exit(4); } };
  reserve_check(uint64_t(2) << 30);
  op1_configure();  // validates P1/C3 resources; prints OP1_RESOURCE x4.
  constexpr int T = 2048, SPLITS = 256;
  const int batch = bench::p1_reference_batch(65536);  // 8 for all cold points
  int total_chunks = 0, total_samples = 0;
  printf("OP1_COLD_BEGIN d7_sha=%s points=4 kinds=2 modes=2 T=2048 batch=%d scope=attention-cold-prefill-multichunk-P1 gate=UNSET boundary=core-post-rope-prescaled-kv Q_values=BF16-exact\n",
         op1cell::d7_sha, batch);
  for (int S : op1cell::cold_prefill) {
    for (int swa = 0; swa < 2; ++swa) {
      int nkv = swa ? 8 : 4, window = swa ? 128 : 0, layer = swa ? 1 : 0;
      m26_geom g{64, nkv, 192, 128, window, 1.0};
      int chunks = S / 2048;
      printf("OP1_COLD_POINT S=%d kind=%s nkv=%d window=%d chunks=%d\n", S, swa ? "swa" : "ga", nkv, window, chunks);
      float* q = alloc<float>(size_t(T) * 64 * 192);
      uint8_t* k = alloc<uint8_t>(size_t(S) * nkv * 192);
      uint8_t* v = alloc<uint8_t>(size_t(S) * nkv * 128);
      int64_t* qpos = alloc<int64_t>(T); int64_t* kpos = alloc<int64_t>(S);
      int32_t* pages = alloc<int32_t>((S + 255) / 256);
      float* out = alloc<float>(size_t(T) * 64 * 128);
      float* ref = alloc<float>(size_t(T) * 64 * 128);
      double* part = alloc<double>(size_t(batch) * 64 * SPLITS * 130);
      float* sink = swa ? alloc<float>(64) : nullptr;
      unsigned* bad = alloc<unsigned>(1);
      std::vector<float> sinks(64);
      for (int h = 0; h < 64; ++h) sinks[h] = (h - 32) * .0625f + layer * .00390625f;
      if (sink) CK(cudaMemcpy(sink, sinks.data(), 64 * 4, cudaMemcpyHostToDevice));
      init_positions<<<8, 256>>>(kpos, S, 0); init_pages<<<1, 256>>>(pages, (S + 255) / 256);
      CK(cudaGetLastError());
      std::vector<double> med[2]; med[0].resize(chunks); med[1].resize(chunks);
      std::vector<double> uf(chunks), ef32(chunks), ef16(chunks);
      for (int i = 0; i < chunks; ++i) {
        budget(); reserve_check(uint64_t(1) << 30);
        int chunk_S = 2048 * (i + 1), qbase = 2048 * i;
        bool full_ref = (i == 0) || (i == chunks / 2) || (i == chunks - 1);
        op1_init_q<<<64, 256>>>(q, size_t(T) * 64 * 192, qbase, unsigned(layer) * 104729u);
        op1_init_kv<<<64, 256>>>(k, size_t(S) * nkv * 192, 192, nkv, (S + 255) / 256, 0, chunk_S, 23u + unsigned(layer) * 104729u);
        op1_init_kv<<<64, 256>>>(v, size_t(S) * nkv * 128, 128, nkv, (S + 255) / 256, 0, chunk_S, 47u + unsigned(layer) * 104729u);
        init_positions<<<8, 256>>>(qpos, T, qbase);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemset(bad, 0, 4)); p1_query_lattice_probe<<<64, 256>>>(q, size_t(T) * 64 * 192, bad);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        unsigned invalid = 0; CK(cudaMemcpy(&invalid, bad, 4, cudaMemcpyDeviceToHost));
        if (invalid) { fprintf(stderr, "RESULT: FAIL OP1 cold query lattice\n"); std::exit(5); }
        // Full scalar reference for the subset chunks (shared across modes).
        std::vector<float> refv;
        if (full_ref) {
          CK(cudaMemset(ref, 0xff, size_t(T) * 64 * 128 * 4));
          for (int first = 0; first < T; first += batch) {
            int n = std::min(batch, T - first); budget();
            CK(cudaMemset(part, 0xff, size_t(n) * 64 * SPLITS * 130 * sizeof(double)));
            CK(m26_attn_decode_splitkv_fp8(&g, q + size_t(first) * 64 * 192, k, nullptr, v, nullptr, pages, 256, qpos + first, kpos, n, chunk_S, SPLITS, 0, part, nullptr));
            CK(m26_attn_reduce(&g, part, sink, n, SPLITS, 0, ref + size_t(first) * 64 * 128, nullptr));
            CK(cudaDeviceSynchronize()); budget();
          }
          refv.assign(size_t(T) * 64 * 128, 0);
          CK(cudaMemcpy(refv.data(), ref, refv.size() * 4, cudaMemcpyDeviceToHost));
          for (float x : refv) if (!std::isfinite(x)) { fprintf(stderr, "RESULT: FAIL OP1 cold reference finite\n"); std::exit(5); }
        }
        double worst_coord = 0, worst_ref = 0;
        for (int native = 0; native < 2; ++native) {
          std::vector<float> times;
          for (int s2 = 0; s2 < 7; ++s2) {
            budget();
            CK(cudaMemset(out, 0xff, size_t(T) * 64 * 128 * 4));
            cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
            CK(cudaEventRecord(a, nullptr));
            auto fn = native ? m26_attn_prefill_fp8_tc_bf16q : m26_attn_prefill_fp8_tc;
            CK(fn(&g, q, k, v, pages, 256, qpos, kpos, T, chunk_S, 0, sink, out, nullptr));
            CK(cudaEventRecord(b, nullptr)); CK(cudaEventSynchronize(b));
            float ms = 0; CK(cudaEventElapsedTime(&ms, a, b)); CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
            times.push_back(ms);
            std::vector<float> outv(size_t(T) * 64 * 128);
            CK(cudaMemcpy(outv.data(), out, outv.size() * 4, cudaMemcpyDeviceToHost));
            for (float x : outv) if (!std::isfinite(x)) { fprintf(stderr, "RESULT: FAIL OP1 cold output finite\n"); std::exit(5); }
            double cand_coord = 0;
            for (int c = 0; c < 3; ++c) {
              int row = c * (T - 1) / 2, head = c * 31, dim = c * 53;
              double value = cold_coordinate(nkv, window, qbase, layer, row, head, dim);
              cand_coord = std::max(cand_coord, std::abs(double(outv[(size_t(row) * 64 + head) * 128 + dim]) - value));
            }
            worst_coord = std::max(worst_coord, cand_coord);
            if (full_ref) {
              auto chk = bench::p1_full_check(outv, refv, outv.size());
              if (!chk.ok) { fprintf(stderr, "RESULT: FAIL OP1 cold full check\n"); std::exit(5); }
              worst_ref = std::max(worst_ref, chk.max_error);
            }
            printf("OP1_COLD_SAMPLE S=%d kind=%s chunk=%d mode=%s index=%d ms=%.9f finite=PASS coord=%.9g\n",
                   S, swa ? "swa" : "ga", i, native ? "bf16q" : "f32q", s2, ms, cand_coord);
          }
          med[native][i] = bench::median(times);
        }
        uf[i] = bench::flops(T, chunk_S, window);
        ef32[i] = bench::p1_mma_flops(T, chunk_S, nkv, window, true);
        ef16[i] = bench::p1_mma_flops(T, chunk_S, nkv, window, false);
        printf("OP1_COLD_CHUNK S=%d kind=%s chunk=%d S_c=%d full_ref=%d checked=%zu coordinates=3 query_exact=PASS reference_finite=PASS useful_flops=%.0f executed_f32=%.0f executed_bf16=%.0f max_error=%.9g max_coordinate_diff=%.9g\n",
               S, swa ? "swa" : "ga", i, chunk_S, full_ref ? 1 : 0, full_ref ? size_t(T) * 64 * 128 : 0,
               uf[i], ef32[i], ef16[i], worst_ref, worst_coord);
        ++total_chunks; total_samples += 14;
      }
      for (int native = 0; native < 2; ++native) {
        double ttft = 0, uf_total = 0, ef_total = 0;
        for (int i = 0; i < chunks; ++i) {
          ttft += med[native][i]; uf_total += uf[i]; ef_total += native ? ef16[i] : ef32[i];
        }
        printf("OP1_COLD_RESULT S=%d kind=%s mode=%s ttft_ms=%.9f tok_s=%.9f useful_flops_total=%.0f executed_flops_total=%.0f useful_tflops=%.9f executed_tflops=%.9f\n",
               S, swa ? "swa" : "ga", native ? "bf16q" : "f32q", ttft, S / (ttft * 1e-3), uf_total, ef_total,
               uf_total / (ttft * 1e9), ef_total / (ttft * 1e9));
      }
      for (void* p : allocations) CK(cudaFree(p)); allocations.clear();
    }
  }
  reserve_check(0);
  printf("OP1_COLD_COMPLETE points=4 kinds=2 modes=2 chunks=%d samples=%d full_ref_chunks=20 gate=UNSET\n", total_chunks, total_samples);
  puts("RESULT: PASS OP1 cold prefill harness (no gate, no promotion)");
}
