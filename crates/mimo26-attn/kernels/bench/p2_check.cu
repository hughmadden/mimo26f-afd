// P2 qualification: the serving prefill kernel (attn_prefill_fa.cu) against the
// P1 split kernel (the device forward's previous prefill) and a CPU FP64 softmax
// reference, on seeded random Q / E4M3 K,V / sink, at the X1a lane shapes.
//
//   nvcc -O3 -std=c++17 --ftz=false -arch=sm_89 -I../include p2_check.cu \
//        ../attn_prefill_fa.cu ../attn_decode_tc.cu -o p2_check && ./p2_check
//
// Prints, per case: max |err| and relative L2 over the sampled rows for P1 and
// P2 vs FP64, P2 vs P1 over all rows, and median kernel times. Exit 1 on a
// NaN/mismatch beyond the gate.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../include/mimo26_attn_kernels.h"

extern "C" cudaError_t m26_attn_prefill_fp8_fa(const m26_geom* g, const float* q, const uint8_t* kc,
                                              const uint8_t* vc, int32_t T, int32_t S, int32_t q_row0,
                                              uint32_t naive, const float* sink, float* out,
                                              m26_stream_t stream);

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
      std::exit(2);                                                                        \
    }                                                                                      \
  } while (0)

static double e4m3_value(uint8_t b) {
  const double s = (b & 0x80) ? -1.0 : 1.0;
  const int e = (b >> 3) & 15, m = b & 7;
  if (e == 0) return s * m * std::ldexp(1.0, -9);
  return s * (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
}

// Nearest E4M3 (ties to larger magnitude, clamp 448): the KV store's codec.
static uint8_t e4m3_encode(float x) {
  const uint8_t sign = x < 0 ? 0x80 : 0;
  const double mag = std::min<double>(std::fabs(x), 448.0);
  int best = 0;
  double bd = 1e300;
  for (int c = 0; c < 127; ++c) {
    const double d = std::fabs(e4m3_value(uint8_t(c)) - mag);
    if (d < bd || (d == bd && e4m3_value(uint8_t(c)) > e4m3_value(uint8_t(best)))) {
      bd = d;
      best = c;
    }
  }
  return uint8_t(sign | best);
}

struct Case {
  const char* name;
  int n_kv, window, T, S, q_row0;
  bool sink;
};

int main() {
  const Case cases[] = {
      {"GA  lane A  T=2014 S=2014", 4, 0, 2014, 2014, 0, false},
      {"GA  lane B  T=2013 S=4027", 4, 0, 2013, 4027, 2014, false},
      {"SWA lane A  T=2014 S=2014 sink", 8, 128, 2014, 2014, 0, true},
      {"SWA lane B  T=2013 S=2140 sink", 8, 128, 2013, 2140, 127, true},
      {"GA  short   T=37   S=300", 4, 0, 37, 300, 263, false},
      {"SWA short   T=5    S=132 sink", 8, 128, 5, 132, 127, true},
  };
  int fails = 0;
  int32_t regs = 0, ctas = 0;
  CK(m26_attn_prefill_tc_config_split(&regs, &ctas));
  for (const Case& cs : cases) {
    std::mt19937 rng(1234 + cs.T + cs.S);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int T = cs.T, S = cs.S, nkv = cs.n_kv;
    std::vector<float> q(size_t(T) * 64 * 192);
    // Post-RoPE Q at a realistic spread (sharper softmax than unit variance).
    for (auto& x : q) x = 2.5f * nd(rng);
    std::vector<uint8_t> kc(size_t(S) * nkv * 192), vc(size_t(S) * nkv * 128);
    for (auto& b : kc) b = e4m3_encode(1.5f * nd(rng));
    for (auto& b : vc) b = e4m3_encode(0.707f * nd(rng));
    std::vector<float> sink(64);
    for (auto& s : sink) s = nd(rng);
    std::vector<int64_t> qpos(T), kpos(S);
    for (int j = 0; j < S; ++j) kpos[j] = 1000 + j;
    for (int t = 0; t < T; ++t) qpos[t] = 1000 + cs.q_row0 + t;

    float *dq, *dsink, *d1, *d2;
    uint8_t *dk, *dv;
    int64_t *dqp, *dkp;
    CK(cudaMalloc(&dq, q.size() * 4));
    CK(cudaMalloc(&dk, kc.size()));
    CK(cudaMalloc(&dv, vc.size()));
    CK(cudaMalloc(&dsink, 64 * 4));
    CK(cudaMalloc(&dqp, T * 8));
    CK(cudaMalloc(&dkp, S * 8));
    CK(cudaMalloc(&d1, size_t(T) * 64 * 128 * 4));
    CK(cudaMalloc(&d2, size_t(T) * 64 * 128 * 4));
    CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk, kc.data(), kc.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, vc.data(), vc.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsink, sink.data(), 64 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqp, qpos.data(), T * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkp, kpos.data(), S * 8, cudaMemcpyHostToDevice));
    m26_geom g{};
    g.n_q = 64;
    g.n_kv = nkv;
    g.d_qk = 192;
    g.d_v = 128;
    g.window = cs.window;
    g.value_scale = 1.0;
    const float* sk = cs.sink ? dsink : nullptr;
    auto p1 = [&] {
      return m26_attn_prefill_fp8_tc_split(&g, dq, dk, dv, nullptr, 0, dqp, dkp, T, S, 0, sk, d1, nullptr);
    };
    auto p2 = [&] { return m26_attn_prefill_fp8_fa(&g, dq, dk, dv, T, S, cs.q_row0, 0, sk, d2, nullptr); };
    CK(p1());
    CK(p2());
    CK(cudaDeviceSynchronize());
    // Timing: median of 11.
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    auto time = [&](auto f) {
      std::vector<float> ms;
      for (int i = 0; i < 11; ++i) {
        CK(cudaEventRecord(a));
        CK(f());
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float x;
        CK(cudaEventElapsedTime(&x, a, b));
        ms.push_back(x);
      }
      std::sort(ms.begin(), ms.end());
      return ms[5];
    };
    const float t1 = time(p1), t2 = time(p2);
    std::vector<float> o1(size_t(T) * 64 * 128), o2(o1.size());
    CK(cudaMemcpy(o1.data(), d1, o1.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o2.data(), d2, o2.size() * 4, cudaMemcpyDeviceToHost));

    // FP64 reference on sampled (token, head) rows, including the first/last tokens.
    const double scale = 1.0 / std::sqrt(192.0);
    double e1 = 0, e2 = 0, n1 = 0, m1 = 0, m2 = 0;
    int nan1 = 0, nan2 = 0, rows = 0;
    const int rep = 64 / nkv;
    std::vector<double> sc(S), acc(128);
    for (int t = 0; t < T; t += std::max(1, T / 23)) {
      for (int h = 0; h < 64; h += 7) {
        const int kvh = h / rep, r = cs.q_row0 + t;
        double mx = -1e300;
        for (int j = 0; j <= r; ++j) {
          if (cs.window > 0 && r - j >= cs.window) { sc[j] = -1e300; continue; }
          double s = 0;
          for (int d = 0; d < 192; ++d)
            s += double(q[(size_t(t) * 64 + h) * 192 + d]) * e4m3_value(kc[(size_t(j) * nkv + kvh) * 192 + d]);
          sc[j] = s * scale;
          mx = std::max(mx, sc[j]);
        }
        if (cs.sink) mx = std::max(mx, double(sink[h]));
        double l = cs.sink ? std::exp(double(sink[h]) - mx) : 0;
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int j = 0; j <= r; ++j) {
          if (sc[j] < -1e299) continue;
          const double p = std::exp(sc[j] - mx);
          l += p;
          for (int d = 0; d < 128; ++d) acc[d] += p * e4m3_value(vc[(size_t(j) * nkv + kvh) * 128 + d]);
        }
        for (int d = 0; d < 128; ++d) {
          const double ref = acc[d] / l;
          const float x1 = o1[(size_t(t) * 64 + h) * 128 + d], x2 = o2[(size_t(t) * 64 + h) * 128 + d];
          nan1 += !std::isfinite(x1);
          nan2 += !std::isfinite(x2);
          e1 += (x1 - ref) * (x1 - ref);
          e2 += (x2 - ref) * (x2 - ref);
          n1 += ref * ref;
          m1 = std::max(m1, std::fabs(x1 - ref));
          m2 = std::max(m2, std::fabs(x2 - ref));
        }
        ++rows;
      }
    }
    double d12 = 0, n12 = 0, m12 = 0;
    int nan_all = 0;
    for (size_t i = 0; i < o1.size(); ++i) {
      nan_all += !std::isfinite(o2[i]);
      d12 += double(o2[i] - o1[i]) * (o2[i] - o1[i]);
      n12 += double(o1[i]) * o1[i];
      m12 = std::max(m12, double(std::fabs(o2[i] - o1[i])));
    }
    const double r1 = std::sqrt(e1 / n1), r2 = std::sqrt(e2 / n1), r12 = std::sqrt(d12 / n12);
    // Gate: P2 within 4e-3 relative L2 of FP64 and no NaN (FP16 Q/P rounding; K,V exact).
    const bool ok = nan2 == 0 && nan_all == 0 && r2 < 4e-3;
    fails += !ok;
    std::printf("%-32s rows=%d  P1 vs f64: max %.2e rel %.2e nan %d | P2 vs f64: max %.2e rel %.2e nan %d | "
                "P2 vs P1 (all): max %.2e rel %.2e | P1 %.3f ms  P2 %.3f ms  (%.1fx) %s\n",
                cs.name, rows, m1, r1, nan1, m2, r2, nan2, m12, r12, t1, t2, t1 / t2, ok ? "PASS" : "FAIL");
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dsink); cudaFree(dqp); cudaFree(dkp); cudaFree(d1);
    cudaFree(d2);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
  }
  std::printf("RESULT: %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
