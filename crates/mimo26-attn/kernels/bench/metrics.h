// Benchmark accounting: useful unique KV bytes, not repeated GQA loads.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bench {
constexpr double target_gbs = 1790.0 * 0.70;
constexpr double target_tflops = 100.0;
constexpr uint64_t reserve_bytes = uint64_t(4) << 30;
inline uint64_t kv_bytes(int s, int nkv) {
  return uint64_t(s) * nkv * (192 + 128); // E4M3, unit scale, one layer
}
inline uint64_t pairs(int t, int s, int window = 0) {
  if (t <= 0 || s < t || window < 0) throw std::invalid_argument("shape");
  uint64_t n = 0;
  for (int i = 0; i < t; ++i) {
    int visible = s - t + i + 1;
    n += window ? std::min(visible, window) : visible;
  }
  return n;
}
inline double flops(int t, int s, int window = 0) {
  return 2.0 * 64 * (192 + 128) * pairs(t, s, window);
}
// T=1 GA, one resident CTA/SM on the coordinator: choose whole 170-SM waves.
constexpr int tc_default_splits(int s) { return s>=1048576?510:s>=131072?85:64; }
constexpr double tc_wave_fill(int splits) { return (4.0*splits)/(170*((4*splits+169)/170)); }
// GA decode: Q has three BF16 products, P has two; count padded tiles.
inline double tc_decode_mma_flops(int s, int splits, bool residual=true, int tile=32) {
  if (s <= 0 || splits <= 0 || (tile!=16 && tile!=32)) throw std::invalid_argument("decode shape/tile");
  uint64_t padded = 0;
  for (int p=0; p<splits; ++p) {
    int64_t length = int64_t(s)*(p+1)/splits - int64_t(s)*p/splits;
    padded += ((length+tile-1)/tile)*tile;
  }
  return 2.0 * 64 * ((residual?3:1)*192 + 2*128) * padded;
}
// P1 M64/N16: benchmark positions are q=[s-t,s), k=[0,s). Charge full
// query-row and key-column tiles whenever ANY valid query sees a key in them.
// This is not an accounting formula for arbitrary/nonmonotonic positions.
inline double p1_mma_flops(int t,int s,int nkv,int window,bool residual=true) {
  if(t<=0||t>65535||s<t||(nkv!=4&&nkv!=8)||window<0)throw std::invalid_argument("P1 shape");
  int qt=64/(64/nkv);uint64_t padded=0;
  for(int base=0;base<t;base+=qt) {
    int64_t hi=int64_t(s)-t+std::min(t,base+qt);
    int64_t lo=window?std::max(int64_t(0),int64_t(s)-t+base-window+1):0;
    padded+=uint64_t((hi+15)/16-lo/16)*16;
  }
  return 2.0*64*nkv*((residual?3:1)*192+2*128)*padded;
}
constexpr size_t eta_shared_bytes(int m,int n) {
  return size_t(m)*192*2 + size_t(n)*320*2 + size_t(m)*n*2 + size_t(m)*4*4 + size_t(n)*320;
}
constexpr double eta_mma_flops(int m,int n,bool residual) {
  return 2.0*m*n*((residual?3:1)*192+2*128);
}
inline double median(std::vector<float> samples) {
  if (samples.empty()) throw std::invalid_argument("no samples");
  for (float x : samples)
    if (!(x > 0) || !std::isfinite(x)) throw std::invalid_argument("invalid timing");
  std::sort(samples.begin(), samples.end());
  size_t n = samples.size();
  return n % 2 ? samples[n/2] : (double(samples[n/2-1]) + samples[n/2]) / 2;
}
inline double gbs(uint64_t bytes, double ms) { return bytes / (ms * 1e6); }
inline double tflops(double ops, double ms) { return ops / (ms * 1e9); }
inline bool memory_fits(uint64_t free, uint64_t needed) {
  return free >= reserve_bytes && needed <= free - reserve_bytes;
}
inline bool aot_matches(int arch, int sms, int baked_arch, int baked_sms) {
  return arch == baked_arch && sms == baked_sms;
}
} // namespace bench
