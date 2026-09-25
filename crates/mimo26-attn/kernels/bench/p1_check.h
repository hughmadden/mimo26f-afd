#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
namespace bench {
// Untimed reference scheduling only; candidate geometry and arithmetic unchanged.
constexpr int p1_reference_batch(int s) { return s==1048576?2:8; }
#ifdef __CUDACC__
__host__ __device__
#endif
inline bool p1_query_bits_ok(uint32_t bits) {
  uint32_t mag=bits&0x7fffffff;
  return !(bits&0xffff)&&mag<0x7f800000&&(!mag||mag>=0x00800000);
}
struct P1Check { bool ok; size_t index; double max_error; };
inline P1Check p1_full_check(const std::vector<float>& got,const std::vector<float>& ref,size_t expected) {
  if(!expected||got.size()!=expected||ref.size()!=expected)return {false,0,INFINITY};
  double worst=0;
  for(size_t i=0;i<expected;++i) {
    double error=std::abs(double(got[i])-ref[i]);
    if(!std::isfinite(got[i])||!std::isfinite(ref[i])||error>2e-5)return {false,i,error};
    worst=std::max(worst,error);
  }
  return {true,expected,worst};
}
} // namespace bench
