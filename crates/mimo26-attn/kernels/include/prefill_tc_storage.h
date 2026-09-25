// Original P1 host/CUDA storage contract. No hardware qualification implied.
#pragma once
#include <cstdint>
namespace m26tc {
constexpr int P1M=64, P1N=16, P1Threads=256;
constexpr int P1M_SPLIT=32, P1Threads_SPLIT=128;
// Split=true is the P1 trial-1 occupancy variant (P1M 64->32, threads 256->128).
// The baseline (Split=false) is unchanged and must remain bitwise-identical.
template<class Bf,bool Residual,bool Split=false> struct alignas(16) PrefillStorage {
  static constexpr int PM = Split ? P1M_SPLIT : P1M;
  Bf q[Residual?3:1][PM*192];
  union alignas(16) Phase {
    Bf k[P1N*192];
    struct { float score[PM*P1N]; Bf ph[PM*P1N],pl[PM*P1N]; float alpha[PM]; } p;
  } phase;
  Bf v[128*P1N];
  int64_t qp[PM],kp[P1N];
  int physical[P1N],visible[P1N],key_bad[P1N];
  int q_bad[PM],row_bad[PM];
  float maximum[PM],sum[PM];
};
static_assert(sizeof(PrefillStorage<uint16_t,true>)==88128,"P1 Q3 shared contract");
static_assert(sizeof(PrefillStorage<uint16_t,false>)==38976,"P1 Q1 shared contract");
static_assert(sizeof(PrefillStorage<uint16_t,true,true>)==48192,"P1 Q3 split shared contract");
} // namespace m26tc
