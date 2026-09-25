#pragma once
#include <cstdint>

namespace m26tc {
// C3 / N4 amendment: single raw stage, fixed M16/N32. No cache ABI change.
// K is dead before P/score/alpha use this union. All QK readers must meet at
// the CTA barrier before the first score store; P/alpha readers must finish
// at the tile-end barrier before the next K expansion.
template<class Bf>
struct alignas(16) CompactMath {
  Bf qh[16*192], ql[16*192], qt[16*192], v[128*32];
  struct Prob {
    Bf ph[16*32], pl[16*32];
    float score[16*32], alpha[16];
  };
  union Phase {
    Bf k[32*192];
    Prob p;
  } phase;
  float maximum[16], sum[16];
  int visible[32];
};
template<class Bf>
struct alignas(16) CompactPipe {
  CompactMath<Bf> math;
  uint8_t raw[1][32*320];
  int physical[1][32];
};
static_assert(sizeof(CompactMath<uint16_t>)==39168,"C3 math layout");
static_assert(sizeof(CompactPipe<uint16_t>)==49536,"C3 total shared footprint");
} // namespace m26tc
