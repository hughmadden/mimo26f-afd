// Typed storage shared by the CPU C1 model and experimental CUDA implementation.
// Inclusion does not establish GPU qualification; C3 remains separately available.
// N16: two converted slots, one raw slot. Each K region aliases its own P phase.
#pragma once
#include <stdint.h>
namespace m26tc {
template<class Bf> struct alignas(16) WarpPipeModel {
  Bf qh[16*192], ql[16*192], qt[16*192];
  struct alignas(16) Slot {
    Bf v[128*16];
    struct Prob { Bf ph[16*16], pl[16*16]; float score[16*16], alpha[16]; };
    union Phase { Bf k[16*192]; Prob p; } phase;
    int visible[16];
  } slot[2];
  uint8_t raw[16*320];
  int physical[16];
  float maximum[16], sum[16];
  alignas(8) uint64_t k_ready[2], p_ready[2], free_slot[2];
  int bad;
};
static_assert(sizeof(WarpPipeModel<uint16_t>::Slot)==10304,"C1 slot model size");
static_assert(sizeof(WarpPipeModel<uint16_t>)==44416,"C1 shared model size");
} // namespace m26tc
