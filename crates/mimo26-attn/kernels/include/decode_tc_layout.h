// Original, host-testable contracts for mma.sync m16n8k16 BF16 fragments.
#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define M26_TC_HD __host__ __device__
#else
#define M26_TC_HD
#endif
namespace m26tc {
// k16-major shared tiles; adjacent BF16 pairs stay adjacent. For each packed
// fragment load, the 32 lanes address 32 distinct four-byte banks.
M26_TC_HD constexpr int tile_index(int rows, int row, int k) {
  return (k / 16) * rows * 16 + row * 16 + ((k & 15) ^ ((row & 7) * 2));
}
// F4: raw cp.async destinations permute whole aligned 16-byte chunks.
// K's 12 chunks permute within groups of four; V's eight permute as a whole.
M26_TC_HD constexpr int raw_k(int row,int d) {
  return row*192+(((d/16)^((row/2)&3))*16)+(d&15);
}
M26_TC_HD constexpr int raw_v(int row,int d) {
  return row*128+(((d/16)^(row&7))*16)+(d&15);
}
// In-place transposed V: d-major E4M3, swizzled at four-byte granularity.
M26_TC_HD constexpr int raw_v_dmajor(int d,int row) {
  return d*32+(row^((d&4)*4));
}
// Each producer warp stores an 8-row by 4-packed-pair fragment, the same
// bank-safe ownership as an MMA A fragment. i enumerates unique output pairs.
M26_TC_HD constexpr int producer_row(int i,int cols) {
  return ((i/32)/(cols/8))*8+(i%32)/4;
}
M26_TC_HD constexpr int producer_col(int i,int cols) {
  return ((i/32)%(cols/8))*8+(i&3)*2;
}
// C0 interleaves the 12 QK k16 steps into two independent chains per term.
M26_TC_HD constexpr int q_acc_group(int k16_step) { return k16_step & 1; }
M26_TC_HD constexpr int a_row(int lane, int reg) { return lane / 4 + (reg & 1) * 8; }
M26_TC_HD constexpr int a_col(int lane, int reg) { return (lane & 3) * 2 + (reg / 2) * 8; }
M26_TC_HD constexpr int b_row(int lane, int reg) { return (lane & 3) * 2 + reg * 8; }
M26_TC_HD constexpr int b_col(int lane) { return lane / 4; }
M26_TC_HD constexpr int c_row(int lane, int reg) { return lane / 4 + (reg / 2) * 8; }
M26_TC_HD constexpr int c_col(int lane, int reg) { return (lane & 3) * 2 + (reg & 1); }
M26_TC_HD constexpr int64_t partial_index(int t, int h, int sp, int splits) {
  return ((int64_t)t * 64 + h) * splits + sp;
}
// Bit-exact E4M3FN -> FP32, including signed zero and the two NaN codes.
M26_TC_HD inline uint32_t e4m3_bits(uint8_t code) {
  uint32_t sign = uint32_t(code & 128) << 24;
  unsigned e = (code >> 3) & 15, m = code & 7;
  if (e == 15 && m == 7) return sign | 0x7fc00000u;
  if (e) return sign | ((e + 120) << 23) | (m << 20);
  if (!m) return sign;
  unsigned lead = m >= 4 ? 2 : m >= 2 ? 1 : 0;
  return sign | ((118 + lead) << 23) | ((m - (1u << lead)) << (23 - lead));
}
// Two adjacent E4M3 codes -> two BF16 lanes, bit-exact including subnormals,
// signed zeros and canonical quiet NaNs. Normal lanes use packed integer math.
M26_TC_HD inline uint32_t e4m3x2_bf16(uint16_t codes) {
  unsigned a=codes&255, b=codes>>8;
  if ((a&120) && (b&120) && (a&127)!=127 && (b&127)!=127) {
    uint32_t spread=(codes&255u)|((uint32_t(codes)&0xff00u)<<8);
    return ((spread&0x00800080u)<<8) |
           ((((spread&0x00780078u)>>3)+0x00780078u)<<7) |
           ((spread&0x00070007u)<<4);
  }
  return (e4m3_bits(uint8_t(a))>>16)|(e4m3_bits(uint8_t(b))&0xffff0000u);
}
} // namespace m26tc
#undef M26_TC_HD
