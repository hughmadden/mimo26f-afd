# Quantizer v1 independent-source freeze

**23 September 2026, 23:47:21 AEST.** Base:
`e0b7bdc43a8f326abde4f73febe1a405ba50dba1`.

The first complete handwritten CUDA and shared host/device scalar implementation
was written from `docs/design/lattice-v1.md` §4 before opening or importing
`oracle/lattice/quant_v1.py`, its tests, or the kernel lead's lattice oracle.
No upstream quantizer implementation was copied or re-read during this implementation.
Earlier upstream design research had inspected the native quantizer and established
that it was nonconforming (BF16 pre-round/clamps); its body was not used here.

Frozen SHA256:

- `quant_v1.cuh`:
  `a7fd7b92a586f496e7701ee6c7b42468cfcf3996826ce0169cdb76f11ea809de`
- `quant_v1.cu`:
  `8e0aadf51eef4ab7e1c567983f0b535d77f525d965f8feba3f78445e009cb38d`

Derivation: derive the round-up power-of-two exponent from IEEE exponent/fraction
bits (the frexp threshold m = 0.875 is fraction 0x600000); perform payload RNE by
integer significand shifts on the exact E4M3FN grid; retain the sign bit for zero;
reject decoded exponent overflow. The scale floor is exactly FP32(1e-4).
The CUDA kernel assigns one K32 block per full warp, amax via integer shuffle,
and one uniquely owned fault record per block. It uses no atomics or FP arithmetic
for quantization, hence no implicit BF16 boundary or FTZ-sensitive scaling multiply.

This freeze is provenance, **not** a correctness claim. CPU/reference comparison
and CUDA compile-only checks follow it. Any later correction must retain these
original hashes and describe the delta; do not rewrite the independent origin.
