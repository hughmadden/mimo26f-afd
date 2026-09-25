"""spike/mxfp4.py — MXFP4 codec twin (E2M1 nibbles + E8M0-32 block scales).

Spec source (external, READ-ONLY): `mimo26/quant/mxfp4.py` of the
mimo-v2.6-flash-ds41rt-port CPU twin ("Storage convention matches the
MiMo-V2.6-Flash-RL expert shards").  Semantics pinned BYTE-EXACT against
`tests/golden/mxfp4_golden.json` (`e2m1_codebook_by_nibble`) by
spike/tests/test_p103_codecs.py.  Independent implementation — no code body
lifted into spike/.

Storage (verified live headers, 23 Sep 2026 AEST): a projection of logical
shape [out, in] is stored as `weight` u8 [out, in//2] (two E2M1 nibbles per
byte along the input dim) + `weight_scale` u8 [out, in//32] (E8M0 scale per
32-element block).  Live example: expert `gate_proj.weight` U8 [2048, 2048]
= logical [2048, 4096], `gate_proj.weight_scale` U8 [2048, 128];
`down_proj.weight` U8 [4096, 1024] = logical [4096, 2048], scale [4096, 64].

  T14 nibble order: input index k lives in byte k//2 — LOW nibble = even k,
      high nibble = odd k.
  T10 E8M0 clamp: scale = 2^(b-127); byte 255 is RESERVED (OCP) and clamps
      to 2^127 (naive: unclamped 2^128 — a poison scale).

Naive flips (MIMO26_SPIKE_NAIVE=1 or naive=True): swapped nibble order (T14)
and the missing clamp (T10).  Decode math runs in float64 and casts at the
end — the golden is byte-exact only on that path.
"""
from __future__ import annotations

import os

import numpy as np

BLOCK = 32
E8M0_BIAS = 127
E8M0_EXP_MAX = 127  # byte 255 reserved -> clamp exponent (T10)
# golden mxfp4_golden.json "e2m1_codebook_by_nibble", indexed directly by nibble
E2M1_CODEBOOK = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float64)


def _naive_default(naive: bool | None) -> bool:
    return os.environ.get("MIMO26_SPIKE_NAIVE") == "1" if naive is None else bool(naive)


def scale_byte_to_float(scales_u8: np.ndarray, naive: bool | None = None) -> np.ndarray:
    """E8M0 byte -> float64 scale 2^(b-127); byte 255 clamps to 2^127 (T10)."""
    naive = _naive_default(naive)
    b = np.asarray(scales_u8, dtype=np.int32)
    if naive:
        return np.exp2(b - E8M0_BIAS)                      # 255 -> 2^128 poison (no clamp)
    e = np.minimum(b, 254) - E8M0_BIAS                     # T10: 255 reserved, clamped
    return np.exp2(e)


def decode_e2m1(nibbles: np.ndarray) -> np.ndarray:
    """Map uint8 nibbles (0..15) to float64 values via the golden codebook."""
    return E2M1_CODEBOOK[np.asarray(nibbles, dtype=np.uint8) & 0xF]


def unpack(packed: np.ndarray, scales_u8: np.ndarray, naive: bool | None = None) -> np.ndarray:
    """Dequantize u8 [out, in//2] + u8 [out, in//32] -> f32 [out, in]."""
    naive = _naive_default(naive)
    p = np.asarray(packed, dtype=np.uint8)
    s = np.asarray(scales_u8, dtype=np.uint8)
    out, half = p.shape
    inn = half * 2
    if s.shape != (out, inn // BLOCK):
        raise ValueError(f"mxfp4: scale shape {s.shape} != {(out, inn // BLOCK)} (T10/T14 layout)")
    nib = np.empty((out, inn), dtype=np.uint8)
    if naive:                                              # T14 naive: nibbles swapped
        nib[:, 0::2] = (p >> 4) & 0x0F
        nib[:, 1::2] = p & 0x0F
    else:                                                  # low nibble = even k
        nib[:, 0::2] = p & 0x0F
        nib[:, 1::2] = (p >> 4) & 0x0F
    vals = decode_e2m1(nib)                                # float64
    scale = scale_byte_to_float(s, naive=naive)            # [out, in//BLOCK] float64
    prod = (vals.reshape(out, inn // BLOCK, BLOCK) * scale[:, :, None]).reshape(out, inn)
    # Saturate to the f32 range BEFORE the cast: |val| x 2^127 can exceed f32
    # max (naive 2^128 poison always does) and the astype raised
    # 'overflow encountered in cast' -> inf (RuntimeWarning, spike tests log).
    # Clamp is a no-op for every pinned in-range value: byte-exact goldens keep
    # bits, and the T10 negative's bad != want detection stays true.
    f32_max = np.finfo(np.float32).max
    return np.clip(prod, -f32_max, f32_max).astype(np.float32)


def unpack_torch(packed, scales_u8, naive: bool | None = None, device=None):
    """Device-side mirror of `unpack` (u8 [out, in//2] + u8 [out, in//32] -> f32
    [out, in]).  BITWISE-identical to the numpy reference on both naive paths by
    construction — same IEEE-754 op order (f64 nibble decode -> f64 exact 2^e
    scale -> f64 product -> +-f32max saturation -> f32 cast) — and pinned
    bitwise (int32-view equality) by
    `test_unpack_torch_bitwise_matches_reference`.  The point is speed: one
    fused device pass replaces the 258 ms/expert-matrix numpy f64 path
    (ADVISOR-I3 §11.3); the needle rerun is I/O-bound after this.
    """
    import torch

    naive = _naive_default(naive)
    if device is None:
        device = packed.device if isinstance(packed, torch.Tensor) else "cpu"
    p = torch.as_tensor(packed, dtype=torch.uint8, device=device)
    s = torch.as_tensor(scales_u8, dtype=torch.uint8, device=device)
    out, half = p.shape
    inn = half * 2
    if tuple(s.shape) != (out, inn // BLOCK):
        raise ValueError(
            f"mxfp4: scale shape {tuple(s.shape)} != {(out, inn // BLOCK)} (T10/T14 layout)")
    lo = torch.bitwise_and(p, 0x0F)                       # T14: low nibble = even k
    hi = torch.bitwise_and(torch.bitwise_right_shift(p, 4), 0x0F)
    nib = torch.stack((hi, lo) if naive else (lo, hi), dim=2).reshape(out, inn)
    lut = torch.tensor(E2M1_CODEBOOK.tolist(), dtype=torch.float64, device=device)
    vals = lut[nib.to(torch.int64)]                       # f64 [out, inn]
    b = s.to(torch.int64)
    e = (b if naive else torch.minimum(b, torch.full_like(b, 254))) - E8M0_BIAS
    # exact 2^e via exponent-bit placement (e in [-127, 128]; f64 domain fine)
    scale = ((e + 1023) << 52).view(torch.float64)        # [out, inn//BLOCK]
    prod = (vals.reshape(out, inn // BLOCK, BLOCK) * scale.unsqueeze(-1)).reshape(out, inn)
    f32_max = float(np.finfo(np.float32).max)
    return torch.clamp(prod, -f32_max, f32_max).to(torch.float32)
