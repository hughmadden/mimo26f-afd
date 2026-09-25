"""Quantizer ``e4m3fn-k32-v1``: the lattice-v1 reference codec (docs/design/lattice-v1.md §4).

This is the codec of record for the E-W4A8-v1 lattice oracle, the coordinator input encoder and the Spark
intermediate encoder. CUDA implementations are written independently from the spec and must match
``encode_blocks`` byte for byte on identical FP32 inputs.

Per block of 32 consecutive FP32 values h:
  a = max(max|h|, 1e-4f) in FP32; k = the smallest integer with a <= 448 * 2**k (exact, from frexp);
  scale byte = k + 127 (always 105..247); payload = RNE-to-E4M3FN of h * 2**-k (ties to even, -0 kept).
Nonfinite input, or a reconstruction outside FP32 range, raises NumericalFault. Nothing is clipped silently.

Two implementations live here, and a test pins them bit-equal:
- ``encode_blocks`` / ``decode_blocks`` (numpy, float64-exact arithmetic): the codec of record.
- ``fake_quant_torch`` (torch, float32-exact arithmetic): the fast quantize-dequantize used by the X1c spike arms.
"""
from __future__ import annotations

import numpy as np

VERSION = "e4m3fn-k32-v1"
BLOCK = 32
FLOOR = np.float32(1e-4)
E4M3_MAX = 448.0
SCALE_BYTE_MIN, SCALE_BYTE_MAX = 105, 247


class NumericalFault(ValueError):
    """Nonfinite input or an out-of-FP32-range reconstruction (lattice-v1 §4 steps 1 and 7)."""


def scale_exponent(a) -> int:
    """Smallest k with a <= 448 * 2**k, for a finite FP32 a >= 1e-4f. Exact: 448 = 0.875 * 2**9."""
    m, e = np.frexp(np.float32(a))  # a = m * 2**e, m in [0.5, 1)
    return int(e) - 9 if m <= np.float32(0.875) else int(e) - 8


def e4m3fn_decode(codes: np.ndarray) -> np.ndarray:
    """E4M3FN codes (uint8) -> float64 values. 0x7F / 0xFF (NaN) raise NumericalFault."""
    c = np.asarray(codes, dtype=np.uint8).astype(np.int64)
    if np.any((c & 0x7F) == 0x7F):
        raise NumericalFault("E4M3FN NaN code")
    s = np.where(c & 0x80, -1.0, 1.0)
    ef, m = (c >> 3) & 0xF, c & 0x7
    mag = np.where(ef == 0, m * 2.0 ** -9, np.ldexp(1.0 + m / 8.0, ef - 7))
    return s * mag


def e4m3fn_encode(y: np.ndarray) -> np.ndarray:
    """RNE (ties to even) of float64 values onto the E4M3FN grid -> uint8 codes. -0 is kept.
    Satfinite is defensive only: under quantizer v1, |y| <= 448 by construction."""
    y = np.asarray(y, dtype=np.float64)
    neg = np.signbit(y)
    ay = np.abs(y)
    codes = np.zeros(y.shape, dtype=np.int64)
    sub = ay < 2.0 ** -6
    # Subnormal range: quantum 2**-9. r == 8 is the smallest normal (field 1, mantissa 0).
    r_sub = np.rint(ay / 2.0 ** -9).astype(np.int64)  # np.rint rounds half to even
    codes = np.where(sub & (r_sub < 8), r_sub, codes)
    codes = np.where(sub & (r_sub == 8), 1 << 3, codes)
    # Normal range: ay = f * 2**g with f in [0.5, 1), so E = g - 1 and the quantum is 2**(E - 3).
    _, g = np.frexp(np.where(sub, 1.0, ay))
    E = g.astype(np.int64) - 1
    r = np.rint(np.ldexp(np.where(sub, 8.0, ay), np.where(sub, 0, 3 - E))).astype(np.int64)  # in [8, 16]
    carry = r == 16
    E = np.where(carry, E + 1, E)
    r = np.where(carry, 8, r)
    over = (E > 8) | ((E == 8) & (r > 14))
    norm = ((E + 7) << 3) | (r - 8)
    norm = np.where(over, (15 << 3) | 6, norm)  # satfinite -> 448 (defensive)
    codes = np.where(sub, codes, norm)
    return (codes | (neg.astype(np.int64) << 7)).astype(np.uint8)


def encode_blocks(h) -> tuple[np.ndarray, np.ndarray]:
    """FP32 [..., n] (n % 32 == 0) -> (payload uint8 [..., n], scale bytes uint8 [..., n/32])."""
    h = np.asarray(h, dtype=np.float32)
    if h.shape[-1] % BLOCK:
        raise ValueError(f"last dimension {h.shape[-1]} is not a multiple of {BLOCK}")
    if not np.all(np.isfinite(h)):
        raise NumericalFault("nonfinite input to quantizer v1")
    hb = h.reshape(h.shape[:-1] + (h.shape[-1] // BLOCK, BLOCK))
    a = np.maximum(np.max(np.abs(hb), axis=-1), FLOOR).astype(np.float32)
    m, e = np.frexp(a)
    k = np.where(m <= np.float32(0.875), e - 9, e - 8).astype(np.int64)
    y = np.ldexp(hb.astype(np.float64), -k[..., None])
    codes = e4m3fn_encode(y).reshape(h.shape)
    return codes, (k + 127).astype(np.uint8)


def decode_blocks(codes, scales) -> np.ndarray:
    """(payload [..., n], scale bytes [..., n/32]) -> FP32 values. Out-of-FP32-range results raise NumericalFault."""
    codes = np.asarray(codes, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.uint8)
    if np.any((scales < SCALE_BYTE_MIN) | (scales > SCALE_BYTE_MAX)):
        raise NumericalFault("activation scale byte outside the quantizer-v1 range 105..247")
    v = e4m3fn_decode(codes).reshape(codes.shape[:-1] + (codes.shape[-1] // BLOCK, BLOCK))
    out = np.ldexp(v, (scales.astype(np.int64) - 127)[..., None])
    if np.any(np.abs(out) > np.finfo(np.float32).max):
        raise NumericalFault("quantizer-v1 reconstruction outside FP32 range")
    return out.astype(np.float32).reshape(codes.shape)


def bf16_rne(x) -> np.ndarray:
    """FP32 -> BF16 (round to nearest, ties to even) -> FP32. Nonfinite in or out raises NumericalFault."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.all(np.isfinite(x)):
        raise NumericalFault("nonfinite input to BF16 RNE")
    b = x.view(np.uint32).astype(np.uint64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)
    if not np.all(np.isfinite(r)):
        raise NumericalFault("BF16 RNE overflow")
    return r


def _pow2(n):
    """Exact float32 2**n for an int tensor n in [-126, 127], built from the exponent bits."""
    import torch

    return ((n.to(torch.int32) + 127) << 23).view(torch.float32)


def fake_quant_torch(h, check: bool = True):
    """torch float32 [..., n] -> quantizer-v1 quantize-dequantize, float32, same device. Bit-equal to
    decode_blocks(*encode_blocks(h)); the tests pin this. All arithmetic is exact in float32: power-of-two
    scaling, integer mantissas, torch.round = half to even."""
    import torch

    if check and not bool(torch.isfinite(h).all()):
        raise NumericalFault("nonfinite input to quantizer v1")
    shp = h.shape
    hb = h.float().reshape(*shp[:-1], shp[-1] // BLOCK, BLOCK)
    a = hb.abs().amax(dim=-1).clamp_min(float(FLOOR))
    m, e = torch.frexp(a)
    k = torch.where(m <= 0.875, e - 9, e - 8)
    y = hb * _pow2(-k).unsqueeze(-1)
    ay = y.abs()
    sub = ay < 2.0 ** -6
    q_sub = torch.round(ay * 512.0) / 512.0  # subnormal quantum 2**-9 (r == 8 lands exactly on 2**-6)
    _, g = torch.frexp(torch.where(sub, torch.ones_like(ay), ay))
    E = (g - 1).to(torch.int32)
    qn = _pow2(E - 3)
    q_norm = torch.round(ay / qn) * qn  # r == 16 carries to 2**(E+1) exactly
    mag = torch.where(sub, q_sub, q_norm).clamp_max(E4M3_MAX)
    out = torch.copysign(mag, y) * _pow2(k).unsqueeze(-1)
    if check and not bool(torch.isfinite(out).all()):
        raise NumericalFault("quantizer-v1 reconstruction outside FP32 range")
    return out.reshape(shp)
