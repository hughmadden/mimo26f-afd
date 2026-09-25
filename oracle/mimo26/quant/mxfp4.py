"""MXFP4 codec: E2M1 nibbles + E8M0 per-32 block scales (OCP microscaling).

Storage convention matches the MiMo-V2.6-Flash-RL expert shards (verified against real
safetensors headers): a projection of logical shape [out, in] is stored as
`weight` u8 [out, in//2] (two E2M1 nibbles per byte along the input dim) and
`weight_scale` u8 [out, in//32] (E8M0 scale per 32-element block along the input dim).

Nibble packing: for input index k, byte k//2, low nibble = even k, high nibble = odd k.
"""

from __future__ import annotations

import numpy as np

BLOCK = 32

# E2M1 magnitudes indexed by (e << 1) | m  — 1 sign bit, 2 exp, 1 mantissa
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
E8M0_BIAS = 127


def decode_e2m1(nibbles: np.ndarray) -> np.ndarray:
    """Map uint8 nibbles (0..15) to float values."""
    n = np.asarray(nibbles, dtype=np.uint8)
    mag = E2M1_MAG[n & 0x7]
    sign = np.where((n & 0x8) != 0, -1.0, 1.0)
    return sign * mag


def encode_e2m1(values: np.ndarray) -> np.ndarray:
    """Round floats to the nearest E2M1 code (ties toward larger magnitude)."""
    v = np.asarray(values, dtype=np.float64)
    sign = np.where(v < 0, 8, 0).astype(np.uint8)
    mag = np.abs(v)
    # midpoints between representable magnitudes: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    idx = np.searchsorted(E2M1_MAG, mag, side="left")
    lo = np.clip(idx - 1, 0, 7)
    hi = np.clip(idx, 0, 7)
    pick_hi = np.abs(mag - E2M1_MAG[hi]) <= np.abs(mag - E2M1_MAG[lo])
    code = np.where(pick_hi, hi, lo).astype(np.uint8)
    return sign | code


def scale_byte_to_float(scale_u8: np.ndarray) -> np.ndarray:
    """E8M0 byte -> float scale 2^(b - 127)."""
    b = np.asarray(scale_u8, dtype=np.int32)
    return np.exp2(b.astype(np.float64) - E8M0_BIAS)


def scale_float_to_byte(scale: np.ndarray) -> np.ndarray:
    s = np.asarray(scale, dtype=np.float64)
    with np.errstate(divide="ignore"):
        e = np.ceil(np.log2(np.where(s > 0, s, 1.0)))
    e = np.clip(e, -E8M0_BIAS, 254 - E8M0_BIAS)  # OCP E8M0: byte = exp+127, 255 reserved
    return (e + E8M0_BIAS).astype(np.uint8)


def _check_shape(w: np.ndarray, block: int) -> None:
    if w.ndim != 2:
        raise ValueError("expected 2D [out, in]")
    if w.shape[1] % block != 0:
        raise ValueError(f"in-dim {w.shape[1]} not divisible by block {block}")
    if w.shape[1] % 2 != 0:
        raise ValueError("in-dim must be even (2 nibbles per byte)")


def pack(w: np.ndarray, block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a float32 [out, in] matrix to (packed u8 [out, in//2], scales u8 [out, in//block]).

    Scale per block: 2^ceil(log2(amax / 6)) so every element is representable.
    """
    w = np.asarray(w, dtype=np.float64)
    _check_shape(w, block)
    out, inn = w.shape
    bl = w.reshape(out, inn // block, block)
    amax = np.abs(bl).max(axis=2)
    with np.errstate(divide="ignore"):
        e = np.ceil(np.log2(np.where(amax > 0, amax / 6.0, 1.0)))
    e = np.clip(e, -E8M0_BIAS, 254 - E8M0_BIAS)  # OCP E8M0: byte = exp+127, 255 reserved
    scale = np.exp2(e)
    scale_u8 = (e + E8M0_BIAS).astype(np.uint8)
    q = encode_e2m1(bl / scale[:, :, None])
    q = q.reshape(out, inn)
    packed = q[:, 0::2] | (q[:, 1::2] << 4)
    return packed.astype(np.uint8), scale_u8


def unpack(packed: np.ndarray, scales_u8: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Dequantize to float32 [out, in]."""
    p = np.asarray(packed, dtype=np.uint8)
    s = np.asarray(scales_u8, dtype=np.uint8)
    out, half = p.shape
    inn = half * 2
    if s.shape != (out, inn // block):
        raise ValueError(f"scale shape {s.shape} != {(out, inn // block)}")
    nib = np.empty((out, inn), dtype=np.uint8)
    nib[:, 0::2] = p & 0x0F
    nib[:, 1::2] = (p >> 4) & 0x0F
    vals = decode_e2m1(nib)
    scale = scale_byte_to_float(s)  # [out, in//block]
    return (vals.reshape(out, inn // block, block) * scale[:, :, None]).reshape(out, inn).astype(np.float32)


def linear(x: np.ndarray, packed: np.ndarray, scales_u8: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Reference MXFP4 GEMM: x [..., in] @ W.T with W [out, in]."""
    w = unpack(packed, scales_u8, block)
    x = np.asarray(x, dtype=np.float32)
    return (x @ w.T).astype(np.float32)
