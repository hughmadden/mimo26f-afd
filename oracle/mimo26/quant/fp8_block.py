"""FP8 E4M3 block-128 codec + fused-QKV reconstruction.

Two responsibilities:

1. Dequantize FP8 (e4m3fn) weights with per-128x128-block `weight_scale_inv` (F32), and a
   reference `fp8_linear`.
2. `split_shard_major_fused` — fix for the MiMo-V2.6 checkpoint convention: fused QKV tensors
   are stored in **tensor-parallel shard order** (shard-major rows, each shard holding its
   local [q; k; v] row segments) and the FP8 scale grid is **padded separately per shard**.
   Loading without this reconstruction "works" and produces corrupted outputs (verified
   trap; see ARCHITECTURE.md §5.3). The test suite pins this with a negative test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# e4m3fn decode table: byte [s eeee mmm] -> float (bias 7, no infinities, S.1111.111 = NaN)
def _e4m3_table() -> np.ndarray:
    t = np.empty(256, dtype=np.float64)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0x0F
        m = b & 0x07
        if e == 0:
            val = (m / 8.0) * (2.0 ** -6)  # subnormal
        else:
            val = (1.0 + m / 8.0) * (2.0 ** (e - 7))
        t[b] = s * val
    t[0x7F] = np.nan  # S.1111.111
    t[0xFF] = np.nan
    return t


E4M3 = _e4m3_table()
E4M3_MAX = 448.0


def decode_e4m3(codes: np.ndarray) -> np.ndarray:
    return E4M3[np.asarray(codes, dtype=np.uint8)]


def encode_e4m3(values: np.ndarray) -> np.ndarray:
    """Nearest-representable encoding (search the decode table's monotone magnitudes)."""
    v = np.asarray(values, dtype=np.float64)
    mag = np.abs(np.nan_to_num(v, nan=0.0))
    mag = np.clip(mag, 0.0, E4M3_MAX)
    # magnitude grid: stable sort of |decode(code)| with NaN magnitudes mapped to +inf
    grid = np.abs(E4M3)
    grid = np.where(np.isnan(grid), np.inf, grid)
    order = np.argsort(grid, kind="stable")
    grid_sorted = grid[order]
    idx = np.searchsorted(grid_sorted, mag, side="left")
    lo = np.clip(idx - 1, 0, 255)
    hi = np.clip(idx, 0, 255)
    pick_hi = np.abs(mag - grid_sorted[hi]) <= np.abs(mag - grid_sorted[lo])
    chosen = np.where(pick_hi, order[hi], order[lo]).astype(np.uint8)
    sign = np.where(v < 0, 0x80, 0).astype(np.uint8)
    return (chosen & 0x7F) | sign


def dequantize_block(w_codes: np.ndarray, scale_inv: np.ndarray, block: tuple[int, int] = (128, 128)) -> np.ndarray:
    """FP8 codes [rows, cols] with scale_inv [ceil(rows/br), ceil(cols/bc)] -> float32."""
    w = np.asarray(w_codes, dtype=np.uint8)
    s = np.asarray(scale_inv, dtype=np.float32)
    rows, cols = w.shape
    br, bc = block
    rb, cb = -(-rows // br), -(-cols // bc)
    if s.shape != (rb, cb):
        raise ValueError(f"scale shape {s.shape} != {(rb, cb)}")
    vals = decode_e4m3(w)
    out = np.zeros((rb * br, cb * bc), dtype=np.float64)
    out[:rows, :cols] = vals
    out = out.reshape(rb, br, cb, bc) * s[:, None, :, None]
    return out.reshape(rb * br, cb * bc)[:rows, :cols].astype(np.float32)


def quantize_block(w: np.ndarray, block: tuple[int, int] = (128, 128)) -> tuple[np.ndarray, np.ndarray]:
    """Quantize float weights per block -> (codes u8, scale_inv f32[rb, cb]).

    Non-finite inputs are sanitized (NaN -> 0, +-inf -> +-1e30 so the block scale
    stays finite and codes saturate at +-448 instead of poisoning the whole block).
    """
    w = np.asarray(w, dtype=np.float64)
    w = np.nan_to_num(w, nan=0.0, posinf=1e30, neginf=-1e30)
    rows, cols = w.shape
    br, bc = block
    rb, cb = -(-rows // br), -(-cols // bc)
    pad = np.zeros((rb * br, cb * bc))
    pad[:rows, :cols] = w
    bl = pad.reshape(rb, br, cb, bc)
    amax = np.abs(bl).max(axis=(1, 3))  # [rb, cb]
    scale = np.where(amax > 0, amax / E4M3_MAX, 1.0).astype(np.float32)
    q = encode_e4m3(bl / scale[:, None, :, None].reshape(rb, 1, cb, 1))
    codes = q.reshape(rb * br, cb * bc)[:rows, :cols]
    scale_inv = scale  # stored weight_scale_inv is the multiplier for dequant
    return codes.astype(np.uint8), scale_inv


def fp8_linear(x: np.ndarray, w_codes: np.ndarray, scale_inv: np.ndarray,
               block: tuple[int, int] = (128, 128)) -> np.ndarray:
    w = dequantize_block(w_codes, scale_inv, block)
    x = np.asarray(x, dtype=np.float32)
    return (x @ w.T).astype(np.float32)


# ---------------------------------------------------------------------------
# fused-QKV reconstruction (the load-fine/outputs-garbage trap)
# ---------------------------------------------------------------------------

@dataclass
class ReconstructedProjection:
    name: str  # "q" | "k" | "v"
    weight: np.ndarray  # u8 [rows, hidden] e4m3 codes, projection-major
    scale_per_row: np.ndarray  # f32 [rows, ceil(hidden/bc)] — scales resolved per row


def split_shard_major_fused(shard_weights: list[np.ndarray], shard_scales: list[np.ndarray],
                            segs_per_shard: tuple[int, int, int],
                            block: tuple[int, int] = (128, 128)) -> dict[str, ReconstructedProjection]:
    """Rebuild per-projection (q, k, v) tensors from shard-major fused storage.

    shard_weights[s]: u8 [rows_s, hidden] — shard s's local [q_s; k_s; v_s] rows.
    shard_scales[s]:  f32 [ceil(rows_s/br), ceil(cols/bc)] — **padded per shard**.
    segs_per_shard:   (q_rows, k_rows, v_rows) within one shard (same for all shards).
    """
    br, bc = block
    q_rows, k_rows, v_rows = segs_per_shard
    if not (len(shard_weights) == len(shard_scales) > 0):
        raise ValueError("need matching non-empty shard weight/scale lists")
    for i, (w, s) in enumerate(zip(shard_weights, shard_scales)):
        if w.shape[0] != q_rows + k_rows + v_rows:
            raise ValueError(f"shard {i}: rows {w.shape[0]} != segment sum "
                             f"{q_rows + k_rows + v_rows} (uneven shards would mis-slice silently)")
    acc: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        "q": ([], []), "k": ([], []), "v": ([], []),
    }
    bounds = (("q", 0, q_rows), ("k", q_rows, q_rows + k_rows), ("v", q_rows + k_rows, q_rows + k_rows + v_rows))
    for w, s in zip(shard_weights, shard_scales):
        for name, lo, hi in bounds:
            rows = hi - lo
            # per-row scale resolution: row r of this shard uses scale row r//br of THIS shard's grid
            local_rows = np.arange(lo, hi)
            scale_rows = local_rows // br
            scale_per_row = s[scale_rows, : w.shape[1] // bc].astype(np.float32)
            acc[name][0].append(w[lo:hi, :])
            acc[name][1].append(scale_per_row)
    out: dict[str, ReconstructedProjection] = {}
    for name in ("q", "k", "v"):
        wts, scs = acc[name]
        out[name] = ReconstructedProjection(
            name=name,
            weight=np.concatenate(wts, axis=0),
            scale_per_row=np.concatenate(scs, axis=0),
        )
    return out


def dequantize_per_row(proj: ReconstructedProjection, block: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Dequantize a reconstructed projection whose scales are resolved per row."""
    br, bc = block
    w = proj.weight
    rows, cols = w.shape
    vals = decode_e4m3(w).reshape(rows, cols // bc, bc)
    return (vals * proj.scale_per_row[:, :, None]).reshape(rows, cols).astype(np.float32)


def dequantize_naive_fused(shard_weights: list[np.ndarray], shard_scales: list[np.ndarray],
                           block: tuple[int, int] = (128, 128)) -> np.ndarray:
    """THE BUG, preserved as a test oracle: treat the concat as one tensor with one grid.

    Wrong in two ways at once: row order stays shard-major (not projection-major) and the
    scale rows are mapped by global row index across per-shard padded grids.
    """
    br, bc = block
    w = np.concatenate(shard_weights, axis=0)
    s = np.concatenate(shard_scales, axis=0)
    rows, cols = w.shape
    vals = decode_e4m3(w).reshape(rows, cols // bc, bc)
    scale_rows = np.arange(rows) // br  # naive: ignores per-shard padding
    scale = s[scale_rows, : cols // bc]
    return (vals * scale[:, :, None]).reshape(rows, cols).astype(np.float32)
