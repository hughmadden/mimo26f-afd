"""spike/real_loader.py — P-102: real fused-QKV geometry + shard-major split.

VERIFIED against the live checkpoint headers (the coordinator, 22 Sep 2026 23:17 AEST):

  model.layers.{l}.self_attn.qkv_proj.weight        F8_E4M3  GA [13568, 4096]
                                                     SWA [14848, 4096]
  model.layers.{l}.self_attn.qkv_proj.weight_scale_inv  F32  GA [108, 32]
                                                     SWA [116, 32]

Geometry (AGENTS.md §3 / mimo26/config.py:45-52 attn_dims): QK 192 / V 128,
q=64 heads, kv=4 (GA) / 8 (SWA).  FULL rows GA 12288/768/512 = 13568,
SWA 12288/1536/1024 = 14848.

ckpt_tp = 4 on EVERY layer including SWA (AGENTS.md §3 "fused-QKV 4-way
TP-ordered on every layer (SWA too)"; patch 01 scale grids 108 GA = 4×27,
116 SWA = 4×29).  The stored tensor is SHARD-MAJOR: 4 contiguous
``[Q_c|K_c|V_c]`` chunks, and the scale grid is PADDED PER SHARD (GA 27 rows
for 26.5 real block-rows — the T2 pad rows; global-grid mapping mis-scales).

Mirrors mimo26/loader.py:59 (qkv_segments), :67 (reconstruct_layer_qkv),
:84-93 (uneven fail-loud) and quant/fp8_block.py:143-146 (local trim) — the
math lives in spike.loader/spike.quant, validated byte-exact against the golden.

Weight access: STRICTLY read-only, per-layer byte-range streaming (safetensors
header parse + pread).  Nothing written, nothing moved host-to-host.
"""
from __future__ import annotations

import numpy as np

from . import loader as L

# Verified real geometry
HIDDEN = 4096
D_QK = 192
D_V = 128
N_Q = 64
N_KV_GA = 4
N_KV_SWA = 8
CKPT_TP = 4  # every layer, SWA included
FP8_BLOCK = (128, 128)

SEG_GA = (N_Q * D_QK, N_KV_GA * D_QK, N_KV_GA * D_V)      # (12288, 768, 512) = 13568
SEG_SWA = (N_Q * D_QK, N_KV_SWA * D_QK, N_KV_SWA * D_V)   # (12288, 1536, 1024) = 14848
PER_SHARD_GA = (SEG_GA[0] // CKPT_TP, SEG_GA[1] // CKPT_TP, SEG_GA[2] // CKPT_TP)   # (3072, 192, 128)
PER_SHARD_SWA = (SEG_SWA[0] // CKPT_TP, SEG_SWA[1] // CKPT_TP, SEG_SWA[2] // CKPT_TP)  # (3072, 384, 256)
SCALE_ROWS_GA = 108   # stored grid rows = 4 shards x 27 (26.5 real + pad)
SCALE_ROWS_SWA = 116  # = 4 x 29


def layer_kind(hybrid_layer_pattern, layer: int) -> str:
    return "ga" if hybrid_layer_pattern[layer] == 0 else "swa"


def expected(kind: str) -> dict:
    if kind == "ga":
        return {"rows": 13568, "segs": SEG_GA, "per": PER_SHARD_GA,
                "grid_rows": SCALE_ROWS_GA, "per_grid": 27}
    return {"rows": 14848, "segs": SEG_SWA, "per": PER_SHARD_SWA,
            "grid_rows": SCALE_ROWS_SWA, "per_grid": 29}


def qkv_segments(n_q: int, d_qk: int, n_kv: int, d_v: int) -> tuple[int, int, int]:
    """mimo26/loader.py:59 — FULL fused projection rows (QK 192 / V 128)."""
    return n_q * d_qk, n_kv * d_qk, n_kv * d_v


def per_shard_segs(segs: tuple[int, int, int], ckpt_tp: int = CKPT_TP) -> tuple[int, int, int]:
    """Per-rank ``[Q_c|K_c|V_c]`` chunk rows (patch 01 @@ -467)."""
    q, k, v = segs
    if q % ckpt_tp or k % ckpt_tp or v % ckpt_tp:
        raise ValueError(f"real_loader: segments {segs} not divisible by ckpt_tp {ckpt_tp}")
    return q // ckpt_tp, k // ckpt_tp, v // ckpt_tp


def verify_stored(kind: str, w_shape: tuple[int, int], s_shape: tuple[int, int]) -> None:
    """Fail-loud against the VERIFIED on-disk shapes (loader.py:84-93 class)."""
    exp = expected(kind)
    if tuple(w_shape) != (exp["rows"], HIDDEN):
        raise ValueError(f"real_loader: fused rows {w_shape} != {(exp['rows'], HIDDEN)} for {kind}")
    if tuple(s_shape) != (exp["grid_rows"], HIDDEN // FP8_BLOCK[1]):
        raise ValueError(f"real_loader: scale grid {s_shape} != "
                         f"{(exp['grid_rows'], HIDDEN // FP8_BLOCK[1])} for {kind}")


def split_stored_fused(fused_w8: np.ndarray, grid: np.ndarray, kind: str):
    """Shard-major stored tensor -> 4 per-rank ``[Q_c|K_c|V_c]`` shard payloads
    with their LOCAL padded scale grids (grid rows ``r//128`` per shard — pad
    rows never indexed, T2)."""
    verify_stored(kind, fused_w8.shape, grid.shape)
    exp = expected(kind)
    per_rows = sum(exp["per"])
    per_grid = exp["per_grid"]
    weights, scales = [], []
    for c in range(4):
        w8 = np.ascontiguousarray(fused_w8[c * per_rows:(c + 1) * per_rows])
        s = np.ascontiguousarray(grid[c * per_grid:(c + 1) * per_grid])
        if w8.shape[0] != per_rows:  # loader.py:84-93 class
            raise ValueError(f"real_loader: shard {c} rows {w8.shape[0]} != {per_rows}")
        max_idx = (per_rows - 1) // FP8_BLOCK[0]
        if max_idx > per_grid - 1:
            raise ValueError(f"real_loader: scale row {max_idx} outside local grid {per_grid}")
        weights.append(w8)
        scales.append(s)
    return weights, scales, exp["per"]


def reconstruct_stored(fused_w8: np.ndarray, grid: np.ndarray, kind: str,
                       naive: bool | None = None) -> np.ndarray:
    """mimo26/loader.py:67 end-to-end on real layout: split -> regroup [Q|K|V]."""
    weights, scales, per = split_stored_fused(fused_w8, grid, kind)
    return L.reconstruct_layer_qkv(weights, scales, per, block=FP8_BLOCK, naive=naive)


def reconstruct_layer_qkv_now(shards: list[dict], segs: tuple[int, int, int],
                              naive: bool | None = None) -> np.ndarray:
    """Reconstruct from pre-split shard dicts (fixture/slice helper)."""
    return L.reconstruct_layer_qkv(
        [s["w8"] for s in shards], [s["s"] for s in shards], segs,
        block=shards[0].get("block", (4, 4)), naive=naive)
