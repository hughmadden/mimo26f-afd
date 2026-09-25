"""spike/loader.py — fused-QKV ckpt_tp=4 TP-order reconstruction (T1) + names.

Mirrors ``mimo26/loader.py`` entry points (map §2):
  - ``reconstruct_layer_qkv`` (:67) ``ckpt_tp = num_key_value_heads = 4`` chunks
    ``[Q_c|K_c|V_c]`` per rank (patch 01 @@ -467) → dequant → regroup ``[Q|K|V]``
  - uneven-shard fail-loud (:84-93)
  - ``canonical_name`` (:28): ``model.mtp.`` → ``mtp.`` tested BEFORE the generic
    ``model.`` strip (order matters); ``is_backbone_weight`` (:43)
  - ``qkv_segments`` (:59) geometry: QK 192 / V 128 first-class (config.py:127
    ``attn_dims``; real per-shard rows GA 12288/768/512, SWA 12288/1536/1024)

Modes (``naive=True`` / env ``MIMO26_SPIKE_NAIVE=1``): the naive path uses
``quant.dequantize_naive_fused`` — shard-major row order + global scale-row
mapping.  Classic scrambled Q/K/V word salad (T1, README.md:123) plus the
padded-scale-grid leak (T2).
"""
from __future__ import annotations

import os

import numpy as np

from . import quant


def _naive_default() -> bool:
    return os.environ.get("MIMO26_SPIKE_NAIVE", "") == "1"


def canonical_name(raw: str) -> str:
    """mimo26/loader.py:28 — MTP prefix first (order matters), then generic strip."""
    if raw.startswith("model.mtp."):
        return "mtp." + raw[len("model.mtp."):]
    if raw.startswith("model."):
        return raw[len("model."):]
    return raw


def is_backbone_weight(raw: str) -> bool:
    """mimo26/loader.py:43 — backbone only; drops mtp/dflash/audio/visual."""
    c = canonical_name(raw)
    return c.startswith("layers.") or c in (
        "embed_tokens.weight", "norm.weight", "lm_head.weight")


def qkv_segments(n_q: int, d_qk: int, n_kv: int, d_v: int) -> tuple[int, int, int]:
    """mimo26/loader.py:59 / config.py:127 attn_dims — (q_rows, k_rows, v_rows)
    of the FULL fused projection (per-shard segments = these // n_ranks).
    Real dims: QK 192 / V 128 (GA n_kv=4, SWA n_kv=8)."""
    return n_q * d_qk, n_kv * d_qk, n_kv * d_v


def reconstruct_layer_qkv(
    shard_weights: list[np.ndarray],
    shard_scales: list[np.ndarray],
    segs_per_shard: tuple[int, int, int],
    block: tuple[int, int] = (128, 128),
    naive: bool | None = None,
) -> np.ndarray:
    """Dequantize the ckpt_tp fused-QKV shards and regroup to ``[Q|K|V]`` rows.

    shard_weights[s]: u8 [sum(segs), hidden] fused ``[Q_c|K_c|V_c]`` for rank c
    (TP-order, patch 01).  shard_scales[s]: per-shard **padded** scale grid.
    Returns f32 [q+k+v, hidden] in projection-major order.

    Correct: per-chunk split + local scale trim (quant.split_shard_major_fused).
    Naive: one tensor / one global grid (quant.dequantize_naive_fused) — the
    word-salad path.  Uneven shard rows raise (loader.py:84-93 fail-loud).
    """
    naive = _naive_default() if naive is None else naive
    if not (len(shard_weights) == len(shard_scales) > 0):
        raise ValueError("need matching non-empty shard weight/scale lists")
    rows = {int(w.shape[0]) for w in shard_weights}
    if len(rows) != 1:  # loader.py:84-93
        raise ValueError(f"spike: uneven shard rows {sorted(rows)} "
                         "(uneven shards would mis-slice silently)")
    total = sum(segs_per_shard)
    for si, w in enumerate(shard_weights):
        if w.shape[0] != total:
            raise ValueError(f"shard {si}: rows {w.shape[0]} != segment sum {total} "
                             "(every shard must match — shard-0-only checks mis-slice silently)")
    if naive:
        return quant.dequantize_naive_fused(shard_weights, shard_scales, block=block)
    parts = quant.split_shard_major_fused(shard_weights, shard_scales, segs_per_shard, block=block)
    return np.concatenate(
        [quant.dequantize_per_row(parts[n], block=block) for n in ("q", "k", "v")], axis=0
    )
