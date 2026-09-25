"""Checkpoint -> model load-path glue (ARCHITECTURE.md §3 checkpoint seam).

Two jobs, both from verified reality (22 Sep 2026):

1. Name canonicalisation — the real index uses `model.`-prefixed names, expert names
   like `model.layers.{l}.mlp.experts.{e}.{gate|up|down}_proj.weight[_scale]`, and MTP
   under `model.mtp.layers.{i}...`. The model and drafter classes consume canonical
   names (no `model.` prefix; `mtp.layers.{i}...`).
2. Fused-QKV reconstruction — the checkpoint stores fused attention tensors in
   tensor-parallel order with **per-shard padded FP8 scale grids**; skipping the
   `split_shard_major_fused` step loads a model that returns broken output. This module
   is the single sanctioned path from shard rows to q/k/v projections.
"""

from __future__ import annotations

import re

import numpy as np

from .config import GA, MiMoConfig
from .quant.fp8_block import ReconstructedProjection, split_shard_major_fused

_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight(_scale)?$")


def canonical_name(raw: str) -> str:
    """Map a real checkpoint tensor name to the canonical model/drafter key.

    'model.layers.3.self_attn.qkv_proj.weight' -> 'layers.3.self_attn.qkv_proj.weight'
    'model.mtp.layers.0.eh_proj.weight'        -> 'mtp.layers.0.eh_proj.weight'
    'model.layers.3.mlp.experts.7.gate_proj.weight_scale'
                                               -> 'layers.3.mlp.experts.7.gate_proj.weight_scale'
    """
    if raw.startswith("model.mtp."):
        return "mtp." + raw[len("model.mtp."):]
    if raw.startswith("model."):
        return raw[len("model."):]
    return raw


def is_backbone_weight(raw: str) -> bool:
    """True for the 48-layer backbone (excludes mtp/dflash/audio/visual tensors)."""
    c = canonical_name(raw)
    return bool(re.match(r"^layers\.\d+\.", c)) or c in (
        "embed_tokens.weight", "norm.weight", "lm_head.weight")


def expert_key(layer: int, expert: int, proj: str, scale: bool = False) -> str:
    return f"layers.{layer}.mlp.experts.{expert}.{proj}_proj.weight" + ("_scale" if scale else "")


def _block(cfg: MiMoConfig) -> tuple[int, int]:
    b = cfg.fp8_block_size
    return (b, b) if isinstance(b, int) else tuple(b)


def qkv_segments(cfg: MiMoConfig, layer: int) -> tuple[int, int, int]:
    """(q_rows, k_rows, v_rows) of the FULL fused-QKV projection for `layer`
    (one shard's segments are these divided by n_ranks)."""
    kind = cfg.hybrid_layer_pattern[layer]
    q_rows, k_rows, v_rows, _ = cfg.attn_dims(kind)
    return q_rows, k_rows, v_rows


def reconstruct_layer_qkv(cfg: MiMoConfig, layer: int,
                          shard_weights: list[np.ndarray],
                          shard_scales: list[np.ndarray],
                          block: tuple[int, int] | None = None,
                          n_ranks: int = 1) -> dict[str, ReconstructedProjection]:
    """Rebuild q/k/v for one attention layer from TP-ordered fused shards.

    shard_weights[s]: u8 [local_rows, hidden] fused [q_s; k_s; v_s] rows for shard s.
    shard_scales[s]:  f32 per-shard **padded** scale grid (rows local_rows//br).
    n_ranks: number of shards the fused rows were split across. Shard-major CONTIGUOUS:
             shard s owns rows [s·rows/n, (s+1)·rows/n) of each projection, with the
             q/k/v segments concatenated per shard (real grid 108 = 4 shards x 27 rows).
             Naive global-grid dequantization is a load-fine/outputs-garbage bug.
    block: FP8 block (default from config: (cfg.fp8_block_size[0], cfg.fp8_block_size[1])).
    """
    br, bc = block if block else _block(cfg)
    q_rows, k_rows, v_rows = qkv_segments(cfg, layer)
    per_shard = (q_rows // n_ranks, k_rows // n_ranks, v_rows // n_ranks)
    for nm, rows, per in (("q", q_rows, per_shard[0]), ("k", k_rows, per_shard[1]),
                          ("v", v_rows, per_shard[2])):
        if per * n_ranks != rows:
            raise ValueError(f"{nm} rows {rows} not divisible by n_ranks {n_ranks} "
                             "(uneven shards would mis-slice silently)")
    for si, sw in enumerate(shard_weights):
        if sum(per_shard) != sw.shape[0]:
            raise ValueError(f"shard {si}: rows {sw.shape[0]} != segments sum {sum(per_shard)} "
                             "(every shard must match — shard-0-only checks mis-slice silently)")
    return split_shard_major_fused(shard_weights, shard_scales, per_shard, block=(br, bc))


def dflash_key_name(raw: str) -> str:
    """DFlash draft-model index names -> drafter keys (strip the 'dflash_draft_model.' root)."""
    for root in ("dflash_draft_model.", "draft_model."):
        if raw.startswith(root):
            return raw[len(root):]
    return raw
