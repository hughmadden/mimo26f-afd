"""spike/tests/test_p102_real_shard_geometry.py — P-102 negatives on real shapes.

Pins the VERIFIED on-disk geometry contract (live headers, the coordinator 22 Sep 2026):
GA fused [13568, 4096] grid [108, 32]; SWA fused [14848, 4096] grid [116, 32].
The loader must honor (map §2 loader.py:59/:67/:84-93, patch 01 ckpt_tp=4,
AGENTS.md §3 "4-way TP-ordered on every layer (SWA too)", T2 grid padding):

  1. FULL rows match attn_dims (QK 192 / V 128): GA 12288/768/512,
     SWA 12288/1536/1024
  2. per-rank ``[Q_c|K_c|V_c]`` chunks divide ckpt_tp=4 exactly (SWA too);
     anything else mis-slices silently and MUST raise
  3. shard-major split + per-shard padded grids (T2) reconstruct to the
     whole-matrix reference; the naive global-grid path does not
  4. ``model.mtp.`` canonicalisation keeps the mtp root (loader.py:28 order)

FAILS under ``MIMO26_SPIKE_NAIVE=1`` where the impl is exercised; PASS on the
correct impl.  Runs recorded in runs/20260923-i1-spike/.
"""
from __future__ import annotations

import numpy as np
import pytest

from spike import real_loader as RL
from spike.loader import canonical_name, is_backbone_weight

from .conftest import make_fused_case


def test_real_fused_qkv_geometry_qk192_v128():
    """AGENTS.md §3: QK 192 / V 128; GA kv=4, SWA kv=8; stored totals 13568/14848."""
    assert RL.qkv_segments(RL.N_Q, RL.D_QK, RL.N_KV_GA, RL.D_V) == (12288, 768, 512)
    assert RL.qkv_segments(RL.N_Q, RL.D_QK, RL.N_KV_SWA, RL.D_V) == (12288, 1536, 1024)
    assert sum(RL.SEG_GA) == 13568 and RL.SEG_GA[0] != RL.SEG_GA[2]  # F9: external anchor = verified header rows 13568; T4 QK192/V128 asymmetry (Q seg != V seg) must stay detectable — no formula-derived constant triple
    assert RL.SEG_SWA == (12288, 1536, 1024) and sum(RL.SEG_SWA) == 14848
    assert (RL.SCALE_ROWS_GA, RL.SCALE_ROWS_SWA) == (108, 116)  # 4x27 / 4x29


def test_ckpt_tp4_every_layer_including_swa():
    """patch 01 / AGENTS.md §3: 4-way TP order on GA AND SWA; chunks divide
    exactly; non-divisors must raise (silent mis-slice guard)."""
    assert RL.CKPT_TP == 4
    assert RL.per_shard_segs(RL.SEG_GA) == (3072, 192, 128)
    assert RL.per_shard_segs(RL.SEG_SWA) == (3072, 384, 256)  # SWA is 4-way too
    with pytest.raises(ValueError):
        RL.per_shard_segs((12288, 768, 512), 7)


def test_verify_stored_shapes_fail_loud():
    """Verified on-disk shapes are the gate: wrong rows/grid raise."""
    RL.verify_stored("ga", (13568, 4096), (108, 32))
    RL.verify_stored("swa", (14848, 4096), (116, 32))
    with pytest.raises(ValueError):
        RL.verify_stored("ga", (14848, 4096), (108, 32))  # SWA rows on GA
    with pytest.raises(ValueError):
        RL.verify_stored("swa", (14848, 4096), (108, 32))  # GA grid on SWA


def test_real_layout_reconstruct_matches_whole_with_padded_grids():
    """T2 at real layout: 4-rank [Q_c|K_c|V_c] shard-major split with per-shard
    PADDED grids reconstructs to the [Q|K|V] whole-matrix reference (the
    env-selected impl — flips on the naive run)."""
    weights, scales, segs, ref = make_fused_case(n_ranks=4, segs=(3, 2, 2), cols=8, block=(4, 4))
    shards = [{"w8": w, "s": s, "rows": w.shape[0]} for w, s in zip(weights, scales)]
    got = RL.reconstruct_layer_qkv_now(shards, segs)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6)


def test_mtp_prefix_canonicalisation_order():
    """loader.py:28 — ``model.mtp.`` -> ``mtp.`` BEFORE the generic strip."""
    assert canonical_name("model.mtp.layers.0.eh_proj.weight") == "mtp.layers.0.eh_proj.weight"
    assert canonical_name("model.layers.3.self_attn.qkv_proj.weight") == "layers.3.self_attn.qkv_proj.weight"
    assert is_backbone_weight("model.layers.3.self_attn.qkv_proj.weight")
    assert not is_backbone_weight("model.mtp.layers.0.eh_proj.weight")
