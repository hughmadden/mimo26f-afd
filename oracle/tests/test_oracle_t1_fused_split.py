"""T1 — fused-QKV ckpt_tp TP-order split (the classic word-salad trap).

Fresh mirror of the spike T1 class (spike/tests/test_t1_fused_qkv_split.py:38-51),
exercised against the TWIN: shard-major ``[Q_c|K_c|V_c]`` storage must regroup
to projection-major ``[Q|K|V]``
(oracle/mimo26/quant/fp8_block.py:120-158 ``split_shard_major_fused``,
oracle/mimo26/loader.py:67-94 ``reconstruct_layer_qkv``).

Two-run classification (suite convention):
  NEGATIVE — FAILS under ``MIMO26_SPIKE_NAIVE=1`` (the twin's
  ``dequantize_naive_fused``, fp8_block.py:170-184 — shard-major rows + global
  scale mapping), PASSES on the correct path:
    test_t1_regroup_projection_major_negative
  BOTH RUNS: the byte-exact split pin, the scramble detection test, and the
  loader fail-loud guards.
"""
from __future__ import annotations

import numpy as np
import pytest

from mimo26.config import MiMoConfig
from mimo26.loader import qkv_segments, reconstruct_layer_qkv
from mimo26.quant import fp8_block as fb

from .conftest import build_fused_case, dequant_fused

BLOCK = (4, 4)


def test_t1_regroup_projection_major_negative():
    """NEGATIVE — flips on the naive run.

    4-rank ``[Q_c|K_c|V_c]`` shards (ckpt_tp=4, patch 01 TP order) must regroup
    to the whole-matrix ``[Q|K|V]`` reference (fp8_block.py:140 bounds +
    :150-158 regroup).  The naive path leaves rows shard-major — word salad.
    """
    weights, scales, segs, expected = build_fused_case(n_ranks=4)
    got = dequant_fused(weights, scales, segs, block=BLOCK)  # env-default impl
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_t1_split_codes_and_scales_byte_exact():
    """Pin (both runs): the split regroups CODES and per-row scales byte-exactly
    — ``[Q_0..Q_n | K_0..K_n | V_0..V_n]`` at the raw-code level, before any
    dequant arithmetic (fp8_block.py:148-157)."""
    weights, scales, segs, _expected = build_fused_case(n_ranks=3)
    parts = fb.split_shard_major_fused(weights, scales, segs, block=BLOCK)
    r_q, r_k, r_v = segs
    bounds = {"q": (0, r_q), "k": (r_q, r_q + r_k), "v": (r_q + r_k, r_q + r_k + r_v)}
    for name, (lo, hi) in bounds.items():
        want_codes = np.concatenate([w[lo:hi, :] for w in weights], axis=0)
        want_scales = np.concatenate(
            [s[np.arange(lo, hi) // BLOCK[0], : w.shape[1] // BLOCK[1]]
             for w, s in zip(weights, scales)], axis=0
        )
        assert parts[name].weight.tobytes() == want_codes.tobytes(), f"{name} codes regroup"
        assert parts[name].scale_per_row.tobytes() == want_scales.tobytes(), f"{name} scale trim"
    n_shards = len(weights)
    assert parts["q"].weight.shape == (n_shards * r_q, weights[0].shape[1])


def test_t1_naive_scramble_observable():
    """Detection power (both runs; spike analogue
    spike/tests/test_t1_fused_qkv_split.py:46-53): the naive scramble must be
    grossly visible — err > 10x correct err + 0.5."""
    weights, scales, segs, expected = build_fused_case(n_ranks=4)
    correct_err = float(np.abs(dequant_fused(weights, scales, segs, block=BLOCK, naive=False)
                               - expected).max())
    naive_err = float(np.abs(dequant_fused(weights, scales, segs, block=BLOCK, naive=True)
                             - expected).max())
    assert naive_err > 10 * correct_err + 0.5, (naive_err, correct_err)


def test_t1_loader_fail_loud_bad_geometry():
    """Fail-loud guards (both runs; loader.py:85-93): non-divisible segments and
    shard-row mismatches raise ValueError — no silent mis-slice."""
    cfg = MiMoConfig.tiny()
    assert qkv_segments(cfg, 0) == (256, 64, 32)  # tiny GA layer geometry
    with pytest.raises(ValueError, match="not divisible"):
        reconstruct_layer_qkv(cfg, 0, [], [], block=BLOCK, n_ranks=7)
    per_shard = sum(qkv_segments(cfg, 0)) // 2  # n_ranks=2 valid shards
    w = np.zeros((per_shard, 8), np.uint8)
    s = np.ones((per_shard // BLOCK[0] + 2, 2), np.float32)  # padded grid
    with pytest.raises(ValueError, match="every shard must match"):
        reconstruct_layer_qkv(cfg, 0, [w, w[:-1]], [s, s], block=BLOCK, n_ranks=2)
