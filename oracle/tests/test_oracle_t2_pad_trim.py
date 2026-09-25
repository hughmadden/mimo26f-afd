"""T2 — per-shard PADDED scale grids: pad rows must NEVER be read.

Fresh mirror of the spike T2 class (spike/tests/test_t1_fused_qkv_split.py:54-70),
exercised against the TWIN's local trim (oracle/mimo26/quant/fp8_block.py:144-147
— shard row ``r`` resolves scale row ``r // br`` of THAT shard's grid).

Two-run classification (suite convention):
  NEGATIVE — FAILS under ``MIMO26_SPIKE_NAIVE=1`` (global ``row // br`` across
  per-shard padded grids — fp8_block.py:182), PASSES on the correct path:
    test_t2_pad_scale_rows_never_read_negative
  BOTH RUNS: the pad-leak detection test and the trim-shape pin.
"""
from __future__ import annotations

import numpy as np

from mimo26.quant import fp8_block as fb

from .conftest import build_fused_case, dequant_fused

BLOCK = (4, 4)


def _poison(scales, real_rows):
    """Pad-row poison: 1e30 in every grid row past the real ones (T2)."""
    out = []
    for s in scales:
        s2 = s.copy()
        s2[real_rows:, :] = 1e30
        out.append(s2)
    return out


def test_t2_pad_scale_rows_never_read_negative():
    """NEGATIVE — flips on the naive run.

    Poison every pad scale row with 1e30: the correct path must be
    BIT-identical to the clean run (pad rows never indexed), and so must the
    env-selected implementation (FAILS under MIMO26_SPIKE_NAIVE=1)."""
    segs = (6, 4, 2)
    per = sum(segs)
    real_rows = -(-per // BLOCK[0])
    weights, scales, segs, _expected = build_fused_case(segs=segs)
    poisoned = _poison(scales, real_rows)
    clean = dequant_fused(weights, scales, segs, block=BLOCK, naive=False)
    np.testing.assert_array_equal(
        dequant_fused(weights, poisoned, segs, block=BLOCK, naive=False), clean
    )
    np.testing.assert_array_equal(
        dequant_fused(weights, poisoned, segs, block=BLOCK), clean
    )  # env-selected impl (flips on the naive run)


def test_t2_naive_pad_leak_observable():
    """Detection power (both runs): the twin's naive function must LEAK the
    poison, or this trap class has no teeth (fp8_block.py:170-184)."""
    segs = (6, 4, 2)
    per = sum(segs)
    real_rows = -(-per // BLOCK[0])
    weights, scales, segs, _expected = build_fused_case(segs=segs)
    poisoned = _poison(scales, real_rows)
    clean = dequant_fused(weights, scales, segs, block=BLOCK, naive=False)
    leaked = dequant_fused(weights, poisoned, segs, block=BLOCK, naive=True)
    assert float(np.abs(leaked - clean).max()) > 1.0, "naive pad-row leak invisible"


def test_t2_scale_per_row_trim_shape_pin():
    """Pin (both runs): ``scale_per_row`` is ``[rows, cols//bc]`` and equals the
    per-shard local resolution ``grid[r // br, : cols//bc]`` — the T2 trim
    contract itself (fp8_block.py:144-147)."""
    segs = (6, 4, 2)
    weights, scales, segs, _expected = build_fused_case(segs=segs, n_ranks=2)
    parts = fb.split_shard_major_fused(weights, scales, segs, block=BLOCK)
    r_q, r_k, r_v = segs
    bounds = {"q": (0, r_q), "k": (r_q, r_q + r_k), "v": (r_q + r_k, sum(segs))}
    for name, (lo, hi) in bounds.items():
        want = np.concatenate(
            [s[np.arange(lo, hi) // BLOCK[0], : w.shape[1] // BLOCK[1]]
             for w, s in zip(weights, scales)], axis=0
        )
        n_shards = len(weights)
        want_shape = ((hi - lo) * n_shards, weights[0].shape[1] // BLOCK[1])
        assert parts[name].scale_per_row.shape == want_shape, f"{name} trim shape"
        assert parts[name].scale_per_row.tobytes() == want.tobytes(), f"{name} trim values"
