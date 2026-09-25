"""spike/tests/conftest.py — shared fixtures for the I1 negative suite.

Golden oracle (map §3), referenced READ-ONLY per I-Gold (no copy-in, no
docs/REUSE.md row — lead, 2026-09-22):
  oracle/goldens/fp8_block_golden.json
  (sha256 d8f45ad930120c29820fe1b63aa56b05624564dfb6325459c3aab52df1abc9c6;
  regen ``code/scripts/gen-golden.py``, seed 20260922).
Missing golden = loud fail unless ``MIMO26_ALLOW_MISSING_GOLDEN=1`` (mirrors
``tests/test_golden.py:14-22``).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

GOLDEN = (
    pathlib.Path(__file__).resolve().parents[2]
    / "oracle/goldens/fp8_block_golden.json"
)


@pytest.fixture(scope="session")
def golden() -> dict:
    if not GOLDEN.exists():
        if os.environ.get("MIMO26_ALLOW_MISSING_GOLDEN") == "1":
            pytest.skip(f"golden missing at {GOLDEN} (MIMO26_ALLOW_MISSING_GOLDEN=1)")
        pytest.fail(f"missing golden {GOLDEN} (regen mimo26 scripts/gen-golden.py)")
    return json.loads(GOLDEN.read_text())


def make_fused_case(n_ranks=2, segs=(3, 2, 2), cols=8, block=(4, 4), seed=7, pad_rows=1):
    """Synthetic shard fixture in REAL layout (map §4: _synth_shards /
    _make_fused_case): per-shard ``[q_c|k_c|v_c]`` rows in TP-rank order, per-shard
    scale grids PADDED to ``ceil(rows/br) + pad_rows`` (the pad rows carry 1.0 and
    are poisonable).

    Returns (shard_weights, shard_scales, segs, ref) where ``ref`` is the
    projection-major ``[Q|K|V]`` f32 matrix built from construction data alone
    (codes decoded times the scale each row was quantized with) — independent of
    the implementation under test.
    """
    from spike import quant as Q

    rng = np.random.default_rng(seed)
    r_q, r_k, r_v = segs
    per = sum(segs)
    br, bc = block
    weights: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    per_shard_rows: list[list[np.ndarray]] = []  # f32 dequantized rows per shard
    for _ in range(n_ranks):
        w = rng.standard_normal((per, cols)).astype(np.float64) * 0.5
        codes, grid = Q.quantize_block(w, block=block)
        real_rows = -(-per // br)
        pad = np.ones((real_rows + pad_rows, grid.shape[1]), np.float32)
        pad[:real_rows] = grid
        # construction-side reference: each row's true scale = pad[row//br, col//bc]
        row_scale = pad[np.arange(per) // br, : cols // bc].astype(np.float64)
        vals = Q.decode_e4m3(codes).reshape(per, cols // bc, bc)
        rows_f32 = (vals * row_scale[:, :, None]).reshape(per, cols).astype(np.float32)
        weights.append(codes)
        scales.append(pad)
        per_shard_rows.append([rows_f32])
    q = np.concatenate([np.concatenate([r[0][0:r_q] for r in [rows]], axis=0) for rows in per_shard_rows], axis=0)
    k = np.concatenate([rows[0][r_q:r_q + r_k] for rows in per_shard_rows], axis=0)
    v = np.concatenate([rows[0][r_q + r_k:] for rows in per_shard_rows], axis=0)
    ref = np.concatenate([q, k, v], axis=0)
    return weights, scales, tuple(segs), ref
