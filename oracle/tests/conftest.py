"""oracle/tests — ORACLE numerics negatives + pins against the imported CPU twin.

Fresh mirrors (NOT verbatim copies) of the T1/T2 trap classes pinned by the I1
spike suite (spike/tests/*), here exercised against `oracle/mimo26` itself — the
numerics contract whose REUSE rows land from runs/20260923-i2/P-201-import.md §5.

Naive discipline: the twin keeps the CORRECT path
(oracle/mimo26/quant/fp8_block.py:120 split_shard_major_fused + :161
dequantize_per_row) and the preserved bug oracle (:170 dequantize_naive_fused)
side by side.  `dequant_fused` below is the suite's env-default selector —
`MIMO26_SPIKE_NAIVE=1` (the same suite convention as spike/tests) picks the
twin's naive function.  Both branches are REAL twin code; nothing is mocked.

Golden oracle (read-only, I-Gold): the external corpus under
`oracle/goldens/` — regenerated/
checked by `oracle/scripts/gen-golden.py --check` (the ci-cpu gate runs it).
Missing corpus = loud fail unless `MIMO26_ALLOW_MISSING_GOLDEN=1` (mirrors
spike/tests/conftest.py:8-9).
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest

ORACLE = pathlib.Path(__file__).resolve().parents[1]  # oracle/
sys.path.insert(0, str(ORACLE))  # `import mimo26` -> oracle/mimo26

from mimo26.quant import fp8_block as fb  # noqa: E402

GOLDEN_DIR = pathlib.Path(os.environ.get(
    "MIMO26_GOLDEN_DIR",
    str(pathlib.Path(__file__).resolve().parents[2] / "oracle/goldens"),
))


def naive_env() -> bool:
    return os.environ.get("MIMO26_SPIKE_NAIVE", "") == "1"


def dequant_fused(shard_weights, shard_scales, segs_per_shard, block, naive=None):
    """Projection-major ``[Q|K|V]`` f32 dequant of the fused shards.

    naive=False -> twin correct path (fp8_block.py:120 + :161-167);
    naive=True  -> twin bug oracle (fp8_block.py:170-184);
    naive=None  -> env default MIMO26_SPIKE_NAIVE (suite convention).
    """
    naive = naive_env() if naive is None else naive
    if naive:
        return fb.dequantize_naive_fused(shard_weights, shard_scales, block=block)
    parts = fb.split_shard_major_fused(shard_weights, shard_scales, segs_per_shard, block=block)
    return np.concatenate(
        [fb.dequantize_per_row(parts[n], block=block) for n in ("q", "k", "v")], axis=0
    )


@pytest.fixture(scope="session")
def golden_dir() -> pathlib.Path:
    if not GOLDEN_DIR.is_dir():
        if os.environ.get("MIMO26_ALLOW_MISSING_GOLDEN") == "1":
            pytest.skip(f"golden dir missing at {GOLDEN_DIR} (MIMO26_ALLOW_MISSING_GOLDEN=1)")
        pytest.fail(f"missing golden corpus {GOLDEN_DIR} (check: oracle/scripts/gen-golden.py --check)")
    return GOLDEN_DIR


def build_fused_case(n_ranks=3, segs=(6, 4, 2), cols=8, block=(4, 4), pad_rows=2, seed=20260923):
    """Synthetic TP-ordered fused shards with PADDED per-shard scale grids.

    Fresh fixture (the I1 analogue is spike/tests/conftest.py:39-76 — cited, not
    copied): shard ``s`` holds local rows ``[q_s; k_s; v_s]`` (TP order) as raw
    e4m3 codes; each shard's grid is ``[ceil(per/br) + pad_rows, ceil(cols/bc)]``
    f32 — real rows carry the scales each row was built with, pad rows carry 1.0
    and are poisonable (T2).

    Returns ``(shard_weights, shard_scales, segs, expected)`` where ``expected``
    is the projection-major ``[Q|K|V]`` f32 matrix built from the construction
    mapping alone — row ``r`` of shard ``s`` is
    ``decode_e4m3(codes[r]) * grid[r // br, c // bc]`` (fp8_block.py:41-42 for
    the decode) — independent of the split/trim path under test.
    """
    br, bc = block
    r_q, r_k, r_v = segs
    per = r_q + r_k + r_v
    rng = np.random.default_rng(seed)
    real_rows = -(-per // br)
    grid_cols = -(-cols // bc)
    weights, scales, rows_all = [], [], []
    for _ in range(n_ranks):
        codes = rng.integers(0, 0x7F, size=(per, cols)).astype(np.uint8)  # no NaN codes
        grid = np.ones((real_rows + pad_rows, grid_cols), np.float32)
        grid[:real_rows] = (rng.integers(1, 5, size=(real_rows, grid_cols)) * 0.5).astype(np.float32)
        row_scale = grid[np.arange(per) // br, : cols // bc].astype(np.float64)
        vals = fb.decode_e4m3(codes).reshape(per, cols // bc, bc)
        rows_f32 = (vals * row_scale[:, :, None]).reshape(per, cols).astype(np.float32)
        weights.append(codes)
        scales.append(grid)
        rows_all.append(rows_f32)
    q = np.concatenate([r[0:r_q] for r in rows_all], axis=0)
    k = np.concatenate([r[r_q:r_q + r_k] for r in rows_all], axis=0)
    v = np.concatenate([r[r_q + r_k:] for r in rows_all], axis=0)
    expected = np.concatenate([q, k, v], axis=0)
    return weights, scales, segs, expected
