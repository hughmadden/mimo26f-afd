"""P-101 / T1+T2 — fused-QKV ckpt_tp=4 TP-order split + poisoned pad rows.

Oracle: golden ``fused_split{}`` (map §3) pins this codec BYTE-EXACT (seed
20260922, regen ``mimo26/scripts/gen-golden.py``; consumed exactly like
``tests/test_golden.py`` ``test_fused_qkv_split_golden``).  Detection power
mirrors map §4 (test_loader.py:136/:146, test_fp8_block.py:105/:121/:148):
  T1  naive shard-major read scrambles Q/K/V -> err > 10x correct + 0.5
  T2  poisoned 1e30 pad row must never leak into data
  +   naive-dequant negative (fp8_block.py:182 global //br bug)

The implementation-under-test calls use the env default, so the suite FAILS
under ``MIMO26_SPIKE_NAIVE=1`` and PASSES on the correct impl (both runs
recorded in runs/20260923-i1-spike/).
"""
from __future__ import annotations

import numpy as np

from spike import loader as L
from spike import quant as Q

from .conftest import GOLDEN, make_fused_case


def test_golden_fused_split_byte_exact(golden):
    """T1/T2 codec pinned byte-exact against the external oracle (map §3)."""
    d = golden["fused_split"]
    fused = np.frombuffer(bytes.fromhex(d["fused_u8_hex"]), np.uint8).reshape(d["fused_rows"], -1)
    fscales = np.frombuffer(bytes.fromhex(d["scales_f32_hex"]), np.float32).reshape(d["scale_shape"])
    parts = Q.split_shard_major_fused([fused], [fscales], tuple(d["segments_rows"]), block=(2, 4))
    for k, v in parts.items():
        exp = d["parts"][k]
        assert list(v.weight.shape) == exp["shape"], k
        assert v.weight.tobytes().hex() == exp["hex"], k
        assert v.scale_per_row.tobytes().hex() == exp["scale_hex"], k


def test_t1_ckpt_tp4_tp_order_split_matches_whole():
    """T1: per-chunk [Q_c|K_c|V_c] -> [Q|K|V] regroup equals the whole-matrix
    order (map §4 test_loader.py:136, n_ranks 4 = ckpt_tp)."""
    weights, scales, segs, ref = make_fused_case(n_ranks=4, segs=(3, 2, 2), cols=8, block=(4, 4))
    got = L.reconstruct_layer_qkv(weights, scales, segs, block=(4, 4))  # env-default impl
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6)


def test_t1_naive_split_scrambles_qkv_negative():
    """Detection power (map §4, test_fp8_block.py:105): naive err > 10x fixed + 0.5."""
    weights, scales, segs, ref = make_fused_case(n_ranks=4, segs=(3, 2, 2), cols=8, block=(4, 4))
    correct_err = float(np.abs(L.reconstruct_layer_qkv(weights, scales, segs, block=(4, 4), naive=False) - ref).max())
    naive_err = float(np.abs(L.reconstruct_layer_qkv(weights, scales, segs, block=(4, 4), naive=True) - ref).max())
    assert naive_err > 10 * correct_err + 0.5, (naive_err, correct_err)


def test_t2_poisoned_pad_row_never_read():
    """T2 (test_fp8_block.py:121): poison pad scale rows with 1e30 — the correct
    path is byte-identical with/without poison; the naive path leaks it."""
    weights, scales, segs, ref = make_fused_case(n_ranks=2, segs=(3, 2, 2), cols=8, block=(4, 4))
    clean = L.reconstruct_layer_qkv(weights, scales, segs, block=(4, 4), naive=False)
    poisoned = []
    for s in scales:
        s2 = s.copy()
        s2[-1, :] = 1e30  # last grid rows are pad for these shapes
        poisoned.append(s2)
    got = L.reconstruct_layer_qkv(weights, poisoned, segs, block=(4, 4), naive=False)
    np.testing.assert_array_equal(got, clean)
    # the env-selected impl must also survive the poison (fails under naive env)
    got_default = L.reconstruct_layer_qkv(weights, poisoned, segs, block=(4, 4))
    np.testing.assert_array_equal(got_default, clean)
    naive = L.reconstruct_layer_qkv(weights, poisoned, segs, block=(4, 4), naive=True)
    assert np.abs(naive - clean).max() > 1.0  # poison leaked -> caught


def test_naive_dequant_global_scale_rows_is_wrong():
    """fp8_block.py:170/:182 bug oracle — global //br over padded grids (T1+T2)."""
    weights, scales, segs, ref = make_fused_case(n_ranks=2, segs=(3, 2, 2), cols=8, block=(4, 4))
    good = L.reconstruct_layer_qkv(weights, scales, segs, block=(4, 4), naive=False)
    bad = Q.dequantize_naive_fused(weights, scales, block=(4, 4))
    assert np.abs(good - bad).max() > 1e-6


def test_uneven_shard_rows_raise():
    """loader.py:84-93 fail-loud (map §2, test_fp8_block.py:148)."""
    weights, scales, segs, ref = make_fused_case(n_ranks=2, segs=(3, 2, 2), cols=8, block=(4, 4))
    weights.append(weights[0][:-1])  # one row short
    scales.append(scales[0])
    try:
        L.reconstruct_layer_qkv(weights, scales, segs, block=(4, 4), naive=False)
    except ValueError:
        return
    raise AssertionError("uneven shard rows did not raise")
