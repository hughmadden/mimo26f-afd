"""P-101 / T9 — incremental decode without start_pos must differ from a full
recompute (the regression, map §4 ``test_model.py:89``).

``mimo26/model.py:110``: ``start_pos = cache.tokens`` default in ``forward``.
The wrong implementation ignores ``start_pos`` and assigns positions from 0 on
every call — RoPE offsets of the new query slide back to the prompt start and
incremental decode diverges from a full recompute (word salad after step 1).

Implementation-under-test calls use the env default, so this file FAILS under
``MIMO26_SPIKE_NAIVE=1`` and PASSES on the correct impl (both runs recorded in
runs/20260923-i1-spike/).

Note on detection geometry: RoPE attention is shift-invariant on an EMPTY cache,
so ``start_pos`` can only be observed against warm-cache rows — every test here
runs the probe token AFTER a prompt.
"""
from __future__ import annotations

import numpy as np

from spike.model import SpikeModel, init_weights

CFG = {
    "n_layers": 2,
    "d_model": 8,
    "vocab": 16,
    "sliding_window": 4,
    "is_ga": [True, False],  # layer 0 GA (T3), layer 1 SWA (eviction T8-class)
    "sink_dim": 0,
    "v_scale": 0.707,
}

PROMPT = [1, 2, 3, 4]
PROBE = np.asarray([5], dtype=np.int64)


def _model(naive: bool | None = None, seed: int = 0) -> SpikeModel:
    return SpikeModel(init_weights(CFG, seed), CFG, naive=naive)


def test_incremental_decode_equals_full_recompute():
    """Twin of test_model.py:25 — incremental step == full recompute row (must
    hold on the correct impl; fails on the naive one)."""
    full = _model().logits(PROMPT + [5])  # env-default impl
    inc_model = _model()
    inc_model.logits(PROMPT)
    inc = inc_model.forward(PROBE)
    np.testing.assert_allclose(inc[-1], full[-1], rtol=1e-5, atol=1e-6)


def test_t9_without_start_pos_differs_from_full_recompute():
    """The regression (map §4 test_model.py:89): incremental decode where the
    model ignores start_pos MUST differ from a full recompute — otherwise this
    test class has no detection power.  And the correct path MUST match."""
    ok_full = _model(naive=False).logits(PROMPT + [5])[-1]
    nv = _model(naive=True)
    nv.logits(PROMPT)
    nv_inc = nv.forward(PROBE)[-1]
    assert not np.allclose(nv_inc, ok_full, rtol=1e-5, atol=1e-6), (
        "naive start_pos path matches full recompute — T9 has no detection power"
    )
    ok = _model(naive=False)
    ok.logits(PROMPT)
    np.testing.assert_allclose(ok.forward(PROBE)[-1], ok_full, rtol=1e-5, atol=1e-6)


def test_explicit_start_pos_overrides_default():
    """start_pos plumbing against a warm cache (empty-cache probes are
    RoPE-shift-invariant and cannot see positions):

    1. ``start_pos=None`` must equal ``start_pos=cache.tokens`` (model.py:110).
    2. A WRONG explicit ``start_pos`` must change the logits — the naive impl
       ignores the parameter and returns identical logits (T9 regression live).
    """
    m = _model()  # env-default impl
    m.logits(PROMPT)
    default = m.forward(PROBE)

    m2 = _model()
    m2.logits(PROMPT)
    explicit = m2.forward(PROBE, start_pos=len(PROMPT))
    np.testing.assert_allclose(explicit, default, rtol=1e-5, atol=1e-6)

    m3 = _model()
    m3.logits(PROMPT)
    wrong = m3.forward(PROBE, start_pos=0)
    assert not np.allclose(wrong, default, rtol=1e-5, atol=1e-6), (
        "start_pos ignored (naive path) — T9 regression live"
    )
