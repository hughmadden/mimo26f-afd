"""P-101 / T3 — GA layers must NOT inherit sliding_window=128 (patch 01 side fix).

tonyd2wild patch 01 @@ -292 copies the SWA cache_config with
``sliding_window=None`` for the GA layers and threads ``cache_config=`` into both
SelfAttn ctors.  The wrong (naive) config copy lets GA attention run with a 128
window — long-context word salad on global layers.

Implementation-under-test calls use the env default, so this file FAILS under
``MIMO26_SPIKE_NAIVE=1`` and PASSES on the correct impl (both runs recorded in
runs/20260923-i1-spike/).
"""
from __future__ import annotations

import numpy as np

import spike.kv as kvmod
from spike import attn


def test_ga_cache_config_drops_sliding_window():
    swa_cfg = {"sliding_window": 128, "kv_dtype": "fp8"}
    ga = attn.ga_cache_config(swa_cfg)  # env-default impl
    assert ga["sliding_window"] is None, f"GA inherited {ga['sliding_window']!r}"


def test_swa_cfg_not_mutated():
    swa_cfg = {"sliding_window": 128, "kv_dtype": "fp8"}
    attn.ga_cache_config(swa_cfg)
    assert swa_cfg["sliding_window"] == 128  # copy, not in-place


def test_ga_layer_has_no_window():
    """Behavioral twin: a GA block built with sliding_window=None keeps
    window=None on the correct impl; the naive impl windows it at 128 (the bug)."""
    d = 8
    rng = np.random.default_rng(3)
    qkv = rng.standard_normal((3 * d, d)).astype(np.float32)
    ow = np.eye(d, dtype=np.float32)
    ga = attn.SelfAttention(d, qkv, ow, sliding_window=None, is_ga=True)  # env default
    assert ga.window is None, f"GA layer carries window {ga.window!r} (T3 bug)"


def test_ga_window_bug_is_visible():
    """Detection power: on a 200-token sequence the naive (windowed) GA block
    and the correct GA block must disagree, or this test class has no power."""
    d = 8
    rng = np.random.default_rng(3)
    qkv = rng.standard_normal((3 * d, d)).astype(np.float32)
    ow = np.eye(d, dtype=np.float32)
    ga_ok = attn.SelfAttention(d, qkv, ow, sliding_window=None, is_ga=True, naive=False)
    ga_nv = attn.SelfAttention(d, qkv, ow, sliding_window=None, is_ga=True, naive=True)
    n = 200
    x = rng.standard_normal((n, d)).astype(np.float32) * 0.1
    pos = np.arange(n)
    out_ok = ga_ok(x, pos, kvmod.KVCache(1, naive=False), 0)
    out_nv = ga_nv(x, pos, kvmod.KVCache(1, naive=True), 0)
    assert not np.allclose(out_ok, out_nv), "GA window bug is invisible — test has no power"
