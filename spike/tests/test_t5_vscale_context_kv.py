"""P-101 / T5 — query-vs-context v_scale mismatch (patch 04 @@ -610 half).

tonyd2wild patch 04 scales the drafter's VALUE stream by ``attention_value_scale``
(0.612 DFlash / 0.707 target) both on the query path (@@ -267 ``v = v*v_scale``)
and on injected context-KV values (``precompute_and_store_context_kv`` @@ -610
``all_v.mul_(v_scale)``).  map §1: the @@ -610 context half has NO twin test
anywhere — this is it.

Method: feed the SAME pre-scale value matrix ``v_pre = x @ wv.T`` through both
halves and compare streams — comparing projected-vs-raw magnitudes (as an early
draft did) is meaningless because the projection itself rescales.

Blocks use the env-default implementation, so this file FAILS under
``MIMO26_SPIKE_NAIVE=1`` (context values stored raw) and PASSES on the correct
impl.  Both runs recorded in runs/20260923-i1-spike/.
"""
from __future__ import annotations

import numpy as np

from spike import attn

V_SCALE = 0.707  # target MiMo-2.6 value scale (AGENTS.md §3)


def _blk(naive: bool | None = None, d: int = 8, seed: int = 11) -> attn.SelfAttention:
    rng = np.random.default_rng(seed)
    qkv = rng.standard_normal((3 * d, d)).astype(np.float32)
    ow = np.eye(d, dtype=np.float32)
    return attn.SelfAttention(d, qkv, ow, sliding_window=None, v_scale=V_SCALE, naive=naive)


def _streams(blk: attn.SelfAttention, seed: int = 9):
    """(v_pre, v_query_path, v_context_path) for one shared pre-scale matrix."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((3, 8)).astype(np.float32)
    v_pre = x @ blk.wv.T
    _, _, v_q = blk.project_qkv(x)               # v_pre * v_scale (both impls)
    _, v_c = blk.inject_context_kv(v_pre, v_pre)  # correct: * v_scale; naive: raw
    return v_pre, v_q, v_c


def test_context_kv_values_are_scaled():
    """@@ -610: injected ``all_v`` must carry the same v_scale as the query path."""
    blk = _blk()  # env-default impl
    _, v_q, v_c = _streams(blk)
    np.testing.assert_allclose(v_c, v_q, rtol=1e-6)  # streams agree


def test_query_and_context_v_scale_match_negative():
    """The regression, both directions of detection power:

    correct  — query-path and context-path values from the same pre-scale matrix
               agree exactly (both scaled);
    naive    — the context stream is off by exactly ``1/v_scale`` vs the query
               stream (query scaled, context raw), and the env-selected impl
               under test must show the agreement or this run fails.
    """
    v_pre, v_q, v_c_env = _streams(_blk())          # env-selected (naive run flips here)
    np.testing.assert_allclose(v_c_env, v_q, rtol=1e-6)

    _, v_q_ok, v_c_ok = _streams(_blk(naive=False))
    _, v_q_nv, v_c_nv = _streams(_blk(naive=True))
    np.testing.assert_allclose(v_c_ok, v_q_ok, rtol=1e-6)
    assert not np.allclose(v_c_nv, v_q_nv, rtol=1e-6)  # F9: assert the mismatch is DETECTABLE, not the naive mock's exact 1/v_scale signature
    assert not np.allclose(v_c_nv, v_q_nv, rtol=1e-6), (
        "context-vs-query mismatch invisible on naive impl — T5 has no detection power"
    )
