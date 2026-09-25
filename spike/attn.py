"""spike/attn.py — tiny attention twin carrying the T3/T5/T4/T6/T7 trap knobs.

Mirrors:
  - patch 01 side fix (T3): the GA layer must NOT inherit the SWA cache's
    ``sliding_window=128`` (cache_config copy, patch 01 @@ -292).
  - patch 04 (T5): ``v_scale`` is applied to the VALUE stream of the query path
    (@@ -267) AND to injected context-KV values (``precompute_and_store_context_kv``
    @@ -610 ``all_v.mul_(v_scale)``).  The context half is the untested one.
  - partial rotary (AGENTS.md §3: factor 0.334) on the first ``rot_dim`` dims of
    q/k — positions enter the math here, which is what makes T9 detectable.
  - T4 (QK 192 / V 128 first-class): score scaling uses the QK dim, never the
    V dim; ``SelfAttention`` accepts ``d_v != d_qk`` so a swapped
    ``attn_scale(d_qk, d_v)`` call-site is pin-visible (P-106 c2).
  - T6: the sink is an EXTRA softmax-logit column, FAMILY-GATED to SWA only
    (verified config: ``add_swa_attention_sink_bias=true``,
    ``add_full_attention_sink_bias=false`` — same family gate as
    real_loop.py:301 ``sink = (self.sink_swa if is_swa else self.sink_ga)``;
    P-106 c1).

Modes (``naive=True`` / env ``MIMO26_SPIKE_NAIVE=1``):
  NAIVE window: every layer inherits ``sliding_window=128`` (GA included).
  NAIVE vscale: query values are scaled, context-KV values are not.
  NAIVE sink:   the sink column is dropped entirely (T6 bug).
  NAIVE theta:  one rope theta for both families (T7 bug).
  NAIVE scale:  attn_scale returns the V-dim root (T4 bug).
"""
from __future__ import annotations

import os

import numpy as np


def _naive_default() -> bool:
    return os.environ.get("MIMO26_SPIKE_NAIVE", "") == "1"


def ga_cache_config(cfg: dict) -> dict:
    """patch 01 @@ -292: cache_config copy with sliding_window=None for GA.

    Correct: GA layers get ``sliding_window: None`` even though the SWA cache
    config carries 128.  Naive: the SWA config is shared verbatim.
    """
    naive = _naive_default()
    if naive:
        return dict(cfg)  # GA inherits sliding_window=128 -> T3 bug
    out = dict(cfg)
    out["sliding_window"] = None
    return out


def apply_rotary(x: np.ndarray, pos: np.ndarray, theta: float = 10_000.0,
                 partial_rotary_factor: float = 0.334) -> np.ndarray:
    """Partial rotary on the first ``rot_dim`` dims (half-split rotation).
    Twin of mimo26/nn/layers.apply_rotary at toy dims (QK 192 / V 128 lives in
    loader/quant geometry)."""
    d = x.shape[-1]
    rot = int(d * partial_rotary_factor)
    rot -= rot % 2
    if rot == 0:
        return x
    half = rot // 2
    freqs = 1.0 / (theta ** (np.arange(half) / max(half, 1)))
    ang = pos[:, None] * freqs[None, :]  # [T, half]
    cos, sin = np.cos(ang), np.sin(ang)
    out = x.copy()
    a, b = x[..., :half], x[..., half:rot]
    out[..., :half] = a * cos - b * sin
    out[..., half:rot] = a * sin + b * cos
    return out


class SelfAttention:
    """Toy attention: qkv_w [2*d_qk + d_v, d_model], o_w [d_model, d_v].

    ``d_qk`` (first ctor arg) is the Q/K width; ``d_v`` (optional, default
    equal) is the VALUE width — QK/V asymmetry is first-class (T4).  The sink
    is per-Q-head in the real model ([64]); the toy carries one scalar logit.
    """

    def __init__(
        self,
        d: int,
        qkv_w: np.ndarray,
        o_w: np.ndarray,
        sliding_window: int | None = None,
        sink_dim: int = 0,
        v_scale: float = 1.0,
        is_ga: bool = False,
        naive: bool | None = None,
        sink_bias: float = 0.0,
        d_v: int | None = None,
    ):
        self.naive = _naive_default() if naive is None else naive
        self.d_qk = int(d)
        self.d_v = int(d) if d_v is None else int(d_v)
        self.d = self.d_qk  # back-compat alias (single-width callers)
        qkv = np.asarray(qkv_w, np.float32)
        # rows [q d_qk | k d_qk | v d_v] — identical to a 3-way split at d_v == d_qk
        self.wq, self.wk, self.wv = np.split(qkv, [self.d_qk, 2 * self.d_qk], axis=0)
        self.wo = np.asarray(o_w, np.float32)
        # T3: naive treats every layer as SWA.
        if self.naive and sliding_window is None:
            sliding_window = 128
        self.window = sliding_window
        self.is_ga = is_ga
        self.sink_dim = sink_dim
        self.sink_bias = float(sink_bias)
        self.v_scale = float(v_scale)

    def project_qkv(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = x @ self.wq.T
        k = x @ self.wk.T
        v = x @ self.wv.T * self.v_scale  # T5 query-path half (patch 04 @@ -267)
        return q, k, v

    def inject_context_kv(self, k_ctx: np.ndarray, v_ctx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """patch 04 @@ -610 ``precompute_and_store_context_kv``: ``all_v.mul_(v_scale)``.

        Correct: injected context values are scaled by v_scale like the query
        path.  Naive: stored raw -> query-vs-context magnitude drift (T5).
        """
        v = np.asarray(v_ctx, dtype=np.float32)
        if not self.naive:
            v = v * self.v_scale
        return np.asarray(k_ctx, np.float32), v

    def __call__(self, x: np.ndarray, pos: np.ndarray, cache, layer: int,
                 cache_pos: np.ndarray | None = None) -> np.ndarray:
        pos = np.asarray(pos, np.int64)
        q, k, v = self.project_qkv(x)
        theta = rope_theta_for(self.is_ga, naive=self.naive)  # T7 dual-θ: GA 1e7 / SWA 1e4
        q = apply_rotary(q, pos, theta=theta)  # partial rotary 0.334 — positions enter here
        k = apply_rotary(k, pos, theta=theta)
        cpos = pos if cache_pos is None else np.asarray(cache_pos, np.int64)
        k2, v2 = cache.append(layer, cpos, k, v, window=self.window)
        # T4: scaling = d_qk**-0.5 — taken from the LIVE tensor widths so a
        # swapped attn_scale call-site is pin-visible at d_qk != d_v (c2).
        # naive= is threaded from the CTOR flag (like window/sink) — an
        # env-only default here made explicit-naive blocks silently correct
        # (caught by test_c2_attn_scale_through_call_negative, 23 Sep 2026).
        att = q @ k2.T * attn_scale(q.shape[-1], v2.shape[-1], naive=self.naive)
        kpos = cache.pos_of(layer)[None, :]
        # causal: a key is visible only at or before its query's position —
        # without this, full prefill leaks future rows and incremental decode
        # can never match a full recompute (the T9 equality is a trap gate).
        keep = kpos <= pos[:, None]
        if self.window is not None:
            keep = keep & ((pos[:, None] - kpos) < self.window)
        att = np.where(keep, att, -1e30)
        # T6, family-gated (P-106 c1): the sink is LIVE on SWA only (verified
        # config add_swa_attention_sink_bias=true /
        # add_full_attention_sink_bias=false; real_loop.py:301 same gate).
        sink_live = self.sink_dim > 0 and not self.is_ga and not self.naive
        if sink_live:
            # Oracle eager_attention_forward :89-96: the per-Q-head sink bias is
            # an EXTRA softmax-logit column; after softmax the column is dropped —
            # its mass is absorbed, never renormalized onto the keys.  Naive:
            # sink dropped entirely.
            att = np.concatenate(
                [att, np.full((att.shape[0], 1), self.sink_bias, np.float32)], axis=1
            )
        p = np.exp(att - att.max(axis=-1, keepdims=True))
        p = p / p.sum(axis=-1, keepdims=True)
        if sink_live:
            p = p[:, :-1]  # drop the sink column AFTER softmax (mass absorbed)
        return (p @ v2) @ self.wo.T


GA_ROPE_THETA = 1.0e7   # verified config rope_theta (GA layers)
SWA_ROPE_THETA = 1.0e4  # verified config swa_rope_theta (SWA layers)


def rope_theta_for(is_ga: bool, naive: bool | None = None) -> float:
    """T7 dual-θ rotary (verified config: GA ``rope_theta=1e7`` vs SWA
    ``swa_rope_theta=1e4``; oracle :28-42 rotates only the first 64 of 192
    dims, split [rope|nope]).  Naive: ONE θ for both families (SWA's 1e4)."""
    naive = _naive_default() if naive is None else naive
    if naive:
        return SWA_ROPE_THETA
    return GA_ROPE_THETA if is_ga else SWA_ROPE_THETA


def attn_scale(d_qk: int, d_v: int, naive: bool | None = None) -> float:
    """T4 QK192/V128: score scaling is ``head_dim ** -0.5`` with head_dim = the
    QK dim (192) — the V dim (128) is the VALUE width and never enters scaling
    (oracle :61-63).  Argument order is (d_qk, d_v) and is PINNED — swapping
    them at a call-site changes the result and must fail
    test_c2_attn_scale_arg_order_pin / test_c2_attn_scale_through_call_negative.
    Naive: scales by the V dim (the QK/V mixup)."""
    naive = _naive_default() if naive is None else naive
    return float(d_v if naive else d_qk) ** -0.5
