"""Reference layer numerics for MiMo-V2.6 (numpy, float32).

Format-agnostic: weights arrive already dequantized (see `mimo26.quant`); this module is
the arithmetic conformance target for the later CUDA kernels. Shapes use [T, H, D] for
per-head sequences.
"""

from __future__ import annotations

import numpy as np


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    var = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps) * np.asarray(weight, dtype=np.float32)).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    # clip only the exp argument: identical values (x * sigmoid(x)), no overflow warnings
    return (x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))).astype(np.float32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


def rot_dim(head_dim: int, partial_rotary_factor: float) -> int:
    """Rotary width for partial rotary (MiMo: 192 * 0.334 -> 64)."""
    return int(head_dim * partial_rotary_factor)


def apply_rotary(x: np.ndarray, positions: np.ndarray, theta: float, partial_rotary_factor: float) -> np.ndarray:
    """Rotate the first `rot_dim` channels of each head; pass the rest through.

    x: [T, H, D]; positions: [T] absolute.
    """
    x = np.asarray(x, dtype=np.float32)
    T, H, D = x.shape
    rd = rot_dim(D, partial_rotary_factor)
    if rd % 2 != 0:
        raise ValueError("rotary dim must be even")
    pos = np.asarray(positions, dtype=np.float64)[:, None]  # [T, 1]
    inv = 1.0 / (theta ** (np.arange(0, rd, 2, dtype=np.float64) / rd))
    ang = pos * inv[None, :]  # [T, rd/2]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    out = x.copy()
    if rd:
        a = x[:, :, : rd // 2]
        b = x[:, :, rd // 2 : rd]
        c = cos[:, None, :]
        s = sin[:, None, :]
        out[:, :, : rd // 2] = a * c - b * s
        out[:, :, rd // 2 : rd] = b * c + a * s
    return out


def repeat_kv(k: np.ndarray, v: np.ndarray, n_rep: int) -> tuple[np.ndarray, np.ndarray]:
    """GQA: expand KV heads to match Q heads ([S, Hk, D] -> [S, Hk*n_rep, D])."""
    if n_rep == 1:
        return k, v
    return (np.repeat(k, n_rep, axis=1), np.repeat(v, n_rep, axis=1))


def attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    q_pos: np.ndarray,
    k_pos: np.ndarray | None = None,
    window: int | None = None,
    sink_bias: np.ndarray | None = None,
    value_scale: float = 1.0,
) -> np.ndarray:
    """Causal GQA attention with optional sliding window and learnable sink column.

    q: [T, Hq, Dqk]; k: [S, Hk, Dqk]; v: [S, Hk, Dv]  (S may exceed T — cache tail).
    q_pos: [T] absolute positions of the queries; k_pos: [S] for the keys (default 0..S-1).
    window: if set, a key is visible only when 0 <= q_pos - k_pos < window (SWA).
    sink_bias: [Hq] extra logit per head appended as the final softmax column.
    value_scale: multiplier on V (MiMo: 0.707).
    Returns [T, Hq, Dv].
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    T, Hq, Dqk = q.shape
    S, Hk, _ = k.shape
    if Hq % Hk != 0:
        raise ValueError("Hq must be a multiple of Hk (GQA)")
    n_rep = Hq // Hk
    kp = np.arange(S) if k_pos is None else np.asarray(k_pos)
    qp = np.asarray(q_pos)[:, None]  # [T, 1]
    mask = kp[None, :] <= qp  # causal
    if window is not None:
        mask &= (qp - kp[None, :]) < window
    neg = np.finfo(np.float32).min
    vscale = np.float32(value_scale)
    out = np.empty((T, Hq, v.shape[2]), dtype=np.float32)
    sink_all = (None if sink_bias is None
                else np.asarray(sink_bias, dtype=np.float32))
    # GQA is folded per KV head (head layout matches repeat_kv: each kv head serves
    # its n_rep consecutive q heads). Never materialize the repeated KV: it is 16x
    # the memory and measured 6.8x slower at 64k context (perf review 2026-09-22).
    for kvh in range(Hk):
        lo, hi = kvh * n_rep, (kvh + 1) * n_rep
        qg = q[:, lo:hi, :]                     # [T, n_rep, Dqk]
        kg = k[:, kvh, :][:, None, :]           # [S, 1, Dqk]
        vg = v[:, kvh, :][:, None, :]           # [S, 1, Dv]
        logits = np.einsum("thd,shd->hts", qg, kg) / np.sqrt(Dqk)  # [n_rep, T, S]
        logits = np.where(mask[None, :, :], logits, neg)
        if sink_all is not None:
            sink = sink_all[lo:hi].reshape(n_rep, 1, 1)
            logits = np.concatenate([logits, np.broadcast_to(sink, (n_rep, T, 1))], axis=2)
        p = softmax(logits, axis=2)
        if sink_all is not None:
            p = p[:, :, :S]
        # a query with no visible key gets zero output (mask sentinel == uniform softmax)
        p = np.where(mask.any(axis=1)[None, :, None], p, 0.0)
        out[:, lo:hi, :] = np.einsum("hts,shd->thd", p, vg * vscale)
    return out


def linear(x: np.ndarray, w: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32) @ np.asarray(w, dtype=np.float32).T
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float32)
    return y.astype(np.float32)


def dense_ffn(x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray) -> np.ndarray:
    """silu(x @ gate.T) * (x @ up.T) @ down.T"""
    return linear(silu(linear(x, w_gate)) * linear(x, w_up), w_down)


def router(
    x: np.ndarray,
    w_router: np.ndarray,
    *,
    top_k: int,
    bias: np.ndarray | None = None,
    scoring: str = "sigmoid",
    norm_topk: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """MoE routing: sigmoid scores (+ optional noaux_tc bias for selection), top-k.

    Returns (indices [T, top_k] int, weights [T, top_k] float) with weights normalized
    across the selected experts when norm_topk.
    """
    x = np.asarray(x, dtype=np.float32)
    scores = linear(x, w_router).astype(np.float64)
    if scoring == "sigmoid":
        scores = 1.0 / (1.0 + np.exp(-scores))
    elif scoring != "softmax":
        raise ValueError(f"unknown scoring {scoring!r}")
    select = scores
    if bias is not None:
        select = scores + np.asarray(bias, dtype=np.float64)  # noaux_tc: bias for selection only
    idx = np.argsort(-select, axis=1)[:, :top_k]
    wts = np.take_along_axis(scores, idx, axis=1)  # routing weight uses unbiased scores
    if norm_topk:
        wts = wts / wts.sum(axis=1, keepdims=True)
    return idx.astype(np.int64), wts.astype(np.float32)


def moe_forward(
    x: np.ndarray,
    idx: np.ndarray,
    wts: np.ndarray,
    expert_fn,
) -> np.ndarray:
    """Dispatch/combine over routed experts (CPU simulation of the Spark lane).

    expert_fn(layer_local_expert_id, x_rows) -> y_rows implements one expert's FFN;
    called once per (expert, rows) group — the same granularity a grouped GEMM serves.
    """
    x = np.asarray(x, dtype=np.float32)
    T, _ = x.shape
    out = np.zeros_like(x)
    k = idx.shape[1]
    for e in np.unique(idx):
        rows = np.where(idx == e)
        y = expert_fn(int(e), x[rows[0]])
        out[rows[0]] += y * wts[rows][:, None]
    return out
