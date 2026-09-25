"""spike/model.py — tiny CPU-twin model for I1 trap tests and the spike loop.

Mirrors ``mimo26/model.py``:
  - fail-loud name audit (model.py:34-42 missing embed/norm/lm_head -> ValueError;
    model.py:85-88 missing ``mlp.gate.e_score_correction_bias`` -> KeyError)
  - T9 companion: ``forward(..., start_pos=None)`` defaults to ``cache.tokens``
    (model.py:110).  The regression: incremental decode that ignores start_pos
    (positions always from 0) must DIFFER from a full recompute.

Positions are used for partial rotary (attn.apply_rotary); KV bookkeeping uses
the cache cursor, so the naive bug shows up exactly as position drift.

Modes (env ``MIMO26_SPIKE_NAIVE=1`` or ctor ``naive=True``):
  NAIVE start_pos: step positions always start at 0 (cache ignored).
  CORRECT:         ``start_pos if start_pos is not None else cache.tokens``.
"""
from __future__ import annotations

import os

import numpy as np

from . import attn, kv


def _naive_default() -> bool:
    return os.environ.get("MIMO26_SPIKE_NAIVE", "") == "1"


class SpikeModel:
    """Tiny decoder stack: embed -> N x (attn + mlp-ish residual) -> lm_head.

    Weights dict (canonical spike names):
      ``embed.weight`` [V, D], ``norm.weight`` [D], ``lm_head.weight`` [V, D],
      per layer ``layers.{i}.attn.qkv_w`` [3D, D], ``layers.{i}.attn.o_w`` [D, D],
      ``layers.{i}.norm.weight``, ``layers.{i}.mlp.gate.e_score_correction_bias``.
    """

    REQUIRED = ("embed.weight", "norm.weight", "lm_head.weight")

    def __init__(self, weights: dict, cfg: dict, naive: bool | None = None):
        self.naive = _naive_default() if naive is None else naive
        self.cfg = cfg
        self.n_layers = int(cfg["n_layers"])
        self.d = int(cfg["d_model"])
        self.v = int(cfg["vocab"])
        self.window = int(cfg["sliding_window"])  # SWA layers only (T3)

        # fail-loud name audit (mirrors model.py:34-42)
        missing = [k for k in self.REQUIRED if k not in weights]
        if missing:
            raise ValueError(f"spike: missing required weights: {missing}")
        for i in range(self.n_layers):
            key = f"layers.{i}.mlp.gate.e_score_correction_bias"
            if key not in weights:  # mirrors model.py:85-88 KeyError
                raise KeyError(f"spike: missing router bias {key}")
        self.w = weights
        self.layers = [
            attn.SelfAttention(
                self.d,
                weights[f"layers.{i}.attn.qkv_w"],
                weights[f"layers.{i}.attn.o_w"],
                sliding_window=None if cfg["is_ga"][i] else self.window,
                sink_dim=cfg.get("sink_dim", 0),
                v_scale=cfg.get("v_scale", 1.0),
                is_ga=bool(cfg["is_ga"][i]),
                naive=self.naive,
            )
            for i in range(self.n_layers)
        ]
        self.kv = kv.KVCache(self.n_layers, naive=self.naive)

    # T9: start_pos default = cache.tokens (model.py:110)
    def forward(self, tokens: np.ndarray, start_pos: int | None = None) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64)
        cache_pos = np.arange(self.kv.tokens, self.kv.tokens + len(tokens), dtype=np.int64)
        if self.naive:
            pos0 = 0  # WRONG: start_pos ignored, cache cursor ignored
        else:
            pos0 = self.kv.tokens if start_pos is None else int(start_pos)
        pos = np.arange(pos0, pos0 + len(tokens), dtype=np.int64)
        h = self.w["embed.weight"][tokens]
        for i, blk in enumerate(self.layers):
            h = blk(h, pos, self.kv, i, cache_pos=cache_pos)
        return h @ self.w["lm_head.weight"].T

    def logits(self, tokens: list[int]) -> np.ndarray:
        return self.forward(np.asarray(tokens, dtype=np.int64))

    def greedy_decode(self, prompt: list[int], n_steps: int) -> list[int]:
        out = list(prompt)
        logits = self.logits(prompt)
        for _ in range(n_steps):
            nxt = int(np.argmax(logits[-1]))
            out.append(nxt)
            logits = self.forward(np.asarray([nxt], dtype=np.int64))  # start_pos=None
        return out


def init_weights(cfg: dict, seed: int = 0, scale: float = 1.0) -> dict:
    """Deterministic tiny weights (twin of mimo26/model.py:132 init_weights).

    ``scale=1.0`` (not the oracle's 0.02) on purpose: the trap tests assert that
    wrong paths DIFFER, and 1e-8-magnitude logits let ``atol`` swallow real
    position/rotation bugs.  O(1) activations keep the negatives honest.
    """
    rng = np.random.default_rng(seed)
    d, v, n = cfg["d_model"], cfg["vocab"], cfg["n_layers"]
    w: dict = {
        "embed.weight": (rng.standard_normal((v, d)) * scale).astype(np.float32),
        "norm.weight": np.ones((d,), np.float32),
        "lm_head.weight": (rng.standard_normal((v, d)) * scale).astype(np.float32),
    }
    for i in range(n):
        w[f"layers.{i}.attn.qkv_w"] = (rng.standard_normal((3 * d, d)) * scale).astype(np.float32)
        w[f"layers.{i}.attn.o_w"] = (rng.standard_normal((d, d)) * scale).astype(np.float32)
        w[f"layers.{i}.norm.weight"] = np.ones((d,), np.float32)
        w[f"layers.{i}.mlp.gate.e_score_correction_bias"] = np.zeros((8,), np.float32)
    return w
