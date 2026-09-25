"""spike/kv.py — KV cache with SWA eviction rule (T8 companion of T9).

Mirrors ``mimo26/kv.py``:
  - ``append`` (:56) with non-decreasing per-sequence positions check (:67-68)
  - eviction (:77-92): ``keep_from = int(pos.min()) - w + 1`` (:82),
    searchsorted drop (:83), old-rule trim ``drop_trim = max(0, n-(w+T))`` (:84)
    as an UPPER BOUND only, ``drop = min(n, max(drop_keep, drop_trim))`` (:85).

Modes (``naive=True`` / ``MIMO26_SPIKE_NAIVE=1``):
  NAIVE eviction: drops ``n-(w+T)`` unconditionally (the old rule) — evicts
  entries newer than ``min(batch_pos) − window + 1`` under a batch, i.e. it can
  evict entries still inside the smallest window.
"""
from __future__ import annotations

import os

import numpy as np


def _naive_default() -> bool:
    return os.environ.get("MIMO26_SPIKE_NAIVE", "") == "1"


class KVCache:
    def __init__(self, n_layers: int, naive: bool | None = None):
        self.naive = _naive_default() if naive is None else naive
        self.n = n_layers
        self.k: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
        self.v: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
        self.p: list[np.ndarray] = [np.zeros((0,), np.int64) for _ in range(n_layers)]
        self.tokens = 0  # T9: default start_pos source (model.py:110)

    def pos_of(self, layer: int) -> np.ndarray:
        return self.p[layer]

    def append(
        self,
        layer: int,
        pos: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        window: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(pos, np.int64)
        if len(self.p[layer]) and pos[0] < self.p[layer][-1]:  # kv.py:67-68
            raise ValueError("spike: KV positions must be non-decreasing")
        self.k[layer].append(np.asarray(k, np.float32))
        self.v[layer].append(np.asarray(v, np.float32))
        self.p[layer] = np.concatenate([self.p[layer], pos])
        self.tokens = int(max(self.tokens, int(pos.max()) + 1))

        n = len(self.p[layer])
        if window is not None and n:
            T = len(pos)
            keep_from = int(pos.min()) - window + 1  # kv.py:82 — the rule
            drop_keep = int(np.searchsorted(self.p[layer], keep_from, side="left"))
            drop_trim = max(0, n - (window + T))  # kv.py:84 old rule, upper bound
            if self.naive:
                drop = drop_trim  # WRONG: ignores keep_from
            else:
                drop = min(n, max(drop_keep, drop_trim))  # kv.py:85
            if drop > 0:
                self.k[layer] = self._trim(self.k[layer], drop)
                self.v[layer] = self._trim(self.v[layer], drop)
                self.p[layer] = self.p[layer][drop:]
        kk = np.concatenate(self.k[layer], axis=0) if self.k[layer] else k
        vv = np.concatenate(self.v[layer], axis=0) if self.v[layer] else v
        return kk, vv

    @staticmethod
    def _trim(chunks: list[np.ndarray], drop: int) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for c in chunks:
            if drop >= len(c):
                drop -= len(c)
                continue
            out.append(c[drop:] if drop else c)
            drop = 0
        return out
