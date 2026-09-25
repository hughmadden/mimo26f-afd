"""KV cache and pool accounting (ARCHITECTURE.md §7).

9 GA layers accumulate every token; 39 SWA layers are ring-capped at `sliding_window`.
Accounting is what the placement planner and the admission stub ship on, so the byte math
is asserted against the config in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .config import SWA

if TYPE_CHECKING:
    from .config import MiMoConfig


class KVCache:
    """Per-sequence KV store. GA layers grow linearly; SWA layers keep a ring of
    `cfg.sliding_window` entries (with their absolute positions).

    Rows live in capacity-doubling buffers: appends are O(1) amortized instead of
    re-copying the whole history per token (perf review 2026-09-22: the old
    concatenate-per-append was O(T^2) bytes per GA sequence). `get` returns views.
    """

    def __init__(self, cfg, dtype=np.float32):
        self.cfg = cfg
        self.dtype = dtype
        self._kb: dict[int, np.ndarray] = {}  # row buffers [capacity, ...]
        self._vb: dict[int, np.ndarray] = {}
        self._pb: dict[int, np.ndarray] = {}  # int64 position rows
        self._n_rows: dict[int, int] = {}     # live rows per layer
        self._n = 0  # tokens appended (sequence position cursor)

    @property
    def tokens(self) -> int:
        return self._n

    def _layer_kind(self, layer: int) -> int:
        return self.cfg.hybrid_layer_pattern[layer]

    @staticmethod
    def _grow(buf: np.ndarray, need: int, rows_shape: tuple, dtype) -> np.ndarray:
        cap = 8 if buf is None else buf.shape[0] * 2
        while cap < need:
            cap *= 2
        new = np.zeros((cap, *rows_shape), dtype=dtype)
        if buf is not None and buf.shape[0]:
            new[: buf.shape[0]] = buf
        return new

    def append(self, layer: int, k: np.ndarray, v: np.ndarray, positions: np.ndarray | None = None) -> None:
        k = np.atleast_2d(np.asarray(k, dtype=self.dtype))
        v = np.atleast_2d(np.asarray(v, dtype=self.dtype))
        if k.shape[0] != v.shape[0]:
            raise ValueError("k/v row mismatch")
        T = k.shape[0]
        if T == 0:
            return
        pos = (np.arange(self._n, self._n + T) if positions is None
               else np.asarray(positions, dtype=np.int64))
        n = self._n_rows.get(layer, 0)
        if n and int(pos[0]) < int(self._pb[layer][n - 1]):
            raise ValueError("positions must be appended non-decreasing (buffer eviction assumes sorted rows)")
        if n + T > (self._kb[layer].shape[0] if layer in self._kb else 0):
            self._kb[layer] = self._grow(self._kb.get(layer), n + T, k.shape[1:], self.dtype)
            self._vb[layer] = self._grow(self._vb.get(layer), n + T, v.shape[1:], self.dtype)
            self._pb[layer] = self._grow(self._pb.get(layer), n + T, (), np.int64)
        self._kb[layer][n:n + T] = k
        self._vb[layer][n:n + T] = v
        self._pb[layer][n:n + T] = pos
        n += T
        if self._layer_kind(layer) == SWA:  # evict entries no query can ever see again
            w = self.cfg.sliding_window
            # an entry is dead only when it is out of window for EVERY query in this batch
            # AND for all future queries (which start at pos.max()+1). Keep from
            # min(batch_pos) - w + 1: evicting earlier breaks full-forward prefill rows.
            keep_from = int(pos.min()) - w + 1
            drop_keep = int(np.searchsorted(self._pb[layer][:n], keep_from))
            drop_trim = max(0, n - (w + T))  # keep the last w+T rows, as before
            drop = min(n, max(drop_keep, drop_trim))
            if drop:
                self._kb[layer][:n - drop] = self._kb[layer][drop:n]
                self._vb[layer][:n - drop] = self._vb[layer][drop:n]
                self._pb[layer][:n - drop] = self._pb[layer][drop:n]
                n -= drop
        self._n_rows[layer] = n

    def bump(self, n: int = 1) -> None:
        self._n += n

    def get(self, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(k, v, positions) currently visible for layer (views into the buffers)."""
        n = self._n_rows[layer]
        return self._kb[layer][:n], self._vb[layer][:n], self._pb[layer][:n]

    def bytes_used(self) -> int:
        return sum(self._kb[l][: self._n_rows[l]].nbytes + self._vb[l][: self._n_rows[l]].nbytes
                   for l in self._kb)

    def swa_entries(self, layer: int) -> int:
        return self._n_rows[layer]


@dataclass
class PoolAccountant:
    """Token-aware admission over a byte budget (ds41rt v10 RAM-KV admission semantics, introduced in v6; CPU).

    Cost of a request = GA KV growth for its tokens + one SWA ring per sequence.
    Drafter KV is a fixed allowance the planner adds separately.
    """

    cfg: "MiMoConfig"
    capacity_bytes: int
    kv_dtype_bytes: int = 1
    resident_tokens: int = 0
    resident_seqs: int = 0
    rejections: int = 0

    def request_cost(self, n_tokens: int, n_seqs: int = 1) -> int:
        ga = self.cfg.ga_kv_bytes_per_token(self.kv_dtype_bytes) * n_tokens
        swa = self.cfg.swa_ring_bytes_per_seq(self.kv_dtype_bytes) * n_seqs
        return ga + swa

    @property
    def used_bytes(self) -> int:
        return self.request_cost(self.resident_tokens, self.resident_seqs)

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    def admit(self, n_tokens: int, n_seqs: int = 1) -> bool:
        cost = self.request_cost(n_tokens, n_seqs)
        if self.used_bytes + cost > self.capacity_bytes:
            self.rejections += 1
            return False
        self.resident_tokens += n_tokens
        self.resident_seqs += n_seqs
        return True

    def release(self, n_tokens: int, n_seqs: int = 1) -> None:
        self.resident_tokens = max(0, self.resident_tokens - n_tokens)
        self.resident_seqs = max(0, self.resident_seqs - n_seqs)

    def max_tokens_for_bytes(self, n_seqs: int = 1) -> int:
        ga = self.cfg.ga_kv_bytes_per_token(self.kv_dtype_bytes)
        swa = self.cfg.swa_ring_bytes_per_seq(self.kv_dtype_bytes) * n_seqs
        return max(0, (self.capacity_bytes - swa) // ga)
