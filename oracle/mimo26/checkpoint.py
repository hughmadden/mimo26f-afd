"""Checkpoint plumbing: safetensors I/O, shard index, repack planning, synthetic writer.

The real MiMo-V2.6-Flash-RL checkpoint is 64 expert-parallel shards
(`model_pp0_ep{0..63}_shard0.safetensors`); ep0 carries all non-expert weights plus its
4-experts-per-MoE-layer share, ep1..63 carry 4 experts x 47 MoE layers each. This module
reads/writes that format (tiny synthetic versions for tests) and computes how experts and
column slices distribute onto N Spark ranks.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_DTYPE_TO_NAME = {"u8": "U8", "f32": "F32", "f8": "F8_E4M3"}
_NAME_TO_DTYPE = {v: k for k, v in _DTYPE_TO_NAME.items()}
_NAME_TO_NP = {"U8": np.uint8, "F32": np.float32, "F8_E4M3": np.uint8}


# ---------------------------------------------------------------------------
# safetensors container (subset: U8 / F32 / F8_E4M3)
# ---------------------------------------------------------------------------

def write_safetensors(path: str | Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict = {}
    offset = 0
    blobs = []
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.uint8:
            # U8 for packed MXFP4 (E2M1 nibbles) and scale grids; F8_E4M3 only for raw
            # fp8 codes (real headers: expert .weight/.weight_scale are both U8).
            dt = ("U8" if ("scale" in name or ".experts." in name or not name.endswith("weight"))
                  else "F8_E4M3")
        elif arr.dtype == np.float32:
            dt = "F32"
        else:
            raise TypeError(f"unsupported dtype {arr.dtype} for {name}; use write_safetensors_typed")
        nbytes = arr.nbytes
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + nbytes]}
        blobs.append(arr.tobytes())
        offset += nbytes
    hdr = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for b in blobs:
            f.write(b)


def write_safetensors_typed(path: str | Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """Explicit-typed variant: name -> (dtype in {U8, F32, F8_E4M3}, array)."""
    header: dict = {}
    offset = 0
    blobs = []
    for name, (dt, arr) in tensors.items():
        arr = np.ascontiguousarray(arr)
        expected = _NAME_TO_NP[dt]
        if arr.dtype != expected:
            raise ValueError(f"{name}: dtype {arr.dtype} does not match declared {dt}")
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + arr.nbytes]}
        blobs.append(arr.tobytes())
        offset += arr.nbytes
    hdr = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for b in blobs:
            f.write(b)


def read_safetensors_header(path: str | Path) -> tuple[dict, int]:
    """Return (tensor metadata dict, header byte length)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    return hdr, n


def read_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    hdr, n = read_safetensors_header(path)
    out: dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        base = f.read()
    data = base[8 + n:]
    for name, meta in hdr.items():
        lo, hi = meta["data_offsets"]
        arr = np.frombuffer(data[lo:hi], dtype=_NAME_TO_NP[meta["dtype"]])
        out[name] = arr.reshape(meta["shape"]).copy()
    return out


# ---------------------------------------------------------------------------
# shard index
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShardIndex:
    """HF `model.safetensors.index.json`: `weight_map` is tensor name -> filename
    (verified against the real index: 73,081 tensors across 65 files, 22 Sep 2026)."""

    tensor_to_file: dict[str, str]

    @classmethod
    def from_index_json(cls, path: str | Path) -> "ShardIndex":
        d = json.loads(Path(path).read_text())
        wm = {name: fn for name, fn in d["weight_map"].items()}
        return cls(tensor_to_file=wm)

    def files(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.tensor_to_file.values())))

    def tensors(self) -> tuple[str, ...]:
        return tuple(sorted(self.tensor_to_file))

    def names_in(self, file: str) -> tuple[str, ...]:
        return tuple(sorted(n for n, f in self.tensor_to_file.items() if f == file))

    @property
    def weight_map(self) -> dict[str, str]:
        return dict(self.tensor_to_file)


# ---------------------------------------------------------------------------
# expert repack planning (N Spark ranks)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankSlice:
    rank: int
    experts: tuple[tuple[int, int], ...]  # (layer, expert_id) for EP distribution
    col_start: int = 0  # TP-over-intermediate column slice (logical moe_inter coords)
    col_end: int = 0  # exclusive, includes padding
    pad_cols: int = 0


@dataclass(frozen=True)
class RepackPlan:
    mode: str  # "ep" | "tp"
    n_ranks: int
    ranks: tuple[RankSlice, ...]
    per_rank_experts: tuple[int, ...]
    pad_cols_total: int
    waste_fraction: float  # padded columns / logical columns (tp) or 0.0 (ep)

    def experts_on(self, rank: int) -> tuple[tuple[int, int], ...]:
        return self.ranks[rank].experts


def plan_experts(cfg, n_ranks: int, mode: str = "ep", align: int = 32) -> RepackPlan:
    """Distribute routed experts onto N Spark ranks.

    ep: every (moe layer, expert id) lands on exactly one rank (uneven allowed, ±1).
    tp: every rank holds all experts but only a column slice of `moe_intermediate_size`,
        padded up to `align` so 3/5/6-way splits stay tensor-core friendly.
    """
    if n_ranks < 1:
        raise ValueError("n_ranks must be positive")
    pairs = [(lid, eid) for lid in cfg.moe_layer_ids for eid in range(cfg.n_routed_experts)]
    ranks: list[RankSlice] = []
    if mode == "ep":
        chunks = [pairs[i::n_ranks] for i in range(n_ranks)]
        for r, ch in enumerate(chunks):
            ranks.append(RankSlice(rank=r, experts=tuple(ch)))
        per = tuple(len(c) for c in chunks)
        return RepackPlan("ep", n_ranks, tuple(ranks), per, 0, 0.0)
    if mode == "tp":
        inter = cfg.moe_intermediate_size
        base = math.ceil(inter / n_ranks / align) * align
        total = base * n_ranks
        for r in range(n_ranks):
            lo = r * base
            ranks.append(RankSlice(rank=r, experts=tuple(pairs), col_start=lo,
                                   col_end=lo + base, pad_cols=max(0, lo + base - inter)))
        waste = (total - inter) / inter
        return RepackPlan("tp", n_ranks, tuple(ranks), (len(pairs),) * n_ranks,
                          total - inter, waste)
    raise ValueError(f"unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# synthetic checkpoint (dummy data in the real on-disk format)
# ---------------------------------------------------------------------------

@dataclass
class SyntheticCheckpoint:
    """Deterministic dummy checkpoint in the real 64-EP layout (scaled via `n_ep_shards`)."""

    cfg: "MiMoConfig"
    n_ep_shards: int = 4
    seed: int = 0
    root: Path = field(default_factory=lambda: Path("synthetic"))
    tensors: dict = field(default_factory=dict)

    @property
    def experts_per_shard(self) -> int:
        q, r = divmod(self.cfg.n_routed_experts, self.n_ep_shards)
        if r != 0:
            raise ValueError("n_routed_experts must be divisible by n_ep_shards")
        return q

    def expert_owner(self, layer: int, expert: int) -> int:
        return expert // self.experts_per_shard

    def build(self) -> "SyntheticCheckpoint":
        rng = np.random.default_rng(self.seed)
        cfg = self.cfg
        epsz = cfg.moe_intermediate_size
        hid = cfg.hidden_size
        block = cfg.mxfp4_block_size
        self.tensors = {i: {} for i in range(self.n_ep_shards)}
        for lid in cfg.moe_layer_ids:
            for eid in range(cfg.n_routed_experts):
                shard = self.expert_owner(lid, eid)
                w = rng.standard_normal((3, epsz, hid)).astype(np.float32) * 0.02
                from .quant import mxfp4
                for name, mat in (("gate_proj", w[0]), ("up_proj", w[1])):
                    packed, scales = mxfp4.pack(mat, block)
                    self.tensors[shard][f"model.layers.{lid}.mlp.experts.{eid}.{name}.weight"] = packed
                    self.tensors[shard][f"model.layers.{lid}.mlp.experts.{eid}.{name}.weight_scale"] = scales
                packed, scales = mxfp4.pack(w[2].T.copy(), block)  # down: [hidden, moe_inter]
                self.tensors[shard][f"model.layers.{lid}.mlp.experts.{eid}.down_proj.weight"] = packed
                self.tensors[shard][f"model.layers.{lid}.mlp.experts.{eid}.down_proj.weight_scale"] = scales
        return self

    def write(self, root: str | Path) -> dict[str, Path]:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for shard, tensors in self.tensors.items():
            p = root / f"model_pp0_ep{shard}_shard0.safetensors"
            typed = {}
            for name, arr in tensors.items():
                dt = ("U8" if (".experts." in name or "scale" in name or not name.endswith(".weight"))
                      else "F8_E4M3")
                typed[name] = (dt, arr)
            write_safetensors_typed(p, typed)
            paths[p.name] = p
        # index in the real shape: {tensor name: file name}
        weight_map = {}
        for shard, tensors in self.tensors.items():
            fn = f"model_pp0_ep{shard}_shard0.safetensors"
            for name in sorted(tensors):
                weight_map[name] = fn
        idx = {"weight_map": dict(sorted(weight_map.items())),
               "metadata": {"total_size": sum(a.nbytes for t in self.tensors.values() for a in t.values())}}
        (root / "model.safetensors.index.json").write_text(json.dumps(idx))
        return paths
