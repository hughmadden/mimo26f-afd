"""External full-expert FFN oracle for the layout-v2 identity test.

Harness only. Reads all 27 fixture tensors from checkpoint with both SHA256s
verified. Existing spike unpack is the independent decoder; this never imports
Rust code, a Rust-produced layout, or a Rust-produced reference answer.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo))
from spike.mxfp4 import unpack

weights, destination = map(Path, sys.argv[1:3])
fixture = json.loads((repo / "bench/fixtures/expert_nibble_fixture.json").read_text())
index = json.loads((weights / "model.safetensors.index.json").read_text())["weight_map"]
by_name = {b["name"]: b for b in fixture["blocks"]}
destination.mkdir(parents=True, exist_ok=False)
rng = np.random.default_rng(0x26AFD)
x = rng.uniform(-0.5, 0.5, (64, 4096)).astype("<f4")
x.tofile(destination / "x.f32")


def tensor(name, shape, digest):
    with (weights / index[name]).open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 64 * 1024 * 1024:
            raise ValueError("oversized safetensors header")
        header = json.loads(f.read(n))
        meta = header[name]
        assert meta["dtype"] == "U8" and meta["shape"] == shape, name
        lo, hi = meta["data_offsets"]
        assert 0 <= lo <= hi and hi-lo == shape[0]*shape[1], name
        f.seek(8+n+lo)
        raw = f.read(hi-lo)
        assert len(raw) == hi-lo and hashlib.sha256(raw).hexdigest() == digest, name
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


for layer in fixture["layers"]:
    for expert in fixture["experts"]:
        tag = f"L{layer:02}_E{expert:03}"
        projections = []
        for proj in ("gate_proj", "up_proj", "down_proj"):
            block = by_name[f"model.layers.{layer}.mlp.experts.{expert}.{proj}"]
            n, k = block["shape"]
            w = tensor(block["name"]+".weight", [n, k//2], block["weight_sha256"])
            s = tensor(block["name"]+".weight_scale", [n, k//32], block["scale_sha256"])
            w.tofile(destination / f"{tag}.{proj}.w")
            s.tofile(destination / f"{tag}.{proj}.s")
            projections.append(unpack(w, s, naive=False).astype(np.float64))
        # FP64 matmul and SiLU to avoid blessing one float32 accumulation order.
        g = x.astype(np.float64) @ projections[0].T
        u = x.astype(np.float64) @ projections[1].T
        h = (g / (1.0 + np.exp(-g))) * u
        y = h @ projections[2].T
        assert np.isfinite(y).all()
        y.astype("<f4").tofile(destination / f"{tag}.y.f32")
        # Additional GPU layerwise goldens come from FULL source matrices, never
        # a repacked slice or the CUDA/Rust implementation under test.
        g.astype("<f4").tofile(destination / f"{tag}.gate.f32")
        u.astype("<f4").tofile(destination / f"{tag}.up.f32")
        h32 = h.astype("<f4")
        h32.tofile(destination / f"{tag}.h.f32")
        for rank in range(4):
            columns = slice(rank * 512, (rank + 1) * 512)
            down = projections[2][:, columns]
            (h[:, columns] @ down.T).astype("<f4").tofile(
                destination / f"{tag}.partial_R{rank}.f32")
            # Isolated down GEMM receives f32 h, not FP64 h from the full FFN.
            (h32[:, columns].astype(np.float64) @ down.T).astype("<f4").tofile(
                destination / f"{tag}.down_R{rank}.f32")
        print(f"ORACLE {tag}: M=64 full FFN + layerwise/partial goldens, FP64 accumulation, six SHA256s verified", flush=True)
