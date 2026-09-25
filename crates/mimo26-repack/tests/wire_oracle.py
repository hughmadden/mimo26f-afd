"""Independent E-FP32 eight-route oracle, harness only; no CUDA/Rust imports."""
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

from checkpoint_reader import tensor

if sys.argv[1:] == ["--selftest"]:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="mimo26f-wire-reader-") as directory:
        root = Path(directory)
        meta = dict(dtype="U8", shape=[2, 2], data_offsets=[0, 4])
        def put(value):
            header = json.dumps(dict(test=value)).encode()
            (root / "test.bin").write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x01\x02\x03")
        put(meta)
        value, digest = tensor(root, dict(test="test.bin"), "test", [2, 2])
        assert value.tolist() == [[0, 1], [2, 3]] and digest == hashlib.sha256(bytes(range(4))).hexdigest()
        for field, bad in [("dtype", "F32"), ("shape", [1, 4]), ("data_offsets", [0, 5]), ("data_offsets", [-1, 3])]:
            put({**meta, field: bad})
            try:
                tensor(root, dict(test="test.bin"), "test", [2, 2])
            except AssertionError:
                pass
            else:
                raise AssertionError("reader negative accepted")
        put(meta)
        try:
            tensor(root, dict(test="../test.bin"), "test", [2, 2])
        except AssertionError:
            pass
        else:
            raise AssertionError("path traversal accepted")
    p = np.arange(4 * 2 * 8 * 3, dtype=np.float64).reshape(4, 2, 8, 3)
    w = np.arange(1, 9, dtype=np.float64) / 64
    expected = np.array([[sum(p[r, t, j, h] * w[j] for r in range(4) for j in range(8)) for h in range(3)] for t in range(2)])
    assert np.array_equal((p.sum(axis=0) * w[None, :, None]).sum(axis=1), expected)
    assert not np.array_equal((p.sum(axis=0) * (w*w)[None, :, None]).sum(axis=1), expected)
    print("WIRE ORACLE SELFTEST PASS: bounded reader, five refusals, weighted rank/token/route axes and double-weight detector")
    sys.exit(0)

if len(sys.argv) == 3 and sys.argv[1] == "--synthetic":
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=False)
    r = np.arange(4)[:, None, None, None]
    t = np.arange(8)[None, :, None, None]
    j = np.arange(8)[None, None, :, None]
    h = np.arange(4096)[None, None, None, :]
    p = ((r * 32 + t * 8 + j + h % 5) / 4096).astype("<f8")
    w = np.asarray([(i + 1) / 36 for i in range(8)], dtype="<f4")
    p.tofile(out / "wire-partial.f64")
    (p.sum(axis=0) * w.astype(np.float64)[None, :, None]).sum(axis=1).astype("<f8").tofile(out / "wire-full.f64")
    w.tofile(out / "wire-weights.f32")
    for m in (1, 2, 4, 8):
        p[:, :m].astype("<f4").tofile(out / f"wire-M{m}.f32")
    bad = p[:, :1].astype("<f4").copy(); bad[0, 0, 0, 0] = 1.0
    bad.tofile(out / "wire-naive-M1.f32")
    print("WIRE SYNTHETIC SELFTEST INPUTS: asymmetric ranks/tokens/routes/channels; one corrupted scalar; no GPU/model claim")
    sys.exit(0)

root, out = map(Path, sys.argv[1:3])
out.mkdir(parents=True, exist_ok=False)
index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
fixture = json.loads((repo / "bench/fixtures/expert_nibble_fixture.json").read_text())
pinned = {b["name"]: b for b in fixture["blocks"]}
experts = [0, 7, 255, 1, 2, 3, 4, 5]
x = np.random.default_rng(0x26AFD).uniform(-0.5, 0.5, (8, 4096)).astype("<f4")
weights = np.asarray([(j + 1) / 36 for j in range(8)], dtype="<f4")
x.tofile(out / "wire-x.f32")
weights.tofile(out / "wire-weights.f32")
partial = np.empty((4, 8, 8, 4096), dtype=np.float64)  # rank, token, route, hidden
full = np.empty((8, 8, 4096), dtype=np.float64)
blocks = []

for route, expert in enumerate(experts):
    matrices = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        name = f"model.layers.1.mlp.experts.{expert}.{projection}"
        n, k = (4096, 2048) if projection == "down_proj" else (2048, 4096)
        w, wh = tensor(root, index, name + ".weight", [n, k // 2])
        s, sh = tensor(root, index, name + ".weight_scale", [n, k // 32])
        if expert in (0, 7, 255):
            assert name in pinned
            assert wh == pinned[name]["weight_sha256"] and sh == pinned[name]["scale_sha256"]
        blocks.append(dict(name=name, weight_sha256=wh, scale_sha256=sh))
        matrices.append(unpack(w, s, naive=False).astype(np.float64))
    gate = x.astype(np.float64) @ matrices[0].T
    up = x.astype(np.float64) @ matrices[1].T
    hidden = gate / (1.0 + np.exp(-gate)) * up
    full[:, route] = hidden @ matrices[2].T
    for rank in range(4):
        sl = slice(rank * 512, (rank + 1) * 512)
        partial[rank, :, route] = hidden[:, sl] @ matrices[2][:, sl].T
    assert np.allclose(partial[:, :, route].sum(axis=0), full[:, route], atol=1e-11, rtol=1e-11)
    print(f"WIRE ORACLE expert={expert} ranks=4 tokens=8 FP64 partial/full identity PASS", flush=True)
assert np.isfinite(partial).all() and np.isfinite(full).all()
partial.astype("<f8").tofile(out / "wire-partial.f64")
(full * weights[None, :, None].astype(np.float64)).sum(axis=1).astype("<f8").tofile(out / "wire-full.f64")
(out / "wire-source.json").write_text(json.dumps(dict(layer=1, experts=experts, blocks=blocks), indent=2) + "\n")
print("WIRE ORACLE PASS: 8 distinct real experts, 48 source hashes, 18 pre-pinned hashes; independent full/TP4 FP64 references", flush=True)
