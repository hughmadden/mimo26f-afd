"""Independent rank-0, all-256-expert v1 benchmark reference; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
from lattice_oracle import REPO, Q, expert, tensor, unpack, np


def generate(root, out):
    out.mkdir(parents=True, exist_ok=False)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    pins = {b["name"]: b for b in json.loads((REPO / "bench/fixtures/expert_nibble_fixture.json").read_text())["blocks"]}
    ids = [0, 7, 255] + [e for e in range(1, 255) if e != 7]
    assert len(ids) == len(set(ids)) == 256
    xp, xs = Q.encode_blocks(np.random.default_rng(0x26AFD).uniform(-.5, .5, (8, 4096)).astype(np.float32))
    partial = np.empty((256, 8, 4096), "<f8")
    blocks = []
    for slot, eid in enumerate(ids):
        matrices = []
        for proj in ("gate", "up", "down"):
            name = f"model.layers.1.mlp.experts.{eid}.{proj}_proj"
            n, k = (4096, 2048) if proj == "down" else (2048, 4096)
            w, wh = tensor(root, index, name + ".weight", [n, k//2])
            s, sh = tensor(root, index, name + ".weight_scale", [n, k//32])
            if eid in (0, 7, 255):
                assert (wh, sh) == (pins[name]["weight_sha256"], pins[name]["scale_sha256"])
            blocks.append(dict(name=name, weight_sha256=wh, scale_sha256=sh))
            matrices.append(unpack(w[:, :256], s[:, :16], naive=False).astype(np.float64) if proj == "down"
                            else unpack(w[:512], s[:512], naive=False).astype(np.float64))
        partial[slot] = expert(xp, xs, *matrices)["full"]
        if slot % 16 == 0: print(f"BENCH ORACLE completed {slot+1}/256 real rank-0 experts", flush=True)
    artifacts = {}
    for name, data in (("b1-x-payload.u8", xp), ("b1-x-scales.u8", xs), ("b1-bench-partial.f64", partial)):
        data.tofile(out / name)
        artifacts[name] = dict(bytes=data.nbytes, sha256=hashlib.sha256((out/name).read_bytes()).hexdigest())
    manifest = dict(lattice="E-W4A8-v1", quantizer=Q.VERSION, experts=ids, blocks=blocks, artifacts=artifacts,
                    scope="CPU rank-0 reference, 256 experts, M8 prefixes; no GPU/timing evidence")
    (out/"b1-bench-source.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("BENCH ORACLE PASS: 256 distinct experts, 1536 source hashes, rank0, 8388608 FP64 coordinates; CPU only", flush=True)


def selftest():
    rng = np.random.default_rng(17)
    xp, xs = Q.encode_blocks(rng.uniform(-.25, .25, (2, 64)).astype(np.float32))
    gate = rng.integers(-4, 5, (512, 64)).astype(np.float64) / 32
    up = rng.integers(-4, 5, (512, 64)).astype(np.float64) / 32
    down = rng.integers(-4, 5, (64, 512)).astype(np.float64) / 32
    full = expert(xp, xs, gate, up, down)
    rank0 = expert(xp, xs, gate[:128], up[:128], down[:, :128])
    np.testing.assert_array_equal(full["partial"][0], rank0["full"])
    wrong = expert(xp, xs, gate[128:256], up[128:256], down[:, 128:256])
    assert np.max(np.abs(wrong["full"] - rank0["full"])) > 1e-5
    print("BENCH ORACLE SELFTEST PASS: independently sliced rank0 equals full expert rank0; wrong rank detected; CPU only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--selftest", action="store_true")
    parser.add_argument("weights", type=Path, nargs="?"); parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args(); selftest()
    if not args.selftest:
        if args.weights is None or args.output is None: parser.error("weights and output required")
        generate(args.weights, args.output)
