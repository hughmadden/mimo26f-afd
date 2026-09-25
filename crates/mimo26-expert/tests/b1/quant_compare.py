"""Byte-exact CPU proof for the independently frozen CUDA/shared codec.

Imports the BUILDER reference only after the source freeze recorded in
kernels/b1/quant-v1-independence.md. No torch/GPU code is called. CUDA execution
still needs its own target-SM receipt; this checks the shared integer scalar code.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
REFERENCE = ROOT / "oracle/lattice/quant_v1.py"
spec = importlib.util.spec_from_file_location("builder_quant_v1", REFERENCE)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def corpus():
    blocks = []
    grid = ref.e4m3fn_decode(np.arange(127, dtype=np.uint8))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for k in range(-22, 121):
            centers = np.ldexp(grid, k).astype(np.float32)
            mid = np.ldexp((grid[:-1] + grid[1:]) / 2, k).astype(np.float32)
            probes = np.concatenate([centers, -centers, mid, -mid,
                np.nextafter(mid, np.float32(-np.inf)), np.nextafter(mid, np.float32(np.inf)),
                np.nextafter(-mid, np.float32(-np.inf)), np.nextafter(-mid, np.float32(np.inf))])
            anchor = np.float32(np.ldexp(240.0 if k == 120 else 448.0, k))
            for start in range(0, len(probes), 31):
                b = np.zeros(32, dtype=np.float32)
                part = probes[start:start + 31]
                b[:len(part)] = part; b[31] = anchor
                blocks.append(b)
        # Explicit amax transitions and their adjacent FP32 values.
        for k in range(-22, 120):
            threshold = np.float32(np.ldexp(448.0, k))
            for sign in (1.0, -1.0):
                for v in (np.nextafter(threshold, np.float32(0)), threshold,
                          np.nextafter(threshold, np.float32(np.inf))):
                    b = np.zeros(32, dtype=np.float32); b[0] = sign * v; blocks.append(b)
    for bits in (0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                 0x38d1b716, 0x38d1b717, 0x38d1b718, 0x7f77ffff, 0x7f780000,
                 0x7f780001, 0x7f7fffff, 0xff77ffff, 0xff780000, 0xff7fffff):
        blocks.append(np.full(32, bits, dtype=np.uint32).view(np.float32))
    for bits in (0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0xffc01234):
        for position in range(32):
            b = np.full(32, .5, dtype=np.float32)
            b.view(np.uint32)[position] = bits
            blocks.append(b)
    rng = np.random.default_rng(0x26B1)
    blocks.extend(rng.integers(0, 2**32, (4096, 32), dtype=np.uint32).view(np.float32))
    with np.errstate(under="ignore"):
        blocks.extend(np.ldexp(rng.uniform(-1, 1, (4096, 32)), rng.integers(-149, 128, (4096, 1))).astype(np.float32))
    return np.ascontiguousarray(blocks, dtype="<f4")


def expected(h):
    payload = np.full(h.shape, 0xa5, dtype=np.uint8)
    scales = np.full(len(h), 0xa5, dtype=np.uint8)
    faults = np.zeros(len(h), dtype="<u4")
    finite = np.isfinite(h).all(axis=1)
    faults[~finite] = 1
    index = np.flatnonzero(finite)
    q, s = ref.encode_blocks(h[finite])
    s = s.reshape(-1)
    decoded_wide = np.ldexp(ref.e4m3fn_decode(q), (s.astype(np.int64) - 127)[:, None])
    overflow = np.any(np.abs(decoded_wide) > np.finfo(np.float32).max, axis=1)
    faults[index[overflow]] = 2
    valid = index[~overflow]
    payload[valid] = q[~overflow]; scales[valid] = s[~overflow]
    assert np.isfinite(ref.decode_blocks(q[~overflow], s[~overflow, None])).all()
    # Confirm the actual reference raises the specified fault, not just a
    # duplicate implementation of its checks in the classifier above.
    for b in np.flatnonzero(faults):
        try:
            ref.decode_blocks(*ref.encode_blocks(h[b:b + 1]))
        except ref.NumericalFault:
            pass
        else:
            raise AssertionError(f"reference did not fault for block {b}")
    return payload, scales, faults


def invoke(executable, path, prefix, mode):
    proc = subprocess.run([str(executable), str(path), str(prefix), mode],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = proc.stdout.decode(errors="replace")
    prefix.with_suffix(".log").write_text(text)
    print(text, end="", flush=True)
    assert proc.returncode == 0, f"adapter failed: {prefix}, exit {proc.returncode}"
    return (np.fromfile(str(prefix) + ".payload", dtype=np.uint8).reshape(-1, 32),
            np.fromfile(str(prefix) + ".scales", dtype=np.uint8),
            np.fromfile(str(prefix) + ".faults", dtype="<u4"))


def main(executable, stage):
    reference_hash = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    h = corpus()
    data = stage / "identical-inputs.f32"; h.tofile(data)
    q, s, f = expected(h)
    got = invoke(executable, data, stage / "correct", "correct")
    for name, actual, want in zip(("payload", "scales", "faults"), got, (q, s, f)):
        bad = np.argwhere(actual != want)
        assert not len(bad), f"{name}: {len(bad)} mismatches, first {bad[:8].tolist()}"
    assert set(s[f == 0].tolist()) == set(range(105, 248)), "missing scale byte coverage"
    assert {int(v) for v in q[f == 0].reshape(-1)} == set(range(256)) - {127, 255}, "missing finite payload code coverage"
    naive = {}
    for mode in ("trunc", "nearest_scale", "no_floor", "bf16_preround", "e4m3_subnormal_flush"):
        candidate = invoke(executable, data, stage / mode, mode)
        mismatch = sum(int(np.count_nonzero(a != b)) for a, b in zip(candidate, (q, s, f)))
        assert mismatch, f"powerless {mode} negative"
        naive[mode] = mismatch
    # Wrong K16 partition, executed using the actual candidate on duplicated
    # half-blocks; two half-scales cannot masquerade as the correct K32 scale.
    valid = h[f == 0]
    halves = valid.reshape(-1, 2, 16)
    k16_input = np.ascontiguousarray(np.concatenate((halves, halves), axis=-1).reshape(-1, 32))
    k16_path = stage / "wrong-k16-inputs.f32"; k16_input.tofile(k16_path)
    kq, ks, kf = invoke(executable, k16_path, stage / "wrong-k16", "correct")
    kq = kq.reshape(-1, 2, 32)[:, :, :16].reshape(-1, 32)
    mismatch = int(np.count_nonzero(kq != q[f == 0])) + int(np.count_nonzero(ks.reshape(-1, 2) != s[f == 0, None])) + int(np.count_nonzero(kf))
    assert mismatch, "powerless K16 negative"
    naive["k16"] = mismatch
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == reference_hash, "reference changed during test"
    receipt = dict(status="PASS", mode="E-W4A8-v1", codec=ref.VERSION, scope="CPU/shared scalar; CUDA compile only",
                   blocks=len(h), elements=h.size, valid_blocks=int(np.count_nonzero(f == 0)),
                   nonfinite_faults=int(np.count_nonzero(f == 1)), reconstruction_faults=int(np.count_nonzero(f == 2)),
                   scale_bytes_covered=143, finite_payload_codes_covered=254,
                   naive_mismatches=naive, reference_sha256=reference_hash,
                   input_sha256=hashlib.sha256(data.read_bytes()).hexdigest())
    (stage / "quant-reference.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("B1 QUANT REFERENCE PASS", json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
