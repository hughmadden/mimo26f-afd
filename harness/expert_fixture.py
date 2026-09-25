"""harness/expert_fixture.py — real-block MXFP4 nibble-order fixture (I4 item 1/3).

ADVISOR-I4 §3.2 step 1: the nibble-order proof must run on **real checkpoint
expert blocks**, not synthetic ones — the kernel's unpack path, dumped to f32 on
a Spark, must equal `spike/mxfp4.unpack` (non-naive: even element in the LOW
nibble) **bitwise**, and the E8M0 scale mapping (T10) must be pinned on real
blocks too.

This tool reads the local checkpoint copy READ-ONLY, computes the reference f32 via
`spike.mxfp4.unpack`, and writes a compact fixture: per block the raw-byte
sha256, a deterministic sample of expected f32 **bit patterns** (hex), the
nibble/scale bytes at those positions, and scale-byte statistics (including
whether the reserved 255 clamp appears in real data).  The Spark-side proof
re-reads the same bytes, runs the kernel unpack, and compares bit patterns at
the sampled positions — no 33 MB arrays in git.

Deterministic: fixed LCG sampling, no RNG, no clocks — two runs are byte-equal
(pinned in harness/selftests/test_expert_fixture.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

LOCAL_WEIGHTS = os.path.expanduser("~/models/XiaomiMiMo/MiMo-V2.6-Flash-RL")
BLOCK = 32
SAMPLE_N = 2048

# Real blocks: 3 layers x 3 experts x 3 projections (layer 0 is dense).
LAYERS = (1, 24, 46)
EXPERTS = (0, 7, 255)
PROJS = ("gate_proj", "up_proj", "down_proj")


def sample_positions(out: int, inn: int, n: int) -> list[tuple[int, int]]:
    """Deterministic (row, col) sample: LCG over the flattened index space."""
    total = out * inn
    n = min(n, total)
    pos, seen, x = [], set(), 0x2545F4914F6CDD1D
    while len(pos) < n:
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        i = (x >> 16) % total
        if i in seen:
            continue
        seen.add(i)
        pos.append((i // inn, i % inn))
    return pos


def block_fixture(name: str, w, s, n: int = SAMPLE_N) -> dict:
    """Fixture for one block: raw sha256 + sampled expected f32 bit patterns."""
    from spike import mxfp4

    out, half = w.shape
    inn = half * 2
    ref = mxfp4.unpack(w, s, naive=False)          # f32 [out, inn], the reference
    pos = sample_positions(out, inn, n)
    bits = [struct.unpack("<I", struct.pack("<f", float(ref[r, c])))[0] for r, c in pos]
    sb = s.reshape(-1)
    return dict(
        name=name,
        shape=[out, inn],
        weight_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
        scale_sha256=hashlib.sha256(s.tobytes()).hexdigest(),
        sample_n=len(pos),
        positions=[[r, c] for r, c in pos],
        expected_f32_bits=[f"{b:08x}" for b in bits],
        scale_byte_min=int(sb.min()),
        scale_byte_max=int(sb.max()),
        scale_byte_255_count=int((sb == 255).sum()),
        saturated_count=int((abs(ref) >= 3.4028234663852886e38).sum()),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/fixtures/expert_nibble_fixture.json")
    ap.add_argument("--sample", type=int, default=SAMPLE_N)
    ap.add_argument("--weights", default=os.environ.get("MIMO26_WEIGHTS_DIR", LOCAL_WEIGHTS))
    args = ap.parse_args(argv)

    os.environ.setdefault("MIMO26_WEIGHTS_DIR", args.weights)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from spike.real_loop import Reader

    r = Reader()
    blocks = []
    for layer in LAYERS:
        for expert in EXPERTS:
            for proj in PROJS:
                base = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
                de, sh, w = r.get(f"{base}.weight")
                sde, ssh, s = r.get(f"{base}.weight_scale")
                if de != "U8" or sde != "U8":
                    raise SystemExit(f"{base}: expected U8 weight+scale, got {de}/{sde}")
                fx = block_fixture(base, w.reshape(sh), s.reshape(ssh), args.sample)
                blocks.append(fx)
                print(f"[fixture] {base} {fx['shape']} scale_max={fx['scale_byte_max']} "
                      f"clamp255={fx['scale_byte_255_count']} sat={fx['saturated_count']}")

    doc = dict(
        source="local checkpoint copy (READ-ONLY)",
        reference="spike/mxfp4.unpack naive=False (even element in the LOW nibble; "
                  "scale byte 255 clamps to 2^127)",
        block=BLOCK,
        layers=list(LAYERS),
        experts=list(EXPERTS),
        projections=list(PROJS),
        blocks=blocks,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    man = args.out + ".sha256"
    with open(man, "w", encoding="utf-8") as f:
        f.write(f"{hashlib.sha256(open(args.out, 'rb').read()).hexdigest()}  "
                f"{os.path.basename(args.out)}\n")
    print(f"[fixture] wrote {args.out} ({len(blocks)} blocks) + {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
