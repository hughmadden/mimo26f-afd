"""Read real raw sample bytes for Rust tests; never reconstruct bytes from answers.

Harness only (not a serving runtime). Uses stdlib, checks both tensor SHA256s,
then emits (packed byte, scale byte, next scale byte) per fixture position.
"""
import hashlib
import json
from pathlib import Path
import struct
import sys

fixture = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]


def tensor(name, shape, expected_sha):
    with (root / index[name]).open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 64 * 1024 * 1024:
            raise ValueError("oversized safetensors header")
        header = json.loads(f.read(n))
        desc = header[name]
        assert desc["dtype"] == "U8" and desc["shape"] == shape, name
        lo, hi = desc["data_offsets"]
        assert 0 <= lo <= hi and hi - lo == shape[0] * shape[1], name
        f.seek(8 + n + lo)
        raw = f.read(hi - lo)
        assert len(raw) == hi - lo, name
        assert hashlib.sha256(raw).hexdigest() == expected_sha, name + " sha mismatch"
        return raw


out = bytearray()
for block in fixture["blocks"]:
    rows, cols = block["shape"]
    w = tensor(block["name"] + ".weight", [rows, cols // 2], block["weight_sha256"])
    s = tensor(block["name"] + ".weight_scale", [rows, cols // 32], block["scale_sha256"])
    for r, c in block["positions"]:
        assert 0 <= r < rows and 0 <= c < cols
        k = c // 32
        out.extend((w[r * (cols // 2) + c // 2], s[r * (cols // 32) + k],
                    s[r * (cols // 32) + k + 1] if k + 1 < cols // 32 else 0))
sys.stdout.buffer.write(out)
