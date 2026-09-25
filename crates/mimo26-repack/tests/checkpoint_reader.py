"""Bounded U8 checkpoint reader shared by independent expert test oracles."""
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def tensor(root, index, name, shape):
    shard = index[name]
    assert Path(shard).name == shard and shard not in (".", "..")
    with (root / shard).open("rb") as f:
        size = f.seek(0, 2)
        f.seek(0)
        n = struct.unpack("<Q", f.read(8))[0]
        assert n <= 64 * 1024 * 1024 and n <= size - 8
        meta = json.loads(f.read(n))[name]
        assert meta["dtype"] == "U8" and meta["shape"] == shape
        lo, hi = meta["data_offsets"]
        assert 0 <= lo <= hi <= size - 8 - n and hi - lo == np.prod(shape)
        f.seek(8 + n + lo)
        data = f.read(hi - lo)
        assert len(data) == hi - lo
    return np.frombuffer(data, np.uint8).reshape(shape), hashlib.sha256(data).hexdigest()
