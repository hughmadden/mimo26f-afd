"""Host-only staged TP4 reader negatives; originals are never modified."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

driver, root, fixture, slot = sys.argv[1:]
root, slot = Path(root).resolve(), Path(slot).resolve()
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MIMO26_BUILDER_GPU="0")
cases = [
    ("marker", "staged layout marker must be v2"),
    ("version", "manifest must be v2"),
    ("offset", "manifest layout offset/length mismatch"),
    ("duplicate", "duplicate/invalid rank"),
    ("descriptor", "wrong source tensor descriptor"),
    ("slice", "slice size/SHA mismatch"),
    ("rectangle", "slice source-coordinate rectangle mismatch"),
    ("nonfinite", "nonfinite oracle/input"),
    ("sum", "independent partial goldens do not sum"),
    ("short-x", "wrong float-file length"),
]
for case, message in cases:
    bad = slot / f"tp4-bad-{case}"
    bad.mkdir()
    for source in root.iterdir():
        if source.is_file():
            (bad / source.name).symlink_to(source)

    def replace(name, data):
        path = bad / name
        path.unlink()  # remove OUR symlink, never write through it to the original
        path.write_bytes(data)

    manifest_name = "L01_E000.manifest.json"
    manifest = json.loads((root / manifest_name).read_text())
    if case == "marker":
        replace("layout.version", b"1\n")
    elif case == "version":
        manifest["version"] = 1
    elif case == "offset":
        manifest["layout"][1]["off"] += 16
    elif case == "duplicate":
        manifest["slices"][1]["rank"] = 0
    elif case == "descriptor":
        manifest["slices"][0]["source"]["tensors"][0]["shape"][0] += 1
    elif case in ("slice", "rectangle"):
        name = manifest["slices"][0]["file"]
        content = bytearray((root / name).read_bytes())
        content[0] ^= 1
        replace(name, content)
        if case == "rectangle":
            # A valid new SHA must NOT excuse a semantically wrong rectangle.
            manifest["slices"][0]["sha256"] = hashlib.sha256(content).hexdigest()
    elif case in ("nonfinite", "sum"):
        name = "L01_E000.partial_R0.f32"
        content = bytearray((root / name).read_bytes())
        first = struct.unpack_from("<f", content)[0]
        struct.pack_into("<f", content, 0, float("nan") if case == "nonfinite" else first + 1.0)
        replace(name, content)
    elif case == "short-x":
        replace("x.f32", b"\x00" * 4)
    replace(manifest_name, json.dumps(manifest).encode())
    proc = subprocess.run([driver, "tp4-audit", str(bad), fixture], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          timeout=120)
    (slot / f"tp4-bad-{case}.log").write_text(proc.stdout)
    assert proc.returncode == 2 and message in proc.stdout, (case, proc.returncode, proc.stdout)
    print(f"TP4 HOST NEGATIVE PASS {case}: typed refusal, originals untouched")
