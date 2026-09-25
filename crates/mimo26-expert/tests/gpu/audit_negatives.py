"""Host-only standalone reader regressions; no CUDA context or model execution."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

driver, weights, fixture_path, slot = sys.argv[1:]
fixture = json.loads(Path(fixture_path).read_text())
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MIMO26_BUILDER_GPU="0")
for name, expected in [
    ("weight-hash", "SHA256 mismatch"),
    ("scale-hash", "SHA256 mismatch"),
    ("shape", "bad fixture geometry"),
    ("position", "sample OOB"),
    ("duplicate", "bad/duplicate block name"),
    ("count", "expected2048 samples per block"),
]:
    bad = copy.deepcopy(fixture)
    block = bad["blocks"][0]
    if name == "weight-hash":
        block["weight_sha256"] = "0" * 64
    elif name == "scale-hash":
        block["scale_sha256"] = "0" * 64
    elif name == "shape":
        block["shape"][0] += 1
    elif name == "position":
        block["positions"][0][1] = block["shape"][1]
    elif name == "duplicate":
        bad["blocks"][1] = copy.deepcopy(block)
    elif name == "count":
        block["expected_f32_bits"].pop()
    path = Path(slot) / f"bad-{name}.json"
    path.write_text(json.dumps(bad))
    proc = subprocess.run([driver, "audit", weights, str(path)], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          timeout=120)
    (Path(slot) / f"bad-{name}.log").write_text(proc.stdout)
    assert proc.returncode == 2 and expected in proc.stdout, (name, proc.returncode, proc.stdout)
    print(f"HOST NEGATIVE PASS {name}: typed reader refusal")
