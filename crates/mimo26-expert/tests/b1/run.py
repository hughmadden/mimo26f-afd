"""B1 host/compile-only cell, invoked by the named Cargo tests via dev.sh.

Numerical mode: E-W4A8-v1. No torch, CUDA driver, GPU launch or timing.
The independent real-image check uses retained Rust output and pinned full-source
weight hashes. It does not regenerate an expected image with the B1 transform.
"""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
B1 = REPO / "crates/mimo26-expert/kernels/b1"
SIZE = 3342336
GEOM = dict(ep_ranks=4, expert_bytes=13369344, experts_per_layer=256,
            hidden=4096, intermediate=2048, moe_layers=47, quarter_slice_bytes=SIZE)
LAYOUT = [
    dict(proj="gate_proj", region="payload", off=0, len=1048576),
    dict(proj="gate_proj", region="scales", off=1048576, len=65536),
    dict(proj="up_proj", region="payload", off=1114112, len=1048576),
    dict(proj="up_proj", region="scales", off=2162688, len=65536),
    dict(proj="down_proj", region="payload", off=2228224, len=1048576),
    dict(proj="down_proj", region="scales", off=3276800, len=65536),
]


def command(argv, log):
    print("COMMAND", " ".join(map(str, argv)), flush=True)
    proc = subprocess.run(list(map(str, argv)), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    log.write_bytes(proc.stdout)
    print(proc.stdout.decode(errors="replace"), end="", flush=True)
    if proc.returncode:
        raise RuntimeError(f"exit {proc.returncode}; retained log {log}")


def validate_manifest(m, layer, expert):
    if m.get("version") != 2:
        raise ValueError("layout version refusal before image access")
    if (m.get("format") != "mimo26-repack-manifest" or m.get("geometry") != GEOM
            or m.get("layout") != LAYOUT or len(m.get("slices", [])) != 4):
        raise ValueError("manifest geometry/layout refusal")
    for rank, entry in enumerate(m["slices"]):
        expected_name = f"L{layer:02}_E{expert:03}_R{rank}.slice"
        digest = entry.get("sha256", "")
        if (entry.get("file") != expected_name or entry.get("rank") != rank
                or entry.get("layer") != layer or entry.get("expert") != expert
                or entry.get("bytes") != SIZE or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("manifest image identity refusal")


def must_refuse(m, layer, expert, reason):
    try:
        validate_manifest(m, layer, expert)
    except ValueError as error:
        if reason not in str(error):
            raise
    else:
        raise AssertionError(f"negative accepted: {reason}")


def prepare(executable, stage):
    root = Path(os.environ["MIMO26_B1_REAL_DIR"]).resolve(strict=True)
    fixture_path = REPO / "bench/fixtures/expert_nibble_fixture.json"
    fixture = json.loads(fixture_path.read_text())
    fixtures = {b["name"]: b for b in fixture["blocks"]}
    image_receipts = []
    command([executable, "synthetic"], stage / "synthetic.log")
    for layer in (1, 24, 46):
        for expert in (0, 7, 255):
            tag = f"L{layer:02}_E{expert:03}"
            manifest = json.loads((root / f"{tag}.manifest.json").read_text())
            validate_manifest(manifest, layer, expert)
            # Actual JSON v1 refusal, even though all region sizes are unchanged.
            bad = copy.deepcopy(manifest); bad["version"] = 1
            must_refuse(bad, layer, expert, "layout version")
            bad = copy.deepcopy(manifest); bad["layout"][0]["off"] = 16
            must_refuse(bad, layer, expert, "geometry/layout")
            bad = copy.deepcopy(manifest); bad["slices"][0]["file"] = "../poison.slice"
            must_refuse(bad, layer, expert, "image identity")
            tensors = []
            for proj in ("gate_proj", "up_proj", "down_proj"):
                block = fixtures[f"model.layers.{layer}.mlp.experts.{expert}.{proj}"]
                expected_shape = [4096, 2048] if proj == "down_proj" else [2048, 4096]
                assert block["shape"] == expected_shape
                w, s = ((root / f"{tag}.{proj}.{suffix}").read_bytes() for suffix in ("w", "s"))
                assert hashlib.sha256(w).hexdigest() == block["weight_sha256"]
                assert hashlib.sha256(s).hexdigest() == block["scale_sha256"]
                assert len(w) == 4194304 and len(s) == 262144
                tensors.append((w, s))
            for rank, entry in enumerate(manifest["slices"]):
                path = root / entry["file"]
                image = path.read_bytes()
                assert len(image) == SIZE and hashlib.sha256(image).hexdigest() == entry["sha256"]
                pieces = []
                for proj, (w, s) in enumerate(tensors):
                    if proj < 2:
                        pieces.extend((w[rank * 1048576:(rank + 1) * 1048576],
                                       s[rank * 65536:(rank + 1) * 65536]))
                    else:
                        pieces.extend((b"".join(w[row * 1024 + rank * 256:row * 1024 + (rank + 1) * 256] for row in range(4096)),
                                       b"".join(s[row * 64 + rank * 16:row * 64 + (rank + 1) * 16] for row in range(4096))))
                assert image == b"".join(pieces), "Rust rank image is not the independent full-source slice"
                command([executable, "prepare", path, rank], stage / f"{path.stem}.log")
                image_receipts.append(dict(file=entry["file"], sha256=entry["sha256"], bytes=len(image)))
    assert len(image_receipts) == 36
    return dict(images=image_receipts, real_bytes=36 * SIZE, source_matrix_hashes=54,
                fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                manifest_negatives=27, numerical_mode="E-W4A8-v1")


def main():
    case, stage_text = sys.argv[1:]
    if case not in ("prepare", "staging", "plan", "route", "quant"):
        raise ValueError("unknown B1 cell")
    stage = Path(stage_text).resolve(strict=True)
    timestamp = datetime.datetime.now(ZoneInfo("Australia/Sydney")).isoformat()
    print("B1 CPU cell", case, timestamp, flush=True)
    source_paths = sorted([*B1.glob("*.cu"), *B1.glob("*.cuh"), *HERE.glob("*.cpp"),
                           *HERE.glob("*.cu"), *HERE.glob("*.py"), HERE.parent / "b1_prep.rs"])
    hashes = lambda: {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    source_hashes = hashes()
    source_commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True)
    executable = stage / "b1-host"
    main_source = HERE / (f"{case}.cpp" if case in ("plan", "route", "quant") else "host.cpp")
    command(["g++", "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
             "-I", B1, "-x", "c++", main_source, B1 / "prepare.cu", B1 / "group_plan.cu",
             B1 / "route_reduce.cu", B1 / "quant_v1.cu", "-o", executable], stage / "host-compile.log")
    if os.environ.get("MIMO26_B1_COMPILE_CUDA") == "1":
        nvcc = Path("/usr/local/cuda-12.8/bin/nvcc")
        command([nvcc, "--version"], stage / "nvcc-version.log")
        for src in (B1 / "prepare.cu", B1 / "group_plan.cu", B1 / "route_reduce.cu", B1 / "quant_v1.cu", HERE / "cuda_compile.cu"):
            command([nvcc, "-O3", "-std=c++17", "-lineinfo", "--ftz=false", "-arch=sm_89",
                     "-I", B1, "-c", src, "-o", stage / (src.stem + ".o")], stage / (src.stem + "-compile.log"))
        print("COMPILE ONLY sm_89; no GPU initialized or launched; not sm_121 qualification", flush=True)
    if case == "prepare":
        receipt = prepare(executable, stage)
    elif case == "quant":
        command(["python3", "-B", HERE / "quant_compare.py", executable, stage], stage / "quant-reference.log")
        receipt = json.loads((stage / "quant-reference.json").read_text())
    else:
        args = [executable, stage] if case == "route" else [executable] if case == "plan" else [executable, "staging"]
        command(args, stage / f"{case}.log")
        receipt = dict(numerical_mode="E-W4A8-v1")
    assert hashes() == source_hashes, "sources changed during the cell"
    artifacts = [executable, *sorted(stage.glob("*.o"))]
    receipt.update(case=case, scope="CPU only; optional sm_89 compile; no GPU execution", status="PASS",
                   timestamp_sydney=timestamp, source_commit=source_commit, source_dirty=dirty,
                   source_sha256=source_hashes,
                   artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},
                   cuda_compile_only=os.environ.get("MIMO26_B1_COMPILE_CUDA") == "1")
    (stage / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("RESULT: PASS", case, "artifacts", stage, flush=True)


if __name__ == "__main__":
    main()
