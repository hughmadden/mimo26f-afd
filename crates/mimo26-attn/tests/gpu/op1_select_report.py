"""Strict OP1 selection-only validator. Exit 0 valid/ungated, 2 invalid."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics

from op1_schedule import CATEGORIES, cases, load_manifest


def fields(line):
    pairs = re.findall(r"([A-Za-z_][A-Za-z_0-9]*)=([^ ]+)", line)
    if len({k for k, _ in pairs}) != len(pairs):
        raise ValueError("duplicate field")
    return dict(pairs)


def expect(d, **wanted):
    for key, value in wanted.items():
        if d.get(key) != str(value):
            raise ValueError(f"wrong {key}: {d.get(key)} != {value}")


def bounded(value, lo=0, hi=2e-5, positive=False):
    x = float(value)
    if not math.isfinite(x) or x < lo or x > hi or (positive and x == 0):
        raise ValueError("nonfinite/out-of-range number")
    return x


def parse(text):
    if re.search(r"RESULT: (FAIL|REFUSE|INCOMPLETE)|\bPROXY\b", text):
        raise ValueError("failed/incomplete/proxy receipt")
    identity = re.findall(r"^IDENTITY (.+)$", text, re.M)
    if len(identity) != 1 or not re.search(r"^[0-9a-f]{64}  .*mimo26f-attn-bench-sm120$", text, re.M):
        raise ValueError("missing/duplicate device or binary identity")
    ident = fields(identity[0])
    expect(ident, arch="sm_120", sms=170, baked_arch="sm_120", baked_sms=170, label="TARGET")
    if "NVIDIA GeForce RTX 5090" not in identity[0] or not re.fullmatch(r"[0-9a-f]{12}", ident.get("source", "")):
        raise ValueError("wrong GPU/dirty source")
    if not re.search(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d AE[SD]T$", text, re.M) or "AOT: PASS architecture, SM-count, and baked kernel launch/readback" not in text:
        raise ValueError("missing Sydney/AOT evidence")
    memory = [fields(x) for x in text.splitlines() if x.startswith("MEMORY ")]
    if not memory:
        raise ValueError("missing memory evidence")
    for row in memory:
        expect(row, reserve_bytes=4294967296)
        if int(row["requested_bytes"]) < 0 or int(row["free_bytes"]) - int(row["requested_bytes"]) < 4294967296:
            raise ValueError("reserve failure")
    manifest = load_manifest()
    wanted = {row["id"]: row for row in cases(manifest) if row["select"]}
    shapes, correct, graphs, samples, resources = {}, {}, {}, {}, {}
    began = ended = False
    for line in text.splitlines():
        if line.startswith("OP1_RESOURCE "):
            d = fields(line); key = (d["mode"], d["family"])
            if key in resources or key[0] not in ("f32q", "bf16q") or key[1] not in ("p1", "c3"):
                raise ValueError("duplicate/wrong resource")
            native, c3 = key[0] == "bf16q", key[1] == "c3"
            expect(d, registers=(70 if native else 92) if c3 else (124 if native else 125),
                   shared=49536 if c3 else (38976 if native else 88128), capacity=2 if c3 or native else 1)
            resources[key] = d
        elif line.startswith("OP1_SELECT_BEGIN "):
            if began or ended or len(resources) != 4:
                raise ValueError("invalid begin")
            expect(fields(line), d7_sha=manifest["inputs"]["d7_sha256"], cases=36, kinds=2, modes=2, paths=5,
                   repeats=16, warmup=3, samples=7, scope="selection-only-context-proxy", gate="UNSET",
                   Q="post-RoPE-f32", Q_values="BF16-exact", KV="E4M3-unit", V="prescaled",
                   page="reverse-256", boundary="attention-core")
            began = True
        elif line.startswith("OP1_SELECT_COMPLETE "):
            if not began or ended or len(samples) != 5040:
                raise ValueError("incomplete/duplicate completion")
            expect(fields(line), shape_kinds=72, candidates=720, samples=5040, all_layer_step_measured=0)
            ended = True
        elif line.startswith("OP1_SELECT_"):
            if not began or ended:
                raise ValueError("row outside begin/end")
            tag = line.split()[0]; d = fields(line)
            cid, kind = int(d["id"]), d["kind"]
            if cid not in wanted or kind not in ("ga", "swa"):
                raise ValueError("unknown shape")
            row = wanted[cid]; key = cid, kind; n = row["t"] * 64 * 128
            if tag == "OP1_SELECT_CASE":
                if key in shapes:
                    raise ValueError("duplicate shape")
                expect(d, category=CATEGORIES[row["category"]], step=row["step"], T=row["t"],
                       S=row["swa_s"] if kind == "swa" else row["s"], prefix=row["prefix"],
                       first=row["swa_start"] if kind == "swa" else 0, query_checked=row["t"]*64*192,
                       reference_checked=n, query_exact="PASS", reference_finite="PASS", coordinates=3)
                bounded(d["max_coordinate_diff"]); shapes[key] = row
                continue
            if key not in shapes:
                raise ValueError("missing reference")
            mode, path = d["mode"], int(d["path"])
            if mode not in ("f32q", "bf16q") or path not in range(5):
                raise ValueError("unknown mode/path")
            key = cid, kind, mode, path
            if tag == "OP1_SELECT_CORRECT":
                if key in correct or any(k[:2] == key[:2] for k in samples):
                    raise ValueError("duplicate or late correctness")
                expect(d, checked=n, guard=128); bounded(d["max_error"]); correct[key] = d
            elif tag == "OP1_SELECT_GRAPH":
                if key in graphs or sum(k[:2] == key[:2] for k in correct) != 10:
                    raise ValueError("graph before both full-mode checks or duplicate")
                expect(d, nodes=16 if path == 0 else 32); graphs[key] = d
            elif tag == "OP1_SELECT_SAMPLE":
                index = int(d["index"])
                if key not in graphs or index not in range(7) or (*key, index) in samples:
                    raise ValueError("missing graph or invalid sample index")
                if index and (*key, index-1) not in samples:
                    raise ValueError("sample order")
                expect(d, checked=n, guard=128); bounded(d["max_error"])
                graph = bounded(d["graph_ms"], hi=10000, positive=True)
                call = bounded(d["per_call_ms"], hi=10000, positive=True)
                if not math.isclose(call*16, graph, rel_tol=2e-6, abs_tol=2e-8):
                    raise ValueError("wrong repeat divisor")
                samples[(*key, index)] = call
            else:
                raise ValueError("unknown OP1 record")
    if not ended or len(shapes) != 72 or len(correct) != 720 or len(graphs) != 720 or len(samples) != 5040:
        raise ValueError("incomplete selection")
    medians = {key: statistics.median(samples[(*key, i)] for i in range(7)) for key in correct}
    recommendations = []
    for phase in ("verification", "prefill"):
        for kind in ("ga", "swa"):
            means = []
            for path in range(5):
                values = [v for (cid, k, mode, p), v in medians.items()
                          if k == kind and mode == "f32q" and p == path
                          and (wanted[cid]["step"] < 0) == (phase == "prefill")]
                means.append(statistics.mean(values))
            best = min(range(5), key=lambda path: (means[path], path))
            recommendations.append({"phase": phase, "kind": kind, "f32_control_mean_ms_by_path": means,
                                    "selected_path": best, "family": "p1" if best == 0 else "c3",
                                    "splits": 0 if best == 0 else 1 << (best-1)})
    return {"status": "VALID", "scope": "selection-controls-only-not-48-layer-steps", "gate": "UNSET",
            "source": ident["source"], "d7_sha256": manifest["inputs"]["d7_sha256"],
            "shape_kinds": len(shapes), "checked_candidates": len(correct), "samples": len(samples),
            "selection_rule": "minimum-f32q-control-mean-per-phase-kind; same-path-for-native-diagnostic",
            "recommendations": recommendations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    try:
        result = parse(args.receipt.read_text())
    except (ValueError, KeyError, TypeError) as exc:
        print(f"INCOMPLETE/INVALID: {exc}"); return 2
    print(json.dumps(result, indent=2, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
