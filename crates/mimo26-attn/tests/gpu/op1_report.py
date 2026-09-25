"""Strict OP1 full-cell validator and statistics. Exit 0 valid/ungated, 2 invalid."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics

from op1_schedule import CATEGORIES, cell_cases, category_statistics, load_manifest


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
        if int(row["free_bytes"]) - int(row["requested_bytes"]) < 4294967296:
            raise ValueError("reserve failure")
    manifest = load_manifest()
    wanted = {row["id"]: row for row in cell_cases(manifest)}
    resources = {}
    began = ended = False
    refs, prechecks, samples, checks = {}, {}, {}, {}
    for line in text.splitlines():
        if line.startswith("OP1_RESOURCE "):
            d = fields(line); key = (d["mode"], d["family"])
            if key in resources or key[0] not in ("f32q", "bf16q") or key[1] not in ("p1", "c3"):
                raise ValueError("duplicate/wrong resource")
            native, c3 = key[0] == "bf16q", key[1] == "c3"
            expect(d, registers=(70 if native else 92) if c3 else (124 if native else 125),
                   shared=49536 if c3 else (38976 if native else 88128), capacity=2 if c3 or native else 1)
            resources[key] = d
        elif line.startswith("OP1_BEGIN "):
            if began or ended or len(resources) != 4:
                raise ValueError("invalid begin")
            expect(fields(line), d7_sha=manifest["inputs"]["d7_sha256"], cases=2651, verification=2642,
                   short_prefill=9, layers=48, ga=9, swa=39, modes=2,
                   paths="ga-ver-c3p8,swa-ver-c3p4,ga-pre-c3p1,swa-pre-p1",
                   timing="direct-48-layer-span", scope="attention-critical-path", gate="UNSET",
                   boundary="core-post-rope-prescaled-kv", Q_values="BF16-exact")
            began = True
        elif line.startswith("OP1_COMPLETE "):
            if not began or ended or len(samples) != 2651 * 2 * 7:
                raise ValueError("incomplete/duplicate completion")
            expect(fields(line), verification_cases=2642, short_prefill_cases=9, cases=2651,
                   samples=2651 * 2 * 7, all_layer_step_measured=1)
            ended = True
        elif line.startswith("OP1_"):
            if not began or ended:
                raise ValueError("row outside begin/end")
            tag = line.split()[0]; d = fields(line)
            cid = int(d["case"])
            if cid not in wanted or d["category"] != CATEGORIES[wanted[cid]["category"]]:
                raise ValueError("unknown case")
            row = wanted[cid]; n = row["t"] * 64 * 128
            if tag == "OP1_REFERENCE":
                if cid in refs or d["request"] != str(row["request"]) or d["step"] != str(row["step"]) or d["phase"] != str(row["phase"]):
                    raise ValueError("duplicate/miskeyed reference")
                expect(d, layers=48, checked=n * 48, coordinates=48, query_checked=row["t"] * 64 * 192 * 48,
                       query_exact="PASS", reference_finite="PASS")
                bounded(d["max_coordinate_diff"])
                refs[cid] = d
            elif tag in ("OP1_PRECHECK", "OP1_CHECK"):
                mode = d["mode"]
                if mode not in ("f32q", "bf16q") or (cid, mode, tag) in prechecks or (cid, mode, tag) in checks:
                    raise ValueError("duplicate/wrong check")
                if tag == "OP1_PRECHECK" and cid in samples:
                    raise ValueError("precheck after samples")
                expect(d, layers=48, checked=n * 48, finite="PASS")
                bounded(d["max_error"])
                (prechecks if tag == "OP1_PRECHECK" else checks)[(cid, mode, tag)] = d
            elif tag == "OP1_SAMPLE":
                mode, index = d["mode"], int(d["index"])
                if mode not in ("f32q", "bf16q") or index not in range(7):
                    raise ValueError("wrong sample")
                key = (cid, mode, index)
                if key in samples or (index and (cid, mode, index - 1) not in samples):
                    raise ValueError("missing graph/duplicate/invalid sample index")
                expect(d, timing="core")
                if not (0 <= int(d["checked_layer"]) < 48):
                    raise ValueError("checked_layer out of range")
                bounded(d["max_error"])
                samples[key] = bounded(d["ms"], positive=True, hi=10000)
            else:
                raise ValueError("unknown OP1 record")
    if not ended or len(refs) != 2651 or len(samples) != 2651 * 2 * 7:
        raise ValueError("incomplete cell")
    for cid in wanted:
        for mode in ("f32q", "bf16q"):
            if (cid, mode, "OP1_PRECHECK") not in prechecks or (cid, mode, "OP1_CHECK") not in checks:
                raise ValueError("missing check")
    # Statistics: per-mode per-category verification step medians.
    def step_values(mode):
        out = {name: [] for name in CATEGORIES}
        for cid, row in wanted.items():
            if row["phase"] != 0:
                continue
            out[CATEGORIES[row["category"]]].append(statistics.median(samples[(cid, mode, i)] for i in range(7)))
        return out
    modes = {}
    for mode in ("f32q", "bf16q"):
        stats = category_statistics(step_values(mode))
        prefill = {}
        for cid, row in wanted.items():
            if row["phase"] == 1:
                prefill[CATEGORIES[row["category"]]] = statistics.median(samples[(cid, mode, i)] for i in range(7))
        modes[mode] = {"categories": stats["categories"],
                       "category_weighted_step_ms": stats["category_weighted_step_ms"],
                       "pooled_step_ms_diagnostic": stats["pooled_step_ms_diagnostic"],
                       "short_prefill_ms": prefill}
    return {"status": "VALID", "scope": "attention-critical-path-48-layer", "gate": "UNSET",
            "source": ident["source"], "d7_sha256": manifest["inputs"]["d7_sha256"],
            "verification_cases": 2642, "short_prefill_cases": 9, "samples": 2651 * 2 * 7,
            "modes": modes}


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
