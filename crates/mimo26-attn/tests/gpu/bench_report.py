#!/usr/bin/env python3
"""Validate one attn_bench stdout receipt on the dev host (stdlib only, never GPU work).

Exit 0: all requested measurements valid and their applicable targets met.
Exit 1: valid measurements include a performance MISS (PROXY is not promotion).
Exit 2: missing/corrupt/failed evidence; never invent numbers for missing cells.
"""
import argparse
from datetime import datetime
import json
import math
import re
import statistics
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

SPECS = {
    "decode-4k": (1, 4096, 4, 0),
    "decode-32k": (1, 32768, 4, 0),
    "decode-128k": (1, 131072, 4, 0),
    "decode-1m": (1, 1048576, 4, 0),
    "prefill-2k": (2048, 2048, 4, 0),
    "prefill-4k": (2048, 4096, 4, 0),
    "prefill-32k": (2048, 32768, 4, 0),
    "prefill-swa": (2048, 32768, 8, 128),
}
TC_KERNELS = ("tc-q3-p2-d01",) + tuple(
    f"tc-{q}-p2-{stage}-w{w}" for q in ("q3", "bf16q") for stage in ("d1", "c0", "c3") for w in (4, 8)) + ("tc-q3-p2-c1-w8", "tc-bf16q-p2-c1-w8")
TARGET_GBS = 1253.0
TARGET_TFLOPS = 100.0


class InvalidReceipt(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise InvalidReceipt(message)


def fields(line):
    # Values such as GPU names contain spaces. Keys cannot repeat: silently
    # taking the final one would permit contradictory identity/metric fields.
    pairs = re.findall(r"(?:^| )([A-Za-z0-9_]+)=(.*?)(?= [A-Za-z0-9_]+=|$)", line)
    result = dict(pairs)
    require(len(result) == len(pairs), "duplicate field")
    return result


def number(row, key, integer=False):
    require(key in row, f"missing {key}")
    try:
        value = int(row[key]) if integer else float(row[key])
    except (ValueError, TypeError) as exc:
        raise InvalidReceipt(f"invalid {key}") from exc
    require(math.isfinite(value), f"nonfinite {key}")
    return value


def near(row, key, expected, tolerance):
    actual = number(row, key)
    require(abs(actual - expected) <= tolerance, f"{key}: reported {actual}, expected {expected}")


def visible_pairs(t, s, window):
    if window:
        return sum(min(window, s - t + i + 1) for i in range(t))
    return t * (s - t) + t * (t + 1) // 2


def parse_receipt(text, required):
    require(required and len(set(required)) == len(required), "empty/duplicate required cell list")
    require(all(c in SPECS for c in required), "unknown required cell")
    stamps = re.findall(r"^(?:TIME )?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} (?:AEST|AEDT))$", text, re.MULTILINE)
    require(stamps, "no Sydney timestamp in receipt")
    identity = None
    aot = False
    completed = False
    cells = {}
    current = None
    memory_free = []
    for line in text.splitlines():
        if completed and line.startswith(("IDENTITY ", "CELL ", "CORRECT ", "SAMPLE ", "METRIC ", "AOT: ")):
            raise InvalidReceipt("measurement after completion")
        if line.startswith("PROFILE_TIMING_NOT_A_GATE"):
            raise InvalidReceipt("profiler-instrumented event timings are not benchmark evidence")
        if line.startswith(("RESULT: FAIL", "RESULT: REFUSE", "RESULT: INCOMPLETE")):
            raise InvalidReceipt(f"retained failed row: {line}")
        if line.startswith("IDENTITY "):
            require(identity is None, "multiple runs in one receipt; validate separately")
            identity = fields(line)
        elif line.startswith("MEMORY "):
            row = fields(line)
            free = number(row, "free_bytes", True)
            total = number(row, "total_bytes", True)
            needed = number(row, "requested_bytes", True)
            reserve = number(row, "reserve_bytes", True)
            require(reserve == 4 * 1024**3 and 0 <= needed and free <= total,
                    "invalid memory accounting")
            require(free >= needed + reserve, "4 GiB memory reserve violated")
            memory_free.append(free)
        elif line == "AOT: PASS architecture, SM-count, and baked kernel launch/readback":
            require(identity is not None and not aot, "AOT gate order/duplication")
            aot = True
        elif line.startswith("CELL "):
            require(aot, "cell before positive hardware AOT gate")
            parts = line.split(" ", 2)
            require(len(parts) == 3, "malformed CELL")
            current = parts[1]
            require(current in SPECS and current not in cells, "unknown/duplicate cell")
            cells[current] = {"shape": fields(parts[2]), "samples": [], "correct": False, "metric": None}
        elif line.startswith("CORRECT "):
            require(current is not None, "orphan correctness row")
            row = fields(line)
            require(not cells[current]["correct"], "duplicate correctness row")
            require(row.get("sampled_oracle") == "3/3" and row.get("full_finite_scan") == "PASS",
                    "numerical sanity gate missing")
            require(0 <= number(row, "max_abs") <= 2e-5, "numerical sanity tolerance exceeded")
            if cells[current]["shape"].get("kernel") in TC_KERNELS:
                require(row.get("full_baseline") == "8192/8192", "TC full baseline comparison missing")
                require(0 <= number(row, "baseline_max_abs") <= 2e-5, "TC baseline mismatch")
            cells[current]["correct"] = True
        elif line.startswith("SAMPLE "):
            require(current is not None and cells[current]["correct"], "sample before numerical sanity")
            row = fields(line)
            cell = cells[current]
            require(cell["metric"] is None, "sample after median")
            require(row.get("cell") == current, "sample cell mismatch")
            require(number(row, "index", True) == len(cell["samples"]), "missing/duplicate/reordered sample")
            ms = number(row, "ms")
            require(ms > 0, "nonpositive event time")
            cell["samples"].append(ms)
        elif line.startswith("METRIC "):
            require(current is not None, "orphan metric")
            row = fields(line)
            require(row.get("cell") == current, "metric cell mismatch")
            require(cells[current]["metric"] is None, "duplicate metric")
            cells[current]["metric"] = row
        elif line == "RESULT: PASS timing harness (performance verdicts are per-cell PASS/MISS)":
            require(not completed, "duplicate completion marker")
            completed = True

    require(identity is not None and aot, "no positive hardware AOT evidence")
    require(completed, "no final timing completion marker")
    require(memory_free, "no memory reserve evidence")
    require(set(required).issubset(cells), "missing required cells: " + ",".join(sorted(set(required)-cells.keys())))
    arch = identity.get("arch")
    require(arch in ("sm_89", "sm_120"), "unsupported architecture")
    proxy = arch == "sm_89"
    label = "PROXY" if proxy else "TARGET"
    expected_sms = 128 if proxy else 170
    require(identity.get("label") == label, "PROXY/TARGET identity mismatch")
    require(("4090" if proxy else "5090") in identity.get("gpu", ""), "GPU name mismatch")
    require(identity.get("baked_arch") == arch, "baked architecture mismatch")
    require(number(identity, "sms", True) == expected_sms, "device SM count mismatch")
    require(number(identity, "baked_sms", True) == expected_sms, "baked SM count mismatch")
    require(re.fullmatch(r"[0-9a-f]{7,40}(?:-dirty)?", identity.get("source", "")), "missing source commit")

    rows = []
    for name, cell in cells.items():
        t, s, nkv, window = SPECS[name]
        shape, metric, samples = cell["shape"], cell["metric"], cell["samples"]
        require(metric is not None and cell["correct"], f"{name}: incomplete numerical/metric evidence")
        for key, expected in (("T", t), ("S", s), ("n_q", 64), ("n_kv", nkv),
                              ("QK", 192), ("V", 128), ("window", window)):
            require(number(shape, key, True) == expected, f"{name}: wrong {key}")
        require(shape.get("KV") == "E4M3-unit", "wrong KV dtype/scale mode")
        bf16q = shape.get("kernel", "").startswith("tc-bf16q-")
        require(shape.get("Q") == ("bf16-RNE-post-RoPE" if bf16q else "f32"), "query dtype/kernel mismatch")
        if bf16q:
            require(shape.get("Q_storage") == "f32" and shape.get("reference") == "bf16q-lattice-local",
                    "BF16-Q rounding/reference scope missing")
            require(shape.get("K_dtype") == "E4M3-unit" and shape.get("V_dtype") == "E4M3-unit", "BF16-Q K/V identity missing")
        require(shape.get("cached_V") == "prescaled", "missing cached V-scale semantics")
        require(shape.get("paged") == "reverse-256", "page indirection not exercised")
        require(shape.get("sink") == ("per-Q-head" if window else "absent"), "wrong sink semantics")
        require(shape.get("label") == label and metric.get("label") == label, "cell label mismatch")
        require(shape.get("kernel"), "missing kernel identity")
        require(not shape["kernel"].startswith("tc-") or shape["kernel"] in
                TC_KERNELS, "unknown TC kernel identity")
        require(1 <= number(shape, "warmup", True) <= 20, "invalid warmup count")
        count = number(shape, "samples", True)
        require(3 <= count <= 101 and len(samples) == count, "incomplete sample set")
        splits = number(shape, "splits", True)
        require(splits == 0 if t > 1 else 1 <= splits <= 4096, "invalid split count")
        ms = number(metric, "median_ms")
        require(ms > 0, "nonpositive median")
        # stdout prints events/medians to 6 decimal places; recompute from raw
        # samples without demanding precision the receipt cannot contain.
        near(metric, "median_ms", statistics.median(samples), 1.1e-6)
        near(metric, "min_ms", min(samples), 1.1e-6)
        near(metric, "max_ms", max(samples), 1.1e-6)
        byte_count = s * nkv * 320
        ops = 2 * 64 * 320 * visible_pairs(t, s, window)
        require(number(metric, "unique_KV_bytes", True) == byte_count, "inflated/incorrect unique KV bytes")
        require(number(metric, "useful_flops", True) == ops, "incorrect useful causal QK+PV FLOPs")
        gb = byte_count / (ms * 1e6)
        tf = ops / (ms * 1e9)
        rounding = 1.1e-6 / ms
        near(metric, "effective_KV_GBs", gb, 0.00051 + gb * rounding)
        near(metric, "pct_5090_peak", gb / 17.9, 0.00051 + gb / 17.9 * rounding)
        near(metric, "BF16_equiv_TFLOPS", tf, 0.00000051 + tf * rounding)
        near(metric, "GA9_extrapolated_ms", 9 * ms if t == 1 else 0, 1e-5)
        near(metric, "target_GBs", TARGET_GBS, 0)
        near(metric, "target_TFLOPS", TARGET_TFLOPS, 0)
        passed = tf >= TARGET_TFLOPS if t > 1 else gb >= TARGET_GBS
        if name == "decode-1m":
            passed = passed and 9 * ms <= 10
        reported = metric.get("verdict")
        require(reported in ("PASS", "MISS"), "invalid performance verdict")
        # Values extremely close to a target need the unrounded samples for a
        # decision: do not certify a threshold crossing from printed rounding.
        boundary = abs(tf-TARGET_TFLOPS) <= tf*rounding if t > 1 else abs(gb-TARGET_GBS) <= gb*rounding
        if name == "decode-1m":
            boundary = boundary or abs(9*ms-10) <= 1e-5
        require(not boundary, "target boundary requires higher-precision receipt")
        require(reported == ("PASS" if passed else "MISS"), "performance verdict contradicts numbers")
        peak_review = t == 1 and s >= 131072 and gb > (1008 if proxy else 1790)
        require(not peak_review, "long-decode bandwidth exceeds device peak; review work/byte formula before accepting")
        # A5 gates long decode only; small cache-resident decode is diagnostic.
        gated = name in required and (t > 1 or s >= 131072)
        tc_metrics = {}
        if shape["kernel"] in TC_KERNELS:
            require(t == 1 and nkv == 4, "unsupported TC benchmark shape")
            require(shape.get("precision_scope") == "bounded-synthetic" and
                    shape.get("Q_values") == "BF16-exact" and
                    number(shape, "KV_abs_max") == 1.875, "TC input scope missing")
            c1 = "-c1-" in shape["kernel"]
            tile = 16 if c1 else 32
            if c1 or "mma_tile_n" in shape:
                require(number(shape, "mma_tile_n", True) == tile, "wrong or missing MMA tile")
            padded = sum(((s*(i+1)//splits - s*i//splits + tile-1)//tile)*tile for i in range(splits))
            executed = 2*64*((1 if bf16q else 3)*192+2*128)*padded
            near(metric, "executed_mma_flops", executed, 0)
            near(metric, "mma_work_factor", executed/ops, 1e-8)
            rate = executed/(ms*1e9)
            near(metric, "executed_mma_TFLOPS", rate, 0.00000051+rate*rounding)
            require(rate <= 209.5, "executed MMA rate exceeds assumed peak; review accounting")
            if shape["kernel"] != "tc-q3-p2-d01":
                require(number(shape, "pipe_warps", True) == int(shape["kernel"][-1]), "D1 warp dispatch mismatch")
                require(0 < number(shape, "pipe_registers", True) <= 255, "D1 register readback missing")
                compact = "-c3-" in shape["kernel"]
                require(number(shape, "pipe_shared_bytes", True) == (44416 if c1 else 49536 if compact else 64192) and
                        number(shape, "pipe_active_ctas", True) == (2 if c1 or compact else 1), "pipe occupancy/shared readback mismatch")
                if c1:
                    require(number(shape, "pipe_registers", True) <= 128, "C1 two-CTA register budget")
                tc_metrics.update({key: number(shape, key, True) for key in
                                   ("pipe_warps", "pipe_registers", "pipe_shared_bytes", "pipe_active_ctas")})
            tc_metrics.update({"mma_tile_n": tile, "executed_mma_flops": executed, "executed_mma_TFLOPS": rate,
                          "mma_work_factor": executed/ops, "precision_scope": "bounded-synthetic",
                          "query_mode": "bf16-RNE-post-RoPE" if bf16q else "f32",
                          "reference_scope": "bf16q-lattice-local" if bf16q else "f32q"})
        rows.append({"cell": name, "label": label, "median_ms": ms, "unique_KV_bytes": byte_count,
                     "useful_flops": ops, "effective_KV_GBs": gb, "BF16_equiv_TFLOPS": tf,
                     "GA9_extrapolated_ms": 9*ms if t == 1 else None,
                     "verdict": reported, "target_gated": gated, "kernel": shape["kernel"], **tc_metrics})
    missed = any(r["target_gated"] and r["verdict"] == "MISS" for r in rows)
    verdict = "MISS" if missed else ("PASS" if any(r["target_gated"] for r in rows) else "NOT_APPLICABLE")
    return {"schema": "mimo26-attn-bench-report-v1", "evidence": "VALID",
            "label": label, "source": identity["source"], "gpu": identity["gpu"],
            "minimum_observed_free_bytes": min(memory_free), "receipt_times_sydney": stamps,
            "required_cells": required, "target_verdict": verdict,
            "promotion": "NONE (microbenchmark; PROXY is not a 5090 gate)" if proxy else
                         "NONE (microbenchmark, not engine promotion)", "cells": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--required", default="all", help="all, decode, prefill, or comma-separated cells")
    args = parser.parse_args()
    if args.required in ("all", "decode", "prefill"):
        required = [c for c in SPECS if args.required == "all" or c.startswith(args.required+"-")]
    else:
        required = args.required.split(",")
    stamp = datetime.now(ZoneInfo("Australia/Sydney")).isoformat(timespec="seconds")
    try:
        result = parse_receipt(args.receipt.read_text(), required)
    except (InvalidReceipt, OSError, UnicodeError) as exc:
        print(json.dumps({"schema": "mimo26-attn-bench-report-v1", "evidence": "INVALID_OR_INCOMPLETE",
                          "generated_at_sydney": stamp, "reason": str(exc)}, indent=2))
        return 2
    result["generated_at_sydney"] = stamp
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result["target_verdict"] == "MISS" else 0


if __name__ == "__main__":
    sys.exit(main())
