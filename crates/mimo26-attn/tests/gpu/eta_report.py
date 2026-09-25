#!/usr/bin/env python3
"""Validate N5 micro receipts; no full-prefill or lattice-eligibility promotion."""
import argparse
import json
import re
import pathlib
import statistics
from bench_report import fields, number, near, require, InvalidReceipt

EXPECTED = {(m, n, mode) for m, n in ((64, 64), (32, 32), (32, 64))
            for mode in ("f32q", "bf16q")}


def validate(text):
    rows, samples, negatives = {}, {}, {}
    identity = None
    complete = aot = False
    for line in text.splitlines():
        require(not line.startswith(("RESULT: FAIL", "RESULT: REFUSE", "RESULT: INCOMPLETE")), "failed receipt")
        if line.startswith("IDENTITY "):
            require(identity is None, "duplicate identity")
            identity = fields(line)
        if line.startswith("AOT: PASS architecture, SM-count, and baked kernel launch/readback"):
            aot = True
        if line.startswith(("ETA ", "ETA_SAMPLE ", "ETA_NEGATIVE ")):
            require(aot and not complete, "row outside qualified interval")
            r = fields(line)
            key = number(r, "M", True), number(r, "N", True), r.get("mode")
            require(key in EXPECTED, "unknown tile/lattice")
            if line.startswith("ETA_SAMPLE "):
                require(key not in rows, "sample after metric")
                values = samples.setdefault(key, [])
                require(number(r, "index", True) == len(values), "sample index")
                value = number(r, "ms")
                require(value > 0, "nonpositive sample")
                values.append(value)
            elif line.startswith("ETA_NEGATIVE "):
                flags = negatives.setdefault(key, set())
                flag = number(r, "flag", True)
                require(flag not in flags and flag in ({1, 2} if key[2] == "f32q" else {2}), "negative flag")
                require(r.get("detected") == "PASS" and number(r, "max_abs") > 2e-5, "missing numerical negative")
                flags.add(flag)
            else:
                require(key not in rows, "duplicate metric")
                rows[key] = r
        if line == "RESULT: PASS eta micro harness (performance verdicts per row)":
            require(not complete, "duplicate completion")
            complete = True
    require(complete and aot and set(rows) == EXPECTED, "incomplete micro receipt")
    require(identity and identity.get("arch") == "sm_120" and number(identity, "sms", True) == 170,
            "wrong target identity")
    require(re.fullmatch(r"[0-9a-f]{7,40}(?:-dirty)?", identity.get("source", "")), "source commit")
    require(re.search(r"2026-\d\d-\d\d \d\d:\d\d:\d\d (?:AEST|AEDT)", text), "Sydney timestamp")
    require(identity.get("gpu") == "NVIDIA GeForce RTX 5090" and identity.get("label") == "TARGET", "wrong GPU")
    summary = []
    for key, r in rows.items():
        m, n, mode = key
        require(len(samples.get(key, [])) == 7, "incomplete samples")
        require(negatives.get(key) == ({1, 2} if mode == "f32q" else {2}), "incomplete negatives")
        ms = number(r, "median_ms")
        near(r, "median_ms", statistics.median(samples[key]), 1.1e-6)
        require(ms > 0 and 0 <= number(r, "max_abs") <= 2e-5, "numerical gate")
        require(r.get("Q_round") == ("Q3" if mode == "f32q" else "BF16-RNE-post-RoPE-once"), "Q lattice")
        require(r.get("K") == r.get("V") == "E4M3-unit" and r.get("cache") == "FP8-prescaled-V", "KV contract")
        require(r.get("stats") == "f32" and r.get("scope") == "interior-resident-micro-not-prefill", "scope")
        require(number(r, "K_abs_max") == 1.875 and number(r, "V_abs_max") == 1.25, "value domain")
        blocks, iterations = number(r, "blocks", True), number(r, "iterations", True)
        require(blocks == 680 and iterations == 32 and number(r, "warps", True) == m//8, "launch geometry")
        shared = m*192*2 + n*320*2 + m*n*2 + m*16 + n*320
        require(number(r, "shared_bytes", True) == shared, "shared budget")
        require(number(r, "registers", True) > 0 and number(r, "active_CTAs_per_SM", True) > 0, "resources")
        useful = 2*m*n*320*blocks*iterations
        executed = 2*m*n*((3 if mode == "f32q" else 1)*192+2*128)*blocks*iterations
        near(r, "useful_flops", useful, 0)
        near(r, "executed_flops", executed, 0)
        tf, etf = useful/(ms*1e9), executed/(ms*1e9)
        near(r, "useful_TFLOPS", tf, 1e-6 + tf*1.1e-6/ms)
        near(r, "executed_TFLOPS", etf, 1e-6 + etf*1.1e-6/ms)
        require(etf <= 209.5, "above assumed peak")
        near(r, "eta", etf/209.5, 1e-6)
        phases = [number(r, p) for p in ("load_cycles", "qk_pack_cycles", "softmax_hi_cycles", "pv_lowpack_cycles")]
        require(all(p > 0 for p in phases), "invalid phase clocks")
        near(r, "softmax_hi_fraction", phases[2]/sum(phases), 1e-6)
        verdict = "PASS" if (etf >= 125.7 if mode == "f32q" else tf >= 100) else "MISS"
        require(r.get("verdict") == verdict, "incorrect performance verdict")
        summary.append(dict(M=m, N=n, mode=mode, median_ms=ms, executed_TFLOPS=etf,
                            useful_TFLOPS=tf, eta=etf/209.5, phase_fractions=[p/sum(phases) for p in phases],
                            verdict=verdict, max_abs=number(r, "max_abs")))
    return dict(evidence="VALID", label="TARGET", source=identity.get("source"),
                scope="N5 interior micro only; not complete prefill or X1c eligibility", rows=summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=pathlib.Path)
    args = parser.parse_args()
    try:
        result = validate(args.receipt.read_text())
    except (InvalidReceipt, OSError) as exc:
        print(json.dumps(dict(evidence="INVALID_OR_INCOMPLETE", error=str(exc)), indent=2))
        raise SystemExit(2)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if any(r["verdict"] == "MISS" for r in result["rows"]) else 0)
