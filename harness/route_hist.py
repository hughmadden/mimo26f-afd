#!/usr/bin/env python3
"""Tokens-per-expert (M) histograms from a route_trace.py .npz, and the effective expert-GEMM bandwidth they imply.

Each active (layer, expert) in a step is one full weight load of equal bytes, so the expert time per step is
sum over active experts of bytes / BW(M). Reported per workload:
  C1-w8 : one sequence, DFlash block-8 verification (8 consecutive tokens per step)
  C1-w1 : one sequence, plain decode
  Cn-w8 : n sequences' block-8 windows drawn from different prompts at random positions, same layer (seed fixed)
BW(M) rows (GB/s, frozen all-256 cell): B2 E-FP32 R6 M1-4 + c01f8cf exact-M M5-8; B1 E-W4A8-v1 record flat at its
worst-M 183.56 (62d1ca5; conservative). M > 8 runs as ceil(M/8) passes of 8 rows (a modelling assumption).
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

GATE = 191.1
BW = {
    "B2 E-FP32": {1: 218.8, 2: 215.9, 3: 196.4, 4: 192.5, 5: 171.1, 6: 156.8, 7: 143.7, 8: 131.7},
    "B1 E-W4A8-v1 (flat worst-M)": {m: 183.56 for m in range(1, 9)},
    "gate (191.1 at every M)": {m: GATE for m in range(1, 9)},
}


def loads(windows):
    """windows: list of [T_w, 8] int arrays for one layer-step -> list of M per active expert."""
    ids = np.concatenate([w.reshape(-1) for w in windows])
    return np.bincount(ids, minlength=256)[np.bincount(ids, minlength=256) > 0]


def eff_bw(ms, table):
    t = 0.0
    for m in ms:
        passes, rem = divmod(int(m), 8)
        t += passes / table[8] + (1 / table[rem] if rem else 0)
    return len(ms) / t if t else float("nan")  # distinct weight bytes moved per unit time (equal bytes per load)


def step_record(windows, layer):
    """One layer-step: the active experts and their M, from the token windows routed together."""
    ids = np.concatenate([w.reshape(-1) for w in windows])
    cnt = np.bincount(ids, minlength=256)
    ex = np.nonzero(cnt)[0]
    return {"layer": int(layer), "experts": ex.tolist(), "M": cnt[ex].tolist()}


def main() -> int:
    """argv: trace.npz [hist.json] [steps.json]. steps.json holds real per-step (expert, M) sets for the R18 cell:
    512 C1-w8 steps sampled uniformly over (prompt, window, layer), and every synthetic C4/C16 step."""
    d = np.load(sys.argv[1])
    steps = {"C1-w8": [], "C4-w8": [], "C16-w8": []}
    names = [k for k in d.files if k != "moe_layers"]
    rng = np.random.default_rng(20260924)
    work = {}
    for w in (1, 8):  # C1
        ms = []
        for nm in names:
            a = d[nm]
            for s in range(0, a.shape[1] - w + 1, w):
                for L in range(a.shape[0]):
                    ms.extend(loads([a[L, s:s + w]]))
        work[f"C1-w{w}"] = np.array(ms)
    for n in (4, 16):  # synthetic concurrency, block-8 windows from different prompts
        ms = []
        for _ in range(400):
            L = int(rng.integers(d[names[0]].shape[0]))
            wins = []
            for nm in rng.choice(names, size=n, replace=n > len(names)):
                a = d[nm]
                s = int(rng.integers(0, a.shape[1] - 8))
                wins.append(a[L, s:s + 8])
            ms.extend(loads(wins))
            steps[f"C{n}-w8"].append(step_record(wins, d["moe_layers"][L]))
        work[f"C{n}-w8"] = np.array(ms)
    srng = np.random.default_rng(20260925)  # C1-w8 step sample, independent of the synthetic draws above
    for _ in range(512):
        nm = names[int(srng.integers(len(names)))]
        a = d[nm]
        w0 = 8 * int(srng.integers(0, a.shape[1] // 8))
        L = int(srng.integers(a.shape[0]))
        steps["C1-w8"].append(step_record([a[L, w0:w0 + 8]], d["moe_layers"][L]))
    rep = {}
    for k, ms in work.items():
        hist = {int(m): int(c) for m, c in zip(*np.unique(ms, return_counts=True))}
        tot = len(ms)
        row = {"loads": tot, "mean_M": round(float(ms.mean()), 3),
               "share_M<=4": round(float((ms <= 4).mean()), 4), "share_M5-8": round(float(((ms >= 5) & (ms <= 8)).mean()), 4),
               "share_M>8": round(float((ms > 8).mean()), 4), "hist": hist,
               "eff_GBps": {tn: round(eff_bw(ms, tb), 1) for tn, tb in BW.items()}}
        rep[k] = row
        print(f"{k:7} loads={tot:>7} meanM={row['mean_M']:.2f} M<=4 {row['share_M<=4']:.1%} M5-8 {row['share_M5-8']:.1%} "
              f"M>8 {row['share_M>8']:.1%} | " + " | ".join(f"{tn}: {v}" for tn, v in row["eff_GBps"].items()))
    if len(sys.argv) > 2:
        json.dump(rep, open(sys.argv[2], "w"), indent=1)
    if len(sys.argv) > 3:
        json.dump(steps, open(sys.argv[3], "w"))
        print("steps:", {k: len(v) for k, v in steps.items()},
              {k: round(float(np.mean([len(x["M"]) for x in v])), 1) for k, v in steps.items()}, "mean active experts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
