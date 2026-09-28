#!/usr/bin/env python3
"""Tokens per speculative round, measured from the client (TensorFold import, 28 Sep 2026).

The engine streams a round's tokens back to back and the next round's after a whole verify step, so the
streamed deltas arrive in bursts, one per round. Clustering arrival times (gap < --gap-ms) counts the
rounds; tokens per round = (tokens after the first) / (bursts after the first). Unlike tok/s this does not
move with other traffic on the engine, so builds can be compared while it serves. Decode tok/s is reported
too (load-dependent).

Cases: prose and code (greedy, T0.7/top_p 0.95, T1.0; salted prompts, unseeded), and copy_bench's quote and
rename-a (greedy). Streams whose tool call is held back to the end (MiMo) cannot be measured this way.

usage: spec_rounds.py --base http://HOST:PORT/v1 --model MODEL [--runs 3] [--gap-ms 3] [--salt S]
                      [--sampled-only | --greedy-only] [--reasoning-off] [--out FILE.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import time
import urllib.request

import copy_bench

PROSE = "Write a detailed essay about the history of lighthouses, their engineering and the lives of their keepers."
CODE = "Write a Python module implementing an LRU cache class with get, put, resize and a thread-safe variant, with docstrings."
ARMS = [("greedy", {}), ("T0.7 top_p0.95", {"temperature": 0.7, "top_p": 0.95}), ("T1.0", {"temperature": 1.0})]


def run(base, model, messages, max_tokens, kw, gap, extra=None):
    body = {"model": model, "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
            "messages": messages, **kw, **(extra or {})}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    times, n = [], 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            for ch in d.get("choices", []):
                if (ch.get("delta") or {}).get("content"):
                    times.append(time.perf_counter())
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
    bursts = 1 + sum(1 for a, b in zip(times, times[1:]) if b - a > gap)
    return {"tokens": n, "bursts": bursts, "tokens_per_round": (n - 1) / max(bursts - 1, 1),
            "decode_tps": (n - 1) / max(times[-1] - times[0], 1e-9) if len(times) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--gap-ms", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--salt", default=None, help="fixed prompt salt, so two builds see the same prompts")
    ap.add_argument("--sampled-only", action="store_true", help="skip the greedy and copy cases")
    ap.add_argument("--greedy-only", action="store_true", help="only the greedy and copy cases")
    ap.add_argument("--reasoning-off", action="store_true", help="send reasoning_effort none (DS41RT thinks by default)")
    a = ap.parse_args()
    gap = a.gap_ms / 1000
    salt = a.salt or f"{random.randrange(1 << 30):x}"
    copy_cases = {name: (mt, msgs) for name, mt, tools, msgs in copy_bench.cases() if name in ("quote", "rename-a")}
    jobs = [(f"{p} {label}", kw, 512, lambda i, p=p, t=t, label=label: [{"role": "user", "content": f"[{salt}-{p}-{label}-{i}] {t}"}])
            for p, t in (("prose", PROSE), ("code", CODE)) for label, kw in ARMS]
    jobs += [(f"{name} greedy", {}, mt, lambda i, msgs=msgs: msgs) for name, (mt, msgs) in copy_cases.items()]
    if a.sampled_only:
        jobs = [j for j in jobs if j[1]]
    if a.greedy_only:
        jobs = [j for j in jobs if not j[1]]
    extra = {"reasoning_effort": "none"} if a.reasoning_off else None
    rows = []
    for label, kw, max_tokens, messages in jobs:
        rs = [run(a.base, a.model, messages(i), max_tokens, kw, gap, extra) for i in range(a.runs)]
        tpr = statistics.median(r["tokens_per_round"] for r in rs)
        tps = statistics.median(r["decode_tps"] for r in rs)
        rows.append({"case": label, "runs": rs, "tokens_per_round": tpr, "decode_tps": tps})
        print(f"{label:22} tokens/round {tpr:5.2f} (runs {', '.join(f'{r['tokens_per_round']:.2f}' for r in rs)})  "
              f"decode {tps:6.1f} tok/s", flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
