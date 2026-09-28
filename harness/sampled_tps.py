#!/usr/bin/env python3
"""Sampled vs greedy decode speed (V3's sampling-tps measurement, extended for the TensorFold import, 28 Sep 2026).

Arms: greedy, temperature 0.7 with top_p 0.95, temperature 1.0. Prompts: prose (the V3 lighthouse essay) and code
(an LRU cache module). Each prompt is salted so no prefix snapshot is reused; sampled arms are unseeded, as agent
clients send them. C1: --runs requests one after another, median per-stream decode tok/s (after the first token).
C16: 16 at once, aggregate tok/s over the wall time and mean per-stream decode.

usage: sampled_tps.py --base http://HOST:PORT/v1 --model MODEL [--tokens 512] [--runs 3] [--out FILE.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import threading
import time
import urllib.request

PROMPTS = {
    "prose": "Write a detailed essay about the history of lighthouses, their engineering and the lives of their keepers.",
    "code": "Write a Python module implementing an LRU cache class with get, put, resize and a thread-safe variant, "
            "with docstrings.",
}
ARMS = [("greedy", {}), ("T0.7 top_p0.95", {"temperature": 0.7, "top_p": 0.95}), ("T1.0", {"temperature": 1.0})]


def run(base, model, prompt, n, kw):
    body = {"model": model, "max_tokens": n, "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}], **kw}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, toks = time.time(), None, 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            for ch in d.get("choices", []):
                if (ch.get("delta") or {}).get("content") and first is None:
                    first = time.time()
            if d.get("usage"):
                toks = d["usage"]["completion_tokens"]
    end = time.time()
    first = first or end
    return {"tokens": toks, "ttft": first - t0, "decode_tps": toks / max(end - first, 1e-9)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    salt = f"{random.randrange(1 << 30):x}"
    rows = []
    for pname, ptext in PROMPTS.items():
        for label, kw in ARMS:
            c1 = [run(a.base, a.model, f"[{salt}-{pname}-{label}-1-{i}] {ptext}", a.tokens, kw) for i in range(a.runs)]
            res = [None] * 16

            def go(i, pname=pname, ptext=ptext, label=label, kw=kw):
                res[i] = run(a.base, a.model, f"[{salt}-{pname}-{label}-16-{i}] {ptext}", a.tokens, kw)

            ts = [threading.Thread(target=go, args=(i,)) for i in range(16)]
            t0 = time.time()
            [t.start() for t in ts]
            [t.join() for t in ts]
            wall = time.time() - t0
            row = {"prompt": pname, "arm": label, "c1": c1, "c1_median": statistics.median(r["decode_tps"] for r in c1),
                   "c16_aggregate": sum(r["tokens"] for r in res) / wall,
                   "c16_per_stream": statistics.mean(r["decode_tps"] for r in res), "c16": res}
            rows.append(row)
            print(f"{pname:5} {label:15} C1 {row['c1_median']:6.1f} tok/s (runs "
                  f"{', '.join(f'{r['decode_tps']:.1f}' for r in c1)})  C16 aggregate {row['c16_aggregate']:6.1f}, "
                  f"per stream {row['c16_per_stream']:5.1f}", flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
