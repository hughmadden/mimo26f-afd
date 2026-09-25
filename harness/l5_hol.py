#!/usr/bin/env python3
"""Head-of-line check (perf reset Q1): a long prefill must not stall the streams already decoding.

1. stream A: a short prompt with a long answer (counting), streamed; every content delta is timestamped;
2. after A's first ``--warm`` seconds of output, prompt B arrives: a needle3 haystack of ``--target`` tokens;
3. while B prefills, A keeps receiving tokens between B's prefill segments.

Reported: A's largest gap between deltas while B prefills (the stall), B's TTFT and answer, and A's
tokens during B's prefill. PASS requires B's codes correct and A's largest gap under ``--max-gap``
seconds. Before Q1 the gap was the whole prefill (about 44 s at 128K).

usage: l5_hol.py --base http://coordinator:8100/v1 [--target 131072] [--max-gap 6] [--out DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import threading
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stream_times(base, messages, max_tokens, out, timeout=1800):
    """Stream a chat completion; append (time, text) per content delta to `out`."""
    body = {"model": "mimo-v2.6-flash", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": True}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            for ch in ev.get("choices", []):
                d = (ch.get("delta") or {}).get("content")
                if d:
                    out.append((time.time(), d))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://coordinator:8100/v1")
    ap.add_argument("--target", type=int, default=131072)
    ap.add_argument("--warm", type=float, default=3.0)
    ap.add_argument("--max-gap", type=float, default=6.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ladder = load("l5_ladder_mod", HERE / "l5_ladder.py")
    print(f"[L5 head-of-line] {ladder.sydney()} base {a.base} target {a.target}", flush=True)
    deltas: list = []
    count = "Count from 1 to 12000, one number per line, and nothing else."
    ta = threading.Thread(target=stream_times, args=(a.base, [{"role": "user", "content": count}], 36000, deltas),
                          daemon=True)
    t0 = time.time()
    ta.start()
    while not deltas or time.time() - deltas[0][0] < a.warm:
        if not ta.is_alive():
            raise SystemExit("stream A ended before B was sent")
        time.sleep(0.05)
    prompt, codes = ladder.needle3_prompt(a.target)
    reuse = load("l5_prefix_reuse_mod", HERE / "l5_prefix_reuse.py")
    tb0 = time.time()
    b = reuse.stream(a.base, [{"role": "user", "content": prompt}], 48)
    tb1 = tb0 + b["ttft_s"]
    got, ok_b = ladder.score_needle3(b["text"], codes)
    alive = ta.is_alive()  # A is left streaming; exiting closes it
    during = [t for t, _ in deltas if tb0 <= t <= tb1]
    edges = [t for t, _ in deltas if t < tb0][-1:] + during + [t for t, _ in deltas if t > tb1][:1]
    gaps = [y - x for x, y in zip(edges, edges[1:])]
    worst = max(gaps) if gaps else float("inf")
    ok = ok_b and worst <= a.max_gap and alive
    print(f"  B prefill      ttft {b['ttft_s']:7.3f} s  prompt {b['usage'] and b['usage'].get('prompt_tokens')}  "
          f"{'PASS' if ok_b else 'FAIL'} {got}", flush=True)
    print(f"  A during B     {len(during)} deltas, largest gap {worst:.3f} s (limit {a.max_gap} s); A still streaming "
          f"when B answered: {alive}", flush=True)
    print(f"  A overall      {len(deltas)} deltas in {deltas[-1][0] - t0:.1f} s", flush=True)
    print(f"RESULT: {'PASS' if ok else 'FAIL'} L5 head-of-line (A's largest gap {worst:.2f} s during a "
          f"{b['ttft_s']:.1f} s prefill)", flush=True)
    if a.out:
        pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
        rec = {"b": b, "b_pass": ok_b, "a_gaps_during_b": gaps, "a_deltas": len(deltas), "pass": ok}
        (pathlib.Path(a.out) / f"l5-hol-{a.target}.json").write_text(json.dumps(rec, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
