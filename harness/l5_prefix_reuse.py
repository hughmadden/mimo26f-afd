#!/usr/bin/env python3
"""Prefix-reuse check (KV host tier 1, perf reset K1): a multi-turn conversation over a long haystack.

1. turn 1, cold: the needle3 haystack (``--target`` tokens) asking for all three codes;
2. turn 2: the whole history (turn 1 + its answer) plus a follow-up question. It should restore from
   turn 1's completion-end snapshot and prefill only the tail;
3. turn 1 again, byte-identical. It should restore turn 1's prompt-end snapshot exactly, with no prefill;
4. turn 2 of a DIFFERENT haystack (a new target). This is a control that must NOT match and runs cold.

TTFT comes from the first streamed content delta. PASS requires:
- every answer is correct (codes retrieved);
- the repeat's answer equals the first turn-1 answer;
- turn 2 and the repeat are at least 5x faster to first token than turn 1.

The server log (`[hostcache] restore ...`) is the evidence of what was restored.

usage: l5_prefix_reuse.py --base http://coordinator:8100/v1 [--target 32000] [--out DIR] [--no-control]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stream(base, messages, max_tokens, timeout=1800):
    body = {"model": "mimo-v2.6-flash", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, text, usage = None, [], None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = (ch.get("delta") or {}).get("content")
                if d:
                    if ttft is None:
                        ttft = time.time() - t0
                    text.append(d)
    return {"text": "".join(text), "ttft_s": round(ttft or 0.0, 3), "wall_s": round(time.time() - t0, 3), "usage": usage}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://coordinator:8100/v1")
    ap.add_argument("--target", type=int, default=32000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--salt", default=None,
                    help="text prepended to every prompt so turn 1 is really cold (default: this run's start time; "
                         "the server keeps snapshots across runs)")
    ap.add_argument("--no-control", action="store_true",
                    help="skip step 4, the second cold prefill (a 1M-token one takes over 20 minutes)")
    a = ap.parse_args()
    ladder = load("l5_ladder_mod", HERE / "l5_ladder.py")
    salt = a.salt if a.salt is not None else f"Run {time.time_ns()}.\n\n"
    prompt, codes = ladder.needle3_prompt(a.target)
    prompt = salt + prompt
    print(f"[L5 prefix reuse] {ladder.sydney()} base {a.base} target {a.target}", flush=True)
    rows = {}

    t1 = stream(a.base, [{"role": "user", "content": prompt}], 48)
    got1, ok1 = ladder.score_needle3(t1["text"], codes)
    rows["turn1_cold"] = dict(t1, pass_=ok1)
    print(f"  turn1 cold     ttft {t1['ttft_s']:7.3f} s  prompt {t1['usage'] and t1['usage'].get('prompt_tokens')}  "
          f"{'PASS' if ok1 else 'FAIL'} {got1}", flush=True)

    follow = "Now reply with only the Beta code, then the Alpha code, one per line, nothing else."
    msgs2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": t1["text"]},
             {"role": "user", "content": follow}]
    t2 = stream(a.base, msgs2, 32)
    ok2 = codes["Beta"] in t2["text"] and codes["Alpha"] in t2["text"]
    rows["turn2_reuse"] = dict(t2, pass_=ok2)
    print(f"  turn2 reuse    ttft {t2['ttft_s']:7.3f} s  prompt {t2['usage'] and t2['usage'].get('prompt_tokens')}  "
          f"{'PASS' if ok2 else 'FAIL'} {t2['text']!r}", flush=True)

    t3 = stream(a.base, [{"role": "user", "content": prompt}], 48)
    ok3 = t3["text"] == t1["text"]
    rows["turn1_repeat"] = dict(t3, pass_=ok3)
    print(f"  turn1 repeat   ttft {t3['ttft_s']:7.3f} s  {'PASS (identical answer)' if ok3 else 'FAIL'} {t3['text']!r}",
          flush=True)

    ok4 = True
    if not a.no_control:
        other, codes_o = ladder.needle3_prompt(a.target + 777)
        other = salt + other
        t4 = stream(a.base, [{"role": "user", "content": other}], 48)
        got4, ok4 = ladder.score_needle3(t4["text"], codes_o)
        rows["control_cold"] = dict(t4, pass_=ok4)
        print(f"  control cold   ttft {t4['ttft_s']:7.3f} s  {'PASS' if ok4 else 'FAIL'} {got4}", flush=True)

    fast = t2["ttft_s"] * 5 <= t1["ttft_s"] and t3["ttft_s"] * 5 <= t1["ttft_s"]
    ok = ok1 and ok2 and ok3 and ok4 and fast
    print(f"RESULT: {'PASS' if ok else 'FAIL'} L5 prefix reuse (speedup turn2 {t1['ttft_s'] / max(t2['ttft_s'], 1e-3):.1f}x, "
          f"repeat {t1['ttft_s'] / max(t3['ttft_s'], 1e-3):.1f}x)", flush=True)
    if a.out:
        pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(a.out) / f"l5-prefix-reuse-{a.target}.json").write_text(json.dumps(rows, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
