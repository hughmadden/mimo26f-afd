#!/usr/bin/env python3
"""KV pressure check (perf reset K3): more long conversations than the GPU can hold, so the device tier
evicts snapshots to the RAM tier and returning conversations restore from there.

1. fill: N conversations, each a needle3 haystack of ``--target`` tokens, prefilled cold (turn 1);
2. return: every conversation's turn 2 in the original order (the history, turn 1's answer and a
   follow-up). The oldest conversations were evicted to RAM and restore from it; the newest are
   still on the device;
3. repeat: every conversation's turn 1 again, byte-identical. It must give the identical answer
   (from the prompt-end snapshot on the device or in RAM).

TTFT comes from the first streamed content delta. PASS requires:
- every answer is correct;
- every repeat equals its turn 1;
- every return and repeat is at least 5x faster to first token than the median cold turn 1.

The server log is the evidence of which tier served each request (``device hit``, ``[hostcache]
restore``) and of the RAM copies (``[hostcache] store``, ``device pressure``).

usage: l5_kv_pressure.py --base http://coordinator:8100/v1 [--conversations 10] [--target 131072] [--order oldest|newest]
       [--out DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://coordinator:8100/v1")
    ap.add_argument("--conversations", type=int, default=10)
    ap.add_argument("--target", type=int, default=131072)
    ap.add_argument("--order", choices=["oldest", "newest"], default="oldest",
                    help="return/repeat order: oldest first (LRU's worst case, every return from RAM) or newest "
                         "first (the device's conversations hit on the device, the rest restore from RAM)")
    ap.add_argument("--salt", default=None,
                    help="text prepended to every prompt so the fill is really cold (default: this run's start time)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ladder = load("l5_ladder_mod", HERE / "l5_ladder.py")
    reuse = load("l5_prefix_reuse_mod", HERE / "l5_prefix_reuse.py")
    print(f"[L5 KV pressure] {ladder.sydney()} base {a.base}: {a.conversations} conversations of {a.target} tokens",
          flush=True)
    import time
    salt = a.salt if a.salt is not None else f"Run {time.time_ns()}.\n\n"
    convs = []
    for i in range(a.conversations):
        prompt, codes = ladder.needle3_prompt(a.target + 997 * i)
        convs.append({"prompt": salt + prompt, "codes": codes})
    rows = []
    follow = "Now reply with only the Beta code, then the Alpha code, one per line, nothing else."

    def run(phase, i, messages, max_tokens, check):
        r = reuse.stream(a.base, messages, max_tokens)
        ok, note = check(r["text"])
        rec = dict(r, phase=phase, conv=i, pass_=ok)
        rows.append(rec)
        prompt_tokens = r["usage"] and r["usage"].get("prompt_tokens")
        print(f"  {phase:6} conv {i:2}  ttft {r['ttft_s']:7.3f} s  prompt {prompt_tokens}  "
              f"{'PASS' if ok else 'FAIL'} {note}", flush=True)
        return r

    def fill_check(text, c):
        got, ok = ladder.score_needle3(text, c["codes"])
        return ok, got

    for i, c in enumerate(convs):
        c["turn1"] = run("fill", i, [{"role": "user", "content": c["prompt"]}], 48, lambda t, c=c: fill_check(t, c))
    order = list(range(len(convs))) if a.order == "oldest" else list(reversed(range(len(convs))))
    for i in order:
        c = convs[i]
        msgs = [{"role": "user", "content": c["prompt"]}, {"role": "assistant", "content": c["turn1"]["text"]},
                {"role": "user", "content": follow}]
        run("return", i, msgs, 32,
            lambda t, c=c: (c["codes"]["Beta"] in t and c["codes"]["Alpha"] in t, repr(t)))
    for i in order:
        c = convs[i]
        run("repeat", i, [{"role": "user", "content": c["prompt"]}], 48,
            lambda t, c=c: (t == c["turn1"]["text"], "identical" if t == c["turn1"]["text"] else repr(t)))

    cold = statistics.median(r["ttft_s"] for r in rows if r["phase"] == "fill")
    warm = [r for r in rows if r["phase"] != "fill"]
    slow = [(r["phase"], r["conv"], r["ttft_s"]) for r in warm if r["ttft_s"] * 5 > cold]
    correct = all(r["pass_"] for r in rows)
    ok = correct and not slow
    worst = max(r["ttft_s"] for r in warm)
    print(f"RESULT: {'PASS' if ok else 'FAIL'} L5 KV pressure (cold median {cold:.2f} s; returns/repeats worst "
          f"{worst:.3f} s = {cold / max(worst, 1e-3):.1f}x faster; {sum(r['pass_'] for r in rows)}/{len(rows)} "
          f"correct{'; slow: ' + str(slow) if slow else ''})", flush=True)
    if a.out:
        pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
        path = pathlib.Path(a.out) / f"l5-kv-pressure-{a.conversations}x{a.target}-{a.order}.json"
        path.write_text(json.dumps(rows, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
