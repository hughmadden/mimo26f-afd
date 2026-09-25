#!/usr/bin/env python3
"""Top-of-memory check (perf reset K3 follow-up): multi-turn reuse when a context nearly fills the GPU.

Run it with the GPU held nearly full, so the context under test is at the top of memory the way a 1M
context is (for example the ballast tool: `ballast <GiB>` on the coordinator host, sized so that
about 1 GiB stays free after turn 1).

1. turn 1, cold: a needle3 haystack of ``--target`` tokens;
2. the exact repeat. The prompt snapshot sits under the turn snapshot, and a fork does not fit, so
   the slot rewinds in place (the turn snapshot goes to RAM). It must be identical to turn 1;
3. turn 2 with a follow-up of ``--follow`` filler tokens. It outgrows the slot's reservation, and
   growing in place does not fit, so the slot is relocated through RAM. The codes must be correct;
4. the exact repeat again, identical.

The server log says which path each step took (``device hit``, ``relocating it through RAM``,
``[hostcache] restore``).

usage: l5_top_memory.py --base http://coordinator:8100/v1 [--target 262144] [--follow 6000] [--out DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://coordinator:8100/v1")
    ap.add_argument("--target", type=int, default=262144)
    ap.add_argument("--follow", type=int, default=6000)
    ap.add_argument("--salt", default=None,
                    help="text prepended to the prompt so turn 1 is really cold (default: this run's start time)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ladder = load("l5_ladder_mod", HERE / "l5_ladder.py")
    reuse = load("l5_prefix_reuse_mod", HERE / "l5_prefix_reuse.py")
    print(f"[L5 top of memory] {ladder.sydney()} base {a.base} target {a.target} follow {a.follow}", flush=True)
    import time
    prompt, codes = ladder.needle3_prompt(a.target)
    prompt = (a.salt if a.salt is not None else f"Run {time.time_ns()}.\n\n") + prompt
    rows = {}

    def step(name, messages, max_tokens, check):
        r = reuse.stream(a.base, messages, max_tokens)
        ok, note = check(r["text"])
        rows[name] = dict(r, pass_=ok)
        print(f"  {name:10} ttft {r['ttft_s']:8.3f} s  prompt {r['usage'] and r['usage'].get('prompt_tokens')}  "
              f"{'PASS' if ok else 'FAIL'} {note}", flush=True)
        return r

    def needle(t):
        got, ok = ladder.score_needle3(t, codes)
        return ok, got

    t1 = step("turn1", [{"role": "user", "content": prompt}], 48, needle)
    step("repeat1", [{"role": "user", "content": prompt}], 48, lambda t: (t == t1["text"], repr(t)))
    import random
    rnd = random.Random(a.follow)
    filler = " ".join(f"{rnd.choice(ladder.FILLER)}{rnd.randint(0, 999)}"
                      for _ in range(int(a.follow * ladder.WORDS_PER_TOKEN)))
    follow = ("Here are some more notes, unrelated to the codes:\n" + filler +
              "\n\nNow reply with only the Beta code, then the Alpha code, one per line, nothing else.")
    step("turn2", [{"role": "user", "content": prompt}, {"role": "assistant", "content": t1["text"]},
                   {"role": "user", "content": follow}], 32,
         lambda t: (codes["Beta"] in t and codes["Alpha"] in t, repr(t)))
    step("repeat2", [{"role": "user", "content": prompt}], 48, lambda t: (t == t1["text"], repr(t)))
    ok = all(r["pass_"] for r in rows.values())
    print(f"RESULT: {'PASS' if ok else 'FAIL'} L5 top of memory", flush=True)
    if a.out:
        pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(a.out) / f"l5-top-memory-{a.target}.json").write_text(json.dumps(rows, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
