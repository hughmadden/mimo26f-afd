#!/usr/bin/env python3
"""Concurrent-serving check (perf reset W4, builder): the batching scheduler under load.

1. N simultaneous three-needle requests of different lengths (the ladder's
   needle3 prompts); every one must retrieve all three codes and stop.
2. Aggregate decode throughput at C1 / C4 / C8 with mimobench's prose batch (the
   ladder's G5 machinery; smoke numbers, not claims).

usage: l5_concurrent.py --base http://coordinator:8100/v1 [--out DIR] [--targets 1000,2000,...]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import pathlib
import time
import types

HERE = pathlib.Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://coordinator:8100/v1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--targets", default="1000,2000,3000,4000,5000,6000,7000,8000")
    ap.add_argument("--concurrency", default="1,4,8")
    a = ap.parse_args()
    ladder = load("l5_ladder_mod", HERE / "l5_ladder.py")
    targets = [int(x) for x in a.targets.split(",")]
    print(f"[l5 concurrent] {ladder.sydney()} base {a.base}: {len(targets)} simultaneous needle3 requests {targets}",
          flush=True)

    def one(target):
        prompt, codes = ladder.needle3_prompt(target)
        r = ladder.chat(a.base, prompt, 48)
        got, ok = ladder.score_needle3(r["content"], codes)
        return {"target": target, "pass": ok and r["finish_reason"] == "stop", "codes": codes, "got": got,
                "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"), "wall_s": r["wall_s"],
                "finish_reason": r["finish_reason"]}

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(len(targets)) as ex:
        rows = list(ex.map(one, targets))
    wall = round(time.time() - t0, 3)
    for r in rows:
        print(f"  needle3 {r['target']:6}  {'PASS' if r['pass'] else 'FAIL'}  {r['prompt_tokens']} prompt tokens, "
              f"{r['wall_s']:.1f} s", flush=True)
    needles_ok = all(r["pass"] for r in rows)
    print(f"  all {len(rows)} concurrent needles: {'PASS' if needles_ok else 'FAIL'} in {wall} s wall", flush=True)

    mb = load("mimobench_l5c", ladder.FLEET / "mimobench.py")
    args = types.SimpleNamespace(base=a.base, model=ladder.MODEL)
    batches = []
    for c in [int(x) for x in a.concurrency.split(",")]:
        b = mb.run_batch(args, c, "prose", ladder.PROSE, 200, "l5c")
        batches.append(b)
        print(f"  C{c}: agg {b['agg_tok_s']} tok/s, ttft {b['ttft_mean_s']} s (smoke, not a claim)", flush=True)
    ok = needles_ok and all(q["completion_tokens"] > 0 for b in batches for q in b["requests"])
    print(f"RESULT: {'PASS' if ok else 'FAIL'} L5 concurrent", flush=True)
    if a.out:
        pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(a.out) / "l5-concurrent.json").write_text(
            json.dumps({"needles": rows, "needles_wall_s": wall, "batches": batches, "pass": ok}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
