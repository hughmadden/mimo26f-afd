#!/usr/bin/env python3
"""Served sampling and the bounded queue (perf reset V3, DS41RT v15's contract) against a live endpoint.

1. greedy: no temperature, temperature 0, and temperature 0.9 with top_k 1 give the same text;
2. seeded: temperature 0.8 / top_p 0.95 / seed 1234 twice gives the same text; seed 99 differs;
3. snapshot: a ~2,500-token prompt sampled with seed 7 cold, then again (an exact device hit: the
   first token is drawn from the snapshot's kept logits) gives the same text; seed 8 on the same
   snapshot gives a different one;
4. diversity: 4 concurrent unseeded requests at temperature 1 do not all agree;
5. spread: 16 seeds asking for a random number from 1 to 10 at temperature 2 with top_k 10 give at
   least 3 different replies (at temperature 1 this model answers 7 nearly every time);
6. stream: a seeded request streamed gives the non-streamed text;
7. refusals: temperature 3 and top_p 0 are 400;
8. queue (with --burst N): N concurrent requests of 256 tokens; beyond the slots, the queue and its
   waiters (3 x 16 by default) they are 429 with Retry-After, the rest complete.

usage: l5_sampling.py [--base http://coordinator:8100/v1] [--model mimo-v2.6-flash] [--burst 60] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import threading
import time
import urllib.error
import urllib.request


def post(base, body, stream=False, timeout=900):
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if not stream:
                d = json.loads(r.read())
                return 200, d["choices"][0]["message"].get("content") or "", time.time() - t0, None
            text = ""
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                for ch in json.loads(line[5:]).get("choices", []):
                    text += (ch.get("delta") or {}).get("content") or ""
            return 200, text, time.time() - t0, None
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace"), time.time() - t0, e.headers.get("Retry-After")


def ask(base, model, prompt, max_tokens=64, stream=False, **kw):
    body = {"model": model, "max_tokens": max_tokens, "stream": stream,
            "messages": [{"role": "user", "content": prompt}], **kw}
    return post(base, body, stream)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://coordinator:8100/v1")
    ap.add_argument("--model", default="mimo-v2.6-flash")
    ap.add_argument("--burst", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = {}
    salt = f"{random.randrange(1 << 30):x}"

    def check(name, ok, note):
        rows[name] = {"pass": bool(ok), "note": note}
        print(f"  {name:10} {'PASS' if ok else 'FAIL'}  {note}", flush=True)

    p1 = f"[{salt}] Write a haiku about the sea."
    g0 = ask(a.base, a.model, p1, 48)
    g1 = ask(a.base, a.model, p1, 48, temperature=0)
    g2 = ask(a.base, a.model, p1, 48, temperature=0.9, top_k=1)
    check("greedy", g0[0] == g1[0] == g2[0] == 200 and g0[1] == g1[1] == g2[1], repr(g0[1][:60]))

    p2 = f"[{salt}] Write two sentences about a lighthouse keeper."
    s = dict(temperature=0.8, top_p=0.95, seed=1234)
    a1 = ask(a.base, a.model, p2, 64, **s)
    a2 = ask(a.base, a.model, p2, 64, **s)
    b1 = ask(a.base, a.model, p2, 64, temperature=0.8, top_p=0.95, seed=99)
    check("seeded", a1[0] == a2[0] == b1[0] == 200 and a1[1] == a2[1] and a1[1] != b1[1],
          f"{a1[1][:50]!r} vs seed 99 {b1[1][:50]!r}")

    doc = "\n".join(f"Line {i} ({salt}): the {['red', 'green', 'blue', 'amber'][i % 4]} lantern number {i * 7 % 101} "
                    f"hangs by door {i % 13}." for i in range(160))
    p3 = doc + "\n\nWrite a short poem inspired by the lanterns above."
    c1 = ask(a.base, a.model, p3, 48, temperature=0.9, seed=7)
    c2 = ask(a.base, a.model, p3, 48, temperature=0.9, seed=7)
    c3 = ask(a.base, a.model, p3, 48, temperature=0.9, seed=8)
    check("snapshot", c1[0] == c2[0] == c3[0] == 200 and c1[1] == c2[1] and c1[1] != c3[1],
          f"cold {c1[2]:.2f} s, repeat {c2[2]:.2f} s, seed 8 {c3[2]:.2f} s; {c1[1][:40]!r} / {c3[1][:40]!r}")

    p4 = f"[{salt}] Invent a name for a fantasy tavern. Reply with the name only."
    outs = [None] * 4

    def run(i):
        outs[i] = ask(a.base, a.model, p4, 16, temperature=1.0)

    ts = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    names = [o[1].strip() for o in outs]
    check("diversity", all(o[0] == 200 for o in outs) and len(set(names)) >= 2, repr(names))

    p5 = f"[{salt}] Pick a random integer from 1 to 10. Reply with the number only."
    nums = [ask(a.base, a.model, p5, 8, temperature=2.0, top_k=10, seed=i)[1].strip() for i in range(16)]
    check("spread", len(set(nums)) >= 3, repr(nums))

    st = ask(a.base, a.model, p2, 64, stream=True, **s)
    check("stream", st[0] == 200 and st[1] == a1[1], repr(st[1][:50]))

    r1 = ask(a.base, a.model, p1, 8, temperature=3)
    r2 = ask(a.base, a.model, p1, 8, temperature=0.5, top_p=0)
    check("refusals", r1[0] == 400 and r2[0] == 400 and "temperature" in r1[1] and "top_p" in r2[1],
          f"{r1[0]} {r2[0]}")

    if a.burst:
        res = [None] * a.burst

        def burst(i):
            res[i] = ask(a.base, a.model, f"[{salt}-{i}] Count from 1 to 200, separated by spaces.", 256, temperature=0)

        ts = [threading.Thread(target=burst, args=(i,)) for i in range(a.burst)]
        t0 = time.time()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        codes = [r[0] for r in res]
        n429 = codes.count(429)
        retry = all(r[3] == "1" for r in res if r[0] == 429)
        worst = max(r[2] for r in res if r[0] == 429) if n429 else 0
        check("queue", n429 > 0 and codes.count(200) + n429 == a.burst and retry,
              f"{codes.count(200)} x 200, {n429} x 429 (Retry-After 1: {retry}; slowest 429 {worst:.1f} s), "
              f"{time.time() - t0:.0f} s")

    ok = all(v["pass"] for v in rows.values())
    print(f"RESULT: {'PASS' if ok else 'FAIL'} V3 sampling ({sum(v['pass'] for v in rows.values())}/{len(rows)})",
          flush=True)
    if a.out:
        pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(a.out) / "l5-sampling.json").write_text(json.dumps(rows, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
