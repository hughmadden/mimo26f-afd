#!/usr/bin/env python3
"""L5 ladder runner (builder). Implements ADVISOR-I5 §3.0 as amended by I5-R6, and TEST-PLAN L5 and §14.

Cells:
  ladder      G1 ready, G2 coherence 3/3, G3 COUNT 5/5, G4 needles at 8K and 32K (the vendored mimo_needle, depths
              0.1/0.5/0.9) plus a three-needle control at 8K, and G5 decode smoke (C1 + C3 prose, 200 out). <= 10 min.
  needle128k  G4 at 128K as its own bounded cell: one cold prefill with three labelled needles (0.1/0.5/0.9).
              Exact prefix reuse is I5b, so "one cold prefill plus three queries" (A1 F7) is one request here.

G0 identity is assembled by the go-window operator and passed with --identity (embedded verbatim).
Every request: chat endpoint, temperature 0, thinking off. Failed rows are retained, and no row is retried.
usage: l5_ladder.py --base http://HOST:PORT/v1 --out DIR [--cell ladder|needle128k] [--identity id.json]
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import pathlib
import random
import re
import subprocess
import sys
import time
import types
import urllib.request
from zoneinfo import ZoneInfo

HERE = pathlib.Path(__file__).resolve().parent
FLEET = HERE / "fleet" / "tonyd2wild"
MODEL = "mimo-v2.6-flash"  # the id the vendored tools send; A8 must accept it
CELL_BUDGET_S = 600

COHERENCE = [
    ("apple", "Reply exactly APPLE.", "APPLE"),
    ("arithmetic", "Calculate 17 times 23. Reply with only the integer answer.", "391"),
    ("json", 'Reply with exactly this JSON object and nothing else: {"ok":true,"n":3}', {"ok": True, "n": 3}),
]
COUNT = [
    ("up20", "Count from 1 to 20, comma separated, nothing else.", list(range(1, 21))),
    ("up50", "Count from 1 to 50, comma separated, nothing else.", list(range(1, 51))),
    ("down15", "Count down from 15 to 1, comma separated, nothing else.", list(range(15, 0, -1))),
    ("even40", "List the even numbers from 2 to 40, comma separated, nothing else.", list(range(2, 41, 2))),
    ("up100", "Count from 100 to 130, comma separated, nothing else.", list(range(100, 131))),
]
PROSE = "Explain in about 120 words how a refrigerator keeps food cold, for a curious 12-year-old."
NEEDLE_LABELS = ("Alpha", "Beta", "Gamma")
NEEDLE_DEPTHS = (0.1, 0.5, 0.9)
WORDS_PER_TOKEN = 0.215  # mimobench/mimo_needle calibration (filler words per MiMo token)
FILLER = ("amber basin cedar delta ember fjord garnet harbor iris juniper kestrel lumen meadow nimbus orchid "
          "pylon quartz raven sierra tundra umber vessel willow xenon yarrow zephyr").split()


def sydney():
    return datetime.datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d %H:%M:%S %Z")


def chat(base, prompt, max_tokens, timeout=1800):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    c = d["choices"][0]
    return {"content": (c["message"].get("content") or ""), "reasoning": c["message"].get("reasoning_content"),
            "finish_reason": c.get("finish_reason"), "usage": d.get("usage"), "wall_s": round(time.time() - t0, 3)}


def needle3_prompt(target_tokens):
    """Deterministic haystack with three labelled needles; returns (prompt, {label: code})."""
    rnd = random.Random(target_tokens)
    words = [f"{rnd.choice(FILLER)}{rnd.randint(0, 999)}" for _ in range(int(target_tokens * WORDS_PER_TOKEN))]
    codes = {lab: f"{random.Random(target_tokens * 10 + k).randint(100000, 999999)}" for k, lab in enumerate(NEEDLE_LABELS)}
    n = len(words)
    for (lab, depth) in reversed(list(zip(NEEDLE_LABELS, NEEDLE_DEPTHS))):  # insert deepest first: indices stay valid
        words.insert(int(n * depth), f". The vault code for {lab} is {codes[lab]}. ")
    q = ("\n\nThree vault codes appear in the text above, for Alpha, Beta and Gamma. Reply with exactly three lines: "
         "Alpha: <code>, Beta: <code>, Gamma: <code>.")
    return " ".join(words) + q, codes


def score_needle3(text, codes):
    got = {lab: (m.group(1) if (m := re.search(rf"{lab}\s*:\s*(\d{{6}})", text)) else None) for lab in NEEDLE_LABELS}
    return got, all(got[lab] == codes[lab] for lab in NEEDLE_LABELS)


def row(name, fn):
    t0 = time.time()
    try:
        out = fn()
    except Exception as e:  # a failed row is retained with its error; the ladder continues
        out = {"pass": False, "error": f"{type(e).__name__}: {e}"}
    out.update(row=name, wall_s=round(time.time() - t0, 3), finished=sydney())
    print(f"  {name:6} {'PASS' if out['pass'] else 'FAIL'}  {out['wall_s']:8.1f} s  {out.get('note', out.get('error', ''))}",
          flush=True)
    return out


def g1(base):
    with urllib.request.urlopen(base + "/models", timeout=30) as r:
        d = json.load(r)
    ids = [m.get("id") for m in d.get("data", [])]
    return {"pass": MODEL in ids, "models": ids, "note": f"models {ids}"}


def g2(base):
    cases = []
    for name, prompt, want in COHERENCE:
        r = chat(base, prompt, 64)
        text = r["content"].strip()
        if name == "json":
            try:
                v = json.loads(text)
                ok = type(v) is dict and v == want and type(v["ok"]) is bool and type(v["n"]) is int
            except ValueError:
                ok = False
        else:
            ok = text == want
        ok = ok and r["finish_reason"] == "stop"
        cases.append(dict(r, case=name, expected=want, pass_=ok))
    n = sum(c["pass_"] for c in cases)
    return {"pass": n == 3, "cases": cases, "note": f"{n}/3"}


def g3(base):
    cases = []
    for name, prompt, want in COUNT:
        r = chat(base, prompt, 256)
        got = [int(x) for x in re.findall(r"-?\d+", r["content"])]
        ok = got == want and r["finish_reason"] == "stop" and bool(r["content"].strip())  # not reasoning-only
        cases.append(dict(r, case=name, parsed=got, pass_=ok))
    n = sum(c["pass_"] for c in cases)
    return {"pass": n == 5, "cases": cases, "note": f"{n}/5"}


NEEDLE_LINE = re.compile(r"^needle\s+(\d+) tok depth ([0-9.]+): (PASS|FAIL)")


def g4_mimo_needle(url, targets):
    p = subprocess.run([sys.executable, str(FLEET / "mimo_needle.py"), url, ",".join(map(str, targets)),
                        ",".join(map(str, NEEDLE_DEPTHS))], capture_output=True, text=True, timeout=CELL_BUDGET_S)
    lines = p.stdout.strip().splitlines()
    parsed = [m.groups() for m in map(NEEDLE_LINE.match, lines) if m]
    npass = sum(1 for g in parsed if g[2] == "PASS")
    want = len(targets) * len(NEEDLE_DEPTHS)
    return {"pass": p.returncode == 0 and len(parsed) == want and npass == want, "lines": lines,
            "stderr": p.stderr[-2000:], "note": f"{npass}/{want} (mimo_needle {targets})"}


def g4_needle3(base, target):
    prompt, codes = needle3_prompt(target)
    r = chat(base, prompt, 48)
    got, ok = score_needle3(r["content"], codes)
    ok = ok and r["finish_reason"] == "stop"
    pt = (r["usage"] or {}).get("prompt_tokens")
    return {"pass": ok, "target": target, "codes": codes, "got": got, "response": r,
            "note": f"{sum(got[k] == codes[k] for k in codes)}/3 at {pt} prompt tokens, {r['wall_s']:.0f} s"}


def g5(base):
    spec = importlib.util.spec_from_file_location("mimobench_l5", FLEET / "mimobench.py")
    mb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mb)
    args = types.SimpleNamespace(base=base, model=MODEL)
    batches = [mb.run_batch(args, c, "prose", PROSE, 200, "l5") for c in (1, 3)]
    ok = all(q["completion_tokens"] > 0 for b in batches for q in b["requests"])
    note = "; ".join(f"C{b['c']} agg {b['agg_tok_s']} tok/s ttft {b['ttft_mean_s']} s (smoke, not a claim)" for b in batches)
    return {"pass": ok, "batches": batches, "note": note}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="OpenAI base URL ending in /v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", choices=("ladder", "needle3", "needle128k"), default="ladder",
                    help="needle3: one cold prefill with three labelled needles at --needle3-target (I5-R18); "
                         "needle128k is needle3 at 128000")
    ap.add_argument("--identity", help="G0 identity JSON assembled by the go-window operator")
    ap.add_argument("--rows", default="", help="ladder cell only: comma list of rows to run (default all); a partial "
                    "run is recorded as PARTIAL and can never pass the ladder")
    ap.add_argument("--needle-targets", default="8000,32000", help=argparse.SUPPRESS)  # selftests shrink these
    ap.add_argument("--needle128k-target", type=int, default=128000, help=argparse.SUPPRESS)
    ap.add_argument("--needle3-target", type=int, default=32000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"cell": a.cell, "base": a.base, "model": MODEL, "started": sydney(),
           "identity": json.loads(pathlib.Path(a.identity).read_text()) if a.identity else None, "rows": []}
    t0 = time.time()
    print(f"[L5 {a.cell}] {rec['started']} base {a.base}", flush=True)
    if a.cell == "ladder":
        targets = [int(x) for x in a.needle_targets.split(",")]
        plan = [("G1", lambda: g1(a.base)), ("G2", lambda: g2(a.base)), ("G3", lambda: g3(a.base)),
                ("G4", lambda: g4_mimo_needle(a.base + "/chat/completions", targets)),
                ("G4c", lambda: g4_needle3(a.base, targets[0])), ("G5", lambda: g5(a.base))]
        want = [r for r in a.rows.split(",") if r]
        rec["partial"] = bool(want) and set(want) != {n for n, _ in plan}
        rec["rows"] += [row(n, f) for n, f in plan if not want or n in want]
    else:
        target = a.needle128k_target if a.cell == "needle128k" else a.needle3_target
        rec["rows"].append(row(f"G4-{round(target / 1000)}K" if a.cell == "needle3" else "G4-128K",
                               lambda: g4_needle3(a.base, target)))
    rec["wall_s"] = round(time.time() - t0, 3)
    rec["finished"] = sydney()
    rec["within_budget"] = rec["wall_s"] <= CELL_BUDGET_S
    rec["pass"] = all(r["pass"] for r in rec["rows"]) and rec["within_budget"] and not rec.get("partial")
    (out / f"l5-{a.cell}.json").write_text(json.dumps(rec, indent=1))
    head = "PARTIAL (rows " + ",".join(r["row"] for r in rec["rows"]) + ")" if rec.get("partial") else ("PASS" if rec["pass"] else "FAIL")
    md = [f"# L5 {a.cell}: {head}", "",
          f"{rec['started']} to {rec['finished']}; base `{a.base}`; model id `{MODEL}`; temperature 0, thinking off.",
          f"Wall clock {rec['wall_s']:.0f} s against the {CELL_BUDGET_S} s cell budget "
          f"({'within' if rec['within_budget'] else 'OVER'}). Failed rows are retained; no row was retried.", "",
          "| Row | Verdict | Wall (s) | Note |", "|---|---|---:|---|"]
    md += [f"| {r['row']} | {'PASS' if r['pass'] else 'FAIL'} | {r['wall_s']:.1f} | {r.get('note', r.get('error', ''))} |"
           for r in rec["rows"]]
    (out / f"l5-{a.cell}.md").write_text("\n".join(md) + "\n")
    print(f"RESULT: {'PASS' if rec['pass'] else 'FAIL'} L5 {a.cell} {rec['wall_s']:.0f} s -> {out}", flush=True)
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
