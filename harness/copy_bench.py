#!/usr/bin/env python3
"""Copy-heavy agent bench (TensorFold import, 28 Sep 2026): decode speed where the output copies its context.

Agent output often repeats earlier text: a whole file with one name changed, an edit tool call quoting the
lines it replaces, a function quoted verbatim. Copy windows (drafts copied from the context) should speed
those up and leave fresh text alone. Cases, all greedy, reasoning off, fixed prompts (so replies can be
compared between builds):

- rename-a / rename-b: output a whole source file with one identifier renamed;
- tool-edit: after a read_file result, an edit_file call whose old_string/new_string quote the file;
- quote: copy one function verbatim, then two sentences about it;
- fresh-code / fresh-prose: controls with nothing to copy.

Each case runs --runs times with the same prompt (the first may prefill, the rest hit the prefix cache;
decode speed is measured after the first token either way). Reports decode tok/s per case (median), and
sha256 of every reply so two builds can be compared. An engine that holds a tool call back until it is complete
(MiMo) streams nothing before the end, so its decode rate is unobservable; `e2e` (tokens over the whole request,
median of the cached repeats, runs 2..N) is the comparable figure there.

usage: copy_bench.py --base http://HOST:PORT/v1 --model MODEL [--runs 3] [--reasoning-off] [--cases a,b] [--out FILE.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
SRC_A = (HERE / "l5_vision.py").read_text()
SRC_B = (HERE / "l5_sampling.py").read_text()

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "edit_file",
     "description": "Replace an exact string in a file. old_string must match the file exactly, including indentation.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"},
                                                     "new_string": {"type": "string"}},
                    "required": ["path", "old_string", "new_string"]}}},
]


def cases():
    fence = "```"
    return [
        ("rename-a", 1800, None, [{"role": "user", "content":
            f"Rename the function `data_url` to `png_data_url` everywhere in this file (its definition and every "
            f"call). Reply with the complete updated file only, no commentary.\n\n{fence}python\n{SRC_A}{fence}"}]),
        ("rename-b", 2600, None, [{"role": "user", "content":
            f"Rename the function `ask` to `chat_request` everywhere in this file (its definition and every "
            f"call). Reply with the complete updated file only, no commentary.\n\n{fence}python\n{SRC_B}{fence}"}]),
        ("tool-edit", 700, TOOLS, [
            {"role": "user", "content": "In harness/l5_vision.py, make post() default to a 900 s timeout instead of 600 "
                                        "and make ask() pass stream through unchanged. Use edit_file; replace the whole "
                                        "post() function in one call."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": "harness/l5_vision.py"})}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": SRC_A}]),
        ("quote", 700, None, [{"role": "user", "content":
            f"Copy the function `post` from this file exactly as written (verbatim, in a python code block), then "
            f"explain in two sentences what it returns.\n\n{fence}python\n{SRC_B}{fence}"}]),
        ("fresh-code", 500, None, [{"role": "user", "content":
            "Write a Python module implementing an LRU cache class with get, put, resize and a thread-safe variant, "
            "with docstrings."}]),
        ("fresh-prose", 500, None, [{"role": "user", "content":
            "Write a detailed essay about the history of lighthouses and the lives of their keepers."}]),
    ]


def run(base, model, messages, tools, max_tokens, reasoning_off, key_header):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "stream": True, "messages": messages,
            "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    if reasoning_off:
        body["reasoning_effort"] = "none"
    headers = {"Content-Type": "application/json", **key_header}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    first = None
    text, args, n = [], [], 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            for ch in d.get("choices", []):
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or "") + "".join(
                    (tc.get("function") or {}).get("arguments") or "" for tc in delta.get("tool_calls") or [])
                if piece:
                    first = first or time.time()
                    text.append(delta.get("content") or "")
                    args.extend((tc.get("function") or {}).get("arguments") or "" for tc in delta.get("tool_calls") or [])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
    end = time.time()
    reply = "".join(text) + "\x00" + "".join(args)
    return {"ttft": (first or end) - t0, "decode_tps": n / max(end - (first or end), 1e-9) if n else 0.0,
            "e2e_tps": n / max(end - t0, 1e-9), "tokens": n, "sha": hashlib.sha256(reply.encode()).hexdigest()[:16]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--reasoning-off", action="store_true")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cases", default=None, help="comma-separated case names (default: all)")
    a = ap.parse_args()
    key_header = {}
    if a.key_file:
        k, v = pathlib.Path(a.key_file).read_text().strip().split(":", 1)
        key_header = {k.strip(): v.strip()}
    rows = {}
    for name, max_tokens, tools, messages in cases():
        if a.cases and name not in a.cases.split(","):
            continue
        rs = [run(a.base, a.model, messages, tools, max_tokens, a.reasoning_off, key_header) for _ in range(a.runs)]
        rows[name] = rs
        tps = [r["decode_tps"] for r in rs]
        e2e = statistics.median(r["e2e_tps"] for r in rs[1:] or rs)
        print(f"  {name:12} decode {statistics.median(tps):7.1f} tok/s (runs {', '.join(f'{x:.1f}' for x in tps)}); "
              f"e2e cached {e2e:6.1f}; {rs[0]['tokens']} tokens; ttft {rs[0]['ttft']:.2f} s; reply {rs[0]['sha']}"
              f"{'' if len({r['sha'] for r in rs}) == 1 else ' (replies differ between runs)'}", flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
