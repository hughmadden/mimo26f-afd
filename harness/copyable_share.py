#!/usr/bin/env python3
"""Copyable share of agent output (TensorFold's copy-window measure) on dsh sessions.

Context = system, user and tool-result text; output = assistant text/reasoning and tool-call arguments, in
session order, tokenized with MiMo's tokenizer. An output token is copyable when the 8 tokens before it occurred
earlier in the stream and the token after that (latest) earlier occurrence equals it: a copy proposer entered on
8 matching tokens would have drafted it. Also: runs of consecutive copyable tokens (long runs are what copy
windows accelerate) and the share inside tool-call arguments.

usage: copyable_share.py [SINCE (YYYY-MM-DD, default 2026-09-19)]   # reads ~/.dsh/sessions/*/session.v3.jsonl.zstd
Needs `zstd` and the `tokenizers` package; TensorFold review, docs/TENSORFOLD-REVIEW-20260926.md item 2."""
import glob, json, os, subprocess, sys, collections
from tokenizers import Tokenizer
N = 8
tok = Tokenizer.from_file(os.path.expanduser("~/models/XiaomiMiMo/MiMo-V2.6-Flash-RL/tokenizer.json"))
since = sys.argv[1] if len(sys.argv) > 1 else "2026-09-19"
files = subprocess.run(["find", os.path.expanduser("~/.dsh/sessions"), "-name", "session.v3.jsonl.zstd", "-newermt", since],
                       capture_output=True, text=True).stdout.split()

def texts(content):
    """Every text in a content value, nested tool-result content included."""
    out = []
    if isinstance(content, str):
        return [content]
    for part in content or []:
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, dict):
            for k in ("text", "reasoning", "thinking"):
                if isinstance(part.get(k), str):
                    out.append(part[k])
            if isinstance(part.get("content"), (list, str)):
                out.extend(texts(part["content"]))
    return out

tot = collections.Counter(); runs = collections.Counter(); models = collections.Counter(); per_session = []
for f in files:
    raw = subprocess.run(["zstd", "-dc", f], capture_output=True).stdout.decode(errors="replace")
    stream, is_out, kind = [], [], []
    def add(text, out, k):
        ids = tok.encode(text, add_special_tokens=False).ids
        stream.extend(ids); is_out.extend([out] * len(ids)); kind.extend([k] * len(ids))
    model = None
    for line in raw.splitlines():
        try: d = json.loads(line)
        except Exception: continue
        t, data = d.get("type"), d.get("data") or {}
        if t == "model/selection": model = data.get("model")
        msg = data.get("message") if isinstance(data.get("message"), dict) else None
        if t in ("system/message", "user/message") and msg:
            for x in texts(msg.get("content")): add(x, False, "ctx")
        elif t == "tool/result" and msg:
            for x in texts(msg.get("content")): add(x, False, "ctx")
        elif t == "assistant/message" and msg:
            for x in texts(msg.get("content")): add(x, True, "text")
        elif t == "tool/call":
            args = data.get("arguments", data.get("input", data.get("args")))
            name = data.get("name") or data.get("toolName") or ""
            add(name + " " + (args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)), True, "tool")
    if not any(is_out): continue
    last = {}; cop = 0; outn = 0; run = 0; tool_out = tool_cop = 0
    for i, t in enumerate(stream):
        if i >= N:
            key = tuple(stream[i - N:i])
            if is_out[i]:
                outn += 1
                j = last.get(key)
                ok = j is not None and stream[j] == t
                cop += ok
                if kind[i] == "tool":
                    tool_out += 1; tool_cop += ok
                if ok: run += 1
                elif run: runs[min(run, 64)] += 1; run = 0
            last[key] = i
    if run: runs[min(run, 64)] += 1
    tot["out"] += outn; tot["copy"] += cop; tot["tool_out"] += tool_out; tot["tool_copy"] += tool_cop
    tot["ctx"] += sum(1 for x in is_out if not x); tot["sessions"] += 1
    models[model] += outn
    per_session.append(cop / max(outn, 1))
print(f"sessions {tot['sessions']} (since {since}); context tokens {tot['ctx']:,}; output tokens {tot['out']:,}")
print(f"copyable output: {tot['copy']:,} = {100*tot['copy']/max(tot['out'],1):.1f}%; in tool-call arguments "
      f"{100*tot['tool_copy']/max(tot['tool_out'],1):.1f}% of {tot['tool_out']:,}")
ps = sorted(per_session); q = lambda p: ps[int(p*(len(ps)-1))]
print(f"per-session copyable share: p10 {100*q(.1):.1f}%  median {100*q(.5):.1f}%  p90 {100*q(.9):.1f}%")
tokens_in_runs = sum(k * v for k, v in runs.items())
for lo, hi in ((1, 3), (4, 7), (8, 15), (16, 31), (32, 64)):
    n = sum(k * v for k, v in runs.items() if lo <= k <= hi)
    print(f"  copyable tokens in runs of {lo}-{hi}{'+' if hi == 64 else ''}: {100*n/max(tokens_in_runs,1):.1f}%")
print("output tokens by model:", models.most_common(6))
