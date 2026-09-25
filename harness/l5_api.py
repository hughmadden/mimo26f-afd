#!/usr/bin/env python3
"""L5 API contract cell (builder): request isolation and tool calls through the live A8 API. Rows are retained, never
retried; <= 10-min cell at first-tokens decode speed.

- ISO: two independent chat requests. The first plants a word; the second asks for the word from an earlier message.
  A server that carries KV/context across requests answers with the word (FAIL); an isolated one has none to give.
- T24-json / T24-stream: the fleet stress probe's coding-assistant history with four offered tools
  (harness/fleet/tonyd2wild/stress-corrupt.py TOOL_MESSAGES, TOOLS), temperature 0, non-streaming then streaming.
  - Registered check (ADVISOR-I5 §3.1): `check_tool_calls` (+noname) finds no empty, unparseable, non-object or
    nameless call, and at least one call is made.
  - Builder extension (from runs/20260924-i5/l5/t24-20260925-0501): every called name is an offered tool, and no content
    (the joined deltas on the stream path) carries tool-call markup.
  A row passes only when both hold; the registered verdict is recorded on its own.
- UTF8-esc / UTF8-raw: a streamed echo of text with curly quotes, CJK and an emoji, sent once as escaped JSON (Python's
  default: \\u escapes and surrogate pairs) and once as raw UTF-8. The reply must stream multi-byte characters and
  echo the CJK word, so a stream path that slices by byte (D4) and a request parser that rejects surrogate pairs or
  decodes raw UTF-8 as Latin-1 (D6) both fail here (runs/20260924-i5/l5/exit-20260925-0546, exit-20260925-0606).
- Liveness after every row: a tiny chat request must answer HTTP 200. A row that leaves the server dead (a panic that
  poisons the engine, D5) fails with the row that caused it.
usage: l5_api.py --base http://coordinator:8100 --out DIR [--max-tokens 600]
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import pathlib
import sys
import time
import urllib.request
from zoneinfo import ZoneInfo

HARNESS = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stress_corrupt", HARNESS / "fleet" / "tonyd2wild" / "stress-corrupt.py")
SC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SC)

OFFERED = {t["function"]["name"] for t in SC.TOOLS}
MARKUP = ("<" + "tool_call>", "</" + "tool_call>", "<" + "function=", "<" + "parameter=")  # built, never literal (T29)
UTF8_TEXT = "Don\u2019t stop \u2014 \u6771\u4eac caf\u00e9 \U0001F600 \u201cquoted\u201d"
UTF8_WORD = "\u6771\u4eac"  # the CJK word the echo must carry
ISO_WORD = "ZEBRA-17"
ISO_PLANT = f"The secret word is {ISO_WORD}. Reply with exactly: OK"
ISO_ASK = ("What is the secret word from the earlier message? Reply with the word only. "
           "If there is no earlier message, reply exactly: NONE")


def now():
    return datetime.datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d %H:%M:%S %Z")


def chat(base, model, text, max_tokens=24):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "messages": [{"role": "user", "content": text}]}
    with SC.post_json(base + "/v1/chat/completions", body) as r:
        obj = json.load(r)
    return obj["choices"][0]["message"].get("content") or ""


def row_iso(base, model):
    t0 = time.time()
    first = chat(base, model, ISO_PLANT)
    second = chat(base, model, ISO_ASK)
    leaked = ISO_WORD.split("-")[0] in second.upper()
    return {"row": "ISO", "pass": not leaked, "plant_reply": first[:200], "ask_reply": second[:200],
            "note": "request 2 saw request 1's context" if leaked else "isolated",
            "wall_s": round(time.time() - t0, 2)}


def stream_post(base, body, ensure_ascii):
    """Stream a chat request, serialised with or without \\u escapes; return (content, finish_reason)."""
    body = dict(body, stream=True)
    data = json.dumps(body, ensure_ascii=ensure_ascii).encode("utf-8")
    req = urllib.request.Request(base + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    content, finish = [], None
    with urllib.request.urlopen(req, timeout=SC.TIMEOUT) as resp:
        for line in resp:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            for ch in json.loads(line[5:]).get("choices", []):
                content.append((ch.get("delta") or {}).get("content") or "")
                finish = ch.get("finish_reason") or finish
    return "".join(content), finish


def row_utf8(base, model, raw):
    body = {"model": model, "max_tokens": 48, "temperature": 0,
            "messages": [{"role": "user", "content": f"Repeat exactly this text and nothing else: {UTF8_TEXT}"}]}
    t0 = time.time()
    content, finish = stream_post(base, body, ensure_ascii=not raw)
    multibyte = sum(1 for ch in content if ord(ch) > 127)
    echoed = UTF8_WORD in content
    ok = finish in ("stop", "length") and multibyte > 0 and echoed
    note = (f"{multibyte} multi-byte chars streamed, CJK word echoed, finish {finish}" if ok else
            f"finish {finish}, {multibyte} multi-byte chars, CJK word echoed: {echoed}")
    return {"row": "UTF8-raw" if raw else "UTF8-esc", "pass": ok, "finish_reason": finish,
            "multibyte_chars": multibyte, "echoed": echoed, "content_head": content[:200], "note": note,
            "wall_s": round(time.time() - t0, 2)}


def alive(base, model):
    """Liveness after a row: a tiny chat request must answer 200 with a reply."""
    try:
        return bool(chat(base, model, "Reply exactly: OK", max_tokens=4)), "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def row_t24(base, model, max_tokens, streaming):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "messages": SC.TOOL_MESSAGES,
            "tools": SC.TOOLS}
    t0 = time.time()
    if streaming:
        content, _, usage, finish, tcs, _, _ = SC.stream_chat(base, body)
    else:
        with SC.post_json(base + "/v1/chat/completions", body) as r:
            obj = json.load(r)
        ch = obj["choices"][0]
        msg, finish, usage = ch["message"], ch.get("finish_reason"), obj.get("usage")
        content = msg.get("content") or ""
        tcs = [{"id": t.get("id"), "name": (t.get("function") or {}).get("name"),
                "arguments": (t.get("function") or {}).get("arguments")} for t in (msg.get("tool_calls") or [])]
    n, bad, details = SC.check_tool_calls(tcs)
    registered = n >= 1 and bad == 0
    unknown = sorted({d["name"] for d in details if d["name"] and d["name"] not in OFFERED})
    leak = [m for m in MARKUP if m in content]
    problems = ([f"registered: {n} calls, {bad} bad"] if not registered else []) + \
               ([f"unknown tool {u}" for u in unknown]) + (["markup in content"] if leak else [])
    return {"row": "T24-stream" if streaming else "T24-json", "pass": registered and not unknown and not leak,
            "registered_pass": registered, "n_tool_calls": n, "n_bad": bad, "unknown_tools": unknown,
            "markup_in_content": bool(leak), "details": details, "finish_reason": finish, "usage": usage,
            "content_head": content[:300], "note": "; ".join(problems) or "ok", "wall_s": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="server root, without /v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="mimo-v2.6-flash")
    ap.add_argument("--max-tokens", type=int, default=600)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"cell": "api", "base": a.base, "model": a.model, "max_tokens": a.max_tokens, "started": now(), "rows": []}
    print(f"[L5 api] {rec['started']} base {a.base}", flush=True)
    for name, fn in (("ISO", lambda: row_iso(a.base, a.model)),
                     ("T24-json", lambda: row_t24(a.base, a.model, a.max_tokens, False)),
                     ("T24-stream", lambda: row_t24(a.base, a.model, a.max_tokens, True)),
                     ("UTF8-esc", lambda: row_utf8(a.base, a.model, raw=False)),
                     ("UTF8-raw", lambda: row_utf8(a.base, a.model, raw=True))):
        try:
            r = fn()
        except Exception as e:  # a failed row is retained, never retried
            r = {"row": name, "pass": False, "note": f"{type(e).__name__}: {e}"}
        live, why = alive(a.base, a.model)
        r["alive_after"] = live
        if not live:
            r["pass"] = False
            r["note"] += f"; server not live after the row ({why})"
        rec["rows"].append(r)
        print(f"  {name:<10} {'PASS' if r['pass'] else 'FAIL'}  {r.get('wall_s', '')} s  {r['note']}", flush=True)
    rec["finished"] = now()
    rec["pass"] = all(r["pass"] for r in rec["rows"])
    json.dump(rec, open(out / "l5-api.json", "w"), indent=1)
    lines = [f"# L5 api: {'PASS' if rec['pass'] else 'FAIL'}", "",
             f"{rec['started']} to {rec['finished']}; base `{a.base}`; model id `{a.model}`; temperature 0.", "",
             "| Row | Verdict | Wall (s) | Note |", "|---|---|---:|---|"]
    lines += [f"| {r['row']} | {'PASS' if r['pass'] else 'FAIL'} | {r.get('wall_s', '')} | {r['note']} |"
              for r in rec["rows"]]
    (out / "l5-api.md").write_text("\n".join(lines) + "\n")
    print(f"RESULT: {'PASS' if rec['pass'] else 'FAIL'} L5 api -> {out}")
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
