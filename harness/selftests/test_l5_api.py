"""Selftests for the L5 API contract cell (harness/l5_api.py), before it is pointed at the engine.

A scripted fake model on a loopback server either behaves (isolated requests, offered tool names, clean content,
multi-byte text echoed intact whether the request is escaped or raw UTF-8, alive after every row) or breaks one
contract: it carries context across requests, invents tool names, streams tool-call markup as content, dies
mid-stream and then answers 500 to everything (D4 + D5), dies right after a clean row, never streams a multi-byte
character, rejects escaped surrogate pairs, or decodes raw UTF-8 as Latin-1 (D6). The cell must pass the good server
and fail exactly the broken rows, keeping the registered T24 verdict separate.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import time

import pytest

HARNESS = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fleet_bench_fake", HARNESS / "selftests" / "test_fleet_bench_tools.py")
FT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FT)  # Fake, send_json, send_sse

MARKUP = "<" + "tool_call><" + "function=read><" + "parameter=file_path>src/theme/palette.js</" + "parameter></" \
         + "function></" + "tool_call>"
ROWS = ["ISO", "T24-json", "T24-stream", "UTF8-esc", "UTF8-raw"]
N_REQUESTS = 11  # ISO 2 + one each for the other 4 rows, and a liveness request after each of the 5 rows


class RawFake(FT.Fake):
    """The shared fake, with the raw request bytes kept on the handler (h.raw) for the encoding modes."""

    def __init__(self, reply):
        super().__init__(reply)
        fake = self

        def do_POST(h):
            h.raw = h.rfile.read(int(h.headers["Content-Length"]))
            body = json.loads(h.raw)
            with fake.lock:
                fake.bodies.append(body)
                fake.arrivals.append(time.time())
            fake.reply(fake, h, body)
        self.srv.RequestHandlerClass.do_POST = do_POST


def reply_factory(mode):
    memory = []  # what a context-carrying server has seen
    dead = []    # set once a dying server has died

    def reply(fake, h, body):
        if dead:
            return FT.send_json(h, {"error": "engine lock poisoned"}, 500)
        text = body["messages"][-1].get("content") or ""
        if text.startswith("Repeat exactly this text"):
            if mode == "surrogate400" and b"\\ud83d" in h.raw:
                return FT.send_json(h, {"error": "json: bad \\u codepoint"}, 400)
            if mode == "panic":
                dead.append(True)  # one delta, then the connection drops: no finish, no [DONE]
                h.send_response(200)
                h.send_header("Content-Type", "text/event-stream")
                h.end_headers()
                h.wfile.write(b"data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "Don"}}]}).encode()
                              + b"\n\n")
                return h.wfile.flush()
            if mode == "dieafter":
                dead.append(True)  # this stream completes cleanly; the server is dead for the next request
            echo = text.split("nothing else: ", 1)[1]
            if mode == "mojibake" and any(b >= 0x80 for b in h.raw):  # raw UTF-8 decoded byte by byte as Latin-1
                echo = echo.encode("utf-8").decode("latin-1")
            if mode == "ascii":
                echo = "Dont stop - Tokyo cafe :) quoted"
            third = max(1, len(echo) // 3)
            parts = [echo[:third], echo[third:2 * third], echo[2 * third:]]
            events = [{"choices": [{"index": 0, "delta": {"content": p}}]} for p in parts]
            return FT.send_sse(h, events + [{"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}])
        if "tools" not in body:
            memory.append(text)
            said = any("ZEBRA-17" in m for m in memory[:-1]) if mode == "carry" else False
            if text.startswith("The secret word") or text.startswith("Reply exactly: OK"):
                answer = "OK"
            else:
                answer = "ZEBRA-17" if said else "NONE"
            return FT.send_json(h, {"choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                                                 "finish_reason": "stop"}]})
        name = "read_file" if mode == "names" else "read"
        args = json.dumps({"file_path": "src/theme/palette.js"})
        if body.get("stream"):
            events = []
            if mode == "leak":
                events.append({"choices": [{"index": 0, "delta": {"content": MARKUP[:20]}}]})
                events.append({"choices": [{"index": 0, "delta": {"content": MARKUP[20:]}}]})
            events += [{"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                                                           "function": {"name": name, "arguments": ""}}]}}]},
                       {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0,
                                                                           "function": {"arguments": args}}]}}]},
                       {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}]
            return FT.send_sse(h, events)
        msg = {"role": "assistant", "content": "",
               "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": args}}]}
        FT.send_json(h, {"choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}]})
    return reply


def run(tmp_path, mode):
    with RawFake(reply_factory(mode)) as fake:
        p = subprocess.run([sys.executable, str(HARNESS / "l5_api.py"), "--base", fake.url, "--out", str(tmp_path)],
                           capture_output=True, text=True, timeout=60)
        n = len(fake.bodies)
    rec = json.loads((tmp_path / "l5-api.json").read_text())
    return p, {r["row"]: r for r in rec["rows"]}, rec, n


def test_good_server_passes_every_row(tmp_path):
    p, rows, rec, n = run(tmp_path, "good")
    assert p.returncode == 0 and rec["pass"], p.stdout + p.stderr
    assert list(rows) == ROWS and n == N_REQUESTS
    assert rows["ISO"]["ask_reply"] == "NONE" and rows["T24-stream"]["details"][0]["name"] == "read"
    assert all(rows[k]["echoed"] and rows[k]["multibyte_chars"] > 0 for k in ("UTF8-esc", "UTF8-raw"))
    assert all(r["alive_after"] for r in rows.values())
    assert (tmp_path / "l5-api.md").read_text().startswith("# L5 api: PASS")


@pytest.mark.parametrize("mode,failing,registered", [
    ("carry", {"ISO"}, None),
    ("names", {"T24-json", "T24-stream"}, True),   # the registered checker alone would pass these
    ("leak", {"T24-stream"}, True),
    ("panic", {"UTF8-esc", "UTF8-raw"}, None),     # dies mid-stream, then 500s: both rows fail, liveness catches it
    ("dieafter", {"UTF8-esc", "UTF8-raw"}, None),  # the first row itself is clean; only the liveness check sees it
    ("ascii", {"UTF8-esc", "UTF8-raw"}, None),     # never exercised multi-byte streaming
    ("surrogate400", {"UTF8-esc"}, None),          # D6: escaped surrogate pairs rejected; raw UTF-8 fine
    ("mojibake", {"UTF8-raw"}, None),              # D6: raw UTF-8 decoded as Latin-1; escaped fine
])
def test_each_broken_contract_fails_exactly_its_row(tmp_path, mode, failing, registered):
    p, rows, rec, n = run(tmp_path, mode)
    assert p.returncode == 1 and not rec["pass"] and n == N_REQUESTS  # failed rows are not retried
    assert {k for k, r in rows.items() if not r["pass"]} == failing
    for k in failing & {"T24-json", "T24-stream"}:
        assert rows[k]["registered_pass"] is registered
    if mode == "names":
        assert rows["T24-json"]["unknown_tools"] == ["read_file"]
    if mode == "leak":
        assert rows["T24-stream"]["markup_in_content"] and not rows["T24-json"]["markup_in_content"]
    if mode == "panic":
        assert rows["UTF8-esc"]["alive_after"] is False and "server not live" in rows["UTF8-esc"]["note"]
    if mode == "dieafter":
        assert rows["UTF8-esc"]["echoed"] and rows["UTF8-esc"]["alive_after"] is False
    if mode == "ascii":
        assert rows["UTF8-esc"]["alive_after"] and rows["UTF8-esc"]["multibyte_chars"] == 0
    if mode == "surrogate400":
        assert rows["UTF8-esc"]["alive_after"] and "HTTP Error 400" in rows["UTF8-esc"]["note"]
    if mode == "mojibake":
        assert rows["UTF8-raw"]["multibyte_chars"] > 0 and not rows["UTF8-raw"]["echoed"]
