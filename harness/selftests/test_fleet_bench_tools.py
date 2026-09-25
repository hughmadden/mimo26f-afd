"""I5 REUSE selftests for the vendored bench tools (docs/REUSE.md rows for replay_exact, mimobench, mimo_needle).

- replay_exact: a canned SSE stream with 7 tool calls is counted as 7, and the request body is replayed exactly.
- mimobench: completion tokens come from the usage block, TTFT is the first content delta, concurrency is real,
  and DFlash acceptance comes from /metrics deltas.
- mimo_needle: the prompt is deterministic per size and depth, and PASS/FAIL parse correctly.

Offline: a loopback fake server on an ephemeral port. The vendored files stay byte-pinned (test_fleet_tools.py);
nothing here edits them.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

FLEET = pathlib.Path(__file__).resolve().parents[1] / "fleet" / "tonyd2wild"
NEEDLE_RE = re.compile(r"The secret vault code is (\d{6})")


class Fake:
    """Loopback OpenAI-shaped server. `reply(fake, handler, body)` writes the response; bodies and arrivals are
    recorded. /metrics serves the two vLLM spec-decode counters that mimobench reads."""

    def __init__(self, reply):
        self.reply, self.bodies, self.arrivals = reply, [], []
        self.drafts = self.accepted = 0
        self.lock = threading.Lock()
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"  # close after each response, so SSE readers stop at EOF

            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path != "/metrics":
                    return send_json(self, {}, 404)
                with fake.lock:
                    txt = (f'vllm:spec_decode_num_drafts_total{{model_name="m"}} {fake.drafts}\n'
                           f'vllm:spec_decode_num_accepted_tokens_total{{model_name="m"}} {fake.accepted}\n')
                data = txt.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake.lock:
                    fake.bodies.append(body)
                    fake.arrivals.append(time.time())
                fake.reply(fake, self, body)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()


def send_json(h, obj, code=200):
    data = json.dumps(obj).encode()
    h.send_response(code)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(data)))
    h.end_headers()
    h.wfile.write(data)


def send_sse(h, events, first_gap=0.0):
    """Stream `events` as SSE, sleeping `first_gap` before the first event that carries content."""
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.end_headers()
    waited = False
    for ev in events:
        has_content = any((c.get("delta") or {}).get("content") for c in ev.get("choices") or [])
        if has_content and not waited:
            time.sleep(first_gap)
            waited = True
        h.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
        h.wfile.flush()
    h.wfile.write(b"data: [DONE]\n\n")
    h.wfile.flush()


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, FLEET / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # mimobench is import-safe (main() is behind __name__ == "__main__")
    return mod


def run_tool(fname, *args):
    p = subprocess.run([sys.executable, str(FLEET / fname), *args], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip().splitlines()


# ---------------------------------------------------------------- replay_exact

def _tool_call_events():
    ev = [{"choices": [{"index": 0, "delta": {"role": "assistant"}}]}]
    for k in range(5):  # calls 0-4: header, then the arguments in two fragments
        ev.append({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": k, "id": f"call_{k}", "type": "function", "function": {"name": f"tool{k}", "arguments": ""}}]}}]})
        ev.append({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": k, "function": {"arguments": '{"a":'}}]}}]})
        ev.append({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": k, "function": {"arguments": f"{k}}}"}}]}}]})
    # calls 5 and 6 packed into one chunk, and call 6's name arrives only in a later fragment
    ev.append({"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 5, "id": "call_5", "type": "function", "function": {"name": "tool5", "arguments": "{}"}},
        {"index": 6, "id": "call_6", "type": "function", "function": {"arguments": "{"}}]}}]})
    ev.append({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 6, "function": {"name": "tool6", "arguments": "}"}}]}}]})
    ev.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    ev.append({"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 123, "total_tokens": 173}})
    return ev


def _replay_reply(fake, h, body):
    if body.get("stream"):
        return send_sse(h, _tool_call_events())
    calls = [{"id": f"call_{k}", "type": "function", "function": {"name": f"tool{k}", "arguments": "{}"}} for k in range(7)]
    send_json(h, {"choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": calls},
                               "finish_reason": "tool_calls"}],
                  "usage": {"prompt_tokens": 50, "completion_tokens": 123, "total_tokens": 173}})


@pytest.mark.parametrize("mode", ["stream", "nostream"])
def test_replay_exact_counts_seven_calls_and_replays_the_body(tmp_path, mode):
    captured = {"model": "m", "messages": [{"role": "user", "content": "use the tools"}],
                "tools": [{"type": "function", "function": {"name": f"tool{k}", "parameters": {}}} for k in range(7)],
                "stream": True, "stream_options": {"include_usage": True}, "temperature": 0}
    body_file = tmp_path / "body.json"
    body_file.write_text(json.dumps(captured))
    with Fake(_replay_reply) as fake:
        out = run_tool("replay_exact.py", fake.url + "/v1/chat/completions", str(body_file), "2", mode)
    assert len(out) == 2
    for i, line in enumerate(out, 1):
        assert line.startswith(f"{mode} run{i}: finish=tool_calls tokens=123 calls=7 "), line
    want = dict(captured, stream=(mode == "stream"))
    if mode == "nostream":
        want.pop("stream_options")
    assert fake.bodies == [want, want]  # exact replay: only `stream` (and stream_options when off) differ


# ---------------------------------------------------------------- mimobench

@pytest.fixture(scope="module")
def mb():
    return load("mimobench_under_test", "mimobench.py")


def _bench_reply(first_gap=0.0, hold=0.0, drafts=0, accepted=0):
    def reply(fake, h, body):
        time.sleep(hold)
        with fake.lock:
            fake.drafts += drafts
            fake.accepted += accepted
        send_sse(h, [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
            {"choices": [{"index": 0, "delta": {"content": "Hel"}}]},
            {"choices": [{"index": 0, "delta": {"content": "lo wor"}}]},  # a speculative chunk packs several tokens
            {"choices": [{"index": 0, "delta": {"content": "ld"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59}},
        ], first_gap=first_gap)
    return reply


def test_mimobench_tokens_from_usage_and_ttft_at_first_content_delta(mb):
    with Fake(_bench_reply(first_gap=0.3)) as fake:
        r = mb.stream_chat(fake.url + "/v1", "m", "hi", 20)
    assert r["completion_tokens"] == 17 and r["prompt_tokens"] == 42  # usage block, not the 3 content chunks
    assert r["chars"] == len("Hello world")
    assert 0.25 <= r["ttft_s"] < r["total_s"]  # the role-only, empty-content delta does not start TTFT
    assert r["decode_tok_s"] == pytest.approx(16 / (r["total_s"] - r["ttft_s"]))
    body = fake.bodies[0]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0 and body["chat_template_kwargs"] == {"enable_thinking": False}


def test_mimobench_batch_is_truly_concurrent_with_unique_tags(mb):
    args = types.SimpleNamespace(base=None, model="m")
    with Fake(_bench_reply(hold=0.5)) as fake:
        args.base = fake.url + "/v1"
        b = mb.run_batch(args, 3, "prose", "Explain", 10, "run")
    assert len(fake.bodies) == 3
    assert max(fake.arrivals) - min(fake.arrivals) < 0.3  # all three were in flight together
    assert b["wall_s"] < 1.2  # sequential would be >= 1.5 s
    prompts = sorted(x["messages"][0]["content"] for x in fake.bodies)
    assert prompts == [f"[bench v1 run prose c3 s{i}] Explain" for i in range(3)]  # unique front tags: no prefix hits
    assert b["tokens"] == 51 and b["accept_per_draft"] is None  # counters flat: no acceptance claimed


def test_mimobench_acceptance_from_metrics_deltas(mb):
    args = types.SimpleNamespace(base=None, model="m")
    with Fake(_bench_reply(drafts=2, accepted=5)) as fake:
        args.base = fake.url + "/v1"
        b = mb.run_batch(args, 2, "math", "Add", 10, "run")
    assert b["accept_per_draft"] == 2.5  # (2 x 5) accepted / (2 x 2) drafts over the batch


def test_mimobench_prompt_set_is_fixed(mb, tmp_path):
    assert mb.PROMPT_SET_VERSION == "v1" and len(mb.CATEGORIES) == 9
    assert mb.filler(430, 2000) == mb.filler(430, 2000) != mb.filler(430, 8000)
    out = tmp_path / "p.json"
    subprocess.run([sys.executable, str(FLEET / "mimobench.py"), "--dump-prompts", str(out)], check=True,
                   capture_output=True, timeout=60)
    d = json.loads(out.read_text())
    assert d["version"] == "v1" and d["temperature"] == 0 and d["thinking"] is False
    assert [c["name"] for c in d["categories"]][:3] == ["coding", "json", "narrative"]


# ---------------------------------------------------------------- mimo_needle

def _needle_reply(correct):
    def reply(fake, h, body):
        m = NEEDLE_RE.search(body["messages"][0]["content"])
        send_json(h, {"choices": [{"index": 0, "message": {"role": "assistant",
                                                           "content": (m.group(1) if correct else "000000")},
                                   "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 2000, "completion_tokens": 3, "total_tokens": 2003}})
    return reply


def test_mimo_needle_pass_and_fail_parse_with_a_deterministic_prompt():
    depths = [0.1, 0.5, 0.9]
    with Fake(_needle_reply(True)) as fake:
        out1 = run_tool("mimo_needle.py", fake.url + "/v1/chat/completions", "2000", "0.1,0.5,0.9")
        out2 = run_tool("mimo_needle.py", fake.url + "/v1/chat/completions", "2000", "0.1,0.5,0.9")
        bodies = list(fake.bodies)
    assert len(out1) == 3 and all(": PASS (want " in line for line in out1), out1
    assert [b["messages"] for b in bodies[:3]] == [b["messages"] for b in bodies[3:]]  # same prompt every run
    codes = [NEEDLE_RE.search(b["messages"][0]["content"]).group(1) for b in bodies[:3]]
    assert len(set(codes)) == 3  # one code per (size, depth)
    for b, d in zip(bodies[:3], depths):
        text = b["messages"][0]["content"]
        assert text.count("The secret vault code is") == 1
        assert NEEDLE_RE.search(text).start() / len(text) == pytest.approx(d, abs=0.03)
        assert b["temperature"] == 0 and b["max_tokens"] == 20
        assert b["chat_template_kwargs"] == {"enable_thinking": False}
        assert b["model"] == "mimo-v2.6-flash"  # the A8 API must accept this model id (integration-i5 §7)
    assert len(bodies[0]["messages"][0]["content"].split()) >= int(2000 * 0.215)
    with Fake(_needle_reply(False)) as fake:
        bad = run_tool("mimo_needle.py", fake.url + "/v1/chat/completions", "2000", "0.5")
    assert len(bad) == 1 and ": FAIL (want " in bad[0] and "got '000000'" in bad[0], bad
