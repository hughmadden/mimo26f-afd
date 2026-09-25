"""harness/t29_goldens.py — T29/T27 parser conformance golden SEED (for I5).

COHERENCE-TRAPS T27/T29 gate: "I5: parser conformance goldens, including a
parameter value that contains the closing parameter tag and one that contains
the chat-eos special token".  This generates the corpus now (I4, CPU-only) so
the I5 tool-call parser port (vLLM `mimo` / Qwen3 parser engine semantics:
split on tags, no escaping, coerce to the tool schema) has its conformance
cases ready.

T29 hygiene (binding on MiMo lanes):
- Every tag and special token is BUILT FROM PARTS below (chr() joins).  The
  literals never appear in this source, in a tool argument, or in the committed
  JSON (the serializer escapes angle brackets as JSON \\u003c/\\u003e).
- The selftest enforces both properties.

Corpus record = {"id", "output", "expected", "must_report", "must_not"} where
"expected" is the parse any conformant parser must produce (documented
split-on-first-tag semantics where the format is ambiguous), "must_report" are
losses it must surface (never silently drop), and "must_not" are failure modes
(wedge, nameless call, wrong coercion) that are always wrong.
"""
from __future__ import annotations

import argparse
import json

# ---- tags and special tokens, built from parts (T29) ------------------------
LT, GT, SLASH, PIPE = chr(60), chr(62), chr(47), chr(124)


def _t(name: str, close: bool = False) -> str:
    return LT + (SLASH if close else "") + name + GT


TC, _TC = _t("tool_call"), _t("tool_call", True)
FN, _FN = _t("function"), _t("function", True)
PA, _PA = _t("parameter"), _t("parameter", True)
TH, _TH = _t("think"), _t("think", True)
IM_START = LT + PIPE + "im_start" + PIPE + GT
IM_END = LT + PIPE + "im_end" + PIPE + GT


def _fn_open(name: str) -> str:
    return LT + "function=" + name + GT


def _pa_open(key: str) -> str:
    return LT + "parameter=" + key + GT


def fn(name: str, *params: tuple[str, str]) -> str:
    body = "".join(_pa_open(k) + v + _PA for k, v in params)
    return TC + _fn_open(name) + body + _FN + _TC


def call_no_close(name: str, *params: tuple[str, str]) -> str:
    """A call whose closing tool_call tag is missing (T27: lost call)."""
    body = "".join(_pa_open(k) + v + _PA for k, v in params)
    return TC + _fn_open(name) + body + _FN


def build() -> list[dict]:
    return [
        {
            "id": "simple_two_params_schema_coercion",
            "output": fn("fetch", ("url", "https://x/y"), ("retries", "5")),
            "expected": {"calls": [{"name": "fetch",
                                    "arguments": {"url": "https://x/y", "retries": 5}}]},
            "must_report": [],
            "must_not": ["wrong coercion: retries must be int 5, not str"],
            "note": "T27 _coerce_value: text '5' + schema type integer -> 5",
        },
        {
            "id": "value_contains_closing_parameter_tag",
            "output": fn("echo", ("v", "abc" + _PA + "def")),
            "expected": {"calls": [{"name": "echo", "arguments": {"v": "abc"}}]},
            "must_report": ["unparsed tail after the embedded closing tag"],
            "must_not": ["wedge", "nameless call", "silent drop of the tail"],
            "note": "T29 gate case. Format has no escaping; documented semantics "
                    "= split on the FIRST closing parameter tag; the tail must be "
                    "reported, never silently dropped.",
        },
        {
            "id": "value_contains_chat_eos_special_token",
            "output": fn("echo", ("v", "abc" + IM_END + "def")),
            "expected": {"calls": [{"name": "echo",
                                    "arguments": {"v": "abc" + IM_END + "def"}}]},
            "must_report": [],
            "must_not": ["truncate the value", "treat it as end of message", "wedge"],
            "note": "T29 gate case. The EOS special token inside an argument is "
                    "plain text to the parser; only the model's generation honors it.",
        },
        {
            "id": "missing_closing_tool_call_tag",
            "output": call_no_close("fetch", ("url", "https://x/y")),
            "expected": {"calls": []},
            "must_report": ["lost call (closing tag missing)"],
            "must_not": ["invent a call", "silent drop"],
            "note": "T27: a call without its closing tag is LOST and must be "
                    "reported, matching vLLM mimo behavior.",
        },
        {
            "id": "nameless_call_is_an_error",
            "output": TC + _pa_open("url") + "x" + _PA + _TC,
            "expected": {"calls": [], "error": "nameless tool call"},
            "must_report": ["nameless tool call"],
            "must_not": ["persist a nameless call", "return 200 with an empty name"],
            "note": "T29: our API (A8) validates every parsed call (non-empty "
                    "name, JSON-parsable arguments) and returns an error.",
        },
        {
            "id": "non_string_renders_with_tojson",
            "output": fn("set", ("n", "5"), ("s", chr(34) + "5" + chr(34))),
            "expected": {"calls": [{"name": "set",
                                    "arguments": {"n": 5, "s": "5"}}]},
            "must_report": [],
            "must_not": ["coerce s to int", "coerce n to str"],
            "note": "T27: the template renders non-strings with tojson (5 vs " +
                    chr(34) + "5" + chr(34) + " are distinct); the parser coerces to the schema.",
        },
        {
            "id": "think_block_then_call",
            "output": TH + "reasoning" + _TH + fn("fetch", ("url", "https://x/y")),
            "expected": {"calls": [{"name": "fetch",
                                    "arguments": {"url": "https://x/y"}}],
                         "reasoning": ["reasoning"]},
            "must_report": [],
            "must_not": ["drop the call after the think block"],
            "note": "T29: think tags feed the reasoning parser; the call still parses.",
        },
        {
            "id": "two_calls_one_message",
            "output": fn("a", ("k", "1")) + fn("b", ("k", "2")),
            "expected": {"calls": [{"name": "a", "arguments": {"k": 1}},
                                   {"name": "b", "arguments": {"k": 2}}]},
            "must_report": [],
            "must_not": ["merge the calls", "drop the second call"],
            "note": "both calls in order.",
        },
    ]


def serialize(cases: list[dict]) -> str:
    """JSON with angle brackets escaped as \\u003c/\\u003e: the literal tags never
    appear even in the committed file text (T29 hygiene for readers)."""
    text = json.dumps({"cases": cases}, indent=1, sort_keys=True, ensure_ascii=True)
    return text.replace("<", r"\u003c").replace(">", r"\u003e") + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="harness/goldens/t29_parser_goldens.json")
    ap.add_argument("--check", action="store_true",
                    help="verify the committed golden matches the generator byte-for-byte")
    args = ap.parse_args(argv)
    want = serialize(build())
    if args.check:
        have = open(args.out, encoding="utf-8").read()
        if have == want:
            print("goldens match generator byte-for-byte")
            return 0
        print(f"GOLDEN MISMATCH: {args.out} differs from the generator")
        return 1
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(want)
    print(f"wrote {args.out} ({len(build())} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
