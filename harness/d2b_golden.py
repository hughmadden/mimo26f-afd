#!/usr/bin/env python3
"""harness/d2b_golden.py — D2b tools-serialisation golden SEED.

Renders the checkpoint chat template (the file on the coordinator at
`/srv/models/XiaomiMiMo/MiMo-V2.6-Flash-RL/chat_template.jinja`) with Jinja2 and
the transformers `tojson` filter (json.dumps, ensure_ascii=False, sort_keys=False,
default ", " / ": " separators), for the four D2b cases, and writes each rendered
byte-string HEX-ENCODED to the fixture file the Rust test reads.

Hex encoding (not the rendered text) is committed so no MiMo tool/chat tag
literal enters the repo (T29): the rendered output contains the template's tags,
which this generator emits only as hex bytes.

Cases (builder attn-d2b.md):
  t24            — the T24 request: four tools with descriptions, a prior read
                   call and its tool result (stress-corrupt.py TOOLS/TOOL_MESSAGES);
  system_tools   — a client system message plus the same tools (the tools turn
                   must come BEFORE the client's system turn);
  nonstring_arg  — a history call whose arguments carry non-string values
                   (int, bool, array, nested object), rendered via tojson in the
                   client's JSON order;
  no_tools       — no tools (must be unchanged from the no-tools render).

Usage:
  python3 harness/d2b_golden.py TEMPLATE_PATH OUTFILE
"""
from __future__ import annotations

import hashlib
import json
import sys

import jinja2


def tojson(obj, ensure_ascii=False):
    return json.dumps(obj, ensure_ascii=ensure_ascii)


TOOLS = [
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search file contents for a regex pattern.",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                       "required": ["pattern", "path"]}}},
    {"type": "function", "function": {
        "name": "read",
        "description": "Read a file and return its contents with line numbers.",
        "parameters": {"type": "object",
                       "properties": {"file_path": {"type": "string"}},
                       "required": ["file_path"]}}},
    {"type": "function", "function": {
        "name": "write",
        "description": "Write content to a file, replacing it entirely.",
        "parameters": {"type": "object",
                       "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["file_path", "content"]}}},
    {"type": "function", "function": {
        "name": "todo_write",
        "description": "Replace the todo list.",
        "parameters": {"type": "object",
                       "properties": {"todos": {"type": "array", "items": {"type": "object",
                           "properties": {"content": {"type": "string"},
                                          "status": {"type": "string",
                                                     "enum": ["pending", "in_progress", "completed"]}},
                           "required": ["content", "status"]}}},
                       "required": ["todos"]}}},
]

T24_MESSAGES = [
    {"role": "system",
     "content": "You are a coding assistant working in the repository /work/app. Use the provided tools to make changes."},
    {"role": "user",
     "content": "There is a typo on line 218 of src/theme/palette.js, the gold entry is broken. Please fix it."},
    {"role": "assistant", "content": "",
     "tool_calls": [{"id": "call_read_1", "type": "function",
                     "function": {"name": "read",
                                  # vLLM json.loads the arguments string before
                                  # templating, so this is the parsed DICT.
                                  "arguments": {"file_path": "src/theme/palette.js"}}}]},
    {"role": "tool", "tool_call_id": "call_read_1",
     "content": "216:   amber: 0xf2a900,\n217:   honey: 0xe8b923,\n218:   gold: 0xf6c council,\n219:   sand: 0xd9c19c,\n220:   ochre: 0xcc7722,\n"},
]

NONSTRING_MESSAGES = [
    {"role": "assistant", "content": "",
     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "grep",
                                  "arguments": {"pattern": "foo", "max_count": 3,
                                                "case_sensitive": False,
                                                "options": {"multiline": True,
                                                            "invert": False}}}}]},
]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    template_path, outfile = sys.argv[1], sys.argv[2]
    with open(template_path) as f:
        template_text = f.read()
    env = jinja2.Environment()
    env.filters["tojson"] = tojson
    tpl = env.from_string(template_text)

    cases = {
        "t24": (TOOLS, T24_MESSAGES),
        "system_tools": (TOOLS, [{"role": "system", "content": "You are a helpful assistant."}]),
        "nonstring_arg": ([], NONSTRING_MESSAGES),
        "no_tools": ([], [{"role": "user", "content": "Hello"}]),
    }
    out = {}
    for name, (tools, messages) in cases.items():
        text = tpl.render(messages=messages, tools=tools,
                          add_generation_prompt=True, enable_thinking=False)
        out[name] = text.encode("utf-8").hex()
        print(f"{name}: {len(text)} chars, sha256={hashlib.sha256(text.encode()).hexdigest()[:16]}")

    with open(outfile, "w") as f:
        for name, hx in out.items():
            f.write(f"{name} {hx}\n")
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main()
