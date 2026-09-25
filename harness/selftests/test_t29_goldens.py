"""Selftests for harness/t29_goldens.py (T29/T27 parser conformance seed).

Pins: (1) the committed golden matches the generator byte-for-byte; (2) both
T29 gate cases are present with their hazards actually embedded; (3) T29
hygiene — neither this file nor the generator contains literal tool-call tags
or chat special tokens (they are built from parts); (4) the committed JSON text
contains no literal angle-bracket tag either (escaped as JSON \\u003c).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness import t29_goldens as G  # noqa: E402

GOLDEN = ROOT / "harness" / "goldens" / "t29_parser_goldens.json"


def _forbidden_literals() -> list[str]:
    """The literals T29 forbids in source/arguments — built from parts here too."""
    lt, gt, slash, pipe = chr(60), chr(62), chr(47), chr(124)
    return [
        lt + "tool_call" + gt,
        lt + slash + "tool_call" + gt,
        lt + "function=",
        lt + slash + "function" + gt,
        lt + "parameter=",
        lt + slash + "parameter" + gt,
        lt + "think" + gt,
        lt + slash + "think" + gt,
        lt + pipe + "im_start" + pipe + gt,
        lt + pipe + "im_end" + pipe + gt,
    ]


def test_golden_matches_generator_byte_for_byte():
    want = G.serialize(G.build())
    assert GOLDEN.read_text(encoding="utf-8") == want


def test_t29_gate_cases_present_with_hazards_embedded():
    cases = {c["id"]: c for c in G.build()}
    closing_pa = G._PA
    eos = G.IM_END
    v1 = cases["value_contains_closing_parameter_tag"]["output"]
    assert closing_pa in v1, "gate case 1 must embed the closing parameter tag"
    v2 = cases["value_contains_chat_eos_special_token"]["output"]
    assert eos in v2, "gate case 2 must embed the chat-eos special token"
    assert cases["value_contains_closing_parameter_tag"]["must_not"]
    assert "truncate the value" in cases["value_contains_chat_eos_special_token"]["must_not"]


def test_expected_values_type_distinctness_t27():
    """T27 coercion pin: int 5 and str 5 are distinct in the golden."""
    cases = {c["id"]: c for c in G.build()}
    args = cases["non_string_renders_with_tojson"]["expected"]["calls"][0]["arguments"]
    assert args["n"] == 5 and isinstance(args["n"], int)
    assert args["s"] == "5" and isinstance(args["s"], str)


def test_no_literal_tags_in_sources():
    """T29 hygiene: the literals never appear in the generator or this test."""
    for path in (ROOT / "harness" / "t29_goldens.py",
                 ROOT / "harness" / "selftests" / "test_t29_goldens.py"):
        text = path.read_text(encoding="utf-8")
        for lit in _forbidden_literals():
            assert lit not in text, f"{path.name} contains a literal special tag"


def test_committed_golden_escapes_angle_brackets():
    """Even the committed corpus text has no literal tags (JSON \\u003c escapes)."""
    text = GOLDEN.read_text(encoding="utf-8")
    for lit in _forbidden_literals():
        assert lit not in text, "committed golden contains a literal special tag"
    # ...but the decoded values do carry them (the corpus is usable).
    doc = json.loads(text)
    joined = json.dumps(doc)
    assert any(G._PA in c["output"] or G.IM_END in c["output"] for c in doc["cases"])
    assert json.loads(joined)["cases"]
