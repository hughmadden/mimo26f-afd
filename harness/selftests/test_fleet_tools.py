"""R6 selftests for the vendored fleet probes (harness/fleet/tonyd2wild/, REUSE rows
23 Sep 2026 AEST, upstream tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe @ 13621bb).

These pin the verdict logic of gate G8 (corruption under concurrency, trap T26) and
of the tool-storm probe (T24) before either tool is pointed at a model: a known-good
PALETTE must score clean, an injected foreign-script line must score hard-bad, and the
tool-call checker must flag empty/unparseable/nameless calls. Offline, no network.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

FLEET = pathlib.Path(__file__).resolve().parents[1] / "fleet" / "tonyd2wild"
PINNED = {  # sha256 at copy time; a changed file must come with a new REUSE row
    "stress-corrupt.py": "d375f144916a5438bcfa544386fb7b49439aff8b596888bfe39c98777b59b319",
    "replay_exact.py": "f5f9035b948f2b079ef7e89357d4bbcf42a43b55058862900fc54d31e0a5dbdd",
    "mimobench.py": "9c2537b171de25f764a9383e6752d47e82e5cff59891e93641ba5e187c092c4c",
    "mimo_needle.py": "5281a81d15b26eda66f9df1e33d8ea5cb01ed64cc56586d23184ffeb5059b41b",
    "toolcap-proxy.cjs": "8b621ed934832f0796f7de703069fd8d323133b4a41ef93e04a08c28375d4cff",
}


def _load_stress():
    spec = importlib.util.spec_from_file_location("stress_corrupt", FLEET / "stress-corrupt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # import-safe: main() is behind __name__ == "__main__"
    return mod


def _palette(lines):
    return "const PALETTE = {\n" + "\n".join(lines) + "\n};"


def _good_lines():
    return [f"  c{i:03d}: 0x{(i * 2654435761) & 0xFFFFFF:06x}," for i in range(300)]


@pytest.mark.parametrize("name", sorted(PINNED))
def test_vendored_tool_is_byte_pinned(name):
    import hashlib
    got = hashlib.sha256((FLEET / name).read_bytes()).hexdigest()
    assert got == PINNED[name], f"{name} drifted from its REUSE row pin"


def test_g8_clean_palette_scores_clean():
    a = _load_stress().analyze_long(_palette(_good_lines()))
    assert a["good_lines"] == 300
    assert a["hard_bad"] == 0 and a["bad_lines"] == 0
    assert a["missing_entries"] == 0 and a["out_of_order"] == 0
    assert a["non_ascii"] == 0 and a["special_tok"] == 0


def test_g8_injected_foreign_script_is_hard_bad():
    lines = _good_lines()
    lines[217] = "  c217: 0x3fабc1,"  # Cyrillic letters inside the hex, the vllm#46669 signature
    a = _load_stress().analyze_long(_palette(lines))
    assert a["hard_bad"] >= 1, "detector missed injected foreign-script characters"
    assert a["non_ascii"] >= 2


def test_g8_relaxed_style_is_bad_but_not_hard_bad():
    lines = _good_lines()
    lines[5] = "  'c005': '0x1A2B3C',"  # style deviation, not corruption
    a = _load_stress().analyze_long(_palette(lines))
    assert a["bad_lines"] == 1 and a["hard_bad"] == 0


def test_g8_missing_and_reordered_entries_are_counted():
    lines = _good_lines()
    del lines[100]
    lines[10], lines[11] = lines[11], lines[10]
    a = _load_stress().analyze_long(_palette(lines))
    assert a["missing_entries"] == 1
    assert a["out_of_order"] >= 2


def test_t24_tool_checker_flags_malformed_calls():
    check = _load_stress().check_tool_calls
    good = {"name": "write", "arguments": json.dumps({"file_path": "a", "content": "b"})}
    n, bad, details = check([good,
                             {"name": "grep", "arguments": ""},
                             {"name": "read", "arguments": "{not json"},
                             {"name": None, "arguments": json.dumps({"x": 1})},
                             {"name": "todo_write", "arguments": "{}"}])
    assert n == 5 and bad == 4
    assert [d["problem"] for d in details] == [None, "empty", "unparseable", "+noname", "empty-object"]
