"""Selftests for harness/nibble_proof.py (I4 item 1 comparator).

Synthetic dumps only — the real proof runs on a Spark window.  These pins make
sure the comparator actually bites: a correct dump passes, and a swapped-nibble
(T14), wrong-bytes (sha) or missing-block dump fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.nibble_proof import compare  # noqa: E402


def _fixture():
    return {
        "blocks": [
            {
                "name": "model.layers.1.mlp.experts.0.gate_proj",
                "weight_sha256": "aa" * 32,
                "positions": [[0, 0], [0, 1], [1, 5]],
                "expected_f32_bits": ["3f800000", "bf800000", "00000000"],
            }
        ]
    }


def _dump(bits, sha="aa" * 32, name="model.layers.1.mlp.experts.0.gate_proj"):
    return {"blocks": [{"name": name, "weight_sha256": sha, "bits": bits}]}


def test_correct_dump_passes():
    ok, lines = compare(_fixture(), _dump(["3f800000", "bf800000", "00000000"]))
    assert ok, lines
    assert any(line.startswith("PASS") for line in lines)


def test_swapped_nibble_dump_fails_and_names_the_position():
    ok, lines = compare(_fixture(), _dump(["3f800000", "3f800000", "00000000"]))
    assert not ok
    assert any("row 0, col 1" in line for line in lines), lines


def test_wrong_bytes_fail_on_sha():
    ok, lines = compare(_fixture(), _dump(["3f800000", "bf800000", "00000000"], sha="bb" * 32))
    assert not ok
    assert any("sha256 differs" in line for line in lines), lines


def test_missing_block_fails():
    ok, lines = compare(_fixture(), {"blocks": []})
    assert not ok
    assert any("missing from the kernel dump" in line for line in lines), lines


def test_bit_count_mismatch_fails():
    ok, lines = compare(_fixture(), _dump(["3f800000"]))
    assert not ok
    assert any("sampled bits" in line for line in lines), lines
