"""Selftests for harness/expert_fixture.py (I4 item 1/3 fixture).

Fast + synthetic: the real-block generation is a bench step; these pins cover the
sampling determinism, the bit-pattern format, and the two trap negatives the
fixture exists to catch (T14 swapped nibbles, T10 reserved-255 clamp).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.expert_fixture import block_fixture, sample_positions  # noqa: E402


def _synth(out: int = 8, inn: int = 128, seed: int = 3):
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 256, size=(out, inn // 2), dtype=np.uint8)
    s = rng.integers(0, 255, size=(out, inn // 32), dtype=np.uint8)
    return w, s


def test_sample_positions_deterministic_unique_in_range():
    a = sample_positions(8, 128, 64)
    b = sample_positions(8, 128, 64)
    assert a == b and len(set(a)) == 64
    assert all(0 <= r < 8 and 0 <= c < 128 for r, c in a)


def test_fixture_bits_match_reference_and_are_deterministic():
    from spike import mxfp4

    w, s = _synth()
    fx1 = block_fixture("t", w, s, n=64)
    fx2 = block_fixture("t", w, s, n=64)
    assert fx1 == fx2, "fixture must be byte-deterministic"
    ref = mxfp4.unpack(w, s, naive=False)
    for (r, c), hexbits in zip(fx1["positions"], fx1["expected_f32_bits"]):
        want = struct.unpack("<I", struct.pack("<f", float(ref[r, c])))[0]
        assert int(hexbits, 16) == want


def test_t14_swapped_nibbles_do_not_match_the_fixture():
    """The fixture must kill a swapped-nibble unpack (T14)."""
    from spike import mxfp4

    w, s = _synth()
    fx = block_fixture("t", w, s, n=64)
    naive = mxfp4.unpack(w, s, naive=True)
    mismatches = 0
    for (r, c), hexbits in zip(fx["positions"], fx["expected_f32_bits"]):
        got = struct.unpack("<I", struct.pack("<f", float(naive[r, c])))[0]
        mismatches += int(got != int(hexbits, 16))
    assert mismatches > 0, "swapped nibbles must differ at some sampled position"


def test_t10_reserved_255_scale_is_clamped_not_inf():
    """A real 255 scale byte must map to the clamped 2^127 product (T10), never
    an inf/2^128 poison — the fixture records the clamped bit pattern."""
    from spike import mxfp4

    w = np.zeros((1, 16), dtype=np.uint8)   # inn = 32 -> one scale block
    w[0, 0] = 0x02                      # low nibble 2 -> 1.0 (even k)
    s = np.array([[255]], dtype=np.uint8)
    fx = block_fixture("clamp", w, s, n=2)
    assert fx["scale_byte_255_count"] == 1
    ref = mxfp4.unpack(w, s, naive=False)
    assert np.isfinite(ref).all(), "clamped path must stay finite"
    assert float(ref[0, 0]) == float(np.float32(2.0 ** 127))
    for (r, c), hexbits in zip(fx["positions"], fx["expected_f32_bits"]):
        assert int(hexbits, 16) == struct.unpack(
            "<I", struct.pack("<f", float(ref[r, c])))[0]
