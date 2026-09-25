"""Codec pins — env-invariant (BOTH RUNS) semantics of the twin's quant codecs.

Complements the T1/T2 traps (test_oracle_t1_fused_split / test_oracle_t2_pad_trim):
these pin the CODEC constants the load path depends on — e4m3fn decode
(oracle/mimo26/quant/fp8_block.py:20-38), MXFP4 nibble order
(oracle/mimo26/quant/mxfp4.py:8, :96-98 — low nibble = even k, the T14 class)
and the OCP E8M0 reserved byte (mxfp4.py:54, :79 — byte 255 is never EMITTED).

Scope note (correction-of-record precision): the twin's
``scale_byte_to_float`` (mxfp4.py:44-47) does NOT clamp a stored 255 (2^128) —
255 is a reserved byte its encoders never produce; the decode-side clamp
(``np.minimum(b, 254) -> 2^127``) is the spike's T10 trap hardening
(spike/mxfp4.py:50).  These pins assert the TWIN contract: 255 never emitted.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from mimo26.quant import fp8_block as fb
from mimo26.quant import mxfp4 as mx


def test_e4m3_table_semantics_pin():
    """fp8_block.py:20-38 — S.1111.111 = NaN (both codes), bias 7, max 448."""
    assert fb.E4M3_MAX == 448.0
    assert np.isnan(fb.E4M3[0x7F]) and np.isnan(fb.E4M3[0xFF])
    assert fb.E4M3[0x00] == 0.0
    assert fb.E4M3[0x38] == 1.0  # s=0 e=7 m=0 -> (1+0) * 2^0
    assert fb.E4M3[0x3C] == 1.5  # s=0 e=7 m=4 -> (1+4/8) * 2^0
    assert fb.E4M3[0x01] == (1 / 8) * 2.0 ** -6  # subnormal


def test_e2m1_nibble_order_byte_exact():
    """T14 class (mxfp4.py:8, :96-98): low nibble = EVEN k, high nibble = odd k.

    Correction-of-record (23 Sep 2026 AEST, captain two-run fire): the first
    version built in-dim 2 with ``block=32`` and tripped the twin's OWN
    scale-shape guard (mxfp4.py:94-95 — ``ValueError: scale shape (1, 1) !=
    (1, 0)``) before any decode — a fixture-geometry bug of mine, NOT a
    nibble-order divergence.  Localized with np prints: twin codebook == golden
    ``e2m1_codebook_by_nibble`` 16/16, ``unpack(golden packed) == golden
    unpacked_f32`` byte-exact, gen-golden --check byte-matched.  Classification
    unchanged: BOTH RUNS pin (the twin's mxfp4 has no naive path).

    Byte 0x12, E8M0 byte 127 (2^0) at real BLOCK=32 geometry: low nibble 2 ->
    1.0 (even k), high nibble 1 -> 0.5 (odd k), tiled over the 32-nibble row."""
    packed = np.full((1, 16), 0x12, np.uint8)  # 32 nibbles = in-dim 32 = 1 block
    scales = np.full((1, 1), 127, np.uint8)  # 2^(127-127) = 1.0
    got = mx.unpack(packed, scales)  # block defaults to BLOCK=32 (mxfp4.py:88)
    want = np.tile(np.array([[1.0, 0.5]], np.float32), (1, 16))
    assert got.shape == (1, 32)
    assert got.tobytes() == want.tobytes(), (got[0, :4].tolist(), want[0, :4].tolist())
    with pytest.raises(ValueError, match="scale shape"):  # mxfp4.py:94-95 fail-loud
        mx.unpack(np.full((1, 1), 0x12, np.uint8), np.full((1, 1), 127, np.uint8),
                  block=32)  # the misflown shape — loud, not silent


def test_e8m0_reserved_255_never_emitted_pin():
    """mxfp4.py:54, :79 — OCP E8M0: 255 is reserved; the encoders clip at 254
    even for extreme magnitudes.  (Stored 255 handling is spike-side T10.)"""
    extreme = mx.scale_float_to_byte(np.array([1e300, 1e-300, 6.0, 0.0]))
    assert int(extreme.max()) <= 254 and int(extreme.min()) >= 0
    w = np.full((2, 64), 1e300)  # saturating amax -> still no 255 scale byte
    _packed, scales = mx.pack(w)
    assert int(scales.max()) <= 254


def test_mxfp4_pack_unpack_roundtrip_pin():
    """mxfp4.py:67-101 — pack->unpack is exactly decode(pack) (the codec's own
    dequant), finite, and shaped [out, in]."""
    rng = np.random.default_rng(11)
    w = (rng.standard_normal((3, 64)) * 2.0).astype(np.float32)
    packed, scales = mx.pack(w)
    assert packed.shape == (3, 32) and scales.shape == (3, 2)
    got = mx.unpack(packed, scales)
    again = mx.unpack(packed, scales)
    assert got.tobytes() == again.tobytes() and np.isfinite(got).all()
    assert float(np.abs(got - w).max()) <= 3.0  # E2M1*E8M0 quantization bound class


def test_e4m3_table_matches_golden(golden_dir):
    """External-oracle pin (AGENTS.md §4; I-Gold consume): the twin's 256-entry
    decode table equals the golden table entry-for-entry (null = NaN) — the same
    corpus `oracle/scripts/gen-golden.py --check` byte-pins."""
    tab = json.loads((golden_dir / "e4m3_decode_table.json").read_text())["table"]
    assert len(tab) == 256
    for b in range(256):
        f = fb.E4M3[b]
        want = None if np.isnan(f) else float(f)
        assert tab[b] == want, f"e4m3 byte {b:#04x}: golden {tab[b]} vs twin {want}"
