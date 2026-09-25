"""Golden pins for the P-103 codec path (external oracles, READ-ONLY):

- tests/golden/e4m3_decode_table.json -> spike.quant._e4m3_table (bitwise) AND
  the F4-fixed torch LUT construction in spike.real_loop (float32 table — the
  pre-fix code viewed float64 bytes as float32 = 512 junk entries).
- tests/golden/mxfp4_golden.json -> spike.mxfp4.unpack, BYTE-EXACT (codebook
  e2m1_codebook_by_nibble + T14 low-nibble=even-k nibble order).

SCOPE (P-106 task-6 close-out; correction-of-record): the golden's
scales_u8_hex contains ONLY the scale bytes {127, 128} (verified against the
fixture), so the T10 E8M0-255 clamp is NOT exercised by the byte-exact
unpack pin above — the golden exercises only {127, 128}. The 255-clamp
negatives (with a SYNTHETIC 255 scale byte; goldens stay READ-ONLY) and the
T14 nibble-order negative live in spike/tests/test_p101_addendum.py:
  test_t10_e8m0_clamp_255_negative
  test_t10_unpack_255_scale_byte_negative
  test_t14_nibble_order_negative
(claimed here earlier BEFORE those tests existed on disk — that claim was
false; the functions are grep-verified present as of this close-out).
This file is the BOTH-RUNS detection-power class (explicit naive=False).

Torch scoping (P-106 c3): test_e4m3_torch_lut_is_float32_256 executes only
where torch is importable (the coordinator env — importorskip on numpy-only
hosts). The torch-construction half of the F4 chain is enforced on EVERY
the coordinator/torch run by real_loop._check_e4m3_lut (RAISES on mismatch — at
import on the torch-built LUT, and at RealModel init on the DEVICE LUT).
The numpy table is pinned on BOTH runs here regardless.
"""
import json
from pathlib import Path

import numpy as np

from spike import mxfp4
from spike.quant import _e4m3_table

GOLDEN = Path(__file__).resolve().parents[2] / "oracle/goldens"


def test_e4m3_table_pinned_to_golden():
    g = json.loads((GOLDEN / "e4m3_decode_table.json").read_text())
    table = np.asarray(g if isinstance(g, list) else g.get("decode_table", g.get("table")),
                       dtype=np.float64)
    assert table.shape == (256,), f"golden table shape {table.shape}"
    ours = _e4m3_table().astype(np.float64)
    assert np.array_equal(ours.view(np.uint64), table.astype(np.float64).view(np.uint64)), \
        "e4m3 decode table diverges from golden (bitwise)"


def test_e4m3_torch_lut_is_float32_256():
    """Torch-scoped pin (P-106 c3). SCOPE: runs ONLY where torch imports
    (the coordinator env; the dev host test env skips it via importorskip — the numpy
    table half above runs on BOTH runs everywhere). The enforcing half for
    torch is real_loop._check_e4m3_lut, which RAISES on every the coordinator/torch
    run — at import on the torch-built LUT and at RealModel init on the
    device LUT — whenever either differs from the golden-pinned numpy
    table (F4 frombuffer-view regression class)."""
    torch = __import__("pytest").importorskip("torch")
    from spike import real_loop as R
    assert R.E4M3.dtype == torch.float32 and tuple(R.E4M3.shape) == (256,), \
        f"F4 class bug: LUT must be 256 f32 entries, got {tuple(R.E4M3.shape)} {R.E4M3.dtype}"
    g = json.loads((GOLDEN / "e4m3_decode_table.json").read_text())
    table = np.asarray(g if isinstance(g, list) else g.get("decode_table", g.get("table")),
                       dtype=np.float32)
    assert np.array_equal(R.E4M3.numpy().view(np.uint32), table.view(np.uint32)), \
        "torch LUT bitwise != golden e4m3_decode_table"


def test_mxfp4_codebook_pinned():
    g = json.loads((GOLDEN / "mxfp4_golden.json").read_text())
    want = [float(x) for x in g["e2m1_codebook_by_nibble"]]
    assert want == [float(x) for x in mxfp4.E2M1_CODEBOOK.tolist()], \
        "e2m1_codebook_by_nibble diverges from golden"


def test_mxfp4_unpack_byte_exact_golden():
    g = json.loads((GOLDEN / "mxfp4_golden.json").read_text())
    packed = np.frombuffer(bytes.fromhex(g["packed_u8_hex"]), dtype=np.uint8).reshape(*g["packed_shape"])
    scales = np.frombuffer(bytes.fromhex(g["scales_u8_hex"]), dtype=np.uint8).reshape(*g["scale_shape"])
    out, half = g["packed_shape"]
    want = np.frombuffer(bytes.fromhex(g["unpacked_f32_hex"]), dtype=np.float32).reshape(out, half * 2)
    got = mxfp4.unpack(packed, scales, naive=False)
    assert got.shape == want.shape
    assert got.view(np.uint32).tolist() == want.view(np.uint32).tolist(), \
        "mxfp4 unpack not byte-exact vs golden (nibble order T14 / clamp T10 / codebook)"
