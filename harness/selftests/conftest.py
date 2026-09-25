"""harness/selftests — R6 offline selftests of the harness layer (TEST-PLAN R6:
"harness selftests before harness runs a model"; S0: "harness selftests exist").

This first tranche is the foundation class (I2 close-out): golden-corpus pins,
the oracle import seam, codec micro-smoke.  The parser/timeout/abort/known-answer
selftests (TEST-PLAN :75 row) land with the harness implementation itself.

Missing goldens = loud fail unless ``MIMO26_ALLOW_MISSING_GOLDEN=1`` (mirror of
spike/tests/conftest.py:8-9 — silence is not green).
"""
from __future__ import annotations

import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]  # mimo26f-afd/
ORACLE = REPO / "oracle"
MIMO26 = ORACLE / "mimo26"
sys.path.insert(0, str(ORACLE))  # `import mimo26` -> oracle/mimo26 (the twin)

GOLDEN_DIR = pathlib.Path(os.environ.get(
    "MIMO26_GOLDEN_DIR",
    str(pathlib.Path(__file__).resolve().parents[2] / "oracle/goldens"),
))

# sha256 --check pins of the consumed golden corpus (I-Gold read-only; digests
# measured 23 Sep 2026 AEST from code/tests/golden/).
GOLDEN_SHA256 = {
    "e4m3_decode_table.json":
        "12ebc31053685c442e9b3f9ccdaf88b67a08d5526616b522dd682334817e5c4c",
    "fp8_block_golden.json":
        "d8f45ad930120c29820fe1b63aa56b05624564dfb6325459c3aab52df1abc9c6",
    "mxfp4_golden.json":
        "1e63c575e6315e884794b29940c75d9f305aba39a90246963e1f9d4a06c826fe",
}

# P-201 byte-verbatim import manifest (runs/20260923-i2/P-201-import.md).
MANIFEST = ORACLE / "SHA256SUMS.import"
