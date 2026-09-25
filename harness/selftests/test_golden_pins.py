"""Golden-corpus pins — `sha256sum --check`-style verification of the two
consumed read-only oracles (I-Gold): the external golden corpus and the P-201
import manifest (runs/20260923-i2/P-201-import.md).

Two-run classification: env-invariant (BOTH RUNS green).  Policy: missing goldens
= loud fail unless ``MIMO26_ALLOW_MISSING_GOLDEN=1`` (spike/tests/conftest.py:8-9).
"""
from __future__ import annotations

import hashlib
import os

import pytest

from .conftest import GOLDEN_DIR, GOLDEN_SHA256, MANIFEST, MIMO26


def _allow_missing() -> bool:
    return os.environ.get("MIMO26_ALLOW_MISSING_GOLDEN") == "1"


def test_golden_corpus_present_fail_loud():
    """Fail loud on a missing corpus (mirror of the spike missing-golden policy;
    TEST-PLAN R1: 'silence is not green')."""
    missing = [n for n in GOLDEN_SHA256 if not (GOLDEN_DIR / n).is_file()]
    if missing and _allow_missing():
        pytest.skip(f"golden corpus missing {missing} at {GOLDEN_DIR} "
                    "(MIMO26_ALLOW_MISSING_GOLDEN=1)")
    assert not missing, f"MISSING golden corpus files {missing} under {GOLDEN_DIR}"


def test_golden_sha256_check_pins():
    """--check pins: each corpus file's sha256 equals the pinned digest — a
    silent corpus swap or drift fails here before any consumer notices."""
    mismatches = []
    for name, want in GOLDEN_SHA256.items():
        p = GOLDEN_DIR / name
        if not p.is_file():
            if _allow_missing():
                pytest.skip(f"golden missing at {GOLDEN_DIR} "
                            "(MIMO26_ALLOW_MISSING_GOLDEN=1)")
            mismatches.append(f"{name}: MISSING")
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            mismatches.append(f"{name}: {got} != pinned {want}")
    assert not mismatches, "GOLDEN DRIFT: " + "; ".join(mismatches)


def test_oracle_import_manifest_pins():
    """oracle/SHA256SUMS.import (P-201) verified sha256sum --check-style against
    oracle/mimo26/ — the byte-verbatim twin pins stay live, not one-shot."""
    assert MANIFEST.is_file(), f"missing {MANIFEST}"
    rows = [ln.split(None, 1) for ln in MANIFEST.read_text().splitlines() if ln.strip()]
    assert len(rows) == 12, f"manifest rows: {len(rows)} (P-201: 12 files)"
    bad = []
    for digest, rel in rows:
        p = MIMO26 / rel.strip()
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"
        if got != digest:
            bad.append(f"{rel.strip()}: {got} != {digest}")
    assert not bad, "TWIN DRIFT: " + "; ".join(bad)
