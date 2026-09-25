#!/usr/bin/env bash
# CPU-only oracle cells; parent dev.sh dispatcher holds the common build lock.
set -euo pipefail
REPO="${1:?repo}"; STAGING="${2:?slot}"; CELL="${3:?cell}"; WEIGHTS="${4:?weights}"
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$REPO/oracle:$REPO${PYTHONPATH:+:$PYTHONPATH}"
python3 "$REPO/crates/mimo26-repack/tests/wire_oracle.py" --selftest | tee "$STAGING/reader.log"
python3 -m pytest -q "$REPO/oracle/tests/test_lattice_quant_v1.py" -k 'not torch' | tee "$STAGING/codec.log"
python3 "$REPO/crates/mimo26-repack/tests/lattice_oracle.py" --selftest | tee "$STAGING/lattice-selftest.log"
python3 "$REPO/crates/mimo26-repack/tests/lattice_compare.py" --selftest | tee "$STAGING/comparator-selftest.log"
if [[ "$CELL" == lattice-compare ]]; then
  python3 "$REPO/crates/mimo26-repack/tests/lattice_compare.py" --compare "${MIMO26_B1_REFERENCE:?reference artifacts required}" "${MIMO26_B1_CANDIDATE:?candidate artifacts required}" | tee "$STAGING/comparison.json"
fi
if [[ "$CELL" == lattice-oracle-selftest && -n "${MIMO26_B1_REFERENCE:-}" ]]; then
  python3 "$REPO/crates/mimo26-repack/tests/lattice_compare.py" --reference-control "$MIMO26_B1_REFERENCE" | tee "$STAGING/reference-control.json"
fi
if [[ "$CELL" == lattice-oracle-real ]]; then
  python3 "$REPO/crates/mimo26-repack/tests/lattice_oracle.py" --real "$WEIGHTS" "$STAGING/oracle" | tee "$STAGING/lattice-real.log"
fi
echo 'RESULT: PASS CPU E-W4A8-v1 oracle; no CUDA kernel, transport or performance qualification'
