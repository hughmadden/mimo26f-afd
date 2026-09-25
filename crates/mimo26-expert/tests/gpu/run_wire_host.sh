#!/usr/bin/env bash
# CPU build/probe helper. Caller holds the dev.sh build-root cargo lock.
set -euo pipefail
REPO="${1:?repo}"; STAGING="${2:?unique local slot}"
export CARGO_TARGET_DIR="$STAGING/wire-target" CUDA_VISIBLE_DEVICES=''
export PATH="$HOME/.cargo/bin:$PATH"
python3 "$REPO/crates/mimo26-repack/tests/wire_oracle.py" --selftest | tee "$STAGING/wire-oracle-selftest.log"
cargo build --locked --manifest-path "$REPO/Cargo.toml" -p mimo26-expert -p mimo26-wire --lib 2>&1 | tee "$STAGING/build.log"
cargo test --locked --manifest-path "$REPO/Cargo.toml" -p mimo26-expert --lib \
  --test cuda_contract --test aot_gates --test bandwidth 2>&1 | tee "$STAGING/expert-unit.log"
rustc --edition=2021 "$REPO/crates/mimo26-expert/tests/wire/seam.rs" \
  --extern "mimo26_expert=$CARGO_TARGET_DIR/debug/libmimo26_expert.rlib" \
  --extern "mimo26_wire=$CARGO_TARGET_DIR/debug/libmimo26_wire.rlib" \
  -L "dependency=$CARGO_TARGET_DIR/debug/deps" -o "$STAGING/wire-seam"
"$STAGING/wire-seam" --selftest | tee "$STAGING/seam-selftest.log"
set +e
"$STAGING/wire-seam" --legacy-bound >"$STAGING/legacy-bound-contract.log" 2>&1
rc=$?
set -e
[[ "$rc" == 3 ]] && grep -q '^WIRE_BOUND_VIOLATION:' "$STAGING/legacy-bound-contract.log" || { echo 'FAIL: legacy BF16-bound counterexample missing'; exit 1; }
"$STAGING/wire-seam" --strict-bound | tee "$STAGING/r8-bound-contract.log"
python3 "$REPO/crates/mimo26-repack/tests/wire_oracle.py" --synthetic "$STAGING/synthetic-wire"
"$STAGING/wire-seam" --real "$STAGING/synthetic-wire" "$STAGING/synthetic-wire" 0 | tee "$STAGING/synthetic-wire.log"
for flag in 1 2; do
  set +e
  "$STAGING/wire-seam" --real "$STAGING/synthetic-wire" "$STAGING/synthetic-wire" "$flag" >"$STAGING/synthetic-negative-$flag.log" 2>&1
  rc=$?
  set -e
  [[ "$rc" == 3 ]] && grep -q '^ORACLE_MISMATCH ' "$STAGING/synthetic-negative-$flag.log" || { echo 'FAIL: synthetic corruption/weight detector'; exit 1; }
done
echo 'R8 CPU PASS: corrected bound, ordered buffering and synthetic artifact probe; real GPU gate separate'
