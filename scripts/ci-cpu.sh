#!/usr/bin/env bash
# mimo26f-afd CPU merge gate. Fail loud if a suite is missing — silence is not green.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Toolchain on PATH for non-login shells (I2 P-203: ~/.cargo/bin/cargo existed
# but `cargo: command not found`). Idempotent with configs/build.env (local).
export PATH="${HOME}/.cargo/bin:${PATH}"
fail=0
note() { printf '%s\n' "$*"; }

note "=== mimo26f-afd ci-cpu ==="
note "root: $ROOT"
note "time: $(TZ=Australia/Sydney date '+%Y-%m-%d %H:%M:%S AEST')"

# Untracked-files check (I2.9, ADVISOR-I3 §4): a tree with untracked source
# under the code dirs cannot be reproduced from a clone of origin — "12/12"
# on a dirty tree is not a merge gate. Fail loud and name the files.
note "--- untracked files (crates oracle harness spike scripts) ---"
if git rev-parse --git-dir >/dev/null 2>&1; then
  untracked="$(git ls-files --others --exclude-standard -- crates oracle harness spike scripts)"
  if [[ -n "$untracked" ]]; then
    note "UNTRACKED (commit by path or ignore):"
    note "$untracked"
    fail=1
  else
    note "none"
  fi
else
  note "NOT A GIT CHECKOUT — the gate runs from a clone of origin. Fail loud."
  fail=1
fi

# Rust workspace
if [[ -f rust/Cargo.toml || -f Cargo.toml ]]; then
  note "--- cargo test (workspace) ---"
  if [[ -f rust/Cargo.toml ]]; then
    (cd rust && cargo test --workspace --no-fail-fast) || fail=1
  else
    cargo test --workspace --no-fail-fast || fail=1
  fi
else
  note "MISSING: Rust workspace (expected rust/Cargo.toml at I1+) — fail loud"
  fail=1
fi

# Oracle / CPU twin
if [[ -d oracle/tests ]]; then
  note "--- pytest oracle/tests ---"
  python3 -m pytest oracle/tests -q || fail=1
  if [[ -f oracle/scripts/gen-golden.py ]]; then
    note "--- gen-golden.py --check ---"
    python3 oracle/scripts/gen-golden.py --check || fail=1
  else
    note "MISSING: oracle/scripts/gen-golden.py --check — fail loud"
    fail=1
  fi
else
  note "MISSING: oracle/tests (import at I1) — fail loud"
  fail=1
fi

# Harness selftests (R6)
if [[ -d harness/selftests ]] && compgen -G 'harness/selftests/test_*.py' >/dev/null; then
  note "--- harness selftests ---"
  python3 -m pytest harness/selftests -q || fail=1
else
  note "MISSING: harness/selftests/test_*.py — fail loud"
  fail=1
fi

if [[ "$fail" -ne 0 ]]; then
  note "RESULT: FAIL (missing suite or red tests)"
  exit 1
fi
note "RESULT: PASS"
