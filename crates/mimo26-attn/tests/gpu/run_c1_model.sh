#!/usr/bin/env bash
# CPU-only draft validation. Does not build or select a CUDA C1 implementation.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CRATE="$REPO/crates/mimo26-attn"
root="${MIMO26F_BUILD_ROOT:-$REPO/target/mimo26f-builds}"
case "$(realpath -m "$root")" in /mnt/scratch|/mnt/scratch/*) exit 3;; esac
mkdir -p "$root"
slot=$(mktemp -d "$root/attn-c1-model.XXXXXX")
export TZ=Australia/Sydney
exec 8>"$root/.cargo.lock"; flock 8
exec 9>"$root/.attn-cuda.lock"; flock 9
set +e
(
  set -e
  date '+%F %T %Z'
  echo 'COMMAND scripts/dev.sh test attn-bench c1-selftest'
  printf 'SOURCE '; git -C "$REPO" rev-parse HEAD
  git -C "$REPO" status --short -- crates/mimo26-attn
  sha256sum "$CRATE/kernels/include/decode_c1_model_storage.h" "$CRATE/tests/gpu/c1_storage_selftest.cpp" "$CRATE/tests/gpu/c1_pipeline_model.py"
  g++ --version
  g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"$CRATE/kernels/include" "$CRATE/tests/gpu/c1_storage_selftest.cpp" -o "$slot/storage-selftest"
  "$slot/storage-selftest"
  python3 "$CRATE/tests/gpu/c1_pipeline_model.py"
  echo 'RESULT: PASS draft CPU preflight; C3 CUDA path unchanged'
) 2>&1 | tee "$slot/receipt.log"
codes=("${PIPESTATUS[@]}")
rc=${codes[0]}; if ((rc==0)); then rc=${codes[1]}; fi
set -e
receipt="$REPO/runs/20260923-i4/attn/$(basename "$slot")"
mkdir -p "$receipt"
cp "$slot/receipt.log" "$receipt/RESULT.md"
echo "C1_MODEL_RECEIPT $receipt/RESULT.md"
exit "$rc"
