#!/usr/bin/env bash
# Local parity or guarded owner-run coordinator parity. CPU build/audit cells too.
# ONLY through scripts/dev.sh test attn [cell]. No service/container mutations.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CRATE="$REPO/crates/mimo26-attn"
source "$CRATE/tests/gpu/gpu_guard.sh"
ARCH="${MIMO26_ATTN_ARCH:-sm_89}"
SHAPES="${MIMO26_ATTN_SHAPES:-tiny}"
REMOTE="${HOST:-local}"
case "${MIMO26_ATTN_PARITY_FAMILY:-c3}" in c3|c1|p1) ;; *) echo 'RESULT: REFUSE unknown parity family'; exit 3;; esac
oracle_extra=(); tc_rows=16
if [[ "${MIMO26_ATTN_PARITY_FAMILY:-c3}" == p1 ]]; then oracle_extra=(--p1); tc_rows=20; fi
case "$REMOTE" in local|coordinator) ;; *) echo 'RESULT: REFUSE unsupported parity host'; exit 3;; esac
if [[ "$REMOTE" == coordinator && ( "$SHAPES" == tc-decode || "$SHAPES" == tc-highv || "$SHAPES" == tc-bf16q || "$SHAPES" == tc-bf16q-highv ) ]]; then
  [[ "${MIMO26_ATTN_BUILDER_WINDOW:-0}" == 1 ]] || { echo 'RESULT: REFUSE remote parity acknowledgement missing'; exit 3; }
  ARCH=sm_120
fi
export TZ=Australia/Sydney
case "$SHAPES" in
  --help|help|-h)
    printf '%s\n' 'scripts/dev.sh test attn [tiny|tc-decode|tc-highv|--all|shape-list|selftest|gpu-check|audit-tc|goldens-tc|goldens-highv|build-sm89|build-sm120]' \
      'selftest/build-* do not query or run a GPU; build-* use static cudart.' \
      'HOST=coordinator MIMO26_ATTN_BUILDER_WINDOW=1 tc-decode checks baseline, pipe4 and pipe8 with negatives.' \
      'Local: MIMO26_ATTN_TWO_RUN=1 checks an explicit numerical naive failure.' \
      'Receipts and goldens persist in the workspace build slot. Allocations leave >=4 GiB free.'
    exit 0 ;;
  build-sm89) ARCH=sm_89 ;;
  build-sm120) ARCH=sm_120 ;;
esac
BUILD_ROOT="${MIMO26F_BUILD_ROOT:-$REPO/target/mimo26f-builds}"
case "$(realpath -m "$BUILD_ROOT")" in /mnt/scratch|/mnt/scratch/*)
  echo 'RESULT: REFUSE forbidden build root' >&2; exit 3;; esac
mkdir -p "$BUILD_ROOT"
STAGING="$(mktemp -d "$BUILD_ROOT/attn-parity.XXXXXX")"
exec > >(tee "$STAGING/receipt.log") 2>&1
printf 'TIME %s\nCOMMAND scripts/dev.sh test attn %s\nSLOT %s\n' "$(date '+%F %T %Z')" "$SHAPES" "$STAGING"
SOURCE="$(git -C "$REPO" rev-parse HEAD)"
printf 'SOURCE %s\n' "$SOURCE"
printf '%s\n' "$SOURCE" > "$STAGING/source.txt"
git -C "$REPO" status --short -- crates/mimo26-attn
# Common dev-host build/GPU lock also used by expert-unit/gemm, then attention lock.
exec 8>"$BUILD_ROOT/.cargo.lock"
flock 8
exec 9>"$BUILD_ROOT/.attn-cuda.lock"
flock 9
c++ -O2 -std=c++17 -Wall -Wextra -Werror "$CRATE/tests/gpu/parity_selftest.cpp" -o "$STAGING/selftest"
"$STAGING/selftest"
c++ -O2 -std=c++17 -Wall -Wextra -Werror "$CRATE/tests/gpu/decode_tc_selftest.cpp" -o "$STAGING/tc-selftest"
"$STAGING/tc-selftest"
python3 "$CRATE/tests/gpu/test_gpu_guard.py"
python3 "$CRATE/tests/gpu/test_tc_highv.py"
python3 "$CRATE/tests/gpu/test_tc_bf16q.py"
python3 "$CRATE/tests/gpu/test_p1_fixtures.py"
c++ -O2 -std=c++17 -Wall -Wextra -Werror "$CRATE/tests/gpu/p1_probe_selftest.cpp" -o "$STAGING/p1-probe-selftest"
"$STAGING/p1-probe-selftest"
bash "$CRATE/tests/gpu/run_remote_parity.sh" --selftest
if [[ "$SHAPES" == gpu-check ]]; then
  [[ "$REMOTE" == local ]] || { echo 'RESULT: REFUSE gpu-check is local-only'; exit 3; }
  nvidia-smi
  m26_gpu_guard
  exit 0
fi
if [[ "$SHAPES" == selftest ]]; then exit 0; fi
if [[ "$SHAPES" == shape-list ]]; then
  python3 "$CRATE/tests/oracle_driver.py" gen --out "$STAGING" --shapes=--all --list-only
  echo 'RESULT: PASS --all argument validation only (no tensors/GPU)'; exit 0
fi
if [[ "$SHAPES" == audit-tc ]]; then
  python3 "$CRATE/tests/oracle_driver.py" audit-tc "${oracle_extra[@]}"
  exit 0
fi
if [[ "$SHAPES" == goldens-tc || "$SHAPES" == goldens-highv || "$SHAPES" == goldens-bf16q || "$SHAPES" == goldens-bf16q-highv ]]; then
  cohort=tc-decode
  [[ "$SHAPES" != goldens-highv ]] || cohort=tc-highv
  [[ "$SHAPES" != goldens-bf16q ]] || cohort=tc-bf16q
  [[ "$SHAPES" != goldens-bf16q-highv ]] || cohort=tc-bf16q-highv
  python3 "$CRATE/tests/oracle_driver.py" gen --out "$STAGING" --shapes "$cohort" "${oracle_extra[@]}"
  echo 'RESULT: PASS CPU oracle generation only (NOT GPU parity)'; exit 0
fi

if [[ "$SHAPES" != build-* ]]; then
  if [[ "$REMOTE" == coordinator ]]; then
    [[ ( "$SHAPES" == tc-decode || "$SHAPES" == tc-highv || "$SHAPES" == tc-bf16q || "$SHAPES" == tc-bf16q-highv ) ]] || { echo 'RESULT: REFUSE remote parity supports tc-decode only'; exit 3; }
  else
  nvidia-smi
  m26_gpu_guard
  if [[ "${MIMO26_ATTN_ALLOW_GPU:-0}" != 1 ]]; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
    case "$GPU_NAME" in *4090*) ;;
      *) echo "RESULT: REFUSE GPU '$GPU_NAME' (want RTX 4090; builder window override required)" >&2; exit 3 ;;
    esac
  fi
  fi
  echo "== goldens: shapes=$SHAPES (oracle/mimo26, I-Gold read-only)"
  python3 "$CRATE/tests/oracle_driver.py" gen --out "$STAGING" --shapes="$SHAPES" "${oracle_extra[@]}"
fi

echo "== build: nvcc -arch=$ARCH (handwritten kernels, correctness-first)"
nvcc -O2 -std=c++17 -cudart static -Xptxas=-v -lineinfo -arch="$ARCH" -DM26_PARITY_ARCH="${ARCH#sm_}" \
  -I"$CRATE/kernels/include" \
  "$CRATE/kernels/attn_decode_splitkv.cu" \
  "$CRATE/kernels/attn_decode_tc.cu" \
  "$CRATE/kernels/attn_reduce.cu" \
  "$CRATE/kernels/attn_prefill_chunk.cu" \
  "$CRATE/kernels/rope.cu" \
  "$CRATE/kernels/kv_cache_fp8.cu" \
  "$CRATE/kernels/parity/attn_parity.cu" \
  -o "$STAGING/mimo26f-attn-parity"
printf 'BINARY %s\n' "$STAGING/mimo26f-attn-parity"
sha256sum "$STAGING/mimo26f-attn-parity"
if [[ "$SHAPES" == build-* ]]; then echo 'RESULT: PASS parity build only (NOT hardware parity)'; exit 0; fi

if [[ "$REMOTE" == coordinator ]]; then
  exec bash "$CRATE/tests/gpu/run_remote_parity.sh" "$STAGING" "$SHAPES"
fi

echo '== run: correct pass (must be green)'
nvidia-smi
m26_gpu_guard
MIMO26_SPIKE_NAIVE=0 M26_NAIVE_FLAGS=0 "$STAGING/mimo26f-attn-parity" "$STAGING" | tee "$STAGING/pass-correct.log"
grep -q '^RESULT: PASS$' "$STAGING/pass-correct.log"
correct_count="$(grep -c '^case ' "$STAGING/pass-correct.log")"
if [[ ( "$SHAPES" == tc-decode || "$SHAPES" == tc-highv || "$SHAPES" == tc-bf16q || "$SHAPES" == tc-bf16q-highv ) ]]; then
  [[ "$correct_count" == "$tc_rows" ]] || { echo 'RESULT: FAIL incomplete tc-decode corpus'; exit 4; }
fi
if [[ "${MIMO26_ATTN_DECODE_IMPL:-baseline}" != baseline ]]; then
  grep -Eq '^TC_COVERAGE launches=[1-9][0-9]* requested=1$' "$STAGING/pass-correct.log"
  if [[ ( "$SHAPES" == tc-decode || "$SHAPES" == tc-highv || "$SHAPES" == tc-bf16q || "$SHAPES" == tc-bf16q-highv ) ]]; then
    grep -q "^TC_COVERAGE launches=$tc_rows requested=1$" "$STAGING/pass-correct.log"
  fi
fi
if [[ "${MIMO26_ATTN_DECODE_IMPL:-baseline}" == p1* ]]; then
  grep -q '^P1_PROTOCOL PASS cases=20 outputs=868352 guards=543232 query=.* scans=full$' "$STAGING/pass-correct.log"
fi
negative_pass() {
  local flags="$1" tag="$2" log="$STAGING/pass-naive-$2.log"
  echo "== run: negative $tag flags=$flags (numerical failure required)"
  nvidia-smi
  m26_gpu_guard
  set +e
  MIMO26_SPIKE_NAIVE=1 M26_NAIVE_FLAGS="$flags" "$STAGING/mimo26f-attn-parity" "$STAGING" | tee "$log"
  local pipe_rc=("${PIPESTATUS[@]}")
  set -e
  local naive_count
  naive_count="$(grep -c '^case ' "$log" || true)"
  if [[ "${pipe_rc[0]}" != 1 || "${pipe_rc[1]}" != 0 || "$naive_count" != "$correct_count" ]] || \
      ! grep -q '^RESULT: FAIL$' "$log" || ! grep -q '^case .*: FAIL' "$log"; then
    echo "RESULT: FAIL negative $tag was not a complete numerical failure (child=${pipe_rc[0]}, tee=${pipe_rc[1]}, cases=$naive_count/$correct_count)" >&2
    exit 4
  fi
  if [[ "${MIMO26_ATTN_DECODE_IMPL:-baseline}" != baseline ]]; then
    [[ "$(grep '^TC_COVERAGE ' "$log")" == "$(grep '^TC_COVERAGE ' "$STAGING/pass-correct.log")" ]] &&
      grep -q '^case .*: FAIL.*tc fp8' "$log" || { echo 'RESULT: FAIL missing candidate-path negative coverage'; exit 4; }
  fi
  echo "negative $tag failed numerically as expected (exit 1, cases=$naive_count/$correct_count)"
}
if [[ "${MIMO26_ATTN_TWO_RUN:-0}" == 1 ]]; then
  negative_pass 16383 all-original
  if [[ ( "$SHAPES" == tc-decode || "$SHAPES" == tc-highv || "$SHAPES" == tc-bf16q || "$SHAPES" == tc-bf16q-highv ) && "${MIMO26_ATTN_DECODE_IMPL:-baseline}" != baseline ]]; then
    # Individual traps must fail inside the same candidate, never by fallback.
    flags_list='1 2 4 8 16 64 128 256 16384 32768 65536 131072'
    [[ "$SHAPES" != tc-bf16q* ]] || flags_list='1 2 4 8 16 64 128 256 32768 65536 262144'
    for flags in $flags_list; do
      negative_pass "$flags" "bit-$flags"
    done
  fi
fi
printf 'RESULT: PASS parity runner (shapes=%s arch=%s cases=%s)\n' "$SHAPES" "$ARCH" "$correct_count"
