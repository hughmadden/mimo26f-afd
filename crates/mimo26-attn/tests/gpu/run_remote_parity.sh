#!/usr/bin/env bash
# Internal adapter. Called only from scripts/dev.sh test attn via the crate runner.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CRATE="$REPO/crates/mimo26-attn"
source "$CRATE/tests/gpu/gpu_guard.sh"
remote_body() {
  printf '%s\n' 'set -euo pipefail'
  declare -f m26_guard_service m26_guard_read_argv m26_gpu_guard
  cat <<'REMOTE'
stage="$1"
shapes="$2"
family="${3:-c3}"
case "$family" in c3|c1|p1) ;; *) exit 3;; esac
rows=16; cases=8
if [[ "$family" == p1 ]]; then rows=20; cases=10; fi
export TZ=Australia/Sydney
exec > >(tee "$stage/receipt.log") 2>&1
exec 9>/var/tmp/mimo26f-attn/.gpu.lock
flock -n 9 || { echo 'RESULT: REFUSE remote GPU lock held'; exit 3; }
date '+%F %T %Z'
nvidia-smi
m26_gpu_guard
sha256sum "$stage/mimo26f-attn-parity"
printf 'SOURCE '; cat "$stage/source.txt"
echo "COMMAND HOST=coordinator MIMO26_ATTN_BUILDER_WINDOW=1 MIMO26_ATTN_PARITY_FAMILY=$family scripts/dev.sh test attn $shapes"
cat "$stage/v-audit.txt"
start=$SECONDS
run_case() {
  local impl="$1" flags="$2" expected="$3" tag="$4" log="$stage/$1-$4.log" splits="${5:-8}"
  (( SECONDS-start < 510 )) || { echo 'RESULT: FAIL remote parity total budget'; exit 4; }
  m26_gpu_guard
  echo "REMOTE_CASE impl=$impl flags=$flags tag=$tag"
  set +e
  MIMO26_ATTN_PARITY_SPLITS="$splits" MIMO26_ATTN_DECODE_IMPL="$impl" M26_NAIVE_FLAGS="$flags" MIMO26_SPIKE_NAIVE=0 \
    timeout 30 "$stage/mimo26f-attn-parity" "$stage" | tee "$log"
  local rc=("${PIPESTATUS[@]}")
  set -e
  [[ "${rc[0]}" == "$expected" && "${rc[1]}" == 0 ]] || {
    echo "RESULT: FAIL child status impl=$impl tag=$tag child=${rc[0]} tee=${rc[1]}"; exit 4;
  }
  [[ "$(grep -c '^case ' "$log")" == "$rows" ]] || { echo 'RESULT: FAIL incomplete case rows'; exit 4; }
  grep -q '^PARITY_AOT PASS arch=sm_120 sms=170 baked=1200 ' "$log"
  if [[ "$impl" != baseline ]]; then
    grep -q "^TC_COVERAGE launches=$rows requested=1$" "$log"
    grep -q '^GPU_CODEC PASS pairs=65536/65536$' "$log"
    [[ "$(grep -c "^PATH case=.* decode_impl=$impl " "$log")" == "$cases" ]]
  fi
  if [[ "$expected" == 0 ]]; then
    grep -q '^RESULT: PASS$' "$log"
    if [[ "$impl" == p1* ]]; then
      grep -q '^P1_PROTOCOL PASS cases=20 outputs=868352 guards=543232 query=.* scans=full$' "$log"
    fi
    if [[ "$impl" == c1* ]]; then
      grep -q '^C1_PROTOCOL PASS cases=8 outputs=65536 query=.* partial_scans=full$' "$log"
    fi
  else
    grep -q '^RESULT: FAIL$' "$log"
    if [[ "$impl" == baseline ]]; then grep -q '^case .*: FAIL' "$log"
    else grep -q '^case .*: FAIL.*tc fp8' "$log"; fi
    echo "NEGATIVE PASS impl=$impl flags=$flags complete_rows=$rows"
  fi
}
run_case baseline 0 0 positive
run_case baseline 16383 1 all-original
impls='pipe4 pipe8'
flags='1 2 4 8 16 64 128 256 16384 32768 65536 131072'
if [[ "$shapes" == tc-bf16q* ]]; then
  impls='pipe4-bf16q pipe8-bf16q'
  flags='1 2 4 8 16 64 128 256 32768 65536 262144'
fi
if [[ "$family" == c1 || "$family" == p1 ]]; then
  impls="$family"; [[ "$shapes" != tc-bf16q* ]] || impls="$family-bf16q"
fi
for impl in $impls; do
  run_case "$impl" 0 0 positive
  run_case "$impl" 0 0 positive-reuse 3
  run_case "$impl" 16383 1 all-original 3
  for flag in $flags; do
    run_case "$impl" "$flag" 1 "bit-$flag" 3
  done
done
echo "RESULT: PASS remote parity shapes=$shapes baseline and $impls ($rows rows each, isolated flags: $flags)"
REMOTE
}
if [[ "${1:-}" == --selftest ]]; then
  remote_body | bash -n
  echo 'RESULT: PASS remote parity shell syntax (CPU only)'
  exit 0
fi
STAGING="$1"
[[ "${MIMO26_ATTN_BUILDER_WINDOW:-0}" == 1 && ( "$2" == tc-decode || "$2" == tc-highv || "$2" == tc-bf16q || "$2" == tc-bf16q-highv ) ]] || {
  echo 'RESULT: REFUSE remote parity requires acknowledged tc-decode cell'; exit 3;
}
oracle_extra=(); [[ "${MIMO26_ATTN_PARITY_FAMILY:-c3}" != p1 ]] || oracle_extra=(--p1)
python3 "$CRATE/tests/oracle_driver.py" audit-tc --shapes "$2" "${oracle_extra[@]}" > "$STAGING/v-audit.txt"
REMOTE_STAGE="$(ssh coordinator 'mkdir -p /var/tmp/mimo26f-attn && mktemp -d /var/tmp/mimo26f-attn/mimo26f-parity-XXXXXXXX')"
[[ "$REMOTE_STAGE" =~ ^/var/tmp/mimo26f-attn/mimo26f-parity-[A-Za-z0-9]{8}$ ]] || exit 3
rsync -a "$STAGING/mimo26f-attn-parity" "$STAGING/manifest.txt" "$STAGING/v-audit.txt" "$STAGING/source.txt" "$STAGING/"*.bin "coordinator:$REMOTE_STAGE/"
flock -u 9
flock -u 8
set +e
remote_body | ssh coordinator bash -s -- "$REMOTE_STAGE" "$2" "${MIMO26_ATTN_PARITY_FAMILY:-c3}"
rc=("${PIPESTATUS[@]}")
set -e
receipt_dir="$REPO/runs/20260923-i4/attn/$(basename "$STAGING")"
mkdir -p "$receipt_dir"
rsync -a "coordinator:$REMOTE_STAGE/receipt.log" "$receipt_dir/RESULT.md"
printf 'REMOTE_RECEIPT %s/RESULT.md\n' "$receipt_dir"
[[ "${rc[0]}" == 0 ]] || exit "${rc[0]}"
exit "${rc[1]}"
