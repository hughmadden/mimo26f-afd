#!/usr/bin/env bash
# Run only by crate dev.sh dispatcher. Agent allowed on the dev host once nvidia-smi works;
# otherwise builder only. No fleet/SSH/container operations. Legacy opt-in name.
set -euo pipefail
[[ "${MIMO26_BUILDER_GPU:-0}" == 1 ]] || { echo 'Explicit authorized GPU opt-in required' >&2; exit 2; }
DRIVER="$1"; WEIGHTS="$2"; FIXTURE="$3"; SLOT="$4"; REPO="$5"
exec > >(tee "$SLOT/gpu-cell.log") 2>&1
if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=normal)" ]]; then
  echo 'Refuse GPU proof from an uncommitted worktree' >&2; exit 2
fi
gpu_guard() {
  local info uuid name free owners
  local -a choices=()
  if ! info="$(nvidia-smi --query-gpu=uuid,name,memory.free --format=csv,noheader,nounits)"; then
    echo 'INFRA_FAILURE: GPU not visible; queue builder, never escalate' >&2; exit 2
  fi
  [[ -f "$SLOT/gpu-before.txt" ]] || printf '%s\n' "$info" >"$SLOT/gpu-before.txt"
  printf '\n%s\n%s\n' "$(TZ=Australia/Sydney date '+%Y-%m-%d %H:%M:%S %Z')" "$info" >>"$SLOT/gpu-checks.log"
  while IFS=, read -r uuid name free; do
    uuid="${uuid//[[:space:]]/}"; free="${free//[[:space:]]/}"
    if [[ "$name" == *'RTX 4090'* && "$free" =~ ^[0-9]+$ && "$free" -ge 4608 ]]; then
      if [[ -z "${MIMO26_GPU_UUID:-}" || "$MIMO26_GPU_UUID" == "$uuid" ]]; then choices+=("$uuid"); fi
    fi
  done <<<"$info"
  [[ ${#choices[@]} == 1 ]] || { echo 'Need one RTX4090 with4GiB reserve+512MiB context/working headroom; optionally set MIMO26_GPU_UUID' >&2; exit 2; }
  export CUDA_VISIBLE_DEVICES="${choices[0]}" MIMO26_GPU_UUID="${choices[0]}"
  if ! owners="$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"; then
    echo 'INFRA_FAILURE: cannot verify CUDA ownership' >&2; exit 2
  fi
  # Resident eye/ear services are authorized; all other owners still block.
  # Match live argv/module, never restart-sensitive PIDs or loose substrings.
  printf '%s\n' "$owners" | python3 "$REPO/crates/mimo26-expert/tests/gpu/cuda_owner_guard.py" | tee -a "$SLOT/gpu-checks.log"
}
if [[ "${6:-unpack}" == capacity ]]; then
  for capacity in 256 2048 4096; do
    DRIVER="$SLOT/class$capacity/gemm-parity"
    gpu_guard
    timeout 180s "$DRIVER" routing 0 | tee "$SLOT/class$capacity/routing-correct.log"
    gpu_guard
    set +e
    timeout 90s "$DRIVER" routing 1 >"$SLOT/class$capacity/routing-ordinal.log" 2>&1
    rc=$?
    set -e
    if [[ "$rc" != 3 ]] || ! grep -q '^ORACLE_MISMATCH ' "$SLOT/class$capacity/routing-ordinal.log"; then
      echo "FAIL: class$capacity ordinal substitution not detected numerically (rc=$rc)" >&2; exit 1
    fi
    echo "CAPACITY GPU PASS class=$capacity: full256 synthetic routing plus ordinal negative"
  done
  echo 'RESULT: PASS pg4090 PROXY capacity256/2048/4096 synthetic routing; no real-checkpoint256 or bandwidth claim'
  exit 0
fi
if [[ "${6:-unpack}" == tp4 ]]; then
  ORACLE="${7:?TP4 staging required}"
  gpu_guard
  timeout 240s "$DRIVER" tp4 "$ORACLE" "$FIXTURE" 0 | tee "$SLOT/tp4-correct.log"
  for flag in 1 4 64 128 8; do
    gpu_guard
    set +e
    timeout 90s "$DRIVER" tp4 "$ORACLE" "$FIXTURE" "$flag" >"$SLOT/tp4-naive-$flag.log" 2>&1
    rc=$?
    set -e
    expected=3; marker='^ORACLE_MISMATCH '
    if [[ "$flag" == 8 ]]; then expected=5; marker='^PADDING_READ_DETECTED '; fi
    if [[ "$rc" != "$expected" ]] || ! grep -q "$marker" "$SLOT/tp4-naive-$flag.log"; then
      echo "FAIL: TP4 mutation$flag did not fail for its intended reason (rc=$rc)" >&2; exit 1
    fi
    echo "TP4 NEGATIVE PASS flag=$flag: actual wrong device implementation detected"
  done
  echo 'RESULT: PASS pg4090 PROXY real TP4/GEMM proof; not full256 prefill, sanitizer or bandwidth qualification'
  exit 0
fi
[[ "${6:-unpack}" == unpack ]] || { echo 'unknown proof mode' >&2; exit 2; }
gpu_guard
timeout 120s "$DRIVER" decoder-check 0 | tee "$SLOT/decoder-correct.log"
gpu_guard
timeout 120s "$DRIVER" dump-unpack "$WEIGHTS" "$FIXTURE" "$SLOT/dump.json" 0 | tee "$SLOT/unpack-correct.log"
python3 "$REPO/harness/nibble_proof.py" --fixture "$FIXTURE" --dump "$SLOT/dump.json" | tee "$SLOT/comparator-correct.log"
# Each independent wrong implementation must fail at the numerical oracle,
# not in CUDA, input parsing, timeout, source identity, or device selection.
for flag in 1 4 128 2; do
  gpu_guard
  set +e
  if [[ "$flag" == 2 ]]; then
    timeout 120s "$DRIVER" decoder-check "$flag" >"$SLOT/naive-$flag.log" 2>&1
  else
    timeout 120s "$DRIVER" dump-unpack "$WEIGHTS" "$FIXTURE" "$SLOT/naive-$flag.json" "$flag" >"$SLOT/naive-$flag.log" 2>&1
  fi
  rc=$?
  set -e
  if [[ "$rc" != 3 ]] || ! grep -q '^ORACLE_MISMATCH ' "$SLOT/naive-$flag.log"; then
    echo "FAIL: mutation$flag did not fail numerically (rc=$rc)" >&2; exit 1
  fi
  if [[ "$flag" != 2 ]]; then
    set +e
    python3 "$REPO/harness/nibble_proof.py" --fixture "$FIXTURE" --dump "$SLOT/naive-$flag.json" >"$SLOT/comparator-naive-$flag.log" 2>&1
    rc=$?
    set -e
    [[ "$rc" == 1 ]] || { echo "FAIL: external comparator did not reject mutation$flag" >&2; exit 1; }
  fi
  echo "NEGATIVE PASS flag=$flag: numerical oracle rejected actual wrong device implementation"
done
echo 'RESULT: PASS pg4090 PROXY unpack only; GEMM/FFN, Spark bandwidth and AOT classes still separate gates'
