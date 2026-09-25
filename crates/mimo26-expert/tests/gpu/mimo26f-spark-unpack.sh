#!/usr/bin/env bash
# Native-only Spark cell; invoked by the dev.sh remote dispatcher. No Python/Rust.
set -euo pipefail
ROOT="${1:?unique scratch slot}"; SOURCE="${2:?source revision}"
MODE="${3:-unpack}"; RESIDENTS="${4:-256}"
export MIMO26_NCU_SUDO="${5:-0}"
case "$MIMO26_NCU_SUDO" in 0) ;; 1) [[ "$MODE" == b1-profile || "$MODE" == b2-profile || "$MODE" == b2-force4 ]] || exit 2 ;; *) exit 2 ;; esac
[[ "$MODE" == unpack || "$MODE" == bench || "$MODE" == wire || "$MODE" == b1-primitive || "$MODE" == b1-sanitize || "$MODE" == b1-ffn || "$MODE" == b1-bench || "$MODE" == b1-profile || "$MODE" == b2-profile || "$MODE" == b2-force4 || "$MODE" == b2-exact || "$MODE" == i4-reference || "$MODE" == silu-exp ]] || exit 2
case "$RESIDENTS" in 256|57|58) ;; *) exit 2 ;; esac
[[ "$ROOT" == /var/tmp/mimo26f-kernel/unpack-* && -d "$ROOT" ]] || exit 2
export TZ=Australia/Sydney LC_ALL=C
mkdir "$ROOT/receipts"
exec > >(tee "$ROOT/receipts/cell.log") 2>&1
exec 9>/var/tmp/mimo26f-kernel/.cell.lock
flock -n 9 || { echo 'Refuse another mimo26f-kernel cell'; exit 2; }
printf 'time=%s source=%s host=%s\n' "$(date --iso-8601=seconds)" "$SOURCE" "$(hostname)"
uname -m; /usr/local/cuda/bin/nvcc --version
sha256sum -c "$ROOT/SOURCE.sha256"
guard() {
  local owners available info uuid name
  info="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader)"
  [[ "$info" != *$'\n'* ]] || { echo 'Need exactly one GB10'; exit 2; }
  IFS=, read -r uuid name <<<"$info"
  [[ "$name" == *GB10* ]] || { echo 'Not a GB10'; exit 2; }
  export CUDA_VISIBLE_DEVICES="${uuid//[[:space:]]/}"
  owners="$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
  [[ -z "${owners//[[:space:]]/}" ]] || { echo "Refuse concurrent CUDA owners: $owners"; exit 2; }
  available="$(free -b | awk '$1=="Mem:" {print $7}')"
  [[ "$available" =~ ^[0-9]+$ && "$available" -ge 8589934592 ]] || { echo 'MemAvailable below8GiB'; exit 2; }
  printf '%s %s MemAvailable=%s CUDA owners=none\n' "$(date --iso-8601=seconds)" "$info" "$available" | tee -a "$ROOT/receipts/guards.log"
}
guard
NVCC=/usr/local/cuda/bin/nvcc
if [[ "$MODE" == silu-exp ]]; then
  source "$ROOT/silu_exp_remote.sh"
  exit 0
fi
if [[ "$MODE" == i4-reference ]]; then
  source "$ROOT/reference_exit_remote.sh"
  exit 0
fi
if [[ "$MODE" == b1-sanitize ]]; then
  SANITIZER=/usr/local/cuda/bin/compute-sanitizer
  [[ -x "$SANITIZER" ]] || { echo 'Compute Sanitizer unavailable; no installation attempted'; exit 2; }
  "$SANITIZER" --version | tee "$ROOT/receipts/sanitizer-version.log"
  source "$ROOT/b1_sanitizer_check.sh"
fi
if [[ "$MODE" == b1-primitive || "$MODE" == b1-sanitize || "$MODE" == b1-ffn || "$MODE" == b1-bench || "$MODE" == b1-profile ]]; then
  FLAGS=(-O3 -std=c++17 -lineinfo --ftz=false --gpu-architecture=compute_121a --gpu-code=sm_121a -DM26B1_TARGET_ARCH=121 -I"$ROOT")
  printf 'B1 primitive nvcc flags: '; printf '%q ' "${FLAGS[@]}"; printf '\n'
  "$NVCC" "${FLAGS[@]}" "$ROOT/primitive.cu" -o "$ROOT/mimo26f-b1-primitive" 2>&1 | tee "$ROOT/receipts/compile.log"
  DRIVER="$ROOT/mimo26f-b1-primitive"
  CUDA_VISIBLE_DEVICES='' "$DRIVER" --selftest | tee "$ROOT/receipts/host.log"
  export MIMO26_BUILDER_GPU=1
  if [[ "$MODE" == b1-profile ]]; then
    source "$ROOT/ncu_profile.sh"
    for m in 1 8; do profile_ncu B1 "$m" "$DRIVER" --profile-connected /var/tmp/models/MiMo-V2.6-Flash-RL "$ROOT/oracle" "$m"; done
    exit 0
  fi
  if [[ "$MODE" == b1-bench ]]; then
    guard
    set +e
    timeout 360s "$DRIVER" --bench-connected /var/tmp/models/MiMo-V2.6-Flash-RL "$ROOT/oracle" | tee "$ROOT/receipts/b1-bench.log"
    rc=${PIPESTATUS[0]}
    set -e
    guard
    echo "RESULT: B1 fixed bandwidth gate native_exit=$rc; no tuning or replacement"
    exit "$rc"
  fi
  if [[ "$MODE" == b1-ffn ]]; then
    mkdir "$ROOT/gpu-output"
    for m in 1 2 3 4 5 6 7 8; do
      guard
      mkdir "$ROOT/gpu-output/m$m"
      timeout 120s "$DRIVER" --ffn /var/tmp/models/MiMo-V2.6-Flash-RL "$ROOT/oracle" "$ROOT/gpu-output/m$m" "$m" 0 | tee "$ROOT/receipts/ffn-m$m.log"
    done
    guard
    mkdir "$ROOT/gpu-output/negative"
    timeout 120s "$DRIVER" --ffn /var/tmp/models/MiMo-V2.6-Flash-RL "$ROOT/oracle" "$ROOT/gpu-output/negative" 4 1 | tee "$ROOT/receipts/ffn-negative.log"
    guard
    echo 'RESULT: connected execution complete; dev-host stage comparison REQUIRED, no numerical/timing qualification yet'
    exit 0
  fi
  if [[ "$MODE" == b1-sanitize ]]; then
    for tool in memcheck initcheck synccheck racecheck; do
      guard
      extra=(); [[ "$tool" != memcheck ]] || extra=(--leak-check full)
      if timeout 120s "$SANITIZER" --tool "$tool" --error-exitcode 4 "${extra[@]}" "$DRIVER" --all-math >"$ROOT/receipts/$tool.log" 2>&1; then rc=0; else rc=$?; fi
      if validate_b1_sanitizer "$tool" "$ROOT/receipts/$tool.log" "$rc"; then :; else
        rc=$?; guard; nvidia-smi >"$ROOT/receipts/gpu-after.txt"; exit "$rc"
      fi
    done
    guard
    nvidia-smi >"$ROOT/receipts/gpu-after.txt"
    echo 'RESULT: PASS B1 compute-only memcheck/initcheck/synccheck/racecheck; not fused FFN or timing qualification'
    exit 0
  fi
  guard
  timeout 120s "$DRIVER" --gpu 0 | tee "$ROOT/receipts/primitive.log"
  for flag in 1 2 4; do
    guard
    set +e
    timeout 120s "$DRIVER" --gpu "$flag" >"$ROOT/receipts/negative-$flag.log" 2>&1
    rc=$?
    set -e
    [[ "$rc" == 3 ]] && grep -q '^ORACLE_MISMATCH ' "$ROOT/receipts/negative-$flag.log" || { echo "FAIL B1 negative flag=$flag rc=$rc"; exit 1; }
    echo "B1 NEGATIVE PASS flag=$flag typed numerical exit 3"
  done
  guard
  timeout 120s "$DRIVER" --compute 0 | tee "$ROOT/receipts/compute.log"
  for flag in 1 2 4 8; do
    guard
    set +e
    timeout 120s "$DRIVER" --compute "$flag" >"$ROOT/receipts/compute-negative-$flag.log" 2>&1
    rc=$?
    set -e
    [[ "$rc" == 3 ]] && grep -q '^ORACLE_MISMATCH ' "$ROOT/receipts/compute-negative-$flag.log" || { echo "FAIL B1 compute negative flag=$flag rc=$rc"; exit 1; }
    echo "B1 COMPUTE NEGATIVE PASS flag=$flag typed numerical exit 3"
  done
  guard
  timeout 120s "$DRIVER" --rank 0 | tee "$ROOT/receipts/rank.log"
  for flag in 1 2 4; do
    guard
    set +e
    timeout 120s "$DRIVER" --rank "$flag" >"$ROOT/receipts/rank-negative-$flag.log" 2>&1
    rc=$?
    set -e
    [[ "$rc" == 3 ]] && grep -q '^ORACLE_MISMATCH ' "$ROOT/receipts/rank-negative-$flag.log" || { echo "FAIL B1 rank negative flag=$flag rc=$rc"; exit 1; }
    echo "B1 RANK NEGATIVE PASS flag=$flag typed numerical exit 3"
  done
  guard
  timeout 120s "$DRIVER" --real /var/tmp/models/MiMo-V2.6-Flash-RL "$ROOT/fixture.json" 0 | tee "$ROOT/receipts/real-container.log"
  guard
  set +e
  timeout 120s "$DRIVER" --real /var/tmp/models/MiMo-V2.6-Flash-RL "$ROOT/fixture.json" 1 >"$ROOT/receipts/real-negative.log" 2>&1
  rc=$?
  set -e
  [[ "$rc" == 3 ]] && grep -q '^ORACLE_MISMATCH ' "$ROOT/receipts/real-negative.log" || { echo "FAIL B1 real negative rc=$rc"; exit 1; }
  guard
  nvidia-smi >"$ROOT/receipts/gpu-after.txt"
  echo 'RESULT: PASS sm_121a B1 primitives and real container K1; synthetic MMA only, no FFN/timing claim'
  exit 0
fi
FLAGS=(-O3 -std=c++17 -lineinfo --ftz=false -arch=sm_121 -DM26X_BAKED_ARCH=121 -DM26X_BAKED_SMS=48 -DM26X_CAPACITY_CLASS=2048 -I"$ROOT")
if [[ "$MODE" == b2-force4 ]]; then FLAGS+=(-DM26X_DIAGNOSTIC_FORCE4=1); fi
if [[ "$MODE" == b2-exact ]]; then FLAGS+=(-DM26X_EXACT_M=1); fi
printf 'nvcc flags: '; printf '%q ' "${FLAGS[@]}"; printf '\n'
"$NVCC" "${FLAGS[@]}" -Xptxas=-v -c "$ROOT/expert_gemm.cu" -o "$ROOT/expert_gemm.o" 2>&1 | tee "$ROOT/receipts/compile.log"
"$NVCC" "${FLAGS[@]}" "$ROOT/gemm_parity.cu" "$ROOT/expert_gemm.o" -o "$ROOT/mimo26f-expert-proof" 2>&1 | tee "$ROOT/receipts/driver-compile.log"
DRIVER="$ROOT/mimo26f-expert-proof"; WEIGHTS=/var/tmp/models/MiMo-V2.6-Flash-RL; FIXTURE="$ROOT/fixture.json"
CUDA_VISIBLE_DEVICES='' "$DRIVER" selftest
CUDA_VISIBLE_DEVICES='' "$DRIVER" audit "$WEIGHTS" "$FIXTURE"
CUDA_VISIBLE_DEVICES='' "$DRIVER" bench-selftest
export MIMO26_BUILDER_GPU=1
if [[ "$MODE" == wire ]]; then
  guard
  mkdir "$ROOT/gpu-output"
  nvidia-smi >"$ROOT/receipts/gpu-before.txt"
  set +e
  timeout 360s "$DRIVER" wire-export "$WEIGHTS" "$ROOT" "$ROOT/gpu-output" | tee "$ROOT/receipts/wire-export.log"
  rc=${PIPESTATUS[0]}
  set -e
  guard
  nvidia-smi >"$ROOT/receipts/gpu-after.txt"
  exit "$rc"
fi
if [[ "$MODE" == b2-profile || "$MODE" == b2-force4 ]]; then
  if [[ "$MODE" == b2-force4 ]]; then
    guard
    set +e
    timeout 120s "$DRIVER" diagnostic "$WEIGHTS" "$FIXTURE" "$ROOT" 5 | tee "$ROOT/receipts/forced4-diagnostic.log"
    rc=${PIPESTATUS[0]}
    set -e
    guard
    echo "FORCED4 DIAGNOSTIC native_exit=$rc; only M5, not all-M qualification"
    case "$rc" in 0|6|7) ;; *) exit "$rc" ;; esac
  fi
  source "$ROOT/ncu_profile.sh"
  for m in 4 5; do profile_ncu B2 "$m" "$DRIVER" profile "$WEIGHTS" "$FIXTURE" "$ROOT" "$m"; done
  exit 0
fi
if [[ "$MODE" == bench || "$MODE" == b2-exact ]]; then
  guard
  nvidia-smi >"$ROOT/receipts/gpu-before.txt"
  set +e
  timeout 360s "$DRIVER" bench "$WEIGHTS" "$FIXTURE" "$ROOT" "$RESIDENTS" | tee "$ROOT/receipts/bench.log"
  rc=${PIPESTATUS[0]}
  set -e
  echo "BENCH native_exit=$rc (6=STOP,7=PIVOT,0=TARGET; other codes are failures)"
  guard
  nvidia-smi >"$ROOT/receipts/gpu-after.txt"
  exit "$rc"
fi
guard
timeout 120s "$DRIVER" decoder-check 0 | tee "$ROOT/receipts/decoder.log"
guard
timeout 120s "$DRIVER" dump-unpack "$WEIGHTS" "$FIXTURE" "$ROOT/receipts/dump.json" 0 | tee "$ROOT/receipts/unpack.log"
for flag in 1 4 128 2; do
  guard
  set +e
  if [[ "$flag" == 2 ]]; then
    timeout 120s "$DRIVER" decoder-check "$flag" >"$ROOT/receipts/negative-$flag.log" 2>&1
  else
    timeout 120s "$DRIVER" dump-unpack "$WEIGHTS" "$FIXTURE" "$ROOT/receipts/negative-$flag.json" "$flag" >"$ROOT/receipts/negative-$flag.log" 2>&1
  fi
  rc=$?
  set -e
  [[ "$rc" == 3 ]] && grep -q '^ORACLE_MISMATCH ' "$ROOT/receipts/negative-$flag.log" || { echo "FAIL negative$flag rc=$rc"; exit 1; }
  echo "NEGATIVE PASS flag=$flag numerical exit3"
done
guard
nvidia-smi >"$ROOT/receipts/gpu-after.txt"
echo 'RESULT: PASS Spark sm121 native real unpack; no bandwidth/FFN/sanitizer promotion'
