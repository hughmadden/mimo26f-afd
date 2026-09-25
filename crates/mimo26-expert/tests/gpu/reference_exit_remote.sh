#!/usr/bin/env bash
# Sourced only by the guarded, source-pinned Spark i4-reference cell.
# Reference correctness/portability only. No kernel tuning or timing promotion.
set -euo pipefail
export MIMO26_BUILDER_GPU=1
SANITIZER=/usr/local/cuda/bin/compute-sanitizer
[[ -x "$SANITIZER" ]] || { echo 'Compute Sanitizer unavailable; no install attempted'; exit 2; }
source "$ROOT/b1_sanitizer_check.sh"
"$SANITIZER" --version >"$ROOT/receipts/sanitizer-version.log"

for capacity in 256 2048 4096; do
  flags=(-O3 -std=c++17 -lineinfo --ftz=false -arch=sm_121 -DM26X_BAKED_ARCH=121 -DM26X_BAKED_SMS=48 -DM26X_CAPACITY_CLASS="$capacity" -I"$ROOT")
  printf 'REFERENCE AOT compile class=%s flags=' "$capacity"; printf '%q ' "${flags[@]}"; printf '\n'
  "$NVCC" "${flags[@]}" -Xptxas=-v -c "$ROOT/expert_gemm.cu" -o "$ROOT/expert-$capacity.o" 2>&1 | tee "$ROOT/receipts/aot-$capacity-compile.log"
  "$NVCC" "${flags[@]}" "$ROOT/gemm_parity.cu" "$ROOT/expert-$capacity.o" -o "$ROOT/mimo26f-reference-$capacity" 2>&1 | tee "$ROOT/receipts/aot-$capacity-driver.log"
  driver="$ROOT/mimo26f-reference-$capacity"
  CUDA_VISIBLE_DEVICES='' "$driver" routing-selftest | tee "$ROOT/receipts/aot-$capacity-host.log"
  guard
  timeout 90s "$driver" routing 0 | tee "$ROOT/receipts/aot-$capacity-positive.log"
  [[ "$(grep -c '^AOT PRELAUNCH PASS ' "$ROOT/receipts/aot-$capacity-positive.log" || true)" == 12 ]]
  grep -q "^ROUTING GPU PASS class=$capacity " "$ROOT/receipts/aot-$capacity-positive.log"
  guard
  rc=0
  timeout 90s "$driver" routing 1 >"$ROOT/receipts/aot-$capacity-ordinal-negative.log" 2>&1 || rc=$?
  [[ "$rc" == 3 ]] && grep -q '^ORACLE_MISMATCH ' "$ROOT/receipts/aot-$capacity-ordinal-negative.log" || { echo "FAIL ordinal negative class=$capacity rc=$rc"; exit 2; }
  echo "REFERENCE AOT PASS class=$capacity live positive, prelaunch negatives, powered bypass and routing mutation"
done

flags=(-O3 -std=c++17 -lineinfo --ftz=false --gpu-architecture=compute_121a --gpu-code=sm_121a -DM26B1_TARGET_ARCH=121 -I"$ROOT")
"$NVCC" "${flags[@]}" "$ROOT/primitive.cu" -o "$ROOT/mimo26f-reference-b1" 2>&1 | tee "$ROOT/receipts/connected-compile.log"
CUDA_VISIBLE_DEVICES='' "$ROOT/mimo26f-reference-b1" --selftest | tee "$ROOT/receipts/connected-host.log"
mkdir "$ROOT/gpu-output"
for tool in memcheck initcheck synccheck racecheck; do
  for m in 1 8; do
    guard
    output="$ROOT/gpu-output/$tool-m$m"; mkdir "$output"
    extra=(); [[ "$tool" != memcheck ]] || extra=(--leak-check full)
    rc=0
    raw_payload=0; [[ "$tool" != initcheck ]] || raw_payload=1
    MIMO26_B1_INITCHECK="$raw_payload" timeout 120s "$SANITIZER" --tool "$tool" --error-exitcode 4 "${extra[@]}" "$ROOT/mimo26f-reference-b1" --ffn /var/tmp/models/MiMo-V2.6-Flash-RL "$ROOT/oracle" "$output" "$m" 0 >"$ROOT/receipts/connected-$tool-m$m.log" 2>&1 || rc=$?
    # Keep partial artifacts on failure; no fallback to compute-only checks.
    validate_b1_sanitizer "$tool" "$ROOT/receipts/connected-$tool-m$m.log" "$rc" connected "$m"
  done
done
guard
rc=0
MIMO26_B1_INITCHECK=1 timeout 60s "$SANITIZER" --tool initcheck --error-exitcode 4 "$ROOT/mimo26f-reference-b1" --initcheck-negative >"$ROOT/receipts/initcheck-powered-negative.log" 2>&1 || rc=$?
[[ "$rc" == 4 ]] && grep -q 'Uninitialized __global__ memory read' "$ROOT/receipts/initcheck-powered-negative.log" && grep -q '^INITCHECK NEGATIVE executed ' "$ROOT/receipts/initcheck-powered-negative.log" || { echo "FAIL uninitialized-read control rc=$rc"; exit 2; }
echo 'INITCHECK POWERED NEGATIVE PASS: real uninitialized read rejected with exit4'
guard
mkdir "$ROOT/gpu-output/negative"
timeout 90s "$ROOT/mimo26f-reference-b1" --ffn /var/tmp/models/MiMo-V2.6-Flash-RL "$ROOT/oracle" "$ROOT/gpu-output/negative" 8 1 | tee "$ROOT/receipts/connected-numerical-negative.log"
echo 'CONNECTED SANITIZER CAPTURE PASS; dev-host independent comparisons and negative refusal remain REQUIRED'

# Cheap single-layer/rank0 residency controls. Keep all native timings, including
# STOP/PIVOT; acceptance here is correctness/residency, not a bandwidth promotion.
for residents in 57 58; do
  guard
  rc=0
  timeout 90s "$ROOT/mimo26f-reference-2048" bench /var/tmp/models/MiMo-V2.6-Flash-RL "$ROOT/fixture.json" "$ROOT" "$residents" >"$ROOT/receipts/residency-$residents.log" 2>&1 || rc=$?
  case "$rc" in 0|6|7) ;; *) echo "RESIDENCY FAILURE residents=$residents rc=$rc"; exit "$rc" ;; esac
  [[ "$(grep -c '^BENCH CORRECT ' "$ROOT/receipts/residency-$residents.log" || true)" == 8 ]]
  [[ "$(grep -c '^BENCH ROW ' "$ROOT/receipts/residency-$residents.log" || true)" == 8 ]]
  echo "RESIDENCY COMPLETE residents=$residents native_exit=$rc; one layer/rank0, NOT full-model residency or performance promotion"
done
guard
nvidia-smi >"$ROOT/receipts/gpu-after.txt"
echo 'REFERENCE REMOTE COMPLETE: AOT3 classes, connected sanitizers M1/M8, residency57/58; local numerical comparisons pending'
