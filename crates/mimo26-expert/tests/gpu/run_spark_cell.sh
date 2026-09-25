#!/usr/bin/env bash
# Local orchestration only. All deployment starts through scripts/dev.sh test gemm.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
MODE="${1:-unpack}"; HOST="${HOST:-}"
case "$HOST" in spark1|spark2|spark3|spark4) ;; *) echo 'HOST must be spark1..4' >&2; exit 2 ;; esac
if [[ "$MODE" == b2-force4 || "$MODE" == --dry-b2-force4 || "$MODE" == b2-exact || "$MODE" == --dry-b2-exact || "$MODE" == i4-reference || "$MODE" == --dry-i4-reference || "$MODE" == silu-exp || "$MODE" == --dry-silu-exp ]]; then
  [[ "$HOST" == spark1 ]] || { echo 'Owned reference/diagnostic cells are spark1-only' >&2; exit 2; }
fi
SOURCE="$(git -C "$REPO" rev-parse HEAD)"
TESTS="$REPO/crates/mimo26-expert/tests/gpu"
RESIDENTS="${MIMO26_BENCH_RESIDENTS:-256}"
NCU_SUDO="${MIMO26_NCU_SUDO:-0}"
case "$NCU_SUDO" in 0|1) ;; *) echo 'MIMO26_NCU_SUDO must be 0 or 1' >&2; exit 2 ;; esac
case "$RESIDENTS" in 256|57|58) ;; *) echo 'Invalid benchmark resident count' >&2; exit 2 ;; esac
if [[ "$MODE" == --dry || "$MODE" == --dry-bench || "$MODE" == --dry-wire || "$MODE" == --dry-b1-primitive || "$MODE" == --dry-b1-sanitize || "$MODE" == --dry-b1-ffn || "$MODE" == --dry-b1-bench || "$MODE" == --dry-b1-profile || "$MODE" == --dry-b2-profile || "$MODE" == --dry-b2-force4 || "$MODE" == --dry-b2-exact || "$MODE" == --dry-i4-reference || "$MODE" == --dry-silu-exp ]]; then
  # Print a runnable command template; no SSH, mkdir, rsync or compilation here.
  dry_mode=unpack; [[ "$MODE" != --dry-bench ]] || dry_mode=bench
  [[ "$MODE" != --dry-wire ]] || dry_mode=wire
  [[ "$MODE" != --dry-b1-primitive ]] || dry_mode=b1-primitive
  [[ "$MODE" != --dry-b1-sanitize ]] || dry_mode=b1-sanitize
  [[ "$MODE" != --dry-b1-ffn ]] || dry_mode=b1-ffn
  [[ "$MODE" != --dry-b1-bench ]] || dry_mode=b1-bench
  [[ "$MODE" != --dry-b1-profile ]] || dry_mode=b1-profile
  [[ "$MODE" != --dry-b2-profile ]] || dry_mode=b2-profile
  [[ "$MODE" != --dry-b2-force4 ]] || dry_mode=b2-force4
  [[ "$MODE" != --dry-b2-exact ]] || dry_mode=b2-exact
  [[ "$MODE" != --dry-i4-reference ]] || dry_mode=i4-reference
  [[ "$MODE" != --dry-silu-exp ]] || dry_mode=silu-exp
  printf 'HOST=%q\nSOURCE=%q\nMODE=%q\nRESIDENTS=%q\nNCU_SUDO=%q\n' "$HOST" "$SOURCE" "$dry_mode" "$RESIDENTS" "$NCU_SUDO"
  printf '%s\n' 'REMOTE=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "mkdir -p /var/tmp/mimo26f-kernel && mktemp -d /var/tmp/mimo26f-kernel/unpack-XXXXXXXX")'
  printf '%s\n' '# BUNDLE is the unique local source-only snapshot; checksum entries use $REMOTE paths.'
  printf '%s\n' 'rsync -a --checksum "$BUNDLE/" "$HOST:$REMOTE/"'
  printf '%s\n' 'ssh -o BatchMode=yes "$HOST" "timeout 540s bash $REMOTE/mimo26f-spark-unpack.sh $REMOTE $SOURCE $MODE $RESIDENTS $NCU_SUDO"'
  printf '%s\n' '# Bench only: independent x/partial goldens are generated on the dev host; no model weights are transferred. 10 warmups,31 CUDA-event samples,M1..8; STOP below163.8, pivot below191.1GB/s.'
  printf '%s\n' 'rsync -a --safe-links --include="*.log" --include="*.json" --include="*.txt" --exclude="*" "$HOST:$REMOTE/receipts/" "$RECEIPTS/"'
  printf '%s\n' '# Remote script: nvcc13 -arch=sm_121, baked arch121/physicalSM48/class2048; native audit, decoder,27 real matrices,4 negatives. No remote Python/Rust.'
  exit 0
fi
[[ "$MODE" == unpack || "$MODE" == bench || "$MODE" == wire || "$MODE" == b1-primitive || "$MODE" == b1-sanitize || "$MODE" == b1-ffn || "$MODE" == b1-bench || "$MODE" == b1-profile || "$MODE" == b2-profile || "$MODE" == b2-force4 || "$MODE" == b2-exact || "$MODE" == i4-reference || "$MODE" == silu-exp ]] || { echo 'unknown Spark mode' >&2; exit 2; }
[[ "${MIMO26_BUILDER_GPU:-0}" == 1 ]] || { echo 'Explicit GPU opt-in required' >&2; exit 2; }
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=normal -- crates/mimo26-expert crates/mimo26-repack bench/fixtures/expert_nibble_fixture.json harness/nibble_proof.py)" ]] || { echo 'Commit expert sources before remote execution' >&2; exit 2; }
BUILD_ROOT="${MIMO26F_BUILD_ROOT:?dev.sh required}"; mkdir -p "$BUILD_ROOT"
STAGING="$(mktemp -d "$BUILD_ROOT/expert-spark-$MODE.XXXXXX")"; BUNDLE="$STAGING/bundle"; mkdir "$BUNDLE"
TAG="$(TZ=Australia/Sydney date +%Y%m%d-%H%M%S)-${SOURCE:0:12}-$(basename "$STAGING")"
RECEIPTS="$REPO/runs/20260923-i4/expert/$HOST-$TAG"; mkdir "$RECEIPTS"
exec > >(tee "$RECEIPTS/local.log") 2>&1
printf 'time=%s source=%s host=%s staging=%s receipts=%s\n' "$(TZ=Australia/Sydney date --iso-8601=seconds)" "$SOURCE" "$HOST" "$STAGING" "$RECEIPTS"
cp "$REPO/crates/mimo26-expert/kernels/"*.cu "$BUNDLE/"
cp "$REPO/crates/mimo26-expert/kernels/include/"*.h "$REPO/crates/mimo26-expert/kernels/include/"*.cuh "$BUNDLE/"
cp "$REPO/crates/mimo26-expert/kernels/parity/gemm_parity.cu" "$REPO/crates/mimo26-expert/kernels/parity/"*.cuh "$BUNDLE/"
cp "$TESTS/fixture_io.h" "$TESTS/tp4_io.h" "$TESTS/routing_proof.h" "$TESTS/bench_io.h" "$TESTS/mimo26f-spark-unpack.sh" "$TESTS/ncu_profile.sh" "$BUNDLE/"
if [[ "$MODE" == silu-exp ]]; then
  map="${MIMO26_EXP_REFERENCE_MAP:?run silu-exp-cpu first}"
  cp "$TESTS/silu_exp_reference.h" "$TESTS/silu_exp_gpu.cu" "$TESTS/silu_exp_remote.sh" "$BUNDLE/"
  cp "$map" "$BUNDLE/reference-map.tsv"
  cp "$(dirname "$map")/cpu-summary.json" "$RECEIPTS/prior-cpu-summary.json"
fi
if [[ "$MODE" == i4-reference ]]; then cp "$TESTS/reference_exit_remote.sh" "$BUNDLE/"; fi
if [[ "$MODE" == b1-sanitize || "$MODE" == i4-reference ]]; then
  mkdir "$STAGING/validator-fixtures"
  bash "$TESTS/b1_sanitizer_check.sh" --selftest "$STAGING/validator-fixtures" | tee "$RECEIPTS/sanitizer-selftest.log"
  cp "$TESTS/b1_sanitizer_check.sh" "$BUNDLE/"
fi
if [[ "$MODE" == b1-primitive || "$MODE" == b1-sanitize || "$MODE" == b1-ffn || "$MODE" == b1-bench || "$MODE" == b1-profile || "$MODE" == i4-reference || "$MODE" == silu-exp ]]; then
  cp "$REPO/crates/mimo26-expert/kernels/b1/"{mxfp4_ptx.cuh,grouped.cu,LICENSE.b1-compute} "$REPO/crates/mimo26-expert/tests/b1_core/"{primitive.cu,compute_test.cuh,rank_compute_test.cuh,ffn_test.cuh,bench_connected.cuh} "$BUNDLE/"
  mkdir "$BUNDLE/b1" "$BUNDLE/include"
  cp "$REPO/crates/mimo26-expert/kernels/b1/"* "$BUNDLE/b1/"
  cp "$REPO/crates/mimo26-expert/kernels/include/"*.h "$REPO/crates/mimo26-expert/kernels/include/"*.cuh "$BUNDLE/include/"
fi
if [[ "$MODE" == b1-ffn || "$MODE" == i4-reference ]]; then
  bash "$REPO/crates/mimo26-expert/tests/b1_core/run_lattice_oracle.sh" "$REPO" "$STAGING" lattice-oracle-real "${MIMO26_WEIGHTS_DIR:-$HOME/models/XiaomiMiMo/MiMo-V2.6-Flash-RL}"
  cp "$STAGING/"*.log "$RECEIPTS/"
  mkdir "$BUNDLE/oracle"
  cp "$STAGING/oracle/"{b1-source.json,b1-x-payload.u8,b1-x-scales.u8,b1-weights.f32} "$BUNDLE/oracle/"
elif [[ "$MODE" == b1-bench || "$MODE" == b1-profile ]]; then
  CUDA_VISIBLE_DEVICES='' python3 -B "$REPO/crates/mimo26-repack/tests/lattice_bench.py" "${MIMO26_WEIGHTS_DIR:-$HOME/models/XiaomiMiMo/MiMo-V2.6-Flash-RL}" "$STAGING/oracle" | tee "$RECEIPTS/bench-oracle.log"
  mkdir "$BUNDLE/oracle"
  cp "$STAGING/oracle/"* "$BUNDLE/oracle/"
fi
if [[ "$MODE" == bench || "$MODE" == b2-profile || "$MODE" == b2-force4 || "$MODE" == b2-exact || "$MODE" == i4-reference ]]; then
  tp4_oracle="$STAGING/oracle"; [[ "$MODE" != i4-reference ]] || tp4_oracle="$STAGING/tp4-oracle"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 "$REPO/crates/mimo26-repack/tests/tp4_oracle.py" "${MIMO26_WEIGHTS_DIR:-$HOME/models/XiaomiMiMo/MiMo-V2.6-Flash-RL}" "$tp4_oracle" | tee "$RECEIPTS/oracle.log"
  cp "$tp4_oracle/x.f32" "$BUNDLE/oracle-x.f32"
  for expert in 0 7 255; do
    printf -v name 'L01_E%03d.partial_R0.f32' "$expert"
    cp "$tp4_oracle/$name" "$BUNDLE/oracle-e$expert.f32"
  done
fi
if [[ "$MODE" == wire ]]; then
  exec 8>"$BUILD_ROOT/.cargo.lock"; flock 8
  bash "$TESTS/run_wire_host.sh" "$REPO" "$STAGING"
  flock -u 8
  cp "$STAGING/"*.log "$RECEIPTS/"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 "$REPO/crates/mimo26-repack/tests/wire_oracle.py" "${MIMO26_WEIGHTS_DIR:-$HOME/models/XiaomiMiMo/MiMo-V2.6-Flash-RL}" "$STAGING/oracle" | tee "$RECEIPTS/oracle.log"
  cp "$STAGING/oracle/wire-x.f32" "$STAGING/oracle/wire-source.json" "$BUNDLE/"
  cp "$STAGING/oracle/wire-source.json" "$RECEIPTS/"
fi
cp "$REPO/bench/fixtures/expert_nibble_fixture.json" "$BUNDLE/fixture.json"
REMOTE="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" 'mkdir -p /var/tmp/mimo26f-kernel && mktemp -d /var/tmp/mimo26f-kernel/unpack-XXXXXXXX')"
[[ "$REMOTE" =~ ^/var/tmp/mimo26f-kernel/unpack-[[:alnum:]]{8}$ ]] || { echo 'Unexpected remote staging path'; exit 2; }
printf 'remote=%s\n' "$REMOTE" | tee "$RECEIPTS/remote.txt"
for file in "$BUNDLE/"* "$BUNDLE/b1/"* "$BUNDLE/include/"* "$BUNDLE/oracle/"*; do
  [[ -f "$file" ]] || continue
  digest="$(sha256sum "$file")"; printf '%s  %s/%s\n' "${digest%% *}" "$REMOTE" "${file#"$BUNDLE/"}"
done >"$STAGING/SOURCE.sha256"
cp "$STAGING/SOURCE.sha256" "$BUNDLE/SOURCE.sha256"
cp "$STAGING/SOURCE.sha256" "$RECEIPTS/source-sha256.txt"
rsync -a --checksum "$BUNDLE/" "$HOST:$REMOTE/"
set +e
ssh -o BatchMode=yes "$HOST" "timeout 540s bash $REMOTE/mimo26f-spark-unpack.sh $REMOTE $SOURCE $MODE $RESIDENTS $NCU_SUDO"
run_rc=$?
rsync -a --safe-links --include='*.log' --include='*.json' --include='*.txt' --include='*.csv' --exclude='*' "$HOST:$REMOTE/receipts/" "$RECEIPTS/"
copy_rc=$?
set -e
printf 'remote_exit=%s retrieve_exit=%s\n' "$run_rc" "$copy_rc"
# The verbose SDK catalog is about90MiB: archive in scratch, never glob-add it
# with text receipts. Keep its hash even when the GPU capture failed.
if [[ -f "$RECEIPTS/ncu-metrics.txt" ]]; then
  mkdir -p "$STAGING/ncu"
  mv "$RECEIPTS/ncu-metrics.txt" "$STAGING/ncu/"
  sha256sum "$STAGING/ncu/ncu-metrics.txt" >"$RECEIPTS/ncu-metrics-sha256.txt"
fi
[[ "$run_rc" == 0 ]] || exit "$run_rc"
[[ "$copy_rc" == 0 ]] || exit 2
if [[ "$MODE" == silu-exp ]]; then
  python3 "$TESTS/silu_exp_decimal.py" --input "$RECEIPTS/gpu-midpoints.csv" --output "$STAGING/gpu-decimal" --gpu-root "$RECEIPTS" --cpu-summary "$RECEIPTS/prior-cpu-summary.json" | tee "$RECEIPTS/decimal-audit.log"
  cp "$STAGING/gpu-decimal/decimal.json" "$RECEIPTS/gpu-decimal.json"
  echo 'RESULT: exhaustive exp/SiLU diagnostic COMPLETE; bitwise mismatch counts retained, no lattice or tolerance change'
  exit 0
fi
if [[ "$MODE" == i4-reference ]]; then
  mkdir "$STAGING/gpu-output"
  rsync -a --safe-links "$HOST:$REMOTE/gpu-output/" "$STAGING/gpu-output/"
  comparator="$REPO/crates/mimo26-repack/tests/lattice_compare.py"
  for tool in memcheck initcheck synccheck racecheck; do for m in 1 8; do
    candidate="$STAGING/gpu-output/$tool-m$m"
    python3 "$comparator" --seal-dump "$STAGING/oracle" "$candidate" "$m" "$SOURCE"
    cp "$candidate/b1-source.json" "$RECEIPTS/dump-$tool-m$m.json"
    python3 "$comparator" --compare "$STAGING/oracle" "$candidate" | tee "$RECEIPTS/compare-$tool-m$m.json"
  done; done
  candidate="$STAGING/gpu-output/negative"
  python3 "$comparator" --seal-dump "$STAGING/oracle" "$candidate" 8 "$SOURCE"
  cp "$candidate/b1-source.json" "$RECEIPTS/dump-negative.json"
  rc=0
  python3 "$comparator" --compare "$STAGING/oracle" "$candidate" >"$RECEIPTS/connected-negative-compare.json" || rc=$?
  [[ "$rc" == 3 ]] || { echo "FAIL connected numerical negative exit=$rc"; exit 2; }
  echo 'RESULT: PASS reference AOT3 classes and connected-v1 sanitizer M1/M8 plus independent comparisons; residency57/58 recorded, NO performance/full-model promotion'
  exit 0
fi
if [[ "$MODE" == b1-ffn ]]; then
  mkdir "$STAGING/gpu-output"
  rsync -a --safe-links "$HOST:$REMOTE/gpu-output/" "$STAGING/gpu-output/"
  comparator="$REPO/crates/mimo26-repack/tests/lattice_compare.py"
  for m in 1 2 3 4 5 6 7 8; do
    candidate="$STAGING/gpu-output/m$m"
    python3 "$comparator" --seal-dump "$STAGING/oracle" "$candidate" "$m" "$SOURCE"
    python3 "$comparator" --compare "$STAGING/oracle" "$candidate" | tee "$RECEIPTS/compare-m$m.json"
  done
  candidate="$STAGING/gpu-output/negative"
  python3 "$comparator" --seal-dump "$STAGING/oracle" "$candidate" 4 "$SOURCE"
  set +e
  python3 "$comparator" --compare "$STAGING/oracle" "$candidate" >"$RECEIPTS/negative-compare.json"
  rc=$?
  set -e
  [[ "$rc" == 3 ]] || { echo "FAIL double-weight detector exit=$rc"; exit 2; }
  echo 'RESULT: PASS connected B1 v1 real FFN M1..8, four ranks, byte-exact same-input codec and ordered return; no performance qualification'
  exit 0
fi
if [[ "$MODE" == b1-profile || "$MODE" == b2-profile || "$MODE" == b2-force4 ]]; then
  mkdir -p "$STAGING/ncu"
  rsync -a --safe-links --include='profile-*.ncu-rep' --exclude='*' "$HOST:$REMOTE/" "$STAGING/ncu/"
  family=B2; [[ "$MODE" != b1-profile ]] || family=B1
  analysis_rc=0
  for raw in "$RECEIPTS/"*.csv; do
    if python3 -B "$TESTS/ncu_report.py" --family "$family" "$raw" >"${raw%.csv}-summary.json"; then :; else analysis_rc=$?; fi
  done
  echo "RESULT: profiling capture complete; analysis_exit=$analysis_rc; missing counters remain explicit, no bandwidth promotion"
  exit "$analysis_rc"
fi
if [[ "$MODE" == b1-bench ]]; then echo 'RESULT: PASS B1 frozen M1..8 bandwidth gate; see fixed-scope native receipt'; exit 0; fi
if [[ "$MODE" == b1-sanitize ]]; then echo 'RESULT: PASS Spark B1 compute-only sanitizers; fused FFN sanitizer gate remains open'; exit 0; fi
if [[ "$MODE" == b1-primitive ]]; then echo 'RESULT: PASS Spark B1 container/MMA primitives only; not FFN/performance qualification'; exit 0; fi
if [[ "$MODE" == bench || "$MODE" == b2-exact ]]; then echo 'RESULT: PASS Spark measured B2 bandwidth target'; exit 0; fi
if [[ "$MODE" == wire ]]; then
  mkdir "$STAGING/gpu-output"
  rsync -a --safe-links --include='*.f32' --exclude='*' "$HOST:$REMOTE/gpu-output/" "$STAGING/gpu-output/"
  sha256sum "$STAGING/gpu-output/"*.f32 >"$RECEIPTS/gpu-artifact-sha256.txt"
  "$STAGING/wire-seam" --real "$STAGING/gpu-output" "$STAGING/oracle" 0 | tee "$RECEIPTS/real-wire.log"
  for flag in 1 2; do
    set +e
    "$STAGING/wire-seam" --real "$STAGING/gpu-output" "$STAGING/oracle" "$flag" >"$RECEIPTS/real-negative-$flag.log" 2>&1
    rc=$?
    set -e
    [[ "$rc" == 3 ]] && grep -q '^ORACLE_MISMATCH ' "$RECEIPTS/real-negative-$flag.log" || { echo "FAIL: real nibble/weight negative flag=$flag rc=$rc"; exit 1; }
  done
  echo 'RESULT: PASS real GPU eight-route weighted wire seam under R8; CPU rank pre-sum, not GPU reducer or performance qualification'
  exit 0
fi
python3 "$REPO/harness/nibble_proof.py" --fixture "$REPO/bench/fixtures/expert_nibble_fixture.json" --dump "$RECEIPTS/dump.json" | tee "$RECEIPTS/comparator.log"
for flag in 1 4 128; do
  set +e
  python3 "$REPO/harness/nibble_proof.py" --fixture "$REPO/bench/fixtures/expert_nibble_fixture.json" --dump "$RECEIPTS/negative-$flag.json" >"$RECEIPTS/comparator-negative-$flag.log" 2>&1
  rc=$?
  set -e
  [[ "$rc" == 1 ]] || { echo "External comparator failed to reject flag$flag (rc=$rc)"; exit 1; }
done
echo 'RESULT: PASS Spark unpack and independent dev-host comparator; text-only receipt retained'
