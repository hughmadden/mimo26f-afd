#!/usr/bin/env bash
# Invoked only through scripts/dev.sh test gemm spark-step-prefill (F5).
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
TESTS="$REPO/crates/mimo26-expert/tests/gpu"; BUILD_ROOT="${MIMO26F_BUILD_ROOT:?use scripts/dev.sh}"
SOURCE="$(git -C "$REPO" rev-parse HEAD)"
owned=(crates/mimo26-expert/tests/gpu/step_{prefill_b1.cu,prefill_b2.cu,prefill_plan.h,prefill_selftest.cpp,hist_prefill.json,prefill_run.sh,prefill_remote.sh,stage.sh,hist.h,gpu_common.cuh} crates/mimo26-repack/tests/{step_oracle.py,lattice_oracle.py})
git -C "$REPO" ls-files --error-unmatch -- "${owned[@]}" >/dev/null
git -C "$REPO" diff --quiet HEAD -- "${owned[@]}" || { echo 'Commit owned prefill sources before device execution'; exit 2; }
mkdir -p "$BUILD_ROOT"; exec 9>"$BUILD_ROOT/.cargo.lock";flock 9
STAGING="$(mktemp -d "$BUILD_ROOT/expert-spark-prefill.XXXXXX")"
TAG="$(TZ=Australia/Sydney date +%Y%m%d-%H%M%S)-${SOURCE:0:12}-$(basename "$STAGING")"
RECEIPTS="$REPO/runs/20260923-i4/expert/spark1-$TAG";mkdir "$RECEIPTS"
printf 'time=%s source=%s staging=%s receipts=%s mode=f5 cases=prefill-M1,prefill-M16,prefill-M64\n' "$(TZ=Australia/Sydney date --iso-8601=seconds)" "$SOURCE" "$STAGING" "$RECEIPTS" | tee "$RECEIPTS/launch.txt"
python3 -B "$TESTS/cuda_owner_guard.py" --selftest | tee "$RECEIPTS/guard-selftest.log"
python3 -B "$TESTS/step_report.py" --selftest | tee "$RECEIPTS/report-selftest.log"
c++ -O2 -std=c++17 -Wall -Wextra -Werror "$TESTS/step_prefill_selftest.cpp" -o "$STAGING/prefill-selftest"
"$STAGING/prefill-selftest" | tee "$RECEIPTS/host-selftest.log"
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B "$REPO/crates/mimo26-repack/tests/step_oracle.py" --selftest | tee "$RECEIPTS/oracle-selftest.log"
bash "$TESTS/step_stage.sh" "$REPO" "$STAGING"
BUNDLE="$STAGING/bundle"
cp "$TESTS/step_hist_prefill.json" "$BUNDLE/histogram.json"
cp "$BUNDLE/histogram.json" "$RECEIPTS/input.json"
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 480s python3 -B "$REPO/crates/mimo26-repack/tests/step_oracle.py" "${MIMO26_WEIGHTS_DIR:-$HOME/models/XiaomiMiMo/MiMo-V2.6-Flash-RL}" "$BUNDLE/oracle" | tee "$RECEIPTS/oracle.log"
cp "$BUNDLE/oracle/source.json" "$RECEIPTS/oracle-source.json"
cp "$TESTS/step_prefill_remote.sh" "$BUNDLE/"
sha256sum "$REPO/crates/mimo26-repack/tests/"{step_oracle.py,lattice_oracle.py,checkpoint_reader.py} "$REPO/oracle/lattice/quant_v1.py" "$REPO/spike/mxfp4.py" >"$RECEIPTS/oracle-producer-sha256.txt"
REMOTE="$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark1 'mkdir -p /var/tmp/mimo26f-kernel && mktemp -d /var/tmp/mimo26f-kernel/step-XXXXXXXX')"
[[ "$REMOTE" =~ ^/var/tmp/mimo26f-kernel/step-[[:alnum:]]{8}$ ]] || exit 2
printf 'remote=%s\n' "$REMOTE" | tee "$RECEIPTS/remote.txt"
for file in "$BUNDLE/"* "$BUNDLE/b1/"* "$BUNDLE/b1/b1/"* "$BUNDLE/b1/include/"* "$BUNDLE/b2/"* "$BUNDLE/oracle/"*; do
  [[ -f "$file" ]] || continue
  hash="$(sha256sum "$file")";printf '%s  %s/%s\n' "${hash%% *}" "$REMOTE" "${file#"$BUNDLE/"}"
done >"$STAGING/SOURCE.sha256"
cp "$STAGING/SOURCE.sha256" "$BUNDLE/SOURCE.sha256"
cp "$STAGING/SOURCE.sha256" "$RECEIPTS/source-sha256.txt"
rsync -a --checksum "$BUNDLE/" "spark1:$REMOTE/"
set +e
ssh -o BatchMode=yes spark1 "timeout 600s bash $REMOTE/step_prefill_remote.sh $REMOTE $SOURCE"
run_rc=$?
rsync -a --safe-links --include='*.log' --include='*.json' --include='*.txt' --exclude='*' "spark1:$REMOTE/receipts/" "$RECEIPTS/"
copy_rc=$?
set -e
printf 'remote_exit=%s retrieve_exit=%s\n' "$run_rc" "$copy_rc" | tee "$RECEIPTS/status.txt"
[[ "$copy_rc" == 0 ]] || exit "$copy_rc"
if [[ -f "$RECEIPTS/cell.log" ]] && [[ "$(stat -c %s "$RECEIPTS/cell.log")" -gt 5000000 ]]; then
  mv "$RECEIPTS/cell.log" "$STAGING/native-cell.log"
  sha256sum "$STAGING/native-cell.log" >"$RECEIPTS/native-cell-sha256.txt"
fi
exit "$run_rc"
