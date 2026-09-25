#!/usr/bin/env bash
# Invoked only through scripts/dev.sh test gemm spark-tensor-calib.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
TESTS="$REPO/crates/mimo26-expert/tests/gpu"; BUILD_ROOT="${MIMO26F_BUILD_ROOT:?use scripts/dev.sh}"
SOURCE="$(git -C "$REPO" rev-parse HEAD)"
owned=(crates/mimo26-expert/tests/gpu/tensor_calib.cu crates/mimo26-expert/tests/gpu/tensor_calib_remote.sh crates/mimo26-expert/tests/gpu/tensor_calib_run.sh crates/mimo26-expert/tests/gpu/tensor_calib_report.py)
git -C "$REPO" ls-files --error-unmatch -- "${owned[@]}" >/dev/null
git -C "$REPO" diff --quiet HEAD -- "${owned[@]}" || { echo 'Commit owned calibration sources before device execution'; exit 2; }
mkdir -p "$BUILD_ROOT"; exec 9>"$BUILD_ROOT/.cargo.lock";flock 9
STAGING="$(mktemp -d "$BUILD_ROOT/expert-calib.XXXXXX")"
TAG="$(TZ=Australia/Sydney date +%Y%m%d-%H%M%S)-${SOURCE:0:12}-$(basename "$STAGING")"
RECEIPTS="$REPO/runs/20260923-i4/expert/spark1-$TAG";mkdir "$RECEIPTS"
printf 'time=%s source=%s staging=%s receipts=%s mode=tensor-calib\n' "$(TZ=Australia/Sydney date --iso-8601=seconds)" "$SOURCE" "$STAGING" "$RECEIPTS" | tee "$RECEIPTS/launch.txt"
python3 -B "$TESTS/tensor_calib_report.py" --selftest | tee "$RECEIPTS/report-selftest.log"
# Stage the frozen B1 FP8xFP4 MMA primitive (62d1ca5) + the owned calibration CU.
mkdir -p "$STAGING/bundle"
git -C "$REPO" archive 62d1ca5797845f65e3ebc41ff180f499fff366cc crates/mimo26-expert/kernels/b1/mxfp4_ptx.cuh | tar -x -C "$STAGING"
cp "$STAGING/crates/mimo26-expert/kernels/b1/mxfp4_ptx.cuh" "$STAGING/bundle/"
cp "$TESTS/tensor_calib.cu" "$TESTS/tensor_calib_remote.sh" "$STAGING/bundle/"
REMOTE="$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark1 'mkdir -p /var/tmp/mimo26f-kernel && mktemp -d /var/tmp/mimo26f-kernel/calib-XXXXXXXX')"
[[ "$REMOTE" =~ ^/var/tmp/mimo26f-kernel/calib-[[:alnum:]]{8}$ ]] || exit 2
printf 'remote=%s\n' "$REMOTE" | tee "$RECEIPTS/remote.txt"
for file in "$STAGING/bundle/"*; do
  hash="$(sha256sum "$file")";printf '%s  %s/%s\n' "${hash%% *}" "$REMOTE" "$(basename "$file")"
done >"$STAGING/SOURCE.sha256"
cp "$STAGING/SOURCE.sha256" "$RECEIPTS/source-sha256.txt"
cp "$STAGING/SOURCE.sha256" "$STAGING/bundle/SOURCE.sha256"
rsync -a --checksum "$STAGING/bundle/" "spark1:$REMOTE/"
set +e
ssh -o BatchMode=yes spark1 "timeout 400s bash $REMOTE/tensor_calib_remote.sh $REMOTE $SOURCE"
run_rc=$?
rsync -a --safe-links --include='*.log' --include='*.txt' --exclude='*' "spark1:$REMOTE/receipts/" "$RECEIPTS/"
copy_rc=$?
set -e
printf 'remote_exit=%s retrieve_exit=%s\n' "$run_rc" "$copy_rc" | tee "$RECEIPTS/run-status.txt"
[[ "$copy_rc" == 0 ]] || exit "$copy_rc"
python3 -B "$TESTS/tensor_calib_report.py" "$RECEIPTS" "$RECEIPTS/analysis.json" | tee "$RECEIPTS/analysis.log"
exit "$run_rc"
