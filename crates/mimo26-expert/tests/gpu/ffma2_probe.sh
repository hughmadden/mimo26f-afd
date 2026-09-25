#!/usr/bin/env bash
# Owned compile-only probe. Remote runs only nvcc/cuobjdump; never generated code.
set -euo pipefail
if [[ "${1:-}" == --remote ]]; then
  ROOT="${2:?}"; SOURCE="${3:?}"
  [[ "$ROOT" == /var/tmp/mimo26f-kernel/ffma2-* && -d "$ROOT" ]] || exit 2
  export CUDA_VISIBLE_DEVICES='' TZ=Australia/Sydney LC_ALL=C
  mkdir "$ROOT/receipts"
  exec > >(tee "$ROOT/receipts/remote.log") 2>&1
  exec 9>/var/tmp/mimo26f-kernel/.cell.lock
  flock -n 9 || { echo 'Refuse concurrent owned cell'; exit 2; }
  avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  [[ "$avail" =~ ^[0-9]+$ && "$avail" -ge 8388608 ]] || exit 2
  printf 'time=%s source=%s host=%s MemAvailable_KiB=%s compile_only=yes GPU_queries=no GPU_launches=no\n' "$(date --iso-8601=seconds)" "$SOURCE" "$(hostname)" "$avail"
  sha256sum -c "$ROOT/SOURCE.sha256"
  NVCC=/usr/local/cuda/bin/nvcc; DUMP=/usr/local/cuda/bin/cuobjdump
  "$NVCC" --version
  "$NVCC" --list-gpu-arch >"$ROOT/receipts/virtual-architectures.txt"
  "$NVCC" --list-gpu-code >"$ROOT/receipts/real-architectures.txt"
  grep -n -B12 -A4 '__ffma2_rn' /usr/local/cuda/include/crt/sm_100_rt.h >"$ROOT/receipts/sdk-declaration.txt"
  printf 'arch\tmode\tptx_rc\tcubin_rc\tsass_rc\n' >"$ROOT/receipts/status.tsv"
  for arch in 100a 121a; do
    for mode in 0 1 2; do
      stem="$ROOT/receipts/$arch-$mode"
      FLAGS=(-O3 -std=c++17 --ftz=false -DPROBE_MODE="$mode")
      printf 'PTX command: %q ' "$NVCC"; printf '%q ' "${FLAGS[@]}" -arch="compute_$arch" --ptx "$ROOT/ffma2_probe.cu" -o "$stem.ptx"; printf '\n'
      p=0; timeout 40s "$NVCC" "${FLAGS[@]}" -arch="compute_$arch" --ptx "$ROOT/ffma2_probe.cu" -o "$stem.ptx" >"$stem-ptx.log" 2>&1 || p=$?
      printf 'CUBIN command: %q ' "$NVCC"; printf '%q ' "${FLAGS[@]}" --gpu-architecture="compute_$arch" --gpu-code="sm_$arch" --cubin -Xptxas=-v "$ROOT/ffma2_probe.cu" -o "$stem.cubin"; printf '\n'
      c=0; timeout 40s "$NVCC" "${FLAGS[@]}" --gpu-architecture="compute_$arch" --gpu-code="sm_$arch" --cubin -Xptxas=-v "$ROOT/ffma2_probe.cu" -o "$stem.cubin" >"$stem-cubin.log" 2>&1 || c=$?
      s=99
      if [[ "$c" == 0 ]]; then
        s=0; "$DUMP" --dump-sass "$stem.cubin" >"$stem.sass" 2>"$stem-sass.log" || s=$?
        sha256sum "$stem.cubin" >>"$ROOT/receipts/cubins-sha256.txt"
      fi
      printf '%s\t%s\t%s\t%s\t%s\n' "$arch" "$mode" "$p" "$c" "$s" >>"$ROOT/receipts/status.tsv"
    done
  done
  echo 'COMPILE PROBE FINISHED; no executable or GPU kernel was run'
  exit 0
fi
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
TESTS="$REPO/crates/mimo26-expert/tests/gpu"
if [[ "${1:-}" == --selftest ]]; then
  bash -n "${BASH_SOURCE[0]}"
  python3 -B "$TESTS/ffma2_probe_report.py" --selftest
  if [[ -n "${MIMO26_FFMA2_REPORT_DIR:-}" ]]; then
    python3 -B "$TESTS/ffma2_probe_report.py" "$MIMO26_FFMA2_REPORT_DIR"
  fi
  exit 0
fi
[[ "${1:-}" == --run && "${HOST:-}" == spark1 ]] || { echo 'compile-only probe requires --run and HOST=spark1'; exit 2; }
files=(ffma2_probe.sh ffma2_probe.cu ffma2_probe_report.py run_gpu_gemm.sh)
for f in "${files[@]}"; do
  [[ -z "$(git -C "$REPO" status --porcelain -- "$TESTS/$f")" ]] || { echo "Commit probe source first: $f"; exit 2; }
done
python3 -B "$TESTS/ffma2_probe_report.py" --selftest
SOURCE="$(git -C "$REPO" rev-parse HEAD)"
STAGING="$(mktemp -d "${MIMO26F_BUILD_ROOT:?dev.sh required}/expert-ffma2.XXXXXX")"
R="$REPO/runs/20260923-i4/expert/spark1-$(TZ=Australia/Sydney date +%Y%m%d-%H%M%S)-${SOURCE:0:12}-$(basename "$STAGING")"
mkdir "$R" "$STAGING/bundle"
exec > >(tee "$R/local.log") 2>&1
printf 'source=%s staging=%s receipts=%s\n' "$SOURCE" "$STAGING" "$R"
for f in "${files[@]}"; do cp "$TESTS/$f" "$STAGING/bundle/"; done
REMOTE="$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark1 'mkdir -p /var/tmp/mimo26f-kernel && mktemp -d /var/tmp/mimo26f-kernel/ffma2-XXXXXXXX')"
printf '%s\n' "$REMOTE" >"$R/remote.txt"
for f in "${files[@]}"; do
  sha="$(sha256sum "$STAGING/bundle/$f")"; printf '%s  %s/%s\n' "${sha%% *}" "$REMOTE" "$f"
done >"$STAGING/bundle/SOURCE.sha256"
cp "$STAGING/bundle/SOURCE.sha256" "$R/source-sha256.txt"
rsync -a --checksum "$STAGING/bundle/" "spark1:$REMOTE/"
rc=0
ssh -o BatchMode=yes spark1 "timeout 300s bash $REMOTE/ffma2_probe.sh --remote $REMOTE $SOURCE" || rc=$?
rsync -a --safe-links --exclude='*.cubin' "spark1:$REMOTE/receipts/" "$R/"
mkdir "$STAGING/cubins"
rsync -a --safe-links --include='*.cubin' --exclude='*' "spark1:$REMOTE/receipts/" "$STAGING/cubins/"
[[ "$rc" == 0 ]] || exit "$rc"
python3 -B "$TESTS/ffma2_probe_report.py" "$R" | tee "$R/analysis.log"
