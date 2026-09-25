#!/usr/bin/env bash
# Arity-3 prefill prototype cell (M64-class up/gate GEMM), native-only spark1.
set -euo pipefail
ROOT="${1:?}"; SOURCE="${2:?}"
[[ "$ROOT" =~ ^/var/tmp/mimo26f-kernel/arity3pf-[[:alnum:]]{8}$ && "$SOURCE" =~ ^[0-9a-f]{40}$ ]] || exit 2
export TZ=Australia/Sydney LC_ALL=C
mkdir "$ROOT/receipts"
cd "$ROOT"
exec > >(tee "$ROOT/receipts/cell.log") 2>&1
exec 9>/var/tmp/mimo26f-kernel/.cell.lock
flock -n 9 || { echo 'Refuse concurrent kernel cell'; exit 2; }
printf 'time=%s source=%s host=%s mode=arity3-prefill\n' "$(date --iso-8601=seconds)" "$SOURCE" "$(hostname)"
[[ "$(hostname)" == "${MIMO26_SPARK1_HOST:-spark1}" ]] || { echo 'spark1 identity required'; exit 2; }
sha256sum -c "$ROOT/SOURCE.sha256"
guard() {
  local info uuid name owners available
  info="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader)"
  [[ "$info" != *$'\n'* ]] || exit 2
  IFS=, read -r uuid name <<<"$info"
  [[ "$name" == *GB10* ]] || exit 2
  export CUDA_VISIBLE_DEVICES="${uuid//[[:space:]]/}"
  [[ "$CUDA_VISIBLE_DEVICES" == GPU-fe473e1a-3f6c-0821-a880-d4d189de1678 ]] || exit 2
  owners="$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
  [[ -z "${owners//[[:space:]]/}" ]] || { echo "Refuse CUDA owners: $owners"; exit 2; }
  available="$(free -b | awk '$1=="Mem:" {print $7}')"
  [[ "$available" =~ ^[0-9]+$ && "$available" -ge 8589934592 ]] || exit 2
  printf '%s %s MemAvailable=%s owners=none\n' "$(date --iso-8601=seconds)" "$info" "$available" | tee -a "$ROOT/receipts/guards.log"
}
guard
NVCC=/usr/local/cuda/bin/nvcc
"$NVCC" --version | tee "$ROOT/receipts/nvcc.txt"
flags=(-O3 -std=c++17 -lineinfo --ftz=false --gpu-architecture=compute_121a --gpu-code=sm_121a -I"$ROOT")
printf 'ARITY3PF FLAGS ';printf '%q ' "${flags[@]}";printf '\n'
"$NVCC" "${flags[@]}" -Xptxas=-v "$ROOT/arity3_prefill.cu" -o "$ROOT/arity3-prefill" 2>&1 | tee "$ROOT/receipts/compile.log"
sha256sum "$ROOT/arity3-prefill" | tee "$ROOT/receipts/binaries-sha256.txt"
export MIMO26_BUILDER_GPU=1
guard
for i in 1 2 3; do nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv,noheader >>"$ROOT/receipts/clocks-before.txt"; sleep 1; done
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv,noheader -l 1 >>"$ROOT/receipts/clocks-during.txt" &
SAMPLE_PID=$!
set +e
timeout 300s "$ROOT/arity3-prefill" | tee "$ROOT/receipts/arity3pf.log"
codes=("${PIPESTATUS[@]}")
set -e
kill "$SAMPLE_PID" 2>/dev/null || true
wait "$SAMPLE_PID" 2>/dev/null || true
printf 'native_exit=%s tee_exit=%s\n' "${codes[0]}" "${codes[1]}" | tee "$ROOT/receipts/status.txt"
for i in 1 2 3; do nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv,noheader >>"$ROOT/receipts/clocks-after.txt"; sleep 1; done
guard
[[ "${codes[1]}" == 0 ]] || exit 3
[[ "${codes[0]}" == 0 ]] || exit "${codes[0]}"
printf 'ARITY3PF BATCH COMPLETE time=%s no_promotion=yes\n' "$(date --iso-8601=seconds)"
