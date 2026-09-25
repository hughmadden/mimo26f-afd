#!/usr/bin/env bash
# F5 native-wide B2 prefill diagnostic, native-only spark1. No remote Python/Rust.
set -euo pipefail
ROOT="${1:?}"; SOURCE="${2:?}"
[[ "$ROOT" =~ ^/var/tmp/mimo26f-kernel/step-[[:alnum:]]{8}$ && "$SOURCE" =~ ^[0-9a-f]{40}$ ]] || exit 2
export TZ=Australia/Sydney LC_ALL=C
mkdir "$ROOT/receipts"
exec > >(tee "$ROOT/receipts/cell.log") 2>&1
exec 9>/var/tmp/mimo26f-kernel/.cell.lock
flock -n 9 || { echo 'Refuse concurrent kernel cell'; exit 2; }
printf 'time=%s source=%s host=%s mode=f5\n' "$(date --iso-8601=seconds)" "$SOURCE" "$(hostname)"
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
b="$ROOT/b2"
flags=(-O3 -std=c++17 -lineinfo --ftz=false -arch=sm_121 -DM26X_BAKED_ARCH=121 -DM26X_BAKED_SMS=48 -DM26X_CAPACITY_CLASS=2048 -DM26X_EXACT_M=1 -I"$b")
printf 'F5 FLAGS ';printf '%q ' "${flags[@]}";printf '\n'
"$NVCC" "${flags[@]}" -Xptxas=-v -c "$b/expert_gemm.cu" -o "$ROOT/b2.o" 2>&1 | tee "$ROOT/receipts/b2-compile.log"
"$NVCC" "${flags[@]}" "$b/step_prefill_b2.cu" "$ROOT/b2.o" -o "$ROOT/step-prefill-b2" 2>&1 | tee "$ROOT/receipts/prefill-driver.log"
b="$ROOT/b1"
flags=(-O3 -std=c++17 -lineinfo --ftz=false --gpu-architecture=compute_121a --gpu-code=sm_121a -DM26B1_TARGET_ARCH=121 -DM26B1_ASYNC_SCALE_FC1=1 -DM26B1_ASYNC_SCALE_FC2=1 -I"$b")
printf 'F5 B1 FLAGS ';printf '%q ' "${flags[@]}";printf '\n'
"$NVCC" "${flags[@]}" "$b/step_prefill_b1.cu" -o "$ROOT/step-prefill-b1" 2>&1 | tee "$ROOT/receipts/prefill-b1-driver.log"
sha256sum "$ROOT/step-prefill-b2" "$ROOT/step-prefill-b1" | tee "$ROOT/receipts/binaries-sha256.txt"
export MIMO26_BUILDER_GPU=1
weights=/var/tmp/models/MiMo-V2.6-Flash-RL
for name in prefill-M1 prefill-M16 prefill-M64; do
  guard
  set +e
  timeout 300s "$ROOT/step-prefill-b2" "$weights" "$ROOT/oracle" "$ROOT/histogram.json" "$name" diagnostic | tee "$ROOT/receipts/B2-$name.log"
  codes=("${PIPESTATUS[@]}")
  set -e
  printf 'native_exit=%s tee_exit=%s\n' "${codes[0]}" "${codes[1]}" | tee "$ROOT/receipts/B2-$name-status.txt"
  guard
  [[ "${codes[1]}" == 0 ]] || exit 3
  [[ "${codes[0]}" == 0 ]] || exit "${codes[0]}"
done
for name in prefill-M16 prefill-M64; do
  guard
  set +e
  timeout 300s "$ROOT/step-prefill-b1" "$weights" "$ROOT/oracle" "$ROOT/histogram.json" "$name" diagnostic | tee "$ROOT/receipts/B1-$name.log"
  codes=("${PIPESTATUS[@]}")
  set -e
  printf 'native_exit=%s tee_exit=%s\n' "${codes[0]}" "${codes[1]}" | tee "$ROOT/receipts/B1-$name-status.txt"
  guard
  [[ "${codes[1]}" == 0 ]] || exit 3
  [[ "${codes[0]}" == 0 ]] || exit "${codes[0]}"
done
guard
printf 'F5 BATCH COMPLETE time=%s no_promotion=yes\n' "$(date --iso-8601=seconds)"
