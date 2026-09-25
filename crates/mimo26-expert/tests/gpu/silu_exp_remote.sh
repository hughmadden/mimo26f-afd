#!/usr/bin/env bash
# Sourced after the standard spark1 ownership/8GiB/source guards.
set -euo pipefail
export TMPDIR="$ROOT/tmp" TMP="$ROOT/tmp" TEMP="$ROOT/tmp" CUDA_CACHE_PATH="$ROOT/cuda-cache"
mkdir "$TMPDIR" "$CUDA_CACHE_PATH"
flags=(-O3 -std=c++17 -lineinfo --ftz=false --prec-div=true --gpu-architecture=compute_121a --gpu-code=sm_121a -Xcompiler=-ffp-contract=off -I"$ROOT" -I"$ROOT/include")
printf 'EXP PROBE FLAGS '; printf '%q ' "${flags[@]}"; printf '\n'
"$NVCC" "${flags[@]}" "$ROOT/silu_exp_gpu.cu" -o "$ROOT/mimo26f-exp-probe" 2>&1 | tee "$ROOT/receipts/exp-compile.log"
sha256sum "$ROOT/mimo26f-exp-probe" >"$ROOT/receipts/binary-sha256.txt"
CUDA_VISIBLE_DEVICES='' "$ROOT/mimo26f-exp-probe" --selftest | tee "$ROOT/receipts/exp-host.log"
guard
export MIMO26_BUILDER_GPU=1
set +e
timeout 480s "$ROOT/mimo26f-exp-probe" --gpu "$ROOT/receipts" "$ROOT/reference-map.tsv" | tee "$ROOT/receipts/exp-gpu.log"
status=("${PIPESTATUS[@]}")
set -e
guard
sha256sum -c "$ROOT/receipts/binary-sha256.txt"
printf 'EXP DIAGNOSTIC native_exit=%s tee_exit=%s; mismatch counts are findings, not waived gates\n' "${status[0]}" "${status[1]}"
[[ "${status[0]}" == 0 ]] || exit "${status[0]}"
[[ "${status[1]}" == 0 ]] || exit 2
