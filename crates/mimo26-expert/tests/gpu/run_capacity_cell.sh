#!/usr/bin/env bash
# Reached only under run_gpu_gemm.sh's dev.sh lock and unique staging directory.
set -euo pipefail
REPO="$1"; STAGING="$2"; CELL="$3"
NVCC=/usr/local/cuda-12.8/bin/nvcc
INCLUDE="$REPO/crates/mimo26-expert/kernels/include"
TESTS="$REPO/crates/mimo26-expert/tests/gpu"
COMMON=(-O3 -std=c++17 -arch=sm_89 -lineinfo --ftz=false -DM26X_BAKED_ARCH=89 -DM26X_BAKED_SMS=128 -I"$INCLUDE")
for script in "$TESTS/"*.sh; do bash -n "$script"; done
"$NVCC" --version | tee "$STAGING/nvcc-version.log"
for capacity in 256 2048 4096; do
  CLASS="$STAGING/class$capacity"; mkdir "$CLASS"
  "$NVCC" "${COMMON[@]}" -DM26X_CAPACITY_CLASS="$capacity" -Xptxas=-v \
    -c "$REPO/crates/mimo26-expert/kernels/expert_gemm.cu" -o "$CLASS/expert_gemm.o" 2>&1 | tee "$CLASS/compile.log"
  "$NVCC" "${COMMON[@]}" -DM26X_CAPACITY_CLASS="$capacity" -DM26X_WITH_OBJECT \
    "$TESTS/kernel_selftest.cpp" "$CLASS/expert_gemm.o" -o "$CLASS/plan-selftest"
  CUDA_VISIBLE_DEVICES='' "$CLASS/plan-selftest" 2>&1 | tee "$CLASS/plan-selftest.log"
  "$NVCC" "${COMMON[@]}" -DM26X_CAPACITY_CLASS="$capacity" -I"$TESTS" \
    "$REPO/crates/mimo26-expert/kernels/parity/gemm_parity.cu" "$CLASS/expert_gemm.o" \
    -o "$CLASS/gemm-parity" 2>&1 | tee "$CLASS/driver-compile.log"
  CUDA_VISIBLE_DEVICES='' "$CLASS/gemm-parity" selftest 2>&1 | tee "$CLASS/driver-selftest.log"
  CUDA_VISIBLE_DEVICES='' "$CLASS/gemm-parity" routing-selftest 2>&1 | tee "$CLASS/routing-selftest.log"
  echo "CAPACITY HOST PASS class=$capacity: compiled and audited synthetic fixture; NO GPU initialized"
done
if [[ "$CELL" == capacity-gpu ]]; then
  timeout 480s bash "$TESTS/run_unpack_cell.sh" "$STAGING/class2048/gemm-parity" '' '' "$STAGING" "$REPO" capacity
else
  echo 'RESULT: PASS capacity HOST checks only; all real-device AOT/routing results pending'
fi
