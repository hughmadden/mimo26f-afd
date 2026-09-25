#!/usr/bin/env bash
# Invoked only from the crate-owned dev.sh dispatcher, already serialized.
set -euo pipefail
REPO="${1:?repo}"; STAGING="${2:?slot}"
NVCC=/usr/local/cuda-12.8/bin/nvcc
"$NVCC" --version | tee "$STAGING/nvcc.log"
"$NVCC" -O3 -std=c++17 -lineinfo --ftz=false --gpu-architecture=compute_120a --gpu-code=sm_120a -DM26B1_TARGET_ARCH=120 \
  -I"$REPO/crates/mimo26-expert/kernels" -I"$REPO/crates/mimo26-expert/kernels/b1" -I"$REPO/crates/mimo26-expert/kernels/include" -I"$REPO/crates/mimo26-expert/tests/gpu" \
  "$REPO/crates/mimo26-expert/tests/b1_core/primitive.cu" -o "$STAGING/mimo26f-b1-primitive" 2>&1 | tee "$STAGING/compile.log"
CUDA_VISIBLE_DEVICES='' "$STAGING/mimo26f-b1-primitive" --selftest | tee "$STAGING/host.log"
CUDA_VISIBLE_DEVICES='' python3 -B "$REPO/crates/mimo26-repack/tests/lattice_bench.py" --selftest | tee "$STAGING/bench-oracle-selftest.log"
echo 'COMPILE/HOST PASS: sm_120a syntax only; NO GPU launched; sm_121a Spark proof still required'
