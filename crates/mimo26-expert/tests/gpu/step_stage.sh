#!/usr/bin/env bash
# Source staging only; invoked by the serialized owned gemm cell.
set -euo pipefail
REPO="$1"; STAGING="$2"; TESTS="$REPO/crates/mimo26-expert/tests/gpu"
for spec in b1:62d1ca5797845f65e3ebc41ff180f499fff366cc b2:769610ae82b463764d4a7b25ca61631afd5afc85; do
  family="${spec%%:*}"; revision="${spec#*:}"; pin="$STAGING/$family-pin"; out="$STAGING/bundle/$family"
  mkdir -p "$pin" "$out"
  git -C "$REPO" archive "$revision" crates/mimo26-expert | tar -x -C "$pin"
  base="$pin/crates/mimo26-expert"
  cp "$base/kernels/"*.cu "$base/kernels/include/"*.h "$base/kernels/include/"*.cuh "$out/"
  cp "$base/kernels/parity/gemm_parity.cu" "$base/kernels/parity/"*.cuh "$out/"
  cp "$base/tests/gpu/"{fixture_io.h,tp4_io.h,routing_proof.h,bench_io.h} "$out/"
  if [[ "$family" == b1 ]]; then
    cp "$base/kernels/b1/"{mxfp4_ptx.cuh,grouped.cu,LICENSE.b1-compute} "$base/tests/b1_core/"{primitive.cu,compute_test.cuh,rank_compute_test.cuh,ffn_test.cuh,bench_connected.cuh} "$out/"
    cp -a "$base/kernels/b1" "$out/b1"
    cp -a "$base/kernels/include" "$out/include"
  fi
  cp "$TESTS/step_$family.cu" "$TESTS/step_hist.h" "$TESTS/step_gpu_common.cuh" "$out/"
  printf '%s\n' "$revision" >"$out/PIN.txt"
done
cp "$TESTS/step_prefill_b2.cu" "$TESTS/step_prefill_plan.h" "$STAGING/bundle/b2/"
cp "$TESTS/step_prefill_b1.cu" "$TESTS/step_prefill_plan.h" "$STAGING/bundle/b1/"
cp "$TESTS/step_b2_mixed.cu" "$STAGING/bundle/b2/"
cp "$REPO/crates/mimo26-expert/kernels/mixed_dispatch.cuh" "$STAGING/bundle/b2/"
cp "$TESTS/step_hist_synthetic.json" "$STAGING/bundle/histogram.json"
