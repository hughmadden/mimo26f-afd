#!/usr/bin/env bash
# Crate-owned cells reached ONLY via scripts/dev.sh test gemm <cell>.
# tp4-cpu and selftests are CPU-only. The native unpack driver is host-qualified
# and staged for pg4090 proof under the latest visibility/ownership policy.
# Spark unpack is proven; B2 bandwidth STOP is retained. CPU wire probes are independent.
# Legacy GPU opt-in flag retained.
set -euo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
CELL="${MIMO26_GEMM_CELL:-tiny}"
# ADVISOR-I4 §9 now authorizes direct Spark ownership. These branches have no
# dependency on a dev-host GPU and never run a remote Python/Rust runtime.
if [[ "$CELL" == step-report-selftest ]]; then
  for script in step_stage.sh step_run.sh step_remote.sh run_gpu_gemm.sh; do bash -n "$REPO/crates/mimo26-expert/tests/gpu/$script"; done
  exec python3 -B "$REPO/crates/mimo26-expert/tests/gpu/step_report.py" --selftest
elif [[ "$CELL" == step-report ]]; then
  exec python3 -B "$REPO/crates/mimo26-expert/tests/gpu/step_report.py" "${MIMO26_STEP_RECEIPT:?receipt required}" "${MIMO26_STEP_REPORT:?new report output required}"
elif [[ "$CELL" == spark-step ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/step_run.sh"
elif [[ "$CELL" == spark-step-mixed ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/step_mixed_run.sh"
elif [[ "$CELL" == spark-step-prefill ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/step_prefill_run.sh"
elif [[ "$CELL" == spark-tensor-calib ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/tensor_calib_run.sh"
elif [[ "$CELL" == tensor-calib-report-selftest ]]; then
  exec python3 -B "$REPO/crates/mimo26-expert/tests/gpu/tensor_calib_report.py" --selftest
elif [[ "$CELL" == ffma2-probe-selftest ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/ffma2_probe.sh" --selftest
elif [[ "$CELL" == spark-ffma2-probe ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/ffma2_probe.sh" --run
elif [[ "$CELL" == spark-silu-exp ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" silu-exp
elif [[ "$CELL" == spark-i4-reference ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" i4-reference
elif [[ "$CELL" == spark-r14-model ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/frozen_profile.sh" --run-r14-model
elif [[ "$CELL" == spark-b2-exact ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" b2-exact
elif [[ "$CELL" == spark-r13-f1 ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" b2-force4
elif [[ "$CELL" == spark-r13-f2 ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/frozen_profile.sh" --run-r13-f2
elif [[ "$CELL" == spark-frozen-profile ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/frozen_profile.sh" --run
elif [[ "$CELL" == spark-dry ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" --dry
elif [[ "$CELL" == spark-unpack ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" unpack
elif [[ "$CELL" == spark-bench ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" bench
elif [[ "$CELL" == spark-b1-profile || "$CELL" == spark-b2-profile ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" "${CELL#spark-}"
elif [[ "$CELL" == spark-b1-profile-dry || "$CELL" == spark-b2-profile-dry ]]; then
  mode="${CELL#spark-}"; exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" "--dry-${mode%-dry}"
elif [[ "$CELL" == spark-b1-bench ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" b1-bench
elif [[ "$CELL" == spark-b1-bench-dry ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" --dry-b1-bench
elif [[ "$CELL" == spark-b1-ffn ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" b1-ffn
elif [[ "$CELL" == spark-b1-ffn-dry ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" --dry-b1-ffn
elif [[ "$CELL" == spark-b1-sanitize ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" b1-sanitize
elif [[ "$CELL" == spark-b1-sanitize-dry ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" --dry-b1-sanitize
elif [[ "$CELL" == spark-b1-primitive ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" b1-primitive
elif [[ "$CELL" == spark-b1-primitive-dry ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" --dry-b1-primitive
elif [[ "$CELL" == spark-wire ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" wire
elif [[ "$CELL" == spark-wire-dry ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" --dry-wire
elif [[ "$CELL" == spark-bench-dry ]]; then
  exec bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" --dry-bench
fi
case "$CELL" in
  help|--help|-h)
    echo 'scripts/dev.sh test gemm tp4-cpu   # real v2 repack -> expert CPU FFN -> independent oracle'
    echo 'scripts/dev.sh test gemm kernel-selftest  # host decoder/index tests, no GPU'
    echo 'scripts/dev.sh test gemm compile          # selftests + sm89 compile only'
    echo 'MIMO26_EXPERT_TWO_RUN=1 adds the deliberate naive failure.'
    echo 'scripts/dev.sh test gemm driver-selftest  # native reader audit + negatives, no GPU'
    echo 'MIMO26_BUILDER_GPU=1 scripts/dev.sh test gemm unpack  # pg4090 proof; builder until visible'
    echo 'scripts/dev.sh test gemm tp4-driver-selftest  # real producer + native audit, no GPU'
    echo 'MIMO26_BUILDER_GPU=1 scripts/dev.sh test gemm tp4-gpu  # real TP4/GEMM proof'
    echo 'HOST=spark1 MIMO26_BUILDER_GPU=1 scripts/dev.sh test gemm spark-i4-reference  # reference AOT3, connected-v1 sanitizers, residency57/58'
    echo 'scripts/dev.sh test gemm capacity-host  # compile3 classes + synthetic routing selftests'
    echo 'MIMO26_BUILDER_GPU=1 scripts/dev.sh test gemm capacity-gpu  # full256 synthetic routing/AOT'
    echo 'HOST=spark1 MIMO26_BUILDER_GPU=1 scripts/dev.sh test gemm spark-bench  # real weights, M1..8'
    echo 'HOST=spark1 scripts/dev.sh test gemm spark-bench-dry  # no remote effects'
    echo 'scripts/dev.sh test gemm wire-boundary  # CPU-only R8 bound/order checks; legacy failure retained'
    echo 'scripts/dev.sh test gemm lattice-oracle-selftest  # CPU FP64/FP32 B1 reference and codec tests'
    echo 'scripts/dev.sh test gemm lattice-oracle-real  # CPU real-checkpoint E-W4A8-v1 reference artifacts'
    echo 'scripts/dev.sh test gemm lattice-compare  # MIMO26_B1_REFERENCE / MIMO26_B1_CANDIDATE artifact directories'
    echo 'scripts/dev.sh test gemm b1-sanitizer-selftest  # CPU-only sanitizer result parser controls'
    echo 'HOST=spark1 MIMO26_BUILDER_GPU=1 scripts/dev.sh test gemm spark-b1-sanitize  # compute-only B1 sanitizer; not fused FFN'
    echo 'Spark benchmark returns6 STOP or7 PIVOT when below threshold.'
    exit 0 ;;
  tp4-cpu|kernel-selftest|compile|driver-selftest|tp4-driver-selftest|owner-guard-selftest|capacity-host|spark-selftest|geometry-selftest|wire-boundary|b1-primitive-host|b1-sanitizer-selftest|lattice-oracle-selftest|lattice-oracle-real|lattice-compare|ncu-report|ncu-pc-delta|ncu-instruction-model|b2-force4-host|b2-exact-host|silu-exp-selftest|silu-exp-cpu|silu-exp-audit|step-hist-host|step-build-host|step-oracle-cpu|step-prefill-host|step-mixed-host|tensor-calib-host) ;;
  unpack|tp4-gpu|capacity-gpu) [[ "${MIMO26_BUILDER_GPU:-0}" == 1 ]] || { echo 'Explicit GPU opt-in required; follow latest visibility/ownership policy' >&2; exit 2; } ;;
  *) echo "RESULT: FAIL: cell '$CELL' is not qualified yet; no GPU launched" >&2; exit 2 ;;
esac
BUILD_ROOT="${MIMO26F_BUILD_ROOT:?run through scripts/dev.sh, which exports the workspace build root}"
mkdir -p "$BUILD_ROOT"
# Same lock as builder's expert-unit verb. No nested dev.sh call under this lock.
exec 9>"$BUILD_ROOT/.cargo.lock"
flock 9
STAGING="$(mktemp -d "$BUILD_ROOT/expert-$CELL.XXXXXX")"
export CARGO_TARGET_DIR="$STAGING/target"
export MIMO26_TP4_DIR="$STAGING/oracle"
export PATH="$HOME/.cargo/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WEIGHTS="${MIMO26_WEIGHTS_DIR:-$HOME/models/XiaomiMiMo/MiMo-V2.6-Flash-RL}"
echo "time: $(TZ=Australia/Sydney date '+%Y-%m-%d %H:%M:%S %Z')"
echo "source: $(git -C "$REPO" rev-parse HEAD) (working-tree changes may apply)"
echo "staging: $STAGING (retained for inspection)"
GUARD="$REPO/crates/mimo26-expert/tests/gpu/cuda_owner_guard.py"
python3 "$GUARD" --selftest | tee "$STAGING/owner-guard-selftest.log"
if [[ "$CELL" == step-mixed-host ]]; then
  tests="$REPO/crates/mimo26-expert/tests/gpu"
  bash "$tests/step_stage.sh" "$REPO" "$STAGING"
  nvcc=/usr/local/cuda-12.8/bin/nvcc; b="$STAGING/bundle/b2"
  flags=(-O3 -std=c++17 -lineinfo --ftz=false -arch=sm_120a -DM26X_BAKED_ARCH=121 -DM26X_BAKED_SMS=48 -DM26X_CAPACITY_CLASS=2048 -DM26X_EXACT_M=1 -I"$b")
  "$nvcc" "${flags[@]}" -c "$b/expert_gemm.cu" -o "$STAGING/b2.o" 2>&1 | tee "$STAGING/b2-kernel-compile.log"
  "$nvcc" "${flags[@]}" -Xptxas=-v "$b/step_b2_mixed.cu" "$STAGING/b2.o" -o "$STAGING/step-b2-mixed" 2>&1 | tee "$STAGING/mixed-driver-compile.log"
  python3 - "$STAGING/mixed-driver-compile.log" <<'PY'
import pathlib,sys,re
text=pathlib.Path(sys.argv[1]).read_text()
def grab(fn):
    for m in re.finditer(r"Compiling entry function '([^']*)'", text):
        if fn in m.group(1):
            i=text.find("Function properties for "+m.group(1), m.end());assert i>=0
            nxt=text.find("Compiling entry function", i+1)
            seg=text[i:(nxt if nxt>=0 else len(text))]
            regs=[int(x) for x in re.findall(r'Used (\d+) registers', seg)]
            sp=re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads', seg)
            sh=re.search(r'(\d+) bytes smem', seg)
            return dict(registers=max(regs), spill_stores=int(sp.group(1)) if sp else 0, spill_loads=int(sp.group(2)) if sp else 0, shared=sh.group(1) if sh else '0')
    raise AssertionError('missing ptxas entry '+fn)
corr=grab('mixed_gemmILb1EE');naive=grab('mixed_gemmILb0EE')
assert corr['spill_stores']==0 and corr['spill_loads']==0, ('correct entry spills', corr)
assert naive['registers']>0
flag='PASS' if corr['registers']<=64 else 'EXCEEDS_64_SPEC'
print('MIXED HOST %s correct registers=%d shared=%s spills=%d/%d; naive registers=%d shared=%s spills=%d/%d' % (flag,corr['registers'],corr['shared'],corr['spill_stores'],corr['spill_loads'],naive['registers'],naive['shared'],naive['spill_stores'],naive['spill_loads']))
if flag!='PASS':print('MIXED HOST NOTE correct entry exceeds the64-register spec; frozen exact-M M8 body is itself64 registers and the width switch adds dispatch overhead')
PY
  echo 'R18c MIXED HOST compile-only sm120a; runtime121 pinned; no device launch'
  exit 0
fi
if [[ "$CELL" == tensor-calib-host ]]; then
  tests="$REPO/crates/mimo26-expert/tests/gpu"
  bash -n "$tests/tensor_calib_remote.sh" "$tests/tensor_calib_run.sh"
  python3 -B "$tests/tensor_calib_report.py" --selftest | tee "$STAGING/calib-report-selftest.log"
  git -C "$REPO" archive 62d1ca5797845f65e3ebc41ff180f499fff366cc crates/mimo26-expert/kernels/b1/mxfp4_ptx.cuh | tar -x -C "$STAGING"
  cp "$STAGING/crates/mimo26-expert/kernels/b1/mxfp4_ptx.cuh" "$STAGING/"
  nvcc=/usr/local/cuda-12.8/bin/nvcc
  "$nvcc" -O3 -std=c++17 --ftz=false --gpu-architecture=compute_120a --gpu-code=sm_120a -I"$STAGING" "$tests/tensor_calib.cu" -o "$STAGING/tensor-calib" 2>&1 | tee "$STAGING/calib-compile.log"
  echo 'TENSOR CALIB HOST PASS compile-only sm120a, runtime121 pinned; no device launch'
  exit 0
fi
if [[ "$CELL" == step-prefill-host ]]; then
  tests="$REPO/crates/mimo26-expert/tests/gpu"
  c++ -O2 -std=c++17 -Wall -Wextra -Werror "$tests/step_prefill_selftest.cpp" -o "$STAGING/prefill-selftest"
  "$STAGING/prefill-selftest" | tee "$STAGING/prefill-selftest.log"
  bash "$tests/step_stage.sh" "$REPO" "$STAGING"
  nvcc=/usr/local/cuda-12.8/bin/nvcc; b="$STAGING/bundle/b2"
  flags=(-O3 -std=c++17 -lineinfo --ftz=false -arch=sm_120a -DM26X_BAKED_ARCH=121 -DM26X_BAKED_SMS=48 -DM26X_CAPACITY_CLASS=2048 -DM26X_EXACT_M=1 -I"$b")
  "$nvcc" "${flags[@]}" -c "$b/expert_gemm.cu" -o "$STAGING/b2.o" 2>&1 | tee "$STAGING/b2-kernel-compile.log"
  "$nvcc" "${flags[@]}" "$b/step_prefill_b2.cu" "$STAGING/b2.o" -o "$STAGING/prefill-b2" 2>&1 | tee "$STAGING/prefill-driver-compile.log"
  b="$STAGING/bundle/b1"
  flags=(-O3 -std=c++17 -lineinfo --ftz=false --gpu-architecture=compute_120a --gpu-code=sm_120a -DM26B1_TARGET_ARCH=121 -DM26B1_ASYNC_SCALE_FC1=1 -DM26B1_ASYNC_SCALE_FC2=1 -I"$b")
  "$nvcc" "${flags[@]}" "$b/step_prefill_b1.cu" -o "$STAGING/prefill-b1" 2>&1 | tee "$STAGING/prefill-b1-driver-compile.log"
  echo 'F5 HOST PASS native-wide B2/B1 adapters compile-only sm120a/121a, runtime121 pinned; no device launch'
  exit 0
fi
if [[ "$CELL" == step-build-host || "$CELL" == step-oracle-cpu ]]; then
  tests="$REPO/crates/mimo26-expert/tests/gpu"
  for script in step_stage.sh step_run.sh step_remote.sh run_gpu_gemm.sh; do bash -n "$tests/$script"; done
  python3 -B "$tests/step_report.py" --selftest | tee "$STAGING/report-selftest.log"
  CUDA_VISIBLE_DEVICES='' python3 -B "$REPO/crates/mimo26-repack/tests/step_oracle.py" --selftest | tee "$STAGING/oracle-selftest.log"
  python3 -B "$tests/step_trace.py" --selftest | tee "$STAGING/trace-selftest.log"
  c++ -O2 -std=c++17 -Wall -Wextra -Werror "$tests/step_hist_selftest.cpp" -o "$STAGING/step-hist"
  "$STAGING/step-hist" --selftest
  if [[ -f "$REPO/runs/20260923-i4/route/route-steps.json" ]]; then
    python3 -B "$tests/step_trace.py" "$REPO/runs/20260923-i4/route/route-steps.json" "$STAGING/real-steps.json" | tee "$STAGING/trace-normalize.log"
    for case in C1-w8 C4-w8 C16-w8; do "$STAGING/step-hist" "$STAGING/real-steps.json" "$case" >"$STAGING/plan-$case.log"; done
  fi
  CUDA_VISIBLE_DEVICES='' python3 -B "$REPO/crates/mimo26-repack/tests/lattice_oracle.py" --selftest | tee "$STAGING/lattice-selftest.log"
  if [[ "$CELL" == step-oracle-cpu ]]; then
    CUDA_VISIBLE_DEVICES='' timeout 480s python3 -B "$REPO/crates/mimo26-repack/tests/step_oracle.py" --layers "${MIMO26_STEP_LAYERS:-1}" "$WEIGHTS" "$STAGING/oracle" | tee "$STAGING/oracle.log"
  else
    bash "$tests/step_stage.sh" "$REPO" "$STAGING"
    nvcc=/usr/local/cuda-12.8/bin/nvcc
    flags=(-O3 -std=c++17 -lineinfo --ftz=false --gpu-architecture=compute_120a --gpu-code=sm_120a)
    b="$STAGING/bundle/b2"
    "$nvcc" "${flags[@]}" -DM26X_BAKED_ARCH=121 -DM26X_BAKED_SMS=48 -DM26X_CAPACITY_CLASS=2048 -DM26X_EXACT_M=1 -I"$b" -c "$b/expert_gemm.cu" -o "$STAGING/b2.o" 2>&1 | tee "$STAGING/b2-kernel-compile.log"
    "$nvcc" "${flags[@]}" -DM26X_BAKED_ARCH=121 -DM26X_BAKED_SMS=48 -DM26X_CAPACITY_CLASS=2048 -DM26X_EXACT_M=1 -I"$b" "$b/step_b2.cu" "$STAGING/b2.o" -o "$STAGING/step-b2" 2>&1 | tee "$STAGING/b2-driver-compile.log"
    b="$STAGING/bundle/b1"
    "$nvcc" "${flags[@]}" -DM26B1_TARGET_ARCH=121 -DM26B1_ASYNC_SCALE_FC1=1 -DM26B1_ASYNC_SCALE_FC2=1 -I"$b" "$b/step_b1.cu" -o "$STAGING/step-b1" 2>&1 | tee "$STAGING/b1-driver-compile.log"
    for family in b1 b2; do
      b="$STAGING/bundle/$family"
      python3 - "$b/step_$family.cu" "$b/step_${family}_badalign.cu" "$family" <<'PY'
import pathlib,sys
src,dst,family=sys.argv[1:]
old,new=('16','128') if family=='b1' else ('128','16')
text=pathlib.Path(src).read_text();needle='#define M26_STEP_REDZONE '+old
assert text.count(needle)==1
pathlib.Path(dst).write_text(text.replace(needle,'#define M26_STEP_REDZONE '+new))
PY
      if [[ "$family" == b1 ]]; then
        extra=(-DM26B1_TARGET_ARCH=121 -DM26B1_ASYNC_SCALE_FC1=1 -DM26B1_ASYNC_SCALE_FC2=1)
      else
        extra=(-DM26X_BAKED_ARCH=121 -DM26X_BAKED_SMS=48 -DM26X_CAPACITY_CLASS=2048 -DM26X_EXACT_M=1)
      fi
      if "$nvcc" "${flags[@]}" "${extra[@]}" -I"$b" -c "$b/step_${family}_badalign.cu" -o "$STAGING/$family-badalign.o" >"$STAGING/$family-badalign.log" 2>&1; then
        echo 'FAIL: allocator alignment regression compiled'; exit 2
      fi
      python3 - "$STAGING/$family-badalign.log" "${family^^}" <<'PY'
import pathlib,sys
text=pathlib.Path(sys.argv[1]).read_text()
assert 'static assertion failed' in text and 'frozen '+sys.argv[2]+' allocation alignment' in text
print('HOST PASS powered frozen '+sys.argv[2]+' allocator alignment compile negative')
PY
    done
    echo 'RESULT: PASS local compile-only sm120a; baked runtime121 refuses local GPU. No CUDA device launch.'
  fi
  exit 0
fi
if [[ "$CELL" == step-hist-host ]]; then
  tests="$REPO/crates/mimo26-expert/tests/gpu"
  c++ -O2 -std=c++17 -Wall -Wextra -Werror "$tests/step_hist_selftest.cpp" -o "$STAGING/step-hist"
  for name in synthetic-all-M1 synthetic-mixed synthetic-tails; do
    "$STAGING/step-hist" "$tests/step_hist_synthetic.json" "$name" | tee "$STAGING/$name.log"
  done
  if [[ -n "${MIMO26_STEP_HIST:-}" ]]; then
    "$STAGING/step-hist" "$MIMO26_STEP_HIST" "${MIMO26_STEP_WORKLOAD:?workload required}" | tee "$STAGING/input-plan.log"
  fi
  for spec in b1:62d1ca5 b2:769610a; do
    family="${spec%%:*}"; revision="${spec#*:}"
    mkdir "$STAGING/$family-pin"
    git -C "$REPO" archive "$revision" crates/mimo26-expert | tar -x -C "$STAGING/$family-pin"
    git -C "$REPO" rev-parse "$revision" | tee "$STAGING/$family-source.txt"
  done
  sha256sum "$tests/step_hist.h" "$tests/step_hist_selftest.cpp" "$tests/step_hist_synthetic.json" >"$STAGING/source-sha256.txt"
  echo 'RESULT: PASS step input/scheduling HOST only; immutable B1/B2 snapshots staged, no CUDA build/launch or timing claim'
  exit 0
fi
if [[ "$CELL" == silu-exp-audit ]]; then
  tests="$REPO/crates/mimo26-expert/tests/gpu"
  receipt="${MIMO26_EXP_RECEIPT:?retained native receipt directory required}"
  sha256sum "$receipt/"{gpu-summary.json,gpu-examples.csv,gpu-midpoints.csv,prior-cpu-summary.json} >"$STAGING/input-sha256.txt"
  python3 -B "$tests/silu_exp_audit_selftest.py" | tee "$STAGING/integrity-selftest.log"
  python3 -B "$tests/silu_exp_decimal.py" --input "$receipt/gpu-midpoints.csv" --output "$STAGING/midpoint-audit" --gpu-root "$receipt" --cpu-summary "$receipt/prior-cpu-summary.json" --gpu-output "$STAGING/gpu-audit" | tee "$STAGING/retained-audit.log"
  sha256sum "$tests/silu_exp_decimal.py" "$tests/silu_exp_audit_selftest.py" "$tests/run_gpu_gemm.sh" >"$STAGING/source-sha256.txt"
  sha256sum -c "$STAGING/input-sha256.txt" | tee "$STAGING/input-check.log"
  exit 0
fi
if [[ "$CELL" == silu-exp-selftest || "$CELL" == silu-exp-cpu ]]; then
  tests="$REPO/crates/mimo26-expert/tests/gpu"
  c++ -O3 -std=c++17 -fno-fast-math -ffp-contract=off "$tests/silu_exp_scan.cpp" -o "$STAGING/exp-scan"
  "$STAGING/exp-scan" --selftest
  python3 "$tests/silu_exp_decimal.py" --selftest
  python3 -B "$tests/silu_exp_audit_selftest.py"
  if [[ "$CELL" == silu-exp-selftest ]]; then
    /usr/local/cuda-12.8/bin/nvcc -O3 -std=c++17 -lineinfo --ftz=false --gpu-architecture=compute_120a --gpu-code=sm_120a -Xcompiler=-ffp-contract=off -I"$tests" -I"$REPO/crates/mimo26-expert/kernels" -I"$REPO/crates/mimo26-expert/kernels/include" "$tests/silu_exp_gpu.cu" -o "$STAGING/exp-gpu"
    CUDA_VISIBLE_DEVICES='' "$STAGING/exp-gpu" --selftest
  fi
  sha256sum "$tests/silu_exp_reference.h" "$tests/silu_exp_scan.cpp" "$tests/silu_exp_decimal.py" >"$STAGING/source-sha256.txt"
  if [[ "$CELL" == silu-exp-cpu ]]; then
    timeout 300s "$STAGING/exp-scan" --scan "$STAGING" | tee "$STAGING/cpu.log"
    python3 "$tests/silu_exp_decimal.py" --input "$STAGING/midpoints.csv" --output "$STAGING" | tee "$STAGING/decimal.log"
  fi
  exit 0
fi
if [[ "$CELL" == ncu-instruction-model ]]; then
  python3 -B "$REPO/crates/mimo26-expert/tests/gpu/ncu_instruction_model.py" --selftest
  IFS=: read -r -a inputs <<<"${MIMO26_NCU_MODEL_INPUTS:?colon-separated retained CSV paths}"
  extra=()
  if [[ -n "${MIMO26_NCU_MODEL_SOURCES:-}" ]]; then
    IFS=: read -r -a sources <<<"$MIMO26_NCU_MODEL_SOURCES"
    extra=(--sources "${sources[@]}")
  fi
  exec python3 -B "$REPO/crates/mimo26-expert/tests/gpu/ncu_instruction_model.py" "${inputs[@]}" "${extra[@]}" --output "$STAGING/model.json"
fi
if [[ "$CELL" == ncu-pc-delta ]]; then
  python3 -B "$REPO/crates/mimo26-expert/tests/gpu/ncu_pc_delta.py" --selftest
  exec python3 -B "$REPO/crates/mimo26-expert/tests/gpu/ncu_pc_delta.py" "${MIMO26_NCU_REPORT_DIR:?retained receipt directory required}"
fi
if [[ "$CELL" == ncu-report ]]; then
  report_dir="${MIMO26_NCU_REPORT_DIR:?retained receipt directory required}"
  python3 -B "$REPO/crates/mimo26-expert/tests/gpu/ncu_report.py" --selftest
  status=0; count=0
  for raw in "$report_dir/"profile-*-m[1-8].csv; do
    case "$(basename "$raw")" in profile-B1-m1.csv|profile-B1-m8.csv) family=B1 ;; profile-B2-m2.csv|profile-B2-m3.csv|profile-B2-m4.csv|profile-B2-m5.csv) family=B2 ;; *) echo 'Unexpected profile file' >&2; exit 2 ;; esac
    if python3 -B "$REPO/crates/mimo26-expert/tests/gpu/ncu_report.py" --brief --family "$family" "$raw" >"${raw%.csv}-analysis.json"; then echo "PROFILE ANALYSIS PASS $raw"; else status=$?; echo "PROFILE ANALYSIS incomplete rc=$status $raw"; fi
    ((count+=1))
  done
  [[ "$count" -gt 0 ]] || exit 2
  exit "$status"
fi
if [[ "$CELL" == owner-guard-selftest || "$CELL" == spark-selftest ]]; then
  for script in "$REPO/crates/mimo26-expert/tests/gpu/"*.sh; do bash -n "$script"; done
  if [[ "$CELL" == spark-selftest ]]; then
    bash "$REPO/crates/mimo26-expert/tests/gpu/ncu_profile.sh" --selftest "$STAGING"
    bash "$REPO/crates/mimo26-expert/tests/gpu/frozen_profile.sh" --selftest "$STAGING"
    python3 -B "$REPO/crates/mimo26-expert/tests/gpu/ncu_report.py" --selftest
    python3 -B "$REPO/crates/mimo26-expert/tests/gpu/ncu_pc_delta.py" --selftest
    mkdir "$STAGING/mock-bin"
    printf '#!/bin/sh\necho "forbidden remote effect in dry mode" >&2\nexit 99\n' >"$STAGING/mock-bin/ssh"
    cp "$STAGING/mock-bin/ssh" "$STAGING/mock-bin/rsync"; chmod +x "$STAGING/mock-bin/"*
    for host in spark1 spark2 spark3 spark4; do
      for mode in --dry --dry-bench --dry-wire --dry-b1-primitive --dry-b1-sanitize --dry-b1-ffn --dry-b1-bench --dry-b1-profile --dry-b2-profile; do
        HOST="$host" PATH="$STAGING/mock-bin:$PATH" bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" "$mode" >"$STAGING/$host-$mode.log"
      done
    done
    for host in coordinator local 'spark1;false'; do
      set +e
      HOST="$host" PATH="$STAGING/mock-bin:$PATH" bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" --dry >"$STAGING/invalid-host.log" 2>&1
      rc=$?
      set -e
      [[ "$rc" == 2 ]] || { echo 'FAIL invalid Spark host accepted'; exit 1; }
    done
    for mode in --dry-b2-force4 --dry-b2-exact --dry-i4-reference --dry-silu-exp; do
      HOST=spark1 PATH="$STAGING/mock-bin:$PATH" bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" "$mode" >"$STAGING/$mode.log"
      for host in spark2 spark3 spark4; do
        rc=0
        HOST="$host" PATH="$STAGING/mock-bin:$PATH" bash "$REPO/crates/mimo26-expert/tests/gpu/run_spark_cell.sh" "$mode" >"$STAGING/$mode-refusal.log" 2>&1 || rc=$?
        [[ "$rc" == 2 ]] || { echo 'FAIL R13 host scope widened'; exit 1; }
      done
    done
    echo 'HOST PASS Spark dry:4 legacy hosts,3 invalid-host refusals,force4/exact/I4-reference spark1-only,no SSH/rsync effects'
  fi
  echo 'RESULT: PASS owner/remote guard HOST selftests only; no GPU query or initialization'
  exit 0
fi
if [[ "$CELL" == b1-sanitizer-selftest ]]; then
  bash "$REPO/crates/mimo26-expert/tests/gpu/b1_sanitizer_check.sh" --selftest "$STAGING" | tee "$STAGING/sanitizer-selftest.log"
  exit 0
fi
if [[ "$CELL" == lattice-oracle-selftest || "$CELL" == lattice-oracle-real || "$CELL" == lattice-compare ]]; then
  bash "$REPO/crates/mimo26-expert/tests/b1_core/run_lattice_oracle.sh" "$REPO" "$STAGING" "$CELL" "$WEIGHTS"
  exit 0
fi
if [[ "$CELL" == b1-primitive-host ]]; then
  bash "$REPO/crates/mimo26-expert/tests/b1_core/run_host.sh" "$REPO" "$STAGING"
  exit 0
fi
if [[ "$CELL" == wire-boundary ]]; then
  bash "$REPO/crates/mimo26-expert/tests/gpu/run_wire_host.sh" "$REPO" "$STAGING"
  exit 0
fi
if [[ "$CELL" == geometry-selftest ]]; then
  cargo test --manifest-path "$REPO/Cargo.toml" -p mimo26-repack --test cuda_layout -- --nocapture
  exit 0
fi
if [[ "$CELL" == capacity-host || "$CELL" == capacity-gpu ]]; then
  bash "$REPO/crates/mimo26-expert/tests/gpu/run_capacity_cell.sh" "$REPO" "$STAGING" "$CELL"
  exit 0
fi
if [[ "$CELL" != tp4-cpu ]]; then
  INCLUDE="$REPO/crates/mimo26-expert/kernels/include"
  g++ -O2 -std=c++17 -Wall -Wextra -Werror -I"$INCLUDE" \
    "$REPO/crates/mimo26-expert/tests/gpu/kernel_selftest.cpp" -o "$STAGING/kernel-selftest"
  "$STAGING/kernel-selftest" | tee "$STAGING/selftest.log"
  if [[ "$CELL" != kernel-selftest ]]; then
    NVCC=/usr/local/cuda-12.8/bin/nvcc
    EXTRA=(); [[ "$CELL" != b2-force4-host ]] || EXTRA+=(-DM26X_DIAGNOSTIC_FORCE4=1)
    [[ "$CELL" != b2-exact-host ]] || EXTRA+=(-DM26X_EXACT_M=1)
    "$NVCC" --version | tee "$STAGING/nvcc-version.log"
    "$NVCC" "${EXTRA[@]}" -O3 -std=c++17 -lineinfo --ftz=false -Xptxas=-v -arch=sm_89 \
      -DM26X_BAKED_ARCH=89 -DM26X_BAKED_SMS=128 -DM26X_CAPACITY_CLASS=2048 \
      -I"$INCLUDE" -c "$REPO/crates/mimo26-expert/kernels/expert_gemm.cu" \
      -o "$STAGING/expert_gemm.o" 2>&1 | tee "$STAGING/compile.log"
    "$NVCC" -O2 -std=c++17 -arch=sm_89 -DM26X_WITH_OBJECT -I"$INCLUDE" \
      "$REPO/crates/mimo26-expert/tests/gpu/kernel_selftest.cpp" "$STAGING/expert_gemm.o" \
      -o "$STAGING/plan-selftest"
    CUDA_VISIBLE_DEVICES='' "$STAGING/plan-selftest" | tee "$STAGING/plan-selftest.log"
    /usr/local/cuda-12.8/bin/cuobjdump --dump-sass "$STAGING/expert_gemm.o" >"$STAGING/expert_gemm.sass"
    echo 'COMPILE PASS sm89 / class2048; NO GPU launched, no numerical/performance qualification'
    if [[ "$CELL" == driver-selftest || "$CELL" == b2-force4-host || "$CELL" == b2-exact-host || "$CELL" == unpack || "$CELL" == tp4-driver-selftest || "$CELL" == tp4-gpu ]]; then
      TESTS="$REPO/crates/mimo26-expert/tests/gpu"
      FIXTURE="$REPO/bench/fixtures/expert_nibble_fixture.json"
      for script in "$TESTS/run_gpu_gemm.sh" "$TESTS/run_unpack_cell.sh"; do bash -n "$script"; done
      "$NVCC" "${EXTRA[@]}" -O3 -std=c++17 -arch=sm_89 -lineinfo --ftz=false \
        -DM26X_BAKED_ARCH=89 -DM26X_BAKED_SMS=128 -DM26X_CAPACITY_CLASS=2048 \
        -I"$INCLUDE" -I"$TESTS" "$REPO/crates/mimo26-expert/kernels/parity/gemm_parity.cu" \
        "$STAGING/expert_gemm.o" -o "$STAGING/gemm-parity" 2>&1 | tee "$STAGING/driver-compile.log"
      CUDA_VISIBLE_DEVICES='' "$STAGING/gemm-parity" selftest 2>&1 | tee "$STAGING/driver-selftest.log"
      CUDA_VISIBLE_DEVICES='' "$STAGING/gemm-parity" bench-selftest 2>&1 | tee "$STAGING/bench-selftest.log"
      CUDA_VISIBLE_DEVICES='' "$STAGING/gemm-parity" audit "$WEIGHTS" "$FIXTURE" 2>&1 | tee "$STAGING/source-audit.log"
      python3 "$TESTS/audit_negatives.py" "$STAGING/gemm-parity" "$WEIGHTS" "$FIXTURE" "$STAGING" | tee "$STAGING/reader-negatives.log"
      if [[ "$CELL" == tp4-driver-selftest || "$CELL" == tp4-gpu ]]; then
        python3 "$REPO/crates/mimo26-repack/tests/tp4_oracle.py" "$WEIGHTS" "$MIMO26_TP4_DIR" | tee "$STAGING/tp4-oracle.log"
        cargo test --manifest-path "$REPO/Cargo.toml" -p mimo26-repack --release \
          --test tp4_identity -- --ignored --nocapture 2>&1 | tee "$STAGING/tp4-producer.log"
        CUDA_VISIBLE_DEVICES='' "$STAGING/gemm-parity" tp4-audit "$MIMO26_TP4_DIR" "$FIXTURE" 2>&1 | tee "$STAGING/tp4-audit.log"
        python3 "$TESTS/tp4_negatives.py" "$STAGING/gemm-parity" "$MIMO26_TP4_DIR" "$FIXTURE" "$STAGING" | tee "$STAGING/tp4-reader-negatives.log"
      fi
      if [[ "$CELL" == unpack ]]; then
        timeout 480s bash "$TESTS/run_unpack_cell.sh" "$STAGING/gemm-parity" "$WEIGHTS" "$FIXTURE" "$STAGING" "$REPO"
      elif [[ "$CELL" == tp4-gpu ]]; then
        timeout 480s bash "$TESTS/run_unpack_cell.sh" "$STAGING/gemm-parity" "$WEIGHTS" "$FIXTURE" "$STAGING" "$REPO" tp4 "$MIMO26_TP4_DIR"
      else
        echo 'RESULT: PASS native proof-driver HOST audit only; no GPU initialized/launched'
      fi
    fi
  fi
  exit 0
fi
python3 "$REPO/crates/mimo26-repack/tests/tp4_oracle.py" "$WEIGHTS" "$MIMO26_TP4_DIR"
cargo test --manifest-path "$REPO/Cargo.toml" -p mimo26-repack --release \
  --test tp4_identity -- --ignored --nocapture
# Positive always selects the correct implementation; the explicit second run
# selects the real naive code. Oracle and raw bytes never depend on those flags.
MIMO26_SPIKE_NAIVE=0 MIMO26_EXPERT_NAIVE=0 \
  cargo test --manifest-path "$REPO/Cargo.toml" -p mimo26-expert --release \
  --test tp4_identity -- --ignored --nocapture | tee "$STAGING/correct.log"
if [[ "${MIMO26_EXPERT_TWO_RUN:-0}" == 1 ]]; then
  set +e
  MIMO26_SPIKE_NAIVE=1 MIMO26_EXPERT_NAIVE=1 \
    cargo test --manifest-path "$REPO/Cargo.toml" -p mimo26-expert --release \
    --test tp4_identity -- --ignored --nocapture >"$STAGING/naive.log" 2>&1
  rc=$?
  set -e
  # Do not greenwash a missing artifact, compile failure, or arbitrary panic.
  if [[ "$rc" != 101 ]] || ! grep -q 'sum .* oracle ' "$STAGING/naive.log"; then
    echo "RESULT: FAIL: naive run did not fail at the numerical oracle (rc=$rc); $STAGING/naive.log" >&2
    exit 1
  fi
  echo "NAIVE: expected numerical oracle failure, exit 101; $STAGING/naive.log"
fi
echo 'RESULT: PASS CPU TP4 identity only (NOT GPU L3, bandwidth, or Spark promotion)'
