#!/usr/bin/env bash
# Sole build/run entry: scripts/dev.sh test attn-bench [cell].
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CRATE="$REPO/crates/mimo26-attn"
source "$CRATE/tests/gpu/gpu_guard.sh"
CELL="${MIMO26_ATTN_BENCH_CELL:-decode-128k}"
PAIRED=0; P1=0; OP1=0
[[ "$CELL" != op1-select && "$CELL" != op1-cell && "$CELL" != op1-cold ]] || OP1=1
case "$CELL" in bf16q-sweep|bf16q-sweep-profile|c1-sweep) PAIRED=1;; esac
case "$CELL" in p1-2k|p1-swa|p1-128k|p1-1m) P1=1;; esac
export TZ=Australia/Sydney
REMOTE="${HOST:-local}"
IMPL="${MIMO26_ATTN_DECODE_IMPL:-baseline}"
case "$IMPL" in baseline|tc|pipe4|pipe8|pipe4-bf16q|pipe8-bf16q|c1|c1-bf16q) ;; *) echo 'RESULT: REFUSE unknown decode implementation' >&2; exit 3;; esac
if [[ "$REMOTE" != coordinator && "$REMOTE" != local ]]; then
  echo "RESULT: REFUSE unsupported HOST=$REMOTE" >&2; exit 3
fi
remote_body() {
  # Embed the same Bash-only policy: coordinator needs no Python installation.
  declare -f m26_guard_service m26_guard_read_argv m26_gpu_guard
  cat <<'REMOTE_SCRIPT'
set -euo pipefail
export TZ=Australia/Sydney
stage="$5"
paired=0
case "$1" in bf16q-sweep|bf16q-sweep-profile|c1-sweep) paired=1;; esac
case "$6" in baseline|tc|pipe4|pipe8|pipe4-bf16q|pipe8-bf16q|c1|c1-bf16q) export MIMO26_ATTN_DECODE_IMPL="$6";; *) exit 3;; esac
[[ "$stage" =~ ^/var/tmp/mimo26f-attn/mimo26f-[A-Za-z0-9]+$ ]] || exit 3
exec 8>/var/tmp/mimo26f-attn/.gpu.lock
flock 8
receipt="$stage/receipt.log"
exec > >(tee "$receipt") 2>&1
date '+%F %T %Z'
nvidia-smi
m26_gpu_guard
sha256sum "$stage/mimo26f-attn-bench-sm120"
if [[ "$1" == d1-sweep || "$paired" == 1 ]]; then
  start=$SECONDS
  run_point() {
    local impl="$1" cell="$2" splits="$3" dir="$stage/$1-$2-p$3"
    (( SECONDS-start < 415 )) || { echo 'RESULT: INCOMPLETE sweep budget'; exit 4; }
    mkdir "$dir"
    {
      date '+%F %T %Z'
      echo "SWEEP_POINT impl=$impl cell=$cell splits=$splits"
      m26_gpu_guard
      MIMO26_ATTN_BUILDER_WINDOW=1 MIMO26_ATTN_DECODE_IMPL="$impl" \
        MIMO26_ATTN_BENCH_WARMUP=3 MIMO26_ATTN_BENCH_SAMPLES=7 MIMO26_ATTN_BENCH_SPLITS="$splits" \
        timeout --signal=TERM --kill-after=5s 120s "$stage/mimo26f-attn-bench-sm120" "$cell"
    } | tee "$dir/RESULT.md"
  }
  if [[ "$paired" == 1 ]]; then
    # Same-binary eight-warp FP32-Q control, plus explicit native BF16-Q rows.
    impls='pipe8 pipe4-bf16q pipe8-bf16q'
    [[ "$1" != c1-sweep ]] || impls='c1 c1-bf16q'
    for impl in $impls; do
      run_point "$impl" decode-128k 85
      for p in 255 510; do run_point "$impl" decode-1m "$p"; done
    done
  else
    for impl in pipe4 pipe8; do
      for p in 64 85 128 256; do run_point "$impl" decode-128k "$p"; done
      for p in 255 256 510 512; do run_point "$impl" decode-1m "$p"; done
    done
  fi
  echo 'RESULT: PASS sweep harness (performance verdicts per point, no promotion)'
fi
if [[ "$1" == d1-profile || "$1" == bf16q-sweep-profile ]]; then
  subdir=.; profile_out=/dev/stdout
  if [[ "$1" == bf16q-sweep-profile ]]; then
    subdir=profile; mkdir "$stage/profile"; profile_out="$stage/profile/RESULT.md"
  fi
  profile_budget=540
  if [[ "$1" == bf16q-sweep-profile ]]; then profile_budget=$((540-(SECONDS-start))); fi
  (( profile_budget > 0 )) || { echo 'RESULT: INCOMPLETE combined sweep/profile budget'; exit 4; }
  image=sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92
  name="mimo26f-attn-ncu-${stage##*-}"
  docker image inspect "$image" --format 'PROFILE_IMAGE {{.Id}}'
  trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
  {
  echo "PROFILE_SCOPE $1 FP32-Q 128K P85 w8; one selected launch after six matching warmups; kernel replay, full set, cold replay caches, no clock locking"
  sha256sum "$stage/mimo26f-attn-bench-sm120"
  echo 'PROFILE_TIMING_NOT_A_GATE: CUDA-event samples under NCU are instrumentation-contaminated'
  m26_gpu_guard
  timeout --signal=TERM --kill-after=5s "${profile_budget}s" docker run --rm --name "$name" --gpus device=0 \
    --cap-add SYS_ADMIN --network none --read-only --tmpfs /tmp:rw,exec,size=2g \
    -e HOME=/tmp -e TZ=Australia/Sydney -e MIMO26_ATTN_BUILDER_WINDOW=1 \
    -e MIMO26_ATTN_DECODE_IMPL=pipe8 -e MIMO26_ATTN_BENCH_SPLITS=85 \
    -e MIMO26_ATTN_BENCH_WARMUP=3 -e MIMO26_ATTN_BENCH_SAMPLES=3 \
    --mount "type=bind,src=$stage,dst=/work" --workdir "/work/$subdir" --entrypoint bash "$image" -lc '
set -euo pipefail
ncu --version
mkdir ncu-details ncu-raw ncu-source ncu-log
ncu --set full --replay-mode kernel --kernel-name-base function --kernel-name regex:decode_pipe \
  --launch-skip 6 --launch-count 1 --clock-control none --pipeline-boost-state dynamic \
  --cache-control all --target-processes application-only --print-details all \
  --export ./d1 --log-file ./ncu-log/RESULT.md /work/mimo26f-attn-bench-sm120 decode-128k
[[ -s ./d1.ncu-rep ]]
ncu --import ./d1.ncu-rep --page details --print-details all --print-rule-details > ./ncu-details/RESULT.md
ncu --import ./d1.ncu-rep --page raw --csv > ./ncu-raw/RESULT.md
ncu --import ./d1.ncu-rep --page source --print-source sass > ./ncu-source/RESULT.md
echo "PROFILE_COMPLETE full_set=1 selected_launches=1 binary_report_retained_remote=1"
'
  } > "$profile_out" 2>&1
elif [[ "$1" == p1-profile ]]; then
  mkdir -p "$stage/profile/f32q" "$stage/profile/bf16q"
  profile_budget=540
  image=sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92
  name="mimo26f-attn-ncu-${stage##*-}"
  docker image inspect "$image" --format 'PROFILE_IMAGE {{.Id}}'
  trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
  {
    echo 'P1_PROFILE_SCOPE T2048/S128K; retained P1 f32q and bf16q; full numerical checks first, then one warmed full-set capture per mode'
    sha256sum "$stage/mimo26f-attn-bench-sm120"
    m26_gpu_guard
    # Uninstrumented p1-128k: full checks + timing retained separately.
    echo 'P1_UNINSTRUMENTED_BEGIN'
    MIMO26_ATTN_BUILDER_WINDOW=1 MIMO26_ATTN_DECODE_IMPL=baseline timeout --signal=TERM --kill-after=5s 480s "$stage/mimo26f-attn-bench-sm120" p1-128k
    echo 'P1_UNINSTRUMENTED_END'
    for mode in f32q bf16q; do
      regex=prefill_tcILb1; [[ "$mode" == bf16q ]] && regex=prefill_tcILb0
      timeout --signal=TERM --kill-after=5s 300s docker run --rm --name "$name" --gpus device=0 \
        --cap-add SYS_ADMIN --network none --read-only --tmpfs /tmp:rw,exec,size=2g \
        -e HOME=/tmp -e TZ=Australia/Sydney -e MIMO26_ATTN_BUILDER_WINDOW=1 \
        -e MIMO26_ATTN_DECODE_IMPL=baseline \
        --mount "type=bind,src=$stage,dst=/work" --workdir /work/profile/$mode --entrypoint bash "$image" -lc "
set -euo pipefail
ncu --version
mkdir -p ncu-details ncu-raw ncu-source
ncu --set full --replay-mode kernel --kernel-name-base mangled --kernel-name regex:$regex \
  --launch-skip 4 --launch-count 1 --clock-control none --pipeline-boost-state dynamic \
  --cache-control all --target-processes application-only --print-details all \
  --export ./p1 --log-file ./ncu-raw/RESULT.md /work/mimo26f-attn-bench-sm120 p1-128k
[[ -s ./p1.ncu-rep ]]
ncu --import ./p1.ncu-rep --page details --print-details all --print-rule-details > ./ncu-details/RESULT.md
ncu --import ./p1.ncu-rep --page raw --csv > ./ncu-raw/RESULT.md
ncu --import ./p1.ncu-rep --page source --print-source sass > ./ncu-source/RESULT.md
echo PROFILE_COMPLETE mode=$mode full_set=1 selected_launches=1
"
    done
    echo 'P1_PROFILE_COMPLETE modes=2 full_checks=1 uninstrumented_timing_retained=1'
  } > "$stage/profile/RESULT.md" 2>&1
elif [[ "$1" != d1-sweep && "$paired" != 1 ]]; then
  MIMO26_ATTN_BUILDER_WINDOW=1 MIMO26_ATTN_BENCH_WARMUP="${2:-2}" MIMO26_ATTN_BENCH_SAMPLES="${3:-5}" MIMO26_ATTN_BENCH_SPLITS="${4:-0}" timeout --signal=TERM --kill-after=5s 540s "$stage/mimo26f-attn-bench-sm120" "$1"
fi
REMOTE_SCRIPT
}
if [[ "$CELL" == --dry ]]; then
  [[ "$REMOTE" == coordinator ]] || { echo '--dry requires HOST=coordinator' >&2; exit 2; }
  # No file writes, compiler, driver probe, ssh, or service operation here.
  cat <<'DRY'
# Run on the dev host. Build requires no GPU. ADVISOR-I4 §9 assigns coordinator to ATTN-LEAD.
set -euo pipefail
scripts/dev.sh test attn-bench selftest
build_log=$(mktemp "$PWD/target/mimo26f-builds/c1-build.XXXXXX.log")
scripts/dev.sh test attn-bench build-sm120 | tee "$build_log"
BINARY=$(sed -n 's/^BINARY //p' "$build_log" | tail -n 1)
CELL=aot # then decode-4k/32k/128k/1m and prefill-2k/4k/32k/swa
SLOT=$(dirname "$BINARY")
REMOTE_STAGE=$(ssh coordinator 'mkdir -p /var/tmp/mimo26f-attn; mktemp -d /var/tmp/mimo26f-attn/mimo26f-XXXXXXXX')
[[ "$REMOTE_STAGE" =~ ^/var/tmp/mimo26f-attn/mimo26f-[A-Za-z0-9]+$ ]]
rsync -a "$BINARY" "coordinator:$REMOTE_STAGE/mimo26f-attn-bench-sm120"
set +e
ssh coordinator bash -s -- "$CELL" 2 5 0 "$REMOTE_STAGE" baseline <<'REMOTE_SCRIPT'
DRY
  remote_body
  cat <<'DRY'
REMOTE_SCRIPT
rc=$?
set -e
mkdir -p "$SLOT/coordinator"
rsync -a "coordinator:$REMOTE_STAGE/receipt.log" "$SLOT/coordinator/"
exit "$rc"
# Equivalent guarded invocation after build (also collects logs after failure):
# HOST=coordinator MIMO26_ATTN_BUILDER_WINDOW=1 MIMO26_ATTN_BENCH_BINARY="$BINARY" scripts/dev.sh test attn-bench aot
DRY
  exit 0
fi
case "$CELL" in
  help|--help|-h)
    printf '%s\n' 'attn-bench cells: selftest | gpu-check | precision-study | summarize | build-sm89 | build-sm120 | aot | decode | prefill | all' \
      'decode-4k | decode-32k | decode-128k | decode-1m' \
      'prefill-2k | prefill-4k | prefill-32k | prefill-swa' \
      'p1-2k | p1-swa | p1-128k | p1-1m: coordinator-only paired modes, full FP64 baseline, 3 warmups/7 samples; 480s budget' \
      'p1-summary: CPU strict paired receipt parser, MIMO26_ATTN_BENCH_LOG and MIMO26_ATTN_BENCH_REQUIRED required' \
      'op1-plan: CPU-only D7 context manifest and negative tests; no GPU or kernel changes' \
      'op1-select: coordinator-only retained P1/C3 selection controls, both modes; NOT all-layer steps' \
      'op1-select-summary: CPU strict selection parser, MIMO26_ATTN_BENCH_LOG required' \
      'op1-cell: coordinator-only full 48-layer verification + short prefill, both modes; no gate' \
      'op1-cell-summary: CPU strict cell parser, MIMO26_ATTN_BENCH_LOG required' \
      'op1-cold: coordinator-only cold 2K/8K/32K/64K multi-chunk P1 prefill, both modes; no gate' \
      'op1-cold-summary: CPU strict cold parser, MIMO26_ATTN_BENCH_LOG required' \
      'm0: coordinator-only dense MMA throughput/latency calibration, three opcode arms; no gate' \
      'Default: sm_89 RTX 4090 PROXY. sm_120 RTX 5090 requires MIMO26_ATTN_BUILDER_WINDOW=1.' \
      'MIMO26_ATTN_BENCH_BINARY: use an already-built executable (no rebuild).' \
      'Warmup=2; samples=5; splits=0 auto (baseline 256; TC 64/64/85/510 by context).' \
      'MIMO26_ATTN_DECODE_IMPL=baseline|tc|pipe4|pipe8|pipe4-bf16q|pipe8-bf16q; decode-long selects 128K and 1M only.' \
      'c1-sweep: 6 same-binary C1 f32q/bf16q rows, N16, 3 warmups/7 samples; fresh static guards.' \
      'bf16q-sweep: 9 same-binary f32q/bf16q rows; fresh build with spill/register/SASS checks before timing.' \
      'bf16q-sweep-profile: same uninstrumented sweep, then one exact-binary full NCU launch in separate profile receipts.' \
      'p1-profile: coordinator-only retained P1 f32q/bf16q T2048/S128K full checks + full-set NCU; no gate' \
      'c0-codegen / c0-sass: CPU-only inventory / full four kernel bodies from MIMO26_ATTN_CODEGEN_SLOT.' \
      'd1-sweep: coordinator-only 16 f32q points (4/8 warps, 128K P64/85/128/256, 1M P255/256/510/512), 3 warmups/7 samples.' \
      'Every run reserves >=4 GiB free. Timeout=540s, including all requested cells.' \
      'HOST=coordinator selects remote driver-only execution; --dry prints commands without actions.' \
      'eta: N5 interior QK/softmax/PV micro-cell, six tile/lattice rows; separate from decode summary.' \
      'precision-study: CPU-only closed-form rounding counterexamples; NOT GPU qualification.' \
      'summarize: MIMO26_ATTN_BENCH_LOG=receipt.log; MIMO26_ATTN_BENCH_REQUIRED=all|decode|prefill|cell,...' \
      'Summary exits 0=valid/no applicable MISS, 1=MISS, 2=incomplete/invalid; never promotes PROXY.'
    exit 0 ;;
  op1-plan)
    receipt=$(mktemp -d "$REPO/runs/20260923-i4/attn/attn-op1-plan.XXXXXX")
    mkdir "$receipt/schedule"
    {
      date '+%F %T %Z'
      printf 'SOURCE %s\n' "$(git -C "$REPO" rev-parse HEAD)"
      echo 'SCOPE CPU-only aggregate-derived OP1 schedule; no GPU, no timing, no promotion'
      sha256sum "$CRATE/tests/gpu/op1_schedule.py" "$CRATE/tests/gpu/test_op1_schedule.py"
      python3 "$CRATE/tests/gpu/test_op1_schedule.py" || exit
      python3 "$CRATE/tests/gpu/op1_schedule.py" > "$receipt/schedule/SUMMARY.md" || exit
      python3 "$CRATE/tests/gpu/op1_schedule.py" --summary || exit
      python3 "$CRATE/tests/gpu/op1_schedule.py" --header > "$receipt/schedule/op1_generated.h" || exit
      python3 "$CRATE/tests/gpu/op1_schedule.py" --header-cell > "$receipt/schedule/op1_cell_generated.h" || exit
      sha256sum "$receipt/schedule/op1_generated.h" "$receipt/schedule/op1_cell_generated.h"
    } 2>&1 | tee "$receipt/RESULT.md"
    printf 'OP1_PLAN_RECEIPT %s\n' "$receipt"
    exit 0 ;;
  op1-select-summary)
    exec python3 "$CRATE/tests/gpu/op1_select_report.py" "${MIMO26_ATTN_BENCH_LOG:?OP1 selection receipt required}" ;;
  op1-cell-summary)
    exec python3 "$CRATE/tests/gpu/op1_report.py" "${MIMO26_ATTN_BENCH_LOG:?OP1 cell receipt required}" ;;
  op1-cold-summary)
    exec python3 "$CRATE/tests/gpu/op1_cold_report.py" "${MIMO26_ATTN_BENCH_LOG:?OP1 cold receipt required}" ;;
  m0-summary)
    exec python3 "$CRATE/tests/gpu/m0_report.py" "${MIMO26_ATTN_BENCH_LOG:?M0 receipt required}" ;;
  op1-select)
    [[ "$REMOTE" == coordinator && "$IMPL" == baseline ]] || { echo 'RESULT: REFUSE OP1 selection requires coordinator and no decode override'; exit 3; } ;;
  op1-cell|op1-cold)
    [[ "$REMOTE" == coordinator && "$IMPL" == baseline ]] || { echo 'RESULT: REFUSE OP1 cell requires coordinator and no decode override'; exit 3; } ;;
  m0)
    [[ "$REMOTE" == coordinator && "$IMPL" == baseline ]] || { echo 'RESULT: REFUSE M0 requires coordinator and no decode override'; exit 3; } ;;
  p1-summary)
    exec python3 "$CRATE/tests/gpu/p1_report.py" "${MIMO26_ATTN_BENCH_LOG:?P1 receipt required}" --required "${MIMO26_ATTN_BENCH_REQUIRED:?P1 context required}" ;;
  p1-2k|p1-swa|p1-128k|p1-1m)
    [[ "$REMOTE" == coordinator && "$IMPL" == baseline ]] || { echo 'RESULT: REFUSE P1 pairs require coordinator and no decode implementation override'; exit 3; } ;;
  p1-codegen|p1-sass)
    python3 "$CRATE/tests/gpu/test_p1_codegen_report.py"
    args=(); [[ "$CELL" != p1-sass ]] || args+=(--emit-sass)
    exec python3 "$CRATE/tests/gpu/p1_codegen_report.py" "${MIMO26_ATTN_CODEGEN_SLOT:?P1 build slot required}" "${args[@]}" ;;
  c1-codegen|c1-sass)
    python3 "$CRATE/tests/gpu/test_c1_codegen_report.py"
    args=(); [[ "$CELL" != c1-sass ]] || args+=(--emit-sass)
    exec python3 "$CRATE/tests/gpu/c1_codegen_report.py" "${MIMO26_ATTN_CODEGEN_SLOT:?C1 build slot required}" "${args[@]}" ;;
  c1-selftest)
    bash -n "$CRATE/tests/gpu/run_c1_model.sh"
    exec bash "$CRATE/tests/gpu/run_c1_model.sh" ;;
  c5-selftest)
    bash -n "$CRATE/tests/gpu/run_c5_reference.sh"
    exec python3 "$CRATE/tests/gpu/c5_reference.py" --selftest ;;
  c5-inspect|c5-import-probe|c5-reference)
    bash -n "$CRATE/tests/gpu/run_c5_reference.sh"
    mode=inspect; [[ "$CELL" != c5-reference ]] || mode=run
    [[ "$CELL" != c5-import-probe ]] || mode=import
    exec bash "$CRATE/tests/gpu/run_c5_reference.sh" "$mode" ;;
  profile-summary)
    exec python3 "$CRATE/tests/gpu/ncu_report.py" "${MIMO26_ATTN_PROFILE_RECEIPT:?profile receipt directory required}" ;;
  p1-profile-summary)
    exec python3 "$CRATE/tests/gpu/p1_profile_report.py" "${MIMO26_ATTN_PROFILE_RECEIPT:?P1 profile receipt directory required}" ;;
  profile-export)
    [[ "$REMOTE" == coordinator ]] || { echo 'RESULT: REFUSE profile-export is coordinator-only'; exit 3; }
    [[ "${MIMO26_ATTN_PROFILE_STAGE:-}" =~ ^/var/tmp/mimo26f-attn/mimo26f-[A-Za-z0-9]+$ ]] || exit 3
    ssh coordinator bash -s -- "$MIMO26_ATTN_PROFILE_STAGE" <<'EXPORT'
set -euo pipefail
stage="$1"
name="mimo26f-attn-ncu-export-$RANDOM-$RANDOM"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
# Offline import only: no GPU device/capability; source report mounted read-only.
timeout --signal=TERM --kill-after=5s 30s docker run --rm --name "$name" --network none --read-only \
  --tmpfs /tmp:rw,exec,size=1g -e HOME=/tmp --mount "type=bind,src=$stage,dst=/work,readonly" \
  --entrypoint /usr/local/cuda/bin/ncu sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92 \
  --import /work/d1.ncu-rep --page details --print-details all --print-rule-details
EXPORT
    exit 0 ;;
  profile-toolkit)
    [[ "$REMOTE" == coordinator ]] || { echo 'RESULT: REFUSE profile-toolkit is coordinator-only'; exit 3; }
    ssh coordinator bash -s <<'TOOLKIT'
set -euo pipefail
export TZ=Australia/Sydney
date '+%F %T %Z'
image=nvidia/cuda:13.0.1-devel-ubuntu24.04
docker image inspect "$image" --format '{{.Id}}'
name="mimo26f-attn-ncu-tools-$RANDOM-$RANDOM"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
# Tool inspection only: no GPU exposure, no network, no image/service mutation.
timeout --signal=TERM --kill-after=5s 20s docker run --rm --name "$name" --network none --read-only \
  --entrypoint bash "$image" -lc 'command -v ncu; ncu --version; ncu --help'
TOOLKIT
    exit 0 ;;
  profile-tools)
    [[ "$REMOTE" == coordinator ]] || { echo 'RESULT: REFUSE profile-tools is coordinator-only'; exit 3; }
    ssh coordinator bash -s <<'TOOLS'
set -euo pipefail
export TZ=Australia/Sydney
date '+%F %T %Z'
printf 'HOST '; hostname
if command -v ncu; then ncu --version; fi
for n in /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu /usr/local/cuda/nsight-compute-*/ncu; do
  [[ ! -x "$n" ]] || "$n" --version
done
if command -v docker; then
  docker ps -a --filter 'name=^/mimo26f-' --format '{{.Names}} {{.Image}} {{.Status}}'
  docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}}'
  while read -r name; do
    [[ "$name" == mimo26f-* ]] || exit 3
    echo "CONTAINER $name"
    docker exec "$name" bash -lc 'command -v ncu || true; for n in /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu /usr/local/cuda/nsight-compute-*/ncu; do [[ ! -x "$n" ]] || "$n" --version; done'
  done < <(docker ps --filter 'name=^/mimo26f-' --format '{{.Names}}')
fi
TOOLS
    exit 0 ;;
  eta-context)
    exec python3 "$CRATE/tests/gpu/eta_context.py" \
      --decode "${MIMO26_ATTN_DECODE_RECEIPT:?decode receipt required}" \
      --eta "${MIMO26_ATTN_ETA_RECEIPT:?eta receipt required}" ;;
  c0-codegen|c0-sass)
    args=(); [[ "$CELL" != c0-sass ]] || args+=(--emit-sass)
    exec python3 "$CRATE/tests/gpu/c0_codegen_report.py" "${MIMO26_ATTN_CODEGEN_SLOT:?codegen build slot required}" "${args[@]}" ;;
  d1-sass-summary)
    exec python3 "$CRATE/tests/gpu/d1_sass_report.py" "${MIMO26_ATTN_SASS_FILE:?SASS file required}" --revision "${MIMO26_ATTN_SASS_REVISION:-ef95e61}" ;;
  d1-sass-extract)
    exec python3 "$CRATE/tests/gpu/d1_sass_report.py" "${MIMO26_ATTN_SASS_FILE:?SASS file required}" --revision "${MIMO26_ATTN_SASS_REVISION:-ef95e61}" --emit-sass ;;
  disassemble)
    [[ -f "${MIMO26_ATTN_BENCH_BINARY:-}" ]] || { echo 'RESULT: FAIL disassemble requires existing MIMO26_ATTN_BENCH_BINARY' >&2; exit 2; }
    exec cuobjdump --dump-sass "$MIMO26_ATTN_BENCH_BINARY" ;;
  eta-summary)
    [[ -n "${MIMO26_ATTN_BENCH_LOG:-}" ]] || { echo 'RESULT: FAIL eta-summary requires MIMO26_ATTN_BENCH_LOG' >&2; exit 2; }
    exec python3 "$CRATE/tests/gpu/eta_report.py" "$MIMO26_ATTN_BENCH_LOG" ;;
  summarize)
    [[ -n "${MIMO26_ATTN_BENCH_LOG:-}" ]] || { echo 'RESULT: FAIL summarize requires MIMO26_ATTN_BENCH_LOG' >&2; exit 2; }
    # Always local/CPU: HOST cannot turn receipt parsing into a remote action.
    exec python3 "$CRATE/tests/gpu/bench_report.py" "$MIMO26_ATTN_BENCH_LOG" \
      --required "${MIMO26_ATTN_BENCH_REQUIRED:-all}" ;;
  selftest|gpu-check|precision-study|build-sm89|build-sm120|aot|eta|m0|d1-sweep|bf16q-sweep|bf16q-sweep-profile|p1-profile|c1-sweep|d1-profile|decode|decode-long|prefill|all|decode-4k|decode-32k|decode-128k|decode-1m|prefill-2k|prefill-4k|prefill-32k|prefill-swa) ;;
  *) echo "RESULT: FAIL unknown attention benchmark cell '$CELL'" >&2; exit 2 ;;
esac
if [[ "$REMOTE" == coordinator && "$CELL" != build-* && "$CELL" != selftest && "$CELL" != precision-study && "${MIMO26_ATTN_BUILDER_WINDOW:-0}" != 1 ]]; then
  echo 'RESULT: REFUSE remote execution requires builder window acknowledgement' >&2; exit 3
fi
if [[ ( "$CELL" == d1-sweep || "$PAIRED" == 1 || "$CELL" == d1-profile || "$CELL" == p1-profile ) && "$REMOTE" != coordinator ]]; then
  echo 'RESULT: REFUSE D1 sweep/profile is coordinator-only'; exit 3
fi
if [[ "$CELL" == d1-profile && ! -x "${MIMO26_ATTN_BENCH_BINARY:-}" ]]; then
  echo 'RESULT: REFUSE profile requires the existing qualified benchmark binary'; exit 3
fi
# Builds/receipts persist in unique slots, never in /mnt/scratch or /tmp.
BUILD_ROOT="${MIMO26F_BUILD_ROOT:-$REPO/target/mimo26f-builds}"
case "$(realpath -m "$BUILD_ROOT")" in /mnt/scratch|/mnt/scratch/*)
  echo 'RESULT: REFUSE forbidden build root' >&2; exit 3;; esac
mkdir -p "$BUILD_ROOT"
SLOT="$(mktemp -d "$BUILD_ROOT/attn-bench.XXXXXX")"
exec > >(tee "$SLOT/receipt.log") 2>&1
printf 'TIME %s\nCOMMAND scripts/dev.sh test attn-bench %s\nSLOT %s\n' "$(date '+%F %T %Z')" "$CELL" "$SLOT"
printf 'SOURCE '; git -C "$REPO" rev-parse HEAD
git -C "$REPO" status --short -- crates/mimo26-attn
# Same dev-host lock as expert-unit/gemm; always take it before the attention lock.
exec 8>"$BUILD_ROOT/.cargo.lock"
flock 8
exec 9>"$BUILD_ROOT/.attn-cuda.lock"
flock 9
c++ -O2 -std=c++17 -Wall -Wextra -Werror "$CRATE/tests/gpu/bench_selftest.cpp" -o "$SLOT/selftest"
"$SLOT/selftest"
c++ -O2 -std=c++17 -Wall -Wextra -Werror "$CRATE/tests/gpu/p1_selftest.cpp" -o "$SLOT/p1-selftest"
"$SLOT/p1-selftest"
c++ -O2 -std=c++17 -Wall -Wextra -Werror "$CRATE/tests/gpu/p1_bench_selftest.cpp" -o "$SLOT/p1-bench-selftest"
"$SLOT/p1-bench-selftest"
MIMO26_ATTN_SELFTEST_DIR="$SLOT" python3 "$CRATE/tests/gpu/test_bench_report.py"
python3 "$CRATE/tests/gpu/test_eta_report.py"
python3 "$CRATE/tests/gpu/test_ncu_report.py"
python3 "$CRATE/tests/gpu/test_c0_codegen_report.py"
python3 "$CRATE/tests/gpu/test_c1_codegen_report.py"
python3 "$CRATE/tests/gpu/test_p1_codegen_report.py"
MIMO26_ATTN_SELFTEST_DIR="$SLOT" python3 "$CRATE/tests/gpu/test_p1_report.py"
python3 "$CRATE/tests/gpu/eta_context.py" --selftest
python3 "$CRATE/tests/gpu/test_gpu_guard.py"
python3 "$CRATE/tests/gpu/test_op1_schedule.py"
python3 "$CRATE/tests/gpu/test_op1_select_report.py"
python3 "$CRATE/tests/gpu/test_op1_report.py"
python3 "$CRATE/tests/gpu/test_op1_cold_report.py"
python3 "$CRATE/tests/gpu/test_m0_report.py"
python3 "$CRATE/tests/gpu/test_p1_profile_report.py"
python3 "$CRATE/tests/gpu/op1_schedule.py" --header > "$SLOT/op1_generated.h"
python3 "$CRATE/tests/gpu/op1_schedule.py" --header-cell > "$SLOT/op1_cell_generated.h"
remote_body | bash -n
# These CPU-only paths return before any driver query, remote action, or CUDA build.
if [[ "$CELL" == precision-study ]]; then
  python3 "$CRATE/tests/gpu/precision_probe.py" | tee "$SLOT/precision-study.log"
  exit 0
fi
python3 "$CRATE/tests/gpu/precision_probe.py" --selftest
if [[ "$CELL" == selftest ]]; then exit 0; fi
if [[ "$CELL" == gpu-check ]]; then
  [[ "$REMOTE" == local ]] || { echo 'RESULT: REFUSE gpu-check is local only'; exit 3; }
  nvidia-smi
  m26_gpu_guard
  exit 0
fi
ARCH=89; SMS=128
if [[ "$CELL" == build-sm120 || ( "$CELL" != build-sm89 && ( "$REMOTE" == coordinator || "${MIMO26_ATTN_BUILDER_WINDOW:-0}" == 1 ) ) ]]; then ARCH=120; SMS=170; fi
# Run paths inspect availability before compiling or allocating. Build-only paths
# deliberately need neither a GPU nor driver (cross-build on the dev host for the builder).
if [[ "$CELL" != build-* && "$REMOTE" != coordinator ]]; then
  nvidia-smi
  m26_gpu_guard
fi
BIN="${MIMO26_ATTN_BENCH_BINARY:-}"
# An explicit build must never reuse a prior architecture's exported binary.
if [[ "$CELL" == build-* ]]; then BIN=""; fi
if [[ -z "$BIN" ]]; then
  BIN="$SLOT/mimo26f-attn-bench-sm${ARCH}"
  SHA="$(git -C "$REPO" rev-parse --short=12 HEAD)"
  if [[ -n "$(git -C "$REPO" status --porcelain -- crates/mimo26-attn)" ]]; then SHA="$SHA-dirty"; fi
  printf 'BUILD arch=sm_%s sms=%s source=%s\n' "$ARCH" "$SMS" "$SHA"
  nvcc --version
  sha256sum "$CRATE"/kernels/attn_{decode_splitkv,decode_tc,reduce,prefill_chunk}.cu \
    "$CRATE"/kernels/include/*.h "$CRATE"/kernels/include/*.cuh \
    "$CRATE"/kernels/bench/* "$SLOT/op1_generated.h" > "$SLOT/source.sha256"
  nvcc -O2 -std=c++17 -cudart static -Xptxas -v -lineinfo -gencode "arch=compute_${ARCH},code=sm_${ARCH}" \
    -DM26_BENCH_ARCH="$ARCH" -DM26_BENCH_SMS="$SMS" -DM26_BENCH_COMMIT="\"$SHA\"" \
    -I"$CRATE/kernels/include" -I"$SLOT" \
    "$CRATE/kernels/attn_decode_splitkv.cu" "$CRATE/kernels/attn_decode_tc.cu" "$CRATE/kernels/attn_reduce.cu" \
    "$CRATE/kernels/attn_prefill_chunk.cu" "$CRATE/kernels/bench/attn_eta.cu" \
    "$CRATE/kernels/bench/attn_bench.cu" -o "$BIN" 2>&1 | tee "$SLOT/ptxas.log"
fi
[[ -x "$BIN" ]] || { echo "RESULT: FAIL missing executable $BIN"; exit 2; }
printf 'BINARY %s\n' "$BIN"
sha256sum "$BIN"
if [[ "$PAIRED" == 1 || "$P1" == 1 || "$OP1" == 1 || "$CELL" == build-sm120 ]]; then
  [[ "$BIN" == "$SLOT/mimo26f-attn-bench-sm120" ]] || { echo 'RESULT: REFUSE C0 sweep needs fresh build and matching ptxas counters'; exit 3; }
  cuobjdump --dump-sass "$BIN" > "$SLOT/SASS.txt"
  python3 "$CRATE/tests/gpu/c0_codegen_report.py" "$SLOT" > "$SLOT/CODEGEN.md"
  echo 'C0_CODEGEN PASS four variants, matching registers, zero spills/local ops, independent MMA destinations'
  if [[ "$CELL" == c1-sweep || "$CELL" == build-sm120 ]]; then
    python3 "$CRATE/tests/gpu/c1_codegen_report.py" "$SLOT" > "$SLOT/C1-CODEGEN.md"
    echo 'C1_CODEGEN PASS both precisions, zero spills/local ops, role MMAs and release boundaries'
  fi
  if [[ "$P1" == 1 || "$OP1" == 1 || "$CELL" == build-sm120 ]]; then
    python3 "$CRATE/tests/gpu/p1_codegen_report.py" "$SLOT" > "$SLOT/P1-CODEGEN.md"
    echo 'P1_CODEGEN PASS both precisions, zero spills/local ops, synchronous CTA phases (not GPU qualification)'
  fi
fi
if [[ "$CELL" == d1-profile ]]; then
  printf '%s  %s\n' b4b006ea25d2b277c27724435b4204b17a1d6ff3d13c710359db69c6b5a54c01 "$BIN" | sha256sum -c -
fi
if [[ "$CELL" == build-* ]]; then echo 'RESULT: PASS build only (NOT a hardware AOT gate)'; exit 0; fi
if [[ "$REMOTE" == coordinator ]]; then
  # Remote execution must not reserve the dev host while KERNEL-LEAD needs that GPU.
  flock -u 9
  exec 9>&-
  flock -u 8
  exec 8>&-
  warmup="${MIMO26_ATTN_BENCH_WARMUP:-2}"
  samples="${MIMO26_ATTN_BENCH_SAMPLES:-5}"
  splits="${MIMO26_ATTN_BENCH_SPLITS:-0}"
  if [[ "$P1" == 1 ]]; then warmup=3; samples=7; splits=256; fi
  for value in "$warmup" "$samples" "$splits"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo 'RESULT: FAIL nonnumeric remote parameter'; exit 2; }
  done
  REMOTE_STAGE=$(ssh coordinator 'mkdir -p /var/tmp/mimo26f-attn; mktemp -d /var/tmp/mimo26f-attn/mimo26f-XXXXXXXX')
  [[ "$REMOTE_STAGE" =~ ^/var/tmp/mimo26f-attn/mimo26f-[A-Za-z0-9]+$ ]] || { echo 'RESULT: REFUSE invalid remote staging path'; exit 3; }
  rsync -a "$BIN" "coordinator:$REMOTE_STAGE/mimo26f-attn-bench-sm120"
  printf 'REMOTE_STAGE %s\n' "$REMOTE_STAGE"
  set +e
  remote_body | ssh coordinator bash -s -- "$CELL" "$warmup" "$samples" "$splits" "$REMOTE_STAGE" "$IMPL"
  rc=$?
  set -e
  receipt_dir="$REPO/runs/20260923-i4/attn/$(basename "$SLOT")"
  mkdir -p "$receipt_dir"
  if [[ "$P1" == 1 || "$OP1" == 1 ]]; then
    mkdir "$receipt_dir/p1-codegen" "$receipt_dir/p1-sass"
    cp "$SLOT/P1-CODEGEN.md" "$receipt_dir/p1-codegen/RESULT.md"
    python3 "$CRATE/tests/gpu/p1_codegen_report.py" "$SLOT" --emit-sass > "$receipt_dir/p1-sass/RESULT.md"
    cp "$SLOT/source.sha256" "$receipt_dir/identity.md"
  fi
  if [[ "$PAIRED" == 1 || "$OP1" == 1 ]]; then
    if [[ "$CELL" == c1-sweep ]]; then
      mkdir "$receipt_dir/c1-codegen" "$receipt_dir/c1-sass"
      cp "$SLOT/C1-CODEGEN.md" "$receipt_dir/c1-codegen/RESULT.md"
      python3 "$CRATE/tests/gpu/c1_codegen_report.py" "$SLOT" --emit-sass > "$receipt_dir/c1-sass/RESULT.md"
      cp "$SLOT/source.sha256" "$receipt_dir/identity.md"
    else
      mkdir "$receipt_dir/codegen" "$receipt_dir/sass"
      cp "$SLOT/CODEGEN.md" "$receipt_dir/codegen/RESULT.md"
      python3 "$CRATE/tests/gpu/c0_codegen_report.py" "$SLOT" --emit-sass > "$receipt_dir/sass/RESULT.md"
    fi
  fi
  if ! rsync -a "coordinator:$REMOTE_STAGE/receipt.log" "$receipt_dir/RESULT.md"; then
    echo 'RESULT: FAIL remote receipt collection failed' >&2
    if (( rc == 0 )); then rc=2; fi
  fi
  printf 'REMOTE_RECEIPT %s/RESULT.md\n' "$receipt_dir"
  if [[ "$OP1" == 1 && "$rc" == 0 ]]; then
    if [[ "$CELL" == op1-select ]]; then
      python3 "$CRATE/tests/gpu/op1_select_report.py" "$receipt_dir/RESULT.md" > "$receipt_dir/SUMMARY.md"
      echo 'OP1_SELECT_SUMMARY VALID selection controls only; all-layer steps remain unmeasured'
    elif [[ "$CELL" == op1-cold ]]; then
      python3 "$CRATE/tests/gpu/op1_cold_report.py" "$receipt_dir/RESULT.md" > "$receipt_dir/SUMMARY.md"
      echo 'OP1_COLD_SUMMARY VALID cold prefill arm; no gate, no promotion'
    else
      python3 "$CRATE/tests/gpu/op1_report.py" "$receipt_dir/RESULT.md" > "$receipt_dir/SUMMARY.md"
      echo 'OP1_SUMMARY VALID full 48-layer cell; no gate, no promotion'
    fi
  fi
  if [[ "$CELL" == m0 && "$rc" == 0 ]]; then
    python3 "$CRATE/tests/gpu/m0_report.py" "$receipt_dir/RESULT.md" > "$receipt_dir/SUMMARY.md"
    echo 'M0_SUMMARY VALID dense MMA calibration; no gate, no ceiling certification'
  fi
  if [[ "$P1" == 1 && "$rc" == 0 ]]; then
    set +e
    python3 "$CRATE/tests/gpu/p1_report.py" "$receipt_dir/RESULT.md" --required "$CELL" > "$receipt_dir/SUMMARY.md"
    report_rc=$?
    set -e
    (( report_rc <= 1 )) || exit "$report_rc"
    echo "P1_SUMMARY valid_modes=2 measured_miss_present=$report_rc (no promotion)"
  fi
  if [[ "$CELL" == d1-profile ]]; then
    # Export text only; retain the binary .ncu-rep in the private remote slot.
    rsync -a --include='*/' --include='RESULT.md' --exclude='*' "coordinator:$REMOTE_STAGE/" "$receipt_dir/"
  fi
  if [[ "$CELL" == p1-profile ]]; then
    rsync -a --include='*/' --include='RESULT.md' --exclude='*' "coordinator:$REMOTE_STAGE/profile/" "$receipt_dir/profile/"
    if [[ "$rc" == 0 ]]; then
      python3 "$CRATE/tests/gpu/p1_profile_report.py" "$receipt_dir/profile" > "$receipt_dir/profile/SUMMARY.md"
      echo 'P1_PROFILE_SUMMARY VALID full-set P1 f32q/bf16q profile; no gate, no promotion'
    fi
  fi
  if [[ "$CELL" == d1-sweep || "$PAIRED" == 1 ]]; then
    rsync -a --include='*/' --include='RESULT.md' --exclude='*' "coordinator:$REMOTE_STAGE/" "$receipt_dir/"
    if (( rc == 0 )) || [[ "$CELL" == bf16q-sweep-profile && -f "$receipt_dir/pipe8-bf16q-decode-1m-p510/RESULT.md" ]]; then
      misses=0; points=0
      impls='pipe4 pipe8'
      point_list='decode-128k-p64 decode-128k-p85 decode-128k-p128 decode-128k-p256 decode-1m-p255 decode-1m-p256 decode-1m-p510 decode-1m-p512'
      if [[ "$PAIRED" == 1 ]]; then
        impls='pipe8 pipe4-bf16q pipe8-bf16q'
        [[ "$CELL" != c1-sweep ]] || impls='c1 c1-bf16q'
        point_list='decode-128k-p85 decode-1m-p255 decode-1m-p510'
      fi
      for impl in $impls; do
        for point in $point_list; do
          dir="$receipt_dir/$impl-$point"
          set +e
          python3 "$CRATE/tests/gpu/bench_report.py" "$dir/RESULT.md" --required "${point%-p*}" > "$dir/SUMMARY.md"
          report_rc=$?
          set -e
          (( report_rc <= 1 )) || exit "$report_rc"
          (( misses += report_rc, points += 1, 1 ))
        done
      done
      echo "SWEEP_SUMMARY valid_points=$points measured_misses=$misses (gate P85 at 128K, P255/510 at 1M; others diagnostic)"
    fi
  fi
  if [[ "$CELL" == bf16q-sweep-profile && "$rc" == 0 ]]; then
    python3 "$CRATE/tests/gpu/ncu_report.py" "$receipt_dir/profile" > "$receipt_dir/profile/SUMMARY.md"
  fi
  exit "$rc"
fi
# Keep the lock across GPU execution too. This does not claim ownership of any
# unrelated user's GPU work, hence the runtime memory reserve checks as well.
nvidia-smi
m26_gpu_guard
timeout --signal=TERM --kill-after=5s 540s "$BIN" "$CELL"
printf 'END %s\n' "$(date '+%F %T %Z')"
