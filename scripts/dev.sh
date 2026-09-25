#!/usr/bin/env bash
# Closed verb list for mimo26f-afd. Agents call this. They do not invent
# cargo / nvcc / cmake / pytest flags. A missing verb exits 2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Machine-local pins (gitignored) — optional, so `check` runs from a fresh
# clone of origin (I2.9 proof). Bake/build verbs fail later and loud if a pin
# they need is absent.
# shellcheck disable=SC1091
[[ -f "$ROOT/configs/build.env" ]] && source "$ROOT/configs/build.env"
# Exported so the crate cell scripts dev.sh execs inherit it. In-workspace by default: the dsh engine
# sessions run workspace-write, and a build root outside the repo needs sandbox escalation (23 Sep).
export MIMO26F_BUILD_ROOT="${MIMO26F_BUILD_ROOT:-$ROOT/target/mimo26f-builds}"

usage() {
  cat <<'EOF'
scripts/dev.sh <verb>

  doctor          read-only preflight (SM, disk, scratch, CUDA owners)
  check           L0/L1 CPU merge gate (fail-loud if a suite is missing)
  retro [path]    scan session logs for bespoke toolchain calls
  spike [args]    I1 load→prefill→decode (execs spike/run.sh)
  build spark     mimo26-spark serving crate + CUDA FFI (compile+link+device smoke)
  test <cell>     attn [shapes] = GPU attention parity (dev-host 4090, default tiny);
                  expert-unit [cargo-test args] = CPU suite of crates/mimo26-expert
                    (serialized; MIMO26_SPIKE_NAIVE / MIMO26_EXPERT_NAIVE pass through);
                  gemm [cell] = expert GEMM GPU cells (crate-owned script; dev-host 4090 only);
                  attn-bench [cell] = attention timing cells (crate-owned script; dev-host 4090 dev numbers);
                  unit | golden | lanesim closed until their scripts exist
  deploy          I5 go-window (not implemented)

RESULT: PASS or RESULT: FAIL. Read the log the verb prints. Do not re-run
the toolchain by hand with a tweaked flag.
EOF
}

closed() {
  local verb="$1"
  printf '%s\n' "RESULT: FAIL"
  printf '%s\n' "verb '${verb}' is closed until its script exists. Do not invent cargo/nvcc/cmake/pytest."
  usage
  exit 2
}

verb="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$verb" in
  doctor)
    exec "$ROOT/scripts/preflight.sh"
    ;;
  check)
    exec "$ROOT/scripts/ci-cpu.sh"
    ;;
  retro)
    exec "$ROOT/scripts/retro-scan.sh" "$@"
    ;;
  spike)
    exec "$ROOT/spike/run.sh" "$@"
    ;;
  test)
    case "${1:-}" in
      attn)
        shift
        export PATH="/usr/local/cuda/bin:$PATH"
        export MIMO26_ATTN_SHAPES="${1:-tiny}"
        exec "$ROOT/crates/mimo26-attn/tests/gpu/run_gpu_parity.sh"
        ;;
      expert-unit)
        # KERNEL-LEAD's scoped CPU suite, opened by the builder (23 Sep, packet kernel-lead.md).
        # Args forward to cargo test and the exit status is preserved. The naive toggles pass
        # through the environment. One cargo at a time per build root (flock).
        shift
        mkdir -p "$MIMO26F_BUILD_ROOT"
        exec flock "$MIMO26F_BUILD_ROOT/.cargo.lock" \
          cargo test --manifest-path "$ROOT/Cargo.toml" -p mimo26-expert "$@"
        ;;
      attn-bench)
        # ATTN-LEAD's attention timing cells (crate-owned script). The dev host's 4090 gives dev proxy numbers;
        # the 5090 (sm_120) C1 measurement runs in a builder window.
        shift
        export PATH="/usr/local/cuda/bin:$PATH"
        export MIMO26_ATTN_BENCH_CELL="${1:-decode-128k}"
        exec "$ROOT/crates/mimo26-attn/tests/gpu/run_gpu_bench.sh"
        ;;
      gemm)
        # Expert GEMM GPU cells: the crate-owned script, fixed by KERNEL-LEAD before first use.
        # The dev host's 4090 only; Spark and 5090 windows are run by the builder.
        shift
        export PATH="/usr/local/cuda/bin:$PATH"
        export MIMO26_GEMM_CELL="${1:-tiny}"
        exec "$ROOT/crates/mimo26-expert/tests/gpu/run_gpu_gemm.sh"
        ;;
      *)
        closed "test ${1:-}"
        ;;
    esac
    ;;
  build)
    case "${1:-}" in
      spark)
        exec "$ROOT/scripts/build_spark.sh" "${@:2}"
        ;;
      *)
        closed "build ${1:-}"
        ;;
    esac
    ;;
  deploy)
    closed "$verb"
    ;;
  ""|-h|--help|help)
    usage
    exit 0
    ;;
  *)
    printf '%s\n' "RESULT: FAIL"
    printf '%s\n' "unknown verb '${verb}'. Do not improvise a toolchain call."
    usage
    exit 2
    ;;
esac
