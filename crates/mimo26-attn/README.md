# mimo26-attn — coordinator attention + KV kernels (task I3-A5)

## Current status — 23 September 2026 AEST

**R7 numerical ruling:** output bound is `2e-5 * max(1, max|v|)` for decoded,
already-prescaled cached V. P remains two terms; Q3/P2 factor stays 2.6.
All eight original TC cases were audited: max V is 0.5 in the first four and
1 in the correction/rescale cases. Their stricter 1e-5 absolute checks remain.
`scripts/dev.sh test attn audit-tc` reproduces the CPU audit; future TC golden
generation includes it in requalification receipts. See the numerical-contract
section in `docs/design/attn-perf.md`. The N5 context addendum explicitly models
128K prefill from tile rates; it is not a full-context GPU measurement.

D1 4/8-warp async variants are implemented: 64,192 B shared, sm_120 registers
61/48, no spills. Host packed-codec checks cover 65,536 pairs. The coordinator parity at source `ef95e61`
passed **80/80 positive outputs**, **27 complete negative child runs** and the
**65536/65536** GPU packed-codec check; both variants use P = 8 and P = 3.
Receipt: `runs/20260923-i4/attn/attn-parity.IsgI7T/SUMMARY.md`.
This qualifies the original bounded corpus, not high-value coverage or D1
performance. Next are the R7 high-value cohort and split/warp timing sweep.

**N5 measured (21:12 AEST, source `841071e`):** all six configurations **MISS**.
M64/N64 f32q reaches **56.873 executed TFLOPS**, bf16q **32.972 useful TFLOPS**.
All 12 full-output checks and 9 negatives passed; max error **1.0165e-6**.
Evidence: `runs/20260923-i4/attn/attn-bench.B9h3B2/`. The assumed eta is not
qualified; D1 is next. Softmax/high-P is only 1–3% of instrumented phase cycles;
load/expand and Q packing + QK dominate this implementation.

N5 runs through `HOST=coordinator MIMO26_ATTN_BUILDER_WINDOW=1 scripts/dev.sh test
attn-bench eta`: six interior tile/lattice configurations, full FP64 references,
isolated Q/P negatives, seven event samples and separate phase-clock diagnostics.
It is not complete prefill or an X1c result. V is pinned to ±1.25, K to ±1.875;
BF16-Q rounding occurs after host FP32 rotation. All modes retain FP32 m/l/O.

**First direct coordinator D0.1 receipt (20:47 AEST):** 128K **2.112512 ms / 79.418
GB/s MISS**; 1M **14.484512 ms / 92.663 GB/s MISS**; GA9 extrapolation
**130.360611 ms MISS**. Both cells passed AOT, full baseline comparisons and
sampled FP64 checks on bounded inputs. Raw receipt and validated summary:
`runs/20260923-i4/attn/attn-bench.QIe3t1/`. Next is N5, then D1.

**20:15 builder ruling:** N3/N4 paper accepted. Order: D0.1 coordinator timing,
N5 eta micro-cell, then D1 pipeline code. F32-Q prefill keeps the **125.7
executed TFLOPS** gate; factor 2.6 gives about **48.3 useful TFLOPS**, not 62.9.

D0.1 timing dispatch is explicit: `MIMO26_ATTN_DECODE_IMPL=tc` with
`HOST=coordinator MIMO26_ATTN_BUILDER_WINDOW=1 scripts/dev.sh test attn-bench decode-long`.
This runs 128K / 1M with auto splits 128 / 512; an explicit
`MIMO26_ATTN_BENCH_SPLITS` overrides them. Synthetic Q values are BF16-exact in
FP32 storage, K/V are bounded by 1.875. Every TC cell checks all 8192 outputs
against the untimed FP64 baseline, plus three independent reference coordinates,
before timing decode + merge. This does not certify the known high-value P case.
Actual padded MMA work is logged and validated; TC prefill never silently falls
back. Raw coordinator receipts land as text `RESULT.md` files under `runs/`.

ADVISOR-I4 §9 now assigns the coordinator's 5090 to ATTN-LEAD for direct timed cells;
The dev host is shared development/parity. The shared Bash `tests/gpu/gpu_guard.sh`
allows only exact Python `-m service.cotenant_a.cotenant_a` / `service.cotenant_b.cotenant_b`
owners, never PID exceptions; unknown/unreadable owners and <4096 MiB free
refuse. Both runners share expert's `.cargo.lock`, then `.attn-cuda.lock`.
The dev-host lock is released before remote execution. `test attn gpu-check` queries
owners/memory without a CUDA launch; 27 CPU guard tests cover spoofing and
negative/error cases. Remote execution embeds the same Bash-only guard,
uses a unique `/var/tmp/mimo26f-attn/mimo26f-*` slot and prefixed executable,
and collects text receipts into `runs/20260923-i4/attn/`.

The I3 close verified GPU tiny **14/14**, scale-up **26/26**, and the two-run
naive gates (HANDOFF I3 close). The original writer notes below are historical,
not the current verification status. Default kernels remain scalar correctness
baselines. Experimental D0 tensor-core decode is opt-in: dev-host smoke **14/14**
passed on the original seven cases. **D0.1 Q-tail correction passed16/16 on the dev host,
plus12/12 isolated negative gates**; sanitizer/scale-up/sm120 qualification and
D1 timing remain ahead.

Timing entrypoint: `scripts/dev.sh test attn-bench --help`. `selftest` runs the
CPU metric/AOT/memory accounting tests; `build-sm89` / `build-sm120` cross-build
static-CUDA-runtime executables without a GPU. `decode` and `prefill` select
cell groups; every result distinguishes harness correctness from performance
PASS/MISS and labels 4090 numbers PROXY. `HOST=coordinator scripts/dev.sh test
attn-bench --dry` prints a driver-only C1 recipe without taking any action.
`MIMO26_ATTN_BENCH_LOG=receipt.log scripts/dev.sh test attn-bench summarize`
checks returned receipts on CPU and emits JSON: exit 0 for valid/no applicable
MISS, 1 for measured MISS, 2 for incomplete/invalid evidence. Required cells
default to all eight (`MIMO26_ATTN_BENCH_REQUIRED` can narrow the scope). The
`selftest` cell runs 20 C++ metric checks, 44 Python receipt checks, 27 owner-guard checks and 24
precision-probe checks; their synthetic fixtures are not GPU measurements.
`test attn-bench precision-study` prints three closed-form CPU counterexamples
for Q/P rounding, with explicit CPU-only labels. It neither emulates GPU MMA
rounding nor changes oracle goldens; this is evidence for the required design review.

`test attn selftest` separately runs 20 host comparator checks and 24 TC
layout/codec checks, plus 27 owner-guard checks. `test attn goldens-tc` generates
eight cases using the unchanged external oracle, with three additional
closed-form cross-checks. `test attn
build-sm89` / `build-sm120` compile the parity executable without GPU access.
The hardened parity harness rejects NaNs/CUDA errors, poisons outputs, reclaims
per-case allocations, and preserves its logs in workspace build slots. Its
naive run must fail numerically (not merely exit nonzero); hardware two-run
requalification is builder-owned. C1 baseline receipts at `c527ba1` show AOT
PASS and all eight performance cells MISS; see the performance design for numbers.

### Experimental decode D0.1 — correctness before timing

Initial D0 authorized at 18:35 AEST; §9 now orders Q-tail correction, amended
D1, coordinator timing, N5 eta micro-cell, then P1 under the adopted named gates.
New `m26_attn_decode_splitkv_fp8_tc` / `m26_attn_reduce_tc` ABIs accept real
geometry, FP32 Q, unit FP8 KV and separate FP32 SoA scratch. They do not replace
the baseline double-scratch ABI. Synchronous shared staging is intentional in
this first correctness increment; asynchronous overlap is not implemented yet.

Run on an idle dev-host GPU from this committed source; retain ≥4 GiB free. Each
command checks `nvidia-smi` and allocation headroom. No remote/container actions.
Estimates: 10–30 s each including build/oracle generation, **not GPU timings**.
Stop on any positive correctness failure and retain every receipt.

```bash
# 8 cases, 16 complete flat/paged comparisons; unchanged 1e-5 absolute bound.
MIMO26_ATTN_DECODE_IMPL=baseline MIMO26_ATTN_TWO_RUN=1 scripts/dev.sh test attn tc-decode
MIMO26_ATTN_DECODE_IMPL=tc MIMO26_ATTN_TWO_RUN=1 scripts/dev.sh test attn tc-decode
```

The candidate must log `TC_COVERAGE launches=16 requested=1`. Its negative run
uses the same path, followed by 12 individually isolated traps: scale, GA sink,
GA window, sink head mapping, absolute positions, V rescale, split sink,
running-max rescale, Q residual, P residual, page indirection and Q tail. Each must
finish all 16 comparisons and fail numerically on a TC row, not on a CUDA error.

**D0.1 precision correction:** a third Q component addresses the **2.4667e-4**
loss in the old two-component decomposition. CPU modeled error becomes
**5.6198e-7** on that construction. Actual MMA work is **2.6×**, not2×;
43,456 B shared,64 sm120 registers, zero spills in the cross-build. The expanded
GPU ladder passed; high-value P-range, sanitizer and scale-up gates remain
before candidate timing. The P-rounding model's error scales with V, so the
current V±1 construction does not certify the full E4M3 value range.
Existing benchmark cells still time the baseline.

`test attn shape-list` validates the `--all` alias on CPU without generating
tensors. The builder's original `--all` argparse failure is retained; the runner
now passes `--shapes=...` and the generator expands the alias. **Do not request
`--all` as a quick smoke:** it includes 128K/1M oracle generation. Prefer the
explicit `tc-decode` cells above for the next batch. See `docs/design/attn-perf.md` for compilation
resources and what remains unverified.

Method, formulas, limitations, and optimization plan: `docs/design/attn-perf.md`.
Builder-only window commands and current evidence:
`runs/20260923-i4/packets/attn-lead.md`. No fleet operations belong to this crate.

Scope: GQA-packed attention (QK 192 / V 128), SWA-128 window, per-Q-head sink
`[64]` (SWA-only), `v_scale` 0.707 BEFORE caching, FP32 on-the-fly partial-rotary
(64/192, dual θ), FP8 KV unit-scale / per-token×head layouts (K/V separate),
paged GA KV (256-token pages), split-KV decode + reduce, chunked prefill over
pages, and the two A10 AOT gates — with a negative test per trap.

Nothing here is toolchain-verified by the writer (LAW: no cargo/nvcc/pytest by
hand): the captain fires `scripts/dev.sh check` (CPU suite) and
`tests/gpu/run_gpu_parity.sh` (GPU parity, dev-host RTX 4090 only).

## Layout

| Path | What |
|---|---|
| `src/lib.rs` | `NaiveBits` trap switches (17 bits), `naive_from_env`/`bits_from_env`, `AttnError` |
| `src/geom.rs` | real dims, `AttnSpec`, window gating (GA ⇒ none), pool byte pins |
| `src/rope.rs` | T19 FP32 on-the-fly angles + derived tolerances, T7 partial/dual-θ |
| `src/fp8kv.rs` | T20 FP8 layouts + code planes + amax clip count |
| `src/cache.rs` | `RowStore` (T18 v_scale before store), `SwaRing` (T8 eviction), `GaPaged` (A2 pages) |
| `src/attn.rs` | two-pass reference + split-KV/chunked/paged decompositions + sink-once reduce |
| `src/aot.rs` | two independent A10 gates (arch ≠ SM count) |
| `tests/` | two-run trap suite + oracle parity harness (manifest protocol) |
| `tests/oracle_driver.py` | numpy-oracle side (consumes `oracle/mimo26` read-only) |
| `kernels/` | handwritten CUDA (correctness-first) + `kernels/parity/attn_parity.cu` |

## Two-run classification (`MIMO26_SPIKE_NAIVE=1`)

Convention (from `crates/mimo26-load`): the CORRECT run is fully green; the
naive run fails **exactly** the NEGATIVE tests and passes every BOTH-RUNS test.

| Test | File | Class | Flips on |
|---|---|---|---|
| `t19_fp32_onthefly_matches_f64_at_1_128k_1m` | rope_t19 | NEGATIVE | 32K-truncated cos table |
| `t7_partial_rotary_64_dims_and_dual_theta` | rope_t19 | NEGATIVE | full-width rotary / single θ |
| `t20_scale_layout_per_token_head_k_v_separate` | kvlayout_t20 | NEGATIVE | block-128 shared scales |
| `amax_clip_count_reported` | kvlayout_t20 | NEGATIVE | silent clamp |
| `t18_vscale_applied_before_fp8_cache` | cache_t18_t8 | NEGATIVE | v_scale after store / on read |
| `t8_swa_eviction_keep_from_min_batch_pos` | cache_t18_t8 | NEGATIVE | keep-last-window eviction |
| `c1_ga_bitwise_sink_free` | attn_props | NEGATIVE | sink applied on GA |
| `t3_ga_layers_are_not_windowed` | attn_props | NEGATIVE | GA windowed |
| `t6_sink_is_per_q_head` | attn_props | NEGATIVE | sink indexed per KV head |
| `t4_attn_scale_uses_d_qk` | attn_props | NEGATIVE | scale 1/√d_v (QK/V mixup) |
| `t9_start_pos_is_honored` | attn_props | NEGATIVE | positions zeroed |
| `splitkv_sink_counted_once_at_reduce` | splitkv_prefill_equiv | NEGATIVE | sink per split |
| `chunked_prefill_rescales_running_max` | splitkv_prefill_equiv | NEGATIVE | no running rescale |
| `aot_arch_gate_rejects_counts_and_foreign_arch` | aot_gates | NEGATIVE | §8 mixed `aot_sm` gate |
| `aot_sm_count_gate_rejects_arch_values` | aot_gates | NEGATIVE | §8 mixed `aot_sm` gate |
| `aot_cross_pairs_rejected` | aot_gates | NEGATIVE | §8 mixed `aot_sm` gate |
| everything else (oracle parity, property pins, detection oracles, manifest selftest) | all files | BOTH RUNS | — |

Expected runs: **correct run — all tests green; naive run — exactly the 16
NEGATIVE tests above fail** (each detection-oracle test keeps passing, proving
the corresponding trap is individually observable).

## GPU parity cell (captain-fired)

`tests/gpu/run_gpu_parity.sh` is the body of the proposed `scripts/dev.sh test
attn` verb (captain lands the one-liner). Flow: unique mktemp staging →
`tests/oracle_driver.py gen` (numpy oracle goldens at tiny / 4K / 32K real
shapes, split-KV at 128K/1M synthetic FP8 KV, rope at 1M) → `nvcc -arch=sm_89`
→ `kernels/parity/attn_parity.cu` compares kernel outputs case-by-case against
the goldens with the manifest's declared tolerances (`RESULT: PASS|FAIL`).

* Self-guards against non-4090 GPUs (`MIMO26_ATTN_ALLOW_GPU=1` to override in a
  maintainer-approved window). Never the 5090, never the Sparks.
* `MIMO26_ATTN_TWO_RUN=1` appends the naive pass (must fail).
* `MIMO26_ATTN_SHAPES` picks shape sets; `decode-1m` writes ~1.3 GB of FP8 goldens.
* Proposed dev.sh cell: `test attn) MIMO26_ATTN_SHAPES="${2:-tiny}" "$ROOT/crates/mimo26-attn/tests/gpu/run_gpu_parity.sh";;`

## What is NOT verified (by the writer)

* **Nothing is compiled or executed by me** — no cargo/nvcc/pytest by contract.
  The Rust twin is written to be obviously-correct std-only code; the CUDA is
  correctness-first (one thread per partial, f64 accumulators) and has never
  been compiled. Expect nvcc friction (small, mechanical) on the first build.
* GPU parity at 4K/32K/128K/1M is a **plan**, not a result, until the captain
  fires the cell on the dev host and pastes `RESULT: PASS`.
* Declared tolerances are derived (module docs in `src/rope.rs`,
  `tests/oracle_driver.py`) but unmeasured until the cell runs.
* `scripts/dev.sh check` will FAIL on untracked files under `crates/` until the
  captain lands+pushes this crate (ci-cpu untracked gate).

## Tolerances (declared, not tuned)

| Compare | Bound | Source |
|---|---|---|
| twin vs oracle, tiny | 1e-5 abs | oracle einsum f32 vs twin f64 |
| GPU vs oracle ≤4K / ≤32K / 128K–1M | 2e-4 / 5e-4 / 2e-3 abs | f32 accumulation over S keys |
| RoPE vs FP64 angles | `3·eps_f32·|pos|·inv[j] + 8·eps_f32` rad | T19 derivation (HF behaviour) |
| FP8 round-trip | 1e-6 abs | exact decode table (golden-pinned) |
