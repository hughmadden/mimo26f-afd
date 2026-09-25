# Attention performance: C1 baseline and tensor-core review R1

## M1 and I4 attention exit audit — 24 September 2026, 12:21 AEST

M1 (`runs/20260923-i4/attn/attn-bench.kYWtSm/notes/SUMMARY.md`): retained P1 full-set NCU
at T2048/S128K, both modes. Tensor pipe 8.4% f32q / 7.6% bf16q, SFU 1.1%/1.9%, achieved
warps 16.7%/33.2%, long-scoreboard 3.36/5.97, shared-store conflicts ~2.2 G. **P1 is
latency/occupancy/operand-supply limited, not tensor/SFU limited**; the resident-operand
slice is not needed. The I4 attention exit audit is
`runs/20260923-i4/packets/i4-attn-exit-audit.md`: decode met (0.805 ms), A5 decode/prefill
MISS, cold prefill MISS, M0/M1 recorded, no unqualified attention PASS. No gate/default/
lattice change; R19 (OP1 → M0 → M1) is complete.

## Previous M0 reconciliation — 24 September 2026, 12:07 AEST

`runs/20260923-i4/attn/attn-bench.aZRO3y/notes/SUMMARY.md`. Per-block `clock64`
SM-cycle deltas in the same launch reconcile the earlier coarse-clock figure:
**511.8 FLOP/SM/measured-cycle** (BF16→FP32 and FP16→FP32) and **1023** (FP16→FP16,
2×). The **512 model is confirmed**; the 2,580 MHz coarse clock was the error (true
SM clock ≈ 2.97 GHz). The 209.5 TFLOPS roof and the 125.7-executed / 100-useful
gates stand unchanged. No roof or gate re-derived. M1 (P1 NCU) and the I4 attention
exit audit remain.

## Previous M0 dense MMA calibration — 24 September 2026, 10:50 AEST

`runs/20260923-i4/attn/attn-bench.AfR7B4/notes/SUMMARY.md`. Three m16n8k16 arms:
BF16→FP32 **258.4 TFLOPS**, FP16→FP32 257.5, FP16→FP16 **514.0 (2×)** at 2580 MHz;
dependent latency 34.2/35.1/29.8 cycles. Measured **589 FLOP/SM/cycle > 512**, so the
512 mode-rate model is falsified (coarse-clock caveat; calibration, not ceiling
certification). The 209.5 TFLOPS peak used in the P1 gate is ~23% low; useful
compute-only roofs re-derive ~99 f32q / ~184 bf16q TFLOPS. Gates/default/lattices
unchanged; M1 (P1 NCU) and the I4 attention exit audit remain.

## Previous OP1 cold-prefill measurement — 24 September 2026, 10:34 AEST

F-1 cold-prefill arm complete: full-request multi-chunk P1 at 2K/8K/32K/64K, GA and
SWA, both lattices. Evidence `runs/20260923-i4/attn/attn-bench.ThWFXX/notes/SUMMARY.md`.
48-layer attention slice (9 GA + 39 SWA) tok/s vs the D7 full-model bar:

| S | f32q | bf16q | D7 bar |
|---|---:|---:|---:|
| 2K | 14,267 | 25,377 | 2,999 |
| 8K | 5,589 | 9,944 | 2,975 |
| 32K | 1,604 | 2,864 | 2,671 |
| 64K | 820 | 1,469 | 2,114 |

GA full-causal attention dominates and crosses the bar between 8K and 32K (f32q) /
32K and 64K (bf16q); that is where the 100-TFLOPS P1 gate matters, not the block-8
decode step (~0.8 ms). SWA rows are an upper bound (general full-scan path, not
ring-specialized). OP1 is now measured end-to-end; M0 → M1 remain.

## Previous OP1 operating-point measurement — 24 September 2026, 10:19 AEST

The R19-amended verification cell measured the **48-layer block-8 attention critical
path** over the full D7 distributions (2,642 T=8 steps, 63 requests, max p+c 434) plus
nine short single-chunk prefills, both lattices, frozen kernel paths (verification GA
C3/P8, SWA C3/P4; short prefill GA C3/P1, SWA P1). Evidence
`runs/20260923-i4/attn/attn-bench.Y67ivh/notes/SUMMARY.md`.

- **Category-weighted step ms (equal 1/9): 0.805 f32q / 0.751 bf16q**; pooled
  0.799 / 0.745. Per-category f32q means 0.736–0.927 ms; p95 0.877–0.930 ms.
- Short prefill (48 layers, T=P): 1.94–9.31 ms f32q, 1.85–5.82 ms bf16q.
- The operating-point attention slice (~0.8 ms/step) is far below the 62.3 ms D7
  parity floor; attention is not the binding constraint there. Long-context rows
  remain recorded I5 debt. No target restatement (R19a's job).

Each case ran an FP64 scalar reference, 48 independent host coordinates, exact-Q proof
and full per-layer output checks; largest error 1.2e-6, coordinate diff 1.3e-8,
short-prefill max 5.5e-6. The first run (q_pos aliased to key positions) was caught by
the independent coordinate fold and fixed in a990eff without kernel changes. Remaining
OP1: the cold 2K/8K/32K/64K multi-chunk P1 arm, then M0 → M1.

## Previous R19 amendment — full distributions + cold-prefill arm — 24 September 2026

R19 **519a721** (MiMo cross-review CHANGES) amends OP1 before it runs:
- Verification replays the **full per-category length distributions** (C1 + C6,
  7 requests each, 2,642 T=8 steps, max prompt+completion **434**) rather than one
  representative per category. Identity note: deterministic decode yields
  non-identical completions per prompt.
- Prefill adds a **cold full-request multi-chunk P1** arm at 2K/8K/32K/64K
  (TTFT, tok/s, per-chunk useful attention FLOPs and ms) beside the short C1 TTFT
  rows (0.251 s bar).
- 128K/1M rows stay recorded MISSes, owner attn-lead, due the I5 plan freeze; the
  I4 close quotes no unqualified attention PASS.

Schedule receipt `runs/20260923-i4/attn/attn-op1-plan.a2xA1x/SUMMARY.md` passes
32+25 CPU tests and freezes the aggregate-derived reconstruction (accepted drafts
plus one bonus, floor-difference acceptance, compact SWA union ≤135 keys). D7 bars
are full-model numbers; our rows are attention-critical-path only. Kernel choices
remain frozen from the 8qKHcw selection, with cold prefill on P1 per the amendment.
No GPU timings yet for the amended OP1; M0/M1 still follow.

## Previous R19 new measurement objective — 24 September 2026, 09:01 AEST

R19 **bd8c71d** orders **OP1 → M0 → M1** on the coordinator. No new kernel design trial
before M0/M1, no target/default/lattice change. Earlier scoped close stands;
long-context MISSes are carried into **I5 design debt**, not waived. R19a will
freeze the service allocation after operating-point evidence; no OP1 PASS gate
is invented here. A-f32q remains default; A-bf16q is diagnostic only.

OP1 CPU preparation now passes **23 tests**:
`runs/20260923-i4/attn/attn-op1-plan.pM7hA2/SUMMARY.md`. The nine D7 C1 categories
exclude ceiling_count and use all **9 GA + 39 SWA** layers, actual model order.
D7 acceptance counters are reconstructed as accepted drafts **plus one bonus**;
first completion comes from prefill. Integer fractional-acceptance reconstruction
yields **369 T8 verification steps**, speculative GA S up to **426**, and compact
SWA union views of at most **135** absolute-position keys. Prefill is one full
chunk at each 44–309-token prompt. These are **aggregate-derived synthetic
contexts, not captured speculative traces**. Full input hashes/step manifest
and mutation/coverage/statistics tests are retained.

Next: implement and time the actual all-layer OP1 sequences, selecting unchanged
C3-class (including reduction) versus P1-class kernels on these shapes. Keep
RoPE, post-RoPE Q casts, V prescale/encoding and cache-commit boundaries explicit.
Report per-category mean/p95 and equal-category weighted step ms, plus full
attention-prefill per request. No GPU timings yet for OP1. M0's three exact MMA
arms and M1's full-set retained P1 T2048/S128K profiles follow, not precede it.

## Previous scoped objective close — 24 September 2026, 08:16 AEST

**Implementation/qualification and required measurements complete; targets
remain MISS, no promotion.** Final audit and receipt index:
`runs/20260923-i4/attn/attn-objective-close/SUMMARY.md`. D0.1/N5/D1,
explicit BF16-Q/C0/C3, C5 actual-rejection SKIP, C1 lifetime/qualification and
six timing MISSes, and P1 bounded qualification plus full-context measurements
are accounted for. C2 remains parked. This does not close engine readiness.

P1 measured set, coordinator sm_120 / 170 SMs, QK/V 192/128, unit E4M3 prescaled V,
BF16-exact values in FP32 Q storage. Every row checks all 16,777,216 outputs
before three warmups/seven samples. FP32 gate is 125.7 **executed** TFLOPS;
native gate is 100 **useful** TFLOPS. All eight rows are valid **MISSes**.

| Cell (T=2048) | FP32 median ms / executed TFLOPS | BF16-Q median ms / useful TFLOPS |
|---|---|---|
| GA S=2048 | 5.508224 / 40.863104 | 3.049344 / 28.183534 |
| GA S=131072 | 652.873840 / 43.447458 | 395.387970 / 27.591277 |
| GA S=1048576 | 6746.384766 / 33.866557 | 3511.933105 / 25.021853 |
| SWA S=32768, window128 + sink | 25.592417 / 1.227197 | 15.429632 / 0.695896 |

GA Hq/Hkv 64/4; SWA 64/8. These are one-layer last chunks, not complete prompt
latencies. SWA uses general logical-tile scanning, not a ring-specialized path.
Final SWA source **8201076b4e78**, evidence
`runs/20260923-i4/attn/attn-bench.kUcqow/notes/SUMMARY.md`; max baseline error
1.05425715e-6. Largest error across all P1 benchmark rows is 5.05708158e-6,
below unchanged 2e-5. Native reference reuse is dataset-local, not general
precision/model equivalence. Exact padded work and independent coordinates
retained. No achieved P1 occupancy/carveout/profile claim.

CPU audit reparses all four complete pairs and rejects the retained 1M
single-query budget failure. No failure rewritten away, R7/target/default
change or automatic repeated ladder. Further optimization is a new objective;
scoped completion uses the explicit honest-measured-MISS allowance.

## Previous execution — 24 September 2026, 08:08 AEST

**P1 1M KV-context pair measured: two valid MISSes.** The coordinator sm_120 / 170 SMs,
source **13391c33b194**, unchanged qualified P1/scalar CUDA bodies. GA T=2048,
S=1048576, Hq/Hkv 64/4, QK/V 192/128, unit E4M3 prescaled cached V and BF16-exact
Q values in FP32 storage. Last-chunk work, not million-token prompt latency.

| Mode | Median ms | Useful TFLOPS | Executed MMA TFLOPS | Gate | Verdict |
|---|---:|---:|---:|---|---|
| FP32-Q Q3/P2 | 6746.384766 | 13.025506 | 33.866557 | 125.7 executed | MISS |
| BF16-Q Q1/P2 | 3511.933105 | 25.021853 | 35.030845 | 100 useful | MISS |

Three warmups/seven samples; both full **16,777,216-output** checks pass before
timing, max baseline error **5.05708158e-6**. All-Q exactness, finite scans and
independent coordinates pass. Two-query/P256 reference stage **318,336.527 ms
wall time**, not scalar-kernel latency. Unchanged 480 s / 540 s budgets held.
Prior single-query failure retained, not replaced; no complete-run speedup
ratio claimed. Evidence:
`runs/20260923-i4/attn/attn-bench.PjeFiF/notes/SUMMARY.md`.

Short/128K/1M now yield **six valid P1 MISSes**. **SWA control remains**, then
full objective audit. No repeated unchanged GA ladder or promotion. All
precision/R7/target/default contracts unchanged; goal still active.

## Previous execution — 24 September 2026, 07:52 AEST

**P1 1M attempt INCOMPLETE: reference budget exhausted.** The coordinator sm_120 / 170 SMs,
source **aab4a7c5b3c1**; candidate/scalar CUDA bodies unchanged. GA T=2048,
S=1048576, Hq/Hkv 64/4, QK/V 192/128, unit E4M3 prescaled V and BF16-exact Q
values in FP32 storage. The one-query/P256 reference reached its last printed
progress **1792/2048 queries at 440.568s**, then hit the **480s soft budget**.
No complete reference, P1 candidate checks or timing samples: **neither a
performance MISS nor a correctness failure**. No 1M candidate performance claim.

Sole round-18 job bash-214 exited 4, collected; no retry. CPU parser rejects the
retained incomplete receipt with exit 2. Evidence:
`runs/20260923-i4/attn/attn-bench.Ge8fK2/notes/SUMMARY.md`.

Next prepare **two-query reference slabs for 1M only**, retaining scalar/P1
kernels, P256, arithmetic order, all output checks and existing 480s/540s budgets.
Source mapping supplies 128 work-bearing scalar CTAs versus 64 at batch 1;
this motivates a changed scheduling test, not an occupancy/cache/speed claim.
CPU mapping/parser/static checks and commit precede another bounded run in a
new round. No same-schedule retry or weakened checks. **1M and SWA remain
unmeasured; goal active.** Previous short/128K P1 and C1/C3 MISSes retained.

## Previous execution — 24 September 2026, 07:30 AEST

**P1 128K KV-context pair measured: two valid MISSes.** The coordinator sm_120 / 170 SMs,
source **50dfed9069ba**, unchanged attention harness 5511391 and qualified CUDA
bodies. **T=2048/S=131072**, GA 64/4, QK/V 192/128, unit E4M3 prescaled V,
BF16-exact values in FP32 Q storage. This is last-chunk work, not full-prompt
latency. Three warmups/seven uninstrumented samples per precision.

| Mode | Median ms | Useful TFLOPS | Executed MMA TFLOPS | Gate | Verdict |
|---|---:|---:|---:|---|---|
| FP32-Q Q3/P2 | 652.873840 | 16.709597 | 43.447458 | 125.7 executed | MISS |
| BF16-Q Q1/P2 | 395.387970 | 27.591277 | 38.630015 | 100 useful | MISS |

All **16,777,216 outputs per mode** checked before timing; max baseline error
**4.6659261e-7**, coordinate error **5.82244823e-9**, complete finite/exact-Q
scans pass. Reference stage **66,596.512 ms wall time**, not scalar-kernel
latency. Evidence: `runs/20260923-i4/attn/attn-bench.Z9Rbos/notes/SUMMARY.md`.

**1M and SWA remain unmeasured.** To address reference-budget risk without
weakening correctness, next prepare one-query reference slabs for 1M only,
retaining the same scalar kernels, P256 and every output. This scheduling
change is not implemented/measured yet; no speed/cache claim. CPU mapping,
parser and static checks must pass before the next single bounded GPU job.
Keep 480s soft / 540s hard budgets and both precision gates. No promotion,
R7/default change or repeated unchanged ladder; whole goal remains incomplete.

## Previous execution — 24 September 2026, 07:15 AEST

**P1 short paired GA measured: two valid MISSes.** The coordinator sm_120 / 170 SMs,
source **deaecc47b72e** (attention identical to harness 5511391), unchanged
qualified P1 CUDA bodies. **T=2048/S=2048**, Hq/Hkv 64/4, QK/V 192/128,
FP32 storage with BF16-exact Q values; E4M3 unit, prescaled cached V.

| Mode | Median ms | Useful TFLOPS | Executed MMA TFLOPS | Gate | Verdict |
|---|---:|---:|---:|---|---|
| FP32-Q Q3/P2 | 5.508224 | 15.602359 | 40.863104 | 125.7 executed | MISS |
| BF16-Q Q1/P2 | 3.049344 | 28.183534 | 39.745797 | 100 useful | MISS |

Three warmups/seven samples per precision, exact padded work, no extra 62.9
gate. Each mode checks **16,777,216 outputs** before any timing; max baseline
error **4.94718552e-6**. Independent coordinates/finite scans and all-Q exactness
proof pass. Reference reuse is dataset-local, not general lattice equivalence.
Reference-stage wall time **1335.708 ms** includes copies/checks/host coordinates;
not scalar-kernel latency or evidence that longer contexts fit the budget.

Evidence: `runs/20260923-i4/attn/attn-bench.yA9LiL/notes/SUMMARY.md`, plus raw receipt,
strict paired-parser summary, source hashes and P1 static inventory/SASS.
**Full 128K/1M KV-context chunks and SWA control remain unmeasured.** Next is
one paired 128K cell, full reference retained, 480s soft / 540s hard budget.
No repetition of unchanged short/C1/C3 ladders, promotion, full-prompt latency
claim or inferred bottleneck. Goal incomplete; all frozen contracts unchanged.

## Previous execution — 24 September 2026, 06:39 AEST

**P1 bounded GPU qualification PASS; performance still unmeasured.** Four
coordinator sm_120 cohorts at source 3139ee4, unchanged CUDA bodies from f8c855b:
240 positive rows (160 P1 + 80 baseline), 54 expected-negative children / 66
total children. All original/high-V bounds pass in both query lattices. Twenty
analytic protocol cases across eight positive children scan 6,946,816 outputs
plus 4,345,856 sentinel/padded-row guards. Mixed-mask poison and query-tile
crossings pass. No sanitizer/universal error proof implied.

Occupancy API reports one CTA/SM FP32-Q, two BF16-Q, with unchanged 125/124
registers and 88,128/38,976 B shared. **Capacity, not achieved occupancy**;
actual carveout unmeasured. Full evidence:
`runs/20260923-i4/attn/attn-p1-qualified-3139ee4/SUMMARY.md`.

Next: add P1 timing dispatch/reporting and prefill-128k/1m cells. Keep all
16,777,216 T2048 outputs checked against the scalar FP64 baseline, potentially
in bounded query slabs. One reference may serve paired modes only for identical
BF16-exact benchmark Q, not general lattice equivalence. Keep FP64 coordinates,
finite scans, three warmups/seven samples and frozen target domains. Current
SWA is the general scanning path, not the earlier proposed specialized tile.
No timing/promotion claim; C1/C3 MISSes and defaults remain unchanged.

## Previous execution — 24 September 2026, 06:06 AEST

**P1 now implemented/cross-built, not GPU-qualified or measured.** Separate
M64/N16 prefill path packs four GA/eight SWA query tokens per CTA/KV head,
with synchronous CTA phase retirement and precision-dependent Q storage.
Typed shared **88,128 B FP32 / 38,976 B BF16**; sm_120 **125/124 registers**,
zero stack/spills/local ops, **52/28 static HMMA sites**. No hardware occupancy
claim. Row-local poison flags plus operand sanitization address masked-query
`0 × NaN` contamination, pending hardware probes.

6,789 enumerated CPU layout/accounting assertions, ten P1 parser tests and
existing host/static guards pass. Full proof/identities:
`runs/20260923-i4/attn/attn-bench.U6giht/SUMMARY.md`. Next is explicit P1 parity
wiring and external-oracle/mixed-mask qualification, then full-output checked
T=2048 prefill chunks with 128K/1M KV. Timing dispatch is not wired yet. No GPU
launch this round. Scalar/default, frozen R7, both precision targets and prior
C1/C3 MISS evidence unchanged; no promotion.

## Previous execution — 24 September 2026, 05:34 AEST

**C1 measured: six valid MISS outcomes, slower than earlier C3.** Source
975b8bc, coordinator sm_120/170 SMs; same binary for both precision modes, three
warmups/seven samples, N16 padded accounting. FP32-Q 128K/P85, 1M/P255,510:
**0.592960 / 4.550976 / 4.639456 ms**, **282.940 / 294.921 / 289.296 GB/s**.
BF16-Q: **0.597056 / 4.588992 / 4.670912 ms**, **280.999 / 292.478 / 287.348 GB/s**.
All full baseline, FP64 coordinates and finite scans pass. Best 1M GA9
40.958782/41.300929 ms, not engine latency. No C1 counters or achieved occupancy;
no causal bottleneck claim. Target remains 1,253 GB/s; no promotion or replan
from absent production reference. Retain C1 as a qualified experimental
regression; C3 remains the faster measured experimental option.

Full data/limitations: `runs/20260923-i4/attn/attn-bench.x1G7FS/SUMMARY.md`.
**Next is required P1 implementation/qualification/full-context measurement**,
not another unchanged decode ladder. FP32 125.7 executed TFLOPS; native 100
useful TFLOPS. Serving default, frozen R7 and parked C2 unchanged.

## Previous execution — 24 September 2026, 05:11 AEST

**C1 GPU-qualified, performance still unmeasured.** Source 5236a04 on the coordinator
sm_120: four original/high-V FP32-Q/BF16-Q cohorts passed **192 positive rows
(128 C1 + 64 baseline), 54 expected-negative children / 66 total children**.
An additional **64 hardware protocol probes** passed full output and partial
scans, covering empty splits, hidden NaNs, phase wrap and poison drains.
Runtime occupancy API reports **two CTA capacity** for both variants, not
achieved occupancy. No sanitizer or performance evidence is implied. Full
receipts/identities: `runs/20260923-i4/attn/attn-c1-qualified-5236a04/SUMMARY.md`.

Ten new N16 work-accounting checks pass, preserving the old N32 default; all
213 benchmark host checks pass. C1 timing dispatch/parser/sweep integration
is next, then 128K/1M both-precision measurements. Do not reuse the N32 padded
formula for C1: 1M/P255 needs 239 × 257 + 16 × 258 N16 tiles. Frozen R7,
separate native lattice, C3 baseline, C5 skip, parked C2 and required P1 remain.

## Previous execution — 24 September 2026, 04:56 AEST

C1 is now **implemented but not yet GPU-qualified or timed**. New explicit
entrypoints retain C3 and the scalar baseline. M16/N16, 2 producer / 2 QK /
4 PV warps, 44,416 B. sm_120 static registers **66 FP32-Q / 71 BF16-Q**, zero
stack/spills/local operations; QK/PV HMMA sites **36/8 and 12/8**. Per-function
shared carveout preference is not measured residency. Concrete subgroup and
mbarrier lifetime mapping, SASS, parser correction and limits:
`runs/20260923-i4/attn/attn-bench.k6voq2/SUMMARY.md`.

Next is the four-cohort C1-only external-oracle ladder plus eight explicit
hardware protocol probes per positive C1 child. Eleven parser tests pass,
including register-only arithmetic after release and rejected shared reads
past release. The bounded CPU model remains a storage-lifetime abstraction,
not proof of actual hardware overlap. Timing integration still needs N16
executed-work accounting. C5 skip remains closed; C2 parked; P1 required.

## Previous execution — 24 September 2026, 04:13 AEST

**C5 complete as an exact-contract availability skip**, not a throughput
measurement. The corrected one-shot at 03:48 reached native vLLM FA2 validation
on the coordinator sm_120: BF16 KV rejected V = 128 against QK = 192; E4M3 KV rejected
BF16 Q because query/key dtypes must match. Both checks used S = 257; no
128K/1M timing or successful attention dispatch occurred. Full evidence:
`runs/20260923-i4/attn/attn-c5.guT66C/SUMMARY.md`. No build, padding, install,
pull or retained-container change. This chooses neither production-bandwidth
branch; retain the target and continue R10, without repeating C5.

**C1 bounded draft, not implemented:** M16/N16, eight warps partitioned
2 producer / 2 QK / 4 PV, two converted slots and one raw slot. Typed shared
model **44,416 B**; two CTAs would fit the C3-observed shared-memory byte
budget, but no register/occupancy/performance claim exists. All packed operand
loads/K stores pass bank checks; known four-way V stores remain. **23 storage
checks + 56 protocol scenarios / 5,963 states** pass, including nine bad
protocol counterexamples and a wrong-global-alpha witness. The model explicitly
allows only **two simultaneous compute-stage owners**, not ideal three-way
overlap. Details and assumptions:
`runs/20260923-i4/attn/attn-c1-model.Zd5dec/SUMMARY.md`.

C3 CUDA behavior remains unchanged. Next is opt-in C1 implementation, actual
subgroup ordering/waits/drains, sm_120 resource checks and fresh qualification
before timing both precision rows. Frozen R7 and full-context P1 stay required.
The C3 timing summary also has a correction of record: **three warmups/seven
samples** in the paired sweep, not the single-cell two/five defaults; measured
numbers and verdicts are unchanged.

## Previous execution — 24 September 2026, 03:33 AEST

**C3 is GPU-qualified and measured**, source **0c948e2** (kernel unchanged
from e822cef). Four cohorts: **320/320 positive rows**, **104 complete
expected-negative children**. All nine timing points remain **MISS**;
full baseline/FP64-coordinate/finite checks passed every timing cell.
Full identity, all nine rows, four parity binaries and exact-binary NCU:
`runs/20260923-i4/attn/attn-bench.7LtCsi/SUMMARY.md`.

| Context / splits | FP32-Q, 8 warps: ms / GB/s | BF16-Q, 8 warps: ms / GB/s |
|---|---|---|
| 128K / 85 | **0.278432 / 602.561** | **0.267264 / 627.739** |
| 1M / 255 | **2.111072 / 635.780** | **1.934944 / 693.652** |
| 1M / 510 | 2.187648 / 613.525 | 2.004992 / 669.418 |

The one-stage/phase-sharing N4 amendment pays off despite its extra barrier:
FP32-Q latency improves about **35.6% / 37.2%** versus C0 at 128K / 1M-P255.
Best GA9 extrapolations **18.999649 ms / 17.414496 ms**, still MISS and not
engine latency. Native remains a separate lattice-local row, no X1c claim.

Actual NCU evidence resolves the carveout/occupancy uncertainty: **102,400 B**
shared configuration, **49,536 B dynamic + 1,024 B driver per CTA**, two
register/shared capacity CTAs, **15.925 active warps/SM / 33.178% occupancy**.
Eligibility rises from historical original-D1 **0.288887 to 0.709882 per
scheduler**; SM issue-active from **24.294% to 45.890%**, no-eligible cycles
from **75.46% to 53.76%**. DRAM **35.647%** and tensor **10.127%** are not
saturated. The V-store site still has **3,198,720 excessive wavefronts**;
post-QK reduction/retirement remains the dominant sampled barrier area.
These profile differences combine C0/K-swizzle and C3; no C0 profile exists.
The 340-CTA P85 grid is now one full two-CTA resident wave, not the old two.

Profile events were isolated from gate timings. Job exit 1 was the CPU
parser's rejection of demangled bool `1` rather than `true`, after the
capture completed. Fixed and CPU-reparsed; false/0 still refused. No GPU
rerun or replacement of measurements.

**C5 is not yet an achievability answer.** The pinned existing vendor image
contains FlashInfer without an AOT/cache package and prebuilt vLLM FA2.
The one-shot reference attempt stopped before native dispatch because our
no-build guard also blocked read-only dependency discovery. CPU-only
`ldconfig -p` / `uname -p` diagnosis is complete; exact read-only forms now
allowed, all build/install paths still denied, import passes. See
`runs/20260923-i4/attn/attn-c5.qlW81d/SUMMARY.md`. This is **incomplete**, not
proof of vendor unavailability. Next allowed GPU job is C5 only, at exact
192/128 dimensions; unsupported means explicit skip, never padding/building.
Do not infer either production bandwidth branch or re-plan the target from
missing data. R11 no-widening decision recorded in the packet; no questions.
C1 design waits for that reference/skip; P1 full-context work still required.

## Previous execution — 24 September 2026, 02:41 AEST

C0 GPU qualification is complete: **320/320 positive rows**, **104 complete
negative children** across FP32-Q/BF16-Q original/high-V cohorts. Source
`4b65aa8`, kernel identical to `0fbafec`. All **9 timing points MISS**; full
identity and limits in `runs/20260923-i4/attn/attn-bench.erWk7T/SUMMARY.md`.

| Context / P | C0 FP32-Q ms / GB/s | C0 BF16-Q ms / GB/s (8 warps) |
|---|---|---|
| 128K / 85 | .432576 / 387.844 | .434752 / 385.903 |
| 1M / 255 | 3.359744 / 399.488 | 3.381056 / 396.970 |
| 1M / 510 | 3.424544 / 391.929 | 3.446080 / 389.479 |

C0 misses the predicted 530–610 GB/s and does not beat original D1 at 128K.
Native BF16-Q regresses roughly 12% from its prior row; retain the better
historical timings, not a precision promotion. Best C0 FP32-Q GA9 is
**30.237697 ms extrapolated**, not engine latency. No new C0 Nsight capture;
no claim of measured eligibility improvement or isolated accumulator benefit.

### N4 amendment for C3 — implemented candidate, not GPU-qualified

C3 keeps M16/N32 and C0 arithmetic, but replaces the two raw stages with one,
removes the unused math-level physical-index array, and reuses converted K's
12,288 B region for disjoint score / P-high / P-low / alpha storage after QK.
Typed storage in `decode_pipe_storage.h` is **39,168 B math + 10,240 B raw +
128 B indices = 49,536 B**. The prior **53,824 B** one-stage-only caveat was
correct; phase reuse provides the additional reduction. The two-CTA model
also fits an assumed 1 KiB per-CTA reservation under 100 KiB. **Actual CUDA
capacity and achieved occupancy/eligibility are still mandatory readbacks.**
Benchmark timing refuses fewer than two theoretical CTAs/SM; this is not proof
of achieved occupancy. Shared-byte metadata comes from the real typed layout.

Lifetimes and costs (this explicitly amends N4's double-raw-stage schedule):

1. Commit raw tile 0, then `wait_group 0` and CTA synchronization before reads.
2. Convert visible K/V; all raw readers meet the recycle barrier. Only then
   prefetch tile t+1 into the same raw slot, concurrently with current math.
   Invisible tiles still advance the queue before their uniform continue.
3. Keep four final QK scores per computing thread in registers. The existing
   bad-score reduction barrier retires **all** K readers before score stores
   enter the overlapping region. Error exits drain async copies before return.
4. An **additional CTA barrier** publishes scores before softmax. Score,
   P-high, P-low and alpha occupy disjoint subregions; m/l and V stay outside
   the union. Existing P publication and tile-end barriers protect consumers
   before the next tile's K expansion. Normal exit retains the explicit drain.
5. All nine Q/K/V/P/score/alpha operand bases preserve their bank residues
   modulo 128 B. Raw/index bases merely rotate banks uniformly. Existing K
   producer/consumer proofs still apply; the known V producer conflict remains.

Host layout checks **67**, benchmark parser **66**, NCU parser **10**; **300**
distinct host checks pass (shared guard suite counted once). Both CUDA binaries
build for sm_120. Pipeline registers remain FP32-Q **124 / 92**, BF16-Q
**124 / 70**, zero stack/spills and no local ops; all static MMA counts pass.
Clean committed source **e822cef** was rebuilt at 02:49 AEST; full counters,
source/binary hashes and four kernel bodies are preserved in
`runs/20260923-i4/attn/attn-bench.RRIROa/SUMMARY.md` (02:51 AEST).
No C3 GPU correctness, occupancy, timing or counter claim yet. Check the
actual shared/L1 carveout and achieved residency, not just capacity math.

Next managed ladder: four fresh precision/cohort gates, then
`bf16q-sweep-profile`. It records nine uninstrumented timing points first,
then one exact-binary FP32-Q 128K/P85/w8 full NCU capture in a **separate**
profile subtree. Instrumented samples never enter timing verdicts. The old
original-D1 profile remains the historical counter baseline, **not an isolated
C0 counter baseline**. C3 vs that profile must be labelled a combined change.
C1 must re-prove lifetimes: it may not expand future K over current P/alpha;
this phase alias does not provide a free second converted buffer. C2 parked;
P1 full-context prefill remains pending. No teammates or expert/repack edits.

## Execution record — 24 September 2026, 01:59 AEST

The explicit BF16-Q decode row is implemented at `bc69aeb` and GPU-qualified
against its **own post-RoPE BF16-RNE lattice**, not the FP32-Q reference. Both
native original/high-V cohorts passed **80/80** positive rows and **25** complete
negative children each. Timing queries remain BF16-exact; the separate parity
fixtures retain unrounded inputs to test the kernel cast. No X1c/model promotion.

Same-binary eight-warp FP32-Q / BF16-Q timings (coordinator, sm_120 / 170 SMs):

| Context / P | FP32-Q ms / GB/s | BF16-Q ms / GB/s |
|---|---|---|
| 128K / 85 | .447008 / 375.322 | .388416 / 431.939 |
| 1M / 255 | 3.467584 / 387.064 | 3.020128 / 444.411 |
| 1M / 510 | 3.546752 / 378.424 | 3.071776 / 436.939 |

All nine points, including four-warp native controls, are **MISS**. Best native
GA9 is **27.181152 ms extrapolated**, not engine latency. Actual native padded
work factors are 1.42358398 / 1.40542603 / 1.41632080, nominal 1.4. Detailed
identity, limits and raw child receipts: `runs/20260923-i4/attn/attn-bench.I3oRDS/SUMMARY.md`.
FP32-Q remains default; the historical FP32-Q best is retained.

Builder's 00:21 R10 adoption supersedes the 23:48 temporary redesign hold:
**C0 accumulator parallelism → C3 occupancy → C1 producer/consumer barriers**;
both FP32-Q and BF16-Q at every timing pass. C2 native FP8 is parked under R7.
The review's bandwidth estimates remain predictions, not passed gates.

C0 is now a **CPU/codegen-validated candidate, not GPU-qualified**. Six QK sets
(three Q terms × two alternating k-groups; native has two), four persistent PV
sets per output fragment (two P terms × two k-groups). Every PV set receives
alpha before accumulation, then the sets combine only at final emission. The
K raw-chunk swizzle / bank-safe producer map is retained as the ride-along.
The known-slower F4 V transpose and its two barriers are removed; V returns to
original D1 traversal, with its known four-way producer-store conflict. This is
a combined candidate, not an isolated accumulator-only attribution experiment.
Cache ABI, consumer layout, Q/P terms, mask/sink logic and final reducer stay
unchanged. Raw-slot recycle barrier, queue advance on invisible tiles, and
normal/error exit drains remain in place. No mbarrier redesign has begun.

Static sm_120 codegen: FP32-Q registers **124 / 92**, BF16-Q **124 / 70** for
four/eight warps; all four have **zero stack/spill bytes and no LDL/STL**. The
four-warp variants exceed the soft 96-register aim; no spilling cap is forced.
Eight-warp FP32-Q fits that aim. New static checks reject spills, incomplete
variant coverage, wrong MMA counts and collapsed accumulator destinations.
Fresh original/high-V GPU parity in both precisions is required before C0 timing.
Committed C0 source **0fbafec** rebuilt cleanly; full four-body SASS, matched
ptxas counters, binary/source hashes and 281 host checks are recorded in
`runs/20260923-i4/attn/attn-bench.nYIv4g/SUMMARY.md`. No C0 GPU run yet.
C3 sizing caveat: one fewer raw stage plus physical-index array yields
**53,824 B**, not ≤50 KiB; that alone cannot establish two CTAs/SM under a
100 KiB budget, even ignoring per-CTA reservation. Prove further reduction
or the N16 choice, then measure occupancy/eligibility; do not assume the gain.
P1 and actual full-context prefill remain pending.

## Latest numerical ruling — R7, 20:58 AEST

The attention output bound is now **`2e-5 · max(1, max|v|)`**. Q3/P2 and
factor **2.6** remain. This explicitly supersedes older fixed-bound/full-range-P
blockers below; no P3 implementation is planned. The numerical-contract section
contains the decoded-V definition and the **8/8** case audit. Prior stricter
checks and raw receipts remain unchanged. N3/N4 are CLOSED by reviewer ACCEPT.
The N5 context addendum now explicitly separates measured 1M decode from
**modeled**, not measured, 128K prefill. A full-context prefill GPU measurement
remains P1 work; interior tile timings are never relabelled as that run.

## Governing update — builder ruling at 20:15 AEST, 23 September 2026

N3/N4 are **accepted as closed on paper** at `77bfe02`; reviewer confirmation
may amend them later, but does not block work. The new order is **D0.1 dev-host parity
and the coordinator timing → N5 eta micro-cell → D1 pipeline code → P1**. Dev-host parity is
complete at 16/16 with 12/12 isolated negative gates; do not rerun that ladder.

The qualifying `attn_prefill_f32q` gate remains **125.7 executed MMA TFLOPS**.
At actual factor **2.6**, the useful equivalent is about **48.3 TFLOPS**, not
62.9. There is no separate 62.9 useful-TFLOPS requirement. Report actual padded
MMA work, never nominal factor 2. `attn_prefill_bf16q` remains a separate opt-in
cell targeting 100 useful TFLOPS at nominal factor 1.4, excluded from oracle
parity/model-semantics claims. One template has a compile-time residual switch.
F32-Q remains the engine default pending I5/X1c; BF16-Q is never silent and
scaled-FP16 P stays rejected. Earlier contradictory gates/order are historical.

ATTN-LEAD owns the coordinator's sm_120 / 170-SM RTX 5090 and runs its own timed cells;
The dev host is shared development/parity. Protected containers remain untouched. Use
own prefixed executables and unique `/var/tmp/mimo26f-attn/mimo26f-*` slots,
owner/memory guards and locks; text receipts return to `runs/`.

**D0.1 timing scope:** the unchanged synthetic workload has BF16-exact values
in its FP32 Q buffer and finite unit E4M3 K/V with |value| ≤ 1.875. Candidate
selection is explicit (`MIMO26_ATTN_DECODE_IMPL=tc`), with separate FP32 scratch,
no prefill fallback, a full finite scan, three independent FP64 coordinates,
and an untimed 8192-output comparison against the FP64 baseline at each context.
Timed events cover candidate decode plus merge only; candidate warmups repeat
after reference execution. The high-value P limitation below stays open for
full-range qualification; these bounded-input timings do not waive it.

## F4 GPU result — 23:35–23:37 AEST: correct, not faster

Source `c595a10`. Original corpus **80/80** positive rows, **27** complete
negative children (`attn-parity.kK1nw1`); high-value cohort the same counts
(`attn-parity.inR73k`). Maxima remain **5.811e-7 / 0.001740** under the existing
original / R7 high-V contracts. Each cohort covers 32 children; no threshold
change or arithmetic-mode change.

All **16** timing points are measured **MISS**, now including corrected 1M
P = 255 / 510 gates. `attn-bench.DWWYyA/SUMMARY.md` has the full table and
matched-run caveats. Eight-warp gate results:

| Context | P | Median ms | Unique KV GB/s | Executed TFLOPS |
|---|---:|---:|---:|---:|
| 128K | 85 | 0.446112 | 376.076 | 31.816644 |
| 1M | 255 | 3.481312 | 385.538 | 32.201063 |
| 1M | 510 | 3.548672 | 378.220 | 31.834714 |

At identical 128K P = 85, old D1 was **0.427296 ms**: F4 is approximately
**4.4% slower**. All twelve matched points are slower. No old P = 255 / 510
control exists; their scheduling improvement must not be credited to F4.
Best corrected 1M GA9 is **31.331808 ms extrapolated**, not engine latency.
Transpose/synchronization costs are real; bank cleanup alone is not sufficient.

Exact-binary SASS and Q-region inventory are archived beside the receipt:
zero global/shared byte loads and shared byte stores in both D1 variants,
no post-prime direct branch back into Q packing. These are static checks;
no new measured conflict counters or profiler speedup claim. Keep the old
best record and this experimental path; no promotion. Explicit BF16-Q,
stall-directed overlap and P1 remain pending.

## F4 measured-hotspot intervention — 23:33 AEST host checkpoint

D1 now permutes aligned raw 16-byte chunks and gives converted K/V producers
an 8-row × 4-packed-pair warp ownership map. Host enumeration proves **32
separate banks per converted store**, with conflict-free raw expansion reads
(including shared-word broadcasts). Negative witnesses reproduce the old
four-way producer stores. MMA consumer layout, Q3/P2 arithmetic and global
cache ABI are unchanged; no warp specialization or BF16-Q switch is bundled.

V is explicitly transposed **inside shared memory** to a swizzled d-major plane.
Each 4 × 4 byte square uses four shuffles and three byte permutations; all reads
finish before the in-place overwrite, and a second barrier publishes the result.
Expansion then uses one aligned 32-bit raw load per BF16 pair rather than two
byte gathers. Both barriers and the transpose are charged to kernel timing.
Transpose ingress remains at most four-way; it is not claimed conflict-free.
No extra shared allocation: **64,192 B**. Compiler registers **96 / 56** for
four / eight warps, zero stack/spills. This is a latency/register tradeoff to
measure, not a forecasted win.

Layout/codec tests **48/48** (22 new producer/transpose checks), host total
**253/253**, sm_120 parity and benchmark builds PASS. GPU original/high-value
corpora and the corrected full-wave timing ladder follow. No new GPU result is
implied by these host proofs.

## D1 Nsight profile before redesign — 23:12 AEST

Per the 23:06 ruling, one unchanged **128K / P = 85 / eight-warp** D1 launch
was captured with NCU **2025.3.1, `--set full`, 39 replay passes**, source
`1e892cd`. Full text and SASS-correlated counters are in
`runs/20260923-i4/attn/attn-bench.hjP3e7/`; `SUMMARY.md` gives scope and attribution.

**Measured:** achieved occupancy **16.64%** (theoretical 16.67%); eligible warps
**0.29 per scheduler**, no eligible warp **75.46%**, issue slots busy **24.29%**.
DRAM **22.87% / 403.66 GB/s** read + write, L2 hits **6.58%**, tensor-pipe active
**6.41%**, ALU / ALU-heavy **13.71% / 22.65%**. Tensor activity is not the same
metric as executed TFLOPS divided by the paper's assumed 209.5 TFLOPS.

Largest warp-state contributions per issued instruction: **Wait 2.43, Barrier
2.06, Short Scoreboard 1.44**, versus **Long Scoreboard 0.21**. Shared stores
incur **9,961,075 bank conflicts**, average **3.6-way**. Converted K/V stores
at PCs **0x2db0 / 0x34a0** contribute **4,798,080 / 3,198,720 excessive shared
wavefronts**. The post-QK barrier accounts for **6,383 sampled barrier stalls**;
short-scoreboard hotspots are expansion's visibility/raw-code dependencies.

Diagnosis: **low latency hiding and producer/shared-memory dependencies plus
CTA synchronization**, not a saturated roof or incomplete CTA waves. Target
F4's producer path and the larger K-store hotspot before the larger overlap
redesign; test producer bank maps, not only MMA consumers. Carry BF16-Q as its
own row, then evaluate overlap against these counters. No assumed speedup.

Cold replay caches; clocks not locked (NCU warning retained). The profiled kernel
alone is **430.144 µs**; instrumented event medians are explicitly rejected as
benchmark evidence. The original **0.427296 ms** uninstrumented split-plus-reduce
result remains the timing record. Host NCU was absent, so an isolated, temporary
`mimo26f-` container used the cached CUDA toolkit; no driver/service changes.
Full baseline/oracle checks pass, and only text receipts returned. Parser has
**8/8** tests plus an instrumented-timing rejection test; host total **231/231**.

## D1 measured results and review corrections — 23:01 AEST

The **22:35–22:36 AEST** run at source `1e892cd` preceded the 22:39 review
ruling. High-value receipt `attn-parity.8rZJXq`: **80/80 positive comparisons**,
**27 complete negative children**, both warp variants, P = 8 and P = 3. Maximum
candidate error **0.001740**, below the relevant R7 **0.00896** bound; baseline
external-oracle maximum **0.001465**. Original fixtures remain unchanged.

Timing receipt `attn-bench.HWQgqN`: **12/12 valid measurements, all MISS**.
Best 128K is **P = 85, eight warps: 0.427296 ms / 392.637 GB/s / 33.217691
executed TFLOPS** (padded factor **2.64379883**). Best measured 1M is **P = 512,
eight warps: 3.587392 ms / 374.137 GB/s / 31.128225 executed TFLOPS**; GA9
**32.286529 ms extrapolated**, not engine latency. That 1M point is now a
partial-wave diagnostic, not a corrected gate point. Eight warps win all matched
pairs. Every point passes full 8192-output baseline, three FP64-coordinate and
finite-scan checks before timing. No stall attribution without counters.

**F1/F5:** auto defaults now **85 / 510** at 128K / 1M; the next f32q sweep also
includes **255** at 1M. Previous 64/128/256/512 points are retained as diagnostics
with wave caps in the aggregate receipt and corrected schedule below. Nine new
CPU schedule checks pass; host total **222/222**, sm_120 build PASS.
**F2:** the R1 schedule/shared budget and `2*F` model are corrected below; old
numerical forecasts are explicitly withdrawn, not re-labelled as measurements.
**F3:** `attn-parity.IsgI7T/{identity.md,sass/RESULT.md}` confirms zero global byte
loads/shared byte stores and Q packing outside the KV loop in the exact qualified
binary. This is instruction evidence, not measured overlap efficiency.

**Open next:** F4 packed d-major raw V, an explicit opt-in BF16-Q decode row,
corrected 1M P = 255/510 timing, and the approved warp-specialized expansion
lever. A contiguous 16-byte copy cannot itself transpose token-major global V;
any in-shared transpose must preserve the cache ABI and charge its real work.
Fresh correctness is required after kernel edits. P1 remains required afterwards.

## R7 stress and D1 timing preparation — 22:32 AEST

A separate `tc-highv` cohort now reuses the eight original Q/K/position/sink
inputs but amplifies decoded cached V by 448 and re-encodes E4M3. Names carry
`tc_highv_`; original fixtures and their 1e-5 absolute thresholds stay unchanged.
Decoded maxima are **224** in the first four cases and **448** in the remaining
four, with explicit R7 bounds **0.00448 / 0.00896**, zero relative tolerance.
This is artificial cache-amplitude stress, not a second 0.707 prescale. Four
CPU isolation/range tests and eight external-oracle generations pass. GPU
requalification follows; no high-value PASS is inferred from CPU preparation.

D1 is wired into the benchmark as explicit `pipe4` / `pipe8`, with distinct
kernel names, resource readback and unchanged full 8192-output scalar-baseline
plus three independent FP64-coordinate checks. Dynamic-shared configuration is
outside timing. Actual padded Q3/P2 MMA work is validated, including P = 85.
The `d1-sweep` wrapper runs the 12 planned points with **3 warmups / 7 samples**,
collects separate per-point raw receipts, and rejects incomplete/invalid reports;
measured MISS remains MISS. P = 256 at 128K is diagnostic. Both dev-host build locks
are released before remote work. Host checks now total **213/213**, plus shell
syntax; sm_120 build passes. High-value parity precedes this sweep.

```bash
HOST=coordinator MIMO26_ATTN_BUILDER_WINDOW=1 scripts/dev.sh test attn tc-highv
HOST=coordinator MIMO26_ATTN_BUILDER_WINDOW=1 scripts/dev.sh test attn-bench d1-sweep
```

## D1 coordinator corpus qualification — 22:14 AEST

Source `ef95e61`, receipt `runs/20260923-i4/attn/attn-parity.IsgI7T/RESULT.md`.
**PASS:** scalar baseline 16/16; each D1 variant (4 and 8 warps) 16/16 at
P = 8 **and** P = 3: **80/80 positive output comparisons** overall. Maximum
candidate error **5.811e-7**, below the retained 1e-5 fixture bound. R7 V audits
are included in the actual GPU receipt. Packed device codec **65536/65536**.
Both candidates detect **12/12 isolated negatives**, and all three paths detect
the original combined negative: **27 complete numerical-negative child runs**.
No missing row, crash or fallback is counted. All baked probes report sm_120,
170 SMs, device code 1200. Hardware resource readback confirms 61/48 registers,
64,192 B shared and **one active CTA per SM**. No dev-host GPU used.

This closes correctness on the original bounded corpus, **not** D1 performance
or high-value qualification. Next: R7 high-value regression, the 4/8-warp split
sweep (128K P = 64/85/128, 256 diagnostic; 1M P = 256/512), counters/sanitizer
as available, then P1. No latency improvement is claimed before measurement.

## N5 context addendum and D1 implementation — 22:06 AEST

The context report at `runs/20260923-i4/attn/n5-context/SUMMARY.md` uses validated
existing receipts, not new GPU data. The measured 1M decode anchor is
**14.484512 ms / eta 0.03680 / MISS**. It separately carries R7's ideal
**0.889 ms compute / 0.895 ms bandwidth** co-bound model.

For **S = 131072, T = 2048** prefill, mechanically applying measured M64/N64
rates to padded causal work gives **498.85 ms f32q / 330.94 ms bf16q**:
**modeled MISS**, not measured prefill latency. The required 3–5× planning
allowances are **1496.54–2494.23 / 992.83–1654.71 ms**. The report also gives
full T = 131072 explicitly; no hidden query-count interpretation. These rates
come from resident/repeated micro-tiles and do not qualify streaming, online
rescale or mask-boundary costs. Context accounting tests **10/10** pass.

**D1 code is now implemented, hardware parity pending.** The original D0.1
entrypoint and scalar baseline remain intact. One 4/8-warp D1 template uses two
16-byte `cp.async` raw stages, zero-filled split tails, wait-current and uniform
barriers, packed bit-exact E4M3→BF16 conversion, and complete final/error drain.
Only four warps execute QK; all warps expand/PV, with four/two PV fragments each.
Invisible tiles still advance the async queue; raw slots are recycled only
after all expansion readers finish. Query remains Q3, P remains P2.

Shared budget **64,192 B**, explicit dynamic opt-in configured outside timing;
sm_120 ptxas **61 registers (4 warps) / 48 registers (8 warps)**, zero stack/spills.
All **65,536** packed code pairs pass the host bit test; an equivalent GPU probe
is in the parity executable. New layout tests include eight-warp output coverage.
Host checks total **204/204**, plus the remote-shell syntax gate. A host-pass
`__CUDA_ARCH__` probe compile error was caught and fixed; slot `attn-parity.yDUm6c`
then built successfully. No GPU qualification is inferred from that build.

The new guarded coordinator parity adapter ships only its executable, corpus and text
metadata into a unique own scratch slot; no Python/CUDA toolkit is needed there.
It runs the scalar baseline, D1 4-warps and D1 8-warps; candidate positives use
P = 8 **and P = 3** so the 512-token case actually recycles both raw slots.
P = 8 alone exercises only two tiles and would miss that lifetime test. Each
candidate then runs the original combined negative and all 12 isolated negatives
at P = 3. Full rows, exact dispatch, packed GPU codec and baked sm_120 / 170-SM
identity are required. No dev-host GPU is used by remote mode. Command:

```bash
HOST=coordinator MIMO26_ATTN_BUILDER_WINDOW=1 scripts/dev.sh test attn tc-decode
```

## N5 measured — 21:12 AEST, source `841071e`

the coordinator RTX 5090, sm_120 / 170 SMs, driver 610.43.02; command
`HOST=coordinator MIMO26_ATTN_BUILDER_WINDOW=1 scripts/dev.sh test attn-bench eta`.
Each row is 680 CTAs × 32 repeated interior tiles, seven event samples.

| M/N | Q mode | Median ms | Executed TFLOPS | Useful TFLOPS | Eta / 209.5 | Verdict |
|---|---|---:|---:|---:|---:|---|
| 64/64 | f32q | 2.607744 | 56.873 | 21.874 | 0.2715 | **MISS** |
| 64/64 | bf16q | 1.730016 | 46.161 | 32.972 | 0.2203 | **MISS** |
| 32/32 | f32q | 1.091328 | 33.975 | 13.067 | 0.1622 | **MISS** |
| 32/32 | bf16q | 0.755936 | 26.411 | 18.865 | 0.1261 | **MISS** |
| 32/64 | f32q | 3.157632 | 23.484 | 9.032 | 0.1121 | **MISS** |
| 32/64 | bf16q | 2.477056 | 16.120 | 11.514 | 0.0769 | **MISS** |

All **12/12** full-output instrumented/uninstrumented checks and **9/9** isolated
numerical negatives passed. Maximum positive error **1.0165e-6**. The validator
reports **VALID / TARGET**, all six rows MISS, expected exit 1. No full-prefill
or lattice eligibility promotion. N5 measurement is complete; the previous
eta 0.75 / 0.85 assumptions are **not qualified** and P1 tiles are not frozen.

Instrumented phase fractions (KV load/expand, Q pack + QK, softmax/high-P, PV/low-P):
M64/N64 f32q **40.9 / 48.6 / 2.0 / 8.5%**; bf16q **61.1 / 23.2 / 3.0 / 12.7%**.
M32/N32 f32q **50.7 / 43.3 / 1.7 / 4.4%**; bf16q **71.3 / 19.7 / 2.8 / 6.2%**.
Softmax/high-P alone is not the dominant measured bucket; neither Q packing
nor QK MMA is separately resolved within their combined bucket. M32/N64 has
only one resident CTA and is worse here. The instrumented phase means are not
additive wall times and do not constitute Nsight stall/DRAM attribution.

SASS confirms byte global/shared loading (`LDG.E.U8`, `STS.U8`), repeated FP32
Q loading/packing and native `HMMA.16816.F32.BF16`. Evidence and exact commands:
`runs/20260923-i4/attn/attn-bench.B9h3B2/{RESULT.md,SUMMARY.md,identity.md}`.
Next is **D1**: two raw 16-byte async stages, packed exact E4M3 expansion,
4/8-warp alternatives, dev-host-independent coordinator correctness/timing, then counters.
P1 must remove unnecessary Q reloading and revisit its measured eta; simply
hiding softmax is not supported as the main fix by this cell.

The new N5 receipt validator has **10/10** synthetic tests; total distinct host
checks **192/192**. `eta-summary` and `disassemble` are CPU-only wrapper verbs.

## N5 micro-cell implementation — before D1

`attn_eta.cu` uses one template with a compile-time Q residual switch, at
M64/N64 (8 warps), M32/N32 (4 warps), and M32/N64 (4 warps, comparison).
M64 inputs use four GA-style positions / 16-head groups; M32 inputs use four
SWA-style positions / 8-head groups. This is an interior tile, not complete
prefill: no boundary masks, sink merge, online inter-tile rescale or streaming
DRAM qualification. Q is generated/rotated in FP32 on the host. BF16-Q rounds
only afterwards; its reference is the explicitly rounded lattice, not f32q
oracle parity. K/V and cache representation are named on every output row.

One shared Q plane is reused across Q3 correction passes; P is also reused
between high and residual PV products. Synchronous raw-FP8 staging is measured,
not a claimed async pipeline. Shared footprints are **95,232 / 45,568 / 78,336 B**.
Uninstrumented sm_120 register counts (f32q / bf16q) are **126/125, 93/92,
124/125**, respectively, with zero stack/spills. Clock-instrumented variants
are separate instantiations and can use more registers; their phase fractions
are diagnostic, not additive wall-clock or uninstrumented cycle accounting.

Per configuration: full FP64 tile reference against every output of every CTA
for both instrumented and uninstrumented variants; isolated missing-Q-correction
negative on f32q and missing-P-residual negative on both modes; then three
warmups and seven uninstrumented CUDA-event samples. No NaN/error exit counts
as a valid negative. GPU checks must finish before throughput is reported.

Following the review at `a8b28f1`, N5 pins **|V| ≤ 1.25** (K ≤ 1.875), within
the review's P2 domain, without relaxing the **2e-5 absolute** check. This is
still empirical accumulation qualification, not a universal bound. The earlier
D0.1 values up to 1.875 were individually checked against all baseline outputs;
that receipt never certified the whole value domain. General high-value P
qualification remains open. No scale-aware tolerance was silently adopted.

Host metrics/layout tests **36/36** and sm_120 build pass (slot
`attn-bench.Garabw`); overall distinct host checks now **182/182**. Hardware
receipt follows the source commit. Run only through the existing guarded runner:

```bash
HOST=coordinator MIMO26_ATTN_BUILDER_WINDOW=1 scripts/dev.sh test attn-bench eta
```

## First direct coordinator D0.1 timing — 20:47 AEST

Clean source `e43fb09`, RTX 5090, sm_120 / 170 SMs, driver 610.43.02,
CUDA 12.8 static cross-build. Command:

```bash
HOST=coordinator MIMO26_ATTN_BUILDER_WINDOW=1 MIMO26_ATTN_DECODE_IMPL=tc \
  scripts/dev.sh test attn-bench decode-long
```

| Context | Splits | Median ms | Unique KV GB/s | Executed MMA TFLOPS | Verdict |
|---|---:|---:|---:|---:|---|
| 128K | 128 | 2.112512 | 79.418 | 6.607604 | **MISS** |
| 1M | 512 | 14.484512 | 92.663 | 7.709555 | **MISS** |

Nine GA layers at 1M extrapolate to **130.360611 ms: MISS** against 10 ms;
this is not engine latency. Each median uses five samples after two warmups
(repeated after reference execution). The kernel remains synchronous D0.1,
not the proposed D1 pipeline. Useful unique-KV rates are only 4.44% / 5.18%
of nominal bandwidth; they are not measured DRAM counters. End-to-end executed
MMA rates are 3.15% / 3.68% of the assumed dense peak, not an isolated eta test.

AOT and full finite scans passed. Each cell passed all **8192** baseline-output
comparisons; max differences were **2.4214e-8 / 1.3039e-8**. Three independent
FP64 coordinates per cell passed, max **7.8794e-9 / 2.3201e-9**. These checks
cover the declared bounded synthetic inputs only. The owner guard found no
CUDA owners before launch; minimum observed free memory was **31,438,995,456 B**.
Only own scratch `/var/tmp/mimo26f-attn/mimo26f-jnXbML00` and prefixed executable
were used; no container/service/driver changes.

Raw receipt: `runs/20260923-i4/attn/attn-bench.QIe3t1/RESULT.md`.
Validator: same directory `SUMMARY.md`, **VALID / TARGET / MISS**, exit 1
(expected performance MISS, not incomplete evidence). HANDOFF line posted.

**Next: N5, not D1 yet.** Measure M64/N64 with 8 warps and M32/N32 with 4 warps
(and M32/N64 as a comparison) in a standalone QK→softmax→PV micro-cell.
Separate load/packing, QK, softmax/P formation and PV intervals; report clock
intervals as diagnostics alongside CUDA-event total throughput. Keep actual
Q3/P2 versus explicit Q1/P2 work accounting and correctness checks. Reuse one
shared Q plane across correction passes if needed: storing all three M64 Q
planes would exceed the original 93-KiB budget. Do not infer a memory-only
bottleneck or promise D1 closes this large gap without measurements.

## Q-tail correction D0.1 — 20:03 AEST

The first correction adds a third BF16 Q component, leaving P high/residual,
FP32 statistics/output, baseline entrypoints, and tolerances intact. Actual MMA
work on unpadded GA tiles is **(3×192 +2×128)/320 =2.6× useful**, not 2×.
CPU ideal-accumulation error on the tail construction falls from **2.4667e-4**
to **5.6198e-7**; this is not itself a GPU qualification. The old failing
2-component variant remains in the probe. 24 probe assertions include numeric
reconstruction of 8192 sampled normal FP32 operands, not a universal proof.

The unchanged external oracle now generates eight cases /16 flat-paged
comparisons at 1e-5. `tc_q_tail` has an independent tanh check and an isolated
`DROP_Q_TAIL` negative. Original seven fixtures keep their inputs/order.
Both architecture builds pass: corrected decode **80 registers sm89 /64 sm120**,
**43,456 B shared**, zero stack/spills; merge remains39/40 registers and544 B.
Host checks159/159, oracle8/8; GPU correction requalification is next.
Receipts: `attn-parity.JxJiIV`, `attn-bench.KvQ9lM`, `attn-parity.LojLBp`,
`attn-parity.YyByuN` under `target/mimo26f-builds`.

### D0.1 dev-host requalification — 20:13 AEST, source77bfe02

Baseline and corrected candidate each passed **16/16** full flat/paged
comparisons. Original negative and **12/12 individual negatives** each completed
16 rows and failed numerically on the candidate path. Max candidate error
**5.811e-7**; `tc_q_tail` error **5.353e-7**. Disabling only the Q tail produces
**2.466e-4** on both flat and paged inputs. Guard accepted only the two resident
modules, with reserve4096 MiB maintained. This is sm89 development parity,
not a sm120 timing/serving claim.

Exact completed text receipts, copied without transformation and byte-compared:
`runs/20260923-i4/attn/d01-baseline-77bfe02/RESULT.md` and
`runs/20260923-i4/attn/d01-tc-77bfe02/RESULT.md` (slots iHnrf1/Ef0qDh).

**Remaining full-range precision stress:** the P-rounding construction used
V = ±1. Its ideal two-term-P error scales with V; multiplying V by 448 predicts
about **6.84e-5** error, above 2e-5. Add this independent-oracle high-value case
and address any failure before general full-range qualification. The current
16-row pass does not qualify the full E4M3 value range or prove a universal
FP32 error bound. The 20:15 ruling puts bounded D0.1 timings first.

## D1 paper amendment — before pipeline code (N3/N4)

**N3:** 128K sweeps P = 64/85/128, all within the adopted 64–128 range;
P = 256 is a diagnostic comparator, not the default. GA grids contain 4P CTAs:
256/340/512, or 1.51/2.00/3.01 CTAs per 170 SMs. P = 85 tests integral grid
waves when a deeper pipeline permits only one resident CTA. Grid count is not
occupancy; padding and tail-wave losses must be measured. 1M initially tests
P = 256/512; short contexts remain diagnostics.

Use the existing byte formula, but replace `2F` with
`2.6*40960*sum_s(32*ceil(split_length_s/32))`. Useful FLOPs remain `40960*S`.
For 128K, the 1253-GB/s gate permits **133.90 µs total**. With 8 µs of
non-bandwidth launch/merge allowance, P = 64–128 requires roughly **78.2–80.1%**
modeled DRAM efficiency and **52.9%** MMA efficiency (P = 85 needs about 53.8%
including padding). With 15 µs allowance the requirements rise to about
**82.8–84.8%** DRAM and **56.0%** MMA (about 57.0% at P = 85). These are
conditional roofline requirements, not forecasts; conversion, barriers and low
residency can invalidate them. Retain the 3–5× planning allowance until actual
receipts. Fewer splits reduce partial traffic but can lose grid balance.

**N4:** first pipeline candidates use two 16-byte `cp.async` raw KV stages,
M16/N32, with 4- and 8-warp instantiations. Query precision stays corrected.
Each raw stage is `32*(192+128)` = 10,240 B; adding two raw stages and separate
position/visibility metadata to D0.1 is approximately **64,192 B shared**.
Use opt-in dynamic shared memory and check the device limit: this is **one**
resident CTA under a 100-KiB SM budget, not two. In the 8-warp variant, four
warps own QK fragments, all eight cooperate on expansion/PV, with two PV
fragments per warp instead of four. Keep both variants until measurements.

Pipeline: prime current raw stage; prefetch the next slot; wait for current
async group plus CTA visibility barrier; expand/compute current; CTA barrier
before reusing its slot. Last tile drains all groups; tail copies zero-fill;
all-masked tiles still participate in uniform synchronization. Maintain
absolute-position/page masks in the matching slot. No source buffer is reused
while async readers or consumers are live. Packed E4M3→BF16 expansion is an
additional intended optimization, requiring exhaustive host codec checks.

Select via ptxas/SASS, occupancy API, Nsight DRAM bytes/throughput, tensor activity
and stall evidence, not occupancy guesses alone. First require baseline/corrected
D0/pipeline flat-paged two-run parity, then memcheck/racecheck and scale-up.
Timed128K/1M cells go to the coordinator only, with explicitly logged implementation,
splits, precision work and source. No P1 tile is frozen before N5's eta cell.

## Bottleneck analysis first — 23 September 2026 AEST

**Static analysis, not measured performance.** The I3 kernels are correctness
baselines: `attn_decode_splitkv.cu` assigns one thread to a query/head/split;
`attn_prefill_chunk.cu` assigns one thread to a query/head. Both loop serially
over keys and QK192/V128 dimensions, using FP64 online softmax/output sums.

Consequences to verify with timings/profiling:

1. No tensor-core MMA instructions. Prefill cannot approach the requested
   100 TFLOPS BF16-equivalent by exploiting the GPU's BF16 tensor capacity.
2. Each thread owns K[192], V[128], and FP64 output[128] arrays. Dynamic
   indexing and register pressure can spill to local memory. Spill volume is
   not measured yet; do not equate useful KV bytes with physical DRAM traffic.
3. Adjacent decode threads traverse different split ranges. Their scalar KV
   loads are not a cooperative contiguous vector load. GQA shares the logical
   KV head but the implementation independently loads it for each Q head
   (16 GA or 8 SWA Q heads per KV head), rather than packing the work together.
4. Prefill `chunk_rows` only groups a scalar loop. It provides no shared tile
   reuse across queries/heads, no asynchronous pipeline, and no MMA tiling.
5. Grid sizing uses `min(total_threads,4096)` **blocks** of 256 threads, rather
   than a rounded-up thread count. Small cells launch many idle threads.

These hypotheses now have a measured baseline below. The agent sandbox has no
GPU visibility; the builder owns all GPU cells. No kernel dispatch has yet been
changed. Physical DRAM traffic, spills, and occupancy still need profiling;
effective useful KV GB/s must not be mislabeled a DRAM-counter measurement.

## First hardware baseline — C1, all 8 cells MISS

Builder window **2026-09-23 17:28:52–17:32:31 AEST**, clean source **`5a2b62f`**,
RTX 5090 sm_120/170 SMs, driver 610.43.02. Command: builder-only
`HOST=coordinator MIMO26_ATTN_BUILDER_WINDOW=1 scripts/dev.sh test attn-bench all`,
using its cross-built static executable. Unit E4M3 KV, f32 Q, f64 accumulators,
64 Q heads, GA 4/SWA 8 KV heads, QK192/V128, 2 warmups +5 timed samples.
Receipts committed by builder in **`c527ba1`**:
`runs/20260923-i4/c1/coordinator-receipts/c1.YHxHeF.log` and `c1.Bu0bFM.log`.
ATTN-LEAD independently ran `summarize` at **17:42 AEST**: VALID, exit 1 = MISS.

| Cell | Median ms, one layer | Useful KV GB/s | BF16-equivalent TFLOPS | Verdict |
|---|---:|---:|---:|---|
| Decode 4K | 4.408448 | 1.189 | 0.038057 | MISS (diagnostic) |
| Decode 32K | 11.704192 | 3.584 | 0.114675 | MISS (diagnostic) |
| Decode 128K | 32.932255 | **5.094** | 0.163023 | **MISS**, target 1253 GB/s |
| Decode 1M | 248.199463 | **5.408** | 0.173045 | **MISS**, target 1253 GB/s |
| Prefill T2048/S2048 | 515.335022 | — | **0.166768** | **MISS**, target 100 TFLOPS |
| Prefill T2048/S4096 | 1604.493286 | — | **0.160636** | **MISS**, target 100 TFLOPS |
| Prefill T2048/S32768 | 17045.064453 | — | **0.156228** | **MISS**, target 100 TFLOPS |
| SWA T2048/S32768 | 74.538780 | — | **0.144051** | **MISS**, target 100 TFLOPS |

Positive architecture/SM-count/baked-kernel AOT gate PASS. All 8 finite scans
and 24/24 sampled scalar-oracle coordinates passed (largest difference
1.96e-9). Minimum observed free memory **31,474,647,040 B**, above 4 GiB.
The 1M nine-GA-layer extrapolation is **2233.795 ms**, not ≤10 ms. This is an
extrapolation, not an engine-step timing. PROXY baseline still awaits the builder.

The near-flat ~0.16 TFLOPS across increasingly deep GA prefill is consistent
with the serial scalar arithmetic/conversion/softmax design, not with reaching
the useful-KV bandwidth roof. Required speedups are about **246× /232×** for
128K/1M bandwidth and **600–640×** for GA prefill. Grid-size tuning alone cannot
credibly close those gaps: remove FP64/scalar per-thread arrays and cooperate
over dimensions first, then tensor-core tile prefill. Hardware counters have
not yet isolated the relative arithmetic/spill/cache contributions.

## Measurement contract

Entry: `scripts/dev.sh test attn-bench [cell]`. CUDA events bracket the complete
split+reduce pair for decode, or the prefill kernel. Allocation, initialization,
H2D/D2H, and CPU validation are excluded. Default 2 warmups, 5 samples; individual
samples, median, min/max printed. Events are on the same default stream as the
launches; each sample synchronizes its end event. Repeated resident inputs are
cache-warm; small decode cells can be L2-resident and are not DRAM roofline gates.

- Decode: S = 4096 / 32768 / 131072 / 1048576, T=1, Hq=64, Hkv=4.
- Prefill: T=2048, **total KV length including this chunk** S=2048/4096/32768;
  query positions S−T through S−1, causal masking. A separate SWA cell at
  S=32768 uses Hkv=8, window=128, nonconstant per-Q-head sink.
- Q is FP32 (exact BF16-representable generated values). KV is synthetic finite
  unit-scale E4M3, V treated as already scaled before caching. No scale planes.
  Physical 256-token pages are reverse-permuted. GA sink is absent.
- Full finite-output scan plus 3 sampled independent FP64 scalar-oracle
  coordinates per cell. This is a benchmark sanity gate, **not a substitute**
  for the external numpy oracle/two-run parity ladder.
- Decode useful bytes = S × 4 × (192+128). At 1M: **1,342,177,280 B per layer**.
  Never multiply by the GQA repetition count to inflate bandwidth.
  GB/s = bytes/(median_ms × 1e6); target = **1253 GB/s** (70% ×1790).
- The A5 ~12 GB and ≤10 ms step refer to **nine GA layers**:
  12,079,595,520 B at 1M. Print `GA9_extrapolated_ms = 9 × one_layer_ms`,
  explicitly an extrapolation, not an end-to-end step measurement. Passing
  1M requires bandwidth and the extrapolated ≤10 ms criterion.
- Useful BF16-equivalent FLOPs = 2 ×64 ×(192+128) × visible query/key pairs.
  GA pairs = T×(S−T) + T×(T+1)/2. SWA pairs sum min(128,S−T+i+1).
  TFLOPS = FLOPs/(median_ms ×1e9). Excludes masked pairs, softmax, conversion,
  and sink work; target ≥100. “BF16-equivalent” is a work accounting unit,
  **not a claim that the baseline executes BF16 MMA**.
- `effective_KV_GBs` counts useful unique KV bytes only, not hardware counters.
  The prefill bandwidth field is informational; its gate uses useful FLOPs.
- sm_89/128-SM RTX 4090 results are **PROXY**, never a 5090 promotion. AOT
  sm_120/170-SM readback is required for the 5090, not the misleading “sm_170”.
- Memory guard: reserve ≥4 GiB, conservative additional 2 GiB allowance for
  context/local-memory overhead before allocating; recheck after warmups and
  each sample. Other processes can allocate concurrently: this is not a
  global VRAM reservation. Refusal/timeout is INCOMPLETE, not a zero-speed MISS.
- Each cell has a 120-second checked budget; the whole executable has a
  540-second wall timeout. A single slow kernel cannot be interrupted by the
  in-process budget. No median is claimed from an incomplete sample set.

## Receipt validation (CPU-only, dev host)

`MIMO26_ATTN_BENCH_LOG=receipt.log scripts/dev.sh test attn-bench summarize`
validates one returned log and emits JSON. Default required coverage is all
8 cells; set `MIMO26_ATTN_BENCH_REQUIRED=decode`, `prefill`, or a comma-separated
cell list for a deliberately narrower receipt. Separate invocations stay
separate receipts; concatenating a failed attempt and a passing retry is rejected.

The validator independently recomputes median/min/max, unique KV bytes, causal
QK+PV FLOPs, GB/s, TFLOPS, percent of 5090 peak, nine-layer extrapolation, and
PASS/MISS. It verifies source/arch/SM-count/PROXY identity, Sydney timestamps,
the hardware launch gate, shape/dtype/sink semantics, memory reserve evidence,
three-coordinate numerical sanity, and complete ordered event samples. Missing,
nonfinite, contradictory, repeated or failed evidence is never promoted.

Exit **0** = valid requested rows with no applicable MISS; **1** = valid rows
with an applicable MISS; **2** = invalid/incomplete evidence. Small decode
cells are diagnostic and alone produce `target_verdict=NOT_APPLICABLE`. PROXY
reports always say no 5090 promotion. This is still microbenchmark evidence,
not a serving-engine gate or external-oracle correctness qualification.

Long-decode effective bandwidth above nominal memory peak (4090: 1008 GB/s;
5090: 1790 GB/s) is refused pending a work/byte/cache review. This is deliberately
conservative: cache effects can make effective bytes/time exceed physical DRAM
throughput. Keep the raw row and explain that before accepting it. Values close
enough to a target that six-decimal millisecond rounding could change PASS/MISS
also require higher-precision evidence, not a rounded promotion.

2026-09-23 17:24 AEST: C++ accounting selftests **20/20**, Python receipt tests
**44/44**, including CLI exits 0/1/2, all 8 shapes, missing-cell and inflation
negatives. All Python test fixtures are synthetic, **not hardware results**.
The real retained local driver-failure log is rejected as INVALID_OR_INCOMPLETE.
No GPU implementation or tolerance changed.

## Correctness-harness hardening before optimized kernels

A source audit found three fail-open risks in the older GPU parity executable:
NaNs can disappear from its max-difference comparison; RoPE/FP8 round-trip
outputs were initialized from expected answers; and CUDA return codes were
mostly ignored. Its shell wrapper accepted any nonzero naive exit, including
CUDA failures. These are harness defects, **not evidence that the retained
finite C1 outputs or historical I3 results were numerically wrong**.

The hardening increment rejects nonfinite actual/expected values and invalid
tolerances; fills output buffers with NaNs (including before the paged rerun);
checks every launch/copy/allocation/free; frees per-case input allocations;
and reserves 4 GiB plus a conservative 2 GiB allocation margin. Naive success
now requires exit 1 with explicit numerical failures and the same case count
as the positive run; CUDA errors/refusals cannot count as trap detections.
`scripts/dev.sh test attn selftest` pins the comparator (20 checks, including
an explicit reproduction of the old NaN bug); `build-sm89`/`build-sm120` qualify
compilation only. Builder must rerun tiny and all-shape two-run parity before
any optimized kernel is promoted. These changes do not alter the kernels.

## Design revision R1 — REVIEW REQUIRED before kernel implementation

**Builder instruction received 23 September 2026 AEST:** redesign around
GQA-packed tensor-core decode and FA2-style prefill; MiMo Pro reviews this
revision **before any new kernel is written**. This supersedes the prior
SIMT-first proposal. No optimized kernel is implemented or approved yet.

### Bottleneck by cell

| Cell | Dominant suspected mechanism (counters still needed) | Required structural remedy |
|---|---|---|
| Decode 4K | 4.41 ms for 5.24 MB: scalar arrays, FP64/SFU work and excessive grid dominate; cache-resident diagnostic | Head-packed MMA; short split ranges; parallel merge, not a bandwidth promotion |
| Decode 32K | 11.70 ms; more work per thread, still only 3.58 useful GB/s | Cooperatively load/convert KV once per KV-head CTA |
| Decode 128K | 32.93 ms, 5.09 GB/s; ~246× below bandwidth target | Enough split CTAs for 170 SMs; remove scalar FP64 and repeated GQA KV work |
| Decode 1M | 248.20 ms, 5.41 GB/s; scaling approaches the scalar work floor | Same tiled algorithm, longer split ranges; include partial write/read and merge costs |
| Prefill 2K | 515.34 ms; no tensor work or cross-query reuse | Pack query positions and GQA heads into an M tile; skip wholly future key tiles |
| Prefill 4K | 1604.49 ms, 0.161 TFLOPS; prefix work repeats independently per query | Resident Q/output tiles, shared converted K/V and online softmax |
| Prefill 32K | 17045.06 ms, 0.156 TFLOPS; same scalar throughput at greater depth | Pipeline global loads with MMA, never materialize global score/probability matrices |
| SWA prefill | 74.54 ms; scalar attention plus scanning 32K positions for a 128-key window | Position-bounded key tiles, compact tail MMA and no prefix-wide scan |

### Hardware envelope — use the correct accumulation mode

The [NVIDIA RTX Blackwell whitepaper, Appendix A](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
lists **209.5 dense TFLOPS for BF16 with FP32 accumulation** on the 5090;
419 is its sparse number (or a different FP16-accumulate mode), not our dense
BF16 ceiling. Its memory figure is 1792 GB/s; retain the task's conservative
**1790 GB/s** for the 1253 GB/s gate. The corresponding 4090 dense BF16/FP32
ceiling is 165.2 TFLOPS and bandwidth 1008 GB/s: PROXY, not a target substitute.

Use `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, `ldmatrix`, and
`cp.async` with CUDA 12.8; do not assume Hopper warp-group MMA or datacenter
Blackwell tensor-memory primitives. Per the [CUDA programming guide, CC 12.0](https://docs.nvidia.com/cuda/archive/12.9.1/pdf/CUDA_C_Programming_Guide.pdf),
budget conservatively for 64K registers/SM, 100 KiB shared/SM and 99 KiB/block;
query actual properties and occupancy in the builder's gate. Above 48 KiB
requires dynamic shared-memory opt-in. Publisher-indexed tables were available
via search; direct page fetches failed DNS in this session. Runtime readback
and compile/resource receipts, not a secondary GPU chart, decide launch legality.

### Numerical contract — ADVISOR-I4 §9 R7 governs

The builder's **20:58 AEST ruling** explicitly sets the attention output bound:

**`|output − reference| ≤ 2e-5 · max(1, max|v|)`**.

Here `v` means **decoded cached V, already prescaled**, in attention-output
units. The audit conservatively takes the maximum across the entire case cache,
including masked entries and all KV heads/components. This is a scale-aware
absolute bound, **not** relative tolerance against the output. FP32 accumulation
and cancellation still require empirical checks; no universal domain proof is
claimed. The scalar FP64 baseline and external oracle remain unchanged.

- **A-f32q:** three BF16 Q components and **two P components**; actual unpadded
  GA MMA factor **2.6**. Keep FP32 m/l/O and separate FP32 partial scratch.
  Three-term Q fixes the demonstrated tail defect; two-term Q is not adequate.
- **A-bf16q:** opt-in, Q rounded **once RNE after RoPE**, P high/residual,
  FP32 m/l/O. Name K, V and KV-cache representation in every receipt. Its
  factor is **1.4** before padding, and it does not establish f32q-oracle parity
  or X1c eligibility. A-f32q remains preferred until X1c and the performance gates.
- One template uses a compile-time Q residual switch. Scaled-FP16 P is
  **rejected**, not an alternative to implement. No silent BF16-Q default.
- F32q performance gate: **125.7 executed TFLOPS**, about **48.3 useful** at
  factor 2.6; no additional 62.9 useful gate. The ideal useful ceiling is about
  **80.6 TFLOPS**. BF16q retains its separate **100 useful TFLOPS** gate.
- R7 supersedes the former fixed 2e-5 full-value-range requirement. At |V| = 448,
  the new bound is **0.00896**; the previous ideal P2 stress error around
  **6.84e-5** is below it. This changes the approved contract, not historical
  measurements. High-value GPU regression evidence is still to be added;
  **P stays at two terms**, not a speculative P3 correction.

The eight original TC cases were audited on CPU at **21:37 AEST**:

| Cases | max decoded V magnitude | R7 bound | Retained fixture bound |
|---|---:|---:|---:|
| `tc_ga_pages`, `tc_swa_sink`, `tc_empty`, `tc_short` | 0.5 | 2e-5 | 1e-5 |
| `tc_q_low`, `tc_p_low`, `tc_rescale`, `tc_q_tail` | 1 | 2e-5 | 1e-5 |

All relative tolerances remain zero. Thus the completed source `77bfe02` GPU
requalification already used a **stricter** bound on all eight cases. This audit
is an addendum, **not a new GPU run**; inputs, RNG ordering and tolerances are
unchanged. `scripts/dev.sh test attn audit-tc` reproduces it, and every subsequent
`tc-decode` golden generation now prints these eight magnitude/bound rows with
the requalification receipt. The existing benchmark checks also remain stricter
than R7; old PASS/MISS results are not reclassified.

R7's co-bound watch remains live: ideal 1M decode compute **0.889 ms** versus
bandwidth **0.895 ms**. The measured D0.1 1M point is **14.484512 ms**, executed
**7.709555 TFLOPS**, eta **0.03680** of 209.5: **MISS**, not an ideal prediction.
The context addendum `runs/20260923-i4/attn/n5-context/SUMMARY.md` labels
128K prefill as a model derived from the measured interior rate, not a GPU
measurement. Both T = 2048 and full T = 131072 interpretations are explicit.

### D1: GQA-packed split-KV tensor-core decode

- CTA = `(query row, KV head, split)`, **4 warps /128 threads**. M=16 is the
  16 GA Q heads sharing that KV head; N=32 keys; QK depth192 =12 k16 steps.
  SWA has 8 Q heads and masks unused M rows if this path is used there.
- Four warps divide QK's N columns, then divide PV's 128 output columns. Each
  K/V tile is loaded once for the group, not once per Q head. Scores/output
  stay in FP32 registers; no per-thread arrays of width192/128.
- Page size256, tile32: page-aligned ranges for long cells; short splits may
  start within a page. Every tail/position is predicated. Stage contiguous
  16-byte FP8 vectors into a raw shared buffer, convert to exact BF16 in a
  bank-swizzled layout, and prefetch the next raw tile while MMA consumes the
  current converted tile. One raw buffer suffices: it is reusable after the
  cooperative conversion barrier. No overwrite of the current BF16 tile
  until both QK and PV consumers finish.
- **Corrected D1 budget:** Q3 planes **18 KiB**, converted K/V **20 KiB**,
  two raw stages **20 KiB**, P2 planes **2 KiB**, scores **2 KiB**, statistics
  and metadata: exactly **64,192 B**. Dynamic opt-in; measured occupancy is
  **one CTA per SM**, not the obsolete R1 two-CTA target. Registers **61/48**
  for 4/8 warps, no spills. The older Q2/single-stage 43.5 KiB proposal is
  superseded; its occupancy assumptions must not enter D1's model.
- **22:39 AEST F1/F5 correction:** T = 1 launches `4P` CTAs. Use full-wave gate
  points; retain the old schedule only as diagnostics. The cap below is the
  equal-duration CTA-wave occupancy model, not a measured DRAM counter.

| Context | P | CTAs | Waves | Wave fill cap | Role |
|---|---:|---:|---:|---:|---|
| 128K | **85** | 340 | 2 | **100%** | Gate/default |
| 1M | **255** | 1020 | 6 | **100%** | Gate |
| 1M | **510** | 2040 | 12 | **100%** | Gate/default |
| 128K | 64 | 256 | 2 | 75.29% | Diagnostic |
| 128K | 128 | 512 | 4 | 75.29% | Diagnostic |
| 128K / 1M | 256 | 1024 | 7 | 86.05% | Diagnostic |
| 1M | 512 | 2048 | 13 | 92.67% | Diagnostic |

Short contexts retain P = 64 as diagnostics. Full waves remove this grid loss,
not the measured in-CTA efficiency deficit. The P = 85 measurement below still
misses the bandwidth target substantially.

- New FP32 partial ABI: separate m and l planes `[query][head][split]`, plus
  an O plane `[query][head][split][128]`, each 128-byte aligned. It still costs
  130 floats/partial, but does **not** interleave an 8-byte m/l header before
  every O row and misalign the 128-byte column-tile loads. This is not the old
  FP64 ABI. Merge uses **256 CTAs for T=1** (`64 heads ×4 value-column tiles`), avoiding
  a 64-CTA merge bottleneck. Each CTA reduces all splits for 32 output columns;
  max/sum reductions are block-cooperative. Denominator is identical across
  column tiles; a SWA sink enters each output's denominator **once**, GA never.
  Empty splits write m=−infinity, l=0, O=0 and are explicitly neutral at merge.

### P1: FA2-style packed prefill, with a separate SWA tile

**GA:** M=64 packs 4 consecutive query positions ×16 GQA heads; N=64 keys,
8 warps /256 threads. Four warp groups along M and two along N handle QK;
PV repartitions output columns without materializing scores globally. Total
CTAs for T2048/Hq64: **2048**, about12 CTAs/SM. A query/head row always retains
its own absolute position: masks are not shared merely because heads are packed.

Shared memory for native BF16 Q: Q 24 KiB + converted KV 40 KiB + raw prefetch
20 KiB + reused P 8 KiB + reductions about 1 KiB = **about 93 KiB**. One CTA/SM,
8 warps; ≤128 registers/thread is the initial register budget. Dynamic shared
opt-in is mandatory. Only the raw *next* tile overlaps current MMA: there is
not room for two expanded BF16 KV tiles. This occupancy/pipeline constraint is
an explicit risk to the utilization assumptions below.

Per tile: QK → scale by 1/sqrt192 → absolute-position/causal mask → row max →
rescale old FP32 O/l → FP32 exp and full-P row sum → PV with P high → reuse P
shared buffer for residual → second PV. Retain full-P registers until residual
formation finishes. Never cast m/l/O to BF16. Final division/output are FP32.

**SWA:** M=32 packs 4 query positions ×8 heads, N=32, 4 warps, roughly44.5 KiB
shared with BF16 Q. Two CTAs/SM and **4096 CTAs** total. For contiguous positions
start at `q_min−127` (clipped to available history), not at context zero. The
union of four 128-key windows is **131 keys**: four N32 tiles plus a 3-key tail.
Tail QK uses N8, tail PV uses a k16 step, with invalid entries zero/masked. This
avoids a whole extra N32 or N64 PV tile. Each row still applies its own mask.
The executed-work factor is `(192×136/128 + 2×128×144/128)/320 = 1.5375`.
Sink is per Q head, incorporated once in the final normalization with zero V.

The range shortcut requires a proved contiguous logical-position contract.
Noncontiguous positions use a correct general path, not guessed range bounds.
SWA prefill must see the retained 127-token prefix plus the **whole current
chunk's KV**; writing only the final ring before early queries consume their
keys is invalid. Preserve T8/min(batch_pos) eviction. GA never inherits SWA bounds.

### Conditional performance model — MODEL, not measurements or promotion

Numbers below are an **engineering budget**, not calibrated forecasts. Useful
FLOPs are the existing causal QK+PV count, not inflated by correction MMAs.
Planning must allow **3–5× model latency** until builder measurements establish
otherwise. An expected target hit is conditional on the stated efficiencies.

Decode, unit FP8, FP32 Q, **three QK / two PV products** (F2 correction):

```
B = 1280*S + 2*(64*P*130*4) + 32*S + 16*ceil(S/256) + 49152 + 32768
F_useful = 40960*S
K_padded = sum_p 32*ceil((floor(S*(p+1)/P)-floor(S*p/P))/32)
F_executed = 2*64*(3*192 + 2*128)*K_padded  # 2.6*F_useful only without padding
wave_fill = 4*P / (170*ceil(4*P/170))
ms = max(B/(1790*beta_eff*1e6), F_executed/(209.5*eta_eff*1e9)) + 0.008
```

`beta_eff` and `eta_eff` must include scheduling and implementation losses;
never silently carry peak-band assumptions through partial waves. The old 0.88
bandwidth / 0.60 compute efficiencies were hypotheses, not measurements. Full
waves remove the modeled grid loss but do not establish those efficiencies.
For an explicit BF16-Q row, replace `3*192` by `192` (factor 1.4 unpadded), and
identify its distinct precision/reference scope.

B includes useful KV, FP32 partial write+read, conservatively repeated position
loads, page entries, Q and final output. Q is tiny/cache-reused; the extra merge
reads of scalar m/l are assumed cache hits. 0.008 ms covers non-bandwidth launch/
merge latency; the bandwidth term includes the data movement of both kernels.
These assumptions are to be challenged by the reviewer and measured, not hidden.

The obsolete R1 table (0.128046 / 0.903098 ms and nine-layer 8.128 ms) is
withdrawn as a current forecast: it used `2*F` and ignored grid quantization.
R7's **0.889 ms compute / 0.895 ms bandwidth** remains a paper co-bound watch,
not an achieved runtime. The first actual D1 pass gives 128K P = 85 / 8 warps
**0.427296 ms**, and 1M P = 512 / 8 warps **3.587392 ms**, both **MISS**.
The latter is now a partial-wave diagnostic; corrected P = 255/510 timings and
BF16-Q decode are still to be collected. Planning allowances remain explicit,
and no model-only number promotes a config.
BF16-KV mode doubles the KV term; report its own byte count, never claim an FP8
KV-read target using BF16 traffic. Short cells are cache/launch diagnostics.

Prefill native BF16 Q, FP32 output, FP8 KV, P correction:
`time = max(modeled_DRAM_bytes/(1790*0.88), executed_MMA_FLOPs/(209.5*eta)) +8 µs`,
using consistent decimal GB/TF units. GA eta=0.75, SWA eta=0.85. These eta values
include the loss from softmax, conversion, barriers and available warps.

| Prefill cell | Modeled bytes GB | Executed MMA GFLOPs | Model ms | Model useful TFLOPS | Modeled traffic GB/s |
|---|---:|---:|---:|---:|---:|
| GA 2K | 0.809501 | 124.017 | 0.797290 | **107.79** | 1015 |
| GA 4K | 2.151678 | 364.535 | 2.328034 | **110.71** | 924 |
| GA 32K | 20.942160 | 3731.790 | 23.758452 | **112.08** | 881 |
| SWA 32K | 0.134611 | 16.509 | 0.100707 | **106.62** | 1337 |

GA conservatively charges every KV tile reload to DRAM: for each 4-position
query group, round its causal key bound up to64; multiply summed key rows by
4 KV heads ×320 bytes. Add BF16 Q reads and FP32 output writes. Executed work
is179.2 FLOPs per loaded KV byte, incorporating P correction and causal padding.
L2 hits could lower actual DRAM bytes; they do not increase credited useful FLOPs.

SWA charges 4096 CTAs ×131 keys ×320 B with **90% KV-reload L2 hits**, plus
50.33 MB of Q and67.11 MB of output. Its used KV union is only2175 positions
(5.568 MB), not all32768 stored positions. The old benchmark's SWA full-plane
GB/s field is informational allocation-normalized throughput, **not read traffic**.
The SWA forecast requires both high tensor utilization and local KV cache reuse;
at70% math efficiency,80% bandwidth and75% KV hits it is about84 TFLOPS (**MISS**).
GA at60% math efficiency is about86–90 TFLOPS (**MISS**).

**SWA dtype warning:** retaining f32 Q adds50.33 MB of input traffic. Even with
perfect KV reuse, Q+FP32-output alone is167.77 MB, nearly the entire time budget
for100 TFLOPS at this small amount of work. Required Q packing cannot be hidden
outside timing. This strengthens the need for an explicit, reviewed native-Q
contract rather than a benchmark-only cast. No cache-priming trick may be used
to claim end-to-end serving throughput.

### MiMo Pro review checklist / implementation gate

1. Approve or reject native BF16-Q prefill semantics against the current FP32
   spike and the external model anchor. If rejected, pick a compatible precision
   scheme and revise the model; don't proceed under a false 100-TFLOPS promise.
2. Review P high/residual error, FP32-Q decomposition for decode, and exact
   zero/empty-row/sink handling; existing tolerances stay unchanged.
3. Check shared-memory accounting/swizzle padding, lifetime barriers, register
   budgets and feasibility of the assumed utilization on170 SMs.
4. Check FP32 partial/merge ABI, traffic model and128K merge sensitivity.
5. Check SWA131-key union/tail execution, absolute-position contract and ring
   lifetime. Challenge the90% KV-hit/85% math-efficiency assumptions.
6. After review only: implement D1 then P1. Add independent oracle fixtures
   with BF16-exact Q that **force** tensor-path execution, and preserve general
   f32/separate-scale coverage. Log selected path/counters so fallback cannot
   fake optimized correctness. Two-run naive switches must operate inside the
   tensor path (sink, scale, mask, rescale, page layout), not route to a scalar
   implementation merely to obtain an expected failure.
7. Builder gates: hardened scalar parity → candidate tiny/scale-up two-run →
   4090 PROXY timing → reviewed5090 C1 window. Retain every failed/MISS row.
   Gather ptxas resource reports and then counters for actual DRAM/L2 traffic,
   tensor utilization and barrier stalls if the modeled budget is missed.

## R1 precision addendum — closed-form CPU counterexamples

**2026-09-23 18:29 AEST.** `scripts/dev.sh test attn-bench precision-study`
ran in `target/mimo26f-builds/attn-bench.3Kn6ny` (a99c534 + probe WIP).
This is a **CPU quantization study**, not a GPU correctness or performance
receipt. It uses FP64 exp/sums to isolate representation loss, not to emulate
FP32 MMA reduction order. Neither oracle goldens nor tolerances changed.
Host checks: metrics 20/20, parser 44/44, new precision checks 20/20.

Both constructions have S=2, QK192, V128, one representative head. Unlisted
Q/K components are zero; V is +1 for key 0 and −1 for key 1 in all 128 columns.
All K/V values are exactly finite E4M3-unit values. They can be embedded into
GQA or early causal rows by repeating the construction across heads.

1. **FP32-Q cancellation:** Q starts `[1.00390625, 1]`, K0 `[448, −448]`,
   K1 `[−448, 448]`. Independent reference is
   `tanh(448 × 2^-8 / sqrt(192)) = 0.125628135846`.
   BF16-only Q rounding makes both scores zero and returns **0**.
2. **BF16-exact Q, probability loss:** Q starts `[1, 0]`, K0 `[1, 0]`,
   K1 `[0, 0]`. Independent reference is
   `tanh(1 / (2 sqrt(192))) = 0.036068738349`.

| Representation change (ideal accumulation) | Cancellation max abs | BF16-exact Q max abs |
|---|---:|---:|
| Round Q only to BF16 | **0.125628** | 1.39e-17 |
| Round P only to BF16; full-P denominator | **3.14e-4** | **3.56e-4** |
| Round P only to scaled FP16; full-P denominator | **3.93e-5** | **1.03e-4** |
| BF16 high/residual Q and P | 6.09e-7 | 1.53e-7 |

Bold errors exceed the unchanged benchmark absolute bound **2e-5**. The
scaled-FP16 alternative in R1 therefore cannot be accepted unconditionally:
avoiding underflow does not remove mantissa error. The high/residual route
passes **these two constructions only**; it still needs the full external-oracle
suite and actual GPU reduction-order tests. A native BF16-Q interface can only
be approved as an explicit input contract, not as a semantics-preserving silent
cast of existing FP32 Q. MiMo Pro review is still required before kernel code.

## Decode D0 implementation increment — 23 September 2026, 19:02 AEST

**Authorization update:** the builder's 18:35 instruction allows decode to
proceed while MiMo reviews R1; it supersedes the decode review hold above.
**Prefill remains held.** The review may still require decode changes.

`attn_decode_tc.cu` adds opt-in M16×N32, four-warp GQA-packed BF16 MMA decode.
Q remains FP32, decomposed into BF16 high/residual; P uses the same two-product
scheme. Unit FP8 K/V expand exactly. GQA8 pads inactive rows; GQA16 fills the
tile. Absolute positions, SWA masks, page indirection, running-max rescaling,
empty splits and per-Q-head SWA-only sink are retained. The new FP32 SoA scratch
and four-output-tile merge are separate ABIs; baseline entrypoints are unchanged.

**Deliberate staging difference from the full D1 proposal:** synchronous
cooperative KV expansion, not cp.async pipelining yet. Packed fragment loads
use original shared-memory swizzles and explicit lane maps. Host tests check
bijections, full fragment ownership, bank addresses, all 256 E4M3 codes and
split/head/query indexing; these do **not** validate actual GPU MMA execution.

Cross-builds at **18:59 AEST**, CUDA12.8, static cudart:

| Kernel | sm_89 registers | sm_120 registers | Shared bytes | Stack/spills |
|---|---:|---:|---:|---:|
| D0 decode | 74 | 70 | 37,312 | 0 /0 |
| D0 merge | 39 | 40 | 544 | 0 /0 |

Build slots: `attn-parity.yBo9a1` / `attn-parity.pNYjEo` under
`target/mimo26f-builds`. These are compile/resource observations, not occupancy,
GPU correctness, or speed receipts. Two resident decode CTAs fit the nominal
100-KiB shared budget; actual occupancy and bandwidth still require hardware.

New `tc-decode` corpus: seven external-oracle cases, 14 flat/paged comparisons,
absolute **1e-5** tolerance. Covers multi-query GA pages with ignored sink,
SWA128/per-head sink, all-masked rows, S=3 with empty splits, Q cancellation,
P residual, and two-tile running-max rescale. The two residual fixtures also
match independent tanh identities during oracle generation. Existing golden
inputs/tolerances and `oracle/mimo26` are unchanged; the FP8 harness explicitly
omits a supplied GA sink when invoking the oracle, matching the layer contract.

Host comparator20 + layout/codec24 checks passed; CPU oracle generation7/7
passed in `attn-parity.HzZ37h`. Builder commands are in the committed crate
README and packet. The candidate logs its path and requires positive TC
coverage; its two-run suite additionally isolates 11 in-kernel negative bits.
No candidate GPU gate has run, no candidate timing is claimed, and C1 remains
8/8 baseline performance MISS. Next: builder layout/correctness smoke, then
precision-edge, sanitizer and scale-up qualification before candidate timing.

**Known D0 precision follow-up (CPU-reproduced, not a GPU receipt):** two BF16 Q
components are not enough to preserve every FP32 query at the existing bound.
Take Q prefix `[1 + 2^-9 + 2^-17, 1 + 2^-9]` and K0 `[448, -448]`, K1 the
negative, with V0=+1/V1=-1. High/residual BF16 rounds both Q components to the
same represented value, so D0 gets zero scores; the exact output is
`tanh(448 * 2^-17 / sqrt(192))`, approximately **2.47e-4**. This case is not
in the seven-case smoke yet. Add it and a corrective Q-tail component before
claiming general-FP32 parity or requesting candidate timing. Passing the
initial smoke must not hide this known limitation; revise the original 2×
MMA budget when the precision fix changes actual work. At **19:15 AEST**, the
extended `precision-study` reproduced error **0.0002466706422337711**, with
21/21 host assertions and 3/3 independent identities. Receipt:
`target/mimo26f-builds/attn-bench.WULMJV/precision-study.log`.

### MiMo Pro review response — received 19:15 AEST

Review `runs/20260923-i4/reviews/a99c534-attn-design-mimo.md` says **CHANGES**.
Accept the structural recommendations: test P=64/128/256 at 128K and budget a
deeper raw-load pipeline or wider CTA, then measure rather than assume bandwidth.
Correction: those split counts launch 256/512/1024 GA CTAs, averaging
**1.51/3.01/6.02** CTAs per 170 SMs (not 3–6 for P=64–128). Residency is distinct
from grid size. Additional raw buffers must fit the actual shared budget;
one cannot assume two stages preserve two resident CTAs.

The new third probe invalidates the review's unconditional approval of the
2-term-Q scheme. Original two-probe margins did not prove general-FP32 accuracy.
Likewise, the review's prefill impossibility argument is conditional on its
chosen work factor/efficiency assumptions, not a universal hardware theorem.
**No target or tolerance is being lowered:** ≥100 useful TFLOPS remains the
requested prefill bar. The proposed executed-throughput gate is a recommendation
for the user/advisor to adjudicate, not an accepted substitute. Prefill remains
held while these contract/precision changes are resolved; decode work continues.

### Returned dev-host baseline evidence — builder batch 1, source b3e4e63

Commit `569f539`, raw `runs/20260923-i4/attn/dev-host-batch1-b3e4e63.log`:
**RTX4090/sm89 PROXY**, baked architecture/SM-count/launch PASS. Hardened tiny
parity passed **14/14**, with a complete **14/14 numerical negative** (exit1)
and zero TC launches. This is baseline qualification, not D0 qualification.

| Completed baseline cell | Median ms | Relevant PROXY rate | Verdict |
|---|---:|---:|---|
| decode-4k | 5.415936 | 0.968 GB/s | MISS |
| decode-32k | 23.348225 | 1.796 GB/s | MISS |
| decode-128k | 83.872162 | 2.000 GB/s | MISS |
| decode-1m | 679.339905 | 1.976 GB/s | MISS |
| prefill-2k | 1063.404663 | 0.080817 TFLOPS | MISS |
| prefill-4k | 3117.454590 | 0.082676 TFLOPS | MISS |

Prefill32K exceeded the 120-second cell budget: **INCOMPLETE, no median**;
SWA was not reached. At 19:20 AEST, the receipt validator correctly returned
exit2/`INVALID_OR_INCOMPLETE`; the group must not be presented as a completed
six-cell qualification. Its six measured rows remain recorded above.

The separate parity `--all` invocation failed argparse before any cases ran.
Fixed both the leading-dash argument transfer (`--shapes=...`) and the missing
alias expansion. `test attn shape-list` verified all10 names on CPU without
allocating tensors at 19:20 AEST. No full-scale parity result is inferred.
Use explicit small cells next; `--all` includes expensive 128K/1M oracle work.

### Direct dev-host D0 smoke — 23 September 2026, 19:30 AEST

The preset change restored visibility at 19:27. Preflight showed RTX4090,
24,564 MiB total/6,281 MiB used, only the resident eye/ear services. No competing
attention/compiler job was found. Under the existing dev-host delegation, the agent
ran `tc-decode` baseline then TC, each with `MIMO26_ATTN_TWO_RUN=1`, through
`scripts/dev.sh`. Every pass repeated `nvidia-smi`; allocation/post-case guards
retained ≥4 GiB. No remote/container operation or candidate timing occurred.

Both receipts report clean attention source **0701318fbb69** (contains D0
b3e4e63 and the follow-up). Baseline **14/14** plus complete numerical negative;
TC **14/14**, max absolute **5.811e-7**, with **14 TC launches**. Original negative
and **11/11 isolated negative runs** each completed 14 comparisons and failed
numerically inside the TC path. No fallback/CUDA-error negative was accepted.

Raw receipts under `target/mimo26f-builds`:
- `attn-parity.iq5LeJ/receipt.log` — baseline, 19:30:02 AEST.
- `attn-parity.VwZJIm/receipt.log` — TC, 19:30:09 AEST; per-bit logs beside it.

**This is the seven-case smoke only.** The independent Q-tail counterexample
still fails the numerical model; no general-FP32 or serving/performance
qualification is inferred. D0-B/D0-T are completed, not rerun requests. Next
implementation work is the precision correction plus its GPU regression.

## Verification at baseline increment

2026-09-23 17:04 AEST: host metric/gate selftests **20/20**; nvcc **12.8.93**
sm_120 static-runtime build PASS. `ldd` shows no libcudart dependency.
At that point no GPU result had landed: local `nvidia-smi` exited 9 and
`/dev/nvidia*` was absent in the sandbox. The then-UNMEASURED status is superseded
by the measured C1 MISS table above. The builder confirmed this is sandbox GPU
visibility, not a host driver failure. Exact commands: the ATTN-LEAD packet.
