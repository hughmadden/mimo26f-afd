# P-305 — Memory & admission policy for full context (A1)

**Packet:** I3-P305 (ADVISOR-I3 §3 A1; ITERATION "P-300 revision", I3 row).
**Status:** design paper for external review. Code lands at I5 (`A1 admission v1`,
ITERATION P-300 revision, I5 row). "P-305 reviewed" is an I3 exit criterion
(ADVISOR-I3 §4, I3 row).
**Date:** 23 September 2026 AEST (Sydney).
**Scope:** coordinator KV pool sizing, per-request byte budget, admission
reservation, the long-context lane, and the pressure ladder for the AFD engine
on one RTX 5090 + 4 GB10 Sparks.

## 0. Provenance policy (read this before any number)

Every number below is labelled **model** or **measured**.

- **model** — derived from checkpoint geometry or the perf model
  `bench/model/afd_vs_tp4_model.py`, or from ADVISOR-I3 §3 arithmetic. Model
  numbers never promote a configuration (R-MODEL, ADVISOR-I3 §1).
- **measured** — carries its receipt (file:line or named run).

No stub-class number is used to size anything in this paper. The one
stub-derived quantity in the neighbourhood (host-cache copy latency, "to be
measured" in `afd-hostcache-design.md:15`) is never used here for sizing; if it
were quoted it would carry the 3–5× discount note and no promotion power
(I-Hon, AGENTS.md §2; TEST-PLAN R5).

**Never-enter list binds this paper** (AGENTS.md §3): no Engram, CED, MLA/sparse
stack, mHC, dSpark body, EXL3/NVFP4 paths, DeepSeek vision, DS/GLM constant
families, or DeepSeek-shaped literals enter the design. The first attempt's
`plan/placement.py` pool math is analysed here as a **negative example only**
(T21) and must never be copied (standing constraint, ITERATION "P-300
revision").

## 1. The constraint

R-CTX: 1,048,576 tokens per request, end to end (ADVISOR-I3 §1). The GA KV of
one such request is 12.1 GB — **model** (1,048,576 × 11,520 B;
`bench/model/afd_vs_tp4_model.py:26`) — while the whole coordinator budget is
32,607 MiB — **measured** (probe 23 Sep, ADVISOR-I3 §2). Capacity is the
binding gap versus 4-Spark TP4 (which holds 186.8 GiB of KV across four ranks,
**measured**, ADVISOR-I3 §10.4.4): the tiers of P-306 close it, and this paper
is what keeps the device side within budget meanwhile.

## 2. KV byte budget per request

All **model** (geometry constants: `bench/model/afd_vs_tp4_model.py:26,170-176`;
AGENTS.md §3 "KV"; ARCHITECTURE §11.1).

| Part | Bytes | Derivation |
|---|---|---|
| GA KV, grow | **11,520 B/token** (FP8) | 9 GA layers × 4 KV heads × (K 192 + V 128) |
| — scale bytes | 0 today | unit-scale FP8 KV: cast to E4M3, upcast on read; exactly 11,520 B/token (ADVISOR-I3 §10.4.2, amending ARCH §11.7). If the amax check or G4n forces per-token scales (T20: per token × head, K/V separate), the scale bytes join the pool math then — not before. |
| SWA rings | **12,779,520 B/seq** (~12.78 MB) | 39 layers × 128 window × 8 KV heads × 320 B |
| DFlash context ring | **20,971,520 B** BF16 (10.5 MB FP8) | 1024 window × 5 layers × 8 heads × (K 128 + V 128) × 2 B |
| MTP rings | **983,040 B** (~0.98 MB) | 3 layers × 128 window × 8 heads × 320 B |
| Per-1M-request GA total | **12.1 GB** | 1,048,576 × 11,520 B |

Only one drafter is resident per deployment (ARCHITECTURE §11.6; ADVISOR-I3
§3 A7), so a request carries the DFlash ring **or** the MTP rings, never both.

## 3. Pool: measured at boot, never pinned from a planner

**Rule (ADVISOR-I3 §3 A1; ARCHITECTURE §11.1, §11.8):** the pool size is a
boot-time measurement of what is actually free after real allocations. It is
not a planner output and not a config constant. A planner number may only be a
pre-boot estimate.

Planning decomposition on the 5090 — **model** (ADVISOR-I3 §3 A1):

| Term | Size |
|---|---|
| Card | 32,607 MiB (**measured**, ADVISOR-I3 §2), ≈ 31.8 GiB usable for the budget |
| Weights, one drafter | ~11.2 GiB (9.8 GiB with FP8 DFlash) |
| CUDA context | ~0.5 GiB |
| Runtime headroom | 2 GiB (the v10 constant, `RUNTIME_HEADROOM`, memory.rs:15) |
| Workspaces + graphs at chunk 2,048 | ~2–3 GiB |
| Per-slot rings (SWA + drafter) | per §2 — **totalled (fold N2, ADVISOR-I4 §2):** (12.78 MB SWA + 20.97 MB DFlash) × 16 slots = **0.54 GB** (1.08 GB at 32 slots), **model** |
| **KV pool left** | **~14–16 GiB ≈ 1.3–1.5M tokens** |

The perf model prints a slightly lower ~1.2–1.4M at 13–15 GiB
(`bench/model/afd_vs_tp4_model.py:167`) with the same arithmetic and a tighter
subtraction; both are **model** ranges and the disagreement is immaterial
because the boot measurement decides. What the estimate must not do is pin the
pool: that is trap T21 (§6).

Precedent for the shape of the check (positive reference, ds41rt v10): pool
sizing is startup-only ("No allocation policy runs in the token loop",
memory.rs:1) and fails loud if the reservation cannot fund the plan
(memory.rs:209-212, an `ensure!` naming the runtime headroom). Keep both
properties.

**Consequence:** one 1M-token request fits (12.1 GB GA + ~34 MB tail, §P-306)
in a 14–16 GiB pool with ~2.7–4.7 GiB to spare, but not two. Hence the long-context
lane.

## 4. Long-context lane

- Requests above **512K tokens** run with **concurrency 1** for that class on a
  32 GB coordinator — **model** (ADVISOR-I3 §3 A1; ARCHITECTURE §11.1). Other
  sessions park in tier 1 and resume from their snapshots (P-306).
- The threshold is 512K, not 1M, so that a parked neighbour plus the
  long-context request can never both claim ~12 GB against a 14–16 GiB pool.
  — **model** (arithmetic from §2/§3).

## 5. Admission: reserve the round, not the lifetime

**Reservation (ADVISOR-I3 §3 A1; ARCHITECTURE §11.1, §11.11 `admit_reserve`):**

```text
reserve = prompt_tokens + min(max_tokens, Q)      with Q = 8192
```

and the reservation **grows per round** by further quanta of Q (bounded by the
request's remaining `max_tokens`) as generation consumes it. All **model** as
policy; the constants come from ARCHITECTURE §11.11 (`admit_reserve = 8192`,
`max_output_tokens = 65536`).

Why not reserve `prompt + max_tokens` (lifetime reservation):

- **T21 arithmetic — model:** an omitted `max_tokens` defaults to
  `max_output_tokens` = 65,536 (ADVISOR-I3 §10.2.1, trap T23), so a lifetime
  reservation locks **755 MB** (65,536 × 11,520 B) of GA KV per request before
  a single token is generated. Two such requests with modest prompts consume
  ~1.6 GB (≈10%) of a 14–16 GiB pool — and the 755 MB/request overhang is what
  never materialises (only ~128K-token prompts would make the pair reach a
  third of the pool).
- v10 does exactly this: every active request's budget is `tokens +
  remaining max_tokens` against committed pages
  (mimo26-flash-tj `v41_native_serve/scheduler/admission.rs:47-51`, wired at
  `scheduler.rs:245-255`), and oversize prompts are refused with "prompt plus
  max_tokens exceeds the GPU KV pool" (`scheduler.rs:269-270`). At DeepSeek's
  890 B/token that is tolerable; at MiMo's 11,520 B/token it starves concurrency
  (T21, COHERENCE-TRAPS §6).

Positive detail to keep from v10's admission accounting: capacity checks are
prefix-sharing-aware and account for partial-page COW (comment at
`scheduler.rs:246-248`). Our per-round reservation keeps that accounting; only
the reserved horizon shrinks from lifetime to prompt + Q.

Under the default output cap the per-request reservation at admission is
`prompt + 8192` tokens = **~94 MB of GA KV per 8K reserved quantum** — **model**
(8192 × 11,520 B = 94.4 MB).

### 5.1 Feasibility vs reservation (fold N1, ADVISOR-I4 §2 — 23 Sep 2026 AEST)

Reservation (§5) is necessary but not sufficient. Admission must also check that
the request **can finish alone**: `prompt + max_tokens ≤ device pool`. Every GA
page must be resident during decode (GA attends to all earlier tokens; tiers
cannot page it in mid-decode), so a request that cannot fit alone can never
finish — refuse it with **400 `context_length_exceeded`** up front. Otherwise
the ladder can preempt the only request that can never fit.

**Progress rule (anti-thrash):**
- the ladder never picks the growing request itself as the victim;
- a resumed request re-enters admission ahead of new arrivals;
- no request is preempted twice inside N rounds.

**Tests:** (a) a single 1M request plus a growing 400K neighbour complete both;
(b) ping-pong preemption is impossible.

## 6. The T21 negative example: the first attempt's planner (analyse, never copy)

The first attempt's planner computes the pool as
`VRAM × 0.97 − weights`
(mimo-v2.6-flash-ds41rt-port `code/mimo26/plan/placement.py:264`,
`memory_reservation = 0.97` at `:214`). Defects, per ADVISOR-I3 §3 A9 and
COHERENCE-TRAPS T21 — **model** analysis of a static file, no code copied:

1. no workspace/graph term (our ~2–3 GiB at chunk 2,048);
2. no CUDA-context term (~0.5 GiB) and no runtime headroom (2 GiB);
3. `VRAM` is the nominal 32 GiB of a 31.8 GiB card;
4. weights counted once for a two-GPU pair whose lanes each hold a full copy;
5. MTP counted all-BF16;
6. pool sized "@ 8 seqs" (`placement.py:268`) while engine concurrency is 16.

Its generated 5090 configs pin an 18.13 GiB pool that the engine's own check
would reject (ADVISOR-I3 §3 A9) — startup failure, or starved concurrency
(T21 symptom). The repack and padding arithmetic in that file stays eligible
for copy-in (ARCHITECTURE §11.8); the pool arithmetic is permanently quarantined
(ITERATION "P-300 revision", standing constraints).

## 7. Pressure ladder

Applied strictly in order, all under the no-tax rules of P-306 (ADVISOR-I3 §3
A1; ARCHITECTURE §11.1):

1. **Drop clean retained snapshots** (device-side retention banks). They are
   already written behind to tier 1, so this is free — **model** (P-306 store
   modes).
2. **Preempt the lowest-priority active request to tier 1.** Its sealed pages
   are already there (streaming write-behind); the transfer cost is its tail
   plus one open page ≈ **37 MB** — **model** (34 MB tail + 2.95 MB open page;
   ADVISOR-I3 §3 A1, P-306 anatomy). The request resumes later from its
   snapshot.
3. **Hold new admissions; return 429 with `Retry-After`.**

**A running request is never failed for pool pressure** (ARCHITECTURE §11.1).
Contrast v10, which on pool pressure kills the affected running request
(error event, `finished = true`, `cacheable = false` —
mimo26-flash-tj `v41_native_serve/scheduler.rs:358-368`, cited in ADVISOR-I3
§3 A1 as the behaviour to avoid). Preemption (step 2) is pause-and-tier, not
failure; only client-visible new work sees 429.

**429 scope (fold N5, ADVISOR-I4 §2):** a 429 on pool pressure can trigger
client-side fallbacks (LiteLLM routing, agent-harness retries). Keep
**429 + `Retry-After`**, and the A8 front-door contract states that the LiteLLM
group for this model has **no cross-model fallback on 429** — a deliberate
backend-down gets a fallback; transient pressure does not.

## 8. Open questions (named, not papered over)

| # | Question | Owner | State in this paper |
|---|---|---|---|
| 1 | **D2 — tier-1 pinned budget on the coordinator.** Proposal of record: **48 GiB of 125 GiB** = 4.5M tokens (64 GiB = 6.0M), **model** (ADVISOR-I3 §3 A2, §7 D2). Coordinator RAM is 125 GiB total / 85 GiB available — **measured** (probe 23 Sep, ADVISOR-I3 §2). | the maintainer | Carried as a **proposal** only. `host_cache_bytes` stays `0` until D2 lands (ARCHITECTURE §11.11). |
| 2 | Exact boot pool figure on the 5090 | I5 measurement | §3's 14–16 GiB is **model**; the boot log becomes the **measured** receipt. |
| 3 | FP8-KV acceptance bar (D5) — if per-token scales are forced in, every B/token and GB figure here grows by the scale layout | the maintainer (D5) | Unit-scale FP8 assumed throughout (ADVISOR-I3 §10.4.2). |
| 4 | 5090 PCIe link trains Gen1 x16 today (coordinator host notes defect 1; ADVISOR-I3 §2) — affects restore budgets in P-306, and preemption step 2's wall time | D1 window | Noted; ladder order is unchanged. |

## 9. Where the code lands (for the reviewer's map)

- **I5:** `A1 admission v1` (ITERATION P-300 revision, I5 row) — §5
  reservation + §7 ladder + pool-at-boot measurement. Tests: "admission and
  pool tests" per T21's Must column (COHERENCE-TRAPS §6).
- **I5b:** tier-1 preemption target exists (P-306).
- Config surface: `admit_reserve`, `max_output_tokens`, `host_cache_bytes`
  (ARCHITECTURE §11.11).

## 10. Claim → citation map

| Claim | Label | Citation |
|---|---|---|
| GA KV 11,520 B/token FP8, 12.1 GB per 1M request | model | bench/model/afd_vs_tp4_model.py:26; ADVISOR-I3 §3 A1; ARCHITECTURE §11.1 |
| Unit-scale FP8 ⇒ no scale bytes today | model | ADVISOR-I3 §10.4.2 (amends ARCH §11.7) |
| SWA 12,779,520 B/seq; DFlash 20,971,520 B / 10.5 MB FP8; MTP 983,040 B | model | bench/model/afd_vs_tp4_model.py:172-174; ARCHITECTURE §11.1; AGENTS.md §3 |
| Pool ~14–16 GiB ≈ 1.3–1.5M tokens, measured at boot | model (until I5 boot log) | ADVISOR-I3 §3 A1; ARCHITECTURE §11.1, §11.8 |
| Budget decomposition (weights 11.2 GiB, context 0.5, headroom 2, workspaces 2–3) | model | ADVISOR-I3 §3 A1; headroom constant mimo26-flash-tj memory.rs:15 |
| Startup-only sizing + fail-loud check | precedent | mimo26-flash-tj memory.rs:1,209-212 |
| Long-context lane: concurrency 1 above 512K | model | ADVISOR-I3 §3 A1; ARCHITECTURE §11.1 |
| Admission reserve prompt + min(max_tokens, 8192), grows per round | policy (model) | ADVISOR-I3 §3 A1; ARCHITECTURE §11.1, §11.11 |
| T21: lifetime reservation = 755 MB/request (65,536 × 11,520 B) | model | COHERENCE-TRAPS §6 T21; ADVISOR-I3 §3 A1; T23 default cap ADVISOR-I3 §10.2.1 |
| v10 lifetime-budget admission | measured code | mimo26-flash-tj scheduler/admission.rs:47-51; scheduler.rs:245-255,269-270 |
| v10 fails a running request under pool pressure (anti-pattern) | measured code | mimo26-flash-tj scheduler.rs:358-368 (via ADVISOR-I3 §3 A1) |
| placement.py pool math defects ×6 | analysis (negative example; never copy) | mimo-v2.6-flash-ds41rt-port code/mimo26/plan/placement.py:214,264-268; ADVISOR-I3 §3 A9; ITERATION P-300 standing constraints |
| Pressure ladder 1→2→3; running request never failed | policy | ADVISOR-I3 §3 A1; ARCHITECTURE §11.1 |
| Preemption cost ~37 MB (34 MB tail + 2.95 MB page) | model | ADVISOR-I3 §3 A1; anatomy in ARCHITECTURE §11.2 (P-306) |
| Card 32,607 MiB; Coordinator RAM 125/85 GiB | measured | ADVISOR-I3 §2 (probes, 23 Sep 2026) |
| TP4 holds 186.8 GiB KV / 46.69 GiB per rank at 500K | measured | ADVISOR-I3 §10.1, §10.4.4; bench/model/afd_vs_tp4_model.py:198-203 (MEASURED_TP4) |
| D2 proposal 48 GiB = 4.5M tokens | model | ADVISOR-I3 §3 A2, §7 D2 |

**Nothing in this paper is uncited except the design judgments themselves**
(the lane threshold rationale in §4 and the ladder ordering in §7 are policy
arguments traceable to ADVISOR-I3 §3 A1, whose inputs are cited above).

## Revision 23 Sep 2026 AEST: corrections of record after adversarial review (F1–F7)

The external review REFUSE was narrow and corrections-only: design, discipline
and every key figure passed (~85 citations spot-checked); the review itself is
the receipt. This paper takes F1, F4, F6 and F7 (F2, F3 and F5 landed in
P-306). All fixes verified against source before editing.

- **F1 (HIGH), §5 T21 bullet — arithmetic error of record.** "Two such requests
  and a modest prompt consume a third of the pool" was wrong by ~3×. Replaced
  with: two such requests with modest prompts consume ~1.6 GB (≈10%) of a
  14–16 GiB pool — and the 755 MB/request overhang is what never materialises
  (only ~128K-token prompts would make the pair reach a third of the pool).
  Verified: 2 × 754,974,720 B + 8,192 × 11,520 B = 1,604,321,280 B = 10.7% of
  14 GiB / 9.3% of 16 GiB; at 128K prompts the pair is 4.53 GB = 30% of 14 GiB.
- **F4 (MED), §0 — wrong rule id.** The stub discount rule is TEST-PLAN **R5**
  ("Stubs design, live gates promote … Discount stub predictions 3–5×",
  TEST-PLAN.md:19); R7 is the bench-ladder/WAIT rule (TEST-PLAN.md:21). Cite
  corrected.
- **F6 (LOW), §2 — wrong section for the one-residency rule.** "Keep only one
  resident per deployment" is ADVISOR-I3 **§3 A7** (ADVISOR-I3:355); §7 D6 only
  picks the default drafter. Cite corrected.
- **F7 (LOW), §3 consequence — spare-capacity band too low.** "~2–3 GiB to
  spare" → "~2.7–4.7 GiB to spare". Verified: 1,048,576 × 11,520 B +
  34,056,192 B tail = 12,113,651,712 B = 11.28 GiB, leaving 2.72–4.72 GiB of a
  14–16 GiB pool.
