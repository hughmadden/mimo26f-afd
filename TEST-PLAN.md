# mimo26-flash-afd — test plan skeleton

**22 September 2026 AEST.** Unit, stub, and bench layers for the greenfield
engine. Companion to [`ARCHITECTURE.md`](ARCHITECTURE.md). Patterns are lifted
from what actually paid rent in `glm-5.2`, `glm-5.3-flash-afd`,
`dsv41-flash-tp4-engram`, `vllm-afd-port`, and the CPU twin in
`mimo-v2.6-flash-ds41rt-port/`.

---

## 0. Rules (non-negotiable)

| # | Rule | Source of the scar |
|---|---|---|
| R1 | **CPU suite is the merge gate.** `cargo test` + `pytest oracle` + harness selftests run with no GPU, no network, no weights. Green or no merge. | vllm-afd-port 274 CPU tests; dsv41 BUILD-RECIPE "unit-test before building" |
| R2 | **External oracle at least once.** Self-consistency ≠ fidelity. One tiny-dim run of Xiaomi `modeling_mimo_v2.py` (or real tensor slices) must agree with our numerics. | First-attempt review Finding 1 / 4: goldens pinned their own bug |
| R3 | **Negative tests for known traps.** Fused-QKV naive path, per-shard scale padding, SWA eviction, E8M0 clamp, missing bias/norm names — each has a test that **fails** on the wrong implementation. | PERF-CORRECTNESS HIGH findings 1–3 |
| R4 | **Invariants as tests.** incremental ≡ full recompute; spec commit stream ≡ target-only at temp 0; pool math ≡ pinned bytes; sink mass / window properties. | CPU twin suite (keep, don't rewrite) |
| R5 | **Stubs design, live gates promote.** A stub-only speedup never flips a config. Discount stub predictions 3–5× and label them. | vllm-afd-port: 4th stub-predicted concurrency fix failed live |
| R6 | **Harness selftests before harness runs a model.** Fake streams, known-correct/incorrect/timeout cases. | glm-5.3 W10 qual packs (38 offline tests) |
| R7 | **Bench ladder, then STOP.** working config → coherence → ≤10-min smoke → notify the maintainer → **WAIT**. No hours-long unasked. | Standing rule; every recipe |
| R8 | **Failed rows retained.** Never retry-and-replace, never fill in, never regrade. Correction-of-record appends. | glm-5.3 R15/R16 COUNT history |
| R9 | **Identity before measurement.** Image/SHA/argv/config readback in every run receipt. | dsv41 image sentinel; glm public-repo safety |
| R10 | **Guards on Sparks.** MemAvailable floor (8 GiB precedent) + refuse concurrent CUDA owners. | glm-5.3 public launchers |
| R11 | **Sydney time** in every artifact name and log line. Dual-stamp UTC when correlating hosts. | House rule |
| R12 | **One harness invocation at a time** for timing-sensitive suites. | dsv41 hostcache rotate suite |

---

## 1. Test pyramid

```text
        ┌─────────────────────────────┐
        │  L6  Bench matrix (gated)   │  minutes–≤10 min cells, PO after
        ├─────────────────────────────┤
        │  L5  Fleet live qual        │  coherence · COUNT · needle · draft
        ├─────────────────────────────┤
        │  L4  Fleet stubbed integ    │  fake sparks / TCP loopback
        ├─────────────────────────────┤
        │  L3  Single-host GPU smoke  │  5090 attention / 1 Spark GEMM
        ├─────────────────────────────┤
        │  L2  Stub / sim             │  CPU fakes of GPU + RPC + latency
        ├─────────────────────────────┤
        │  L1  Contract / golden      │  byte pins, real headers, oracle
        ├─────────────────────────────┤
        │  L0  Unit (CPU)             │  tiny dims, property + negative
        └─────────────────────────────┘
              merge gate ─────────────────────────► L3+ need the maintainer
```

---

## 2. L0 — Unit (CPU, milliseconds)

**Location:** `oracle/tests/` (existing CPU twin — consume) +
`crates/*/src/**/*_tests` (new engine units).

**What lives here**

| Area | Cases (skeleton) | Notes |
|---|---|---|
| MXFP4 codec | pack/unpack round-trip, E8M0 decode range, **clamp ≠ 1.0 bug**, nibble order, amax>6 regression | Goldens byte-pin encode |
| FP8 block-128 | encode/decode representable set, block scale apply, non-finite sanitization | |
| Fused-QKV split | synthetic 4-shard set: naive path corrupts / fixed path exact; scale-grid padding trim; uneven-shard mis-slice | **R3 negative** |
| Config/derived | GA 11520 B/token, SWA ring 12 779 520 B/seq, expert params 302 795 194 368; `tiny()` twin scales | Fail-loud if HF config drifts |
| Attention props | window-128 isolation, sink mass, value-scale linearity, partial-rotary first-64 dims, QK≠V shapes | |
| KV eviction | incremental ≡ full; SWA cut = `min(batch_pos)−window+1`; O(1) append amortization smoke | |
| Router | sigmoid+ bias selection, top-8 renorm, sum dispatched weights = 1 | |
| Sampler | temp 0 ≡ greedy; top-p/k truncation; eos set; seeded RNG determinism | |
| Spec transaction | greedy accept prefix; rollback; adaptive-K bounds; commit stream ≡ target-only (temp 0) | R4 |
| MTP / DFlash | chain length, block-8 fill, mask token, injection layer ids present | |
| Loader names | every consumed tensor name exists in the real 73 081-entry index | Caught bias/norm HIGH bugs |
| Wire codec | magic/version/96/40/12 lengths, flags, compact_id, SHA debug checksum | Twins Python ↔ Rust |
| Placement | 2/4 clean; 3/5/6 padding waste bounds; coordinator footprints | |
| Harness selftests | parser, timeout, abort, known-good/bad answers, census math | R6 |

**Pass bar:** 100% of L0 green in CI; any skip is a written waiver in HANDOFF.
Target runtime < 5 s (tiny dims only — no 300 B allocations).

**Proven pattern:** `python3 -m pytest oracle/tests -q` + `cargo test --workspace`.

---

## 3. L1 — Contract / golden (CPU, still no GPU)

**Location:** `oracle/tests/golden/`, `oracle/tests/test_real_slices.py`,
`oracle/tests/test_snapshots.py`.

| Case | Input | Assert |
|---|---|---|
| Byte-exact corpus | `gen-golden.py --check` | Python ↔ Rust identical files |
| Real safetensors headers | cached ep0 / mtp / dflash headers | shapes, dtypes, scale-grid geometry |
| Real tensor slices | qkv / scale / expert / embed bytes | E8M0 sane range; dequant stats as nibble-order canary |
| External oracle (once per numerics change) | Xiaomi `modeling_mimo_v2.py` at `tiny()` | logits / attn out within declared tol |
| Name audit | loader-consumed set vs real index | zero missing, zero silent extras |

**Pass bar:** `--check` byte-for-byte; header suite green without network (headers
cached). External oracle is a **recorded receipt**, not a CI dependency (the
upstream file may move — pin a SHA copy under `vendor/oracle-src/`).

**Proven pattern:** first-attempt goldens (keep) + glm wire-integrity CPU tests
(CPU-first validate the diagnostic before trusting it).

---

## 4. L2 — Stub / simulation (CPU)

**Purpose:** exercise seams that need hardware without hardware. **May rank
design choices. May not promote them (R5).**

**Location:** `harness/stubs/` + unit tests beside them.

| Stub | Fakes | What it can decide | What it cannot |
|---|---|---|---|
| `LaneSim` | expert dispatch/combine RTT, loss, reorder | wire/frame logic, rank map, waste math | GEMM numerics, real bandwidth |
| Fake RPC actor | multi-lane proposals, stream vs chunk grants | scheduler deadlocks, ordering invariants | ms-level speedups (apply 3–5× discount) |
| CPU GEMM oracle | MXFP4 grouped GEMM at 2048×4096×3 | conformance to goldens | throughput |
| Admit sim | queue depth, 429 vs 503, Retry-After | policy correctness | tail latency under real load |
| Draft sim | acceptance traces from recorded token streams | adaptive-K policy, rollback | true acceptance rate on distribution |
| Prefill/decode chunker | forced interleavings | starvation, fairness properties | CUDA graph / capture bugs |
| AOT SM gate | 170/121 assert | rebuild-don't-patch rule | actual kernel launch |
| Fault inject | lane drop, partial frame, timeout | recovery paths | silent corruption below the frame layer (need L4 integrity) |

**Required stub tests (minimum)**

1. Two lanes in-flight do not starve decode (fake-RPC).
2. Chunk grant vs decode grant same-round contract (vllm-afd-port Fix A shape).
3. LaneSim drops N frames → session recovers or fails closed (no silent truncate).
4. Draft reject does not advance committed history.
5. Admission: pool empty → deferred/requeue, not silent 500 (HC-13 lesson).
6. Frame round-trip + intentional bitflip → detector fires (wire-integrity pattern).

**Pass bar:** suite green; every stub has a header comment: *"prediction only;
promotion requires L5 gate X"*. Any write-up that quotes a stub number quotes the
discounted band too (R5).

---

## 5. L3 — Single-host GPU smoke (bounded, one device)

**Purpose:** prove kernels on real silicon before any fleet path.

| Cell | Device | Budget | Assert |
|---|---|---|---|
| Attention kernel vs CPU oracle | 5090 | < 2 min | rel tol on outputs at tiny + real shapes |
| MXFP4 grouped GEMM vs CPU oracle | 1 Spark *or* 5090 stand-in | < 2 min | rel tol; nibble order hardware-proof |
| FP8 QKV/o_proj GEMM | 5090 | < 2 min | vs dequantized FP32 |
| AOT bake + SM gate | each class | build-time | refuses wrong SM (170/121) |
| KV pool alloc/accounting | 5090 | seconds | bytes match §7 pins under alloc pressure |

**Pass bar:** numerics within tol; no multi-GPU; **no model-wide forward yet**.
Do not run a second CUDA process beside a loaded engine (R10).

**Proven pattern:** dsv41 `gateA-5090-raw` spine GEMM bench receipts;
KERNEL-SPECS AOT bake/check chain.

---

## 6. L4 — Integration on a bench (fake or partial fleet)

**Location:** `harness/` + `configs/dev/`. Loopback TCP or 1 real Spark + CPU-sim
peers.

| Case | Topology | Assert |
|---|---|---|
| Wire round-trip multi-rank | 4 fake sparks | encode/decode + reorder tolerance |
| Frame integrity ladder | Ts 128, 2048, … | no payload corruption (wire-integrity pattern) |
| Target-only e2e | 1–4 ranks, tiny weights | greedy tokens match CPU oracle stream |
| Prefix reuse | exact hit | `cached_tokens` visible through the API (dsv41 streaming-usage lesson) |
| SWA eviction e2e | long synthetic seq | incremental ≡ full still holds across ring wrap |
| Draft on/off identity | temp 0 | committed stream identical with draft disabled |
| Restart / reconnect | kill one fake spark | fail closed or recover; no partial rows committed |
| Concurrency C1/C2/C4 | fake RPC | invariants hold; counters consistent (`stores = evictions + resident` if hostcache on) |

**Pass bar:** all green with `harness selftests` first (R6). Wall clock < 10 min.
This is the last layer before touching the real 4+1 fleet.

---

## 7. L5 — Fleet live qualification (gated, short)

**Trigger:** the maintainer's explicit go. **Instrumentation:** receipts under
`runs/<date>-<slice>/`, identity readback first (R9), Spark guards armed (R10).
**Budget:** the whole ladder ≤ 10 min unless the maintainer extends (R7).

Ordered gates — fail stops the ladder and is recorded (R8):

| Gate | What | Pass bar | Pattern from |
|---|---|---|---|
| G0 Identity | image SHA, argv, config, MemAvailable, `nvidia-smi` snapshot | match plan | dsv41 sentinel |
| G1 Ready | readiness probe + weight residency + transport up | clean readback | ds41rt readiness contract |
| G2 Coherence | **3/3**: simple factual, arithmetic, JSON shape (temp 0) | 3/3 exact | glm-5.3 `APPLE / 17*23 / JSON` |
| G3 COUNT | 5× deterministic count tasks, full answers, temp 0 | 5/5 complete answers (not reasoning-only) | glm R15/R16 hard lesson |
| G4 Needle | exact-reuse prefix + one long-context needle (glm-5.2 100% bar is the aspiration; start with 3 needles) | 3/3 retrieved | glm-5.2 needle harness |
| G5 Decode smoke | ≤10 min: single-stream + C3, prose prompt, 200 out | 0 errors; tok/s recorded as **smoke**, not a claim | dsv41 decode5 |
| G6 Draft identity *(S5+)* | draft on vs off, temp 0 | identical committed stream | R4 live |
| G7 Acceptance *(S5+)* | matched prompts, acceptance + all-token/s | record only; no promotion claim without a baseline | glm matched-smoke honesty |

**On any fail:** restore known-good if a deploy happened; notify the maintainer;
write RESULT with the failed row intact.

**Proven pattern:** dsv41 "equivalence 3/3"; glm COUNT discipline; W10 prose-code
qualification pack (selftested validators).

---

## 8. L6 — Bench matrix (explicitly gated)

**Only after L5 green and the maintainer names the question.** Cells are minutes, not hours.
Three runs for final qualification; medians + variance; config/SHA in every table.

| Cell ID | Question | Shape | Metric |
|---|---|---|---|
| B-prefill | TTFT / prefill tok/s vs prompt ladder | 8K / 64K / 256K | TTFT p50, prefill tok/s |
| B-decode1 | single-stream decode | 8K + 200 out | tok/s, accept len |
| B-decodeC | aggregate at C3 / C6 | same | aggregate + per-stream |
| B-draft | MTP vs DFlash vs off | matched prompts | accept rate, all-token/s, wall |
| B-accept | temp 1.0 / top-p 0.95 sampling acceptance identity | fixed corpus | common-nonce vs min(1,p/q) (ARCH §9.1) |
| B-cache | cold vs exact-reuse TTFT | prefix salt / no salt | ratio |
| B-pool | concurrency ceiling vs pool | C ladder until 429 | admitted vs deferred |
| B-repack | 4-Spark pad/waste sanity | plan only + one load | bytes/rank vs planner |

**Reporting:** `bench/<date>/` with raw JSONL + one SUMMARY table. Failed rows
stay.

**Proven pattern:** dsv41 `bench_c1c6.py`, `decode5`, hostcache JSONL with
`/v1/stats` brackets; glm-5.2 findings tables; vllm-afd ABAB scorecard (and its
stub-discount column).

---

## 9. Acceptance / rollback special cases

Speculative decoding needs its own identity tests (cheap, L0/L2/L4):

1. **Greedy identity:** draft on/off, temp 0 → bit-identical token streams.
2. **Rollback identity:** force mid-block reject → committed prefix equals
   target-only prefix.
3. **Bonus-token rule:** commit = longest accepted prefix + 1; asserted, not assumed.
4. **Sampling identity (once chosen):** seeded common-nonce run is deterministic;
   Leviathan path passes the residual-resample invariant in unit form.
5. **Adaptive-K bounds:** never proposes past block size; never commits unverified.

Live form is G6/G7 — short, matched prompts, no performance claim until B-draft.

---

## 10. Artifact layout

```text
runs/YYYYMMDD-HHMM-<slice>-<what>/
  identity.md          # R9 readback
  config.sha
  raw/*.jsonl
  RESULT.md            # pass/fail per gate; failed rows intact
harness/
  selftests/           # R6 — offline
  stubs/               # L2
  fleet/               # L5/L6 drivers (coherence, count, needle, decode)
bench/YYYYMMDD-<topic>/
  SUMMARY.md
  raw/
```

---

## 11. Anti-patterns (do not repeat)

1. **Self-consistency goldens without an external oracle** — pins your own bug.
2. **Stub-predicted speedup as a promotion gate** — needs live A/B.
3. **LAUNCH-READY meaning "docs ready"** — label must mean "binary can serve".
4. **Tree-wide rename / dual-model keep** before first tokens.
5. **Reasoning-only answers counting as COUNT pass.**
6. **Retrying a failed cell and replacing the row.**
7. **Harness that was never selftested** against fakes.
8. **Hours-long benches unasked** or notifications that wake the maintainer.
9. **Multi-agent freeze protocols** on one engine — assign ownership instead.
10. **Doc correction-of-record as a substitute for a fix** — fix the three call
    sites *and* the test that would have caught it.
11. **Bespoke toolchain calls** — `cargo` / `nvcc` / `pytest` invented in chat.
    The verb is `scripts/dev.sh`. A repeated workaround becomes a verb, a pin
    in `configs/build.env`, or a skill, before the iteration closes.

---

## 12. Mapping: slice → minimum tests

| Slice | Must be green before exit |
|---|---|
| S0 | L0 smoke of scaffold, harness selftests exist |
| S1 | L0 full + L1 goldens + external oracle receipt |
| S2 | L3 attention cells + L0/L1 still green |
| S3 | L3 GEMM cells + L4 wire/LaneSim |
| S4 | L4 e2e target-only + **L5 G0–G5** |
| S5 | L4 draft identity + **L5 G0–G7** + B-accept (short) |
| S6 | same as S5 vs DFlash + B-draft (short) |
| S7 | L5 regression + named L6 cells only |

---

## 13. Owner checklist per PR (skeleton)

- [ ] L0/L1 green (paste counts)
- [ ] New trap ⇒ new negative test
- [ ] Stub numbers labelled + discounted (if any)
- [ ] No config promotion without a live gate ID
- [ ] Identity fields present in any receipt
- [ ] Sydney timestamp
- [ ] HANDOFF status line updated (append, don't rewrite)
## 14. P-300 additions (23 September 2026 AEST — folds ADVISOR-I3 §5)

- **G4 becomes a ladder:** 8K, 32K and 128K needles (3 each). 512K and 1M are named L6 cells.
  **Amended by ADVISOR-I5 I5-R6 (2026-09-24 15:36 AEST):** the ≤ 10 min L5 holds G0–G5 with G4 at 8K and 32K. The G4 128K row is its own bounded cell (≤ 10 min) right after it, because a first-tokens 128K cold prefill is modelled at about 483 s (`docs/design/integration-i5.md` §4).
- **B-prefill** extends to 512K and 1M, and adds a chunk 256 vs 2048 A/B.
- **B-tier:** restore 170K and 1M from tier 1 and tier 2 — latency and bytes, then continue-exactness.
- **Tax gate (R-NOTAX):** cache on vs off within 1%.
- **B-mix:** C1/C4 decode latency while a 256K prefill runs (A4).
- **G4n FP8-KV quality:** needle plus log-prob deltas vs a BF16-KV reference at 32K/128K.
- **First-attempt smoke lessons:** concurrency cells must actually run concurrently (the first attempt's "C6" was 3 sequential calls summed); prefill tok/s comes from `usage.prompt_tokens`, not a nominal count; coherence and thinking checks go through the chat endpoint, not `/v1/completions`.
