# mimo26-flash-afd — architecture skeleton

**22 September 2026 AEST.** Greenfield AFD inference engine for
`XiaomiMiMo/MiMo-V2.6-Flash-RL`, first topology **4× DGX Spark + 1× RTX 5090**.
Design pressure: extreme performance; method pressure: *do not refactor ds41rt*.

This is the skeleton to argue over before any engine code is written. Companion:
[`TEST-PLAN.md`](TEST-PLAN.md). Research inputs stay in
`../mimo-v2.6-flash-ds41rt-port/` (model facts, goldens, PORT-SURFACE) and
`../dsv41-flash-tp4-engram/` + `../glm-5.3-flash-afd/` (ops, bench discipline).

---

## 1. Goal and non-goals

**Goal.** Serve MiMo-V2.6-Flash-RL over OpenAI-compatible HTTP on 4 Sparks + 1 5090,
with both shipped drafters (MTP-3, then DFlash-5) behind one draft seam, under the
LiteLLM front-door contract (user URL / keys / model-group names never change).

**Non-goals (v1).** Vision/audio encoders. Multi-coordinator. TP5/6 topologies.
Engram / CED / mHC / MLA stack. Full-tree rename of any upstream. Keeping a
DeepSeek serving path. Public release. Long unattended benches.

---

## 2. Build principles (anti-lessons from the first attempt)

| # | Principle | Why |
|---|---|---|
| P1 | **Greenfield core.** New repo (`mimo26-afd` or similar). Upstream is *source material*, not a base to evolve. | The first attempt froze a 355K-line tree, then started renaming it. |
| P2 | **Copy-in allowlist only.** Each borrowed unit has a row in §5 with source path + why. Anything not listed is not copied. | Prevents "just add code to the existing codebase". |
| P3 | **Vertical slices with a running path.** First coherent tokens before any cosmetic work. | Docs and CPU twins accumulated for a day without a serving binary. |
| P4 | **One engine owner.** No multi-agent freeze/merge protocol. Reviews are pull-style. | `agent-collab.txt` became the product. |
| P5 | **Consume goldens, don't rewrite them.** The CPU `mimo26/` package + corpus is the numerics contract. | 270 tests and byte-pinned codecs already exist. |
| P6 | **Docs stay thin.** One ARCHITECTURE, one TEST-PLAN, one REUSE list, living HANDOFF. Corrections append; history is not rewritten. | First attempt grew 1800 lines of contract before a kernel. |
| P7 | **Stubs may design, never promote.** Live gates decide. Stub speedups get a 3–5× discount. | vLLM-afd-port: four stub-predicted concurrency wins failed live. |
| P8 | **Bench-gating.** Working config → coherence → ≤10-min smoke → notify the maintainer → WAIT. | Standing rule across every recipe. |

---

## 3. System shape (AFD seam)

```text
                    ┌──────────────────────────────────────┐
   OpenAI HTTP ───► │  Coordinator · 1× RTX 5090           │
   (LiteLLM)        │  ─────────────────────────────────   │
                    │  REUSE   api · admit · sched · kv    │
                    │          wire client · AOT bake      │
                    │  WRITE   attn (GQA+SWA+sink)         │
                    │          load (fused-QKV) · router   │
                    │          sample · draft MTP/DFlash   │
                    └───────────────┬──────────────────────┘
                                    │  proto v2 · RoCE verbs
                    ┌───────────────▼──────────────────────┐
                    │  Spark expert ranks · ×4 (TP4/EP4)   │
                    │  ─────────────────────────────────   │
                    │  REUSE   dispatch/combine loop       │
                    │          MXFP4 E2M1+E8M0-32 family   │
                    │          role labels · AOT SM 121    │
                    │  WRITE   expert shard map (64→N)     │
                    └──────────────────────────────────────┘
```

Seam rule (from ds41rt `architecture.md`): split at the attention/FFN boundary.
Coordinator owns embeddings, all attention, sampling, API, KV pool, drafting.
Sparks own routed-expert FFN only.

**Delete-on-sight list** (never enter the new tree): Engram, CED compressor/indexer,
MLA/sparse-attention stack, mHC, dSpark implementation, EXL3/NVFP4 converted paths,
DeepSeek vision encoder, DS/GLM constant families, `real_full` bring-up harness,
DeepSeek-shaped literals (5120 / 40 layers / 384 experts / top-6 / 890 B/token).

---

## 4. Module skeleton (new tree)

```text
mimo26-afd/
  crates/                     # or src/ — name is free; keep it small
    mimo26-api/               # OpenAI-compatible surface (REUSE shape)
    mimo26-admit/             # admission, queue, 429/503 policy (REUSE)
    mimo26-sched/             # waves around remote expert boundary (REUSE)
    mimo26-kv/                # GA grow + SWA ring-128 + pool accountant (WRITE)
    mimo26-attn/              # GQA+SWA+sink+v-scale+partial-rotary (WRITE)
    mimo26-moe/               # router top-8 sigmoid + dispatch/combine (WRITE thin)
    mimo26-load/              # safetensors, fused-QKV shard-major (WRITE)
    mimo26-draft/             # MTP-3, DFlash-5 behind DraftAdapter (WRITE)
    mimo26-sample/            # temp/top-p/top-k/greedy (WRITE)
    mimo26-wire/              # DS41RTE3-v3 frame codec (COPY)
    mimo26-expert/            # Spark MXFP4 grouped GEMM service (COPY family)
    mimo26-plan/              # placement / RepackPlan (FROM CPU twin)
    mimo26-bin-coordinator/
    mimo26-bin-spark/
  vendor/                     # pinned upstream copies, never edited in place
    ds41rt/                   # git pin only — reference + copy source
  oracle/                     # CPU numerics contract — IMPORT, do not fork
    mimo26/                   # existing package from the first attempt
    tests/golden/
  harness/                    # see TEST-PLAN.md
  configs/
  docs/
    ARCHITECTURE.md           # this file (or relocated)
    REUSE.md                  # §5 expanded as rows land
    HANDOFF.md                # living state
```

Language split stays Rust + CUDA, same as the proven engine: Rust for daemon/API/
wire/schedule, CUDA for attention + expert GEMMs. Python is **oracle and harness
only** — not a third runtime in the serving path.

---

## 5. REUSE allowlist (copy-in, pin the source SHA)

Copy means: take the file/crate as-is into `vendor/` or a `mimo26-*` crate, adapt
names at the boundary only, record SHA. No "while I'm here" cleanups.

| Unit | Source (`tpurtell/ds41rt` unless noted) | Action | Notes |
|---|---|---|---|
| Wire frame codec | `rust/crates/ds41rt-transport/src/protocol_v2.rs` | **COPY** | MAGIC `DS41RTE3` + VERSION 3, 96/40/12 B headers. Keep magic (P: wire is a fleet contract). Goldens already re-based in CPU twin. |
| Verbs/RoCE + TCP fallback | `ds41rt-transport` (v41_expert/roce.rs, tcp.rs) | **COPY** | Role labels: start TP4 only; leave the 2/3/6 gate strings alone until needed. |
| Admission + HTTP front door | `ds41rt-api` | **COPY shape** | Strip DeepSeek model registry; keep 429/Retry-After vs 503 policy (dsv41 HC-13 lesson). |
| Scheduler waves | `ds41rt-daemon` schedule/admission | **COPY shape** | Alternating local/remote waves around expert boundary. No Engram/CED calls. |
| KV pool + hostcache | `ds41rt-hostcache` + daemon prefix path | ~~**DEFER**~~ **REQUIRED** (§11.2) | ~~Optional S7. v1 is device pool only + exact prefix reuse.~~ Superseded 23 Sep 2026: RAM write-behind tier 1 in I5b, Spark tier 2 in I8 (§11.2). |
| MXFP4 pack/GEMM family | `native/v41_expert*.cc`, `nibbles`, `expert_format.rs` | **COPY family** | Already FP4 E2M1 + E8M0-per-32. Retarget shapes to 2048×4096×3, 256 experts. |
| AOT SM gate + bake | `justfile`, `scripts/`, KERNEL-SPECS §6 | **COPY** | 170 = 5090, 121 = GB10 Spark. Rebuild-don't-patch rule encoded as a gate. **Two gates, not one (§11.9):** arch sm_120/sm_121 and SM count 170/188/48. |
| Draft seam traits | `DraftChain` / `VerificationTarget` | **COPY traits** | Replace dSpark body with MTP-3 / DFlash-5. |
| FP8 block-128 path | `v41_fp8.cc` + loader sites | **COPY family** | Attention/MLP projections; o_proj stays BF16 (ignored_layers). |
| Placement planner math | CPU twin `plan/placement.py` | **PORT to Rust** (repack/padding only) | Do not rewrite the repack arithmetic. **Do not port its KV-pool arithmetic** (§11.8: VRAM×0.97 − weights, no workspace/headroom). |
| API smoke / doctor | `scripts/doctor.sh`, `api-smoke.sh` | **COPY** | Reuse as `harness/preflight`. |
| Build/launch scripts | `build.sh`, `run.sh`, `wip.sh` | **ADAPT slim** | One coordinator + one spark launcher. Drop multi-model WIP matrix. |

**Explicitly not copied:** entire `native/src` tree as a unit; xgrammar (add later if
structured output is required); quantization toolchain (checkpoint is already
MXFP4 — only repack-to-engine-layout survives); `third_party/` except SparkInfer
pins the expert GEMM actually needs.

Provenance row format for `docs/REUSE.md` (fill as rows land):

```text
| unit | upstream path | upstream SHA | new path | delta allowed | tests that pin it |
```

---

## 6. Model write list (the real work)

Sourced from `mimo-v2.6-flash-ds41rt-port/ARCHITECTURE.md` + REVIEW (already
header-verified against the real HF checkpoint). Treat those numbers as pinned
unless a live gate disagrees.

1. **Loader** — 64 EP shards, ep0 dense; fused-QKV **always 4-way TP-ordered**
   (SWA layers too — the "SWA 8" reading is a known HIGH defect in older docs);
   per-shard FP8 scale-grid padding trim; MTP tensors carry `model.` prefix;
   `load_json_lenient` for `dflash/config.json` trailing comma; fail-loud on
   missing `e_score_correction_bias` / `post_attention_layernorm`.
2. **Attention** — GQA (GA 4 / SWA 8 KV heads), QK 192 / V 128, partial rotary 0.334
   (64 dims, dual θ 10M/10k), SWA window 128, learnable sink bias **per-Q-head [64]**
   as extra logit column, `attention_value_scale` 0.707 on V, o_proj BF16.
   No KV-head broadcast; QK≠V head-dim is first-class. This is the one kernel that
   is a true rewrite vs ds41rt (MLA → GQA).
3. **KV layout** — 9 GA layers grow per token (11 520 B/token FP8); 39 SWA layers
   ring-capped at 128 (12 779 520 B/seq). Evict only entries older than
   `min(batch_pos) − window + 1` (not "keep last window rows").
4. **Router** — sigmoid + `noaux_tc` bias (`mlp.gate.e_score_correction_bias`),
   top-8, `norm_topk_prob`, no shared experts, `n_group=1` (no group-limited hook).
5. **Sampler** — temp / top-p / top-k / greedy; defaults 1.0 / 0.95; eos
   `[151643, 151645, 151672]`; temperature 0 ≡ greedy exactly. Decide
   common-nonce vs Leviathan min(1,p/q) acceptance **by measurement**, not inheritance.
6. **Drafters** — MTP-3 first (chained `eh_proj`, 3 dense SWA layers, shared head);
   DFlash-5 second (block-8, mask 151675, anchors 4096, target layers
   `[0,11,23,35,47]`, value scale 0.612, sliding 1024). Adaptive-K reuses ds41rt
   machinery. **Superseded in part by §11.6** (MTP window is 128 and untied
   embeddings; DFlash conditions through `fc [4096,20480]` over five hidden states —
   `anchors 4096` is a training parameter; DFlash is the serving default).
7. **Expert repack** — 256 experts × 47 MoE layers → 4 ranks (4 experts/layer/shard
   matches the native 64-shard layout at TP4). Payload ~40.2 GB/rank at TP4.
   **Superseded by §11.5:** TP4EP1 quarter-intermediate slices of every expert (balanced
   per token), not whole experts per rank (EP4).

---

## 7. Vertical slice ladder

Each slice ends green on its gates (see TEST-PLAN) and is independently revertable.
No slice starts on the fleet without the maintainer's go.

| Slice | Outcome | Fleet? |
|---|---|---|
| **S0 Scaffold** | Repo, REUSE rows started, CI: `cargo test` + `pytest oracle` green, doctor/preflight copy | no |
| **S1 Load + CPU forward** | Real-geometry loader; tiny coherent forward; golden suite consumed; one run of Xiaomi `modeling_mimo_v2.py` as external oracle | no |
| **S2 Coordinator attention** | GQA+SWA+sink kernel on 5090; matches CPU oracle at tiny and real shapes (rel tol); AOT 170 bake | no |
| **S3 One Spark expert** | MXFP4 grouped GEMM + dispatch/combine on 1 Spark (or CPU-sim loopback first); wire codec round-trip | local only |
| **S4 Target-only text** | 4 Sparks + 5090, no drafters; coherence 3/3; one exact-reuse needle | yes, gated |
| **S5 MTP-3** | Drafter on the seam; acceptance measured; commit ≡ target-only at temp 0 | yes, gated |
| **S6 DFlash-5** | Block drafter; vs MTP-3 on the same ladder | yes, gated |
| **S7 Ops polish** | Sampling defaults live, LiteLLM mapping, optional hostcache, 2/3/5/6 Spark topologies | yes, gated |

Order is deliberate: **attention before experts** if the 5090 is free, because that
kernel is the highest-risk rewrite; experts reuse a known family. S3 can proceed in
parallel on paper but S4 needs both.

---

## 8. Config / launch surface

Minimal config (one file, fail-loud unknown keys):

```text
role            = coordinator | spark
model_path      = host path to HF snapshot
expert_hosts    = 4 names or addrs
spark_tp        = 4
kv_pool_bytes   = derived + override
prefill_capacity= 256 | 1024 | 4096     # ds41rt operating-point knob — superseded §11.4 (chunk 2048/4096; classes 256|2048|4096)
draft           = none | mtp | dflash
aot_sm          = 170 | 121             # gate, not a hint — split §11.9 (aot_arch + aot_sm_count)
```

Launch: `bin/mimo26-coordinator --config …` and `bin/mimo26-spark --config …`.
Self-guarding go-window scripts (from dsv41 v3port): refuse to start while the
previous generation is live; image/identity readback before readiness.

---

## 9. Open decisions (need the maintainer, not agent drift)

1. **Acceptance identity** for sampling-aware spec: common-nonce (seed-lossless)
   vs Leviathan min(1, p/q). Measure both on S5 smoke; pick one.
2. **Wire magic:** keep `DS41RTE3` (fleet-compatible) vs new `MIMO26E1`.
   Skeleton default: **keep**, unless we own both ends and want the split obvious.
3. **Repo home:** new public `mimo26f-afd` vs private until S4.
4. **Hostcache:** defer (v1 device-only) or cut into S4 if cold TTFT hurts. — **RESOLVED
   23 Sep 2026: required** (R-NOTAX, §11.2); budget is the maintainer's decision D2 (`docs/ADVISOR-I3.md` §7).
5. **Scope name:** Flash only (this doc) vs also plan Pro later on a bigger fleet.

---

## 10. Standing constraints (inherited)

Upstream pins are not stoppers: move to a newer upstream or fork it. The API
front door keeps its URL, keys and model names for clients.
Sydney time in every record. Failed bench rows retained, never
backfilled.

---

## 11. Revisions, 23 September 2026 (architectural advisor, after I2)

_Tags such as `[tool_call]`, `[/parameter]`, `[think]` and `[im_end]` are written in square brackets on purpose: literal ones break MiMo tool calls (COHERENCE-TRAPS T29)._

This section appends; it does not rewrite. It supersedes the §5–§9 lines marked
"superseded" above. Evidence, model numbers and the iteration mapping are in
[`docs/ADVISOR-I3.md`](docs/ADVISOR-I3.md). Model numbers come from
`bench/model/afd_vs_tp4_model.py`.

### 11.1 Full-context memory and admission

**KV per request:**
- GA KV is 11,520 B/token (FP8) plus scales, so a 1M-token request needs
  12.1 GB.
- SWA rings: 12.78 MB per sequence.
- DFlash context ring: 21 MB at BF16, 10.5 MB at FP8.
- MTP rings: 0.98 MB.

**The pool is measured at boot, never pinned from a planner.** On a 5090 it
is ~14–16 GiB, about 1.3–1.5M tokens: one 1M request, run in a long-context
lane with concurrency 1 for requests over 512K.

**Admission** reserves the prompt plus `min(max_tokens, 8192)` and grows per
round.

**Pressure ladder,** in order:
1. drop clean retained snapshots;
2. preempt the lowest-priority active request to tier 1;
3. return 429 with `Retry-After` for new work.

A running request is never failed for pool pressure.

### 11.2 KV tiers (R-NOTAX, R-TIER2)

The order is device → tier 1 (the coordinator pinned host RAM, write-behind) → tier 2
(Spark memory over RDMA) → recompute. Every tier follows host-cache design v3
R1–R11 (unpublished `dsv41-flash-tp4-engram/research/afd-hostcache-design.md`):
- zero overhead when off;
- no request ever waits on the cache until there is pressure (≤1% tax gate);
- restores go through the engine fill path within a budget, and fall through
  to prefill on timeout;
- bounded pinned memory;
- exact page sharing;
- single-threaded and event-driven.

| Snapshot part | Bytes | Notes |
|---|---|---|
| Sealed GA page (256 tokens) | 2,949,120 + scales | Immutable once sealed; COW-shared; written behind once ((page id, generation) identity map). |
| 39 SWA ring states | 12,779,520 + scales | Captured at an exact position (§11.3). |
| Drafter state | DFlash ring 20,971,520 (BF16); MTP 983,040 | Only the resident drafter. |
| Logit row | 305,152 (BF16) | First token after restore. |

**Store modes:**
- `on-retain` (v3);
- **streaming write-behind** of each GA page as it seals (~85 MB/s at 7.4k
  tok/s prefill). A long session is then already in RAM when it retires or is
  preempted.

**Restore targets (MiMo):** 170K tokens in ≤ 100 ms and 1M in ≤ 600 ms. Both
assume the 5090 PCIe link trains at Gen5; at the Gen1 it reads today, 1M takes
3–4 s.

**Tier 2:**
- ~50–60 GB free per Spark beyond its 40.2 GB expert slice, about 17–21M
  tokens across four.
- Stores are demotions from tier-1 LRU: rate-limited, lowest-priority QP,
  paused during prefill bursts.
- Restores are striped across the Sparks and staged through a pinned bounce
  buffer on the coordinator.

**Reuse** the `ds41rt-hostcache` crate (`ds41rt-persistence`, branch
`hostcache/rc6`). Only the anatomy and byte constants change.

### 11.3 Exact prefix reuse

- Restore only at an exact snapshot at or before the divergence point, then
  recompute the rest. No empty-window SWA replay (trap T17).
- Snapshots are retained at prompt end and completion end, plus optional
  in-prompt checkpoints every 32K tokens.
- The prompt-end bank is what makes thinking-on multi-turn reuse work.
  Clients drop `reasoning_content`, so the next prompt diverges right after
  the previous `[think]`.

### 11.4 Prefill chunk size and scheduling

**Chunk classes.** A standalone prefill chunk is 2,048 tokens (4,096 on 96 GB
coordinators). AOT expert capacity classes are {256, 2048, 4096}.

**Why not 256.** Every chunk of 256 or more tokens touches ~all 256 experts,
so each Spark re-reads its 40.2 GB slice from local memory for each chunk.
Model prefill rates: 256-token chunks give ~1.0–1.4k tok/s; 2,048-token
chunks give ~5.3–7.4k tok/s.

**Mixing prefill with decode:**
- If decode rounds touch under ~50% of experts (low concurrency), time-slice:
  a decode round, then a prefill chunk, under a decode-latency budget.
- Otherwise, merge prefill slices into the decode rounds.

**Long prefills.** A 1M prefill (~20–40 min on a 5090) is preemptible at chunk
boundaries.

### 11.5 Expert layout and wire

**Layout.** TP4EP1: each Spark holds a quarter intermediate slice (512 columns)
of every expert, so per-token work is balanced. Repack at load time, or once
offline to each Spark's NVMe (never to the DC).

**Returns.** One compact BF16 partial row (8 KB) per token per Spark, summed on
the coordinator. Never per-route FP32: that would be 24.6 MB/token, a ~1.3k
tok/s ceiling (ds41rt `docs/ds41-prefill-return-bottleneck.md`).

**Budget.** The coordinator carries 0.82 MB out and 1.54 MB in per token, a ~20k tok/s
ceiling at ~250 Gb/s (model). Keep the coordinator bond boot balance gate for every
measurement.

### 11.6 Drafters (amended by §11.12: `k` = 7 default)

**DFlash** is the serving default (tech report §6: +31.3% accepted length
versus MTP-3). Specification (trap T15):
- context feature `hidden_norm(fc(concat(h[l+1], l ∈ {0,11,23,35,47})))`,
  with `fc [4096, 20480]`;
- context K/V from each drafter layer's own `k_proj`/`v_proj`, plus `k_norm`
  and RoPE (θ 1e4, partial 0.5);
- anchor slot holds a real token; mask slots use `mask_embedding.pt`;
- seven predictions through the target's `lm_head` and `embed_tokens`;
- `q_norm` and `k_norm`; sink; `v_scale` 0.612 on query V and context V;
- window 1024.

**MTP-3** (trap T16): window 128, `v_scale` 0.707, `embed_tokens` inputs
(untied from `lm_head`).

**Deployment:**
- One resident drafter per deployment.
- Block 6 or 8 (width 5 or 7).
- `k` adapts to occupancy.
- Sample-then-match acceptance by default.

**Hidden taps.** The target pass exports layers 0/11/23/35/47 and the final
hidden state (ABI designed in I3).

### 11.7 Coordinator attention for full context (amended by §11.12: unit-scale FP8 KV first; FP32 router)

**KV and kernels:**
- Paged GA KV (256-token pages).
- Split-KV decode with a reduce kernel; target ≥70% of 1.79 TB/s.
- Chunked prefill over pages; target ≥100 TFLOPS BF16-equivalent.
- GQA-packed; QK 192 / V 128.
- Sink column on SWA layers only.
- `v_scale` before caching.

**RoPE** is computed in FP32 on the fly: partial 64 of 192 dims, θ 1e7 (GA) /
1e4 (SWA), no 1M-row tables.

**FP8 KV** uses an explicit per token × head scale layout for K and V, counted
in the pool bytes, with quality gated against BF16-KV at 32K and 128K.

### 11.8 Planner

Port only the repack and padding arithmetic. The first attempt's pool
arithmetic (`VRAM × 0.97 − weights`) omits workspace, runtime headroom and the
CUDA context, counts weights once for two GPUs, and pinned an unfundable
18.13 GiB pool. The engine sizes its pool at boot.

### 11.9 AOT gates

Two gates:
- `aot_arch`: sm_120 on the coordinator (5090 / RTX PRO 6000), sm_121 on GB10.
- `aot_sm_count`: 170 / 188 / 48.

Each has its own negative test.

### 11.10 API for long contexts (amended by §11.12: sampling defaults, tool-call cap, parser semantics)

- SSE keep-alive comments every ≤ 15 s until the first token.
- Server-side thinking default OFF (template layer).
- Qwen3-Coder-style XML tool-call parser, typed per the tool schema.
- `[think]` reasoning parser.
- Media parts return 400.
- LiteLLM `model_info` advertises `max_input_tokens` / `max_output_tokens`.

### 11.11 Config surface (adds to §8)

```text
max_context_tokens  = 1048576          # R-CTX; gate-backed, not a constant
max_output_tokens   = 65536            # default when the client omits max_tokens
admit_reserve       = 8192             # output quantum reserved at admission (grows)
prefill_chunk       = 2048 | 4096      # standalone; interleave policy per §11.4
aot_arch            = sm_120 | sm_121
aot_sm_count        = 170 | 188 | 48
host_cache_bytes    = 0 | <GiB>        # tier 1 (0 = byte-for-byte off); D2 sizes it
host_cache_store    = on-retain | streaming
spark_tier_bytes    = 0 | <GB/Spark>   # tier 2 (I8)
sse_keepalive_s     = 15
draft               = none | dflash | mtp   # one resident
draft_block         = 6 | 8
```

### 11.12 Amendments, 23 Sep 2026 09:40 AEST (tonyd2wild recipe @ `13621bb`, see ADVISOR-I3 §10)

**Drafters (§11.6).** `k` = 7 (block 8) is the default. At vLLM TP4 on 4
Sparks, `k` = 4 lost at C1 and C6 in every category except code, and never
helped prose. Adaptive `k` is an experiment cell, not a design element.
Verify blocks use position-dependent penalty state (T25).

**KV and router (§11.7):**
- FP8 KV starts with unit scales: cast to E4M3, upcast on read — vLLM's
  working path, with needles passing at 100K and 250K. That is exactly
  11,520 B/token. Add a per-layer amax clip check; add per-token scales only
  if that check or G4n fails.
- Router logits, `e_score_correction_bias` (F32 [256] in the checkpoint) and
  top-k selection all run in FP32 (T22).

**Sampling defaults (§11.10).** When a request omits sampling parameters, use
temperature 1.0 and top_p 0.95 (checkpoint), plus `repetition_penalty` 1.05.
- Never use the checkpoint's `max_new_tokens: 2048` as the default output
  limit (T23); the default is `max_output_tokens`.
- `repetition_penalty` uses a per-request seen-token bitmap (152,576 bits) on
  the GPU.

**Tool-call storm cap (§11.10, T24).** `tool_call_cap` defaults to 6 (1 when
`parallel_tool_calls` is false). At the opening of call cap+1: stop
generation, free the lane, never emit the partial call, and finish with
`finish_reason: "tool_calls"`. Client disconnect cancels generation at once.

**Parser (§11.10, T27).** Semantics match vLLM's `mimo` parser (Qwen3 parser
engine):
- REASONING is the initial state only when thinking is on;
- `[tool_call]` implicitly ends reasoning, and a duplicate `[/think]` is
  dropped;
- `[function=` is accepted without `[tool_call]`, and consecutive calls are
  accepted;
- one leading and one trailing newline are trimmed from each parameter value;
- values are coerced to the tool schema's types recursively;
- argument deltas stream.

A constrained `tool_choice` needs a `qwen_3_coder` structural-tag grammar; v1
answers such requests with 400 until it exists.

**Concurrency correctness (T26).** No asynchronous overlap may let one
request's tokens or KV bleed into another's. Enforced by gate G8, the
stress-corrupt probe.

**Spark memory.** Budget from `cudaMemGetInfo` after dropping caches, never
from MemAvailable (a GB10 reporting 116 GB MemAvailable showed ~105 GB free to
CUDA). Tier 2 is planned at ~45 GiB per Spark.

**Config additions (§11.11):**

```text
default_temperature        = 1.0     # applied only when the request omits it
default_top_p              = 0.95
default_repetition_penalty = 1.05
tool_call_cap              = 6       # 0 = off; parallel_tool_calls=false -> 1
kv_fp8_scale               = unit | per_token_head   # unit first (amax-gated)
```
