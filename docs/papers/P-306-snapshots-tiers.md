# P-306 — Snapshot anatomy & KV tier protocol (A2/A3)

**Packet:** I3-P306 (ADVISOR-I3 §3 A2/A3; ITERATION "P-300 revision", I3 row).
**Status:** design paper for external review. Code lands at I5b (tier 1 +
exact reuse) and I8 (tier 2) — ITERATION "P-300 revision". "P-306 reviewed" is
an I3 exit criterion (ADVISOR-I3 §4, I3 row).
**Date:** 23 September 2026 AEST (Sydney).
**Scope:** what a retained snapshot is, how it moves between device, the coordinator host
RAM and Spark memory, and the exact-prefix-reuse rule for a hybrid-SWA model.

## 0. Provenance policy (read this before any number)

Every number below is labelled **model** or **measured**. **model** = checkpoint
geometry or `bench/model/afd_vs_tp4_model.py` / ADVISOR-I3 §3 arithmetic;
**measured** = carries its receipt. Model numbers never promote a configuration
(R-MODEL, ADVISOR-I3 §1). No stub number sizes anything here; the hostcache
v3 copy-latency estimate marked "to be measured"
(`afd-hostcache-design.md:15`) would carry the 3–5× discount if quoted
(I-Hon, AGENTS.md §2).

**Never-enter list binds this paper** (AGENTS.md §3). One extra rule for this
packet: sibling-repo sources are **cited, never copied**. Code copy-in of the
hostcache crate is a later packet (I5b) and needs its `docs/REUSE.md` row first
(I-Allow). The first attempt's `plan/placement.py` pool math stays quarantined
(T21, P-305 §6).

## 1. Tier order and the no-tax contract

**Device → tier 1 (the coordinator pinned host RAM, write-behind) → tier 2 (Spark memory
over RDMA) → recompute** (ADVISOR-I3 §3 A2; ARCHITECTURE §11.2). Every tier
follows host-cache design v3 R1–R11
(unpublished `dsv41-flash-tp4-engram/research/afd-hostcache-design.md` §2
:20-48), summarised:

| Rule | Content (v3 wording, trimmed) | Cite |
|---|---|---|
| R1 | Zero overhead when off (`0` = byte-for-byte current paths) | afd-hostcache-design.md:20-21 |
| R2 | Zero overhead until pressure when on: no request waits on the cache; async copies on a dedicated stream; **acceptance = prefill ladder + C1 decode within 1% of cache-off** | :22-26 |
| R3 | Restore instead of recompute via the engine's own fill path — **MiMo replaces v3's targets** (§4) | :27-30 |
| R4 | Same reuse rule: the host cache answers with the engine's own retention radix; prompt and turn banks keep separate identities and tie-break | :31-33 |
| R5 | Bounded pinned memory, allocated once at boot; LRU within bank order (prompts before turns); never evict a snapshot with a restore in flight | :34-36 |
| R6 | Exact sharing: pages immutable once written; COW-shared pages stored once; no hashing | :37-39 |
| R7 | Single-threaded, event-driven on the scheduler thread; GPU copies async, completion polled per tick | :40-41 |
| R8 | Everything configurable, logged at boot, metrics exported always | :42-43 |
| R9 | Lost on restart is acceptable | :44 |
| R10 | Testable without the fleet: host-only crate, stub copy engine, virtual clock | :45-46 |
| R11 | Upstreamable: hook points + a copy engine; engine types used, not copied | :47-48 |

**R-NOTAX gate (binding):** cache on vs off within **≤1%** on the prefill ladder
and C1 decode, and no request ever waits on the cache until there is pressure
(ADVISOR-I3 §1 R-NOTAX, §5). `host_cache_bytes = 0` is a byte-for-byte off
switch (ARCHITECTURE §11.11).

## 2. Snapshot anatomy

Unit of copy and of sharing: **the sealed GA page**. All values **model**
(checkpoint geometry; `bench/model/afd_vs_tp4_model.py:170-176`;
ARCHITECTURE §11.2 table :291-296). "Scales" below are zero under today's
unit-scale FP8 KV (ADVISOR-I3 §10.4.2); the columns exist so a per-token-scale
fallback (T20) is a constant change, not a redesign.

| Part | Bytes | Identity / capture rule |
|---|---|---|
| Sealed GA page (256 tokens) | **2,949,120 + scales** | 256 × 9 layers × 4 KV heads × 320 B FP8. Immutable once sealed; COW-shared across requests; written behind **once**, mapped `(page id, generation)` → host page (hostcache v3 §4.2 :77-84; `DevicePageId { compressor: u8, page: u32, generation: u32 }`, ds41rt-hostcache
`snapshot.rs:18-28` on `hostcache/rc6`; the `compressor` bank is what
distinguishes reused page indices — see §8 for its fate on MiMo). |
| 39 SWA ring states | **12,779,520 + scales** | 39 × 128 × 8 heads × 320 B. Captured **at an exact position** (§5). One ring state per layer per slot. |
| Drafter state, DFlash | **20,971,520** (BF16) | Context K/V ring, 1024 × 5 × 8 × 512 B (K 128 + V 128, BF16). FP8 halves it (10.5 MB). |
| Drafter state, MTP | **983,040** | 3 × 128 × 8 × 320 B. Only the **resident** drafter is ever captured (ARCHITECTURE §11.6). |
| MTP seed hidden state | **8,192** | 4,096 × 2 B BF16 (fold N3, ADVISOR-I4 §2): MTP's first layer consumes the target's final hidden state at the last position — without it the first speculative step after a restore has nothing to draft from. Exactness is unaffected (greedy verification is target-exact); acceptance is. DFlash needs no extra state (its context ring is the projected context). |
| Logit row | **305,152** (BF16) | 152,576 vocab × 2 B; the first token after restore. |

**Snapshot tail totals (model):** ~**34 MB** with DFlash
(12,779,520 + 20,971,520 + 305,152 = 34,056,192 B; ADVISOR-I3 §3 A2 "about
34 MB") or ~**14.1 MB** with MTP (derived from the same rows; +8,192 B seed
hidden state per fold N3 ≈ 14.11 MB). Pages are shared;
the tail is per snapshot. Each optional in-prompt checkpoint therefore costs one
tail (ADVISOR-I3 §3 A3).

Cited for anatomy, not copied: ds41rt's equivalents at its 890 B/token size are
documented in `afd-hostcache-design.md` §3 (:52-58) and pinned as crate
constants (`ds41rt-hostcache` `lib.rs:33-57` on `hostcache/rc6`:
`KV_BYTES_PER_TOKEN = 890` :45,:69, `TAIL_BYTES = 2,720,064` :52-53,:70).
The MiMo port changes exactly the anatomy constants — `KV_BYTES_PER_TOKEN`
890 → 11,520, `TAIL_BYTES` → ~34 MB, compressor-page constants → the GA page
(ADVISOR-I3 §3 A2) — and keeps the structures.

## 3. Store modes

- **`on-retain`** — the v3 default: pages copied behind at retention, so an
  eviction later drops a clean snapshot for free
  (afd-hostcache-design.md §4.3 :87-93).
- **`streaming write-behind`** — new for MiMo (ADVISOR-I3 §3 A2): each GA page
  is copied to host **as it seals**, on the store stream, rate-limited. Traffic
  is **~85 MB/s even at 7.4k tok/s prefill** — **model** (7,400 tok/s ×
  11,520 B/token = 85.2 MB/s; `bench/model/afd_vs_tp4_model.py:195-196`) —
  a rounding error next to even the degraded Gen1 link. A long session is thus
  already in RAM when it retires or is preempted (this is what makes P-305's
  pressure-ladder step 2 cost ~37 MB instead of a full session).

Config: `host_cache_store = on-retain | streaming` (ARCHITECTURE §11.11).
Both modes ride the store stream and never block prefill or decode (R2,
afd-hostcache-design.md:23-24).

## 4. Restore protocol and budgets

Mechanics (ADVISOR-I3 §3 A2; v3 §4.3 :102-109): H2D copies into device pages
reserved for the restore; the tail goes into ring slots; the rebuilt snapshot
enters the device bank and the lookup proceeds as a device hit. Restores are
**budgeted**: on timeout the reservation is cancelled and the request **falls
through to prefill** (v3 behaviour, ds41rt commit `5173e44`, cited at
ADVISOR-I3 §3 A2; v3's `--host-cache-restore-budget-ms` default 500 ms,
afd-hostcache-design.md:105-106, is a ds41rt-size constant to re-derive).

**Targets (MiMo, replacing v3 R3's 890 B/token assumption — ADVISOR-I3 §3
A2):** all **model**:

| Case | Bytes | Target | Arithmetic check |
|---|---|---|---|
| 170K tokens | 2.0 GB (170,000 × 11,520 B + tail) | **≤ 100 ms** | ~50–80 ms at 25–40 GB/s Gen5 |
| 1M tokens | 12.1 GB | **≤ 600 ms** | ~303–484 ms at 25–40 GB/s Gen5 |
| 1M from tier 2 | 12.1 GB | **~0.4–0.6 s** | 20–31 GB/s through the coordinator inbound |

(`bench/model/afd_vs_tp4_model.py:182-189` runs exactly these link cases.)

**Gen1-PCIe caveat (must ship with every restore number):** the 5090 link reads
**Gen1 x16 today** (coordinator host notes defect 1; ADVISOR-I3 §2 — **measured** probe,
may be idle downclocking; verify under load, D1 window). At Gen1 (~3–4 GB/s) a
1M restore takes **3–4 s** instead of ≤600 ms — **model**
(bench/model/afd_vs_tp4_model.py:183). The I5b exit allows "or a documented
PCIe cause" for exactly this reason (ITERATION "P-300 revision", I5b row).

**Why the budget is generous anyway:** recomputing a 1M-token prompt takes
**20–40 min** on the 5090 — **model** (ADVISOR-I3 §3 A2). Even the Gen1 restore
is ~500× faster than the recompute it replaces. Restores only ever replace
recompute (ADVISOR-I3 §3 A2 no-tax rules); they never preempt live work.

**Tier-1 capacity (model):** 48 GiB pinned = **4.5M tokens**; 64 GiB = 6.0M
(ADVISOR-I3 §3 A2). Sized by D2 (§7).

## 5. Exact prefix reuse for hybrid SWA (A3)

**Rule: restore only at an exact snapshot at or before the divergence point,
then recompute the rest** (ARCHITECTURE §11.3; R-EXACT, ADVISOR-I3 §1:
token-exact against a cold compute at temperature 0).

**Empty-window SWA replay is forbidden (trap T17).** Replaying 128 tokens from
an empty window rebuilds the ring's *size*, not its *contents*: the replayed
rows are computed with a truncated window at all 39 SWA layers, and their GA KV
then carries that error for the rest of the session — subtle drift, restore ≠
cold compute (COHERENCE-TRAPS §6 T17; ADVISOR-I3 §3 A3). Two concrete refusals:

1. ds41rt's replay is built on DeepSeek's layer-19/20 encoder/decoder split,
   which MiMo does not have — visible in code as `begin_replay` asserting
   `layer >= 20` and `begin_encoder_replay` asserting `layer < 20`
   (mimo26-flash-tj `v41_window.rs:131-141`) and the 128-token
   `restore_encoder_continuation` rule (`prefix.rs:275-278`). Cited as the
   mechanism **not** being ported; MiMo's restore path may copy
   the `restore` shape (prefix.rs:248-285) but never its replay rule.
2. The first attempt planned to reuse `REPLAY_WINDOW_TOKENS = 128` unchanged.
   Do not (ADVISOR-I3 §3 A3).

**Retain points — two banks (prompt end, completion end)**, as in ds41rt
(prompt and turn banks keep separate identities and tie-break: R4,
afd-hostcache-design.md:31-33; `Saved` retained in
`v41_native_serve/prefix.rs`, v3 §3 :52). Optionally an in-prompt checkpoint
every 32K tokens for very long prompts — each costs one 34 MB tail, pages
shared (ADVISOR-I3 §3 A3).

**Why the prompt-end bank matters (thinking-on multi-turn):** the chat template
re-renders earlier assistant turns as `` thinking{reasoning_content or ''} /thinking``
(`chat_template.jinja`, sha256 `853650be…`, ADVISOR-I3 §9), so a client that
drops `reasoning_content` diverges right after the previous
`` /thinking``. The prompt-end snapshot is exactly the right restore
point: only the last turn's content and the new messages are recomputed
(ADVISOR-I3 §3 A3).

**Gates (I5b exit, ITERATION "P-300 revision"; ADVISOR-I3 §3 A3, §5):**
L4 tiny weights: restore → continue ≡ cold compute at temperature 0. L5 real
weights: same at 32K. A thinking-on multi-turn trace must hit the prompt-end
snapshot. B-tier cell: restore 170K and 1M from tier 1 and tier 2 — latency,
bytes, then continue-exactness (ADVISOR-I3 §5).

## 6. Tier 2 protocol (Spark memory over RDMA) — lands at I8

All **model** (ADVISOR-I3 §3 A2; §10.4.3 narrows capacity). Open decision D3
governs whether this happens at all (§7).

- **Capacity:** ~50–60 GB usable per Spark after the 40.2 GB expert slice,
  workspace and the 8 GiB MemAvailable floor → **17–21M tokens across four**
  (ADVISOR-I3 §3 A2). §10.4.3 narrows to **4 × 45 GiB ≈ 16.8M tokens** from the
  measured TP4 memory facts (42.5 GiB weights + 46.69 GiB KV per rank;
  **measured**, ADVISOR-I3 §10.1/:701-704). Budget from `cudaMemGetInfo` after
  dropping caches, never from MemAvailable (ADVISOR-I3 §10.4.3).
- **Format:** the same page format as tier 1. `CopyEngine` gains an **RDMA
  backend**: a pinned region on each Spark, pages striped across all four
  (ADVISOR-I3 §3 A2). The trait seam exists (ds41rt-hostcache `copy.rs:36`
  `CopyEngine`, `copy.rs:182` `StubCopyEngine`, `hostcache/rc6` — cited for
  anatomy; the RDMA backend is new code).
- **Restores** are striped across the Sparks and staged through a **pinned
  bounce buffer on the coordinator** into the 5090, **unless GPUDirect is qualified**
  (ADVISOR-I3 §3 A2). GPUDirect qualification is an open item (§7).
- **No-tax rules for demotions** (the tier-2 direction is the dangerous one):
  tier-2 stores are demotions from the tier-1 LRU, **rate-limited on the
  lowest-priority QP** and **paused during prefill bursts** — the coordinator inbound
  direction already carries expert returns at 1.54 MB/token
  (**model**, ADVISOR-I3 §3 A2, §3 A6). Restores only ever replace recompute.
- **Integrity (fold N4, ADVISOR-I4 §2):** every tier-2 page transfer carries a
  CRC32C and its `(page, generation)` in the transfer header, verified on
  restore — pages can be corrupted in transit or placed out of order, and a
  corrupted page must never silently poison the rest of the session. Mismatch
  falls through to recompute and bumps a counter. Same detection ladder as the
  expert wire (L4 bitflip/reorder).
- 1M from tier 2 ≈ **0.4–0.6 s** (**model**, §4 table).

## 7. Open questions (named, not papered over)

| # | Question | Owner | State in this paper |
|---|---|---|---|
| 1 | **D2 — tier-1 pinned budget.** Proposal: **48 GiB of the coordinator's 125 GiB** = 4.5M tokens (64 GiB = 6.0M) — **model** (ADVISOR-I3 §7 D2). | the maintainer | Carried as a **proposal**; `host_cache_bytes` stays 0 until D2 lands. |
| 2 | **D3 — tier 2 at all**, and whether the Sparks will be dedicated to MiMo (production `ds41-flash` experts live there today) | the maintainer | §6 is design-ready but gated on D3. |
| 3 | GPUDirect RDMA GPU-to-GPU qualification on this fabric | engineering | Bounce-buffer staging is the default until qualified (§6). |
| 4 | PCIe link: Gen1 readback today (defect 1) — is it idle downclocking? | D1 window | Every restore number in §4 carries the caveat. |
| 5 | Restore budget constant (`restore-budget-ms`) re-derived for MiMo | I5b | v3's 500 ms is ds41rt-size (afd-hostcache-design.md:105-106). |
| 6 | FP8-KV acceptance (D5): if per-token scales return (T20), every page/GB figure grows by the scale layout | the maintainer (D5) | Unit-scale assumed (ADVISOR-I3 §10.4.2). |

## 8. Crate port map (cite now, copy at I5b with a REUSE row)

`ds41rt-hostcache` on branch `hostcache/rc6` of `ds41rt-persistence` —
cited read-only (ADVISOR-I3 §3 A2 "Reuse, not rewrite"; I-Allow):

| Unit | Cite | Port delta (ADVISOR-I3 §3 A2) |
|---|---|---|
| Module map: config, pool, copy, snapshot, cache, metrics, sim | `lib.rs:18-29` | Keep |
| Source-page / KV constants | `lib.rs:33-45` | Compressor pages → GA page (2,949,120 B); `KV_BYTES_PER_TOKEN` 890 → 11,520 |
| Tail constants | `lib.rs:46-57` | `TAIL_BYTES` 2,720,064 → ~34 MB (SWA rings + resident drafter + logit row) |
| Page identity + exact sharing: `DevicePageId { compressor, page, generation }`, `PageRef` | `snapshot.rs:18-35` | Keep the generation semantics exactly. `compressor: u8` **retires**: it disambiguated reused page indices across ds41rt's four compressor banks, while MiMo's GA pages live in one flat paged pool where `(page, generation)` alone is a unique identity. |
| `CopyEngine` trait + stub | `copy.rs:36,182` | Add RDMA backend (§6) |
| Pinned slab pool, four size classes, O(1) alloc | `pool.rs:1-5,44-60` | Class sizes follow the MiMo anatomy |
| Simulator + suites (R10) | `sim.rs` | Keep; drive with MiMo constants |

Every copied unit gets a `docs/REUSE.md` row before it is useful in a build
(I-Allow, AGENTS.md §2). This paper deliberately contains **no code**.

## 9. Claim → citation map

| Claim | Label | Citation |
|---|---|---|
| Tier order device → t1 → t2 → recompute | policy | ADVISOR-I3 §3 A2; ARCHITECTURE §11.2 |
| R1–R11 no-tax contract | policy | afd-hostcache-design.md §2 :20-48 |
| ≤1% tax gate (prefill ladder + C1 decode, on vs off) | policy / gate | ADVISOR-I3 §1 R-NOTAX, §5; afd-hostcache-design.md:22-26 |
| GA page 256 tokens = 2,949,120 B + scales, COW-shared, (page id, generation) | model | bench/model/afd_vs_tp4_model.py:170-171; ARCHITECTURE §11.2; afd-hostcache-design.md §4.2 :77-84; ds41rt-hostcache snapshot.rs:18-28 |
| 39 SWA rings 12,779,520 B + scales at an exact position | model | bench/model/afd_vs_tp4_model.py:172; ARCHITECTURE §11.2 |
| DFlash 20,971,520 / MTP 983,040; logit row 305,152 | model | bench/model/afd_vs_tp4_model.py:173-175; ARCHITECTURE §11.2 |
| Tail ~34 MB (DFlash) / ~14.1 MB (MTP) | model | ADVISOR-I3 §3 A2; arithmetic from ARCHITECTURE §11.2 rows |
| Streaming write-behind ~85 MB/s at 7.4k tok/s | model | bench/model/afd_vs_tp4_model.py:195-196; ADVISOR-I3 §3 A2 |
| Restore 170K ≤ 100 ms (2.0 GB), 1M ≤ 600 ms (12.1 GB) | model (targets) | ADVISOR-I3 §3 A2 (replaces v3 R3, afd-hostcache-design.md:27-30) |
| Gen1 caveat: 1M = 3–4 s at Gen1 | model + measured link state | bench/model/afd_vs_tp4_model.py:183; ADVISOR-I3 §2 (coordinator host notes defect 1) |
| 1M recompute 20–40 min | model | ADVISOR-I3 §3 A2, §4 |
| Timeout falls through to prefill | measured code (ds41rt `5173e44`) | ADVISOR-I3 §3 A2; afd-hostcache-design.md §4.3 :102-109 |
| Tier-1 48 GiB = 4.5M / 64 GiB = 6.0M tokens | model | ADVISOR-I3 §3 A2, §7 D2 |
| Tier-2 17–21M (narrowed 16.8M) tokens; 1M restore 0.4–0.6 s | model | ADVISOR-I3 §3 A2, §10.4.3 (measured inputs §10.1) |
| Demotions: rate-limited, lowest-priority QP, paused in prefill bursts; expert returns 1.54 MB/token | policy / model | ADVISOR-I3 §3 A2, §3 A6 |
| Striped restores via pinned bounce (GPUDirect pending) | policy | ADVISOR-I3 §3 A2 |
| T17: empty-window SWA replay forbidden; error enters GA KV permanently | trap | COHERENCE-TRAPS §6 T17; ADVISOR-I3 §3 A3 |
| ds41rt replay rests on layer-19/20 split (not ported) | measured code | mimo26-flash-tj v41_window.rs:131-141; prefix.rs:275-278 |
| Restore only at exact snapshots; recompute the rest | policy | ARCHITECTURE §11.3; ADVISOR-I3 §1 R-EXACT |
| Two banks at prompt end / completion end | policy | ARCHITECTURE §11.3; afd-hostcache-design.md:31-33, §3 :52 |
| Prompt-end bank for thinking-on multi-turn (` /thinking` divergence) | model (template fact) | ADVISOR-I3 §3 A3; chat_template.jinja sha256 `853650be…` (ADVISOR-I3 §9) |
| In-prompt checkpoint every 32K, one tail each | model | ADVISOR-I3 §3 A2/A3 |
| Hostcache crate anatomy + port deltas | measured code (cite-only) | ds41rt-hostcache lib.rs:18-57, snapshot.rs:18-35, copy.rs:36,182, pool.rs:1-60 (`hostcache/rc6`); ADVISOR-I3 §3 A2 |
| I5b/I8 landing + gates | plan | ITERATION "P-300 revision"; ADVISOR-I3 §4, §5 |

**Uncited residue:** the design judgments themselves (bounce-buffer default
over unqualified GPUDirect; demotion pausing policy) are engineering arguments
traceable to ADVISOR-I3 §3 A2, whose inputs are cited above. Nothing numeric is
uncited.

## Revision 23 Sep 2026 AEST: corrections of record after adversarial review (F1–F7)

The external review REFUSE was narrow and corrections-only: design, discipline
and every key figure passed (~85 citations spot-checked); the review itself is
the receipt. This paper takes F2, F3 and F5 (F1, F4, F6 and F7 landed in
P-305). All fixes verified against source before editing.

- **F2 (MED), §2 anatomy — derivation expression off by 2×.** "1024 × 5 × 8 ×
  256 B" evaluates to 10,485,760, half of the stated 20,971,520 (the missing
  factor 2 is BF16). Corrected to "1024 × 5 × 8 × 512 B (K 128 + V 128, BF16)",
  matching P-305 §2's derivation. Verified: 1024 × 5 × 8 × 512 = 20,971,520.
- **F3 (MED), §2 + §8 + §9 map — `DevicePageId` struct misdescribed and
  mis-cited.** The real struct is
  `DevicePageId { compressor: u8, page: u32, generation: u32 }`
  (ds41rt-hostcache `snapshot.rs:18-28` on `hostcache/rc6`) — three fields, not
  two; the `compressor` bank is what distinguishes reused page indices. Struct
  text and all line cites fixed (19-27/19-35 → 18-28/18-35). The §8 port-map
  row now states the field's fate: `compressor: u8` retires, because MiMo's GA
  pages live in one flat paged pool where `(page, generation)` alone is unique;
  generation semantics are kept exactly.
- **F5 (LOW), §5 — `restore` conflated with `restore_prefix`.** `prefix.rs`
  248-285 is fn **`restore`**, which *calls* `restore_prefix` at :280;
  `restore_prefix` proper is `v41_requests.rs:210` (grep-verified). Reworded to
  "may copy the `restore` shape (prefix.rs:248-285) but never its replay rule".
