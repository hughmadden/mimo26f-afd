# REUSE allowlist — provenance ledger

**Format:** one row per copy-in. Start empty; fill as iterations land. Anything
without a row is not allowed into the engine tree (AGENTS.md I-Allow).

| Unit | Upstream path | Upstream SHA | New path | Delta allowed | Tests that pin it | Date / who |
|---|---|---|---|---|---|---|
| _(none yet)_ | | | | | | |

### Vision input rows (perf reset V2, 26 September 2026 AEST)

Reimplemented, not copied. To match libjpeg-turbo bit for bit, the `mimo26-image` JPEG decoder follows two of its behaviours, read from a vendored source tree on the dev host:
- the block-smoothing kernel's constants, which are transcribed;
- the 16-bit wrap of the x86-64 AVX2 islow IDCT.

No libjpeg-turbo source file is in this repo. The checkpoint's vision modeling code is never copied either: `harness/vision_ref.py` executes it from the checkpoint directory at run time.

| Unit | Upstream path | Upstream SHA | New path | Delta allowed | Tests that pin it | Date / who |
|---|---|---|---|---|---|---|
| JPEG block smoothing (progressive, incompletely refined coefficients); AVX2 islow IDCT wrap | libjpeg-turbo 3.1.0 `jdcoefct.c` (`decompress_smooth_data`), `simd/x86_64/jidctint-avx2.asm`, read-only from `~/.cargo/registry/.../turbojpeg-sys-1.2.0` | libjpeg-turbo 3.1.0 | `crates/mimo26-image/src/jpeg.rs` | reimplementation in Rust; smoothing constants transcribed | `crates/mimo26-image/tests/goldens.rs` (74 JPEG goldens incl. 36 truncated progressive cuts, bit-exact vs Pillow 12.2 / libjpeg-turbo 3.1.4.1) | 2026-09-26 / image subagent + builder |

### Oracle import rows (P-201, 23 September 2026 AEST)

Upstream `code/` is **git-ignored** in the unpublished port workspace (non-git source) — these
rows pin **file digests**, not commits. Delta `none` ×12 (byte-verbatim; ARCHITECTURE
§5: no "while I'm here" cleanups). Glue rows completed with named transitive pins.

| `mimo26/__init__.py` | `port-workspace/code/mimo26/__init__.py` | `sha256:a96b5b115552195e83910a4a8b17928f23aec8f38c8edf84ca7503d428cca1f2` | `oracle/mimo26/__init__.py` | `none` | transitive: `crates/mimo26-load/tests/golden_fused_split.rs` (imports the package) — package glue | 2026-09-23 / spike-writer-3 + captain |
| `mimo26/config.py` | `port-workspace/code/mimo26/config.py` | `sha256:d72694133466b05959f88fb0be934331b7321898c582d8c6edf25e435779a6af` | `oracle/mimo26/config.py` | `none` | `spike/tests/test_p102_real_shard_geometry.py::test_real_fused_qkv_geometry_qk192_v128`; `spike/tests/test_t3_swa_window.py::test_ga_cache_config_drops_sliding_window`; `spike/tests/test_p101_addendum.py::test_f8_t11_missing_router_bias_negative` | 2026-09-23 / spike-writer-3 + captain |
| `mimo26/kv.py` | `port-workspace/code/mimo26/kv.py` | `sha256:d08d63f3a63c478ef3692f6ffd3b408e6474a13bf70efa08953bcd9d75eea1c7` | `oracle/mimo26/kv.py` | `none` | `spike/tests/test_t3_swa_window.py::{test_ga_layer_has_no_window,test_ga_window_bug_is_visible}`; `spike/tests/test_p101_addendum.py::test_f7_t8_eviction_keeps_rule_set_negative`; `spike/tests/test_t9_start_pos.py::*` | 2026-09-23 / spike-writer-3 + captain |
| `mimo26/util.py` | `port-workspace/code/mimo26/util.py` | `sha256:13e2716292cf94b1eb0d77ee91471ca06a37751652dd2a5ac8a80241a0239110` | `oracle/mimo26/util.py` | `none` | transitive: `config.py` row (load_json_lenient) — glue | 2026-09-23 / spike-writer-3 + captain |
| `mimo26/loader.py` | `port-workspace/code/mimo26/loader.py` | `sha256:20e44fbfd0af0b81de1f163c043a1f2714028f7dfcd2f67e999bb26fcd8618a2` | `oracle/mimo26/loader.py` | `none` | `spike/tests/test_p102_real_shard_geometry.py::*`; `spike/tests/test_t1_fused_qkv_split.py::*`; `spike/tests/test_p101_addendum.py::test_t2_real_pad64_half_pad_scale_row_negative`; `crates/mimo26-load/tests/{golden_fused_split.rs,t1_t2_negatives.rs}` | 2026-09-23 / spike-writer-3 + captain |
| `mimo26/checkpoint.py` | `port-workspace/code/mimo26/checkpoint.py` | `sha256:fb9d4b8a1f33ddcce573393cc21d7c5f121762929c6737576868622bd7fbee51` | `oracle/mimo26/checkpoint.py` | `none` | `spike/tests/test_p102_real_shard_geometry.py::{test_verify_stored_shapes_fail_loud,test_mtp_prefix_canonicalisation_order}`; `spike/tests/test_t1_fused_qkv_split.py::test_t2_poisoned_pad_row_never_read` | 2026-09-23 / spike-writer-3 + captain |
| `mimo26/model.py` | `port-workspace/code/mimo26/model.py` | `sha256:954b4a828050197ef3972edd3220a4e72aeb21aacbcbdb621638d02ef926f24e` | `oracle/mimo26/model.py` | `none` | `spike/tests/test_p101_addendum.py::{test_f8_*,test_f6_window_on_model_built_layers_negative}`; `spike/tests/test_t9_start_pos.py::*`; `crates/mimo26-load/tests/t1_t2_negatives.rs::{missing_required_weight_raises,missing_router_bias_raises}` | 2026-09-23 / spike-writer-3 + captain |
| `mimo26/quant/__init__.py` | `port-workspace/code/mimo26/quant/__init__.py` | `sha256:4bdb90cde3a233a9ae0b468d5c1238e4556033d6ffd3ee756de4afb8e96f4d02` | `oracle/mimo26/quant/__init__.py` | `none` | transitive: `quant/{fp8_block,mxfp4}.py` rows — package glue | 2026-09-23 / spike-writer-3 + captain |
| `mimo26/quant/fp8_block.py` | `port-workspace/code/mimo26/quant/fp8_block.py` | `sha256:d5009f779fd85613d4ee1d3cd4b399e17cd21b4644efd8a3fdb170b98c94635d` | `oracle/mimo26/quant/fp8_block.py` | `none` | `spike/tests/test_p103_codecs.py::test_e4m3_decode_table_golden_pin`; `spike/tests/test_t1_fused_qkv_split.py::{test_golden_fused_split_byte_exact,test_naive_dequant_global_scale_rows_is_wrong}`; `crates/mimo26-load/tests/golden_fused_split.rs` | 2026-09-23 / spike-writer-3 + captain |
| `mimo26/quant/mxfp4.py` | `port-workspace/code/mimo26/quant/mxfp4.py` | `sha256:22f81245773cdb327c54a15865f3d52fb221cf0ec45a89931e6297a281ce8d4c` | `oracle/mimo26/quant/mxfp4.py` | `none` | `spike/tests/test_p103_codecs.py::{test_mxfp4_codebook_golden_pin,test_mxfp4_unpack_byte_exact_golden}`; `spike/tests/test_p101_addendum.py::{test_t10_*,test_t14_nibble_order_negative}` | 2026-09-23 / spike-writer-3 + captain |
| `mimo26/nn/__init__.py` | `port-workspace/code/mimo26/nn/__init__.py` | `sha256:e752cf7cf09d247555a8114f950e5fa1be1b2f5dfb16306bc777e671d0ec40e7` | `oracle/mimo26/nn/__init__.py` | `none` | transitive: `nn/layers.py` row — package glue | 2026-09-23 / spike-writer-3 + captain |
| `mimo26/nn/layers.py` | `port-workspace/code/mimo26/nn/layers.py` | `sha256:cd6fd4a6acb2fc5c5226ac8987ec36f6e39c8e8485af2cb2edf83809430972fc` | `oracle/mimo26/nn/layers.py` | `none` | `spike/tests/test_p101_addendum.py::{test_t4_score_scale_uses_qk_dim_negative,test_c2_attn_scale_through_call_negative,test_t7_*,test_c1_sink_family_gated_negative,test_t6_sink_live_on_swa_negative}` | 2026-09-23 / spike-writer-3 + captain |
| `golden generator` | `port-workspace/code/scripts/gen-golden.py` | `sha256:ebf5e0be4bc5f8a4597c356146269c2f9c81706cba98fcb79c01ca8751dc3022` | `oracle/scripts/gen-golden.py` | `boundary-adapt` (import retarget to our byte-identical twin + `--check` mode) | `scripts/ci-cpu.sh` gen-golden `--check` (byte-compare regen) | 2026-09-23 / spike-writer-3 + captain |

### Fleet harness rows (ADVISOR-I3 §10.3, 23 September 2026 AEST)

Upstream: `github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe` @ `13621bb3cc6fd30a94d53609320599d1f1134686` (MIT, `harness/fleet/tonyd2wild/LICENSE`). Delta `none` (byte-verbatim; sha256 pinned in `harness/selftests/test_fleet_tools.py`). These are L5/L6 fleet drivers — never run by `ci-cpu`, never pointed at a model before their selftest row is complete (R6).

| Unit | Upstream path | Upstream SHA | New path | Delta allowed | Tests that pin it | Date / who |
|---|---|---|---|---|---|---|
| stress-corrupt probe (G8 corruption under concurrency, T26; tool-storm probe, T24) | `tools/stress-corrupt.py` | `13621bb` (file sha256 `d375f144…`) | `harness/fleet/tonyd2wild/stress-corrupt.py` | `none` | `harness/selftests/test_fleet_tools.py::{test_g8_*,test_t24_tool_checker_flags_malformed_calls,test_vendored_tool_is_byte_pinned}` | 2026-09-23 / advisor (Claude) |
| exact replay of captured agent bodies (T24 regression corpus) | `tools/replay_exact.py` | `13621bb` (`f5f9035b…`) | `harness/fleet/tonyd2wild/replay_exact.py` | `none` | byte pin + selftest `harness/selftests/test_fleet_bench_tools.py` (2026-09-24): canned SSE with 7 calls (two packed in one chunk, one late name) → counted 7, stream and non-stream; body replayed exactly | 2026-09-23 / advisor (Claude) |
| bench of record (prompt set v1, C1–C6, cold prefill; comparable to the measured TP4 bar) | `bench/mimobench.py` | `13621bb` (`9c2537b1…`) | `harness/fleet/tonyd2wild/mimobench.py` | `none` | byte pin + selftest `harness/selftests/test_fleet_bench_tools.py` (2026-09-24): fake SSE server: usage-block token counts, TTFT at first content delta, true concurrency with unique tags, /metrics acceptance deltas, fixed prompt set | 2026-09-23 / advisor (Claude) |
| needle ladder (G4: filler + code at depths 0.1/0.5/0.9) | `tests/mimo_needle.py` | `13621bb` (`5281a81d…`) | `harness/fleet/tonyd2wild/mimo_needle.py` | `none` | byte pin + selftest `harness/selftests/test_fleet_bench_tools.py` (2026-09-24): deterministic prompt per size/depth (needle at depth ±0.03), PASS/FAIL parse; sends model id `mimo-v2.6-flash`, which the A8 API must accept | 2026-09-23 / advisor (Claude) |
| tool-call cap reference (semantics for the native `tool_call_cap`, T24; not deployed) | `tools/toolcap-proxy.cjs` | `13621bb` (`8b621ed9…`) | `harness/fleet/tonyd2wild/toolcap-proxy.cjs` | `none` (reference only) | byte pin; the engine-side cap gets its own I5 tests | 2026-09-23 / advisor (Claude) |

## Row rules

- **Upstream path** is the real file/crate (`tpurtell/ds41rt`, CPU
  twin under `port-workspace/code/`, or other named
  source).
- **Upstream SHA** is the git commit of the source tree at copy time (or file
  digest for non-git sources).
- **Delta allowed:** `none` / `names-only` / `boundary-adapt` / `family-retarget`.
  "Cleanup" is not a delta class.
- **Tests that pin it** must exist before the row is complete — golden, property,
  or L4 case ID from `TEST-PLAN.md`.

## Planned rows (from ARCHITECTURE §5)

| Unit | Planned action | Target iteration | Actual at I4 close (builder, 24 Sep 2026; scan `15af89a`) |
|---|---|---|---|
| Wire codec (`protocol_v2.rs`) | COPY | I3 | **DERIVED, not copied**: `crates/mimo26-wire`, under the "Wire codec (DS41RTE3 v3) format derivation" source row; no code body copied |
| Verbs/RoCE + TCP | COPY | I3 | **not yet**: the network transport is I5 Track S (ADVISOR-I5 §3.1); a copy-in gets a row when it lands |
| MXFP4 pack/GEMM family | COPY family | I3 | **MIXED**: B1 via the registered hedge rows (above); B2 `expert_gemm.cu` and repack `crates/mimo26-repack` in-project (re-derived from `spike/mxfp4.py`) |
| FP8 block-128 family | COPY family | I2 | **in-project**: `crates/mimo26-load/src/e4m3.rs`, ported from the project spike `spike/quant.py` (internal) |
| Draft traits (`DraftChain`/`VerificationTarget`) | COPY traits | I5 | **not yet**: drafters are I6/I7 (ITERATION P-300) |
| AOT SM gate + bake | COPY | I2 | **in-project**: `crates/mimo26-expert/src/aot.rs` (no copy-in marker) |
| Doctor / api-smoke | COPY | I0–I1 | **not present in tree**: preflight via `scripts/preflight.sh`; api-smoke is due with the I5 A8 API |
| API/admission shape | COPY shape | I4 | **not yet**: I5 A1/A8 (ADVISOR-I5 §3.1) |
| Scheduler waves shape | COPY shape | I4 | **not yet**: I5 A4 (ADVISOR-I5 §3.1) |
| Placement planner | PORT from CPU twin | I4 | **not yet**: I5 TP4 placement, Spark side (ADVISOR-I5 A1) |
| Hostcache | DEFER | I7 | **deferred** (I5b/I7) |
| CPU twin `mimo26/` + goldens | IMPORT (oracle/) | I1 | **IMPORTED**: P-201 oracle import rows plus the golden generator row (above) |

## Source pools (the maintainer, 2026-09-22: source of code, tests, and know-how)

The sister repos inventoried in `HANDOFF.md` are **sanctioned sources**: we copy
code and tests out of them and take know-how freely (cite path + SHA). We never
write to them (I-Green). A borrowed test, golden corpus, or script is a copy-in
unit like code — it gets a row here. Know-how (facts, configs, proven fixes)
needs no row but gets a citation. The AGENTS.md never-enter list still binds
regardless of source.

## Source pins (verified 22 September 2026 AEST)

| Source | Path | SHA / tag | Notes |
|---|---|---|---|
| ds41rt-afd-native | `tpurtell/ds41rt` | `52130044944dffd437336afe375bae98311b46fe` | copy source (primary) |
| ds41rt engine configs | `ds41rt-persistence` + `codex/afd-*` worktrees | parent `68b11a27414dc9767bf2aa14a8e8343cc17db99a`; worktree head at copy time | code + tests + know-how (esp. `graph-retention-r88`, `plan-g-r89`, hostcache lanes) |
| b12x engine configs | `b12x-native-aot.git` worktrees | `2683868f…` (`wob-split1`), `4d0e4094…` (`wob-m16-split2`) | split-K determinism know-how |
| First attempt (code + tests) | `mimo26-flash-tj` (+ `-mimo-only` worktree) | `8710e8af358bd6bfc3d6a3056cd4f8941e53206a` / `8d706d900339e3dcbb6064e2b697145cf82b63f2` | `mimo26-*` crates, test patterns, goldens |
| CPU twin | `port-workspace/code/mimo26` | file digest at copy time (untracked in the unpublished port workspace @ `b200cb32…`) | oracle import (I-Gold) |
| Golden corpus | `.../code/tests/golden/` (`mxfp4_golden.json`, `fp8_block_golden.json`, `e4m3_decode_table.json`) | file digest at copy time | byte pins (I-Gold) |
| Xiaomi modeling ref | HF `XiaomiMiMo/MiMo-V2.6-Flash-RL` `modeling_mimo_v2.py` (the coordinator copy) | repo pin `3b38d063180c3e4aed9691fdc735f3d10b266ee4` | external oracle |
| tonyd2wild 2x-Spark patches | `.../code/vendor/tonyd2wild-mimo26-2x-spark/patches/` | repo as of 22 Sep 2026 | T1/T3/T4/T5/T12 fixes (see COHERENCE-TRAPS) |
| vllm-afd evidence | `vllm-afd-workspace/` (see REPO-MAP) | port workspace `b200cb32…` | transport/reduction know-how |
| Wire codec (DS41RTE3 v3) format derivation | ds41rt-v10 `ds41rt-transport` `protocol_v2.rs` (`:6-16` magic/v3/lengths, `:34-37` checksum concept, `:45-60` dtype codes, `:122-128` source kinds, `:261-292` header/row/route structs, `:580-620` row/route layouts) + `request.rs:364-410` (frame sectioning) — via PORT-SURFACE row `ds41rt-transport protocol_v2` | port workspace `b200cb32…` (READ-ONLY) | **clean-room re-implementation, no verbatim bodies** (`crates/mimo26-wire`). Deltas: unified 128-B header, L4 seq+CRC32C tail (SHA-256 debug checksum not copied), frame-level identity fields, bare BF16 return rows, MiMo constants 4096/8/256/4. I4 items 5-6. |
| LaneSim design shape | ds41rt `mimo26/afd/stubs.py::LaneSim` + `tests/test_afd_stubs.py` lane-simulation section (via PORT-SURFACE) | port workspace `b200cb32…` (READ-ONLY) | **no code bodies copied**; shape ported to `crates/mimo26-lanesim` (`src/fault.rs::LaneFaults`, `tests/negatives.rs::lane_faults_shape_ported_from_ds41rt`). Deltas: splitmix64 instead of numpy RNG, `dispatch` returns 0..=2 copies (duplicates first-class), `duplicate_rate` knob added, virtual clock. I4 item 7. |

### B1 expert hedge rows (builder approval, 2026-09-23 20:05 AEST)

Source note: `docs/design/expert-b1-hedge.md` §3 (`bcca0c4`). **R/** = `https://github.com/tpurtell/ds41rt`
@ `e2a6f2ca5af56b8b567fb5086f82597427f28477`; **S/** = `https://github.com/tpurtell/sparkinfer-glmrt.git` @
`3882b935ede761d6c73a5d6fd68e690f1e3f5380` (https://github.com/local-inference-lab/b12x). Approved here: the four
lattice-independent units (prepare, staging, group plan, route-reduce scaffold). **Held pending the P-LATTICE ruling**
(`runs/20260923-i4/packets/builder-p-lattice.md`): E2M1 register-container transform, block-scaled MMA primitive,
FC1/FC2 slice schedule. Translation into CUDA is a copy-in under these rows; the "Delta allowed" column binds.

| Unit | Upstream path | Upstream SHA | New path | Delta allowed | Tests that pin it | Date / who |
|---|---|---|---|---|---|---|
| Lossless N256/K128 prepare transform | R/`native/cuda/kernels/v41_expert_pack.cu:8–43,52–100` | `e2a6f2ca5af56b8b567fb5086f82597427f28477`; file SHA256 `c72c6cee143fba9522b7485573eeb2889d49eef848a3b7b809bb7434edfbe23e` | `crates/mimo26-expert/kernels/b1/prepare.cu` | H4096/I512; v2 region binding; separately tagged prepared output; up/gate order explicit; no quantization/padded640; preserve bounds/overlap rejection | proposed `b1_prepare_roundtrip`:all36 real rank images +manifest-v1 refusal +inverse-coordinate/poison tests | 2026-09-23 / lead proposal (`bcca0c4`); **builder approved 2026-09-23 20:05 AEST** |
| Exact-width staging/addressing | S/`b12x/moe/_shared/kernels/w4a8_staging.py:13–146` | `3882b935ede761d6c73a5d6fd68e690f1e3f5380`; file SHA256 `ab95db03abdc06152ab47878cfa4e2e7b7bb19f450516d76ad472f04f1e147f0` | `crates/mimo26-expert/kernels/b1/staging.cuh` | Translate to handwritten CUDA; bind tagged prepared strides; preserve64-bit offsets/copy-completion rules; new width128 tail where needed; do not reinterpret row-major bytes | proposed `b1_stage_bounds`:N/K boundary, tail, sparse/inactive/poison +sanitizer | 2026-09-23 / lead proposal (`bcca0c4`); **builder approved 2026-09-23 20:05 AEST** |
| Stable group/inverse-map and ordered slice-reduction algorithm | S/`b12x/moe/_shared/kernels/v41_route_plan.py:14–165` | same S commit; file SHA256 `f5ee61e9e0e3c8cbbf0c6aa50af74be57d33d32b88b09f69debe93c996175da4` | `crates/mimo26-expert/kernels/b1/group_plan.cu` | C++/CUDA implementation, E256/top8/H4096, fault-on-invalid, MiMo validated IDs/offsets; bounded scratch; preserve stable association; no hidden capacity→atomic heuristic | proposed `b1_plan_replay`:sparse IDs, inactive, tails,M64 split,changed graph rows,invalid IDs,capacity classes | 2026-09-23 / lead proposal (`bcca0c4`); **builder approved 2026-09-23 20:05 AEST** |
| Local route compaction / BF16 RNE scaffold only | R/`native/cuda/kernels/v41_route_reduce.cu:36–55,73–97` | same R commit; file SHA256 `49fcb3270485505bfe57e91bc55e3eba005f68efb1c0b8b5d014f487095b3466` | `crates/mimo26-expert/kernels/b1/route_reduce.cu` | H4096/top8; unweighted raw input, apply route weight once; expose FP32 pre-sum before explicit BF16 wire cast; no shared expert or per-route rank-rounding | proposed weighted8-route `ReturnRow`/`CoordinatorSum` identity; weight-once negative; separate adopted wire bound | 2026-09-23 / lead proposal (`bcca0c4`); **builder approved 2026-09-23 20:05 AEST** |

#### Released 2026-09-23 20:34 AEST after the P-LATTICE ruling (ADVISOR-I4 §9 R5)

| Unit | Upstream path | Upstream SHA | New path | Delta allowed | Tests that pin it | Date / who |
|---|---|---|---|---|---|---|
| E2M1 register-container transform | S/`b12x/_lib/intrinsics.py:4847–4881` (optional relabel `:4884–4924`) | `3882b935ede761d6c73a5d6fd68e690f1e3f5380`; file SHA256 `9357389a24880c8a17d2b8718df9268f55cd79b58ad7802c910f0dc77dc1ee28` | `crates/mimo26-expert/kernels/b1/mxfp4_ptx.cuh` | Extract only named PTX sequence into CUDA wrapper; preserve bit convention; use MiMo's scale/special handling, no entire intrinsics module | proposed `b1_container_k1`:4096 code/scale pairs +55,296 real bits +nibble negatives | 2026-09-23 / lead proposal (`bcca0c4`); **builder approved 2026-09-23 20:34 AEST for E-W4A8-v1 only** (ADVISOR-I4 §9 R5; deltas per `docs/design/lattice-v1.md` §3–4: no FC1-output BF16 cast, no DS clamps, route weight once after FC2, quantizer v1 without BF16 pre-round, no atomics) |
| Block-scaled MMA primitive, conditional numeric-mode arm | S/`b12x/_lib/intrinsics.py:4059–4120` | same S commit and intrinsics file hash | same `b1/mxfp4_ptx.cuh` | CUDA inline-PTX wrapper only; preserve opcode/operand/scale-byte mapping; no silent FP32→FP8 cast; **hold until mode decision** | proposed `b1_mma_fragment`; scale-byte-ID, signed-zero and asymmetric K32 cases; separate lattice and FP32 parity gates | 2026-09-23 / lead proposal (`bcca0c4`); **builder approved 2026-09-23 20:34 AEST for E-W4A8-v1 only** (ADVISOR-I4 §9 R5; deltas per `docs/design/lattice-v1.md` §3–4: no FC1-output BF16 cast, no DS clamps, route weight once after FC2, quantizer v1 without BF16 pre-round, no atomics) |
| FC1/FC2 slice schedule, not DS activation | S/`b12x/moe/_shared/kernels/w4a8_v41_slice.py:48–256,290–364` | same S commit; file SHA256 `4fbaacaeea6791dcb03b41a2b4e2a99725f98ede8c4c09e67aa4e350068145d7` | `crates/mimo26-expert/kernels/b1/grouped.cu` | H4096/I512/E256; M16 groups,64/128 baseline, optional192+128tail; MiMo ABI; remove clamps/pre-FC2 route weight/implicit rounding; ordered output; numerical implementation explicitly approved first | existing real TP4 identity extended to B1, proposed clamp/route-weight/lattice negatives and M64 tiled cases | 2026-09-23 / lead proposal (`bcca0c4`); **builder approved 2026-09-23 20:34 AEST for E-W4A8-v1 only** (ADVISOR-I4 §9 R5; deltas per `docs/design/lattice-v1.md` §3–4: no FC1-output BF16 cast, no DS clamps, route weight once after FC2, quantizer v1 without BF16 pre-round, no atomics) |

### System libraries — CUDA runtime/cuBLAS FFI (24 September 2026 AEST)

NVIDIA CUDA toolkit system libraries, linked through FFI declarations only (no
source copied). The CUDA runtime (`libcudart`) row covers the `mimo26-attn`
device layer; the cuBLAS row covers the coordinator GPU dense path (I5-R8a).

| Unit | Upstream | Version | New path | Delta allowed | Tests that pin it | Date / who |
|---|---|---|---|---|---|---|
| cuBLAS SGEMM (`cublasSgemm_v2`) | NVIDIA CUDA toolkit (`/usr/local/cuda/lib64/libcublas.so`) | 12.8 | `crates/mimo26-coordinator/src/cublas.rs` (FFI declarations + `CUBLAS_PEDANTIC_MATH` TF32-off) | FFI only — declarations + `-lcublas` link, no source | `examples/dense_golden.rs` (`sgemm_nt` vs the CPU `linear` twin, accumulation-order only, verified on the coordinator) | 2026-09-24 / attn-lead |
