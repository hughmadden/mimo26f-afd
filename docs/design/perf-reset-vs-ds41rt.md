# Performance reset: our serving path against the DS41RT reference

**Builder (Claude), 2026-09-25 07:40 AEST, at the maintainer's direction:** "compare base to the spark reference implementation, the referenced ds41rt for how things should be done and then look at which parts of this implementation are just completely wrong."

- **Reference:** tj's DS41RT as it ships, `tpurtell/ds41rt` @ `521300449`.
  - Release entry points: `run.sh:175` runs `expertd-native` (Spark); `run.sh:191` runs `serve-native` (5090 coordinator). The `real_full/*` tree is the legacy DS4 path and is not the reference.
  - The expert kernels come from SparkInfer/b12x @ `3882b935`.
- **Ours:** `mimo26f-afd` @ `21b743a`, the I5 exit engine (`ADVISOR-I5` I5-R20).
- Reference citations were read by three Opus readers and spot-checked by the builder: `run.sh:175,191`, `native/src/ds41rt_native.cc:3132-3198`, `scheduler/independent.rs:17-20`, `docs/ds41-expert-boundary-breakdown.md:5-15`.

## 0. Verdict

The gap is **architectural, not tuning**. Our serving path is the CPU oracle with GPU calls bolted onto individual ops:
- the hidden state lives in host memory in FP32;
- every dense GEMM round-trips over PCIe in pedantic FP32;
- the KV cache lives on the host and is re-uploaded in full every layer;
- one request at a time runs behind a global mutex;
- the expert exchange is a blocking TCP RPC with CPU framing;
- the Spark runs a CUDA-core FP32 kernel.

The reference does none of this. The I6 proposals (`docs/design/spark-roundtrip-i6.md`, 15–25 ms of about 250 ms per layer) polish that structure and cannot reach the target. **The fix is to adopt the reference's structure**, not to tune ours.

The per-token expert work is essentially the same in both models:
- **DS41:** 6 routes × 3 × 5,120 × 576 ≈ 53 M MAC per rank.
- **MiMo:** 8 routes × 3 × 4,096 × 512 ≈ 50 M MAC per rank.

So the reference's Spark numbers are a fair bar for ours.

## 1. Measured gap (same fleet class: 4 GB10 Sparks + one RTX coordinator)

| Metric | Ours (I5 exit) | DS41RT | Ratio |
|---|---:|---:|---:|
| Prefill, 2K prompt | 174 tok/s (`6650099`) | 2,668 tok/s best median (release report, `docs/ENGINEERING.md`); 7,726 at +16K (`docs/phase2-release-performance.md:14`) | 15–44× |
| Per-layer wall, 2K-row prefill | ~250 ms (`c6c93a0`) | Spark 2,048-row call 10.9–12.3 ms (`docs/ds41-intermediate-prefill-batches.md:47`); kernel median 9.90 ms (`docs/ds41-encoder-stream.md:60-63`) | ~20× |
| Decode, 1 stream | ~172 ms/token = 5.8 tok/s (G5, `5a6802a`) | 26.35 ms/step at 16K context (`docs/ds41-decode-lanes.md:96-97`); 60.7 tok/s with dSpark | ~6.5× before drafting |
| Expert phase, 1-row decode layer | not isolated (ours ≈ 3.6 ms per layer, all-in) | 308–328 µs (`docs/ds41-expert-boundary-breakdown.md:9-15`) | ~10× |
| Concurrency | 1 (global mutex, `api.rs:136-194`) | 16 slots in 2 lanes (`scheduler/independent.rs:56`) | — |

## 2. One 2K-row prefill layer, stage by stage

| Stage | Ours (measured: `c6c93a0`, `a6f3d50`) | Reference |
|---|---|---|
| Residual / hidden state | Host `Vec<f32>`; RMSNorm and residual adds are CPU loops (`serving.rs:466-477`) | Device-resident, with device-to-device copies between layers (`v41_block.rs:160-161,512-515`) |
| Dense projections | cuBLAS **SGEMM, pedantic FP32 (TF32 off), FP32 weights** (`gpu_dense.rs:1-3`). Every call uploads the input and downloads the output (`gpu_dense.rs:81-127`): qkv 2.5 ms, o_proj 9.5 ms | The checkpoint's **native FP8** weights through SparkInfer's MXFP8 GEMM, FP32 accumulate (`export_b12x_v41_fp8_aot.py:88-121`; `docs/ds41-fp8-qualification.md:3,7`) |
| Router | FP32 SGEMM + download + **CPU** sigmoid/bias/top-k: **12.6 ms** | GPU router kernel with top-k (`v41_router.cu:13-93`) |
| KV cache | **Host** `Vec<u8>`. Every layer of every prefill chunk downloads the new codes, then **re-uploads the whole accumulated prefix** into fresh `cudaMalloc`s (`serving.rs:309-329`) | Device-resident FP8 window ring plus a paged pool; the attention reads it in place (`v41_window.rs:87-88`; `source_cache.rs`) |
| Attention kernel | 3.2 ms on the device (fine) | ~1.95 ms (`docs/ds41-attention-head-reuse.md:41-42`) |
| Wire quantization | CPU scales + device encode + download: **15 ms** | Inside the router's CUDA graph (`v41_backbone_router.rs:386-387`) |
| Frame build | Per-row `RowDescriptor`/`RouteEntry`/`HiddenRow` structs, **cloned 4×**, encoded 4× (`wire.rs:319-364`) | One frame, memcpy'd into 4 pre-registered slots (`verbs.rs:2620-2667`); some Vec churn remains (`router.rs:706-765`) |
| Transport | **Kernel TCP** (`std::net::TcpStream`), with **OS threads spawned per layer** for send and again for receive (`wire.rs:367-415`) | **RDMA RC** SEND into pre-posted RECV slots in pinned rings registered once (`ds41rt_native.cc:3132-3198,3345-3402`); CQ polling, no threads spawned |
| Integrity | CRC32C over every frame, both directions | Off on the hot path (SHA-256 only under `DEBUG_CHECKSUM`, `request.rs:394-399`) |
| Spark receive | Blocking `read_exact` into a fresh `vec!` per frame: 8.85 ms. CPU E4M3→**f32** decode of 8.4 M elements + a 33.5 MB H2D: 16.0 ms (`decode.rs:613-691`) | NIC lands in a registered slot. The kernel reads **E4M3 as-is** (`export_b12x_v41_slices_aot.py:68-69`); a small staged H2D remains (`execution.rs:635-654`) |
| Expert GEMM | **B2 E-FP32 CUDA-core (SIMT) kernel, designed for M ≤ 8, run for prefill**: **51.8 ms** (≈3 TFLOPS effective) | b12x fused W4A8 slice kernel: `mma…mxf8f6f4.block_scale` E4M3 × E2M1, FP32 accumulate; FC1→SiLU·up→FC2 fused on-chip: **~10 ms** |
| Spark return | D2H of 16.8 MB + clone + encode + CRC + blocking `write_all`: 3.4 + 6.9 + 12.4 ms | Compaction kernel writes BF16 **directly into the mapped registered send slot** (`execution.rs:584-598`) |
| Rank reduce | CPU FP32 sum of 4 × 8.4 M elements (`CoordinatorSum`) | GPU `reduce_compact` kernel (`v41_route_reduce.cu:59-70`) |
| Overlap | **None.** Attention → RPC → wait → next layer, strictly serial; the 5090 idles while the Sparks work and vice versa | **Two lanes.** Prefill chunks alternate lanes; chunk i+1's layer-L attention waits only for chunk i's layer-L KV (`encoder_pair.rs:92-107`). Decode lanes are `tokio::join!`ed |
| Graphs / syncs | None; every op synchronous; `cudaMalloc` per decode-attention call (`serve.rs:303-318`) | Per-component per-layer CUDA graphs, captured lazily (`v41_layer_graphs.rs:1-86`) |

## 3. What is completely wrong (ordered by cost)

1. **W1: the forward is host-resident and FP32** (`serving.rs:443-515`).
   - The CPU runs embedding, both RMSNorms, both residual adds, decode-time RoPE and the KV encode.
   - The GPU is called per op, with a host round trip each time.
   - **Reference:** everything between expert boundaries stays on the device.
2. **W2: dense GEMMs run in pedantic FP32 SGEMM on FP32 weights** (`gpu_dense.rs`).
   - The weights take about 21 GB of the 5090's 32 GB, against about 5.3 GB as FP8.
   - The math runs about 5–10× slower than BF16/FP8 tensor cores.
   - Every call adds its own upload and download.
   - **Reference:** native FP8 weights on tensor cores with FP32 accumulate.
3. **W3: the KV cache is on the host and re-uploaded whole every layer.**
   - **Prefill** re-uploads the prefix every chunk, which is O(n²) PCIe (`serving.rs:319-329`).
   - **Decode** re-uploads it every token, with eight `cudaMalloc`s per call (`serve.rs:303-318`). At 32K that is about 380 MB per token; at 1M it would be about 13 GB per token. Decode can never be fast this way.
   - **Reference:** a device ring plus a paged pool, read in place.
4. **W4: one request at a time.**
   - `api.rs` holds one KV-cache set and one wire client behind mutexes, clears the cache and runs prefill and decode to completion per request.
   - With no batching, the Sparks see M ≈ 1 per expert in decode forever.
   - `scheduler.rs`, `kv_pool.rs` and `paging.rs` exist but nothing on the serving path uses them.
   - **Reference:** 16 slots in two lanes, continuous at round boundaries.
5. **W5: no overlap anywhere.** The design's core idea (ARCHITECTURE §3: "waves around the remote expert boundary") was never built (REUSE audit: "Scheduler waves: not yet").
   - **Reference:** two lanes; prefill chunk pipelining across layers; the shared expert runs during the remote wait.
6. **W6: the expert exchange is a blocking TCP RPC with CPU framing.**
   - The CPU handles the per-row structs, four clones, two CRCs, the per-layer thread spawns and the FP32 rank sum.
   - About 120 ms of the 250 ms layer is coordinator wire handling alone.
   - **Reference:** RDMA RC into registered rings, GPU quantize and GPU reduce.
7. **W7: the Spark path is wrong for prefill.**
   - It decodes E4M3 to FP32 on the CPU and ships 33.5 MB H2D.
   - It runs the decode-shaped FP32 SIMT kernel at M ≈ 64 rows per expert.
   - It encodes, clones and CRCs the return on the host.
   - **Reference:** E4M3 feeds a fused tensor-core W4A8 kernel directly, and the output is written straight into the send slot.
8. **W8: the precision policy chose the slow path.**
   - lattice-v1 §6 ("the most precise eligible lattice") made E-FP32 experts, A-f32q attention and pedantic FP32 projections the serving defaults.
   - The reference serves the checkpoint's native FP8/FP4 formats with FP32 accumulation and qualifies them semantically.
   - E-W4A8-v1 (the reference's own format) is already **CLEAR** in X1c v2 (I5-R10). Serving still uses E-FP32.
9. **W9: process.**
   - I4 spent its effort on the decode-shaped GEMM (≥ 191.1 GB/s at M ≤ 8) before a serving architecture existed.
   - The REUSE allowlist that would have brought the reference's hot path over was never executed. At I4 close:
     - verbs: not yet;
     - scheduler waves: not yet;
     - admission: not yet;
     - FP8 family: replaced by FP32 SGEMM;
     - MXFP4 family: replaced by B2 SIMT.

## 4. What is sound (keep)

- **Model semantics, all verified against the oracle and X1a:**
  - GQA + SWA-128 + sink;
  - value scale (fixed `87e708d`);
  - partial rotary with dual θ;
  - sigmoid top-8 routing with the correction bias;
  - the dense layer 0.
- **The attention kernels,** device-side: prefill about 3.2 ms per 2K layer; decode TC C3.
- **The B1 W4A8 tensor-core expert kernel** (correct end to end; about 184 GB/s at M ≤ 8, the reference kernel's class).
- The repack and resident MXFP4 images.
- The wire codec format.
- The API layer with the D1–D7 fixes and `harness/l5_api.py`.
- The CPU oracle and goldens as the correctness reference, used with tolerances rather than as the serving lattice.

## 4b. Upstream DS41RT v15 (checked 2026-09-25 10:20 AEST, at the maintainer's direction)

- **The reference moved.** `tpurtell/ds41rt` main is `b4517141` (v15), 587 commits after this fork's base `63235c6a` (2026-09-16). On one RTX:
  - **prefill 7,824–7,878 tok/s**;
  - **target-only decode ~43–45 tok/s**;
  - **weighted dSpark decode ~105 tok/s** (`docs/release-v14-notes.md`, `docs/phase2-release-performance.md`).
- **The Spark expert kernel is unchanged.** SparkInfer/b12x `w4a8_v41_slice.py` is identical between `3882b935` (the source of our B1 port, REUSE row 118) and the current pin `7fcc094`.
  - The only kernel-side change is `7fcc094e`, *plan V4.1 decode routes in one CTA* (`v41_route_plan.py`, +103 lines).
  - So upstream's speedups are orchestration, and **B1 remains the right Spark kernel**.
- **Upstream changes adopted into this plan:**
  1. **Device-ordered layer chains:** stages ordered with CUDA events, no host drain per stage. Cache producers overlap the query projections. Layers are timed with events (v14: +12–19% decode). This goes into R5.
  2. **Zero-copy Spark input:** workers copy hidden rows on their own stream from the mapped RDMA request frame (`827383a7`). This goes into R2.
  3. **Two lanes on two RoCE ports:** each lane has its own subnet and port (`ds41rt.config` `SPARK_n_LANE_A` 10.55.0.x / `LANE_B` 10.55.1.x). Our Sparks have a second 200G port (`enP2p1s0f1np1`), and the coordinator has the coordinator bond 2×200G. This goes into R2/R4.
  4. **The single-CTA decode route planner** replaces B1's serial scaffold planner for decode capacities. This goes into R3.
  5. **Local expert layers on the coordinator:** upstream runs routed experts for layers 0–4 on the RTX, saving 5 round trips per pass. This is new step R6, gated on R1b freeing 5090 memory.
- **Not adopted:** W4A4 NVFP4 experts (FP4 activations; a quality risk, not our checkpoint format), dSpark tuning, the device sampler (a later decode item), and persisting-L2 expert prefetch (upstream measured it slower).

## 5. The reset: adopt the reference's structure

**Recommendation: a hybrid.** Keep our MiMo-specific model code and kernels, and rebuild the serving orchestration on the reference's design. Transplant the reference's model-agnostic parts:
- the verbs transport;
- the expert-service structure;
- the lane scheduler shape.

Do **not** port the DS41 coordinator: its CED/mHC/Engram/compressed-KV code runs through every `v41_*` file, which is the first attempt's failure mode (ARCHITECTURE P1).

| Step | Change | Removes | Expected (model, per 2K layer unless noted) | State |
|---|---|---|---|---|
| R0 | **Use the 200G fabric.** Spark hostnames resolve to the 10 GbE addresses; use 192.0.2.1/.2/.4/.5 | 10 GbE TCP | measured prefill 145 → 193 tok/s at 4K | **DONE** `5687b1c` |
| R1a | **Device-resident coordinator forward:** device hidden, device RMSNorm/residual/RoPE/router top-8/wire quantize, device KV appended in place, persistent scratch | W1, W3 | measured ≤ 5% (the coordinator was not the wall) | **DONE** `5687b1c`, X1a PASS |
| R1b | BF16, then FP8-block, dense GEMMs with FP32 accumulate (weights 21 → 10.6 → 5.3 GB) | W2 | coordinator ~15 → ~5 ms; frees memory for R6 | **BF16 DONE** `8f64da9` (decode 85.9 → 75.0 ms/token, KL 1.4e-4); FP8-block open |
| R2 | **Expert exchange (upstream v15 design).** RC SEND/RECV into registered rings, **one lane per RoCE port**. Spark: mapped-frame zero-copy input on its stream, output straight into the send slot. Coordinator: GPU quantize, GPU reduce of the 4 planes. No CRC or threads on the hot path | W6, most of W7 | ~120 ms coordinator wire + ~45 ms Spark host → ~5 ms | **DONE, one port:** GPU rank sum + CRC off `9998cc2`; RDMA RC `4da4d9e`; pinned rings + in-place return `2abd3ad`; in-place receive, double-buffered send, ACK timeout 8 `d136065`. Lossy fabric: see §6 |
| R3 | **B1 (E-W4A8-v1) on the Spark for all M:** E4M3 input used directly; upstream's single-CTA decode planner; a parallel prefill planner | W7, W8 | FFN 51.8 → ~10–15 ms; decode 2 ms → ~0.2–0.3 ms per layer | **DONE** `9998cc2`; R3b `c424ed5` (reduce 14.9 → 1.4 ms, FC1 M64 12.7 → 4.7 ms: 2K-row rank FFN 30.8 → 9.4 ms); D2 `e18fe84` (decode FC1 depth) |
| R4 | **Scheduler:** 16 slots in 2 lanes (one per port); prefill chunks alternate lanes; batched decode | W4, W5 | prefill → Spark-bound, ~3,000+ tok/s | **Two-lane prefill DONE** `68d9d10`; multi-request slots and batched decode open |
| P2 | **Serving prefill attention**, FlashAttention-2 style: FP16 Q/P, FP32 accumulate, 64-key tiles, range-limited (not in the original plan: the P1 kernel was 14.3 ms per 4K GA lane) | W8 | GA layers stop being the wall | **DONE** `8434547`: GA 20.6 → 1.53 ms and SWA 2.0 → 0.18 ms per lane; KL 1.3e-4 |
| D1 | **Decode attention:** SWA reads only its window; splits scale with keys | — | — | **DONE** `ae1d541`: 40.7 → 36.3 ms/token |
| R5 | **Device-ordered stage chain** (CUDA events; no host drains), cache producers overlapped with the query projections, CUDA graphs for decode | residual | decode toward ~30–45 tok/s per stream (upstream target-only 43–45) | |
| R6 | **Local expert layers on the 5090** (upstream: layers 0–4), after R1b frees memory | 5 round trips per pass | ~10% per pass | |

**Throughput model,** labelled as model:
- **After R1–R3, one request, no overlap:** about 4 + 3 (wire) + 12 + 2 ≈ 21 ms per layer × 47 ≈ 1.0 s per 2K chunk, **about 2,000 tok/s (11× today)**.
- **After R4:** the Spark-bound ceiling is 2,048 / (47 × ~12 ms) ≈ **3,600 tok/s**.
- **Decode after R1–R3:** about 0.2 ms coordinator + about 0.33 ms expert phase (the reference's measured class) ≈ 0.55 ms per layer, about 26 ms per token, **about 38 tok/s single-stream**.

**Order (revised 2026-09-25 after R0/R1a measurements and the upstream check):** R3, since the Spark is now the wall for both prefill and decode. Then R2 (upstream's transport and lane layout), R1b, R4, R5, R6.

Every step keeps the working fallback, the I5 exit ladder, `l5_api.py` and X1a as regression gates, plus the tight golden at an explicit tolerance.

## 6. Measured progression (X1a 4,027 tokens; every step PASS with gen_ids identical)

| Step | Commit | 4K prefill (tok/s) | Decode (ms/token) |
|---|---|---:|---:|
| I5 exit | `d1bb9ed` | ~150 | ~168 (G5) |
| R0 200G fabric | `5687b1c` | 193 | 180 |
| R3 B1 + R2 part 1 | `9998cc2` | 660–670 | 86–93 |
| R1b BF16 | `8f64da9` | 663 | 75 |
| R2 RDMA | `4da4d9e` | 748 | 41 |
| R2 pinned + in-place return | `2abd3ad` | 815 | 42 |
| R4 two lanes | `68d9d10` | 1,114–1,208 | 41 |
| R3b reduce fix + FC1 M64 | `c424ed5` | 2,191–2,328 | 40.8 |
| P2 attention | `8434547` | 2,636–3,021 | 40.8 |
| R2d in-place receive, double send, ACK timeout 8 | `d136065` | 3,566–3,659 | 40.7 |
| D1 decode attention | `ae1d541` | 3,612–3,644 | 36.3 |
| D2 decode FC1 depth | `e18fe84` | 3,584–3,691 | 35.3 |
| D3 fused attention prep | `1105381` | 3,721 | 30.9 |
| D4 prepare-time scale scan (unchecked B1 MMAs) | `d5ba68c` | 3,948–4,106 | 28.3 |
| Switch 802.3x flow control (fabric) | `4632a7e` | 4,046–4,150 | 28.3 |

**From here on the bench of record is D7's own `mimobench` (prompt set v1, temperature 0) through the A8 API.** It runs 16 slots, reports per-stream decode, and prefills unique prefixes cold.

| Step | Commit | Prefill 2K / 8K / 32K / 64K (tok/s) | C1 per stream / C6 agg / C16 agg (tok/s) |
|---|---|---|---|
| D7 reference (vLLM TP4, DFlash k=7) | — | 2,999 / 2,975 / 2,671 / 2,114 | 71.6 / 166.0 / 264.8 |
| W4 v2, no speculative decoding (`runs/20260925-reset/compare`) | `20925e6` | 3,520 / 4,235 / 3,928 / 3,409 | 38.3 / 102.7 / 145.1 |
| **S1** DFlash on the 5090 (`s1`) | `66697d4` | 3,423 / 4,306 / 3,987 / 3,484 | **84.1 / 184.2 / 287.0** |
| **P5/P6** 4K expert chunks + mapped RDMA send (`p6`) | `f41d404` | **3,582 / 5,168 / 4,669 / 4,030** | 83.9 / 184.6 / 286.8 |
| **S2** adaptive verify length (chain 0.3); host tier + fabric guard (`s2`) | `298cab4` | 3,572 / 5,044 / 4,601 / 3,993 | **90.4 / 199.7 / 306.5** |
| **L2** quiet Spark hot path + SWA reserve + Q1b scheduler (`l2`) | `0013320` | 3,509 / 4,880–4,960 / 4,621 / 3,974 | 97.3 / 200.3 / 310.7 (9 categories; corrected) |
| **L4** + KN4 INT8 drafter head, L3 device fault reduction, packed Spark uploads (`l4`) | `f2e15a5` | 3,592 / 4,868 / 4,622 / 4,013 | **102.6 / 209.1 / 310.3** (9 categories; corrected) |
| **B1** batched prefill of short prompts (`../20260926-b1`) | `c500e2b` | (not rerun) | 102.4 / **256.8 / 395.2** (9 categories) |
| **P3** GA prefill attention, two 64-row CTAs per SM (`../20260926-p3`) | `307f1ef` | 3,485 / **5,105 / 4,799 / 4,163** | decode unchanged (B1) |
| **L5** Spark FC1 at decode sizes, `fc1_decode` (`../20260926-l5`) | `4f6f215` | (P3) | **110.0 / 274.5 / 426.2** (9 categories) |

**P6 ÷ D7:** prefill 1.19× / 1.74× / 1.75× / 1.91×; decode 1.17× / 1.11× / 1.08×.
**S2 ÷ D7:** decode **1.26× / 1.20× / 1.16×**; prefill unchanged within ~2%.
**L5 ÷ D7:** decode **1.54× / 1.65× / 1.61×**; C16 mean time to first token 0.650 s (D7 0.795).
**P3 ÷ D7:** prefill 1.16× / 1.72× / 1.80× / 1.97×; cold 128K / 256K / 512K / ~1M in 40.6 / 107.8 / 333.2 / 1,049.3 s (−7% / −10% / −13% / −15% on K3).
**B1 ÷ D7:** decode 1.43× / **1.55× / 1.49×**; C16 mean time to first token 0.667 s (D7 0.795).
**L4 ÷ D7:** decode **1.43× / 1.26× / 1.17×** (9 categories like D7; an earlier version of this line said 1.49× / 1.37× / 1.31×, from 10-category means); prefill 1.20× / 1.64× / 1.73× / 1.90×. C16 mean time to first token is 1.42 s against D7's 0.80 s.

**K1** (`0f7efae`, `runs/20260925-reset/k1`): KV host tier 1, exact prefix reuse from page-locked RAM.
- 64K follow-up turn: time to first token **16.17 → 0.266 s** (restore 63,858 tokens in 42 ms).
- Exact repeat: 0.121 s.

**K2** (`b02b199`): async host-tier copies; a 64K restore takes 22.9 ms.

**L1** (`runs/20260925-reset/l1`): KV right-sized, admission, 1,048,576-token maximum context.
- Idle GPU use 28.9 → 17.6 GB.

**K3** (`runs/20260925-reset/k3`): the DS41RT host cache design v3 `on-evict`.
- Snapshots stay on the GPU; RAM copies happen only when the device evicts. K1's per-request RAM copies are gone.
- 64K turn 2 in **0.226 s** with no RAM traffic.
- 10 × 128K under pressure: 30/30 correct, returns ≤ 0.46 s.
- 256K / 512K / ~1M cold 119 / 384 / 1,229 s, all correct. Two top-of-memory defects are open at ~1M (receipt).

**What limits each side now** (`runs/20260925-reset/p6/RESULT.md`):
- **Prefill:**
  - the return path: 4 × BF16 partials into the coordinator's CX7 at PCIe x8, then H2D;
  - the Spark FFN, which is within ~17% of its dataflow floor.
- **Decode:** each verify step is ~55 ms at 8 rows, ~31 ms of it Spark weight streaming, which runs near GB10's ~238 GB/s DRAM read rate.

The regression ladder (API cell, cell A 8K, cell B 32K) passes:
- on `d136065`: wall 71 s, against 621 s at the I5 exit;
- **on the final build (15:26): cell A 37 s, G5 C1 decode 37.1 tok/s, G4-32K 9.0 s** (was 226.7 at the exit).

Receipts: `runs/20260925-reset/ladder-20260925-{1417,1526}/RESULT.md`.

**The fabric is lossy** (`runs/20260925-reset/r2d/RESULT.md`).
- Four Spark returns converge on the coordinator's single 200G port, and the switch drops packets: coordinator logs about 4.7 K out-of-sequence packets per 4K run and sends global pause.
- No PFC or ECN is configured, by policy, and RC packet pacing is unsupported on these CX7s.
- The QP ACK timeout of 8 (about 1 ms) bounds each loss.
- **Update 15:06:** with the maintainer's approval the switch now runs 802.3x flow control on the six RoCE ports (`runs/20260925-reset/fabric-fc/RESULT.md`).
  - It honours the coordinator's pauses, and the per-layer cycle is flat at 19 ms.
  - It still sends no pause to the Sparks, so some drops remain.
  - WRED+ECN was tried and reverted: the packets are Not-ECT.
  - PFC or a pull model would remove the rest.
