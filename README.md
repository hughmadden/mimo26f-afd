# mimo26f-afd

An inference engine for [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) that runs the model across **one RTX 5090 and four NVIDIA DGX Spark (GB10) systems**, connected by RoCE v2 RDMA.

The GPU runs attention and everything outside the routed experts. The four Sparks run the experts. This split is attention–FFN disaggregation (AFD).

On the reference setup it serves an OpenAI-compatible API, with text and image input, and a context of up to 1,048,576 tokens. Compared with the 4-Spark vLLM reference (tensor-parallel across the same four Sparks, without the 5090), it decodes at 110 tok/s on one stream (+53%) and 417 tok/s across sixteen (+57%), and prefills at up to 5,123 tok/s (+72%; +103% at 64K) ([BENCHMARKS.md](BENCHMARKS.md)). The uplift is the added RTX 5090 and the engine together: the GPU takes attention, the KV cache and the drafter off the Sparks, and the engine is written to make that split pay.

It is written in Rust and handwritten CUDA, with no external Rust crates. It is a research engine and has been tested on one hardware setup (see [Status](#status)).

## How it works

- **Coordinator (RTX 5090).** It runs the embeddings, attention (GQA; sliding-window and global layers with attention sinks; FP8 KV cache), the dense layers, the FP32 router, sampling, and DFlash speculative decoding with the checkpoint's own drafter. It serves `/v1/chat/completions` (streaming and tool calls) and `/v1/models`.
- **Expert ranks (four DGX Sparks).** Each Spark holds one quarter of every routed expert, split tensor-parallel over the expert's intermediate dimension (TP4). The weights stay in the checkpoint's MXFP4 format. A W4A8 tensor-core kernel (B1) computes the expert FFN.
- **Fabric.** For each layer, the coordinator sends the routed tokens to all four ranks and sums the partial outputs they return. Frames use DS41RT's `DS41RTE3` v3 format over RDMA RC queue pairs. Both ends refuse inference traffic on anything but a RoCE v2 port of at least 100 Gb/s. The LAN carries only the API.
- **KV cache.**
  - Each request's KV grows on demand inside one GPU pool.
  - Admission reserves the prompt plus 1K–8K output rows (from `max_tokens`). A prompt at or over the maximum context is refused with a 400.
  - Prompt and turn ends are kept as snapshots. A follow-up with an exact prefix resumes from the GPU copy. When memory is short, it resumes from a page-locked RAM tier (up to 32 GiB) instead.
  - A 64K follow-up turn starts in 0.16 s. Cold, the same prompt takes 15 s.
- **Serving.**
  - Continuous batching over `MIMO26_MAX_SLOTS` requests (16 on the reference setup).
  - Short prompts arriving together prefill in one pass.
  - Long prefills run in segments of about 2 s, with decode steps for the other streams in between.
  - **A bounded request queue (v1.2.0).**
    - Up to `MIMO26_QUEUE_DEPTH` requests (default: the slot count) wait up to `MIMO26_QUEUE_WAIT_MS` (25 s). Beyond that, the API answers 429 with `Retry-After: 1` before the response starts.
    - A request that doesn't fit in GPU memory waits for running ones instead of being refused.
- **Sampling (v1.2.0).**
  - Greedy is the default.
  - `temperature`, `top_p`, `top_k`, `min_p` and `seed` follow DS41RT v15's contract, with filters applied in vLLM's order.
  - Each draw is a function of the seed and the token's position, so batching, caching and speculative decoding leave a seeded request's output unchanged.
- **Vision (v1.1.0).**
  - Images arrive as PNG or JPEG in inline data URLs. A dependency-free crate decodes and preprocesses them, matching Pillow bit for bit.
  - The checkpoint's own vision encoder runs on the coordinator. Its weights stay in page-locked host RAM and are uploaded per request, so the KV budget and the 1,048,576-token context are unchanged.
  - The language model, the Sparks and the drafter are unchanged.

## Results

These were measured on the reference setup with tonyd2wild's `mimobench.py` (prompt set v1, temperature 0), vendored in `harness/fleet/tonyd2wild/`. The comparison is the 4-Spark vLLM reference: vLLM TP4 on the same Sparks, without the 5090, using [tonyd2wild's recipe](https://github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe) with DFlash k=7. The deltas therefore compare two deployments, and a large part of the uplift is the fifth device. Decode figures are means over the nine prompt categories. Each cell is the median of three runs on the v1.1.0 build. With greedy decoding, v1.2.0's regression battery matches. Sampled prose decodes 3–6% slower than greedy at one stream and 7–14% slower at 16, because the drafter's guesses are accepted less often.

| | This engine (4 Sparks + 5090) | 4-Spark vLLM reference | Over the reference |
|---|---:|---:|---:|
| Decode, 1 stream (tok/s per stream) | 109.7 | 71.6 | +53% |
| Decode, 6 streams (tok/s total) | 279.1 | 166.0 | +68% |
| Decode, 16 streams (tok/s total) | 416.7 | 264.8 | +57% |
| Time to first token, 16 streams (mean) | 0.65–0.66 s | 0.80 s | 17–18% sooner |
| Cold prefill 2K / 8K / 32K / 64K (tok/s) | 3,630 / 5,123 / 4,816 / 4,283 | 2,999 / 2,975 / 2,671 / 2,114 | +21 / +72 / +80 / +103% |

A 130K-token prompt prefills in 38 s, and a 993,795-token prompt in 17.3 minutes. Both answer correctly. [BENCHMARKS.md](BENCHMARKS.md) has the method, the long-context and cache results, and the correctness checks.

## Requirements

| Part | Requirement | Reference setup |
|---|---|---|
| Coordinator GPU | NVIDIA Blackwell, ≥ 32 GB. The build bakes the SM architecture and count. | RTX 5090 (sm_120, 170 SMs) |
| Expert ranks | Exactly four DGX Spark (GB10, sm_121) | 48 SMs, 128 GB unified memory each |
| Inference fabric | RoCE v2, ≥ 100 Gb/s per port, from the coordinator to every rank | ConnectX-7 200G through one switch |
| LAN | API clients and operations only | 10 GbE |
| Coordinator host RAM | Enough to load the weights; the KV RAM tier takes min(32 GiB, 40% of available) | 125 GB |
| Software | CUDA 12.8+ (coordinator), CUDA 13.0 (Sparks), rdma-core (libibverbs), Rust stable | Rust 1.98 |
| Checkpoint | `XiaomiMiMo/MiMo-V2.6-Flash-RL` with its `dflash/` drafter, on the coordinator and on each Spark (to stage that rank's expert slices) | |

## Quick start

[docs/DEPLOY.md](docs/DEPLOY.md) is the full procedure.

1. **Build** the coordinator on a machine with nvcc, and the rank daemon on a Spark. The commands are in DEPLOY §2.
2. **Stage the expert slices** on each Spark. This is a byte permutation of the checkpoint's MXFP4 tensors, plus a sha256 manifest:
   ```bash
   cargo build --release -p mimo26-repack
   ./target/release/mimo26-repack --rank <0-3> --checkpoint <checkpoint dir> --out <slice dir>
   ```
3. **Start the four ranks, then the coordinator** (DEPLOY §3):
   ```bash
   # on each Spark
   MIMO26_SPARK_B1=1 MIMO26_WIRE_NOCRC=1 ./mimo26-spark --rank <r> --dir <slice dir> --listen 0.0.0.0:8600
   # on the coordinator host
   MIMO26_RDMA=1 MIMO26_WIRE_NOCRC=1 MIMO26_SPARK_ADDRS=<rank0 fabric ip>:8600,<rank1>:8600,<rank2>:8600,<rank3>:8600 \
     MIMO26_WEIGHTS_DIR=<checkpoint dir> MIMO26_MAX_SLOTS=16 ./mimo26-coordinator
   ```
4. **Verify** with the ladder in DEPLOY §7: API contract, needles, concurrency, prefix reuse, KV pressure, top of memory, and corruption stress.

## Tests without the hardware

```bash
cargo test --workspace                                        # Rust (CPU; GPU cells are separate)
python3 -m pytest oracle/tests harness/selftests spike/tests  # numpy + pytest; torch-only tests skip
python3 oracle/scripts/gen-golden.py --check                  # golden vectors reproduce byte-for-byte
```

`scripts/ci-cpu.sh` runs all three. `oracle/` is a CPU reference of the model's numerics. `oracle/goldens/` holds its pinned golden vectors. The GPU test cells live under `crates/*/tests/gpu/` and are driven through `scripts/dev.sh test <cell>`.

## Repository layout

```text
crates/
  mimo26-coordinator   coordinator binary: model forward, scheduler, KV cache and snapshots, DFlash, expert-wire client
  mimo26-api           OpenAI-compatible HTTP API and the MiMo tool-call parser
  mimo26-attn          attention kernels (GQA + SWA + sink), paged KV, split-KV decode, AOT gates
  mimo26-expert        MXFP4 expert GEMM kernels (B1) and their reference harness
  mimo26-spark         the Spark expert-rank daemon
  mimo26-wire          DS41RTE3 v3 frames and CRC32C
  mimo26-rdma          RDMA RC transport (rdma-core verbs)
  mimo26-repack        expert slice staging for the ranks
  mimo26-load          checkpoint loading (fused QKV, scale grids) and name audit
  mimo26-lanesim       discrete-event simulator of the expert path
  mimo26-image         image decoding and preprocessing for vision input (PNG/JPEG, Pillow-exact resize)
oracle/     CPU reference of the model's numerics, and the golden vectors
spike/      Python reference implementation used to validate the engine on real weights
harness/    serving checks, vendored bench tools, selftests
bench/      fixtures and a performance model (AFD vs TP4)
configs/    build pins (build.env.example)
scripts/    dev.sh (build and test verbs), build_spark.sh, ci-cpu.sh, preflight.sh
docs/       DEPLOY.md, design notes, papers, REUSE.md (provenance of copied code)
```

## Documents

- [docs/DEPLOY.md](docs/DEPLOY.md): requirements, builds, launch, configuration, the KV cache, tuning, and the verification ladder.
- [ARCHITECTURE.md](ARCHITECTURE.md): the design contract. [TEST-PLAN.md](TEST-PLAN.md): the test pyramid.
- [docs/design/perf-reset-vs-ds41rt.md](docs/design/perf-reset-vs-ds41rt.md): the measured progression, step by step.
- [docs/COHERENCE-TRAPS.md](docs/COHERENCE-TRAPS.md): model-specific traps that produce wrong output.
- [docs/REUSE.md](docs/REUSE.md): the source and licence of every copied unit.

The documents are the project's working record, so read them with three things in mind:

- They cite internal planning notes (`ADVISOR-I3/I4/I5`, `HANDOFF`, `AGENTS`) and measurement receipts under `runs/`. None of these are in this repository.
- Hosts are named by role:
  - `coordinator` is the RTX 5090 host;
  - `spark1`–`spark4` are the ranks;
  - the "dev host" is a development machine with an RTX 4090.
- The GPU test drivers reach remote hosts as `ssh coordinator` and `ssh sparkN`. Define those names in `~/.ssh/config`.

## Status

- **Serving:** exactly four Spark ranks and one coordinator GPU, with text and image input (v1.1.0) but no audio or video. Tested on the reference setup only.
- **Designed but not built:** other topologies (2, 3, 5 or 6 Sparks) and larger coordinator GPUs.
- **Correctness gates:**
  - against the CPU oracle and golden vectors;
  - end to end with needle, stress, API-contract and top-of-memory checks.
- **Numerics:** the expert path runs W4A8 (MXFP4 weights, E4M3 activations).

## Credits

- **[DS41RT](https://github.com/tpurtell/ds41rt)** by T.J. Purtell ([@wrldsuksgo2mars](https://x.com/wrldsuksgo2mars)): the AFD design this engine follows, the wire format, the expert prepare and route-reduce units, and the v15 sampling contract and request queue (MIT).
- **[b12x](https://github.com/local-inference-lab/b12x)** by the b12x authors: the block-scaled MMA primitive and the W4A8 slice schedule behind the B1 kernel (Apache-2.0).
- **[tonyd2wild's MiMo-V2.6-Flash DGX Spark recipe](https://github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe)** by Tech2wild ([@Tech2Wild](https://x.com/Tech2Wild)): the vLLM TP4 reference and the vendored bench and stress tools (MIT).
- **[@majewskizby](https://x.com/majewskizby)**: the 8-bit drafter-head idea, from the [DeepSeek-V4.1-Flash four-Spark TP4 recipe](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4).
- **The Xiaomi MiMo team**: the model and its drafters.

## License

MIT; see [LICENSE](LICENSE). Third-party components keep their own licences, listed in [NOTICE.md](NOTICE.md). The model weights are not part of this repository. They are distributed by Xiaomi under the model's own licence.
