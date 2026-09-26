# Deploying the MiMo-V2.6-Flash AFD recipe

This recipe runs the attention/dense half of MiMo-V2.6-Flash on one coordinator GPU and the routed experts on four Spark ranks. It must carry to other Spark, 5090 and switch fleets, so this page separates three things:
- what the engine **requires**;
- what it **detects by itself**;
- what was **tuned on the reference setup** and is optional elsewhere.

Measured results and the step-by-step history are in `docs/design/perf-reset-vs-ds41rt.md` §6 and `runs/20260925-reset/*/RESULT.md`.

## 1. Requirements

| Part | Requirement | Reference setup |
|---|---|---|
| Coordinator GPU | NVIDIA Blackwell with ≥ 32 GB; the build bakes the arch (§2) | RTX 5090 (sm_120, 170 SMs) |
| Expert ranks | Exactly 4 (TP4 over each expert's intermediate), GB10 / sm_121 | spark1–4 (48 SMs, 128 GB unified) |
| **Inference fabric** | RoCE v2, **≥ 100 Gb/s** per port (`MIMO26_WIRE_MIN_GBPS`), from the coordinator to every rank | CX7 200G; two coordinator ports bonded (2×200G LACP), each Spark `enp1s0f1np1` |
| LAN | API clients (LiteLLM) and operations **only** | 10 GbE |
| Coordinator host RAM | Enough for the weights load; the KV host tier takes min(32, 40% of MemAvailable) GiB page-locked | 125 GB, tier 32 GiB |
| Checkpoint | `XiaomiMiMo/MiMo-V2.6-Flash-RL` on the coordinator; per-rank expert slices on each Spark | `/srv/models/…`; `/var/tmp/mimo26f-kernel/slices` |

**The fabric rule is enforced in code** (`mimo26_rdma::fabric_port`, both ends).
- An inference connection whose **local** address is not on a RoCE v2 port of at least the floor is refused.
- A Spark daemon refuses a coordinator that dialled its LAN address. A coordinator refuses to dial through its LAN interface.
- There is **no default for `MIMO26_SPARK_ADDRS`**.
- Loopback is exempt (single-host tests). `MIMO26_WIRE_ALLOW_LAN=1` is a test-only override.

## 2. Builds

**Coordinator** (the build machine needs nvcc; the arch must match the GPU):

```bash
PATH=/usr/local/cuda/bin:$PATH MIMO26F_CUDA_ARCH=sm_120 cargo build --release --offline --features cuda \
  -p mimo26-coordinator --bin mimo26-coordinator --example x1a_run
```

**Spark daemon** (build on a Spark). The baked arch, SM count and capacity class are an identity checked at boot (the AOT gate), not tuning:

```bash
PATH=$HOME/.cargo/bin:/usr/local/cuda-13.0/bin:$PATH MIMO26F_NVCC=/usr/local/cuda-13.0/bin/nvcc \
  MIMO26F_CUDA_ARCH=sm_121a MIMO26F_BAKED_ARCH=121 MIMO26F_BAKED_SMS=48 MIMO26F_CAPACITY_CLASS=2048 \
  MIMO26F_CUDA_LIB=/usr/local/cuda-13.0/lib64 cargo build --release --offline --features cuda -p mimo26-spark --bin mimo26-spark
```

## 3. Launch order

**1. Spark ranks 0–3**, one per Spark:

```bash
MIMO26_SPARK_B1=1 MIMO26_WIRE_NOCRC=1 ./mimo26-spark --rank <r> --dir <slices> --listen 0.0.0.0:8600
```

- Ready when the log says `listening`, after `boot readback: … slices match`.
- The first request after a start pays a one-time lazy expert prepare (about 25 s).
- Over ssh, launch detached: `ssh -n host "(setsid nohup … > log 2>&1 < /dev/null &)"`.

**2. Coordinator:**

```bash
MIMO26_RDMA=1 MIMO26_WIRE_NOCRC=1 MIMO26_SPARK_ADDRS=<rank0 fabric ip>:8600,<rank1>:8600,<rank2>:8600,<rank3>:8600 \
  MIMO26_WEIGHTS_DIR=<checkpoint> MIMO26_MAX_SLOTS=16 ./mimo26-coordinator
```

It is serving when the log shows `[wire] rank r …: fabric <dev> port <p> at <N> Gb/s` for all four ranks, then `serving A8 API on 0.0.0.0:8100`.

## 4. Configuration reference

### Required

| Variable | Where | Meaning |
|---|---|---|
| `MIMO26_SPARK_ADDRS` | coordinator | The four ranks' **fabric** addresses, rank order |
| `MIMO26_RDMA=1` | coordinator | RDMA RC transport (TCP remains only for tests) |
| `MIMO26_WIRE_NOCRC=1` | both | Frame CRC off on the hot path (RDMA needs it on both ends) |
| `MIMO26_SPARK_B1=1` | Spark | The B1 tensor-core expert path |
| `MIMO26_WEIGHTS_DIR` | coordinator | Checkpoint directory; the DFlash drafter loads from its `dflash/` if present |

### Common

| Variable | Default | Meaning |
|---|---|---|
| `MIMO26_API_ADDR` | `0.0.0.0:8100` | OpenAI-compatible API |
| `MIMO26_MAX_SLOTS` | 8 | Concurrent requests (the reference setup runs 16) |
| `MIMO26_SLOT_KV_TOKENS` | 4096 | Initial GA rows per extra slot (grows on demand; admission reserves the prompt) |
| `MIMO26_KV_TOKENS` | 4096 | Initial GA rows of the first slot |
| `MIMO26_PREFILL_CHUNK` | 4096 | Expert rows per lane exchange (2048 restores the old cut) |
| `MIMO26_HOST_CACHE_GB` | auto: min(32, 40% MemAvailable) | KV RAM tier (§5); 0 = off |
| `MIMO26_PREFIX_CACHE_ENTRIES` | 24 | Snapshots kept on the GPU per bank (prompt, turn) before the oldest goes to RAM (§5) |
| `MIMO26_VISION` | on | The image encoder (§5a); `0` = off, and image parts are refused with 400 |
| `MIMO26_QUEUE_DEPTH` | `MIMO26_MAX_SLOTS` | Requests waiting for a slot before callers queue behind them (§5b); the reference setup runs 64 |
| `MIMO26_QUEUE_WAIT_MS` | 25000 | How long a caller beyond the queue waits for a place before a 429 (§5b) |
| `MIMO26_PREFILL_SEGMENT_MS` | 2000 | Prefill segment target while other requests decode (×4 when nothing waits) (§5) |
| `MIMO26_DFLASH` | on if `dflash/` exists | 0 disables the drafter |
| `MIMO26_SPEC` | on with a drafter | 0 = one token per decode step |
| `MIMO26_SPEC_POLICY` / `MIMO26_SPEC_TAU` | `chain` / 0.3 | Verify drafts while the drafter's confidence chain stays at or above τ; `fixed` = verify all 7 (D7's k); `conf` = cost-model greedy (`MIMO26_SPEC_COST_A`/`_B`) |
| `MIMO26_WIRE_MIN_GBPS` | 100 | Fabric floor for inference connections |
| `MIMO26_RDMA_TIMEOUT` | 8 | QP local ACK timeout exponent (≈ 1 ms). Bounds loss recovery on a lossy fabric |

### A/B and diagnostics

Leave these unset in production:
- `MIMO26_B1_Y=bf16`: BF16 route outputs on the Sparks. A numerical-mode change, awaiting a decision; `runs/20260925-reset/p8`.
- `MIMO26_DRAFT_LM8=0`: back to the BF16 `lm_head` for the drafter. The default is an INT8 copy (+0.59 GiB of GPU memory) for draft passes of up to 8 rows; it gains C1 +0.7% (`runs/20260925-reset/kn4`). Target outputs are unchanged, since the target verifies every draft with its own BF16 `lm_head`.
- `MIMO26_B1_FC1_GROUPS=8`, `MIMO26_B1_FC2=m64`, `MIMO26_B1_FC1=m16` or `m64` (`m64`: `fc1_m64` below 256 rows too), `MIMO26_B1_FC1_STAGES`, `MIMO26_B1_CHECKED=1`.
- `MIMO26_ATTN_PREFILL=p1`, `MIMO26_ATTN_PREP=split`, `MIMO26_DECODE_LANES=1`, `MIMO26_DENSE=fp32`, `MIMO26_HOST_FORWARD`.
- `MIMO26_PROFILE`, `MIMO26_PROFILE_STAGES`, `MIMO26_SPEC_TRACE`, `MIMO26_TIMELINE`, `MIMO26_SPARK_DUMP_FRAME`.
- `MIMO26_SPARK_TRACE=1`: per-request stage lines on the Sparks (off, a rank prints one summary line a minute). Per-request logging costs a decode step about 5%, and a log of ~0.5 GB a day.
- `MIMO26_WIRE_ALLOW_LAN=1`: tests only.

## 5. Context and KV cache

**Maximum context.** At boot the coordinator computes what one slot can grow to with the pool idle, capped at 1,048,576 tokens. It logs this as `max context Some(N)`. The API refuses a prompt at or over it with **400**, before any work.

**Admission.** A request reserves `prompt + clamp(max_tokens, 1024, 8192) + 64` GA rows (11,592 B each) in one allocation, checked against free GPU memory. The check includes the growth transient (one layer's old buffer, held while the new one fills). Admission evicts retained snapshots first. When the request still does not fit, it is refused with "retry when other requests finish". A retained conversation too big to grow in place near the top of memory is relocated through RAM. All per-step scratch is sized at boot, so the free memory at boot is what stays free.

**Prefix reuse**, the DS41RT host snapshot cache v3 in its `on-evict` mode:
- **Snapshots.** Every prompt end and completion end of 512 or more tokens stays on the GPU as a snapshot inside its slot. Saving one is a device copy of 35 MB (SWA window + DFlash rings); nothing goes to RAM.
- **Resume.** A new prompt resumes at its longest exact snapshot, in this order:
  - on the GPU, in place or forked (≈ 1 ms per 100K tokens);
  - restored from RAM (≈ 42 ms per 130K tokens);
  - else prefilled cold.
- **RAM copies happen only under pressure:**
  - a bank over `MIMO26_PREFIX_CACHE_ENTRIES`;
  - no free slot;
  - not enough GPU memory for a request.
- **RAM eviction deletes:** the least recently used snapshot first; at equal use, a prompt snapshot before a turn snapshot. DS41RT deletes every prompt snapshot before any turn snapshot, but a prompt snapshot shares its pages with its turn snapshot, so under page pressure that order wiped out recent prompt snapshots while stale turns stayed (`runs/20260926-v110`).
- **Exact prefixes only.** A prompt that shares only part of a snapshot (e.g. the same system prompt with a different question) prefills cold. MiMo's SWA state cannot be rebuilt at an arbitrary position (trap T17).

Evidence: `runs/20260925-reset/{l1,k3}/RESULT.md`.

**Short prompts arriving together prefill in one pass** (up to 1,024 rows; perf reset B1), so a burst's first tokens come back together (C16 mean 0.67 s).

**Long prefills are interleaved with decode.** Prompts prefill in segments of about `MIMO26_PREFILL_SEGMENT_MS` (whole 8,192-token groups), with a decode step for the running streams between segments. A 128K prefill (about 41 s alone) stretches decoding streams' token gaps to about 4 s instead of freezing them. About 1M takes about 17.5 min on one 5090 (perf reset P3).

**Clients.**
- A streaming request whose client disconnects stops within a step. Detection is by the failed write: about 30 s during a prefill, from the keepalive writes.
- While a first token is slow, the stream carries an SSE comment (`: keepalive`) every 15 s.
- A prefill abandoned by its client is kept as a snapshot, so a retry of the same prompt resumes where it stopped.
- Thinking is off unless the request sends `chat_template_kwargs: {enable_thinking: true}`. A stream then carries the think block as `reasoning` deltas, live, and never as content.
- **Behind a proxy:** a proxy in front of the API (for example LiteLLM) may not forward the SSE keepalives. Clients going through it need a stream idle timeout longer than their longest cold prefill (about 20 minutes at 1M tokens).

## 5a. Images

Chat requests may carry images: Chat Completions `image_url` parts (and `input_image` or `image`) with inline data URLs (`data:image/png;base64,...`). PNG and JPEG are decoded; remote URLs, other formats, audio and video get a 400.

**Each image is preprocessed like the model's HF processor**, then encoded on the coordinator GPU:
- EXIF orientation; transparency composited on white.
- Resized so both sides are multiples of 32, between 3,136 and 12,845,056 pixels (Qwen2-VL `smart_resize`).
- 16-pixel patches, CLIP normalisation.
- Encoded by the checkpoint's 28-block vision transformer (`crates/mimo26-coordinator/src/vision.rs`).
- Every 2x2 patch group becomes one prompt position. A 640x480 image is 300 tokens; a 2560x1440 screenshot is 3,600.

**At most 16 images per request.** Older ones are replaced by a text note, so a long conversation keeps working.

**GPU memory.** The encoder's weights (1.46 GB BF16) stay in page-locked RAM and are copied to the GPU only while a request's images are encoded, then freed.
- An idle server's KV budget and 1M-token context are unchanged.
- Encoding needs the weights plus about 30 KB per 16-pixel patch; retained snapshots are evicted to RAM to make room.
- Loading the weights adds about 1.5 GB of page-locked RAM and a few seconds to startup.

**The prefix caches know which image a prompt holds.** An image's positions carry ids derived from its bytes (past the vocabulary), so a follow-up turn about the same image reuses the cache, and a different image never matches.

**Accuracy** (`harness/vision_ref.py` + `examples/vision_check.rs`): against the reference module in FP32, the encoder is within relative L2 2e-2, worst-token cosine 0.994. The reference itself in BF16 is at 7e-2 / 0.933.

## 5b. Sampling and the request queue

Both follow DS41RT v15.

**Greedy is the default.** A request without `temperature`, with `temperature` below 1e-5, or with `top_k` 1 takes the argmax, as every request did before V3. The checkpoint's `generation_config` has `do_sample: false`.

**With a temperature, a request samples.**
- Parameters: `temperature` (0–2), `top_p` (0–1], `top_k` (0 or -1 = off), `min_p` [0–1] and `seed`. Filters left out are off; out-of-range values get a 400.
- Filters run in vLLM's order: temperature, `min_p`, `top_k`, `top_p`. Ties at a top-k or top-p boundary are kept.
- Draws come only from the 151,675 ids the tokenizer knows, never from the lm_head's 901 padding rows.
- Kernel: `kernels/sample.cu`, one CTA per sampled row; about 0.4 ms for a whole C1 or C16 verify step on the 5090. Greedy rows pay nothing.

**Draws depend only on the seed and the token's position.**
- Batching, speculation, streaming and the caches cannot change which draw a token gets. A seeded request repeats its text as exactly as greedy decoding does (logits can move in their last bits with the batch shape, which matters only at a near-tie).
- Without a seed the server picks one.
- Speculation stays exact (DS41RT's sample-and-match): every verify row draws its own token, and a draft is accepted only while it equals that draw.

**Snapshots keep what a sampled repeat needs.**
- A prompt snapshot keeps its last logit row (0.6 MB, in host RAM, device and RAM tiers). An exact repeat of a sampled prompt draws its first token from it without a forward.
- A sampled request's turn snapshot does not record a greedy next token. A greedy exact repeat of one resumes from the longest shorter snapshot.

**The queue is bounded.**
- At most `MIMO26_QUEUE_DEPTH` requests wait for a slot. Up to as many more callers wait for a place, for at most `MIMO26_QUEUE_WAIT_MS`.
- Any other caller gets `429` with `Retry-After: 1`, before the response starts, so streams are refused cleanly too.
- A request that does not fit in GPU memory while others run waits for them (first in, first out) instead of being refused. It is refused only when it would not fit even alone.

## 6. Fabric and host tuning (optional)

The engine does not depend on any of these:
- **Switch:** 802.3x flow control on the RoCE ports. About 2–3% prefill on the reference setup. PFC or ECN would suit a lossless fabric better.
- **Coordinator CPU:** performance governor and C2 idle states off.
- **Coordinator NIC link width:** the reference coordinator's CX7 trains at PCIe x8. At x16 the four ranks' returns would land about 2× faster, and they are the long-prompt prefill limit at 4K lanes.
- **LACP hashing:** with two bonded coordinator ports, the flow hash can leave the bond unbalanced, and the balance can change at each boot; check the per-port counters.
- **Queue depth 64** (`MIMO26_QUEUE_DEPTH=64`): agent fan-outs of 30–100 requests queue instead of getting 429s. The default (the slot count, as DS41RT) turned a 60-request burst into 14–23 × 429.

## 7. Verification ladder

Run it after any deploy. All steps run against the live endpoint except X1a.

| Step | Command | Pass |
|---|---|---|
| X1a (coordinator stopped) | `MIMO26_SPARK_ADDRS=… ./x1a_run` | `RESULT: PASS` (`KESTREL-41`) |
| API contract | `harness/l5_api.py --base http://<coord>:8100` | 5 rows PASS (ISO, T24 json/stream, UTF-8 escaped/raw) |
| Ladder | `harness/l5_ladder.py --base http://<coord>:8100/v1 --cell ladder --needle-targets 8000` | G1–G5 PASS |
| Concurrency | `harness/l5_concurrent.py --base …/v1` | 8 simultaneous needles PASS |
| Prefix reuse | `harness/l5_prefix_reuse.py --base …/v1 --target 64000` | PASS (turn 2 and repeat ≥ 5× faster to first token); log shows `device hit`, no `[hostcache] store` |
| KV pressure | `harness/l5_kv_pressure.py --base …/v1 --conversations 10 --target 131072` | PASS (30/30 correct, repeats identical, returns ≥ 5× faster); log shows `device pressure` and `[hostcache] restore` |
| Batched prefill | `harness/l5_first_token.py http://<coord>:8100/v1` | PASS (burst vs solo first tokens differ in ≤ 2 of 30) |
| Head-of-line | `harness/l5_hol.py --base …/v1 --target 131072` | PASS (a decoding stream's largest gap ≤ 6 s during a 128K prefill) |
| Top of memory | `harness/l5_top_memory.py --base …/v1 --target 262144`, with `harness/tools/ballast.cu` holding the GPU nearly full (build: `nvcc -arch=sm_120 -cudart static`) | 4/4 PASS; log shows an in-place rewind and a relocation through RAM |
| Corruption | `harness/fleet/tonyd2wild/stress-corrupt.py --lane A=http://<coord>:8100 --probe both` | 0 bad lines, 0 storms |
| Images | `harness/l5_vision.py --base …/v1` | 6/6 PASS |
| Sampling and queue | `harness/l5_sampling.py --base …/v1 --burst 60` | 8/8 PASS (greedy unchanged, seeded repeats, sampled snapshot repeat, 429 with `Retry-After`); run `--burst` only while the server is otherwise idle |
| Bench of record | `harness/fleet/tonyd2wild/mimobench.py --levels 1,6,16 --prefill 2000,8000,32000,64000` | Compare with §6 of the perf-reset doc |
