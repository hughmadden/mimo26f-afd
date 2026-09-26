# Benchmarks

These numbers were measured on 26 September 2026, on the reference setup below.

- **Decode and prefill:** three benchmark runs on the v1.1.0 build. Each cell gives the median, with the range in brackets.
- **Long-context, cache and memory checks:** measured on v1.1.0 and on v1.1.1. v1.1.1 changes only the RAM tier's eviction order; see Prefix reuse. The step-by-step history, with the build behind each row, is in [docs/design/perf-reset-vs-ds41rt.md](docs/design/perf-reset-vs-ds41rt.md) §6.

## Reference setup

- **Coordinator:** one RTX 5090 (32 GB, sm_120, 170 SMs) in a host with 125 GB of RAM. The host's ConnectX-7 has two 200G ports, bonded, and trains at PCIe x8 in this host.
- **Expert ranks:** four DGX Spark (GB10, 48 SMs, 128 GB unified memory), each on one 200G ConnectX-7 port.
- **Fabric:** RoCE v2 through one 200G switch, with 802.3x flow control.
- **Engine settings:** `MIMO26_MAX_SLOTS=16`, the DFlash drafter on (chain policy, τ = 0.3), everything else at its default.

## Method

- **Tool.** Decode and prefill come from tonyd2wild's `mimobench.py`, vendored byte-for-byte in `harness/fleet/tonyd2wild/`. It uses prompt set v1 at temperature 0, through the OpenAI API: `mimobench.py --levels 1,6,16 --prefill 2000,8000,32000,64000`.
- **Decode.** Tokens per second come from the usage block. At one stream the figure is per stream; at 6 and 16 streams it is the total across streams. Each figure is the mean over the nine prompt categories. The tool's `ceiling_count` row is left out, as it is in the reference results.
- **Prefill.** Prefill prompts carry unique prefixes, so every prefill is cold.
- **Reference.** The comparison is the 4-Spark vLLM reference: vLLM with tensor parallelism across the same four Sparks, using [tonyd2wild's recipe](https://github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe) with DFlash k = 7, measured with the same tool. It has no 5090. This engine adds one, so the deltas compare two deployments, and a large part of the uplift is that GPU. No measurement separates the GPU's share from the code's: vLLM does not run this split, and this engine needs its coordinator GPU.

## Decode and time to first token

| Streams | This engine (4 Sparks + 5090) | 4-Spark vLLM reference | Over the reference |
|---|---:|---:|---:|
| 1 (tok/s per stream) | 109.7 (109.6–110.1) | 71.6 | +53% |
| 6 (tok/s total) | 279.1 (274.3–279.1) | 166.0 | +68% |
| 16 (tok/s total) | 416.7 (413.7–424.7) | 264.8 | +57% |
| 1, mean time to first token | 0.152 s | 0.251 s | 39% sooner |
| 6, mean time to first token | 0.329–0.331 s | 0.606 s | 45–46% sooner |
| 16, mean time to first token | 0.648–0.663 s | 0.795 s | 17–18% sooner |

- Each decode step verifies the drafter's tokens for as long as the drafter's confidence chain stays at or above τ.
- The verify step is bound by the Sparks' weight streaming.

## Cold prefill

| Prompt | This engine (tok/s) | 4-Spark vLLM reference (tok/s) | Over the reference |
|---|---:|---:|---:|
| 2K | 3,630 (3,391–3,632) | 2,999 | +21% |
| 8K | 5,123 (5,032–5,125) | 2,975 | +72% |
| 32K | 4,816 (4,811–4,831) | 2,671 | +80% |
| 64K | 4,283 (4,281–4,298) | 2,114 | +103% |

- **The low 2K reading:** the first 2K run after a restart is the slowest, at 3,391 tok/s.
- **The limit on long prompts:** the ranks' partial outputs arriving at the coordinator's NIC. On this host that NIC runs at PCIe x8.

## Long context: one request, cold

| Prompt (tokens) | Time to first token | Prefill rate | Answer |
|---:|---:|---:|---|
| 63,851 | 14.9 s | 4,294 tok/s | 3 of 3 codes correct |
| 130,301 | 37.6 s | 3,465 tok/s | 3 of 3 codes correct |
| 260,306 | 104.4 s | 2,492 tok/s | 3 of 3 codes correct |
| 521,347 | 327.8 s | 1,590 tok/s | 3 of 3 codes correct |
| 993,795 | 1,037.0 s | 958 tok/s | 3 of 3 codes correct |

- **Why the rate falls with length.** Attention over the global layers grows with context, and it all runs on the one GPU.
- **Other streams keep decoding.** Prefill runs in segments of about 2 s, with decode steps for the other streams in between. During a 128K prefill, a decoding stream's largest gap is 4.1 s; before segmentation it stalled for the whole 44 s.

## Prefix reuse

| Prompt (tokens) | Cold | Follow-up turn | Exact repeat |
|---:|---:|---:|---:|
| 63,851 | 14.9 s | 0.159 s | 0.041 s |
| 130,301 | 37.6 s | 0.222 s | 0.081 s |
| 260,306 | 104.4 s | 0.346 s | 0.229 s |
| 521,347 | 327.8 s | 0.593 s | 0.455 s |
| 993,795 | 1,037.0 s | 1.050 s | 0.871 s |

At about 1M tokens, most of a warm turn is spent tokenising the prompt once, about 0.7 s.

**Under memory pressure** (v1.1.1), with 10 conversations of 128K each:
- **Correctness:** 30 of 30.
- **Returns:** the longest took 0.317 s.
- **Exact repeats:** the longest took 0.147 s. All came from the RAM tier, and the outputs were identical.
- **Restores:** 42–46 ms per 130–139K tokens from the RAM tier.

**The v1.1.1 fix.** Under RAM pressure, v1.1.0's RAM tier evicted every prompt snapshot before any turn snapshot. That could drop a fresh conversation's prompt snapshot while a stale conversation's turn survived, so an exact repeat (a retry, a regenerate, an agent resending a prompt) prefilled cold: 38–41 s at 128K. v1.1.1 evicts the least recently used snapshot first.
- **Exact prefixes only.** A prompt that shares only part of a snapshot prefills cold. MiMo's sliding-window state cannot be rebuilt at an arbitrary position.

## Vision (v1.1.0)

- **Input:** PNG or JPEG as inline data URLs: OpenAI `image_url` or `input_image` parts, or Anthropic base64 image blocks through a gateway.
  - Remote URLs are not fetched, and they get a 400, as do audio and video.
  - Up to 16 images per request. Older ones are replaced by a note.
- **Encoder:** the checkpoint's own vision transformer, on the coordinator GPU. Its weights (1.46 GB) stay in page-locked host RAM and are uploaded per request.
- **Accuracy against the FP32 reference module:**
  - relative L2 error 1.5–1.7 × 10⁻²;
  - worst-token cosine similarity at least 0.99.
  - For scale, the reference module itself in BF16 is at 7–9 × 10⁻².
- **Cost:** 30–100 ms per request for the image upload and encode on the 5090. A 640×480 image is 300 tokens.
- **Checks on the release build:** `harness/l5_vision.py` passed 6 of 6 end to end, and the API contract passed 5 of 5.

## Memory

- **Maximum context:** 1,048,576 tokens. The coordinator computes it at boot from free GPU memory, capped at 2^20.
- **Global-attention KV:** 11,592 bytes per token.
- **RAM tier:** min(32 GiB, 40% of available host RAM), page-locked.

## Correctness checks, run on every build

The commands are in [docs/DEPLOY.md](docs/DEPLOY.md) §7.

- **Stress:** tonyd2wild's `stress-corrupt.py` found 0 bad lines, and the outputs were identical across runs (12 of 12).
- **Needle ladder:** G1–G5.
- **API contract:** 5 rows.
- **Concurrency:** 8 simultaneous needles.
- **Batched prefill:** first tokens match solo runs in 29 of 30 cases. The one difference is a near-tie.
- **Prefix reuse and KV pressure.**
- **Head-of-line.**
- **Top of memory:** 4 of 4 at 256K under a 9 GiB GPU ballast.
  - The exact repeat rewinds in place on the GPU in 0.23 s.
  - A 6K follow-up that cannot grow in place is relocated through RAM in 4.2 s; 260K tokens restore in 81–82 ms.
  - The repeat from RAM takes 0.26 s.
