# Benchmarks

These numbers were measured on 26 September 2026, on the v1 build, on the reference setup below. They will be re-measured for each release tag. The step-by-step history, with the build behind each row, is in [docs/design/perf-reset-vs-ds41rt.md](docs/design/perf-reset-vs-ds41rt.md) §6.

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
| 1 (tok/s per stream) | 110.0 | 71.6 | +54% |
| 6 (tok/s total) | 274.5 | 166.0 | +65% |
| 16 (tok/s total) | 426.2 | 264.8 | +61% |
| 1, mean time to first token | 0.155 s | 0.251 s | 38% sooner |
| 6, mean time to first token | 0.332 s | 0.606 s | 45% sooner |
| 16, mean time to first token | 0.650 s | 0.795 s | 18% sooner |

- Each decode step verifies the drafter's tokens for as long as the drafter's confidence chain stays at or above τ.
- The verify step is bound by the Sparks' weight streaming.

## Cold prefill

| Prompt | This engine (tok/s) | 4-Spark vLLM reference (tok/s) | Over the reference |
|---|---:|---:|---:|
| 2K | 3,485 | 2,999 | +16% |
| 8K | 5,105 | 2,975 | +72% |
| 32K | 4,799 | 2,671 | +80% |
| 64K | 4,163 | 2,114 | +97% |

On long prompts, the ranks' partial outputs arriving at the coordinator's NIC set the limit. On this host that NIC runs at PCIe x8.

## Long context: one request, cold

| Prompt (tokens) | Time to first token | Prefill rate | Answer |
|---:|---:|---:|---|
| 130,279 | 40.6 s | 3,209 tok/s | 3 of 3 codes correct |
| 260,284 | 107.8 s | 2,415 tok/s | 3 of 3 codes correct |
| 521,325 | 333.2 s | 1,565 tok/s | 3 of 3 codes correct |
| 993,795 | 1,049.3 s | 947 tok/s | 3 of 3 codes correct |

- **Why the rate falls with length.** Attention over the global layers grows with context, and it all runs on the one GPU.
- **Other streams keep decoding.** Prefill runs in segments of about 2 s, with decode steps for the other streams in between. During a 128K prefill, a decoding stream's largest gap is about 4 s; before segmentation it stalled for the whole 44 s.

## Prefix reuse

| Case | Time to first token |
|---|---:|
| 64K prompt, cold | 15.0 s |
| 64K follow-up turn (exact prefix, GPU snapshot) | 0.165 s |
| 64K exact repeat | 0.042 s |
| 128K follow-up turn | 0.28 s |
| 993,795-token prompt, follow-up turn | 1.04 s |
| 993,795-token prompt, exact repeat | 0.87 s |
| 10 conversations of 128K under memory pressure | 30 of 30 correct, returns ≤ 0.46 s |

- **Restores.** A restore from the RAM tier takes about 42 ms per 130K tokens.
- **Exact prefixes only.** A prompt that shares only part of a snapshot prefills cold. MiMo's sliding-window state cannot be rebuilt at an arbitrary position.

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
- **Top of memory:** 4 of 4, at 256K under a GPU ballast and at about 1M.
