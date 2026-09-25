# COHERENCE-TRAPS — known gibberish / word-salad causes

**22 September 2026 AEST.** Online chatter about MiMo-V2.6-Flash producing
gibberish is almost always a **load-path / attention-layout** bug, not a broken
checkpoint. We already have the fixes from the reference Spark deployment
(**tonyd2wild/MiMo-V2.6-Flash-2x-DGX-Spark**, vendored at
`port-workspace/code/vendor/tonyd2wild-mimo26-2x-spark/`)
plus our header-verified research.

**Spike rule:** the I1 bare e2e is not allowed to call its output "coherent"
unless every trap in §1 is either fixed or explicitly listed as out-of-scope.

---

## 1. Trap checklist (apply before any coherence claim)

| ID | Trap | Symptom | Fix (source) | Spike must |
|---|---|---|---|---|
| **T1** | Fused-QKV **ckpt_tp = 4 on every layer** (not 4 GA / 8 SWA) | loads fine, **word salad**; scrambled Q/K/V | tonyd2wild patch 01 + README: "a naive fix scrambled Q/K/V and produced word salad"; vLLM#57508; real headers: SWA scale grid 116=4×29 not 120 | **must fix** (negative test required) |
| **T2** | Per-shard FP8 **scale-grid padding** trim | garbage / silent wrong scales | Vontra MLX note: "loads but returns broken output"; our `split_shard_major_fused` | **must fix** |
| **T3** | Full-attention layers must **not** inherit SWA `sliding_window` (128) | long-context incoherence; GA layers windowed | tonyd2wild patch 01 (`cache_config.sliding_window = None` for GA) | **must fix** |
| **T4** | **QK head_dim 192 ≠ V head_dim 128** (no broadcast) | shape-OK garbage or crash | tonyd2wild DiffKV backend + our kernel specs | **must fix** in attention path |
| **T5** | **attention_value_scale 0.707** on V (target) / 0.612 on DFlash (query **and** context KV-injection) | wrong magnitude; soft failures | tonyd2wild patch 04 (vLLM#57784); our ARCHITECTURE §5 | **must fix** on target path (DFlash = later) |
| **T6** | **SWA sink bias per-Q-head [64]** as extra logit column | attention mass wrong → rambling/repeat | Xiaomi `modeling_mimo_v2.py` + first-attempt REVERSAL of record | **must fix** |
| **T7** | Partial rotary **0.334 → 64 dims**, dual θ (10M GA / 10k SWA) | position/repeat failures | Xiaomi ref + tonyd2wild `swa_rope_theta` | **must fix** |
| **T8** | SWA **eviction** `min(batch_pos) − window + 1` (not "keep last window") | incremental ≠ full; drift after 128 tok | first-attempt KV invariant | decode step must use correct ring |
| **T9** | `start_pos` ignored → every step at position 0 | silent garbage after step 1 | PERF-CORRECTNESS finding 4 | **must fix** in spike loop |
| **T10** | E8M0 scale clamp to 1.0 | saturated experts → word salad on MoE tokens | PERF-CORRECTNESS finding 1 | if experts in spike |
| **T11** | Missing `e_score_correction_bias` / wrong norm name | KeyError or silent bias drop | findings 2–3 | fail-loud loader |
| **T12** | `dflash/config.json` **trailing comma** / broken JSON | loader crash or partial parse | tonyd2wild `dflash-config.fixed.json` + our `load_json_lenient` | later (drafter) |
| **T13** | Thinking **ON** leaks reasoning into `content` | looks like incoherent/duplicated answers at the API | tonyd2wild serve default `enable_thinking: false` | API layer (not spike) |
| **T14** | MXFP4 **nibble order** (even-low vs U2×4) | expert garbage | our goldens + hardware-proof fixture | if experts in spike |

## 2. Reference fixes inventory (tonyd2wild 2x Spark)

| Artifact | What it fixes |
|---|---|
| `patches/01-mimo_v2-fused-qkv-chunks-and-fp8-kv.patch` | T1 + T3 (+ cites ckpt_tp model-wide 4) |
| `patches/03-triton_attn_diffkv-fp8-kv.patch` | T4 (K/V head-dim split) + FP8 KV on that path |
| `patches/04-qwen3_dflash-value-scale.patch` | T5 DFlash V-scale on query + context injection |
| `patches/02-mimo_v2_omni-supports-eagle3.patch` | drafter interface (Eagle3/DFlash hook) |
| `patches/files/dflash-config.fixed.json` | T12 well-formed dflash config |
| `patches/files/mimo_v2.py`, `triton_attn_diffkv.py`, `qwen3_dflash.py` | drop-in reference implementations |
| README operational notes | T13 thinking-off default; restart MemAvailable wait; marlin MoE; DeepGEMM off |

Vendor path:
`port-workspace/code/vendor/tonyd2wild-mimo26-2x-spark/`
Digest: `.../research/tonyd2wild-mimo26-dflash-2x-spark.md`.

## 3. Our prior HIGH fixes (already in the CPU twin — do not regress)

1. E8M0 clamp (T10) + regenerated goldens.
2. Router bias real name `mlp.gate.e_score_correction_bias` (T11).
3. Backbone `post_attention_layernorm` vs MTP `pre_mlp_layernorm` (T11).
4. `start_pos` default (T9).
5. Fused-QKV negative test (T1/T2) — **keep and port to the spike**.

## 4. Spike coherence bar (I1)

A spike run is **COHERENT** only if:

1. Trap checklist §1 T1–T11 either fixed in the spike path or explicitly waived
   in the run receipt (waive only if that path is not used, e.g. no experts).
2. At least three prompts produce non-salad continuations at greedy:
   - factual short ("The capital of France is")
   - arithmetic ("17*23 =")
   - continuation of a fixed sentence
3. Output tokens are recorded raw in `runs/`; a human can read them.
4. If salad: stop and localize (loader vs attention vs rope vs sink) — do not
   "tune" sampling to hide it (T13-class mistake).

## 5. Not this document

Performance, acceptance rates, drafters, multi-GPU scheduling. See TEST-PLAN /
ITERATION. Online HF discussion threads currently show little beyond eval PRs;
the **production word-salad evidence is the tonyd2wild patch series**, which is
authoritative enough to gate on.

---

## 6. Addendum, 23 September 2026 AEST: T15–T21 (architectural advisor)

Evidence is in [`docs/ADVISOR-I3.md`](ADVISOR-I3.md) §3 and §9. Each trap
needs a negative test in the iteration that first touches its path.

| ID | Trap | Symptom | Fix (source) | Must |
|---|---|---|---|---|
| **T15** | DFlash conditioning is not "target-KV injection". The real path is `target_hidden = hidden_norm(fc(concat(h[l+1] for l in [0,11,23,35,47])))` with `fc.weight [4096, 20480]`. Each drafter layer then applies its own `k_proj`/`v_proj` + `k_norm` + RoPE to that feature. Slot 0 is the anchor, a real token; slots 1–7 hold the mask embedding. Seven predictions go through the target's `lm_head`/`embed_tokens`; the drafter has no `lm_head`. | Weights fail to load (`fc` shape), or near-zero acceptance. | Xiaomi `dflash/dflash.py` (sha256 `da5ab173…`); vLLM `qwen3_dflash.py`; real header (63 tensors). **Never import** the first attempt's `draft/dflash.py`: it pins `fc (4096,4096)` and a drafter `lm_head` (`tests/test_dflash.py:178-179`). | I6/I7: negative test on the `fc` shape and a missing tap. |
| **T16** | MTP layers use SWA window **128** (`sliding_window_size`) and `v_scale` 0.707. Inputs are `model.embed_tokens` rows, because `tie_word_embeddings: false`. | Low acceptance; wrong drafter KV budget (the first attempt assumed 1024 → 7.9 MB/seq instead of 0.98 MB). | vLLM `mimo_v2_mtp.py`; SGLang `mimo_v2_flash_nextn.py:78`. **Never import** the first attempt's `draft/mtp.py` (window 1024, `lm_head`-row embeddings). | I6/I7: window and embedding-source negatives. |
| **T17** | SWA state cannot be rebuilt by replaying 128 tokens from an empty window. That fixes the ring's size, not its contents. The error enters the replayed positions' GA KV permanently. | Subtle drift after partial-prefix restores; restore ≠ cold compute. | Restore only at an exact snapshot at or before the divergence point (ARCH §11.3). ds41rt's replay rests on DeepSeek's layer-19/20 split, which MiMo lacks. | I5b: restore→continue ≡ cold at temperature 0. |
| **T18** | `v_scale` is applied **before** caching; the reference caches V×0.707 (`modeling_mimo_v2.py:301-302`). | Small FP8-KV rounding differences if it is applied after the cache read with a different scale layout. | Match the reference, or fold into `o_proj` with a golden-equivalence test. | I3. |
| **T19** | RoPE angles at long positions: θ 1e7 (GA) with positions up to 1,048,575. | Long-range incoherence if angles are computed in BF16/FP16 or taken from a truncated table. | FP32 `float(pos) × inv_freq`, as HF does, computed on the fly (no 256 MB tables). | I3: parity at positions 1, 128K and 1M against an FP64 reference, with a declared tolerance. |
| **T20** | FP8 KV scale layout: "block-128" does not divide the 192-dim K. Scales must be per token × head, with K and V separate, and counted in the pool bytes. | Silent quality loss at long range; pool over-commit. | ARCH §11.7. | I3: layout pin. I5: G4n quality gate at 32K/128K. |
| **T21** | KV budgeting: `VRAM × 0.97 − weights` ignores workspace, headroom and the CUDA context. Lifetime reservation (prompt + full `max_tokens`) costs 755 MB per request at MiMo's size. | Startup failure (an 18.13 GiB pool pinned on a 5090), or starved concurrency. | Measure the pool at boot; grow-on-demand admission (ARCH §11.1/§11.8). | I5: admission and pool tests. |

### 6.1 Addendum, 23 Sep 2026 09:40 AEST: T22–T27 (tonyd2wild recipe @ `13621bb`; ADVISOR-I3 §10)

| ID | Trap | Symptom | Fix (source) | Must |
|---|---|---|---|---|
| **T22** | Router bias precision. `e_score_correction_bias` is **F32** [256] in the checkpoint; the reference computes logits in FP32 and selects top-k on FP32 biased scores. | Near-tie expert flips: quality drift, not salad. vLLM keeps BF16 (the recipe's open item). | FP32 logits, bias and top-k (`modeling_mimo_v2.py` `MiMoV2MoEGate.forward`). | I5: a near-tie negative that flips under a BF16 bias. |
| **T23** | Sampling defaults when the client omits them. Near-greedy defaults cause tool-call loops (148 or 446 identical calls). Inheriting the checkpoint's `max_new_tokens: 2048` as the default `max_tokens` (vLLM `--generation-config auto`) silently truncates long outputs. | Agent loops; truncated file writes. | Defaults: temperature 1.0, top_p 0.95, `repetition_penalty` 1.05. Default `max_tokens` = `max_output_tokens`, never 2048. | I5: sampling-defaults gate. |
| **T24** | Tool-call storms: dozens to hundreds of calls in one response after a large write. It is the model's likely continuation, not a serving bug. Temperature 0.6 is worse; repetition penalty does not help. | 15-minute turns; hundreds of calls. | Native `tool_call_cap` (6): stop at the opening of call 7, drop the partial, `finish_reason: tool_calls`. | I5: captured-body replay gate. |
| **T25** | Penalties inside a speculative verify block must be position-dependent: position *i* sees the draft tokens before *i*. | Speculation changes the sampled distribution. | Per-position seen-bitmap deltas in verify. | I6: draft on/off distribution identity with penalty 1.05. |
| **T26** | Asynchronous overlap bleeds across requests. vLLM's async scheduling with speculative decoding injected foreign-script characters under concurrency (vllm#46669). | Corrupted tokens only under load. | No async stage may read or write another slot's buffers; gate G8 (stress-corrupt: 24 concurrent strict outputs → 0 bad lines). | I5: G8. |
| **T27** | Tool-call parsing must match vLLM `mimo` (Qwen3 parser engine) and coerce values to the tool schema. The template renders non-strings with `tojson`. | Arguments with the wrong types (`"5"` for 5); lost calls when the closing `[/tool_call]` tag is missing (tags in square brackets: T29). | Port the parser state machine and schema coercion (`vllm/parser/qwen3.py`, `parser_engine.py` `_coerce_value`). | I5: parser conformance goldens. |

### 6.2 Addendum, 23 Sep 2026 AEST: T28 (ADVISOR-I4 §3.5; first seen I1b row 4)

| id | Symptom | Damage | Must | Gate |
|---|---|---|---|---|
| **T28** | Raw-mode output from the `-RL` checkpoint: given a prompt WITHOUT the chat template, MiMo-V2.6-Flash-RL emits the EOS marker first; if decoding continues past it, it produces a role-less assistant-marker turn and `REWARD:True` (an RL-rollout artifact). Seen in I1b row 4 (raw gen ended `787, 28166, 25, 2514, 151645` = `REWARD:True`). | A false "PASS" read past EOS (ADVISOR-I4 F1); a server fed the raw prompt returns an EMPTY completion (first token is EOS). | Every coherence or needle gate renders the chat template and honours the full EOS set `[151643, 151645, 151672]`; any gate that ignores EOS labels itself a raw-mode probe; `/v1/completions` in raw mode is documented out-of-distribution for this checkpoint; the chat path's output checks assert: no `REWARD:`, no role-less assistant marker, no text after EOS. | I4 X1a: done, clean (receipt `runs/20260923-i4/X1a-receipt.md`). I5: chat-path output checks in the gate suite. |

### 6.3 Addendum, 23 Sep 2026 AEST: T29 (seen live in the captain session)

Tags below are written in **square brackets on purpose**. MiMo's real tags use angle brackets, and writing them
literally is the trap itself.

| id | Symptom | Damage | Must | Gate |
|---|---|---|---|---|
| **T29** | Tool-call markup or chat special tokens inside **tool-call arguments**. MiMo emits tool calls in an XML-like format (`[tool_call]`, `[function=NAME]`, `[parameter=K]`, plus their closing tags) with no escaping. Both the hosted API and vLLM's `mimo` parser split on those tags. `[think]`/`[/think]` feed the reasoning parser. `im_start` and `im_end` are special tokens, and `im_end` is EOS. | 23 Sep, captain on `mimo-token-plan`:<br>- an edit whose argument quoted the closing tool_call tag was cut short, garbling the T27/T28 rows;<br>- the fix attempts made the provider return a **nameless tool call**;<br>- dsh persisted it, and every later request failed with `400 … tool_calls[0] is missing a function name`;<br>- the session was wedged twice (12:36 and 13:14 AEST) and recovered by forking.<br>An `im_end` inside an argument ends generation mid-call. | Agents on MiMo lanes never emit these tags or tokens literally in any tool argument (edit, write, bash, commit message).<br>- Docs write them in square brackets.<br>- Code and tests build them from parts or escapes (e.g. `"\x3c" + "tool_call>"`), so the literal never appears in a model-emitted argument.<br>- Our API (A8) validates every parsed call (non-empty name, JSON-parsable arguments) and returns an error rather than a nameless call.<br>- dsh-side: drop or repair a nameless call before persisting it. | I5:<br>- parser conformance goldens, including a parameter value that contains the closing parameter tag and one that contains `im_end`;<br>- the L5 API gate runs the T24 checker (`+noname`). |
