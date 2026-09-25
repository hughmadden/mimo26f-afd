"""spike/footprint_probe.py — DRY device-footprint probe for the I3-I1b needle
spike on the dev host's RTX 4090 (24564 MiB, measured via nvidia-smi 2026-09-23 AEST).

NO RUN, NO torch: reads ``config.json`` and the 65 safetensors HEADERS
read-only (the same 2-pread header pattern as spike/real_loop.py:85-97) and
mirrors ``RealModel``'s device allocations line-by-line as f32 matrices — the
exact way real_loop materializes them (to_f32 always promotes to f32,
real_loop.py:110-115).

Hard scope (board task-9): measurement only — never the coordinator's 5090, never the 4
Sparks, no model forward, stdlib only (the dev host spike venv has no torch).

Structural finding this probe encodes: the spike ALREADY streams weights
per-call (load_proj/load_qkv/load_norm re-pread and dequant per forward,
real_loop.py:236-269 / :365-369) — layerwise weight offload is its load model
by construction (docstring :3-4: 166 G weights >> VRAM).  The 24 GB question is
therefore the T=4096 prefill ACTIVATION chain (the [n_q, T, T] attention
matrices), not resident weights.

Usage: python3 -m spike.footprint_probe [--base DIR] [--budget-mib 24564]
                                       [--tokens 4096] [--steps 16]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

LOCAL_WEIGHTS = os.path.expanduser("~/models/XiaomiMiMo/MiMo-V2.6-Flash-RL")
MIB = 1 << 20


def read_headers(base: Path) -> dict:
    """name -> (dtype, shape) from every shard header (real_loop.py:85-97)."""
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for shard in sorted(set(index.values())):
        with open(base / shard, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            meta = json.loads(f.read(hlen))
        for name, info in meta.items():
            if name == "__metadata__":
                continue
            out[name] = (info["dtype"], tuple(info["shape"]))
    return out


def f32_bytes(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n * 4


def device_bytes(dtype: str, shape) -> int:
    """Device f32 bytes after the real_loop dequant path (to_f32 :110-115 f32;
    dequant_block :125-134 f32; mxfp4.unpack :244 -> f32 [out, in] with the
    packed nibble dim doubled)."""
    if dtype == "U8":  # MXFP4 packed [out, in//2] -> f32 [out, in]
        return f32_bytes((shape[0], shape[1] * 2))
    return f32_bytes(shape)  # F32 / BF16 (promoted) / F8_E4M3 (LUT-indexed)


def probe(base: Path, budget_mib: int, T: int, steps: int, chunk: int = 512, seg: int = 512) -> int:
    cfg = json.loads((base / "config.json").read_text())
    H = cfg["hidden_size"]
    n_q = cfg["num_attention_heads"]
    hd, vhd = cfg["head_dim"], cfg["v_head_dim"]
    ga_kv, swa_kv = cfg["num_key_value_heads"], cfg["swa_num_key_value_heads"]
    pattern = cfg["hybrid_layer_pattern"]
    moe_freq = cfg.get("moe_layer_freq", [1] * len(pattern))
    vocab = cfg["vocab_size"]
    inter = cfg["intermediate_size"]
    moe_inter = cfg["moe_intermediate_size"]
    top_k = cfg["num_experts_per_tok"]
    n_ga = sum(1 for k in pattern if k == 0)
    n_swa = len(pattern) - n_ga

    th = read_headers(base)
    need = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight"]
    missing = [n for n in need if n not in th]
    if missing:
        print(f"RESULT: FAIL probe: missing tensors {missing} in headers")
        return 1
    exp0 = next(n for n in th if ".mlp.experts.0.gate_proj.weight" in n and ".layers.1." in n)
    lines = []  # (MiB, label, real_loop cite)

    def add(b, label, cite):
        lines.append((b / MIB, label, cite))  # MiB for display
        return b  # BYTES for the peak math (all peak sums are bytes)

    # persistent: embed + lm_head + final_norm + e4m3 LUT (real_loop.py:205-211, :174)
    persistent = (
        add(device_bytes(*th["model.embed_tokens.weight"]), "embed f32 [vocab, hidden]", ":205-207")
        + add(device_bytes(*th["lm_head.weight"]), "lm_head f32 [vocab, hidden]", ":208-209")
        + add(device_bytes(*th["model.norm.weight"]), "final_norm f32", ":210-211")
        + add(256 * 4, "e4m3 LUT f32 [256]", ":174"))

    # KV at prefill END: one append of T entries/layer; SWA eviction starts only
    # at later appends (keep_from = min(pos)-window+1 <= 0 at prefill, :150-157)
    kv = 0
    kv_ga = T * (ga_kv * hd + ga_kv * vhd) * 4 + T * 8
    kv_swa = T * (swa_kv * hd + swa_kv * vhd) * 4 + T * 8
    kv = n_ga * kv_ga + n_swa * kv_swa
    add(kv, f"KV @prefill-end {T} tok: {n_ga}GA x {kv_ga/MIB:.1f} + {n_swa}SWA x {kv_swa/MIB:.1f}", ":146-157")

    # attention transient at one layer at SEGMENT size Tt (SWA worst for k/v;
    # the sink cat is a 2x-chain like masked_fill/softmax — no 3rd live copy).
    # Tt = min(seg, T): fix (2) segmentwise prefill caps every T-shaped
    # transient at segment size (KV above stays at full T).
    Tt = min(seg, T) if seg > 0 else T
    qkv_b = device_bytes(*th["model.layers.0.self_attn.qkv_proj.weight"])
    o_b = device_bytes(*th["model.layers.0.self_attn.o_proj.weight"])
    attn_items = [
        (qkv_b, "Wq+Wk+Wv f32 (load_qkv output)", ":252-269"),
        (2 * (qkv_b * (n_q * hd) // (n_q * hd + ga_kv * hd + ga_kv * vhd)),
         "load_qkv build temp (wt*st, Wq-sized)", ":265-268"),
        (o_b, "o_proj f32", ":310"),
        (o_b, "o_proj dequant temp (LUT copy)", ":125-134"),
        (Tt * (n_q * hd) * 4, "q act [Tt, 12288]", ":280"),
        (Tt * (swa_kv * hd) * 4, "k act [Tt, 1536] (SWA worst)", ":280"),
        (Tt * (swa_kv * vhd) * 4, "v act [Tt, 1024] (SWA worst)", ":281-283"),
        (Tt * n_q * hd * 4, "kk repeat_interleave [Tt, 64, 192]", ":297"),
        (Tt * n_q * vhd * 4, "vv repeat_interleave [Tt, 64, 128]", ":298"),
        (2 * (n_q * min(chunk if chunk > 0 else Tt, Tt) * Tt * 4),
         f"att chain 2x [64, {min(chunk if chunk > 0 else Tt, Tt)}, {Tt}] "
         "(query-chunked; masked_fill copy; softmax p)", ":299-306"),
        (Tt * n_q * vhd * 4 + Tt * H * 4, "out + o result", ":309-310"),
        (4 * Tt * H * 4, "x/h/residuals x4", ":354-361"),
    ]
    for b, label, cite in attn_items:
        add(b, label, cite)
    attn_peak = sum(b for b, _, _ in attn_items)

    mlp_l0 = sum(device_bytes(*th[n]) for n in
                 ["model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight",
                  "model.layers.0.mlp.down_proj.weight"])
    exp_gate, exp_up, exp_down = (device_bytes(*th[n]) for n in
                                  (exp0, exp0.replace("gate_proj", "up_proj"),
                                   exp0.replace("gate_proj", "down_proj")))
    mlp_dense_peak = mlp_l0 + 2 * mlp_l0 // 3 + 3 * (Tt * inter * 4)
    add(mlp_dense_peak, "MLP dense peak (layer 0 SwiGLU; Tt acts)", ":349-352")
    moe_peak = (exp_gate + exp_up + exp_down + 2 * exp_gate
                + 256 * H * 4 + 256 * 4 + 3 * (Tt * H * 4))
    add(moe_peak, "MLP MoE peak (fix 2b: ONE expert set + unpack temp + router)", ":356-392")
    add(Tt * vocab * 4, "logits [Tt, vocab] (per segment; forward return)", ":407")

    # peak instants (attention transient, MLP transient and logits are sequential)
    peak_attn = persistent + kv + attn_peak
    peak_mlp = persistent + kv + max(mlp_dense_peak, moe_peak)
    peak_logits = persistent + kv + Tt * vocab * 4 + 4 * Tt * H * 4
    reserve = (512 + 1024) * MIB  # CUDA context ~512 MiB + cuBLAS workspace/fragmentation
    peak = max(peak_attn, peak_mlp, peak_logits) + reserve

    print(f"== spike device-footprint (dry, f32 as real_loop materializes) base={base}")
    print(f"   tokens T={T} seg={seg} Tt={min(seg, T) if seg > 0 else T} chunk={chunk} "
          f"decode_steps={steps} budget={budget_mib} MiB")
    for b, label, cite in lines:
        print(f"  {b:10.1f} MiB  {label}  (real_loop.py{cite})")
    print(f"  -- peak instants (sequential lifetimes) --")
    print(f"  {peak_attn / MIB:10.1f} MiB  persistent+KV+attention transient")
    print(f"  {peak_mlp / MIB:10.1f} MiB  persistent+KV+MLP transient")
    print(f"  {peak_logits / MIB:10.1f} MiB  persistent+KV+logits")
    print(f"  {reserve / MIB:10.1f} MiB  reserve (CUDA context + cuBLAS workspace + fragmentation)")
    print(f"  {peak / MIB:10.1f} MiB  PEAK ESTIMATE vs {budget_mib} MiB")

    # -- pre-fix death model (fire rows #1/#2 explained; CORRECTION OF RECORD) --
    # Rows #1/#2 died IDENTICALLY (17.35 / 17.36 GiB process) with grad on and
    # inference_mode active -> the grad-retention attribution is REFUTED (spike
    # tensors never require grad; no graph was built either way).  Real class:
    # the MoE expert_cache held idx.unique() across ALL T tokens — 256 experts x
    # 96 MiB = 24 GiB at T=4018 — dying mid-population at the same deterministic
    # expert (the 32.0 MiB alloc = one expert f32 [2048, 4096]; => identical
    # rows).  Suspects (a) full-score materialization and (d) KV cat-history are
    # excluded at code level; (b) mxfp4.unpack f64 refuted (unpack -> f32).
    # --memtrace names the live set at the next fire.
    exp_set = exp_gate + exp_up + exp_down
    base = persistent + kv_ga + kv_swa + 2 * (T * H * 4)  # embed/lm_head + 2 layers' KV + x/out
    for label_m, meas_gib in (("row #1", 16.77), ("row #2", 16.86)):
        n_res = (meas_gib * 1024 - base / MIB) / (exp_set / MIB)
        print(f"  -- pre-fix death model vs {label_m} ({meas_gib * 1024:.0f} MiB measured): "
              f"{n_res:.0f} of 256 experts resident mid-population")
    print("  (fix 2b: moe_expert_major caps residency at ONE expert set; fix 2:")
    print(f"   segmentwise prefill seg={seg}.  MODELS: live f32 + 1536 MiB reserve;")
    print("   NOT modeled: allocator pages, cuBLAS algo choice, expandable_segments.)")
    print(f"  RE-FIRE PREDICTION (seg={seg}, chunk={chunk}, fix 2b): peak {peak / MIB:.0f} MiB"
          f" — claim: within 10% of the measured peak.")
    margin = budget_mib - peak / MIB
    if peak / MIB <= 0.95 * budget_mib:
        print(f"VERDICT: FIT — margin {margin:.0f} MiB ({100 * margin / budget_mib:.1f}%)")
        print("note: layerwise weight streaming is ALREADY the load model "
              "(real_loop.py:236-269) — no weight offload fallback is needed at this shape.")
        return 0
    print(f"VERDICT: NON-FIT — over by {-margin:.0f} MiB")
    print("""FALLBACK PLAN (do not improvise beyond this):
  1. Query-chunked prefill is IMPLEMENTED (MIMO26_SPIKE_CHUNK ->
     real_loop.attention_core, needle cell default 512): lower the chunk further
     (256, 128) — att is 2 x [64, chunk, T].  Semantics unchanged (each query
     row's softmax spans its full key row; equality pinned atol=1e-6 in
     spike/tests/test_needle_prompt.py).  This is the structural fix: the peak
     is the activation chain, not resident weights.
  2. If logits dominate on a longer T: chunk the lm_head matmul over vocab.
  3. Layerwise WEIGHT offload is a no-op here — weights already stream per call;
     only pin-then-free the per-layer set if the allocator fragments (load_qkv
     result lives exactly one layer turn).
  4. Re-run this probe after any change; it is the receipt's §1 source.""")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("MIMO26_WEIGHTS_DIR", LOCAL_WEIGHTS))
    ap.add_argument("--budget-mib", type=int, default=24564)  # RTX 4090, measured
    ap.add_argument("--tokens", type=int, default=4096)       # I1b ~4K prompt
    ap.add_argument("--steps", type=int, default=16)          # greedy decode steps
    ap.add_argument("--chunk", type=int, default=512,          # MIMO26_SPIKE_CHUNK (needle cell default)
                    help="query-chunk rows for prefill (0 = unchunked legacy)")
    ap.add_argument("--seg", type=int, default=512,             # MIMO26_SPIKE_SEG (needle cell default)
                    help="segmentwise-prefill segment size (0 = monolithic legacy)")
    args = ap.parse_args(argv)
    return probe(Path(args.base), args.budget_mib, args.tokens, args.steps, args.chunk, args.seg)


if __name__ == "__main__":
    raise SystemExit(main())
