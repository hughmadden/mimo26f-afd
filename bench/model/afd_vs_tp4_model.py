"""First-principles performance model: MiMo-V2.6-Flash on
  (A) 4x DGX Spark, tensor-parallel TP4 (vLLM-style: every Spark holds 1/4 of every weight)
  (B) AFD: RTX 5090 coordinator (attention, KV, embed/lm_head, drafter) + 4x Spark experts
      (ds41rt TP4EP1: every Spark holds a 1/4 intermediate slice of every expert).

Every number is (low, high) = (pessimistic, optimistic) under stated assumptions.
Calibration anchors (measured):
  - tonyd2wild MiMo TP2 on 2 Sparks (vLLM): prefill 1,947 tok/s @2K, 656 @248K; C1 53.3 tok/s (DFlash)
  - ds41rt v10 DSV4.1 on 3 Sparks + RTX PRO 6000: target-only decode 46.4 tok/s, prefill 6.9k tok/s @32K suffix
"""

import math

GB = 1e9
# ---------------- model geometry (real config / headers) ----------------
H = 4096
L_MOE = 47                      # MoE layers (block 0 is dense)
N_EXP, TOPK = 256, 8
EXPERT_BYTES = 13_369_344       # MXFP4 gate+up+down incl. E8M0 scales (13.37 MB)
SLICE = EXPERT_BYTES / 4        # 1/4 intermediate slice per Spark (both designs)
ATTN_W = 6.237e9                # fused qkv FP8 (3.016) + o_proj BF16 (3.221) over 48 layers
LMHEAD_W = 1.25e9               # BF16 152,576 x 4096
DENSE_W = 0.211e9               # layer-0 dense FFN FP8
ROUTER_W = 0.1e9
DFLASH_W = 2.936e9              # BF16 drafter (fc + 5 layers)
GA_KV_TOK = 11_520              # FP8 bytes/token, 9 GA layers x 4 KV heads x 320
# FLOPs per token (prefill)
F_EXPERT = TOPK * 3 * 2 * H * 2048 * L_MOE          # 18.93 GF
F_ATTN_PROJ = (39 * 2 * H * (14848 + 4096 * 2) + 9 * 2 * H * (13568 + 4096 * 2))  # qkv + o_proj
F_DENSE = 3 * 2 * H * 16384
F_ROUTER = 2 * H * N_EXP * L_MOE
F_SWA = 39 * 64 * 128 * 320 * 2
F_NONEXP = F_ATTN_PROJ + F_DENSE + F_ROUTER + F_SWA  # ~9.6 GF
def f_ga_avg(n_prompt):                              # average GA FLOPs/token over a prompt of length n
    return 9 * 64 * 320 * 2 * n_prompt / 2           # = 184,320 * n

# ---------------- wire sizes (ds41rt protocol v2, MiMo geometry) ----------------
REQ_ROW = 40 + TOPK * 12 + (H + H // 32)   # descriptor + 8 routes + FP8 row + UE8M0 scales = 4,360 B
RESP_ROW = H * 2                           # compact BF16 rank partial = 8,192 B
AR_MSG = H * 2                             # TP all-reduce message per token (BF16 hidden)
AR_PER_RANK = 2 * (4 - 1) / 4 * AR_MSG     # ring all-reduce bytes sent (= received) per rank = 12,288 B
N_AR = 2 * 48 + 1                          # attention + MoE all-reduce per layer, + embedding

# ---------------- hardware ----------------
SPARK_BW = 273 * GB
C5090_BW = 1792 * GB
SPARK_LINK = 15.75 * GB                     # Spark CX7 port trains Gen5 x4 ~126 Gb/s (coordinator host notes defect 5)
ROMEO_LINK = 31 * GB                        # 2x200G bond behind PCIe Gen5 x8 (~250 Gb/s host ceiling)

A = dict(  # assumption ranges: (pessimistic, optimistic)
    eta_spark=(0.60, 0.80),        # achieved fraction of LPDDR5x BW for weight streaming
    eta_5090=(0.70, 0.85),
    kappa=(1.00, 0.63),            # unique-expert factor vs independent routing (ds41rt measured 22/34.7 = 0.63)
    t_ar=(60e-6, 25e-6),           # small-message 4-rank all-reduce latency over RoCE (NCCL)
    t_b=(200e-6, 80e-6),           # AFD per-layer boundary: post, RDMA RTT, poll, 4-way reduce (v10-calibrated ~0.13 ms)
    f_spark=(21e12, 30e12),        # effective prefill TFLOPS per Spark: 21.9 implied by MEASURED vLLM TP4 8K
                                   # (tonyd2wild recipe 2026-09-22, 2,911 tok/s); 28 by TP2; ds41rt v10 ~34
    f_5090=(83e12, 200e12),        # 5090 = 3.3x a GB10 peak: pessimistic = same efficiency as vLLM-on-GB10, optimistic = tuned FA-class kernel
)

def unique_experts(rows, kappa):
    return N_EXP * (1 - (1 - TOPK / N_EXP) ** rows) * kappa

def decode_step(design, batch, rows, ctx, i, draft=True):
    """Seconds per decode/verify step. i=0 pessimistic, 1 optimistic."""
    es, ec, k = A["eta_spark"][i], A["eta_5090"][i], A["kappa"][i]
    n_rows = batch * rows
    u = min(N_EXP, unique_experts(n_rows, k)) if n_rows > 1 else TOPK
    expert_bytes_per_spark = u * L_MOE * SLICE
    t_exp = expert_bytes_per_spark / (SPARK_BW * es)
    nonexp = ATTN_W + LMHEAD_W + DENSE_W + ROUTER_W
    draft_bytes = (DFLASH_W + LMHEAD_W) if draft else 0.0
    kv = batch * ctx * GA_KV_TOK
    if design == "tp4":
        t_side = (nonexp + draft_bytes + kv) / 4 / (SPARK_BW * es)
        # all-reduces: latency + bandwidth term for the row payload
        n_ar = N_AR + (10 if draft else 0)
        t_sync = n_ar * (A["t_ar"][i] + n_rows * AR_PER_RANK / SPARK_LINK)
        return dict(expert=t_exp, attn_side=t_side, sync=t_sync, total=t_exp + t_side + t_sync)
    else:
        t_side = (nonexp + draft_bytes + kv) / (C5090_BW * ec)
        wire_in = L_MOE * 4 * n_rows * RESP_ROW / ROMEO_LINK      # serialization of 4 partial planes
        t_sync = L_MOE * A["t_b"][i] + wire_in
        if batch == 1:            # one lane: coordinator and Sparks alternate every layer
            total = t_exp + t_side + t_sync
        else:                     # two execution lanes overlap coordinator and Spark work
            total = max(t_exp, t_side) + t_sync
        return dict(expert=t_exp, attn_side=t_side, sync=t_sync, total=total)

def prefill_rate(design, n_prompt, chunk, i):
    """Tokens/s for a prompt of length n_prompt (average over the prompt), pipelined (max of stages)."""
    es, fs, fc = A["eta_spark"][i], A["f_spark"][i], A["f_5090"][i]
    ga = f_ga_avg(n_prompt)
    if design == "tp4":
        t_comp = (F_EXPERT + F_NONEXP + ga) / (4 * fs)
        stream = (N_EXP * L_MOE * SLICE + (ATTN_W + DENSE_W) / 4) / (SPARK_BW * es) / chunk
        t_net = N_AR * AR_PER_RANK / SPARK_LINK
        stages = dict(spark_compute=t_comp, spark_stream=stream, network=t_net)
    else:
        t_sp = F_EXPERT / (4 * fs)
        stream = N_EXP * L_MOE * SLICE / (SPARK_BW * es) / chunk
        t_co = (F_NONEXP + ga) / fc
        t_net = L_MOE * 4 * RESP_ROW / ROMEO_LINK
        stages = dict(spark_compute=t_sp, spark_stream=stream, coordinator=t_co, network=t_net)
    bound = max(stages, key=stages.get)
    return 1 / stages[bound], bound, stages

def fmt(x):
    return f"{x:,.0f}"

if __name__ == "__main__":
    print("== wire / memory facts ==")
    print(f"AFD request row {REQ_ROW} B, response row {RESP_ROW} B")
    per_tok_out = L_MOE * 4 * REQ_ROW; per_tok_in = L_MOE * 4 * RESP_ROW
    print(f"AFD coordinator link per token: out {per_tok_out/1e6:.2f} MB, in {per_tok_in/1e6:.2f} MB; "
          f"per Spark: in {L_MOE*REQ_ROW/1e6:.2f} MB, out {L_MOE*RESP_ROW/1e6:.2f} MB")
    print(f"TP4 per Spark per token: sent {N_AR*AR_PER_RANK/1e6:.2f} MB, received {N_AR*AR_PER_RANK/1e6:.2f} MB "
          f"({N_AR} all-reduces)")
    print(f"AFD coord inbound ceiling {ROMEO_LINK/per_tok_in:,.0f} tok/s; TP4 per-Spark link ceiling "
          f"{SPARK_LINK/(N_AR*AR_PER_RANK):,.0f} tok/s")
    print(f"old ds41rt per-route FP32 return, MiMo shape: {TOPK*H*4*4*L_MOE/1e6:.1f} MB/token -> "
          f"{ROMEO_LINK/(TOPK*H*4*4*L_MOE):,.0f} tok/s ceiling")
    print(f"FLOPs/token: experts {F_EXPERT/1e9:.2f} GF, non-expert {F_NONEXP/1e9:.2f} GF, "
          f"GA avg @8K {f_ga_avg(8192)/1e9:.1f}, @128K {f_ga_avg(131072)/1e9:.1f}, @1M {f_ga_avg(1048576)/1e9:.1f} GF")
    print()
    print("== decode (ms per step; tokens/step is the same drafter in both designs) ==")
    cases = [("C1 no-spec, 8K ctx", 1, 1, 8192, False),
             ("C1 DFlash R=8, 8K ctx", 1, 8, 8192, True),
             ("C1 DFlash R=8, 256K ctx", 1, 8, 262144, True),
             ("C16 DFlash R=8, 32K ctx", 16, 8, 32768, True)]
    for name, b, r, ctx, dr in cases:
        out = {}
        for d in ("tp4", "afd"):
            lo = decode_step(d, b, r, ctx, 0, dr); hi = decode_step(d, b, r, ctx, 1, dr)
            out[d] = (lo, hi)
        print(name)
        for d in ("tp4", "afd"):
            lo, hi = out[d]
            print(f"   {d}: total {hi['total']*1e3:5.1f}-{lo['total']*1e3:5.1f} ms | "
                  f"experts {hi['expert']*1e3:5.1f}-{lo['expert']*1e3:5.1f} | "
                  f"attn-side {hi['attn_side']*1e3:5.1f}-{lo['attn_side']*1e3:5.1f} | "
                  f"sync {hi['sync']*1e3:4.1f}-{lo['sync']*1e3:4.1f}")
        tl, th = out["tp4"][0]["total"], out["tp4"][1]["total"]
        al, ah = out["afd"][0]["total"], out["afd"][1]["total"]
        print(f"   AFD speed vs TP4: {tl/al:.2f}x (pessimistic both) .. {th/ah:.2f}x (optimistic both)")
    print()
    print("== prefill tok/s (pipelined, max-stage bound) ==")
    for n in (8192, 32768, 131072, 262144, 1048576):
        row = [f"{n//1024:>5}K"]
        for d, ch in (("tp4", 8192), ("afd", 256), ("afd", 2048)):
            lo = prefill_rate(d, n, ch, 0); hi = prefill_rate(d, n, ch, 1)
            row.append(f"{d}{'' if d=='tp4' else '@'+str(ch)}: {fmt(lo[0])}-{fmt(hi[0])} [{lo[1]}/{hi[1]}]")
        print("  ".join(row))
    print()
    for n in (131072, 1048576):
        for d, ch in (("tp4", 8192), ("afd", 2048)):
            lo = prefill_rate(d, n, ch, 0)[0]; hi = prefill_rate(d, n, ch, 1)[0]
            print(f"time to prefill {n//1024}K on {d}: {n/hi/60:.1f}-{n/lo/60:.1f} min")
    print()
    print("== calibration check: TP2 on 2 Sparks (tonyd2wild measured 1,947 @2K, 656 @248K) ==")
    for n, meas in ((2048, 1947), (253952, 656)):
        f = (F_EXPERT + F_NONEXP + f_ga_avg(n))
        print(f"  n={n}: implied effective TFLOPS per Spark = {meas*f/2/1e12:.1f}")
    print("== KV capacity ==")
    spark_kv = 4 * (0.85 * 121.7 * 2**30 - (N_EXP*L_MOE*EXPERT_BYTES + ATTN_W + LMHEAD_W*2 + DENSE_W + DFLASH_W) / 4) / GA_KV_TOK
    print(f"  TP4: ~{spark_kv/1e6:.1f}M tokens (GMU 0.85 of 121.7 GiB per Spark, minus 1/4 weights)")
    print(f"  AFD 5090: ~1.2-1.4M tokens (13-15 GiB after weights, workspace, 2 GiB headroom)")

# ---------------- KV tiers (A2): capacity and restore time, MiMo geometry ----------------
PAGE_TOKENS = 256
PAGE_BYTES = PAGE_TOKENS * GA_KV_TOK                # 2,949,120 B per sealed GA page (FP8, scales excluded)
SWA_TAIL = 39 * 128 * 8 * 320                       # 12,779,520 B: all 39 SWA ring states at one position
DFLASH_RING = 1024 * 5 * 8 * (128 + 128) * 2        # 20,971,520 B BF16 context K/V ring (FP8 halves it)
MTP_RING = 3 * 128 * 8 * 320                        # 983,040 B
LOGIT_ROW = 152_576 * 2                             # BF16 first-token logits
TAIL = SWA_TAIL + DFLASH_RING + LOGIT_ROW

def tier_report():
    print("== KV tiers ==")
    print(f"  sealed GA page {PAGE_BYTES:,} B; snapshot tail {TAIL/1e6:.1f} MB "
          f"(SWA rings {SWA_TAIL/1e6:.1f} + DFlash ring {DFLASH_RING/1e6:.1f} + logits {LOGIT_ROW/1e3:.0f} KB)")
    links = [("5090 PCIe Gen5 x16, ~25-40 GB/s achieved", 25e9, 40e9),
             ("5090 PCIe as read today (Gen1 x16, ~3-4 GB/s)", 3e9, 4e9),
             ("tier 2 from Sparks via the coordinator inbound (~31 GB/s ceiling)", 20e9, 31e9)]
    for n in (170_000, 1_048_576):
        size = n * GA_KV_TOK + TAIL
        print(f"  restore {n:,} tokens = {size/1e9:.2f} GB:")
        for name, lo, hi in links:
            print(f"     {name}: {size/hi*1e3:,.0f}-{size/lo*1e3:,.0f} ms")
    for gib in (48, 64):
        print(f"  tier 1 host pinned {gib} GiB -> {gib*2**30/GA_KV_TOK/1e6:.1f}M tokens")
    per_spark = (50e9, 60e9)
    print(f"  tier 2 Spark memory 4 x {per_spark[0]/1e9:.0f}-{per_spark[1]/1e9:.0f} GB -> "
          f"{4*per_spark[0]/GA_KV_TOK/1e6:.1f}-{4*per_spark[1]/GA_KV_TOK/1e6:.1f}M tokens")
    streaming = 7400 * GA_KV_TOK
    print(f"  streaming write-behind at 7.4k tok/s prefill = {streaming/1e6:.0f} MB/s of store traffic")

MEASURED_TP4 = {  # tonyd2wild MiMo-V2.6-Flash-DGX-Spark-Recipe results/tp4 (vLLM TP4, 4 Sparks, 2026-09-22)
    "decode_agg": {"C1": 59.8, "C6": 191.6, "C8": 221, "C12": 252, "C16": 269, "C24": 363, "C32": 397},
    "decode_per_stream_C1": 70.66, "ttft_C1_s": 0.235,
    "prefill_tok_s": {2004: 2952, 7919: 2911, 31836: 2603, 63764: 2172},
    "kv_pool_tokens_at_500k": 15_045_038, "kv_gib_per_rank": 46.69,
}

if __name__ == "__main__":
    print()
    tier_report()
    print()
    print("== measured vLLM TP4 baseline (the bar AFD must beat) ==")
    for k, v in MEASURED_TP4.items():
        print(f"  {k}: {v}")
    for n, meas in MEASURED_TP4["prefill_tok_s"].items():
        f = (F_EXPERT + F_NONEXP + f_ga_avg(n))
        print(f"  TP4 prefill @{n}: implied {meas*f/4/1e12:.1f} effective TFLOPS per Spark")
