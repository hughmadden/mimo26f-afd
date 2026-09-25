"""spike/real_loop.py — P-103 topology A: bare greedy loop on REAL weights.

Single-process on the coordinator's RTX 5090; per-layer READ-ONLY streaming with
on-demand top-8 expert fetch (166 G weights >> 32 G VRAM / 110 GiB RAM).
Weights at /srv/models/XiaomiMiMo/MiMo-V2.6-Flash-RL — never moved, never
written.  Run ONLY via ``scripts/dev.sh spike real`` (stages this code to
the coordinator; code travels, weights do not).

EXTERNAL-ORACLE RECEIPT (AGENTS §4.4) — forward semantics from the READ-ONLY
  /srv/models/XiaomiMiMo/MiMo-V2.6-Flash-RL/modeling_mimo_v2.py
  sha256 a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad
Spec use only — NO code bodies lifted into spike/.  Semantics taken:
  rotary : partial 0.334 -> rope_dim = int(192*0.334) = 64 (even); split
           [rope|nope]; rotate-half with emb=cat((freqs,freqs));
           inv_freq = 1/(theta**(arange(0,dim,2)/dim)); DUAL theta:
           swa_rope_theta / swa_head_dim on SWA layers.
  sink   : per-Q-head learned bias appended as an EXTRA logits column before
           softmax, dropped after (add_full/add_swa_attention_sink_bias).
  router : sigmoid; e_score_correction_bias on CHOICE only; noaux_tc group
           top-2 sum -> topk_group groups -> top-8 in masked scores; weights =
           RAW sigmoid scores at topk_idx (not bias-adjusted) -> norm_topk_prob
           -> * routed_scaling_factor.
  v-scale: value_states * attention_value_scale (0.707) BEFORE the KV cache,
           so query path and cached context share one scale (T5).
  attn   : scaling = head_dim**-0.5 = 192**-0.5; GQA 64/4 GA, 64/8 SWA;
           causal always; sliding_window mask ONLY on SWA layers (GA None).
  layer  : pre-norm residual x2 (input_layernorm->attn +res;
           post_attention_layernorm->mlp +res); RMSNorm eps layernorm_epsilon.
  mlp    : SwiGLU down(silu(gate)*up) (gate/up/down_proj naming); experts
           mlp.experts.{e}.* are MXFP4 (E2M1 nibble + E8M0-32; T10 clamp,
           T14 low-nibble=even) — NOT FP8 block-128 (P-105 correction).

Trap seams (loader=spike.loader/real_loader, exact golden-pinned codec):
T1 ckpt_tp=4 shard-major [Q_c|K_c|V_c] regroup · T2 per-shard grid trim ·
T3 GA drops window · T5 v-scale both paths · T8 evict < min(batch_pos)-w+1 ·
T9 start_pos continuation.  Salad => name the seam; greedy = argmax only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct

import numpy as np
import torch

from spike import mxfp4, real_loader as RL
from spike.quant import _e4m3_table, split_shard_major_fused

# Weights root: the coordinator default unchanged; dev-host runs (I1b needle, D7: the dev host's
# 4090 only) override via MIMO26_WEIGHTS_DIR -> the dev host copy.
BASE = os.environ.get("MIMO26_WEIGHTS_DIR", "/srv/models/XiaomiMiMo/MiMo-V2.6-Flash-RL")
INDEX = f"{BASE}/model.safetensors.index.json"
V_SCALE_DEFAULT = 0.707
PROMPTS = {
    "factual": "Paris is the capital of",
    "arithmetic": "2+3=",
    "continuation": "The quick brown",
}
E4M3 = torch.from_numpy(_e4m3_table().astype(np.float32))  # F4 fix: 256 f32 entries
# (the table is float64; view()ing its bytes as float32 yielded 512 junk entries)


def _check_e4m3_lut(lut: torch.Tensor, where: str) -> None:
    """(c3) torch-LUT self-check — RAISES unless ``lut`` is bitwise the
    golden-pinned numpy table (the numpy table is pinned to
    e4m3_decode_table.json by spike/tests/test_p103_codecs.py).

    EXECUTES on every the coordinator/torch run: at IMPORT on the torch-built LUT and
    at RealModel init on the DEVICE LUT (the one dequant actually indexes).
    The dev-host test env has no torch and skips this path (importorskip pin).
    Regression guard for the F4 frombuffer-view bug class (512 junk entries).
    """
    ref = _e4m3_table().astype(np.float32).view(np.uint32).tolist()
    got = lut.detach().cpu().numpy().view(np.uint32).tolist()
    if got != ref:
        raise RuntimeError(f"spike: {where} E4M3 LUT != golden-pinned numpy table (F4 class)")


_check_e4m3_lut(E4M3, "torch-built")  # (c3) raises at import on ANY torch run


class Reader:
    """Read-only safetensors streaming: headers once, per-tensor pread."""

    def __init__(self):
        self.wm: dict = json.load(open(INDEX))["weight_map"]
        self._meta: dict = {}
        self.names = set(self.wm)

    def _header(self, shard: str):
        if shard not in self._meta:
            with open(f"{BASE}/{shard}", "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                meta = json.loads(f.read(n))
                self._meta[shard] = (8 + n, meta)
        return self._meta[shard]

    def get(self, name: str) -> tuple:
        shard = self.wm[name]
        hlen, meta = self._header(shard)
        info = meta[name]
        lo, hi = info["data_offsets"]
        with open(f"{BASE}/{shard}", "rb") as f:
            f.seek(hlen + lo)
            raw = f.read(hi - lo)
        dt = {"F32": np.float32, "BF16": np.uint16, "U8": np.uint8, "F8_E4M3": np.uint8}[info["dtype"]]
        return info["dtype"], tuple(info["shape"]), np.frombuffer(raw, dtype=dt)


def to_f32(dtype: str, arr: np.ndarray) -> torch.Tensor:
    if dtype == "F32":
        return torch.from_numpy(arr.copy())
    if dtype == "BF16":
        return torch.from_numpy(arr.copy()).view(torch.bfloat16).float()
    raise ValueError(f"to_f32: unexpected dtype {dtype}")


def dequant_fp8(dtype: str, arr: np.ndarray, shape) -> torch.Tensor:
    if dtype != "F8_E4M3":
        return to_f32(dtype, arr.reshape(shape))
    scale = None
    return None  # caller uses dequant_block


def dequant_block(w8: np.ndarray, shape, scale_f32: np.ndarray, sshape) -> torch.Tensor:
    """FP8 E4M3 block-(128,128) dequant with padded-grid TRIM (T2): block rows
    map to grid row r//128 of the FULL grid; pad rows beyond `shape` never
    index because we slice the expanded grid back to shape."""
    r, c = shape
    w = E4M3[torch.from_numpy(w8.astype(np.int64)).reshape(r, c)]
    s = torch.from_numpy(np.array(scale_f32.reshape(sshape)))  # copy: source is read-only (UserWarning in fire row #1 log)
    s = s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:r, :c]
    return w * s


class KV:
    def __init__(self, n_layers: int, device):
        self.k = [None] * n_layers
        self.v = [None] * n_layers
        self.pos = [None] * n_layers
        self.device = device

    def tokens(self) -> int:
        return 0 if self.pos[0] is None else int(self.pos[0][-1]) + 1

    def append(self, layer, pos, k, v, window):
        pos = torch.as_tensor(pos, dtype=torch.long, device=self.device)
        self.k[layer] = k if self.k[layer] is None else torch.cat([self.k[layer], k])
        self.v[layer] = v if self.v[layer] is None else torch.cat([self.v[layer], v])
        self.pos[layer] = pos if self.pos[layer] is None else torch.cat([self.pos[layer], pos])
        if window is not None:
            keep_from = int(pos.min()) - window + 1  # T8: min(batch_pos)-w+1
            drop = int(torch.searchsorted(self.pos[layer], torch.as_tensor(keep_from, device=self.device)))
            if drop:
                self.pos[layer] = self.pos[layer][drop:]
                self.k[layer] = self.k[layer][drop:]
                self.v[layer] = self.v[layer][drop:]


def _chunk_env() -> int:
    """Query-chunk size for prefill (``MIMO26_SPIKE_CHUNK``; 0/unset = legacy
    single-pass).  ``spike/run.sh needle`` defaults it to 512 on the dev host: the 4090 is
    tenant-adjusted (doctor: cotenant_a 3418 MiB + cotenant_b 2848 MiB resident ->
    ~18.3 GiB free), and the single-pass 2 x [n_q, T, Tk] score chain (8 GiB at
    T=4096) leaves too little margin.  Chunked peak drops to ~10.6 GiB."""
    raw = os.environ.get("MIMO26_SPIKE_CHUNK", "0") or "0"
    try:
        return max(int(raw), 0)
    except ValueError:
        raise SystemExit(f"MIMO26_SPIKE_CHUNK must be an integer (got {raw!r}; 0 = unchunked)")


def attention_core(q, kk, vv, keep, sink_bias=None, chunk=0):
    """Score/softmax/aggregate block of ``RealModel.attention`` — the one big
    transient (I1b query-chunked prefill seam, MIMO26_SPIKE_CHUNK).

    q [T, n_q, d_qk]; kk/vv [Tk, n_q, d_qk / d_v] (rep-expanded); keep [T, Tk]
    bool (causal + T3/T8 window); sink_bias [n_q, 1, 1] or None (the T6 family
    gate is the caller's); chunk 0 = legacy single pass.  -> [T, n_q, d_v].

    ``chunk > 0`` slices the QUERY rows only — every query row's softmax still
    spans its full key row — so chunked == unchunked semantically.  Equality is
    pinned atol=1e-6 (spike/tests/test_needle_prompt.py::
    test_query_chunked_prefill_equals_full); bitwise is not promised because
    GEMM-shape changes can reorder the d_qk reduction.  The transient score
    chain drops from 2 x [n_q, T, Tk] to 2 x [n_q, chunk, Tk].  Scaling uses
    the QK dim (T4): ``q.shape[-1] ** -0.5`` == ``head_dim ** -0.5``.
    """
    T = q.shape[0]
    step = chunk if chunk > 0 else T
    outs = []
    for t0 in range(0, T, step):
        t1 = min(T, t0 + step)
        att = torch.einsum("thd,shd->hts", q[t0:t1], kk) * (q.shape[-1] ** -0.5)
        att = att.masked_fill(~keep[t0:t1][None], float("-inf"))
        if sink_bias is not None:
            att = torch.cat([att, sink_bias.expand(q.shape[1], t1 - t0, 1)], -1)
        p = torch.softmax(att, -1)
        if sink_bias is not None:
            p = p[..., :-1]                                        # drop sink mass
        outs.append(torch.einsum("hts,shd->thd", p, vv))
    return torch.cat(outs, 0)


def _seg_env() -> int:
    """Segmentwise-prefill segment size (``MIMO26_SPIKE_SEG``; 0 = monolithic).
    ``spike/run.sh needle`` defaults 512: KV-accumulating segments cap EVERY
    T-shaped transient at segment size (fire rows #1/#2 fix (2))."""
    raw = os.environ.get("MIMO26_SPIKE_SEG", "0") or "0"
    try:
        return max(int(raw), 0)
    except ValueError:
        raise SystemExit(f"MIMO26_SPIKE_SEG must be an integer (got {raw!r}; 0 = monolithic)")


TRACE: list[str] = []


def _mtrace(tag: str, device=None) -> None:
    """Append one memtrace line: live bytes + top live tensors (gc-tracked).

    I1b --memtrace (fire rows #1/#2 left ~10 GiB unattributed by model) — this
    NAMES the live set at op-class boundaries instead of modeling it.  Stable
    format (dry-runnable on CPU): ``[memtrace] <tag> cuda_alloc=<f>MiB
    gc_live=<f>MiB n=<count> top=[('<shape>', '<dtype>', <f>MiB), ...]``
    """
    import gc
    import warnings
    live = 0
    tops: list[tuple[tuple, str, float]] = []
    with warnings.catch_warnings():          # gc surfaces torch internals; keep the trace quiet
        warnings.simplefilter("ignore")
        for o in gc.get_objects():
            try:
                if isinstance(o, torch.Tensor):
                    b = o.numel() * o.element_size()
                    live += b
                    tops.append((tuple(o.shape), str(o.dtype).replace("torch.", ""), b / (1 << 20)))
            except Exception:  # noqa: BLE001 — gc can surface half-initialized tensors
                continue
    tops.sort(key=lambda t: -t[2])
    dev = (torch.cuda.memory_allocated(device) / (1 << 20)
           if device is not None and str(device).startswith("cuda") else float("nan"))
    TRACE.append(
        f"[memtrace] {tag} cuda_alloc={dev:.1f}MiB gc_live={live / (1 << 20):.1f}MiB "
        f"n={len(tops)} top={[(s, d, round(m, 1)) for s, d, m in tops[:6]]}")


def moe_expert_major(x, idx, w, load_expert):
    """MoE token dispatch, expert-major RESIDENCY (I1b fix 2b).

    ``load_expert(e) -> (gg, uu, dd)`` streams one expert set (96 MiB) and it
    is freed before the next — resident cap ONE expert set, independent of T and
    of expert count.  Pre-fix mlp() cached ``idx.unique()`` across ALL T tokens
    before dispatch: at T=4018 that is all 256 experts = 24 GiB live at once
    (fire rows #1/#2 died populating it at the same deterministic expert).
    Equals the cached loop up to f32 accumulation order — pinned atol=1e-6 in
    spike/tests/test_needle_prompt.py::test_moe_expert_major_equals_cached.
    """
    out = torch.zeros_like(x)
    hits: dict[int, list[tuple[int, int]]] = {}
    for tok in range(x.shape[0]):
        for j in range(idx.shape[1]):
            hits.setdefault(int(idx[tok, j]), []).append((tok, j))
    for e, places in hits.items():
        gg, uu, dd = load_expert(int(e))
        for tok, j in places:
            h = x[tok : tok + 1]
            out[tok] += ((torch.nn.functional.silu(h @ gg.T) * (h @ uu.T)) @ dd.T).squeeze(0) * w[tok, j]
        del gg, uu, dd
    return out


class RealModel:
    def __init__(self, r: Reader, device="cuda", memtrace: bool = False,
                 layer_cap: int | None = None):
        self.r = r
        self.device = device
        self.memtrace = memtrace      # I1b --memtrace: _mtrace at op-class boundaries
        self.layer_cap = layer_cap    # memtrace runs: first N layers only
        if device.startswith("cuda"):
            import subprocess
            free, total = torch.cuda.mem_get_info()
            if free < 8 << 30:   # AGENTS.md §6: refuse concurrent CUDA owners
                who = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"],
                    capture_output=True, text=True).stdout
                raise SystemExit(
                    f"real_loop: only {free / 2**30:.1f} GiB GPU free — refusing to run "
                    f"beside another CUDA owner (AGENTS.md §6). Current owners:\n{who}")
        self.e4m3 = E4M3.to(self.device)  # LUT indexed by u8 codes (device-side)
        _check_e4m3_lut(self.e4m3, "device")  # (c3) the DEVICE LUT, every real run
        self.cfg = json.load(open(f"{BASE}/config.json"))
        g = self.cfg.get
        self.hidden = g("hidden_size", 4096)
        self.n_q = g("num_attention_heads", 64)
        self.head_dim = g("head_dim", 192)
        self.v_head_dim = g("v_head_dim", 128)
        self.rope_dim = int(self.head_dim * g("partial_rotary_factor", 0.334))
        assert self.rope_dim % 2 == 0
        self.window = g("sliding_window", 128)
        self.v_scale = g("attention_value_scale", V_SCALE_DEFAULT)
        self.eps = g("layernorm_epsilon", 1e-6)
        self.pattern = g("hybrid_layer_pattern")           # 0 = GA, 1 = SWA
        self.moe_freq = g("moe_layer_freq", [1] * len(self.pattern))
        self.n_layers = len(self.pattern)
        self.swa_kv = g("swa_num_key_value_heads", 8)
        self.ga_kv = g("num_key_value_heads", 4)
        self.theta = g("rope_theta", 10000.0)
        self.swa_theta = g("swa_rope_theta", self.theta)
        self.sink_ga = g("add_full_attention_sink_bias", False)
        self.sink_swa = g("add_swa_attention_sink_bias", False)
        self.top_k = g("num_experts_per_tok", 8)
        self.n_exp = g("n_routed_experts")
        self.n_group = g("n_group", 1)
        self.topk_group = g("topk_group", self.n_group)
        self.norm_topk = g("norm_topk_prob", True)
        self.routed_scale = g("routed_scaling_factor", 1.0) or 1.0
        self.moe_inter = g("moe_intermediate_size")
        self.inter = g("intermediate_size", self.moe_inter)
        self.vocab = g("vocab_size", 151675)
        # pinned once (small vs 166G): embeddings + head
        de, sh, emb = r.get("model.embed_tokens.weight")
        self.embed = to_f32(de, emb.reshape(sh)).to(device)
        de, sh, lh = r.get("lm_head.weight")
        self.lm_head = to_f32(de, lh.reshape(sh)).to(device)
        de, sh, nw = r.get("model.norm.weight")
        self.final_norm = to_f32(de, nw.reshape(sh)).to(device)
        # expert name discovery (fail-loud on missing per-layer tensors)
        self.exp_names = {}
        pat = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
        for nm in r.names:
            m = pat.match(nm)
            if m:
                self.exp_names.setdefault(int(m.group(1)), {}).setdefault(int(m.group(2)), {})[m.group(3)] = nm

    # ---- helpers -----------------------------------------------------------
    def is_swa(self, layer: int) -> bool:
        return self.pattern[layer] == 1

    def kv_heads(self, layer: int) -> int:
        return self.swa_kv if self.is_swa(layer) else self.ga_kv

    def rope(self, pos, dim, theta):
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=self.device).float() / dim))
        f = torch.as_tensor(pos, device=self.device).float()[:, None] * inv[None, :]
        return torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()

    @staticmethod
    def _rotate_half(x, half):
        return torch.cat([-x[..., half:], x[..., :half]], -1)

    def load_proj(self, base: str, out_rows: int):
        de, sh, w = self.r.get(f"{base}.weight")
        if de in ("F8_E4M3", "U8"):
            if f"{base}.weight_scale_inv" in self.r.names:       # FP8 E4M3 block-128 (fused qkv)
                sde, ssh, sw = self.r.get(f"{base}.weight_scale_inv")
                t = dequant_block(w, sh, sw, ssh)
            elif f"{base}.weight_scale" in self.r.names:         # MXFP4: E2M1 nibbles + E8M0-32
                sde, ssh, sw = self.r.get(f"{base}.weight_scale")
                # device-side dequant, bitwise-pinned to mxfp4.unpack — replaces
                # the 258 ms/expert-matrix numpy f64 path (ADVISOR-I3 §11.3)
                t = mxfp4.unpack_torch(w.reshape(sh), sw.reshape(ssh), device=self.device)
            else:
                raise KeyError(f"{base}: quantized weight with no scale tensor; prefix names: "
                               f"{sorted(n for n in self.r.names if n.startswith(base))}")
        else:
            t = to_f32(de, w.reshape(sh))
        return t.to(self.device)  # [out_rows, in]

    def load_qkv(self, layer: int):
        base = f"model.layers.{layer}.self_attn.qkv_proj"
        de, sh, w = self.r.get(f"{base}.weight")
        sde, ssh, sw = self.r.get(f"{base}.weight_scale_inv")
        kind = "swa" if self.is_swa(layer) else "ga"
        # T1+T2: shard-major stored tensor -> 4 per-rank payloads with LOCAL
        # padded grids (pad rows never indexed), then regroup [Q|K|V] via the
        # golden-pinned codec (P-102 real-layout reconstruct test).
        weights, scales, per = RL.split_stored_fused(w.reshape(sh), sw.reshape(ssh), kind)
        parts = split_shard_major_fused(weights, scales, per, block=RL.FP8_BLOCK)
        out = []
        for name in ("q", "k", "v"):
            p = parts[name]
            wt = self.e4m3[torch.from_numpy(p.weight.astype(np.int64)).to(self.device)]
            st = torch.from_numpy(p.scale_per_row).to(self.device)
            st = st.repeat_interleave(RL.FP8_BLOCK[1], 1)[:, : wt.shape[1]]
            out.append(wt * st)
        return out  # [Wq, Wk, Wv] dequant f32 on device

    # ---- forward -----------------------------------------------------------
    def layer_norm(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def attention(self, layer, x, pos0, kv: KV):
        is_swa = self.is_swa(layer)
        kvh = self.kv_heads(layer)
        Wq, Wk, Wv = self.load_qkv(layer)
        T = x.shape[0]
        q, k, v = x @ Wq.T, x @ Wk.T, x @ Wv.T
        q = q.view(T, self.n_q, self.head_dim)
        k = k.view(T, kvh, self.head_dim)
        v = (v.view(T, kvh, self.v_head_dim) * self.v_scale)  # T5: before cache
        cos, sin = self.rope(range(pos0, pos0 + T), self.rope_dim, self.swa_theta if is_swa else self.theta)
        half = self.rope_dim // 2
        for t_ in (q, k):
            r_, n_ = t_[..., : self.rope_dim], t_[..., self.rope_dim :]
            t_[..., : self.rope_dim] = r_ * cos[:, None, :] + self._rotate_half(r_, half) * sin[:, None, :]
        window = self.window if is_swa else None
        kv.append(layer, list(range(pos0, pos0 + T)), k, v, window)
        if self.memtrace:
            _mtrace(f"post_kv_append L{layer}", self.device)
        kk, vv, kp = kv.k[layer], kv.v[layer], kv.pos[layer]
        qpos = torch.arange(pos0, pos0 + T, device=self.device)
        keep = kp[None, :] <= qpos[:, None]                       # causal
        if window is not None:
            keep = keep & ((qpos[:, None] - kp[None, :]) < window)  # T3/T8 window
        rep = self.n_q // kvh
        kk = kk.repeat_interleave(rep, 1)  # [Tk, n_q, head]
        vv = vv.repeat_interleave(rep, 1)
        sink = (self.sink_swa if is_swa else self.sink_ga)          # T6 family gate
        bias = None
        if sink:
            de, sh, sb = self.r.get(f"model.layers.{layer}.self_attn.attention_sink_bias")
            bias = to_f32(de, sb).view(self.n_q, 1, 1).to(self.device)
        out = attention_core(q, kk, vv, keep, bias, chunk=_chunk_env())  # I1b chunked prefill
        out = out.reshape(T, self.n_q * self.v_head_dim)
        return out @ self.load_proj(f"model.layers.{layer}.self_attn.o_proj", None).T

    def mlp(self, layer, x):
        if self.moe_freq[layer]:
            dt, sh, gw = self.r.get(f"model.layers.{layer}.mlp.gate.weight")
            g = to_f32(dt, gw.reshape(sh)).to(self.device)
            dt, sh, eb = self.r.get(f"model.layers.{layer}.mlp.gate.e_score_correction_bias")
            e_bias = to_f32(dt, eb.reshape(sh)).to(self.device)
            logits = (x.float() @ g.T).float()
            scores = torch.sigmoid(logits)
            choice = scores + e_bias                       # bias on CHOICE only
            gs = choice.view(-1, self.n_group, self.n_exp // self.n_group).topk(2, -1)[0].sum(-1)
            gi = gs.topk(self.topk_group, -1)[1]
            mask = torch.zeros_like(gs, dtype=torch.bool).scatter_(1, gi, True)
            sm = mask[:, :, None].expand(-1, -1, self.n_exp // self.n_group).reshape(-1, self.n_exp)
            tmp = choice.masked_fill(~sm, float("-inf"))
            idx = tmp.topk(self.top_k, -1)[1]
            w = scores.gather(1, idx)                      # RAW sigmoid weights
            if self.norm_topk:
                w = w / (w.sum(-1, keepdim=True) + 1e-20)
            w = w * self.routed_scale
            names_l = self.exp_names.get(layer, {})

            def load_expert(e: int):
                nm = names_l.get(int(e))
                if nm is None:
                    raise KeyError(f"expert {e} weights missing for layer {layer}")
                return (self.load_proj(nm["gate"].removesuffix(".weight"), None),
                        self.load_proj(nm["up"].removesuffix(".weight"), None),
                        self.load_proj(nm["down"].removesuffix(".weight"), None))

            return moe_expert_major(x, idx, w, load_expert)  # I1b fix 2b: one set resident
        gg = self.load_proj(f"model.layers.{layer}.mlp.gate_proj", None)
        uu = self.load_proj(f"model.layers.{layer}.mlp.up_proj", None)
        dd = self.load_proj(f"model.layers.{layer}.mlp.down_proj", None)
        return (torch.nn.functional.silu(x @ gg.T) * (x @ uu.T)) @ dd.T

    def forward(self, ids: list[int], kv: KV, start_pos: int):
        x = self.embed[torch.as_tensor(ids, device=self.device)]
        pos0 = kv.tokens() if start_pos is None else start_pos   # T9
        if self.memtrace:
            _mtrace("post_embed", self.device)
        for layer in range(self.layer_cap or self.n_layers):
            h = self.layer_norm(x, self.load_norm(f"model.layers.{layer}.input_layernorm"))
            x = x + self.attention(layer, h, pos0, kv)
            if self.memtrace:
                _mtrace(f"post_attn L{layer}", self.device)
            h = self.layer_norm(x, self.load_norm(f"model.layers.{layer}.post_attention_layernorm"))
            x = x + self.mlp(layer, h)
            if self.memtrace:
                _mtrace(f"post_mlp L{layer}", self.device)
        x = self.layer_norm(x, self.final_norm)
        # lm_head on the LAST ROW only (ADVISOR-I3 §11.3 optional lever): every
        # consumer (greedy transcript) reads only logits[-1]; the full [T, vocab]
        # f32 materialisation is ~2.45 GB at T=4018.  Returns [1, vocab].
        x = x[-1:] @ self.lm_head.T
        if self.memtrace:
            _mtrace("post_logits", self.device)
        return x

    def load_norm(self, name):
        if not name.endswith(".weight"):
            name += ".weight"
        de, sh, w = self.r.get(name)
        return to_f32(de, w.reshape(sh)).to(self.device)


# I1b fire rows #1/#2 (identical OOMs 09:22/09:24 AEST) — CORRECTION OF RECORD:
# the first diagnosis (grad retention) is REFUTED (row #2 died identically with
# inference_mode ACTIVE; spike tensors never require grad, so no graph was built
# either way).  Real cause: the MoE expert_cache held idx.unique() across ALL T
# tokens — 256 experts x 96 MiB = 24 GiB at T=4018 — dying mid-population at the
# same deterministic expert (=> identical rows).  moe_expert_major caps that at
# one expert set.  inference_mode stays as correct inference hygiene (audited:
# no param mutation; in-place only on fresh activations :336/:367/:391).
@torch.inference_mode()
def greedy(model: RealModel, ids: list[int], steps: int,
           eos: set[int] | None = None) -> tuple[list[int], int]:
    """Greedy decode.  ``eos`` (X1a): generation stops at the first sampled id in
    the set (the EOS id IS appended, then we stop — nothing after EOS, T28)."""
    kv = KV(model.n_layers, model.device)
    # I1b fix (2): segmentwise prefill — KV-accumulating segments (T9: pos0 =
    # kv.tokens()) cap every T-shaped transient at segment size.  Equality pin:
    # test_segmentwise_prefill_equals_monolithic (atol=1e-6).
    seg = _seg_env()
    step = seg if seg > 0 else len(ids)
    for i in range(0, len(ids), step):
        logits = model.forward(ids[i : i + step], kv, None)
    tok = int(logits[-1].argmax())
    gen = []
    for _ in range(steps):
        gen.append(tok)
        if eos and tok in eos:                           # X1a: honour EOS; stop after it
            break
        logits = model.forward([tok], kv, None)          # start_pos -> cache (T9)
        tok = int(logits[-1].argmax())
    return gen, kv.tokens()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--prompt", default="all")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{BASE}/tokenizer.json")
    model = RealModel(Reader())
    todo = PROMPTS if args.prompt == "all" else {args.prompt: PROMPTS[args.prompt]}
    print("== spike real loop (topology A, the coordinator 5090) — greedy argmax only")
    for name, text in todo.items():
        ids = tok.encode(text).ids
        gen, n = greedy(model, ids, args.steps)
        out = tok.decode(gen)
        print(f"[{name}] prompt={text!r} -> {out!r} ids={gen}")
    print("RESULT: PASS transcripts emitted (judge salad with the seam names above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
