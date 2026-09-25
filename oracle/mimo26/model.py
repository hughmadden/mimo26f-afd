"""MiMoModel — full CPU reference forward (prefill + incremental decode).

Weights are float arrays keyed by canonical names (HF naming minus the `model.` prefix).
Quantized checkpoints are decoded by `mimo26.quant`/`mimo26.checkpoint` first: this module
is format-agnostic and serves as the arithmetic conformance target (ARCHITECTURE.md §3, §5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import GA, MOE_FFN, SWA, MiMoConfig
from .kv import KVCache
from .nn import layers as L


@dataclass
class ModelOutput:
    logits: np.ndarray  # [T, vocab]
    hidden: np.ndarray  # [T, hidden] (final-norm input, i.e. last block output)
    captured: dict[int, np.ndarray] = field(default_factory=dict)  # layer id -> [T, hidden]
    cache: KVCache | None = None


def w_name(layer: int, part: str) -> str:
    return f"layers.{layer}.{part}"


class MiMoModel:
    def __init__(self, cfg: MiMoConfig, weights: dict[str, np.ndarray]):
        self.cfg = cfg
        self.w = weights
        for name in ("embed_tokens.weight", "norm.weight", "lm_head.weight"):
            if name not in weights:
                raise ValueError(f"missing required weight {name}")
        if weights["embed_tokens.weight"].shape != (cfg.vocab_size, cfg.hidden_size):
            raise ValueError("embed_tokens shape mismatch")
        if weights["lm_head.weight"].shape != (cfg.vocab_size, cfg.hidden_size):
            raise ValueError("lm_head shape mismatch")

    # ------------------------------------------------------------------

    def _attn_block(self, layer: int, x: np.ndarray, pos: np.ndarray, cache: KVCache) -> np.ndarray:
        cfg = self.cfg
        kind = cfg.hybrid_layer_pattern[layer]
        Hq = cfg.num_attention_heads
        if kind == GA:
            Hk, Dqk, Dv = cfg.num_key_value_heads, cfg.head_dim, cfg.v_head_dim
            theta, window = cfg.rope_theta, None
        else:
            Hk, Dqk, Dv = cfg.swa_num_key_value_heads, cfg.swa_head_dim, cfg.swa_v_head_dim
            theta, window = cfg.swa_rope_theta, cfg.sliding_window
        q_rows, k_rows, v_rows, _ = cfg.attn_dims(kind)
        qkv = L.linear(x, self.w[w_name(layer, "self_attn.qkv_proj.weight")])
        q, k, v = np.split(qkv, [q_rows, q_rows + k_rows], axis=1)
        q = q.reshape(-1, Hq, Dqk)
        k = k.reshape(-1, Hk, Dqk)
        v = v.reshape(-1, Hk, Dv)
        q = L.apply_rotary(q, pos, theta, cfg.partial_rotary_factor)
        k = L.apply_rotary(k, pos, theta, cfg.partial_rotary_factor)
        cache.append(layer, k, v, positions=pos)
        kk, vv, kpos = cache.get(layer)
        sink = None
        if kind == SWA and cfg.add_swa_attention_sink_bias:
            sink = self.w.get(w_name(layer, "self_attn.attention_sink_bias"))
        out = L.attention(q, kk, vv, q_pos=pos, k_pos=kpos, window=window,
                          sink_bias=sink, value_scale=cfg.attention_value_scale)
        out = out.reshape(-1, Hq * Dv)
        return L.linear(out, self.w[w_name(layer, "self_attn.o_proj.weight")])

    def _ffn_block(self, layer: int, x: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        if cfg.moe_layer_freq[layer] != MOE_FFN:
            return L.dense_ffn(x,
                               self.w[w_name(layer, "mlp.gate_proj.weight")],
                               self.w[w_name(layer, "mlp.up_proj.weight")],
                               self.w[w_name(layer, "mlp.down_proj.weight")])
        gate = self.w[w_name(layer, "mlp.gate.weight")]
        # noaux_tc selection bias. Real checkpoint name (verified vs the 73k-tensor
        # index): model.layers.N.mlp.gate.e_score_correction_bias. Dropping it silently
        # changes top-k selection, so this is fail-loud (correctness review 2026-09-22).
        bias = self.w.get(w_name(layer, "mlp.gate.e_score_correction_bias"))
        if bias is None:
            raise KeyError(f"missing noaux_tc router bias "
                           f"{w_name(layer, 'mlp.gate.e_score_correction_bias')}")
        idx, wts = L.router(x, gate, top_k=cfg.num_experts_per_tok, bias=bias,
                            scoring=cfg.scoring_func, norm_topk=cfg.norm_topk_prob)

        def expert_fn(e: int, rows: np.ndarray) -> np.ndarray:
            return L.dense_ffn(rows,
                               self.w[w_name(layer, f"mlp.experts.{e}.gate_proj.weight")],
                               self.w[w_name(layer, f"mlp.experts.{e}.up_proj.weight")],
                               self.w[w_name(layer, f"mlp.experts.{e}.down_proj.weight")])

        return L.moe_forward(x, idx, wts, expert_fn)

    # ------------------------------------------------------------------

    def forward(self, token_ids: Sequence[int], *, cache: KVCache | None = None,
                start_pos: int | None = None, capture_layers: Sequence[int] = ()) -> ModelOutput:
        cfg = self.cfg
        ids = np.asarray(token_ids, dtype=np.int64)
        T = ids.shape[0]
        if cache is None:
            cache = KVCache(cfg)
        if start_pos is None:
            start_pos = cache.tokens  # incremental decode must continue the cache (silent-garbage fix)
        pos = np.arange(start_pos, start_pos + T)
        h = self.w["embed_tokens.weight"][ids]
        captured: dict[int, np.ndarray] = {}
        for layer in range(cfg.num_hidden_layers):
            x = L.rmsnorm(h, self.w[w_name(layer, "input_layernorm.weight")], cfg.layernorm_epsilon)
            h = h + self._attn_block(layer, x, pos, cache)
            x = L.rmsnorm(h, self.w[w_name(layer, "post_attention_layernorm.weight")], cfg.layernorm_epsilon)
            h = h + self._ffn_block(layer, x)
            if layer in capture_layers:
                captured[layer] = h
        cache.bump(T)
        hidden = h
        logits = L.linear(L.rmsnorm(hidden, self.w["norm.weight"], cfg.layernorm_epsilon),
                          self.w["lm_head.weight"])
        return ModelOutput(logits=logits, hidden=hidden, captured=captured, cache=cache)


# ---------------------------------------------------------------------------
# deterministic dummy weights (ARCHITECTURE.md §9 — synthetic in the real shapes)
# ---------------------------------------------------------------------------

def init_weights(cfg: MiMoConfig, seed: int = 0, scale: float = 0.02) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    hid = cfg.hidden_size

    def n(*shape):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    w: dict[str, np.ndarray] = {
        "embed_tokens.weight": n(cfg.vocab_size, hid),
        "lm_head.weight": n(cfg.vocab_size, hid),
        "norm.weight": np.ones(hid, dtype=np.float32),
    }
    for layer in range(cfg.num_hidden_layers):
        kind = cfg.hybrid_layer_pattern[layer]
        q_rows, k_rows, v_rows, o_in = cfg.attn_dims(kind)
        w[w_name(layer, "input_layernorm.weight")] = np.ones(hid, np.float32)
        w[w_name(layer, "post_attention_layernorm.weight")] = np.ones(hid, np.float32)  # backbone spelling (pre_mlp_layernorm is MTP-only)
        w[w_name(layer, "self_attn.qkv_proj.weight")] = n(q_rows + k_rows + v_rows, hid)
        w[w_name(layer, "self_attn.o_proj.weight")] = n(hid, o_in)
        if kind == SWA and cfg.add_swa_attention_sink_bias:
            w[w_name(layer, "self_attn.attention_sink_bias")] = np.zeros(
                cfg.num_attention_heads, np.float32)
        if cfg.moe_layer_freq[layer] == MOE_FFN:
            w[w_name(layer, "mlp.gate.weight")] = n(cfg.n_routed_experts, hid)
            w[w_name(layer, "mlp.gate.e_score_correction_bias")] = np.zeros(cfg.n_routed_experts, np.float32)
            for e in range(cfg.n_routed_experts):
                w[w_name(layer, f"mlp.experts.{e}.gate_proj.weight")] = n(cfg.moe_intermediate_size, hid)
                w[w_name(layer, f"mlp.experts.{e}.up_proj.weight")] = n(cfg.moe_intermediate_size, hid)
                w[w_name(layer, f"mlp.experts.{e}.down_proj.weight")] = n(hid, cfg.moe_intermediate_size)
        else:
            w[w_name(layer, "mlp.gate_proj.weight")] = n(cfg.intermediate_size, hid)
            w[w_name(layer, "mlp.up_proj.weight")] = n(cfg.intermediate_size, hid)
            w[w_name(layer, "mlp.down_proj.weight")] = n(hid, cfg.intermediate_size)
    return w
