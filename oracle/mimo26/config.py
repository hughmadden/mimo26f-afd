"""MiMo-V2.6-Flash (`mimo_v2`) model configuration.

Single source of truth for hyper-parameters and derived geometry. Defaults are the
real XiaomiMiMo/MiMo-V2.6-Flash-RL values (see ARCHITECTURE.md §4); `MiMoConfig.tiny()`
returns a scaled-down, internally consistent twin for CPU tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

# hybrid_layer_pattern / moe_layer_freq codes
GA = 0  # global attention (full causal)
SWA = 1  # sliding-window attention
DENSE_FFN = 0
MOE_FFN = 1


@dataclass(frozen=True)
class MiMoConfig:
    # --- backbone ---
    vocab_size: int = 152576
    hidden_size: int = 4096
    intermediate_size: int = 16384  # dense FFN (layer 0 only)
    num_hidden_layers: int = 48
    # hybrid_layer_pattern: per-layer attention kind, 0=GA, 1=SWA (length = num_hidden_layers)
    hybrid_layer_pattern: tuple[int, ...] = (
        0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
    )
    # moe_layer_freq: per-layer FFN kind, 0=dense, 1=MoE (length = num_hidden_layers)
    moe_layer_freq: tuple[int, ...] = (0,) + (1,) * 47

    # --- attention ---
    num_attention_heads: int = 64
    num_key_value_heads: int = 4  # GA layers
    swa_num_key_value_heads: int = 8  # SWA layers
    head_dim: int = 192  # QK head dim (GA and SWA)
    v_head_dim: int = 128
    swa_head_dim: int = 192
    swa_v_head_dim: int = 128
    sliding_window: int = 128
    partial_rotary_factor: float = 0.334
    rope_theta: float = 10_000_000.0  # GA
    swa_rope_theta: float = 10_000.0  # SWA
    attention_value_scale: float = 0.707
    add_swa_attention_sink_bias: bool = True
    add_full_attention_sink_bias: bool = False
    layernorm_epsilon: float = 1e-6
    max_position_embeddings: int = 1_048_576

    # --- MoE ---
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    n_shared_experts: int = 0  # MiMo-V2.6 has no shared experts
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float | None = None

    # --- MTP ---
    num_nextn_predict_layers: int = 3

    # --- tokens / generation (generation_config.json wins on bos) ---
    bos_token_id: int = 151643
    eos_token_ids: tuple[int, ...] = (151643, 151645, 151672)
    pad_token_id: int = 151643
    mask_token_id: int = 151675  # DFlash mask embedding token

    # --- storage conventions observed in the checkpoint ---
    o_proj_storage: str = "bf16"  # every self_attn.o_proj is in quantization ignored_layers
    fused_qkv: bool = True  # attention_projection_layout == "fused_qkv"
    expert_quant: str = "mxfp4"  # E2M1 nibbles + E8M0 per-32 scales
    weight_quant: str = "fp8_e4m3_block128"
    mxfp4_block_size: int = 32
    fp8_block_size: int = 128

    def __post_init__(self) -> None:
        L = self.num_hidden_layers
        if len(self.hybrid_layer_pattern) != L:
            raise ValueError("hybrid_layer_pattern length must equal num_hidden_layers")
        if len(self.moe_layer_freq) != L:
            raise ValueError("moe_layer_freq length must equal num_hidden_layers")
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("top-k exceeds expert count")
        if self.sliding_window < 1:
            raise ValueError("sliding_window must be positive")
        if self.n_routed_experts < self.num_experts_per_tok or self.moe_intermediate_size < 1:
            raise ValueError("bad MoE geometry")

    # ---------- derived geometry ----------

    @property
    def ga_layer_ids(self) -> tuple[int, ...]:
        return tuple(i for i, k in enumerate(self.hybrid_layer_pattern) if k == GA)

    @property
    def swa_layer_ids(self) -> tuple[int, ...]:
        return tuple(i for i, k in enumerate(self.hybrid_layer_pattern) if k == SWA)

    @property
    def moe_layer_ids(self) -> tuple[int, ...]:
        return tuple(i for i, k in enumerate(self.moe_layer_freq) if k == MOE_FFN)

    @property
    def dense_layer_ids(self) -> tuple[int, ...]:
        return tuple(i for i, k in enumerate(self.moe_layer_freq) if k == DENSE_FFN)

    @property
    def n_ga(self) -> int:
        return len(self.ga_layer_ids)

    @property
    def n_swa(self) -> int:
        return len(self.swa_layer_ids)

    @property
    def n_moe(self) -> int:
        return len(self.moe_layer_ids)

    def attn_dims(self, kind: int) -> tuple[int, int, int, int]:
        """(q_rows, k_rows, v_rows, o_in) projection geometry for a GA or SWA layer."""
        if kind == GA:
            kv = self.num_key_value_heads
            hd, vhd = self.head_dim, self.v_head_dim
        else:
            kv = self.swa_num_key_value_heads
            hd, vhd = self.swa_head_dim, self.swa_v_head_dim
        q = self.num_attention_heads * hd
        k = kv * hd
        v = kv * vhd
        o_in = self.num_attention_heads * vhd  # o_proj is [hidden, o_in]
        return q, k, v, o_in

    @property
    def fused_qkv_rows_swa(self) -> int:
        q, k, v, _ = self.attn_dims(SWA)
        return q + k + v

    @property
    def fused_qkv_rows_ga(self) -> int:
        q, k, v, _ = self.attn_dims(GA)
        return q + k + v

    # ---------- memory arithmetic (ARCHITECTURE.md §7) ----------

    def ga_kv_bytes_per_token(self, dtype_bytes: int = 1) -> int:
        per = self.num_key_value_heads * (self.head_dim + self.v_head_dim)
        return self.n_ga * per * dtype_bytes

    def swa_ring_bytes_per_seq(self, dtype_bytes: int = 1) -> int:
        per = self.swa_num_key_value_heads * (self.swa_head_dim + self.swa_v_head_dim)
        return self.n_swa * per * self.sliding_window * dtype_bytes

    def expert_params_per_expert(self) -> int:
        # gate [moe_inter, hidden] + up [moe_inter, hidden] + down [hidden, moe_inter]
        return 3 * self.hidden_size * self.moe_intermediate_size

    def expert_params_total(self) -> int:
        return self.n_moe * self.n_routed_experts * self.expert_params_per_expert()

    def active_expert_params_per_token(self) -> int:
        return self.n_moe * self.num_experts_per_tok * self.expert_params_per_expert()

    def expert_bytes_total(self) -> int:
        """Stored size of all routed experts: 4-bit packed + E8M0 per-32 scales."""
        elems = self.expert_params_total()
        return elems // 2 + elems // self.mxfp4_block_size

    def attn_params_total(self) -> int:
        n = 0
        for kind in (GA, SWA):
            q, k, v, o_in = self.attn_dims(kind)
            layers = self.n_ga if kind == GA else self.n_swa
            n += layers * ((q + k + v) * self.hidden_size + self.hidden_size * o_in)
        return n

    def dense_ffn_params(self) -> int:
        return len(self.dense_layer_ids) * 3 * self.hidden_size * self.intermediate_size

    def expert_shard_bytes(self, n_experts: int) -> int:
        per = self.expert_params_per_expert()
        return n_experts * (per // 2 + per // self.mxfp4_block_size)

    # ---------- construction ----------

    @classmethod
    def from_dict(cls, d: dict) -> "MiMoConfig":
        """Build from a Hugging Face `config.json` dict (unknown keys ignored)."""
        pattern = d.get("hybrid_layer_pattern")
        freq = d.get("moe_layer_freq")
        kw = {}
        for f in (
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "swa_num_key_value_heads",
            "head_dim", "v_head_dim", "swa_head_dim", "swa_v_head_dim", "sliding_window",
            "partial_rotary_factor", "rope_theta", "swa_rope_theta",
            "attention_value_scale", "layernorm_epsilon", "max_position_embeddings",
            "moe_intermediate_size", "n_routed_experts", "num_experts_per_tok",
            "n_shared_experts", "scoring_func", "topk_method", "norm_topk_prob",
            "n_group", "topk_group", "routed_scaling_factor", "num_nextn_predict_layers",
            "mask_token_id", "add_swa_attention_sink_bias", "add_full_attention_sink_bias",
        ):
            if f in d:
                kw[f] = d[f]
        if pattern is not None:
            kw["hybrid_layer_pattern"] = tuple(pattern)
        if freq is not None:
            kw["moe_layer_freq"] = tuple(freq)
        if "eos_token_id" in d:
            eos = d["eos_token_id"]
            kw["eos_token_ids"] = tuple(eos) if isinstance(eos, list) else (eos,)
        if "bos_token_id" in d and d["bos_token_id"] is not None:
            kw["bos_token_id"] = d["bos_token_id"]
        if "pad_token_id" in d and d["pad_token_id"] is not None:
            kw["pad_token_id"] = d["pad_token_id"]
        q = d.get("quantization_config") or {}
        if q.get("store_dtype") == "mxfp4":
            kw["mxfp4_block_size"] = q.get("mxfp4_block_size", 32)
        return cls(**kw)

    @classmethod
    def from_file(cls, path: str | Path) -> "MiMoConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def tiny(cls, **overrides) -> "MiMoConfig":
        """Scaled-down, internally consistent config for CPU tests."""
        base = dict(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=32,
            num_hidden_layers=4,
            hybrid_layer_pattern=(GA, SWA, SWA, SWA),
            moe_layer_freq=(DENSE_FFN, MOE_FFN, MOE_FFN, MOE_FFN),
            num_attention_heads=8,
            num_key_value_heads=2,
            swa_num_key_value_heads=4,
            head_dim=32,
            v_head_dim=16,
            swa_head_dim=32,
            swa_v_head_dim=16,
            sliding_window=4,
            partial_rotary_factor=0.5,
            rope_theta=10_000.0,
            swa_rope_theta=10_000.0,
            max_position_embeddings=256,
            moe_intermediate_size=32,  # >= mxfp4 block 32 so expert matrices pack cleanly
            n_routed_experts=8,
            num_experts_per_tok=2,
            num_nextn_predict_layers=3,
        )
        base.update(overrides)
        return cls(**base)

    def replace(self, **overrides) -> "MiMoConfig":
        return replace(self, **overrides)
