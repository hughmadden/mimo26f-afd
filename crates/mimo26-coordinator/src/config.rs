//! Model configuration — the Rust twin of `oracle/mimo26/config.py` (I-Gold:
//! consumed read-only). Single source of truth for the MiMo-V2.6-Flash geometry
//! The coordinator serves. A1 F1: "config" is a coordinator-side spine item.

/// GA layer ids (hybrid_layer_pattern == 0): {0,5,11,17,23,29,35,41,47}.
pub const GA_LAYER_IDS: [usize; 9] = [0, 5, 11, 17, 23, 29, 35, 41, 47];
/// Dense-FFN layer ids (moe_layer_freq == 0): layer 0 only.
pub const DENSE_LAYER_IDS: [usize; 1] = [0];

/// Attention kind per layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Ga,
    Swa,
}

/// MiMo-V2.6-Flash configuration (real defaults; `tiny()` scales it down).
#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize, // dense FFN (layer 0 only)
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,     // GA layers
    pub swa_num_key_value_heads: usize, // SWA layers
    pub head_dim: usize,                // QK head dim (GA and SWA)
    pub v_head_dim: usize,
    pub sliding_window: usize,
    pub partial_rotary_factor: f32,
    pub rope_theta: f32,     // GA
    pub swa_rope_theta: f32, // SWA
    pub attention_value_scale: f32,
    pub add_swa_attention_sink_bias: bool,
    pub layernorm_epsilon: f32,
    pub max_position_embeddings: u64,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub bos_token_id: u32,
    pub eos_token_ids: [u32; 3],
    pub pad_token_id: u32,
    pub mask_token_id: u32,
    /// GA layer ids for this config's pattern (real = the fixed 9).
    pub ga_layer_ids: Vec<usize>,
    /// Dense-FFN layer ids (real = layer 0 only).
    pub dense_layer_ids: Vec<usize>,
}

impl Config {
    /// The real XiaomiMiMo/MiMo-V2.6-Flash-RL defaults.
    pub fn real() -> Self {
        Config {
            vocab_size: 152_576,
            hidden_size: 4096,
            intermediate_size: 16_384,
            num_hidden_layers: 48,
            num_attention_heads: 64,
            num_key_value_heads: 4,
            swa_num_key_value_heads: 8,
            head_dim: 192,
            v_head_dim: 128,
            sliding_window: 128,
            partial_rotary_factor: 0.334,
            rope_theta: 10_000_000.0,
            swa_rope_theta: 10_000.0,
            attention_value_scale: 0.707,
            add_swa_attention_sink_bias: true,
            layernorm_epsilon: 1e-6,
            max_position_embeddings: 1_048_576,
            moe_intermediate_size: 2048,
            n_routed_experts: 256,
            num_experts_per_tok: 8,
            norm_topk_prob: true,
            bos_token_id: 151_643,
            eos_token_ids: [151_643, 151_645, 151_672],
            pad_token_id: 151_643,
            mask_token_id: 151_675,
            ga_layer_ids: GA_LAYER_IDS.to_vec(),
            dense_layer_ids: DENSE_LAYER_IDS.to_vec(),
        }
    }

    /// A scaled-down, internally consistent twin for CPU tests (mirrors
    /// `MiMoConfig.tiny()`).
    pub fn tiny() -> Self {
        Config {
            vocab_size: 128,
            hidden_size: 64,
            intermediate_size: 32,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            swa_num_key_value_heads: 4,
            head_dim: 32,
            v_head_dim: 16,
            sliding_window: 4,
            partial_rotary_factor: 0.5,
            rope_theta: 10_000.0,
            swa_rope_theta: 10_000.0,
            attention_value_scale: 0.707,
            add_swa_attention_sink_bias: true,
            layernorm_epsilon: 1e-6,
            max_position_embeddings: 256,
            moe_intermediate_size: 32,
            n_routed_experts: 8,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
            bos_token_id: 151_643,
            eos_token_ids: [151_643, 151_645, 151_672],
            pad_token_id: 151_643,
            mask_token_id: 151_675,
            ga_layer_ids: vec![0],
            dense_layer_ids: vec![0],
        }
    }

    /// Attention kind for a layer (from this config's GA ids; the rest SWA).
    pub fn layer_kind(&self, layer: usize) -> LayerKind {
        if self.ga_layer_ids.contains(&layer) {
            LayerKind::Ga
        } else {
            LayerKind::Swa
        }
    }

    /// Whether a layer runs the routed MoE (layers outside `dense_layer_ids`).
    pub fn is_moe_layer(&self, layer: usize) -> bool {
        !self.dense_layer_ids.contains(&layer)
    }

    pub fn n_ga(&self) -> usize {
        self.ga_layer_ids.len()
    }

    pub fn n_swa(&self) -> usize {
        self.num_hidden_layers - self.ga_layer_ids.len()
    }

    /// `(q_rows, k_rows, v_rows, o_in)` projection geometry for a layer kind.
    pub fn attn_dims(&self, kind: LayerKind) -> (usize, usize, usize, usize) {
        let (kv, hd, vhd) = match kind {
            LayerKind::Ga => (self.num_key_value_heads, self.head_dim, self.v_head_dim),
            LayerKind::Swa => (self.swa_num_key_value_heads, self.head_dim, self.v_head_dim),
        };
        let q = self.num_attention_heads * hd;
        let k = kv * hd;
        let v = kv * vhd;
        let o_in = self.num_attention_heads * vhd;
        (q, k, v, o_in)
    }

    /// GA KV bytes per token (FP8, scales excluded): n_ga × kv × (QK + V).
    pub fn ga_kv_bytes_per_token(&self) -> u64 {
        let (_, k, v, _) = self.attn_dims(LayerKind::Ga);
        (self.n_ga() * (k + v)) as u64
    }

    /// SWA ring bytes per sequence (FP8): n_swa × kv × (QK + V) × window.
    pub fn swa_ring_bytes_per_seq(&self) -> u64 {
        let (_, k, v, _) = self.attn_dims(LayerKind::Swa);
        (self.n_swa() * (k + v) * self.sliding_window) as u64
    }
}
