//! Attention geometry + KV byte pins (ARCHITECTURE.md §11.7/§11.11, ADVISOR-I3
//! §3 A5/A1). One source of truth for dims, family gating, and the pool byte
//! arithmetic the loader/admission layers depend on (`ga_kv_bytes_per_token`
//! 11,520 B/token FP8 unit-scale is pinned in `tests/kvlayout_t20.rs`).
//!
//! Real-model constants come from `oracle/mimo26/config.py:38-53` (the
//! byte-verified CPU twin's config): 64 Q heads, 4 GA KV heads / 8 SWA KV heads,
//! QK 192 / V 128, window 128, `partial_rotary_factor` 0.334 → 64 dims, θ 1e7
//! (GA) / 1e4 (SWA), `attention_value_scale` 0.707, sink on SWA layers only
//! (`add_swa_attention_sink_bias: true`, `add_full_attention_sink_bias: false`).

use crate::Family;

/// QK head dim (GA and SWA) — T4: this is NOT the V width.
pub const D_QK: usize = 192;
/// V head dim — T4: `d_qk ≠ d_v` is first-class, no broadcast.
pub const D_V: usize = 128;
/// Q heads (both families).
pub const N_Q: usize = 64;
/// GA KV heads (9 GA layers).
pub const GA_KV_HEADS: usize = 4;
/// SWA KV heads (39 SWA layers).
pub const SWA_KV_HEADS: usize = 8;
/// SWA window (T3: GA layers must never inherit this).
pub const WINDOW: usize = 128;
/// Partial rotary width: `int(192 * 0.334) = 64` (T7).
pub const ROT_DIM: usize = 64;
/// GA rope θ (T19: angles up to position 1,048,575).
pub const ROPE_THETA: f64 = 10_000_000.0;
/// SWA rope θ (T7 dual-θ).
pub const SWA_ROPE_THETA: f64 = 10_000.0;
/// `attention_value_scale` — applied BEFORE caching (T18).
pub const VALUE_SCALE: f32 = 0.707;
/// Partial rotary factor (kept for config parity; ROT_DIM is the pinned width).
pub const PARTIAL_ROTARY_FACTOR: f64 = 0.334;
/// R-CTX: max positions (RoPE must stay exact to this).
pub const MAX_POSITION: i64 = 1_048_576;

/// GA layers in the real `hybrid_layer_pattern` (config.py:29-33): 9.
pub const N_GA: usize = 9;
/// SWA layers: 39.
pub const N_SWA: usize = 39;
/// GA KV page length in tokens (ADVISOR-I3 §3 A2).
pub const PAGE_TOKENS: usize = 256;

/// Per-layer-per-slot KV storage mode (ARCHITECTURE.md §11.12: unit-scale FP8 KV
/// first — exactly 11,520 B/token, no scale bytes — with the per-token×head
/// layout pinned as the extension, T20).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleMode {
    /// FP8 E4M3, scale = 1.0, no scale bytes (the vLLM working path; §11.12).
    Unit,
    /// FP8 E4M3 with one f32 scale per token × KV head, **K and V in separate
    /// scale planes** (T20 — never block-128, which does not divide 192).
    PerTokenHead,
}

/// Attention layer spec — the contract every kernel launch is derived from.
#[derive(Clone, Debug, PartialEq)]
pub struct AttnSpec {
    pub family: Family,
    pub n_q: usize,
    pub n_kv: usize,
    pub d_qk: usize,
    pub d_v: usize,
    /// SWA window; informational for GA (the family gate [`AttnSpec::window_gated`]
    /// is what the math uses).
    pub window: usize,
    pub theta: f64,
    pub partial_rotary_factor: f64,
    pub value_scale: f32,
    /// Whether this family carries a sink bias at all (SWA only).
    pub sink_enabled: bool,
}

impl AttnSpec {
    /// Real MiMo-V2.6 dims for one family (config.py:38-53).
    pub fn real(family: Family) -> Self {
        let (n_kv, theta) = match family {
            Family::Ga => (GA_KV_HEADS, ROPE_THETA),
            Family::Swa => (SWA_KV_HEADS, SWA_ROPE_THETA),
        };
        AttnSpec {
            family,
            n_q: N_Q,
            n_kv,
            d_qk: D_QK,
            d_v: D_V,
            window: WINDOW,
            theta,
            partial_rotary_factor: PARTIAL_ROTARY_FACTOR,
            value_scale: VALUE_SCALE,
            // config.py:50-51: SWA sink ON, full-attention sink OFF.
            sink_enabled: family == Family::Swa,
        }
    }

    /// GQA group width: 16 Q heads per KV head on GA, 8 on SWA (real dims).
    pub fn n_rep(&self) -> usize {
        self.n_q / self.n_kv
    }

    /// Partial rotary width (`int(d_qk * factor)`; 64 at 192 * 0.334).
    pub fn rot_dim(&self) -> usize {
        crate::rope::rot_dim(self.d_qk, self.partial_rotary_factor)
    }

    /// **Family gate (c1/T3).** GA has no window and no sink, ever — this is the
    /// single place that decides it, mirroring `spike/real_loop.py:281`
    /// (`window = self.window if is_swa else None`).
    pub fn window_gated(&self) -> Option<usize> {
        match self.family {
            Family::Ga => None,
            Family::Swa => Some(self.window),
        }
    }

    /// c1: a sink column exists only on SWA layers.
    pub fn sink_allowed(&self) -> bool {
        self.family == Family::Swa && self.sink_enabled
    }

    /// KV bytes for `n_tok` cached tokens at this layer (codes only, F32 mode
    /// excluded — see `fp8kv::kv_bytes_per_token`).
    pub fn kv_row_elems(&self) -> usize {
        self.n_kv * (self.d_qk + self.d_v)
    }
}

/// Pool byte arithmetic (ARCHITECTURE.md §7 pins; ADVISOR-I3 §3 A1/A2).
/// All figures are **model** numbers derived from the real config — pinned by
/// `tests/kvlayout_t20.rs::pool_byte_pins`.
pub mod bytes {
    use super::{ScaleMode, D_QK, D_V, GA_KV_HEADS, N_GA, N_SWA, PAGE_TOKENS, SWA_KV_HEADS, WINDOW};

    /// FP8 codes per token at one layer's KV heads: `n_kv * (d_qk + d_v)`.
    pub fn layer_kv_code_bytes(n_kv: usize) -> usize {
        n_kv * (D_QK + D_V)
    }

    /// f32 scale bytes per token at one layer, K and V separate (T20):
    /// `n_kv * 2 * 4`.
    pub fn layer_scale_bytes(n_kv: usize) -> usize {
        n_kv * 2 * std::mem::size_of::<f32>()
    }

    fn per_token(n_layers: usize, n_kv: usize, mode: ScaleMode) -> usize {
        let codes = n_layers * layer_kv_code_bytes(n_kv);
        match mode {
            ScaleMode::Unit => codes,
            ScaleMode::PerTokenHead => codes + n_layers * layer_scale_bytes(n_kv),
        }
    }

    /// GA KV per token across the 9 GA layers. FP8 unit-scale: exactly
    /// **11,520 B/token** (ADVISOR-I3 §3 A1). PerTokenHead: +288 B of scales.
    pub fn ga_kv_bytes_per_token(mode: ScaleMode) -> usize {
        per_token(N_GA, GA_KV_HEADS, mode)
    }

    /// One SWA ring per sequence across the 39 SWA layers (window rows each).
    /// FP8 unit-scale: exactly **12,779,520 B/seq** (ADVISOR-I3 §3 A1).
    pub fn swa_ring_bytes_per_seq(mode: ScaleMode) -> usize {
        per_token(N_SWA, SWA_KV_HEADS, mode) * WINDOW
    }

    /// One sealed GA page: `PAGE_TOKENS` tokens of the GA layers' KV.
    /// FP8 unit-scale: exactly **2,949,120 B** (ADVISOR-I3 §3 A2).
    pub fn ga_page_bytes(mode: ScaleMode) -> usize {
        ga_kv_bytes_per_token(mode) * PAGE_TOKENS
    }
}
