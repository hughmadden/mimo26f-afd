//! `mimo26-attn` — coordinator attention + KV foundation (I3-A5, A5 kernels + A10
//! gates per `docs/ADVISOR-I3.md` §3, `ARCHITECTURE.md` §11.7/§11.9).
//!
//! # What this crate is
//!
//! 1. A **golden-locked Rust twin** of `oracle/mimo26` attention numerics
//!    (I-Gold: the byte-verified CPU twin is CONSUMED read-only by
//!    `tests/oracle_driver.py` — parity tests compare against it, never against
//!    this crate's own output). Every kernel decomposition tested here (paged
//!    reads, chunked prefill, split-KV decode + reduce) is pinned equivalent to
//!    the monolithic oracle math before any CUDA is trusted.
//! 2. The **layout + gate contracts** the CUDA kernels in `kernels/` implement:
//!    FP8 KV unit-scale / per-token×head layout (T20), `v_scale` BEFORE caching
//!    (T18), FP32 on-the-fly partial RoPE with dual θ (T19/T7), paged GA KV
//!    (256-token pages), SWA-128 rings with `min(batch_pos) − window + 1`
//!    eviction (T8), per-Q-head sink `[64]` **SWA-only** (T6/c1), and the two
//!    AOT gates (A10: `aot_arch` sm_120 coord / sm_121 GB10,
//!    `aot_sm_count` 170/188/48).
//!
//! # Trap coverage (docs/COHERENCE-TRAPS.md §1/§6) — each with a NEGATIVE test
//!
//! | Trap | Where pinned | Naive misfeature |
//! |---|---|---|
//! | T3 GA must not inherit SWA window | `attn` + `cache` | [`NaiveBits::GA_WINDOWED`] |
//! | T4 QK 192 / V 128, no broadcast | `attn` | [`NaiveBits::SCALE_BY_DV`], [`NaiveBits::V_BROADCAST`] |
//! | T6 sink per-Q-head `[64]` | `attn` | [`NaiveBits::SINK_PER_KV`] |
//! | T7 partial rotary 64/192, dual θ | `rope` | [`NaiveBits::ROPE_FULL_WIDTH`], [`NaiveBits::ROPE_SINGLE_THETA`] |
//! | T8 SWA eviction `min(batch_pos)−window+1` | `cache` | [`NaiveBits::EVICT_KEEP_LAST`] |
//! | T9 `start_pos` honored | `attn`/`rope` | [`NaiveBits::POS_ZEROED`] |
//! | T18 `v_scale` BEFORE caching | `cache`/`fp8kv` | [`NaiveBits::VSCALE_AFTER_STORE`], [`NaiveBits::VSCALE_ON_READ`] |
//! | T19 RoPE angle precision at 1M | `rope` | [`NaiveBits::ROPE_TRUNC_TABLE`] |
//! | T20 FP8 KV scale layout | `fp8kv` | [`NaiveBits::BLOCK128_SHARED_SCALES`] |
//! | c1 sink is SWA-only (GA bitwise sink-free) | `attn` | [`NaiveBits::SINK_ON_GA`] |
//! | c2 `attn_scale(d_qk, d_v)` distinct params | `attn` | [`NaiveBits::SCALE_BY_DV`] |
//! | A10 two AOT gates, each negative-tested | `aot` | [`NaiveBits::AOT_MIXED_GATE`] |
//!
//! # Naive discipline (suite convention — see `mimo26-load` `tests/t1_t2_negatives.rs`)
//!
//! The **naive implementation** of each module is real code selected by
//! [`NaiveBits`] — never a test-side mock. [`naive_from_env`] reads
//! `MIMO26_SPIKE_NAIVE=1` (alias `MIMO26_ATTN_NAIVE=1`) and maps to
//! [`NaiveBits::ALL`] ("the naive implementation" = every misfeature at once,
//! as a bad port would have them). NEGATIVE tests call env-default entry points
//! (`bits_from_env()`), so they FAIL on the naive run and PASS on the correct
//! impl. Detection/attribution tests use single-bit oracles (e.g.
//! [`rope::angles_truncated_table_naive`]) and pass both runs. Classification is
//! listed in each `tests/*.rs` header.

pub mod aot;
pub mod attn;
pub mod cache;
pub mod cuda;
pub mod device;
pub mod ffi;
pub mod fp8kv;
pub mod geom;
pub mod rope;
pub mod serve;

use std::fmt;

/// Family code, mirroring `oracle/mimo26/config.py:15-16` (`hybrid_layer_pattern`:
/// 0 = global attention, 1 = sliding-window attention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Global (full causal) attention — 9 layers of the real model.
    Ga,
    /// Sliding-window attention — 39 layers of the real model.
    Swa,
}

/// Misfeature bits — the individual wrong choices a naive port makes. Unit
/// tests set single bits to attribute a trap; [`NaiveBits::ALL`] is "the naive
/// implementation".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NaiveBits(pub u32);

impl NaiveBits {
    pub const NONE: NaiveBits = NaiveBits(0);
    /// T19: angles from a truncated 32K cos/sin table (`pos % 32768`).
    pub const ROPE_TRUNC_TABLE: NaiveBits = NaiveBits(1 << 0);
    /// T7: rotate all `d_qk` dims instead of the partial 64.
    pub const ROPE_FULL_WIDTH: NaiveBits = NaiveBits(1 << 1);
    /// T7: one θ everywhere (the SWA 1e4 θ applied to GA layers too).
    pub const ROPE_SINGLE_THETA: NaiveBits = NaiveBits(1 << 2);
    /// c2/T4: `attn_scale` uses the V width (`d_v`) — the QK/V mixup
    /// (spike `tests/test_p101_addendum.py:234`).
    pub const SCALE_BY_DV: NaiveBits = NaiveBits(1 << 3);
    /// c1: the sink column applied on GA layers too.
    pub const SINK_ON_GA: NaiveBits = NaiveBits(1 << 4);
    /// T3: GA layers inherit the SWA window.
    pub const GA_WINDOWED: NaiveBits = NaiveBits(1 << 5);
    /// T6: sink bias broadcast per KV head instead of per Q head `[64]`.
    pub const SINK_PER_KV: NaiveBits = NaiveBits(1 << 6);
    /// T4: V broadcast across the QK width (shape-OK garbage).
    pub const V_BROADCAST: NaiveBits = NaiveBits(1 << 7);
    /// T18: `v_scale` applied AFTER caching (the FP8 codes pin the difference).
    pub const VSCALE_AFTER_STORE: NaiveBits = NaiveBits(1 << 8);
    /// T18: `v_scale` applied AGAIN on the read of an already-scaled cache.
    pub const VSCALE_ON_READ: NaiveBits = NaiveBits(1 << 9);
    /// T8: SWA eviction "keep the last window" instead of
    /// `min(batch_pos) − window + 1`.
    pub const EVICT_KEEP_LAST: NaiveBits = NaiveBits(1 << 10);
    /// T9: positions zeroed (`start_pos` ignored).
    pub const POS_ZEROED: NaiveBits = NaiveBits(1 << 11);
    /// T20: block-128 shared K/V scales over the flattened 192-dim K.
    pub const BLOCK128_SHARED_SCALES: NaiveBits = NaiveBits(1 << 12);
    /// amax clip check: silently clamp and report 0 clips.
    pub const SILENT_CLAMP: NaiveBits = NaiveBits(1 << 13);
    /// Split-KV: the sink counted once PER SPLIT instead of once at reduce.
    pub const SINK_PER_SPLIT: NaiveBits = NaiveBits(1 << 14);
    /// Chunked prefill: no running-max rescale between chunks.
    pub const NO_RUNNING_RESCALE: NaiveBits = NaiveBits(1 << 15);
    /// A10: the §8 mixed gate `aot_sm = 170 | 121` on one field.
    pub const AOT_MIXED_GATE: NaiveBits = NaiveBits(1 << 16);
    /// T9 (paged): `attention_paged` hardcodes 0-based key positions
    /// (`k_pos = 0..n_tokens`) instead of the caller's absolute `k_pos` — wrong
    /// for any paged cache that doesn't start at position 0.
    pub const PAGED_KPOS_ZERO_BASED: NaiveBits = NaiveBits(1 << 17);

    /// "The naive implementation": every misfeature at once.
    pub const ALL: NaiveBits = NaiveBits(u32::MAX);

    pub fn none() -> Self {
        Self::NONE
    }

    pub fn has(self, other: NaiveBits) -> bool {
        self.0 & other.0 != 0
    }

    pub fn of(misfeature: NaiveBits) -> Self {
        misfeature
    }
}

/// Suite convention: naive implementation selected by env `MIMO26_SPIKE_NAIVE=1`
/// (alias `MIMO26_ATTN_NAIVE=1`), exactly like `mimo26-load`'s `naive_from_env`.
pub fn naive_from_env() -> bool {
    std::env::var("MIMO26_SPIKE_NAIVE").map(|v| v == "1").unwrap_or(false)
        || std::env::var("MIMO26_ATTN_NAIVE").map(|v| v == "1").unwrap_or(false)
}

/// Env-default entry-point selector (NEGATIVE tests pass this through).
pub fn bits_from_env() -> NaiveBits {
    if naive_from_env() {
        NaiveBits::ALL
    } else {
        NaiveBits::NONE
    }
}

/// Attention-path failures — all fail loud (a silent mis-slice is word salad).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttnError {
    /// QK/V widths must stay distinct (T4): `d_qk = 192 ≠ d_v = 128`.
    ShapeMismatch { what: String },
    /// A sink was supplied for a layer family that has none (GA has no sink —
    /// `add_full_attention_sink_bias: false` in the checkpoint config).
    SinkNotSupported { family: String },
    /// Sink bias length must be `n_q` (per Q head, `[64]` on the real model).
    SinkLength { got: usize, expected: usize },
    /// Page/ring accounting violation (would mis-read silently).
    KvLayout { what: String },
}

impl fmt::Display for AttnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttnError::ShapeMismatch { what } => write!(f, "shape: {what}"),
            AttnError::SinkNotSupported { family } => {
                write!(f, "sink on {family} layer — GA must be bitwise sink-free (c1)")
            }
            AttnError::SinkLength { got, expected } => {
                write!(f, "sink bias length {got} != n_q {expected} (per-Q-head [64], T6)")
            }
            AttnError::KvLayout { what } => write!(f, "kv layout: {what}"),
        }
    }
}

impl std::error::Error for AttnError {}
