//! The ADVISOR-I4 §3.1 arithmetic, analytic. **All MODEL** except the three
//! labelled MEASURED D7-bar constants at the bottom of the constant block —
//! those are quoted from `runs/20260923-d7-tp4-baseline/RESULT.md` and are the
//! bar the model is compared against, never model output.
//!
//! The §3.1 table (reproduced by [`table_cases`]) is the number that decides the
//! project: decode on both designs is bound by streaming expert bytes from Spark
//! LPDDR5x at 273 GB/s.

use crate::geom::ModelGeom;

/// LPDDR5x streaming bandwidth per Spark (the §3.1 denominator). MODEL.
pub const LPDDR5X_BYTES_PER_SEC: u64 = 273_000_000_000;
/// Per-rank wire link (Spark CX7 port trains Gen5 x4 ~126 Gb/s;
/// `bench/model/afd_vs_tp4_model.py::SPARK_LINK`). MODEL.
pub const SPARK_LINK_BYTES_PER_SEC: u64 = 15_750_000_000;
/// Wire RTT per layer boundary ("~40 us wire RTT", §3.1 modelled AFD step). MODEL.
pub const WIRE_RTT_NS: u64 = 40_000;
/// Coordinator attention per layer boundary ("~80 us coordinator attention"). MODEL.
pub const COORD_ATTN_NS: u64 = 80_000;
/// DFlash drafter per step. MODEL: inferred from the §3.1 rounded totals
/// (54/65/76 ms at 75/60/50% efficiency land at a ~4.3 ms drafter each).
pub const DRAFTER_NS: u64 = 4_300_000;
/// C1 DFlash k=7 = 8 tokens/step (1 target + 7 drafts). MODEL.
pub const C1_TOKENS_PER_STEP: usize = 8;

/// Mean DFlash acceptance over the 9 D7 categories, accepted tokens per step.
/// **MEASURED** (D7 bar).
pub const MEAN_DFLASH_ACCEPTANCE: f64 = 4.46;
/// vLLM TP4 D7 bar: 62 ms per C1 step. **MEASURED** (D7 bar).
pub const D7_STEP_MS: f64 = 62.0;
/// vLLM TP4 D7 bar: 71.6 tok/s per stream at C1. **MEASURED** (D7 bar).
pub const D7_TOK_PER_S: f64 = 71.6;

/// The kernel-efficiency levels the §3.1 table is quoted at: 100%, 75%, 60%.
pub const TABLE_EFFICIENCIES: [f64; 3] = [1.0, 0.75, 0.6];

/// Expected unique experts per layer under uniform top-8 routing over `rows`
/// routed rows: `N_EXP * (1 - (1 - TOPK/N_EXP)^rows)` (kappa = 1;
/// `bench/model/afd_vs_tp4_model.py::unique_experts`). MODEL.
pub fn unique_experts_expected(rows: usize) -> f64 {
    let g = ModelGeom::REAL;
    g.experts_per_layer as f64
        * (1.0 - (1.0 - g.top_k as f64 / g.experts_per_layer as f64).powi(rows as i32))
}

/// Streaming floor in ms: `bytes / (273 GB/s x kernel_efficiency)`. MODEL.
pub fn stream_floor_ms(bytes: u64, kernel_efficiency: f64) -> f64 {
    bytes as f64 / (LPDDR5X_BYTES_PER_SEC as f64 * kernel_efficiency) * 1e3
}

/// One §3.1 table row. MODEL (`label` says so on every instance).
#[derive(Clone, Debug, PartialEq)]
pub struct TableCase {
    /// Case name as printed in ADVISOR-I4 §3.1.
    pub name: &'static str,
    /// Routed rows per step (tokens x 1 row each): 8 / 48 / 128 / 2048.
    pub rows_per_step: usize,
    /// Unique experts per layer (uniform routing expectation).
    pub unique_experts: f64,
    /// Bytes streamed per Spark per step (u x 47 x 3.34 MB).
    pub bytes_per_spark_per_step: u64,
    /// Streaming floor in ms at [`TABLE_EFFICIENCIES`] = [100%, 75%, 60%].
    pub floor_ms: [f64; 3],
    /// Always `MODEL`.
    pub label: &'static str,
}

/// The ADVISOR-I4 §3.1 table, reproduced from the checkpoint geometry. MODEL.
pub fn table_cases() -> [TableCase; 4] {
    let g = ModelGeom::REAL;
    let mk = |name: &'static str, rows_per_step: usize| {
        let unique_experts = unique_experts_expected(rows_per_step);
        let bytes = g.bytes_per_spark_per_step(unique_experts);
        let floor_ms = TABLE_EFFICIENCIES.map(|e| stream_floor_ms(bytes, e));
        TableCase {
            name,
            rows_per_step,
            unique_experts,
            bytes_per_spark_per_step: bytes,
            floor_ms,
            label: crate::MODEL,
        }
    };
    [
        mk("C1, DFlash k=7 (8 tokens/step)", C1_TOKENS_PER_STEP),
        mk("C6 × 8", 6 * C1_TOKENS_PER_STEP),
        mk("C16 × 8", 16 * C1_TOKENS_PER_STEP),
        mk("Prefill chunk 2,048", 2048),
    ]
}

/// Modelled AFD C1 step (§3.1): 47 layers x (expert stream + 40 us wire RTT +
/// 80 us coordinator attention) + drafter. MODEL. Lands at 54 / 65 / 76 ms for
/// 75% / 60% / 50% kernel efficiency.
pub fn afd_c1_step_ms(kernel_efficiency: f64) -> f64 {
    let g = ModelGeom::REAL;
    let bytes = g.bytes_per_spark_per_step(unique_experts_expected(C1_TOKENS_PER_STEP));
    let expert = stream_floor_ms(bytes, kernel_efficiency);
    let boundary = g.moe_layers as f64 * (WIRE_RTT_NS + COORD_ATTN_NS) as f64 / 1e6;
    expert + boundary + DRAFTER_NS as f64 / 1e6
}

/// Tokens/s at mean DFlash acceptance over a step of `step_ms`. The AFD rows of
/// §3.1 count accepted tokens (4.46/step, MEASURED), not drafted positions.
pub fn toks_per_s(step_ms: f64) -> f64 {
    MEAN_DFLASH_ACCEPTANCE / (step_ms / 1e3)
}
