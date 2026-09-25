//! Bandwidth harness (I4 item 4) — achieved GB/s of the grouped expert GEMM.
//!
//! ADVISOR-I4 §3.1: "**Kernel bandwidth check** (<=10 min, one Spark window):
//! achieved GB/s of the grouped expert GEMM at M in {1, 2, 4, 8, 16} tokens per
//! expert and at M ~ 64 (prefill), against 273 GB/s. **Target >= 70% at
//! M <= 8.** Below 60%: stop and redesign the kernel before I5. Do not build
//! the scheduler on a slow kernel."
//!
//! # What this module is
//!
//! The **pure** half of the harness: the byte/FLOP accounting, the verdict
//! thresholds, and the report formatting. The GPU half is
//! `kernels/parity/gemm_bench.cu` (times the kernel) driven by
//! `tests/gpu/run_gpu_gemm.sh`; the Rust side turns its timings into a verdict
//! and a printed table. Splitting it this way means the thresholds and the
//! arithmetic are unit-tested on CPU (no GPU needed) and the GPU cell only has
//! to produce honest timings.
//!
//! # The number
//!
//! Spark LPDDR5x peak is **273 GB/s** (ADVISOR-I4 §3.1). The kernel is
//! bandwidth-bound: at M = 1 the arithmetic intensity is ~2 FLOP/byte, so the
//! achieved GB/s is `streamed_bytes / elapsed`. `streamed_bytes` counts the
//! expert slice bytes actually read (payload + scales), the token rows read,
//! and the output rows written — see [`crate::grouped::streamed_bytes`].
//!
//! # Verdicts
//!
//! | achieved | verdict |
//! |---|---|
//! | >= 70% at M <= 8 | [`Verdict::Target`] — the §3.1 target |
//! | 60%..70% at M <= 8 | [`Verdict::BelowTarget`] — usable, not the target |
//! | < 60% at M <= 8 | [`Verdict::Stop`] — a **documented STOP**, not a pass |
//! | any at M > 8 | [`Verdict::Prefill`] — reported, no §3.1 threshold |
//!
//! A STOP is reported with the number, never rationalised (ADVISOR-I4 §3.1).

use crate::grouped::{self, GroupedPlan};
use crate::slice::Proj;

/// Spark LPDDR5x peak bandwidth, GB/s (ADVISOR-I4 §3.1).
pub const SPARK_PEAK_GBPS: f64 = 273.0;

/// The §3.1 target: >= 70% of peak at M <= 8.
pub const TARGET_FRACTION: f64 = 0.70;

/// The §3.1 STOP line: below 60% at M <= 8 is a documented STOP.
pub const STOP_FRACTION: f64 = 0.60;

/// The M sizes the harness reports (§3.1: M in {1,2,4,8,16} and M ~ 64).
pub const BENCH_M: [usize; 6] = [1, 2, 4, 8, 16, 64];

/// The M sizes the >= 70% target applies to.
pub const TARGET_M_MAX: usize = 8;

/// One measured row of the harness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchRow {
    /// Tokens per expert.
    pub m: usize,
    /// Experts in the group.
    pub experts: usize,
    /// Bytes streamed (weights + activations + output).
    pub bytes: u64,
    /// FLOPs performed.
    pub flops: u64,
    /// Measured elapsed seconds.
    pub seconds: f64,
}

impl BenchRow {
    /// Achieved GB/s (decimal GB: 1e9 bytes/s).
    pub fn gbps(&self) -> f64 {
        if self.seconds <= 0.0 {
            return 0.0;
        }
        self.bytes as f64 / self.seconds / 1e9
    }

    /// Achieved fraction of the 273 GB/s peak.
    pub fn fraction(&self) -> f64 {
        self.gbps() / SPARK_PEAK_GBPS
    }

    /// Achieved TFLOP/s.
    pub fn tflops(&self) -> f64 {
        if self.seconds <= 0.0 {
            return 0.0;
        }
        self.flops as f64 / self.seconds / 1e12
    }

    /// Arithmetic intensity, FLOP/byte.
    pub fn intensity(&self) -> f64 {
        if self.bytes == 0 {
            return 0.0;
        }
        self.flops as f64 / self.bytes as f64
    }

    /// The §3.1 verdict for this row.
    pub fn verdict(&self) -> Verdict {
        let f = self.fraction();
        if self.m > TARGET_M_MAX {
            Verdict::Prefill
        } else if f >= TARGET_FRACTION {
            Verdict::Target
        } else if f >= STOP_FRACTION {
            Verdict::BelowTarget
        } else {
            Verdict::Stop
        }
    }
}

/// The §3.1 verdict for one row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// >= 70% at M <= 8 — the target.
    Target,
    /// 60%..70% at M <= 8 — usable, below the target.
    BelowTarget,
    /// < 60% at M <= 8 — a documented STOP.
    Stop,
    /// M > 8 — reported, no §3.1 threshold.
    Prefill,
}

impl Verdict {
    /// The one-word label the harness prints.
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Target => "TARGET",
            Verdict::BelowTarget => "BELOW-TARGET",
            Verdict::Stop => "STOP",
            Verdict::Prefill => "prefill",
        }
    }
}

/// Build the bench rows for one projection at every M in [`BENCH_M`].
///
/// `experts` is the group size (the number of resident expert slices the kernel
/// streams). The §3.1 model uses 256 unique experts/layer for prefill and 57.4
/// for C1 decode; the harness takes it as a parameter so both can be reported.
pub fn plan_rows(p: Proj, experts: usize) -> Vec<(usize, GroupedPlan, u64, u64)> {
    BENCH_M
        .iter()
        .map(|&m| {
            let plan = GroupedPlan::uniform(experts, m, 0);
            let bytes = grouped::streamed_bytes(&plan, p, m);
            let flops = grouped::gemm_flops(&plan, p);
            (m, plan, bytes, flops)
        })
        .collect()
}

/// The full expert FFN's rows (gate + up + down) — the number that matters for
/// the §3.1 decode model.
pub fn ffn_rows(experts: usize) -> Vec<(usize, GroupedPlan, u64, u64)> {
    BENCH_M
        .iter()
        .map(|&m| {
            let plan = GroupedPlan::uniform(experts, m, 0);
            let bytes = grouped::ffn_streamed_bytes(&plan, m);
            let flops = grouped::ffn_flops(&plan);
            (m, plan, bytes, flops)
        })
        .collect()
}

/// The overall verdict for a set of rows: the worst verdict at M <= 8.
///
/// A STOP anywhere at M <= 8 makes the whole run a STOP — the §3.1 rule is
/// "below 60% at M <= 8 is a documented STOP", not "on average".
pub fn overall(rows: &[BenchRow]) -> Verdict {
    let mut worst = Verdict::Prefill;
    for r in rows.iter().filter(|r| r.m <= TARGET_M_MAX) {
        let v = r.verdict();
        worst = match (worst, v) {
            (Verdict::Stop, _) | (_, Verdict::Stop) => Verdict::Stop,
            (Verdict::BelowTarget, _) | (_, Verdict::BelowTarget) => Verdict::BelowTarget,
            (Verdict::Target, _) | (_, Verdict::Target) => Verdict::Target,
            _ => Verdict::Prefill,
        };
    }
    worst
}

/// Format the harness report (the receipt body).
///
/// `gpu` and `arch` are the identity fields AGENTS.md §4.2 requires with every
/// number ("Report GPU, dtype, shape, command, and commit with every number").
pub fn report(gpu: &str, arch: &str, rows: &[BenchRow]) -> String {
    let mut s = String::new();
    s.push_str("mimo26-expert grouped MXFP4 expert GEMM — bandwidth check (ADVISOR-I4 §3.1)\n");
    s.push_str(&format!("gpu: {gpu}\narch: {arch}\n"));
    s.push_str(&format!(
        "peak: {SPARK_PEAK_GBPS:.0} GB/s (Spark LPDDR5x) | target >= {:.0}% at M <= {TARGET_M_MAX} | STOP < {:.0}%\n",
        TARGET_FRACTION * 100.0,
        STOP_FRACTION * 100.0
    ));
    s.push_str("dtype: MXFP4 weights (E2M1+E8M0-32), f32 accumulate, f32 out\n");
    s.push_str(
        "shape: quarter slice gate/up [512, 4096] (in 4096), down [4096, 512] (in 512, layout v2)\n\n",
    );
    s.push_str("     M   experts        bytes        GB/s    %peak    TFLOP/s   FLOP/B  verdict\n");
    for r in rows {
        s.push_str(&format!(
            "{:>6} {:>9} {:>12} {:>11.1} {:>7.1}% {:>10.2} {:>8.2}  {}\n",
            r.m,
            r.experts,
            r.bytes,
            r.gbps(),
            r.fraction() * 100.0,
            r.tflops(),
            r.intensity(),
            r.verdict().label()
        ));
    }
    let v = overall(rows);
    s.push_str(&format!("\nOVERALL (M <= {TARGET_M_MAX}): {}\n", v.label()));
    match v {
        Verdict::Stop => s.push_str(
            "STOP: below 60% at M <= 8 — redesign the kernel before I5 (ADVISOR-I4 §3.1). \
             This is a documented STOP, not a pass.\n",
        ),
        Verdict::BelowTarget => s.push_str(
            "Below the 70% target but above the 60% STOP line: usable, not the §3.1 target.\n",
        ),
        Verdict::Target => s.push_str("Target met (>= 70% at M <= 8).\n"),
        Verdict::Prefill => s.push_str("No M <= 8 rows measured.\n"),
    }
    s
}

/// The §3.1 model's decode-time floor for one Spark, in milliseconds.
///
/// `bytes` is the per-Spark per-step expert bytes (ADVISOR-I4 §3.1 table:
/// 9.0 GB at C1/DFlash k=7, 40.2 GB at prefill chunk 2048). At 100% of peak the
/// floor is `bytes / 273 GB/s`; the table's 75%/60% columns are this divided by
/// the fraction.
pub fn floor_ms(bytes: u64, fraction: f64) -> f64 {
    if fraction <= 0.0 {
        return f64::INFINITY;
    }
    bytes as f64 / (SPARK_PEAK_GBPS * 1e9 * fraction) * 1e3
}

/// The §3.1 table's three columns for one byte count.
pub fn model_row(bytes: u64) -> (f64, f64, f64) {
    (floor_ms(bytes, 1.0), floor_ms(bytes, 0.75), floor_ms(bytes, 0.60))
}

/// The §3.1 resident expert bytes per Spark: 47 MoE layers x 256 experts x
/// 3.34 MB quarter slice = 40.2 GB.
pub fn resident_bytes_per_spark() -> u64 {
    let per_expert = (Proj::Gate.slice_payload_bytes()
        + Proj::Gate.slice_scale_bytes()
        + Proj::Up.slice_payload_bytes()
        + Proj::Up.slice_scale_bytes()
        + Proj::Down.slice_payload_bytes()
        + Proj::Down.slice_scale_bytes()) as u64;
    per_expert * 256 * 47
}
