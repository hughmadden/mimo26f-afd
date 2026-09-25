//! Bandwidth harness — the §3.1 verdict logic and report (CPU, no GPU).
//!
//! ADVISOR-I4 §3.1: "achieved GB/s of the grouped expert GEMM at M in
//! {1, 2, 4, 8, 16} tokens per expert and at M ~ 64 (prefill), against
//! 273 GB/s. **Target >= 70% at M <= 8.** Below 60%: stop and redesign the
//! kernel before I5."
//!
//! The GPU half is `kernels/parity/gemm_parity.cu bench` (it times the kernel);
//! this file pins the arithmetic and the thresholds so the GPU cell only has to
//! produce honest timings.
//!
//! Classification: every test here is **BOTH RUNS** (the harness is not a trap).

mod common;

use mimo26_expert::bench::{
    self, floor_ms, model_row, overall, report, BenchRow, Verdict, BENCH_M, SPARK_PEAK_GBPS,
    STOP_FRACTION, TARGET_FRACTION, TARGET_M_MAX,
};
use mimo26_expert::grouped;
use mimo26_expert::slice::Proj;

/// BOTH RUNS. The §3.1 constants.
#[test]
fn section_3_1_constants_are_pinned() {
    assert_eq!(SPARK_PEAK_GBPS, 273.0);
    assert_eq!(TARGET_FRACTION, 0.70);
    assert_eq!(STOP_FRACTION, 0.60);
    assert_eq!(TARGET_M_MAX, 8);
    assert_eq!(BENCH_M, [1, 2, 4, 8, 16, 64]);
    assert_eq!(bench::BENCH_M, grouped::BENCH_M_SIZES);
}

/// BOTH RUNS. GB/s and the % of peak.
#[test]
fn gbps_and_fraction_are_the_definition() {
    let r = BenchRow {
        m: 1,
        experts: 256,
        bytes: 273_000_000_000,
        flops: 1_000_000_000,
        seconds: 1.0,
    };
    assert!((r.gbps() - 273.0).abs() < 1e-9);
    assert!((r.fraction() - 1.0).abs() < 1e-9);
    // Half the bytes in the same time is half the bandwidth.
    let r2 = BenchRow { bytes: 136_500_000_000, ..r };
    assert!((r2.fraction() - 0.5).abs() < 1e-9);
    // A zero-time row is 0, not inf/NaN.
    let r3 = BenchRow { seconds: 0.0, ..r };
    assert_eq!(r3.gbps(), 0.0);
    assert_eq!(r3.fraction(), 0.0);
}

/// BOTH RUNS. The verdict thresholds: >= 70% is TARGET, 60..70% is
/// BELOW-TARGET, < 60% is STOP, and M > 8 is prefill (no threshold).
#[test]
fn verdict_thresholds_are_the_section_3_1_rule() {
    let mk = |m: usize, frac: f64| BenchRow {
        m,
        experts: 256,
        bytes: (SPARK_PEAK_GBPS * frac * 1e9).round() as u64,
        flops: 1,
        seconds: 1.0,
    };
    assert_eq!(mk(1, 0.70).verdict(), Verdict::Target);
    assert_eq!(mk(1, 0.95).verdict(), Verdict::Target);
    assert_eq!(mk(8, 0.70).verdict(), Verdict::Target);
    assert_eq!(mk(1, 0.699).verdict(), Verdict::BelowTarget);
    assert_eq!(mk(8, 0.60).verdict(), Verdict::BelowTarget);
    assert_eq!(mk(1, 0.599).verdict(), Verdict::Stop);
    assert_eq!(mk(8, 0.10).verdict(), Verdict::Stop);
    // M > 8 has no §3.1 threshold.
    assert_eq!(mk(16, 0.10).verdict(), Verdict::Prefill);
    assert_eq!(mk(64, 0.99).verdict(), Verdict::Prefill);
}

/// BOTH RUNS. The overall verdict is the WORST at M <= 8 — a STOP anywhere is
/// a STOP, not an average.
#[test]
fn overall_verdict_is_the_worst_at_m_le_8() {
    let mk = |m: usize, frac: f64| BenchRow {
        m,
        experts: 256,
        bytes: (SPARK_PEAK_GBPS * frac * 1e9).round() as u64,
        flops: 1,
        seconds: 1.0,
    };
    assert_eq!(overall(&[mk(1, 0.9), mk(8, 0.8)]), Verdict::Target);
    assert_eq!(overall(&[mk(1, 0.9), mk(8, 0.65)]), Verdict::BelowTarget);
    assert_eq!(overall(&[mk(1, 0.9), mk(8, 0.59)]), Verdict::Stop);
    assert_eq!(overall(&[mk(1, 0.9), mk(4, 0.1), mk(8, 0.9)]), Verdict::Stop);
    // M > 8 rows do not affect the overall verdict.
    assert_eq!(overall(&[mk(1, 0.9), mk(64, 0.1)]), Verdict::Target);
    // No M <= 8 rows at all.
    assert_eq!(overall(&[mk(64, 0.9)]), Verdict::Prefill);
}

/// BOTH RUNS. The report prints the identity fields, the peak, the % of peak
/// and the verdict — the receipt body.
#[test]
fn report_prints_the_receipt_fields() {
    let rows = vec![
        BenchRow { m: 1, experts: 256, bytes: 273_000_000_000, flops: 1, seconds: 1.0 },
        BenchRow { m: 64, experts: 256, bytes: 273_000_000_000, flops: 1, seconds: 2.0 },
    ];
    let text = report("NVIDIA GeForce RTX 4090", "sm_89", &rows);
    assert!(text.contains("NVIDIA GeForce RTX 4090"), "gpu identity");
    assert!(text.contains("sm_89"), "arch identity");
    assert!(text.contains("273 GB/s"), "the peak");
    assert!(text.contains("70%"), "the target");
    assert!(text.contains("60%"), "the STOP line");
    assert!(text.contains("100.0%"), "the % of peak");
    assert!(text.contains("TARGET"), "the verdict");
    assert!(text.contains("OVERALL"), "the overall line");
    assert!(text.contains("MXFP4"), "the dtype");
}

/// BOTH RUNS. A STOP report says STOP and does not rationalise it.
#[test]
fn stop_report_is_a_documented_stop() {
    let rows = vec![BenchRow {
        m: 1,
        experts: 256,
        bytes: 273_000_000_000,
        flops: 1,
        seconds: 10.0, // 27.3 GB/s = 10% of peak
    }];
    let text = report("NVIDIA GeForce RTX 4090", "sm_89", &rows);
    assert!(text.contains("STOP"), "a < 60% run must print STOP");
    assert!(
        text.contains("documented STOP, not a pass"),
        "the STOP must be documented, not rationalised"
    );
    assert_eq!(overall(&rows), Verdict::Stop);
}

/// BOTH RUNS. The §3.1 model's floor arithmetic: 9.0 GB at C1 is 33 ms at 100%,
/// 44 ms at 75%, 55 ms at 60% (the table in ADVISOR-I4 §3.1).
#[test]
fn model_floor_matches_the_section_3_1_table() {
    let (f100, f75, f60) = model_row(9_000_000_000);
    assert!((f100 - 33.0).abs() < 0.5, "9.0 GB at 100% is ~33 ms, got {f100:.1}");
    assert!((f75 - 44.0).abs() < 0.7, "9.0 GB at 75% is ~44 ms, got {f75:.1}");
    assert!((f60 - 55.0).abs() < 0.9, "9.0 GB at 60% is ~55 ms, got {f60:.1}");
    // Prefill chunk 2048: 40.2 GB -> 147 / 196 / 246 ms.
    let (p100, p75, p60) = model_row(40_200_000_000);
    assert!((p100 - 147.0).abs() < 2.0, "40.2 GB at 100% is ~147 ms, got {p100:.1}");
    assert!((p75 - 196.0).abs() < 3.0, "40.2 GB at 75% is ~196 ms, got {p75:.1}");
    assert!((p60 - 246.0).abs() < 3.0, "40.2 GB at 60% is ~246 ms, got {p60:.1}");
    // A zero fraction is infinite, not a division by zero.
    assert!(floor_ms(1, 0.0).is_infinite());
}

/// BOTH RUNS. The harness's byte accounting for the bench rows: the weights
/// dominate at M = 1 (the whole point of the §3.1 model).
#[test]
fn bench_rows_are_bandwidth_dominated_at_small_m() {
    let rows = bench::plan_rows(Proj::Gate, 256);
    assert_eq!(rows.len(), 6);
    for (m, plan, bytes, flops) in &rows {
        assert_eq!(plan.expert_count(), 256);
        assert_eq!(plan.total_tokens(), 256 * m);
        assert_eq!(*bytes, grouped::streamed_bytes(&plan, Proj::Gate, *m));
        assert_eq!(*flops, grouped::gemm_flops(&plan, Proj::Gate));
    }
    // At M = 1 weights are 98.4% of logical traffic (1,114,112 / 1,132,544).
    let (_, _, bytes1, _) = rows[0];
    let weights = 256u64 * (1_048_576 + 65_536);
    assert!(
        weights as f64 / bytes1 as f64 > 0.98,
        "at M=1 the expert weights must dominate the streamed bytes"
    );
    // Arithmetic intensity at M = 1 is ~3.70 FLOP/byte (MXFP4, 2 FLOPs/weight).
    let (_, _, b1, f1) = rows[0];
    let intensity = f1 as f64 / b1 as f64;
    assert!(
        (3.6..3.8).contains(&intensity),
        "M=1 arithmetic intensity must be ~3.70 FLOP/byte, got {intensity:.2}"
    );
    // At M = 64 the intensity is ~64x higher (still bandwidth-bound, but the
    // weights are amortised).
    let (_, _, b64, f64_) = rows[5];
    let i64 = f64_ as f64 / b64 as f64;
    assert!(i64 > 20.0, "M=64 intensity must be much higher, got {i64:.2}");
}

/// BOTH RUNS. The FFN rows cover gate + up + down.
#[test]
fn ffn_rows_cover_all_three_projections() {
    let rows = bench::ffn_rows(256);
    assert_eq!(rows.len(), 6);
    for (m, plan, bytes, flops) in &rows {
        assert_eq!(*bytes, grouped::ffn_streamed_bytes(plan, *m));
        assert_eq!(*flops, grouped::ffn_flops(plan));
        let single = grouped::streamed_bytes(plan, Proj::Gate, *m)
            + grouped::streamed_bytes(plan, Proj::Up, *m)
            + grouped::streamed_bytes(plan, Proj::Down, *m);
        assert_eq!(*bytes, single);
    }
}
