//! The scheduler-facing API surface in action: wire row shapes (what
//! `mimo26-wire` must carry), the per-step expert-byte stream, and the virtual
//! clock measurements. All MODEL.

use mimo26_lanesim::geom::ModelGeom;
use mimo26_lanesim::model::unique_experts_expected;
use mimo26_lanesim::rng::step_rows;
use mimo26_lanesim::{
    f32_to_bf16, FrameSeq, LaneSim, LaneSimConfig, RequestRow, ReturnFrame,
};

/// Kills: request-row byte accounting that drops the descriptor or scales
/// (would understate the coordinator->Spark stream).
#[test]
fn request_row_is_4360_b() {
    let rows = step_rows(0, 0, 1, &ModelGeom::REAL);
    let row: &RequestRow = &rows[0];
    assert_eq!(row.hidden.len(), 4096);
    assert_eq!(row.routes.len(), 8);
    assert_eq!(row.wire_bytes(), 4_360);
}

/// Kills: any compact-return sizing that is not one BF16 hidden per Spark per
/// token (8 KB asserted, per §3.3).
#[test]
fn compact_return_is_8192_b_per_spark_per_token() {
    let frame = ReturnFrame {
        seq: FrameSeq { rank: 0, row: 0, seq: 0 },
        partial: vec![0.0; 4096],
    };
    assert_eq!(frame.wire_bytes(), 8_192);
}

/// The full expert path, one C1 step (47 layers x 8 rows), measured on the
/// virtual clock. Kills: per-layer vs per-step byte bookkeeping errors (the
/// streamed bytes must equal sum_layers unique(U_l) x 3.34 MB), wrong stream
/// denominators (bytes must map to time at 273 GB/s x eff), and RTT accounting
/// that is not one boundary per layer.
#[test]
fn full_c1_step_byte_stream_and_virtual_clock() {
    let mut cfg = LaneSimConfig::model_default();
    cfg.kernel_efficiency = 1.0;
    cfg.seed = 4;
    let mut sim = LaneSim::new(cfg).unwrap();
    let r = sim.run_decode_step(8).unwrap();

    assert_eq!(r.layers, 47);
    assert_eq!(r.rows_per_step, 8);
    assert_eq!(r.label, "MODEL");

    let expect_bytes: u64 = r
        .unique_experts_per_layer
        .iter()
        .map(|&u| u as u64 * ModelGeom::REAL.quarter_slice_bytes())
        .sum();
    assert_eq!(r.bytes_per_spark_per_step, expect_bytes);
    assert_eq!(r.wire_rtt_ns, 47 * 40_000);

    // expert time == bytes / (273 GB/s x eff), up to per-layer ns ceiling
    let expect_ns = r.bytes_per_spark_per_step as f64 * 1e9 / 273e9;
    assert!((r.expert_stream_ns as f64 - expect_ns).abs() < 48.0);
    assert!(r.total_ns >= r.expert_stream_ns + r.wire_rtt_ns);

    // realized unique-expert counts land near the uniform-routing expectation
    let mean_u = r.unique_experts_per_layer.iter().sum::<usize>() as f64 / 47.0;
    assert!(
        (mean_u - unique_experts_expected(8)).abs() <= 8.0,
        "mean unique experts {mean_u} vs expectation {}",
        unique_experts_expected(8)
    );
}

/// The stream accounting identity on a collected submit: each Spark streams
/// unique-experts x quarter-slice bytes for the layer.
#[test]
fn submit_accounts_the_expert_byte_stream() {
    let mut sim = LaneSim::new(LaneSimConfig::model_default()).unwrap();
    let rows = step_rows(2, 0, 2, &ModelGeom::REAL);
    let t = sim.submit(0, rows).unwrap();
    let res = sim.collect(t).unwrap();
    assert_eq!(res.combined.len(), 2);
    for row in &res.combined {
        assert_eq!(row.len(), 4096); // -> 8,192 B BF16 on the wire
    }
    assert_eq!(
        res.acct.stream_bytes_per_rank,
        res.acct.unique_experts as u64 * ModelGeom::REAL.quarter_slice_bytes()
    );
    assert_eq!(res.label, "MODEL");
    assert!(sim.now_ns() > 0);
}

/// Kills: BF16 quantization that is lossier than the compact return promises
/// (the conservation tolerance is budgeted against one bf16 half-ulp).
#[test]
fn bf16_quantization_within_one_half_ulp() {
    assert_eq!(f32_to_bf16(1.0), 1.0);
    assert_eq!(f32_to_bf16(0.0), 0.0);
    for k in 1..2000u32 {
        let x = k as f32 * 0.017;
        let q = f32_to_bf16(x);
        assert!((q - x).abs() <= x.abs() / 256.0 + f32::EPSILON, "{x} -> {q}");
    }
}
