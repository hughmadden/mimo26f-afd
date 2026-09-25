//! Negatives: every test here proves the sum-conservation check (or the loud
//! error paths) KILLS a specific wrong implementation. AGENTS §4.5: the test
//! must fail on the wrong implementation.

use mimo26_lanesim::geom::ModelGeom;
use mimo26_lanesim::model::WIRE_RTT_NS;
use mimo26_lanesim::rng::step_rows;
use mimo26_lanesim::{
    reference_expert_sum, CollectMode, ConfigError, ConError, LaneFaults, LaneSim, LaneSimConfig,
    RequestRow, SimError, StubBehavior,
};

const TOL: f64 = 0.05; // BF16 rounding lands < 1%; the wrong paths land > 80%

fn build(
    behavior: StubBehavior,
    mode: CollectMode,
    faults: Option<LaneFaults>,
) -> (LaneSim, Vec<RequestRow>, Vec<Vec<f64>>) {
    let mut cfg = LaneSimConfig::model_default();
    cfg.seed = 11;
    cfg.behavior = behavior;
    cfg.collect_mode = mode;
    let sim = match faults {
        Some(f) => LaneSim::with_faults(cfg, f).unwrap(),
        None => LaneSim::new(cfg).unwrap(),
    };
    let rows = step_rows(11, 3, 3, &ModelGeom::REAL);
    let expected = reference_expert_sum(11, 3, &rows);
    (sim, rows, expected)
}

/// The spec'd negative (ADVISOR-I4 packet item 4): **a sim that forgets the
/// pre-sum must fail a sum-conservation test**. `ForgotPreSum` returns one
/// route's partial instead of the weighted pre-sum over the 8 routes;
/// `Unweighted` drops the route weights. Both must trip the checker, while the
/// correct compact return passes the very same check.
#[test]
fn wrong_pre_sum_behaviors_fail_conservation() {
    for behavior in [StubBehavior::ForgotPreSum, StubBehavior::Unweighted] {
        let (mut sim, rows, expected) = build(behavior, CollectMode::DedupBySeq, None);
        let t = sim.submit(3, rows).unwrap();
        let res = sim.collect(t).unwrap();
        assert!(
            matches!(res.check_sum_conservation(&expected, TOL), Err(ConError::Deviation { .. })),
            "{behavior:?} must violate sum-conservation"
        );
    }
    let (mut sim, rows, expected) = build(StubBehavior::Correct, CollectMode::DedupBySeq, None);
    let t = sim.submit(3, rows).unwrap();
    let res = sim.collect(t).unwrap();
    assert!(res.check_sum_conservation(&expected, TOL).is_ok());
}

/// The spec'd negative (packet item 4): **duplicate frame handling must not
/// double-count**. Every frame arrives twice (L4 duplicate injection); the
/// dedup-by-sequence collect counts each key once and conserves the sum.
#[test]
fn duplicate_frames_do_not_double_count() {
    let faults = LaneFaults::new(4, 0.0, 1.0, 5).unwrap();
    let (mut sim, rows, expected) = build(StubBehavior::Correct, CollectMode::DedupBySeq, Some(faults));
    let t = sim.submit(3, rows).unwrap();
    let res = sim.collect(t).unwrap();
    assert_eq!(res.frames_received, 3 * 4); // unique keys: 3 rows x 4 ranks
    assert_eq!(res.duplicates_ignored, 3 * 4); // every key delivered twice
    assert!(res.check_sum_conservation(&expected, TOL).is_ok());
}

/// The meta-negative: the naive collect (sums every delivered copy) must FAIL
/// the same conservation check — proof the checker bites double-counting and
/// the dedup above is load-bearing.
#[test]
fn naive_collect_double_counts_and_the_checker_kills_it() {
    let faults = LaneFaults::new(4, 0.0, 1.0, 5).unwrap();
    let (mut sim, rows, expected) =
        build(StubBehavior::Correct, CollectMode::NaiveNoDedup, Some(faults));
    let t = sim.submit(3, rows).unwrap();
    let res = sim.collect(t).unwrap();
    assert_eq!(res.duplicates_ignored, 0); // the naive collect sees nothing wrong
    assert!(matches!(res.check_sum_conservation(&expected, TOL), Err(ConError::Deviation { .. })));
}

/// Kills: a collect that silently assembles whatever arrived. A lost return
/// frame is detected by its sequence gap and fails loud (L4: "detected and
/// retried or fails loud, and none produces a silent wrong sum").
#[test]
fn lost_frame_fails_loud_never_a_silent_partial_sum() {
    let faults = LaneFaults::new(4, 0.5, 0.0, 9).unwrap();
    let (mut sim, rows, _expected) = build(StubBehavior::Correct, CollectMode::DedupBySeq, Some(faults));
    let t = sim.submit(3, rows).unwrap();
    assert!(matches!(sim.collect(t), Err(SimError::FrameLost { .. })));
}

/// Kills: kernel efficiency accepted outside the modelled band [0.5, 1.0], or
/// applied in the wrong direction in the sim's streaming clock.
#[test]
fn kernel_efficiency_band_is_enforced_and_scales_the_clock() {
    for bad in [0.4, 1.1, 0.0] {
        let mut cfg = LaneSimConfig::model_default();
        cfg.kernel_efficiency = bad;
        assert!(matches!(
            LaneSim::new(cfg),
            Err(ConfigError::KernelEfficiencyOutOfRange(_))
        ));
    }
    let ns_at = |eff: f64| {
        let mut cfg = LaneSimConfig::model_default();
        cfg.kernel_efficiency = eff;
        let mut s = LaneSim::new(cfg).unwrap();
        s.run_decode_step(2).unwrap().expert_stream_ns
    };
    let full = ns_at(1.0);
    let half = ns_at(0.5);
    assert!((half as f64 / full as f64 - 2.0).abs() < 0.01);
}

/// Kills: wire RTT forgotten from the step, or applied per token instead of per
/// layer boundary (47 x 40 us on the C1 step's critical path).
#[test]
fn wire_rtt_accounted_once_per_layer() {
    let mut s = LaneSim::new(LaneSimConfig::model_default()).unwrap();
    let report = s.run_decode_step(2).unwrap();
    assert_eq!(report.wire_rtt_ns, 47 * WIRE_RTT_NS);
    assert_eq!(report.label, "MODEL");
}

/// Kills: silent acceptance of malformed request rows (wrong hidden width,
/// wrong route count, duplicate tokens) — submit must fail loud.
#[test]
fn submit_validates_row_shape_loud() {
    let mut s = LaneSim::new(LaneSimConfig::model_default()).unwrap();
    let good = step_rows(11, 3, 1, &ModelGeom::REAL);

    let mut short = good.clone();
    short[0].hidden.truncate(64);
    assert!(matches!(
        s.submit(3, short),
        Err(SimError::Config(ConfigError::HiddenLen { .. }))
    ));

    let mut few_routes = good.clone();
    few_routes[0].routes.truncate(4);
    assert!(matches!(
        s.submit(3, few_routes),
        Err(SimError::Config(ConfigError::RouteCount { .. }))
    ));

    let mut dup = good.clone();
    dup.push(dup[0].clone());
    assert!(matches!(
        s.submit(3, dup),
        Err(SimError::Config(ConfigError::DuplicateToken(_)))
    ));

    assert!(matches!(
        s.submit(3, Vec::new()),
        Err(SimError::Config(ConfigError::EmptyRows))
    ));
}

/// Ported ds41rt lane-sim shape (design-shape reuse; deltas in the receipt):
/// same-seed determinism, lossless never drops, injected loss tracks the rate,
/// combine accounting, loud validation.
#[test]
fn lane_faults_shape_ported_from_ds41rt() {
    let mut a = LaneFaults::new(4, 0.3, 0.0, 5).unwrap();
    let mut b = LaneFaults::new(4, 0.3, 0.0, 5).unwrap();
    let pa: Vec<bool> = (0..100).map(|_| !a.dispatch(0u8).is_empty()).collect();
    let pb: Vec<bool> = (0..100).map(|_| !b.dispatch(0u8).is_empty()).collect();
    assert_eq!(pa, pb);

    let mut lossless = LaneFaults::new(4, 0.0, 0.0, 1).unwrap();
    for i in 0..500u16 {
        assert_eq!(lossless.dispatch((i % 256) as u8).len(), 1);
    }
    let st = lossless.stats();
    assert_eq!((st.sent, st.delivered, st.dropped, st.duplicated), (500, 500, 0, 0));

    let mut lossy = LaneFaults::new(4, 0.2, 0.0, 9).unwrap();
    for _ in 0..2000 {
        lossy.dispatch(0u8);
    }
    let st = lossy.stats();
    assert_eq!(st.sent, 2000);
    assert!((st.dropped as f64 / 2000.0 - 0.2).abs() < 0.05);

    let mut lanes = LaneFaults::new(8, 0.0, 0.0, 0).unwrap();
    assert_eq!(lanes.combine(3), 3);
    assert_eq!(lanes.combine(2), 5);
    assert_eq!(lanes.stats().combine_records, 5);
    assert_eq!(lanes.stats().lanes, 8);

    assert!(LaneFaults::new(0, 0.0, 0.0, 0).is_err());
    assert!(LaneFaults::new(4, 1.0, 0.0, 0).is_err());
    // duplicate_rate 1.0 is VALID (duplicate every frame — the L4 duplicate
    // injection extreme); out-of-range still fails loud.
    assert!(LaneFaults::new(4, 0.0, 1.0, 0).is_ok());
    assert!(LaneFaults::new(4, 0.0, 1.0001, 0).is_err());
    assert!(LaneFaults::new(4, 0.0, -0.1, 0).is_err());
}
