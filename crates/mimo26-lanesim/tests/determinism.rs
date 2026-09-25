//! Determinism: two runs of the same scenario must agree bit-for-bit.

use mimo26_lanesim::geom::ModelGeom;
use mimo26_lanesim::rng::step_rows;
use mimo26_lanesim::{LaneFaults, LaneSim, LaneSimConfig};

fn sim(seed: u64) -> LaneSim {
    let mut cfg = LaneSimConfig::model_default();
    cfg.seed = seed;
    LaneSim::new(cfg).unwrap()
}

/// Kills: HashMap iteration order anywhere in the event path, arrival-order
/// f32 accumulation, wall-clock or thread-scheduling leakage, unseeded fault
/// injection. Two identical runs — reports, combined outputs and virtual-clock
/// totals included — must be bit-for-bit equal.
#[test]
fn two_runs_bitwise_identical() {
    let mut a = sim(7);
    let ra = a.run_decode_step(4).unwrap();
    let mut b = sim(7);
    let rb = b.run_decode_step(4).unwrap();
    assert_eq!(ra, rb); // full StepReport, including total_ns
    assert!(ra.total_ns > 0);
    assert_eq!(ra.label, "MODEL");

    // the interleaved manual path (two tickets in flight, duplicate frames
    // injected on every delivery) has the same claim
    let run = |seed: u64| {
        let mut cfg = LaneSimConfig::model_default();
        cfg.seed = seed;
        let faults = LaneFaults::new(4, 0.0, 1.0, seed).unwrap(); // every frame duplicated
        let mut s = LaneSim::with_faults(cfg, faults).unwrap();
        let rows0 = step_rows(seed, 0, 2, &ModelGeom::REAL);
        let rows1 = step_rows(seed, 1, 2, &ModelGeom::REAL);
        let t0 = s.submit(0, rows0).unwrap();
        let t1 = s.submit(1, rows1).unwrap();
        let r1 = s.collect(t1).unwrap();
        let r0 = s.collect(t0).unwrap();
        (r0.combined, r1.combined, s.fault_stats().duplicated, s.now_ns())
    };
    let first = run(21);
    let second = run(21);
    assert_eq!(first, second);
    assert_eq!(first.2, 2 * 2 * 4); // 2 tickets x 2 rows x 4 ranks, every frame duplicated
}

/// Kills: cross-ticket accumulation state (shared buffers, running sums) — the
/// per-ticket combined output must not depend on the order the scheduler
/// collects its tickets in (I5 needs collect-order freedom for two-batch
/// overlap).
#[test]
fn collect_order_does_not_change_sums() {
    let scenario = |order: [usize; 2]| {
        let cfg = LaneSimConfig::model_default();
        let mut s = LaneSim::new(cfg).unwrap();
        let rows0 = step_rows(3, 0, 2, &ModelGeom::REAL);
        let rows1 = step_rows(3, 1, 2, &ModelGeom::REAL);
        let tickets = [s.submit(0, rows0).unwrap(), s.submit(1, rows1).unwrap()];
        let mut got: [Option<Vec<Vec<f32>>>; 2] = [None, None];
        for &i in &order {
            got[i] = Some(s.collect(tickets[i]).unwrap().combined);
        }
        [got[0].clone().unwrap(), got[1].clone().unwrap()]
    };
    let ab = scenario([0, 1]);
    let ba = scenario([1, 0]);
    assert_eq!(ab, ba);
}
