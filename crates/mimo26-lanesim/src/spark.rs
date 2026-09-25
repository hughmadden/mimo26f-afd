//! StubSpark: the TP4EP1 quarter-slice expert path, algebraically. **All MODEL;
//! never a numerics claim** (I-Hon).
//!
//! Real shape being stood in for: expert = gate[2048x4096] + up[2048x4096] +
//! down[4096x2048]; quarter slice `s` owns intermediate rows/columns
//! `[512s, 512(s+1))`; Spark `s` computes `c_s = down_s . act(gate_s.h, up_s.h)`
//! and returns the weighted pre-sum over its top-8 routes as one BF16 hidden
//! vector (8,192 B). The algebra under test here is the **assembly law**
//! (quarter linearity + weighted pre-sum + FP32 rank combine), not GEMM math.
//!
//! The stub kernel is deliberately cheap (O(H) per route) and strictly
//! positive-valued, so the sum-conservation check has no cancellation to hide
//! in: the wrong assembly paths land hundreds of percent off while BF16 rounding
//! alone stays under 1%.

use std::collections::BTreeMap;

use crate::rows::{f32_to_bf16, RequestRow};

/// Quarter slices per expert (TP4). Matches `ModelGeom::REAL.spark_ranks`.
const N_QUARTERS: u16 = 4;
/// Stub neurons per quarter slice: 512 of the 2048 intermediate neurons.
const NQ: usize = 512;

/// What the stub Spark returns. [`StubBehavior::Correct`] is the compact-return
/// contract (ADVISOR-I4 §3.2 step 5); the other two are WRONG implementations
/// kept as negative-test hooks so `check_sum_conservation` can be shown to kill
/// them. Never use them outside tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StubBehavior {
    /// Weighted pre-sum of all top-8 route partials — the correct compact return.
    Correct,
    /// WRONG: "forgets the pre-sum" — returns only the first route's weighted
    /// partial and drops the other routes.
    ForgotPreSum,
    /// WRONG: pre-sums the routes but drops their weights (`norm_topk_prob`
    /// forgotten).
    Unweighted,
}

/// One stub Spark rank: owns quarter slice `rank` of every expert. MODEL.
#[derive(Clone, Debug)]
pub struct StubSpark {
    pub rank: u16,
    pub behavior: StubBehavior,
    /// Per-expert output fan vectors (rho depends on (seed, expert, quarter)
    /// only), cached across `pre_sum` calls so a 47-layer step stays cheap.
    rho_cache: BTreeMap<u32, Vec<f64>>,
}

impl StubSpark {
    pub fn new(rank: u16, behavior: StubBehavior) -> Self {
        Self { rank, behavior, rho_cache: BTreeMap::new() }
    }

    /// The compact return rows, one hidden-sized partial per request row: this
    /// rank's quarter-slice partials, combined over routes per
    /// [`StubBehavior`], then BF16-quantized like the 8,192-B wire row.
    pub fn pre_sum(&mut self, seed: u64, layer: u32, rows: &[RequestRow]) -> Vec<Vec<f32>> {
        let rank = self.rank;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let mut acc = vec![0.0f64; row.hidden.len()];
            for (route_idx, (expert, weight)) in row.routes.iter().enumerate() {
                let z = quarter_z(seed, layer, *expert, rank, &row.hidden);
                let k = match self.behavior {
                    StubBehavior::Correct => weight * z,
                    StubBehavior::ForgotPreSum if route_idx == 0 => weight * z,
                    StubBehavior::ForgotPreSum => 0.0,
                    StubBehavior::Unweighted => z,
                };
                if k == 0.0 {
                    continue;
                }
                let rho = self
                    .rho_cache
                    .entry(*expert)
                    .or_insert_with(|| rho_vec(seed, *expert, rank, row.hidden.len()));
                for (i, r) in rho.iter().enumerate() {
                    acc[i] += k * r;
                }
            }
            out.push(acc.iter().map(|&v| f32_to_bf16(v as f32)).collect());
        }
        out
    }
}

/// The un-sharded reference: `sum_routes w_e * Expert_e(h)` with
/// `Expert_e(h) = sum_{s<4} c_{s,e}(h)` — computed independently of the rank
/// assembly (no pre-sum, no frames, no dedup), in f64. This is the expected
/// value for [`crate::StepResult::check_sum_conservation`].
///
/// Self-consistency caveat (I-Gold discipline): it shares the stub kernel
/// helpers [`quarter_z`]/[`rho_vec`] with the assembly path — what the
/// conservation tests pin is the *assembly law* (pre-sum, weights, duplicate
/// handling, FP32 combine), which is where the wrong implementations live.
pub fn reference_expert_sum(seed: u64, layer: u32, rows: &[RequestRow]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|row| {
            let mut out = vec![0.0f64; row.hidden.len()];
            for (expert, weight) in &row.routes {
                for q in 0..N_QUARTERS {
                    let z = quarter_z(seed, layer, *expert, q, &row.hidden);
                    if *weight == 0.0 || z == 0.0 {
                        continue;
                    }
                    let rho = rho_vec(seed, *expert, q, row.hidden.len());
                    for (i, r) in rho.iter().enumerate() {
                        out[i] += weight * z * r;
                    }
                }
            }
            out
        })
        .collect()
}

/// Quarter `quarter`'s scalar activation mass for expert `layer/expert` on one
/// hidden row: `sum_j d_j * act(g_j*x_j + b_j)` over the quarter's 512 stub
/// neurons, with `act(v) = |v| / (1 + |v|)` (bounded, non-negative) and
/// hash-derived `g_j, b_j, d_j`. MODEL stand-in for `down_s . act(...)`.
fn quarter_z(seed: u64, layer: u32, expert: u32, quarter: u16, hidden: &[f32]) -> f64 {
    let mut rng = crate::rng::SplitMix64::new(crate::rng::mix(&[
        seed,
        layer as u64,
        expert as u64,
        quarter as u64,
        0x5A17_0BEE,
    ]));
    let base = quarter as usize * NQ;
    let n = hidden.len();
    let mut z = 0.0f64;
    for j in 0..NQ {
        let g = 2.0 * rng.next_f64() - 1.0;
        let b = 2.0 * rng.next_f64() - 1.0;
        let d = 0.5 + rng.next_f64();
        let x = hidden[(base + j) % n] as f64;
        let v = g * x + b;
        let a = v.abs() / (1.0 + v.abs());
        z += d * a;
    }
    z
}

/// Output fan vector for expert `expert`, quarter `quarter`: positive
/// hash-derived coefficients in [0.5, 1.5), one per hidden element. Shared
/// verbatim by the assembly path and the reference so quarter linearity holds
/// exactly before BF16 rounding. Depends on (seed, expert, quarter) only, so the
/// stub Sparks can cache it across layers. MODEL.
fn rho_vec(seed: u64, expert: u32, quarter: u16, n: usize) -> Vec<f64> {
    let mut rng = crate::rng::SplitMix64::new(crate::rng::mix(&[
        seed,
        expert as u64,
        quarter as u64,
        0x5248_4F54_4147_0001,
    ]));
    (0..n).map(|_| 0.5 + rng.next_f64()).collect()
}
