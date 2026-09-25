//! Deterministic PRNG streams (splitmix64) and uniform top-k routing. All MODEL.
//!
//! No dependency on any RNG crate: same seed -> same streams, bit-for-bit, on
//! every run. Nothing here touches the wall clock.

use std::collections::BTreeSet;

use crate::geom::ModelGeom;
use crate::rows::RequestRow;

/// splitmix64: tiny, deterministic, std-only.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [0, 1), f32.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }
}

/// Fold values into an independent stream seed.
pub fn mix(vals: &[u64]) -> u64 {
    let mut acc = 0x243F_6A88_85A3_08D3u64;
    for &v in vals {
        let mut z = acc ^ v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        acc = z ^ (z >> 31);
    }
    acc
}

/// Uniform top-`k` over `experts` without replacement (distinct route ids), with
/// positive weights normalized to sum 1 (stand-in for `norm_topk_prob`). MODEL.
pub fn uniform_top_k(rng: &mut SplitMix64, experts: usize, k: usize) -> Vec<(u32, f64)> {
    let want = k.min(experts);
    let mut chosen = BTreeSet::new();
    while chosen.len() < want {
        chosen.insert((rng.next_u64() as usize) % experts);
    }
    let mut routes: Vec<(u32, f64)> = chosen
        .into_iter()
        .map(|e| (e as u32, 0.5 + rng.next_f64()))
        .collect();
    let sum: f64 = routes.iter().map(|(_, w)| *w).sum();
    for r in &mut routes {
        r.1 /= sum;
    }
    routes
}

/// One step's request rows (hidden + top-8 routes) for one layer: pure in the
/// seed, so two runs of the same scenario are identical. MODEL.
pub fn step_rows(seed: u64, layer: u32, rows: usize, geom: &ModelGeom) -> Vec<RequestRow> {
    (0..rows)
        .map(|t| {
            let mut rng = SplitMix64::new(mix(&[seed, layer as u64, t as u64, 0x524F_5753]));
            let hidden: Vec<f32> = (0..geom.hidden).map(|_| rng.next_f32()).collect();
            let routes = uniform_top_k(&mut rng, geom.experts_per_layer, geom.top_k);
            RequestRow::new(t as u32, hidden, routes)
        })
        .collect()
}
