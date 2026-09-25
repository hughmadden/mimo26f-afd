//! CPU lane-fault injection.
//!
//! Design-shape reuse of ds41rt `mimo26/afd/stubs.py::LaneSim` (READ-ONLY
//! reference: `port-workspace/`): deterministic
//! seeded loss, `loss_rate = 0` never drops, `combine` response accounting,
//! `stats` counters, loud validation. Deltas vs the ds41rt unit (full list in
//! the packet receipt for the captain's `docs/REUSE.md` row):
//!
//! 1. Rust splitmix64 instead of `numpy.random.default_rng` (same-shape
//!    determinism per seed, different bit stream),
//! 2. `dispatch` returns `Vec<T>` of 0..=2 copies (duplicates are first-class
//!    here for the L4 duplicate-injection contract) instead of `bytes | None`,
//! 3. `duplicate_rate` knob added (the ds41rt unit models loss only),
//! 4. no `latency_bytes_per_tok` knob — latency is the virtual clock's job.

use crate::error::ConfigError;
use crate::rng::SplitMix64;

/// Lane counters (ds41rt `LaneSim::stats` shape + `duplicated`). MODEL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneStats {
    pub lanes: usize,
    pub sent: u64,
    pub delivered: u64,
    pub dropped: u64,
    pub duplicated: u64,
    pub combine_records: u64,
}

/// Deterministic seeded fault injection over the simulated rank lanes. MODEL.
#[derive(Clone, Debug)]
pub struct LaneFaults {
    n_lanes: usize,
    loss_rate: f64,
    duplicate_rate: f64,
    rng: SplitMix64,
    sent: u64,
    delivered: u64,
    dropped: u64,
    duplicated: u64,
    combine_records: u64,
}

impl LaneFaults {
    /// `n_lanes >= 1`; `loss_rate` in [0, 1) and `duplicate_rate` in [0, 1]
    /// (1.0 = duplicate every frame — the L4 duplicate-injection extreme).
    /// Anything else is a loud [`ConfigError`], never a silent clamp.
    pub fn new(
        n_lanes: usize,
        loss_rate: f64,
        duplicate_rate: f64,
        seed: u64,
    ) -> Result<Self, ConfigError> {
        if n_lanes < 1 {
            return Err(ConfigError::Lanes);
        }
        if !(0.0..1.0).contains(&loss_rate) {
            return Err(ConfigError::LossRate(loss_rate));
        }
        if !(0.0..=1.0).contains(&duplicate_rate) {
            return Err(ConfigError::DuplicateRate(duplicate_rate));
        }
        Ok(Self {
            n_lanes,
            loss_rate,
            duplicate_rate,
            rng: SplitMix64::new(seed),
            sent: 0,
            delivered: 0,
            dropped: 0,
            duplicated: 0,
            combine_records: 0,
        })
    }

    /// Dispatch one payload over the simulated lanes. Returns 0 copies (drop),
    /// 1 copy (clean delivery) or 2 copies (duplicate delivery). Deterministic
    /// per seed; `loss_rate = 0` never drops and `duplicate_rate = 0` never
    /// duplicates (both tested).
    pub fn dispatch<T: Clone>(&mut self, payload: T) -> Vec<T> {
        self.sent += 1;
        if self.loss_rate > 0.0 && self.rng.next_f64() < self.loss_rate {
            self.dropped += 1;
            return Vec::new();
        }
        let mut out = Vec::new();
        if self.duplicate_rate > 0.0 && self.rng.next_f64() < self.duplicate_rate {
            self.duplicated += 1;
            out.push(payload.clone());
        }
        self.delivered += 1;
        out.push(payload);
        out
    }

    /// Account `n` combine records on the response path (running total).
    pub fn combine(&mut self, n: u64) -> u64 {
        self.combine_records += n;
        self.combine_records
    }

    pub fn stats(&self) -> LaneStats {
        LaneStats {
            lanes: self.n_lanes,
            sent: self.sent,
            delivered: self.delivered,
            dropped: self.dropped,
            duplicated: self.duplicated,
            combine_records: self.combine_records,
        }
    }
}
