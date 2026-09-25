//! LaneSim: the full expert path against a stub Spark, on a **virtual clock**.
//!
//! Deterministic total event order `(time_ns, event_seq)`; no wall-clock, no
//! sleeps, no threads. Every number produced here is MODEL.
//!
//! This is where the scheduler-facing API is fixed (ADVISOR-I4 §3.2 steps 7–8):
//! `submit(layer, rows) -> Ticket` is non-blocking and returns immediately;
//! `poll(ticket)` advances the virtual clock by one event at most;
//! `collect(ticket)` drives the clock to readiness. Multiple tickets in flight
//! give the I5 scheduler its two-batch overlap (attention of micro-batch B over
//! experts of micro-batch A) — the stub ranks are FIFS queues on the clock.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{ConfigError, ConError, SimError};
use crate::fault::{LaneFaults, LaneStats};
use crate::geom::ModelGeom;
use crate::rows::{FrameSeq, RequestRow, ReturnFrame};
use crate::spark::{StubBehavior, StubSpark};
use crate::MODEL;

/// Opaque async handle from [`LaneSim::submit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ticket(pub u64);

/// How [`LaneSim::collect`] assembles delivered return frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectMode {
    /// Correct: each [`FrameSeq`] key counts exactly once; re-delivered
    /// duplicates are ignored. This is the L4 duplicate-injection contract.
    DedupBySeq,
    /// WRONG (negative-test hook): sums every delivered copy, so duplicates
    /// double-count. Exists so [`StepResult::check_sum_conservation`] can be
    /// shown to kill it. Never use outside tests.
    NaiveNoDedup,
}

/// LaneSim configuration. [`LaneSimConfig::model_default`] is the §3.1 design
/// point (75% kernel efficiency). All MODEL.
#[derive(Clone, Debug, PartialEq)]
pub struct LaneSimConfig {
    pub geom: ModelGeom,
    /// Achieved fraction of LPDDR5x bandwidth for expert weight streaming,
    /// pluggable in [0.5, 1.0] (§3.1). Outside the band is a loud
    /// [`ConfigError`] — the band widens only with a measurement.
    pub kernel_efficiency: f64,
    /// Wire RTT per layer round trip (default 40 us, §3.1). MODEL.
    pub wire_rtt_ns: u64,
    /// Per-rank link bandwidth for request/return serialization.
    pub link_bytes_per_sec: u64,
    /// Spark LPDDR5x streaming bandwidth (default 273 GB/s, §3.1).
    pub lpddr5x_bytes_per_sec: u64,
    /// Seed for routing, stub numerics and fault injection. MODEL.
    pub seed: u64,
    /// Stub Spark behavior (negative-test hooks included).
    pub behavior: StubBehavior,
    /// Frame assembly mode (negative-test hook included).
    pub collect_mode: CollectMode,
}

impl LaneSimConfig {
    /// The §3.1 model configuration: 75% kernel efficiency, 40 us wire RTT,
    /// 273 GB/s LPDDR5x, 15.75 GB/s per-rank link, lossless lanes.
    pub fn model_default() -> Self {
        Self {
            geom: ModelGeom::REAL,
            kernel_efficiency: 0.75,
            wire_rtt_ns: crate::model::WIRE_RTT_NS,
            link_bytes_per_sec: crate::model::SPARK_LINK_BYTES_PER_SEC,
            lpddr5x_bytes_per_sec: crate::model::LPDDR5X_BYTES_PER_SEC,
            seed: 0,
            behavior: StubBehavior::Correct,
            collect_mode: CollectMode::DedupBySeq,
        }
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if !(0.5..=1.0).contains(&self.kernel_efficiency) {
            return Err(ConfigError::KernelEfficiencyOutOfRange(self.kernel_efficiency));
        }
        if self.link_bytes_per_sec == 0 {
            return Err(ConfigError::NonPositive("link_bytes_per_sec"));
        }
        if self.lpddr5x_bytes_per_sec == 0 {
            return Err(ConfigError::NonPositive("lpddr5x_bytes_per_sec"));
        }
        if self.geom.hidden == 0 || self.geom.moe_layers == 0 || self.geom.spark_ranks == 0 {
            return Err(ConfigError::NonPositive("geom.hidden/moe_layers/spark_ranks"));
        }
        Ok(())
    }
}

/// Per-submit critical-path accounting (MODEL), folded into the step report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerAcct {
    /// Streaming time per rank: slice bytes / (LPDDR5x x kernel_efficiency).
    pub compute_ns: u64,
    /// Wire RTT for the layer round trip (one per layer boundary).
    pub rtt_ns: u64,
    /// Request + return serialization on the per-rank link.
    pub ser_ns: u64,
    /// Quarter-slice bytes each Spark streams for this submit.
    pub stream_bytes_per_rank: u64,
    /// Unique experts routed across the submit's rows (drives the byte stream).
    pub unique_experts: usize,
}

/// One collected submit. `combined` is the coordinator's FP32 sum of the 4 rank
/// partials, one hidden vector per request row. MODEL.
#[derive(Clone, Debug, PartialEq)]
pub struct StepResult {
    pub ticket: Ticket,
    pub layer: u32,
    /// Per-row combined output: FP32 sum of the 4 rank partials.
    pub combined: Vec<Vec<f32>>,
    /// Unique frame keys received.
    pub frames_received: usize,
    /// Duplicate deliveries ignored by [`CollectMode::DedupBySeq`].
    pub duplicates_ignored: usize,
    /// Virtual-clock time (ns) of this ticket's last frame arrival.
    pub finished_ns: u64,
    pub acct: LayerAcct,
    /// Always `MODEL`.
    pub label: &'static str,
}

impl StepResult {
    /// The compact-return sum-conservation law (ADVISOR-I4 §3.2 step 5):
    /// the FP32 sum of the 4 pre-summed rank partials must equal
    /// `sum_routes w_e * Expert_e(h)` (`expected` from
    /// [`crate::spark::reference_expert_sum`]) within `rel_tol` of the row's max magnitude.
    ///
    /// BF16 rounding of the compact returns alone lands well under 1%; the
    /// wrong assembly paths (forgotten pre-sum, dropped weights,
    /// double-counted duplicates) land at tens to hundreds of percent — so a
    /// default `rel_tol` of ~0.05 separates them cleanly. MODEL.
    pub fn check_sum_conservation(
        &self,
        expected: &[Vec<f64>],
        rel_tol: f64,
    ) -> Result<(), ConError> {
        if expected.len() != self.combined.len() {
            return Err(ConError::RowsMismatch {
                expected: expected.len(),
                got: self.combined.len(),
            });
        }
        for (row, (got, want)) in self.combined.iter().zip(expected.iter()).enumerate() {
            if got.len() != want.len() {
                return Err(ConError::RowLenMismatch {
                    row,
                    expected: want.len(),
                    got: got.len(),
                });
            }
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let bound = rel_tol * scale;
            let max_abs_err = got
                .iter()
                .zip(want.iter())
                .map(|(g, w)| (*g as f64 - *w).abs())
                .fold(0.0f64, f64::max);
            if max_abs_err > bound {
                return Err(ConError::Deviation { row, max_abs_err, bound });
            }
        }
        Ok(())
    }
}

/// Per-step virtual-clock measurements for later scheduler work (ADVISOR-I4
/// §3.1). All MODEL — quote only with the stub discount.
#[derive(Clone, Debug, PartialEq)]
pub struct StepReport {
    /// MoE layers walked (47).
    pub layers: usize,
    /// Request rows (tokens) per layer.
    pub rows_per_step: usize,
    /// Expert streaming time on the critical path: sum over layers of the
    /// per-rank stream time at `kernel_efficiency` x 273 GB/s.
    pub expert_stream_ns: u64,
    /// Wire RTT on the critical path: one `wire_rtt_ns` per layer (47 x 40 us).
    pub wire_rtt_ns: u64,
    /// Request + return serialization on the critical path.
    pub wire_ser_ns: u64,
    /// Total virtual-clock time of the step (expert + wire, per-layer sequential).
    pub total_ns: u64,
    /// Bytes each Spark streams from its own LPDDR5x over the step
    /// (sum over layers of unique-experts x 3.34 MB — the §3.1 byte stream).
    pub bytes_per_spark_per_step: u64,
    /// Realized unique experts per layer (uniform routing; ~the §3.1 expectation).
    pub unique_experts_per_layer: Vec<usize>,
    /// The efficiency knob this report was produced at.
    pub kernel_efficiency: f64,
    /// Always `MODEL`.
    pub label: &'static str,
}

enum Event {
    Deliver { ticket: Ticket, frame: ReturnFrame },
}

struct TicketState {
    layer: u32,
    row_count: usize,
    expected: BTreeSet<FrameSeq>,
    received: Vec<ReturnFrame>, // arrival order
    received_seqs: BTreeSet<FrameSeq>,
    lost: BTreeSet<FrameSeq>,
    acct: LayerAcct,
    finished_ns: u64,
}

/// The simulator: 4 stub Spark ranks as FIFS resources on a virtual clock,
/// deterministic event order, seeded routing/numerics/faults. MODEL.
pub struct LaneSim {
    config: LaneSimConfig,
    sparks: Vec<StubSpark>,
    faults: LaneFaults,
    clock_ns: u64,
    queue: BTreeMap<(u64, u64), Event>,
    next_event_key: u64,
    next_ticket: u64,
    next_seq: u64,
    rank_free_at_ns: Vec<u64>,
    tickets: BTreeMap<Ticket, TicketState>,
}

impl LaneSim {
    /// Lossless lanes (no fault injection). Fails loud on a bad config.
    pub fn new(config: LaneSimConfig) -> Result<Self, ConfigError> {
        let faults = LaneFaults::new(config.geom.spark_ranks, 0.0, 0.0, config.seed)?;
        Self::with_faults(config, faults)
    }

    /// Bring your own [`LaneFaults`] (deterministic loss/duplicate injection).
    pub fn with_faults(config: LaneSimConfig, faults: LaneFaults) -> Result<Self, ConfigError> {
        config.validate()?;
        let sparks = (0..config.geom.spark_ranks as u16)
            .map(|rank| StubSpark::new(rank, config.behavior))
            .collect();
        let ranks = config.geom.spark_ranks;
        Ok(Self {
            config,
            sparks,
            faults,
            clock_ns: 0,
            queue: BTreeMap::new(),
            next_event_key: 0,
            next_ticket: 0,
            next_seq: 0,
            rank_free_at_ns: vec![0; ranks],
            tickets: BTreeMap::new(),
        })
    }

    /// Async expert seam (ADVISOR-I4 §3.2 step 8): queue one layer's request
    /// rows on the stub Sparks and return **immediately** (non-blocking).
    ///
    /// Per rank the schedule is: request serialization + half RTT to arrive,
    /// FIFS behind that rank's earlier work, stream
    /// `unique_experts x quarter-slice` bytes at
    /// `LPDDR5x x kernel_efficiency`, then half RTT + return serialization per
    /// compact return (8,192 B per token). MODEL.
    pub fn submit(&mut self, layer: u32, rows: Vec<RequestRow>) -> Result<Ticket, SimError> {
        let g = self.config.geom;
        if rows.is_empty() {
            return Err(ConfigError::EmptyRows.into());
        }
        let mut tokens = BTreeSet::new();
        let mut experts = BTreeSet::new();
        for r in &rows {
            if r.hidden.len() != g.hidden {
                return Err(ConfigError::HiddenLen { expected: g.hidden, got: r.hidden.len() }.into());
            }
            if r.routes.len() != g.top_k {
                return Err(ConfigError::RouteCount { expected: g.top_k, got: r.routes.len() }.into());
            }
            if !tokens.insert(r.token) {
                return Err(ConfigError::DuplicateToken(r.token).into());
            }
            for (e, _) in &r.routes {
                experts.insert(*e);
            }
        }

        let ticket = Ticket(self.next_ticket);
        self.next_ticket += 1;
        let now = self.clock_ns;
        let req_bytes: u64 = rows.iter().map(|r| r.wire_bytes() as u64).sum();
        let ret_bytes: u64 = rows.len() as u64 * g.return_bytes_per_token() as u64;
        let link = self.config.link_bytes_per_sec;
        let req_ser = ceil_ns(req_bytes, link);
        let ret_ser = ceil_ns(ret_bytes, link);
        let stream_bytes_per_rank = experts.len() as u64 * g.quarter_slice_bytes();
        let compute_ns = (stream_bytes_per_rank as f64 * 1e9
            / (self.config.lpddr5x_bytes_per_sec as f64 * self.config.kernel_efficiency))
            .ceil() as u64;
        let half_rtt = self.config.wire_rtt_ns / 2;

        let mut expected = BTreeSet::new();
        let mut finished_ns = 0u64;
        let seed = self.config.seed;
        for rank in 0..self.sparks.len() {
            let partials = self.sparks[rank].pre_sum(seed, layer, &rows);
            let arrive = now + half_rtt + req_ser;
            let start = arrive.max(self.rank_free_at_ns[rank]);
            let finish = start + compute_ns;
            self.rank_free_at_ns[rank] = finish;
            let deliver = finish + half_rtt + ret_ser;
            finished_ns = finished_ns.max(deliver);
            for (row, partial) in partials.into_iter().enumerate() {
                let seq = FrameSeq { rank: rank as u16, row: row as u32, seq: self.next_seq };
                self.next_seq += 1;
                expected.insert(seq);
                self.queue.insert(
                    (deliver, self.next_event_key),
                    Event::Deliver { ticket, frame: ReturnFrame { seq, partial } },
                );
                self.next_event_key += 1;
            }
        }

        let acct = LayerAcct {
            compute_ns,
            rtt_ns: self.config.wire_rtt_ns,
            ser_ns: req_ser + ret_ser,
            stream_bytes_per_rank,
            unique_experts: experts.len(),
        };
        self.tickets.insert(
            ticket,
            TicketState {
                layer,
                row_count: rows.len(),
                expected,
                received: Vec::new(),
                received_seqs: BTreeSet::new(),
                lost: BTreeSet::new(),
                acct,
                finished_ns,
            },
        );
        Ok(ticket)
    }

    /// Non-blocking: advance the virtual clock by at most one event, then report
    /// the ticket if it is ready (or failed loud).
    pub fn poll(&mut self, ticket: Ticket) -> Result<Option<StepResult>, SimError> {
        if !self.tickets.contains_key(&ticket) {
            return Err(SimError::UnknownTicket(ticket.0));
        }
        if let Some(result) = self.try_finish(ticket)? {
            return Ok(Some(result));
        }
        self.advance_one_event()?;
        self.try_finish(ticket)
    }

    /// Drive the virtual clock until the ticket is ready. Deterministic; never a
    /// wall-clock wait. Fails loud on a lost frame or a drained queue.
    pub fn collect(&mut self, ticket: Ticket) -> Result<StepResult, SimError> {
        if !self.tickets.contains_key(&ticket) {
            return Err(SimError::UnknownTicket(ticket.0));
        }
        loop {
            if let Some(result) = self.try_finish(ticket)? {
                return Ok(result);
            }
            if !self.advance_one_event()? {
                return Err(SimError::Stalled(ticket.0));
            }
        }
    }

    /// Current virtual-clock time (ns).
    pub fn now_ns(&self) -> u64 {
        self.clock_ns
    }

    /// Lane-fault counters (ds41rt `stats` shape).
    pub fn fault_stats(&self) -> LaneStats {
        self.faults.stats()
    }

    /// The full expert path for one decode/prefill step: 47 MoE layers of
    /// `tokens_per_step` routed rows each, submitted and collected
    /// layer-sequentially (the §3.1 shape), measured on the virtual clock.
    /// MODEL.
    pub fn run_decode_step(&mut self, tokens_per_step: usize) -> Result<StepReport, SimError> {
        let g = self.config.geom;
        if tokens_per_step == 0 {
            return Err(ConfigError::EmptyRows.into());
        }
        let start = self.clock_ns;
        let mut report = StepReport {
            layers: g.moe_layers,
            rows_per_step: tokens_per_step,
            expert_stream_ns: 0,
            wire_rtt_ns: 0,
            wire_ser_ns: 0,
            total_ns: 0,
            bytes_per_spark_per_step: 0,
            unique_experts_per_layer: Vec::with_capacity(g.moe_layers),
            kernel_efficiency: self.config.kernel_efficiency,
            label: MODEL,
        };
        for layer in 0..g.moe_layers as u32 {
            let rows = crate::rng::step_rows(self.config.seed, layer, tokens_per_step, &g);
            let ticket = self.submit(layer, rows)?;
            let res = self.collect(ticket)?;
            report.expert_stream_ns += res.acct.compute_ns;
            report.wire_rtt_ns += res.acct.rtt_ns;
            report.wire_ser_ns += res.acct.ser_ns;
            report.bytes_per_spark_per_step += res.acct.stream_bytes_per_rank;
            report.unique_experts_per_layer.push(res.acct.unique_experts);
        }
        report.total_ns = self.clock_ns - start;
        Ok(report)
    }

    fn advance_one_event(&mut self) -> Result<bool, SimError> {
        let Some(((t, _), event)) = self.queue.pop_first() else {
            return Ok(false);
        };
        self.clock_ns = self.clock_ns.max(t);
        match event {
            Event::Deliver { ticket, frame } => {
                let seq = frame.seq;
                let copies = self.faults.dispatch(frame);
                if !copies.is_empty() {
                    self.faults.combine(copies.len() as u64);
                }
                let state = self
                    .tickets
                    .get_mut(&ticket)
                    .ok_or(SimError::UnknownTicket(ticket.0))?;
                if copies.is_empty() {
                    // Detected by its sequence gap at finish time: fail loud,
                    // never silently combine 3-of-4 partials (L4).
                    state.lost.insert(seq);
                } else {
                    for c in copies {
                        state.received_seqs.insert(c.seq);
                        state.received.push(c);
                    }
                }
                Ok(true)
            }
        }
    }

    fn try_finish(&mut self, ticket: Ticket) -> Result<Option<StepResult>, SimError> {
        let (lost, complete) = {
            let state = self
                .tickets
                .get(&ticket)
                .ok_or(SimError::UnknownTicket(ticket.0))?;
            (
                state.lost.iter().next().copied(),
                state.received_seqs.len() >= state.expected.len(),
            )
        };
        if lost.is_none() && !complete {
            return Ok(None);
        }
        let state = self.tickets.remove(&ticket).expect("checked above");
        if let Some(l) = lost {
            return Err(SimError::FrameLost { rank: l.rank, row: l.row, seq: l.seq });
        }

        let dim = state.received.first().map(|f| f.partial.len()).unwrap_or(0);
        let mut combined = vec![vec![0.0f32; dim]; state.row_count];
        let mut frames: Vec<&ReturnFrame> = state.received.iter().collect();
        // Deterministic sum order: (rank, row, seq) — never arrival order.
        frames.sort_by_key(|f| f.seq);
        let mut seen: BTreeSet<FrameSeq> = BTreeSet::new();
        let mut duplicates_ignored = 0usize;
        for f in frames {
            match self.config.collect_mode {
                CollectMode::DedupBySeq => {
                    if !seen.insert(f.seq) {
                        duplicates_ignored += 1;
                        continue;
                    }
                }
                CollectMode::NaiveNoDedup => {}
            }
            for (i, v) in f.partial.iter().enumerate() {
                combined[f.seq.row as usize][i] += *v;
            }
        }
        Ok(Some(StepResult {
            ticket,
            layer: state.layer,
            combined,
            frames_received: state.received_seqs.len(),
            duplicates_ignored,
            finished_ns: state.finished_ns,
            acct: state.acct,
            label: MODEL,
        }))
    }
}

/// Ceiling division of `bytes` over `bytes_per_sec`, in whole ns.
fn ceil_ns(bytes: u64, bytes_per_sec: u64) -> u64 {
    (((bytes as u128 * 1_000_000_000u128) + bytes_per_sec as u128 - 1) / bytes_per_sec as u128) as u64
}
