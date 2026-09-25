//! `mimo26-lanesim` — **LaneSim**: the full expert path simulated against a
//! STUB Spark with a virtual clock (I4 item 7; ADVISOR-I4 §3.2 step 7).
//!
//! # What this crate is
//!
//! 1. **The scheduler-facing seam, fixed here** (§3.2 steps 7–8):
//!    [`LaneSim::submit`]`(layer, rows) -> `[`Ticket`]` (non-blocking),
//!    [`LaneSim::poll`] (advance the virtual clock by one event),
//!    [`LaneSim::collect`] (drive the clock to readiness). Multiple tickets in
//!    flight give I5 its two-batch overlap.
//! 2. **The wire row shapes** the `mimo26-wire` DS41RTE3 v3 codec must carry
//!    (§3.2 step 5): request row **4,360 B** (FP8 hidden + UE8M0 scales +
//!    top-8 route ids/weights), compact return **8,192 B BF16 per Spark per
//!    token** with the Spark pre-summing its 8 weighted route partials and the
//!    coordinator summing the 4 rank partials in FP32. Per-route FP32 returns
//!    are forbidden (the 24.6 MB/token wall). This crate deliberately does not
//!    depend on `mimo26-wire`; it fixes the shapes, the captain wires the codec.
//! 3. **The sum-conservation law**, checkable at runtime:
//!    [`StepResult::check_sum_conservation`] — the assembled FP32 sum must equal
//!    `sum_routes w_e * Expert_e(h)` ([`spark::reference_expert_sum`]). The
//!    wrong assembly paths ([`StubBehavior::ForgotPreSum`],
//!    [`StubBehavior::Unweighted`], [`CollectMode::NaiveNoDedup`] double
//!    counting) all trip it by a wide margin.
//! 4. **The §3.1 measurements for later scheduler work** ([`model`]):
//!    per-step expert time at 273 GB/s LPDDR5x with a pluggable kernel
//!    efficiency in [0.5, 1.0], the ~40 us wire-RTT model, and the
//!    C1 DFlash k=7 / C6 x 8 / C16 x 8 / prefill-2,048 cases at 100% / 75% /
//!    60% ([`model::table_cases`]).
//!
//! # Labels — MODEL vs MEASURED
//!
//! **Every number this crate emits is MODEL** (virtual clock + stub Spark +
//! checkpoint geometry from `bench/model/afd_vs_tp4_model.py`) and its reports
//! carry `label: "MODEL"` ([`MODEL`]). Stubs may design, never promote
//! (AGENTS I-Hon): quote these numbers only with the 3–5× discount. The only
//! **MEASURED** constants carried are the D7 bar ([`model::MEAN_DFLASH_ACCEPTANCE`],
//! [`model::D7_STEP_MS`], [`model::D7_TOK_PER_S`]), quoted from
//! `runs/20260923-d7-tp4-baseline/RESULT.md` and labelled at their definitions.
//!
//! # Determinism
//!
//! CPU-only, std-only: no wall-clock, no sleeps, no threads. Event order is the
//! total order `(time_ns, event_seq)`; routing, stub numerics and fault
//! injection are seeded splitmix64 streams. Two runs of the same scenario are
//! bit-for-bit identical (tested).

pub mod error;
pub mod fault;
pub mod geom;
pub mod model;
pub mod rng;
pub mod rows;
pub mod spark;
pub mod sim;

pub use error::{ConfigError, ConError, SimError};
pub use fault::{LaneFaults, LaneStats};
pub use geom::{ModelGeom, MODEL};
pub use rows::{f32_to_bf16, FrameSeq, RequestRow, ReturnFrame, ROUTE_ENTRY_BYTES, ROW_DESCRIPTOR_BYTES};
pub use spark::{reference_expert_sum, StubBehavior, StubSpark};
pub use sim::{CollectMode, LaneSim, LaneSimConfig, LayerAcct, StepReport, StepResult, Ticket};
