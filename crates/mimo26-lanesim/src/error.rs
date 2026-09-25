//! Loud error types. A lost frame, a bad config or a broken conservation law is
//! never a silent wrong sum (ADVISOR-I4 §3.2 step 6).

use std::fmt;

/// Configuration and request-row validation failures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConfigError {
    /// `kernel_efficiency` outside the modelled band [0.5, 1.0].
    KernelEfficiencyOutOfRange(f64),
    /// A named numeric knob must be > 0 (link / LPDDR5x bandwidth, geometry).
    NonPositive(&'static str),
    /// `submit` with zero request rows.
    EmptyRows,
    /// A request row's hidden width != the configured geometry.
    HiddenLen { expected: usize, got: usize },
    /// A request row carried != `geom.top_k` routes (the row shape is fixed top-8).
    RouteCount { expected: usize, got: usize },
    /// Two request rows in one submit carried the same token id (frame keys collide).
    DuplicateToken(u32),
    /// `LaneFaults`: `n_lanes` must be >= 1.
    Lanes,
    /// `LaneFaults`: `loss_rate` must be in [0, 1).
    LossRate(f64),
    /// `LaneFaults`: `duplicate_rate` must be in [0, 1] (1.0 = every frame).
    DuplicateRate(f64),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelEfficiencyOutOfRange(e) => write!(
                f,
                "kernel_efficiency {e} outside the modelled band [0.5, 1.0]; widen the band only with a measurement, never to make a number pass"
            ),
            Self::NonPositive(what) => write!(f, "{what} must be > 0"),
            Self::EmptyRows => write!(f, "submit needs at least one request row"),
            Self::HiddenLen { expected, got } => {
                write!(f, "request row hidden width {got}, geometry says {expected}")
            }
            Self::RouteCount { expected, got } => write!(
                f,
                "request row carries {got} routes, the row shape is fixed top-{expected}"
            ),
            Self::DuplicateToken(t) => {
                write!(f, "token {t} appears twice in one submit (frame keys would collide)")
            }
            Self::Lanes => write!(f, "n_lanes must be >= 1"),
            Self::LossRate(r) => write!(f, "loss_rate {r} must be in [0, 1)"),
            Self::DuplicateRate(r) => write!(f, "duplicate_rate {r} must be in [0, 1]"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Simulation-level failures: every one of these is fail-loud, never a partial sum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SimError {
    /// Configuration rejected (see [`ConfigError`]).
    Config(ConfigError),
    /// `poll`/`collect` on an unknown or already-collected ticket.
    UnknownTicket(u64),
    /// A return frame was dropped by the fault model and detected by its sequence
    /// gap — the ticket fails instead of silently combining 3-of-4 partials.
    FrameLost { rank: u16, row: u32, seq: u64 },
    /// The event queue drained before the ticket completed (sim bug or misuse).
    Stalled(u64),
}

impl From<ConfigError> for SimError {
    fn from(e: ConfigError) -> Self {
        Self::Config(e)
    }
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(e) => write!(f, "config: {e}"),
            Self::UnknownTicket(t) => write!(f, "ticket {t} unknown or already collected"),
            Self::FrameLost { rank, row, seq } => {
                write!(f, "return frame lost (rank {rank}, row {row}, seq {seq}): failing loud, not summing a partial")
            }
            Self::Stalled(t) => write!(f, "event queue drained before ticket {t} completed"),
        }
    }
}

impl std::error::Error for SimError {}

/// Sum-conservation check failures ([`crate::StepResult::check_sum_conservation`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConError {
    /// Row-count mismatch between the combined output and the reference.
    RowsMismatch { expected: usize, got: usize },
    /// Hidden-width mismatch on one row.
    RowLenMismatch { row: usize, expected: usize, got: usize },
    /// The assembled sum deviates from `sum_routes w_e * Expert_e(h)` beyond tolerance.
    Deviation { row: usize, max_abs_err: f64, bound: f64 },
}

impl fmt::Display for ConError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowsMismatch { expected, got } => {
                write!(f, "conservation: {got} combined rows vs {expected} reference rows")
            }
            Self::RowLenMismatch { row, expected, got } => {
                write!(f, "conservation: row {row} width {got} vs {expected}")
            }
            Self::Deviation { row, max_abs_err, bound } => write!(
                f,
                "conservation violated on row {row}: max |combined - sum_routes w_e Expert_e| = {max_abs_err} > {bound} (pre-sum, weights, or duplicate handling is wrong)"
            ),
        }
    }
}

impl std::error::Error for ConError {}
