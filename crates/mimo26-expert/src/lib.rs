//! `mimo26-expert` — grouped MXFP4 expert GEMM for the Spark ranks (I4 items
//! 3+4; ADVISOR-I4 §3.1, §3.2 steps 1/3/4, §3.3).
//!
//! # What this crate is
//!
//! 1. A **golden-locked Rust twin** of the MXFP4 expert dequant + grouped GEMM
//!    semantics. The reference is `spike/mxfp4.py` (READ-ONLY; I-Gold — the
//!    spike is consumed, never imported) and the pinned quarter-slice layout in
//!    `crates/mimo26-repack/src/geom.rs` (READ-ONLY; that crate is the sole
//!    writer of the layout). Nothing here imports code under test.
//! 2. The **layout + gate contracts** the CUDA kernels in `kernels/` implement:
//!    the quarter-slice byte image, the grouped-GEMM launch geometry, the
//!    device dequant path (T14 nibble order, T10 E8M0 clamp, f32 saturation),
//!    the L3 tolerance, and the sm_121 AOT capacity classes.
//! 3. CPU synthetic GEMM reference tests plus a separate ignored real-byte
//!    unpack proof (`bench/fixtures/expert_nibble_fixture.json`, 27 blocks,
//!    2048 sampled oracle bit patterns each). CPU results are NOT GPU L3.
//!    Explicit negatives must FAIL under the naive implementation.
//! 4. The **bandwidth harness** (item 4): achieved GB/s of the grouped expert
//!    GEMM at M in {1,2,4,8,16} and M ~ 64 against 273 GB/s (Spark LPDDR5x),
//!    printing the % of peak. Target >= 70% at M <= 8; below 60% is a
//!    documented STOP.
//!
//! # Trap coverage (docs/COHERENCE-TRAPS.md §1/§6) — each with a NEGATIVE test
//!
//! | Trap | Where pinned | Naive misfeature |
//! |---|---|---|
//! | T14 MXFP4 nibble order (even element in the LOW nibble) | [`mxfp4`] | [`NaiveBits::NIBBLE_SWAP`] |
//! | T10 E8M0 byte 255 clamps to `2^127` | [`mxfp4`] | [`NaiveBits::E8M0_NO_CLAMP`] |
//! | T10-adjacent scale block off by one | [`mxfp4`] | [`NaiveBits::SCALE_OFF_BY_ONE`] |
//! | padded expert row must never be read | [`grouped`] | [`NaiveBits::PAD_ROW_READ`] |
//! | AOT capacity class / wrong SM refuses | [`aot`] | [`NaiveBits::AOT_MIXED_GATE`] |
//!
//! # Naive discipline (suite convention — see `mimo26-load` `tests/t1_t2_negatives.rs`)
//!
//! The **naive implementation** of each module is real code selected by
//! [`NaiveBits`] — never a test-side mock. [`naive_from_env`] reads
//! `MIMO26_SPIKE_NAIVE=1` (alias `MIMO26_EXPERT_NAIVE=1`) and maps to
//! [`NaiveBits::ALL`] ("the naive implementation" = every misfeature at once,
//! as a bad port would have them). NEGATIVE tests call env-default entry points
//! ([`bits_from_env`]), so they FAIL on the naive run and PASS on the correct
//! impl. Detection/attribution tests use single-bit oracles and pass both runs.
//! Classification is listed in each `tests/*.rs` header.

pub mod aot;
pub mod bench;
pub mod fixture;
pub mod grouped;
pub mod mxfp4;
pub mod rank_sum;
pub mod slice;
pub mod split;

use std::fmt;

/// Misfeature bits — the individual wrong choices a naive port makes. Unit
/// tests set single bits to attribute a trap; [`NaiveBits::ALL`] is "the naive
/// implementation".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NaiveBits(pub u32);

impl NaiveBits {
    /// The correct implementation.
    pub const NONE: NaiveBits = NaiveBits(0);
    /// Every trap at once (the naive run).
    pub const ALL: NaiveBits = NaiveBits(u32::MAX);

    /// **T14** — nibble order swapped: high nibble = even `k`.
    pub const NIBBLE_SWAP: NaiveBits = NaiveBits(1 << 0);
    /// **T10** — E8M0 byte 255 not clamped (`2^128` poison scale).
    pub const E8M0_NO_CLAMP: NaiveBits = NaiveBits(1 << 1);
    /// **T10-adjacent** — scale block off by one: block `b` reads scale `b+1`
    /// (the final fallback byte0 means scale2^-127, not zero).
    pub const SCALE_OFF_BY_ONE: NaiveBits = NaiveBits(1 << 2);
    /// **Padded-expert** — the grouped GEMM reads a padded (out-of-range) row
    /// instead of masking it to zero. A padded expert row must never be read:
    /// the padding is not a real expert, and reading it silently injects
    /// whatever bytes follow the slice into the sum.
    pub const PAD_ROW_READ: NaiveBits = NaiveBits(1 << 3);
    /// **AOT** — the §8 mixed gate `aot_sm = 170 | 121` on one field: accepts
    /// an SM count where an arch belongs and `sm_121` where a count belongs.
    pub const AOT_MIXED_GATE: NaiveBits = NaiveBits(1 << 4);
    /// **AOT capacity** — the capacity class is not checked against the
    /// manifest's declared class (a 4096-class bake served to a 256-class
    /// manifest, or vice versa).
    pub const AOT_CAPACITY_IGNORED: NaiveBits = NaiveBits(1 << 5);
    /// **BF16 accumulate** — the grouped GEMM accumulates in BF16 instead of
    /// f32 (the L3 tolerance is declared for f32 accumulate; a BF16
    /// accumulator is a different, worse kernel).
    pub const BF16_ACCUM: NaiveBits = NaiveBits(1 << 6);
    /// **T10** — literally force every decoded E8M0 scale to 1.0.
    pub const SCALE_ONE: NaiveBits = NaiveBits(1 << 7);
    /// CPU rank-return boundary trap: multiply the route weight twice.
    pub const ROUTE_WEIGHT_TWICE: NaiveBits = NaiveBits(1 << 8);

    /// Is `bit` selected?
    pub const fn has(self, other: NaiveBits) -> bool {
        self.0 & other.0 != 0
    }

    /// Single-bit flag (per-trap negatives).
    pub const fn of(misfeature: NaiveBits) -> Self {
        misfeature
    }
}

/// Suite convention: naive implementation selected by env `MIMO26_SPIKE_NAIVE=1`
/// (alias `MIMO26_EXPERT_NAIVE=1`), exactly like `mimo26-load`'s
/// `naive_from_env`.
pub fn naive_from_env() -> bool {
    let on = |key: &str| std::env::var(key).map(|v| v == "1").unwrap_or(false);
    on("MIMO26_SPIKE_NAIVE") || on("MIMO26_EXPERT_NAIVE")
}

/// Env-default entry-point selector (NEGATIVE tests pass this through).
pub fn bits_from_env() -> NaiveBits {
    if naive_from_env() {
        NaiveBits::ALL
    } else {
        NaiveBits::NONE
    }
}

/// Expert-path failures — all fail loud (a silent mis-slice is expert garbage).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertError {
    /// A tensor's declared shape is not the pinned MXFP4 expert geometry.
    ShapeMismatch { what: String },
    /// A quarter slice is not the pinned size.
    SliceSize { want: usize, got: usize },
    /// The grouped GEMM's expert count / token count is inconsistent.
    Grouped { what: String },
    /// A padded expert row was read (the padding is not a real expert).
    PaddedRowRead { expert: usize, row: usize },
    /// The AOT gate refused the device or the manifest.
    Aot { what: String },
    /// The golden fixture is missing, malformed, or its sha256 disagrees.
    Fixture { what: String },
}

impl fmt::Display for ExpertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExpertError::ShapeMismatch { what } => write!(f, "shape: {what}"),
            ExpertError::SliceSize { want, got } => {
                write!(f, "slice size {got} B != {want} B (truncated or wrong layout)")
            }
            ExpertError::Grouped { what } => write!(f, "grouped gemm: {what}"),
            ExpertError::PaddedRowRead { expert, row } => write!(
                f,
                "padded expert row read: expert {expert} row {row} is padding, not a real expert \
                 (mask it to zero, never read it)"
            ),
            ExpertError::Aot { what } => write!(f, "aot: {what}"),
            ExpertError::Fixture { what } => write!(f, "fixture: {what}"),
        }
    }
}

impl std::error::Error for ExpertError {}
