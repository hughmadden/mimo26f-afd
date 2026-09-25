//! `mimo26-load` — MiMo-V2.6-Flash checkpoint load path (P-202, I2).
//!
//! Ports the PROVEN I1 spike loader logic. Per the row rules ("know-how cited,
//! not rowed") this is a PORT of our own spike, not a copy-in — no
//! `docs/REUSE.md` row is required for it. Citations (spike, P-101/P-102
//! golden-verified; recorded in `runs/20260923-i1-spike/`):
//!
//! * `spike/loader.py` — fused-QKV `ckpt_tp=4` TP-order reconstruction
//!   (`reconstruct_layer_qkv` :53-87), `canonical_name` :30 (``model.mtp.``
//!   tested BEFORE the generic ``model.`` strip — order matters),
//!   `is_backbone_weight` :39, uneven-shard fail-loud :73-81.
//! * `spike/quant.py` — `split_shard_major_fused` :114-156 (shard-major
//!   `[Q_c|K_c|V_c]` → projection-major `[Q|K|V]`, per-shard scale-grid
//!   padding trim :143-145), `dequantize_per_row` :159-165, and the bug oracle
//!   `dequantize_naive_fused` :168-183 (global `row // br` across per-shard
//!   padded grids).
//! * `spike/model.py:39-56` + `spike/shard_index.py:43-74` — fail-loud name
//!   audit (required weights, router bias, shard-index classification).
//!
//! Coherence traps pinned here: **T1** (fused-QKV ckpt_tp=4 is the classic word
//! salad — scrambled Q/K/V rows) and **T2** (per-shard padded scale grids — pad
//! rows must NEVER be read), per `docs/COHERENCE-TRAPS.md`.
//!
//! Golden oracle (read-only per I-Gold — referenced, never copied):
//! `oracle/goldens/fp8_block_golden.json`
//! (sha256 `d8f45ad930120c29820fe1b63aa56b05624564dfb6325459c3aab52df1abc9c6`,
//! regen `code/scripts/gen-golden.py`, seed 20260922). Its `fused_split{}` case
//! pins this codec BYTE-EXACT (see `tests/golden_fused_split.rs`) — the
//! external-oracle receipt that keeps this a golden-locked twin rather than
//! self-consistency (AGENTS.md §4).
//!
//! # Naive discipline (suite convention)
//!
//! The naive implementation (shard-major row order + global scale-row mapping —
//! loads fine, outputs garbage) is selected by `naive == true`, or by env
//! `MIMO26_SPIKE_NAIVE=1` (alias `MIMO26_LOAD_NAIVE=1`) through
//! [`naive_from_env`]. Trap-negative tests call the env-default entry points so
//! they FAIL on the naive run and PASS on the correct impl; detection tests use
//! explicit `naive` flags and pass both runs. Classification is listed in
//! `tests/t1_t2_negatives.rs`.

use std::fmt;

pub mod e4m3;
pub mod fused;
pub mod names;

/// Row-major matrix — the loader's tensor currency (u8 e4m3 codes / f32 scales
/// / f64 fixture data). Row-major like the numpy arrays it mirrors.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl<T> Mat<T> {
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, LoadError> {
        if data.len() != rows * cols {
            return Err(LoadError::ShapeMismatch {
                what: format!("Mat data len {} != {}x{}", data.len(), rows, cols),
            });
        }
        Ok(Self { rows, cols, data })
    }

    pub fn row(&self, r: usize) -> &[T] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }
}

impl<T: Copy + Default> Mat<T> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![T::default(); rows * cols] }
    }

    pub fn get(&self, r: usize, c: usize) -> T {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: T) {
        self.data[r * self.cols + c] = v;
    }
}

impl<T: Clone> Mat<T> {
    /// `m[lo:hi, :]` — rows `lo..hi`.
    pub fn slice_rows(&self, lo: usize, hi: usize) -> Mat<T> {
        assert!(hi <= self.rows && lo <= hi, "slice_rows out of bounds");
        Mat {
            rows: hi - lo,
            cols: self.cols,
            data: self.data[lo * self.cols..hi * self.cols].to_vec(),
        }
    }

    /// `np.concatenate(parts, axis=0)` — all parts must share `cols`.
    pub fn stack_rows(parts: &[Mat<T>]) -> Result<Mat<T>, LoadError> {
        let cols = match parts.first() {
            Some(p) => p.cols,
            None => return Ok(Mat { rows: 0, cols: 0, data: Vec::new() }),
        };
        let mut data = Vec::new();
        let mut rows = 0usize;
        for p in parts {
            if p.cols != cols {
                return Err(LoadError::ShapeMismatch {
                    what: format!("stack_rows: cols {} != {}", p.cols, cols),
                });
            }
            data.extend_from_slice(&p.data);
            rows += p.rows;
        }
        Ok(Mat { rows, cols, data })
    }
}

pub fn ceil_div(a: usize, b: usize) -> usize {
    a / b + usize::from(a % b != 0)
}

/// Load-path failures — all fail loud, none silently mis-slice (AGENTS.md
/// "gibberish is a load-path bug until proven otherwise").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// spike/loader.py:73-76 — uneven shard rows would mis-slice silently.
    UnevenShardRows { rows: Vec<usize> },
    /// spike/loader.py:78-81 — every shard must equal the segment sum
    /// (shard-0-only checks mis-slice silently).
    SegmentMismatch { shard: usize, got: usize, expected: usize },
    /// "need matching non-empty shard weight/scale lists" (loader.py:71-72).
    ShardListMismatch,
    /// spike/model.py:50-52 — missing required weights (ValueError there).
    MissingRequiredWeights { missing: Vec<String> },
    /// spike/model.py:54-56 — missing router bias (KeyError there).
    MissingRouterBias { key: String },
    /// spike/shard_index.py:63-72 — UNCLASSIFIED names / lost mtp root /
    /// fused-QKV count mismatch.
    NameAudit { problems: Vec<String> },
    /// Shape/contract violation (dequantize_block :77-78 and friends).
    ShapeMismatch { what: String },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::UnevenShardRows { rows } => write!(
                f,
                "spike: uneven shard rows {rows:?} (uneven shards would mis-slice silently)"
            ),
            LoadError::SegmentMismatch { shard, got, expected } => write!(
                f,
                "shard {shard}: rows {got} != segment sum {expected} \
                 (every shard must match — shard-0-only checks mis-slice silently)"
            ),
            LoadError::ShardListMismatch => {
                write!(f, "need matching non-empty shard weight/scale lists")
            }
            LoadError::MissingRequiredWeights { missing } => {
                write!(f, "missing required weights: {missing:?}")
            }
            LoadError::MissingRouterBias { key } => write!(f, "missing router bias {key}"),
            LoadError::NameAudit { problems } => {
                write!(f, "name audit: {} problems: {:?}", problems.len(), &problems[..problems.len().min(8)])
            }
            LoadError::ShapeMismatch { what } => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Suite convention: the naive implementation is selected by env
/// `MIMO26_SPIKE_NAIVE=1` (alias `MIMO26_LOAD_NAIVE=1`). Test harnesses call
/// the entry points with this value so the trap negatives flip behind the
/// env-default naive flag, never behind test-side mocks (NAIVE DISCIPLINE).
pub fn naive_from_env() -> bool {
    std::env::var("MIMO26_SPIKE_NAIVE").map(|v| v == "1").unwrap_or(false)
        || std::env::var("MIMO26_LOAD_NAIVE").map(|v| v == "1").unwrap_or(false)
}
