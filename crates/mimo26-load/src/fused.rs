//! Fused-QKV ckpt_tp=4 reconstruction — T1 (TP-order split) + T2 (per-shard
//! scale-grid padding trim).
//!
//! Citations: `spike/loader.py:53-87` (`reconstruct_layer_qkv`, uneven-shard
//! fail-loud) and `spike/quant.py:103-183` (`split_shard_major_fused`,
//! `dequantize_per_row`, bug oracle `dequantize_naive_fused`). Storage layout:
//! each ckpt_tp rank stores its fused `[Q_c|K_c|V_c]` rows (TP order, tonyd2wild
//! patch 01 @@ -467) with its own **padded** scale grid; the whole projection is
//! `[Q_0..Q_3 | K_0..K_3 | V_0..V_3]` (projection-major).
//!
//! T1: regrouping must yield `[Q|K|V]` — reading the concat as one matrix
//! scrambles Q/K/V (the classic word salad).
//! T2: shard row `r` reads scale row `r // br` of **that shard's** grid — the
//! padding rows at the end of each per-shard grid are NEVER indexed.

use crate::e4m3::decode_e4m3;
use crate::{LoadError, Mat};

pub const DEFAULT_BLOCK: (usize, usize) = (128, 128);

/// A rebuilt projection in projection-major order (`spike/quant.py:107-111`).
#[derive(Clone, Debug)]
pub struct ReconstructedProjection {
    pub name: &'static str,
    /// u8 `[rows, hidden]` e4m3 codes.
    pub weight: Mat<u8>,
    /// f32 `[rows, hidden / bc]` — scales resolved per row (local trim applied).
    pub scale_per_row: Mat<f32>,
}

/// The three rebuilt projections (`spike/quant.py` returns them keyed
/// `"q"`/`"k"`/`"v"`; typed fields here — same data).
#[derive(Clone, Debug)]
pub struct SplitParts {
    pub q: ReconstructedProjection,
    pub k: ReconstructedProjection,
    pub v: ReconstructedProjection,
}

/// `spike/quant.py:114-156` — rebuild per-projection tensors from shard-major
/// fused storage. `shard_weights[s]`: u8 `[sum(segs), hidden]` — shard `s`'s
/// local `[q_s; k_s; v_s]` rows in TP-rank order. `shard_scales[s]`: f32
/// `[grid_rows, ceil(hidden/bc)]` — padded per shard.
pub fn split_shard_major_fused(
    shard_weights: &[Mat<u8>],
    shard_scales: &[Mat<f32>],
    segs_per_shard: (usize, usize, usize),
    block: (usize, usize),
) -> Result<SplitParts, LoadError> {
    if shard_weights.is_empty() || shard_weights.len() != shard_scales.len() {
        return Err(LoadError::ShardListMismatch);
    }
    let (br, bc) = block;
    let (q_rows, k_rows, v_rows) = segs_per_shard;
    let total = q_rows + k_rows + v_rows;
    for (i, w) in shard_weights.iter().enumerate() {
        if w.rows != total {
            return Err(LoadError::SegmentMismatch { shard: i, got: w.rows, expected: total });
        }
    }
    let bounds: [(&'static str, usize, usize); 3] = [
        ("q", 0, q_rows),
        ("k", q_rows, q_rows + k_rows),
        ("v", q_rows + k_rows, total),
    ];
    let mut acc: [(&'static str, Vec<Mat<u8>>, Vec<Mat<f32>>); 3] = [
        ("q", Vec::new(), Vec::new()),
        ("k", Vec::new(), Vec::new()),
        ("v", Vec::new(), Vec::new()),
    ];
    for (w, s) in shard_weights.iter().zip(shard_scales.iter()) {
        for (ai, (name, lo, hi)) in bounds.iter().enumerate() {
            let mut scale_per_row = Mat::<f32>::zeros(hi - lo, w.cols / bc);
            for (i, r) in (*lo..*hi).enumerate() {
                // T2 local trim (spike/quant.py:143-145; fp8_block.py:143-146):
                // shard-local row r -> scale row r//br of THIS shard's grid;
                // pad rows beyond sum(segs)//br are never indexed.
                let srow = r / br;
                if srow >= s.rows {
                    return Err(LoadError::ShapeMismatch {
                        what: format!("shard scale grid {} rows too small for row {r}", s.rows),
                    });
                }
                for c in 0..w.cols / bc {
                    scale_per_row.set(i, c, s.get(srow, c));
                }
            }
            acc[ai].1.push(w.slice_rows(*lo, *hi));
            acc[ai].2.push(scale_per_row);
            let _ = name;
        }
    }
    let mut out: [Option<ReconstructedProjection>; 3] = [None, None, None];
    for (oi, (name, wts, scs)) in acc.into_iter().enumerate() {
        let weight = Mat::stack_rows(&wts)?;
        let scale_per_row = Mat::stack_rows(&scs)?;
        out[oi] = Some(ReconstructedProjection { name, weight, scale_per_row });
    }
    let [Some(q), Some(k), Some(v)] = out else {
        unreachable!("three accumulators")
    };
    Ok(SplitParts { q, k, v })
}

/// `spike/quant.py:159-165` — dequantize a reconstructed projection whose
/// scales are resolved per row.
pub fn dequantize_per_row(
    proj: &ReconstructedProjection,
    block: (usize, usize),
) -> Result<Mat<f32>, LoadError> {
    let (_br, bc) = block;
    let rows = proj.weight.rows;
    let cols = proj.weight.cols;
    if proj.scale_per_row.rows != rows || proj.scale_per_row.cols != cols / bc {
        return Err(LoadError::ShapeMismatch {
            what: format!(
                "scale_per_row {}x{} != {rows}x{}",
                proj.scale_per_row.rows,
                proj.scale_per_row.cols,
                cols / bc
            ),
        });
    }
    let mut out = Mat::<f32>::zeros(rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            let v = decode_e4m3(proj.weight.get(r, c)) * f64::from(proj.scale_per_row.get(r, c / bc));
            out.set(r, c, v as f32);
        }
    }
    Ok(out)
}

/// `spike/loader.py:53-87` — dequantize the ckpt_tp fused-QKV shards and
/// regroup to `[Q|K|V]` rows. Correct path: per-chunk split + local scale trim.
/// `naive == true`: the word-salad path (one tensor, one global grid).
pub fn reconstruct_layer_qkv(
    shard_weights: &[Mat<u8>],
    shard_scales: &[Mat<f32>],
    segs_per_shard: (usize, usize, usize),
    block: (usize, usize),
    naive: bool,
) -> Result<Mat<f32>, LoadError> {
    if shard_weights.is_empty() || shard_weights.len() != shard_scales.len() {
        return Err(LoadError::ShardListMismatch);
    }
    // loader.py:73-76 — uneven shard rows would mis-slice silently.
    let mut distinct: Vec<usize> = shard_weights.iter().map(|w| w.rows).collect();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() != 1 {
        return Err(LoadError::UnevenShardRows { rows: distinct });
    }
    let total = segs_per_shard.0 + segs_per_shard.1 + segs_per_shard.2;
    for (si, w) in shard_weights.iter().enumerate() {
        if w.rows != total {
            return Err(LoadError::SegmentMismatch { shard: si, got: w.rows, expected: total });
        }
    }
    if naive {
        return dequantize_naive_fused(shard_weights, shard_scales, block);
    }
    let parts = split_shard_major_fused(shard_weights, shard_scales, segs_per_shard, block)?;
    let q = dequantize_per_row(&parts.q, block)?;
    let k = dequantize_per_row(&parts.k, block)?;
    let v = dequantize_per_row(&parts.v, block)?;
    // projection-major [Q|K|V] — loader.py:85-86.
    Mat::stack_rows(&[q, k, v])
}

/// `spike/quant.py:168-183` — THE BUG, preserved as a test oracle
/// (`mimo26/quant/fp8_block.py:170`, global `row // br` at :182). Wrong in two
/// ways at once: row order stays shard-major (not projection-major) and scale
/// rows are mapped by global row index across per-shard **padded** grids —
/// scrambles Q/K/V (T1) and leaks padded scale rows (T2).
pub fn dequantize_naive_fused(
    shard_weights: &[Mat<u8>],
    shard_scales: &[Mat<f32>],
    block: (usize, usize),
) -> Result<Mat<f32>, LoadError> {
    if shard_weights.is_empty() || shard_weights.len() != shard_scales.len() {
        return Err(LoadError::ShardListMismatch);
    }
    let (br, bc) = block;
    let w = Mat::stack_rows(shard_weights)?;
    let s = Mat::stack_rows(shard_scales)?;
    let rows = w.rows;
    let cols = w.cols;
    let mut out = Mat::<f32>::zeros(rows, cols);
    for r in 0..rows {
        let srow = r / br; // naive: ignores per-shard padding
        if srow >= s.rows {
            return Err(LoadError::ShapeMismatch {
                what: format!("naive global scale row {srow} out of {} grid rows", s.rows),
            });
        }
        for c in 0..cols {
            let v = decode_e4m3(w.get(r, c)) * f64::from(s.get(srow, c / bc));
            out.set(r, c, v as f32);
        }
    }
    Ok(out)
}
