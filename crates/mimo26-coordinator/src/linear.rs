//! Dense projections — FP32 accumulate, twin of `oracle/mimo26/nn/layers.py`.
//!
//! `linear(x, w) == x @ w.T` with `w` row-major `[out, in]`. The oracle uses
//! NumPy `@` (BLAS); we use a plain FP32 dot so the reduction order is frozen
//! and reproducible (phase-contract INV-7 within-arm determinism). FP32
//! accumulation differs from BLAS by well under the R7 tolerance at serving
//! shapes.

/// `y = x @ w.T` (+ optional bias). `x` is `[rows, in_dim]`; `w` is
/// `[out_dim, in_dim]`; `bias`, if present, is `[out_dim]`.
pub fn linear(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    assert_eq!(x.len() % in_dim, 0, "x length must be a multiple of in_dim");
    assert_eq!(w.len(), out_dim * in_dim, "w shape must be [out_dim, in_dim]");
    let rows = x.len() / in_dim;
    let mut y = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            let wrow = &w[o * in_dim..(o + 1) * in_dim];
            let xrow = &x[r * in_dim..(r + 1) * in_dim];
            for i in 0..in_dim {
                acc += xrow[i] * wrow[i];
            }
            y[r * out_dim + o] = acc;
        }
    }
    y
}

/// Dense FFN: `linear(silu(linear(x, gate)) * linear(x, up), down)`.
/// `gate`/`up` are `[inter_dim, in_dim]`; `down` is `[in_dim, inter_dim]`.
pub fn dense_ffn(
    x: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    in_dim: usize,
    inter_dim: usize,
) -> Vec<f32> {
    let rows = x.len() / in_dim;
    let g = linear(x, gate, in_dim, inter_dim);
    let u = linear(x, up, in_dim, inter_dim);
    let mut h = Vec::with_capacity(g.len());
    for r in 0..rows {
        for j in 0..inter_dim {
            let a = g[r * inter_dim + j];
            let b = u[r * inter_dim + j];
            h.push(crate::norm::silu(a) * b);
        }
    }
    linear(&h, down, inter_dim, in_dim)
}
