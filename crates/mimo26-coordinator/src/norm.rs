//! RMSNorm and SiLU — FP64-shared math, twin of `oracle/mimo26/nn/layers.py`.
//!
//! The oracle computes the RMSNorm mean-of-squares and the SiLU denominator in
//! FP64 (`x.astype(float64) ** 2` and `1.0 + exp(-clip(x))` with FP64 `1.0`),
//! then casts back to FP32. We reproduce that exactly so a golden-lock is
//! bit-comparable rather than merely tolerance-close.

/// RMSNorm over the last axis: `x / sqrt(mean(x^2) + eps) * weight`, in FP32
/// out with the mean-of-squares and the normalize+scale chain evaluated in
/// FP64 (the oracle's precision). `x` is row-major `[rows, dim]`; `weight` is
/// `[dim]`.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = weight.len();
    assert_eq!(x.len() % dim, 0, "x length must be a multiple of dim");
    let rows = x.len() / dim;
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        // mean of squares in FP64 (oracle: `x.astype(float64) ** 2`).
        let mut sum_sq = 0.0f64;
        for &v in row {
            let v = v as f64;
            sum_sq += v * v;
        }
        let var = sum_sq / dim as f64;
        let inv = 1.0 / (var + eps as f64).sqrt();
        for i in 0..dim {
            // (x / sqrt(var+eps)) * weight, both promoted to FP64.
            out[r * dim + i] = ((row[i] as f64) * inv * (weight[i] as f64)) as f32;
        }
    }
    out
}

/// SiLU: `x / (1 + exp(-clip(x, -60, 60)))`, FP64 denominator (oracle).
pub fn silu(x: f32) -> f32 {
    let xc = x.clamp(-60.0f32, 60.0f32) as f64;
    let denom = 1.0f64 + (-xc).exp();
    ((x as f64) / denom) as f32
}

/// Elementwise SiLU over a slice.
pub fn silu_all(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| silu(v)).collect()
}
