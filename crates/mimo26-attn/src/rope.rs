//! Partial rotary embeddings — FP32 on-the-fly angles, dual θ (T19/T7).
//!
//! Semantics mirror `oracle/mimo26/nn/layers.py:32-59` (the byte-verified CPU
//! twin): the FIRST `rot_dim = int(d_qk * factor)` channels of each head rotate
//! as a half-split pair set — `(a, b) = (x[..rd/2], x[rd/2..rd])`,
//! `a' = a·cos − b·sin`, `b' = b·cos + a·sin` — and the remaining channels pass
//! through. `inv_freq[j] = θ^(−2j/rd)`, `j = 0..rd/2−1`; at 192 × 0.334 that is
//! **64 of 192 dims** (T7) with **θ = 1e7 on GA and 1e4 on SWA** (dual θ).
//!
//! # T19 — angle precision at long positions
//!
//! The angle is computed **on the fly in FP32 as HF does** —
//! `float(pos) × inv_freq` (`modeling_mimo_v2.py` convention, ADVISOR-I3 §3
//! A5) — and never from a 1M-row cos/sin table (256 MB each) or a truncated
//! table. Positions run to 1,048,575 (R-CTX), where an f16 `inv_freq` or a
//! 32K-row table shifts the angle by O(10²–10⁵) radians: word salad at range.
//!
//! The FP64 reference ([`angles_f64`]) computes `f64(pos) × inv_freq_f64`. The
//! **declared tolerance** for FP32-vs-FP64 is derived, not guessed
//! ([`angle_tolerance`]): the FP32 angle error is bounded by the rounding of
//! `inv_freq` and of the product, `3·eps_f32·|pos|·inv[j] + 8·eps_f32`, and the
//! cosine error by that angle error (|Δcos| ≤ |Δangle|). At position 1 the bound
//! is ~1e-6; at 1M the largest pair (j = 0, angle ≈ 1.05e6 rad) allows ~0.19 —
//! that band IS HF behaviour (the model was trained on f32 angles); anything
//! outside it is a bug. `tests/rope_t19.rs` asserts both directions.

use crate::NaiveBits;

/// `int(head_dim * partial_rotary_factor)` — 64 at 192 × 0.334 (T7).
/// F64 multiply then truncate, exactly like `layers.py:32-34`.
pub fn rot_dim(d: usize, partial_rotary_factor: f64) -> usize {
    (d as f64 * partial_rotary_factor) as usize
}

/// `inv_freq[j] = 1 / θ^(2j/rd)` in f64 (`layers.py:48`).
pub fn inv_freq_f64(theta: f64, rd: usize) -> Vec<f64> {
    let half = rd / 2;
    (0..half)
        .map(|j| 1.0 / theta.powf(2.0 * j as f64 / rd as f64))
        .collect()
}

/// The same in f32 — the device-side table shape (HF keeps `inv_freq` f32).
pub fn inv_freq_f32(theta: f64, rd: usize) -> Vec<f32> {
    inv_freq_f64(theta, rd).iter().map(|&x| x as f32).collect()
}

/// f16-rounded `inv_freq` — bug oracle for the "BF16/FP16 angles" half of T19
/// (atlas: a 1M-row BF16 table or an f16 inv tensor). Deterministic: rounds the
/// f32 mantissa to 10 bits, round-to-nearest.
pub fn inv_freq_f16(theta: f64, rd: usize) -> Vec<f32> {
    inv_freq_f32(theta, rd)
        .into_iter()
        .map(round_f32_to_f16_precision)
        .collect()
}

fn round_f32_to_f16_precision(x: f32) -> f32 {
    // f16 keeps 10 explicit mantissa bits; f32 has 23 — drop 13 with
    // round-to-nearest (ties up). `inv_freq` values are finite magnitudes ≤ 1.
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    let mag = bits & 0x7FFF_FFFF;
    let rounded = (mag + (1 << 12)) & !((1 << 13) - 1);
    f32::from_bits(sign | rounded)
}

/// FP32 on-the-fly angles (T19 spec): `f32(pos) × f32(inv_freq[j])`, one per
/// rotated pair. Returns `rd/2` angles.
pub fn angles_f32(pos: i64, theta: f64, rd: usize) -> Vec<f32> {
    let inv = inv_freq_f32(theta, rd);
    let p = pos as f32;
    inv.iter().map(|&i| p * i).collect()
}

/// FP64 reference angles: `f64(pos) × f64(inv_freq[j])`.
pub fn angles_f64(pos: i64, theta: f64, rd: usize) -> Vec<f64> {
    let inv = inv_freq_f64(theta, rd);
    let p = pos as f64;
    inv.iter().map(|&i| p * i).collect()
}

/// Truncated-table bug oracle (the other T19 half): angles as if taken from a
/// 32K-row cos/sin table indexed `pos % 32768`. Right at the origin, wrong by
/// O(|pos|) radians past 32K.
pub fn angles_truncated_table_naive(pos: i64, theta: f64, rd: usize) -> Vec<f32> {
    angles_f32(pos.rem_euclid(32_768), theta, rd)
}

/// f16-`inv_freq` angle path — bug oracle combining [`inv_freq_f16`] with the
/// FP32 product (a BF16/FP16 inv table on device).
pub fn angles_f16_path(pos: i64, theta: f64, rd: usize) -> Vec<f32> {
    let inv = inv_freq_f16(theta, rd);
    let p = pos as f32;
    inv.iter().map(|&i| p * i).collect()
}

/// Declared tolerance (per rotated pair `j`) for FP32 angle vs the FP64
/// reference at `pos` — derived from f32 rounding of `inv_freq` (rel `eps/2`)
/// and of the product (rel `eps/2`), plus a f32-cast slack of `8·eps`:
/// `3·eps_f32·|pos|·inv[j] + 8·eps_f32`. See the module docs for the 1M figure.
pub fn angle_tolerance(pos: i64, theta: f64, rd: usize) -> Vec<f64> {
    let eps = f64::from(f32::EPSILON);
    let inv = inv_freq_f64(theta, rd);
    let p = pos.abs() as f64;
    inv.iter()
        .map(|&i| 3.0 * eps * p * i + 8.0 * eps)
        .collect()
}

/// Tolerance for `cos`/`sin` values derived from the FP32 angle vs those from
/// the FP64 angle: the angle bound (|Δcos| ≤ |Δangle|) plus f32 storage.
pub fn cos_tolerance(pos: i64, theta: f64, rd: usize) -> Vec<f64> {
    angle_tolerance(pos, theta, rd)
        .into_iter()
        .map(|t| t + 4.0 * f64::from(f32::EPSILON))
        .collect()
}

/// Rotate `x` ([T, H, D], row-major) at absolute `positions` ([T]).
///
/// Correct path: partial rotary (`rot_dim` of `d`), half-split pairing, FP32
/// on-the-fly angles, θ from the caller (dual-θ is the caller's family gate).
///
/// Naive misfeatures (see `NaiveBits`):
/// * [`NaiveBits::ROPE_TRUNC_TABLE`] — angles from `pos % 32768` (T19);
/// * [`NaiveBits::ROPE_FULL_WIDTH`] — rotate all `d` dims (T7);
/// * [`NaiveBits::ROPE_SINGLE_THETA`] — force θ = 1e4 everywhere (T7 dual-θ);
/// * [`NaiveBits::POS_ZEROED`] — ignore `positions` (T9).
pub fn apply_rotary(
    x: &[f32],
    n_tok: usize,
    n_heads: usize,
    d: usize,
    positions: &[i64],
    theta: f64,
    partial_rotary_factor: f64,
    naive: NaiveBits,
) -> Vec<f32> {
    assert_eq!(x.len(), n_tok * n_heads * d, "x shape");
    assert_eq!(positions.len(), n_tok, "positions length");
    let mut out = x.to_vec();
    let rd = if naive.has(NaiveBits::ROPE_FULL_WIDTH) {
        d
    } else {
        rot_dim(d, partial_rotary_factor)
    };
    assert!(rd <= d && rd % 2 == 0, "rotary dim must be even and <= d");
    let eff_theta = if naive.has(NaiveBits::ROPE_SINGLE_THETA) {
        crate::geom::SWA_ROPE_THETA
    } else {
        theta
    };
    let inv = inv_freq_f32(eff_theta, rd);
    for t in 0..n_tok {
        let pos = if naive.has(NaiveBits::POS_ZEROED) {
            0
        } else {
            positions[t]
        };
        let ang: Vec<f32> = if naive.has(NaiveBits::ROPE_TRUNC_TABLE) {
            angles_truncated_table_naive(pos, eff_theta, rd)
        } else {
            // T19 spec: float(pos) × inv_freq, on the fly, FP32.
            let p = pos as f32;
            inv.iter().map(|&i| p * i).collect()
        };
        for h in 0..n_heads {
            let base = (t * n_heads + h) * d;
            for j in 0..rd / 2 {
                // cos/sin evaluated in FP32 from the FP32 angle — mirroring the
                // device kernel (cosf/sinf) and the HF convention (torch.cos on a
                // f32 tensor). The f64-then-cast form was ~1 ulp off the device,
                // which flips e4m3 codes at the rounding boundary.
                let (c, s) = (ang[j].cos(), ang[j].sin());
                let a = out[base + j];
                let b = out[base + rd / 2 + j];
                out[base + j] = a * c - b * s;
                out[base + rd / 2 + j] = b * c + a * s;
            }
        }
    }
    out
}

/// FP64-angle reference path (same layout) — the T19 comparison target.
/// Angles come from [`angles_f64`]; everything else matches [`apply_rotary`].
pub fn apply_rotary_f64_ref(
    x: &[f32],
    n_tok: usize,
    n_heads: usize,
    d: usize,
    positions: &[i64],
    theta: f64,
    partial_rotary_factor: f64,
) -> Vec<f32> {
    assert_eq!(x.len(), n_tok * n_heads * d, "x shape");
    assert_eq!(positions.len(), n_tok, "positions length");
    let mut out = x.to_vec();
    let rd = rot_dim(d, partial_rotary_factor);
    for t in 0..n_tok {
        let ang = angles_f64(positions[t], theta, rd);
        for h in 0..n_heads {
            let base = (t * n_heads + h) * d;
            for j in 0..rd / 2 {
                let (c, s) = (ang[j].cos() as f32, ang[j].sin() as f32);
                let a = out[base + j];
                let b = out[base + rd / 2 + j];
                out[base + j] = a * c - b * s;
                out[base + rd / 2 + j] = b * c + a * s;
            }
        }
    }
    out
}

/// cos/sin per (position, rotary pair), computed **exactly** as [`apply_rotary`]'s
/// inner loop does — the FP32 angle `f32(pos) × inv_freq_f32[j]`, trig in f64
/// then cast to f32 (T19). This hoists the trig out of the per-head loop so it is
/// evaluated once per position instead of `n_heads` times, and lets the caller
/// share one table across all layers of the same θ. Returns `(cos, sin)`, each
/// `[n_tok, rd/2]` row-major (pair index fastest).
pub fn rope_cos_sin(theta: f64, positions: &[i64], rd: usize) -> (Vec<f32>, Vec<f32>) {
    assert!(rd % 2 == 0, "rotary dim must be even");
    let inv = inv_freq_f32(theta, rd);
    let mut cos = Vec::with_capacity(positions.len() * (rd / 2));
    let mut sin = Vec::with_capacity(positions.len() * (rd / 2));
    for &pos in positions {
        let p = pos as f32;
        for j in 0..rd / 2 {
            let ang = p * inv[j];
            // f32 trig, mirroring the device kernel (`rope.cu` cosf/sinf) and the
            // HF convention (`torch.cos` on f32 angles). The f64-then-cast form
            // diverged by ~1 ULP of f32, which flips e4m3 codes at the rounding
            // boundary and red-shifted the tight golden.
            cos.push(ang.cos());
            sin.push(ang.sin());
        }
    }
    (cos, sin)
}

/// Partial rotary over precomputed cos/sin — the same arithmetic as
/// [`apply_rotary`] (correct path, `NaiveBits::NONE`), but the trig is supplied
/// by the caller (see [`rope_cos_sin`]). Bit-identical to `apply_rotary` when the
/// tables come from `rope_cos_sin` at the same θ and positions.
pub fn apply_rotary_precomputed(
    x: &[f32],
    n_tok: usize,
    n_heads: usize,
    d: usize,
    rd: usize,
    cos: &[f32],
    sin: &[f32],
) -> Vec<f32> {
    assert_eq!(x.len(), n_tok * n_heads * d, "x shape");
    assert_eq!(cos.len(), n_tok * (rd / 2), "cos shape");
    assert_eq!(sin.len(), n_tok * (rd / 2), "sin shape");
    let mut out = x.to_vec();
    for t in 0..n_tok {
        for h in 0..n_heads {
            let base = (t * n_heads + h) * d;
            for j in 0..rd / 2 {
                let c = cos[t * (rd / 2) + j];
                let s = sin[t * (rd / 2) + j];
                let a = out[base + j];
                let b = out[base + rd / 2 + j];
                out[base + j] = a * c - b * s;
                out[base + rd / 2 + j] = b * c + a * s;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The precomputed path must be bit-identical to the on-the-fly `apply_rotary`
    /// at the same θ and positions (the perf optimization must not move numerics).
    #[test]
    fn precomputed_matches_on_the_fly() {
        let (n_tok, n_heads, d) = (7usize, 4usize, 192usize);
        let factor = 0.334f64;
        let theta = 1e7f64;
        let rd = rot_dim(d, factor);
        // Deterministic fill (no test-only RNG in the lib crate).
        let x: Vec<f32> = (0..n_tok * n_heads * d)
            .map(|i| (i as f32).sin() * 0.5 + (i as f32).cos() * 0.3)
            .collect();
        let positions: Vec<i64> = vec![0, 1, 10, 1000, 1_000_000, 123_456, 999_999];
        let want = apply_rotary(&x, n_tok, n_heads, d, &positions, theta, factor, NaiveBits::NONE);
        let (cos, sin) = rope_cos_sin(theta, &positions, rd);
        let got = apply_rotary_precomputed(&x, n_tok, n_heads, d, rd, &cos, &sin);
        assert_eq!(want.len(), got.len());
        for (i, (&w, &g)) in want.iter().zip(got.iter()).enumerate() {
            assert_eq!(w.to_bits(), g.to_bits(), "bit mismatch at {i}: {w} vs {g}");
        }
    }
}
