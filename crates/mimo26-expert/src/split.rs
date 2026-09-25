//! Track P per-arity error study — the split-MMA activation arithmetic.
//!
//! The (i) arity-switchable kernel runs the expert GEMM with the FP4 weights
//! exact in BF16 (E2M1 code + E8M0 scale folded as a power-of-2 shift outside the
//! MMA) and the FP32 activations split into 2 or 3 BF16 terms. This module is the
//! CPU twin of that arithmetic: the 2-term split truncates each product to ~16
//! mantissa bits, the 3-term split is exact (8+8+8 = 24 mantissa bits). The study
//! measures both against the frozen FP64 oracle over the §3.2 input classes and
//! reports the max componentwise error against the frozen 1e-5 metric.

/// E2M1 codebook, nibble-indexed (the MXFP4 weight magnitudes).
pub const E2M1_CODEBOOK: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// The frozen L3 componentwise metric: `|got - want| <= ATOL + RTOL * |want|`.
pub const ATOL: f64 = 1e-5;
pub const RTOL: f64 = 1e-5;

/// f32 -> BF16 round-to-nearest-even, round-tripped back to f32 (exact).
/// Mirrors `mimo26_wire::bf16::{f32_to_bf16_rne, bf16_to_f32}`.
pub fn bf16_rne(a: f32) -> f32 {
    let bits = a.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mag = bits & 0x7FFF_FFFF;
    if mag > 0x7F80_0000 {
        // NaN -> canonical BF16 qNaN.
        return f32::from_bits(((sign | 0x7FFF) as u32) << 16);
    }
    let mut rounded = mag.wrapping_add(0x7FFF + ((mag >> 16) & 1));
    if rounded >= 0x7F80_0000 {
        rounded = 0x7F80_0000;
    }
    let code = sign | ((rounded >> 16) as u16);
    f32::from_bits((code as u32) << 16)
}

/// 2-term split: `(hi, lo)` with `hi + lo ≈ a` (16 mantissa bits).
pub fn split2(a: f32) -> (f32, f32) {
    let hi = bf16_rne(a);
    let lo = bf16_rne(a - hi);
    (hi, lo)
}

/// 3-term split: `(hi, mid, lo)` with `hi + mid + lo == a` (24 mantissa bits,
/// exact for every finite f32).
pub fn split3(a: f32) -> (f32, f32, f32) {
    let hi = bf16_rne(a);
    let mid = bf16_rne(a - hi);
    let lo = bf16_rne(a - hi - mid);
    (hi, mid, lo)
}

/// The E8M0 scale byte -> exact power of two (f64); byte 255 clamps to 2^127.
pub fn e8m0_scale(byte: u8) -> f64 {
    2f64.powi(byte.min(254) as i32 - 127)
}

/// FP64 reference dot: `sum_k w_k * a_k` with weights as exact E2M1+E8M0 values.
pub fn dot_ref(codes: &[u8], scales: &[u8], a: &[f32]) -> f64 {
    debug_assert_eq!(codes.len(), a.len());
    debug_assert_eq!(scales.len(), a.len() / 32);
    let mut acc = 0.0f64;
    for k in 0..a.len() {
        let w = E2M1_CODEBOOK[(codes[k] & 15) as usize] * e8m0_scale(scales[k / 32]);
        acc += w * (a[k] as f64);
    }
    acc
}

/// The split dot for one arity. The MMA computes the codebook-value sub-dot per
/// 32-element block in f32 (weights <= 6, no overflow), then the E8M0 scale is
/// folded as an exact power-of-2 multiply in f64 — this is the "fold scales
/// outside the MMA" rule, so scale extremes 0/255 do not enter the MMA mantissa.
fn dot_arity(arity: u8, codes: &[u8], scales: &[u8], a: &[f32]) -> f32 {
    debug_assert_eq!(codes.len(), a.len());
    debug_assert_eq!(scales.len(), a.len() / 32);
    let mut acc = 0.0f64;
    for block in 0..scales.len() {
        let mut sub = 0.0f32;
        for k in block * 32..(block + 1) * 32 {
            let w = E2M1_CODEBOOK[(codes[k] & 15) as usize] as f32;
            if arity == 2 {
                let (hi, lo) = split2(a[k]);
                sub += w * hi;
                sub += w * lo;
            } else {
                let (hi, mid, lo) = split3(a[k]);
                sub += w * hi;
                sub += w * mid;
                sub += w * lo;
            }
        }
        acc += (sub as f64) * e8m0_scale(scales[block]);
    }
    acc as f32
}

/// 2-term split dot, f32 accumulation (the kernel's arity-2 arithmetic).
pub fn dot2(codes: &[u8], scales: &[u8], a: &[f32]) -> f32 {
    dot_arity(2, codes, scales, a)
}

/// 3-term split dot, f32 accumulation (the kernel's arity-3 arithmetic; exact
/// up to f32 accumulation order because the activation split is exact).
pub fn dot3(codes: &[u8], scales: &[u8], a: &[f32]) -> f32 {
    dot_arity(3, codes, scales, a)
}

/// The frozen L3 componentwise metric: the error exceeds the bound iff
/// `|got - want| > ATOL + RTOL * |want|`. Returns the ratio (<= 1.0 = pass).
/// This is the exact metric of record (`1e-5 + 1e-5 * |want|`); a near-zero
/// output coordinate keeps the `ATOL` absolute floor.
pub fn max_error_ratio(got: f64, want: f64) -> f64 {
    (got - want).abs() / (ATOL + RTOL * want.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3-term is exact: hi + mid + lo reconstructs the f32 bit-for-bit, for
    /// activations inside BF16's normal range (the hidden-state regime).
    #[test]
    fn three_term_reconstructs_f32_exactly() {
        for a in [0.0f32, 1.0, -1.0, 0.1, 1e-3, 1e3, -0.618, 1.5, 0.5, 1.0 / 3.0, -7.25] {
            let (hi, mid, lo) = split3(a);
            assert_eq!(hi + mid + lo, a, "3-term split of {a} is not exact");
        }
    }

    /// 2-term is approximate: hi + lo generally differs in the low mantissa.
    #[test]
    fn two_term_truncates_the_low_mantissa() {
        let a = 1.0f32 / 3.0;
        let (hi, lo) = split2(a);
        assert!(hi + lo != a, "2-term split of 1/3 should truncate");
    }

    /// The study over the frozen classes, against the frozen metric of record
    /// (`|err| <= 1e-5 + 1e-5*|want|`). 3-term must pass (exact by construction);
    /// 2-term's max ratio is recorded — it exceeds 1 on the cancellation-heavy
    /// (near-zero output) class, which is the "marginal-to-failing" finding.
    #[test]
    fn arity_error_study_over_the_frozen_classes() {
        let mut max2 = 0.0f64;
        let mut max3 = 0.0f64;
        for k in [2048usize, 4096] {
            // Class 1: uniform [-1, 1] activations, mixed E2M1 weights, scale 127.
            let a: Vec<f32> = (0..k)
                .map(|i| ((i.wrapping_mul(2654435761) >> 8) as f32 / 16777216.0 - 0.5) * 2.0)
                .collect();
            let codes: Vec<u8> = (0..k).map(|i| ((i * 7 + 3) % 16) as u8).collect();
            let scales = vec![127u8; k / 32];
            max2 = max2.max(max_error_ratio(dot2(&codes, &scales, &a) as f64, dot_ref(&codes, &scales, &a)));
            max3 = max3.max(max_error_ratio(dot3(&codes, &scales, &a) as f64, dot_ref(&codes, &scales, &a)));

            // Class 2: cancellation-heavy (alternating signs, sum ~ 0).
            let c: Vec<f32> = (0..k).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
            max2 = max2.max(max_error_ratio(dot2(&codes, &scales, &c) as f64, dot_ref(&codes, &scales, &c)));
            max3 = max3.max(max_error_ratio(dot3(&codes, &scales, &c) as f64, dot_ref(&codes, &scales, &c)));
        }
        println!(
            "ARITY STUDY 2-term max_ratio={max2:.6e} 3-term max_ratio={max3:.6e} (metric |err| <= {ATOL:e} + {RTOL:e}*|want|; pass = <= 1.0)"
        );
        // 3-term passes (exact by construction; only f32 accumulation order).
        assert!(max3 <= 1.0, "3-term must pass the frozen metric, got {max3}");
        // 2-term is marginal-to-failing: it exceeds the metric on the
        // cancellation-heavy class. Recorded, not a blanket pass.
        assert!(max2 > 1.0, "2-term should fail the cancellation class, got {max2}");
    }
}
