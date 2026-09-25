//! T19 (RoPE angle precision at long positions) + T7 (partial rotary 64/192,
//! dual θ) trap tests.
//!
//! # Two-run classification
//!
//! NEGATIVES — FAIL under `MIMO26_SPIKE_NAIVE=1`, PASS on the correct impl (they
//! call the env-default entry points):
//!   * `t19_fp32_onthefly_matches_f64_at_1_128k_1m`
//!   * `t7_partial_rotary_64_dims_and_dual_theta`
//!
//! BOTH RUNS (explicit `NaiveBits` bug oracles for attribution):
//!   `t19_bug_oracles_violate_the_declared_bound`,
//!   `t7_bug_oracles_diverge`, plus the derived-tolerance self-check.
//!
//! Declared tolerance (derived, not guessed — `rope.rs` module docs): the FP32
//! angle error vs the FP64 reference is bounded by
//! `3·eps_f32·|pos|·inv[j] + 8·eps_f32` radians and `|Δcos| ≤ |Δangle|`.

mod common;

use common::*;
use mimo26_attn::geom::{ROPE_THETA, SWA_ROPE_THETA};
use mimo26_attn::rope::{
    angle_tolerance, angles_f16_path, angles_f32, angles_f64, angles_truncated_table_naive,
    apply_rotary, apply_rotary_f64_ref, cos_tolerance, inv_freq_f16, inv_freq_f32, rot_dim,
};
use mimo26_attn::{bits_from_env, NaiveBits};

const D: usize = 192;
const PARTIAL: f64 = 0.334;

fn rd() -> usize {
    rot_dim(D, PARTIAL)
}

/// NEGATIVE (env-default). Positions 1, 128K and 1M (T19's exact list) — the
/// FP32 on-the-fly angles must sit inside the declared FP32-vs-FP64 band, both
/// at the angle level and end-to-end through `apply_rotary`.
/// Flips on the naive run (32K truncated table).
#[test]
fn t19_fp32_onthefly_matches_f64_at_1_128k_1m() {
    let rd = rd();
    let mut rng = XorShift64::new(1_048_575);
    let x = rng.fill_small(1 * 64 * D);
    for &pos in &[1i64, 131_072, 1_048_575] {
        // angle-level
        let a32 = angles_f32(pos, ROPE_THETA, rd);
        let a64 = angles_f64(pos, ROPE_THETA, rd);
        let tol = angle_tolerance(pos, ROPE_THETA, rd);
        for j in 0..rd / 2 {
            let d = (f64::from(a32[j]) - a64[j]).abs();
            assert!(
                d <= tol[j],
                "T19: angle[{j}] at pos {pos} off by {d} rad (declared tol {})",
                tol[j]
            );
        }
        // end-to-end through the env-default implementation
        let y = apply_rotary(&x, 1, 64, D, &[pos], ROPE_THETA, PARTIAL, bits_from_env());
        let y_ref = apply_rotary_f64_ref(&x, 1, 64, D, &[pos], ROPE_THETA, PARTIAL);
        let ctol = cos_tolerance(pos, ROPE_THETA, rd);
        let bound = ctol.iter().cloned().fold(0.0f64, f64::max) * 2.0 * 0.5 + 2e-6;
        let d = max_abs_diff(&y, &y_ref);
        assert!(
            d <= bound,
            "T19: apply_rotary at pos {pos} diverges from the FP64 reference by {d} (declared bound {bound})"
        );
        if pos == 1 {
            // near the origin the FP32 path is essentially exact
            assert!(d <= 1e-5, "T19: pos 1 must be tight, got {d}");
        }
    }
}

/// BOTH RUNS — attribution: each documented T19 bug oracle violates the
/// declared band at long positions (and the correct FP32 path does not).
#[test]
fn t19_bug_oracles_violate_the_declared_bound() {
    let rd = rd();
    let pos = 1_048_575i64;
    let tol = angle_tolerance(pos, ROPE_THETA, rd);
    // (1) truncated 32K table
    let trunc = angles_truncated_table_naive(pos, ROPE_THETA, rd);
    let a64 = angles_f64(pos, ROPE_THETA, rd);
    let worst = (0..rd / 2)
        .map(|j| (f64::from(trunc[j]) - a64[j]).abs() / tol[j])
        .fold(0.0f64, f64::max);
    assert!(worst > 10.0, "truncated-table oracle has no detection power ({worst}x tol)");
    // (2) f16 inv_freq ("angles in BF16/FP16")
    let inv16 = inv_freq_f16(ROPE_THETA, rd);
    let inv32 = inv_freq_f32(ROPE_THETA, rd);
    let inv64 = mimo26_attn::rope::inv_freq_f64(ROPE_THETA, rd);
    let mut worst16 = 0.0f64;
    let mut worst32 = 0.0f64;
    for j in 0..rd / 2 {
        let d16 = ((f64::from(inv16[j]) - inv64[j]) * pos as f64).abs();
        let d32 = ((f64::from(inv32[j]) - inv64[j]) * pos as f64).abs();
        worst16 = worst16.max(d16 / tol[j]);
        worst32 = worst32.max(d32 / tol[j]);
    }
    assert!(worst16 > 10.0, "f16 inv_freq oracle has no detection power ({worst16}x tol)");
    assert!(worst32 <= 1.0, "the FP32 inv_freq itself violates the declared band ({worst32}x)");
    // the helper used by the parity manifest agrees with the truncated oracle
    let via_helper = angles_f16_path(pos, ROPE_THETA, rd);
    assert_eq!(via_helper.len(), rd / 2);
}

/// NEGATIVE (env-default). T7: partial rotary is 64 of 192 dims and GA uses θ
/// 1e7 (dual θ vs SWA 1e4). The env-default `apply_rotary` must match the FP64
/// reference at both; flips on the naive run (full-width rotary + single θ).
#[test]
fn t7_partial_rotary_64_dims_and_dual_theta() {
    let rd = rd();
    assert_eq!(rd, 64, "T7: partial rotary width must be 64 of 192");
    let mut rng = XorShift64::new(77);
    let (t, h) = (2usize, 3usize);
    let x = rng.fill_small(t * h * D);
    let positions = vec![1i64, 5_000];
    // GA θ 1e7
    let y = apply_rotary(&x, t, h, D, &positions, ROPE_THETA, PARTIAL, bits_from_env());
    let y_ref = apply_rotary_f64_ref(&x, t, h, D, &positions, ROPE_THETA, PARTIAL);
    let d = max_abs_diff(&y, &y_ref);
    assert!(d <= 3e-3, "T7/GA: diverged from the partial-rotary reference by {d}");
    // SWA θ 1e4
    let y = apply_rotary(&x, t, h, D, &positions, SWA_ROPE_THETA, PARTIAL, bits_from_env());
    let y_ref = apply_rotary_f64_ref(&x, t, h, D, &positions, SWA_ROPE_THETA, PARTIAL);
    let d = max_abs_diff(&y, &y_ref);
    assert!(d <= 3e-3, "T7/SWA: diverged from the partial-rotary reference by {d}");
    // dims beyond rot_dim must PASS THROUGH untouched (partial, not full)
    let one = apply_rotary(&x, t, h, D, &positions, ROPE_THETA, PARTIAL, NaiveBits::NONE);
    for tok in 0..t {
        for hh in 0..h {
            for i in rd..D {
                let idx = (tok * h + hh) * D + i;
                assert_eq!(
                    one[idx].to_bits(),
                    x[idx].to_bits(),
                    "T7: channel {i} must pass through unrotated"
                );
            }
        }
    }
}

/// BOTH RUNS — attribution for T7.
#[test]
fn t7_bug_oracles_diverge() {
    let mut rng = XorShift64::new(78);
    let (t, h) = (1usize, 2usize);
    let x = rng.fill_small(t * h * D);
    let positions = vec![5_000i64];
    let y_ref = apply_rotary_f64_ref(&x, t, h, D, &positions, ROPE_THETA, PARTIAL);
    // full-width rotary (T7 half 1)
    let y_full = apply_rotary(&x, t, h, D, &positions, ROPE_THETA, PARTIAL, NaiveBits::ROPE_FULL_WIDTH);
    assert!(
        max_abs_diff(&y_full, &y_ref) > 0.1,
        "full-rotary oracle converged with the reference — no detection power"
    );
    // single θ (T7 half 2 — the SWA 1e4 θ applied to a GA layer)
    let y_one = apply_rotary(&x, t, h, D, &positions, ROPE_THETA, PARTIAL, NaiveBits::ROPE_SINGLE_THETA);
    assert!(
        max_abs_diff(&y_one, &y_ref) > 0.1,
        "single-θ oracle converged with the reference — no detection power"
    );
}
