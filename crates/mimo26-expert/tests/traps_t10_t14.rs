//! Trap negatives — T14 (nibble order) and T10 (E8M0 clamp / scale mapping).
//!
//! Classification:
//!
//! * **NEGATIVE** (must FAIL under `MIMO26_SPIKE_NAIVE=1` /
//!   `MIMO26_EXPERT_NAIVE=1`): `t14_*`, `t10_*` — they call the env-default
//!   entry points, so the naive implementation is selected and the check must
//!   catch it.
//! * **BOTH-RUNS** (pass both runs): `detector_*` — they pass an explicit
//!   [`NaiveBits`] flag and assert the check *detects* the wrong
//!   implementation.
//!
//! # What each negative kills
//!
//! | test | wrong implementation killed |
//! |---|---|
//! | `t14_unpack_matches_the_reference_semantics` | nibble order swapped (high nibble = even k) |
//! | `t14_unpack_is_bit_exact_not_tolerance_based` | a nibble swap that a tolerance would hide |
//! | `t14_grouped_gemm_matches_the_reference` | the swap surviving into the GEMM |
//! | `t10_scale_byte_255_clamps_to_2_pow_127` | E8M0 byte 255 unclamped (`2^128` poison) |
//! | `t10_scale_mapping_is_2_pow_b_minus_127` | a scale off by one (block b reads b+1) |
//! | `t10_grouped_gemm_matches_the_reference` | either scale bug surviving into the GEMM |

mod common;

use common::*;
use mimo26_expert::grouped::{self, GroupedPlan};
use mimo26_expert::mxfp4::{self, E2M1_CODEBOOK};
use mimo26_expert::slice::Proj;
use mimo26_expert::NaiveBits;

// ---------------------------------------------------------------------------
// T14 — nibble order
// ---------------------------------------------------------------------------

/// NEGATIVE. The unpack path must equal the reference semantics bitwise.
///
/// The fixture uses only exactly-representable values, so the comparison is
/// bitwise: a single swapped nibble anywhere in the slice fails.
#[test]
fn t14_unpack_matches_the_reference_semantics() {
    let naive = naive_env();
    let s = exact_slice(1401);
    for p in Proj::ALL {
        let b = s.proj(p);
        let got = mxfp4::unpack_matrix(
            &b.payload,
            &b.scales,
            p.slice_rows(),
            p.slice_in_cols(),
            naive,
        );
        // The reference: decode each byte's LOW nibble as the even element.
        let mut want = Vec::with_capacity(got.len());
        for r in 0..p.slice_rows() {
            for k in 0..p.slice_in_cols() {
                let byte = b.payload[r * p.payload_cols() + k / 2];
                let nib = if k % 2 == 0 { byte & 0x0f } else { (byte >> 4) & 0x0f };
                let scale = mxfp4::e8m0_scale(b.scales[r * p.scale_cols() + k / 32], naive);
                let prod = f64::from(E2M1_CODEBOOK[nib as usize]) * scale;
                want.push(prod.clamp(-(f32::MAX as f64), f32::MAX as f64) as f32);
            }
        }
        assert!(
            bits_eq(&got, &want),
            "T14: {} unpack is not bitwise the reference (nibble order swapped?) \
             first diff at {:?}",
            p.name(),
            first_bit_diff(&got, &want)
        );
    }
}

/// NEGATIVE. The check must be bitwise, not tolerance-based: a nibble swap
/// changes values by O(1) relative, so a tolerance would hide it only if the
/// tolerance were absurd — this test pins that the *bit* comparison is what
/// runs, by asserting the exact bit patterns of a hand-built byte.
#[test]
fn t14_unpack_is_bit_exact_not_tolerance_based() {
    let naive = naive_env();
    // One byte: low nibble 1 (0.5), high nibble 7 (6.0). Scale byte 127 -> 1.0.
    let payload = [0x71u8];
    let scales = [127u8];
    let got = mxfp4::unpack_row(&payload, &scales, naive);
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].to_bits(), 0.5f32.to_bits(), "even k must be the LOW nibble (T14)");
    assert_eq!(got[1].to_bits(), 6.0f32.to_bits(), "odd k must be the HIGH nibble (T14)");
}

/// NEGATIVE. The swap must not survive into the grouped GEMM.
#[test]
fn t14_grouped_gemm_matches_the_reference() {
    let naive = naive_env();
    let s = exact_slice(1402);
    let image = grouped_image(&[s.clone()]);
    let plan = GroupedPlan::uniform(1, 4, 0);
    let x = tokens(7, 4, Proj::Gate.slice_in_cols());
    let got = grouped::grouped_gemm(&image, &x, &plan, Proj::Gate, naive).expect("gemm");
    let want = reference_gemm(&s, &x, &plan, Proj::Gate, NaiveBits::NONE);
    assert!(
        bits_eq(&got.data, &want),
        "T14: grouped GEMM is not bitwise the reference (nibble order swapped?) \
         first diff at {:?}",
        first_bit_diff(&got.data, &want)
    );
}

// ---------------------------------------------------------------------------
// T10 — E8M0 clamp and the scale mapping
// ---------------------------------------------------------------------------

/// NEGATIVE. Byte 255 is reserved and clamps to `2^127`; the naive path
/// computes the unclamped `2^128`, a poison scale.
#[test]
fn t10_scale_byte_255_clamps_to_2_pow_127() {
    let naive = naive_env();
    let s = mxfp4::e8m0_scale(255, naive);
    assert!(
        s.is_finite(),
        "T10: scale byte 255 must clamp to a finite 2^127, got {s}"
    );
    assert_eq!(s, mxfp4::exp2(127), "T10: byte 255 must clamp to 2^127");
    // And the clamp must not touch any other byte.
    for b in 0u8..=254 {
        assert_eq!(
            mxfp4::e8m0_scale(b, naive),
            mxfp4::exp2(i32::from(b) - 127),
            "T10: byte {b} must map to 2^(b-127)"
        );
    }
}

/// NEGATIVE. The scale mapping is `2^(b-127)` per 32-element block; a scale off
/// by one (block b reads b+1) must be caught.
#[test]
fn t10_scale_mapping_is_2_pow_b_minus_127() {
    let naive = naive_env();
    // Two blocks: scale 127 (1.0) then 128 (2.0). Payload all nibble 1 (0.5).
    let payload = [0x11u8; 32]; // 64 elements, two complete blocks
    let scales = [127u8, 128u8];
    let got = mxfp4::unpack_row(&payload, &scales, naive);
    for k in 0..64 {
        let want = if k < 32 { 0.5f32 } else { 1.0 };
        assert_eq!(
            got[k].to_bits(),
            want.to_bits(),
            "T10: element {k} must use block {}'s scale",
            k / 32
        );
    }
    // A second row with a different second block, to catch an off-by-one that
    // happens to be invisible above.
    let scales2 = [120u8, 130u8];
    let got2 = mxfp4::unpack_row(&payload, &scales2, naive);
    assert_eq!(got2[0].to_bits(), (0.5f32 * mxfp4::exp2(-7) as f32).to_bits());
    assert_eq!(got2[31].to_bits(), (0.5f32 * mxfp4::exp2(-7) as f32).to_bits());
}

/// NEGATIVE. Either scale bug must not survive into the grouped GEMM.
#[test]
fn t10_grouped_gemm_matches_the_reference() {
    let naive = naive_env();
    let s = exact_slice(1002);
    let image = grouped_image(&[s.clone()]);
    let plan = GroupedPlan::uniform(1, 2, 0);
    let x = tokens(11, 2, Proj::Down.slice_in_cols());
    let got = grouped::grouped_gemm(&image, &x, &plan, Proj::Down, naive).expect("gemm");
    let want = reference_gemm(&s, &x, &plan, Proj::Down, NaiveBits::NONE);
    assert!(
        bits_eq(&got.data, &want),
        "T10: grouped GEMM is not bitwise the reference (scale mapping wrong?) \
         first diff at {:?}",
        first_bit_diff(&got.data, &want)
    );
}

// ---------------------------------------------------------------------------
// BOTH-RUNS — the detectors (explicit flags; pass on both runs)
// ---------------------------------------------------------------------------

/// BOTH RUNS. The nibble-swap detector: with the flag set, the unpack must
/// differ from the correct one on a block whose even and odd nibbles differ.
#[test]
fn detector_nibble_swap_is_detected() {
    let payload = [0x71u8]; // even 0.5, odd 6.0
    let scales = [127u8];
    let correct = mxfp4::unpack_row(&payload, &scales, NaiveBits::NONE);
    let swapped = mxfp4::unpack_row(&payload, &scales, NaiveBits::NIBBLE_SWAP);
    assert_eq!(correct[0].to_bits(), 0.5f32.to_bits());
    assert_eq!(correct[1].to_bits(), 6.0f32.to_bits());
    assert_eq!(swapped[0].to_bits(), 6.0f32.to_bits());
    assert_eq!(swapped[1].to_bits(), 0.5f32.to_bits());
    assert!(!bits_eq(&correct, &swapped), "the swap must be detectable");
}

/// BOTH RUNS. The E8M0 clamp detector: the unclamped byte 255 is `2^128`, which
/// is finite in f64 but saturates every f32 product.
#[test]
fn detector_e8m0_no_clamp_is_detected() {
    let clamped = mxfp4::e8m0_scale(255, NaiveBits::NONE);
    let unclamped = mxfp4::e8m0_scale(255, NaiveBits::E8M0_NO_CLAMP);
    assert_eq!(clamped, mxfp4::exp2(127));
    assert_eq!(unclamped, mxfp4::exp2(128));
    assert_ne!(clamped, unclamped, "the missing clamp must be detectable");
    // The poison shows up as saturation in f32.
    let payload = [0x22u8]; // 1.0: clamped product finite, unclamped saturates
    let scales = [255u8];
    let ok = mxfp4::unpack_row(&payload, &scales, NaiveBits::NONE);
    let bad = mxfp4::unpack_row(&payload, &scales, NaiveBits::E8M0_NO_CLAMP);
    assert_eq!(ok[0], 2.0f32.powi(127), "clamped 1.0 * 2^127 is finite");
    assert_eq!(bad[0], f32::MAX, "unclamped 1.0 * 2^128 saturates to f32::MAX");
}

/// BOTH RUNS. The scale-off-by-one detector: block b reading b+1 changes the
/// last block to a zero scale (a silent zeroing of the last 32 columns).
#[test]
fn detector_scale_off_by_one_is_detected() {
    let payload = [0x11u8; 32]; // two complete 32-element blocks
    let scales = [127u8, 128u8];
    let correct = mxfp4::unpack_row(&payload, &scales, NaiveBits::NONE);
    let shifted = mxfp4::unpack_row(&payload, &scales, NaiveBits::SCALE_OFF_BY_ONE);
    assert_eq!(correct[0].to_bits(), 0.5f32.to_bits());
    assert_eq!(correct[31].to_bits(), 0.5f32.to_bits());
    // Shifted: block 0 reads scale[1] = 128 -> 1.0; block 1 falls back to
    // E8M0 byte zero, which means 2^-127, NOT a floating-point zero scale.
    assert_eq!(shifted[0].to_bits(), 1.0f32.to_bits());
    assert_eq!(shifted[31].to_bits(), 1.0f32.to_bits());
    assert_eq!(shifted[32].to_bits(), ((0.5f64 * 2.0f64.powi(-127)) as f32).to_bits());
    assert!(!bits_eq(&correct, &shifted), "the off-by-one must be detectable");
}

/// BOTH RUNS. The E2M1 codebook is the golden 16-entry table, and nibble 8 is
/// `-0.0` (a different bit pattern from nibble 0's `0.0`).
#[test]
fn detector_e2m1_codebook_is_the_golden_table() {
    let want: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    for i in 0..16 {
        assert_eq!(
            E2M1_CODEBOOK[i].to_bits(),
            want[i].to_bits(),
            "E2M1 codebook entry {i}"
        );
    }
    assert_ne!(
        E2M1_CODEBOOK[8].to_bits(),
        E2M1_CODEBOOK[0].to_bits(),
        "nibble 8 is -0.0, a different bit pattern from nibble 0's 0.0"
    );
    // The device LUT's contents, as bit patterns, are the same table.
    let bits = mxfp4::e2m1_codebook_bits();
    for i in 0..16 {
        assert_eq!(bits[i], want[i].to_bits(), "device LUT entry {i}");
    }
}

// ---------------------------------------------------------------------------
// the reference GEMM (independent of the crate's own implementation)
// ---------------------------------------------------------------------------

/// A deliberately naive, independent reference: dequantize the whole slice to
/// f32 with the documented semantics, then do a plain triple loop.
///
/// This is NOT the crate's `grouped_gemm` — it is the oracle the crate's
/// implementation is checked against, so a bug in the crate's loop structure
/// cannot hide behind itself.
fn reference_gemm(
    s: &SliceBytes,
    x: &[f32],
    plan: &GroupedPlan,
    p: Proj,
    naive: NaiveBits,
) -> Vec<f32> {
    let b = s.proj(p);
    let in_cols = p.slice_in_cols();
    let out_rows = p.slice_rows();
    // Dequantize the whole projection.
    let mut w = vec![0.0f32; out_rows * in_cols];
    for r in 0..out_rows {
        for k in 0..in_cols {
            let byte = b.payload[r * p.payload_cols() + k / 2];
            let nib = if naive.has(NaiveBits::NIBBLE_SWAP) {
                if k % 2 == 0 {
                    (byte >> 4) & 0x0f
                } else {
                    byte & 0x0f
                }
            } else if k % 2 == 0 {
                byte & 0x0f
            } else {
                (byte >> 4) & 0x0f
            };
            let sbyte = if naive.has(NaiveBits::SCALE_OFF_BY_ONE) {
                *b.scales.get(r * p.scale_cols() + k / 32 + 1).unwrap_or(&0)
            } else {
                b.scales[r * p.scale_cols() + k / 32]
            };
            let scale = mxfp4::e8m0_scale(sbyte, naive);
            let prod = f64::from(E2M1_CODEBOOK[nib as usize]) * scale;
            w[r * in_cols + k] = prod.clamp(-(f32::MAX as f64), f32::MAX as f64) as f32;
        }
    }
    let mut out = vec![0.0f32; plan.total_tokens() * out_rows];
    for g in &plan.groups {
        for t in 0..g.tokens {
            let xrow = &x[(g.token_offset + t) * in_cols..(g.token_offset + t + 1) * in_cols];
            for r in 0..out_rows {
                let mut acc = 0.0f32;
                for k in 0..in_cols {
                    acc += w[r * in_cols + k] * xrow[k];
                }
                out[(g.token_offset + t) * out_rows + r] = acc;
            }
        }
    }
    out
}
