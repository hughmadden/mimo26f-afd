//! CPU reference checks for the expert WIP. NOT a hardware L3 receipt.
//!
//! Numerical cases below use synthetic checkpoint-shaped bytes and an independently
//! decoded scalar reference, never the implementation's naive flags. Real fixture
//! metadata is checked separately. Full real-weight GPU oracle parity remains a
//! distinct required gate. M=2048*8/256 is 64, not 8 or 2048.
//!
//! NEGATIVE: l3_gemm_matches_the_reference_at_every_m, l3_ffn_matches_the_reference.
//! All other tests use explicit modes and must pass in both runs.
mod common;
use common::*;
use mimo26_expert::grouped::{self, GroupedPlan};
use mimo26_expert::slice::{self, Proj};
use mimo26_expert::NaiveBits;

const L3_M_NUMERIC: [usize; 3] = [1, 8, 64];
const RTOL: f64 = 1e-5;
const ATOL: f64 = 1e-5;

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite() && w.is_finite(), "nonfinite output at {i}");
        assert!((g as f64 - w as f64).abs() <= ATOL + RTOL * (w as f64).abs(),
            "output {i}: got {g}, want {w}");
    }
}

#[test]
fn fixture_contract_is_the_pinned_27_real_blocks() {
    let fx = load_fixture().expect("real-block fixture");
    assert_eq!(fx.block, 32);
    assert_eq!(fx.layers, vec![1, 24, 46]);
    assert_eq!(fx.experts, vec![0, 7, 255]);
    assert_eq!(fx.projections, vec!["gate_proj", "up_proj", "down_proj"]);
    assert_eq!(fx.blocks.len(), 27);
    assert_eq!(fx.total_samples(), 27 * 2048);
    for b in &fx.blocks {
        let p = b.proj().unwrap();
        assert_eq!(b.shape, [p.out_rows(), p.in_cols()]);
        assert_eq!(b.positions.len(), b.expected_f32_bits.len());
        assert_eq!(b.weight_sha256.len(), 64);
        assert_eq!(b.scale_sha256.len(), 64);
        assert!(b.positions.iter().all(|&(r,c)| r < b.shape[0] && c < b.shape[1]));
    }
}

// Verify actual bytes, not just the presence of a non-empty digest string.
#[test]
fn fixture_sha256_companion_matches() {
    let path = fixture_path();
    let text = std::fs::read_to_string(path.with_extension("json.sha256")).unwrap();
    let want = text.split_whitespace().next().unwrap();
    let output = std::process::Command::new("sha256sum").arg(&path).output().unwrap();
    assert!(output.status.success());
    let output = String::from_utf8(output.stdout).unwrap();
    assert_eq!(output.split_whitespace().next().unwrap(), want);
}

#[test]
fn fixture_records_no_255_scale_and_no_saturation_in_real_data() {
    for b in load_fixture().unwrap().blocks {
        assert_eq!(b.scale_byte_255_count, 0);
        assert_eq!(b.saturated_count, 0);
        assert!(b.scale_byte_min >= 118 && b.scale_byte_max <= 125);
    }
}

#[test]
fn fixture_expected_bits_are_finite() {
    for b in load_fixture().unwrap().blocks {
        assert!(b.expected_f32_bits.iter().all(|&v| f32::from_bits(v).is_finite()));
    }
}

#[test]
fn l3_gemm_matches_the_reference_at_every_m() {
    let slices: Vec<_> = (0..2).map(|e| exact_slice(3000 + e)).collect();
    let image = grouped_image(&slices);
    for m in L3_M_NUMERIC {
        let plan = GroupedPlan::uniform(2, m, 0);
        for p in Proj::ALL {
            let x = tokens(31 + m as u64, plan.total_tokens(), p.slice_in_cols());
            let got = grouped::grouped_gemm(&image, &x, &plan, p, naive_env()).unwrap();
            assert_close(&got.data, &reference_gemm(&slices, &x, &plan, p));
        }
    }
}

#[test]
fn l3_prefill_routed_row_count_is_pinned() {
    let plan = GroupedPlan::uniform(256, 2048 * 8 / 256, 0);
    plan.validate().unwrap();
    assert_eq!(plan.max_tokens(), 64);
    assert_eq!(plan.total_tokens(), 16384);
    // This is structural coverage, not a 256-expert numerical receipt.
    assert_eq!(L3_M_NUMERIC, [1, 8, 64]);
}

#[test]
fn l3_ffn_matches_the_reference() {
    let slices = vec![exact_slice(4000), exact_slice(4001)];
    let image = grouped_image(&slices);
    let plan = GroupedPlan::uniform(2, 4, 0);
    let x = tokens(41, plan.total_tokens(), slice::HIDDEN);
    let got = grouped::expert_ffn_self_contained(&image, &x, &plan, naive_env()).unwrap();
    let g = reference_gemm(&slices, &x, &plan, Proj::Gate);
    let u = reference_gemm(&slices, &x, &plan, Proj::Up);
    // V2 intermediate stays local: no gather and no zero embedding.
    let mut h = vec![0.0; plan.total_tokens() * Proj::Down.slice_in_cols()];
    for t in 0..plan.total_tokens() {
        for c in 0..Proj::Gate.slice_rows() {
            let i = t * Proj::Gate.slice_rows() + c;
            h[t * Proj::Down.slice_in_cols() + c] = g[i] / (1.0 + (-g[i]).exp()) * u[i];
        }
    }
    assert_close(&got.data, &reference_gemm(&slices, &h, &plan, Proj::Down));
}

#[test]
fn l3_f32_tolerance_catches_a_single_swapped_nibble() {
    let mut s = exact_slice(5001);
    // Guarantee unequal nibbles and a nonzero input; a random byte can have
    // identical nibbles and give a powerless mutation.
    s.gate.payload[0] = 0x71;
    let mut bad = s.clone();
    bad.gate.payload[0] = 0x17;
    let plan = GroupedPlan::uniform(1, 1, 0);
    let mut x = vec![0.0; Proj::Gate.slice_in_cols()];
    x[0] = 1.0;
    let good = grouped::grouped_gemm(&grouped_image(&[s]), &x, &plan, Proj::Gate, NaiveBits::NONE).unwrap();
    let wrong = grouped::grouped_gemm(&grouped_image(&[bad]), &x, &plan, Proj::Gate, NaiveBits::NONE).unwrap();
    assert!(max_abs_diff(&good.data, &wrong.data) > ATOL);
}

#[test]
fn l3_bf16_accumulator_is_detectably_wrong() {
    let image = grouped_image(&[exact_slice(5002)]);
    let plan = GroupedPlan::uniform(1, 1, 0);
    let x = tokens(52, 1, Proj::Gate.slice_in_cols());
    let good = grouped::grouped_gemm(&image, &x, &plan, Proj::Gate, NaiveBits::NONE).unwrap();
    let bad = grouped::grouped_gemm(&image, &x, &plan, Proj::Gate, NaiveBits::BF16_ACCUM).unwrap();
    // No claim that BF16 accumulation meets 1e-2 over a 4096-term dot product.
    assert!(max_abs_diff(&good.data, &bad.data) > ATOL);
}

#[test]
fn l3_m_sizes_are_the_packet_set() {
    assert_eq!(grouped::M_SIZES, [1, 2, 4, 8, 16, 64, 256]);
    for m in grouped::M_SIZES { GroupedPlan::uniform(2, m, 0).validate().unwrap(); }
    assert!(GroupedPlan::uniform(2, 3, 0).validate().is_err());
}

// Independent codec expression: no crate LUT, scale helper, or naive selector.
fn reference_weight(payload: &[u8], scales: &[u8], k: usize) -> f32 {
    let nib = (payload[k / 2] >> (4 * (k % 2))) & 15;
    let magnitude = [0.0f64, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(nib & 7) as usize];
    let sign = if nib & 8 == 0 { 1.0 } else { -1.0 };
    let scale = 2.0f64.powi(i32::from(scales[k / 32].min(254)) - 127);
    (sign * magnitude * scale).clamp(-(f32::MAX as f64), f32::MAX as f64) as f32
}

fn reference_gemm(slices: &[SliceBytes], x: &[f32], plan: &GroupedPlan, p: Proj) -> Vec<f32> {
    let mut out = vec![0.0; plan.total_tokens() * p.slice_rows()];
    for g in &plan.groups {
        let b = slices[g.expert].proj(p);
        for r in 0..p.slice_rows() {
            let payload = &b.payload[r * p.payload_cols()..(r+1) * p.payload_cols()];
            let scales = &b.scales[r * p.scale_cols()..(r+1) * p.scale_cols()];
            let w: Vec<_> = (0..p.slice_in_cols()).map(|k| reference_weight(payload, scales, k)).collect();
            for t in 0..g.tokens {
                let token = g.token_offset + t;
                let mut acc = 0.0f32;
                for k in 0..p.slice_in_cols() { acc += w[k] * x[token * p.slice_in_cols() + k]; }
                out[token * p.slice_rows() + r] = acc;
            }
        }
    }
    out
}
