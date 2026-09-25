//! Padded-expert negative — **a padded expert row must never be read**.
//!
//! ADVISOR-I4 §3.2 step 3: "negatives: a swapped nibble order and a scale off
//! by one must both fail; **a padded-expert negative**."
//!
//! # Why this is a trap
//!
//! A grouped batch is padded to a fixed expert count so the grid is uniform
//! (`gridDim.y = n_experts + padded`). The padding is not a real expert: its
//! slice bytes are whatever follows the resident slices — the next layer's
//! experts, a different rank's slice, or uninitialised memory. Reading them
//! injects arbitrary values into the sum, and the result is *plausible*: the
//! output has the right shape and the right magnitude, so nothing downstream
//! notices. It is the same failure class as the scale-grid padding trim (T2).
//!
//! # The contract
//!
//! * `GroupedPlan::expert_count()` is the number of **real** experts;
//! * `padded_expert_count()` is the grid's `gridDim.y`;
//! * a padded expert has zero tokens and produces no output rows;
//! * the kernel returns before any load when `blockIdx.y >= expert_count`.
//!
//! # Classification
//!
//! * **NEGATIVE** (must FAIL under `MIMO26_SPIKE_NAIVE=1`):
//!   `padded_expert_rows_are_never_read`,
//!   `padded_expert_does_not_change_the_output`.
//! * **BOTH-RUNS**: `detector_pad_read_is_detected`,
//!   `padded_plan_geometry_is_uniform`.

mod common;

use common::*;
use mimo26_expert::grouped::{self, GroupedPlan};
use mimo26_expert::slice::{self, Proj};
use mimo26_expert::NaiveBits;

/// Build a grouped image where the padding slots hold a **poison** slice: a
/// slice whose bytes are all `0xFF` (nibble 15 = -6.0, scale byte 255 = the
/// reserved clamp). If the kernel reads it, the output changes visibly.
fn poisoned_image(real: &[SliceBytes], padded: usize) -> Vec<u8> {
    let mut out = grouped_image(real);
    for _ in 0..padded {
        out.extend(std::iter::repeat(0xFFu8).take(slice::QUARTER_SLICE_BYTES));
    }
    out
}

/// NEGATIVE. The padded slots must not change the output at all.
#[test]
fn padded_expert_rows_are_never_read() {
    let naive = naive_env();
    let s = exact_slice(2001);
    let plan = GroupedPlan::uniform(2, 4, 3);
    let x = tokens(21, plan.total_tokens(), Proj::Gate.in_cols());

    let clean = grouped_image(&[s.clone(), s.clone()]);
    let poisoned = poisoned_image(&[s.clone(), s.clone()], 3);

    let a = grouped::grouped_gemm(&clean, &x, &plan, Proj::Gate, naive).expect("gemm clean");
    let b = grouped::grouped_gemm(&poisoned, &x, &plan, Proj::Gate, naive).expect("gemm poisoned");

    assert!(
        bits_eq(&a.data, &b.data),
        "padded expert rows were READ: the poisoned padding changed the output \
         (first diff at {:?})",
        first_bit_diff(&a.data, &b.data)
    );
}

/// NEGATIVE. The same, through the full expert FFN (gate/up/down), where a
/// padded read would also poison the silu activation.
#[test]
fn padded_expert_does_not_change_the_output() {
    let naive = naive_env();
    let s = exact_slice(2002);
    let plan = GroupedPlan::uniform(1, 2, 2);
    let x = tokens(22, plan.total_tokens(), slice::HIDDEN);

    let clean = grouped_image(&[s.clone()]);
    let poisoned = poisoned_image(&[s.clone()], 2);

    let a = grouped::expert_ffn_self_contained(&clean, &x, &plan, naive).expect("ffn clean");
    let b = grouped::expert_ffn_self_contained(&poisoned, &x, &plan, naive).expect("ffn poisoned");
    assert!(
        bits_eq(&a.data, &b.data),
        "padded expert rows were READ in the FFN: the poisoned padding changed \
         the output (first diff at {:?})",
        first_bit_diff(&a.data, &b.data)
    );
}

/// BOTH RUNS. The detector: with [`NaiveBits::PAD_ROW_READ`] the padding IS
/// read, and the output changes — so the check above has real detection power.
#[test]
fn detector_pad_read_is_detected() {
    let s = exact_slice(2003);
    let plan = GroupedPlan::uniform(2, 4, 3);
    let x = tokens(23, plan.total_tokens(), Proj::Gate.in_cols());
    let poisoned = poisoned_image(&[s.clone(), s.clone()], 3);

    let correct =
        grouped::grouped_gemm(&poisoned, &x, &plan, Proj::Gate, NaiveBits::NONE).expect("gemm");
    assert_eq!(correct.data.len(), plan.total_tokens() * Proj::Gate.slice_rows());
    let err = grouped::grouped_gemm(&poisoned, &x, &plan, Proj::Gate, NaiveBits::PAD_ROW_READ)
        .expect_err("the naive loop must report its attempted padded access");
    assert!(matches!(err, mimo26_expert::ExpertError::PaddedRowRead { expert: 2, row: 0 }));
}

/// BOTH RUNS. The plan's geometry: the grid is uniform over real + padded, but
/// only the real experts do work, and a padded expert has no output rows.
#[test]
fn padded_plan_geometry_is_uniform() {
    let plan = GroupedPlan::uniform(4, 8, 4);
    assert_eq!(plan.expert_count(), 4);
    assert_eq!(plan.padded_expert_count(), 8);
    assert_eq!(plan.total_tokens(), 32);
    assert_eq!(plan.max_tokens(), 8);
    assert_eq!(plan.padded_output_rows(), 0);
    assert!(!plan.is_padded(3));
    assert!(plan.is_padded(4));
    assert!(plan.is_padded(7));
    plan.validate().expect("plan validates");

    let g = grouped::launch_geometry(Proj::Gate, 8, plan.padded_expert_count());
    assert_eq!(g.grid_y, 8, "the grid covers real + padded experts");
    assert_eq!(g.live_blocks(plan.expert_count()), u64::from(g.grid_x) * 4);
    assert_eq!(g.blocks(), u64::from(g.grid_x) * 8);
}

/// BOTH RUNS. A padded expert's slice is never addressed: `grouped_proj` on a
/// padded index is out of range and must fail loud rather than read.
#[test]
fn padded_expert_slice_is_out_of_range() {
    let s = exact_slice(2004);
    let image = grouped_image(&[s.clone()]);
    // Expert 0 is real; expert 1 is padding and is not in the image.
    assert!(slice::grouped_proj(&image, 0, Proj::Gate).is_ok());
    let err = slice::grouped_proj(&image, 1, Proj::Gate).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("outside the grouped image"),
        "a padded expert's slice must be out of range, got: {msg}"
    );
}
