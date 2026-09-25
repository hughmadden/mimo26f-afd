//! Trap negatives — T14 (nibble order) and T10 (E8M0 clamp / scale mapping).
//!
//! Classification:
//!
//! * **NEGATIVE** (must FAIL under `MIMO26_SPIKE_NAIVE=1` /
//!   `MIMO26_REPACK_NAIVE=1`): `t14_*`, `t10_*` — they call the env-default
//!   entry points, so the naive implementation is selected and the round-trip
//!   check must catch it.
//! * **BOTH-RUNS** (pass both runs): `detector_*` — they pass an explicit
//!   `Mxfp4Naive` flag and assert the check *detects* the wrong implementation.
//!
//! # The round-trip check
//!
//! The repack is a **pure byte permutation**: it must not touch a nibble, a
//! scale byte, or a row order. So the check is:
//!
//! ```text
//! decode_correct(slice_proj(build_slice(t, rank)))  ==  rows_of(decode_correct(t), rank)
//! ```
//!
//! The decode side is always the **correct** semantics; only the repack varies.
//! A nibble-swapped or scale-shifted slice is the right size and "loads" — only
//! this check catches it (AGENTS.md §4.5).
//!
//! A second, independent detector varies the **decode** semantics against a
//! correct slice, which pins the semantics themselves (T14/T10) rather than
//! just the permutation.

mod common;

use mimo26_repack::geom::{self, Proj};
use mimo26_repack::mxfp4::{self, Mxfp4Naive};
use mimo26_repack::repack::{build_slice, expert_to_f32, slice_rows_of_full, slice_to_f32};

/// The reference decode of the source tensors, always with correct semantics.
fn reference(t: &mimo26_repack::ExpertTensors, p: Proj, rank: usize) -> Vec<f32> {
    let full = expert_to_f32(t, p, Mxfp4Naive::NONE);
    slice_rows_of_full(&full, p, rank)
}

/// Round-trip one projection of one rank: repack under `repack_naive`, decode
/// with the CORRECT semantics, compare bitwise against the source.
fn roundtrip_proj(
    t: &mimo26_repack::ExpertTensors,
    rank: usize,
    p: Proj,
    repack_naive: Mxfp4Naive,
) -> bool {
    let slice = build_slice(t, rank, repack_naive).expect("build slice");
    let got = slice_to_f32(&slice, p, Mxfp4Naive::NONE).expect("slice_to_f32");
    common::bits_eq(&got, &reference(t, p, rank))
}

/// Round-trip all three projections of one rank.
fn roundtrip_all(t: &mimo26_repack::ExpertTensors, rank: usize, repack_naive: Mxfp4Naive) -> bool {
    Proj::ALL.iter().all(|p| roundtrip_proj(t, rank, *p, repack_naive))
}

// ---------------------------------------------------------------------------
// NEGATIVE — T14 nibble order
// ---------------------------------------------------------------------------

#[test]
fn t14_roundtrip_matches_the_reference_semantics() {
    // NEGATIVE: under the naive run the slice is nibble-swapped and this fails.
    let t = common::exact_expert(101);
    let naive = common::naive_env();
    for rank in 0..geom::EP_RANKS {
        assert!(
            roundtrip_all(&t, rank, naive),
            "T14: rank {rank} slice does not round-trip to the source tensors \
             (nibble order swapped?)"
        );
    }
}

#[test]
fn t14_roundtrip_is_bit_exact_not_tolerance_based() {
    // NEGATIVE. The fixture uses only exactly-representable values, so the
    // comparison is bitwise: a single swapped nibble anywhere in 3.3 MB fails.
    let t = common::exact_expert(102);
    let naive = common::naive_env();
    let slice = build_slice(&t, 1, naive).expect("build slice");
    let got = slice_to_f32(&slice, Proj::Gate, Mxfp4Naive::NONE).expect("slice_to_f32");
    let want = reference(&t, Proj::Gate, 1);
    assert_eq!(got.len(), want.len());
    assert_eq!(
        common::first_bit_diff(&got, &want),
        None,
        "T14: first bitwise difference at index {:?}",
        common::first_bit_diff(&got, &want)
    );
}

#[test]
fn t14_every_rank_and_projection_round_trips() {
    // NEGATIVE. 4 ranks x 3 projections x 2 fixtures.
    let naive = common::naive_env();
    for seed in [201u64, 202] {
        let t = common::exact_expert(seed);
        for rank in 0..geom::EP_RANKS {
            for p in Proj::ALL {
                assert!(
                    roundtrip_proj(&t, rank, p, naive),
                    "T14: seed {seed} rank {rank} {} failed round-trip",
                    p.name()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NEGATIVE — T10 E8M0 clamp and scale mapping
// ---------------------------------------------------------------------------

#[test]
fn t10_scale_byte_255_clamps_to_2_pow_127() {
    // NEGATIVE: the naive path returns 2^128 (a poison scale).
    let naive = common::naive_env();
    let s = mxfp4::e8m0_scale(255, naive);
    assert_eq!(
        s,
        mxfp4::exp2(127),
        "T10: E8M0 byte 255 must clamp to 2^127, got 2^{}",
        s.log2()
    );
    assert_eq!(s, 1.7014118346046923e38);
    // 254 is the largest unclamped byte and must be untouched.
    assert_eq!(mxfp4::e8m0_scale(254, naive), mxfp4::exp2(127));
    assert_eq!(mxfp4::e8m0_scale(127, naive), 1.0);
    assert_eq!(mxfp4::e8m0_scale(0, naive), mxfp4::exp2(-127));
}

#[test]
fn t10_roundtrip_with_reserved_scale_bytes() {
    // NEGATIVE. The fixture deliberately plants byte 255 in 1/8 of the scale
    // blocks; the round-trip must still be bit-exact.
    let t = common::synth_expert(301);
    let naive = common::naive_env();
    for rank in 0..geom::EP_RANKS {
        assert!(
            roundtrip_all(&t, rank, naive),
            "T10: rank {rank} slice does not round-trip with reserved scale bytes"
        );
    }
}

#[test]
fn t10_scale_mapping_is_per_32_element_block() {
    // NEGATIVE. A scale grid that is off by one block (or 16-wide) changes the
    // decoded values; the round-trip must catch it.
    let t = common::exact_expert(302);
    let naive = common::naive_env();
    let slice = build_slice(&t, 0, naive).expect("build slice");
    let got = slice_to_f32(&slice, Proj::Up, Mxfp4Naive::NONE).expect("slice_to_f32");
    let want = reference(&t, Proj::Up, 0);
    assert!(
        common::bits_eq(&got, &want),
        "T10: scale block mapping drifted (first diff {:?})",
        common::first_bit_diff(&got, &want)
    );
    // And the block width itself is pinned.
    assert_eq!(mxfp4::BLOCK, 32);
    assert_eq!(mxfp4::E8M0_BYTE_MAX, 254);
}

// ---------------------------------------------------------------------------
// BOTH-RUNS — the detector proves the check kills the wrong implementation
// ---------------------------------------------------------------------------

#[test]
fn detector_kills_swapped_nibble_order() {
    // BOTH-RUNS: explicit flags, so this passes under the naive run too.
    let t = common::exact_expert(401);
    // Correct repack: round-trips.
    assert!(roundtrip_all(&t, 0, Mxfp4Naive::NONE));
    // Nibble-swapped repack: must NOT round-trip.
    assert!(
        !roundtrip_all(&t, 0, Mxfp4Naive::of(Mxfp4Naive::NIBBLE_SWAP)),
        "T14 detector failed: a nibble-swapped slice round-tripped"
    );
    // The swap is observable on the bytes themselves.
    let good = build_slice(&t, 0, Mxfp4Naive::NONE).unwrap();
    let bad = build_slice(&t, 0, Mxfp4Naive::of(Mxfp4Naive::NIBBLE_SWAP)).unwrap();
    assert_ne!(good, bad, "the nibble swap must change the slice bytes");
    assert_eq!(good.len(), bad.len(), "the swap must not change the size");
}

#[test]
fn detector_kills_swapped_decode_semantics() {
    // BOTH-RUNS. The other direction: a correct slice decoded with swapped
    // semantics must not match the reference. This pins the semantics (T14),
    // not just the permutation.
    let t = common::exact_expert(402);
    let slice = build_slice(&t, 0, Mxfp4Naive::NONE).expect("build slice");
    let got = slice_to_f32(&slice, Proj::Gate, Mxfp4Naive::of(Mxfp4Naive::NIBBLE_SWAP))
        .expect("slice_to_f32");
    let want = reference(&t, Proj::Gate, 0);
    assert!(
        !common::bits_eq(&got, &want),
        "T14 detector failed: swapped decode semantics matched the correct slice"
    );
}

#[test]
fn detector_kills_unclamped_e8m0() {
    // BOTH-RUNS.
    //
    // NOTE on the fixture: `unpack_row` saturates to +-f32::MAX, so a scale of
    // 2^127 or 2^128 both saturate and the difference would be masked. The
    // detector therefore uses a fixture whose reserved byte 255 sits in a block
    // whose E2M1 values are ZERO — then the product is 0 * 2^127 = 0 vs
    // 0 * 2^128 = 0, which is also masked. So instead the check is done at the
    // scale level (exact, unmasked) plus a value-level check with a small
    // magnitude: byte 255 is compared against byte 254, whose scales are
    // 2^127 and 2^127 — identical — so the *clamp* is what makes 255 equal 254.
    let t = common::synth_expert(403);
    assert!(roundtrip_all(&t, 0, Mxfp4Naive::NONE));
    // The clamp is observable on the value itself, exactly.
    assert_eq!(
        mxfp4::e8m0_scale(255, Mxfp4Naive::NONE),
        mxfp4::e8m0_scale(254, Mxfp4Naive::NONE),
        "T10: byte 255 must clamp to the same scale as byte 254"
    );
    assert_ne!(
        mxfp4::e8m0_scale(255, Mxfp4Naive::NONE),
        mxfp4::e8m0_scale(255, Mxfp4Naive::of(Mxfp4Naive::E8M0_NO_CLAMP)),
        "T10: the naive path must produce a different scale for byte 255"
    );
    // And the unclamped path is exactly 2x the clamped one (2^128 vs 2^127).
    assert_eq!(
        mxfp4::e8m0_scale(255, Mxfp4Naive::of(Mxfp4Naive::E8M0_NO_CLAMP)),
        2.0 * mxfp4::e8m0_scale(255, Mxfp4Naive::NONE)
    );
    // Value-level: a block whose scale byte is 255 and whose E2M1 values are
    // small enough not to saturate. 0.5 * 2^127 = 8.5e37 < f32::MAX, so the
    // clamp is visible in the decoded value; the unclamped 0.5 * 2^128 =
    // 1.7e38 is still < f32::MAX, so it is visible too (no saturation mask).
    let packed = vec![mxfp4::pack_byte(1, 1)]; // two 0.5 codes
    let scales = vec![255u8];
    let clamped = mxfp4::unpack_row(&packed, &scales, Mxfp4Naive::NONE);
    let unclamped = mxfp4::unpack_row(&packed, &scales, Mxfp4Naive::of(Mxfp4Naive::E8M0_NO_CLAMP));
    assert_eq!(clamped[0], 0.5 * mxfp4::exp2(127) as f32);
    // NOTE (captain fix): cast AFTER the f64 multiply — `x * exp2(128) as f32`
    // casts 2^128 to f32 first (-> inf) because `as` binds tighter than `*`.
    assert_eq!(unclamped[0], (0.5 * mxfp4::exp2(128)) as f32);
    assert_ne!(
        clamped[0].to_bits(),
        unclamped[0].to_bits(),
        "T10 detector failed: the unclamped scale decoded to the same value"
    );
    assert!(
        unclamped[0] < f32::MAX,
        "the fixture must not saturate, or the clamp would be masked"
    );
}

#[test]
fn detector_kills_off_by_one_scale_mapping() {
    // BOTH-RUNS. A correct slice decoded with the scale block shifted by one
    // must differ.
    let t = common::exact_expert(404);
    assert!(roundtrip_all(&t, 0, Mxfp4Naive::NONE));
    let slice = build_slice(&t, 0, Mxfp4Naive::NONE).expect("build slice");
    let got = slice_to_f32(&slice, Proj::Gate, Mxfp4Naive::of(Mxfp4Naive::SCALE_OFF_BY_ONE))
        .expect("slice_to_f32");
    let want = reference(&t, Proj::Gate, 0);
    assert!(
        !common::bits_eq(&got, &want),
        "T10 detector failed: an off-by-one scale mapping matched the reference"
    );
}

#[test]
fn detector_kills_a_scale_shifted_repack() {
    // BOTH-RUNS. The repack itself shifts every scale row by one byte.
    let t = common::exact_expert(405);
    assert!(roundtrip_all(&t, 0, Mxfp4Naive::NONE));
    assert!(
        !roundtrip_all(&t, 0, Mxfp4Naive::of(Mxfp4Naive::SCALE_OFF_BY_ONE)),
        "T10 detector failed: a scale-shifted repack round-tripped"
    );
}

#[test]
fn detector_kills_a_shifted_slice_row() {
    // BOTH-RUNS. A slice built for the wrong rank is the right size and loads
    // fine; the round-trip must reject it.
    let t = common::exact_expert(406);
    let slice_r0 = build_slice(&t, 0, Mxfp4Naive::NONE).expect("build slice");
    let got = slice_to_f32(&slice_r0, Proj::Gate, Mxfp4Naive::NONE).expect("slice_to_f32");
    let want_r1 = reference(&t, Proj::Gate, 1);
    assert!(
        !common::bits_eq(&got, &want_r1),
        "rank detector failed: rank 0's slice matched rank 1's rows"
    );
}

#[test]
fn detector_kills_a_swapped_projection_order() {
    // BOTH-RUNS. gate and up have identical geometry, so a slice with them
    // swapped is the right size and loads — the round-trip must reject it.
    let t = common::exact_expert(407);
    let mut swapped = t.clone();
    std::mem::swap(&mut swapped.gate_w, &mut swapped.up_w);
    std::mem::swap(&mut swapped.gate_s, &mut swapped.up_s);
    let a = build_slice(&t, 0, Mxfp4Naive::NONE).expect("build slice");
    let b = build_slice(&swapped, 0, Mxfp4Naive::NONE).expect("build slice");
    assert_ne!(a, b, "gate/up swap must change the slice bytes");
    let got = slice_to_f32(&b, Proj::Gate, Mxfp4Naive::NONE).expect("slice_to_f32");
    let want = reference(&t, Proj::Gate, 0);
    assert!(
        !common::bits_eq(&got, &want),
        "projection-order detector failed: a gate/up-swapped slice round-tripped"
    );
}

#[test]
fn detector_kills_a_scale_region_shifted_by_one_byte() {
    // BOTH-RUNS. Shift the whole scale region by one byte inside the slice —
    // the classic "scale grid starts one byte early" bug.
    let t = common::exact_expert(408);
    let mut slice = build_slice(&t, 0, Mxfp4Naive::NONE).expect("build slice");
    let off = Proj::Gate.slice_scale_off();
    let len = Proj::Gate.slice_scale_bytes();
    slice.copy_within(off + 1..off + len, off);
    slice[off + len - 1] = 0;
    let got = slice_to_f32(&slice, Proj::Gate, Mxfp4Naive::NONE).expect("slice_to_f32");
    let want = reference(&t, Proj::Gate, 0);
    assert!(
        !common::bits_eq(&got, &want),
        "scale-shift detector failed: a one-byte-shifted scale region round-tripped"
    );
}

#[test]
fn detector_kills_a_payload_region_shifted_by_one_byte() {
    // BOTH-RUNS. Same for the payload: a one-byte shift re-pairs every nibble.
    let t = common::exact_expert(409);
    let mut slice = build_slice(&t, 0, Mxfp4Naive::NONE).expect("build slice");
    let off = Proj::Up.slice_payload_off();
    let len = Proj::Up.slice_payload_bytes();
    slice.copy_within(off + 1..off + len, off);
    slice[off + len - 1] = 0;
    let got = slice_to_f32(&slice, Proj::Up, Mxfp4Naive::NONE).expect("slice_to_f32");
    let want = reference(&t, Proj::Up, 0);
    assert!(
        !common::bits_eq(&got, &want),
        "payload-shift detector failed: a one-byte-shifted payload round-tripped"
    );
}

#[test]
fn detector_kills_a_16_wide_scale_grid() {
    // BOTH-RUNS. A scale grid that is 16 elements wide instead of 32 is the
    // classic K-block misalignment; the decoded values must differ.
    let t = common::exact_expert(410);
    let slice = build_slice(&t, 0, Mxfp4Naive::NONE).expect("build slice");
    let (payload, scales) = mimo26_repack::repack::slice_proj(&slice, Proj::Gate).expect("proj");
    let in_cols = Proj::Gate.in_cols();
    let srow = in_cols / 32;
    // Decode row 0 with a 16-wide grid (two scale bytes per 32-element block).
    let mut got = vec![0.0f32; in_cols];
    for k in 0..in_cols {
        let byte = payload[k / 2];
        let pair = mxfp4::decode_byte(byte, Mxfp4Naive::NONE);
        let block16 = k / 16;
        let sbyte = scales[block16.min(srow - 1)];
        got[k] = (pair[k % 2] as f64 * mxfp4::e8m0_scale(sbyte, Mxfp4Naive::NONE)) as f32;
    }
    let full = expert_to_f32(&t, Proj::Gate, Mxfp4Naive::NONE);
    let want = &full[0..in_cols];
    assert!(
        !common::bits_eq(&got, want),
        "K16 detector failed: a 16-wide scale grid matched the 32-wide reference"
    );
}

#[test]
fn detector_kills_a_swapped_nibble_pair_within_a_byte() {
    // BOTH-RUNS. The minimal T14 corruption: swap the two nibbles of ONE byte
    // in the middle of the gate payload. Everything else is correct.
    let t = common::exact_expert(411);
    let mut slice = build_slice(&t, 0, Mxfp4Naive::NONE).expect("build slice");
    let off = Proj::Gate.slice_payload_off();
    let len = Proj::Gate.slice_payload_bytes();
    // Pick a byte whose two nibbles actually differ, so the swap is observable.
    let idx = (off..off + len)
        .find(|i| (slice[*i] & 0x0f) != (slice[*i] >> 4))
        .expect("the fixture must contain a byte with two different nibbles");
    slice[idx] = slice[idx].rotate_left(4);
    let got = slice_to_f32(&slice, Proj::Gate, Mxfp4Naive::NONE).expect("slice_to_f32");
    let want = reference(&t, Proj::Gate, 0);
    assert!(
        !common::bits_eq(&got, &want),
        "T14 detector failed: a single swapped nibble pair round-tripped"
    );
}
