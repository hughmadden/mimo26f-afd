//! Determinism — the same input must produce byte-identical slice files.
//!
//! Classification: **BOTH-RUNS** (no naive flag; the naive run produces
//! different bytes but is still deterministic, so these pass both runs).
//!
//! Kills: any dependence on iteration order, hash-map ordering, uninitialised
//! memory, the region offsets, or the process environment.

mod common;

use mimo26_repack::geom::{self, Proj};
use mimo26_repack::mxfp4::Mxfp4Naive;
use mimo26_repack::repack::{build_slice, read_expert};
use mimo26_repack::safetensors::SafetensorsHeader;
use mimo26_repack::sha256;

#[test]
fn same_input_same_slice_bytes() {
    let t = common::synth_expert(501);
    for rank in 0..geom::EP_RANKS {
        let a = build_slice(&t, rank, Mxfp4Naive::NONE).expect("build slice");
        let b = build_slice(&t, rank, Mxfp4Naive::NONE).expect("build slice");
        assert_eq!(a, b, "rank {rank} slice is not deterministic");
        assert_eq!(
            sha256::hex(&sha256::sha256(&a)),
            sha256::hex(&sha256::sha256(&b)),
            "rank {rank} sha256 is not deterministic"
        );
    }
}

#[test]
fn slice_bytes_are_independent_of_construction_order() {
    // Build the same expert twice from independently generated tensors with the
    // same seed — the fixture RNG is deterministic, so the bytes must match.
    let t1 = common::synth_expert(502);
    let t2 = common::synth_expert(502);
    assert_eq!(t1, t2, "fixture is not deterministic");
    for rank in 0..geom::EP_RANKS {
        assert_eq!(
            build_slice(&t1, rank, Mxfp4Naive::NONE).unwrap(),
            build_slice(&t2, rank, Mxfp4Naive::NONE).unwrap()
        );
    }
}

#[test]
fn different_ranks_produce_different_slices() {
    let t = common::synth_expert(503);
    let mut shas = Vec::new();
    for rank in 0..geom::EP_RANKS {
        let s = build_slice(&t, rank, Mxfp4Naive::NONE).expect("build slice");
        shas.push(sha256::hex(&sha256::sha256(&s)));
    }
    let mut uniq = shas.clone();
    uniq.sort();
    uniq.dedup();
    assert_eq!(uniq.len(), geom::EP_RANKS, "ranks produced duplicate slices");
}

#[test]
fn repack_from_a_real_safetensors_file_is_deterministic() {
    // End to end through the reader: write a real safetensors file, read it
    // twice, repack twice, compare bytes and shas.
    let dir = common::scratch_dir("determinism");
    let t = common::synth_expert(504);
    let path = common::write_expert_safetensors(&dir, "shard.safetensors", 3, 7, &t);
    let h1 = SafetensorsHeader::read(&path).expect("read header");
    let h2 = SafetensorsHeader::read(&path).expect("read header");
    let e1 = read_expert(&h1, &path, 3, 7).expect("read expert");
    let e2 = read_expert(&h2, &path, 3, 7).expect("read expert");
    assert_eq!(e1, e2);
    assert_eq!(e1, t, "reader must return the tensors verbatim");
    for rank in 0..geom::EP_RANKS {
        let a = build_slice(&e1, rank, Mxfp4Naive::NONE).unwrap();
        let b = build_slice(&e2, rank, Mxfp4Naive::NONE).unwrap();
        assert_eq!(a, b);
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn slice_bytes_do_not_depend_on_the_environment() {
    // The naive env var must not leak into the correct path: an explicit
    // Mxfp4Naive::NONE is NONE regardless of MIMO26_SPIKE_NAIVE.
    let t = common::synth_expert(505);
    let before = build_slice(&t, 0, Mxfp4Naive::NONE).unwrap();
    std::env::set_var("MIMO26_REPACK_NAIVE", "1");
    let after = build_slice(&t, 0, Mxfp4Naive::NONE).unwrap();
    std::env::remove_var("MIMO26_REPACK_NAIVE");
    assert_eq!(before, after, "explicit NONE must ignore the env");
}

#[test]
fn slice_tiles_the_whole_buffer_with_no_padding() {
    // Every byte of the slice belongs to a projection region: the last scale
    // byte ends exactly at the slice end.
    let t = common::synth_expert(506);
    let end = Proj::Down.slice_scale_off() + Proj::Down.slice_scale_bytes();
    assert_eq!(end, geom::QUARTER_SLICE_BYTES);
    for rank in 0..geom::EP_RANKS {
        let s = build_slice(&t, rank, Mxfp4Naive::NONE).unwrap();
        assert_eq!(s.len(), geom::QUARTER_SLICE_BYTES);
    }
}
