//! Decomposition equivalence (split-KV decode + reduce ≡ chunked prefill ≡
//! two-pass) — the CPU proof behind `kernels/attn_decode_splitkv.cu` +
//! `attn_reduce.cu` + `attn_prefill_chunk.cu` before any GPU run is trusted.
//!
//! # Two-run classification
//!
//! NEGATIVES — FAIL under `MIMO26_SPIKE_NAIVE=1`, PASS on the correct impl:
//!   * `splitkv_sink_counted_once_at_reduce`
//!   * `chunked_prefill_rescales_running_max`
//!
//! BOTH RUNS (invariants + detection oracles):
//!   `decompositions_match_two_pass`,
//!   `splitkv_sink_per_split_oracle_diverges`,
//!   `no_running_rescale_oracle_diverges`.
//!
//! Trap statements: the sink column exists once per query, so the split-KV
//! reduce must add it ONCE (a sink folded into every split's `(m, l)` counts it
//! `n_splits` times); chunked prefill must rescale `(l, o)` by `exp(m_old−m_new)`
//! as the running max moves.

mod common;

use common::*;
use mimo26_attn::attn::{attention, attention_chunked, decode_split_kv};
use mimo26_attn::geom::AttnSpec;
use mimo26_attn::{bits_from_env, Family, NaiveBits};

const TOL_EQUIV: f64 = 1e-6;

fn case_swa() -> (AttnSpec, Vec<f32>, Vec<f32>, Vec<f32>, Vec<i64>, Vec<i64>, Vec<f32>) {
    let spec = AttnSpec {
        n_q: 8,
        n_kv: 2,
        d_qk: 16,
        d_v: 8,
        window: 32,
        ..AttnSpec::real(Family::Swa)
    };
    let (t, sn) = (3usize, 23usize);
    let mut rng = XorShift64::new(5150);
    let q = rng.fill_small(t * spec.n_q * spec.d_qk);
    let k = rng.fill_small(sn * spec.n_kv * spec.d_qk);
    let v = rng.fill_small(sn * spec.n_kv * spec.d_v);
    let q_pos = vec![20i64, 21, 22];
    let k_pos: Vec<i64> = (0..sn as i64).collect();
    let sink: Vec<f32> = (0..spec.n_q).map(|h| 1.5 - 0.3 * h as f32).collect();
    (spec, q, k, v, q_pos, k_pos, sink)
}

/// BOTH RUNS — every decomposition equals the two-pass reference (and the sink
/// reduce is exact across split counts).
#[test]
fn decompositions_match_two_pass() {
    let (spec, q, k, v, q_pos, k_pos, sink) = case_swa();
    let ref_out = attention(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&sink), NaiveBits::NONE).expect("two-pass");
    for n_splits in [1usize, 2, 3, 5, 8, 23] {
        let out = decode_split_kv(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&sink), n_splits, NaiveBits::NONE)
            .expect("split-kv");
        let d = max_abs_diff(&out, &ref_out);
        assert!(d <= TOL_EQUIV, "split-kv x{n_splits} ≠ two-pass (diff {d})");
    }
    for chunk in [1usize, 4, 7, 23] {
        let out = attention_chunked(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&sink), chunk, NaiveBits::NONE)
            .expect("chunked");
        let d = max_abs_diff(&out, &ref_out);
        assert!(d <= TOL_EQUIV, "chunked {chunk} ≠ two-pass (diff {d})");
    }
    // and no-sink GA flavor
    let spec_ga = AttnSpec {
        n_q: 8,
        n_kv: 2,
        d_qk: 16,
        d_v: 8,
        ..AttnSpec::real(Family::Ga)
    };
    let ref_ga = attention(&spec_ga, &q, &k, &v, &q_pos, &k_pos, None, NaiveBits::NONE).expect("two-pass");
    for n_splits in [2usize, 5] {
        let out = decode_split_kv(&spec_ga, &q, &k, &v, &q_pos, &k_pos, None, n_splits, NaiveBits::NONE)
            .expect("split-kv");
        let d = max_abs_diff(&out, &ref_ga);
        assert!(d <= TOL_EQUIV, "GA split-kv x{n_splits} ≠ two-pass (diff {d})");
    }
}

/// NEGATIVE (env-default). The sink column must be added ONCE at the reduce.
/// Flips on the naive run (one sink per split/chunk).
#[test]
fn splitkv_sink_counted_once_at_reduce() {
    let (spec, q, k, v, q_pos, k_pos, sink) = case_swa();
    let ref_out = attention(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&sink), NaiveBits::NONE).expect("two-pass");
    let out = decode_split_kv(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&sink), 4, bits_from_env()).expect("split-kv");
    let d = max_abs_diff(&out, &ref_out);
    assert!(
        d <= TOL_EQUIV,
        "split-KV sink accounting wrong: diverged {d} from the two-pass reference (sink must enter the denominator ONCE)"
    );
}

/// BOTH RUNS — attribution: per-split sink really diverges when the sink is
/// strong and split maxima differ.
#[test]
fn splitkv_sink_per_split_oracle_diverges() {
    let (spec, q, k, v, q_pos, k_pos, _sink) = case_swa();
    let sink: Vec<f32> = vec![8.0; spec.n_q]; // dominant sink
    let ref_out = attention(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&sink), NaiveBits::NONE).expect("two-pass");
    let bad = decode_split_kv(
        &spec,
        &q,
        &k,
        &v,
        &q_pos,
        &k_pos,
        Some(&sink),
        6,
        NaiveBits::SINK_PER_SPLIT,
    )
    .expect("split-kv");
    let d = max_abs_diff(&bad, &ref_out);
    assert!(d > 1e-3, "per-split-sink oracle has no detection power ({d})");
}

/// NEGATIVE (env-default). Chunked prefill must rescale the running `(m, l, o)`
/// as the max moves. Flips on the naive run.
#[test]
fn chunked_prefill_rescales_running_max() {
    let (spec, q, k, v, q_pos, k_pos, sink) = case_swa();
    // logits grow with j (later chunks dominate) — the rescale's worst case
    let k_growing: Vec<f32> = k
        .iter()
        .enumerate()
        .map(|(i, &x)| x + 0.05 * (i / (spec.n_kv * spec.d_qk)) as f32)
        .collect();
    let ref_out =
        attention(&spec, &q, &k_growing, &v, &q_pos, &k_pos, Some(&sink), NaiveBits::NONE).expect("two-pass");
    let out = attention_chunked(
        &spec,
        &q,
        &k_growing,
        &v,
        &q_pos,
        &k_pos,
        Some(&sink),
        4,
        bits_from_env(),
    )
    .expect("chunked");
    let d = max_abs_diff(&out, &ref_out);
    assert!(
        d <= TOL_EQUIV,
        "chunked prefill running-max rescale broken: diverged {d} from two-pass"
    );
}

/// BOTH RUNS — attribution for the rescale.
#[test]
fn no_running_rescale_oracle_diverges() {
    let (spec, q, k, v, q_pos, k_pos, sink) = case_swa();
    let k_growing: Vec<f32> = k
        .iter()
        .enumerate()
        .map(|(i, &x)| x + 0.05 * (i / (spec.n_kv * spec.d_qk)) as f32)
        .collect();
    let ref_out =
        attention(&spec, &q, &k_growing, &v, &q_pos, &k_pos, Some(&sink), NaiveBits::NONE).expect("two-pass");
    let bad = attention_chunked(
        &spec,
        &q,
        &k_growing,
        &v,
        &q_pos,
        &k_pos,
        Some(&sink),
        4,
        NaiveBits::NO_RUNNING_RESCALE,
    )
    .expect("chunked");
    let d = max_abs_diff(&bad, &ref_out);
    assert!(d > 1e-3, "no-rescale oracle has no detection power ({d})");
}
