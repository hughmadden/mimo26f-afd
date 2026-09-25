//! Attention properties (R4-style invariants) + the T3/T4/T6/T9/c1 trap
//! negatives. Expectations for the trap negatives are computed **in this test
//! from the math definition** (a two-pass softmax written out longhand) — never
//! from the implementation under test.
//!
//! # Two-run classification
//!
//! NEGATIVES — FAIL under `MIMO26_SPIKE_NAIVE=1`, PASS on the correct impl:
//!   * `c1_ga_bitwise_sink_free`            (sink applied on GA)
//!   * `t3_ga_layers_are_not_windowed`      (GA inherits the SWA window)
//!   * `t6_sink_is_per_q_head`              (sink broadcast per KV head)
//!   * `t4_attn_scale_uses_d_qk`            (scale 1/√d_v — QK/V mixup)
//!   * `t9_start_pos_is_honored`            (positions zeroed)
//!
//! BOTH RUNS (properties + detection oracles):
//!   `sink_mass_zero_value_and_vanishing_sink`, `value_scale_linearity`,
//!   `gqa_packing_matches_kv_repeat`, `window_128_isolation`,
//!   `no_visible_key_gives_zero_row`, `fail_loud_shape_and_sink_checks`,
//!   `v_broadcast_bug_oracle_diverges`.

mod common;

use common::*;
use mimo26_attn::attn::{attention, attention_checked, attn_scale};
use mimo26_attn::geom::AttnSpec;
use mimo26_attn::{bits_from_env, AttnError, Family, NaiveBits};

/// The softmax definition written out longhand (f64), independent of
/// `mimo26_attn::attn`. `keys` = `(k_pos, k_row[d_qk], v_row[d_v])`.
fn analytic_softmax(
    scale: f64,
    sink: Option<f64>,
    q_row: &[f32],
    keys: &[(i64, Vec<f32>, Vec<f32>)],
) -> Vec<f32> {
    let d_v = keys[0].2.len();
    let mut m = f64::NEG_INFINITY;
    let mut s = Vec::new();
    for (_, k, _) in keys {
        let mut dot = 0.0f64;
        for i in 0..q_row.len() {
            dot += f64::from(q_row[i]) * f64::from(k[i]);
        }
        dot *= scale;
        s.push(dot);
        m = m.max(dot);
    }
    if let Some(b) = sink {
        m = m.max(b);
    }
    let mut l = 0.0f64;
    let mut o = vec![0.0f64; d_v];
    for (j, (_, _, v)) in keys.iter().enumerate() {
        let p = (s[j] - m).exp();
        l += p;
        for i in 0..d_v {
            o[i] += p * f64::from(v[i]);
        }
    }
    if let Some(b) = sink {
        l += (b - m).exp();
    }
    o.iter().map(|&x| (x / l) as f32).collect()
}

fn tiny_swa(n_q: usize, n_kv: usize) -> AttnSpec {
    AttnSpec {
        n_q,
        n_kv,
        d_qk: 4,
        d_v: 2,
        window: 8,
        theta: mimo26_attn::geom::SWA_ROPE_THETA,
        ..AttnSpec::real(Family::Swa)
    }
}

fn tiny_ga(n_q: usize, n_kv: usize) -> AttnSpec {
    AttnSpec {
        n_q,
        n_kv,
        d_qk: 4,
        d_v: 2,
        window: 4, // must be IGNORED on GA (T3)
        theta: mimo26_attn::geom::ROPE_THETA,
        ..AttnSpec::real(Family::Ga)
    }
}

fn rows(rng: &mut XorShift64, n: usize, d: usize) -> Vec<f32> {
    rng.fill_small(n * d)
}

/// NEGATIVE (env-default). c1: a GA layer with a sink argument must be
/// **bitwise identical** to one without. Flips on the naive run.
#[test]
fn c1_ga_bitwise_sink_free() {
    let spec = tiny_ga(4, 2);
    let (t, sn) = (2usize, 3usize);
    let mut rng = XorShift64::new(11);
    let q = rows(&mut rng, t * spec.n_q, spec.d_qk);
    let k = rows(&mut rng, sn * spec.n_kv, spec.d_qk);
    let v = rows(&mut rng, sn * spec.n_kv, spec.d_v);
    let q_pos = vec![6i64, 7];
    let k_pos = vec![0i64, 5, 6];
    let sink = vec![0.9f32, -0.3, 1.7, 0.25];
    let without = attention(&spec, &q, &k, &v, &q_pos, &k_pos, None, bits_from_env()).expect("no sink");
    let with = attention(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&sink), bits_from_env()).expect("sink");
    assert!(
        bit_eq(&without, &with),
        "c1: GA output moved when a sink was passed — GA must be bitwise sink-free"
    );
}

/// BOTH RUNS — c1 detection: the sink DOES move SWA outputs (it is real there).
#[test]
fn sink_moves_swa_outputs() {
    let spec = tiny_swa(4, 2);
    let (t, sn) = (1usize, 3usize);
    let mut rng = XorShift64::new(12);
    let q = rows(&mut rng, t * spec.n_q, spec.d_qk);
    let k = rows(&mut rng, sn * spec.n_kv, spec.d_qk);
    let v = rows(&mut rng, sn * spec.n_kv, spec.d_v);
    let q_pos = vec![7i64];
    let k_pos = vec![5i64, 6, 7];
    let sink = vec![3.0f32; 4];
    let without = attention(&spec, &q, &k, &v, &q_pos, &k_pos, None, NaiveBits::NONE).expect("no sink");
    let with = attention(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&sink), NaiveBits::NONE).expect("sink");
    assert!(
        max_abs_diff(&without, &with) > 0.05,
        "sink had no effect on SWA — detection power gone"
    );
}

/// NEGATIVE (env-default). T3: GA sees the whole past — with `q_pos = 10` over
/// keys {0, 9, 10} the GA output must match the UNWINDOWED analytic
/// expectation (key 0 visible). Flips on the naive run (GA inherits the SWA
/// window 4 and drops key 0).
#[test]
fn t3_ga_layers_are_not_windowed() {
    let spec = tiny_ga(2, 2); // window field = 4, must be IGNORED on GA
    assert_eq!(spec.n_rep(), 1);
    let (t, sn) = (1usize, 3usize);
    let mut rng = XorShift64::new(33);
    let q = rows(&mut rng, t * spec.n_q, spec.d_qk);
    let k = rows(&mut rng, sn * spec.n_kv, spec.d_qk);
    let v = rows(&mut rng, sn * spec.n_kv, spec.d_v);
    let q_pos = vec![10i64];
    let k_pos = vec![0i64, 9, 10];
    let out = attention(&spec, &q, &k, &v, &q_pos, &k_pos, None, bits_from_env()).expect("attn");
    let scale = 1.0 / (spec.d_qk as f64).sqrt();
    for h in 0..spec.n_q {
        let kvh = h; // n_rep == 1
        let q_row: Vec<f32> = (0..spec.d_qk).map(|i| q[h * spec.d_qk + i]).collect();
        let keys: Vec<(i64, Vec<f32>, Vec<f32>)> = (0..sn)
            .map(|j| {
                (
                    k_pos[j],
                    k[(j * spec.n_kv + kvh) * spec.d_qk..(j * spec.n_kv + kvh + 1) * spec.d_qk].to_vec(),
                    v[(j * spec.n_kv + kvh) * spec.d_v..(j * spec.n_kv + kvh + 1) * spec.d_v].to_vec(),
                )
            })
            .collect();
        let exp = analytic_softmax(scale, None, &q_row, &keys); // all 3 keys visible
        let got: Vec<f32> = (0..spec.d_v).map(|i| out[h * spec.d_v + i]).collect();
        let d = max_abs_diff(&got, &exp);
        assert!(d <= 1e-5, "T3: GA head {h} diverges from the UNWINDOWED expectation ({d})");
    }
}

/// NEGATIVE (env-default). T6: the sink bias is per Q head (`[64]` real);
/// expectations are per-head hand computations with `sink[h]`. Flips on the
/// naive run (bias indexed per KV head).
#[test]
fn t6_sink_is_per_q_head() {
    let spec = tiny_swa(4, 2); // n_rep 2
    let (t, sn) = (1usize, 3usize);
    let mut rng = XorShift64::new(66);
    let q = rows(&mut rng, t * spec.n_q, spec.d_qk);
    let k = rows(&mut rng, sn * spec.n_kv, spec.d_qk);
    let v = rows(&mut rng, sn * spec.n_kv, spec.d_v);
    let q_pos = vec![7i64];
    let k_pos = vec![5i64, 6, 7];
    let sink = vec![1.0f32, 2.0, 3.0, 4.0]; // distinct within every GQA group
    let out = attention(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&sink), bits_from_env()).expect("attn");
    let scale = 1.0 / (spec.d_qk as f64).sqrt();
    for h in 0..spec.n_q {
        let kvh = h / spec.n_rep();
        let q_row: Vec<f32> = (0..spec.d_qk).map(|i| q[h * spec.d_qk + i]).collect();
        let keys: Vec<(i64, Vec<f32>, Vec<f32>)> = (0..sn)
            .map(|j| {
                (
                    k_pos[j],
                    k[(j * spec.n_kv + kvh) * spec.d_qk..(j * spec.n_kv + kvh + 1) * spec.d_qk].to_vec(),
                    v[(j * spec.n_kv + kvh) * spec.d_v..(j * spec.n_kv + kvh + 1) * spec.d_v].to_vec(),
                )
            })
            .collect();
        let exp = analytic_softmax(scale, Some(f64::from(sink[h])), &q_row, &keys);
        let got: Vec<f32> = (0..spec.d_v).map(|i| out[h * spec.d_v + i]).collect();
        let d = max_abs_diff(&got, &exp);
        assert!(d <= 1e-5, "T6: head {h} uses the wrong sink bias (diff {d} vs sink[{h}])");
    }
}

/// NEGATIVE (env-default). c2/T4: the logit scale is `1/√d_qk` (d_qk = 4 here,
/// d_v = 2) — never the V width. Flips on the naive run (QK/V mixup).
#[test]
fn t4_attn_scale_uses_d_qk() {
    let spec = tiny_swa(2, 2);
    assert_ne!(spec.d_qk, spec.d_v, "the pin needs distinct widths");
    let (t, sn) = (1usize, 3usize);
    let mut rng = XorShift64::new(44);
    let q = rows(&mut rng, t * spec.n_q, spec.d_qk);
    let k = rows(&mut rng, sn * spec.n_kv, spec.d_qk);
    let v = rows(&mut rng, sn * spec.n_kv, spec.d_v);
    let q_pos = vec![7i64];
    let k_pos = vec![5i64, 6, 7];
    let out = attention(&spec, &q, &k, &v, &q_pos, &k_pos, None, bits_from_env()).expect("attn");
    let scale = 1.0 / (spec.d_qk as f64).sqrt();
    for h in 0..spec.n_q {
        let kvh = h / spec.n_rep();
        let q_row: Vec<f32> = (0..spec.d_qk).map(|i| q[h * spec.d_qk + i]).collect();
        let keys: Vec<(i64, Vec<f32>, Vec<f32>)> = (0..sn)
            .map(|j| {
                (
                    k_pos[j],
                    k[(j * spec.n_kv + kvh) * spec.d_qk..(j * spec.n_kv + kvh + 1) * spec.d_qk].to_vec(),
                    v[(j * spec.n_kv + kvh) * spec.d_v..(j * spec.n_kv + kvh + 1) * spec.d_v].to_vec(),
                )
            })
            .collect();
        let exp = analytic_softmax(scale, None, &q_row, &keys);
        let got: Vec<f32> = (0..spec.d_v).map(|i| out[h * spec.d_v + i]).collect();
        let d = max_abs_diff(&got, &exp);
        assert!(d <= 1e-5, "c2/T4: head {h} used the wrong logit scale (diff {d})");
    }
    // detection: the two scales genuinely differ on this width pair
    assert!((attn_scale(spec.d_qk, spec.d_v, NaiveBits::NONE) - attn_scale(spec.d_qk, spec.d_v, NaiveBits::SCALE_BY_DV)).abs() > 0.05);
}

/// NEGATIVE (env-default). T9: absolute positions drive the window. Query at
/// 100 over keys {0, 50, 100} with window 60: key 0 is dead. Flips on the
/// naive run (everything pinned at position 0 → key 0 leaks in).
#[test]
fn t9_start_pos_is_honored() {
    let spec = tiny_swa(2, 2);
    let spec = AttnSpec { window: 60, ..spec };
    let (t, sn) = (1usize, 3usize);
    let mut rng = XorShift64::new(99);
    let q = rows(&mut rng, t * spec.n_q, spec.d_qk);
    let k = rows(&mut rng, sn * spec.n_kv, spec.d_qk);
    let v = rows(&mut rng, sn * spec.n_kv, spec.d_v);
    let q_pos = vec![100i64];
    let k_pos = vec![0i64, 50, 100];
    let out = attention(&spec, &q, &k, &v, &q_pos, &k_pos, None, bits_from_env()).expect("attn");
    let scale = 1.0 / (spec.d_qk as f64).sqrt();
    for h in 0..spec.n_q {
        let kvh = h / spec.n_rep();
        let q_row: Vec<f32> = (0..spec.d_qk).map(|i| q[h * spec.d_qk + i]).collect();
        // ONLY keys 50 and 100 are visible at q=100 with window 60 (T9/T3 mask)
        let keys: Vec<(i64, Vec<f32>, Vec<f32>)> = [1usize, 2]
            .iter()
            .map(|&j| {
                (
                    k_pos[j],
                    k[(j * spec.n_kv + kvh) * spec.d_qk..(j * spec.n_kv + kvh + 1) * spec.d_qk].to_vec(),
                    v[(j * spec.n_kv + kvh) * spec.d_v..(j * spec.n_kv + kvh + 1) * spec.d_v].to_vec(),
                )
            })
            .collect();
        let exp = analytic_softmax(scale, None, &q_row, &keys);
        let got: Vec<f32> = (0..spec.d_v).map(|i| out[h * spec.d_v + i]).collect();
        let d = max_abs_diff(&got, &exp);
        assert!(d <= 1e-5, "T9: head {h} ignored start_pos (diff {d})");
    }
}

/// BOTH RUNS — the sink column absorbs mass but carries ZERO value; and a
/// very negative sink logit ≡ no sink.
#[test]
fn sink_mass_zero_value_and_vanishing_sink() {
    let spec = tiny_swa(2, 2);
    let (t, sn) = (1usize, 3usize);
    let mut rng = XorShift64::new(71);
    let q = rows(&mut rng, t * spec.n_q, spec.d_qk);
    let k = rows(&mut rng, sn * spec.n_kv, spec.d_qk);
    let zero_v = vec![0.0f32; sn * spec.n_kv * spec.d_v];
    let q_pos = vec![7i64];
    let k_pos = vec![5i64, 6, 7];
    let sink = vec![20.0f32; spec.n_q]; // sink dominates the softmax
    let out = attention(&spec, &q, &k, &zero_v, &q_pos, &k_pos, Some(&sink), NaiveBits::NONE).expect("attn");
    assert!(
        out.iter().all(|&x| x == 0.0),
        "sink value must be zero: all-zero V must give exactly zero output even with a dominant sink"
    );
    // vanishing sink ≡ no sink
    let v = rows(&mut rng, sn * spec.n_kv, spec.d_v);
    let no_sink = attention(&spec, &q, &k, &v, &q_pos, &k_pos, None, NaiveBits::NONE).expect("attn");
    let dead_sink = vec![-1e9f32; spec.n_q];
    let with_dead = attention(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&dead_sink), NaiveBits::NONE).expect("attn");
    assert!(
        max_abs_diff(&no_sink, &with_dead) <= 1e-6,
        "a −1e9 sink logit must reproduce the no-sink output"
    );
}

/// BOTH RUNS — attention is linear in cached V (`value_scale` already folded in
/// by the cache, T18).
#[test]
fn value_scale_linearity() {
    let spec = tiny_swa(2, 2);
    let (t, sn) = (2usize, 3usize);
    let mut rng = XorShift64::new(72);
    let q = rows(&mut rng, t * spec.n_q, spec.d_qk);
    let k = rows(&mut rng, sn * spec.n_kv, spec.d_qk);
    let v = rows(&mut rng, sn * spec.n_kv, spec.d_v);
    let v2: Vec<f32> = v.iter().map(|&x| x * 2.0).collect();
    let q_pos = vec![6i64, 7];
    let k_pos = vec![5i64, 6, 7];
    let sink = vec![0.5f32; spec.n_q];
    let o1 = attention(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&sink), NaiveBits::NONE).expect("attn");
    let o2 = attention(&spec, &q, &k, &v2, &q_pos, &k_pos, Some(&sink), NaiveBits::NONE).expect("attn");
    for i in 0..o1.len() {
        assert!((f64::from(o2[i]) - 2.0 * f64::from(o1[i])).abs() <= 1e-6, "value linearity at {i}");
    }
}

/// BOTH RUNS — GQA packing: head h reads KV head `h / n_rep` — equals computing
/// with the KV heads physically repeated.
#[test]
fn gqa_packing_matches_kv_repeat() {
    let spec = tiny_swa(4, 2); // n_rep 2
    let (t, sn) = (2usize, 5usize);
    let mut rng = XorShift64::new(73);
    let q = rows(&mut rng, t * spec.n_q, spec.d_qk);
    let k = rows(&mut rng, sn * spec.n_kv, spec.d_qk);
    let v = rows(&mut rng, sn * spec.n_kv, spec.d_v);
    let q_pos = vec![8i64, 9];
    let k_pos = vec![4i64, 5, 6, 7, 8];
    let sink = vec![0.1f32, 0.2, 0.3, 0.4];
    let packed = attention(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&sink), NaiveBits::NONE).expect("packed");
    // physically repeat each KV head n_rep times
    let mut k2 = vec![0f32; sn * spec.n_q * spec.d_qk];
    let mut v2 = vec![0f32; sn * spec.n_q * spec.d_v];
    for j in 0..sn {
        for kvh in 0..spec.n_kv {
            for r in 0..spec.n_rep() {
                let h2 = kvh * spec.n_rep() + r;
                k2[(j * spec.n_q + h2) * spec.d_qk..(j * spec.n_q + h2 + 1) * spec.d_qk]
                    .copy_from_slice(&k[(j * spec.n_kv + kvh) * spec.d_qk..(j * spec.n_kv + kvh + 1) * spec.d_qk]);
                v2[(j * spec.n_q + h2) * spec.d_v..(j * spec.n_q + h2 + 1) * spec.d_v]
                    .copy_from_slice(&v[(j * spec.n_kv + kvh) * spec.d_v..(j * spec.n_kv + kvh + 1) * spec.d_v]);
            }
        }
    }
    let spec2 = AttnSpec { n_kv: spec.n_q, ..spec.clone() };
    let repeated = attention(&spec2, &q, &k2, &v2, &q_pos, &k_pos, Some(&sink), NaiveBits::NONE).expect("repeat");
    let d = max_abs_diff(&packed, &repeated);
    assert!(d <= 1e-6, "GQA packing ≠ KV repeat (diff {d})");
}

/// BOTH RUNS — window isolation: perturbing keys outside the window cannot move
/// the output (SWA-128 contract, T3/T8 mask semantics).
#[test]
fn window_128_isolation() {
    let spec = AttnSpec { window: 4, ..tiny_swa(2, 2) };
    let (t, sn) = (1usize, 9usize);
    let mut rng = XorShift64::new(74);
    let q = rows(&mut rng, t * spec.n_q, spec.d_qk);
    let k = rows(&mut rng, sn * spec.n_kv, spec.d_qk);
    let v = rows(&mut rng, sn * spec.n_kv, spec.d_v);
    let q_pos = vec![8i64];
    let k_pos: Vec<i64> = (0..sn as i64).collect();
    let base = attention(&spec, &q, &k, &v, &q_pos, &k_pos, None, NaiveBits::NONE).expect("attn");
    // perturb key at position 0 (distance 8 ≥ window 4)
    let mut k2 = k.clone();
    let mut v2 = v.clone();
    for e in k2[..spec.n_kv * spec.d_qk].iter_mut() {
        *e += 3.0;
    }
    for e in v2[..spec.n_kv * spec.d_v].iter_mut() {
        *e += 3.0;
    }
    let moved = attention(&spec, &q, &k2, &v2, &q_pos, &k_pos, None, NaiveBits::NONE).expect("attn");
    assert!(bit_eq(&base, &moved), "a key outside the window changed the output");
}

/// BOTH RUNS — a query with no visible key gets a zero row (oracle
/// layers.py:123-124), sink or not.
#[test]
fn no_visible_key_gives_zero_row() {
    let spec = tiny_swa(2, 2);
    let (t, sn) = (1usize, 3usize);
    let mut rng = XorShift64::new(75);
    let q = rows(&mut rng, t * spec.n_q, spec.d_qk);
    let k = rows(&mut rng, sn * spec.n_kv, spec.d_qk);
    let v = rows(&mut rng, sn * spec.n_kv, spec.d_v);
    let q_pos = vec![5i64];
    let k_pos = vec![6i64, 7, 8]; // all in the future — nothing visible
    let sink = vec![9.0f32; spec.n_q];
    for s in [None, Some(sink.as_slice())] {
        let out = attention(&spec, &q, &k, &v, &q_pos, &k_pos, s, NaiveBits::NONE).expect("attn");
        assert!(out.iter().all(|&x| x == 0.0), "no-visible-key row must be exactly zero");
    }
}

/// BOTH RUNS — fail-loud checks (T4 shape contract, T6 sink length, c1 strict).
#[test]
fn fail_loud_shape_and_sink_checks() {
    let spec = tiny_swa(2, 2);
    let (t, sn) = (1usize, 2usize);
    let mut rng = XorShift64::new(76);
    let q = rows(&mut rng, t * spec.n_q, spec.d_qk);
    let k = rows(&mut rng, sn * spec.n_kv, spec.d_qk);
    let v = rows(&mut rng, sn * spec.n_kv, spec.d_v);
    let q_pos = vec![3i64];
    let k_pos = vec![2i64, 3];
    // wrong V width (a d_v == d_qk assumption) must error, not broadcast
    let v_bad = rows(&mut rng, sn * spec.n_kv, spec.d_qk);
    let err = attention(&spec, &q, &k, &v_bad, &q_pos, &k_pos, None, NaiveBits::NONE);
    assert!(matches!(err, Err(AttnError::ShapeMismatch { .. })), "T4: V width must fail loud");
    // wrong sink length (must be exactly n_q=2 per Q head, T6) — 3 ≠ 2 fails loud
    let err = attention(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&[0.1f32, 0.2, 0.3]), NaiveBits::NONE);
    assert!(matches!(err, Err(AttnError::SinkLength { .. })), "T6: sink length must fail loud");
    // strict entry rejects a sink on GA (c1 fail-loud twin of the bitwise pin)
    let ga = tiny_ga(2, 2);
    let err = attention_checked(&ga, &q, &k, &v, &q_pos, &k_pos, Some(&[0.1f32; 2]), NaiveBits::NONE);
    assert!(matches!(err, Err(AttnError::SinkNotSupported { .. })), "c1: GA+sink must fail loud here");
}

/// BOTH RUNS — attribution for T4's broadcast half.
#[test]
fn v_broadcast_bug_oracle_diverges() {
    let spec = tiny_swa(2, 2);
    let (t, sn) = (1usize, 3usize);
    let mut rng = XorShift64::new(77);
    let q = rows(&mut rng, t * spec.n_q, spec.d_qk);
    let k = rows(&mut rng, sn * spec.n_kv, spec.d_qk);
    let v = rows(&mut rng, sn * spec.n_kv, spec.d_v);
    let q_pos = vec![7i64];
    let k_pos = vec![5i64, 6, 7];
    let good = attention(&spec, &q, &k, &v, &q_pos, &k_pos, None, NaiveBits::NONE).expect("attn");
    let bad = attention(&spec, &q, &k, &v, &q_pos, &k_pos, None, NaiveBits::V_BROADCAST).expect("attn");
    assert!(
        max_abs_diff(&good, &bad) > 0.05,
        "V-broadcast oracle converged with correct — no detection power"
    );
}
