//! Two-run trap suite for the P-202 loader port — T1 (fused-QKV ckpt_tp=4
//! TP-order split) + T2 (per-shard scale-grid padding trim) negatives and the
//! fail-loud name-audit guards.
//!
//! Ports `spike/tests/test_t1_fused_qkv_split.py` + the name half of
//! `spike/tests/test_p102_real_shard_geometry.py` + `spike/model.py:49-56`.
//!
//! # Two-run classification (suite convention)
//!
//! NEGATIVES — FAIL under `MIMO26_SPIKE_NAIVE=1` (alias `MIMO26_LOAD_NAIVE=1`),
//! PASS on the correct impl (they call the env-default entry points):
//!   * `t1_ckpt_tp4_tp_order_split_matches_whole`
//!   * `t2_poisoned_pad_row_never_read` (its env-default half)
//!
//! BOTH RUNS (explicit `naive=` flags for detection power; fail-loud guards):
//!   `t1_naive_split_scrambles_qkv_detection`,
//!   `naive_dequant_global_scale_rows_is_wrong`, `uneven_shard_rows_raise`,
//!   `mtp_prefix_canonicalisation_order`, `missing_required_weight_raises`,
//!   `missing_router_bias_raises`, `name_audit_unclassified_fails_loud`,
//!   `name_audit_qkv_count_mismatch_fails_loud`, `name_audit_passes_on_clean_map`
//! (plus `tests/golden_fused_split.rs`, both runs).

mod common;

use common::*;
use mimo26_load::fused::{dequantize_naive_fused, reconstruct_layer_qkv};
use mimo26_load::names::{
    audit_required_weights, audit_router_bias, audit_weight_map, canonical_name, is_backbone_weight,
    REQUIRED_WEIGHTS,
};
use mimo26_load::{naive_from_env, LoadError, Mat};

const BLOCK: (usize, usize) = (4, 4);
const SEGS: (usize, usize, usize) = (3, 2, 2);
const COLS: usize = 8;

fn max_abs_diff(a: &Mat<f32>, b: &Mat<f32>) -> f32 {
    assert_eq!((a.rows, a.cols), (b.rows, b.cols), "shape mismatch");
    let mut m = 0f32;
    for r in 0..a.rows {
        for c in 0..a.cols {
            m = m.max((a.get(r, c) - b.get(r, c)).abs());
        }
    }
    m
}

fn approx_eq(a: &Mat<f32>, b: &Mat<f32>, atol: f32) -> bool {
    a.rows == b.rows && a.cols == b.cols && max_abs_diff(a, b) <= atol
}

fn bit_eq(a: &Mat<f32>, b: &Mat<f32>) -> bool {
    a.rows == b.rows
        && a.cols == b.cols
        && (0..a.rows).all(|r| (0..a.cols).all(|c| a.get(r, c).to_bits() == b.get(r, c).to_bits()))
}

// ---------------------------------------------------------------------------
// NEGATIVES (flip behind the env-default naive flag)
// ---------------------------------------------------------------------------

#[test]
fn t1_ckpt_tp4_tp_order_split_matches_whole() {
    // NEGATIVE — flips on the naive run (map §4 test_loader.py:136; n_ranks 4 =
    // ckpt_tp). Per-chunk [Q_c|K_c|V_c] -> [Q|K|V] regroup must equal the
    // whole-matrix reference (spike/tests/test_t1_fused_qkv_split.py:38-43).
    let (w, s, refm) = make_fused_case(4, SEGS, COLS, BLOCK, 1);
    let got = reconstruct_layer_qkv(&w, &s, SEGS, BLOCK, naive_from_env()).expect("reconstruct");
    assert!(
        approx_eq(&got, &refm, 1e-6),
        "T1: ckpt_tp=4 TP-order split must regroup to [Q|K|V] whole-matrix order (max diff {})",
        max_abs_diff(&got, &refm)
    );
}

#[test]
fn t2_poisoned_pad_row_never_read() {
    // NEGATIVE (env-default half) + detection halves (explicit naive) —
    // spike/tests/test_t1_fused_qkv_split.py:54-70. Poison the pad scale rows
    // with 1e30: the correct path is BIT-identical with/without poison; the
    // naive path leaks it (T2).
    let (w, scales, _refm) = make_fused_case(2, SEGS, COLS, BLOCK, 1);
    let clean = reconstruct_layer_qkv(&w, &scales, SEGS, BLOCK, false).expect("clean");
    let poisoned: Vec<Mat<f32>> = scales
        .iter()
        .map(|s| {
            let mut s2 = s.clone();
            for c in 0..s2.cols {
                s2.set(s2.rows - 1, c, 1e30); // last grid rows are pad for these shapes
            }
            s2
        })
        .collect();
    let got = reconstruct_layer_qkv(&w, &poisoned, SEGS, BLOCK, false).expect("poisoned correct");
    assert!(bit_eq(&got, &clean), "T2: pad scale rows must never be read");
    // the env-selected impl must also survive the poison (FAILS on the naive run)
    let got_default =
        reconstruct_layer_qkv(&w, &poisoned, SEGS, BLOCK, naive_from_env()).expect("poisoned env");
    assert!(bit_eq(&got_default, &clean), "T2: env-default impl leaked a pad scale row");
    let naive = reconstruct_layer_qkv(&w, &poisoned, SEGS, BLOCK, true).expect("poisoned naive");
    assert!(
        max_abs_diff(&naive, &clean) > 1.0,
        "poison not leaked by the naive oracle — T2 detection power is gone"
    );
}

// ---------------------------------------------------------------------------
// BOTH RUNS (detection power + fail-loud guards)
// ---------------------------------------------------------------------------

#[test]
fn t1_naive_split_scrambles_qkv_detection() {
    // Detection power (map §4, test_fp8_block.py:105): naive err > 10x fixed + 0.5.
    let (w, s, refm) = make_fused_case(4, SEGS, COLS, BLOCK, 1);
    let correct_err = max_abs_diff(
        &reconstruct_layer_qkv(&w, &s, SEGS, BLOCK, false).expect("correct"),
        &refm,
    );
    let naive_err = max_abs_diff(
        &reconstruct_layer_qkv(&w, &s, SEGS, BLOCK, true).expect("naive"),
        &refm,
    );
    assert!(
        naive_err > 10.0 * correct_err + 0.5,
        "naive scramble invisible ({naive_err} vs {correct_err}) — T1 has no detection power"
    );
}

#[test]
fn naive_dequant_global_scale_rows_is_wrong() {
    // fp8_block.py:170/:182 bug oracle — global row//br over padded grids (T1+T2).
    let (w, s, _refm) = make_fused_case(2, SEGS, COLS, BLOCK, 1);
    let good = reconstruct_layer_qkv(&w, &s, SEGS, BLOCK, false).expect("good");
    let bad = dequantize_naive_fused(&w, &s, BLOCK).expect("bad");
    assert!(
        max_abs_diff(&good, &bad) > 1e-6,
        "bug oracle converged with the correct path — detection power is gone"
    );
}

#[test]
fn uneven_shard_rows_raise() {
    // loader.py:84-93 fail-loud (map §2, test_fp8_block.py:148).
    let (mut w, mut s, _refm) = make_fused_case(2, SEGS, COLS, BLOCK, 1);
    let short = w[0].slice_rows(0, w[0].rows - 1); // one row short
    w.push(short);
    s.push(s[0].clone());
    let err = reconstruct_layer_qkv(&w, &s, SEGS, BLOCK, false).expect_err("must fail loud");
    assert!(
        matches!(err, LoadError::UnevenShardRows { .. }),
        "uneven shard rows must raise UnevenShardRows, got {err:?}"
    );
}

#[test]
fn mtp_prefix_canonicalisation_order() {
    // loader.py:28 — `model.mtp.` -> `mtp.` BEFORE the generic strip.
    assert_eq!(
        canonical_name("model.mtp.layers.0.eh_proj.weight"),
        "mtp.layers.0.eh_proj.weight"
    );
    assert_eq!(
        canonical_name("model.layers.3.self_attn.qkv_proj.weight"),
        "layers.3.self_attn.qkv_proj.weight"
    );
    assert!(is_backbone_weight("model.layers.3.self_attn.qkv_proj.weight"));
    assert!(!is_backbone_weight("model.mtp.layers.0.eh_proj.weight"));
}

#[test]
fn missing_required_weight_raises() {
    // spike/model.py:49-52 — no silent default, no skip.
    let mut names: Vec<String> = REQUIRED_WEIGHTS.iter().map(|r| (*r).to_string()).collect();
    names.retain(|n| n != "lm_head.weight");
    let err = audit_required_weights(names.iter().map(|s| s.as_str())).expect_err("must fail loud");
    match err {
        LoadError::MissingRequiredWeights { missing } => {
            assert_eq!(missing, vec!["lm_head.weight".to_string()])
        }
        other => panic!("expected MissingRequiredWeights, got {other:?}"),
    }
}

#[test]
fn missing_router_bias_raises() {
    // spike/model.py:53-56 — T11 name-audit class (KeyError there).
    let names = [
        "layers.0.mlp.gate.e_score_correction_bias",
        "layers.1.norm.weight", // layer 1 router bias missing
    ];
    let err = audit_router_bias(names.iter().copied(), 2).expect_err("must fail loud");
    match err {
        LoadError::MissingRouterBias { key } => {
            assert_eq!(key, "layers.1.mlp.gate.e_score_correction_bias")
        }
        other => panic!("expected MissingRouterBias, got {other:?}"),
    }
}

#[test]
fn name_audit_unclassified_fails_loud() {
    // spike/shard_index.py:63-64 — anything UNCLASSIFIED raises.
    let names = [
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.experts.3.up_proj.weight",
        "model.mtp.layers.0.eh_proj.weight",
        "model.weird.new_thing", // UNCLASSIFIED
    ];
    let err = audit_weight_map(names.iter().copied(), 1).expect_err("must fail loud");
    match err {
        LoadError::NameAudit { problems } => {
            assert!(problems.iter().any(|p| p == "model.weird.new_thing"))
        }
        other => panic!("expected NameAudit, got {other:?}"),
    }
}

#[test]
fn name_audit_qkv_count_mismatch_fails_loud() {
    // spike/shard_index.py:67-70 — fused-QKV count is an external anchor
    // (48 on the live index; parameterised here).
    let names = ["model.layers.0.self_attn.qkv_proj.weight"];
    let err = audit_weight_map(names.iter().copied(), 48).expect_err("must fail loud");
    match err {
        LoadError::NameAudit { problems } => assert!(problems
            .iter()
            .any(|p| p.contains("fused-QKV count 1 != 48"))),
        other => panic!("expected NameAudit, got {other:?}"),
    }
}

#[test]
fn name_audit_passes_on_clean_map() {
    // Happy path over one of each kind (shard_index.py:43-52 classes).
    let names = [
        "model.embed_tokens.weight",                      // backbone
        "model.layers.0.self_attn.qkv_proj.weight",       // backbone + qkv count
        "model.layers.0.mlp.experts.0.gate_proj.weight",  // expert
        "model.mtp.layers.0.eh_proj.weight",              // mtp (root kept)
        "model.audio_encoder.layers.0.q.weight",          // expected non-backbone
    ];
    let rep = audit_weight_map(names.iter().copied(), 1).expect("clean map must pass");
    assert_eq!(rep.qkv, 1);
    assert_eq!(rep.tensors, 5);
    assert_eq!(rep.kinds.get("expert"), Some(&1));
    assert_eq!(rep.kinds.get("mtp"), Some(&1));
}
