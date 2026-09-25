//! T20 (FP8 KV scale layout) + pool byte pins + the amax clip gate.
//!
//! # Two-run classification
//!
//! NEGATIVES — FAIL under `MIMO26_SPIKE_NAIVE=1`, PASS on the correct impl:
//!   * `t20_scale_layout_per_token_head_k_v_separate`
//!   * `amax_clip_count_reported`
//!
//! BOTH RUNS (explicit `NaiveBits` bug oracles / pins):
//!   `pool_byte_pins`, `t20_detection_block128_shared_diverges`,
//!   `amax_detection_silent_clamp_hides_clips`, `layout_index_pins`.
//!
//! T20 trap statement (docs/COHERENCE-TRAPS.md §6): "block-128 does not divide
//! the 192-dim K; scales must be per token × head, with K and V separate, and
//! counted in the pool bytes" — all three clauses are pinned below.

mod common;

use common::*;
use mimo26_attn::fp8kv::{
    decode_kv, encode_kv, k_scale_off, kv_bytes_per_token, shared_block128_off,
    shared_blocks_per_row, v_scale_off, Layout,
};
use mimo26_attn::geom::{bytes, AttnSpec, ScaleMode, D_QK, D_V, GA_KV_HEADS, SWA_KV_HEADS};
use mimo26_attn::{bits_from_env, NaiveBits, Family};

/// BOTH RUNS — the pool arithmetic ARCHITECTURE §7 / ADVISOR-I3 §3 A1–A2 pin
/// (model numbers derived from the real config, `oracle/mimo26/config.py`).
#[test]
fn pool_byte_pins() {
    // 9 GA layers × 4 KV heads × (192 + 128) B = 11,520 B/token (FP8 unit-scale)
    assert_eq!(bytes::ga_kv_bytes_per_token(ScaleMode::Unit), 11_520);
    // + 9 × 4 × 2 × 4 B of per-token×head scales (T20: K and V separate)
    assert_eq!(bytes::ga_kv_bytes_per_token(ScaleMode::PerTokenHead), 11_808);
    // 39 SWA layers × 8 KV heads × 320 B × 128 rows = 12,779,520 B/seq
    assert_eq!(bytes::swa_ring_bytes_per_seq(ScaleMode::Unit), 12_779_520);
    assert_eq!(bytes::swa_ring_bytes_per_seq(ScaleMode::PerTokenHead), 13_099_008);
    // 256-token page × 11,520 B/token = 2,949,120 B (ADVISOR-I3 §3 A2)
    assert_eq!(bytes::ga_page_bytes(ScaleMode::Unit), 2_949_120);
    assert_eq!(bytes::ga_page_bytes(ScaleMode::PerTokenHead), 3_022_848);
    // per-layer view agrees
    assert_eq!(kv_bytes_per_token(Layout::Unit, GA_KV_HEADS, D_QK, D_V), 1_280);
    assert_eq!(kv_bytes_per_token(Layout::PerTokenHead, GA_KV_HEADS, D_QK, D_V), 1_280 + 32);
    assert_eq!(kv_bytes_per_token(Layout::Unit, SWA_KV_HEADS, D_QK, D_V), 2_560);
}

/// BOTH RUNS — the three T20 clauses at the index level.
#[test]
fn layout_index_pins() {
    let n_kv = 4usize;
    // per token × head (not per 128-element block)
    assert_eq!(k_scale_off(2, 3, n_kv), 2 * n_kv + 3);
    assert_eq!(v_scale_off(2, 3, n_kv), 2 * n_kv + 3);
    // K and V separate planes: same index, different plane (poison test in the
    // NEGATIVE below proves independence end to end)
    // block-128 cannot express the 192-dim K: dims 128..192 of K share a scale
    // block with V dims 0..64 of the same head — the exact T20 trap.
    assert_eq!(shared_block128_off(127), 0);
    assert_eq!(shared_block128_off(128), 1);
    assert_eq!(shared_block128_off(191), 1); // K tail -> block 1
    assert_eq!(shared_block128_off(192), 1); // V head -> SAME block (trap)
    assert_eq!(shared_block128_off(255), 1);
    assert_eq!(shared_block128_off(256), 2);
    assert_eq!(shared_blocks_per_row(n_kv, D_QK, D_V), (n_kv * (D_QK + D_V) + 127) / 128);
}

/// Crafted rows where the T20 trap bites: the tail of K (dims 128..192) has a
/// huge amax while V is tiny — the naive shared block scale destroys V (and the
/// correct per-token×head layout keeps both exact to E4M3).
fn crafted_case(n_tok: usize, n_kv: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = XorShift64::new(2020);
    let mut k = rng.fill_small(n_tok * n_kv * D_QK);
    for t in 0..n_tok {
        for h in 0..n_kv {
            for i in 0..D_QK {
                let idx = (t * n_kv + h) * D_QK + i;
                k[idx] = if i < 128 {
                    k[idx] * 2e-3 // tiny amax block
                } else {
                    k[idx] * 600.0 // huge amax tail (drives the shared block)
                };
            }
        }
    }
    let mut v = rng.fill_small(n_tok * n_kv * D_V);
    for x in v.iter_mut() {
        *x *= 2e-3;
    }
    (k, v)
}

/// Construction-side round-trip expectation (independent of `fp8kv`): the
/// per-token×head path, using the externally golden-pinned codec directly.
fn expected_per_token_head(rows: &[f32], n_tok: usize, n_kv: usize, d: usize) -> Vec<f32> {
    use mimo26_load::e4m3::{decode_e4m3, encode_e4m3, E4M3_MAX};
    let mut out = vec![0f32; rows.len()];
    for t in 0..n_tok {
        for h in 0..n_kv {
            let base = (t * n_kv + h) * d;
            let plane = &rows[base..base + d];
            let mut amax = 0f64;
            for &x in plane {
                amax = amax.max(f64::from(x).abs());
            }
            let s = if amax > 0.0 { (amax / E4M3_MAX) as f32 } else { 1.0 };
            for i in 0..d {
                let code = encode_e4m3(f64::from(plane[i]) / f64::from(s));
                out[base + i] = decode_e4m3(code) as f32 * s;
            }
        }
    }
    out
}

/// NEGATIVE (env-default). T20: per-token×head scales, K and V separate, layout
/// shape asserted. Flips on the naive run (block-128 shared grid).
#[test]
fn t20_scale_layout_per_token_head_k_v_separate() {
    let (n_tok, n_kv) = (3usize, 4usize);
    let (k, v) = crafted_case(n_tok, n_kv);
    let enc = encode_kv(&k, &v, n_tok, n_kv, D_QK, D_V, ScaleMode::PerTokenHead, bits_from_env())
        .expect("encode");
    // (1) layout pin: per-token×head planes, K and V SEPARATE (T20)
    assert_eq!(enc.layout, Layout::PerTokenHead, "T20: scales must be per token × head");
    assert_eq!(enc.k_scales.len(), n_tok * n_kv, "T20: one K scale per token × head");
    assert_eq!(enc.v_scales.len(), n_tok * n_kv, "T20: one V scale per token × head, separate plane");
    assert!(enc.shared_scales.is_empty(), "T20: no block-128 shared grid");
    // (2) round-trip == the per-token×head construction truth
    let (k_dec, v_dec) = decode_kv(&enc, 1.0);
    let exp_k = expected_per_token_head(&k, n_tok, n_kv, D_QK);
    let exp_v = expected_per_token_head(&v, n_tok, n_kv, D_V);
    let dk = max_abs_diff(&k_dec, &exp_k);
    let dv = max_abs_diff(&v_dec, &exp_v);
    assert!(dk <= 1e-6, "T20: K round-trip off by {dk}");
    assert!(dv <= 1e-6, "T20: V round-trip off by {dv}");
    // (3) K/V plane independence: poisoning the V scales must not move K at all
    let mut poisoned = enc.clone();
    for s in poisoned.v_scales.iter_mut() {
        *s = 1e30;
    }
    let (k_poisoned, _) = decode_kv(&poisoned, 1.0);
    assert!(
        bytes_eq(
            &k_dec.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
            &k_poisoned.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()
        ),
        "T20: K decode changed when the V scale plane was poisoned — the planes are not separate"
    );
}

/// BOTH-RUNS detection fixture — asymmetric SCALE POISON (the spike's
/// `test_p101_addendum.py` pad-poison pattern: poison ONE shared block, then
/// assert the collapsed set is exactly its partners — not a tolerance boundary).
/// Every V element is a clean 8.0 (the per-token×head plane scale reconstructs
/// it to ≈8.0). Exactly one K-tail element (head 0, dim 128) is poisoned to
/// 1e30 — it lives in the SAME block-128 shared scale block as head 0's V dims
/// 0..63 (`shared_block128_off(191) == 1 == shared_block128_off(192)`, the
/// documented T20 trap), so that block's shared scale is dragged to ≈2.2e27 and
/// those V dims underflow to EXACTLY 0.0. Only head 0 straddles a 128-boundary
/// with its V (head 1's K tail and V head land in different blocks), so the
/// collapsed V set is exactly head 0's block-1 partners: asymmetric and exactly
/// countable — the divergence is O(1), never a sub-threshold tolerance gap.
fn poison_case(n_tok: usize, n_kv: usize) -> (Vec<f32>, Vec<f32>) {
    let v = vec![8.0f32; n_tok * n_kv * D_V];
    let mut k = vec![1.0f32; n_tok * n_kv * D_QK]; // nonzero everywhere (no zero-amax block → no NaN)
    for t in 0..n_tok {
        // head 0's K tail (dim 128) shares block 1 with head 0's V dims 0..63
        k[(t * n_kv) * D_QK + 128] = 1e30f32;
    }
    (k, v)
}

/// BOTH RUNS — detection power for T20: the block-128 shared scale grid must
/// destroy V GROSSLY on the asymmetric scale-poison fixture. This pins an O(1)
/// magnitude divergence AND an exact collapsed-element count (the spike's
/// `expect = ...` hardening) — NOT a 1e-3 tolerance boundary — so the oracle can
/// never "refuse to vouch for a negative it cannot see" (the old tiny-V
/// crafted case gave only 0.00066 < 1e-3).
#[test]
fn t20_detection_block128_shared_diverges() {
    let (n_tok, n_kv) = (2usize, 2usize);
    let (k, v) = poison_case(n_tok, n_kv);
    let naive_enc = encode_kv(
        &k,
        &v,
        n_tok,
        n_kv,
        D_QK,
        D_V,
        ScaleMode::PerTokenHead,
        NaiveBits::BLOCK128_SHARED_SCALES,
    )
    .expect("encode naive");
    assert_eq!(naive_enc.layout, Layout::Block128Shared);
    let (_, v_naive) = decode_kv(&naive_enc, 1.0);
    let exp_v = expected_per_token_head(&v, n_tok, n_kv, D_V); // clean V plane ≈ 8.0
    // (1) GROSS magnitude divergence: the poisoned block's V partners underflow
    //     to exactly 0.0 while the per-token×head truth is ≈8.0 → dv ≈ 8.0.
    let dv = max_abs_diff(&v_naive, &exp_v);
    assert!(dv > 1.0, "block-128 shared-grid oracle has no detection power (V error {dv})");
    // (2) EXACT discrete asymmetry pin: the shared grid collapses exactly head
    //     0's block-1 V partners (64 per token); the correct path collapses none.
    let collapsed = |xs: &[f32]| xs.iter().filter(|&&x| x.abs() < 4.0).count();
    assert_eq!(
        collapsed(&exp_v),
        0,
        "per-token×head must reconstruct the clean V plane (nothing collapses)"
    );
    assert_eq!(
        collapsed(&v_naive),
        n_tok * 64,
        "block-128 shared grid must collapse exactly the poisoned block's V partners (head 0, dims 0..63)"
    );
}

/// NEGATIVE (env-default). Unit-scale FP8 + amax clip gate (ADVISOR-I3 §10.4.2:
/// "the clip count must be 0" on real activations — so it must COUNT when it is
/// not). Flips on the naive run (silent clamp reports 0).
#[test]
fn amax_clip_count_reported() {
    let (n_tok, n_kv) = (1usize, 1usize);
    let mut k = vec![0.1f32; n_tok * n_kv * D_QK];
    let mut v = vec![0.1f32; n_tok * n_kv * D_V];
    k[0] = 500.0; // > E4M3_MAX (448)
    v[3] = -700.0;
    let expected_clips = 2u64;
    let enc = encode_kv(&k, &v, n_tok, n_kv, D_QK, D_V, ScaleMode::Unit, bits_from_env()).expect("encode");
    assert_eq!(
        enc.clip_count, expected_clips,
        "amax clip gate: every clamped value must be counted (ADVISOR-I3 §10.4.2)"
    );
    // and the clamp itself is visible in the decode (not silent garbage)
    let (_, v_dec) = decode_kv(&enc, 1.0);
    assert!(v_dec[3].is_finite(), "clamped decode must stay finite");
    assert!(f64::from(v_dec[3]).abs() <= 448.0 + 1e-6, "clamp bound");
}

/// BOTH RUNS — attribution for the amax gate.
#[test]
fn amax_detection_silent_clamp_hides_clips() {
    let (n_tok, n_kv) = (1usize, 1usize);
    let mut k = vec![0.1f32; n_tok * n_kv * D_QK];
    let mut v = vec![0.1f32; n_tok * n_kv * D_V];
    k[0] = 500.0;
    v[3] = -700.0;
    let honest = encode_kv(&k, &v, n_tok, n_kv, D_QK, D_V, ScaleMode::Unit, NaiveBits::NONE).expect("encode");
    let silent = encode_kv(&k, &v, n_tok, n_kv, D_QK, D_V, ScaleMode::Unit, NaiveBits::SILENT_CLAMP)
        .expect("encode");
    assert_eq!(honest.clip_count, 2, "correct path counts");
    assert_eq!(silent.clip_count, 0, "silent-clamp oracle must hide the count");
}

/// BOTH RUNS — the geometry sanity the byte pins derive from.
#[test]
fn real_dims_agree_with_oracle_config() {
    let ga = AttnSpec::real(Family::Ga);
    let swa = AttnSpec::real(Family::Swa);
    assert_eq!((ga.d_qk, ga.d_v, ga.n_q, ga.n_kv), (192, 128, 64, 4));
    assert_eq!((swa.d_qk, swa.d_v, swa.n_q, swa.n_kv), (192, 128, 64, 8));
    assert_eq!(ga.n_rep(), 16);
    assert_eq!(swa.n_rep(), 8);
    assert_eq!(ga.window_gated(), None, "T3: GA is never windowed");
    assert_eq!(swa.window_gated(), Some(128));
    assert!(!ga.sink_allowed(), "c1: GA has no sink");
    assert!(swa.sink_allowed(), "T6: SWA carries the sink");
}
