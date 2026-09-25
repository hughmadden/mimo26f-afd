//! T18 (`v_scale` BEFORE caching) + T8 (SWA eviction `min(batch_pos) − window +
//! 1`) + GA paging pins.
//!
//! # Two-run classification
//!
//! NEGATIVES — FAIL under `MIMO26_SPIKE_NAIVE=1`, PASS on the correct impl:
//!   * `t18_vscale_applied_before_fp8_cache`
//!   * `t8_swa_eviction_keep_from_min_batch_pos`
//!
//! BOTH RUNS (explicit `NaiveBits` bug oracles / invariants):
//!   `t18_detection_scaled_after_store_differs`,
//!   `incremental_equiv_full_across_ring_wrap`, `ga_page_table_and_pool_pins`.
//!
//! T18 statement (ARCHITECTURE.md §11.7, `modeling_mimo_v2.py:301-302`): the
//! reference caches `V×0.707`. Applied after the cache read instead, the FP8
//! codes quantize the WRONG quantity (and with per-token scales the rounding
//! lands differently) — small divergence at short range, salad at long range.

mod common;

use common::*;
use mimo26_attn::attn::attention;
use mimo26_attn::cache::{GaPaged, RowStore, StoreMode, SwaRing};
use mimo26_attn::geom::{AttnSpec, D_QK, D_V, VALUE_SCALE};
use mimo26_attn::{bits_from_env, AttnError, Family, NaiveBits};
use mimo26_load::e4m3::encode_e4m3;

fn v_crafted(n: usize) -> Vec<f32> {
    // values whose E4M3 rounding separates "scale before" from "scale after":
    // e4m3(0.707·1.0) = 0.6875 ≠ 0.707·e4m3(1.0) = 0.707
    (0..n)
        .map(|i| match i % 3 {
            0 => 1.0f32,
            1 => -1.0f32,
            _ => 0.812f32,
        })
        .collect()
}

fn expected_codes_scaled_before(rows: &[f32], scale: f32) -> Vec<u8> {
    // construction-side truth for T18: codes of (scale · v), unit-scale codec
    rows.iter().map(|&x| encode_e4m3(f64::from(x * scale))).collect()
}

/// NEGATIVE (env-default). The FP8 codes must encode `v_scale·V` (T18), and the
/// F32 store must hold `v_scale·V` bitwise. Flips on the naive run (raw V
/// stored, scale applied at read).
#[test]
fn t18_vscale_applied_before_fp8_cache() {
    let (n_tok, n_kv) = (2usize, 2usize);
    let klen = n_tok * n_kv * D_QK;
    let vlen = n_tok * n_kv * D_V;
    let mut rng = XorShift64::new(1801);
    let k = rng.fill_small(klen);
    let v_raw = v_crafted(vlen);

    // FP8 unit-scale store: codes pin
    let mut store = RowStore::new(StoreMode::Fp8Unit, n_kv, D_QK, D_V, VALUE_SCALE);
    store.append(&k, &v_raw, n_tok, bits_from_env()).expect("append");
    let enc = store.encoded().expect("fp8 codes");
    let exp_v_codes = expected_codes_scaled_before(&v_raw, VALUE_SCALE);
    assert!(
        bytes_eq(&enc.v_codes, &exp_v_codes),
        "T18: cached V codes must be e4m3(0.707·V) — v_scale applies BEFORE caching"
    );

    // F32 store: stored-value pin (bitwise)
    let mut f32s = RowStore::new(StoreMode::F32, n_kv, D_QK, D_V, VALUE_SCALE);
    f32s.append(&k, &v_raw, n_tok, bits_from_env()).expect("append");
    let stored = f32s.stored_v_f32();
    let exp_stored: Vec<f32> = v_raw.iter().map(|&x| x * VALUE_SCALE).collect();
    assert!(
        bit_eq(stored, &exp_stored),
        "T18: stored V must be bitwise (0.707·V)"
    );
}

/// BOTH RUNS — attribution: scaling after the store moves the codes, and
/// rescaling on read double-scales the output.
#[test]
fn t18_detection_scaled_after_store_differs() {
    let (n_tok, n_kv) = (1usize, 1usize);
    let k = vec![0.1f32; n_tok * n_kv * D_QK];
    let v_raw = v_crafted(n_tok * n_kv * D_V);
    let mut after = RowStore::new(StoreMode::Fp8Unit, n_kv, D_QK, D_V, VALUE_SCALE);
    after
        .append(&k, &v_raw, n_tok, NaiveBits::VSCALE_AFTER_STORE)
        .expect("append");
    let enc = after.encoded().expect("codes");
    let exp_v_codes = expected_codes_scaled_before(&v_raw, VALUE_SCALE);
    assert!(
        !bytes_eq(&enc.v_codes, &exp_v_codes),
        "after-store oracle produced identical codes — no detection power"
    );
    // double-scale detection (T18 read side)
    let mut f32s = RowStore::new(StoreMode::F32, n_kv, D_QK, D_V, VALUE_SCALE);
    f32s.append(&k, &v_raw, n_tok, NaiveBits::NONE).expect("append");
    let (_, v_once) = f32s.rows(NaiveBits::NONE);
    let (_, v_twice) = f32s.rows(NaiveBits::VSCALE_ON_READ);
    assert!(
        max_abs_diff(&v_once, &v_twice) > 0.1,
        "on-read rescale oracle converged with the correct read — no detection power"
    );
}

/// Run one chunked incremental pass over an SWA ring and return the per-chunk
/// outputs computed FROM THE RING, plus the full-history reference outputs.
fn chunked_ring_pass(
    spec: &AttnSpec,
    naive: NaiveBits,
    chunks: &[(Vec<f32>, Vec<f32>, Vec<i64>)],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut ring = SwaRing::new(
        StoreMode::F32,
        spec.n_kv,
        spec.d_qk,
        spec.d_v,
        spec.window,
        spec.value_scale,
    );
    let mut from_ring = Vec::new();
    let mut from_full = Vec::new();
    let mut all_k: Vec<f32> = Vec::new();
    let mut all_v: Vec<f32> = Vec::new();
    let mut all_pos: Vec<i64> = Vec::new();
    for (k, v, pos) in chunks {
        ring.append(k, v, pos, naive).expect("ring append");
        all_k.extend_from_slice(k);
        // full-history reference uses CACHED form (post-v_scale, T18) like the ring
        all_v.extend(v.iter().map(|&x| x * VALUE_SCALE));
        all_pos.extend_from_slice(pos);
        // one query row per batch row, at the batch's own positions
        let (rk, rv, rp) = ring.get(naive);
        let q = attention_query_rows(spec, k, pos);
        let out_r = attention(&spec, &q, &rk, &rv, pos, &rp, None, NaiveBits::NONE).expect("ring attn");
        let out_f =
            attention(&spec, &q, &all_k, &all_v, pos, &all_pos, None, NaiveBits::NONE).expect("full attn");
        from_ring.push(out_r);
        from_full.push(out_f);
    }
    (from_ring, from_full)
}

/// Query rows for a batch: K rows of the batch stand in as Q (any deterministic
/// input works — the comparison is ring-vs-full on the same queries).
fn attention_query_rows(spec: &AttnSpec, k_batch: &[f32], pos: &[i64]) -> Vec<f32> {
    let _ = pos;
    // use the batch's K rows projected to Q width (same d_qk) per Q head by
    // tiling over heads deterministically
    let n_tok = k_batch.len() / (spec.n_kv * spec.d_qk);
    let mut q = Vec::with_capacity(n_tok * spec.n_q * spec.d_qk);
    for t in 0..n_tok {
        for h in 0..spec.n_q {
            let kvh = h / spec.n_rep();
            let base = (t * spec.n_kv + kvh) * spec.d_qk;
            q.extend_from_slice(&k_batch[base..base + spec.d_qk]);
        }
    }
    q
}

fn ring_chunks(n_chunks: usize, chunk_rows: usize) -> Vec<(Vec<f32>, Vec<f32>, Vec<i64>)> {
    let mut rng = XorShift64::new(8080);
    (0..n_chunks)
        .map(|c| {
            let n = chunk_rows;
            let pos: Vec<i64> = ((c * n) as i64..((c + 1) * n) as i64).collect();
            let k = rng.fill_small(n * 8 * D_QK);
            let v = rng.fill_small(n * 8 * D_V);
            (k, v, pos)
        })
        .collect()
}

fn swa_spec() -> AttnSpec {
    AttnSpec {
        n_kv: 8,
        n_q: 8,
        window: 8,
        ..AttnSpec::real(Family::Swa)
    }
}

/// NEGATIVE (env-default). T8: the ring must keep every row visible to the
/// just-appended batch (`keep_from = min(batch_pos) − window + 1`, keeping
/// `window + T` rows) — incremental ring output must equal the full-history
/// reference for each chunk. Flips on the naive run ("keep the last window"
/// eats rows the early queries of the chunk still need).
#[test]
fn t8_swa_eviction_keep_from_min_batch_pos() {
    let spec = swa_spec();
    let chunks = ring_chunks(5, 4); // window 8, batch 4 -> needs 12 rows
    let (from_ring, from_full) = chunked_ring_pass(&spec, bits_from_env(), &chunks);
    for (c, (r, f)) in from_ring.iter().zip(from_full.iter()).enumerate() {
        let d = max_abs_diff(r, f);
        assert!(
            d <= 1e-12,
            "T8: chunk {c} ring output diverges from full history by {d} — eviction dropped a visible row"
        );
    }
}

/// BOTH RUNS — the R4 invariant (incremental ≡ full recompute) across ring
/// wrap, on the correct implementation.
#[test]
fn incremental_equiv_full_across_ring_wrap() {
    let spec = swa_spec();
    let chunks = ring_chunks(8, 4);
    let (from_ring, from_full) = chunked_ring_pass(&spec, NaiveBits::NONE, &chunks);
    for (c, (r, f)) in from_ring.iter().zip(from_full.iter()).enumerate() {
        let d = max_abs_diff(r, f);
        assert!(d <= 1e-12, "incremental ≠ full at chunk {c} (diff {d})");
    }
    // and the naive eviction really is wrong on the same input (detection)
    let (bad_ring, bad_full) = chunked_ring_pass(&spec, NaiveBits::EVICT_KEEP_LAST, &chunks);
    let mut worst = 0.0f64;
    for (r, f) in bad_ring.iter().zip(bad_full.iter()) {
        worst = worst.max(max_abs_diff(r, f));
    }
    assert!(worst > 1e-6, "keep-last-window oracle has no detection power ({worst})");
}

/// BOTH RUNS — GA paging (A2) + T18 through the paged store: page arithmetic at
/// the 255/256 boundary, page-table indirection, pool byte accounting, and the
/// GA-never-evicts rule (T3).
#[test]
fn ga_page_table_and_pool_pins() {
    // real page size arithmetic (256-token pages)
    let ga = GaPaged::new(StoreMode::F32, 4, D_QK, D_V, VALUE_SCALE);
    assert_eq!(ga.page_of(255), (0, 255));
    assert_eq!(ga.page_of(256), (1, 0));
    assert_eq!(ga.page_of(511), (1, 255));
    assert_eq!(ga.page_tokens(), mimo26_attn::geom::PAGE_TOKENS);

    // multi-page store through the table (tiny pages for the test)
    let (n_kv, n_tok) = (2usize, 10usize);
    let mut rng = XorShift64::new(256);
    let k = rng.fill_small(n_tok * n_kv * D_QK);
    let v_raw = rng.fill_small(n_tok * n_kv * D_V);
    let mut paged = GaPaged::new(StoreMode::F32, n_kv, D_QK, D_V, VALUE_SCALE).with_page_tokens(4);
    paged.append(&k, &v_raw, n_tok, NaiveBits::NONE).expect("append");
    assert_eq!(paged.n_tokens(), n_tok, "T3: GA never evicts");
    assert_eq!(paged.allocated_pages(), 3, "10 rows / 4 per page -> 3 pages");
    assert_eq!(paged.page_of(9), (2, 1));
    assert_eq!(paged.slot_of(0), 0);
    // rows through the page table == flat rows, and V arrives scaled (T18)
    let (fk, fv) = paged.rows_range(0, n_tok, NaiveBits::NONE);
    assert!(bit_eq(&fk, &k), "paged K read must be exact");
    let exp_v: Vec<f32> = v_raw.iter().map(|&x| x * VALUE_SCALE).collect();
    assert!(bit_eq(&fv, &exp_v), "T18 through pages: cached V is 0.707·V");
    // per-page reads assemble the same rows (chunked prefill over pages unit)
    for p in 0..3 {
        let (pk, pv) = paged.page_rows(p, NaiveBits::NONE);
        let lo = p * 4;
        let hi = (lo + 4).min(n_tok);
        let (ck, cv) = paged.rows_range(lo, hi, NaiveBits::NONE);
        assert!(bit_eq(&pk, &ck), "page {p} rows");
        assert!(bit_eq(&pv, &cv), "page {p} values");
    }
    // pool bytes grow by page granularity
    let b = paged.pool_bytes(NaiveBits::NONE);
    assert_eq!(b, 3 * 4 * n_kv * (D_QK + D_V) * 4, "f32 twin pool accounting");
    let mut fp8 = GaPaged::new(StoreMode::Fp8Unit, n_kv, D_QK, D_V, VALUE_SCALE).with_page_tokens(4);
    fp8.append(&k, &v_raw, n_tok, NaiveBits::NONE).expect("append");
    assert_eq!(
        fp8.pool_bytes(NaiveBits::NONE),
        3 * 4 * fp8kv_per_token_unit(n_kv),
        "FP8 unit-scale pool accounting"
    );
}

fn fp8kv_per_token_unit(n_kv: usize) -> usize {
    mimo26_attn::fp8kv::kv_bytes_per_token(mimo26_attn::fp8kv::Layout::Unit, n_kv, D_QK, D_V)
}

/// BOTH RUNS — the ring's fail-loud position guard (oracle kv.py:67-68).
#[test]
fn ring_rejects_decreasing_positions() {
    let spec = swa_spec();
    let mut ring = SwaRing::new(StoreMode::F32, spec.n_kv, D_QK, D_V, spec.window, VALUE_SCALE);
    let k = vec![0.1f32; 2 * spec.n_kv * D_QK];
    let v = vec![0.1f32; 2 * spec.n_kv * D_V];
    ring.append(&k, &v, &[5, 6], NaiveBits::NONE).expect("first append");
    let err = ring.append(&k, &v, &[3, 4], NaiveBits::NONE);
    assert!(
        matches!(err, Err(AttnError::KvLayout { .. })),
        "decreasing positions must fail loud, got {err:?}"
    );
}
