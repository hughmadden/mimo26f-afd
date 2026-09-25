//! BOTH RUNS — live parity of the Rust twin + its decompositions against
//! `oracle/mimo26` (the byte-verified numpy CPU twin, consumed read-only —
//! AGENTS.md §4: "a golden that imports the code under test only proves
//! self-consistency"). Tiny dims first (real head dims 192/128, few tokens),
//! then the captain-fired GPU cell repeats the same manifest format at 4K/32K
//! (`tests/gpu/run_gpu_parity.sh` → `kernels/parity/attn_parity.cu`).
//!
//! Classification: every test here passes `NaiveBits::NONE` explicitly and
//! PASSES BOTH RUNS. Trap flips live in the NEGATIVE tests of the other files.
//!
//! `v` tensors are CACHED V (post-`v_scale`, T18) and the oracle is called with
//! `value_scale=1.0` on them — the engine contract `cache::RowStore` enforces.

mod common;

use common::*;
use mimo26_attn::attn::{attention, attention_chunked, attention_paged, decode_split_kv};
use mimo26_attn::cache::{GaPaged, RowStore, StoreMode};
use mimo26_attn::geom::{AttnSpec, ScaleMode, VALUE_SCALE};
use mimo26_attn::rope::apply_rotary;
use mimo26_attn::{Family, NaiveBits};

const TOL_TINY: f64 = 1e-5;
const TOL_ROPE: f64 = 2e-4;
const TOL_KV: f64 = 1e-6;

fn case_manifest(
    name: &str,
    ty: &str,
    family: &str,
    window: usize,
    sink: u8,
    theta: f64,
    partial: f64,
    vscale: f64,
    tol_abs: f64,
) -> ManifestCase {
    ManifestCase {
        name: name.into(),
        ty: ty.into(),
        family: family.into(),
        window,
        sink,
        theta,
        partial,
        vscale,
        tol_abs,
        tol_rel: 0.0,
        tensors: Vec::new(),
    }
}

fn tensor(name: &str, dtype: &str, shape: Vec<usize>, file: &str) -> ManifestTensor {
    ManifestTensor {
        name: name.into(),
        dtype: dtype.into(),
        shape,
        file: file.into(),
        is_expect: false,
    }
}

fn run_case(case: &ManifestCase, in_dir: &std::path::Path, out_dir: &std::path::Path) -> ManifestCase {
    write_manifest(&in_dir.join("manifest.txt"), std::slice::from_ref(case));
    run_oracle_eval(in_dir, out_dir);
    let mut got = parse_manifest(&out_dir.join("manifest.txt"));
    assert_eq!(got.len(), 1, "expected exactly one case back from the oracle");
    let mut c = got.remove(0);
    c.name = case.name.clone();
    c
}

fn assert_close(what: &str, got: &[f32], exp: &[f32], tol: f64) {
    let d = max_abs_diff(got, exp);
    assert!(d <= tol, "{what}: max abs diff {d} > tol {tol}");
}

/// `tiny_ga_basic` + `tiny_decode_start_pos`: GA family, real dims (64 Q / 4 KV,
/// QK 192 / V 128), through ALL four paths (two-pass, split-KV ×3, chunked,
/// paged). `start_pos` is non-trivial in the decode case (T9).
#[test]
fn parity_tiny_ga_basic_and_decode_start_pos() {
    let spec = AttnSpec::real(Family::Ga);
    for (name, ty, t_len, q_pos) in [
        ("tiny_ga_basic", "attn", 3usize, vec![3i64, 7, 10]),
        ("tiny_decode_start_pos", "decode", 1usize, vec![1000i64]),
    ] {
        let s_len = 11usize;
        let k_pos: Vec<i64> = match name {
            "tiny_decode_start_pos" => (987..1000).collect(),
            _ => (0..s_len as i64).collect(),
        };
        let s_len = k_pos.len();
        let mut rng = XorShift64::new(0x9E37_79B9 ^ name.len() as u64);
        let q = rng.fill_small(t_len * spec.n_q * spec.d_qk);
        let k = rng.fill_small(s_len * spec.n_kv * spec.d_qk);
        let v_raw = rng.fill_small(s_len * spec.n_kv * spec.d_v);
        // cached V (T18): the cache stores v_scale·v_raw; attention/oracle read
        // the cached form (read_scale=1). ga.append below takes RAW V.
        let v: Vec<f32> = v_raw.iter().map(|&x| x * VALUE_SCALE).collect();

        let mut case = case_manifest(name, ty, "ga", 0, 0, spec.theta, spec.partial_rotary_factor, 1.0, TOL_TINY);
        case.tensors.push(tensor("q", "f32", vec![t_len, spec.n_q, spec.d_qk], "q.bin"));
        case.tensors.push(tensor("k", "f32", vec![s_len, spec.n_kv, spec.d_qk], "k.bin"));
        case.tensors.push(tensor("v", "f32", vec![s_len, spec.n_kv, spec.d_v], "v.bin"));
        case.tensors.push(tensor("q_pos", "i64", vec![t_len], "q_pos.bin"));
        case.tensors.push(tensor("k_pos", "i64", vec![s_len], "k_pos.bin"));

        let in_dir = tmp_dir(&format!("parity-{name}-in"));
        let out_dir = tmp_dir(&format!("parity-{name}-out"));
        write_bin_f32(&in_dir, "q.bin", &q);
        write_bin_f32(&in_dir, "k.bin", &k);
        write_bin_f32(&in_dir, "v.bin", &v);
        write_bin_i64(&in_dir, "q_pos.bin", &q_pos);
        write_bin_i64(&in_dir, "k_pos.bin", &k_pos);

        let got = run_case(&case, &in_dir, &out_dir);
        let exp_t = got.tensor("o");
        assert_eq!(exp_t.shape, vec![t_len, spec.n_q, spec.d_v], "oracle out shape");
        let exp = read_tensor_f32(&out_dir, exp_t);

        let two_pass = attention(&spec, &q, &k, &v, &q_pos, &k_pos, None, NaiveBits::NONE).expect("two-pass");
        assert_close(&format!("{name} two-pass"), &two_pass, &exp, TOL_TINY);
        for n_splits in [2usize, 3, 5] {
            let split = decode_split_kv(&spec, &q, &k, &v, &q_pos, &k_pos, None, n_splits, NaiveBits::NONE)
                .expect("split-kv");
            assert_close(&format!("{name} split-kv x{n_splits}"), &split, &exp, TOL_TINY);
        }
        for chunk in [3usize, 7] {
            let ch = attention_chunked(&spec, &q, &k, &v, &q_pos, &k_pos, None, chunk, NaiveBits::NONE)
                .expect("chunked");
            assert_close(&format!("{name} chunked {chunk}"), &ch, &exp, TOL_TINY);
        }
        // paged GA path (256-token pages at real page_tokens == s_len < 256 is
        // one page; force multi-page with a 4-token page in the tiny case)
        let mut ga = GaPaged::new(StoreMode::F32, spec.n_kv, spec.d_qk, spec.d_v, spec.value_scale)
            .with_page_tokens(4);
        ga.append(&k, &v_raw, s_len, NaiveBits::NONE).expect("paged append"); // RAW V; store applies T18
        let paged = attention_paged(&spec, &q, &q_pos, &k_pos, None, &ga, 1, NaiveBits::NONE).expect("paged");
        assert_close(&format!("{name} paged"), &paged, &exp, TOL_TINY);
    }
}

/// Pin: a NON-0-based paged cache (rows at absolute positions 987..999) with a
/// query whose causal mask hides the tail (`q_pos=988` ⇒ only keys ≤ 988 are
/// visible). `attention_paged` must honor the caller's absolute `k_pos`; the old
/// hardcode (`k_pos = 0..n_tokens`) saw all 13 rows as visible and diverged from
/// the independent oracle. BOTH RUNS (not a naive-16 negative): (a) `NONE` must
/// match the oracle — this FAILS the hardcode if it is reintroduced; (b) the
/// single-bit `PAGED_KPOS_ZERO_BASED` must be *detected* in isolation (NOT
/// `ALL`, where T9's `POS_ZEROED` zeroes `kp` and masks the hardcode — see the
/// report for why this cannot be a clean #17).
#[test]
fn parity_paged_offset_k_pos_partial_mask() {
    let spec = AttnSpec::real(Family::Ga);
    let (t_len, s_len) = (1usize, 13usize);
    let q_pos = vec![988i64];
    // offset 987 — the paged cache starts mid-sequence, so k_pos is NOT 0-based
    let k_pos: Vec<i64> = (987..987 + s_len as i64).collect();
    let mut rng = XorShift64::new(0x0FF5_E77A);
    let q = rng.fill_small(t_len * spec.n_q * spec.d_qk);
    let k = rng.fill_small(s_len * spec.n_kv * spec.d_qk);
    let v_raw = rng.fill_small(s_len * spec.n_kv * spec.d_v);
    let v: Vec<f32> = v_raw.iter().map(|&x| x * VALUE_SCALE).collect(); // cached V (T18)

    let mut case = case_manifest(
        "paged_offset_k_pos",
        "decode",
        "ga",
        0,
        0,
        spec.theta,
        spec.partial_rotary_factor,
        1.0,
        TOL_TINY,
    );
    case.tensors.push(tensor("q", "f32", vec![t_len, spec.n_q, spec.d_qk], "q.bin"));
    case.tensors.push(tensor("k", "f32", vec![s_len, spec.n_kv, spec.d_qk], "k.bin"));
    case.tensors.push(tensor("v", "f32", vec![s_len, spec.n_kv, spec.d_v], "v.bin"));
    case.tensors.push(tensor("q_pos", "i64", vec![t_len], "q_pos.bin"));
    case.tensors.push(tensor("k_pos", "i64", vec![s_len], "k_pos.bin"));
    let in_dir = tmp_dir("parity-paged-offset-in");
    let out_dir = tmp_dir("parity-paged-offset-out");
    write_bin_f32(&in_dir, "q.bin", &q);
    write_bin_f32(&in_dir, "k.bin", &k);
    write_bin_f32(&in_dir, "v.bin", &v);
    write_bin_i64(&in_dir, "q_pos.bin", &q_pos);
    write_bin_i64(&in_dir, "k_pos.bin", &k_pos);
    let got = run_case(&case, &in_dir, &out_dir);
    let exp = read_tensor_f32(&out_dir, got.tensor("o"));

    let mut ga = GaPaged::new(StoreMode::F32, spec.n_kv, spec.d_qk, spec.d_v, spec.value_scale)
        .with_page_tokens(4);
    ga.append(&k, &v_raw, s_len, NaiveBits::NONE).expect("paged append"); // RAW V; store applies T18

    // (a) real k_pos ⇒ matches the oracle. FAILS the hardcode: with
    // `k_pos = 0..n`, the tail rows (real kp 989..999 > 988) become visible.
    let paged = attention_paged(&spec, &q, &q_pos, &k_pos, None, &ga, 1, NaiveBits::NONE).expect("paged");
    assert_close("paged offset k_pos", &paged, &exp, TOL_TINY);

    // (b) the 0-based hardcode is DETECTED in isolation (single bit, not `ALL`).
    let hardcoded = attention_paged(
        &spec,
        &q,
        &q_pos,
        &k_pos,
        None,
        &ga,
        1,
        NaiveBits::of(NaiveBits::PAGED_KPOS_ZERO_BASED),
    )
    .expect("paged naive");
    let d = max_abs_diff(&hardcoded, &exp);
    assert!(d > TOL_TINY, "the paged 0-based k_pos hardcode must be detected (diff {d})");
}

/// `tiny_swa_sink_window`: SWA family, window 4 (wraps), per-Q-head sink `[64]`
/// (T6), all decompositions incl. the split-KV sink-once reduce.
#[test]
fn parity_tiny_swa_sink_window() {
    let spec = AttnSpec {
        window: 4,
        ..AttnSpec::real(Family::Swa)
    };
    let (t_len, s_len) = (3usize, 13usize);
    let q_pos = vec![5i64, 11, 12];
    let k_pos: Vec<i64> = (0..s_len as i64).collect();
    let mut rng = XorShift64::new(2026_0923);
    let q = rng.fill_small(t_len * spec.n_q * spec.d_qk);
    let k = rng.fill_small(s_len * spec.n_kv * spec.d_qk);
    let v = rng.fill_small(s_len * spec.n_kv * spec.d_v);
    let sink: Vec<f32> = (0..spec.n_q).map(|h| rng.next_small() + 0.05 * h as f32).collect();

    let mut case = case_manifest(
        "tiny_swa_sink_window",
        "attn",
        "swa",
        spec.window,
        1,
        spec.theta,
        spec.partial_rotary_factor,
        1.0,
        TOL_TINY,
    );
    case.tensors.push(tensor("q", "f32", vec![t_len, spec.n_q, spec.d_qk], "q.bin"));
    case.tensors.push(tensor("k", "f32", vec![s_len, spec.n_kv, spec.d_qk], "k.bin"));
    case.tensors.push(tensor("v", "f32", vec![s_len, spec.n_kv, spec.d_v], "v.bin"));
    case.tensors.push(tensor("q_pos", "i64", vec![t_len], "q_pos.bin"));
    case.tensors.push(tensor("k_pos", "i64", vec![s_len], "k_pos.bin"));
    case.tensors.push(tensor("sink", "f32", vec![spec.n_q], "sink.bin"));

    let in_dir = tmp_dir("parity-swa-in");
    let out_dir = tmp_dir("parity-swa-out");
    write_bin_f32(&in_dir, "q.bin", &q);
    write_bin_f32(&in_dir, "k.bin", &k);
    write_bin_f32(&in_dir, "v.bin", &v);
    write_bin_i64(&in_dir, "q_pos.bin", &q_pos);
    write_bin_i64(&in_dir, "k_pos.bin", &k_pos);
    write_bin_f32(&in_dir, "sink.bin", &sink);

    let got = run_case(&case, &in_dir, &out_dir);
    let exp = read_tensor_f32(&out_dir, got.tensor("o"));

    let two_pass = attention(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&sink), NaiveBits::NONE).expect("two-pass");
    assert_close("swa two-pass", &two_pass, &exp, TOL_TINY);
    for n_splits in [1usize, 4, 8] {
        let split =
            decode_split_kv(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&sink), n_splits, NaiveBits::NONE)
                .expect("split-kv");
        assert_close(&format!("swa split-kv x{n_splits}"), &split, &exp, TOL_TINY);
    }
    let ch = attention_chunked(&spec, &q, &k, &v, &q_pos, &k_pos, Some(&sink), 5, NaiveBits::NONE)
        .expect("chunked");
    assert_close("swa chunked", &ch, &exp, TOL_TINY);
}

/// Rope parity vs the oracle's `apply_rotary` (f64 angles there; ours are FP32
/// on-the-fly per T19 — at these positions the two agree inside `TOL_ROPE`).
#[test]
fn parity_tiny_rope_ga_and_swa() {
    for (name, heads, theta) in [
        ("tiny_rope_ga", 4usize, mimo26_attn::geom::ROPE_THETA),
        ("tiny_rope_swa", 8usize, mimo26_attn::geom::SWA_ROPE_THETA),
    ] {
        let (t_len, d) = (2usize, 192usize);
        let positions = vec![1i64, 300];
        let mut rng = XorShift64::new(0x51ED_0000 ^ name.len() as u64);
        let x = rng.fill_small(t_len * heads * d);
        let mut case = case_manifest(name, "rope", "na", 0, 0, theta, 0.334, 1.0, TOL_ROPE);
        case.tensors.push(tensor("x", "f32", vec![t_len, heads, d], "x.bin"));
        case.tensors.push(tensor("pos", "i64", vec![t_len], "pos.bin"));
        let in_dir = tmp_dir(&format!("parity-{name}-in"));
        let out_dir = tmp_dir(&format!("parity-{name}-out"));
        write_bin_f32(&in_dir, "x.bin", &x);
        write_bin_i64(&in_dir, "pos.bin", &positions);
        let got = run_case(&case, &in_dir, &out_dir);
        let exp = read_tensor_f32(&out_dir, got.tensor("y"));
        let y = apply_rotary(&x, t_len, heads, d, &positions, theta, 0.334, NaiveBits::NONE);
        assert_close(name, &y, &exp, TOL_ROPE);
    }
}

/// FP8 KV store parity (T18 + T20): the store path applies `v_scale` BEFORE
/// quantization; expected codes/decodes come from the oracle codec
/// (`mimo26.quant.fp8_block`), not from this crate's codec chain.
#[test]
fn parity_tiny_kv_store_fp8() {
    let (n_tok, n_kv, d_qk, d_v) = (3usize, 4usize, 192usize, 128usize);
    let mut rng = XorShift64::new(4242);
    let k = rng.fill_small(n_tok * n_kv * d_qk);
    let mut v_raw = rng.fill_small(n_tok * n_kv * d_v);
    v_raw[0] = 700.0; // amax clip probe: survives the 0.707 pre-scale (T18) —
    // unit-scale expects clip == 1; pth's plane scale absorbs it (clip == 0)

    for (name, mode) in [
        ("tiny_kv_store_pth", ScaleMode::PerTokenHead),
        ("tiny_kv_store_unit", ScaleMode::Unit),
    ] {
        let mut case = case_manifest(
            name,
            if mode == ScaleMode::Unit { "kv_store_unit" } else { "kv_store_pth" },
            "na",
            0,
            0,
            0.0,
            0.0,
            f64::from(VALUE_SCALE),
            TOL_KV,
        );
        case.tensors.push(tensor("k", "f32", vec![n_tok, n_kv, d_qk], "k.bin"));
        case.tensors.push(tensor("v_raw", "f32", vec![n_tok, n_kv, d_v], "v_raw.bin"));
        let in_dir = tmp_dir(&format!("parity-{name}-in"));
        let out_dir = tmp_dir(&format!("parity-{name}-out"));
        write_bin_f32(&in_dir, "k.bin", &k);
        write_bin_f32(&in_dir, "v_raw.bin", &v_raw);
        let got = run_case(&case, &in_dir, &out_dir);

        // The STORE path applies `v_scale` before quantization (T18) — exactly
        // what the manifest promises ("v_raw is RAW V").
        let store_mode = match mode {
            ScaleMode::Unit => StoreMode::Fp8Unit,
            ScaleMode::PerTokenHead => StoreMode::Fp8PerTokenHead,
        };
        let mut store = RowStore::new(store_mode, n_kv, d_qk, d_v, VALUE_SCALE);
        store.append(&k, &v_raw, n_tok, NaiveBits::NONE).expect("RowStore::append");
        let enc = store.encoded().expect("fp8 encoding");
        let (k_dec, v_dec) = store.rows(NaiveBits::NONE);
        assert_close(&format!("{name} k_dec"), &k_dec, &read_tensor_f32(&out_dir, got.tensor("k_dec")), TOL_KV);
        assert_close(&format!("{name} v_dec"), &v_dec, &read_tensor_f32(&out_dir, got.tensor("v_dec")), TOL_KV);
        let clip_exp = read_tensor_i64(&out_dir, got.tensor("clip"))[0] as u64;
        assert_eq!(
            enc.clip_count, clip_exp,
            "{name}: amax clip count must match the oracle's ({})",
            enc.clip_count
        );
    }
}
