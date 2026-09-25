//! FP8 KV codec + scale layout (T20, ARCHITECTURE.md §11.7/§11.12).
//!
//! **Unit-scale first (§11.12 amendment, binding):** cast K/V to E4M3 with
//! scale 1.0 and upcast on read — exactly `11,520 B/token`, **no scale bytes**
//! (vLLM's working path; needles pass at 100K/250K). A per-layer **amax clip
//! check** guards it ([`EncodedKv::clip_count`]; the clip count must be 0 on
//! real activations). Per-token scales are the extension, adopted only if that
//! check or G4n fails.
//!
//! **T20 layout pin:** when scales exist they are **per token × head** (not
//! per-128-element blocks) with **K and V in separate planes** and the scale
//! bytes counted in the pool math (`geom::bytes`). "block-128" cannot express
//! this: 128 does not divide the 192-dim K, so a block-128 grid over the
//! flattened `K‖V` row mixes the tail of K (dims 128..192) into the same scale
//! block as the head of V — silent quality loss at long range (T20) and pool
//! over-commit. [`Layout::Block128Shared`] is that wrong layout, kept as the
//! bug oracle behind [`NaiveBits::BLOCK128_SHARED_SCALES`].
//!
//! The E4M3 codec itself is `mimo26_load::e4m3` — the externally golden-pinned
//! codec (`fp8_block_golden.json`, sha256 in `mimo26-load/src/lib.rs:26`);
//! one codec for weights and KV, never a second copy (I-Gold).

use crate::geom::ScaleMode;
use crate::{AttnError, NaiveBits};
use mimo26_load::e4m3::{decode_e4m3, encode_e4m3, E4M3_MAX};

/// Scale-plane element size (f32 per token × head).
pub const SCALE_BYTES: usize = std::mem::size_of::<f32>();

/// Which scale layout the codes were (or would be) paired with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// scale = 1.0, no scale plane (the §11.12 default).
    Unit,
    /// One f32 scale per token × head per plane; K and V SEPARATE (T20).
    PerTokenHead,
    /// BUG ORACLE (T20): block-128 scales over the flattened `K‖V` row, K and V
    /// sharing one grid. Selected by [`NaiveBits::BLOCK128_SHARED_SCALES`].
    Block128Shared,
}

impl Layout {
    pub fn of(mode: ScaleMode, naive: NaiveBits) -> Self {
        if naive.has(NaiveBits::BLOCK128_SHARED_SCALES) {
            Layout::Block128Shared
        } else {
            match mode {
                ScaleMode::Unit => Layout::Unit,
                ScaleMode::PerTokenHead => Layout::PerTokenHead,
            }
        }
    }
}

/// One layer's cached K/V rows for `n_tok` tokens, FP8-encoded.
///
/// Row-major index convention (shared with `kernels/kv_cache_fp8.cu`):
/// `k_codes[(t*n_kv + h)*d_qk + i]`, `v_codes[(t*n_kv + h)*d_v + j]`.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedKv {
    pub layout: Layout,
    pub n_tok: usize,
    pub n_kv: usize,
    pub d_qk: usize,
    pub d_v: usize,
    /// `[n_tok*n_kv*d_qk]`.
    pub k_codes: Vec<u8>,
    /// `[n_tok*n_kv*d_v]`.
    pub v_codes: Vec<u8>,
    /// Layout::PerTokenHead only — `[n_tok*n_kv]`, K plane.
    pub k_scales: Vec<f32>,
    /// Layout::PerTokenHead only — `[n_tok*n_kv]`, **separate** V plane.
    pub v_scales: Vec<f32>,
    /// Layout::Block128Shared only — `[ceil(n_tok*n_kv*(d_qk+d_v)/128)]`.
    pub shared_scales: Vec<f32>,
    /// Values whose magnitude exceeded `E4M3_MAX` and were clamped on encode.
    /// The per-layer amax gate (ADVISOR-I3 §10.4.2) requires 0 on real
    /// activations; [`NaiveBits::SILENT_CLAMP`] hides it (that is the bug).
    pub clip_count: u64,
}

// ---------------------------------------------------------------------------
// layout index arithmetic (the T20 pin — mirrored by the CUDA store kernel)
// ---------------------------------------------------------------------------

/// K code byte offset within the K plane. Plane size: `n_tok*n_kv*d_qk`.
pub fn k_code_off(t: usize, h: usize, i: usize, n_kv: usize, d_qk: usize) -> usize {
    (t * n_kv + h) * d_qk + i
}

/// V code byte offset within the V plane (a SEPARATE allocation from K).
pub fn v_code_off(t: usize, h: usize, j: usize, n_kv: usize, d_v: usize) -> usize {
    (t * n_kv + h) * d_v + j
}

/// K scale offset within the K scale plane (`[n_tok*n_kv]`, f32 elements).
/// Per token × head — never per-128-element block (T20).
pub fn k_scale_off(t: usize, h: usize, n_kv: usize) -> usize {
    t * n_kv + h
}

/// V scale offset within the **separate** V scale plane (`[n_tok*n_kv]`).
pub fn v_scale_off(t: usize, h: usize, n_kv: usize) -> usize {
    t * n_kv + h
}

/// BUG ORACLE (T20): block-128 scale index over the flattened `K‖V` row of one
/// token — `flat = h*(d_qk+d_v) + (i or d_qk + j)`, `scale = shared[flat / 128]`.
/// With `d_qk = 192` the second K block (128..192) shares a scale with the
/// first 64 V dims of the same head: exactly the trap.
pub fn shared_block128_off(flat_in_row: usize) -> usize {
    flat_in_row / 128
}

/// Number of f32 scales one token needs in the shared bug layout.
pub fn shared_blocks_per_row(n_kv: usize, d_qk: usize, d_v: usize) -> usize {
    (n_kv * (d_qk + d_v) + 127) / 128
}

/// Per-layer KV bytes per token for the given layout (codes + scales).
pub fn kv_bytes_per_token(layout: Layout, n_kv: usize, d_qk: usize, d_v: usize) -> usize {
    let codes = n_kv * (d_qk + d_v);
    match layout {
        Layout::Unit => codes,
        Layout::PerTokenHead => codes + n_kv * 2 * SCALE_BYTES,
        Layout::Block128Shared => codes + shared_blocks_per_row(n_kv, d_qk, d_v) * SCALE_BYTES,
    }
}

// ---------------------------------------------------------------------------
// encode / decode
// ---------------------------------------------------------------------------

fn encode_rows(
    values: &[f32],
    n_tok: usize,
    n_kv: usize,
    d: usize,
    scale_of: impl Fn(usize, usize) -> f32,
    naive: NaiveBits,
) -> (Vec<u8>, u64) {
    assert_eq!(values.len(), n_tok * n_kv * d, "row plane length");
    let mut codes = vec![0u8; values.len()];
    let mut clips = 0u64;
    for t in 0..n_tok {
        for h in 0..n_kv {
            let s = scale_of(t, h);
            let base = (t * n_kv + h) * d;
            for i in 0..d {
                // Kernel semantics (kv_cache_fp8.cu `store_kernel` /
                // `m26::e4m3_encode`): the pre-encode quotient is computed in
                // f32. An f64 quotient nudges values sitting exactly at the
                // ±448 boundary (700 probe → amax → 448 after the plane scale)
                // and over-counts clips. Mirror the kernel exactly: f32
                // division, then widen (exact) for the compare/encode.
                let q = values[base + i] / s;
                if f64::from(q).abs() > E4M3_MAX {
                    clips += 1;
                }
                codes[base + i] = encode_e4m3(f64::from(q)); // encode_e4m3 clamps to ±448
            }
        }
    }
    if naive.has(NaiveBits::SILENT_CLAMP) {
        // The bug: clamp happened but the gate is told nothing (P-106 class).
        (codes, 0)
    } else {
        (codes, clips)
    }
}

/// Encode one layer's K rows (`[n_tok][n_kv][d_qk]`) and V rows
/// (`[n_tok][n_kv][d_v]`) — **V must arrive already `v_scale`d (T18: the cache
/// scales before storing)**; this function quantizes exactly what it is given.
pub fn encode_kv(
    k_rows: &[f32],
    v_rows: &[f32],
    n_tok: usize,
    n_kv: usize,
    d_qk: usize,
    d_v: usize,
    mode: ScaleMode,
    naive: NaiveBits,
) -> Result<EncodedKv, AttnError> {
    if n_tok == 0 {
        return Err(AttnError::KvLayout { what: "encode_kv: empty".into() });
    }
    let layout = Layout::of(mode, naive);
    match layout {
        Layout::Unit => {
            let (k_codes, kc) = encode_rows(k_rows, n_tok, n_kv, d_qk, |_, _| 1.0, naive);
            let (v_codes, vc) = encode_rows(v_rows, n_tok, n_kv, d_v, |_, _| 1.0, naive);
            Ok(EncodedKv {
                layout,
                n_tok,
                n_kv,
                d_qk,
                d_v,
                k_codes,
                v_codes,
                k_scales: Vec::new(),
                v_scales: Vec::new(),
                shared_scales: Vec::new(),
                clip_count: kc + vc,
            })
        }
        Layout::PerTokenHead => {
            // Per token × head amax/448 scale; K and V computed independently
            // into SEPARATE planes (T20).
            let mut k_scales = vec![0f32; n_tok * n_kv];
            let mut v_scales = vec![0f32; n_tok * n_kv];
            for t in 0..n_tok {
                for h in 0..n_kv {
                    let kb = (t * n_kv + h) * d_qk;
                    let vb = (t * n_kv + h) * d_v;
                    k_scales[k_scale_off(t, h, n_kv)] = plane_scale(&k_rows[kb..kb + d_qk]);
                    v_scales[v_scale_off(t, h, n_kv)] = plane_scale(&v_rows[vb..vb + d_v]);
                }
            }
            let (k_codes, kc) =
                encode_rows(k_rows, n_tok, n_kv, d_qk, |t, h| k_scales[k_scale_off(t, h, n_kv)], naive);
            let (v_codes, vc) =
                encode_rows(v_rows, n_tok, n_kv, d_v, |t, h| v_scales[v_scale_off(t, h, n_kv)], naive);
            Ok(EncodedKv {
                layout,
                n_tok,
                n_kv,
                d_qk,
                d_v,
                k_codes,
                v_codes,
                k_scales,
                v_scales,
                shared_scales: Vec::new(),
                clip_count: kc + vc,
            })
        }
        Layout::Block128Shared => {
            // BUG ORACLE: one scale per 128 flattened K‖V elements per token.
            let blocks = shared_blocks_per_row(n_kv, d_qk, d_v);
            let mut shared = vec![0f32; n_tok * blocks];
            for t in 0..n_tok {
                for b in 0..blocks {
                    let mut amax = 0f64;
                    for flat in b * 128..((b + 1) * 128).min(n_kv * (d_qk + d_v)) {
                        let (val, _) = flat_lookup(k_rows, v_rows, t, n_kv, d_qk, d_v, flat);
                        amax = amax.max(f64::from(val).abs());
                    }
                    shared[t * blocks + b] = if amax > 0.0 { (amax / E4M3_MAX) as f32 } else { 1.0 };
                }
            }
            let mut k_codes = vec![0u8; n_tok * n_kv * d_qk];
            let mut v_codes = vec![0u8; n_tok * n_kv * d_v];
            let mut clips = 0u64;
            for t in 0..n_tok {
                for h in 0..n_kv {
                    for i in 0..d_qk {
                        let flat = h * (d_qk + d_v) + i;
                        let s = shared[t * blocks + shared_block128_off(flat)];
                        let q = f64::from(k_rows[k_code_off(t, h, i, n_kv, d_qk)]) / f64::from(s);
                        if q.abs() > E4M3_MAX {
                            clips += 1;
                        }
                        k_codes[k_code_off(t, h, i, n_kv, d_qk)] = encode_e4m3(q);
                    }
                    for j in 0..d_v {
                        let flat = h * (d_qk + d_v) + d_qk + j;
                        let s = shared[t * blocks + shared_block128_off(flat)];
                        let q = f64::from(v_rows[v_code_off(t, h, j, n_kv, d_v)]) / f64::from(s);
                        if q.abs() > E4M3_MAX {
                            clips += 1;
                        }
                        v_codes[v_code_off(t, h, j, n_kv, d_v)] = encode_e4m3(q);
                    }
                }
            }
            let clips = if naive.has(NaiveBits::SILENT_CLAMP) { 0 } else { clips };
            Ok(EncodedKv {
                layout,
                n_tok,
                n_kv,
                d_qk,
                d_v,
                k_codes,
                v_codes,
                k_scales: Vec::new(),
                v_scales: Vec::new(),
                shared_scales: shared,
                clip_count: clips,
            })
        }
    }
}

fn plane_scale(row: &[f32]) -> f32 {
    let mut amax = 0f64;
    for &v in row {
        amax = amax.max(f64::from(v).abs());
    }
    if amax > 0.0 {
        (amax / E4M3_MAX) as f32
    } else {
        1.0
    }
}

fn flat_lookup(
    k_rows: &[f32],
    v_rows: &[f32],
    t: usize,
    n_kv: usize,
    d_qk: usize,
    d_v: usize,
    flat: usize,
) -> (f32, bool) {
    // returns (value, is_k) for flattened [K‖V] position `flat` in token t's row
    let per_head = d_qk + d_v;
    let h = flat / per_head;
    let within = flat % per_head;
    if within < d_qk {
        (k_rows[k_code_off(t, h, within, n_kv, d_qk)], true)
    } else {
        (v_rows[v_code_off(t, h, within - d_qk, n_kv, d_v)], false)
    }
}

/// Decode back to f32 rows. `NaiveBits::VSCALE_ON_READ` re-applies `value_scale`
/// on the read of an already-scaled cache (T18 double-scale bug) — pass the
/// spec's `value_scale` as `read_scale` (1.0 on the correct path).
pub fn decode_kv(enc: &EncodedKv, read_scale: f32) -> (Vec<f32>, Vec<f32>) {
    let EncodedKv { layout, n_tok, n_kv, d_qk, d_v, .. } = *enc;
    let mut k = vec![0f32; n_tok * n_kv * d_qk];
    let mut v = vec![0f32; n_tok * n_kv * d_v];
    for t in 0..n_tok {
        for h in 0..n_kv {
            for i in 0..d_qk {
                let off = k_code_off(t, h, i, n_kv, d_qk);
                let s = match layout {
                    Layout::Unit => 1.0,
                    Layout::PerTokenHead => enc.k_scales[k_scale_off(t, h, n_kv)],
                    Layout::Block128Shared => {
                        let flat = h * (d_qk + d_v) + i;
                        enc.shared_scales[shared_block128_off(flat)]
                    }
                };
                k[off] = decode_e4m3(enc.k_codes[off]) as f32 * s;
            }
            for j in 0..d_v {
                let off = v_code_off(t, h, j, n_kv, d_v);
                let s = match layout {
                    Layout::Unit => 1.0,
                    Layout::PerTokenHead => enc.v_scales[v_scale_off(t, h, n_kv)],
                    Layout::Block128Shared => {
                        let flat = h * (d_qk + d_v) + d_qk + j;
                        enc.shared_scales[shared_block128_off(flat)]
                    }
                };
                v[off] = decode_e4m3(enc.v_codes[off]) as f32 * s * read_scale;
            }
        }
    }
    (k, v)
}
