//! MXFP4 semantics — E2M1 nibbles + E8M0-32 block scales, device-dequant path.
//!
//! # Reference (READ-ONLY, re-derived — no `spike/` code is imported)
//!
//! `spike/mxfp4.py` (I1 spike, golden-pinned against
//! `tests/golden/mxfp4_golden.json::e2m1_codebook_by_nibble`). The semantics
//! below are re-derived from that file's documented storage convention and
//! pinned here by `tests/unpack_fixture.rs` against the **real-block** golden
//! `bench/fixtures/expert_nibble_fixture.json`; the crate never links or copies
//! spike code (AGENTS.md §3: the I1 spike is throwaway and does not enter
//! `crates/` without I2 productization — this is a re-derivation, not a port).
//!
//! Storage (verified live against the real checkpoint headers, 23 Sep 2026
//! AEST): a projection of logical shape `[out, in]` is stored as
//! `weight` `u8 [out, in/2]` (two E2M1 nibbles per byte along the input dim)
//! plus `weight_scale` `u8 [out, in/32]` (one E8M0 scale per 32-element block).
//! Live example: `gate_proj.weight` U8 [2048, 2048] = logical [2048, 4096],
//! `gate_proj.weight_scale` U8 [2048, 128]; `down_proj.weight` U8 [4096, 1024]
//! = logical [4096, 2048], scale [4096, 64].
//!
//! * **T14 nibble order** — input index `k` lives in byte `k/2`; the **LOW**
//!   nibble is the **even** `k`, the high nibble the odd `k`. The naive
//!   (swapped) order loads fine and produces expert garbage.
//! * **T10 E8M0 clamp** — `scale = 2^(b-127)`; byte **255 is RESERVED** (OCP)
//!   and clamps to `2^127`. The naive path computes the unclamped `2^128`, a
//!   poison scale that saturates every expert it touches.
//! * **f32-max saturation** — `|val| * 2^127` can exceed `f32::MAX`, so the
//!   product is saturated to `±f32::MAX` **before** the f32 cast. The naive
//!   unclamped `2^128` always does.
//!
//! # The device path this mirrors
//!
//! `kernels/include/mimo26_expert_bits.h` composes sign/exponent/mantissa
//! directly, including subnormals and finite saturation, without a constant
//! LUT or FP64 arithmetic. Independent host exhaustive tests pin its bits;
//! real GPU bitwise parity remains a separate required gate.

use crate::NaiveBits;

/// E8M0 block width: one scale byte per 32 contiguous input elements.
pub const BLOCK: usize = 32;

/// E8M0 exponent bias.
pub const E8M0_BIAS: i32 = 127;

/// Highest usable E8M0 byte. 255 is reserved by OCP and clamps to `2^127` (T10).
pub const E8M0_BYTE_MAX: u8 = 254;

/// E2M1 codebook indexed directly by nibble (0..15).
///
/// Mirrors `spike/mxfp4.py::E2M1_CODEBOOK` / the golden
/// `e2m1_codebook_by_nibble`. Note nibble 8 is `-0.0` (sign bit set, zero
/// magnitude) — it decodes to `-0.0f32`, which compares equal to `0.0` but has
/// a different bit pattern; the fixture check compares bits, not values.
pub const E2M1_CODEBOOK: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// E8M0 byte -> scale as an exact power of two.
///
/// `2^(b-127)`; byte 255 clamps to `2^127` (T10). Returned as `f64` because
/// the product with a large E2M1 value can exceed `f32::MAX` — the spike's
/// reference path is float64 for exactly this reason (`spike/mxfp4.py:44-51`).
pub fn e8m0_scale(byte: u8, naive: NaiveBits) -> f64 {
    if naive.has(NaiveBits::SCALE_ONE) { return 1.0; }
    let b = if naive.has(NaiveBits::E8M0_NO_CLAMP) {
        byte as i32
    } else {
        core::cmp::min(byte, E8M0_BYTE_MAX) as i32
    };
    exp2(b - E8M0_BIAS)
}

/// Exact `2^e` for `e` in `[-1074, 1023]` (the f64 normal range plus subnormals
/// down to `2^-1074`). E8M0 exponents live in `[-127, 128]`, so this is exact
/// for every byte including the naive unclamped 255 -> `2^128`.
pub fn exp2(e: i32) -> f64 {
    debug_assert!((-1074..=1023).contains(&e), "exp2 exponent out of f64 range");
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        // Subnormal: 2^-1074 is the smallest positive f64.
        f64::from_bits(1u64 << (e + 1074))
    }
}

/// Decode one packed byte into its two E2M1 values, `[even_k, odd_k]`.
///
/// Correct (T14): low nibble = even `k`, high nibble = odd `k`.
pub fn decode_byte(byte: u8, naive: NaiveBits) -> [f32; 2] {
    let lo = E2M1_CODEBOOK[(byte & 0x0f) as usize];
    let hi = E2M1_CODEBOOK[((byte >> 4) & 0x0f) as usize];
    if naive.has(NaiveBits::NIBBLE_SWAP) {
        [hi, lo]
    } else {
        [lo, hi]
    }
}

/// The scale byte for input column `k` of a row, under `naive`.
///
/// Correct: block `k / 32`. Naive ([`NaiveBits::SCALE_OFF_BY_ONE`]): block
/// `k / 32 + 1`, with a missing final byte replaced by byte0 (scale2^-127,
/// NOT floating-point zero). The wrong path never performs an actual OOB read.
#[inline]
pub fn scale_byte_for(scales: &[u8], k: usize, naive: NaiveBits) -> u8 {
    let block = k / BLOCK;
    if naive.has(NaiveBits::SCALE_OFF_BY_ONE) {
        *scales.get(block + 1).unwrap_or(&0)
    } else {
        scales[block]
    }
}

/// Dequantize one row: `packed` `[in/2]` + `scales` `[in/32]` -> `f32 [in]`.
///
/// Mirrors `spike/mxfp4.py::unpack` for a single row: float64 nibble decode,
/// exact `2^e` scale, float64 product, saturate to `±f32::MAX`, then cast.
/// The saturation is not cosmetic — `|val| * 2^127` overflows `f32`, and the
/// naive unclamped `2^128` always does.
pub fn unpack_row(packed: &[u8], scales: &[u8], naive: NaiveBits) -> Vec<f32> {
    let inn = packed.len() * 2;
    let mut out = vec![0.0f32; inn];
    let f32_max = f32::MAX as f64;
    for k in 0..inn {
        let byte = packed[k / 2];
        let pair = decode_byte(byte, naive);
        let val = pair[k % 2] as f64;
        let sbyte = scale_byte_for(scales, k, naive);
        let prod = val * e8m0_scale(sbyte, naive);
        out[k] = prod.clamp(-f32_max, f32_max) as f32;
    }
    out
}

/// Dequantize a whole matrix: `packed` `[out, in/2]` + `scales` `[out, in/32]`
/// -> `f32 [out, in]` row-major.
pub fn unpack_matrix(
    packed: &[u8],
    scales: &[u8],
    out_rows: usize,
    in_cols: usize,
    naive: NaiveBits,
) -> Vec<f32> {
    let half = in_cols / 2;
    let srow = in_cols / BLOCK;
    let mut res = Vec::with_capacity(out_rows * in_cols);
    for r in 0..out_rows {
        let p = &packed[r * half..(r + 1) * half];
        let s = &scales[r * srow..(r + 1) * srow];
        res.extend_from_slice(&unpack_row(p, s, naive));
    }
    res
}

/// Dequantize one element `(row, col)` without materializing the row — the
/// device kernel's access pattern, and the fixture check's inner loop.
///
/// `packed` is the row's `[in/2]` payload, `scales` the row's `[in/32]` scales.
#[inline]
pub fn unpack_element(packed: &[u8], scales: &[u8], col: usize, naive: NaiveBits) -> f32 {
    let byte = packed[col / 2];
    let pair = decode_byte(byte, naive);
    let val = pair[col % 2] as f64;
    let sbyte = scale_byte_for(scales, col, naive);
    let prod = val * e8m0_scale(sbyte, naive);
    let f32_max = f32::MAX as f64;
    prod.clamp(-f32_max, f32_max) as f32
}

/// Quantize one `f32` value to the nearest E2M1 code (round-half-away-from-zero
/// on magnitude, ties to the larger magnitude — the OCP E2M1 rounding rule).
///
/// Used only by the synthetic fixture builder: the checkpoint is already
/// quantized, so the kernel never quantizes. Kept here so the fixture is built
/// from the same codebook the decoder uses.
pub fn quantize_e2m1(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 8u8 } else { 0u8 };
    let a = x.abs();
    let mags = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, m) in mags.iter().enumerate() {
        let d = (a - m).abs();
        // `<=` picks the larger magnitude on an exact tie.
        if d <= best_d {
            best_d = d;
            best = i;
        }
    }
    sign | best as u8
}

/// Pack two E2M1 codes into one byte with the correct (T14) nibble order:
/// low nibble = even `k`.
pub fn pack_byte(even_code: u8, odd_code: u8) -> u8 {
    (even_code & 0x0f) | ((odd_code & 0x0f) << 4)
}

/// The E2M1 codebook as raw f32 bit patterns — the device LUT's contents, so a
/// test can pin the CUDA `__constant__` table against this crate's table.
pub fn e2m1_codebook_bits() -> [u32; 16] {
    let mut out = [0u32; 16];
    let mut i = 0;
    while i < 16 {
        out[i] = E2M1_CODEBOOK[i].to_bits();
        i += 1;
    }
    out
}
