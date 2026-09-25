//! FP8 E4M3 block-128 codec (allowlist family: FP8 block-128 — planned
//! "COPY family" row is for the native kernel side; this is the CPU numerics
//! the loader dequantizes with, ported from `spike/quant.py:27-100` which
//! mirrors `mimo26/quant/fp8_block.py`).
//!
//! Golden pin: `code/tests/golden/e4m3_decode_table.json` +
//! `fp8_block_golden.json` (read-only); spike twin pins the same table in
//! `spike/tests/test_p103_codecs.py::test_e4m3_decode_table_golden_pin`.
//!
//! e4m3fn: byte `[s eeee mmm]`, bias 7, no infinities, `S.1111.111` = NaN.

use crate::{LoadError, Mat};

pub const E4M3_MAX: f64 = 448.0;

/// `spike/quant.py:28-41` — decode one code byte. Table magnitudes are exact
/// in f32; the multiplies accumulate in f64 and cast to f32 at the end, mirroring
/// the numpy codec bit-for-bit (`decode_e4m3` there returns f64).
pub fn decode_e4m3(code: u8) -> f64 {
    if code == 0x7F || code == 0xFF {
        return f64::NAN; // S.1111.111
    }
    let s = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((code >> 3) & 0x0F);
    let m = f64::from(code & 0x07);
    let val = if e == 0 {
        (m / 8.0) * 2f64.powi(-6) // subnormal
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    s * val
}

/// Full 256-code decode table (NaN for `0x7f`/`0xff`), cached once. The Spark
/// serve path decodes `[tokens, 4096]` hidden rows per request, so the per-element
/// `decode_e4m3` (which calls `2f64.powi` on every element) was the hot spot;
/// a table lookup is O(1) and bit-identical (computed with the same `decode_e4m3`).
pub fn decode_table() -> &'static [f64] {
    use std::sync::OnceLock;
    static T: OnceLock<Vec<f64>> = OnceLock::new();
    T.get_or_init(|| (0u16..=255).map(|c| decode_e4m3(c as u8)).collect())
}

/// `spike/quant.py:52-67` — nearest-representable encoding (search the decode
/// table's monotone magnitudes; ties go to the larger magnitude, as there).
/// O(1) bit-level: the E4M3 magnitudes are monotone in `code`, so the nearest
/// code is read directly from the f64's IEEE-754 exponent/mantissa bits with
/// round-half-up (which is exactly the linear scan's "tie prefers the larger
/// code"). No `log2`/`powi` in the hot path.
pub fn encode_e4m3(value: f64) -> u8 {
    let v = if value.is_nan() { 0.0 } else { value }; // nan_to_num(nan=0)
    let mag = v.abs().min(E4M3_MAX);
    let sign: u8 = if v < 0.0 { 0x80 } else { 0 };
    if mag == 0.0 {
        return sign; // +0/-0 encode to code 0
    }
    let bits = mag.to_bits();
    let exp = ((bits >> 52) & 0x7FF) as i32;
    let mant = bits & 0x000F_FFFF_FFFF_FFFF;
    // Subnormal region: mag < 2^-6 (exp < 1017). m = round(mag * 2^9); m == 0
    // encodes to zero (0x00), m in [1, 7] to the subnormal codes, and m == 8
    // rounds up to the FIRST normal code (e=1,m=0 == 2^-6) — the linear scan's
    // tie-break (larger magnitude) picks that code, not the subnormal clamp.
    if exp < 1017 {
        let m = (mag * 512.0).round() as u8;
        return if m >= 8 { sign | 0x08 } else { sign | m };
    }
    // Normal: mag = (1 + mant/2^52) * 2^(exp-1023). E4M3: e = exp-1016;
    // m = round(mant / 2^49) = (mant + 2^48) >> 49.
    let e = exp - 1016;
    let m = ((mant + 0x0001_0000_0000_0000u64) >> 49) as i32;
    let (e, m) = if m >= 8 {
        (e + 1, 0) // mantissa round-up carries into the exponent
    } else {
        (e, m)
    };
    let e = e.clamp(1, 15) as u8;
    // m in [0, 7]; e=15,m=7 is NaN but never arises (mag <= 448 == e=15,m=6).
    let m = m.clamp(0, 7) as u8;
    sign | (e << 3) | m
}

/// `spike/quant.py:86-100` — quantize per `(br, bc)` block → (codes, scale_inv).
pub fn quantize_block(
    w: &Mat<f64>,
    block: (usize, usize),
) -> Result<(Mat<u8>, Mat<f32>), LoadError> {
    let (br, bc) = block;
    let rb = crate::ceil_div(w.rows, br);
    let cb = crate::ceil_div(w.cols, bc);
    let mut codes = Mat::<u8>::zeros(w.rows, w.cols);
    let mut scale = Mat::<f32>::zeros(rb, cb);
    for bi in 0..rb {
        for bj in 0..cb {
            let r_hi = (bi * br + br).min(w.rows);
            let c_hi = (bj * bc + bc).min(w.cols);
            let mut amax = 0f64;
            for r in bi * br..r_hi {
                for c in bj * bc..c_hi {
                    amax = amax.max(w.get(r, c).abs());
                }
            }
            let s = if amax > 0.0 { (amax / E4M3_MAX) as f32 } else { 1.0 };
            scale.set(bi, bj, s);
            for r in bi * br..r_hi {
                for c in bj * bc..c_hi {
                    codes.set(r, c, encode_e4m3(w.get(r, c) / f64::from(s)));
                }
            }
        }
    }
    Ok((codes, scale))
}

/// `spike/quant.py:70-83` — FP8 codes `[rows, cols]` + scale_inv
/// `[ceil(rows/br), ceil(cols/bc)]` → f32.
pub fn dequantize_block(
    codes: &Mat<u8>,
    scale_inv: &Mat<f32>,
    block: (usize, usize),
) -> Result<Mat<f32>, LoadError> {
    let (br, bc) = block;
    let rb = crate::ceil_div(codes.rows, br);
    let cb = crate::ceil_div(codes.cols, bc);
    if scale_inv.rows != rb || scale_inv.cols != cb {
        return Err(LoadError::ShapeMismatch {
            what: format!(
                "scale shape {}x{} != {rb}x{cb}",
                scale_inv.rows, scale_inv.cols
            ),
        });
    }
    let mut out = Mat::<f32>::zeros(codes.rows, codes.cols);
    for r in 0..codes.rows {
        for c in 0..codes.cols {
            let v = decode_e4m3(codes.get(r, c)) * f64::from(scale_inv.get(r / br, c / bc));
            out.set(r, c, v as f32);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    /// The b22b583 binary-search encoder, kept as the golden reference for the
    /// O(1) encoder (builder R27: reference, not the linear scan). Same
    /// tie-break as the original 256-code scan (ties prefer the larger code).
    fn encode_e4m3_reference(value: f64) -> u8 {
        let v = if value.is_nan() { 0.0 } else { value }; // nan_to_num(nan=0)
        let mag = v.abs().min(E4M3_MAX);
        let sign: u8 = if v < 0.0 { 0x80 } else { 0 };
        static TAB: OnceLock<Vec<f64>> = OnceLock::new();
        let tab = TAB.get_or_init(|| (0..0x7Fu8).map(|c| decode_e4m3(c).abs()).collect());
        // First index with tab[i] >= mag (mag <= 448 == tab[0x7e], so hi <= 0x7e).
        let hi = tab.partition_point(|&x| x < mag);
        let best = if hi == 0 {
            0usize
        } else {
            let lo = hi - 1;
            let e_lo = (mag - tab[lo]).abs();
            let e_hi = (tab[hi] - mag).abs();
            if e_lo < e_hi { lo } else { hi } // tie -> larger code
        };
        ((best as u8) & 0x7F) | sign
    }

    #[test]
    fn fast_encode_matches_reference() {
        // Dense grid over the E4M3 range plus edge cases, positive and negative.
        for i in 0..200_000 {
            let x = (i as f64 / 199_999.0) * 448.0;
            for v in [x, -x, 448.0 - x, 0.5f64.powi((i % 40) as i32 - 20)] {
                assert_eq!(encode_e4m3(v), encode_e4m3_reference(v), "mismatch at {v:e}");
            }
        }
        // Exact code midpoints and near-zero.
        for c in 0..0x7Eu8 {
            let v = decode_e4m3(c).abs();
            assert_eq!(encode_e4m3(v), encode_e4m3_reference(v), "mismatch at code {c}");
            assert_eq!(encode_e4m3(v), c, "code {c} not its own encode");
        }
    }

    /// The subnormal→normal boundary (7/512 .. 2^-6) is a ~2e-3-wide window the
    /// dense 0..448 grid can step over; a dense sweep there pins the promote-to-
    /// normal rule (m rounds to 8 → code 0x08, not the subnormal clamp 0x07).
    /// This stays the fast merge-gate test; the exhaustive sweep is the ignored
    /// test below.
    #[test]
    fn encode_matches_reference_across_subnormal_normal_boundary() {
        let lo = 7.0f64 / 512.0;
        let hi = 2.0f64.powi(-6);
        for i in 0..200_000 {
            let x = lo + (hi - lo) * (i as f64 / 199_999.0);
            for v in [x, -x] {
                assert_eq!(encode_e4m3(v), encode_e4m3_reference(v), "boundary mismatch at {v:e}");
            }
        }
        // Exact midpoint (7.5/512) ties to the larger code (0x08), and the two
        // endpoints encode to their own codes.
        let mid = 7.5f64 / 512.0;
        assert_eq!(encode_e4m3(mid), 0x08, "midpoint must promote to e=1,m=0");
        assert_eq!(encode_e4m3(7.0f64 / 512.0), 0x07, "subnormal code 7 must be its own encode");
        assert_eq!(encode_e4m3(hi), 0x08, "2^-6 must encode to e=1,m=0");
    }

    /// Exhaustive golden (builder R27): every f32 bit pattern with |x| <= 448
    /// through the O(1) encoder and the binary-search reference — ~2.28e9
    /// inputs at O(log 256) each, minutes in release. Plus the specials the
    /// |x|<=448 filter excludes (±inf, NaN, the >448 saturation edge).
    ///
    /// Run once per encoder change:
    /// `cargo test --release -p mimo26-load -- --ignored exhaustive_encode_matches_reference --nocapture`
    #[test]
    #[ignore]
    fn exhaustive_encode_matches_reference() {
        let t0 = std::time::Instant::now();
        let mut checked: u64 = 0;
        for bits in 0u32..=u32::MAX {
            let x = f32::from_bits(bits);
            if x.abs() <= 448.0 {
                let v = f64::from(x);
                assert_eq!(
                    encode_e4m3(v),
                    encode_e4m3_reference(v),
                    "mismatch at bits {bits:#010x} (x={x:e})"
                );
                checked += 1;
            }
        }
        for v in [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::MAX,
            f64::MIN,
            448.0000001,
            464.0,
            480.0,
            512.0,
            1024.0,
        ] {
            assert_eq!(encode_e4m3(v), encode_e4m3_reference(v), "special mismatch at {v:e}");
        }
        eprintln!(
            "exhaustive encode: {checked} f32 inputs + 10 specials matched in {:?}",
            t0.elapsed()
        );
    }

    /// The GPU `m26::e4m3_encode` (kernels/include/mimo26_attn_device.cuh) uses an
    /// f32 `ilogbf`/`floorf`/`ldexpf` formulation; this replicates it and checks it
    /// is bit-identical to the f64 bit-level encoder, so a device-side KV store is
    /// numerics-preserving. Covers a dense grid + every E4M3 code midpoint + the
    /// subnormal→normal boundary.
    fn device_encode_replica(x: f32) -> u8 {
        let sign = if x < 0.0 { 0x80u8 } else { 0x00u8 };
        let mut mag = x.abs();
        if mag > 448.0 {
            mag = 448.0;
        }
        if mag == 0.0 {
            return sign;
        }
        // ilogbf for a normal f32 (E4M3's subnormal band is all f32 normals).
        let exp = ((mag.to_bits() >> 23) & 0xFF) as i32;
        let e = if exp == 0 { -127 } else { exp - 127 };
        if e <= -7 {
            let code = (mag * 512.0 + 0.5).floor() as i32;
            return sign | (code as u8);
        }
        let step = 2.0f32.powi(e - 3);
        let mut q = (mag / step + 0.5).floor() as i32;
        let mut ee = e + 7;
        if q >= 16 {
            q = 8;
            ee += 1;
        }
        let code = (ee << 3) | (q - 8);
        let code = if code > 0x7E { 0x7E } else { code };
        sign | (code as u8)
    }

    #[test]
    fn device_encode_matches_bit_level() {
        for i in 0..200_000 {
            let x = (i as f32 / 199_999.0) * 448.0;
            for v in [x, -x, 448.0 - x, 0.5f32.powi((i % 40) as i32 - 20)] {
                assert_eq!(
                    device_encode_replica(v),
                    encode_e4m3(v as f64),
                    "device replica mismatch at {v:e}"
                );
            }
        }
        for c in 0..0x7Eu8 {
            let v = decode_e4m3(c).abs() as f32;
            assert_eq!(device_encode_replica(v), encode_e4m3(v as f64), "code {c} mismatch");
        }
        // Subnormal→normal boundary (the window the dense grid can skip).
        let lo = 7.0f32 / 512.0;
        let hi = 2.0f32.powi(-6);
        for i in 0..100_000 {
            let x = lo + (hi - lo) * (i as f32 / 99_999.0);
            for v in [x, -x] {
                assert_eq!(device_encode_replica(v), encode_e4m3(v as f64), "boundary mismatch at {v:e}");
            }
        }
    }
}
