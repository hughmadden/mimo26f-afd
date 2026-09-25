//! Pillow `Image.resize(size, Image.BICUBIC)` for 8-bit RGB (Resample.c).
//!
//! Coefficients are computed in f64 with Pillow's exact operation order
//! (`precompute_coeffs`), normalized per output pixel, then converted to
//! 22-bit fixed point (`PRECISION_BITS = 32 - 8 - 2`) rounding half away from
//! zero. The horizontal pass runs first over only the input rows the vertical
//! pass needs; each pass rounds (`+ 1 << 21`) and clips to 0..255. A pass is
//! skipped when its dimension is unchanged, as in `ImagingResampleInner`.

use crate::RgbImage;

const PRECISION_BITS: u32 = 32 - 8 - 2;

fn bicubic(x: f64) -> f64 {
    // Pillow bicubic_filter with a = -0.5.
    const A: f64 = -0.5;
    let x = if x < 0.0 { -x } else { x };
    if x < 1.0 {
        return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    }
    if x < 2.0 {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * A;
    }
    0.0
}

/// Per-output-pixel taps: first input index, fixed-point weights.
struct Coeffs {
    ksize: usize,
    bounds: Vec<(usize, usize)>, // (xmin, count)
    k: Vec<i32>,                 // out_size * ksize
}

fn precompute(in_size: usize, out_size: usize) -> Coeffs {
    const SUPPORT: f64 = 2.0;
    // box = (0, 0, in_w, in_h) as C floats; (in1 - in0) promoted to double.
    let in0 = 0f32;
    let in1 = in_size as f32;
    let scale = (in1 - in0) as f64 / out_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = SUPPORT * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut k = vec![0i32; out_size * ksize];
    let mut w = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = in0 as f64 + (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // C casts truncate toward zero.
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64);
        let count = (xmax - xmin).max(0) as usize;
        let count = count.min(ksize);
        let mut ww = 0.0f64;
        for (x, wx) in w.iter_mut().enumerate().take(count) {
            let v = bicubic(((x as i64 + xmin) as f64 - center + 0.5) * ss);
            *wx = v;
            ww += v;
        }
        let row = &mut k[xx * ksize..(xx + 1) * ksize];
        for (x, dst) in row.iter_mut().enumerate().take(count) {
            let mut v = w[x];
            if ww != 0.0 {
                v /= ww;
            }
            *dst = if v < 0.0 {
                (-0.5 + v * (1u32 << PRECISION_BITS) as f64) as i32
            } else {
                (0.5 + v * (1u32 << PRECISION_BITS) as f64) as i32
            };
        }
        bounds.push((xmin as usize, count));
    }
    Coeffs { ksize, bounds, k }
}

#[inline(always)]
fn clip8(v: i32) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

pub(crate) fn resize_bicubic(img: &RgbImage, width: u32, height: u32) -> RgbImage {
    let (iw, ih) = (img.width as usize, img.height as usize);
    let (ow, oh) = (width as usize, height as usize);
    if ow == 0 || oh == 0 {
        return RgbImage {
            width,
            height,
            data: Vec::new(),
        };
    }
    if iw == 0 || ih == 0 {
        // No input taps: Pillow's accumulator stays at the rounding bias -> 0.
        return RgbImage {
            width,
            height,
            data: vec![0; ow * oh * 3],
        };
    }
    // Tolerate an inconsistent buffer (public fields) without panicking.
    let padded;
    let src: &[u8] = if img.data.len() >= iw * ih * 3 {
        &img.data[..iw * ih * 3]
    } else {
        let mut v = img.data.clone();
        v.resize(iw * ih * 3, 0);
        padded = v;
        &padded
    };
    if iw == ow && ih == oh {
        return RgbImage {
            width,
            height,
            data: src.to_vec(),
        };
    }
    let need_h = ow != iw;
    let need_v = oh != ih;
    let cv = precompute(ih, oh);
    let ybox_first = cv.bounds[0].0;
    let (lx, lc) = cv.bounds[oh - 1];
    let ybox_last = lx + lc;

    // Horizontal pass over input rows [ybox_first, ybox_last).
    let (tmp, tmp_w, row_off) = if need_h {
        let ch = precompute(iw, ow);
        let rows = ybox_last - ybox_first;
        let mut t = vec![0u8; ow * rows * 3];
        for y in 0..rows {
            let s = &src[(y + ybox_first) * iw * 3..(y + ybox_first + 1) * iw * 3];
            let d = &mut t[y * ow * 3..(y + 1) * ow * 3];
            horizontal_row(s, d, &ch);
        }
        (t, ow, ybox_first)
    } else {
        (Vec::new(), iw, 0)
    };
    let hsrc: &[u8] = if need_h { &tmp } else { src };
    if !need_v {
        return RgbImage {
            width,
            height,
            data: hsrc.to_vec(),
        };
    }
    let mut out = vec![0u8; tmp_w * oh * 3];
    let rowlen = tmp_w * 3;
    let mut acc = vec![0i32; rowlen];
    for yy in 0..oh {
        let (ymin, cnt) = cv.bounds[yy];
        let ymin = ymin - row_off;
        let kk = &cv.k[yy * cv.ksize..yy * cv.ksize + cnt];
        acc.fill(1 << (PRECISION_BITS - 1));
        for (j, &kw) in kk.iter().enumerate() {
            let r = &hsrc[(ymin + j) * rowlen..(ymin + j + 1) * rowlen];
            for (a, &p) in acc.iter_mut().zip(r) {
                *a = a.wrapping_add((p as i32).wrapping_mul(kw));
            }
        }
        for (d, &a) in out[yy * rowlen..(yy + 1) * rowlen].iter_mut().zip(&acc) {
            *d = clip8(a);
        }
    }
    RgbImage {
        width,
        height,
        data: out,
    }
}

fn horizontal_row(s: &[u8], d: &mut [u8], c: &Coeffs) {
    for (xx, px) in d.chunks_exact_mut(3).enumerate() {
        let (xmin, cnt) = c.bounds[xx];
        let kk = &c.k[xx * c.ksize..xx * c.ksize + cnt];
        let base = 1i32 << (PRECISION_BITS - 1);
        let (mut s0, mut s1, mut s2) = (base, base, base);
        for (x, &kw) in kk.iter().enumerate() {
            let p = (xmin + x) * 3;
            s0 = s0.wrapping_add((s[p] as i32).wrapping_mul(kw));
            s1 = s1.wrapping_add((s[p + 1] as i32).wrapping_mul(kw));
            s2 = s2.wrapping_add((s[p + 2] as i32).wrapping_mul(kw));
        }
        px[0] = clip8(s0);
        px[1] = clip8(s1);
        px[2] = clip8(s2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_degenerate_sizes() {
        let img = RgbImage {
            width: 2,
            height: 1,
            data: vec![1, 2, 3, 4, 5, 6],
        };
        assert_eq!(resize_bicubic(&img, 2, 1), img);
        assert!(resize_bicubic(&img, 0, 5).data.is_empty());
        let one = RgbImage {
            width: 1,
            height: 1,
            data: vec![9, 8, 7],
        };
        let up = resize_bicubic(&one, 3, 2);
        assert_eq!(up.data, [9, 8, 7].repeat(6));
    }

    #[test]
    fn weights_sum_to_one_in_fixed_point() {
        for (i, o) in [(10, 3), (3, 10), (1000, 999), (7, 1), (1, 7)] {
            let c = precompute(i, o);
            for xx in 0..o {
                let (_, cnt) = c.bounds[xx];
                let s: i64 = c.k[xx * c.ksize..xx * c.ksize + cnt]
                    .iter()
                    .map(|&v| v as i64)
                    .sum();
                assert!(
                    (s - (1 << PRECISION_BITS)).abs() <= cnt as i64,
                    "{i}->{o} px {xx}: {s}"
                );
            }
        }
    }
}
