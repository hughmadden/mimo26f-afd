//! HF `Qwen2VLImageProcessor._preprocess` (slow/PIL path) for one RGB image:
//! smart_resize -> PIL BICUBIC resize -> rescale -> normalize -> patchify.

use crate::{
    bail, resize_bicubic, smart_resize, ImageError, Patches, RgbImage, IMAGE_MEAN, IMAGE_STD,
    MERGE, PATCH, PATCH_DIM, TEMPORAL,
};

/// The exact float HF produces for 8-bit value `v` of channel `c`:
///
/// ```text
/// x = float32(float64(v) * (1/255))          # rescale: astype(f64) * scale, astype(f32)
/// y = (x - float32(mean[c])) / float32(std[c])  # normalize in float32
/// ```
///
/// (`transformers.image_transforms.rescale` / `normalize`; numpy evaluates
/// the subtraction and division in IEEE float32, as Rust does.)
pub fn normalize_value(c: usize, v: u8) -> f32 {
    let x = (v as f64 * (1.0f64 / 255.0)) as f32;
    (x - IMAGE_MEAN[c] as f32) / IMAGE_STD[c] as f32
}

fn lut() -> [[f32; 256]; 3] {
    let mut t = [[0f32; 256]; 3];
    for (c, row) in t.iter_mut().enumerate() {
        for (v, d) in row.iter_mut().enumerate() {
            *d = normalize_value(c, v as u8);
        }
    }
    t
}

pub(crate) fn preprocess(img: &RgbImage) -> Result<Patches, ImageError> {
    let (w, h) = (img.width as usize, img.height as usize);
    if img.data.len() != w * h * 3 {
        bail!(
            "RGB buffer length {} does not match {}x{}x3",
            img.data.len(),
            img.width,
            img.height
        );
    }
    let (rh, rw) = smart_resize(img.height, img.width)?;
    let resized;
    let src: &RgbImage = if rw == img.width && rh == img.height {
        img
    } else {
        resized = resize_bicubic(img, rw, rh);
        &resized
    };
    let (rw, rh) = (rw as usize, rh as usize);
    let gh = rh / PATCH;
    let gw = rw / PATCH;
    let lut = lut();
    let mut data = vec![0f32; gh * gw * PATCH_DIM];
    let plane = PATCH * PATCH; // floats per (channel, temporal) slice
    let mut row = 0usize;
    for bh in 0..gh / MERGE {
        for bw in 0..gw / MERGE {
            for mh in 0..MERGE {
                for mw in 0..MERGE {
                    let py0 = (bh * MERGE + mh) * PATCH;
                    let px0 = (bw * MERGE + mw) * PATCH;
                    let dst = &mut data[row * PATCH_DIM..(row + 1) * PATCH_DIM];
                    for py in 0..PATCH {
                        let s = ((py0 + py) * rw + px0) * 3;
                        let pixels = &src.data[s..s + PATCH * 3];
                        for (c, table) in lut.iter().enumerate() {
                            let base = c * TEMPORAL * plane + py * PATCH;
                            let (t0, t1) = dst[base..base + plane + PATCH].split_at_mut(plane);
                            for (px, p) in pixels.chunks_exact(3).enumerate() {
                                let v = table[p[c] as usize];
                                t0[px] = v;
                                t1[px] = v;
                            }
                        }
                    }
                    row += 1;
                }
            }
        }
    }
    Ok(Patches {
        grid_t: 1,
        grid_h: gh as u32,
        grid_w: gw as u32,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_endpoints() {
        // (0 - mean) / std and (1 - mean) / std in float32.
        for c in 0..3 {
            let m = IMAGE_MEAN[c] as f32;
            let s = IMAGE_STD[c] as f32;
            assert_eq!(normalize_value(c, 0), (0f32 - m) / s);
            assert_eq!(normalize_value(c, 255), (1f32 - m) / s);
        }
    }

    #[test]
    fn patch_layout_small() {
        // 32x32 image (one merged unit, 4 patches); pixel value encodes x, y.
        let mut data = vec![0u8; 32 * 32 * 3];
        for y in 0..32 {
            for x in 0..32 {
                let p = (y * 32 + x) * 3;
                data[p] = x as u8;
                data[p + 1] = y as u8;
                data[p + 2] = 200;
            }
        }
        // smart_resize(32, 32) -> 64x64, so check the layout on the resized copy.
        let img = RgbImage {
            width: 32,
            height: 32,
            data,
        };
        let p = preprocess(&img).unwrap();
        assert_eq!((p.grid_t, p.grid_h, p.grid_w), (1, 4, 4));
        assert_eq!(p.data.len(), 16 * PATCH_DIM);
        let r = resize_bicubic(&img, 64, 64);
        // Row 5 = merged unit (0, 1), patch (dy=0, dx=1): pixels y 0..16, x 48..64.
        let row = &p.data[5 * PATCH_DIM..6 * PATCH_DIM];
        for c in 0..3 {
            for t in 0..2 {
                for py in 0..16 {
                    for px in 0..16 {
                        let v = r.data[(py * 64 + 48 + px) * 3 + c];
                        let got = row[c * 512 + t * 256 + py * 16 + px];
                        assert_eq!(got, normalize_value(c, v));
                    }
                }
            }
        }
    }
}
