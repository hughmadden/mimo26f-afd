//! PNG decoder producing exactly what HF gets from Pillow 12:
//! `Image.open` (PngImagePlugin modes) -> `exif_transpose` -> `convert_to_rgb`.
//!
//! Pillow modes by (bit depth, color type) and how each reaches RGB:
//!
//! | PNG            | Pillow mode | RGB                                             |
//! |----------------|-------------|-------------------------------------------------|
//! | gray 1         | `1`         | 0/255; tRNS: `255 if t else 0` -> alpha 0         |
//! | gray 2/4/8     | `L`         | v*85 / v*17 / v; tRNS: L == t & 0xff -> alpha 0  |
//! | gray 16        | `I;16`      | min(v, 255) (clip!); tRNS as above on the clip  |
//! | RGB 8/16       | `RGB`       | returned as-is (16: high byte); tRNS ignored    |
//! | palette 1..8   | `P`         | PLTE (missing entries or PLTE: opaque black) + tRNS alpha |
//! | gray+alpha 8   | `LA`        | (L,L,L,A) over white                            |
//! | gray+alpha 16  | `RGBA`      | high bytes, over white                          |
//! | RGBA 8/16      | `RGBA`      | high bytes (16), over white                     |
//!
//! "Over white" is `Image.alpha_composite(white, im.convert("RGBA"))`
//! (AlphaComposite.c integer math). Chunk CRCs are not verified.

use crate::exif;
use crate::inflate;
use crate::{bail, ImageError, RgbImage, MAX_DECODE_PIXELS};
use std::sync::OnceLock;

pub(crate) const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

#[derive(Clone, Copy)]
struct Header {
    width: u32,
    height: u32,
    depth: u8,
    ctype: u8,
    interlaced: bool,
}

/// tRNS as Pillow stores it in `info["transparency"]` for the IHDR mode.
enum Trns {
    /// Mode `1`/`L`/`I;16`: pixels whose 8-bit value equals this become transparent.
    Gray(u8),
    /// Mode `P`: per-entry alpha.
    Palette(Vec<u8>),
}

fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

fn chunk_type_ok(t: &[u8]) -> bool {
    t.iter().all(|&c| c.is_ascii_alphanumeric() || c == b'_')
}

pub(crate) fn decode(data: &[u8]) -> Result<RgbImage, ImageError> {
    let mut pos = SIGNATURE.len();
    let mut hdr: Option<Header> = None;
    let mut plte: Option<&[u8]> = None;
    let mut trns: Option<Trns> = None;
    let mut exif_blob: Option<&[u8]> = None;
    let mut idat: Vec<u8> = Vec::new();
    let mut found_idat = false;

    // Chunks up to the first IDAT (all must be complete, like Pillow's _open).
    loop {
        let Some(head) = data.get(pos..pos + 8) else {
            bail!("truncated PNG (no image data)");
        };
        let len = be32(head) as usize;
        let ty = &head[4..8];
        if !chunk_type_ok(ty) {
            bail!("broken PNG file (invalid chunk type)");
        }
        let body_start = pos + 8;
        if ty == b"IDAT" {
            found_idat = true;
            break;
        }
        let Some(body) = data.get(body_start..body_start.saturating_add(len)) else {
            bail!(
                "truncated PNG (chunk {} runs past end of file)",
                String::from_utf8_lossy(ty)
            );
        };
        if data.len() < body_start + len + 4 {
            bail!(
                "truncated PNG (chunk {} has no CRC)",
                String::from_utf8_lossy(ty)
            );
        }
        match ty {
            b"IHDR" => {
                if body.len() < 13 {
                    bail!("truncated PNG IHDR chunk");
                }
                if body[11] != 0 {
                    bail!("PNG uses an unknown filter method {}", body[11]);
                }
                hdr = Some(Header {
                    width: be32(&body[0..4]),
                    height: be32(&body[4..8]),
                    depth: body[8],
                    ctype: body[9],
                    interlaced: body[12] != 0,
                });
            }
            b"PLTE" => {
                if hdr.is_some_and(|h| h.ctype == 3) {
                    plte = Some(body);
                }
            }
            b"tRNS" => {
                if let Some(h) = hdr {
                    trns = parse_trns(h, body)?;
                }
            }
            b"eXIf" => exif_blob = Some(body),
            b"IEND" => break,
            _ => {}
        }
        pos = body_start + len + 4;
    }
    if !found_idat {
        bail!("PNG has no image data (IEND before IDAT)");
    }
    let Some(h) = hdr else {
        bail!("PNG has no IHDR chunk before the image data");
    };

    // The first run of consecutive IDAT chunks (a truncated last chunk
    // contributes what is present; the inflater decides if that suffices).
    while let Some(head) = data.get(pos..pos + 8) {
        if &head[4..8] != b"IDAT" {
            break;
        }
        let len = be32(head) as usize;
        let start = pos + 8;
        let end = start.saturating_add(len).min(data.len());
        idat.extend_from_slice(&data[start..end]);
        pos = start.saturating_add(len).saturating_add(4);
    }
    // After the image data Pillow reads remaining chunks on load; a later
    // eXIf replaces an earlier one. Stop quietly at anything malformed.
    while let Some(head) = data.get(pos..pos + 8) {
        let len = be32(head) as usize;
        let ty = &head[4..8];
        if !chunk_type_ok(ty) || ty == b"IEND" {
            break;
        }
        let start = pos + 8;
        let Some(body) = data.get(start..start.saturating_add(len)) else {
            break;
        };
        if ty == b"eXIf" {
            exif_blob = Some(body);
        }
        pos = start + len + 4;
    }

    let img = decode_image(h, &idat, plte, trns.as_ref())?;
    let o = exif::orientation(exif_blob, None);
    Ok(exif::apply(img, o))
}

fn parse_trns(h: Header, body: &[u8]) -> Result<Option<Trns>, ImageError> {
    let i16be = |o: usize| -> Result<u16, ImageError> {
        match body.get(o..o + 2) {
            Some(b) => Ok(u16::from_be_bytes([b[0], b[1]])),
            None => bail!("PNG tRNS chunk is too short"),
        }
    };
    Ok(match (h.ctype, h.depth) {
        (3, _) => {
            // Pillow: a tRNS of all 0xFF except one 0x00 becomes an int index
            // (putpalettealpha); anything else is putpalettealphas(bytes),
            // which rejects more than 256 entries.
            let zeros = body.iter().filter(|&&b| b == 0).count();
            let simple = zeros == 1 && body.iter().all(|&b| b == 0 || b == 0xFF);
            if simple {
                let i = body.iter().position(|&b| b == 0).unwrap_or(0);
                if i >= 256 {
                    bail!("PNG tRNS palette index out of range");
                }
                let mut alpha = vec![0xFF; i + 1];
                alpha[i] = 0;
                Some(Trns::Palette(alpha))
            } else if body.len() > 256 {
                bail!("PNG tRNS chunk has more than 256 palette entries");
            } else {
                Some(Trns::Palette(body.to_vec()))
            }
        }
        (0, 1) => Some(Trns::Gray(if i16be(0)? != 0 { 255 } else { 0 })),
        (0, _) => Some(Trns::Gray(i16be(0)? as u8)),
        (2, _) => {
            // Mode RGB stores (r, g, b) but convert_to_rgb returns RGB as-is;
            // only the length check (Pillow's i16 reads) matters.
            i16be(4)?;
            None
        }
        _ => None,
    })
}

fn channels(ctype: u8) -> usize {
    match ctype {
        0 | 3 => 1,
        4 => 2,
        2 => 3,
        _ => 4,
    }
}

fn decode_image(
    h: Header,
    idat: &[u8],
    plte: Option<&[u8]>,
    trns: Option<&Trns>,
) -> Result<RgbImage, ImageError> {
    let valid = matches!(
        (h.ctype, h.depth),
        (0, 1 | 2 | 4 | 8 | 16) | (2, 8 | 16) | (3, 1 | 2 | 4 | 8) | (4, 8 | 16) | (6, 8 | 16)
    );
    if !valid {
        bail!(
            "unsupported PNG bit depth {} for color type {}",
            h.depth,
            h.ctype
        );
    }
    if h.width == 0 || h.height == 0 {
        bail!("PNG has zero width or height ({}x{})", h.width, h.height);
    }
    let pixels = h.width as u64 * h.height as u64;
    if pixels > MAX_DECODE_PIXELS {
        bail!(
            "PNG is too large: {}x{} exceeds the {} pixel limit",
            h.width,
            h.height,
            MAX_DECODE_PIXELS
        );
    }
    let (w, ht) = (h.width as usize, h.height as usize);
    let ch = channels(h.ctype);
    let bits = h.depth as usize * ch;
    let row_bytes = |width: usize| (width * bits).div_ceil(8);
    let fbpp = (bits / 8).max(1);

    let passes: Vec<(usize, usize, usize, usize)> = if h.interlaced {
        [
            (0, 0, 8, 8),
            (4, 0, 8, 8),
            (0, 4, 4, 8),
            (2, 0, 4, 4),
            (0, 2, 2, 4),
            (1, 0, 2, 2),
            (0, 1, 1, 2),
        ]
        .to_vec()
    } else {
        vec![(0, 0, 1, 1)]
    };
    let pass_dims = |&(x0, y0, dx, dy): &(usize, usize, usize, usize)| {
        let pw = if w > x0 { (w - x0).div_ceil(dx) } else { 0 };
        let ph = if ht > y0 { (ht - y0).div_ceil(dy) } else { 0 };
        (pw, ph)
    };
    let mut expected = 0usize;
    for p in &passes {
        let (pw, ph) = pass_dims(p);
        if pw > 0 && ph > 0 {
            expected += ph * (1 + row_bytes(pw));
        }
    }
    let mut raw = inflate::zlib_decompress(idat, expected)?;

    // Canonical 8-bit samples per pixel (see module docs), `ch` per pixel.
    let mut canon = vec![0u8; w * ht * ch];
    let mut off = 0usize;
    let mut line = Vec::new();
    for p in &passes {
        let (pw, ph) = pass_dims(p);
        if pw == 0 || ph == 0 {
            continue;
        }
        let rb = row_bytes(pw);
        let stride = rb + 1;
        let (x0, y0, dx, dy) = *p;
        for r in 0..ph {
            let (before, rest) = raw.split_at_mut(off);
            let prev = if r == 0 {
                None
            } else {
                Some(&before[off - rb..off])
            };
            let (filter, cur) = rest[..stride].split_first_mut().expect("stride >= 1");
            unfilter(*filter, cur, prev, fbpp)?;
            let y = y0 + r * dy;
            if !h.interlaced {
                let dst = &mut canon[y * w * ch..(y + 1) * w * ch];
                canonicalize(h, cur, pw, dst);
            } else {
                line.resize(pw * ch, 0);
                canonicalize(h, cur, pw, &mut line);
                for i in 0..pw {
                    let x = x0 + i * dx;
                    let d = (y * w + x) * ch;
                    canon[d..d + ch].copy_from_slice(&line[i * ch..(i + 1) * ch]);
                }
            }
            off += stride;
        }
    }
    drop(raw);

    let data = to_rgb(h, canon, plte, trns);
    Ok(RgbImage {
        width: h.width,
        height: h.height,
        data,
    })
}

/// Undo one PNG filter in place (`prev` = previous unfiltered row of the pass).
fn unfilter(filter: u8, cur: &mut [u8], prev: Option<&[u8]>, bpp: usize) -> Result<(), ImageError> {
    let n = cur.len();
    match (filter, prev) {
        (0, _) => {}
        (1, _) => {
            for i in bpp..n {
                cur[i] = cur[i].wrapping_add(cur[i - bpp]);
            }
        }
        (2, None) => {}
        (2, Some(up)) => {
            for (c, &u) in cur.iter_mut().zip(up) {
                *c = c.wrapping_add(u);
            }
        }
        (3, None) => {
            for i in bpp..n {
                cur[i] = cur[i].wrapping_add(cur[i - bpp] / 2);
            }
        }
        (3, Some(up)) => {
            for i in 0..bpp.min(n) {
                cur[i] = cur[i].wrapping_add(up[i] / 2);
            }
            for i in bpp..n {
                let avg = ((cur[i - bpp] as u16 + up[i] as u16) / 2) as u8;
                cur[i] = cur[i].wrapping_add(avg);
            }
        }
        (4, None) => {
            // Paeth with b = c = 0 reduces to Sub.
            for i in bpp..n {
                cur[i] = cur[i].wrapping_add(cur[i - bpp]);
            }
        }
        (4, Some(up)) => {
            for i in 0..bpp.min(n) {
                cur[i] = cur[i].wrapping_add(up[i]);
            }
            for i in bpp..n {
                let a = cur[i - bpp] as i16;
                let b = up[i] as i16;
                let c = up[i - bpp] as i16;
                let pa = (b - c).abs();
                let pb = (a - c).abs();
                let pc = (a + b - 2 * c).abs();
                let pred = if pa <= pb && pa <= pc {
                    a
                } else if pb <= pc {
                    b
                } else {
                    c
                };
                cur[i] = cur[i].wrapping_add(pred as u8);
            }
        }
        _ => bail!("PNG row uses unknown filter type {filter}"),
    }
    Ok(())
}

/// Unpack one unfiltered row of `width` pixels into canonical 8-bit samples.
fn canonicalize(h: Header, row: &[u8], width: usize, dst: &mut [u8]) {
    match (h.ctype, h.depth) {
        (_, 8) => dst.copy_from_slice(&row[..dst.len()]),
        (0, 16) => {
            // Pillow I;16 -> L/RGB clips (does not scale).
            for (d, s) in dst.iter_mut().zip(row.chunks_exact(2)) {
                *d = if s[0] != 0 { 255 } else { s[1] };
            }
        }
        (_, 16) => {
            // RGB;16B / RGBA;16B / LA;16B unpackers keep the high byte.
            for (d, s) in dst.iter_mut().zip(row.chunks_exact(2)) {
                *d = s[0];
            }
        }
        (ctype, depth) => {
            // 1/2/4-bit gray or palette, MSB first.
            let depth = depth as usize;
            let per_byte = 8 / depth;
            let mask = (1u8 << depth) - 1;
            let scale: u8 = match (ctype, depth) {
                (0, 1) => 255,
                (0, 2) => 85,
                (0, 4) => 17,
                _ => 1, // palette indices stay raw
            };
            for (x, d) in dst.iter_mut().enumerate().take(width) {
                let byte = row[x / per_byte];
                let shift = 8 - depth * (x % per_byte + 1);
                *d = ((byte >> shift) & mask) * scale;
            }
        }
    }
}

/// `alpha_composite(white, rgba).convert("RGB")` for one channel value `c`
/// with alpha `a` (AlphaComposite.c, PRECISION_BITS 7, dst = 255/255).
fn over_white_exact(c: u8, a: u8) -> u8 {
    if a == 0 {
        return 255;
    }
    let (c, a) = (c as u32, a as u32);
    let blend = 255 * (255 - a);
    let outa255 = a * 255 + blend;
    let coef1 = a * 255 * 255 * (1 << 7) / outa255;
    let coef2 = 255 * (1 << 7) - coef1;
    let tmp = c * coef1 + 255 * coef2 + (0x80 << 7);
    ((((tmp >> 8) + tmp) >> 8) >> 7) as u8
}

/// 64 KiB table: `[alpha][value]`.
fn over_white_table() -> &'static [u8] {
    static T: OnceLock<Vec<u8>> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = vec![0u8; 256 * 256];
        for a in 0..256 {
            for c in 0..256 {
                t[a * 256 + c] = over_white_exact(c as u8, a as u8);
            }
        }
        t
    })
}

fn to_rgb(h: Header, canon: Vec<u8>, plte: Option<&[u8]>, trns: Option<&Trns>) -> Vec<u8> {
    let ow = over_white_table();
    let px = h.width as usize * h.height as usize;
    match h.ctype {
        2 => canon, // mode RGB: convert_to_rgb returns it unchanged
        6 => {
            let mut out = Vec::with_capacity(px * 3);
            for p in canon.chunks_exact(4) {
                let t = &ow[p[3] as usize * 256..][..256];
                out.extend_from_slice(&[t[p[0] as usize], t[p[1] as usize], t[p[2] as usize]]);
            }
            out
        }
        4 => {
            let mut out = Vec::with_capacity(px * 3);
            for p in canon.chunks_exact(2) {
                let v = ow[p[1] as usize * 256 + p[0] as usize];
                out.extend_from_slice(&[v, v, v]);
            }
            out
        }
        3 => {
            // Pillow palette: 256 entries, unset ones opaque black; tRNS sets
            // the alpha of its first len entries.
            let mut pal = [[0u8, 0, 0, 255]; 256];
            if let Some(p) = plte {
                for (i, rgb) in p.chunks_exact(3).take(256).enumerate() {
                    pal[i] = [rgb[0], rgb[1], rgb[2], 255];
                }
            }
            if let Some(Trns::Palette(alpha)) = trns {
                for (i, &a) in alpha.iter().take(256).enumerate() {
                    pal[i][3] = a;
                }
            }
            let rgb: Vec<[u8; 3]> = pal
                .iter()
                .map(|e| {
                    let t = &ow[e[3] as usize * 256..][..256];
                    [t[e[0] as usize], t[e[1] as usize], t[e[2] as usize]]
                })
                .collect();
            let mut out = Vec::with_capacity(px * 3);
            for &i in &canon {
                out.extend_from_slice(&rgb[i as usize]);
            }
            out
        }
        _ => {
            let key = match trns {
                Some(Trns::Gray(t)) => Some(*t),
                _ => None,
            };
            let mut out = Vec::with_capacity(px * 3);
            for &v in &canon {
                let v = if key == Some(v) { 255 } else { v };
                out.extend_from_slice(&[v, v, v]);
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn over_white_endpoints() {
        for c in 0..=255u8 {
            assert_eq!(over_white_exact(c, 255), c);
            assert_eq!(over_white_exact(c, 0), 255);
        }
        // Spot values printed by Pillow 12.2 alpha_composite over white.
        assert_eq!(over_white_exact(10, 128), 132);
        assert_eq!(over_white_exact(40, 7), 249);
    }
}
