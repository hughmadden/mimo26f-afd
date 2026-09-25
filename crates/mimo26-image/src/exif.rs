//! EXIF / XMP orientation, following Pillow 12: `Image.getexif()` reads tag
//! 0x0112 from IFD0 of the Exif blob (falling back to the XMP
//! `tiff:Orientation` regex only when the tag is absent), and
//! `ImageOps.exif_transpose` maps orientations 2..8 to a transpose.
//!
//! Deviations (documented, error cases only): where Pillow raises on a
//! malformed Exif header (e.g. not a TIFF), we ignore the Exif instead; BigTIFF
//! Exif blobs are ignored.

use crate::RgbImage;

/// Tag 0x0112 as Pillow sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tag {
    /// Not in IFD0 (or no/unparseable Exif): the XMP fallback applies.
    Absent,
    /// Present with a value that never equals an int key (BYTE, ASCII,
    /// UNDEFINED, non-integral rationals/floats).
    Unusable,
    /// Present with this integral value.
    Value(i64),
}

/// Look up Orientation in an Exif blob (leading `Exif\0\0` headers stripped,
/// as `Image.Exif.load` does).
pub(crate) fn exif_tag(mut blob: &[u8]) -> Tag {
    while blob.starts_with(b"Exif\0\0") {
        blob = &blob[6..];
    }
    ifd0_orientation(blob).unwrap_or(Tag::Absent)
}

fn ifd0_orientation(t: &[u8]) -> Option<Tag> {
    let head = t.get(..8)?;
    let le = match &head[..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    // Pillow accepts the classic magic in either byte order (MM\0* II*\0 MM*\0
    // II\0*); 43 (BigTIFF) is not handled here.
    if !matches!((head[2], head[3]), (0x2A, 0x00) | (0x00, 0x2A)) {
        return None;
    }
    let u16_at = |o: usize| -> Option<u16> {
        let b = t.get(o..o + 2)?;
        Some(if le {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        })
    };
    let u32_at = |o: usize| -> Option<u32> {
        let b = t.get(o..o + 4)?;
        let a = [b[0], b[1], b[2], b[3]];
        Some(if le {
            u32::from_le_bytes(a)
        } else {
            u32::from_be_bytes(a)
        })
    };
    let ifd = u32_at(4)? as usize;
    let count = u16_at(ifd)? as usize;
    let mut result = Tag::Absent;
    for i in 0..count {
        let e = ifd + 2 + i * 12;
        // A truncated entry ends the IFD; entries read so far are kept.
        let Some(entry) = t.get(e..e + 12) else { break };
        let tag = u16_at(e)?;
        if tag != 0x0112 {
            continue;
        }
        let typ = u16_at(e + 2)?;
        let cnt = u32_at(e + 4)? as u64;
        let unit: u64 = match typ {
            1 | 2 | 6 | 7 => 1,
            3 | 8 => 2,
            4 | 9 | 11 | 13 => 4,
            5 | 10 | 12 | 16 | 17 | 18 => 8,
            _ => continue, // unsupported type: Pillow skips the entry
        };
        let size = cnt * unit;
        let data: &[u8] = if size > 4 {
            let off = u32_at(e + 8)? as u64;
            match t.get(off as usize..(off + size).min(usize::MAX as u64) as usize) {
                Some(d) if off + size <= t.len() as u64 => d,
                _ => continue, // short read: "Possibly corrupt EXIF data", tag skipped
            }
        } else {
            &entry[8..8 + size as usize]
        };
        if data.is_empty() {
            continue;
        }
        result = first_value(typ, data, le);
    }
    Some(result)
}

/// The first value of a tag (Pillow keeps `values[0]` for a length-1 tag).
fn first_value(typ: u16, d: &[u8], le: bool) -> Tag {
    let u16v = |o: usize| {
        let a = [d[o], d[o + 1]];
        if le {
            u16::from_le_bytes(a)
        } else {
            u16::from_be_bytes(a)
        }
    };
    let u32v = |o: usize| {
        let a = [d[o], d[o + 1], d[o + 2], d[o + 3]];
        if le {
            u32::from_le_bytes(a)
        } else {
            u32::from_be_bytes(a)
        }
    };
    let u64v = |o: usize| {
        let mut a = [0u8; 8];
        a.copy_from_slice(&d[o..o + 8]);
        if le {
            u64::from_le_bytes(a)
        } else {
            u64::from_be_bytes(a)
        }
    };
    let ratio = |n: i64, den: i64| {
        // IFDRational: den 0 is NaN; otherwise equal to an int iff exact.
        if den != 0 && n % den == 0 {
            Tag::Value(n / den)
        } else {
            Tag::Unusable
        }
    };
    let float = |v: f64| {
        if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e18 {
            Tag::Value(v as i64)
        } else {
            Tag::Unusable
        }
    };
    match typ {
        3 => Tag::Value(u16v(0) as i64),
        8 => Tag::Value(u16v(0) as i16 as i64),
        4 | 13 => Tag::Value(u32v(0) as i64),
        9 => Tag::Value(u32v(0) as i32 as i64),
        6 => Tag::Value(d[0] as i8 as i64),
        16 | 18 => i64::try_from(u64v(0))
            .map(Tag::Value)
            .unwrap_or(Tag::Unusable),
        17 => Tag::Value(u64v(0) as i64),
        5 => ratio(u32v(0) as i64, u32v(4) as i64),
        10 => ratio(u32v(0) as i32 as i64, u32v(4) as i32 as i64),
        11 => float(f32::from_bits(u32v(0)) as f64),
        12 => float(f64::from_bits(u64v(0))),
        _ => Tag::Unusable, // BYTE / ASCII / UNDEFINED hold bytes or str
    }
}

/// Pillow's XMP fallback: the first match of `tiff:Orientation(="|>)([0-9])`.
pub(crate) fn xmp_orientation(xmp: &[u8]) -> Option<i64> {
    const KEY: &[u8] = b"tiff:Orientation";
    let mut i = 0;
    while let Some(off) = find(&xmp[i..], KEY) {
        let rest = &xmp[i + off + KEY.len()..];
        let digit = match rest {
            [b'=', b'"', d, ..] if d.is_ascii_digit() => Some(*d),
            [b'>', d, ..] if d.is_ascii_digit() => Some(*d),
            _ => None,
        };
        if let Some(d) = digit {
            return Some((d - b'0') as i64);
        }
        i += off + 1;
    }
    None
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Resolve the orientation exactly as `exif_transpose` would read it.
pub(crate) fn orientation(exif: Option<&[u8]>, xmp: Option<&[u8]>) -> i64 {
    match exif.map(exif_tag).unwrap_or(Tag::Absent) {
        Tag::Value(v) => v,
        Tag::Unusable => 1,
        Tag::Absent => xmp.and_then(xmp_orientation).unwrap_or(1),
    }
}

/// Apply `ImageOps.exif_transpose` for `orientation` (2..8; others: no-op).
pub(crate) fn apply(img: RgbImage, orientation: i64) -> RgbImage {
    if !(2..=8).contains(&orientation) || img.width == 0 || img.height == 0 {
        return img;
    }
    let (w, h) = (img.width as usize, img.height as usize);
    let swap = orientation >= 5;
    let (ow, oh) = if swap { (h, w) } else { (w, h) };
    let mut out = vec![0u8; img.data.len()];
    let src = &img.data;
    for r in 0..oh {
        let dst_row = &mut out[r * ow * 3..(r + 1) * ow * 3];
        for c in 0..ow {
            // Pillow Geometry.c: output (col c, row r) <- input (x, y).
            let (x, y) = match orientation {
                2 => (w - 1 - c, r),         // FLIP_LEFT_RIGHT
                3 => (w - 1 - c, h - 1 - r), // ROTATE_180
                4 => (c, h - 1 - r),         // FLIP_TOP_BOTTOM
                5 => (r, c),                 // TRANSPOSE
                6 => (r, h - 1 - c),         // ROTATE_270
                7 => (w - 1 - r, h - 1 - c), // TRANSVERSE
                _ => (w - 1 - r, c),         // 8: ROTATE_90
            };
            let s = (y * w + x) * 3;
            dst_row[c * 3..c * 3 + 3].copy_from_slice(&src[s..s + 3]);
        }
    }
    RgbImage {
        width: ow as u32,
        height: oh as u32,
        data: out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiff(le: bool, entries: &[(u16, u16, u32, [u8; 4])]) -> Vec<u8> {
        let mut v = Vec::new();
        let p16 = |v: &mut Vec<u8>, x: u16| {
            v.extend_from_slice(&if le { x.to_le_bytes() } else { x.to_be_bytes() })
        };
        let p32 = |v: &mut Vec<u8>, x: u32| {
            v.extend_from_slice(&if le { x.to_le_bytes() } else { x.to_be_bytes() })
        };
        v.extend_from_slice(if le { b"II" } else { b"MM" });
        p16(&mut v, 42);
        p32(&mut v, 8);
        p16(&mut v, entries.len() as u16);
        for &(tag, typ, cnt, val) in entries {
            p16(&mut v, tag);
            p16(&mut v, typ);
            p32(&mut v, cnt);
            v.extend_from_slice(&val);
        }
        p32(&mut v, 0);
        v
    }

    #[test]
    fn short_orientation_both_endians() {
        let t = tiff(true, &[(0x0112, 3, 1, [6, 0, 0, 0])]);
        assert_eq!(exif_tag(&t), Tag::Value(6));
        let t = tiff(
            false,
            &[(0x0110, 2, 4, *b"abc\0"), (0x0112, 3, 1, [0, 8, 0, 0])],
        );
        assert_eq!(exif_tag(&t), Tag::Value(8));
        let mut with_hdr = b"Exif\0\0".to_vec();
        with_hdr.extend_from_slice(&t);
        assert_eq!(exif_tag(&with_hdr), Tag::Value(8));
    }

    #[test]
    fn byte_type_is_unusable_and_blocks_xmp() {
        let t = tiff(true, &[(0x0112, 1, 1, [6, 0, 0, 0])]);
        assert_eq!(exif_tag(&t), Tag::Unusable);
        assert_eq!(orientation(Some(&t), Some(b"<tiff:Orientation>3<")), 1);
        let none = tiff(true, &[(0x0100, 3, 1, [1, 0, 0, 0])]);
        assert_eq!(
            orientation(Some(&none), Some(b"x tiff:Orientation=\"3\" y")),
            3
        );
    }

    #[test]
    fn xmp_regex_semantics() {
        assert_eq!(
            xmp_orientation(b"tiff:Orientation='6' tiff:Orientation>7"),
            Some(7)
        );
        assert_eq!(xmp_orientation(b"tiff:Orientation=\"x\""), None);
        assert_eq!(xmp_orientation(b"<tiff:Orientation>0</"), Some(0));
    }

    #[test]
    fn truncated_ifd_keeps_earlier_entries() {
        let mut t = tiff(
            true,
            &[(0x0112, 3, 1, [3, 0, 0, 0]), (0x0100, 3, 1, [1, 0, 0, 0])],
        );
        t.truncate(8 + 2 + 12 + 5);
        assert_eq!(exif_tag(&t), Tag::Value(3));
        assert_eq!(exif_tag(b"garbage"), Tag::Absent);
    }

    #[test]
    fn transposes_match_pillow_definitions() {
        // 3x2 image, pixel value = index; check each orientation's corner.
        let data: Vec<u8> = (0..6u8).flat_map(|i| [i, i, i]).collect();
        let img = RgbImage {
            width: 3,
            height: 2,
            data,
        };
        let px = |im: &RgbImage| im.data.chunks(3).map(|c| c[0]).collect::<Vec<_>>();
        // Pillow: im.transpose(m) for [[0,1,2],[3,4,5]]
        let want: [(i64, (u32, u32), [u8; 6]); 7] = [
            (2, (3, 2), [2, 1, 0, 5, 4, 3]),
            (3, (3, 2), [5, 4, 3, 2, 1, 0]),
            (4, (3, 2), [3, 4, 5, 0, 1, 2]),
            (5, (2, 3), [0, 3, 1, 4, 2, 5]),
            (6, (2, 3), [3, 0, 4, 1, 5, 2]),
            (7, (2, 3), [5, 2, 4, 1, 3, 0]),
            (8, (2, 3), [2, 5, 1, 4, 0, 3]),
        ];
        for (o, dims, pix) in want {
            let t = apply(img.clone(), o);
            assert_eq!((t.width, t.height), dims, "orientation {o}");
            assert_eq!(px(&t), pix, "orientation {o}");
        }
    }
}
