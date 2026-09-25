//! Hostile-input robustness: every golden input truncated and corrupted at
//! many offsets must decode to Ok or Err, never panic. Also header-level
//! guards (size limits, decompression bombs) and data-URL misuse.

mod common;

use common::*;
use mimo26_image::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn all_inputs() -> Vec<String> {
    let mut v: Vec<String> = manifest("decode.txt")
        .into_iter()
        .map(|f| f[0].clone())
        .collect();
    for f in manifest("errors.txt") {
        v.push(f[0].clone());
    }
    // The photo-sized inputs only in release builds (seconds in debug).
    if !cfg!(debug_assertions) {
        for p in [
            "pixel_values/photo_640x480.jpg",
            "pixel_values/photo_500x375.jpg",
            "pixel_values/rgba_97x61.png",
        ] {
            v.push(p.to_string());
        }
    }
    v
}

/// Returns a description if decoding `data` panicked.
fn try_decode(data: &[u8]) -> Option<String> {
    match catch_unwind(AssertUnwindSafe(|| {
        if let Ok(img) = decode(data) {
            assert_eq!(img.data.len(), img.width as usize * img.height as usize * 3);
        }
    })) {
        Ok(()) => None,
        Err(e) => Some(
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default(),
        ),
    }
}

#[test]
fn truncated_and_corrupted_goldens_never_panic() {
    std::panic::set_hook(Box::new(|_| {})); // failures are collected below
    let heavy = !cfg!(debug_assertions);
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let mut failures = Vec::new();
    let mut runs = 0usize;
    let t0 = Instant::now();
    for input in all_inputs() {
        let data = read(&input);
        let n = data.len();
        if !heavy && n > 16 * 1024 {
            continue; // debug builds: small inputs only (release covers all)
        }
        // Truncations: every offset for small files (release), else a spread
        // plus the first 64 bytes (headers).
        let mut cuts: Vec<usize> = if heavy && n <= 4096 {
            (0..n).collect()
        } else {
            let k = if heavy { 400 } else { 16 };
            let mut c: Vec<usize> = (0..k).map(|i| i * n / k).collect();
            c.extend(0..n.min(if heavy { 64 } else { 12 }));
            c
        };
        cuts.sort_unstable();
        cuts.dedup();
        for &cut in &cuts {
            runs += 1;
            if let Some(p) = try_decode(&data[..cut]) {
                failures.push(format!("{input} truncated to {cut}: panic {p}"));
            }
        }
        // Single-byte corruptions at spread + random offsets.
        let k = if heavy { 300 } else { 10 };
        for i in 0..k {
            let off = if i % 2 == 0 { i * n / k } else { rng.below(n) };
            for op in 0..4 {
                let mut d = data.clone();
                if d.is_empty() {
                    continue;
                }
                d[off] = match op {
                    0 => d[off] ^ 0xFF,
                    1 => 0x00,
                    2 => 0xFF,
                    _ => d[off].wrapping_add(1),
                };
                runs += 1;
                if let Some(p) = try_decode(&d) {
                    failures.push(format!("{input} byte {off} op {op}: panic {p}"));
                }
            }
        }
        // Random multi-byte mutations.
        let k = if heavy { 60 } else { 4 };
        for _ in 0..k {
            let mut d = data.clone();
            for _ in 0..1 + rng.below(8) {
                if !d.is_empty() {
                    let o = rng.below(d.len());
                    d[o] = rng.next() as u8;
                }
            }
            runs += 1;
            if let Some(p) = try_decode(&d) {
                failures.push(format!("{input} random mutation: panic {p}"));
            }
        }
    }
    let _ = std::panic::take_hook();
    eprintln!(
        "robustness: {runs} decodes in {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    assert!(
        failures.is_empty(),
        "{} panics:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn png_chunk(ty: &[u8], data: &[u8]) -> Vec<u8> {
    let mut v = (data.len() as u32).to_be_bytes().to_vec();
    v.extend_from_slice(ty);
    v.extend_from_slice(data);
    v.extend_from_slice(&[0, 0, 0, 0]); // CRC is not checked
    v
}

fn png_with_ihdr(w: u32, h: u32, depth: u8, ctype: u8, idat: &[u8]) -> Vec<u8> {
    let mut ihdr = w.to_be_bytes().to_vec();
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[depth, ctype, 0, 0, 0]);
    let mut v = b"\x89PNG\r\n\x1a\n".to_vec();
    v.extend(png_chunk(b"IHDR", &ihdr));
    v.extend(png_chunk(b"IDAT", idat));
    v.extend(png_chunk(b"IEND", b""));
    v
}

#[test]
fn size_limits_are_checked_from_the_header() {
    // One pixel over the limit, tiny body: rejected before any allocation.
    let e = decode(&png_with_ihdr(8193, 8192, 8, 6, b"")).unwrap_err();
    assert!(e.0.contains("too large"), "{e}");
    let e = decode(&png_with_ihdr(u32::MAX, u32::MAX, 16, 6, b"")).unwrap_err();
    assert!(e.0.contains("too large"), "{e}");
    // At the limit with no data: fails fast as truncated.
    let t = Instant::now();
    let e = decode(&png_with_ihdr(8192, 8192, 16, 6, &[0x78, 0x9c])).unwrap_err();
    assert!(e.0.contains("truncated"), "{e}");
    assert!(t.elapsed().as_secs_f64() < 5.0);
    let e = decode(&png_with_ihdr(0, 5, 8, 2, b"")).unwrap_err();
    assert!(e.0.contains("zero width or height"), "{e}");
}

#[test]
fn deflate_bomb_is_bounded_by_the_image_size() {
    // The golden's IDAT inflates to 10 MiB but a 1x1 gray image needs 2 bytes.
    let data = read("inputs/png_zlib_bomb_1x1.png");
    let t = Instant::now();
    let img = decode(&data).unwrap();
    assert_eq!(
        (img.width, img.height, img.data.as_slice()),
        (1, 1, &[0u8, 0, 0][..])
    );
    assert!(
        t.elapsed().as_secs_f64() < 0.5,
        "bomb took {:?}",
        t.elapsed()
    );
}

#[test]
fn data_url_rejections() {
    for (url, needle) in [
        (
            "http://example.com/a.png",
            "remote image URLs are not supported",
        ),
        ("data:image/png;base64,@@@@", "invalid base64"),
        ("data:image/png;utf8,<svg/>", "not base64-encoded"),
        (
            "data:image/png;base64,aGVsbG8gd29ybGQ=",
            "unsupported image format",
        ),
        ("", "must be a base64 data URL"),
    ] {
        let e = decode_data_url(url).unwrap_err();
        assert!(e.0.contains(needle), "{url:?}: {e}");
    }
}

#[test]
fn preprocess_rejects_inconsistent_buffers() {
    let img = RgbImage {
        width: 10,
        height: 10,
        data: vec![0; 299],
    };
    assert!(preprocess(&img).unwrap_err().0.contains("does not match"));
    let img = RgbImage {
        width: 0,
        height: 10,
        data: vec![],
    };
    assert!(preprocess(&img).is_err());
    let img = RgbImage {
        width: 1,
        height: 201,
        data: vec![0; 603],
    };
    assert!(preprocess(&img).unwrap_err().0.contains("aspect ratio"));
}
