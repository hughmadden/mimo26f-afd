//! Golden tests against Pillow / numpy outputs written by tests/gen_goldens.py.
//! Each test runs every case of its manifest and reports all failures at once.

mod common;

use common::*;
use mimo26_image::*;

#[test]
fn sha256_known_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

/// Decode every `decode.txt` case whose input has extension `ext`; all must
/// be bit-exact. Prints per-case diff statistics for any mismatch.
fn decode_goldens(ext: &str) -> usize {
    let mut failures = Vec::new();
    let mut n = 0;
    for f in manifest("decode.txt") {
        let (input, w, h, expected) = (&f[0], &f[1], &f[2], &f[3]);
        if !input.ends_with(ext) {
            continue;
        }
        n += 1;
        let (w, h): (u32, u32) = (w.parse().unwrap(), h.parse().unwrap());
        let want = read(expected);
        match decode(&read(input)) {
            Err(e) => failures.push(format!("{input}: decode error: {e}")),
            Ok(img) if (img.width, img.height) != (w, h) => failures.push(format!(
                "{input}: size {}x{} want {w}x{h}",
                img.width, img.height
            )),
            Ok(img) if img.data != want => {
                let (max, mean) = diff_stats(&img.data, &want);
                let differing = img.data.iter().zip(&want).filter(|(a, b)| a != b).count();
                failures.push(format!(
                    "{input}: {differing} of {} samples differ; max {:?} mean [{:.4}, {:.4}, {:.4}]",
                    want.len(),
                    max,
                    mean[0],
                    mean[1],
                    mean[2]
                ));
            }
            Ok(_) => {}
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {n} {ext} goldens differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
    n
}

#[test]
fn png_goldens_bit_exact() {
    let n = decode_goldens(".png");
    assert!(n >= 60, "expected the full PNG golden set, found {n}");
}

#[test]
fn jpeg_goldens_bit_exact() {
    let n = decode_goldens(".jpg");
    assert!(n >= 55, "expected the full JPEG golden set, found {n}");
}

#[test]
fn error_goldens() {
    let text = String::from_utf8(read("errors.txt")).unwrap();
    let mut failures = Vec::new();
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let (left, needle) = line.split_once(" | ").expect("'|' separator");
        let input = left.split_whitespace().next().unwrap();
        match decode(&read(input)) {
            Ok(img) => failures.push(format!(
                "{input}: decoded {}x{}, expected an error",
                img.width, img.height
            )),
            Err(e) if !e.0.contains(needle.trim()) => failures.push(format!(
                "{input}: error {:?} lacks {:?}",
                e.0,
                needle.trim()
            )),
            Err(_) => {}
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn resize_goldens_bit_exact() {
    let mut failures = Vec::new();
    let cases = manifest("resize.txt");
    for f in &cases {
        let p: Vec<u32> = f[1..5].iter().map(|s| s.parse().unwrap()).collect();
        let src = RgbImage {
            width: p[0],
            height: p[1],
            data: read(&f[0]),
        };
        let want = read(&f[5]);
        let got = resize_bicubic(&src, p[2], p[3]);
        if (got.width, got.height) != (p[2], p[3]) || got.data != want {
            let (max, _) = diff_stats(&got.data, &want);
            failures.push(format!("{} -> {}x{}: max diff {max:?}", f[0], p[2], p[3]));
        }
    }
    assert!(cases.len() >= 20);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn smart_resize_and_merged_tokens_match_python() {
    let mut failures = Vec::new();
    let cases = manifest("smart_resize.txt");
    for f in &cases {
        let (h, w): (u32, u32) = (f[0].parse().unwrap(), f[1].parse().unwrap());
        let got = smart_resize(h, w);
        if f[2] == "ERR" {
            if got.is_ok() || merged_tokens(h, w).is_ok() {
                failures.push(format!("{h}x{w}: expected error, got {got:?}"));
            }
            continue;
        }
        let want = (f[2].parse::<u32>().unwrap(), f[3].parse::<u32>().unwrap());
        let tokens: usize = f[4].parse().unwrap();
        match got {
            Ok(g) if g == want && merged_tokens(h, w) == Ok(tokens) => {}
            other => failures.push(format!(
                "{h}x{w}: got {other:?}, want {want:?} / {tokens} tokens"
            )),
        }
    }
    assert!(cases.len() > 1500);
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// One `pixel_values.txt` case: compare grid, resized-u8 hash, full
/// pixel_values hash and the stored rows (max abs diff <= 1e-5 required,
/// exact expected).
fn check_pixel_values(f: &[String], img: &RgbImage) -> Result<(), String> {
    let name = &f[0];
    let n = |i: usize| f[i].parse::<u32>().unwrap();
    let (gt, gh, gw, rw, rh) = (n(4), n(5), n(6), n(7), n(8));
    let (sha_rs, sha_pv) = (&f[9], &f[10]);
    let rows: Vec<usize> = f[11].split(',').map(|s| s.parse().unwrap()).collect();
    if (img.width, img.height) != (n(2), n(3)) {
        return Err(format!(
            "{name}: input {}x{} want {}x{}",
            img.width,
            img.height,
            n(2),
            n(3)
        ));
    }
    let (sh, sw) = smart_resize(img.height, img.width).map_err(|e| e.0)?;
    if (sw, sh) != (rw, rh) {
        return Err(format!("{name}: smart_resize {sw}x{sh} want {rw}x{rh}"));
    }
    let resized = if (sw, sh) == (img.width, img.height) {
        img.clone()
    } else {
        resize_bicubic(img, sw, sh)
    };
    let rs_ok = sha256_hex(&resized.data) == *sha_rs;
    let p = preprocess(img).map_err(|e| e.0)?;
    if (p.grid_t, p.grid_h, p.grid_w) != (gt, gh, gw) {
        return Err(format!(
            "{name}: grid {:?} want {:?}",
            (p.grid_t, p.grid_h, p.grid_w),
            (gt, gh, gw)
        ));
    }
    if p.data.len() != p.rows() * PATCH_DIM || p.merged_tokens() != (gh * gw / 4) as usize {
        return Err(format!("{name}: bad data length {}", p.data.len()));
    }
    let stored = f32s(&read(&format!("pixel_values/{name}.rows.f32")));
    let mut max_diff = 0f32;
    for (i, &r) in rows.iter().enumerate() {
        let got = &p.data[r * PATCH_DIM..(r + 1) * PATCH_DIM];
        let want = &stored[i * PATCH_DIM..(i + 1) * PATCH_DIM];
        for (a, b) in got.iter().zip(want) {
            max_diff = max_diff.max((a - b).abs());
        }
    }
    let pv_ok = sha256_hex(&f32_bytes(&p.data)) == *sha_pv;
    if !rs_ok || !pv_ok || max_diff > 1e-5 {
        return Err(format!(
            "{name}: resized-u8 hash {}, pixel_values hash {}, stored-row max abs diff {max_diff:e}",
            if rs_ok { "ok" } else { "MISMATCH" },
            if pv_ok { "ok" } else { "MISMATCH" },
        ));
    }
    Ok(())
}

fn pv_case(name: &str) -> Vec<String> {
    manifest("pixel_values.txt")
        .into_iter()
        .find(|f| f[0] == name)
        .unwrap_or_else(|| panic!("no pixel_values case {name}"))
}

fn pv_input(f: &[String]) -> RgbImage {
    let (kind, path) = f[1].split_once(':').unwrap();
    let (w, h) = (f[2].parse().unwrap(), f[3].parse().unwrap());
    match kind {
        "rgb" => RgbImage {
            width: w,
            height: h,
            data: read(path),
        },
        "procedural" => procedural(w, h),
        // Encoded inputs go through the public data-URL entry point.
        _ => {
            let mime = if kind == "jpeg" {
                "image/jpeg"
            } else {
                "image/png"
            };
            let url = format!("data:{mime};base64,{}", b64(&read(path)));
            decode_data_url(&url).expect("decode_data_url")
        }
    }
}

fn b64(data: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for (i, c) in data.chunks(3).enumerate() {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        s.push(A[(n >> 18) as usize & 63] as char);
        s.push(A[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 {
            A[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        s.push(if c.len() > 2 {
            A[n as usize & 63] as char
        } else {
            '='
        });
        if i % 19 == 18 {
            s.push('\n'); // line breaks inside the payload must be tolerated
        }
    }
    s
}

#[test]
fn pixel_values_small_full_rows() {
    for name in ["min_20x30", "tiny_5x3", "rgba_97x61"] {
        let f = pv_case(name);
        check_pixel_values(&f, &pv_input(&f)).unwrap();
    }
}

#[test]
fn pixel_values_photos_via_data_url() {
    for name in ["photo_640x480", "photo_500x375"] {
        let f = pv_case(name);
        check_pixel_values(&f, &pv_input(&f)).unwrap();
    }
}

/// 3900x3400 -> max_pixels downscale to 3808x3328 (~12.7 M px, 290 MB of
/// f32). Release builds only; debug builds would take minutes.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn pixel_values_max_pixels_procedural() {
    let f = pv_case("procedural_3900x3400");
    check_pixel_values(&f, &pv_input(&f)).unwrap();
}
