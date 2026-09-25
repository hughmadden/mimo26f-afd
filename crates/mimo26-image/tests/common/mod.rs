//! Shared helpers for the golden tests (goldens come from tests/gen_goldens.py).
#![allow(dead_code)]

use mimo26_image::RgbImage;
use std::path::PathBuf;

pub fn goldens() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
}

pub fn read(rel: &str) -> Vec<u8> {
    let p = goldens().join(rel);
    std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// Manifest lines split on whitespace, comments/blank lines skipped and a
/// trailing `# note` removed.
pub fn manifest(name: &str) -> Vec<Vec<String>> {
    let text = String::from_utf8(read(name)).expect("utf-8 manifest");
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let l = l.split(" # ").next().unwrap_or(l);
            l.split_whitespace().map(str::to_string).collect()
        })
        .collect()
}

/// Deterministic test pattern, identical to `procedural()` in gen_goldens.py.
pub fn procedural(w: u32, h: u32) -> RgbImage {
    let mut data = Vec::with_capacity(w as usize * h as usize * 3);
    for y in 0..h as u64 {
        for x in 0..w as u64 {
            let n = ((x
                .wrapping_mul(2654435761)
                .wrapping_add(y.wrapping_mul(40503)))
                & 0xFFFF_FFFF)
                >> 16;
            let r = (x + 2 * y + ((x * y) >> 9)) & 255;
            let g = (128 + ((x >> 3) ^ (y >> 3)) * 3) & 255;
            let b = ((x >> 2) + (y >> 2) + (n & 31)) & 255;
            data.extend_from_slice(&[r as u8, g as u8, b as u8]);
        }
    }
    RgbImage {
        width: w,
        height: h,
        data,
    }
}

/// Per-channel (max abs diff, mean abs diff) between two RGB buffers.
pub fn diff_stats(a: &[u8], b: &[u8]) -> ([u8; 3], [f64; 3]) {
    let mut max = [0u8; 3];
    let mut sum = [0u64; 3];
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = x.abs_diff(y);
        max[i % 3] = max[i % 3].max(d);
        sum[i % 3] += d as u64;
    }
    let n = (a.len() / 3).max(1) as f64;
    (
        max,
        [sum[0] as f64 / n, sum[1] as f64 / n, sum[2] as f64 / n],
    )
}

pub fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// SHA-256 (FIPS 180-4), test-only.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut compress = |block: &[u8]| {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[4 * i],
                block[4 * i + 1],
                block[4 * i + 2],
                block[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    };
    let full = data.len() / 64 * 64;
    for block in data[..full].chunks_exact(64) {
        compress(block);
    }
    let mut tail = data[full..].to_vec();
    tail.push(0x80);
    while tail.len() % 64 != 56 {
        tail.push(0);
    }
    tail.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in tail.chunks_exact(64) {
        compress(block);
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}
