//! Shared fixtures for the mimo26-repack two-run trap suite.
//!
//! The fixture builds a **synthetic expert** in the real checkpoint layout:
//! `gate/up` `[2048, 4096]` and `down` `[4096, 2048]` logical, stored as
//! `weight` `u8 [out, in/2]` + `weight_scale` `u8 [out, in/32]`, with the
//! E2M1 nibble order and E8M0 scale bytes chosen by the fixture RNG.
//!
//! It also writes a **real safetensors file** (header + data) so the reader,
//! the repack and the manifest are exercised end to end without touching the
//! 13 GB checkpoint. The real checkpoint is read READ-ONLY by
//! `tests/real_headers.rs`, which is `#[ignore]`d by default (it needs the dev host
//! weights copy and is not part of the L0/L1 merge gate).

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use mimo26_repack::geom::{self, Proj};
use mimo26_repack::mxfp4::{self, Mxfp4Naive};
use mimo26_repack::repack::ExpertTensors;

/// Deterministic xorshift64 (the suite convention: `mimo26-load/tests/common`).
pub struct XorShift64(u64);

impl XorShift64 {
    /// Seed (0 is remapped to 1 so the stream is never all-zero).
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    /// Next raw word.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Next byte.
    pub fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 24) as u8
    }

    /// Uniform in `[0, n)`.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
}

/// Build one projection's packed payload + scales with the fixture RNG.
///
/// The payload bytes are random (every nibble pattern appears), and the scale
/// bytes are drawn from a range that **includes 255** so the T10 clamp is
/// exercised on real data rather than only on a hand-made vector.
pub fn synth_proj(rng: &mut XorShift64, p: Proj) -> (Vec<u8>, Vec<u8>) {
    let mut w = vec![0u8; p.out_rows() * p.in_cols() / 2];
    for b in w.iter_mut() {
        *b = rng.next_u8();
    }
    let mut s = vec![0u8; p.out_rows() * p.in_cols() / 32];
    for b in s.iter_mut() {
        // 1 in 8 blocks gets the reserved byte 255 (T10), the rest 100..=140.
        *b = if rng.below(8) == 0 {
            255
        } else {
            (100 + rng.below(41)) as u8
        };
    }
    (w, s)
}

/// A synthetic expert in the real checkpoint layout.
pub fn synth_expert(seed: u64) -> ExpertTensors {
    let mut rng = XorShift64::new(seed);
    let (gate_w, gate_s) = synth_proj(&mut rng, Proj::Gate);
    let (up_w, up_s) = synth_proj(&mut rng, Proj::Up);
    let (down_w, down_s) = synth_proj(&mut rng, Proj::Down);
    ExpertTensors {
        gate_w,
        gate_s,
        up_w,
        up_s,
        down_w,
        down_s,
    }
}

/// A synthetic expert whose values are all exactly representable, so the
/// round-trip comparison is bit-exact rather than tolerance-based.
///
/// Every payload byte is `pack_byte(even, odd)` with codes drawn from the
/// codebook, and every scale byte is in `100..=140` (never 255), so
/// `unpack` is exact in f32 for both the correct and the naive path.
pub fn exact_expert(seed: u64) -> ExpertTensors {
    let mut rng = XorShift64::new(seed);
    let mut proj = |p: Proj| -> (Vec<u8>, Vec<u8>) {
        let mut w = vec![0u8; p.out_rows() * p.in_cols() / 2];
        for b in w.iter_mut() {
            let even = (rng.below(16)) as u8;
            let odd = (rng.below(16)) as u8;
            *b = mxfp4::pack_byte(even, odd);
        }
        let mut s = vec![0u8; p.out_rows() * p.in_cols() / 32];
        for b in s.iter_mut() {
            *b = (100 + rng.below(41)) as u8;
        }
        (w, s)
    };
    let (gate_w, gate_s) = proj(Proj::Gate);
    let (up_w, up_s) = proj(Proj::Up);
    let (down_w, down_s) = proj(Proj::Down);
    ExpertTensors {
        gate_w,
        gate_s,
        up_w,
        up_s,
        down_w,
        down_s,
    }
}

/// Write a real safetensors file holding one expert's six tensors.
///
/// Layout: `<u64 LE header_len><JSON header><data>`, exactly as the checkpoint
/// shards are written. Returns the path.
pub fn write_expert_safetensors(
    dir: &Path,
    file_name: &str,
    layer: usize,
    expert: usize,
    t: &ExpertTensors,
) -> PathBuf {
    let mut entries: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for p in Proj::ALL {
        let (w, s) = t.proj(p);
        entries.push((
            geom::tensor_name(layer, expert, p, false),
            vec![p.out_rows(), p.in_cols() / 2],
            w.to_vec(),
        ));
        entries.push((
            geom::tensor_name(layer, expert, p, true),
            vec![p.out_rows(), p.in_cols() / 32],
            s.to_vec(),
        ));
    }
    let mut header = String::from("{");
    let mut offset = 0u64;
    for (i, (name, shape, data)) in entries.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let shape: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            name,
            shape.join(","),
            offset,
            offset + data.len() as u64
        ));
        offset += data.len() as u64;
    }
    header.push('}');
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for (_, _, data) in &entries {
        out.extend_from_slice(data);
    }
    let path = dir.join(file_name);
    std::fs::write(&path, &out).expect("write fixture safetensors");
    path
}

/// A unique scratch directory under the target tmp dir (no external crates).
pub fn scratch_dir(tag: &str) -> PathBuf {
    let base = std::env::var("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = base.join(format!("mimo26-repack-{tag}-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// The local weights copy (READ-ONLY). Used only by the `#[ignore]`d real-header
/// test; never written to.
pub fn local_weights_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    PathBuf::from(home).join("models/XiaomiMiMo/MiMo-V2.6-Flash-RL")
}

/// Compare two f32 slices bitwise (so `-0.0` and `0.0` are distinguished).
pub fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// First index where two f32 slices differ bitwise.
pub fn first_bit_diff(a: &[f32], b: &[f32]) -> Option<usize> {
    a.iter()
        .zip(b.iter())
        .position(|(x, y)| x.to_bits() != y.to_bits())
}

/// The naive flag for a NEGATIVE test: env default (fails under the naive run).
pub fn naive_env() -> Mxfp4Naive {
    mxfp4::naive_from_env()
}
