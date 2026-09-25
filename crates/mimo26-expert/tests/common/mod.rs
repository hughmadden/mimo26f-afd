//! Shared fixtures for the `mimo26-expert` two-run trap suite.
//!
//! The fixture builds a **synthetic expert quarter slice** in the pinned
//! layout: gate/up `[512, 4096]` and down `[4096, 512]` logical, stored as
//! `weight` `u8 [rows, in/2]` + `weight_scale` `u8 [rows, in/32]`, with the
//! E2M1 nibble order and E8M0 scale bytes chosen by the fixture RNG.
//!
//! It also builds a **real-block** fixture from
//! `bench/fixtures/expert_nibble_fixture.json` (27 real expert blocks,
//! sha-pinned) — the L3 golden. The real fixture carries only the sampled bit
//! patterns, not the 33 MB of payload, so the CPU tests use the synthetic
//! blocks for the GEMM and the real fixture for the unpack-path contract
//! (positions, shapes, scale statistics, and the reference semantics).

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use mimo26_expert::mxfp4::{self, BLOCK};
use mimo26_expert::slice::{self, Proj};
use mimo26_expert::{ExpertError, NaiveBits};

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

    /// Uniform in `[-0.5, 0.5)`.
    pub fn next_small(&mut self) -> f32 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        (u - 0.5) as f32
    }
}

/// One projection's packed payload + scales, in the slice layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjBytes {
    /// `u8 [slice_rows, in/2]`.
    pub payload: Vec<u8>,
    /// `u8 [slice_rows, in/32]`.
    pub scales: Vec<u8>,
}

/// A synthetic expert quarter slice: the three projections' bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceBytes {
    /// gate.
    pub gate: ProjBytes,
    /// up.
    pub up: ProjBytes,
    /// down.
    pub down: ProjBytes,
}

impl SliceBytes {
    /// The `(payload, scales)` pair for one projection.
    pub fn proj(&self, p: Proj) -> &ProjBytes {
        match p {
            Proj::Gate => &self.gate,
            Proj::Up => &self.up,
            Proj::Down => &self.down,
        }
    }

    /// Serialize into the pinned flat slice image.
    pub fn to_slice(&self) -> Vec<u8> {
        let mut out = vec![0u8; slice::QUARTER_SLICE_BYTES];
        for p in Proj::ALL {
            let b = self.proj(p);
            let poff = p.slice_payload_off();
            let soff = p.slice_scale_off();
            out[poff..poff + b.payload.len()].copy_from_slice(&b.payload);
            out[soff..soff + b.scales.len()].copy_from_slice(&b.scales);
        }
        out
    }
}

/// Build one projection's bytes with the fixture RNG.
///
/// The payload bytes are random (every nibble pattern appears), and the scale
/// bytes are drawn from a range that **includes 255** so the T10 clamp is
/// exercised on synthetic data (the real fixture has no 255 byte — see
/// `src/fixture.rs`).
pub fn synth_proj(rng: &mut XorShift64, p: Proj) -> ProjBytes {
    let rows = p.slice_rows();
    let mut payload = vec![0u8; rows * p.payload_cols()];
    for b in payload.iter_mut() {
        *b = rng.next_u8();
    }
    let mut scales = vec![0u8; rows * p.scale_cols()];
    for b in scales.iter_mut() {
        // 1 in 8 blocks gets the reserved byte 255 (T10), the rest 100..=140.
        *b = if rng.below(8) == 0 {
            255
        } else {
            (100 + rng.below(41)) as u8
        };
    }
    ProjBytes { payload, scales }
}

/// A synthetic expert quarter slice.
pub fn synth_slice(seed: u64) -> SliceBytes {
    let mut rng = XorShift64::new(seed);
    SliceBytes {
        gate: synth_proj(&mut rng, Proj::Gate),
        up: synth_proj(&mut rng, Proj::Up),
        down: synth_proj(&mut rng, Proj::Down),
    }
}

/// A synthetic slice whose values are all exactly representable, so the
/// comparisons are bit-exact rather than tolerance-based.
///
/// Every payload byte is `pack_byte(even, odd)` with codes drawn from the
/// codebook, and every scale byte is in `118..=125` (never 255), so `unpack` is
/// exact in f32 for both the correct and the naive path.
pub fn exact_slice(seed: u64) -> SliceBytes {
    let mut rng = XorShift64::new(seed);
    let mut proj = |p: Proj| -> ProjBytes {
        let rows = p.slice_rows();
        let mut payload = vec![0u8; rows * p.payload_cols()];
        for b in payload.iter_mut() {
            let even = rng.below(16) as u8;
            let odd = rng.below(16) as u8;
            *b = mxfp4::pack_byte(even, odd);
        }
        let mut scales = vec![0u8; rows * p.scale_cols()];
        for b in scales.iter_mut() {
            *b = (118 + rng.below(8)) as u8;
        }
        ProjBytes { payload, scales }
    };
    SliceBytes {
        gate: proj(Proj::Gate),
        up: proj(Proj::Up),
        down: proj(Proj::Down),
    }
}

/// `n_experts` synthetic slices back-to-back (the grouped image).
pub fn grouped_image(slices: &[SliceBytes]) -> Vec<u8> {
    let mut out = Vec::with_capacity(slices.len() * slice::QUARTER_SLICE_BYTES);
    for s in slices {
        out.extend_from_slice(&s.to_slice());
    }
    out
}

/// A deterministic token matrix `[total_tokens, in_cols]`.
pub fn tokens(seed: u64, total_tokens: usize, in_cols: usize) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    (0..total_tokens * in_cols).map(|_| rng.next_small()).collect()
}

/// Compare two f32 slices bitwise (so `-0.0` and `0.0` are distinguished).
pub fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// First index where two f32 slices differ bitwise.
pub fn first_bit_diff(a: &[f32], b: &[f32]) -> Option<usize> {
    a.iter().zip(b.iter()).position(|(x, y)| x.to_bits() != y.to_bits())
}

/// Max absolute difference.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    let mut m = 0.0f64;
    for i in 0..a.len() {
        assert!(a[i].is_finite() && b[i].is_finite(), "nonfinite output at {i}");
        let d = (f64::from(a[i]) - f64::from(b[i])).abs();
        if d > m {
            m = d;
        }
    }
    m
}

/// Max relative difference (denominator floored at `atol`).
pub fn max_rel_diff(a: &[f32], b: &[f32], atol: f64) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    let mut m = 0.0f64;
    for i in 0..a.len() {
        assert!(a[i].is_finite() && b[i].is_finite(), "nonfinite output at {i}");
        let d = (f64::from(a[i]) - f64::from(b[i])).abs();
        let denom = f64::from(b[i]).abs().max(atol);
        let r = d / denom;
        if r > m {
            m = r;
        }
    }
    m
}

/// The naive flag for a NEGATIVE test: env default (fails under the naive run).
pub fn naive_env() -> NaiveBits {
    mimo26_expert::bits_from_env()
}

/// The repo root.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The real-block fixture path.
pub fn fixture_path() -> PathBuf {
    mimo26_expert::fixture::fixture_path()
}

/// Load the real-block fixture, or `None` when it is absent (the L0/L1 gate
/// must not depend on a 2.7 MB JSON that a fresh clone may not have — but the
/// fixture IS committed, so the tests assert it loads).
pub fn load_fixture() -> Result<mimo26_expert::fixture::Fixture, ExpertError> {
    mimo26_expert::fixture::load()
}

/// A unique scratch directory (no external crates).
pub fn scratch_dir(tag: &str) -> PathBuf {
    let base = std::env::var("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = base.join(format!("mimo26-expert-{tag}-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// The local weights copy (READ-ONLY). Used only by `#[ignore]`d real-block tests.
pub fn local_weights_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    PathBuf::from(home).join("models/XiaomiMiMo/MiMo-V2.6-Flash-RL")
}

/// The E8M0 block width, re-exported for readability.
pub const BLOCK_W: usize = BLOCK;
