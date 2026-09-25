//! Shared fixtures for the mimo26-load two-run trap suite.
//!
//! `make_fused_case` ports `spike/tests/conftest.py:39-76`: synthetic shards in
//! REAL layout — per-shard `[q_c|k_c; v_c]` rows in TP-rank order, per-shard
//! scale grids PADDED to `ceil(rows/br) + pad_rows` (pad rows carry 1.0 and are
//! poisonable), and a construction-side reference built from construction data
//! alone (codes decoded times the scale each row was quantized with) —
//! independent of the implementation under test.
//!
//! The fixture RNG is a local xorshift (the Python side uses
//! `numpy.random.default_rng(7)`); the case is synthetic and self-referential,
//! so only determinism matters here — the cross-language pin is
//! `tests/golden_fused_split.rs` against the external golden corpus.

use mimo26_load::e4m3::{decode_e4m3, quantize_block};
use mimo26_load::Mat;

pub struct XorShift64(u64);

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[-0.5, 0.5)` (stands in for the Python `standard_normal*0.5`).
    pub fn next_small(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
        u - 0.5
    }
}

/// `(shard_weights, shard_scales, ref)` — see `make_fused_case` in
/// `spike/tests/conftest.py:39-76`.
pub fn make_fused_case(
    n_ranks: usize,
    segs: (usize, usize, usize),
    cols: usize,
    block: (usize, usize),
    pad_rows: usize,
) -> (Vec<Mat<u8>>, Vec<Mat<f32>>, Mat<f32>) {
    let (br, bc) = block;
    let (r_q, r_k, r_v) = segs;
    let per = r_q + r_k + r_v;
    let mut rng = XorShift64::new(7);
    let mut weights: Vec<Mat<u8>> = Vec::new();
    let mut scales: Vec<Mat<f32>> = Vec::new();
    let mut per_shard_rows: Vec<Mat<f32>> = Vec::new();
    for _ in 0..n_ranks {
        let mut w = Mat::<f64>::zeros(per, cols);
        for r in 0..per {
            for c in 0..cols {
                w.set(r, c, rng.next_small());
            }
        }
        let (codes, grid) = quantize_block(&w, block).expect("fixture quantize");
        let real_rows = mimo26_load::ceil_div(per, br);
        let mut pad = Mat::<f32>::zeros(real_rows + pad_rows, grid.cols);
        for r in 0..pad.rows {
            for c in 0..pad.cols {
                pad.set(r, c, 1.0);
            }
        }
        for r in 0..grid.rows {
            for c in 0..grid.cols {
                pad.set(r, c, grid.get(r, c));
            }
        }
        // construction-side truth: row r's scale = pad[r // br, c // bc]
        let mut rows_f32 = Mat::<f32>::zeros(per, cols);
        for r in 0..per {
            for c in 0..cols {
                let s = pad.get(r / br, c / bc);
                rows_f32.set(r, c, (decode_e4m3(codes.get(r, c)) * f64::from(s)) as f32);
            }
        }
        weights.push(codes);
        scales.push(pad);
        per_shard_rows.push(rows_f32);
    }
    // reference: projection-major [Q|K|V] (conftest.py:72-75)
    let mut q_parts = Vec::new();
    let mut k_parts = Vec::new();
    let mut v_parts = Vec::new();
    for rows in &per_shard_rows {
        q_parts.push(rows.slice_rows(0, r_q));
        k_parts.push(rows.slice_rows(r_q, r_q + r_k));
        v_parts.push(rows.slice_rows(r_q + r_k, per));
    }
    let q = Mat::stack_rows(&q_parts).expect("fixture q");
    let k = Mat::stack_rows(&k_parts).expect("fixture k");
    let v = Mat::stack_rows(&v_parts).expect("fixture v");
    let refm = Mat::stack_rows(&[q, k, v]).expect("fixture ref");
    (weights, scales, refm)
}

// ---------------------------------------------------------------------------
// external golden corpus (read-only per I-Gold — referenced, never copied;
// mirrors spike/tests/conftest.py:24-36)
// ---------------------------------------------------------------------------

pub fn golden_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("MIMO26_GOLDEN") {
        return p.into();
    }
    let home = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    std::path::PathBuf::from(home)
        .join("oracle/goldens/fp8_block_golden.json")
}

/// Read the golden JSON; `None` = skip mode (`MIMO26_ALLOW_MISSING_GOLDEN=1`);
/// otherwise a missing golden is a LOUD failure (conftest.py:30-36).
pub fn load_golden() -> Option<String> {
    match std::fs::read_to_string(golden_path()) {
        Ok(s) => Some(s),
        Err(e) => {
            if std::env::var("MIMO26_ALLOW_MISSING_GOLDEN").as_deref() == Ok("1") {
                eprintln!("golden missing at {:?} ({e}) — MIMO26_ALLOW_MISSING_GOLDEN=1", golden_path());
                None
            } else {
                panic!(
                    "missing golden {:?} ({e}) — regen mimo26 scripts/gen-golden.py (mirrors spike/tests/conftest.py:30-36)",
                    golden_path()
                );
            }
        }
    }
}

// Minimal targeted JSON readers — the golden is machine-generated and flat; a
// full parser is out of scope. Fail loud on any structural surprise.

pub fn json_object<'a>(hay: &'a str, key: &str) -> &'a str {
    let pat = format!("\"{key}\"");
    let start = hay.find(&pat).unwrap_or_else(|| panic!("golden: object key {key:?} not found"));
    let after = &hay[start + pat.len()..];
    let ob = after.find('{').unwrap_or_else(|| panic!("golden: no object after {key:?}"));
    let mut depth = 0usize;
    for (i, ch) in after[ob..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &after[ob..ob + i + 1];
                }
            }
            _ => {}
        }
    }
    panic!("golden: unbalanced object after {key:?}");
}

pub fn json_str_field(hay: &str, key: &str) -> String {
    let pat = format!("\"{key}\"");
    let start = hay.find(&pat).unwrap_or_else(|| panic!("golden: field {key:?} not found"));
    let after = &hay[start + pat.len()..];
    let q1 = after.find('"').expect("golden: string open");
    let rest = &after[q1 + 1..];
    let q2 = rest.find('"').expect("golden: string close");
    rest[..q2].to_string()
}

pub fn json_usize_field(hay: &str, key: &str) -> usize {
    let pat = format!("\"{key}\"");
    let start = hay.find(&pat).unwrap_or_else(|| panic!("golden: field {key:?} not found"));
    let after = &hay[start + pat.len()..];
    let colon = after.find(':').expect("golden: colon");
    let rest = after[colon + 1..].trim_start();
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().expect("golden: integer parse")
}

pub fn json_usize_array_field(hay: &str, key: &str) -> Vec<usize> {
    let pat = format!("\"{key}\"");
    let start = hay.find(&pat).unwrap_or_else(|| panic!("golden: field {key:?} not found"));
    let after = &hay[start + pat.len()..];
    let ob = after.find('[').expect("golden: array open");
    let cb = after[ob..].find(']').expect("golden: array close");
    after[ob + 1..ob + cb]
        .split(',')
        .map(|t| t.trim().parse::<usize>().expect("golden: array int"))
        .collect()
}

pub fn hex_to_bytes(hex: &str) -> Vec<u8> {
    assert!(hex.len() % 2 == 0, "golden: odd hex length");
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("golden: hex byte"))
        .collect()
}

pub fn bytes_to_f32le(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 4 == 0, "golden: f32 byte length");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
