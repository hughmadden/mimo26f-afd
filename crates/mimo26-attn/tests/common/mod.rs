//! Shared fixtures for the `mimo26-attn` two-run trap suite + the oracle
//! parity harness plumbing.
//!
//! (Per-binary `mod common` — not every test binary uses every helper.)
//!
//! # Parity manifest format (`mimo26-attn-parity-manifest 1`)
//!
//! Line-oriented, whitespace-separated, little-endian raw binaries beside the
//! manifest — the SAME file drives `tests/oracle_driver.py` (numpy oracle) and
//! `kernels/parity/attn_parity.cu` (GPU harness), so a case written once is
//! checked on both sides:
//!
//! ```text
//! mimo26-attn-parity-manifest 1
//! case <name> <type> <family> <window> <sink> <theta> <partial> <vscale> <tol_abs> <tol_rel>
//! tensor <name> <f32|i64|u8> <d0,d1,d2> <file>
//! expect <name> <f32|i64|u8> <d0,d1,d2> <file>
//! ```
//!
//! `window` 0 = none (GA), `sink` 1 = a `sink` tensor carries the per-Q-head
//! `[n_q]` bias. Case types: `attn` / `decode` / `prefill` (tensor `v` is
//! CACHED V — post-`v_scale`, T18), `rope`, `kv_store` (tensor `v_raw` is RAW V
//! — the store path applies `v_scale`, T18).
//!
//! `run_oracle_eval` shells `python3 tests/oracle_driver.py` against
//! `oracle/mimo26` — the byte-verified CPU twin (I-Gold: consumed read-only;
//! nothing here imports code under test). If python3/numpy ever disappear the
//! gate FAILS LOUD: ci-cpu already runs `pytest oracle/tests` on them.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// deterministic RNG (same generator shape as mimo26-load's test fixtures)
// ---------------------------------------------------------------------------

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

    /// Uniform in `[-0.5, 0.5)`.
    pub fn next_small(&mut self) -> f32 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        (u - 0.5) as f32
    }

    pub fn fill_small(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_small()).collect()
    }
}

// ---------------------------------------------------------------------------
// numeric comparison
// ---------------------------------------------------------------------------

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    let mut m = 0.0f64;
    for i in 0..a.len() {
        let d = (f64::from(a[i]) - f64::from(b[i])).abs();
        if d > m {
            m = d;
        }
    }
    m
}

pub fn bit_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && (0..a.len()).all(|i| a[i].to_bits() == b[i].to_bits())
}

pub fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
    a == b
}

// ---------------------------------------------------------------------------
// manifest + raw binary I/O
// ---------------------------------------------------------------------------

pub const MANIFEST_HEADER: &str = "mimo26-attn-parity-manifest 1";

#[derive(Clone, Debug)]
pub struct ManifestTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub file: String,
    pub is_expect: bool,
}

#[derive(Clone, Debug)]
pub struct ManifestCase {
    pub name: String,
    pub ty: String,
    pub family: String,
    pub window: usize,
    pub sink: u8,
    pub theta: f64,
    pub partial: f64,
    pub vscale: f64,
    pub tol_abs: f64,
    pub tol_rel: f64,
    pub tensors: Vec<ManifestTensor>,
}

impl ManifestCase {
    pub fn tensor(&self, name: &str) -> &ManifestTensor {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("case {}: tensor {name:?} missing from manifest", self.name))
    }

    pub fn line(&self) -> String {
        format!(
            "case {} {} {} {} {} {} {} {} {} {}",
            self.name, self.ty, self.family, self.window, self.sink, self.theta, self.partial,
            self.vscale, self.tol_abs, self.tol_rel
        )
    }
}

pub fn tensor_line(t: &ManifestTensor) -> String {
    let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
    format!(
        "{} {} {} {} {}",
        if t.is_expect { "expect" } else { "tensor" },
        t.name,
        t.dtype,
        shape.join(","),
        t.file
    )
}

pub fn write_bin_f32(dir: &Path, file: &str, data: &[f32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(dir.join(file), bytes).expect("write f32 bin");
}

pub fn write_bin_i64(dir: &Path, file: &str, data: &[i64]) {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(dir.join(file), bytes).expect("write i64 bin");
}

pub fn write_bin_u8(dir: &Path, file: &str, data: &[u8]) {
    std::fs::write(dir.join(file), data).expect("write u8 bin");
}

pub fn read_bin_f32(dir: &Path, file: &str) -> Vec<f32> {
    let bytes = std::fs::read(dir.join(file)).unwrap_or_else(|e| panic!("read {file}: {e}"));
    assert!(bytes.len() % 4 == 0, "{file}: f32 byte length");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn read_bin_i64(dir: &Path, file: &str) -> Vec<i64> {
    let bytes = std::fs::read(dir.join(file)).unwrap_or_else(|e| panic!("read {file}: {e}"));
    assert!(bytes.len() % 8 == 0, "{file}: i64 byte length");
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

pub fn read_bin_u8(dir: &Path, file: &str) -> Vec<u8> {
    std::fs::read(dir.join(file)).unwrap_or_else(|e| panic!("read {file}: {e}"))
}

/// Read a tensor referenced by a manifest entry (dtype-dispatched, f32 out).
pub fn read_tensor_f32(dir: &Path, t: &ManifestTensor) -> Vec<f32> {
    assert_eq!(t.dtype, "f32", "tensor {} is not f32", t.name);
    read_bin_f32(dir, &t.file)
}

pub fn read_tensor_i64(dir: &Path, t: &ManifestTensor) -> Vec<i64> {
    assert_eq!(t.dtype, "i64", "tensor {} is not i64", t.name);
    read_bin_i64(dir, &t.file)
}

pub fn parse_manifest(path: &Path) -> Vec<ManifestCase> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {:?}: {e}", path));
    let mut cases: Vec<ManifestCase> = Vec::new();
    for (ln, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        match f[0] {
            "mimo26-attn-parity-manifest" => {
                assert_eq!(f[1], "1", "manifest version (line {})", ln + 1)
            }
            "case" => {
                assert_eq!(f.len(), 11, "case line {} needs 10 fields", ln + 1);
                cases.push(ManifestCase {
                    name: f[1].to_string(),
                    ty: f[2].to_string(),
                    family: f[3].to_string(),
                    window: f[4].parse().expect("window"),
                    sink: f[5].parse().expect("sink"),
                    theta: f[6].parse().expect("theta"),
                    partial: f[7].parse().expect("partial"),
                    vscale: f[8].parse().expect("vscale"),
                    tol_abs: f[9].parse().expect("tol_abs"),
                    tol_rel: f[10].parse().expect("tol_rel"),
                    tensors: Vec::new(),
                })
            }
            "tensor" | "expect" => {
                assert_eq!(f.len(), 5, "tensor line {} needs 4 fields", ln + 1);
                let shape: Vec<usize> = f[3]
                    .split(',')
                    .map(|d| d.parse().expect("shape dim"))
                    .collect();
                let case = cases
                    .last_mut()
                    .unwrap_or_else(|| panic!("tensor before case at line {}", ln + 1));
                case.tensors.push(ManifestTensor {
                    name: f[1].to_string(),
                    dtype: f[2].to_string(),
                    shape,
                    file: f[4].to_string(),
                    is_expect: f[0] == "expect",
                });
            }
            other => panic!("manifest line {}: unknown record {other:?}", ln + 1),
        }
    }
    cases
}

/// Write a manifest of cases (case lines + their tensor/expect records).
pub fn write_manifest(path: &Path, cases: &[ManifestCase]) {
    let mut out = String::from(MANIFEST_HEADER);
    out.push('\n');
    for c in cases {
        out.push_str(&c.line());
        out.push('\n');
        for t in &c.tensors {
            out.push_str(&tensor_line(t));
            out.push('\n');
        }
    }
    std::fs::write(path, out).expect("write manifest");
}

// ---------------------------------------------------------------------------
// oracle driver invocation (python3 + numpy — same deps as the ci-cpu gate)
// ---------------------------------------------------------------------------

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Unique staging dir per run + tag (mimo26f-build rule 3: never a fixed path).
pub fn tmp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "mimo26-attn-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).expect("create tmp dir");
    d
}

fn driver() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_driver.py")
}

fn run_driver(args: &[&str]) {
    let out = Command::new("python3")
        .arg(driver())
        .args(args)
        .output()
        .expect("python3 must be on PATH (the ci-cpu gate runs pytest oracle on it)");
    if !out.status.success() {
        panic!(
            "oracle_driver.py {:?} FAILED\n--- stdout ---\n{}\n--- stderr ---\n{}",
            args,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// `oracle_driver.py eval --in <in_dir> --out <out_dir>` — compute the numpy
/// oracle's expected outputs for the input manifest written by the test.
pub fn run_oracle_eval(in_dir: &Path, out_dir: &Path) {
    run_driver(&[
        "eval",
        "--in",
        in_dir.to_str().expect("utf8 path"),
        "--out",
        out_dir.to_str().expect("utf8 path"),
    ]);
}

/// `oracle_driver.py gen --out <out_dir> --shapes ...` — self-contained golden
/// generation (inputs + oracle outputs) for the GPU parity cell / harness
/// selftest.
pub fn run_oracle_gen(out_dir: &Path, shapes: &str) {
    run_driver(&["gen", "--out", out_dir.to_str().expect("utf8 path"), "--shapes", shapes]);
}
