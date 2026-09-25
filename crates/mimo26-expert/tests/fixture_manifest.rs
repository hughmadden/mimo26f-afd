//! Fixture manifest emitter — the bridge between the JSON fixture and the CUDA
//! harness.
//!
//! The CUDA harness (`kernels/parity/gemm_parity.cu`) must not carry a second
//! JSON parser: the Rust side parses `bench/fixtures/expert_nibble_fixture.json`
//! once (`mimo26_expert::fixture`) and writes the flat, line-oriented manifest
//! the harness reads:
//!
//! ```text
//! block <name> <out> <in> <n_samples>
//! sample <row> <col> <hex_bits>
//! ...
//! ```
//!
//! This test is `#[ignore]`d by default (it writes a file); the GPU cell runs it
//! with `--ignored --nocapture` and `MIMO26_EXPERT_FIXTURE_MANIFEST=<path>`.
//!
//! Classification: **BOTH RUNS** (it is a tool, not a trap).

mod common;

use std::io::Write;

use common::*;

/// Emit the manifest for the GPU cell.
#[test]
#[ignore = "writes a manifest file; run by tests/gpu/run_gpu_gemm.sh"]
fn emit_fixture_manifest() {
    let out = std::env::var("MIMO26_EXPERT_FIXTURE_MANIFEST")
        .expect("MIMO26_EXPERT_FIXTURE_MANIFEST must point at the output path");
    let fx = load_fixture().expect("the real-block fixture must load");
    let mut f = std::fs::File::create(&out).expect("create the manifest");
    for b in &fx.blocks {
        writeln!(
            f,
            "block {} {} {} {}",
            b.name,
            b.shape[0],
            b.shape[1],
            b.positions.len()
        )
        .expect("write block line");
        for (i, &(r, c)) in b.positions.iter().enumerate() {
            writeln!(f, "sample {} {} {:08x}", r, c, b.expected_f32_bits[i])
                .expect("write sample line");
        }
    }
    f.flush().expect("flush");
    println!(
        "wrote {} blocks ({} samples) to {out}",
        fx.blocks.len(),
        fx.total_samples()
    );
}

/// BOTH RUNS. The manifest format round-trips: every block line's sample count
/// matches the number of sample lines that follow it.
#[test]
fn manifest_format_is_self_consistent() {
    let fx = load_fixture().expect("fixture");
    let mut text = String::new();
    for b in &fx.blocks {
        text.push_str(&format!(
            "block {} {} {} {}\n",
            b.name,
            b.shape[0],
            b.shape[1],
            b.positions.len()
        ));
        for (i, &(r, c)) in b.positions.iter().enumerate() {
            text.push_str(&format!("sample {} {} {:08x}\n", r, c, b.expected_f32_bits[i]));
        }
    }
    // Re-parse it the way the CUDA harness does.
    let mut blocks = 0usize;
    let mut samples = 0usize;
    let mut declared = 0usize;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("block ") {
            if blocks > 0 {
                assert_eq!(samples, declared, "block {blocks} sample count");
            }
            let f: Vec<&str> = rest.split_whitespace().collect();
            assert_eq!(f.len(), 4, "block line fields");
            declared = f[3].parse().expect("n_samples");
            samples = 0;
            blocks += 1;
        } else if line.starts_with("sample ") {
            samples += 1;
        }
    }
    assert_eq!(samples, declared, "last block sample count");
    assert_eq!(blocks, 27);
    assert_eq!(samples, 2048);
}
