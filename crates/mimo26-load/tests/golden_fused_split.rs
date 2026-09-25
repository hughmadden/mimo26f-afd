//! Golden lock — the golden-locked Rust twin (AGENTS.md §4: "Numpy oracle →
//! golden-locked Rust twin → engine"; a golden that imports the code under test
//! only proves self-consistency).
//!
//! The `fused_split{}` case of `code/tests/golden/fp8_block_golden.json` pins
//! `split_shard_major_fused` + the per-shard scale-grid padding trim BYTE-EXACT
//! — the same external corpus the spike pins in
//! `spike/tests/test_t1_fused_qkv_split.py::test_golden_fused_split_byte_exact`
//! (regen `code/scripts/gen-golden.py`, seed 20260922).
//!
//! Two-run classification: BOTH RUNS (the split/trim path has no naive seam —
//! `dequantize_naive_fused` is a separate function). Skips only in
//! `MIMO26_ALLOW_MISSING_GOLDEN=1` mode (mirrors the spike's loud-fail default).

mod common;

use common::*;
use mimo26_load::fused::split_shard_major_fused;
use mimo26_load::Mat;

#[test]
fn golden_fused_split_byte_exact() {
    let Some(text) = load_golden() else {
        return; // skip mode
    };
    let fs = json_object(&text, "fused_split");
    let fused_rows = json_usize_field(fs, "fused_rows");
    let fused = hex_to_bytes(&json_str_field(fs, "fused_u8_hex"));
    let cols = fused.len() / fused_rows;
    let w = Mat::from_row_major(fused_rows, cols, fused).expect("fused shape");
    let scale_shape = json_usize_array_field(fs, "scale_shape");
    let s = Mat::from_row_major(
        scale_shape[0],
        scale_shape[1],
        bytes_to_f32le(&hex_to_bytes(&json_str_field(fs, "scales_f32_hex"))),
    )
    .expect("scale shape");
    let segs = json_usize_array_field(fs, "segments_rows");
    let parts = split_shard_major_fused(&[w], &[s], (segs[0], segs[1], segs[2]), (2, 4))
        .expect("split");
    let parts_json = json_object(fs, "parts");
    for (name, proj) in [("q", &parts.q), ("k", &parts.k), ("v", &parts.v)] {
        let pj = json_object(parts_json, name);
        let shape = json_usize_array_field(pj, "shape");
        assert_eq!(
            (proj.weight.rows, proj.weight.cols),
            (shape[0], shape[1]),
            "{name}: weight shape"
        );
        assert_eq!(
            proj.weight.data,
            hex_to_bytes(&json_str_field(pj, "hex")),
            "{name}: codes must be BYTE-EXACT vs the external golden"
        );
        assert_eq!(
            proj.scale_per_row.data,
            bytes_to_f32le(&hex_to_bytes(&json_str_field(pj, "scale_hex"))),
            "{name}: per-row scales (padding trim) must be BYTE-EXACT vs the external golden"
        );
    }
}
