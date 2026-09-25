//! BOTH RUNS — harness selftest (TEST-PLAN R6: "harness selftests before the
//! harness runs a model"): the `oracle_driver.py gen` path that feeds
//! `kernels/parity/attn_parity.cu` must produce a well-formed manifest with
//! inputs AND oracle outputs, from a clean staging dir. No GPU here — this is
//! the plumbing the captain-fired `scripts/dev.sh test attn` cell stands on.

mod common;

use common::*;

/// `gen --shapes tiny` emits every case class the GPU harness knows
/// (`attn`/`decode`/`prefill`/`rope`/`kv_store_*`) with its expect tensors.
#[test]
fn gen_tiny_manifest_is_well_formed() {
    let out_dir = tmp_dir("gen-tiny");
    run_oracle_gen(&out_dir, "tiny");
    let cases = parse_manifest(&out_dir.join("manifest.txt"));
    assert!(!cases.is_empty(), "gen must emit at least one case");
    let mut seen_types = Vec::new();
    for c in &cases {
        assert!(!c.tensors.is_empty(), "case {} has no tensors", c.name);
        let expects: Vec<&str> = c
            .tensors
            .iter()
            .filter(|t| t.is_expect)
            .map(|t| t.name.as_str())
            .collect();
        assert!(!expects.is_empty(), "case {} has no expect tensors", c.name);
        // every referenced file exists and has the declared element count
        for t in &c.tensors {
            let data = read_bin_u8(&out_dir, &t.file);
            let n: usize = t.shape.iter().product();
            let width = match t.dtype.as_str() {
                "f32" => 4,
                "i64" => 8,
                "u8" => 1,
                other => panic!("unknown dtype {other}"),
            };
            assert_eq!(data.len(), n * width, "case {}: {} byte length", c.name, t.name);
        }
        if !seen_types.contains(&c.ty.as_str()) {
            seen_types.push(c.ty.as_str());
        }
    }
    for want in ["attn", "decode", "prefill", "rope", "kv_store_unit", "kv_store_pth"] {
        assert!(seen_types.contains(&want), "gen tiny must cover case type {want}");
    }
}
