//! Manifest — format pin, round-trip, and the fail-loud parse rules.
//!
//! Classification: **BOTH-RUNS** (no naive flag).
//!
//! Kills: a manifest that omits the source tensor names or the geometry, a
//! parser that silently accepts a malformed sha, a geometry block that
//! disagrees with the compiled-in constants, and a layout table that drifts.

mod common;

use mimo26_repack::geom::{self, Proj};
use mimo26_repack::manifest::{self, Manifest, SliceEntry, SourceTensor};
use mimo26_repack::sha256;

fn entry_for(file: &str, layer: usize, expert: usize, rank: usize, bytes: &[u8]) -> SliceEntry {
    SliceEntry {
        file: file.to_string(),
        sha256: sha256::hex(&sha256::sha256(bytes)),
        bytes: bytes.len() as u64,
        layer,
        expert,
        rank,
        shard: geom::shard_file(expert),
        tensors: manifest::source_tensors(layer, expert),
    }
}

#[test]
fn manifest_round_trips_through_json() {
    let mut m = Manifest::new();
    let bytes = vec![7u8; geom::QUARTER_SLICE_BYTES];
    m.push(entry_for("L01_E000_R0.slice", 1, 0, 0, &bytes));
    m.push(entry_for("L01_E000_R1.slice", 1, 0, 1, &bytes));
    let json = m.to_json();
    let back = Manifest::parse(&json).expect("parse manifest");
    assert_eq!(back, m, "manifest did not round-trip");
    assert_eq!(back.slices.len(), 2);
    assert_eq!(back.slices[0].file, "L01_E000_R0.slice");
    assert_eq!(back.slices[1].file, "L01_E000_R1.slice");
}

#[test]
fn manifest_carries_the_source_tensor_names_and_shapes() {
    let m = Manifest::new();
    let t = manifest::source_tensors(1, 0);
    assert_eq!(t.len(), 6);
    let names: Vec<&str> = t.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "model.layers.1.mlp.experts.0.gate_proj.weight",
            "model.layers.1.mlp.experts.0.gate_proj.weight_scale",
            "model.layers.1.mlp.experts.0.up_proj.weight",
            "model.layers.1.mlp.experts.0.up_proj.weight_scale",
            "model.layers.1.mlp.experts.0.down_proj.weight",
            "model.layers.1.mlp.experts.0.down_proj.weight_scale",
        ]
    );
    let shapes: Vec<Vec<usize>> = t.iter().map(|s| s.shape.clone()).collect();
    assert_eq!(
        shapes,
        vec![
            vec![2048, 2048],
            vec![2048, 128],
            vec![2048, 2048],
            vec![2048, 128],
            vec![4096, 1024],
            vec![4096, 64],
        ]
    );
    assert!(t.iter().all(|s| s.dtype == "U8"));
    // The geometry block is present and pinned.
    assert_eq!(m.geometry.get("expert_bytes"), Some(&13_369_344));
    assert_eq!(m.geometry.get("quarter_slice_bytes"), Some(&3_342_336));
    assert_eq!(m.geometry.get("moe_layers"), Some(&47));
    assert_eq!(m.geometry.get("experts_per_layer"), Some(&256));
    assert_eq!(m.geometry.get("ep_ranks"), Some(&4));
    // The layout table is in the manifest too.
    assert_eq!(m.layout.len(), 6);
    assert_eq!(m.layout[0].proj, "gate_proj");
    assert_eq!(m.layout[0].region, "payload");
    assert_eq!(m.layout[0].off, 0);
    assert_eq!(m.layout[0].len, 1_048_576);
    assert_eq!(m.layout[5].proj, "down_proj");
    assert_eq!(m.layout[5].region, "scales");
    assert_eq!(m.layout[5].off, 3_276_800);
    assert_eq!(m.layout[5].len, 65_536);
}

#[test]
fn manifest_json_is_stable_byte_for_byte() {
    // The manifest is a receipt: two runs over the same input must produce the
    // same bytes (modulo nothing — the producer string is a constant).
    let mut a = Manifest::new();
    let mut b = Manifest::new();
    let bytes = vec![3u8; geom::QUARTER_SLICE_BYTES];
    a.push(entry_for("L02_E004_R3.slice", 2, 4, 3, &bytes));
    b.push(entry_for("L02_E004_R3.slice", 2, 4, 3, &bytes));
    assert_eq!(a.to_json(), b.to_json());
    assert!(a.to_json().ends_with("}\n"));
}

#[test]
fn manifest_rejects_a_malformed_sha() {
    let mut m = Manifest::new();
    let bytes = vec![0u8; geom::QUARTER_SLICE_BYTES];
    m.push(entry_for("L01_E000_R0.slice", 1, 0, 0, &bytes));
    let json = m.to_json().replace(
        &sha256::hex(&sha256::sha256(&bytes)),
        &"z".repeat(64),
    );
    let err = Manifest::parse(&json).unwrap_err();
    assert!(
        format!("{err}").contains("bad sha256"),
        "a malformed sha must fail loud, got {err}"
    );
}

#[test]
fn manifest_rejects_a_short_sha() {
    let mut m = Manifest::new();
    let bytes = vec![0u8; geom::QUARTER_SLICE_BYTES];
    m.push(entry_for("L01_E000_R0.slice", 1, 0, 0, &bytes));
    let json = m.to_json().replace(
        &sha256::hex(&sha256::sha256(&bytes)),
        &"a".repeat(63),
    );
    assert!(Manifest::parse(&json).is_err());
}

#[test]
fn manifest_rejects_a_geometry_drift() {
    let m = Manifest::new();
    let json = m.to_json().replace("\"moe_layers\": 47", "\"moe_layers\": 48");
    let err = Manifest::parse(&json).unwrap_err();
    assert!(
        matches!(err, mimo26_repack::RepackError::GeometryMismatch { field: "moe_layers", .. }),
        "a geometry drift must fail loud, got {err:?}"
    );
    // And the slice size.
    let json = m
        .to_json()
        .replace("\"quarter_slice_bytes\": 3342336", "\"quarter_slice_bytes\": 3342337");
    assert!(Manifest::parse(&json).is_err());
}

#[test]
fn manifest_rejects_a_layout_drift() {
    let m = Manifest::new();
    let json = m
        .to_json()
        .replace("\"off\": 1048576, \"len\": 65536", "\"off\": 1048577, \"len\": 65536");
    let err = Manifest::parse(&json).unwrap_err();
    assert!(
        format!("{err}").contains("layout row drift"),
        "a layout drift must fail loud, got {err}"
    );
}

#[test]
fn manifest_rejects_a_wrong_format_or_version() {
    let m = Manifest::new();
    let json = m.to_json().replace("mimo26-repack-manifest", "something-else");
    assert!(Manifest::parse(&json).is_err());
    let json = m.to_json().replace("\"version\": 2", "\"version\": 1");
    assert!(Manifest::parse(&json).is_err());
}

#[test]
fn manifest_rejects_missing_fields_and_trailing_bytes() {
    let m = Manifest::new();
    let json = m.to_json();
    // Drop the slices array entirely.
    let cut = json.replace("  \"slices\": [\n  ]\n", "");
    assert!(Manifest::parse(&cut).is_err());
    // Trailing garbage.
    assert!(Manifest::parse(&format!("{json}garbage")).is_err());
    // Duplicate keys.
    let dup = json.replace(
        "  \"version\": 2,",
        "  \"version\": 2,\n  \"version\": 2,",
    );
    assert!(Manifest::parse(&dup).is_err());
}

#[test]
fn manifest_writes_and_reads_from_disk() {
    let dir = common::scratch_dir("manifest");
    let mut m = Manifest::new();
    let bytes = vec![9u8; geom::QUARTER_SLICE_BYTES];
    m.push(entry_for("L01_E000_R0.slice", 1, 0, 0, &bytes));
    let path = dir.join("manifest.json");
    m.write(&path).expect("write manifest");
    let back = Manifest::read(&path).expect("read manifest");
    assert_eq!(back, m);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn manifest_entries_are_sorted_by_file_name() {
    let mut m = Manifest::new();
    let bytes = vec![0u8; geom::QUARTER_SLICE_BYTES];
    m.push(entry_for("L02_E000_R0.slice", 2, 0, 0, &bytes));
    m.push(entry_for("L01_E000_R0.slice", 1, 0, 0, &bytes));
    m.push(entry_for("L01_E000_R1.slice", 1, 0, 1, &bytes));
    let files: Vec<&str> = m.slices.iter().map(|s| s.file.as_str()).collect();
    assert_eq!(
        files,
        vec!["L01_E000_R0.slice", "L01_E000_R1.slice", "L02_E000_R0.slice"]
    );
    assert!(m.get("L01_E000_R1.slice").is_some());
    assert!(m.get("nope.slice").is_none());
}

#[test]
fn source_tensor_shapes_match_the_projection_geometry() {
    for p in Proj::ALL {
        let t: Vec<SourceTensor> = manifest::source_tensors(5, 9)
            .into_iter()
            .filter(|s| s.name.contains(p.name()))
            .collect();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].shape, vec![p.out_rows(), p.in_cols() / 2]);
        assert_eq!(t[1].shape, vec![p.out_rows(), p.in_cols() / 32]);
    }
}
