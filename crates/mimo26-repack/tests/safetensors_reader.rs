//! Safetensors reader — header parse, tensor ranges, and the fail-loud rules.
//!
//! Classification: **BOTH-RUNS** (no naive flag).
//!
//! Kills: a reader that trusts a declared length without checking the file, one
//! that accepts a non-U8 dtype, one that accepts a shape that is not the pinned
//! expert geometry, and one that silently returns a missing tensor as empty.

mod common;

use mimo26_repack::geom::{self, Proj};
use mimo26_repack::repack::read_expert;
use mimo26_repack::safetensors::{parse_header_json, SafetensorsHeader};

#[test]
fn reads_a_real_safetensors_file() {
    let dir = common::scratch_dir("st-read");
    let t = common::synth_expert(601);
    let path = common::write_expert_safetensors(&dir, "shard.safetensors", 1, 0, &t);
    let h = SafetensorsHeader::read(&path).expect("read header");
    assert_eq!(h.tensors.len(), 6);
    let e = read_expert(&h, &path, 1, 0).expect("read expert");
    assert_eq!(e, t);
    assert_eq!(e.total_bytes(), geom::EXPERT_BYTES);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn header_parse_handles_metadata_and_whitespace() {
    let text = r#"{
      "__metadata__": {"format": "pt", "nested": {"a": [1, 2, 3]}},
      "model.layers.1.mlp.experts.0.gate_proj.weight": {"dtype": "U8", "shape": [2048, 2048], "data_offsets": [0, 4194304]},
      "model.layers.1.mlp.experts.0.gate_proj.weight_scale": {"dtype": "U8", "shape": [2048, 128], "data_offsets": [4194304, 4456448]}
    }"#;
    let m = parse_header_json(text).expect("parse");
    assert_eq!(m.len(), 2);
    let e = m
        .get("model.layers.1.mlp.experts.0.gate_proj.weight")
        .expect("entry");
    assert_eq!(e.dtype, "U8");
    assert_eq!(e.shape, vec![2048, 2048]);
    assert_eq!(e.data_offsets, (0, 4_194_304));
    assert_eq!(e.byte_len(), 4_194_304);
}

#[test]
fn header_parse_rejects_malformed_json() {
    assert!(parse_header_json("").is_err());
    assert!(parse_header_json("{").is_err());
    assert!(parse_header_json("{} trailing").is_err());
    assert!(parse_header_json(r#"{"a": {"dtype": "U8"}}"#).is_err()); // no shape
    assert!(parse_header_json(r#"{"a": {"shape": [1], "data_offsets": [0, 1]}}"#).is_err());
    assert!(parse_header_json(r#"{"a": {"dtype": "U8", "shape": [1], "data_offsets": [1, 0]}}"#).is_err());
    assert!(parse_header_json(r#"{"a": {"dtype": "U8", "shape": [1], "data_offsets": [0]}}"#).is_err());
}

#[test]
fn reader_rejects_a_truncated_file() {
    let dir = common::scratch_dir("st-truncated");
    let t = common::synth_expert(602);
    let path = common::write_expert_safetensors(&dir, "shard.safetensors", 1, 0, &t);
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 1024]).unwrap();
    let err = SafetensorsHeader::read(&path).unwrap_err();
    assert!(
        matches!(err, mimo26_repack::RepackError::Truncated { .. }),
        "a truncated shard must fail loud, got {err:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn reader_rejects_a_file_shorter_than_the_header_length() {
    let dir = common::scratch_dir("st-short");
    let path = dir.join("short.safetensors");
    std::fs::write(&path, [0u8; 4]).unwrap();
    assert!(SafetensorsHeader::read(&path).is_err());
    std::fs::write(&path, 999_999u64.to_le_bytes()).unwrap();
    assert!(SafetensorsHeader::read(&path).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn reader_rejects_a_missing_tensor() {
    let dir = common::scratch_dir("st-missing");
    let t = common::synth_expert(603);
    let path = common::write_expert_safetensors(&dir, "shard.safetensors", 1, 0, &t);
    let h = SafetensorsHeader::read(&path).expect("read header");
    // Expert 1 is not in this file.
    let err = read_expert(&h, &path, 1, 1).unwrap_err();
    assert!(
        matches!(err, mimo26_repack::RepackError::MissingTensor(_)),
        "a missing tensor must fail loud, got {err:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn reader_rejects_a_wrong_shape() {
    let dir = common::scratch_dir("st-shape");
    let path = dir.join("bad.safetensors");
    // A header that declares gate_proj.weight as [2048, 1024] (half the real
    // width) — the classic "wrong in/2" bug.
    let header = r#"{"model.layers.1.mlp.experts.0.gate_proj.weight":{"dtype":"U8","shape":[2048,1024],"data_offsets":[0,2097152]}}"#;
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&vec![0u8; 2_097_152]);
    std::fs::write(&path, &out).unwrap();
    let h = SafetensorsHeader::read(&path).expect("read header");
    let err = read_expert(&h, &path, 1, 0).unwrap_err();
    assert!(
        matches!(err, mimo26_repack::RepackError::BadGeometry { .. }),
        "a wrong shape must fail loud, got {err:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn reader_rejects_a_non_u8_dtype() {
    let dir = common::scratch_dir("st-dtype");
    let path = dir.join("bad.safetensors");
    let header = r#"{"model.layers.1.mlp.experts.0.gate_proj.weight":{"dtype":"F32","shape":[2048,2048],"data_offsets":[0,16777216]}}"#;
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&vec![0u8; 16_777_216]);
    std::fs::write(&path, &out).unwrap();
    let h = SafetensorsHeader::read(&path).expect("read header");
    let err = read_expert(&h, &path, 1, 0).unwrap_err();
    assert!(
        matches!(err, mimo26_repack::RepackError::BadGeometry { .. }),
        "a non-U8 expert weight must fail loud, got {err:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn reader_rejects_a_dense_layer() {
    let dir = common::scratch_dir("st-layer0");
    let t = common::synth_expert(604);
    let path = common::write_expert_safetensors(&dir, "shard.safetensors", 1, 0, &t);
    let h = SafetensorsHeader::read(&path).expect("read header");
    // Layer 0 is dense: it has no expert tensors, so this must fail loud.
    assert!(read_expert(&h, &path, 0, 0).is_err());
    assert!(read_expert(&h, &path, 48, 0).is_err());
    assert!(read_expert(&h, &path, 1, 256).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tensor_names_are_the_checkpoint_names() {
    // The names the reader looks for must be exactly the checkpoint's.
    for p in Proj::ALL {
        assert_eq!(
            geom::tensor_name(1, 0, p, false),
            format!("model.layers.1.mlp.experts.0.{}.weight", p.name())
        );
        assert_eq!(
            geom::tensor_name(1, 0, p, true),
            format!("model.layers.1.mlp.experts.0.{}.weight_scale", p.name())
        );
    }
}
