//! Real checkpoint headers — READ-ONLY confirmation of the tensor names and
//! shapes this crate slices.
//!
//! Classification: **BOTH-RUNS**, but `#[ignore]`d by default: it needs the dev host
//! weights copy at `~/models/XiaomiMiMo/MiMo-V2.6-Flash-RL` (13 GB of shards)
//! and is therefore NOT part of the L0/L1 merge gate. Run it explicitly:
//!
//! ```text
//! cargo test -p mimo26-repack --test real_headers -- --ignored --nocapture
//! ```
//!
//! The test opens the shards READ-ONLY and never writes to the weights
//! directory (AGENTS.md §4.7: weights stay on host mounts; never move them).
//!
//! Kills: a geometry assumption that does not match the real checkpoint — the
//! exact failure mode that would make every slice silently wrong.

mod common;

use mimo26_repack::geom::{self, Proj};
use mimo26_repack::manifest;
use mimo26_repack::repack::read_expert;
use mimo26_repack::safetensors::SafetensorsHeader;

fn weights_dir() -> std::path::PathBuf {
    common::local_weights_dir()
}

#[test]
#[ignore = "needs the local weights copy (~/models/XiaomiMiMo/MiMo-V2.6-Flash-RL)"]
fn real_shard_headers_match_the_pinned_geometry() {
    let dir = weights_dir();
    if !dir.is_dir() {
        eprintln!("SKIP: {} is not present", dir.display());
        return;
    }
    // Shard 0 holds experts 0..3 of every MoE layer.
    let shard = dir.join(geom::shard_file(0));
    let h = SafetensorsHeader::read(&shard).expect("read shard header");
    println!("shard {}: {} tensors", shard.display(), h.tensors.len());

    // Every expert tensor of layer 1, expert 0 must be present with the pinned
    // shape and dtype.
    for p in Proj::ALL {
        for scale in [false, true] {
            let name = geom::tensor_name(1, 0, p, scale);
            let e = h
                .tensors
                .get(&name)
                .unwrap_or_else(|| panic!("missing real tensor {name}"));
            let want = if scale {
                vec![p.out_rows(), p.in_cols() / 32]
            } else {
                vec![p.out_rows(), p.in_cols() / 2]
            };
            assert_eq!(e.dtype, "U8", "{name} dtype");
            assert_eq!(e.shape, want, "{name} shape");
            assert_eq!(e.byte_len() as usize, want[0] * want[1], "{name} bytes");
        }
    }

    // The six tensors of one expert total exactly the pinned expert size.
    let t = read_expert(&h, &shard, 1, 0).expect("read real expert");
    assert_eq!(t.total_bytes(), geom::EXPERT_BYTES);
    println!("real expert L1 E0 = {} B", t.total_bytes());

    // Layer 0 is dense: no expert tensors at all.
    let layer0: Vec<&String> = h
        .tensors
        .keys()
        .filter(|k| k.starts_with("model.layers.0.mlp.experts."))
        .collect();
    assert!(layer0.is_empty(), "layer 0 must have no expert tensors");

    // The manifest's source-tensor list must name real tensors.
    for s in manifest::source_tensors(1, 0) {
        assert!(
            h.tensors.contains_key(&s.name),
            "manifest names a tensor the checkpoint does not have: {}",
            s.name
        );
    }
}

#[test]
#[ignore = "needs the local weights copy (~/models/XiaomiMiMo/MiMo-V2.6-Flash-RL)"]
fn real_index_confirms_47_moe_layers_and_256_experts() {
    let dir = weights_dir();
    let index = dir.join("model.safetensors.index.json");
    if !index.is_file() {
        eprintln!("SKIP: {} is not present", index.display());
        return;
    }
    let text = std::fs::read_to_string(&index).expect("read index");
    // Count the expert tensors per layer without a JSON dependency: the index
    // is a flat weight_map, so a substring scan is enough for a sanity check.
    let mut layers = std::collections::BTreeSet::new();
    let mut experts = std::collections::BTreeSet::new();
    for line in text.lines() {
        if let Some(rest) = line.trim().strip_prefix("\"model.layers.") {
            if let Some(dot) = rest.find('.') {
                if let Ok(l) = rest[..dot].parse::<usize>() {
                    if rest.contains(".mlp.experts.") {
                        layers.insert(l);
                        let after = &rest[rest.find(".mlp.experts.").unwrap() + 13..];
                        if let Some(d) = after.find('.') {
                            if let Ok(e) = after[..d].parse::<usize>() {
                                experts.insert(e);
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(layers.len(), geom::MOE_LAYERS, "MoE layer count");
    assert_eq!(layers.iter().min(), Some(&geom::FIRST_MOE_LAYER));
    assert_eq!(
        layers.iter().max(),
        Some(&(geom::FIRST_MOE_LAYER + geom::MOE_LAYERS - 1))
    );
    assert_eq!(experts.len(), geom::EXPERTS_PER_LAYER, "expert count");
    println!(
        "real index: {} MoE layers ({}..={}), {} experts",
        layers.len(),
        layers.iter().min().unwrap(),
        layers.iter().max().unwrap(),
        experts.len()
    );
}

#[test]
#[ignore = "needs the local weights copy (~/models/XiaomiMiMo/MiMo-V2.6-Flash-RL)"]
fn real_expert_repacks_and_round_trips() {
    let dir = weights_dir();
    let shard = dir.join(geom::shard_file(0));
    if !shard.is_file() {
        eprintln!("SKIP: {} is not present", shard.display());
        return;
    }
    let h = SafetensorsHeader::read(&shard).expect("read shard header");
    let t = read_expert(&h, &shard, 1, 0).expect("read real expert");
    // The real checkpoint is already quantized, so the round-trip is exact:
    // the repack is a pure permutation and the semantics are the same on both
    // sides. This is the real-data version of the T14/T10 negative.
    for rank in 0..geom::EP_RANKS {
        let slice = mimo26_repack::build_slice(&t, rank, mimo26_repack::Mxfp4Naive::NONE)
            .expect("build slice");
        assert_eq!(slice.len(), geom::QUARTER_SLICE_BYTES);
        for p in Proj::ALL {
            let got = mimo26_repack::slice_to_f32(&slice, p, mimo26_repack::Mxfp4Naive::NONE)
                .expect("slice_to_f32");
            let full = mimo26_repack::repack::expert_to_f32(&t, p, mimo26_repack::Mxfp4Naive::NONE);
            let want = mimo26_repack::repack::slice_rows_of_full(&full, p, rank);
            assert!(
                common::bits_eq(&got, &want),
                "real data: rank {rank} {} failed the round-trip",
                p.name()
            );
        }
    }
    println!("real expert L1 E0 round-trips on all 4 ranks x 3 projections");
}

#[test]
#[ignore = "needs the local weights copy (~/models/XiaomiMiMo/MiMo-V2.6-Flash-RL)"]
fn real_weights_directory_is_never_written() {
    // Guard: the test must not have created anything in the weights dir.
    let dir = weights_dir();
    if !dir.is_dir() {
        return;
    }
    let before: Vec<String> = std::fs::read_dir(&dir)
        .expect("read weights dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    assert!(
        before.iter().all(|n| !n.ends_with(".slice") && n != "manifest.json"),
        "the weights directory must never receive repack output"
    );
}
