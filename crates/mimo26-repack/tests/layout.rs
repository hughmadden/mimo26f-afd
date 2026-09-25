//! Layout-v2 pins. Both runs: no naive flags here. Do not infer layout from size.
mod common;
use mimo26_repack::geom::{self, Proj};
use mimo26_repack::mxfp4::Mxfp4Naive;
use mimo26_repack::repack::{build_slice, slice_proj};

#[test]
fn expert_and_slice_sizes_are_the_advisor_numbers() {
    assert_eq!(geom::EXPERT_BYTES, 13_369_344);
    assert_eq!(geom::QUARTER_SLICE_BYTES, 3_342_336);
    assert_eq!(geom::EXPERT_BYTES / 4, geom::QUARTER_SLICE_BYTES);
    assert_eq!(3 * 2048 * 4096 * 17 / 32, geom::EXPERT_BYTES);
}

#[test]
fn layout_table_is_the_pinned_table() {
    let table = geom::layout_table();
    let got: Vec<_> = table.iter().map(|r| (r.proj.name(), r.region, r.off, r.len)).collect();
    assert_eq!(got, vec![
        ("gate_proj", "payload", 0, 1_048_576),
        ("gate_proj", "scales", 1_048_576, 65_536),
        ("up_proj", "payload", 1_114_112, 1_048_576),
        ("up_proj", "scales", 2_162_688, 65_536),
        ("down_proj", "payload", 2_228_224, 1_048_576),
        ("down_proj", "scales", 3_276_800, 65_536),
    ]);
    let mut cursor = 0;
    for r in table { assert_eq!(r.off, cursor); cursor += r.len; }
    assert_eq!(cursor, geom::QUARTER_SLICE_BYTES);
}

#[test]
fn per_projection_slice_geometry() {
    for p in [Proj::Gate, Proj::Up] {
        assert_eq!((p.slice_rows(), p.slice_in_cols()), (512, 4096));
    }
    assert_eq!((Proj::Down.slice_rows(), Proj::Down.slice_in_cols()), (4096, 512));
    for p in Proj::ALL {
        assert_eq!(p.slice_payload_bytes(), 1_048_576);
        assert_eq!(p.slice_scale_bytes(), 65_536);
        for rank in 0..4 { assert_eq!(p.slice_col_start(rank) % 32, 0); }
    }
}

#[test]
fn contiguous_intermediate_partition_covers_exactly_once() {
    for p in [Proj::Gate, Proj::Up] {
        let mut all = Vec::new();
        for rank in 0..4 {
            let rows = geom::slice_row_indices(p, rank);
            assert_eq!(rows, (rank * 512..(rank+1) * 512).collect::<Vec<_>>());
            all.extend(rows);
        }
        assert_eq!(all, (0..2048).collect::<Vec<_>>());
    }
    for rank in 0..4 {
        assert_eq!(geom::slice_row_indices(Proj::Down, rank), (0..4096).collect::<Vec<_>>());
        assert_eq!(Proj::Down.slice_col_start(rank), rank * 512);
    }
}

#[test]
fn slice_is_exactly_the_pinned_size_and_tiles_without_padding() {
    let t = common::synth_expert(11);
    for rank in 0..4 {
        assert_eq!(build_slice(&t, rank, Mxfp4Naive::NONE).unwrap().len(), 3_342_336);
    }
    assert_eq!(Proj::Down.slice_scale_off() + Proj::Down.slice_scale_bytes(), 3_342_336);
}

#[test]
fn slice_proj_round_trips_the_regions() {
    let t = common::synth_expert(12);
    for rank in 0..4 {
        let image = build_slice(&t, rank, Mxfp4Naive::NONE).unwrap();
        for p in Proj::ALL {
            let (payload, scales) = slice_proj(&image, p).unwrap();
            let (w, s) = t.proj(p);
            // Independently spell out v2 source coordinates rather than sharing
            // the repacker's index helpers with its expected-answer path.
            let (rows, cols, source_row0, source_k0) = match p {
                Proj::Gate | Proj::Up => (512, 4096, rank * 512, 0),
                Proj::Down => (4096, 512, 0, rank * 512),
            };
            for row in 0..rows {
                for unit in [2, 32] {
                    let (src, dst) = if unit == 2 { (w, payload) } else { (s, scales) };
                    let src0 = (source_row0 + row) * (p.in_cols() / unit) + source_k0 / unit;
                    let dst0 = row * (cols / unit);
                    assert_eq!(&dst[dst0..dst0 + cols/unit], &src[src0..src0 + cols/unit],
                        "{p:?} rank {rank} row {row} unit {unit}");
                }
            }
        }
    }
}

#[test]
fn slice_size_rejects_a_wrong_length() {
    assert!(slice_proj(&vec![0; geom::QUARTER_SLICE_BYTES - 1], Proj::Gate).is_err());
    let mut t = common::synth_expert(13);
    t.down_s.pop();
    assert!(build_slice(&t, 0, Mxfp4Naive::NONE).is_err());
}

#[test]
fn rank_and_layer_and_expert_bounds_are_checked() {
    assert!(geom::check_rank(3).is_ok());
    assert!(geom::check_rank(4).is_err());
    assert!(geom::check_layer(1).is_ok());
    assert!(geom::check_layer(47).is_ok());
    assert!(geom::check_layer(0).is_err());
    assert!(geom::check_layer(48).is_err());
    assert!(geom::check_expert(255).is_ok());
    assert!(geom::check_expert(256).is_err());
}

#[test]
fn shard_and_file_names_follow_the_checkpoint() {
    assert_eq!(geom::shard_file(0), "model_pp0_ep0_shard0.safetensors");
    assert_eq!(geom::shard_file(3), "model_pp0_ep0_shard0.safetensors");
    assert_eq!(geom::shard_file(4), "model_pp0_ep1_shard0.safetensors");
    assert_eq!(geom::shard_file(255), "model_pp0_ep63_shard0.safetensors");
    assert_eq!(geom::slice_file_name(47,255,3), "L47_E255_R3.slice");
    assert_eq!(geom::tensor_name(47,255,Proj::Down,true), "model.layers.47.mlp.experts.255.down_proj.weight_scale");
}
