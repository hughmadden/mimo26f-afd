//! Launch geometry + slice layout — the kernel's contract, pinned on CPU.
//!
//! The CUDA kernel's `__launch_bounds__`, grid computation and slice offsets
//! must match these numbers exactly; a drift is a silent wrong answer, not a
//! crash. `kernels/expert_gemm.cu` carries the same constants in
//! `mimo26_expert_kernels.h` and `proj_geom()`.
//!
//! Classification: every test here is **BOTH RUNS** (the geometry is not a
//! trap; the traps are T10/T14/padding, tested elsewhere).

mod common;

use common::*;
use mimo26_expert::grouped::{self, GroupedPlan, K_STEP, K_TILE, ROWS_PER_BLOCK, LANES_PER_ROW, THREADS, WARPS};
use mimo26_expert::slice::{self, Proj};

/// BOTH RUNS. The pinned launch constants.
#[test]
fn launch_constants_are_pinned() {
    assert_eq!(THREADS, 256);
    assert_eq!(ROWS_PER_BLOCK, 32);
    assert_eq!(K_STEP, 32, "one E8M0 scale block");
    assert_eq!(K_TILE, 256);
    assert_eq!(WARPS, 8);
    assert_eq!(LANES_PER_ROW, 8);
    assert_eq!(THREADS / 32, WARPS);
    assert_eq!(THREADS / LANES_PER_ROW, ROWS_PER_BLOCK);
}

/// BOTH RUNS. The grid covers every output row of every projection.
#[test]
fn grid_covers_every_output_row() {
    for p in Proj::ALL {
        for &experts in &[1usize, 4, 256] {
            let g = grouped::launch_geometry(p, 8, experts);
            assert_eq!(g.grid_y, experts as u32);
            assert_eq!(g.block_x, THREADS as u32);
            assert_eq!(g.rows_per_block, ROWS_PER_BLOCK as u32);
            assert_eq!(g.k_step, K_STEP as u32);
            let covered = u64::from(g.grid_x) * ROWS_PER_BLOCK as u64;
            assert!(
                covered >= p.slice_rows() as u64,
                "{}: grid_x {} x {} rows covers {covered} < {}",
                p.name(),
                g.grid_x,
                ROWS_PER_BLOCK,
                p.slice_rows()
            );
            // And not more than one block's worth of waste.
            assert!(covered - p.slice_rows() as u64 <= ROWS_PER_BLOCK as u64);
        }
    }
    assert_eq!(grouped::launch_geometry(Proj::Gate, 1, 1).grid_x, 16);
    assert_eq!(grouped::launch_geometry(Proj::Up, 1, 1).grid_x, 16);
    assert_eq!(grouped::launch_geometry(Proj::Down, 1, 1).grid_x, 128);
    assert_eq!(grouped::launch_geometry(Proj::Down, 64, 1).grid_z, 8);
    assert_eq!(grouped::launch_geometry(Proj::Down, 0, 1).blocks(), 0);
}

/// BOTH RUNS. K256 double buffering, never a full-K shared allocation.
#[test]
fn token_stage_bytes_are_pinned() {
    for (m,kib) in [(0,0),(1,2),(2,4),(3,8),(4,8),(8,16),(16,16),(64,16)] {
        for cols in [512,4096] {
            assert_eq!(grouped::token_stage_bytes(m,cols),kib*1024);
            assert!(grouped::token_stage_bytes(m,cols)<=16*1024);
        }
    }
}

/// BOTH RUNS. The slice layout: every offset and the total, exactly as the
/// repack crate pins them (`crates/mimo26-repack/src/geom.rs`).
#[test]
fn slice_layout_offsets_are_pinned() {
    assert_eq!(slice::LAYOUT_VERSION, 2);
    assert!(slice::check_layout_version(2).is_ok());
    assert!(slice::check_layout_version(1).is_err());
    assert!(slice::check_layout_version(0).is_err());
    assert_eq!(slice::HIDDEN, 4096);
    assert_eq!(slice::INTERMEDIATE, 2048);
    assert_eq!(slice::EXPERTS_PER_LAYER, 256);
    assert_eq!(slice::MOE_LAYERS, 47);
    assert_eq!(slice::EP_RANKS, 4);
    assert_eq!(slice::EXPERT_BYTES, 13_369_344);
    assert_eq!(slice::QUARTER_SLICE_BYTES, 3_342_336);
    assert_eq!(slice::EXPERT_BYTES / 4, slice::QUARTER_SLICE_BYTES);

    assert_eq!(Proj::Gate.slice_rows(), 512);
    assert_eq!(Proj::Up.slice_rows(), 512);
    assert_eq!(Proj::Down.slice_rows(), 4096);
    assert_eq!(Proj::Down.slice_in_cols(), 512);
    assert_eq!(Proj::Gate.in_cols(), 4096);
    assert_eq!(Proj::Down.in_cols(), 2048);

    assert_eq!(Proj::Gate.slice_payload_off(), 0);
    assert_eq!(Proj::Gate.slice_payload_bytes(), 1_048_576);
    assert_eq!(Proj::Gate.slice_scale_off(), 1_048_576);
    assert_eq!(Proj::Gate.slice_scale_bytes(), 65_536);
    assert_eq!(Proj::Up.slice_payload_off(), 1_114_112);
    assert_eq!(Proj::Up.slice_scale_off(), 2_162_688);
    assert_eq!(Proj::Down.slice_payload_off(), 2_228_224);
    assert_eq!(Proj::Down.slice_payload_bytes(), 1_048_576);
    assert_eq!(Proj::Down.slice_scale_off(), 3_276_800);
    assert_eq!(Proj::Down.slice_scale_bytes(), 65_536);
    assert_eq!(Proj::Down.reserved_bytes(), 0);
    assert_eq!(
        Proj::Down.slice_scale_off() + Proj::Down.slice_scale_bytes() + Proj::Down.reserved_bytes(),
        slice::QUARTER_SLICE_BYTES
    );

    // The layout table is contiguous and covers the slice.
    let table = slice::layout_table();
    let mut off = 0usize;
    for row in &table {
        assert_eq!(row.off, off, "{:?} {} is not contiguous", row.proj, row.region);
        off += row.len;
    }
    assert_eq!(off, slice::QUARTER_SLICE_BYTES);
}

/// BOTH RUNS. V2 shards contiguous intermediates, never output rows of down.
#[test]
fn slice_row_indices_are_the_tp4_contiguous_split() {
    for rank in 0..4 {
        for p in [Proj::Gate, Proj::Up] {
            assert_eq!(slice::slice_row_indices(p, rank), (rank*512..(rank+1)*512).collect::<Vec<_>>());
        }
        assert_eq!(slice::slice_row_indices(Proj::Down, rank), (0..4096).collect::<Vec<_>>());
    }
}

/// BOTH RUNS. A slice image round-trips through `slice_proj` at the pinned
/// offsets, and a wrong-size slice fails loud.
#[test]
fn slice_proj_reads_the_pinned_regions() {
    let s = synth_slice(9001);
    let image = s.to_slice();
    assert_eq!(image.len(), slice::QUARTER_SLICE_BYTES);
    for p in Proj::ALL {
        let (payload, scales) = slice::slice_proj(&image, p).expect("slice_proj");
        let b = s.proj(p);
        assert_eq!(payload, &b.payload[..], "{} payload", p.name());
        assert_eq!(scales, &b.scales[..], "{} scales", p.name());
    }
    // A truncated slice refuses.
    let short = vec![0u8; slice::QUARTER_SLICE_BYTES - 1];
    let err = slice::slice_proj(&short, Proj::Gate).unwrap_err();
    assert!(err.to_string().contains("slice size"), "got: {err}");
}

/// BOTH RUNS. The grouped image addressing: expert e's slice is at
/// `e * QUARTER_SLICE_BYTES`, and an out-of-range expert fails loud.
#[test]
fn grouped_proj_addresses_each_expert_slice() {
    let a = synth_slice(9101);
    let b = synth_slice(9102);
    let image = grouped_image(&[a.clone(), b.clone()]);
    assert_eq!(image.len(), 2 * slice::QUARTER_SLICE_BYTES);
    for p in Proj::ALL {
        let (pa, sa) = slice::grouped_proj(&image, 0, p).expect("expert 0");
        assert_eq!(pa, &a.proj(p).payload[..]);
        assert_eq!(sa, &a.proj(p).scales[..]);
        let (pb, sb) = slice::grouped_proj(&image, 1, p).expect("expert 1");
        assert_eq!(pb, &b.proj(p).payload[..]);
        assert_eq!(sb, &b.proj(p).scales[..]);
    }
    assert!(slice::grouped_proj(&image, 2, Proj::Gate).is_err());
}

/// BOTH RUNS. The streamed-bytes and FLOP accounting the bandwidth harness
/// divides by — the §3.1 arithmetic.
#[test]
fn streamed_bytes_and_flops_are_the_section_3_1_arithmetic() {
    let plan = GroupedPlan::uniform(256, 8, 0);
    // gate: 256 experts x (524,288 + 65,536) B = 150,994,944 B of weights.
    let weights = 256u64 * (1_048_576 + 65_536);
    let tokens = 2048u64 * 4096 * 4;
    let out = 2048u64 * 512 * 4;
    assert_eq!(
        grouped::streamed_bytes(&plan, Proj::Gate, 8),
        weights + tokens + out
    );
    // FLOPs: 2 x 2048 x 512 x 4096.
    assert_eq!(grouped::gemm_flops(&plan, Proj::Gate), 2 * 2048 * 512 * 4096);
    // The full FFN's resident bytes per Spark: 47 layers x 256 experts x 3.34 MB.
    let per_expert = (1_048_576u64 + 65_536) * 2 + (1_048_576 + 65_536);
    assert_eq!(per_expert, 3_342_336);
    // 3,342,336 x 256 x 47 = 40,214,986,752 B = 40.2 GB (ADVISOR-I4 §3.1).
    assert_eq!(per_expert * 256 * 47, 40_214_986_752);
}

/// BOTH RUNS. The §3.1 model's resident bytes per Spark is 40.2 GB.
#[test]
fn resident_bytes_per_spark_is_40_2_gb() {
    let b = mimo26_expert::bench::resident_bytes_per_spark();
    assert_eq!(b, 3_342_336u64 * 256 * 47);
    let gb = b as f64 / 1e9;
    assert!(
        (40.1..40.3).contains(&gb),
        "resident bytes per Spark must be ~40.2 GB, got {gb:.3} GB"
    );
}
