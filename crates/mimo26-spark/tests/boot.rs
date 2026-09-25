//! Daemon boot identity readback + slice residency (CPU-only).
//!
//! Classification: **BOTH-RUNS** (no naive flag). Kills a Spark that serves a
//! corrupted/truncated/missing slice, one that serves a foreign resident slice,
//! and one that loads another rank's slices.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use mimo26_repack::geom;
use mimo26_repack::manifest::{Manifest, SliceEntry};
use mimo26_repack::sha256;
use mimo26_spark::boot;
use mimo26_spark::resident::Resident;

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn scratch_dir(name: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("mimo26-spark-{name}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Write `(layer, expert, rank)` slice files and the matching manifest.
fn write_slices(dir: &Path, specs: &[(usize, usize, usize)]) -> Manifest {
    let mut m = Manifest::new();
    for (i, &(layer, expert, rank)) in specs.iter().enumerate() {
        let file = geom::slice_file_name(layer, expert, rank);
        let mut bytes = vec![0u8; geom::QUARTER_SLICE_BYTES];
        bytes[0] = i as u8;
        bytes[geom::QUARTER_SLICE_BYTES - 1] = (i as u8).wrapping_mul(3);
        std::fs::write(dir.join(&file), &bytes).expect("write slice");
        m.push(SliceEntry {
            file,
            sha256: sha256::hex(&sha256::sha256(&bytes)),
            bytes: bytes.len() as u64,
            layer,
            expert,
            rank,
            shard: geom::shard_file(expert),
            tensors: mimo26_repack::manifest::source_tensors(layer, expert),
        });
    }
    m.write(&dir.join("manifest.json")).expect("write manifest");
    m
}

const QUARTER: usize = 3_342_336;

#[test]
fn clean_boot_matches_every_slice() {
    let dir = scratch_dir("boot-clean");
    let specs = [(1, 0, 0), (1, 1, 1), (2, 0, 0), (2, 1, 1)];
    write_slices(&dir, &specs);
    let receipt = boot::readback(&dir).expect("clean readback");
    assert_eq!(receipt.matched, 4);
    assert_eq!(receipt.total, 4);
    assert_eq!(receipt.matched_bytes, 4 * QUARTER as u64);
    assert!(receipt.summary.contains("4/4"), "{}", receipt.summary);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn corrupt_slice_refuses_to_serve() {
    let dir = scratch_dir("boot-corrupt");
    let specs = [(1, 0, 0), (1, 1, 1)];
    write_slices(&dir, &specs);
    let victim = dir.join("L01_E001_R1.slice");
    let mut bytes = std::fs::read(&victim).expect("read slice");
    bytes[1_000_000] ^= 0x01;
    std::fs::write(&victim, &bytes).expect("write corrupt");
    let err = boot::readback(&dir).unwrap_err();
    assert!(format!("{err}").contains("refusing to serve"), "got {err}");
    assert!(format!("{err}").contains("L01_E001_R1.slice"), "got {err}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn unlisted_slice_refuses_to_serve() {
    let dir = scratch_dir("boot-unlisted");
    let specs = [(1, 0, 0)];
    write_slices(&dir, &specs);
    std::fs::write(dir.join("L47_E255_R3.slice"), vec![0u8; QUARTER]).expect("stray");
    let err = boot::readback(&dir).unwrap_err();
    assert!(format!("{err}").contains("not listed"), "got {err}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_rank_loads_only_its_rank() {
    let dir = scratch_dir("resident-rank");
    let specs = [
        (1, 0, 0),
        (1, 1, 0),
        (1, 2, 1),
        (2, 0, 0),
        (2, 1, 1),
        (2, 2, 1),
    ];
    write_slices(&dir, &specs);
    let resident = Resident::load_manifest(&dir, 0).expect("load rank 0");
    assert_eq!(resident.slices, 3, "rank 0 owns 3 of 6 slices");
    for &(layer, expert) in &[(1usize, 0usize), (1, 1), (2, 0)] {
        let s = resident.slice(layer, expert).expect("resident slice");
        assert_eq!(s.len(), QUARTER);
    }
    // A rank-1 slice is not resident on rank 0.
    assert!(resident.slice(1, 2).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn grouped_image_is_the_expert_ordered_placement() {
    let dir = scratch_dir("resident-grouped");
    let specs = [(1, 0, 0), (1, 1, 0), (1, 2, 0), (2, 0, 0)];
    write_slices(&dir, &specs);
    let resident = Resident::load_manifest(&dir, 0).expect("load rank 0");

    // TP4 placement: layer 1's three experts concatenated in expert-id order.
    let image = resident.grouped_image(1, 3).expect("grouped image");
    assert_eq!(image.len(), 3 * QUARTER);
    for e in 0..3 {
        let part = &image[e * QUARTER..(e + 1) * QUARTER];
        assert_eq!(part, resident.slice(1, e).expect("slice"), "expert {e} misplaced");
    }
    // A missing expert fails loud (no silent short image).
    assert!(resident.grouped_image(1, 4).is_err());
    std::fs::remove_dir_all(&dir).ok();
}
