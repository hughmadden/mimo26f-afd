//! Identity readback — the I5 G0 boot check.
//!
//! Classification: **BOTH-RUNS** (no naive flag).
//!
//! Kills: a verifier that reports "ok" on a corrupted slice, one that misses a
//! truncated file, one that ignores a missing file, one that ignores an
//! unlisted resident file, and one that stops at the first failure instead of
//! naming every bad slice.

mod common;

use std::path::Path;

use mimo26_repack::geom;
use mimo26_repack::identity::{verify_bytes, verify_dir, FileStatus};
use mimo26_repack::manifest::{Manifest, SliceEntry};
use mimo26_repack::sha256;

/// Write `n` slice files into `dir` and return the matching manifest.
fn write_slices(dir: &Path, n: usize) -> Manifest {
    let mut m = Manifest::new();
    for i in 0..n {
        let layer = 1 + i / 4;
        let expert = i % 4;
        let rank = i % geom::EP_RANKS;
        let file = geom::slice_file_name(layer, expert, rank);
        // Distinct content per file so a swap is detectable.
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
    m
}

#[test]
fn clean_readback_matches_every_file() {
    let dir = common::scratch_dir("identity-clean");
    let m = write_slices(&dir, 8);
    let report = verify_dir(&dir, &m).expect("verify");
    assert!(report.all_match(), "clean readback must match: {}", report.summary());
    assert_eq!(report.matched(), 8);
    assert_eq!(report.failed(), 0);
    assert_eq!(
        report.matched_bytes,
        8 * geom::QUARTER_SLICE_BYTES as u64
    );
    assert!(report.clone().into_result().is_ok());
    assert!(report.summary().contains("8/8"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn corrupted_slice_fails_loud() {
    // The I5 G0 requirement: a deliberately corrupted slice must fail loud.
    let dir = common::scratch_dir("identity-corrupt");
    let m = write_slices(&dir, 4);
    let victim = dir.join("L01_E001_R1.slice");
    let mut bytes = std::fs::read(&victim).expect("read slice");
    // Flip one bit deep inside the payload — the smallest possible corruption.
    bytes[1_000_000] ^= 0x01;
    std::fs::write(&victim, &bytes).expect("write corrupted slice");

    let report = verify_dir(&dir, &m).expect("verify");
    assert!(!report.all_match(), "a corrupted slice must not match");
    assert_eq!(report.matched(), 3);
    assert_eq!(report.failed(), 1);
    match report.first_failure() {
        Some(FileStatus::ShaMismatch { file, want, got }) => {
            assert_eq!(file, "L01_E001_R1.slice");
            assert_ne!(want, got);
        }
        other => panic!("expected a ShaMismatch, got {other:?}"),
    }
    // Refusing to serve: the error must name the file and both shas.
    let err = report.into_result().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("IDENTITY MISMATCH"), "got {msg}");
    assert!(msg.contains("L01_E001_R1.slice"), "got {msg}");
    assert!(msg.contains("refusing to serve"), "got {msg}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn truncated_slice_fails_loud() {
    let dir = common::scratch_dir("identity-truncated");
    let m = write_slices(&dir, 4);
    let victim = dir.join("L01_E002_R2.slice");
    let bytes = std::fs::read(&victim).expect("read slice");
    std::fs::write(&victim, &bytes[..bytes.len() - 4096]).expect("truncate slice");

    let report = verify_dir(&dir, &m).expect("verify");
    assert!(!report.all_match());
    match report.first_failure() {
        Some(FileStatus::SizeMismatch { file, want, got }) => {
            assert_eq!(file, "L01_E002_R2.slice");
            assert_eq!(*want, geom::QUARTER_SLICE_BYTES as u64);
            assert_eq!(*got, geom::QUARTER_SLICE_BYTES as u64 - 4096);
        }
        other => panic!("expected a SizeMismatch, got {other:?}"),
    }
    let err = report.into_result().unwrap_err();
    assert!(format!("{err}").contains("truncated or wrong layout"), "got {err}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn empty_slice_fails_loud() {
    let dir = common::scratch_dir("identity-empty");
    let m = write_slices(&dir, 2);
    std::fs::write(dir.join("L01_E000_R0.slice"), b"").expect("empty slice");
    let report = verify_dir(&dir, &m).expect("verify");
    assert!(matches!(
        report.first_failure(),
        Some(FileStatus::SizeMismatch { got: 0, .. })
    ));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn missing_slice_fails_loud() {
    let dir = common::scratch_dir("identity-missing");
    let m = write_slices(&dir, 3);
    std::fs::remove_file(dir.join("L01_E002_R2.slice")).expect("remove slice");
    let report = verify_dir(&dir, &m).expect("verify");
    assert!(matches!(
        report.first_failure(),
        Some(FileStatus::Missing { file }) if file == "L01_E002_R2.slice"
    ));
    let err = report.into_result().unwrap_err();
    assert!(format!("{err}").contains("not resident"), "got {err}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn unlisted_resident_slice_fails_loud() {
    // A stale or foreign slice on the Spark's NVMe must not be silently
    // ignored: serving it would be serving an unknown build.
    let dir = common::scratch_dir("identity-unlisted");
    let m = write_slices(&dir, 2);
    std::fs::write(
        dir.join("L47_E255_R3.slice"),
        vec![0u8; geom::QUARTER_SLICE_BYTES],
    )
    .expect("write stray slice");
    let report = verify_dir(&dir, &m).expect("verify");
    assert!(!report.all_match());
    assert!(matches!(
        report.first_failure(),
        Some(FileStatus::Unlisted { file }) if file == "L47_E255_R3.slice"
    ));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn swapped_slice_files_fail_loud() {
    // Two slices with each other's content: both shas mismatch. This is the
    // "right size, right count, wrong bytes" case a size check alone misses.
    let dir = common::scratch_dir("identity-swapped");
    let m = write_slices(&dir, 2);
    let a = std::fs::read(dir.join("L01_E000_R0.slice")).unwrap();
    let b = std::fs::read(dir.join("L01_E001_R1.slice")).unwrap();
    std::fs::write(dir.join("L01_E000_R0.slice"), &b).unwrap();
    std::fs::write(dir.join("L01_E001_R1.slice"), &a).unwrap();
    let report = verify_dir(&dir, &m).expect("verify");
    assert_eq!(report.failed(), 2);
    assert!(report
        .files
        .iter()
        .all(|f| matches!(f, FileStatus::ShaMismatch { .. })));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn report_names_every_failure_not_just_the_first() {
    let dir = common::scratch_dir("identity-multi");
    let m = write_slices(&dir, 4);
    // Corrupt two, truncate one, delete one.
    let mut bytes = std::fs::read(dir.join("L01_E000_R0.slice")).unwrap();
    bytes[10] ^= 0xff;
    std::fs::write(dir.join("L01_E000_R0.slice"), &bytes).unwrap();
    let mut bytes = std::fs::read(dir.join("L01_E001_R1.slice")).unwrap();
    bytes[20] ^= 0xff;
    std::fs::write(dir.join("L01_E001_R1.slice"), &bytes).unwrap();
    let bytes = std::fs::read(dir.join("L01_E002_R2.slice")).unwrap();
    std::fs::write(dir.join("L01_E002_R2.slice"), &bytes[..100]).unwrap();
    std::fs::remove_file(dir.join("L01_E003_R3.slice")).unwrap();

    let report = verify_dir(&dir, &m).expect("verify");
    assert_eq!(report.failed(), 4);
    assert_eq!(report.matched(), 0);
    let names: Vec<&str> = report
        .files
        .iter()
        .filter(|f| !f.is_match())
        .map(|f| f.file())
        .collect();
    assert_eq!(
        names,
        vec![
            "L01_E000_R0.slice",
            "L01_E001_R1.slice",
            "L01_E002_R2.slice",
            "L01_E003_R3.slice"
        ]
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn verify_bytes_matches_verify_dir() {
    let dir = common::scratch_dir("identity-bytes");
    let m = write_slices(&dir, 1);
    let entry = &m.slices[0];
    let bytes = std::fs::read(dir.join(&entry.file)).unwrap();
    assert!(verify_bytes(&bytes, entry).is_match());
    let mut bad = bytes.clone();
    bad[0] ^= 0x80;
    assert!(matches!(
        verify_bytes(&bad, entry),
        FileStatus::ShaMismatch { .. }
    ));
    assert!(matches!(
        verify_bytes(&bytes[..100], entry),
        FileStatus::SizeMismatch { .. }
    ));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn v1_manifest_is_refused_at_parse_readback_and_load() {
    let dir = common::scratch_dir("identity-v1");
    let mut manifest = write_slices(&dir, 1);
    let file = manifest.slices[0].file.clone();
    assert_eq!(mimo26_repack::manifest::VERSION, 2);
    let loaded = mimo26_repack::identity::load_slice(&dir, &manifest, &file).unwrap();
    assert_eq!(loaded.len(), geom::QUARTER_SLICE_BYTES);
    // Same size, regions and matching SHA do not make a v1 artifact compatible.
    manifest.version = 1;
    assert!(Manifest::parse(&manifest.to_json()).is_err());
    assert!(verify_dir(&dir, &manifest).is_err());
    let err = mimo26_repack::identity::load_slice(&dir, &manifest, &file).unwrap_err();
    assert!(err.to_string().contains("version 1"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn checked_load_refuses_corruption_and_ambiguous_identity() {
    let dir = common::scratch_dir("identity-load");
    let mut manifest = write_slices(&dir, 1);
    let file = manifest.slices[0].file.clone();
    let mut bytes = std::fs::read(dir.join(&file)).unwrap();
    bytes[42] ^= 1;
    std::fs::write(dir.join(&file), bytes).unwrap();
    assert!(mimo26_repack::identity::load_slice(&dir, &manifest, &file).is_err());
    manifest.slices.push(manifest.slices[0].clone());
    assert!(mimo26_repack::identity::load_slice(&dir, &manifest, &file).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn readback_rejects_a_manifest_with_drifted_geometry() {
    let dir = common::scratch_dir("identity-geom");
    let m = write_slices(&dir, 1);
    let json = m.to_json().replace("\"ep_ranks\": 4", "\"ep_ranks\": 8");
    let bad = Manifest::parse(&json).unwrap_err();
    assert!(matches!(
        bad,
        mimo26_repack::RepackError::GeometryMismatch { field: "ep_ranks", .. }
    ));
    std::fs::remove_dir_all(&dir).ok();
}
