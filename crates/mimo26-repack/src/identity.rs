//! Identity readback — the I5 G0 boot check.
//!
//! A Spark reads its manifest plus the slice files resident on its local NVMe
//! and reports match/mismatch per file. **Any mismatch refuses to serve**
//! (ADVISOR-I4 §3.2 step 2: "a slice that doesn't match its sha refuses to
//! serve"). The verifier is deliberately total: it checks every file and
//! reports all of them, then returns a single loud error if any failed, so the
//! boot log names every bad slice rather than only the first.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::error::{io_err, RepackError};
use crate::geom;
use crate::manifest::Manifest;
use crate::sha256;

/// Per-file readback result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileStatus {
    /// Resident, right size, sha matches the manifest.
    Match {
        /// File name.
        file: String,
        /// sha256 from the manifest (== the resident sha).
        sha256: String,
        /// File length.
        bytes: u64,
    },
    /// Resident but the sha disagrees — the file is corrupt or the wrong build.
    ShaMismatch {
        /// File name.
        file: String,
        /// sha256 the manifest pins.
        want: String,
        /// sha256 actually resident.
        got: String,
    },
    /// Resident but not the pinned quarter-slice size (truncated / wrong layout).
    SizeMismatch {
        /// File name.
        file: String,
        /// Pinned size.
        want: u64,
        /// Actual size.
        got: u64,
    },
    /// Listed in the manifest but not resident.
    Missing {
        /// File name.
        file: String,
    },
    /// Resident but not listed in the manifest.
    Unlisted {
        /// File name.
        file: String,
    },
}

impl FileStatus {
    /// Is this file servable?
    pub fn is_match(&self) -> bool {
        matches!(self, FileStatus::Match { .. })
    }

    /// The file name this status refers to.
    pub fn file(&self) -> &str {
        match self {
            FileStatus::Match { file, .. }
            | FileStatus::ShaMismatch { file, .. }
            | FileStatus::SizeMismatch { file, .. }
            | FileStatus::Missing { file }
            | FileStatus::Unlisted { file } => file,
        }
    }
}

/// The full readback report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadbackReport {
    /// Per-file status, in file-name order.
    pub files: Vec<FileStatus>,
    /// Total bytes of the matching slices.
    pub matched_bytes: u64,
}

impl ReadbackReport {
    /// Every file matched?
    pub fn all_match(&self) -> bool {
        self.files.iter().all(|f| f.is_match())
    }

    /// Number of matching files.
    pub fn matched(&self) -> usize {
        self.files.iter().filter(|f| f.is_match()).count()
    }

    /// Number of non-matching files.
    pub fn failed(&self) -> usize {
        self.files.len() - self.matched()
    }

    /// The first failure, if any.
    pub fn first_failure(&self) -> Option<&FileStatus> {
        self.files.iter().find(|f| !f.is_match())
    }

    /// A one-line boot-log summary.
    pub fn summary(&self) -> String {
        format!(
            "identity readback: {}/{} slices match ({} B), {} failed",
            self.matched(),
            self.files.len(),
            self.matched_bytes,
            self.failed()
        )
    }

    /// Convert to a loud error when anything failed. `None` means servable.
    pub fn into_result(self) -> Result<ReadbackReport, RepackError> {
        match self.first_failure() {
            None => Ok(self),
            Some(FileStatus::ShaMismatch { file, want, got }) => Err(RepackError::ShaMismatch {
                path: file.clone(),
                want: want.clone(),
                got: got.clone(),
            }),
            Some(FileStatus::SizeMismatch { file, want, got }) => Err(RepackError::SliceSize {
                path: file.clone(),
                want: *want,
                got: *got,
            }),
            Some(FileStatus::Missing { file }) => Err(RepackError::MissingSlice(file.clone())),
            Some(FileStatus::Unlisted { file }) => Err(RepackError::UnlistedSlice(file.clone())),
            Some(FileStatus::Match { .. }) => unreachable!("first_failure returned a match"),
        }
    }
}

/// Verify the resident slices in `dir` against `manifest`.
///
/// Checks, per manifest entry: the file exists, its length is the pinned
/// quarter-slice size, and its sha256 matches. Then checks the other direction:
/// every `*.slice` file in `dir` is listed in the manifest (an unlisted slice
/// is a stale or foreign file — serving it would be serving an unknown build).
pub fn verify_dir(dir: &Path, manifest: &Manifest) -> Result<ReadbackReport, RepackError> {
    manifest.validate_geometry()?;
    let mut files = Vec::with_capacity(manifest.slices.len());
    let mut matched_bytes = 0u64;
    for entry in &manifest.slices {
        let path = dir.join(&entry.file);
        let status = verify_one(&path, entry);
        if let FileStatus::Match { bytes, .. } = &status {
            matched_bytes += *bytes;
        }
        files.push(status);
    }
    // Reverse direction: resident slices not in the manifest.
    let listed: BTreeSet<&str> = manifest.slices.iter().map(|s| s.file.as_str()).collect();
    let mut resident: Vec<String> = Vec::new();
    let rd = std::fs::read_dir(dir).map_err(|e| io_err(&dir.display().to_string(), e))?;
    for e in rd {
        let e = e.map_err(|e| io_err(&dir.display().to_string(), e))?;
        let name = e.file_name().to_string_lossy().to_string();
        if name.ends_with(".slice") {
            resident.push(name);
        }
    }
    resident.sort();
    for name in resident {
        if !listed.contains(name.as_str()) {
            files.push(FileStatus::Unlisted { file: name });
        }
    }
    files.sort_by(|a, b| a.file().cmp(b.file()));
    Ok(ReadbackReport {
        files,
        matched_bytes,
    })
}

fn verify_one(path: &Path, entry: &crate::manifest::SliceEntry) -> FileStatus {
    let file = entry.file.clone();
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return FileStatus::Missing { file },
    };
    let got_len = meta.len();
    if got_len != geom::QUARTER_SLICE_BYTES as u64 {
        return FileStatus::SizeMismatch {
            file,
            want: geom::QUARTER_SLICE_BYTES as u64,
            got: got_len,
        };
    }
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return FileStatus::Missing { file },
    };
    let got = sha256::hex(&sha256::sha256(&bytes));
    if got != entry.sha256 {
        return FileStatus::ShaMismatch {
            file,
            want: entry.sha256.clone(),
            got,
        };
    }
    FileStatus::Match {
        file,
        sha256: got,
        bytes: got_len,
    }
}

/// Load a v2 slice only after validating the manifest, declared identity, size,
/// and SHA256. Raw slices have no header: a filename/size alone is NOT a layout
/// discriminator. Never relabel a v1 manifest as v2; regenerate every slice.
pub fn load_slice(dir: &Path, manifest: &Manifest, file: &str) -> Result<Vec<u8>, RepackError> {
    manifest.validate_geometry()?;
    let entries: Vec<_> = manifest.slices.iter().filter(|s| s.file == file).collect();
    if entries.len() != 1 {
        return Err(RepackError::BadManifest(format!("{file}: expected exactly one manifest entry")));
    }
    let entry = entries[0];
    geom::check_layer(entry.layer)?;
    geom::check_expert(entry.expert)?;
    geom::check_rank(entry.rank)?;
    if entry.file != geom::slice_file_name(entry.layer, entry.expert, entry.rank)
        || entry.bytes != geom::QUARTER_SLICE_BYTES as u64
        || entry.shard != geom::shard_file(entry.expert)
        || entry.tensors != crate::manifest::source_tensors(entry.layer, entry.expert)
    {
        return Err(RepackError::BadManifest(format!("{file}: slice identity mismatch")));
    }
    let path = dir.join(file);
    let bytes = std::fs::read(&path).map_err(|e| io_err(&path.display().to_string(), e))?;
    let status = verify_bytes(&bytes, entry);
    ReadbackReport { files: vec![status], matched_bytes: 0 }.into_result()?;
    Ok(bytes)
}

/// Low-level hash/length helper. This does NOT validate a layout version;
/// consumers loading slices must use [`load_slice`] or validate the manifest
/// before using this helper on already-resident bytes.
pub fn verify_bytes(bytes: &[u8], entry: &crate::manifest::SliceEntry) -> FileStatus {
    if bytes.len() as u64 != geom::QUARTER_SLICE_BYTES as u64 {
        return FileStatus::SizeMismatch {
            file: entry.file.clone(),
            want: geom::QUARTER_SLICE_BYTES as u64,
            got: bytes.len() as u64,
        };
    }
    let got = sha256::hex(&sha256::sha256(bytes));
    if got != entry.sha256 {
        return FileStatus::ShaMismatch {
            file: entry.file.clone(),
            want: entry.sha256.clone(),
            got,
        };
    }
    FileStatus::Match {
        file: entry.file.clone(),
        sha256: got,
        bytes: bytes.len() as u64,
    }
}

/// Convenience: the manifest path next to a slice directory.
pub fn manifest_path(dir: &Path) -> PathBuf {
    dir.join("manifest.json")
}
