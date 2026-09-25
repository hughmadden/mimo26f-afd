//! Boot identity readback — the I5 G0 check (ADVISOR-I4 §3.2).
//!
//! A Spark reads its manifest plus the slice files resident on its local NVMe
//! and refuses to serve on any mismatch. The sha256/size/geometry verification
//! lives in `mimo26-repack::identity`; this module is the daemon's boot policy:
//! read the manifest, verify every slice, and fail loud — a Spark that cannot
//! prove its resident slices are the pinned build never serves.

use std::path::Path;

use mimo26_repack::identity::{self, ReadbackReport};
use mimo26_repack::manifest::Manifest;
use mimo26_repack::RepackError;

/// The boot receipt: the readback summary logged at Spark start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootReceipt {
    /// Number of slices that matched their sha256 and size.
    pub matched: usize,
    /// Total slices checked (every manifest entry plus any unlisted resident).
    pub total: usize,
    /// Bytes of the matching slices.
    pub matched_bytes: u64,
    /// One-line boot-log summary.
    pub summary: String,
}

/// Verify every resident slice against the manifest and refuse on any mismatch.
///
/// Returns the receipt only when every listed slice matches (sha256 + size) and
/// no unlisted slice is resident; otherwise a loud [`RepackError`] naming every
/// failure class. This is the "refuse to serve" gate — partial residency is not
/// a servable state.
pub fn readback(dir: &Path) -> Result<BootReceipt, RepackError> {
    let manifest = Manifest::read(&identity::manifest_path(dir))?;
    manifest.validate_geometry()?;
    let report: ReadbackReport = identity::verify_dir(dir, &manifest)?;
    // Refuse to serve on any mismatch (sha, size, missing, or unlisted).
    report.clone().into_result()?;
    Ok(BootReceipt {
        matched: report.matched(),
        total: report.files.len(),
        matched_bytes: report.matched_bytes,
        summary: report.summary(),
    })
}
