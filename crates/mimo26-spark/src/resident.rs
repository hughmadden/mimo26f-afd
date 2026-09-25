//! Resident slice image — the layout-v2 quarter slices for one rank, held as
//! file paths with a `(layer, expert)` index. The grouped image for a layer is
//! read on demand (from the page-cached slice files) and uploaded to the device,
//! then the host copy is dropped — so the daemon does not hold the full 40 GB
//! host residency in addition to the device-resident grouped images (the boot
//! readback hashes from disk anyway).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mimo26_repack::error::io_err;
use mimo26_repack::identity;
use mimo26_repack::manifest::Manifest;
use mimo26_repack::RepackError;

/// One rank's resident slices, indexed by `(layer, expert)` -> slice file name.
#[derive(Debug)]
pub struct Resident {
    /// The slice directory (the manifest lives here too).
    pub dir: PathBuf,
    /// `(layer, expert)` -> slice file name (rank-owned slices only).
    pub files: BTreeMap<(usize, usize), String>,
    /// Number of rank-owned slices.
    pub slices: usize,
}

impl Resident {
    /// Load the manifest and index this rank's slice files (no 40 GB read — the
    /// boot readback has already verified every slice from disk). The grouped
    /// image for a layer is read on demand by [`Resident::grouped_image`].
    pub fn load_manifest(dir: &Path, rank: usize) -> Result<Self, RepackError> {
        let manifest = Manifest::read(&identity::manifest_path(dir))?;
        manifest.validate_geometry()?;
        let mut files = BTreeMap::new();
        for entry in &manifest.slices {
            if entry.rank != rank {
                continue;
            }
            files.insert((entry.layer, entry.expert), entry.file.clone());
        }
        let slices = files.len();
        Ok(Self { dir: dir.to_path_buf(), files, slices })
    }

    /// The quarter-slice bytes for one expert, read on demand from disk.
    pub fn slice(&self, layer: usize, expert: usize) -> Result<Vec<u8>, RepackError> {
        let file = self
            .files
            .get(&(layer, expert))
            .ok_or_else(|| RepackError::MissingSlice(format!("L{layer:02}_E{expert:03}")))?;
        std::fs::read(self.dir.join(file)).map_err(|e| io_err(file, e))
    }

    /// Build the FFN grouped image for one layer — the resident experts' quarter
    /// slices concatenated in expert-id order (the kernel's `expert_id == slot`
    /// convention), read on demand.
    pub fn grouped_image(&self, layer: usize, experts: usize) -> Result<Vec<u8>, RepackError> {
        let mut image = Vec::with_capacity(experts * mimo26_repack::geom::QUARTER_SLICE_BYTES);
        for expert in 0..experts {
            let slice = self.slice(layer, expert)?;
            image.extend_from_slice(&slice);
        }
        Ok(image)
    }
}
