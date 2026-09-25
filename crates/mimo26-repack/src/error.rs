//! Repack error taxonomy. Every failure is loud and names the file/field that
//! disagreed — a silent wrong slice is the failure mode this crate exists to
//! prevent (ADVISOR-I4 §3.2 step 2: "a slice that doesn't match its sha refuses
//! to serve").

use std::fmt;

/// Errors from repack, manifest parsing and identity readback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepackError {
    /// A tensor's byte length disagrees with its declared shape.
    ShapeMismatch {
        tensor: String,
        want: usize,
        got: usize,
    },
    /// A tensor's declared shape is not the MXFP4 expert geometry.
    BadGeometry {
        tensor: String,
        detail: &'static str,
    },
    /// A required tensor is absent from the source.
    MissingTensor(String),
    /// A tensor name is not part of the expert geometry (fail-loud name audit).
    UnexpectedTensor(String),
    /// The source is not a safetensors file (bad header length / JSON / offsets).
    BadSafetensors(String),
    /// The source file is shorter than its header declares.
    Truncated {
        path: String,
        declared: u64,
        got: u64,
    },
    /// A slice file on disk is not the pinned quarter-slice size.
    SliceSize {
        path: String,
        want: u64,
        got: u64,
    },
    /// A slice file's sha256 disagrees with the manifest (identity readback).
    ShaMismatch {
        path: String,
        want: String,
        got: String,
    },
    /// A manifest entry names a file that is not resident.
    MissingSlice(String),
    /// A resident slice file is not listed in the manifest.
    UnlistedSlice(String),
    /// The manifest itself is malformed.
    BadManifest(String),
    /// The manifest's geometry disagrees with the compiled-in geometry.
    GeometryMismatch {
        field: &'static str,
        want: u64,
        got: u64,
    },
    /// I/O failure, with the path that caused it.
    Io { path: String, detail: String },
}

impl fmt::Display for RepackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RepackError::ShapeMismatch { tensor, want, got } => write!(
                f,
                "repack: tensor {tensor} byte length {got} != declared {want}"
            ),
            RepackError::BadGeometry { tensor, detail } => {
                write!(f, "repack: tensor {tensor} has bad geometry: {detail}")
            }
            RepackError::MissingTensor(t) => write!(f, "repack: missing tensor {t}"),
            RepackError::UnexpectedTensor(t) => {
                write!(f, "repack: unexpected tensor {t} (name audit)")
            }
            RepackError::BadSafetensors(d) => write!(f, "repack: bad safetensors: {d}"),
            RepackError::Truncated {
                path,
                declared,
                got,
            } => write!(
                f,
                "repack: {path} is truncated: header declares {declared} B, file has {got} B"
            ),
            RepackError::SliceSize { path, want, got } => write!(
                f,
                "repack: slice {path} is {got} B, expected {want} B (truncated or wrong layout)"
            ),
            RepackError::ShaMismatch { path, want, got } => write!(
                f,
                "repack: IDENTITY MISMATCH {path}: manifest sha256 {want}, resident {got} — refusing to serve"
            ),
            RepackError::MissingSlice(p) => {
                write!(f, "repack: manifest lists {p} but it is not resident")
            }
            RepackError::UnlistedSlice(p) => {
                write!(f, "repack: {p} is resident but not listed in the manifest")
            }
            RepackError::BadManifest(d) => write!(f, "repack: bad manifest: {d}"),
            RepackError::GeometryMismatch { field, want, got } => write!(
                f,
                "repack: manifest geometry {field}={got} != expected {want}"
            ),
            RepackError::Io { path, detail } => write!(f, "repack: io {path}: {detail}"),
        }
    }
}

impl std::error::Error for RepackError {}

impl From<std::io::Error> for RepackError {
    fn from(e: std::io::Error) -> Self {
        RepackError::Io {
            path: String::new(),
            detail: e.to_string(),
        }
    }
}

/// Attach a path to an I/O error.
pub fn io_err(path: &str, e: std::io::Error) -> RepackError {
    RepackError::Io {
        path: path.to_string(),
        detail: e.to_string(),
    }
}
