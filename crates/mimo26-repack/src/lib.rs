//! `mimo26-repack` — I4 item 2 (ADVISOR-I4 §3.2 step 2, §3.3).
//!
//! Checkpoint MXFP4 expert weights (`u8 [out, in/2]` + `u8 [out, in/32]`
//! E8M0-32 block scales) -> the kernel's per-quarter-slice layout, with a
//! sha256 manifest and an identity readback verifier (the I5 G0 boot check).
//!
//! **Offline tooling.** No GPU, no hardware, no docker, no remote hosts, no
//! network. std-only, zero dependencies.
//!
//! # What this crate does
//!
//! 1. [`repack`] — read the six expert tensors for one (layer, expert) out of a
//!    checkpoint shard and permute them into the pinned quarter-slice layout
//!    ([`geom`]). The repack is a pure byte permutation: no dequantization, no
//!    requantization, no arithmetic on the payload. That is what makes it
//!    deterministic (pinned by `tests/determinism.rs`).
//! 2. [`manifest`] — a JSON manifest listing every slice file, its sha256, its
//!    source tensor names/shapes and the geometry, written next to the slices.
//! 3. [`identity`] — the readback verifier: reads a manifest plus the resident
//!    slice files and reports match/mismatch per file, refusing to serve on
//!    mismatch.
//! 4. [`mxfp4`] — the MXFP4 semantics (E2M1 codebook, E8M0-32 scales) the
//!    round-trip check uses, re-derived from `spike/mxfp4.py` (READ-ONLY
//!    reference; no spike code is imported).
//!
//! # Geometry (ADVISOR-I4 §3.1, confirmed against the real checkpoint headers)
//!
//! expert = 3 x 2048 x 4096 MXFP4 = 13,369,344 B; quarter slice = 3,342,336 B;
//! 47 MoE layers (layer 0 dense); 256 experts; 4 EP ranks. Per quarter slice:
//! 512 contiguous output rows of gate/up, 512 matching input columns of down.
//! All 4096 down output rows are retained (layout v2).
//!
//! # Traps pinned here
//!
//! * **T14** — MXFP4 nibble order (even element in the LOW nibble).
//! * **T10** — E8M0 scale byte 255 clamps to `2^127` (naive: `2^128` poison).
//!
//! Both are selectable through [`mxfp4::Mxfp4Naive`] so the negative tests can
//! prove the round-trip check kills the wrong implementation (AGENTS.md §4.5).
//!
//! # Naive discipline (suite convention)
//!
//! NEGATIVE tests call the env-default entry points ([`mxfp4::naive_from_env`])
//! so they FAIL under `MIMO26_SPIKE_NAIVE=1` (alias `MIMO26_REPACK_NAIVE=1`);
//! BOTH-RUNS tests pass explicit flags and pass both runs. Classification is
//! listed in the header of each `tests/*.rs` file.

pub mod error;
pub mod geom;
pub mod identity;
pub mod json;
pub mod manifest;
pub mod mxfp4;
pub mod repack;
pub mod safetensors;
pub mod sha256;

pub use error::RepackError;
pub use geom::{Proj, EXPERT_BYTES, QUARTER_SLICE_BYTES};
pub use identity::{load_slice, verify_dir, verify_bytes, FileStatus, ReadbackReport};
pub use manifest::{Manifest, SliceEntry, SourceTensor};
pub use mxfp4::{naive_from_env, Mxfp4Naive};
pub use repack::{build_slice, read_expert, slice_to_f32, ExpertTensors};
