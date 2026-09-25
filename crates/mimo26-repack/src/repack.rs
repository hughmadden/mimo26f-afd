//! Repack: checkpoint MXFP4 expert tensors -> the pinned quarter-slice layout.
//!
//! The repack is a **pure byte permutation**: no dequantization, no
//! requantization, no arithmetic on the payload. That is what makes it
//! deterministic and byte-identical across runs (pinned by
//! `tests/determinism.rs`), and it is why the round-trip check in
//! `tests/mxfp4_roundtrip.rs` can compare against `spike/mxfp4.py` semantics
//! without the repack itself being able to hide a nibble-order bug.
//!
//! Source (READ-ONLY): one `model_pp0_ep{N}_shard0.safetensors` per 4 experts,
//! holding `model.layers.{L}.mlp.experts.{E}.{proj}.weight` `u8 [out, in/2]`
//! and `.weight_scale` `u8 [out, in/32]` for all 47 MoE layers.

use std::path::Path;

use crate::error::RepackError;
use crate::geom::{self, Proj};
use crate::mxfp4::Mxfp4Naive;
use crate::safetensors::SafetensorsHeader;

/// One expert's three projections, as read from the checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertTensors {
    /// `gate_proj.weight` `[2048, 2048]`.
    pub gate_w: Vec<u8>,
    /// `gate_proj.weight_scale` `[2048, 128]`.
    pub gate_s: Vec<u8>,
    /// `up_proj.weight` `[2048, 2048]`.
    pub up_w: Vec<u8>,
    /// `up_proj.weight_scale` `[2048, 128]`.
    pub up_s: Vec<u8>,
    /// `down_proj.weight` `[4096, 1024]`.
    pub down_w: Vec<u8>,
    /// `down_proj.weight_scale` `[4096, 64]`.
    pub down_s: Vec<u8>,
}

impl ExpertTensors {
    /// Total bytes: must equal [`geom::EXPERT_BYTES`].
    pub fn total_bytes(&self) -> usize {
        self.gate_w.len()
            + self.gate_s.len()
            + self.up_w.len()
            + self.up_s.len()
            + self.down_w.len()
            + self.down_s.len()
    }

    /// The `(payload, scales)` pair for one projection.
    pub fn proj(&self, p: Proj) -> (&[u8], &[u8]) {
        match p {
            Proj::Gate => (&self.gate_w, &self.gate_s),
            Proj::Up => (&self.up_w, &self.up_s),
            Proj::Down => (&self.down_w, &self.down_s),
        }
    }
}

/// Validate one tensor's declared shape against the pinned geometry.
fn check_shape(name: &str, shape: &[usize], want: [usize; 2]) -> Result<(), RepackError> {
    if shape != want {
        return Err(RepackError::BadGeometry {
            tensor: name.to_string(),
            detail: "shape is not the pinned MXFP4 expert geometry",
        });
    }
    Ok(())
}

/// Read one expert's six tensors from a shard header (READ-ONLY).
pub fn read_expert(
    header: &SafetensorsHeader,
    shard_path: &Path,
    layer: usize,
    expert: usize,
) -> Result<ExpertTensors, RepackError> {
    geom::check_layer(layer)?;
    geom::check_expert(expert)?;
    // Read one tensor with the full fail-loud audit: present, U8, pinned shape,
    // and a byte length that matches the declared shape.
    let read = |p: Proj, scale: bool| -> Result<Vec<u8>, RepackError> {
        let name = geom::tensor_name(layer, expert, p, scale);
        let shape = if scale {
            [p.out_rows(), p.in_cols() / 32]
        } else {
            [p.out_rows(), p.in_cols() / 2]
        };
        let entry = header
            .tensors
            .get(&name)
            .ok_or_else(|| RepackError::MissingTensor(name.clone()))?;
        if entry.dtype != "U8" {
            return Err(RepackError::BadGeometry {
                tensor: name.clone(),
                detail: "expert weights must be U8 (MXFP4 packed)",
            });
        }
        check_shape(&name, &entry.shape, shape)?;
        let bytes = header.read_tensor(shard_path, &name)?;
        if bytes.len() != shape[0] * shape[1] {
            return Err(RepackError::ShapeMismatch {
                tensor: name,
                want: shape[0] * shape[1],
                got: bytes.len(),
            });
        }
        Ok(bytes)
    };
    let t = ExpertTensors {
        gate_w: read(Proj::Gate, false)?,
        gate_s: read(Proj::Gate, true)?,
        up_w: read(Proj::Up, false)?,
        up_s: read(Proj::Up, true)?,
        down_w: read(Proj::Down, false)?,
        down_s: read(Proj::Down, true)?,
    };
    if t.total_bytes() != geom::EXPERT_BYTES {
        return Err(RepackError::ShapeMismatch {
            tensor: format!("expert L{layer} E{expert}"),
            want: geom::EXPERT_BYTES,
            got: t.total_bytes(),
        });
    }
    Ok(t)
}

/// Build one quarter slice from an expert's tensors.
///
/// Pure byte permutation into the pinned layout (see [`crate::geom`]): for each
/// projection, contiguous gate/up rows and matching contiguous down K columns,
/// followed by their scales, projections in `gate, up, down` order. The three
/// projections tile the slice exactly — no padding, no reserved tail.
/// `naive` selects known-wrong implementations for the negative tests; the
/// correct implementation is [`Mxfp4Naive::NONE`].
pub fn build_slice(
    t: &ExpertTensors,
    rank: usize,
    naive: Mxfp4Naive,
) -> Result<Vec<u8>, RepackError> {
    geom::check_rank(rank)?;
    let mut out = vec![0u8; geom::QUARTER_SLICE_BYTES];
    for p in Proj::ALL {
        let (w, s) = t.proj(p);
        let full_half = p.in_cols() / 2;
        let full_srow = p.in_cols() / 32;
        if w.len() != p.out_rows() * full_half || s.len() != p.out_rows() * full_srow {
            return Err(RepackError::BadGeometry {
                tensor: p.name().into(), detail: "source payload/scale length mismatch",
            });
        }
        let half = p.slice_in_cols() / 2;
        let srow = p.slice_in_cols() / 32;
        let col = p.slice_col_start(rank);
        let poff = p.slice_payload_off();
        let soff = p.slice_scale_off();
        for (i, r) in geom::slice_row_indices(p, rank).into_iter().enumerate() {
            let src = r * full_half + col / 2;
            let dst = poff + i * half;
            out[dst..dst + half].copy_from_slice(&w[src..src + half]);
            let ssrc = r * full_srow + col / 32;
            let sdst = soff + i * srow;
            out[sdst..sdst + srow].copy_from_slice(&s[ssrc..ssrc + srow]);
        }
    }
    if naive.has(Mxfp4Naive::NIBBLE_SWAP) {
        // T14: swap the two nibbles of every payload byte. The slice is still
        // the right size and still "loads" — only a round-trip check catches it.
        for p in Proj::ALL {
            let off = p.slice_payload_off();
            let len = p.slice_payload_bytes();
            for b in &mut out[off..off + len] {
                *b = b.rotate_left(4);
            }
        }
    }
    if naive.has(Mxfp4Naive::SCALE_OFF_BY_ONE) {
        // T10-adjacent: shift every scale row by one byte within its row.
        for p in Proj::ALL {
            let off = p.slice_scale_off();
            let srow = p.slice_in_cols() / 32;
            let rows = p.slice_rows();
            for i in 0..rows {
                let base = off + i * srow;
                out.copy_within(base + 1..base + srow, base);
                out[base + srow - 1] = 0;
            }
        }
    }
    Ok(out)
}

/// Read one projection's payload+scales back out of a slice, as the kernel
/// would: `(payload [slice_rows, in/2], scales [slice_rows, in/32])`.
pub fn slice_proj(slice: &[u8], p: Proj) -> Result<(&[u8], &[u8]), RepackError> {
    if slice.len() != geom::QUARTER_SLICE_BYTES {
        return Err(RepackError::SliceSize {
            path: "<memory>".to_string(),
            want: geom::QUARTER_SLICE_BYTES as u64,
            got: slice.len() as u64,
        });
    }
    let poff = p.slice_payload_off();
    let plen = p.slice_payload_bytes();
    let soff = p.slice_scale_off();
    let slen = p.slice_scale_bytes();
    Ok((&slice[poff..poff + plen], &slice[soff..soff + slen]))
}

/// Reassemble the full logical `f32 [out, in]` matrix for one projection from a
/// slice, using the MXFP4 semantics under `naive`.
///
/// This is the **round-trip check** the trap negatives run against: it is the
/// only thing that can tell a correct slice from a nibble-swapped or
/// scale-shifted one, because both wrong slices are the right size and both
/// "load".
pub fn slice_to_f32(slice: &[u8], p: Proj, naive: Mxfp4Naive) -> Result<Vec<f32>, RepackError> {
    let (payload, scales) = slice_proj(slice, p)?;
    let in_cols = p.slice_in_cols();
    let half = in_cols / 2;
    let srow = in_cols / 32;
    let rows = p.slice_rows();
    let mut out = vec![0.0f32; rows * in_cols];
    for i in 0..rows {
        let row = crate::mxfp4::unpack_row(
            &payload[i * half..(i + 1) * half],
            &scales[i * srow..(i + 1) * srow],
            naive,
        );
        out[i * in_cols..(i + 1) * in_cols].copy_from_slice(&row);
    }
    Ok(out)
}

/// Reassemble the full logical `f32 [out, in]` matrix for one projection
/// directly from the checkpoint tensors (the reference the slice must match).
pub fn expert_to_f32(t: &ExpertTensors, p: Proj, naive: Mxfp4Naive) -> Vec<f32> {
    let (w, s) = t.proj(p);
    crate::mxfp4::unpack_matrix(w, s, p.out_rows(), p.in_cols(), naive)
}

/// Extract this rank's rectangle from a full projection (rows for gate/up,
/// K columns for down). Retains the historical function name for callers.
pub fn slice_rows_of_full(full: &[f32], p: Proj, rank: usize) -> Vec<f32> {
    let cols = p.slice_in_cols();
    let start = p.slice_col_start(rank);
    let mut out = Vec::with_capacity(p.slice_rows() * cols);
    for r in geom::slice_row_indices(p, rank) {
        out.extend_from_slice(&full[r * p.in_cols() + start..r * p.in_cols() + start + cols]);
    }
    out
}
