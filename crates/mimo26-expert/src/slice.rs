//! The pinned quarter-slice layout — the kernel's input contract.
//!
//! # Source (READ-ONLY)
//!
//! `crates/mimo26-repack/src/geom.rs` is the **sole writer** of this layout
//! (I4 item 2, ADVISOR-I4 §3.2 step 2). This module re-states the same
//! constants so the kernel and its harness can address a slice without
//! depending on the offline repack crate. `tests/launch_geometry.rs`
//! pins every offset here, and the repack crate's `tests/layout.rs` pins the
//! same numbers — a drift between the two is a test failure on one side.
//!
//! # Geometry (ADVISOR-I4 §3.1; confirmed against the real checkpoint headers)
//!
//! | quantity | value |
//! |---|---|
//! | hidden width `H` | 4,096 |
//! | expert intermediate width `I` | 2,048 |
//! | experts per MoE layer | 256 |
//! | MoE layers (layer 0 dense) | 47 |
//! | EP ranks (TP4EP1 quarter slices) | 4 |
//! | expert bytes | 13,369,344 |
//! | quarter slice bytes | 3,342,336 |
//!
//! Per expert, three projections, each stored as `weight` `u8 [out, in/2]` +
//! `weight_scale` `u8 [out, in/32]`:
//!
//! | projection | logical `[out, in]` | `weight` | `weight_scale` |
//! |---|---|---|---|
//! | `gate_proj` | [2048, 4096] | [2048, 2048] | [2048, 128] |
//! | `up_proj` | [2048, 4096] | [2048, 2048] | [2048, 128] |
//! | `down_proj` | [4096, 2048] | [4096, 1024] | [4096, 64] |
//!
//! # Quarter-slice layout (PINNED — this is the kernel contract)
//!
//! Layout v2: rank `r` takes contiguous gate/up output rows
//! `[r*512,(r+1)*512)` and matching down K columns. Gate/up is `[512,4096]`,
//! down is `[4096,512]`. Each rank computes a full-hidden partial; sum four
//! partials without gathering intermediates. V1 had the same byte length but
//! incompatible semantics: reject its manifest before loading.
//!
//! The slice is a flat little-endian byte image, projections in the order
//! `gate, up, down`, and within each projection the payload region followed by
//! the scale region:
//!
//! | off | size | field |
//! |---|---|---|
//! | 0 | 1,048,576 | gate payload `u8 [512, 2048]` |
//! | 1,048,576 | 65,536 | gate scales `u8 [512, 128]` |
//! | 1,114,112 | 1,048,576 | up payload `u8 [512, 2048]` |
//! | 2,162,688 | 65,536 | up scales `u8 [512, 128]` |
//! | 2,228,224 | 1,048,576 | down payload `u8 [4096, 256]` |
//! | 3,276,800 | 65,536 | down scales `u8 [4096, 16]` |
//! | **3,342,336** | | **total** |
//!
//! # The grouped GEMM's view of a slice
//!
//! The kernel reads a **grouped** batch: `n_experts` slices resident
//! back-to-back, each `QUARTER_SLICE_BYTES` long, plus a per-expert token
//! count. `M` tokens per expert means the GEMM is
//! `[M, in] x [in, slice_rows]^T -> [M, slice_rows]` per expert, with the
//! expert's slice selected by the group index. The launch geometry is in
//! [`crate::grouped`].

use crate::ExpertError;

/// Hidden width.
pub const HIDDEN: usize = 4096;
/// Expert intermediate width (gate/up output columns, down input rows).
pub const INTERMEDIATE: usize = 2048;
/// Routed experts per MoE layer.
pub const EXPERTS_PER_LAYER: usize = 256;
/// MoE layers (layer 0 is dense and has no expert tensors).
pub const MOE_LAYERS: usize = 47;
/// EP ranks / TP4EP1 quarter slices per expert.
pub const EP_RANKS: usize = 4;
/// First MoE layer id in the checkpoint (`model.layers.1...`).
pub const FIRST_MOE_LAYER: usize = 1;

/// Bytes of one full expert: 3 x 2048 x 4096 x 0.53125 = 13,369,344.
pub const EXPERT_BYTES: usize = 13_369_344;
/// Bytes of one quarter slice: 3,342,336.
pub const QUARTER_SLICE_BYTES: usize = 3_342_336;
/// Raw byte lengths cannot distinguish v1 from v2. Read this from the manifest.
pub const LAYOUT_VERSION: u32 = 2;

pub fn check_layout_version(version: u32) -> Result<(), ExpertError> {
    if version != LAYOUT_VERSION {
        return Err(ExpertError::ShapeMismatch {
            what: format!("layout version {version} is not v{LAYOUT_VERSION}; repack required"),
        });
    }
    Ok(())
}

/// The three expert projections, in slice order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Proj {
    /// `gate_proj`: logical `[2048, 4096]`.
    Gate,
    /// `up_proj`: logical `[2048, 4096]`.
    Up,
    /// `down_proj`: logical `[4096, 2048]`.
    Down,
}

impl Proj {
    /// All three, in slice order.
    pub const ALL: [Proj; 3] = [Proj::Gate, Proj::Up, Proj::Down];

    /// Checkpoint tensor-name suffix (`gate_proj`, `up_proj`, `down_proj`).
    pub const fn name(self) -> &'static str {
        match self {
            Proj::Gate => "gate_proj",
            Proj::Up => "up_proj",
            Proj::Down => "down_proj",
        }
    }

    /// Logical output rows of the full projection.
    pub const fn out_rows(self) -> usize {
        match self {
            Proj::Gate | Proj::Up => INTERMEDIATE,
            Proj::Down => HIDDEN,
        }
    }

    /// Logical input columns of the full projection.
    pub const fn in_cols(self) -> usize {
        match self {
            Proj::Gate | Proj::Up => HIDDEN,
            Proj::Down => INTERMEDIATE,
        }
    }

    /// Output rows in a v2 slice: 512 gate/up, all 4096 down outputs.
    pub const fn slice_rows(self) -> usize {
        match self {
            Proj::Gate | Proj::Up => INTERMEDIATE / EP_RANKS,
            Proj::Down => HIDDEN,
        }
    }

    /// Local input width: full hidden for gate/up, 512 intermediate for down.
    pub const fn slice_in_cols(self) -> usize {
        match self {
            Proj::Gate | Proj::Up => HIDDEN,
            Proj::Down => INTERMEDIATE / EP_RANKS,
        }
    }

    /// Payload bytes in one quarter slice.
    pub const fn slice_payload_bytes(self) -> usize {
        self.slice_rows() * self.slice_in_cols() / 2
    }

    /// Scale bytes in one quarter slice.
    pub const fn slice_scale_bytes(self) -> usize {
        self.slice_rows() * self.slice_in_cols() / 32
    }

    /// Byte offset of this projection's payload inside the slice.
    pub const fn slice_payload_off(self) -> usize {
        match self {
            Proj::Gate => 0,
            Proj::Up => Proj::Gate.slice_payload_bytes() + Proj::Gate.slice_scale_bytes(),
            Proj::Down => {
                Proj::Up.slice_payload_off()
                    + Proj::Up.slice_payload_bytes()
                    + Proj::Up.slice_scale_bytes()
            }
        }
    }

    /// Byte offset of this projection's scale region inside the slice.
    pub const fn slice_scale_off(self) -> usize {
        self.slice_payload_off() + self.slice_payload_bytes()
    }

    /// Bytes of the reserved zero-filled tail.
    pub const fn reserved_bytes(self) -> usize {
        match self {
            Proj::Gate | Proj::Up => 0,
            Proj::Down => {
                QUARTER_SLICE_BYTES
                    - (Proj::Down.slice_scale_off() + Proj::Down.slice_scale_bytes())
            }
        }
    }

    /// Scale bytes per row (`in_cols / 32`).
    pub const fn scale_cols(self) -> usize {
        self.slice_in_cols() / 32
    }

    /// Payload bytes per row (`in_cols / 2`).
    pub const fn payload_cols(self) -> usize {
        self.slice_in_cols() / 2
    }
}

/// One row of the pinned layout table (used by the tests and the report).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutRow {
    /// Projection.
    pub proj: Proj,
    /// `"payload"` or `"scales"` or `"reserved"`.
    pub region: &'static str,
    /// Byte offset in the slice.
    pub off: usize,
    /// Byte length.
    pub len: usize,
}

/// The pinned layout table, in offset order.
pub fn layout_table() -> Vec<LayoutRow> {
    let mut rows = Vec::new();
    for p in Proj::ALL {
        rows.push(LayoutRow {
            proj: p,
            region: "payload",
            off: p.slice_payload_off(),
            len: p.slice_payload_bytes(),
        });
        rows.push(LayoutRow {
            proj: p,
            region: "scales",
            off: p.slice_scale_off(),
            len: p.slice_scale_bytes(),
        });
    }
    rows.push(LayoutRow {
        proj: Proj::Down,
        region: "reserved",
        off: Proj::Down.slice_scale_off() + Proj::Down.slice_scale_bytes(),
        len: Proj::Down.reserved_bytes(),
    });
    rows
}

/// Which output rows of the full projection belong to `rank`'s quarter slice.
///
/// Contiguous gate/up intermediate rows; every down output row.
pub fn slice_row_indices(proj: Proj, rank: usize) -> Vec<usize> {
    assert!(rank < EP_RANKS, "rank outside TP4");
    let start = match proj {
        Proj::Gate | Proj::Up => rank * proj.slice_rows(),
        Proj::Down => 0,
    };
    (start..start + proj.slice_rows()).collect()
}

/// The `(payload, scales)` regions of one projection inside a slice.
///
/// This is the kernel's addressing function: `payload` is
/// `[slice_rows, in/2]` row-major, `scales` is `[slice_rows, in/32]` row-major.
pub fn slice_proj(slice: &[u8], p: Proj) -> Result<(&[u8], &[u8]), ExpertError> {
    if slice.len() != QUARTER_SLICE_BYTES {
        return Err(ExpertError::SliceSize {
            want: QUARTER_SLICE_BYTES,
            got: slice.len(),
        });
    }
    let poff = p.slice_payload_off();
    let plen = p.slice_payload_bytes();
    let soff = p.slice_scale_off();
    let slen = p.slice_scale_bytes();
    Ok((&slice[poff..poff + plen], &slice[soff..soff + slen]))
}

/// The `(payload, scales)` regions of one projection of one expert inside a
/// **grouped** slice image (`n_experts` slices back-to-back).
pub fn grouped_proj<'a>(
    grouped: &'a [u8],
    expert: usize,
    p: Proj,
) -> Result<(&'a [u8], &'a [u8]), ExpertError> {
    let base = expert
        .checked_mul(QUARTER_SLICE_BYTES)
        .ok_or_else(|| ExpertError::Grouped {
            what: format!("expert {expert} offset overflows"),
        })?;
    let end = base + QUARTER_SLICE_BYTES;
    if end > grouped.len() {
        return Err(ExpertError::Grouped {
            what: format!(
                "expert {expert} slice [{base}, {end}) is outside the grouped image ({} B)",
                grouped.len()
            ),
        });
    }
    slice_proj(&grouped[base..end], p)
}

/// Validate a rank index.
pub fn check_rank(rank: usize) -> Result<(), ExpertError> {
    if rank >= EP_RANKS {
        return Err(ExpertError::ShapeMismatch {
            what: format!("EP rank {rank} must be 0..{EP_RANKS}"),
        });
    }
    Ok(())
}

/// Validate a layer id (1..=47; layer 0 is dense).
pub fn check_layer(layer: usize) -> Result<(), ExpertError> {
    if layer < FIRST_MOE_LAYER || layer >= FIRST_MOE_LAYER + MOE_LAYERS {
        return Err(ExpertError::ShapeMismatch {
            what: format!("MoE layer {layer} must be 1..={} (layer 0 is dense)", MOE_LAYERS),
        });
    }
    Ok(())
}

/// Validate an expert id.
pub fn check_expert(expert: usize) -> Result<(), ExpertError> {
    if expert >= EXPERTS_PER_LAYER {
        return Err(ExpertError::ShapeMismatch {
            what: format!("expert {expert} must be 0..{EXPERTS_PER_LAYER}"),
        });
    }
    Ok(())
}

/// The checkpoint tensor name for one projection of one expert.
pub fn tensor_name(layer: usize, expert: usize, proj: Proj, scale: bool) -> String {
    let suffix = if scale { "weight_scale" } else { "weight" };
    format!(
        "model.layers.{layer}.mlp.experts.{expert}.{}.{suffix}",
        proj.name()
    )
}

/// Slice file name for one (layer, expert, rank) — the repack crate's naming.
pub fn slice_file_name(layer: usize, expert: usize, rank: usize) -> String {
    format!("L{layer:02}_E{expert:03}_R{rank}.slice")
}
