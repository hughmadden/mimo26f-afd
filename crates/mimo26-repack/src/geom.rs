//! Checkpoint geometry and the **pinned quarter-slice layout**.
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
//! `[r*512,(r+1)*512)` and matching down input columns. Local matrices are
//! gate/up `[512,4096]` and down `[4096,512]`. Sum four full-hidden FFN
//! partials; no intermediate gather. V1 strided-output artifacts must be
//! regenerated, not relabelled: byte lengths alone cannot distinguish them.
//!
//! The slice is a flat little-endian byte image, projections in the order
//! `gate, up, down`, and within each projection the payload region followed by
//! the scale region:
//!
//! | off | size | field |
//! |---|---|---|
//! | 0 | 1,048,576 | gate payload `u8 [512, 2048]` (contiguous 512-row range) |
//! | 1,048,576 | 65,536 | gate scales `u8 [512, 128]` |
//! | 1,114,112 | 1,048,576 | up payload `u8 [512, 2048]` |
//! | 2,162,688 | 65,536 | up scales `u8 [512, 128]` |
//! | 2,228,224 | 1,048,576 | down payload `u8 [4096, 256]` (all output rows, contiguous K512) |
//! | 3,276,800 | 65,536 | down scales `u8 [4096, 16]` |
//! | **3,342,336** | | **total** |
//!
//! The three projections tile the slice exactly — no padding, no reserved tail —
//! and the total is exactly `expert_bytes / 4` (3,342,336), so the resident-bytes
//! arithmetic in ADVISOR-I4 §3.1 holds byte-for-byte. `tests/layout.rs` pins
//! every offset and the total.

use crate::error::RepackError;

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

    /// Checkpoint K offset owned by this rank (aligned to an E8M0 block).
    pub const fn slice_col_start(self, rank: usize) -> usize {
        match self {
            Proj::Gate | Proj::Up => 0,
            Proj::Down => rank * (INTERMEDIATE / EP_RANKS),
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
}

/// One row of the pinned layout table (used by the tests and the report).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutRow {
    /// Projection.
    pub proj: Proj,
    /// `"payload"` or `"scales"`.
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

/// Validate a rank index.
pub fn check_rank(rank: usize) -> Result<(), RepackError> {
    if rank >= EP_RANKS {
        return Err(RepackError::BadGeometry {
            tensor: format!("rank {rank}"),
            detail: "EP rank must be 0..4",
        });
    }
    Ok(())
}

/// Validate a layer id (1..=47; layer 0 is dense).
pub fn check_layer(layer: usize) -> Result<(), RepackError> {
    if layer < FIRST_MOE_LAYER || layer >= FIRST_MOE_LAYER + MOE_LAYERS {
        return Err(RepackError::BadGeometry {
            tensor: format!("layer {layer}"),
            detail: "MoE layer must be 1..=47 (layer 0 is dense)",
        });
    }
    Ok(())
}

/// Validate an expert id.
pub fn check_expert(expert: usize) -> Result<(), RepackError> {
    if expert >= EXPERTS_PER_LAYER {
        return Err(RepackError::BadGeometry {
            tensor: format!("expert {expert}"),
            detail: "expert id must be 0..256",
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

/// The checkpoint shard file that holds expert `expert` (64-way EP sharding:
/// shard `N` holds experts `4N..4N+3` of every MoE layer).
pub fn shard_file(expert: usize) -> String {
    format!("model_pp0_ep{}_shard0.safetensors", expert / 4)
}

/// Slice file name for one (layer, expert, rank).
pub fn slice_file_name(layer: usize, expert: usize, rank: usize) -> String {
    format!("L{layer:02}_E{expert:03}_R{rank}.slice")
}
