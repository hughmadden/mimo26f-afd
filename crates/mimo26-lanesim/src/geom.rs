//! Checkpoint geometry and wire-shape byte arithmetic. All MODEL.
//!
//! Constants mirror `bench/model/afd_vs_tp4_model.py` (the §3.1 source of truth):
//! expert = 3 x 2048 x 4096 params x 0.53125 B (MXFP4 E2M1 + E8M0-32) =
//! 13,369,344 B; TP4EP1 quarter slice = expert / 4 = 3,342,336 B; 47 MoE layers
//! (layer 0 dense, not modelled here); 256 experts/layer; top-8 routing.

/// Label carried by every number this crate emits. MODEL — never measured.
pub const MODEL: &str = "MODEL";

/// Model geometry. [`ModelGeom::REAL`] is the MiMo-V2.6-Flash checkpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelGeom {
    /// Hidden width (4,096).
    pub hidden: usize,
    /// Expert intermediate width (2,048: gate/up columns, down rows).
    pub intermediate: usize,
    /// Routed experts per MoE layer (256).
    pub experts_per_layer: usize,
    /// Routes per token (top-8).
    pub top_k: usize,
    /// MoE layers streamed per step (47; layer 0 is dense).
    pub moe_layers: usize,
    /// Spark ranks / TP4EP1 quarter slices per expert (4).
    pub spark_ranks: usize,
    /// Bytes per expert parameter incl. scales: MXFP4 0.5 B + E8M0-32 1/32 B.
    pub quant_bytes_per_param: f64,
}

impl ModelGeom {
    /// Real MiMo-V2.6-Flash geometry (ADVISOR-I4 §3.1).
    pub const REAL: ModelGeom = ModelGeom {
        hidden: 4096,
        intermediate: 2048,
        experts_per_layer: 256,
        top_k: 8,
        moe_layers: 47,
        spark_ranks: 4,
        quant_bytes_per_param: 0.53125,
    };

    /// One full expert (gate + up + down): 13,369,344 B = 13.37 MB. MODEL.
    pub fn expert_bytes(&self) -> u64 {
        (3.0 * self.intermediate as f64 * self.hidden as f64 * self.quant_bytes_per_param) as u64
    }

    /// One TP4EP1 quarter slice, resident on every Spark: 3,342,336 B = 3.34 MB. MODEL.
    pub fn quarter_slice_bytes(&self) -> u64 {
        self.expert_bytes() / self.spark_ranks as u64
    }

    /// Coordinator->Spark request row: 40-B row descriptor + 12-B route entries +
    /// FP8 hidden row + one UE8M0 scale per 32 hidden values = 4,360 B at
    /// top-8/H=4096 (`bench/model/afd_vs_tp4_model.py::REQ_ROW`). MODEL.
    pub fn request_row_bytes(&self) -> usize {
        40 + self.top_k * 12 + self.hidden + self.hidden / 32
    }

    /// Compact return per Spark per token: one BF16 hidden vector = 8,192 B at
    /// H=4096. The Spark pre-sums its weighted route partials into this one row;
    /// per-route FP32 is forbidden (see [`Self::per_route_fp32_wall_per_token`]). MODEL.
    pub fn return_bytes_per_token(&self) -> usize {
        self.hidden * 2
    }

    /// Expert bytes resident per Spark: 256 x 47 x 3.34 MB = 40.2 GB (all experts,
    /// quarter slices). MODEL.
    pub fn resident_bytes_per_spark(&self) -> u64 {
        self.experts_per_layer as u64 * self.moe_layers as u64 * self.quarter_slice_bytes()
    }

    /// The wall the compact return exists to avoid: old per-route FP32 returns
    /// cost top_k x H x 4 B x ranks x layers = 24.6 MB/token. MODEL.
    pub fn per_route_fp32_wall_per_token(&self) -> u64 {
        self.top_k as u64
            * self.hidden as u64
            * 4
            * self.spark_ranks as u64
            * self.moe_layers as u64
    }

    /// Bytes each Spark streams from its own LPDDR5x in one step, given the mean
    /// unique experts per layer (uniform routing): u x 47 x 3.34 MB. MODEL.
    pub fn bytes_per_spark_per_step(&self, unique_experts_per_layer: f64) -> u64 {
        (unique_experts_per_layer
            * self.moe_layers as f64
            * self.quarter_slice_bytes() as f64) as u64
    }
}
