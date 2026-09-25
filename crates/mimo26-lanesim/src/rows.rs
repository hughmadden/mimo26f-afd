//! Wire row shapes — MODELED locally; the real DS41RTE3 v3 codec is
//! `mimo26-wire` (wire-writer, integrated by the captain). These types pin the
//! shapes and sizes the codec must carry (ADVISOR-I4 §3.2 step 5):
//!
//! - request row 4,360 B (FP8 hidden + scales, route ids + weights),
//! - compact return 8,192 B BF16 per Spark per token (the Spark pre-sums its 8
//!   weighted route partials; the coordinator sums the 4 rank partials in FP32),
//! - per-route FP32 returns are forbidden (the 24.6 MB/token wall).
//!
//! Assumption for integration: `mimo26-wire` frames carry a per-frame sequence
//! number with the same duplicate semantics as [`FrameSeq`] here — a re-delivered
//! frame repeats its key and is counted exactly once (L4 duplicate injection).

/// Row-descriptor bytes (ds41rt protocol: 40-B row descriptors).
pub const ROW_DESCRIPTOR_BYTES: usize = 40;
/// Route-entry bytes (ds41rt protocol: 12-B route entries: id + weight + pad).
pub const ROUTE_ENTRY_BYTES: usize = 12;

/// One coordinator->Spark request row: FP8 hidden + UE8M0 scales + route ids +
/// normalized route weights. The hidden state rides as f32 in the stub; the wire
/// size models the FP8 encoding (1 B + 1 scale per 32 values). MODEL.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestRow {
    /// Token id, unique within one submit — it keys the return frames.
    pub token: u32,
    /// Hidden state, width = `ModelGeom::hidden`.
    pub hidden: Vec<f32>,
    /// Top-k routes as (expert id, normalized weight), distinct ids.
    pub routes: Vec<(u32, f64)>,
}

impl RequestRow {
    /// Plain constructor; geometry validation happens at `LaneSim::submit`
    /// (one loud choke point).
    pub fn new(token: u32, hidden: Vec<f32>, routes: Vec<(u32, f64)>) -> Self {
        Self { token, hidden, routes }
    }

    /// Wire bytes: 40-B descriptor + 12 B/route + FP8 hidden + UE8M0 scales.
    /// 4,360 B at top-8 / H=4096. MODEL.
    pub fn wire_bytes(&self) -> usize {
        ROW_DESCRIPTOR_BYTES
            + self.routes.len() * ROUTE_ENTRY_BYTES
            + self.hidden.len()
            + self.hidden.len() / 32
    }
}

/// Frame key: (`rank`, `row`) identifies one compact return; `seq` is the wire
/// sequence number. A duplicate delivery repeats the identical key and must be
/// counted exactly once ([`crate::CollectMode::DedupBySeq`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameSeq {
    /// Spark rank / TP4EP1 quarter slice (0..4).
    pub rank: u16,
    /// Index of the request row (token) within the submit.
    pub row: u32,
    /// Monotonic frame sequence number per simulated lane.
    pub seq: u64,
}

/// The compact return frame: one pre-summed hidden vector per Spark per token,
/// BF16 on the wire (8,192 B at H=4096). MODEL.
#[derive(Clone, Debug, PartialEq)]
pub struct ReturnFrame {
    pub seq: FrameSeq,
    /// The rank's pre-summed weighted route partials, BF16-quantized.
    pub partial: Vec<f32>,
}

impl ReturnFrame {
    /// Wire bytes = BF16 hidden = 8,192 B at H=4096. MODEL.
    pub fn wire_bytes(&self) -> usize {
        self.partial.len() * 2
    }
}

/// BF16 round-to-nearest-even quantize + dequantize: models the compact return's
/// numeric precision (bit surgery, no unsafe).
pub fn f32_to_bf16(f: f32) -> f32 {
    let bits = f.to_bits();
    let rounding_bias = 0x7fffu32 + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(rounding_bias) >> 16;
    f32::from_bits(rounded << 16)
}
