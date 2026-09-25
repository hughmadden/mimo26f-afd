//! Grouped MXFP4 expert GEMM — the kernel's semantics, launch geometry and
//! padded-expert contract.
//!
//! # The computation
//!
//! For one quarter slice (TP4EP1) and a group of experts, each with `M` tokens:
//!
//! ```text
//! gate = x[M, 4096] @ gate_w[512, 4096]^T   -> [M, 512]     (MXFP4 weights)
//! up   = x[M, 4096] @ up_w  [512, 4096]^T   -> [M, 512]
//! h    = silu(gate) * up                    -> [M, 512]
//! down = h[M, 512] @ down_w[4096, 512]^T  -> [M, 4096]    (v2 rank partial)
//! // Sum four rank partials; no intermediate gather.
//! ```
//!
//! `M` is the tokens-per-expert count, in `{1, 2, 4, 8, 16, 64, 256}`. The
//! weights are MXFP4: `u8 [rows, in/2]` payload + `u8 [rows, in/32]` E8M0-32
//! scales, dequantized **on the device** ([`crate::mxfp4`] semantics: T14
//! nibble order, T10 clamp, f32 saturation). Accumulation is f32; the output is
//! written as f32 or BF16.
//!
//! # Why grouped, and what "grouped" costs
//!
//! Decode routes 8 experts per token, so at C1 the batch is 8 rows spread over
//! 8 different experts — `M = 1` per expert. The kernel must therefore stream
//! **whole expert slices** (3.34 MB each) to serve a handful of rows: the
//! weight-only arithmetic intensity is ~3.765 FLOP/byte, so decode is a
//! bandwidth-bound streaming problem. That is exactly the §3.1 model: decode
//! time = expert bytes / 273 GB/s. The launch geometry below is chosen for
//! streaming, not for FLOPs.
//!
//! # V2 CUDA launch-planning contract (GPU qualification separate)
//!
//! 32 rows/CTA, 256 threads, eight cooperating lanes/row; one leader stores.
//! Each lane streams an aligned 16-byte vector per K256 tile. Token-inner
//! reuse has M-capacity 1/2/4/8. Two shared activation stages use at most16KiB,
//! independent of full K. M>8 uses additional grid-z tiles and rereads weights;
//! this is a SIMT prefill fallback, not a tensor-core performance claim.
//!
//! # Padded experts (the negative)
//!
//! A grouped batch is padded to a fixed expert count so the grid is uniform.
//! **A padded expert row must never be read.** The padding is not a real
//! expert: its slice bytes are whatever follows the resident slices, and
//! reading them injects arbitrary values into the sum. The contract is:
//!
//! * `expert_count` is the number of **real** experts; `padded_count` is the
//!   number of padding slots;
//! * a block whose `blockIdx.y >= expert_count` returns immediately, before any
//!   load;
//! * the token count for a padded expert is 0, and the output rows for it are
//!   written as zero (or not written at all — the caller's choice, pinned by
//!   [`GroupedPlan::padded_output_rows`]).
//!
//! [`NaiveBits::PAD_ROW_READ`] selects the wrong implementation (read the
//! padded slice anyway), and `tests/padded_expert.rs` proves the check kills
//! it.

use crate::mxfp4::{self, BLOCK};
use crate::slice::{self, Proj};
use crate::{ExpertError, NaiveBits};

/// Threads per block (pinned).
pub const THREADS: usize = 256;
/// Output rows of the projection handled by one block (pinned).
pub const ROWS_PER_BLOCK: usize = 32;
/// Input-dimension step: one E8M0 scale block per step (pinned).
pub const K_STEP: usize = BLOCK;
/// Warps per block.
pub const WARPS: usize = THREADS / 32;
/// Cooperating lanes per output row (only lane zero stores).
pub const LANES_PER_ROW: usize = 8;
/// Elements per double-buffered K tile.
pub const K_TILE: usize = 256;

/// The tokens-per-expert sizes the kernel is built and benched for.
pub const M_SIZES: [usize; 7] = [1, 2, 4, 8, 16, 64, 256];

/// The M sizes the bandwidth harness reports (§3.1: M in {1,2,4,8,16} and M~64).
pub const BENCH_M_SIZES: [usize; 6] = [1, 2, 4, 8, 16, 64];

/// Launch geometry for one projection of a grouped batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchGeometry {
    /// `gridDim.x` — output-row tiles.
    pub grid_x: u32,
    /// `gridDim.y` — experts (real + padded).
    pub grid_y: u32,
    /// `gridDim.z` — M-capacity tiles; M>8 explicitly rereads weights.
    pub grid_z: u32,
    /// `blockDim.x`.
    pub block_x: u32,
    /// Output rows per block.
    pub rows_per_block: u32,
    /// Input-dimension step (one E8M0 block).
    pub k_step: u32,
    /// Static shared-memory bytes per block (two activation K tiles).
    pub smem_bytes: u32,
}

impl LaunchGeometry {
    /// Total blocks launched.
    pub fn blocks(&self) -> u64 {
        u64::from(self.grid_x) * u64::from(self.grid_y) * u64::from(self.grid_z)
    }

    /// Blocks that do real work (the padded experts return immediately).
    pub fn live_blocks(&self, expert_count: usize) -> u64 {
        u64::from(self.grid_x) * expert_count as u64 * u64::from(self.grid_z)
    }
}

/// Bounded shared-memory footprint; the full activation matrix is never staged.
pub fn token_stage_bytes(m: usize, in_cols: usize) -> usize {
    debug_assert!(in_cols >= K_TILE && in_cols % K_TILE == 0);
    if m == 0 { return 0; }
    2 * m.min(8).next_power_of_two() * K_TILE * 4
}

/// The pinned launch geometry for one projection.
///
/// `n_experts` is the **padded** expert count (the grid is uniform); the real
/// count is carried separately in [`GroupedPlan`].
pub fn launch_geometry(p: Proj, m: usize, n_experts: usize) -> LaunchGeometry {
    let rows = p.slice_rows();
    let grid_x = rows.div_ceil(ROWS_PER_BLOCK) as u32;
    LaunchGeometry {
        grid_x,
        grid_y: n_experts as u32,
        grid_z: m.div_ceil(8) as u32,
        block_x: THREADS as u32,
        rows_per_block: ROWS_PER_BLOCK as u32,
        k_step: K_STEP as u32,
        smem_bytes: token_stage_bytes(m, p.slice_in_cols()) as u32,
    }
}

/// One expert's share of a grouped batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Group {
    /// Expert id (0..256).
    pub expert: usize,
    /// Tokens routed to this expert.
    pub tokens: usize,
    /// Row offset of this expert's tokens in the batch's token matrix.
    pub token_offset: usize,
}

/// A grouped batch: real experts, padding, and the token routing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedPlan {
    /// The real groups, in order.
    pub groups: Vec<Group>,
    /// Padding slots appended after the real groups (never read).
    pub padded: usize,
}

impl GroupedPlan {
    /// A plan with `n_experts` real experts, each with `m` tokens, and
    /// `padded` padding slots.
    pub fn uniform(n_experts: usize, m: usize, padded: usize) -> Self {
        let mut groups = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            groups.push(Group {
                expert: e,
                tokens: m,
                token_offset: e * m,
            });
        }
        Self { groups, padded }
    }

    /// Real expert count.
    pub fn expert_count(&self) -> usize {
        self.groups.len()
    }

    /// Padded expert count (the grid's `gridDim.y`).
    pub fn padded_expert_count(&self) -> usize {
        self.groups.len() + self.padded
    }

    /// Total tokens in the batch.
    pub fn total_tokens(&self) -> usize {
        self.groups.iter().map(|g| g.tokens).sum()
    }

    /// The largest per-expert token count (the `M` the geometry is built for).
    pub fn max_tokens(&self) -> usize {
        self.groups.iter().map(|g| g.tokens).max().unwrap_or(0)
    }

    /// Is `expert` a padding slot?
    pub fn is_padded(&self, expert: usize) -> bool {
        expert >= self.groups.len()
    }

    /// The output rows a padded expert would write — always empty: a padded
    /// expert produces no output, and its rows are never read.
    pub fn padded_output_rows(&self) -> usize {
        0
    }

    /// Validate the plan against the pinned geometry.
    pub fn validate(&self) -> Result<(), ExpertError> {
        let mut expect_off = 0usize;
        for (i, g) in self.groups.iter().enumerate() {
            slice::check_expert(g.expert)?;
            if g.token_offset != expect_off {
                return Err(ExpertError::Grouped {
                    what: format!(
                        "group {i} token_offset {} != running offset {expect_off}",
                        g.token_offset
                    ),
                });
            }
            if !M_SIZES.contains(&g.tokens) {
                return Err(ExpertError::Grouped {
                    what: format!(
                        "group {i} has {} tokens; M must be one of {M_SIZES:?}",
                        g.tokens
                    ),
                });
            }
            expect_off += g.tokens;
        }
        Ok(())
    }
}

/// The result of one grouped GEMM: `[total_tokens, out_rows]` f32.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupedOutput {
    /// Row-major `[total_tokens, out_rows]`.
    pub data: Vec<f32>,
    /// Output rows per token.
    pub out_rows: usize,
}

impl GroupedOutput {
    /// The row of one token.
    pub fn row(&self, token: usize) -> &[f32] {
        &self.data[token * self.out_rows..(token + 1) * self.out_rows]
    }
}

/// **The reference grouped GEMM** — the CPU twin of the kernel.
///
/// `grouped` is `n_experts` quarter slices back-to-back; `x` is
/// `[total_tokens, in_cols]` f32; the result is `[total_tokens, slice_rows]`.
///
/// The dequant is [`crate::mxfp4::unpack_element`] — the same function the
/// device kernel mirrors — so this is the golden-locked twin, not a second
/// implementation of the codec.
///
/// # Padded experts
///
/// A padded expert's slice is **never read**: the loop runs over
/// `plan.groups` only, and `plan.padded` slots are skipped entirely. Under
/// [`NaiveBits::PAD_ROW_READ`] the loop instead runs to
/// `plan.padded_expert_count()` and reads the padding — the wrong
/// implementation the negative test kills.
pub fn grouped_gemm(
    grouped: &[u8],
    x: &[f32],
    plan: &GroupedPlan,
    p: Proj,
    naive: NaiveBits,
) -> Result<GroupedOutput, ExpertError> {
    plan.validate()?;
    let in_cols = p.slice_in_cols();
    let out_rows = p.slice_rows();
    let total = plan.total_tokens();
    if x.len() != total * in_cols {
        return Err(ExpertError::Grouped {
            what: format!(
                "x has {} elements, expected {total} tokens x {in_cols} cols = {}",
                x.len(),
                total * in_cols
            ),
        });
    }

    let mut out = vec![0.0f32; total * out_rows];
    let n_loop = if naive.has(NaiveBits::PAD_ROW_READ) {
        plan.padded_expert_count()
    } else {
        plan.expert_count()
    };

    for e in 0..n_loop {
        // A padded expert has no group: under the naive flag we invent one
        // (tokens = M of the first group) so the wrong implementation actually
        // reads the padding instead of silently doing nothing.
        let (tokens, token_offset) = match plan.groups.get(e) {
            Some(g) => (g.tokens, g.token_offset),
            None => {
                let m = plan.max_tokens();
                (m, plan.total_tokens())
            }
        };
        if tokens == 0 {
            continue;
        }
        let expert = plan.groups.get(e).map_or(e, |g| g.expert);
        let (payload, scales) = slice::grouped_proj(grouped, expert, p)?;
        // The naive loop has addressed padding. Report it before indexing token
        // rows that do not exist, rather than panicking or fabricating tokens.
        if plan.is_padded(e) {
            return Err(ExpertError::PaddedRowRead { expert, row: 0 });
        }
        let half = in_cols / 2;
        let srow = in_cols / BLOCK;

        for t in 0..tokens {
            let xrow = &x[(token_offset + t) * in_cols..(token_offset + t + 1) * in_cols];
            let orow = &mut out[(token_offset + t) * out_rows..(token_offset + t + 1) * out_rows];
            for (r, o) in orow.iter_mut().enumerate() {
                let prow = &payload[r * half..(r + 1) * half];
                let srowb = &scales[r * srow..(r + 1) * srow];
                let mut acc = 0.0f32;
                for k in 0..in_cols {
                    let w = mxfp4::unpack_element(prow, srowb, k, naive);
                    acc += w * xrow[k];
                    if naive.has(NaiveBits::BF16_ACCUM) && acc.is_finite() {
                        // Deliberately wrong accumulator: BF16 round-to-nearest-even
                        // after every addition, not merely at the final output.
                        let bits = acc.to_bits();
                        acc = f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
                    }
                }
                *o = acc;
            }
        }
    }
    Ok(GroupedOutput { data: out, out_rows })
}

/// The gate/up half of the expert FFN: `gate = x @ gate_w^T`,
/// `up = x @ up_w^T`, `h = silu(gate) * up`.
///
/// `silu(x) = x * sigmoid(x)`, computed in f32 (the reference's
/// `F.silu` on the f32 activation).
pub fn gate_up(
    grouped: &[u8],
    x: &[f32],
    plan: &GroupedPlan,
    naive: NaiveBits,
) -> Result<GroupedOutput, ExpertError> {
    let g = grouped_gemm(grouped, x, plan, Proj::Gate, naive)?;
    let u = grouped_gemm(grouped, x, plan, Proj::Up, naive)?;
    let mut data = vec![0.0f32; g.data.len()];
    for i in 0..data.len() {
        data[i] = silu(g.data[i]) * u.data[i];
    }
    Ok(GroupedOutput {
        data,
        out_rows: g.out_rows,
    })
}

/// `silu(x) = x * sigmoid(x)` in f32.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The full expert FFN for one quarter slice:
/// `down(silu(gate(x)) * up(x))`.
///
/// Layout v2: `h` is this rank's contiguous `[total_tokens,512]` intermediate;
/// result is a `[total_tokens,4096]` partial. Sum the four ranks' partials.
/// There is no intermediate gather and no zero padding.
pub fn expert_ffn(
    grouped: &[u8],
    h: &[f32],
    plan: &GroupedPlan,
    naive: NaiveBits,
) -> Result<GroupedOutput, ExpertError> {
    grouped_gemm(grouped, h, plan, Proj::Down, naive)
}

/// Complete rank-local v2 FFN, with 512 intermediates and all 4096 outputs.
/// The coordinator sums four rank partials to obtain the full expert FFN.
pub fn expert_ffn_self_contained(
    grouped: &[u8],
    x: &[f32],
    plan: &GroupedPlan,
    naive: NaiveBits,
) -> Result<GroupedOutput, ExpertError> {
    let h = gate_up(grouped, x, plan, naive)?;
    grouped_gemm(grouped, &h.data, plan, Proj::Down, naive)
}

/// Bytes streamed by one grouped GEMM: every real expert's slice for the
/// projection, plus the token rows read and the output rows written.
///
/// This is the §3.1 arithmetic: the kernel is bandwidth-bound, so the achieved
/// GB/s is `bytes / time` with `bytes` from here.
///
/// `m` is the tokens-per-expert the geometry was built for. It does not change
/// the byte count (the token rows are `total_tokens x in_cols` regardless of
/// how they are grouped), but it is part of the signature because the harness
/// reports the row per M and the kernel's launch geometry is a function of it —
/// keeping it here makes a mislabelled row impossible.
pub fn streamed_bytes(plan: &GroupedPlan, p: Proj, m: usize) -> u64 {
    debug_assert!(
        m == 0 || plan.max_tokens() == m,
        "streamed_bytes: m={m} does not match the plan's max_tokens={}",
        plan.max_tokens()
    );
    let per_expert = (p.slice_payload_bytes() + p.slice_scale_bytes()) as u64;
    let weights = per_expert * plan.expert_count() as u64;
    let tokens = (plan.total_tokens() * p.slice_in_cols() * 4) as u64;
    let out = (plan.total_tokens() * p.slice_rows() * 4) as u64;
    weights + tokens + out
}

/// FLOPs of one grouped GEMM (2 per multiply-add).
pub fn gemm_flops(plan: &GroupedPlan, p: Proj) -> u64 {
    2 * plan.total_tokens() as u64 * p.slice_rows() as u64 * p.slice_in_cols() as u64
}

/// The full expert FFN's streamed bytes (gate + up + down).
pub fn ffn_streamed_bytes(plan: &GroupedPlan, m: usize) -> u64 {
    streamed_bytes(plan, Proj::Gate, m)
        + streamed_bytes(plan, Proj::Up, m)
        + streamed_bytes(plan, Proj::Down, m)
}

/// The full expert FFN's FLOPs.
pub fn ffn_flops(plan: &GroupedPlan) -> u64 {
    gemm_flops(plan, Proj::Gate) + gemm_flops(plan, Proj::Up) + gemm_flops(plan, Proj::Down)
}
