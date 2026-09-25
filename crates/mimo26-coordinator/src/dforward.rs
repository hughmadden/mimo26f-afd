//! Device-resident serving forward (perf reset R1a, `docs/design/perf-reset-vs-ds41rt.md`).
//!
//! The hidden state, every glue op (RMSNorm, residual adds, RoPE, SiLU, the FP64
//! router top-k, the wire quantizer) and the KV cache stay on the 5090 for the
//! whole forward. The only per-layer host traffic left is the MoE wire exchange:
//! the E4M3 payload, scales and routes go down, the four-rank sum comes back up
//! (R2 moves that to RDMA). Buffers are persistent and grown to the largest
//! chunk; nothing is allocated per layer.
//!
//! Arithmetic is the host path's (`serving.rs`): pedantic FP32 GEMMs over the same
//! resident weights, FP64 norms and router, the same KV codec and attention
//! kernels. `serving.rs` stays as the reference and fallback.
//!
//! Perf reset R4: over a pipelined wire (RDMA) the prefill runs two lanes, each
//! with its own hidden state and positions; every other scratch buffer is used
//! by one lane at a time and shared.

use std::collections::HashMap;

use mimo26_attn::cuda;
use mimo26_attn::device::DeviceBuffer;
use mimo26_attn::ffi::{self as af, CudaError, CudaStream, M26Geom};

use crate::config::{Config, LayerKind};
use crate::forward::w_name;
use crate::gpu_dense::DenseDevice;
use crate::wire::WireClient;

mod dflash;
pub use dflash::{BLOCK as DFLASH_BLOCK, DRAFTS as DFLASH_DRAFTS};

unsafe extern "C" {
    fn m26c_rmsnorm(x: *const f32, w: *const f32, out: *mut f32, rows: i32, dim: i32, eps: f64, s: CudaStream) -> CudaError;
    fn m26c_add_inplace(h: *mut f32, a: *const f32, n: i64, s: CudaStream) -> CudaError;
    fn m26c_silu_mul(gate: *mut f32, up: *const f32, n: i64, s: CudaStream) -> CudaError;
    fn m26c_router_topk(
        logits: *const f32,
        bias: *const f32,
        rows: i32,
        n_experts: i32,
        top_k: i32,
        idx: *mut i32,
        w: *mut f32,
        s: CudaStream,
    ) -> CudaError;
    fn m26c_quant_scales(x: *const f32, n_blocks: i64, scales: *mut u8, scale_inv: *mut f32, s: CudaStream) -> CudaError;
    fn m26c_copy_i64(src: *const i64, dst: *mut i64, n: i32, s: CudaStream) -> CudaError;
    fn m26c_attn_prep(
        qkv: *const f32,
        pos: *const i64,
        t: i32,
        nq: i32,
        nkv: i32,
        dqk: i32,
        dv: i32,
        rot_dim: i32,
        theta: f64,
        value_scale: f32,
        q_rot: *mut f32,
        kc: *mut u8,
        vc: *mut u8,
        kpos: *mut i64,
        clip: *mut u64,
        s: CudaStream,
    ) -> CudaError;
    fn m26c_f32_to_bf16(x: *const f32, y: *mut u16, n: i64, s: CudaStream) -> CudaError;
    fn m26c_frame_fill(
        idx: *const i32,
        wts: *const f32,
        payload: *const u8,
        scales: *const u8,
        t: i32,
        topk: i32,
        hid: i32,
        routes: *mut u8,
        hidden: *mut u8,
        pitch: i32,
        s: CudaStream,
    ) -> CudaError;
    fn cudaHostGetDevicePointer(dev: *mut *mut core::ffi::c_void, host: *mut core::ffi::c_void, flags: u32) -> CudaError;
    fn cudaMemcpyAsync(
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        count: usize,
        kind: core::ffi::c_int,
        s: CudaStream,
    ) -> CudaError;
    fn m26c_iota_i64(dst: *mut i64, start: i64, n: i64, s: CudaStream) -> CudaError;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> CudaError;
    fn m26c_rank_sum_bf16(
        p0: *const u16,
        p1: *const u16,
        p2: *const u16,
        p3: *const u16,
        out: *mut f32,
        n: i64,
        s: CudaStream,
    ) -> CudaError;
}

const STREAM: CudaStream = core::ptr::null_mut();
/// Split-KV decode: one split per 64 visible keys, at most this many. The fixed
/// 8 of `serving.rs` gave a 4K GA layer 32 CTAs on a 170-SM GPU (0.36 ms).
const MAX_DECODE_SPLITS: usize = 256;

fn decode_splits(keys: usize) -> i32 {
    keys.div_ceil(64).clamp(1, MAX_DECODE_SPLITS) as i32
}
/// The Spark B1 FFN takes up to 4,096 token rows per request (`m26s_b1_ffn`);
/// the device forward's chunk is clamped to it, so one lane is one exchange.
/// Perf reset P5: 4,096 (was 2,048): a 4K chunk streams each rank's expert
/// weights once for twice the tokens (b1_bench: 13.6 ms per 4K vs 2 x 8.2 ms).
const MOE_CHUNK: usize = 4096;
/// Rows one batched prefill ([`DeviceForward::prefill_batch`]) takes: the DFlash
/// context tail, so every row's aux fits the buffer sized at boot and all rows lie
/// in each prompt's kept tail.
pub const BATCH_ROWS: usize = dflash::CTX_TAIL;
/// Largest MoE return plane (bytes) the rank sum reads in place from the mapped
/// RDMA ring instead of uploading first (perf reset P9): decode and verify sizes.
const ZERO_COPY_PLANE: usize = 64 * 4096 * 2;
/// Free device memory now (`cudaMemGetInfo`).
pub fn device_free_bytes() -> Result<usize, String> {
    let (mut free, mut total) = (0usize, 0usize);
    ck(unsafe { cudaMemGetInfo(&mut free, &mut total) }, "cudaMemGetInfo")?;
    Ok(free)
}

/// Device memory kept free for allocator slack, cuBLAS and kernel scratch growth.
pub const KV_MARGIN_BYTES: usize = 512 << 20;

/// SWA rows a slot keeps between prefills (perf reset L1): the window before the
/// next query plus room for decode and verify appends. A prefill chunk grows its
/// slot's SWA buffers for the chunk and [`DeviceKv::shrink_swa`] returns them, so
/// idle slots do not each hold a whole prefill chunk (421 MB per slot at 4K).
const SWA_SLACK_ROWS: usize = 64;
/// GA rows per host KV page (ARCHITECTURE §11.2 "sealed GA page").
pub const KV_PAGE_ROWS: usize = 256;
/// Smallest lane worth a second exchange: a prompt under two of these stays in
/// one lane (the extra exchange would cost more than the overlap returns).
const PAIR_MIN: usize = 64;

fn ck(rc: CudaError, what: &str) -> Result<(), String> {
    if rc == cuda::SUCCESS {
        Ok(())
    } else {
        Err(format!("{what}: {}", cuda::error_string(rc)))
    }
}

fn bytes_of<T>(x: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices viewed as bytes (little-endian host).
    unsafe { core::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(x)) }
}

fn bytes_of_mut<T>(x: &mut [T]) -> &mut [u8] {
    // SAFETY: plain-old-data slices viewed as bytes.
    unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut u8, std::mem::size_of_val(x)) }
}

/// A device buffer that grows (never shrinks) to the largest request seen.
struct Grow {
    buf: DeviceBuffer,
    cap: usize,
}

impl Grow {
    fn new() -> Result<Self, String> {
        Ok(Self { buf: DeviceBuffer::alloc(256).map_err(cuda::error_string)?, cap: 256 })
    }

    fn ensure(&mut self, bytes: usize) -> Result<(), String> {
        if bytes > self.cap {
            self.buf = DeviceBuffer::alloc(bytes).map_err(cuda::error_string)?;
            self.cap = bytes;
        }
        Ok(())
    }

    fn p<T>(&self) -> *mut T {
        self.buf.as_ptr() as *mut T
    }
}

/// Per-forward device scratch, sized by `ensure(t)` for a `t`-row chunk. `h`
/// (the residual stream) and `pos` are per lane (R4); the rest is shared.
struct Scratch {
    h: [Grow; 2],
    x: Grow,
    qkv: Grow,
    q: Grow,
    k: Grow,
    v: Grow,
    q_rot: Grow,
    k_rot: Grow,
    attn: Grow,
    o: Grow,
    f: Grow,
    gate: Grow,
    up: Grow,
    logits: Grow,
    idx: Grow,
    wts: Grow,
    scales: Grow,
    scale_inv: Grow,
    payload: Grow,
    pos: [Grow; 2],
    partials: Grow,
    clip: Grow,
    lm: Grow,
    planes: Grow,
    xb: Grow,
    /// DFlash aux features `[rows, 5 * hidden]` (perf reset S1).
    aux: Grow,
}

impl Scratch {
    fn new() -> Result<Self, String> {
        Ok(Self {
            h: [Grow::new()?, Grow::new()?],
            x: Grow::new()?,
            qkv: Grow::new()?,
            q: Grow::new()?,
            k: Grow::new()?,
            v: Grow::new()?,
            q_rot: Grow::new()?,
            k_rot: Grow::new()?,
            attn: Grow::new()?,
            o: Grow::new()?,
            f: Grow::new()?,
            gate: Grow::new()?,
            up: Grow::new()?,
            logits: Grow::new()?,
            idx: Grow::new()?,
            wts: Grow::new()?,
            scales: Grow::new()?,
            scale_inv: Grow::new()?,
            payload: Grow::new()?,
            pos: [Grow::new()?, Grow::new()?],
            partials: Grow::new()?,
            clip: Grow::new()?,
            lm: Grow::new()?,
            planes: Grow::new()?,
            xb: Grow::new()?,
            aux: Grow::new()?,
        })
    }

    fn ensure(&mut self, cfg: &Config, t: usize) -> Result<(), String> {
        let hid = cfg.hidden_size;
        let (qg, kg, vg, og) = cfg.attn_dims(LayerKind::Ga);
        let (qs, ks, vs, os) = cfg.attn_dims(LayerKind::Swa);
        let (qr, kr, vr, orr) = (qg.max(qs), kg.max(ks), vg.max(vs), og.max(os));
        let f = 4usize;
        for lane in 0..2 {
            self.h[lane].ensure(t * hid * f)?;
            self.pos[lane].ensure(t * 8)?;
        }
        self.x.ensure(t * hid * f)?;
        self.qkv.ensure(t * (qr + kr + vr) * f)?;
        self.q.ensure(t * qr * f)?;
        self.k.ensure(t * kr * f)?;
        self.v.ensure(t * vr * f)?;
        self.q_rot.ensure(t * qr * f)?;
        self.k_rot.ensure(t * kr * f)?;
        self.attn.ensure(t * orr * f)?;
        self.o.ensure(t * hid * f)?;
        self.f.ensure(t * hid * f)?;
        self.gate.ensure(t * cfg.intermediate_size * f)?;
        self.up.ensure(t * cfg.intermediate_size * f)?;
        self.logits.ensure(t * cfg.n_routed_experts * f)?;
        self.idx.ensure(t * cfg.num_experts_per_tok * 4)?;
        self.wts.ensure(t * cfg.num_experts_per_tok * 4)?;
        self.scales.ensure(t * hid / 32)?;
        self.scale_inv.ensure(t * hid / 32 * f)?;
        self.payload.ensure(t * hid)?;
        // Decode partials (one row, or one verify block of up to 8 rows): TC
        // `64*splits*130` f32 or naive `n_q*splits*(2+d_v)` f64 per row; size for
        // the larger at the split cap.
        let tc = 8 * 64 * MAX_DECODE_SPLITS * 130 * 4;
        let naive = 8 * cfg.num_attention_heads * MAX_DECODE_SPLITS * (2 + cfg.v_head_dim) * 8;
        self.partials.ensure(tc.max(naive))?;
        self.clip.ensure(8)?;
        self.lm.ensure(cfg.vocab_size * f)?;
        // The four ranks' BF16 return planes of one MoE chunk.
        self.planes.ensure(4 * t.min(MOE_CHUNK) * hid * 2)?;
        // BF16 GEMM input (the widest input is the dense layer-0 down projection).
        self.xb.ensure(t * hid.max(orr).max(cfg.intermediate_size) * 2)?;
        Ok(())
    }
}

/// One layer's device KV: FP8 K/V codes plus i64 positions, `rows` live rows.
/// GA layers append linearly and grow; SWA layers keep the last `window - 1`
/// rows plus the batch being appended (the host twin's `append_codes` eviction).
struct LayerKv {
    kind: LayerKind,
    row_k: usize, // bytes per row: n_kv * d_qk
    row_v: usize, // bytes per row: n_kv * d_v
    window: usize,
    cap: usize,
    rows: usize,
    k: DeviceBuffer,
    v: DeviceBuffer,
    pos: DeviceBuffer,
}

/// A request's device KV cache (all layers) plus the processed-token count.
pub struct DeviceKv {
    layers: Vec<LayerKv>,
    tokens: usize,
    max_chunk: usize,
    /// GA capacity the cache was built with (K1: `shrink_ga` returns to it).
    ga_cap0: usize,
    /// SWA capacity between prefills (window - 1 + [`SWA_SLACK_ROWS`]).
    swa_cap0: usize,
    tmp: DeviceBuffer,
    /// The DFlash draft KV rings (perf reset S1), when a drafter is loaded.
    draft: Option<dflash::DraftKv>,
}

impl DeviceKv {
    fn new(cfg: &Config, ga_cap: usize, max_chunk: usize) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let mut tmp_need = 256usize;
        for layer in 0..cfg.num_hidden_layers {
            let kind = cfg.layer_kind(layer);
            let n_kv = match kind {
                LayerKind::Ga => cfg.num_key_value_heads,
                LayerKind::Swa => cfg.swa_num_key_value_heads,
            };
            let row_k = n_kv * cfg.head_dim;
            let row_v = n_kv * cfg.v_head_dim;
            let (window, cap) = match kind {
                LayerKind::Ga => (0, ga_cap),
                LayerKind::Swa => (cfg.sliding_window, cfg.sliding_window - 1 + SWA_SLACK_ROWS),
            };
            if kind == LayerKind::Swa {
                tmp_need = tmp_need.max((window - 1) * row_k.max(row_v).max(8));
            }
            layers.push(LayerKv {
                kind,
                row_k,
                row_v,
                window,
                cap,
                rows: 0,
                k: DeviceBuffer::alloc(cap * row_k).map_err(cuda::error_string)?,
                v: DeviceBuffer::alloc(cap * row_v).map_err(cuda::error_string)?,
                pos: DeviceBuffer::alloc(cap * 8).map_err(cuda::error_string)?,
            });
        }
        Ok(Self {
            layers,
            tokens: 0,
            max_chunk,
            ga_cap0: ga_cap,
            swa_cap0: cfg.sliding_window - 1 + SWA_SLACK_ROWS,
            tmp: DeviceBuffer::alloc(tmp_need).map_err(cuda::error_string)?,
            draft: None,
        })
    }

    /// Drop every row (D1: a fresh cache per request).
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.rows = 0;
        }
        self.tokens = 0;
    }

    /// Tokens processed so far (the next position).
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Drop the newest rows back to `to` tokens (a speculative block's rejected
    /// tail). Every layer appended those rows last, so each loses its last
    /// `tokens - to` rows; an SWA compaction during the append kept the window
    /// before the block, which is all the next query needs.
    pub fn truncate(&mut self, to: usize) -> Result<(), String> {
        let drop = self.tokens.checked_sub(to).ok_or_else(|| format!("truncate to {to} past {}", self.tokens))?;
        for (i, l) in self.layers.iter_mut().enumerate() {
            l.rows = l.rows.checked_sub(drop).ok_or_else(|| format!("truncate: layer {i} has {} rows < {drop}", l.rows))?;
        }
        self.tokens = to;
        Ok(())
    }

    /// Give back GA memory grown past the build capacity (perf reset K1: a
    /// retired request's KV lives in the host tier, so an idle slot need not hold
    /// its long context on the 5090). Drops every row.
    pub fn shrink_ga(&mut self) -> Result<(), String> {
        self.reset();
        for l in self.layers.iter_mut().filter(|l| l.kind == LayerKind::Ga && l.cap > self.ga_cap0) {
            l.k = DeviceBuffer::alloc(self.ga_cap0 * l.row_k).map_err(cuda::error_string)?;
            l.v = DeviceBuffer::alloc(self.ga_cap0 * l.row_v).map_err(cuda::error_string)?;
            l.pos = DeviceBuffer::alloc(self.ga_cap0 * 8).map_err(cuda::error_string)?;
            l.cap = self.ga_cap0;
        }
        Ok(())
    }

    /// Return every SWA layer grown by a prefill to its between-prefills size,
    /// keeping the rows a later query can see (the last `window - 1`).
    pub fn shrink_swa(&mut self) -> Result<(), String> {
        let cap0 = self.swa_cap0;
        for l in self.layers.iter_mut().filter(|l| l.kind == LayerKind::Swa && l.cap > cap0) {
            let keep = l.rows.min(l.window - 1);
            let from = l.rows - keep;
            let k = DeviceBuffer::alloc(cap0 * l.row_k).map_err(cuda::error_string)?;
            let v = DeviceBuffer::alloc(cap0 * l.row_v).map_err(cuda::error_string)?;
            let pos = DeviceBuffer::alloc(cap0 * 8).map_err(cuda::error_string)?;
            unsafe {
                ck(cuda::cudaMemcpy(k.as_ptr(), (l.k.as_ptr() as *const u8).add(from * l.row_k) as _, keep * l.row_k,
                    cuda::D2D), "swa shrink k")?;
                ck(cuda::cudaMemcpy(v.as_ptr(), (l.v.as_ptr() as *const u8).add(from * l.row_v) as _, keep * l.row_v,
                    cuda::D2D), "swa shrink v")?;
                ck(cuda::cudaMemcpy(pos.as_ptr(), (l.pos.as_ptr() as *const u8).add(from * 8) as _, keep * 8, cuda::D2D),
                    "swa shrink pos")?;
            }
            l.k = k;
            l.v = v;
            l.pos = pos;
            l.cap = cap0;
            l.rows = keep;
        }
        Ok(())
    }

    /// Device bytes of one token's GA rows across the GA layers (11,520 for MiMo).
    pub fn ga_token_bytes(&self) -> usize {
        self.layers.iter().filter(|l| l.kind == LayerKind::Ga).map(|l| l.row_k + l.row_v + 8).sum()
    }

    /// Device bytes a prefill of `max_chunk` rows adds to the SWA layers while it runs.
    pub fn swa_prefill_bytes(&self) -> usize {
        self.layers.iter().filter(|l| l.kind == LayerKind::Swa).map(|l| self.max_chunk * (l.row_k + l.row_v + 8)).sum()
    }

    /// GA rows this cache can hold without growing.
    pub fn ga_capacity(&self) -> usize {
        self.layers.iter().find(|l| l.kind == LayerKind::Ga).map_or(0, |l| l.cap)
    }

    /// Restore target (KV host tier, perf reset K1): the next position after an
    /// imported snapshot.
    pub fn set_tokens(&mut self, n: usize) {
        self.tokens = n;
    }

    /// Bytes of one host GA page: [`KV_PAGE_ROWS`] rows of every GA layer, laid
    /// out `[GA layer][K rows][V rows]` at the full page stride (a partial page
    /// uses the same stride). ARCHITECTURE §11.2: 2,949,120 B for MiMo.
    pub fn ga_page_bytes(&self) -> usize {
        self.layers.iter().filter(|l| l.kind == LayerKind::Ga).map(|l| KV_PAGE_ROWS * (l.row_k + l.row_v)).sum()
    }

    /// Copy GA rows `[r0, r0 + n)` (`n <= KV_PAGE_ROWS`) of every GA layer into
    /// the host page `page`.
    pub fn export_ga_page(&self, r0: usize, n: usize, page: &mut [u8]) -> Result<(), String> {
        if n > KV_PAGE_ROWS || page.len() < self.ga_page_bytes() {
            return Err(format!("export_ga_page: {n} rows into {} B", page.len()));
        }
        let mut off = 0;
        for l in self.layers.iter().filter(|l| l.kind == LayerKind::Ga) {
            if r0 + n > l.rows {
                return Err(format!("export_ga_page: rows {r0}+{n} past {}", l.rows));
            }
            // Async on the default stream (the pages are page-locked); the next
            // synchronous copy or `sync()` orders them (perf reset K2).
            unsafe {
                ck(cudaMemcpyAsync(page[off..].as_mut_ptr() as _, (l.k.as_ptr() as *const u8).add(r0 * l.row_k) as _,
                    n * l.row_k, cuda::D2H, STREAM), "export ga k")?;
                let vo = off + KV_PAGE_ROWS * l.row_k;
                ck(cudaMemcpyAsync(page[vo..].as_mut_ptr() as _, (l.v.as_ptr() as *const u8).add(r0 * l.row_v) as _,
                    n * l.row_v, cuda::D2H, STREAM), "export ga v")?;
            }
            off += KV_PAGE_ROWS * (l.row_k + l.row_v);
        }
        Ok(())
    }

    /// Grow every GA layer to hold `rows` rows (one reallocation per layer, at
    /// admission or before a restore): to `rows` rounded up to 4,096, not the
    /// 12.5% step, since the request's whole reservation is known.
    pub fn reserve_ga(&mut self, rows: usize) -> Result<(), String> {
        let cap = rows.next_multiple_of(4096);
        for i in 0..self.layers.len() {
            if self.layers[i].kind == LayerKind::Ga && self.layers[i].cap < rows {
                self.grow_ga(i, cap)?;
            }
        }
        Ok(())
    }

    /// Append `n` rows from a host page (as [`Self::export_ga_page`] wrote it)
    /// to every GA layer, at positions `rows..rows + n`.
    pub fn import_ga_page(&mut self, n: usize, page: &[u8]) -> Result<(), String> {
        if n > KV_PAGE_ROWS || page.len() < self.ga_page_bytes() {
            return Err(format!("import_ga_page: {n} rows from {} B", page.len()));
        }
        let mut off = 0;
        for i in 0..self.layers.len() {
            if self.layers[i].kind != LayerKind::Ga {
                continue;
            }
            self.prepare(i, n)?;
            let l = &mut self.layers[i];
            let r0 = l.rows;
            // Async (page-locked source); positions and the sync come from
            // `finish_ga_import` (perf reset K2).
            unsafe {
                ck(cudaMemcpyAsync((l.k.as_ptr() as *mut u8).add(r0 * l.row_k) as _, page[off..].as_ptr() as _,
                    n * l.row_k, cuda::H2D, STREAM), "import ga k")?;
                let vo = off + KV_PAGE_ROWS * l.row_k;
                ck(cudaMemcpyAsync((l.v.as_ptr() as *mut u8).add(r0 * l.row_v) as _, page[vo..].as_ptr() as _,
                    n * l.row_v, cuda::H2D, STREAM), "import ga v")?;
            }
            l.rows += n;
            off += KV_PAGE_ROWS * (l.row_k + l.row_v);
        }
        Ok(())
    }

    /// After a run of [`Self::import_ga_page`] from row 0: write every GA row's
    /// position (row r holds position r) and wait for the copies.
    pub fn finish_ga_import(&mut self) -> Result<(), String> {
        for l in self.layers.iter().filter(|l| l.kind == LayerKind::Ga) {
            ck(unsafe { m26c_iota_i64(l.pos.as_ptr() as *mut i64, 0, l.rows as i64, STREAM) }, "ga positions")?;
        }
        ck(unsafe { cuda::cudaDeviceSynchronize() }, "ga import sync")
    }

    /// Wait for queued snapshot copies (after a run of [`Self::export_ga_page`]).
    pub fn sync_export(&self) -> Result<(), String> {
        ck(unsafe { cuda::cudaDeviceSynchronize() }, "ga export sync")
    }

    fn swa_keep_max(&self) -> usize {
        self.layers.iter().find(|l| l.kind == LayerKind::Swa).map_or(0, |l| l.window - 1)
    }

    /// Bytes of the SWA state: the last `window - 1` rows of every SWA layer (K,
    /// V, positions), `[SWA layer][K][V][pos]` at the full stride.
    pub fn swa_state_bytes(&self) -> usize {
        let keep = self.swa_keep_max();
        self.layers.iter().filter(|l| l.kind == LayerKind::Swa).map(|l| keep * (l.row_k + l.row_v + 8)).sum()
    }

    /// Copy the SWA state (every row a later query can see) to `dst`; returns
    /// the rows kept per layer (the same for every SWA layer).
    pub fn export_swa(&self, dst: &mut [u8]) -> Result<usize, String> {
        let keep_max = self.swa_keep_max();
        if dst.len() < self.swa_state_bytes() {
            return Err("export_swa: short buffer".into());
        }
        let mut keep_seen = None;
        let mut off = 0;
        for l in self.layers.iter().filter(|l| l.kind == LayerKind::Swa) {
            let keep = l.rows.min(keep_max);
            if *keep_seen.get_or_insert(keep) != keep {
                return Err("export_swa: SWA layers disagree on rows".into());
            }
            let r0 = l.rows - keep;
            unsafe {
                ck(cuda::cudaMemcpy(dst[off..].as_mut_ptr() as _, (l.k.as_ptr() as *const u8).add(r0 * l.row_k) as _,
                    keep * l.row_k, cuda::D2H), "export swa k")?;
                let vo = off + keep_max * l.row_k;
                ck(cuda::cudaMemcpy(dst[vo..].as_mut_ptr() as _, (l.v.as_ptr() as *const u8).add(r0 * l.row_v) as _,
                    keep * l.row_v, cuda::D2H), "export swa v")?;
                let po = vo + keep_max * l.row_v;
                ck(cuda::cudaMemcpy(dst[po..].as_mut_ptr() as _, (l.pos.as_ptr() as *const i64).add(r0) as _, keep * 8,
                    cuda::D2H), "export swa pos")?;
            }
            off += keep_max * (l.row_k + l.row_v + 8);
        }
        Ok(keep_seen.unwrap_or(0))
    }

    /// Load an SWA state (as [`Self::export_swa`] wrote it, `rows` per layer)
    /// into every SWA layer's rows `[0, rows)`.
    pub fn import_swa(&mut self, rows: usize, src: &[u8]) -> Result<(), String> {
        let keep_max = self.swa_keep_max();
        if rows > keep_max || src.len() < self.swa_state_bytes() {
            return Err(format!("import_swa: {rows} rows from {} B", src.len()));
        }
        let mut off = 0;
        for l in self.layers.iter_mut().filter(|l| l.kind == LayerKind::Swa) {
            unsafe {
                ck(cuda::cudaMemcpy(l.k.as_ptr() as _, src[off..].as_ptr() as _, rows * l.row_k, cuda::H2D),
                    "import swa k")?;
                let vo = off + keep_max * l.row_k;
                ck(cuda::cudaMemcpy(l.v.as_ptr() as _, src[vo..].as_ptr() as _, rows * l.row_v, cuda::H2D),
                    "import swa v")?;
                let po = vo + keep_max * l.row_v;
                ck(cuda::cudaMemcpy(l.pos.as_ptr() as _, src[po..].as_ptr() as _, rows * 8, cuda::H2D),
                    "import swa pos")?;
            }
            l.rows = rows;
            off += keep_max * (l.row_k + l.row_v + 8);
        }
        Ok(())
    }

    /// Bytes of the whole position state beside the GA rows: the SWA state
    /// ([`Self::swa_state_bytes`]) then the draft rings, the host tier's state
    /// slot layout.
    pub fn state_bytes(&self) -> usize {
        self.swa_state_bytes() + self.draft_state_bytes()
    }

    /// Save the SWA state and draft rings to the device buffer `dst` (at least
    /// [`Self::state_bytes`]) in the host tier's layout: a device-side retained
    /// snapshot of this position (perf reset K3, ARCHITECTURE §11.2's retained
    /// prompt bank). Returns the SWA rows kept per layer.
    pub fn save_state_dev(&self, dst: &DeviceBuffer) -> Result<usize, String> {
        if dst.bytes() < self.state_bytes() {
            return Err("save_state_dev: short buffer".into());
        }
        let keep_max = self.swa_keep_max();
        let base = dst.as_ptr() as *mut u8;
        let mut keep_seen = None;
        let mut off = 0;
        for l in self.layers.iter().filter(|l| l.kind == LayerKind::Swa) {
            let keep = l.rows.min(keep_max);
            if *keep_seen.get_or_insert(keep) != keep {
                return Err("save_state_dev: SWA layers disagree on rows".into());
            }
            let r0 = l.rows - keep;
            unsafe {
                ck(cudaMemcpyAsync(base.add(off) as _, (l.k.as_ptr() as *const u8).add(r0 * l.row_k) as _,
                    keep * l.row_k, cuda::D2D, STREAM), "save swa k")?;
                let vo = off + keep_max * l.row_k;
                ck(cudaMemcpyAsync(base.add(vo) as _, (l.v.as_ptr() as *const u8).add(r0 * l.row_v) as _,
                    keep * l.row_v, cuda::D2D, STREAM), "save swa v")?;
                let po = vo + keep_max * l.row_v;
                ck(cudaMemcpyAsync(base.add(po) as _, (l.pos.as_ptr() as *const i64).add(r0) as _, keep * 8, cuda::D2D,
                    STREAM), "save swa pos")?;
            }
            off += keep_max * (l.row_k + l.row_v + 8);
        }
        if let Some(d) = self.draft.as_ref() {
            d.save_dev(unsafe { base.add(self.swa_state_bytes()) })?;
        }
        Ok(keep_seen.unwrap_or(0))
    }

    /// Load a state saved by [`Self::save_state_dev`] (`rows` SWA rows per layer)
    /// into this cache's SWA layers and draft rings.
    pub fn load_state_dev(&mut self, src: &DeviceBuffer, rows: usize) -> Result<(), String> {
        let keep_max = self.swa_keep_max();
        if rows > keep_max || src.bytes() < self.state_bytes() {
            return Err(format!("load_state_dev: {rows} rows from {} B", src.bytes()));
        }
        let base = src.as_ptr() as *const u8;
        let swa_bytes = self.swa_state_bytes();
        let mut off = 0;
        for l in self.layers.iter_mut().filter(|l| l.kind == LayerKind::Swa) {
            unsafe {
                ck(cudaMemcpyAsync(l.k.as_ptr() as _, base.add(off) as _, rows * l.row_k, cuda::D2D, STREAM),
                    "load swa k")?;
                let vo = off + keep_max * l.row_k;
                ck(cudaMemcpyAsync(l.v.as_ptr() as _, base.add(vo) as _, rows * l.row_v, cuda::D2D, STREAM),
                    "load swa v")?;
                let po = vo + keep_max * l.row_v;
                ck(cudaMemcpyAsync(l.pos.as_ptr() as _, base.add(po) as _, rows * 8, cuda::D2D, STREAM),
                    "load swa pos")?;
            }
            l.rows = rows;
            off += keep_max * (l.row_k + l.row_v + 8);
        }
        if let Some(d) = self.draft.as_ref() {
            d.load_dev(unsafe { base.add(swa_bytes) })?;
        }
        Ok(())
    }

    /// Back to an earlier exact snapshot inside this cache: keep the first `p`
    /// GA rows (append-only, so they are still the prefix's rows) and set the
    /// position; the caller then loads that position's SWA/draft state.
    pub fn rewind_ga(&mut self, p: usize) -> Result<(), String> {
        for l in self.layers.iter_mut().filter(|l| l.kind == LayerKind::Ga) {
            if p > l.rows {
                return Err(format!("rewind_ga to {p} past {} rows", l.rows));
            }
            l.rows = p;
        }
        self.tokens = p;
        Ok(())
    }

    /// Start this (reset) cache as a copy of `src`'s first `n` GA rows, device to
    /// device: the fork of a retained snapshot inside `src` (perf reset K3; the
    /// design shares these pages copy-on-write, a slot here copies them, ≈ 1 ms
    /// per 100K tokens). The caller then loads the snapshot's SWA/draft state.
    pub fn fork_ga(&mut self, src: &DeviceKv, n: usize) -> Result<(), String> {
        self.reserve_ga(n)?;
        for (d, s) in self.layers.iter_mut().zip(&src.layers).filter(|(d, _)| d.kind == LayerKind::Ga) {
            if n > s.rows || n > d.cap {
                return Err(format!("fork_ga: {n} rows from {} into capacity {}", s.rows, d.cap));
            }
            unsafe {
                ck(cudaMemcpyAsync(d.k.as_ptr(), s.k.as_ptr() as _, n * d.row_k, cuda::D2D, STREAM), "fork ga k")?;
                ck(cudaMemcpyAsync(d.v.as_ptr(), s.v.as_ptr() as _, n * d.row_v, cuda::D2D, STREAM), "fork ga v")?;
                ck(cudaMemcpyAsync(d.pos.as_ptr(), s.pos.as_ptr() as _, n * 8, cuda::D2D, STREAM), "fork ga pos")?;
            }
            d.rows = n;
        }
        self.tokens = n;
        Ok(())
    }

    /// Bytes of the DFlash draft state (0 without a drafter).
    pub fn draft_state_bytes(&self) -> usize {
        self.draft.as_ref().map_or(0, |d| d.bytes())
    }

    /// Copy the DFlash draft rings to `dst` (no-op without a drafter).
    pub fn export_draft(&self, dst: &mut [u8]) -> Result<(), String> {
        self.draft.as_ref().map_or(Ok(()), |d| d.export(dst))
    }

    /// Load DFlash draft rings from `src` (no-op without a drafter).
    pub fn import_draft(&mut self, src: &[u8]) -> Result<(), String> {
        self.draft.as_mut().map_or(Ok(()), |d| d.import(src))
    }

    /// Make room to append `t` rows to `layer`.
    fn prepare(&mut self, layer: usize, t: usize) -> Result<(), String> {
        let tmp = self.tmp.as_ptr() as *mut u8;
        let l = &mut self.layers[layer];
        if l.rows + t <= l.cap {
            return Ok(());
        }
        match l.kind {
            LayerKind::Swa => {
                // Keep the last window-1 rows (every earlier row is outside the
                // window of all later queries), moved to the front via `tmp`.
                let keep = l.rows.min(l.window - 1);
                let from = l.rows - keep;
                for (buf, rb) in [(&l.k, l.row_k), (&l.v, l.row_v), (&l.pos, 8usize)] {
                    let base = buf.as_ptr() as *mut u8;
                    unsafe {
                        ck(cuda::cudaMemcpy(tmp as _, base.add(from * rb) as _, keep * rb, cuda::D2D), "swa compact 1")?;
                        ck(cuda::cudaMemcpy(base as _, tmp as _, keep * rb, cuda::D2D), "swa compact 2")?;
                    }
                }
                l.rows = keep;
                if l.rows + t > l.cap {
                    // A prefill chunk: grow for it (perf reset L1; `shrink_swa`
                    // returns the memory when the prefill ends). The scheduler
                    // reserves this before a prefill ([`Self::reserve_swa`]), so the
                    // reallocation (whose free synchronizes the device) stays out of
                    // the pipelined layer loop.
                    let cap = (l.rows + t).div_ceil(64) * 64;
                    self.grow_swa(layer, cap)?;
                }
            }
            LayerKind::Ga => {
                // Perf reset L1: 12.5% steps (at least 4,096 rows), not doubling: a
                // doubling step near the top of the pool (e.g. 512K -> 1M rows)
                // would ask for 6 GB it does not need. Admission reserves the
                // request up front ([`Self::reserve_ga`], exact), so this step is
                // for output that outgrows its reservation.
                let cap = (l.cap + (l.cap / 8).max(4096)).max(l.rows + t);
                self.grow_ga(layer, cap)?;
            }
        }
        Ok(())
    }

    /// Reallocate SWA layer `layer` to `cap` rows, keeping its rows.
    fn grow_swa(&mut self, layer: usize, cap: usize) -> Result<(), String> {
        let l = &mut self.layers[layer];
        let k = DeviceBuffer::alloc(cap * l.row_k).map_err(cuda::error_string)?;
        let v = DeviceBuffer::alloc(cap * l.row_v).map_err(cuda::error_string)?;
        let pos = DeviceBuffer::alloc(cap * 8).map_err(cuda::error_string)?;
        unsafe {
            ck(cuda::cudaMemcpy(k.as_ptr(), l.k.as_ptr(), l.rows * l.row_k, cuda::D2D), "swa grow k")?;
            ck(cuda::cudaMemcpy(v.as_ptr(), l.v.as_ptr(), l.rows * l.row_v, cuda::D2D), "swa grow v")?;
            ck(cuda::cudaMemcpy(pos.as_ptr(), l.pos.as_ptr(), l.rows * 8, cuda::D2D), "swa grow pos")?;
        }
        l.k = k;
        l.v = v;
        l.pos = pos;
        l.cap = cap;
        Ok(())
    }

    /// Grow every SWA layer to its prefill working set (the window plus one
    /// chunk) before a prefill starts. Growing inside `forward` freed the old
    /// buffers mid-pipeline, and each `cudaFree` synchronizes the device: 39 syncs
    /// in a prompt's first two-lane group cost an 8K prefill ~11%.
    pub fn reserve_swa(&mut self) -> Result<(), String> {
        let cap = (self.swa_keep_max() + self.max_chunk).div_ceil(64) * 64;
        for i in 0..self.layers.len() {
            if self.layers[i].kind == LayerKind::Swa && self.layers[i].cap < cap {
                self.grow_swa(i, cap)?;
            }
        }
        Ok(())
    }

    /// Reallocate GA layer `layer` to `cap` rows, keeping its rows. The new
    /// buffers exist beside the old ones until the copy is done: the transient
    /// [`Self::ga_grow_transient`] accounts for.
    fn grow_ga(&mut self, layer: usize, cap: usize) -> Result<(), String> {
        let l = &mut self.layers[layer];
        let k = DeviceBuffer::alloc(cap * l.row_k).map_err(cuda::error_string)?;
        let v = DeviceBuffer::alloc(cap * l.row_v).map_err(cuda::error_string)?;
        let pos = DeviceBuffer::alloc(cap * 8).map_err(cuda::error_string)?;
        unsafe {
            ck(cuda::cudaMemcpy(k.as_ptr(), l.k.as_ptr(), l.rows * l.row_k, cuda::D2D), "ga grow k")?;
            ck(cuda::cudaMemcpy(v.as_ptr(), l.v.as_ptr(), l.rows * l.row_v, cuda::D2D), "ga grow v")?;
            ck(cuda::cudaMemcpy(pos.as_ptr(), l.pos.as_ptr(), l.rows * 8, cuda::D2D), "ga grow pos")?;
        }
        l.k = k;
        l.v = v;
        l.pos = pos;
        l.cap = cap;
        Ok(())
    }

    /// Extra device bytes a GA growth to `rows` needs while it runs, beyond the
    /// net growth: one layer's new buffers beside its old ones (layers grow one
    /// at a time). Zero when `rows` fits.
    pub fn ga_grow_transient(&self, rows: usize) -> usize {
        let cap = self.ga_capacity();
        if rows <= cap {
            return 0;
        }
        self.layers.iter().find(|l| l.kind == LayerKind::Ga).map_or(0, |l| cap * (l.row_k + l.row_v + 8))
    }
}

/// The device-resident serving model.
/// Image rows of a prompt (perf reset V2): per image, its first prompt position and its encoded
/// rows (`[tokens, hidden]`, BF16-rounded), which replace the embeddings of the prompt's image
/// tokens.
#[derive(Default)]
pub struct EmbedOverlay {
    pub spans: Vec<(usize, Vec<f32>)>,
}

impl EmbedOverlay {
    fn row(&self, pos: usize, hid: usize) -> Option<&[f32]> {
        self.spans.iter().find_map(|(start, rows)| {
            let n = rows.len() / hid;
            (pos >= *start && pos < start + n).then(|| &rows[(pos - start) * hid..(pos - start + 1) * hid])
        })
    }
}

pub struct DeviceForward {
    cfg: Config,
    dense: DenseDevice,
    ln_in: Vec<DeviceBuffer>,
    ln_post: Vec<DeviceBuffer>,
    router_bias: Vec<Option<DeviceBuffer>>,
    sink: Vec<Option<DeviceBuffer>>,
    final_norm: DeviceBuffer,
    embed: Vec<f32>,
    sc: Scratch,
    max_chunk: usize,
    // Reused host staging for the wire exchange.
    h_payload: Vec<u8>,
    h_scales: Vec<u8>,
    h_idx: Vec<i32>,
    h_wts: Vec<f32>,
    h_embed: Vec<f32>,
    /// Image rows for the prompt being prefilled (perf reset V2): set by the scheduler around a
    /// prefill segment whose ids include image tokens.
    pub overlay: Option<EmbedOverlay>,
    /// Prefill attention kernel: P2 (FlashAttention-2-style FP16, default) or
    /// the P1 A-f32q split kernel (`MIMO26_ATTN_PREFILL=p1`).
    prefill_p2: bool,
    /// Attention prep: one fused kernel (default) or the split/RoPE/store ops
    /// (`MIMO26_ATTN_PREP=split`); bit-identical.
    fused_prep: bool,
    /// Batched decode lanes: 2 (default, W4 v2) or 1 (`MIMO26_DECODE_LANES=1`).
    decode_lanes: usize,
    /// Speculative verify lanes: 2 (default) or 1 (`MIMO26_VERIFY_LANES=1`). Two
    /// lanes overlap one lane's exchange with the other's attention, but a
    /// request's block split in two re-streams the experts both halves route to.
    verify_lanes: usize,
    /// `MIMO26_PROFILE_STAGES=1`: per-stage device-synchronized wall times.
    stages: std::cell::RefCell<Option<std::collections::BTreeMap<&'static str, (f64, usize)>>>,
    stage_t0: std::cell::Cell<std::time::Instant>,
    /// The DFlash drafter (perf reset S1), when loaded.
    dflash: Option<dflash::Dflash>,
    /// How many drafts each speculative step verifies (`MIMO26_SPEC_POLICY`).
    spec_policy: SpecPolicy,
    /// `MIMO26_SPEC_TRACE=1`: one line per request per step (drafts' probabilities, matches).
    spec_trace: bool,
    /// Page-locked host ranges mapped into the device address space, as
    /// `(host base, bytes, device base)` (perf reset P9: the RDMA rings and body).
    mapped: std::cell::RefCell<Vec<(usize, usize, usize)>>,
}

/// Verify-length policy for speculative steps.
#[derive(Clone, Copy, Debug)]
enum SpecPolicy {
    /// Verify every draft the budget allows (D7's fixed k = 7).
    Fixed,
    /// Adaptive (`MIMO26_SPEC_POLICY=conf`): estimate P(draft j accepted) as the
    /// product of the drafter's probabilities of drafts 1..j, then add verify rows
    /// best-first while a row's expected tokens per ms beat the step's average,
    /// under the cost model `a_ms + b_ms * rows` (`MIMO26_SPEC_COST_A`/`_B`).
    Confidence { a_ms: f64, b_ms: f64 },
    /// Chain cut (the default): verify drafts while the product of the drafter's
    /// probabilities stays at or above `tau` (`MIMO26_SPEC_TAU`, default 0.3).
    Chain { tau: f64 },
}

/// [`SpecPolicy::Chain`]'s choice for one request (at most `cap` drafts).
fn chain_length(p: &[f32; DFLASH_DRAFTS], cap: usize, tau: f64) -> usize {
    let mut q = 1.0f64;
    for j in 0..cap.min(DFLASH_DRAFTS) {
        q *= f64::from(p[j]).clamp(0.0, 1.0);
        if q < tau {
            return j;
        }
    }
    cap.min(DFLASH_DRAFTS)
}

/// [`SpecPolicy::Confidence`]'s choice: per request, how many of its drafts to
/// verify (each at most `caps[i]`). Every request always gets its bonus row.
fn verify_lengths(probs: &[[f32; DFLASH_DRAFTS]], caps: &[usize], a_ms: f64, b_ms: f64) -> Vec<usize> {
    let n = probs.len();
    // Candidate rows (request, depth, P(accepted through depth)), best first.
    let mut cand: Vec<(usize, usize, f64)> = Vec::new();
    for (i, p) in probs.iter().enumerate() {
        let mut q = 1.0f64;
        for j in 0..caps[i].min(DFLASH_DRAFTS) {
            q *= f64::from(p[j]).clamp(0.0, 1.0);
            cand.push((i, j + 1, q));
        }
    }
    cand.sort_by(|x, y| y.2.total_cmp(&x.2));
    let mut ks = vec![0usize; n];
    let (mut tokens, mut rows) = (n as f64, n as f64);
    for (i, depth, q) in cand {
        // Rows of one request must stay a prefix: the chain is monotone in depth,
        // so a deeper row never outranks a shallower one of the same request.
        if depth != ks[i] + 1 {
            continue;
        }
        if q / b_ms < tokens / (a_ms + b_ms * rows) {
            break;
        }
        ks[i] = depth;
        tokens += q;
        rows += 1.0;
    }
    ks
}

fn upload_f32(x: &[f32]) -> Result<DeviceBuffer, String> {
    let b = DeviceBuffer::alloc(x.len().max(1) * 4).map_err(cuda::error_string)?;
    b.upload(bytes_of(x)).map_err(cuda::error_string)?;
    Ok(b)
}

impl DeviceForward {
    /// Upload the dense weights (the `ServingModel::new` set) and the small
    /// per-layer tensors; keep only the embedding table on the host.
    pub fn new(cfg: Config, mut w: HashMap<String, Vec<f32>>, max_chunk: usize) -> Result<Self, String> {
        let hid = cfg.hidden_size;
        let mut dense = DenseDevice::new()?;
        let mut ln_in = Vec::new();
        let mut ln_post = Vec::new();
        let mut router_bias = Vec::new();
        let mut sink = Vec::new();
        let take = |w: &mut HashMap<String, Vec<f32>>, n: &str| -> Result<Vec<f32>, String> {
            w.remove(n).ok_or_else(|| format!("missing weight {n}"))
        };
        // Perf reset R1b: dense projections in BF16 on tensor cores (FP32
        // accumulate) unless MIMO26_DENSE=fp32. The router gate stays FP32 (T22).
        let bf16 = std::env::var("MIMO26_DENSE").map(|v| v != "fp32").unwrap_or(true);
        eprintln!("[dforward] dense projections: {}", if bf16 { "BF16 tensor core" } else { "FP32 pedantic" });
        eprintln!("[dforward] prefill attention: {}",
            if std::env::var("MIMO26_ATTN_PREFILL").map(|v| v != "p1").unwrap_or(true) { "P2 (FP16 flash)" } else { "P1 (A-f32q split)" });
        let up = |d: &mut DenseDevice, n: &str, x: &[f32], o: usize, i: usize| -> Result<(), String> {
            if bf16 { d.upload_bf16(n, x, o, i) } else { d.upload(n, x, o, i) }
        };
        for layer in 0..cfg.num_hidden_layers {
            let kind = cfg.layer_kind(layer);
            let (q, k, v, o_in) = cfg.attn_dims(kind);
            let n = w_name(layer, "self_attn.qkv_proj.weight");
            up(&mut dense, &n, &take(&mut w, &n)?, q + k + v, hid)?;
            let n = w_name(layer, "self_attn.o_proj.weight");
            up(&mut dense, &n, &take(&mut w, &n)?, hid, o_in)?;
            if cfg.is_moe_layer(layer) {
                let n = w_name(layer, "mlp.gate.weight");
                dense.upload(&n, &take(&mut w, &n)?, cfg.n_routed_experts, hid)?;
                router_bias.push(Some(upload_f32(&take(&mut w, &w_name(layer, "mlp.gate.e_score_correction_bias"))?)?));
            } else {
                for (s, o, i) in [
                    ("mlp.gate_proj.weight", cfg.intermediate_size, hid),
                    ("mlp.up_proj.weight", cfg.intermediate_size, hid),
                    ("mlp.down_proj.weight", hid, cfg.intermediate_size),
                ] {
                    let n = w_name(layer, s);
                    up(&mut dense, &n, &take(&mut w, &n)?, o, i)?;
                }
                router_bias.push(None);
            }
            ln_in.push(upload_f32(&take(&mut w, &w_name(layer, "input_layernorm.weight"))?)?);
            ln_post.push(upload_f32(&take(&mut w, &w_name(layer, "post_attention_layernorm.weight"))?)?);
            sink.push(if cfg.add_swa_attention_sink_bias && kind == LayerKind::Swa {
                w.remove(&w_name(layer, "self_attn.attention_sink_bias")).map(|s| upload_f32(&s)).transpose()?
            } else {
                None
            });
        }
        up(&mut dense, "lm_head.weight", &take(&mut w, "lm_head.weight")?, cfg.vocab_size, hid)?;
        let final_norm = upload_f32(&take(&mut w, "norm.weight")?)?;
        let embed = take(&mut w, "embed_tokens.weight")?;
        drop(w);
        // One-time P1 prefill config (dynamic shared memory + carveout).
        let (mut reg, mut ctas) = (0i32, 0i32);
        ck(unsafe { af::m26_attn_prefill_tc_config_split(&mut reg, &mut ctas) }, "prefill tc config")?;
        let mut me = Self {
            cfg,
            dense,
            ln_in,
            ln_post,
            router_bias,
            sink,
            final_norm,
            embed,
            sc: Scratch::new()?,
            max_chunk: max_chunk.clamp(1, MOE_CHUNK),
            h_payload: Vec::new(),
            h_scales: Vec::new(),
            h_idx: Vec::new(),
            h_wts: Vec::new(),
            h_embed: Vec::new(),
            overlay: None,
            prefill_p2: std::env::var("MIMO26_ATTN_PREFILL").map(|v| v != "p1").unwrap_or(true),
            fused_prep: std::env::var("MIMO26_ATTN_PREP").map(|v| v != "split").unwrap_or(true),
            decode_lanes: if std::env::var("MIMO26_DECODE_LANES").map(|v| v == "1").unwrap_or(false) { 1 } else { 2 },
            verify_lanes: if std::env::var("MIMO26_VERIFY_LANES").map(|v| v == "1").unwrap_or(false) { 1 } else { 2 },
            stages: std::cell::RefCell::new(std::env::var_os("MIMO26_PROFILE_STAGES").map(|_| Default::default())),
            stage_t0: std::cell::Cell::new(std::time::Instant::now()),
            dflash: None,
            // Perf reset S2: the chain cut at 0.3 is the default (D7's bench: C1 per
            // stream 90.05 vs 83.9 tok/s for fixed k = 7, C16 311.6 vs 286.8, no
            // category worse); `fixed` restores D7's k = 7.
            spec_policy: match std::env::var("MIMO26_SPEC_POLICY").as_deref() {
                Ok("fixed") => SpecPolicy::Fixed,
                Ok("conf") => SpecPolicy::Confidence {
                    a_ms: std::env::var("MIMO26_SPEC_COST_A").ok().and_then(|v| v.parse().ok()).unwrap_or(25.0),
                    b_ms: std::env::var("MIMO26_SPEC_COST_B").ok().and_then(|v| v.parse().ok()).unwrap_or(3.7),
                },
                _ => SpecPolicy::Chain {
                    tau: std::env::var("MIMO26_SPEC_TAU").ok().and_then(|v| v.parse().ok()).unwrap_or(0.3),
                },
            },
            spec_trace: std::env::var("MIMO26_SPEC_TRACE").map(|v| v == "1").unwrap_or(false),
            mapped: std::cell::RefCell::new(Vec::new()),
        };
        let (cfg, mc) = (me.cfg.clone(), me.max_chunk);
        me.sc.ensure(&cfg, mc)?;
        Ok(me)
    }

    /// Page-lock host buffers the MoE return planes arrive in (the RDMA receive
    /// rings, [`WireClient::plane_buffers`]) so their uploads run at full PCIe
    /// rate. Idempotent per buffer for the process lifetime (never unregistered:
    /// the rings live as long as the wire client).
    pub fn register_plane_buffers(&self, bufs: &[(*mut u8, usize)]) -> Result<(), String> {
        for &(p, n) in bufs {
            // SAFETY: a live page-aligned host allocation of `n` bytes. Mapped
            // (cudaHostRegisterMapped) so kernels can read/write it in place.
            ck(unsafe { cuda::cudaHostRegister(p as _, n, 2) }, "cudaHostRegister plane ring")?;
            let mut dev = core::ptr::null_mut();
            ck(unsafe { cudaHostGetDevicePointer(&mut dev, p as _, 0) }, "cudaHostGetDevicePointer")?;
            self.mapped.borrow_mut().push((p as usize, n, dev as usize));
        }
        Ok(())
    }

    /// The device address of a byte inside a registered host range.
    fn dev_ptr(&self, p: *const u8) -> Option<*mut u8> {
        let a = p as usize;
        self.mapped.borrow().iter().find(|&&(h, n, _)| a >= h && a < h + n).map(|&(h, _, d)| (d + (a - h)) as *mut u8)
    }

    /// Size the per-step scratch for `max_seqs` concurrent requests now instead
    /// of on the first prefill or step (the base scratch is sized at build).
    pub fn warm_scratch(&mut self, max_seqs: usize) -> Result<(), String> {
        self.warm_dflash(max_seqs)
    }

    /// A fresh device KV cache for one request (GA capacity grows on demand).
    pub fn new_kv(&self, ga_cap: usize) -> Result<DeviceKv, String> {
        let mut kv = DeviceKv::new(&self.cfg, ga_cap, self.max_chunk)?;
        self.attach_draft(&mut kv)?;
        Ok(kv)
    }

    /// Give `kv` its draft KV rings if a drafter is loaded and it has none.
    pub fn attach_draft(&self, kv: &mut DeviceKv) -> Result<(), String> {
        if self.dflash.is_some() && kv.draft.is_none() {
            kv.draft = Some(dflash::DraftKv::new()?);
        }
        Ok(())
    }

    /// `c = x @ w.T` for a resident weight: BF16 tensor core (casting `x` to BF16
    /// in scratch) when the weight is BF16, else pedantic FP32 SGEMM.
    unsafe fn lin(&self, x: *const f32, wname: &str, m: usize, c: *mut f32) -> Result<(), String> {
        if self.dense.is_bf16(wname) {
            let (_, inp) = self.dense.shape(wname)?;
            unsafe {
                ck(m26c_f32_to_bf16(x, self.sc.xb.p(), (m * inp) as i64, STREAM), "bf16 cast")?;
                self.dense.gemm_bf16_dev(self.sc.xb.p(), wname, m, c)
            }
        } else {
            unsafe { self.dense.gemm_dev(x, wname, m, c) }
        }
    }

    /// Close a profiled stage (device-synchronized) when `MIMO26_PROFILE_STAGES` is set.
    fn stage(&self, name: &'static str) {
        if let Some(map) = self.stages.borrow_mut().as_mut() {
            unsafe { cuda::cudaDeviceSynchronize() };
            let e = map.entry(name).or_insert((0.0, 0));
            e.0 += self.stage_t0.get().elapsed().as_secs_f64() * 1e3;
            e.1 += 1;
            self.stage_t0.set(std::time::Instant::now());
        }
    }

    /// Print and clear the per-stage table (one line per stage, per forward call).
    fn stage_report(&self, t: usize) {
        if let Some(map) = self.stages.borrow_mut().as_mut() {
            for (k, (ms, n)) in map.iter() {
                eprintln!("STAGE t={t} {k:<12} total={ms:9.3} ms n={n:4} mean={:8.4} ms", ms / *n as f64);
            }
            map.clear();
        }
    }

    fn geom(&self, kind: LayerKind) -> M26Geom {
        let cfg = &self.cfg;
        let (n_kv, window) = match kind {
            LayerKind::Ga => (cfg.num_key_value_heads as i32, 0i64),
            LayerKind::Swa => (cfg.swa_num_key_value_heads as i32, cfg.sliding_window as i64),
        };
        M26Geom {
            n_q: cfg.num_attention_heads as i32,
            n_kv,
            d_qk: cfg.head_dim as i32,
            d_v: cfg.v_head_dim as i32,
            window,
            value_scale: f64::from(cfg.attention_value_scale),
        }
    }

    fn use_tc(g: &M26Geom) -> bool {
        g.n_q == 64 && g.d_qk == 192 && g.d_v == 128 && (g.n_kv == 4 || g.n_kv == 8)
    }

    /// Forward `ids` (appended after `kv.tokens()`) in chunks of at most
    /// `max_chunk`. Returns the last row's logits `[vocab]`.
    ///
    /// Perf reset R4: over a pipelined wire (RDMA) the chunks are balanced to an
    /// even count and run as pairs in two lanes. Per layer, lane A's attention and
    /// MoE dispatch run, then lane B's, and each lane's expert return is collected
    /// at its next layer, so one lane's coordinator work and transfers hide under
    /// the other lane's Spark FFN (DS41RT's prefill chunk alternation). Per layer
    /// the KV append order stays A then B, the sequence order.
    pub fn forward(&mut self, ids: &[usize], kv: &mut DeviceKv, wire: &mut WireClient) -> Result<Vec<f32>, String> {
        if ids.is_empty() {
            return Err("forward: no tokens".into());
        }
        if kv.max_chunk != self.max_chunk {
            return Err("forward: kv built for a different chunk size".into());
        }
        let cfg = self.cfg.clone();
        let hid = cfg.hidden_size;
        let lanes = if wire.pipelined() { 2 } else { 1 };
        let spans = chunk_spans(ids.len(), self.max_chunk, lanes == 2);
        let groups: Vec<&[(usize, usize)]> = spans.chunks(lanes).collect();
        // DFlash (perf reset S1): keep the aux features of the prompt's last
        // CTX_TAIL tokens, the only ones a block query can ever see.
        let start = kv.tokens;
        let aux_from = (self.dflash.is_some() && kv.draft.is_some())
            .then(|| start.max((start + ids.len()).saturating_sub(dflash::CTX_TAIL)));
        if aux_from.is_some() {
            self.sc.aux.ensure(ids.len().min(dflash::CTX_TAIL) * dflash::TARGET_LAYERS.len() * hid * 4)?;
        }
        for g in &groups {
            self.forward_group(ids, g, kv, wire, aux_from)?;
        }
        if let Some(from) = aux_from {
            let rows: Vec<(i32, i64)> = (from..kv.tokens).map(|p| (0, p as i64)).collect();
            let aux = self.sc.aux.p::<f32>();
            self.dflash_commit(std::slice::from_ref(&kv), aux, &rows)?;
        }
        // The SWA prefill working set stays until the caller's prefill is done
        // (`DeviceKv::shrink_swa`; the scheduler shrinks once per prompt, not per
        // segment).
        // Final norm + lm_head on the last row of the last lane.
        let last_group = groups.last().expect("non-empty prompt");
        let (lane, (_, t)) = (last_group.len() - 1, *last_group.last().expect("non-empty group"));
        let last = unsafe { self.sc.h[lane].p::<f32>().add((t - 1) * hid) };
        unsafe {
            ck(m26c_rmsnorm(last, self.final_norm.as_ptr() as _, self.sc.x.p(), 1, hid as i32,
                f64::from(cfg.layernorm_epsilon), STREAM), "final norm")?;
            self.lin(self.sc.x.p(), "lm_head.weight", 1, self.sc.lm.p())?;
        }
        let mut logits = vec![0f32; cfg.vocab_size];
        self.sc.lm.buf.download_prefix(bytes_of_mut(&mut logits)).map_err(cuda::error_string)?;
        Ok(logits)
    }

    /// Copy `lane`'s residual rows after `layer` into the DFlash aux buffer if
    /// `layer` is a drafter target layer: lane row `j` goes to aux row `base + j`
    /// (rows landing below 0 are outside the kept tail and skipped).
    fn capture(&self, lane: usize, layer: usize, t: usize, base: Option<isize>) -> Result<(), String> {
        let Some(base) = base else { return Ok(()) };
        let Some(slot) = dflash::TARGET_LAYERS.iter().position(|&l| l == layer) else { return Ok(()) };
        let skip = (-base).max(0) as usize;
        if skip >= t {
            return Ok(());
        }
        let hid = self.cfg.hidden_size;
        let n = dflash::TARGET_LAYERS.len();
        unsafe {
            let dst = self.sc.aux.p::<f32>().add(((base + skip as isize) as usize * n + slot) * hid);
            let src = self.sc.h[lane].p::<f32>().add(skip * hid);
            ck(cuda::cudaMemcpy2D(dst as _, n * hid * 4, src as _, hid * 4, hid * 4, t - skip, cuda::D2D), "aux capture")
        }
    }

    /// Run one group of one or two lanes (`(start, len)` spans of `ids`, in
    /// sequence order) through every layer.
    fn forward_group(
        &mut self,
        ids: &[usize],
        group: &[(usize, usize)],
        kv: &mut DeviceKv,
        wire: &mut WireClient,
        aux_from: Option<usize>,
    ) -> Result<(), String> {
        let cfg = self.cfg.clone();
        let hid = cfg.hidden_size;
        let eps = f64::from(cfg.layernorm_epsilon);
        let tg0 = std::time::Instant::now();
        let mut next = kv.tokens;
        let mut cap: [Option<isize>; 2] = [None; 2];
        for (lane, &(s, t)) in group.iter().enumerate() {
            cap[lane] = aux_from.map(|f| next as isize - f as isize);
            let pos: Vec<i64> = (next as i64..(next + t) as i64).collect();
            self.sc.pos[lane].buf.upload_prefix(bytes_of(&pos)).map_err(cuda::error_string)?;
            // Embedding rows gathered on the host, one upload per lane. An id past the
            // vocabulary is an image token (perf reset V2): its row is the image encoder's.
            self.h_embed.clear();
            for (k, &id) in ids[s..s + t].iter().enumerate() {
                if id < cfg.vocab_size {
                    self.h_embed.extend_from_slice(&self.embed[id * hid..(id + 1) * hid]);
                } else {
                    let row = self.overlay.as_ref().and_then(|o| o.row(next + k, hid));
                    let row = row.ok_or_else(|| format!("token {id} at position {} is not a known image row", next + k))?;
                    self.h_embed.extend_from_slice(row);
                }
            }
            self.sc.h[lane].buf.upload_prefix(bytes_of(&self.h_embed)).map_err(cuda::error_string)?;
            next += t;
        }
        self.stage_t0.set(std::time::Instant::now());
        let mut pending = [false; 2];
        for layer in 0..cfg.num_hidden_layers {
            for (lane, &(_, t)) in group.iter().enumerate() {
                if std::mem::take(&mut pending[lane]) {
                    self.moe_finish(lane, t, wire)?;
                    self.capture(lane, layer - 1, t, cap[lane])?;
                }
                let h = self.sc.h[lane].p::<f32>();
                unsafe {
                    ck(m26c_rmsnorm(h, self.ln_in[layer].as_ptr() as _, self.sc.x.p(), t as i32, hid as i32, eps, STREAM),
                        "rmsnorm in")?;
                }
                self.stage("norm_resid");
                self.attn(layer, t, lane, kv)?;
                self.stage(if cfg.layer_kind(layer) == LayerKind::Ga { "a_oproj_ga" } else { "a_oproj_swa" });
                unsafe {
                    ck(m26c_add_inplace(h, self.sc.o.p(), (t * hid) as i64, STREAM), "residual attn")?;
                    ck(m26c_rmsnorm(h, self.ln_post[layer].as_ptr() as _, self.sc.x.p(), t as i32, hid as i32, eps, STREAM),
                        "rmsnorm post")?;
                }
                self.stage("norm_resid");
                pending[lane] = self.ffn_start(layer, t, lane, wire)?;
                if !pending[lane] {
                    self.capture(lane, layer, t, cap[lane])?;
                }
            }
        }
        for (lane, &(_, t)) in group.iter().enumerate() {
            if pending[lane] {
                self.moe_finish(lane, t, wire)?;
                self.capture(lane, cfg.num_hidden_layers - 1, t, cap[lane])?;
            }
        }
        kv.tokens = next;
        let rows: usize = group.iter().map(|&(_, t)| t).sum();
        self.stage_report(rows);
        if std::env::var_os("MIMO26_PROFILE").is_some() {
            eprintln!("PROFILE dfwd_group t={rows} lanes={} {:.3}", group.len(), tg0.elapsed().as_secs_f64() * 1e3);
        }
        Ok(())
    }

    fn attn(&mut self, layer: usize, t: usize, lane: usize, kv: &mut DeviceKv) -> Result<(), String> {
        let cfg = &self.cfg;
        let kind = cfg.layer_kind(layer);
        let (q_rows, k_rows, v_rows, _o_in) = cfg.attn_dims(kind);
        let g = self.geom(kind);
        let n_q = cfg.num_attention_heads as i32;
        let n_kv = g.n_kv;
        let d_qk = cfg.head_dim as i32;
        let theta = match kind {
            LayerKind::Ga => f64::from(cfg.rope_theta),
            LayerKind::Swa => f64::from(cfg.swa_rope_theta),
        };
        let factor = f64::from(cfg.partial_rotary_factor);
        let total = q_rows + k_rows + v_rows;
        let sc = &self.sc;
        let pos = sc.pos[lane].p::<i64>();
        let ga = kind == LayerKind::Ga;
        unsafe {
            self.lin(sc.x.p(), &w_name(layer, "self_attn.qkv_proj.weight"), t, sc.qkv.p())?;
        }
        self.stage(if ga { "a_qkv_ga" } else { "a_qkv_swa" });
        if !self.fused_prep {
            unsafe {
                let qkv = sc.qkv.p::<u8>();
                for (dst, off, rows) in
                    [(sc.q.p::<u8>(), 0usize, q_rows), (sc.k.p(), q_rows, k_rows), (sc.v.p(), q_rows + k_rows, v_rows)]
                {
                    ck(cuda::cudaMemcpy2D(dst as _, rows * 4, qkv.add(off * 4) as _, total * 4, rows * 4, t, cuda::D2D),
                        "qkv split")?;
                }
                ck(af::m26_rope_apply(theta, factor, sc.q.p(), sc.q_rot.p(), pos, t as i32, n_q, d_qk, 0, STREAM),
                    "rope q")?;
                ck(af::m26_rope_apply(theta, factor, sc.k.p(), sc.k_rot.p(), pos, t as i32, n_kv, d_qk, 0, STREAM),
                    "rope k")?;
            }
        }
        kv.prepare(layer, t)?;
        let l = &mut kv.layers[layer];
        let row0 = l.rows;
        unsafe {
            let kdst = (l.k.as_ptr() as *mut u8).add(row0 * l.row_k);
            let vdst = (l.v.as_ptr() as *mut u8).add(row0 * l.row_v);
            let pdst = (l.pos.as_ptr() as *mut i64).add(row0);
            if self.fused_prep {
                // rope.cu's rotary width: (int)(factor * d + 1e-6).
                let rot_dim = (factor * f64::from(d_qk) + 1e-6) as i32;
                ck(m26c_attn_prep(sc.qkv.p(), pos, t as i32, n_q, n_kv, d_qk, g.d_v, rot_dim, theta,
                    cfg.attention_value_scale, sc.q_rot.p(), kdst, vdst, pdst, sc.clip.p(), STREAM), "attn prep")?;
            } else {
                ck(af::m26_kv_store_fp8(&g, sc.k_rot.p(), sc.v.p(), t as i32, 1, 0, kdst, core::ptr::null_mut(), vdst,
                    core::ptr::null_mut(), sc.clip.p(), STREAM), "kv store")?;
                ck(m26c_copy_i64(pos, pdst, t as i32, STREAM), "kv pos")?;
            }
        }
        l.rows += t;
        let s = l.rows as i32;
        self.stage(if ga { "a_prep_ga" } else { "a_prep_swa" });
        let sink_ptr = self.sink[layer].as_ref().map_or(core::ptr::null(), |b| b.as_ptr() as *const f32);
        let tc = Self::use_tc(&g);
        unsafe {
            if t > 1 && tc && self.prefill_p2 {
                // P2 (perf reset): the queries are this chunk's rows row0..row0+t of
                // a position-contiguous cache (GA appends; SWA compaction keeps order).
                ck(af::m26_attn_prefill_fp8_fa(&g, sc.q_rot.p(), l.k.as_ptr() as _, l.v.as_ptr() as _, t as i32, s,
                    row0 as i32, 0, sink_ptr, sc.attn.p(), STREAM), "prefill p2")?;
            } else if t > 1 {
                if tc {
                    ck(af::m26_attn_prefill_fp8_tc_split(&g, sc.q_rot.p(), l.k.as_ptr() as _, l.v.as_ptr() as _,
                        core::ptr::null(), 0, pos, l.pos.as_ptr() as _, t as i32, s, 0, sink_ptr,
                        sc.attn.p(), STREAM), "prefill tc")?;
                } else {
                    ck(af::m26_attn_prefill_fp8(&g, sc.q_rot.p(), l.k.as_ptr() as _, core::ptr::null(),
                        l.v.as_ptr() as _, core::ptr::null(), core::ptr::null(), 0, pos, l.pos.as_ptr() as _,
                        t as i32, s, 64, 0, sink_ptr, sc.attn.p(), STREAM), "prefill naive")?;
                }
            } else {
                // Decode. A windowed layer sees only its last `window` rows (every
                // earlier row is outside the window of this query), so the kernel
                // reads just those; the splits scale with the visible keys.
                let rows = l.rows;
                let k0 = if g.window > 0 { rows - (g.window as usize).min(rows) } else { 0 };
                let keys = rows - k0;
                let splits = decode_splits(keys);
                let kc = (l.k.as_ptr() as *const u8).add(k0 * l.row_k);
                let vc = (l.v.as_ptr() as *const u8).add(k0 * l.row_v);
                let kpos = (l.pos.as_ptr() as *const i64).add(k0);
                if tc {
                    ck(af::m26_attn_decode_splitkv_fp8_tc(&g, sc.q_rot.p(), kc, vc, core::ptr::null(), 0, pos, kpos, 1,
                        keys as i32, splits, 0, sc.partials.p(), STREAM), "decode tc")?;
                    ck(af::m26_attn_reduce_tc(&g, sc.partials.p(), sink_ptr, 1, splits, 0, sc.attn.p(), STREAM),
                        "reduce tc")?;
                } else {
                    ck(af::m26_attn_decode_splitkv_fp8(&g, sc.q_rot.p(), kc, core::ptr::null(), vc, core::ptr::null(),
                        core::ptr::null(), 0, pos, kpos, 1, keys as i32, splits, 0, sc.partials.p(), STREAM),
                        "decode naive")?;
                    ck(af::m26_attn_reduce(&g, sc.partials.p(), sink_ptr, 1, splits, 0, sc.attn.p(), STREAM),
                        "reduce naive")?;
                }
            }
        }
        self.stage(if ga { "a_kern_ga" } else { "a_kern_swa" });
        unsafe {
            self.lin(sc.attn.p(), &w_name(layer, "self_attn.o_proj.weight"), t, sc.o.p())?;
        }
        Ok(())
    }

    /// One decode step for several requests at once (batched serving, perf reset
    /// W4): row i is request i's last token `ids[i]` at position `kvs[i].tokens()`.
    /// The dense GEMMs, router, expert exchange and lm_head run over all rows; RoPE
    /// + KV store and the split-KV attention run per row against that request's
    /// cache. Over a pipelined wire the rows split into two lanes that alternate
    /// like the prefill's (W4 v2): one lane's attention runs while the Sparks
    /// compute the other's experts. Returns every row's logits, `[ids.len() * vocab]`.
    pub fn decode_batch(
        &mut self,
        ids: &[usize],
        kvs: &mut [&mut DeviceKv],
        wire: &mut WireClient,
    ) -> Result<Vec<f32>, String> {
        let n = ids.len();
        if n == 0 || n != kvs.len() || n > self.max_chunk {
            return Err(format!("decode_batch: {n} ids for {} caches (max {})", kvs.len(), self.max_chunk));
        }
        if kvs.iter().any(|kv| kv.max_chunk != self.max_chunk) {
            return Err("decode_batch: kv built for a different chunk size".into());
        }
        let cfg = self.cfg.clone();
        let hid = cfg.hidden_size;
        let eps = f64::from(cfg.layernorm_epsilon);
        // Lane l holds rows [start[l], start[l] + len[l]).
        let lanes = if wire.pipelined() && n >= 2 && self.decode_lanes > 1 { 2 } else { 1 };
        let n0 = if lanes == 2 { n.div_ceil(2) } else { n };
        let spans = [(0usize, n0), (n0, n - n0)];
        for (lane, &(st, len)) in spans.iter().enumerate().take(lanes) {
            let pos: Vec<i64> = kvs[st..st + len].iter().map(|kv| kv.tokens as i64).collect();
            self.sc.pos[lane].buf.upload_prefix(bytes_of(&pos)).map_err(cuda::error_string)?;
            self.h_embed.clear();
            for &id in &ids[st..st + len] {
                if id >= cfg.vocab_size {
                    return Err(format!("token id {id} out of vocab"));
                }
                self.h_embed.extend_from_slice(&self.embed[id * hid..(id + 1) * hid]);
            }
            self.sc.h[lane].buf.upload_prefix(bytes_of(&self.h_embed)).map_err(cuda::error_string)?;
        }
        let (kv0, kv1) = kvs.split_at_mut(n0);
        let mut pending = [false; 2];
        for layer in 0..cfg.num_hidden_layers {
            for lane in 0..lanes {
                let (_, t) = spans[lane];
                if std::mem::take(&mut pending[lane]) {
                    self.moe_finish(lane, t, wire)?;
                }
                let h = self.sc.h[lane].p::<f32>();
                unsafe {
                    ck(m26c_rmsnorm(h, self.ln_in[layer].as_ptr() as _, self.sc.x.p(), t as i32, hid as i32, eps,
                        STREAM), "rmsnorm in")?;
                }
                let rows: &mut [&mut DeviceKv] = if lane == 0 { &mut *kv0 } else { &mut *kv1 };
                self.attn_decode_rows(layer, lane, rows)?;
                unsafe {
                    ck(m26c_add_inplace(h, self.sc.o.p(), (t * hid) as i64, STREAM), "residual attn")?;
                    ck(m26c_rmsnorm(h, self.ln_post[layer].as_ptr() as _, self.sc.x.p(), t as i32, hid as i32, eps,
                        STREAM), "rmsnorm post")?;
                }
                pending[lane] = self.ffn_start(layer, t, lane, wire)?;
            }
        }
        for lane in 0..lanes {
            if pending[lane] {
                self.moe_finish(lane, spans[lane].1, wire)?;
            }
        }
        for kv in kvs.iter_mut() {
            kv.tokens += 1;
        }
        let vocab = cfg.vocab_size;
        self.sc.lm.ensure(n * vocab * 4)?;
        for (lane, &(st, len)) in spans.iter().enumerate().take(lanes) {
            unsafe {
                ck(m26c_rmsnorm(self.sc.h[lane].p(), self.final_norm.as_ptr() as _, self.sc.x.p(), len as i32,
                    hid as i32, eps, STREAM), "final norm")?;
                self.lin(self.sc.x.p(), "lm_head.weight", len, self.sc.lm.p::<f32>().add(st * vocab))?;
            }
        }
        let mut logits = vec![0f32; n * vocab];
        self.sc.lm.buf.download_prefix(bytes_of_mut(&mut logits)).map_err(cuda::error_string)?;
        Ok(logits)
    }

    /// [`Self::decode_batch`]'s attention: the qkv and o_proj GEMMs over all rows,
    /// the fused prep and the split-KV decode per row (its own cache and position).
    fn attn_decode_rows(&mut self, layer: usize, lane: usize, kvs: &mut [&mut DeviceKv]) -> Result<(), String> {
        let n = kvs.len();
        let cfg = &self.cfg;
        let kind = cfg.layer_kind(layer);
        let (q_rows, k_rows, v_rows, _o_in) = cfg.attn_dims(kind);
        let total = q_rows + k_rows + v_rows;
        let g = self.geom(kind);
        let n_q = cfg.num_attention_heads as i32;
        let n_kv = g.n_kv;
        let d_qk = cfg.head_dim as i32;
        let theta = match kind {
            LayerKind::Ga => f64::from(cfg.rope_theta),
            LayerKind::Swa => f64::from(cfg.swa_rope_theta),
        };
        let rot_dim = (f64::from(cfg.partial_rotary_factor) * f64::from(d_qk) + 1e-6) as i32;
        let vscale = cfg.attention_value_scale;
        let tc = Self::use_tc(&g);
        let sink_ptr = self.sink[layer].as_ref().map_or(core::ptr::null(), |b| b.as_ptr() as *const f32);
        let sc = &self.sc;
        let pos = sc.pos[lane].p::<i64>();
        unsafe {
            self.lin(sc.x.p(), &w_name(layer, "self_attn.qkv_proj.weight"), n, sc.qkv.p())?;
        }
        let (q_row, a_row) = (cfg.num_attention_heads * cfg.head_dim, cfg.num_attention_heads * cfg.v_head_dim);
        for (i, kv) in kvs.iter_mut().enumerate() {
            kv.prepare(layer, 1)?;
            let l = &mut kv.layers[layer];
            let row0 = l.rows;
            unsafe {
                let qi = sc.q_rot.p::<f32>().add(i * q_row);
                let pi = pos.add(i);
                ck(m26c_attn_prep(sc.qkv.p::<f32>().add(i * total), pi, 1, n_q, n_kv, d_qk, g.d_v, rot_dim, theta, vscale,
                    qi, (l.k.as_ptr() as *mut u8).add(row0 * l.row_k), (l.v.as_ptr() as *mut u8).add(row0 * l.row_v),
                    (l.pos.as_ptr() as *mut i64).add(row0), sc.clip.p(), STREAM), "attn prep")?;
                l.rows += 1;
                let rows = l.rows;
                let k0 = if g.window > 0 { rows - (g.window as usize).min(rows) } else { 0 };
                let keys = rows - k0;
                let splits = decode_splits(keys);
                let kc = (l.k.as_ptr() as *const u8).add(k0 * l.row_k);
                let vc = (l.v.as_ptr() as *const u8).add(k0 * l.row_v);
                let kpos = (l.pos.as_ptr() as *const i64).add(k0);
                let out = sc.attn.p::<f32>().add(i * a_row);
                if tc {
                    ck(af::m26_attn_decode_splitkv_fp8_tc(&g, qi, kc, vc, core::ptr::null(), 0, pi, kpos, 1, keys as i32,
                        splits, 0, sc.partials.p(), STREAM), "decode tc")?;
                    ck(af::m26_attn_reduce_tc(&g, sc.partials.p(), sink_ptr, 1, splits, 0, out, STREAM), "reduce tc")?;
                } else {
                    ck(af::m26_attn_decode_splitkv_fp8(&g, qi, kc, core::ptr::null(), vc, core::ptr::null(),
                        core::ptr::null(), 0, pi, kpos, 1, keys as i32, splits, 0, sc.partials.p(), STREAM),
                        "decode naive")?;
                    ck(af::m26_attn_reduce(&g, sc.partials.p(), sink_ptr, 1, splits, 0, out, STREAM), "reduce naive")?;
                }
            }
        }
        unsafe {
            self.lin(sc.attn.p(), &w_name(layer, "self_attn.o_proj.weight"), n, sc.o.p())?;
        }
        Ok(())
    }

    /// One speculative step (perf reset S1, DFlash): draft 7 tokens per request,
    /// verify `[last, d1..d_k]` through the target (`ks[i]` = drafts verified for
    /// request i, at most 7), accept the longest prefix the target's argmax
    /// agrees with, roll the rejected rows out of the KV and feed the accepted
    /// rows to the drafter. Returns each request's new tokens: the accepted
    /// drafts then the target's next token (`acc + 1` of them).
    pub fn spec_step(
        &mut self,
        kvs: &mut [&mut DeviceKv],
        lasts: &[usize],
        ks: &[usize],
        wire: &mut WireClient,
    ) -> Result<Vec<Vec<usize>>, String> {
        let n = kvs.len();
        if n == 0 || lasts.len() != n || ks.len() != n {
            return Err(format!("spec_step: {n} caches, {} tokens, {} ks", lasts.len(), ks.len()));
        }
        let starts: Vec<usize> = kvs.iter().map(|kv| kv.tokens).collect();
        let seqs: Vec<(usize, usize)> = (0..n).map(|i| (lasts[i], starts[i])).collect();
        let drafts = self.dflash_draft(kvs, &seqs)?;
        self.stage("draft");
        let ks: Vec<usize> = match self.spec_policy {
            SpecPolicy::Fixed => ks.iter().map(|&k| k.min(DFLASH_DRAFTS)).collect(),
            SpecPolicy::Confidence { a_ms, b_ms } => {
                let probs: Vec<[f32; DFLASH_DRAFTS]> = drafts.iter().map(|d| d.1).collect();
                verify_lengths(&probs, ks, a_ms, b_ms)
            }
            SpecPolicy::Chain { tau } => drafts.iter().zip(ks).map(|(d, &k)| chain_length(&d.1, k, tau)).collect(),
        };
        let blocks: Vec<Vec<usize>> = (0..n)
            .map(|i| std::iter::once(lasts[i]).chain(drafts[i].0[..ks[i]].iter().copied()).collect())
            .collect();
        let am = self.verify_batch(&blocks, kvs, wire)?;
        let mut out = Vec::with_capacity(n);
        let mut rows = Vec::with_capacity(am.len());
        let mut off = 0;
        for (i, b) in blocks.iter().enumerate() {
            let g = &am[off..off + b.len()];
            let mut acc = 0;
            while acc + 1 < b.len() && b[acc + 1] == g[acc] {
                acc += 1;
            }
            if self.spec_trace {
                let q: Vec<String> = drafts[i].1.iter().map(|p| format!("{p:.3}")).collect();
                eprintln!("SPEC req={i} k={} acc={acc} p=[{}] match=[{}]", b.len() - 1, q.join(","),
                    (0..DFLASH_DRAFTS).map(|j| if j < b.len() - 1 { if drafts[i].0[j] == g[j] { '1' } else { '0' } } else { '.' })
                        .collect::<String>());
            }
            kvs[i].truncate(starts[i] + acc + 1)?;
            for r in 0..b.len() {
                rows.push((if r <= acc { i as i32 } else { -1 }, (starts[i] + r) as i64));
            }
            out.push(b[1..=acc].iter().copied().chain(std::iter::once(g[acc])).collect());
            off += b.len();
        }
        let aux = self.sc.aux.p::<f32>();
        self.dflash_commit(kvs, aux, &rows)?;
        self.stage("commit");
        Ok(out)
    }

    /// The target's verify pass (perf reset S1): request i appends `blocks[i]` at
    /// `kvs[i].tokens()`. Dense GEMMs, router, exchange and lm_head run over all
    /// rows; attention runs per request over its own cache with the split-KV
    /// kernel (causal by position, so a block row sees the rows before it). Over
    /// a pipelined wire one request's block splits across the two lanes, several
    /// requests split by request. Returns every row's argmax in block order and
    /// leaves the rows' aux features in `sc.aux` (same order).
    fn verify_batch(
        &mut self,
        blocks: &[Vec<usize>],
        kvs: &mut [&mut DeviceKv],
        wire: &mut WireClient,
    ) -> Result<Vec<usize>, String> {
        let n = blocks.len();
        let total: usize = blocks.iter().map(Vec::len).sum();
        if n == 0 || n != kvs.len() || blocks.iter().any(|b| b.is_empty() || b.len() > DFLASH_BLOCK) {
            return Err(format!("verify_batch: {n} blocks for {} caches", kvs.len()));
        }
        if total > 2 * self.max_chunk {
            return Err(format!("verify_batch: {total} rows exceed two lanes of {}", self.max_chunk));
        }
        let cfg = self.cfg.clone();
        let hid = cfg.hidden_size;
        let eps = f64::from(cfg.layernorm_epsilon);
        // Lanes of (request, first block row, rows) segments, contiguous in block order.
        let lanes: Vec<Vec<(usize, usize, usize)>> = if !wire.pipelined() || self.verify_lanes == 1 || (n == 1 && total < 2) {
            vec![(0..n).map(|i| (i, 0, blocks[i].len())).collect()]
        } else if n == 1 {
            let a = total.div_ceil(2);
            vec![vec![(0, 0, a)], vec![(0, a, total - a)]]
        } else {
            let n0 = n.div_ceil(2);
            vec![(0..n0).map(|i| (i, 0, blocks[i].len())).collect(), (n0..n).map(|i| (i, 0, blocks[i].len())).collect()]
        };
        let rows: Vec<usize> = lanes.iter().map(|l| l.iter().map(|x| x.2).sum()).collect();
        if rows.iter().any(|&r| r > self.max_chunk) {
            return Err(format!("verify_batch: lane rows {rows:?} exceed {}", self.max_chunk));
        }
        let capture = self.dflash.is_some();
        if capture {
            self.sc.aux.ensure(total * dflash::TARGET_LAYERS.len() * hid * 4)?;
        }
        let mut base = [0usize; 2];
        for (lane, segs) in lanes.iter().enumerate() {
            base[lane] = if lane == 0 { 0 } else { rows[0] };
            let mut pos = Vec::with_capacity(rows[lane]);
            self.h_embed.clear();
            for &(i, r0, len) in segs {
                for j in r0..r0 + len {
                    let id = blocks[i][j];
                    if id >= cfg.vocab_size {
                        return Err(format!("token id {id} out of vocab"));
                    }
                    pos.push((kvs[i].tokens + j) as i64);
                    self.h_embed.extend_from_slice(&self.embed[id * hid..(id + 1) * hid]);
                }
            }
            self.sc.pos[lane].buf.upload_prefix(bytes_of(&pos)).map_err(cuda::error_string)?;
            self.sc.h[lane].buf.upload_prefix(bytes_of(&self.h_embed)).map_err(cuda::error_string)?;
        }
        let cap = |lane: usize| capture.then_some(base[lane] as isize);
        let mut pending = [false; 2];
        for layer in 0..cfg.num_hidden_layers {
            for lane in 0..lanes.len() {
                let t = rows[lane];
                if std::mem::take(&mut pending[lane]) {
                    self.moe_finish(lane, t, wire)?;
                    self.capture(lane, layer - 1, t, cap(lane))?;
                }
                let h = self.sc.h[lane].p::<f32>();
                unsafe {
                    ck(m26c_rmsnorm(h, self.ln_in[layer].as_ptr() as _, self.sc.x.p(), t as i32, hid as i32, eps,
                        STREAM), "rmsnorm in")?;
                }
                self.attn_segs(layer, lane, &lanes[lane], kvs)?;
                unsafe {
                    ck(m26c_add_inplace(h, self.sc.o.p(), (t * hid) as i64, STREAM), "residual attn")?;
                    ck(m26c_rmsnorm(h, self.ln_post[layer].as_ptr() as _, self.sc.x.p(), t as i32, hid as i32, eps,
                        STREAM), "rmsnorm post")?;
                }
                pending[lane] = self.ffn_start(layer, t, lane, wire)?;
                if !pending[lane] {
                    self.capture(lane, layer, t, cap(lane))?;
                }
            }
        }
        for lane in 0..lanes.len() {
            if pending[lane] {
                self.moe_finish(lane, rows[lane], wire)?;
                self.capture(lane, cfg.num_hidden_layers - 1, rows[lane], cap(lane))?;
            }
        }
        for (i, b) in blocks.iter().enumerate() {
            kvs[i].tokens += b.len();
        }
        self.stage("verify");
        let vocab = cfg.vocab_size;
        self.sc.lm.ensure(total * vocab * 4)?;
        self.sc.idx.ensure(total * 4)?;
        for lane in 0..lanes.len() {
            unsafe {
                ck(m26c_rmsnorm(self.sc.h[lane].p(), self.final_norm.as_ptr() as _, self.sc.x.p(), rows[lane] as i32,
                    hid as i32, eps, STREAM), "final norm")?;
                self.lin(self.sc.x.p(), "lm_head.weight", rows[lane], self.sc.lm.p::<f32>().add(base[lane] * vocab))?;
            }
        }
        unsafe {
            ck(dflash::m26c_argmax_rows(self.sc.lm.p(), vocab as i64, total as i32, vocab as i32, self.sc.idx.p(), STREAM),
                "verify argmax")?;
        }
        let mut am = vec![0i32; total];
        self.sc.idx.buf.download_prefix(bytes_of_mut(&mut am)).map_err(cuda::error_string)?;
        self.stage("lm_head");
        Ok(am.into_iter().map(|x| x as usize).collect())
    }

    /// Prefill several short prompts in one pass (perf reset B1). Request i's rows
    /// `segs[i]` are appended after `kvs[i].tokens()`. The dense GEMMs, router and
    /// expert exchange run over every request's rows together, so a burst of short
    /// prompts pays the per-layer latency (48 exchanges) once instead of once per
    /// prompt, and the Sparks stream each routed expert once per lane instead of
    /// once per prompt. Attention runs per request against its own cache (fused
    /// prep, then P2). A request's segment stays whole in one lane, so its rows
    /// append in order; segments are dealt to two lanes by row count. At most
    /// [`BATCH_ROWS`] rows in all: every row's DFlash aux is captured into the
    /// buffer sized at boot. Returns each request's greedy next token.
    pub fn prefill_batch(&mut self, segs: &[&[usize]], kvs: &mut [&mut DeviceKv], wire: &mut WireClient)
        -> Result<Vec<usize>, String> {
        let n = segs.len();
        let total: usize = segs.iter().map(|s| s.len()).sum();
        if n == 0 || n != kvs.len() || segs.iter().any(|s| s.is_empty()) || total > BATCH_ROWS {
            return Err(format!("prefill_batch: {n} prompts of {total} rows for {} caches", kvs.len()));
        }
        let cfg = self.cfg.clone();
        let hid = cfg.hidden_size;
        let eps = f64::from(cfg.layernorm_epsilon);
        let nl = if wire.pipelined() && n >= 2 { 2 } else { 1 };
        // (request, row offset in the lane, rows)
        let mut lanes: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); nl];
        let mut rows = vec![0usize; nl];
        for (i, seg) in segs.iter().enumerate() {
            let l = (0..nl).min_by_key(|&l| rows[l]).expect("a lane");
            lanes[l].push((i, rows[l], seg.len()));
            rows[l] += seg.len();
        }
        if rows.iter().any(|&r| r > self.max_chunk) {
            return Err(format!("prefill_batch: lane rows {rows:?} exceed {}", self.max_chunk));
        }
        let capture = self.dflash.is_some() && kvs.iter().all(|k| k.draft.is_some());
        if capture {
            self.sc.aux.ensure(total * dflash::TARGET_LAYERS.len() * hid * 4)?;
        }
        let mut base = [0usize; 2];
        for (lane, segs_l) in lanes.iter().enumerate() {
            base[lane] = if lane == 0 { 0 } else { rows[0] };
            let mut pos = Vec::with_capacity(rows[lane]);
            self.h_embed.clear();
            for &(i, _, _) in segs_l {
                for (j, &id) in segs[i].iter().enumerate() {
                    if id >= cfg.vocab_size {
                        return Err(format!("token id {id} out of vocab"));
                    }
                    pos.push((kvs[i].tokens + j) as i64);
                    self.h_embed.extend_from_slice(&self.embed[id * hid..(id + 1) * hid]);
                }
            }
            self.sc.pos[lane].buf.upload_prefix(bytes_of(&pos)).map_err(cuda::error_string)?;
            self.sc.h[lane].buf.upload_prefix(bytes_of(&self.h_embed)).map_err(cuda::error_string)?;
        }
        let cap = |lane: usize| capture.then_some(base[lane] as isize);
        let mut pending = [false; 2];
        for layer in 0..cfg.num_hidden_layers {
            for lane in 0..nl {
                let t = rows[lane];
                if std::mem::take(&mut pending[lane]) {
                    self.moe_finish(lane, t, wire)?;
                    self.capture(lane, layer - 1, t, cap(lane))?;
                }
                let h = self.sc.h[lane].p::<f32>();
                unsafe {
                    ck(m26c_rmsnorm(h, self.ln_in[layer].as_ptr() as _, self.sc.x.p(), t as i32, hid as i32, eps,
                        STREAM), "rmsnorm in")?;
                }
                self.attn_prefill_segs(layer, lane, &lanes[lane], kvs)?;
                unsafe {
                    ck(m26c_add_inplace(h, self.sc.o.p(), (t * hid) as i64, STREAM), "residual attn")?;
                    ck(m26c_rmsnorm(h, self.ln_post[layer].as_ptr() as _, self.sc.x.p(), t as i32, hid as i32, eps,
                        STREAM), "rmsnorm post")?;
                }
                pending[lane] = self.ffn_start(layer, t, lane, wire)?;
                if !pending[lane] {
                    self.capture(lane, layer, t, cap(lane))?;
                }
            }
        }
        for lane in 0..nl {
            if pending[lane] {
                self.moe_finish(lane, rows[lane], wire)?;
                self.capture(lane, cfg.num_hidden_layers - 1, rows[lane], cap(lane))?;
            }
        }
        for (i, seg) in segs.iter().enumerate() {
            kvs[i].tokens += seg.len();
        }
        // DFlash context: every captured row (a batch is at most CTX_TAIL rows, so
        // all lie in each prompt's kept tail), in aux order: lane 0, then lane 1.
        if capture {
            let mut crows = Vec::with_capacity(total);
            for segs_l in &lanes {
                for &(i, _, len) in segs_l {
                    let start = kvs[i].tokens - len;
                    crows.extend((0..len).map(|j| (i as i32, (start + j) as i64)));
                }
            }
            let aux = self.sc.aux.p::<f32>();
            self.dflash_commit(kvs, aux, &crows)?;
        }
        // Each request's last row: gathered, final norm, lm_head over n rows (one
        // pass over the lm_head weights), argmax.
        let vocab = cfg.vocab_size;
        self.sc.lm.ensure(n * vocab * 4)?;
        self.sc.idx.ensure(n * 4)?;
        unsafe {
            for (lane, segs_l) in lanes.iter().enumerate() {
                for &(i, off, len) in segs_l {
                    let src = self.sc.h[lane].p::<f32>().add((off + len - 1) * hid);
                    ck(cudaMemcpyAsync(self.sc.f.p::<f32>().add(i * hid) as _, src as _, hid * 4, cuda::D2D, STREAM),
                        "gather last rows")?;
                }
            }
            ck(m26c_rmsnorm(self.sc.f.p(), self.final_norm.as_ptr() as _, self.sc.x.p(), n as i32, hid as i32, eps,
                STREAM), "final norm")?;
            self.lin(self.sc.x.p(), "lm_head.weight", n, self.sc.lm.p())?;
            ck(dflash::m26c_argmax_rows(self.sc.lm.p(), vocab as i64, n as i32, vocab as i32, self.sc.idx.p(), STREAM),
                "prefill batch argmax")?;
        }
        let mut am = vec![0i32; n];
        self.sc.idx.buf.download_prefix(bytes_of_mut(&mut am)).map_err(cuda::error_string)?;
        Ok(am.into_iter().map(|x| x as usize).collect())
    }

    /// [`Self::prefill_batch`]'s attention for one lane: the qkv and o_proj GEMMs
    /// over the lane's rows; per request segment `(request, lane row, rows)`, the
    /// fused prep appends its rows to that request's cache and P2 attends them (the
    /// single-prompt `attn` path, per segment).
    fn attn_prefill_segs(
        &mut self,
        layer: usize,
        lane: usize,
        segs: &[(usize, usize, usize)],
        kvs: &mut [&mut DeviceKv],
    ) -> Result<(), String> {
        let t: usize = segs.iter().map(|x| x.2).sum();
        let cfg = &self.cfg;
        let kind = cfg.layer_kind(layer);
        let (q_rows, k_rows, v_rows, _o_in) = cfg.attn_dims(kind);
        let total = q_rows + k_rows + v_rows;
        let g = self.geom(kind);
        let n_q = cfg.num_attention_heads as i32;
        let n_kv = g.n_kv;
        let d_qk = cfg.head_dim as i32;
        let theta = match kind {
            LayerKind::Ga => f64::from(cfg.rope_theta),
            LayerKind::Swa => f64::from(cfg.swa_rope_theta),
        };
        let rot_dim = (f64::from(cfg.partial_rotary_factor) * f64::from(d_qk) + 1e-6) as i32;
        let vscale = cfg.attention_value_scale;
        let tc = Self::use_tc(&g);
        let p2 = tc && self.prefill_p2;
        let sink_ptr = self.sink[layer].as_ref().map_or(core::ptr::null(), |b| b.as_ptr() as *const f32);
        let sc = &self.sc;
        let pos = sc.pos[lane].p::<i64>();
        unsafe {
            self.lin(sc.x.p(), &w_name(layer, "self_attn.qkv_proj.weight"), t, sc.qkv.p())?;
        }
        let (q_row, a_row) = (cfg.num_attention_heads * cfg.head_dim, cfg.num_attention_heads * cfg.v_head_dim);
        for &(i, row, len) in segs {
            let kv = &mut *kvs[i];
            kv.prepare(layer, len)?;
            let l = &mut kv.layers[layer];
            let row0 = l.rows;
            unsafe {
                let qi = sc.q_rot.p::<f32>().add(row * q_row);
                let pi = pos.add(row);
                ck(m26c_attn_prep(sc.qkv.p::<f32>().add(row * total), pi, len as i32, n_q, n_kv, d_qk, g.d_v, rot_dim,
                    theta, vscale, qi, (l.k.as_ptr() as *mut u8).add(row0 * l.row_k),
                    (l.v.as_ptr() as *mut u8).add(row0 * l.row_v), (l.pos.as_ptr() as *mut i64).add(row0), sc.clip.p(),
                    STREAM), "batch prep")?;
                l.rows += len;
                let rows = l.rows;
                let out = sc.attn.p::<f32>().add(row * a_row);
                if len > 1 && p2 {
                    ck(af::m26_attn_prefill_fp8_fa(&g, qi, l.k.as_ptr() as _, l.v.as_ptr() as _, len as i32, rows as i32,
                        row0 as i32, 0, sink_ptr, out, STREAM), "batch prefill p2")?;
                } else {
                    // One row (or no P2): the split-KV kernel, as a verify segment.
                    let k0 = if g.window > 0 { rows - (g.window as usize + len - 1).min(rows) } else { 0 };
                    let keys = rows - k0;
                    let splits = decode_splits(keys);
                    let kc = (l.k.as_ptr() as *const u8).add(k0 * l.row_k);
                    let vc = (l.v.as_ptr() as *const u8).add(k0 * l.row_v);
                    let kpos = (l.pos.as_ptr() as *const i64).add(k0);
                    if tc {
                        ck(af::m26_attn_decode_splitkv_fp8_tc(&g, qi, kc, vc, core::ptr::null(), 0, pi, kpos, len as i32,
                            keys as i32, splits, 0, sc.partials.p(), STREAM), "batch split tc")?;
                        ck(af::m26_attn_reduce_tc(&g, sc.partials.p(), sink_ptr, len as i32, splits, 0, out, STREAM),
                            "batch reduce tc")?;
                    } else {
                        ck(af::m26_attn_decode_splitkv_fp8(&g, qi, kc, core::ptr::null(), vc, core::ptr::null(),
                            core::ptr::null(), 0, pi, kpos, len as i32, keys as i32, splits, 0, sc.partials.p(), STREAM),
                            "batch split naive")?;
                        ck(af::m26_attn_reduce(&g, sc.partials.p(), sink_ptr, len as i32, splits, 0, out, STREAM),
                            "batch reduce naive")?;
                    }
                }
            }
        }
        unsafe {
            self.lin(sc.attn.p(), &w_name(layer, "self_attn.o_proj.weight"), t, sc.o.p())?;
        }
        Ok(())
    }

    /// [`Self::verify_batch`]'s attention for one lane: the qkv and o_proj GEMMs
    /// over the lane's rows; per segment, the fused prep appends its rows to that
    /// request's cache and the split-KV kernel attends them (T = segment rows).
    fn attn_segs(
        &mut self,
        layer: usize,
        lane: usize,
        segs: &[(usize, usize, usize)],
        kvs: &mut [&mut DeviceKv],
    ) -> Result<(), String> {
        let t: usize = segs.iter().map(|x| x.2).sum();
        let cfg = &self.cfg;
        let kind = cfg.layer_kind(layer);
        let (q_rows, k_rows, v_rows, _o_in) = cfg.attn_dims(kind);
        let total = q_rows + k_rows + v_rows;
        let g = self.geom(kind);
        let n_q = cfg.num_attention_heads as i32;
        let n_kv = g.n_kv;
        let d_qk = cfg.head_dim as i32;
        let theta = match kind {
            LayerKind::Ga => f64::from(cfg.rope_theta),
            LayerKind::Swa => f64::from(cfg.swa_rope_theta),
        };
        let rot_dim = (f64::from(cfg.partial_rotary_factor) * f64::from(d_qk) + 1e-6) as i32;
        let vscale = cfg.attention_value_scale;
        let tc = Self::use_tc(&g);
        let sink_ptr = self.sink[layer].as_ref().map_or(core::ptr::null(), |b| b.as_ptr() as *const f32);
        let sc = &self.sc;
        let pos = sc.pos[lane].p::<i64>();
        unsafe {
            self.lin(sc.x.p(), &w_name(layer, "self_attn.qkv_proj.weight"), t, sc.qkv.p())?;
        }
        let (q_row, a_row) = (cfg.num_attention_heads * cfg.head_dim, cfg.num_attention_heads * cfg.v_head_dim);
        let mut row = 0usize;
        for &(i, _, len) in segs {
            let kv = &mut *kvs[i];
            kv.prepare(layer, len)?;
            let l = &mut kv.layers[layer];
            let row0 = l.rows;
            unsafe {
                let qi = sc.q_rot.p::<f32>().add(row * q_row);
                let pi = pos.add(row);
                ck(m26c_attn_prep(sc.qkv.p::<f32>().add(row * total), pi, len as i32, n_q, n_kv, d_qk, g.d_v, rot_dim, theta,
                    vscale, qi, (l.k.as_ptr() as *mut u8).add(row0 * l.row_k), (l.v.as_ptr() as *mut u8).add(row0 * l.row_v),
                    (l.pos.as_ptr() as *mut i64).add(row0), sc.clip.p(), STREAM), "attn prep")?;
                l.rows += len;
                // A windowed layer: the block's first query needs the window
                // before it, the last one its own rows too.
                let rows = l.rows;
                let k0 = if g.window > 0 { rows - (g.window as usize + len - 1).min(rows) } else { 0 };
                let keys = rows - k0;
                let splits = decode_splits(keys);
                let kc = (l.k.as_ptr() as *const u8).add(k0 * l.row_k);
                let vc = (l.v.as_ptr() as *const u8).add(k0 * l.row_v);
                let kpos = (l.pos.as_ptr() as *const i64).add(k0);
                let out = sc.attn.p::<f32>().add(row * a_row);
                if tc {
                    ck(af::m26_attn_decode_splitkv_fp8_tc(&g, qi, kc, vc, core::ptr::null(), 0, pi, kpos, len as i32,
                        keys as i32, splits, 0, sc.partials.p(), STREAM), "verify tc")?;
                    ck(af::m26_attn_reduce_tc(&g, sc.partials.p(), sink_ptr, len as i32, splits, 0, out, STREAM),
                        "verify reduce tc")?;
                } else {
                    ck(af::m26_attn_decode_splitkv_fp8(&g, qi, kc, core::ptr::null(), vc, core::ptr::null(),
                        core::ptr::null(), 0, pi, kpos, len as i32, keys as i32, splits, 0, sc.partials.p(), STREAM),
                        "verify naive")?;
                    ck(af::m26_attn_reduce(&g, sc.partials.p(), sink_ptr, len as i32, splits, 0, out, STREAM),
                        "verify reduce naive")?;
                }
            }
            row += len;
        }
        unsafe {
            self.lin(sc.attn.p(), &w_name(layer, "self_attn.o_proj.weight"), t, sc.o.p())?;
        }
        Ok(())
    }

    /// Post-attention half of a layer for `lane` (its normed input in `sc.x`).
    /// A dense layer runs its MLP and residual now (`Ok(false)`); a MoE layer
    /// routes, quantizes and sends the exchange (`Ok(true)`, collected by
    /// [`Self::moe_finish`]).
    fn ffn_start(&mut self, layer: usize, t: usize, lane: usize, wire: &mut WireClient) -> Result<bool, String> {
        let cfg = self.cfg.clone();
        let hid = cfg.hidden_size;
        if !cfg.is_moe_layer(layer) {
            let inter = cfg.intermediate_size;
            unsafe {
                self.lin(self.sc.x.p(), &w_name(layer, "mlp.gate_proj.weight"), t, self.sc.gate.p())?;
                self.lin(self.sc.x.p(), &w_name(layer, "mlp.up_proj.weight"), t, self.sc.up.p())?;
                ck(m26c_silu_mul(self.sc.gate.p(), self.sc.up.p(), (t * inter) as i64, STREAM), "silu mul")?;
                self.lin(self.sc.gate.p(), &w_name(layer, "mlp.down_proj.weight"), t, self.sc.f.p())?;
                ck(m26c_add_inplace(self.sc.h[lane].p(), self.sc.f.p(), (t * hid) as i64, STREAM), "residual ffn")?;
            }
            self.stage("dense_mlp");
            return Ok(false);
        }
        let topk = cfg.num_experts_per_tok;
        let bias = self.router_bias[layer].as_ref().ok_or("moe layer without router bias")?;
        unsafe {
            self.dense.gemm_dev(self.sc.x.p(), &w_name(layer, "mlp.gate.weight"), t, self.sc.logits.p())?;
            ck(m26c_router_topk(self.sc.logits.p(), bias.as_ptr() as _, t as i32, cfg.n_routed_experts as i32,
                topk as i32, self.sc.idx.p(), self.sc.wts.p(), STREAM), "router")?;
            ck(m26c_quant_scales(self.sc.x.p(), (t * hid / 32) as i64, self.sc.scales.p(), self.sc.scale_inv.p(), STREAM),
                "quant scales")?;
            ck(af::m26_quantize_hidden_fp8(self.sc.x.p(), self.sc.scale_inv.p(), self.sc.payload.p(), (t * hid) as i64,
                STREAM), "quant payload")?;
        }
        self.stage("router_quant");
        if wire.pipelined() && t * hid * 2 <= ZERO_COPY_PLANE && !self.mapped.borrow().is_empty() {
            // Perf reset P9: at decode/verify sizes the GPU writes the route entries
            // and hidden rows into the mapped request frame; one sync, no route
            // download. (At prefill sizes the kernel's PCIe writes are slower than
            // P6's DMA copies: X1a 3,675 vs ~4,000 tok/s.)
            let me: &Self = self;
            let (idx, wts) = (me.sc.idx.p::<i32>() as *const i32, me.sc.wts.p::<f32>() as *const f32);
            let (payload, scales) = (me.sc.payload.p::<u8>() as *const u8, me.sc.scales.p::<u8>() as *const u8);
            wire.moe_send_device(layer as u32, t, topk, |routes, hidden, pitch| unsafe {
                let r = me.dev_ptr(routes).ok_or("request frame body is not mapped")?;
                let h = me.dev_ptr(hidden).ok_or("request frame body is not mapped")?;
                ck(m26c_frame_fill(idx, wts, payload, scales, t as i32, topk as i32, hid as i32, r, h, pitch as i32,
                    STREAM), "frame fill")?;
                ck(cuda::cudaDeviceSynchronize(), "frame fill sync")
            })?;
            self.stage("send");
            return Ok(true);
        }
        self.h_idx.resize(t * topk, 0);
        self.h_wts.resize(t * topk, 0.0);
        self.sc.idx.buf.download_prefix(bytes_of_mut(&mut self.h_idx)).map_err(cuda::error_string)?;
        self.sc.wts.buf.download_prefix(bytes_of_mut(&mut self.h_wts)).map_err(cuda::error_string)?;
        let routes: Vec<(u32, f32)> = self.h_idx.iter().zip(&self.h_wts).map(|(&e, &w)| (e as u32, w)).collect();
        if wire.pipelined() {
            // Perf reset P6: over RDMA the payload and scales go from the device
            // straight into the registered request frame (hidden row pitch 4,224),
            // and the send is posted without waiting for its completion.
            let (payload, scales) = (self.sc.payload.p::<u8>() as *const u8, self.sc.scales.p::<u8>() as *const u8);
            wire.moe_send_mapped(layer as u32, &routes, topk, |dst, pitch| unsafe {
                ck(cuda::cudaMemcpy2D(dst as _, pitch, payload as _, hid, hid, t, cuda::D2H), "payload to frame")?;
                ck(cuda::cudaMemcpy2D(dst.add(hid) as _, pitch, scales as _, hid / 32, hid / 32, t, cuda::D2H),
                    "scales to frame")
            })?;
            self.stage("send");
            return Ok(true);
        }
        self.h_payload.resize(t * hid, 0);
        self.h_scales.resize(t * hid / 32, 0);
        self.sc.payload.buf.download_prefix(&mut self.h_payload).map_err(cuda::error_string)?;
        self.sc.scales.buf.download_prefix(&mut self.h_scales).map_err(cuda::error_string)?;
        self.stage("d2h");
        wire.moe_send_raw(layer as u32, &self.h_payload, &self.h_scales, &routes, topk)?;
        self.stage("send");
        Ok(true)
    }

    /// Collect `lane`'s oldest in-flight exchange: the ranks' BF16 planes go
    /// straight to the device and are summed there in rank order (bit-identical
    /// to the host CoordinatorSum, perf reset R2), then added to the lane's
    /// residual stream.
    fn moe_finish(&mut self, lane: usize, t: usize, wire: &mut WireClient) -> Result<(), String> {
        let hid = self.cfg.hidden_size;
        let (layer, n) = wire.moe_recv_raw()?;
        if n != t {
            return Err(format!("moe_finish: layer {layer} returned {n} rows, lane {lane} has {t}"));
        }
        self.stage("recv_wait");
        let plane_bytes = n * hid * 2;
        let base = self.sc.planes.p::<u8>();
        // Perf reset P9: small returns (decode, verify) are summed in place from
        // the mapped RDMA ring; large ones are uploaded first (DMA).
        let mapped: Option<Vec<*const u16>> = if plane_bytes <= ZERO_COPY_PLANE {
            (0..4).map(|r| self.dev_ptr(wire.rank_plane(r, n).as_ptr()).map(|p| p as *const u16)).collect()
        } else {
            None
        };
        let planes: Vec<*const u16> = match mapped {
            Some(p) => p,
            None => {
                for r in 0..4 {
                    let src = wire.rank_plane(r, n);
                    unsafe {
                        ck(cuda::cudaMemcpy(base.add(r * plane_bytes) as _, src.as_ptr() as _, plane_bytes, cuda::H2D),
                            "plane upload")?;
                    }
                }
                (0..4).map(|r| unsafe { base.add(r * plane_bytes) } as *const u16).collect()
            }
        };
        unsafe {
            ck(m26c_rank_sum_bf16(planes[0], planes[1], planes[2], planes[3], self.sc.f.p(), (n * hid) as i64, STREAM),
                "rank sum")?;
            ck(m26c_add_inplace(self.sc.h[lane].p(), self.sc.f.p(), (n * hid) as i64, STREAM), "residual ffn")?;
        }
        self.stage("planes_sum");
        Ok(())
    }
}

/// Cut `n` tokens into `(start, len)` chunks of at most `max_chunk`. Unpaired:
/// full chunks then the remainder (the serial path's historical cut). Paired
/// (R4): an even count of balanced chunks, unless the prompt is too short for
/// two lanes of [`PAIR_MIN`] rows.
fn chunk_spans(n: usize, max_chunk: usize, paired: bool) -> Vec<(usize, usize)> {
    if !paired {
        return (0..n).step_by(max_chunk).map(|s| (s, max_chunk.min(n - s))).collect();
    }
    let mut c = n.div_ceil(max_chunk);
    if c % 2 == 1 && n >= 2 * PAIR_MIN {
        c += 1;
    }
    let (base, rem) = (n / c, n % c);
    let mut spans = Vec::with_capacity(c);
    let mut s = 0;
    for i in 0..c {
        let t = base + usize::from(i < rem);
        spans.push((s, t));
        s += t;
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_lengths_follow_confidence_and_caps() {
        let hi = [0.99f32, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93];
        let lo = [0.30f32, 0.20, 0.10, 0.05, 0.05, 0.05, 0.05];
        // Confident drafts are all verified; unconfident ones barely at all.
        assert_eq!(verify_lengths(&[hi], &[7], 25.0, 3.7), vec![7]);
        assert!(verify_lengths(&[lo], &[7], 25.0, 3.7)[0] <= 1);
        // Caps (the token budget) bind.
        assert_eq!(verify_lengths(&[hi], &[3], 25.0, 3.7), vec![3]);
        assert_eq!(verify_lengths(&[hi, hi], &[0, 7], 25.0, 3.7), vec![0, 7]);
        // Mixed batch: rows go to the confident request first.
        let ks = verify_lengths(&[lo, hi], &[7, 7], 25.0, 3.7);
        assert!(ks[1] == 7 && ks[0] <= 2, "{ks:?}");
        // Zero per-row cost verifies everything the caps allow.
        assert_eq!(verify_lengths(&[lo], &[7], 25.0, 0.0), vec![7]);
    }

    #[test]
    fn chunk_spans_cover_and_pair() {
        for &(n, mc) in &[(1, 2048), (100, 2048), (127, 2048), (128, 2048), (2048, 2048), (2049, 2048), (4096, 2048),
            (5000, 2048), (1_000_000, 2048), (7, 3)]
        {
            for paired in [false, true] {
                let sp = chunk_spans(n, mc, paired);
                let mut s = 0;
                for &(a, t) in &sp {
                    assert_eq!(a, s);
                    assert!(t >= 1 && t <= mc, "n={n} mc={mc} paired={paired} t={t}");
                    s += t;
                }
                assert_eq!(s, n);
                if paired && n >= 2 * PAIR_MIN {
                    assert_eq!(sp.len() % 2, 0, "n={n}");
                    let (lo, hi) = (sp.iter().map(|x| x.1).min().unwrap(), sp.iter().map(|x| x.1).max().unwrap());
                    assert!(hi - lo <= 1);
                }
            }
        }
        assert_eq!(chunk_spans(4096, 2048, true), vec![(0, 2048), (2048, 2048)]);
        assert_eq!(chunk_spans(4096, 2048, false), vec![(0, 2048), (2048, 2048)]);
        assert_eq!(chunk_spans(1000, 2048, true), vec![(0, 500), (500, 500)]);
        assert_eq!(chunk_spans(100, 2048, true), vec![(0, 100)]);
        assert_eq!(chunk_spans(1, 2048, true), vec![(0, 1)]);
    }
}
