//! Vision tower (perf reset V2): MiMo-V2.6-Flash's image encoder (`visual.*`) on the
//! coordinator GPU, for chat requests with images.
//!
//! The checkpoint's `MiMoVisionTransformer` (kernels in `kernels/vision.cu`): a patch embedding,
//! 28 blocks (RMSNorm, 32/8-head attention with 2-D rotary and head dim 64, SwiGLU), and a merger
//! that turns each 2x2 group of 16-pixel patches into one 4096-wide language-model embedding. Those
//! rows replace the `<|image_pad|>` embeddings of the prompt; the language model itself does not
//! change (1-D positions; see `api.rs` for how image tokens are carried).
//!
//! The 728.6M parameters (1.46 GB BF16) stay in page-locked host memory and are copied to the GPU
//! only while images are encoded, then freed, so the KV budget and the 1M-token context are
//! unchanged when no image is in flight. An image of `n` patches needs about 30 KB of GPU scratch
//! per patch plus ~0.4 GB of fixed chunk buffers.

use std::path::Path;

use mimo26_attn::cuda;
use mimo26_attn::device::DeviceBuffer;
use mimo26_attn::ffi::{CudaError, CudaStream};
use mimo26_repack::safetensors::SafetensorsHeader;

use crate::cublas::{self, Handle};
use crate::gpu_dense::f32_to_bf16_rne;
use crate::json::{self, JsonValue};

const STREAM: CudaStream = core::ptr::null_mut();

pub const HIDDEN: usize = 1280;
pub const OUT_DIM: usize = 4096;
pub const PATCH_DIM: usize = 1536;
const DEPTH: usize = 28;
const HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const QKV: usize = (HEADS + 2 * KV_HEADS) * HEAD_DIM;
const ATTN: usize = HEADS * HEAD_DIM;
const INTER: usize = 4608;
const MERGED: usize = 4 * HIDDEN;
const EPS: f32 = 1e-6;
/// Rows per GEMM chunk: bounds the MLP and QKV scratch at any image size.
const CHUNK: usize = 4096;

unsafe extern "C" {
    fn m26v_rmsnorm_bf16(x: *const f32, w: *const f32, y: *mut u16, rows: i32, dim: i32, eps: f32, s: CudaStream) -> CudaError;
    fn m26v_layernorm_bf16(x: *const f32, w: *const f32, y: *mut u16, rows: i32, dim: i32, eps: f32, s: CudaStream)
        -> CudaError;
    fn m26v_add_bias_residual(x: *mut f32, y: *const f32, b: *const f32, rows: i64, cols: i32, s: CudaStream) -> CudaError;
    fn m26v_swiglu_bias_bf16(gu: *const f32, b: *const f32, y: *mut u16, rows: i64, inter: i32, s: CudaStream) -> CudaError;
    fn m26v_gelu_bf16(x: *const f32, y: *mut u16, n: i64, s: CudaStream) -> CudaError;
    fn m26v_gather_units(x: *const f32, y: *mut f32, idx: *const i32, units: i64, cols: i32, s: CudaStream) -> CudaError;
    fn m26v_rope_qkv(qkv: *const f32, bias: *const f32, hw: *const i32, inv_freq: *const f32, t0: i32, rows: i32,
        q: *mut u16, k: *mut u16, v: *mut u16, s: CudaStream) -> CudaError;
    fn m26v_attn(q: *const u16, k: *const u16, v: *const u16, n: i32, window: i32, sinks: *const f32, out: *mut u16,
        s: CudaStream) -> CudaError;
}

fn ck(rc: CudaError, what: &str) -> Result<(), String> {
    if rc == cuda::SUCCESS { Ok(()) } else { Err(format!("vision {what}: {}", cuda::error_string(rc))) }
}

/// One block's weights: byte offsets into the blob (matrices BF16, vectors FP32).
struct Block {
    qkv: usize,
    qkv_b: usize,
    proj: usize,
    proj_b: usize,
    /// gate rows then up rows: one [9216, 1280] GEMM gives `[gate | up]` per row.
    gu: usize,
    gu_b: usize,
    down: usize,
    down_b: usize,
    n1: usize,
    n2: usize,
    sinks: Option<usize>,
    /// 0 = full attention, else the band half-width.
    window: i32,
    /// Window type 1: the block sees the merge units in column-major order.
    col: bool,
}

pub struct VisionTower {
    blob: Vec<u8>,
    pinned: bool,
    patch: usize,
    ln: usize,
    m0: usize,
    m2: usize,
    inv_freq: usize,
    blocks: Vec<Block>,
    handle: Handle,
    /// `<|vision_start|>` and `<|vision_end|>`, which bracket an image's tokens in a prompt.
    pub vision_start: u32,
    pub vision_end: u32,
}

// SAFETY: the blob's page-locked registration and the cuBLAS handle are only used from the
// thread that owns the tower (the scheduler); moving it there is fine.
unsafe impl Send for VisionTower {}

fn field<'a>(o: &'a JsonValue, k: &str) -> Option<&'a JsonValue> {
    match o {
        JsonValue::Object(m) => m.get(k),
        _ => None,
    }
}

fn int(o: &JsonValue, k: &str) -> Result<i64, String> {
    field(o, k).and_then(|v| v.as_int()).ok_or_else(|| format!("config {k} missing"))
}

fn ints(o: &JsonValue, k: &str) -> Result<Vec<i64>, String> {
    match field(o, k) {
        Some(JsonValue::Array(a)) => a.iter().map(|v| v.as_int().ok_or_else(|| format!("vision_config.{k}"))).collect(),
        _ => Err(format!("vision_config.{k} missing")),
    }
}

fn push(blob: &mut Vec<u8>, bytes: &[u8]) -> usize {
    while blob.len() % 256 != 0 {
        blob.push(0);
    }
    let off = blob.len();
    blob.extend_from_slice(bytes);
    off
}

fn bf16_to_f32_bytes(raw: &[u8]) -> Vec<u8> {
    raw.chunks_exact(2)
        .flat_map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).to_le_bytes())
        .collect()
}

impl VisionTower {
    /// From a checkpoint directory: `config.json`'s `vision_config` and the shard holding `visual.*`.
    pub fn load_dir(dir: &Path) -> Result<Self, String> {
        let cfg = std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?;
        for i in 0..64usize {
            let path = dir.join(format!("model_pp0_ep{i}_shard0.safetensors"));
            let Ok(hdr) = SafetensorsHeader::read(&path) else { continue };
            if hdr.tensors.contains_key("visual.patch_embed.proj.weight") {
                return Self::load(&path, &cfg);
            }
        }
        Err("no visual.* tensors in the checkpoint".into())
    }

    /// From one safetensors file holding every `visual.*` tensor, and the checkpoint config text.
    pub fn load(path: &Path, config_json: &str) -> Result<Self, String> {
        let cfg = json::parse(config_json).map_err(|e| format!("config.json: {e:?}"))?;
        let vc = field(&cfg, "vision_config").ok_or("config.json has no vision_config")?;
        let head_dim = field(vc, "qk_channels").and_then(|v| v.as_int()).unwrap_or(64);
        let expect = [
            ("depth", DEPTH as i64),
            ("hidden_size", HIDDEN as i64),
            ("num_heads", HEADS as i64),
            ("num_key_value_heads", KV_HEADS as i64),
            ("intermediate_size", INTER as i64),
            ("out_hidden_size", OUT_DIM as i64),
            ("patch_size", 16),
            ("temporal_patch_size", 2),
            ("spatial_merge_size", 2),
        ];
        for (k, want) in expect {
            let got = int(vc, k)?;
            if got != want {
                return Err(format!("vision_config.{k} = {got}, this engine implements {want}"));
            }
        }
        if head_dim != HEAD_DIM as i64 {
            return Err(format!("vision head dim {head_dim}, this engine implements {HEAD_DIM}"));
        }
        let full = ints(vc, "fullatt_block_indexes")?;
        let types = ints(vc, "vit_window_attn_types")?;
        if types.len() != DEPTH {
            return Err("vision_config.vit_window_attn_types must have one entry per block".into());
        }
        let band = int(vc, "visual_token_window_size")?;
        let vision_start = int(&cfg, "vision_start_token_id")? as u32;
        let vision_end = int(&cfg, "vision_end_token_id")? as u32;
        let use_sink = matches!(field(vc, "use_sink"), Some(JsonValue::Bool(true)));
        if types.last() == Some(&1) {
            return Err("the last vision block must see row order".into());
        }

        let hdr = SafetensorsHeader::read(path).map_err(|e| format!("{}: {e:?}", path.display()))?;
        let get = |name: &str, want: &[usize]| -> Result<Vec<u8>, String> {
            let full_name = format!("visual.{name}");
            let e = hdr.tensors.get(&full_name).ok_or_else(|| format!("missing {full_name}"))?;
            if e.dtype != "BF16" || e.shape != want {
                return Err(format!("{full_name}: {} {:?}, want BF16 {want:?}", e.dtype, e.shape));
            }
            hdr.read_tensor(path, &full_name).map_err(|e| format!("{full_name}: {e:?}"))
        };
        let mut blob = Vec::with_capacity(1_470_000_000);
        let patch = push(&mut blob, &get("patch_embed.proj.weight", &[HIDDEN, 3, 2, 16, 16])?);
        let mut blocks = Vec::with_capacity(DEPTH);
        for i in 0..DEPTH {
            let b = |s: &str| format!("blocks.{i}.{s}");
            let is_full = full.contains(&(i as i64));
            let qkv = push(&mut blob, &get(&b("attn.qkv.weight"), &[QKV, HIDDEN])?);
            let qkv_b = push(&mut blob, &bf16_to_f32_bytes(&get(&b("attn.qkv.bias"), &[QKV])?));
            let proj = push(&mut blob, &get(&b("attn.proj.weight"), &[HIDDEN, ATTN])?);
            let proj_b = push(&mut blob, &bf16_to_f32_bytes(&get(&b("attn.proj.bias"), &[HIDDEN])?));
            let mut gu_w = get(&b("mlp.gate_proj.weight"), &[INTER, HIDDEN])?;
            gu_w.extend_from_slice(&get(&b("mlp.up_proj.weight"), &[INTER, HIDDEN])?);
            let gu = push(&mut blob, &gu_w);
            let mut gu_bias = bf16_to_f32_bytes(&get(&b("mlp.gate_proj.bias"), &[INTER])?);
            gu_bias.extend_from_slice(&bf16_to_f32_bytes(&get(&b("mlp.up_proj.bias"), &[INTER])?));
            let gu_b = push(&mut blob, &gu_bias);
            let down = push(&mut blob, &get(&b("mlp.down_proj.weight"), &[HIDDEN, INTER])?);
            let down_b = push(&mut blob, &bf16_to_f32_bytes(&get(&b("mlp.down_proj.bias"), &[HIDDEN])?));
            let n1 = push(&mut blob, &bf16_to_f32_bytes(&get(&b("norm1.weight"), &[HIDDEN])?));
            let n2 = push(&mut blob, &bf16_to_f32_bytes(&get(&b("norm2.weight"), &[HIDDEN])?));
            let sinks = if use_sink && !is_full {
                Some(push(&mut blob, &bf16_to_f32_bytes(&get(&b("attn.sinks"), &[HEADS])?)))
            } else {
                None
            };
            let window = if is_full { 0 } else { band as i32 };
            blocks.push(Block { qkv, qkv_b, proj, proj_b, gu, gu_b, down, down_b, n1, n2, sinks, window, col: types[i] == 1 });
        }
        let ln = push(&mut blob, &bf16_to_f32_bytes(&get("merger.ln_q.weight", &[HIDDEN])?));
        let m0 = push(&mut blob, &get("merger.mlp.0.weight", &[MERGED, MERGED])?);
        let m2 = push(&mut blob, &get("merger.mlp.2.weight", &[OUT_DIM, MERGED])?);
        // MiMoVisionRotaryEmbedding(head_dim / 2): 1 / 10000^(arange(0, 32, 2) / 32), in FP32.
        let inv: Vec<u8> = (0..16)
            .flat_map(|i| (1.0f32 / 10000f32.powf((2 * i) as f32 / 32.0)).to_le_bytes())
            .collect();
        let inv_freq = push(&mut blob, &inv);
        let pinned = unsafe { cuda::cudaHostRegister(blob.as_mut_ptr() as _, blob.len(), 0) } == cuda::SUCCESS;
        let handle = Handle::new_with_math(cublas::DEFAULT_MATH)?;
        Ok(VisionTower { blob, pinned, patch, ln, m0, m2, inv_freq, blocks, handle, vision_start, vision_end })
    }

    /// Weight bytes copied to the GPU per encode.
    pub fn weight_bytes(&self) -> usize {
        self.blob.len()
    }

    /// GPU bytes an image of `n` patches needs beyond the weights.
    pub fn scratch_bytes(n: usize) -> usize {
        let units = n / 4;
        n * (HIDDEN * 4 * 2 + HIDDEN * 2 + PATCH_DIM * 2 + ATTN * 2 * 2 + 2 * KV_HEADS * HEAD_DIM * 2 + 2 * 4 * 2)
            + units * (OUT_DIM * 4 + 2 * 4)
            + CHUNK * (QKV * 4 + HIDDEN * 4 + 2 * INTER * 4 + INTER * 2 + MERGED * 4 + MERGED * 2)
    }

    /// Copy the weights to the GPU for one request's images (freed when the result is dropped).
    pub fn upload(&self) -> Result<DeviceTower<'_>, String> {
        let w = DeviceBuffer::alloc(self.blob.len()).map_err(|e| format!("vision alloc: {}", cuda::error_string(e)))?;
        w.upload(&self.blob).map_err(|e| format!("vision weights upload: {}", cuda::error_string(e)))?;
        Ok(DeviceTower { t: self, w })
    }
}

/// The tower's weights on the GPU, for the images of one request.
pub struct DeviceTower<'a> {
    t: &'a VisionTower,
    w: DeviceBuffer,
}

impl DeviceTower<'_> {
    /// Encode one image: `pixels` is `[grid_h * grid_w, 1536]` in the HF patch order. Returns
    /// `[grid_h * grid_w / 4, 4096]`, rounded to BF16 like the reference's cast to the embedding
    /// dtype. With `dump`, also the residual stream after each block, in row order.
    pub fn encode(&self, pixels: &[f32], grid_h: usize, grid_w: usize, dump: bool) -> Result<(Vec<f32>, Vec<Vec<f32>>), String> {
        let tw = self.t;
        let n = grid_h * grid_w;
        if grid_h % 2 != 0 || grid_w % 2 != 0 || n == 0 || pixels.len() != n * PATCH_DIM {
            return Err(format!("vision: bad patch grid {grid_h}x{grid_w} for {} values", pixels.len()));
        }
        let units = n / 4;
        let (hw_row, hw_col, col_idx, inv_idx) = positions(grid_h, grid_w);
        let al = |bytes: usize| DeviceBuffer::alloc(bytes.max(256)).map_err(|e| format!("vision alloc: {}", cuda::error_string(e)));
        let wp = |off: usize| unsafe { (self.w.as_ptr() as *const u8).add(off) };
        let x = al(n * HIDDEN * 4)?;
        let xt = al(n * HIDDEN * 4)?;
        let h = al(n * HIDDEN * 2)?;
        let q = al(n * ATTN * 2)?;
        let k = al(n * KV_HEADS * HEAD_DIM * 2)?;
        let v = al(n * KV_HEADS * HEAD_DIM * 2)?;
        let a = al(n * ATTN * 2)?;
        let out = al(units * OUT_DIM * 4)?;
        let qkv_c = al(CHUNK * QKV * 4)?;
        let o_c = al(CHUNK * HIDDEN * 4)?;
        let gu_c = al(CHUNK * 2 * INTER * 4)?;
        let m_c = al(CHUNK * INTER * 2)?;
        let m1_c = al(CHUNK * MERGED * 4)?;
        let g_c = al(CHUNK * MERGED * 2)?;
        let dev_i32 = |v: &[i32]| -> Result<DeviceBuffer, String> {
            let b = al(v.len() * 4)?;
            b.upload_prefix(bytes_of(v)).map_err(|e| format!("vision upload: {}", cuda::error_string(e)))?;
            Ok(b)
        };
        let (hw_r, hw_c, col_d, inv_d) = (dev_i32(&hw_row)?, dev_i32(&hw_col)?, dev_i32(&col_idx)?, dev_i32(&inv_idx)?);

        let gemm = |a_: *const u8, w_: usize, m: usize, n_out: usize, kdim: usize, c: *mut u8| unsafe {
            cublas::gemm_bf16_nt(&tw.handle, a_ as *const u16, wp(w_) as *const u16, m, n_out, kdim, c as *mut f32)
        };
        let fp = |b: &DeviceBuffer| b.as_ptr() as *mut u8;
        let mut dumps = Vec::new();
        unsafe {
            {
                let pix: Vec<u16> = pixels.iter().map(|&p| f32_to_bf16_rne(p)).collect();
                let pb = al(pix.len() * 2)?;
                pb.upload_prefix(bytes_of(&pix)).map_err(|e| format!("vision pixels upload: {}", cuda::error_string(e)))?;
                gemm(fp(&pb), tw.patch, n, HIDDEN, PATCH_DIM, fp(&x));
            }
            let (mut xs, mut xo) = (&x, &xt);
            let mut col = false;
            for blk in &tw.blocks {
                if blk.col != col {
                    let idx = if blk.col { &col_d } else { &inv_d };
                    ck(m26v_gather_units(fp(xs) as _, fp(xo) as _, idx.as_ptr() as _, units as i64, HIDDEN as i32, STREAM), "gather")?;
                    std::mem::swap(&mut xs, &mut xo);
                    col = blk.col;
                }
                let hw = if col { &hw_c } else { &hw_r };
                let xp = fp(xs) as *mut f32;
                ck(m26v_rmsnorm_bf16(xp, wp(blk.n1) as _, fp(&h) as _, n as i32, HIDDEN as i32, EPS, STREAM), "norm1")?;
                for r0 in (0..n).step_by(CHUNK) {
                    let rows = CHUNK.min(n - r0);
                    gemm(fp(&h).add(r0 * HIDDEN * 2), blk.qkv, rows, QKV, HIDDEN, fp(&qkv_c));
                    ck(m26v_rope_qkv(fp(&qkv_c) as _, wp(blk.qkv_b) as _, hw.as_ptr() as _, wp(tw.inv_freq) as _, r0 as i32,
                        rows as i32, fp(&q) as _, fp(&k) as _, fp(&v) as _, STREAM), "rope")?;
                }
                let sinks = blk.sinks.map_or(core::ptr::null(), |o| wp(o) as *const f32);
                ck(m26v_attn(fp(&q) as _, fp(&k) as _, fp(&v) as _, n as i32, blk.window, sinks, fp(&a) as _, STREAM), "attention")?;
                for r0 in (0..n).step_by(CHUNK) {
                    let rows = CHUNK.min(n - r0);
                    gemm(fp(&a).add(r0 * ATTN * 2), blk.proj, rows, HIDDEN, ATTN, fp(&o_c));
                    ck(m26v_add_bias_residual(xp.add(r0 * HIDDEN), fp(&o_c) as _, wp(blk.proj_b) as _, rows as i64,
                        HIDDEN as i32, STREAM), "proj residual")?;
                }
                ck(m26v_rmsnorm_bf16(xp, wp(blk.n2) as _, fp(&h) as _, n as i32, HIDDEN as i32, EPS, STREAM), "norm2")?;
                for r0 in (0..n).step_by(CHUNK) {
                    let rows = CHUNK.min(n - r0);
                    gemm(fp(&h).add(r0 * HIDDEN * 2), blk.gu, rows, 2 * INTER, HIDDEN, fp(&gu_c));
                    ck(m26v_swiglu_bias_bf16(fp(&gu_c) as _, wp(blk.gu_b) as _, fp(&m_c) as _, rows as i64, INTER as i32, STREAM),
                        "swiglu")?;
                    gemm(fp(&m_c), blk.down, rows, HIDDEN, INTER, fp(&o_c));
                    ck(m26v_add_bias_residual(xp.add(r0 * HIDDEN), fp(&o_c) as _, wp(blk.down_b) as _, rows as i64,
                        HIDDEN as i32, STREAM), "mlp residual")?;
                }
                if dump {
                    let mut host = vec![0f32; n * HIDDEN];
                    xs.download_prefix(bytes_of_mut(&mut host)).map_err(|e| format!("vision dump: {}", cuda::error_string(e)))?;
                    if col {
                        let mut row = vec![0f32; n * HIDDEN];
                        for (kk, &u) in col_idx.iter().enumerate() {
                            let (src, dst) = (kk * 4 * HIDDEN, u as usize * 4 * HIDDEN);
                            row[dst..dst + 4 * HIDDEN].copy_from_slice(&host[src..src + 4 * HIDDEN]);
                        }
                        host = row;
                    }
                    dumps.push(host);
                }
            }
            if col {
                return Err("vision: the last block left column order".into());
            }
            ck(m26v_layernorm_bf16(fp(xs) as _, wp(tw.ln) as _, fp(&h) as _, n as i32, HIDDEN as i32, EPS, STREAM), "ln_q")?;
            for u0 in (0..units).step_by(CHUNK) {
                let rows = CHUNK.min(units - u0);
                gemm(fp(&h).add(u0 * MERGED * 2), tw.m0, rows, MERGED, MERGED, fp(&m1_c));
                ck(m26v_gelu_bf16(fp(&m1_c) as _, fp(&g_c) as _, (rows * MERGED) as i64, STREAM), "gelu")?;
                gemm(fp(&g_c), tw.m2, rows, OUT_DIM, MERGED, fp(&out).add(u0 * OUT_DIM * 4));
            }
        }
        let mut emb = vec![0f32; units * OUT_DIM];
        out.download_prefix(bytes_of_mut(&mut emb)).map_err(|e| format!("vision download: {}", cuda::error_string(e)))?;
        for e in emb.iter_mut() {
            *e = f32::from_bits((f32_to_bf16_rne(*e) as u32) << 16);
        }
        Ok((emb, dumps))
    }
}

impl Drop for VisionTower {
    fn drop(&mut self) {
        if self.pinned {
            unsafe { cuda::cudaHostUnregister(self.blob.as_mut_ptr() as _) };
        }
    }
}

/// Per-patch (h, w) positions in row order (merge units row-major, the 4 patches of a unit in
/// (dy, dx) order), the same in column-major unit order, the column-major unit permutation and its
/// inverse — `rot_pos_emb` and `get_window_index_1d(col=True)` of the reference.
pub fn positions(gh: usize, gw: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let (uh, uw) = (gh / 2, gw / 2);
    let n = gh * gw;
    let mut hw_row = vec![0i32; 2 * n];
    for u in 0..uh * uw {
        let (uy, ux) = (u / uw, u % uw);
        for r in 0..4 {
            let t = u * 4 + r;
            hw_row[2 * t] = (uy * 2 + r / 2) as i32;
            hw_row[2 * t + 1] = (ux * 2 + r % 2) as i32;
        }
    }
    let mut col_idx = vec![0i32; uh * uw];
    for xx in 0..uw {
        for yy in 0..uh {
            col_idx[xx * uh + yy] = (yy * uw + xx) as i32;
        }
    }
    let mut inv = vec![0i32; uh * uw];
    for (kk, &u) in col_idx.iter().enumerate() {
        inv[u as usize] = kk as i32;
    }
    let mut hw_col = vec![0i32; 2 * n];
    for (kk, &u) in col_idx.iter().enumerate() {
        let (src, dst) = (2 * 4 * u as usize, 2 * 4 * kk);
        hw_col[dst..dst + 8].copy_from_slice(&hw_row[src..src + 8]);
    }
    (hw_row, hw_col, col_idx, inv)
}

fn bytes_of<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytes_of_mut<T>(v: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

#[cfg(test)]
mod tests {
    use super::positions;

    #[test]
    fn positions_match_the_reference_orders() {
        // grid 4x6 patches: 2x3 merge units.
        let (row, col, idx, inv) = positions(4, 6);
        // Unit 0 = patches (0,0) (0,1) (1,0) (1,1); unit 1 starts at column 2.
        assert_eq!(&row[..8], &[0, 0, 0, 1, 1, 0, 1, 1]);
        assert_eq!(&row[8..10], &[0, 2]);
        // Column-major units: (0,0) (1,0) (0,1) (1,1) (0,2) (1,2) -> row-major ids 0 3 1 4 2 5.
        assert_eq!(idx, vec![0, 3, 1, 4, 2, 5]);
        for (k, &u) in idx.iter().enumerate() {
            assert_eq!(inv[u as usize], k as i32);
        }
        // The second column-ordered unit is row-major unit 3: patch rows 2..3, columns 0..1.
        assert_eq!(&col[8..16], &[2, 0, 2, 1, 3, 0, 3, 1]);
    }
}
