//! DFlash block drafter on the 5090 (perf reset S1).
//!
//! The checkpoint ships `dflash/`: a 5-layer Qwen3-style block-diffusion drafter
//! (hidden 4096, 64 Q / 8 KV heads of 128, MLP 16384, partial rotary 64 of 128 at
//! θ 1e4, SWA 1024, non-causal, per-Q-head sink, V × 0.612) that proposes 7
//! tokens per step from the target's hidden features. Semantics follow vLLM's
//! implementation as the D7 reference ran it (`runs/20260923-d7-tp4-baseline/
//! stage/qwen3_dflash.py`, `v1/spec_decode/dflash.py` and `eagle3_utils.py` from
//! its image):
//!
//! - **Aux features:** the target's residual stream after layers 0, 11, 23, 35
//!   and 47 (`target_layer_ids` + 1 = vLLM's aux layers; the value is
//!   `hidden + residual`, before the final norm), concatenated per token.
//! - **Context:** `hidden_norm(fc(aux))`, projected per draft layer by that
//!   layer's k/v (k_norm, RoPE at the token's position, V × 0.612) and cached.
//!   Only accepted tokens become context. A block query at `p` sees keys from
//!   `p - 1023`, so the cache is a ring of [`RING`] positions and a prompt
//!   contributes only its last [`CTX_TAIL`] tokens.
//! - **Block:** `[bonus, mask × 7]` at the next 8 positions (mask = the shipped
//!   `mask_embedding.pt`, not embedding row 151675), through the 5 layers
//!   non-causally over context + block; the final norm and the target's
//!   `lm_head` at the 7 mask rows, argmax = the drafts.

use std::path::Path;

use mimo26_repack::safetensors::SafetensorsHeader;

use super::*;

/// vLLM aux layer `i + 1` = the output of target layer `i`.
pub(crate) const TARGET_LAYERS: [usize; 5] = [0, 11, 23, 35, 47];
/// Block size: the bonus token plus 7 drafts (D7: `num_speculative_tokens` 7).
pub const BLOCK: usize = 8;
pub const DRAFTS: usize = BLOCK - 1;
/// Prompt positions that can ever be visible to a block query (window 1024).
pub(crate) const CTX_TAIL: usize = 1024;
/// Draft KV ring length: >= window 1024 + block 8 + the prompt tail's extra row.
const RING: usize = 1088;
const WINDOW: usize = 1024;
const LAYERS: usize = 5;
const HID: usize = 4096;
const NQ: usize = 64;
const NKV: usize = 8;
const HD: usize = 128;
const INTER: usize = 16384;
const V_SCALE: f32 = 0.612;
const THETA: f64 = 10000.0;
const ROT: f64 = 0.5;
const EPS: f64 = 1e-6;
const SPLIT_KEYS: usize = 128;
const MASK_TOKEN: usize = 151675;

unsafe extern "C" {
    fn m26c_dflash_store_kv(
        k: *const f32,
        v: *const f32,
        ld: i32,
        row_seq: *const i32,
        pos: *const i64,
        n: i32,
        hd: i32,
        ring: i32,
        v_scale: f32,
        pk: *const u64,
        pv: *const u64,
        s: CudaStream,
    ) -> CudaError;
    fn m26c_dflash_attn(
        q: *const f32,
        qpos: *const i64,
        seq_klo: *const i64,
        seq_khi: *const i64,
        nseq: i32,
        pk: *const u64,
        pv: *const u64,
        ring: i32,
        n_kv: i32,
        window: i32,
        scale: f32,
        split_keys: i32,
        splits: i32,
        sink: *const f32,
        part: *mut f32,
        out: *mut f32,
        s: CudaStream,
    ) -> CudaError;
    pub(crate) fn m26c_argmax_rows(x: *const f32, ld: i64, rows: i32, vocab: i32, out: *mut i32, s: CudaStream) -> CudaError;
    fn m26c_lm8_quantize(w: *const u16, v: i64, k: i32, q: *mut i8, scale: *mut f32, s: CudaStream) -> CudaError;
    fn m26c_lm8_gemv(x: *const f32, rows: i32, w: *const i8, scale: *const f32, out: *mut f32, ld: i64, v: i32, k: i32,
        s: CudaStream) -> CudaError;
    fn m26c_argmax_prob_rows(x: *const f32, ld: i64, rows: i32, vocab: i32, out: *mut i32, prob: *mut f32, s: CudaStream)
        -> CudaError;
}

/// A request's draft KV: per layer, K and V rings of [`RING`] rows of 8 × 128 BF16.
pub(crate) struct DraftKv {
    k: Vec<DeviceBuffer>,
    v: Vec<DeviceBuffer>,
}

impl DraftKv {
    /// Bytes of all rings (host snapshot layout `[layer][K ring][V ring]`).
    pub(crate) fn bytes(&self) -> usize {
        LAYERS * 2 * RING * NKV * HD * 2
    }

    pub(crate) fn export(&self, dst: &mut [u8]) -> Result<(), String> {
        let ring = RING * NKV * HD * 2;
        if dst.len() < self.bytes() {
            return Err("draft export: short buffer".into());
        }
        for l in 0..LAYERS {
            for (j, b) in [&self.k[l], &self.v[l]].into_iter().enumerate() {
                let o = (l * 2 + j) * ring;
                ck(unsafe { cuda::cudaMemcpy(dst[o..].as_mut_ptr() as _, b.as_ptr() as _, ring, cuda::D2H) },
                    "draft export")?;
            }
        }
        Ok(())
    }

    pub(crate) fn import(&mut self, src: &[u8]) -> Result<(), String> {
        let ring = RING * NKV * HD * 2;
        if src.len() < self.bytes() {
            return Err("draft import: short buffer".into());
        }
        for l in 0..LAYERS {
            for (j, b) in [&self.k[l], &self.v[l]].into_iter().enumerate() {
                let o = (l * 2 + j) * ring;
                ck(unsafe { cuda::cudaMemcpy(b.as_ptr() as _, src[o..].as_ptr() as _, ring, cuda::H2D) }, "draft import")?;
            }
        }
        Ok(())
    }

    /// Device-to-device save of all rings to `dst` (same layout as `export`).
    pub(crate) fn save_dev(&self, dst: *mut u8) -> Result<(), String> {
        let ring = RING * NKV * HD * 2;
        for l in 0..LAYERS {
            for (j, b) in [&self.k[l], &self.v[l]].into_iter().enumerate() {
                let o = (l * 2 + j) * ring;
                ck(unsafe { cuda::cudaMemcpy(dst.add(o) as _, b.as_ptr() as _, ring, cuda::D2D) }, "draft save")?;
            }
        }
        Ok(())
    }

    /// Device-to-device load of all rings from `src` (as `save_dev` wrote them).
    pub(crate) fn load_dev(&self, src: *const u8) -> Result<(), String> {
        let ring = RING * NKV * HD * 2;
        for l in 0..LAYERS {
            for (j, b) in [&self.k[l], &self.v[l]].into_iter().enumerate() {
                let o = (l * 2 + j) * ring;
                ck(unsafe { cuda::cudaMemcpy(b.as_ptr() as _, src.add(o) as _, ring, cuda::D2D) }, "draft load")?;
            }
        }
        Ok(())
    }

    pub(crate) fn new() -> Result<Self, String> {
        let bytes = RING * NKV * HD * 2;
        let alloc = || DeviceBuffer::alloc(bytes).map_err(cuda::error_string);
        Ok(Self { k: (0..LAYERS).map(|_| alloc()).collect::<Result<_, _>>()?, v: (0..LAYERS).map(|_| alloc()).collect::<Result<_, _>>()? })
    }
}

/// The drafter's resident small tensors and scratch (its matrices live in the
/// shared [`DenseDevice`] as `dflash.*` BF16 weights).
pub(crate) struct Dflash {
    ln_in: Vec<DeviceBuffer>,
    ln_post: Vec<DeviceBuffer>,
    q_norm: Vec<DeviceBuffer>,
    k_norm: Vec<DeviceBuffer>,
    sink: Vec<DeviceBuffer>,
    hidden_norm: DeviceBuffer,
    norm: DeviceBuffer,
    mask_emb: Vec<f32>,
    h: Grow,
    x: Grow,
    q: Grow,
    qn: Grow,
    k: Grow,
    kn: Grow,
    v: Grow,
    a: Grow,
    o: Grow,
    g: Grow,
    u: Grow,
    pos: Grow,
    row_seq: Grow,
    ptrs: Grow,
    klo: Grow,
    khi: Grow,
    part: Grow,
    am: Grow,
    /// The drafter's softmax probability of each draft (adaptive verify length).
    pr: Grow,
    /// 8-bit lm_head for the drafter (perf reset KN4, default on; `MIMO26_DRAFT_LM8=0` off): INT8
    /// rows and per-row scales; used for draft passes of up to 8 rows.
    lm8: Option<(DeviceBuffer, DeviceBuffer)>,
}

/// `torch.save`'d `{"mask_token_id": 151675, "embedding": bf16[4096]}`: a stored
/// (uncompressed) zip whose entry sizes live in the central directory (the local
/// headers use data descriptors); the tensor is the `*/data/0` entry.
fn read_mask_embedding(path: &Path) -> Result<Vec<f32>, String> {
    let z = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let bad = |what: &str| format!("{}: {what}", path.display());
    let u16_at = |o: usize| -> Result<usize, String> {
        z.get(o..o + 2).map(|b| u16::from_le_bytes([b[0], b[1]]) as usize).ok_or_else(|| bad("truncated"))
    };
    let u32_at = |o: usize| -> Result<usize, String> {
        z.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize).ok_or_else(|| bad("truncated"))
    };
    let eocd = (0..z.len().saturating_sub(21)).rev().find(|&i| z[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .ok_or_else(|| bad("no end of central directory"))?;
    let (count, mut c) = (u16_at(eocd + 10)?, u32_at(eocd + 16)?);
    let (mut tensor, mut token_ok) = (None, false);
    for _ in 0..count {
        if z.get(c..c + 4) != Some(&[0x50, 0x4b, 0x01, 0x02][..]) {
            return Err(bad("bad central directory entry"));
        }
        let (method, size) = (u16_at(c + 10)?, u32_at(c + 20)?);
        let (name_len, extra_len, comment_len, local) = (u16_at(c + 28)?, u16_at(c + 30)?, u16_at(c + 32)?, u32_at(c + 42)?);
        let name = String::from_utf8_lossy(z.get(c + 46..c + 46 + name_len).ok_or_else(|| bad("truncated"))?).to_string();
        c += 46 + name_len + extra_len + comment_len;
        let data = local + 30 + u16_at(local + 26)? + u16_at(local + 28)?;
        let bytes = z.get(data..data + size).ok_or_else(|| bad("truncated entry"))?;
        if method != 0 {
            return Err(bad(&format!("entry {name} is compressed")));
        }
        if name.ends_with("/data/0") {
            tensor = Some(bytes.to_vec());
        } else if name.ends_with("/data.pkl") {
            // BININT (`J`) 151675: the file's mask_token_id matches the config's.
            let want = [&[b'J'][..], &(MASK_TOKEN as u32).to_le_bytes()[..]].concat();
            token_ok = bytes.windows(5).any(|w| w == want.as_slice());
        }
    }
    let raw = tensor.ok_or_else(|| bad("no tensor data"))?;
    if raw.len() != HID * 2 || !token_ok {
        return Err(bad(&format!("{} bytes (want {}), mask_token_id match {token_ok}", raw.len(), HID * 2)));
    }
    Ok(crate::load::bf16_to_f32(&raw))
}

impl DeviceForward {
    /// Load the drafter from `dir` (the checkpoint's `dflash/`). Its matrices go
    /// to the device as the checkpoint's BF16; the norms, sinks and the mask
    /// embedding as f32.
    pub fn load_dflash(&mut self, dir: &Path) -> Result<(), String> {
        let cfg_text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("dflash config: {e}"))?;
        let flat: String = cfg_text.chars().filter(|c| !c.is_whitespace()).collect();
        for want in ["\"target_layer_ids\":[0,11,23,35,47]", "\"mask_token_id\":151675", "\"block_size\":8",
            "\"attention_value_scale\":0.612", "\"attention_sink_bias\":true", "\"sliding_window\":1024",
            "\"partial_rotary_factor\":0.5", "\"rope_theta\":10000.0", "\"num_hidden_layers\":5", "\"is_causal\":false"]
        {
            if !flat.contains(want) {
                return Err(format!("dflash config: expected {want}"));
            }
        }
        let path = dir.join("dflash_draft_model.safetensors");
        let hdr = SafetensorsHeader::read(&path).map_err(|e| format!("dflash header: {e:?}"))?;
        let raw = |name: &str, want: &[usize]| -> Result<Vec<u8>, String> {
            let e = hdr.tensors.get(name).ok_or_else(|| format!("dflash: missing {name}"))?;
            if e.dtype != "BF16" || e.shape != want {
                return Err(format!("dflash {name}: {} {:?}, want BF16 {want:?}", e.dtype, e.shape));
            }
            hdr.read_tensor(&path, name).map_err(|e| format!("dflash {name}: {e:?}"))
        };
        let small = |name: &str, n: usize| -> Result<DeviceBuffer, String> { upload_f32(&crate::load::bf16_to_f32(&raw(name, &[n])?)) };
        self.dense.upload_bf16_raw("dflash.fc", &raw("fc.weight", &[HID, HID * TARGET_LAYERS.len()])?, HID,
            HID * TARGET_LAYERS.len())?;
        let (mut ln_in, mut ln_post, mut q_norm, mut k_norm, mut sink) = (vec![], vec![], vec![], vec![], vec![]);
        for l in 0..LAYERS {
            for (part, out, inp) in [
                ("self_attn.q_proj", NQ * HD, HID),
                ("self_attn.k_proj", NKV * HD, HID),
                ("self_attn.v_proj", NKV * HD, HID),
                ("self_attn.o_proj", HID, NQ * HD),
                ("mlp.gate_proj", INTER, HID),
                ("mlp.up_proj", INTER, HID),
                ("mlp.down_proj", HID, INTER),
            ] {
                self.dense.upload_bf16_raw(&format!("dflash.{l}.{part}"), &raw(&format!("layers.{l}.{part}.weight"), &[out, inp])?,
                    out, inp)?;
            }
            ln_in.push(small(&format!("layers.{l}.input_layernorm.weight"), HID)?);
            ln_post.push(small(&format!("layers.{l}.post_attention_layernorm.weight"), HID)?);
            q_norm.push(small(&format!("layers.{l}.self_attn.q_norm.weight"), HD)?);
            k_norm.push(small(&format!("layers.{l}.self_attn.k_norm.weight"), HD)?);
            sink.push(small(&format!("layers.{l}.self_attn.attention_sink_bias"), NQ)?);
        }
        let d = Dflash {
            ln_in,
            ln_post,
            q_norm,
            k_norm,
            sink,
            hidden_norm: small("hidden_norm.weight", HID)?,
            norm: small("norm.weight", HID)?,
            mask_emb: read_mask_embedding(&dir.join("mask_embedding.pt"))?,
            h: Grow::new()?,
            x: Grow::new()?,
            q: Grow::new()?,
            qn: Grow::new()?,
            k: Grow::new()?,
            kn: Grow::new()?,
            v: Grow::new()?,
            a: Grow::new()?,
            o: Grow::new()?,
            g: Grow::new()?,
            u: Grow::new()?,
            pos: Grow::new()?,
            row_seq: Grow::new()?,
            ptrs: Grow::new()?,
            klo: Grow::new()?,
            khi: Grow::new()?,
            part: Grow::new()?,
            am: Grow::new()?,
            pr: Grow::new()?,
            lm8: None,
        };
        self.dflash = Some(d);
        // On by default (perf reset KN4: C1 +0.7%, every category up, for 0.59 GiB);
        // MIMO26_DRAFT_LM8=0 keeps the drafter on the BF16 lm_head.
        let lm8 = std::env::var("MIMO26_DRAFT_LM8").map(|v| v != "0").unwrap_or(true);
        if lm8 {
            let vocab = self.cfg.vocab_size;
            let q = DeviceBuffer::alloc(vocab * HID).map_err(cuda::error_string)?;
            let sc = DeviceBuffer::alloc(vocab * 4).map_err(cuda::error_string)?;
            let w = self.dense.weight_ptr("lm_head.weight")?;
            ck(unsafe { m26c_lm8_quantize(w as _, vocab as i64, HID as i32, q.as_ptr() as _, sc.as_ptr() as _, STREAM) },
                "lm8 quantize")?;
            ck(unsafe { cuda::cudaDeviceSynchronize() }, "lm8 quantize sync")?;
            if let Some(d) = self.dflash.as_mut() {
                d.lm8 = Some((q, sc));
            }
        }
        eprintln!("[dforward] DFlash drafter loaded from {} (5 layers, block {BLOCK}, BF16{})", dir.display(),
            if lm8 { "; INT8 lm_head for draft passes of up to 8 rows" } else { "" });
        Ok(())
    }

    /// Whether a drafter is loaded (aux capture and speculative steps on).
    pub fn has_dflash(&self) -> bool {
        self.dflash.is_some()
    }

    /// Scratch for a context commit of `rows` rows: `fc`, the norms and the K/V
    /// projections only. Sizing a prefill's 1,024-row commit like a draft pass
    /// held ~0.9 GB of query, MLP and logits buffers it never used (perf reset K3
    /// follow-up).
    fn dflash_ensure_commit(&mut self, rows: usize, nseq: usize) -> Result<(), String> {
        let d = self.dflash.as_mut().ok_or("no drafter")?;
        let f = 4usize;
        d.h.ensure(rows * HID * f)?;
        d.x.ensure(rows * HID * f)?;
        d.k.ensure(rows * NKV * HD * f)?;
        d.kn.ensure(rows * NKV * HD * f)?;
        d.v.ensure(rows * NKV * HD * f)?;
        d.pos.ensure(rows * 8)?;
        d.row_seq.ensure(rows * 4)?;
        d.ptrs.ensure(LAYERS * 2 * nseq * 8)?;
        self.sc.xb.ensure(rows * HID * TARGET_LAYERS.len() * 2)?;
        Ok(())
    }

    /// Size the drafter's per-step buffers for the scheduler's steady state at
    /// startup: a prefill's commit of [`CTX_TAIL`] rows and a speculative step of
    /// `max_seqs` requests. Then the free memory the maximum context and
    /// admission are computed from stays free (perf reset K3 follow-up: these
    /// grew on the first prefill, and at the top of memory that failed).
    pub(crate) fn warm_dflash(&mut self, max_seqs: usize) -> Result<(), String> {
        if self.dflash.is_none() {
            return Ok(());
        }
        let rows = max_seqs * BLOCK;
        self.sc.aux.ensure(rows.max(CTX_TAIL) * TARGET_LAYERS.len() * HID * 4)?;
        self.dflash_ensure_commit(rows.max(CTX_TAIL), max_seqs)?;
        self.dflash_ensure(rows, max_seqs)?;
        self.sc.idx.ensure(rows * 4)
    }

    fn dflash_ensure(&mut self, rows: usize, nseq: usize) -> Result<(), String> {
        let d = self.dflash.as_mut().ok_or("no drafter")?;
        let f = 4usize;
        d.h.ensure(rows * HID * f)?;
        d.x.ensure(rows * HID * f)?;
        d.q.ensure(rows * NQ * HD * f)?;
        d.qn.ensure(rows * NQ * HD * f)?;
        d.k.ensure(rows * NKV * HD * f)?;
        d.kn.ensure(rows * NKV * HD * f)?;
        d.v.ensure(rows * NKV * HD * f)?;
        d.a.ensure(rows * NQ * HD * f)?;
        d.o.ensure(rows * HID * f)?;
        d.g.ensure(rows * INTER * f)?;
        d.u.ensure(rows * INTER * f)?;
        d.pos.ensure(rows * 8)?;
        d.row_seq.ensure(rows * 4)?;
        d.ptrs.ensure(LAYERS * 2 * nseq * 8)?;
        d.klo.ensure(nseq * 8)?;
        d.khi.ensure(nseq * 8)?;
        let splits = (WINDOW + BLOCK).div_ceil(SPLIT_KEYS);
        d.part.ensure(nseq * splits * BLOCK * NQ * (2 + HD) * f)?;
        d.am.ensure(rows * 4)?;
        d.pr.ensure(rows * 4)?;
        self.sc.xb.ensure(rows * (HID * TARGET_LAYERS.len()).max(INTER) * 2)?;
        self.sc.lm.ensure(rows * self.cfg.vocab_size * f)?;
        Ok(())
    }

    /// Upload the per-layer ring pointer tables `[layer][k|v][seq]`.
    fn dflash_ptrs(&mut self, kvs: &[&mut DeviceKv]) -> Result<(), String> {
        let mut t = vec![0u64; LAYERS * 2 * kvs.len()];
        for (i, kv) in kvs.iter().enumerate() {
            let dk = kv.draft.as_ref().ok_or("dflash: a request without a draft KV")?;
            for l in 0..LAYERS {
                t[(l * 2) * kvs.len() + i] = dk.k[l].as_ptr() as u64;
                t[(l * 2 + 1) * kvs.len() + i] = dk.v[l].as_ptr() as u64;
            }
        }
        let d = self.dflash.as_ref().ok_or("no drafter")?;
        d.ptrs.buf.upload_prefix(bytes_of(&t)).map_err(cuda::error_string)
    }

    /// Add context to the draft KV: `aux` holds `rows.len()` aux feature rows
    /// (`[n, 5 * 4096]` f32, device); row `r` belongs to `kvs[rows[r].0]` at
    /// position `rows[r].1`, or is skipped when `rows[r].0` is negative (a
    /// rejected block row).
    pub(crate) fn dflash_commit(&mut self, kvs: &[&mut DeviceKv], aux: *const f32, rows: &[(i32, i64)]) -> Result<(), String> {
        let n = rows.len();
        if n == 0 {
            return Ok(());
        }
        self.dflash_ensure_commit(n, kvs.len())?;
        self.dflash_ptrs(kvs)?;
        let d = self.dflash.as_ref().ok_or("no drafter")?;
        let seqv: Vec<i32> = rows.iter().map(|r| r.0).collect();
        let posv: Vec<i64> = rows.iter().map(|r| r.1).collect();
        d.row_seq.buf.upload_prefix(bytes_of(&seqv)).map_err(cuda::error_string)?;
        d.pos.buf.upload_prefix(bytes_of(&posv)).map_err(cuda::error_string)?;
        let nseq = kvs.len();
        unsafe {
            self.lin(aux, "dflash.fc", n, d.x.p())?;
            ck(m26c_rmsnorm(d.x.p(), d.hidden_norm.as_ptr() as _, d.h.p(), n as i32, HID as i32, EPS, STREAM), "dflash hidden_norm")?;
            for l in 0..LAYERS {
                self.lin(d.h.p(), &format!("dflash.{l}.self_attn.k_proj"), n, d.k.p())?;
                self.lin(d.h.p(), &format!("dflash.{l}.self_attn.v_proj"), n, d.v.p())?;
                ck(m26c_rmsnorm(d.k.p(), d.k_norm[l].as_ptr() as _, d.kn.p(), (n * NKV) as i32, HD as i32, EPS, STREAM),
                    "dflash k_norm")?;
                ck(af::m26_rope_apply(THETA, ROT, d.kn.p(), d.k.p(), d.pos.p(), n as i32, NKV as i32, HD as i32, 0, STREAM),
                    "dflash rope k")?;
                let pk = d.ptrs.p::<u64>().add(l * 2 * nseq);
                ck(m26c_dflash_store_kv(d.k.p(), d.v.p(), (NKV * HD) as i32, d.row_seq.p(), d.pos.p(), n as i32,
                    (NKV * HD) as i32, RING as i32, V_SCALE, pk, pk.add(nseq), STREAM), "dflash store ctx")?;
            }
        }
        Ok(())
    }

    /// Draft [`DRAFTS`] tokens for each request: `seqs[i]` is `kvs[i]`'s bonus
    /// token (sampled, not yet in the target cache) and its position. Each draft
    /// comes with the drafter's softmax probability of it.
    pub(crate) fn dflash_draft(
        &mut self,
        kvs: &[&mut DeviceKv],
        seqs: &[(usize, usize)],
    ) -> Result<Vec<([usize; DRAFTS], [f32; DRAFTS])>, String> {
        let nseq = seqs.len();
        if nseq == 0 || nseq != kvs.len() {
            return Err(format!("dflash_draft: {nseq} requests for {} caches", kvs.len()));
        }
        let rows = nseq * BLOCK;
        self.dflash_ensure(rows, nseq)?;
        self.dflash_ptrs(kvs)?;
        let vocab = self.cfg.vocab_size;
        let d = self.dflash.as_ref().ok_or("no drafter")?;
        // Block embeddings [bonus, mask x 7], positions, owners, key ranges.
        let mut emb = Vec::with_capacity(rows * HID);
        let (mut posv, mut seqv, mut klo, mut khi) = (Vec::with_capacity(rows), Vec::with_capacity(rows), vec![], vec![]);
        for (i, &(bonus, start)) in seqs.iter().enumerate() {
            if bonus >= vocab {
                return Err(format!("dflash_draft: token {bonus} out of vocab"));
            }
            emb.extend_from_slice(&self.embed[bonus * HID..(bonus + 1) * HID]);
            for _ in 1..BLOCK {
                emb.extend_from_slice(&d.mask_emb);
            }
            for j in 0..BLOCK {
                posv.push((start + j) as i64);
                seqv.push(i as i32);
            }
            klo.push(start.saturating_sub(WINDOW - 1) as i64);
            khi.push((start + BLOCK) as i64);
        }
        d.h.buf.upload_prefix(bytes_of(&emb)).map_err(cuda::error_string)?;
        d.pos.buf.upload_prefix(bytes_of(&posv)).map_err(cuda::error_string)?;
        d.row_seq.buf.upload_prefix(bytes_of(&seqv)).map_err(cuda::error_string)?;
        d.klo.buf.upload_prefix(bytes_of(&klo)).map_err(cuda::error_string)?;
        d.khi.buf.upload_prefix(bytes_of(&khi)).map_err(cuda::error_string)?;
        let splits = seqs.iter().map(|&(_, s)| (s + BLOCK - s.saturating_sub(WINDOW - 1)).div_ceil(SPLIT_KEYS)).max().unwrap_or(1);
        let r = rows as i32;
        unsafe {
            for l in 0..LAYERS {
                ck(m26c_rmsnorm(d.h.p(), d.ln_in[l].as_ptr() as _, d.x.p(), r, HID as i32, EPS, STREAM), "dflash ln_in")?;
                self.lin(d.x.p(), &format!("dflash.{l}.self_attn.q_proj"), rows, d.q.p())?;
                self.lin(d.x.p(), &format!("dflash.{l}.self_attn.k_proj"), rows, d.k.p())?;
                self.lin(d.x.p(), &format!("dflash.{l}.self_attn.v_proj"), rows, d.v.p())?;
                ck(m26c_rmsnorm(d.q.p(), d.q_norm[l].as_ptr() as _, d.qn.p(), r * NQ as i32, HD as i32, EPS, STREAM),
                    "dflash q_norm")?;
                ck(m26c_rmsnorm(d.k.p(), d.k_norm[l].as_ptr() as _, d.kn.p(), r * NKV as i32, HD as i32, EPS, STREAM),
                    "dflash k_norm")?;
                ck(af::m26_rope_apply(THETA, ROT, d.qn.p(), d.q.p(), d.pos.p(), r, NQ as i32, HD as i32, 0, STREAM),
                    "dflash rope q")?;
                ck(af::m26_rope_apply(THETA, ROT, d.kn.p(), d.k.p(), d.pos.p(), r, NKV as i32, HD as i32, 0, STREAM),
                    "dflash rope k")?;
                let pk = d.ptrs.p::<u64>().add(l * 2 * nseq);
                ck(m26c_dflash_store_kv(d.k.p(), d.v.p(), (NKV * HD) as i32, d.row_seq.p(), d.pos.p(), r,
                    (NKV * HD) as i32, RING as i32, V_SCALE, pk, pk.add(nseq), STREAM), "dflash store block")?;
                ck(m26c_dflash_attn(d.q.p(), d.pos.p(), d.klo.p(), d.khi.p(), nseq as i32, pk, pk.add(nseq), RING as i32,
                    NKV as i32, WINDOW as i32, 1.0 / (HD as f32).sqrt(), SPLIT_KEYS as i32, splits as i32,
                    d.sink[l].as_ptr() as _, d.part.p(), d.a.p(), STREAM), "dflash attn")?;
                self.lin(d.a.p(), &format!("dflash.{l}.self_attn.o_proj"), rows, d.o.p())?;
                ck(m26c_add_inplace(d.h.p(), d.o.p(), (rows * HID) as i64, STREAM), "dflash resid attn")?;
                ck(m26c_rmsnorm(d.h.p(), d.ln_post[l].as_ptr() as _, d.x.p(), r, HID as i32, EPS, STREAM), "dflash ln_post")?;
                self.lin(d.x.p(), &format!("dflash.{l}.mlp.gate_proj"), rows, d.g.p())?;
                self.lin(d.x.p(), &format!("dflash.{l}.mlp.up_proj"), rows, d.u.p())?;
                ck(m26c_silu_mul(d.g.p(), d.u.p(), (rows * INTER) as i64, STREAM), "dflash silu")?;
                self.lin(d.g.p(), &format!("dflash.{l}.mlp.down_proj"), rows, d.o.p())?;
                ck(m26c_add_inplace(d.h.p(), d.o.p(), (rows * HID) as i64, STREAM), "dflash resid mlp")?;
            }
            ck(m26c_rmsnorm(d.h.p(), d.norm.as_ptr() as _, d.x.p(), r, HID as i32, EPS, STREAM), "dflash norm")?;
            match d.lm8.as_ref().filter(|_| rows <= 8) {
                Some((q, sc)) => ck(m26c_lm8_gemv(d.x.p(), r, q.as_ptr() as _, sc.as_ptr() as _, self.sc.lm.p(),
                    vocab as i64, vocab as i32, HID as i32, STREAM), "dflash lm8")?,
                None => self.lin(d.x.p(), "lm_head.weight", rows, self.sc.lm.p())?,
            }
            ck(m26c_argmax_prob_rows(self.sc.lm.p(), vocab as i64, r, vocab as i32, d.am.p(), d.pr.p(), STREAM),
                "dflash argmax")?;
        }
        let mut am = vec![0i32; rows];
        let mut pr = vec![0f32; rows];
        d.am.buf.download_prefix(bytes_of_mut(&mut am)).map_err(cuda::error_string)?;
        d.pr.buf.download_prefix(bytes_of_mut(&mut pr)).map_err(cuda::error_string)?;
        Ok((0..nseq)
            .map(|i| {
                let (mut out, mut p) = ([0usize; DRAFTS], [0f32; DRAFTS]);
                for j in 0..DRAFTS {
                    out[j] = am[i * BLOCK + 1 + j] as usize;
                    p[j] = pr[i * BLOCK + 1 + j];
                }
                (out, p)
            })
            .collect())
    }
}
