//! Serving forward (I5-R8a): dense GEMMs on the GPU (`DenseDevice` cuBLAS SGEMM,
//! TF32 off) and attention on the GPU (`mimo26-attn::serve`), with the MoE on
//! CPU for now (the wire seam swaps it next).
//!
//! Mirrors `forward::Model::forward` — same layer order, same frozen CPU
//! reduction orders for the norms/RoPE/router-sigmoid. The CPU `forward::Model`
//! stays the golden bisect reference. Only compiled under the `cuda` feature.

use std::collections::HashMap;

use mimo26_attn::device::DeviceBuffer;
use mimo26_attn::ffi::M26Geom;
use mimo26_attn::fp8kv::{encode_kv, EncodedKv};
use mimo26_attn::geom::ScaleMode;
use mimo26_attn::rope::{apply_rotary_precomputed, rope_cos_sin, rot_dim};
use mimo26_attn::serve::{
    decode_attention_device, kv_store_fp8_device, pos_to_device, prefill_attention_dev,
    rope_apply_device, split_qkv_device,
};
use mimo26_attn::NaiveBits;

/// Split-KV decode split count (flash decoding chunks; correctness-first value).
const DECODE_SPLITS: i32 = 8;

/// Prefill chunk size (tokens per forward pass), read once from
/// `MIMO26_PREFILL_CHUNK` (default 2048) and logged. A single-shot 32K prefill
/// OOMs the 5090 (the device Q/K/V buffers are O(prompt)); chunking bounds them
/// to O(chunk) while the attention reads the accumulated KV cache. The env hook
/// lets a test force a tiny chunk (e.g. 128) so a T=300 prefill crosses several
/// chunk boundaries and exercises the cross-chunk causal mask.
fn prefill_chunk() -> usize {
    use std::sync::OnceLock;
    static CHUNK: OnceLock<usize> = OnceLock::new();
    *CHUNK.get_or_init(|| {
        let c = std::env::var("MIMO26_PREFILL_CHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2048)
            .max(1);
        eprintln!("[coordinator] PREFILL_CHUNK={c}");
        c
    })
}

use crate::config::{Config, LayerKind};
use crate::forward::{w_name, Model};
use crate::gpu_dense::DenseDevice;
use crate::wire::WireClient;
use crate::{dense_ffn, embed, moe_forward, rmsnorm, router_from_logits};

/// Per-stage wall timers, enabled by `MIMO26_PROFILE=1` (host-path performance
/// pass, I5-R13). Prints `PROFILE <stage> <ms>` to stderr.
fn profile(stage: &str, ms: f64) {
    if std::env::var_os("MIMO26_PROFILE").is_some() {
        eprintln!("PROFILE {stage} {ms:.3}");
    }
}

fn now_ms(t: &std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Precomputed RoPE cos/sin per position for one forward, shared by every layer
/// (exact: computed once per θ, not per layer × head). `rd` is the rotary dim
/// (`rot_dim(head_dim, factor)`).
struct RopeTables {
    rd: usize,
    cos_ga: Vec<f32>,
    sin_ga: Vec<f32>,
    cos_swa: Vec<f32>,
    sin_swa: Vec<f32>,
}

impl RopeTables {
    fn new(cfg: &Config, positions: &[i64]) -> Self {
        let rd = rot_dim(cfg.head_dim, f64::from(cfg.partial_rotary_factor));
        let (cos_ga, sin_ga) = rope_cos_sin(f64::from(cfg.rope_theta), positions, rd);
        let (cos_swa, sin_swa) = rope_cos_sin(f64::from(cfg.swa_rope_theta), positions, rd);
        Self { rd, cos_ga, sin_ga, cos_swa, sin_swa }
    }

    fn get(&self, kind: LayerKind) -> (&[f32], &[f32]) {
        match kind {
            LayerKind::Ga => (&self.cos_ga, &self.sin_ga),
            LayerKind::Swa => (&self.cos_swa, &self.sin_swa),
        }
    }
}

/// One layer's FP8 KV cache (unit scale, T18 V pre-scaled). GA grows; SWA keeps
/// the last `window` tokens (a ring).
pub struct Fp8KvCache {
    kind: LayerKind,
    n_kv: usize,
    d_qk: usize,
    d_v: usize,
    window: usize,
    value_scale: f32,
    k_codes: Vec<u8>, // [S, n_kv*d_qk]
    v_codes: Vec<u8>, // [S, n_kv*d_v]
    k_pos: Vec<i64>,  // [S]
}

impl Fp8KvCache {
    pub fn for_layer(cfg: &Config, layer: usize) -> Self {
        let kind = cfg.layer_kind(layer);
        let (n_kv, d_qk, d_v) = match kind {
            LayerKind::Ga => (cfg.num_key_value_heads, cfg.head_dim, cfg.v_head_dim),
            LayerKind::Swa => (cfg.swa_num_key_value_heads, cfg.head_dim, cfg.v_head_dim),
        };
        Self {
            kind,
            n_kv,
            d_qk,
            d_v,
            window: if kind == LayerKind::Ga { 0 } else { cfg.sliding_window },
            value_scale: cfg.attention_value_scale,
            k_codes: Vec::new(),
            v_codes: Vec::new(),
            k_pos: Vec::new(),
        }
    }

    /// Encode + append one step's K/V (V is pre-scaled by `value_scale`, T18),
    /// then evict the SWA ring to the window.
    pub fn append(&mut self, k_rot: &[f32], v_raw: &[f32], pos: &[i64]) {
        let t = pos.len();
        let v_scaled: Vec<f32> = v_raw.iter().map(|&x| x * self.value_scale).collect();
        let enc: EncodedKv = encode_kv(
            k_rot,
            &v_scaled,
            t,
            self.n_kv,
            self.d_qk,
            self.d_v,
            ScaleMode::Unit,
            NaiveBits::NONE,
        )
        .expect("encode_kv");
        self.append_codes(&enc.k_codes, &enc.v_codes, pos);
    }

    /// Append pre-encoded FP8 codes (the device-KV path: the encode happened on
    /// the GPU via `m26_kv_store_fp8`), then evict the SWA ring to the window.
    pub fn append_codes(&mut self, k_codes: &[u8], v_codes: &[u8], pos: &[i64]) {
        let t = pos.len();
        self.k_codes.extend_from_slice(k_codes);
        self.v_codes.extend_from_slice(v_codes);
        self.k_pos.extend_from_slice(pos);
        if self.kind == LayerKind::Swa {
            // T8 eviction (mirror SwaRing::evict_count): keep_from =
            // min(batch_pos) - window + 1; drop = min(n, max(drop_keep, drop_trim)).
            // NOT "keep the last window" — that eats rows the early queries of a
            // multi-row prefill still need (the EVICT_KEEP_LAST trap).
            let n = self.k_pos.len();
            let batch_rows = t;
            let batch_min = self.k_pos[n - batch_rows..].iter().copied().min().unwrap_or(0);
            let keep_from = batch_min - self.window as i64 + 1;
            let drop_keep = self.k_pos.iter().take_while(|&&p| p < keep_from).count();
            let drop_trim = n.saturating_sub(self.window + batch_rows);
            let drop = n.min(drop_keep.max(drop_trim));
            if drop > 0 {
                self.k_codes.drain(..drop * self.n_kv * self.d_qk);
                self.v_codes.drain(..drop * self.n_kv * self.d_v);
                self.k_pos.drain(..drop);
            }
        }
    }

    pub fn get(&self) -> (&[u8], &[u8], &[i64]) {
        (&self.k_codes, &self.v_codes, &self.k_pos)
    }

    /// Drop every KV row (D1: a fresh cache per request — the KV cache must not
    /// leak a prior request's conversation into the next one).
    pub fn clear(&mut self) {
        self.k_codes.clear();
        self.v_codes.clear();
        self.k_pos.clear();
    }
}

pub struct ServingModel {
    model: Model,
    dense: DenseDevice,
}

impl ServingModel {
    /// Build the serving model: keep every weight on the CPU `Model` (for the
    /// norms/embed/sink/router-bias/MoE experts) and upload the dense GEMM
    /// weights to the device.
    pub fn new(cfg: Config, w: HashMap<String, Vec<f32>>) -> Result<Self, String> {
        let mut dense = DenseDevice::new()?;
        let hid = cfg.hidden_size;
        for layer in 0..cfg.num_hidden_layers {
            let kind = cfg.layer_kind(layer);
            let (q, k, v, o_in) = cfg.attn_dims(kind);
            dense.upload(
                &w_name(layer, "self_attn.qkv_proj.weight"),
                &w[&w_name(layer, "self_attn.qkv_proj.weight")],
                q + k + v,
                hid,
            )?;
            dense.upload(
                &w_name(layer, "self_attn.o_proj.weight"),
                &w[&w_name(layer, "self_attn.o_proj.weight")],
                hid,
                o_in,
            )?;
            if cfg.is_moe_layer(layer) {
                dense.upload(
                    &w_name(layer, "mlp.gate.weight"),
                    &w[&w_name(layer, "mlp.gate.weight")],
                    cfg.n_routed_experts,
                    hid,
                )?;
            } else {
                dense.upload(
                    &w_name(layer, "mlp.gate_proj.weight"),
                    &w[&w_name(layer, "mlp.gate_proj.weight")],
                    cfg.intermediate_size,
                    hid,
                )?;
                dense.upload(
                    &w_name(layer, "mlp.up_proj.weight"),
                    &w[&w_name(layer, "mlp.up_proj.weight")],
                    cfg.intermediate_size,
                    hid,
                )?;
                dense.upload(
                    &w_name(layer, "mlp.down_proj.weight"),
                    &w[&w_name(layer, "mlp.down_proj.weight")],
                    hid,
                    cfg.intermediate_size,
                )?;
            }
        }
        dense.upload("lm_head.weight", &w["lm_head.weight"], cfg.vocab_size, hid)?;
        Ok(Self { model: Model::new(cfg, w), dense })
    }

    fn geom(&self, kind: LayerKind) -> M26Geom {
        let cfg = &self.model.cfg;
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
            value_scale: f64::from(cfg.attention_value_scale), // T18: scale V before the codec
        }
    }

    fn attn_block(
        &self,
        layer: usize,
        x: &[f32],
        pos: &[i64],
        cache: &mut Fp8KvCache,
        rope: &RopeTables,
    ) -> Vec<f32> {
        let cfg = &self.model.cfg;
        let kind = cfg.layer_kind(layer);
        let (q_rows, k_rows, v_rows, _o_in) = cfg.attn_dims(kind);
        let t = pos.len();
        let g = self.geom(kind);
        let sink: Option<&[f32]> = if cfg.add_swa_attention_sink_bias && kind == LayerKind::Swa {
            Some(self.model.w.get(&w_name(layer, "self_attn.attention_sink_bias"))
                .map(|v| v.as_slice()).unwrap_or(&[]))
        } else {
            None
        };
        let n_q = cfg.num_attention_heads;
        let n_kv = match kind {
            LayerKind::Ga => cfg.num_key_value_heads,
            LayerKind::Swa => cfg.swa_num_key_value_heads,
        };
        let d_qk = cfg.head_dim;
        let theta = match kind {
            LayerKind::Ga => f64::from(cfg.rope_theta),
            LayerKind::Swa => f64::from(cfg.swa_rope_theta),
        };
        let factor = f64::from(cfg.partial_rotary_factor);

        if t > 1 {
            // Prefill: Q-on-device + device-KV (I5-R17 step 3). Keep QKV on the
            // device, split/RoPE/store on the device, attention reads device KV,
            // then sync the FP8 codes to the host cache for the decode steps.
            let tq = std::time::Instant::now();
            let d_qkv = self
                .dense
                .linear_to_device(x, &w_name(layer, "self_attn.qkv_proj.weight"), t)
                .expect("qkv sgemm dev");
            profile("qkv_sgemm", now_ms(&tq));
            let (d_q, d_k, d_v) =
                split_qkv_device(&d_qkv, t, q_rows, k_rows, v_rows).expect("qkv split");
            let d_pos = pos_to_device(pos).expect("pos upload");
            let ts = std::time::Instant::now();
            let d_q_rot = rope_apply_device(theta, factor, &d_q, &d_pos, t as i32, n_q as i32, d_qk as i32)
                .expect("rope q");
            let d_k_rot = rope_apply_device(theta, factor, &d_k, &d_pos, t as i32, n_kv as i32, d_qk as i32)
                .expect("rope k");
            profile("split_rope", now_ms(&ts));
            let ta = std::time::Instant::now();
            let (d_k_codes, d_v_codes) =
                kv_store_fp8_device(&g, &d_k_rot, &d_v, t as i32).expect("kv store");
            // Host cache sync (decode steps read the host codes).
            let mut k_codes_host = vec![0u8; t * n_kv * d_qk];
            let mut v_codes_host = vec![0u8; t * n_kv * cfg.v_head_dim];
            d_k_codes.download(&mut k_codes_host).expect("k codes download");
            d_v_codes.download(&mut v_codes_host).expect("v codes download");
            cache.append_codes(&k_codes_host, &v_codes_host, pos);
            profile("kv_append_get", now_ms(&ta));
            let to = std::time::Instant::now();
            // Cross-chunk attention: read the accumulated host KV cache (the full
            // prefix), upload it to the device, and attend [chunk Q] x [prefix K/V].
            // The kernel already takes t (Q rows) != s (K rows), so this is the same
            // causal result as the single-shot call, only chunk-bounded in memory.
            let (k_codes_all, v_codes_all, k_pos_all) = cache.get();
            let s = k_pos_all.len();
            let d_k_all = DeviceBuffer::alloc(k_codes_all.len()).expect("k cache alloc");
            d_k_all.upload(k_codes_all).expect("k cache upload");
            let d_v_all = DeviceBuffer::alloc(v_codes_all.len()).expect("v cache alloc");
            d_v_all.upload(v_codes_all).expect("v cache upload");
            let d_kpos_all = pos_to_device(k_pos_all).expect("k pos upload");
            let d_out = prefill_attention_dev(
                &g, &d_q_rot, &d_k_all, &d_v_all, &d_pos, &d_kpos_all,
                t as i32, s as i32, sink,
            )
            .expect("prefill dev");
            profile("attention", now_ms(&to));
            let tp = std::time::Instant::now();
            let o = self.dense.linear_from_device(&d_out, &w_name(layer, "self_attn.o_proj.weight"), t)
                .expect("o_proj sgemm");
            profile("o_proj_sgemm", now_ms(&tp));
            return o;
        }

        // Decode (T == 1): host path (RoPE cache + host KV encode + host cache).
        let tq = std::time::Instant::now();
        let qkv = self
            .dense
            .linear(x, &w_name(layer, "self_attn.qkv_proj.weight"), t)
            .expect("qkv sgemm");
        profile("qkv_sgemm", now_ms(&tq));
        let total = q_rows + k_rows + v_rows;
        let mut q = Vec::with_capacity(t * q_rows);
        let mut k = Vec::with_capacity(t * k_rows);
        let mut v = Vec::with_capacity(t * v_rows);
        for r in 0..t {
            let base = r * total;
            q.extend_from_slice(&qkv[base..base + q_rows]);
            k.extend_from_slice(&qkv[base + q_rows..base + q_rows + k_rows]);
            v.extend_from_slice(&qkv[base + q_rows + k_rows..base + total]);
        }

        let ts = std::time::Instant::now();
        let (cos, sin) = rope.get(kind);
        let q_rot = apply_rotary_precomputed(&q, t, n_q, d_qk, rope.rd, cos, sin);
        let k_rot = apply_rotary_precomputed(&k, t, n_kv, d_qk, rope.rd, cos, sin);
        profile("split_rope", now_ms(&ts));

        let ta = std::time::Instant::now();
        cache.append(&k_rot, &v, pos);
        let (k_codes, v_codes, k_pos) = cache.get();
        profile("kv_append_get", now_ms(&ta));

        let to = std::time::Instant::now();
        let d_out = decode_attention_device(&g, &q_rot, k_codes, v_codes, pos, k_pos, DECODE_SPLITS, sink)
            .expect("decode_attention");
        profile("attention", now_ms(&to));
        let tp = std::time::Instant::now();
        let o = self.dense.linear_from_device(&d_out, &w_name(layer, "self_attn.o_proj.weight"), t)
            .expect("o_proj sgemm");
        profile("o_proj_sgemm", now_ms(&tp));
        o
    }

    fn ffn_block(&self, layer: usize, x: &[f32], wire: Option<&mut WireClient>) -> Vec<f32> {
        let cfg = &self.model.cfg;
        let hid = cfg.hidden_size;
        let t = x.len() / hid;
        if !cfg.is_moe_layer(layer) {
            let gate = self.dense.linear(x, &w_name(layer, "mlp.gate_proj.weight"), t).expect("gate sgemm");
            let up = self.dense.linear(x, &w_name(layer, "mlp.up_proj.weight"), t).expect("up sgemm");
            let inter = cfg.intermediate_size;
            let mut h = Vec::with_capacity(gate.len());
            for r in 0..t {
                for j in 0..inter {
                    h.push(crate::norm::silu(gate[r * inter + j]) * up[r * inter + j]);
                }
            }
            return self.dense.linear(&h, &w_name(layer, "mlp.down_proj.weight"), t).expect("down sgemm");
        }
        let tg = std::time::Instant::now();
        let logits = self.dense.linear(x, &w_name(layer, "mlp.gate.weight"), t).expect("gate sgemm");
        let bias = &self.model.w[&w_name(layer, "mlp.gate.e_score_correction_bias")];
        let (idx, wts) = router_from_logits(&logits, bias, t, cfg.n_routed_experts, cfg.num_experts_per_tok);
        profile("router_gemm_topk", now_ms(&tg));
        if let Some(wire) = wire {
            // Wire MoE: router -> quantize -> 4-Spark RPC -> R8 rank sum, chunked
            // to <= 2,048 tokens per request (ARCHITECTURE §11.4 standalone
            // prefill chunk; the Spark kernel caps a launch at 8 x class = 16,384
            // routed rows). The chunk returns are concatenated in token order.
            const MOE_CHUNK: usize = 2048;
            let topk = cfg.num_experts_per_tok;
            let mut out = Vec::with_capacity(t * hid);
            for cs in (0..t).step_by(MOE_CHUNK) {
                let ce = (cs + MOE_CHUNK).min(t);
                let tw = std::time::Instant::now();
                let idx_c = &idx[cs * topk..ce * topk];
                let wts_c = &wts[cs * topk..ce * topk];
                let routes: Vec<(u32, f32)> = idx_c.iter().zip(wts_c.iter())
                    .map(|(&e, &w)| (e as u32, w)).collect();
                let chunk_out = wire
                    .moe_layer(layer as u32, &x[cs * hid..ce * hid], &routes, topk)
                    .expect("wire moe_layer");
                profile("wire_moe_chunk", now_ms(&tw));
                out.extend_from_slice(&chunk_out);
            }
            return out;
        }
        let w_ref = &self.model.w;
        moe_forward(x, &idx, &wts, hid, cfg.num_experts_per_tok, |e, rows| {
            dense_ffn(
                rows,
                &w_ref[&w_name(layer, &format!("mlp.experts.{e}.gate_proj.weight"))],
                &w_ref[&w_name(layer, &format!("mlp.experts.{e}.up_proj.weight"))],
                &w_ref[&w_name(layer, &format!("mlp.experts.{e}.down_proj.weight"))],
                hid,
                cfg.moe_intermediate_size,
            )
        })
    }

    /// Full forward (GPU attention + GPU dense + CPU MoE/norms). `token_ids` are
    /// the new tokens; the per-layer `caches` persist across calls (prefill then
    /// decode). Returns `(logits, hidden)`.
    pub fn forward(
        &self,
        token_ids: &[usize],
        caches: &mut [Fp8KvCache],
        mut wire: Option<&mut WireClient>,
    ) -> (Vec<f32>, Vec<f32>) {
        let cfg = &self.model.cfg;
        let hid = cfg.hidden_size;
        let t = token_ids.len();
        // Positions continue from the cache length (single-sequence assumption).
        let base = caches[0].k_pos.len() as i64;
        let pos: Vec<i64> = (base..base + t as i64).collect();
        let rope = RopeTables::new(cfg, &pos);
        let h_all = embed(token_ids, &self.model.w["embed_tokens.weight"], cfg.vocab_size, hid);
        // Chunked prefill (PREFILL_CHUNK tokens per pass) so the device Q/K/V
        // buffers stay bounded; the attention reads the accumulated KV cache
        // (cross-chunk), which keeps the numerics identical to the single-shot
        // causal attention. The decode (t == 1) is a single one-token chunk.
        let mut last_hidden: Option<Vec<f32>> = None;
        for cs in (0..t).step_by(prefill_chunk()) {
            let ce = (cs + prefill_chunk()).min(t);
            let mut h = h_all[cs * hid..ce * hid].to_vec();
            let pos_c = &pos[cs..ce];
            for layer in 0..cfg.num_hidden_layers {
                crate::wire::tl("layer_start", Some(layer as u32), None);
                let x = rmsnorm(&h, &self.model.w[&w_name(layer, "input_layernorm.weight")], cfg.layernorm_epsilon);
                let a = self.attn_block(layer, &x, pos_c, &mut caches[layer], &rope);
                for i in 0..h.len() {
                    h[i] += a[i];
                }
                crate::wire::tl("attn_done", Some(layer as u32), None);
                let x = rmsnorm(&h, &self.model.w[&w_name(layer, "post_attention_layernorm.weight")], cfg.layernorm_epsilon);
                let f = self.ffn_block(layer, &x, wire.as_mut().map(|w| &mut **w));
                for i in 0..h.len() {
                    h[i] += f[i];
                }
                // Numerics (b) hook: dump this chunk's post-layer hidden rows
                // (absolute position >= `MIMO26_DUMP_ROW_START`) as f32 LE, one
                // file per (chunk, layer), so a harness can diff chunk 4096 vs
                // 2048 at the SAME token positions and locate the first layer
                // whose cross-chunk t != s attention diverges.
                if let Some(dir) = std::env::var_os("MIMO26_DUMP_HIDDENS") {
                    let start = std::env::var("MIMO26_DUMP_ROW_START")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(0);
                    let nrows = h.len() / hid;
                    let r0 = start.saturating_sub(cs).min(nrows);
                    if r0 < nrows {
                        use std::io::Write;
                        let path = std::path::Path::new(&dir)
                            .join(format!("c{cs}.l{layer}.f32"));
                        if let Ok(mut f) = std::fs::File::create(&path) {
                            for r in r0..nrows {
                                let row = &h[r * hid..(r + 1) * hid];
                                for v in row {
                                    let _ = f.write_all(&v.to_le_bytes());
                                }
                            }
                        }
                    }
                }
            }
            last_hidden = Some(h);
        }
        let h = last_hidden.unwrap_or_default();
        let hidden = h.clone();
        let logits = self
            .dense
            .linear(&rmsnorm(&h, &self.model.w["norm.weight"], cfg.layernorm_epsilon), "lm_head.weight", h.len() / hid)
            .expect("lm_head sgemm");
        (logits, hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::Fp8KvCache;
    use crate::config::Config;

    /// D1: `clear()` must empty every KV row so a request's cache cannot leak
    /// into the next request (the ISO black-box proof).
    #[test]
    fn clear_drops_all_kv_rows() {
        let cfg = Config::tiny();
        for layer in 0..cfg.num_hidden_layers {
            let mut c = Fp8KvCache::for_layer(&cfg, layer);
            let n_kv = if cfg.layer_kind(layer) == crate::config::LayerKind::Ga {
                cfg.num_key_value_heads
            } else {
                cfg.swa_num_key_value_heads
            };
            let k: Vec<u8> = vec![7u8; 5 * n_kv * cfg.head_dim];
            let v: Vec<u8> = vec![3u8; 5 * n_kv * cfg.v_head_dim];
            let pos: Vec<i64> = vec![0, 1, 2, 3, 4];
            c.append_codes(&k, &v, &pos);
            assert!(!c.k_codes.is_empty());
            c.clear();
            assert!(c.k_codes.is_empty());
            assert!(c.v_codes.is_empty());
            assert!(c.k_pos.is_empty());
        }
    }
}
