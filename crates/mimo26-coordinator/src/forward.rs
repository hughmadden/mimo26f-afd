//! The model forward — embeddings → 48/tiny layers (attention + FFN) → final
//! norm → lm_head. The CPU golden-reference twin of `oracle/mimo26/model.py`
//! (`MiMoModel.forward`): composes this crate's glue (rmsnorm/linear/dense_ffn/
//! router/moe_forward/embed) with `mimo26-attn` (attention/RoPE/KV cache).
//!
//! This is the bridge the GPU serving path is tested against (I-Gold: Python
//! oracle → Rust twin → engine). Weights are `HashMap<name, Vec<f32>>` keyed by
//! the oracle `w_name(layer, part)` convention, row-major `[out, in]`.

use std::collections::HashMap;

use mimo26_attn::attn::attention;
use mimo26_attn::cache::{RowStore, StoreMode, SwaRing};
use mimo26_attn::geom::AttnSpec;
use mimo26_attn::rope::apply_rotary;
use mimo26_attn::{Family, NaiveBits};

use crate::config::{Config, LayerKind};
use crate::{dense_ffn, embed, linear, moe_forward, rmsnorm, router};

/// One layer's KV cache: GA grows (no eviction), SWA is a window ring.
pub enum LayerCache {
    Ga(RowStore),
    Swa(SwaRing),
}

impl LayerCache {
    pub fn for_layer(cfg: &Config, layer: usize) -> Self {
        Self::for_layer_mode(cfg, layer, StoreMode::F32)
    }

    /// Build a layer cache with an explicit store mode (`F32` or `Fp8Unit`).
    /// `Fp8Unit` gives the CPU twin the same FP8 unit-scale KV round-trip the
    /// GPU attention uses, so a golden drops to the accumulation-order class.
    pub fn for_layer_mode(cfg: &Config, layer: usize, mode: StoreMode) -> Self {
        let kind = cfg.layer_kind(layer);
        let (n_kv, d_qk, d_v) = match kind {
            LayerKind::Ga => (cfg.num_key_value_heads, cfg.head_dim, cfg.v_head_dim),
            LayerKind::Swa => (cfg.swa_num_key_value_heads, cfg.head_dim, cfg.v_head_dim),
        };
        match kind {
            LayerKind::Ga => LayerCache::Ga(RowStore::new(
                mode,
                n_kv,
                d_qk,
                d_v,
                cfg.attention_value_scale,
            )),
            LayerKind::Swa => LayerCache::Swa(SwaRing::new(
                mode,
                n_kv,
                d_qk,
                d_v,
                cfg.sliding_window,
                cfg.attention_value_scale,
            )),
        }
    }

    pub fn append(&mut self, k: &[f32], v: &[f32], pos: &[i64]) {
        match self {
            LayerCache::Ga(s) => s.append(k, v, pos.len(), NaiveBits::NONE).expect("append"),
            LayerCache::Swa(s) => s.append(k, v, pos, NaiveBits::NONE).expect("append"),
        }
    }

    pub fn get(&self) -> (Vec<f32>, Vec<f32>, Vec<i64>) {
        match self {
            LayerCache::Ga(s) => {
                let (k, v) = s.rows(NaiveBits::NONE);
                (k, v, (0..s.len() as i64).collect())
            }
            LayerCache::Swa(s) => s.get(NaiveBits::NONE),
        }
    }
}

/// The weight-name helper (oracle `w_name`).
pub fn w_name(layer: usize, part: &str) -> String {
    format!("layers.{layer}.{part}")
}

/// A small deterministic LCG (Numerical Recipes) so the same weights can be
/// regenerated bit-exactly in Rust and Python for the golden-lock.
pub struct Lcg(pub u32);
impl Lcg {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / 16_777_216.0
    }
    pub fn next_signed(&mut self, scale: f32) -> f32 {
        (self.next_f32() - 0.5) * 2.0 * scale
    }
}

/// Generate a deterministic weight set (uniform in `[-scale, scale]`) in the
/// oracle `init_weights` order. Mirrors `MiMoConfig.tiny()` geometry; norms are
/// ones, the sink bias and `e_score_correction_bias` are zeros (as the oracle).
pub fn gen_weights(cfg: &Config, seed: u32, scale: f32) -> HashMap<String, Vec<f32>> {
    let mut rng = Lcg(seed);
    let hid = cfg.hidden_size;
    let mut draw = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.next_signed(scale)).collect() };
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    w.insert("embed_tokens.weight".into(), draw(cfg.vocab_size * hid));
    w.insert("lm_head.weight".into(), draw(cfg.vocab_size * hid));
    w.insert("norm.weight".into(), vec![1.0f32; hid]);
    for layer in 0..cfg.num_hidden_layers {
        let kind = cfg.layer_kind(layer);
        let (q, k, v, o_in) = cfg.attn_dims(kind);
        w.insert(w_name(layer, "input_layernorm.weight"), vec![1.0f32; hid]);
        w.insert(w_name(layer, "post_attention_layernorm.weight"), vec![1.0f32; hid]);
        w.insert(w_name(layer, "self_attn.qkv_proj.weight"), draw((q + k + v) * hid));
        w.insert(w_name(layer, "self_attn.o_proj.weight"), draw(hid * o_in));
        if kind == LayerKind::Swa && cfg.add_swa_attention_sink_bias {
            w.insert(w_name(layer, "self_attn.attention_sink_bias"),
                vec![0.0f32; cfg.num_attention_heads]);
        }
        if cfg.is_moe_layer(layer) {
            w.insert(w_name(layer, "mlp.gate.weight"), draw(cfg.n_routed_experts * hid));
            w.insert(w_name(layer, "mlp.gate.e_score_correction_bias"),
                vec![0.0f32; cfg.n_routed_experts]);
            for e in 0..cfg.n_routed_experts {
                w.insert(w_name(layer, &format!("mlp.experts.{e}.gate_proj.weight")),
                    draw(cfg.moe_intermediate_size * hid));
                w.insert(w_name(layer, &format!("mlp.experts.{e}.up_proj.weight")),
                    draw(cfg.moe_intermediate_size * hid));
                w.insert(w_name(layer, &format!("mlp.experts.{e}.down_proj.weight")),
                    draw(hid * cfg.moe_intermediate_size));
            }
        } else {
            w.insert(w_name(layer, "mlp.gate_proj.weight"), draw(cfg.intermediate_size * hid));
            w.insert(w_name(layer, "mlp.up_proj.weight"), draw(cfg.intermediate_size * hid));
            w.insert(w_name(layer, "mlp.down_proj.weight"), draw(hid * cfg.intermediate_size));
        }
    }
    w
}

/// The composed model forward (CPU reference).
pub struct Model {
    pub cfg: Config,
    pub w: HashMap<String, Vec<f32>>,
}

impl Model {
    pub fn new(cfg: Config, w: HashMap<String, Vec<f32>>) -> Self {
        Model { cfg, w }
    }

    fn spec(&self, kind: LayerKind) -> AttnSpec {
        let cfg = &self.cfg;
        let (n_kv, theta) = match kind {
            LayerKind::Ga => (cfg.num_key_value_heads, cfg.rope_theta),
            LayerKind::Swa => (cfg.swa_num_key_value_heads, cfg.swa_rope_theta),
        };
        AttnSpec {
            family: if kind == LayerKind::Ga { Family::Ga } else { Family::Swa },
            n_q: cfg.num_attention_heads,
            n_kv,
            d_qk: cfg.head_dim,
            d_v: cfg.v_head_dim,
            window: cfg.sliding_window,
            theta: theta as f64,
            partial_rotary_factor: cfg.partial_rotary_factor as f64,
            value_scale: cfg.attention_value_scale,
            sink_enabled: kind == LayerKind::Swa && cfg.add_swa_attention_sink_bias,
        }
    }

    fn attn_block(&self, layer: usize, x: &[f32], pos: &[i64], cache: &mut LayerCache) -> Vec<f32> {
        let cfg = &self.cfg;
        let kind = cfg.layer_kind(layer);
        let (q_rows, k_rows, v_rows, o_in) = cfg.attn_dims(kind);
        let hid = cfg.hidden_size;
        let spec = self.spec(kind);
        let t = pos.len();

        let qkv = linear(x, &self.w[&w_name(layer, "self_attn.qkv_proj.weight")], hid, q_rows + k_rows + v_rows);
        // qkv is [T, q+k+v] row-major; split the COLUMNS per row (stride total).
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

        let q_rot = apply_rotary(&q, t, spec.n_q, spec.d_qk, pos, spec.theta,
            spec.partial_rotary_factor, NaiveBits::NONE);
        let k_rot = apply_rotary(&k, t, spec.n_kv, spec.d_qk, pos, spec.theta,
            spec.partial_rotary_factor, NaiveBits::NONE);

        cache.append(&k_rot, &v, pos);
        let (kk, vv, k_pos) = cache.get();

        let sink: Option<&[f32]> = if spec.sink_enabled {
            Some(self.w.get(&w_name(layer, "self_attn.attention_sink_bias"))
                .map(|v| v.as_slice()).unwrap_or(&[]))
        } else {
            None
        };
        let out = attention(&spec, &q_rot, &kk, &vv, pos, &k_pos, sink, NaiveBits::NONE)
            .expect("attention");
        // out is [t, n_q, d_v] = [t, o_in] already (n_q*d_v == o_in).
        linear(&out, &self.w[&w_name(layer, "self_attn.o_proj.weight")], o_in, hid)
    }

    fn ffn_block(&self, layer: usize, x: &[f32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let hid = cfg.hidden_size;
        if !cfg.is_moe_layer(layer) {
            return dense_ffn(
                x,
                &self.w[&w_name(layer, "mlp.gate_proj.weight")],
                &self.w[&w_name(layer, "mlp.up_proj.weight")],
                &self.w[&w_name(layer, "mlp.down_proj.weight")],
                hid,
                cfg.intermediate_size,
            );
        }
        let gate = &self.w[&w_name(layer, "mlp.gate.weight")];
        let bias = &self.w[&w_name(layer, "mlp.gate.e_score_correction_bias")];
        let (idx, wts) = router(x, gate, bias, hid, cfg.n_routed_experts, cfg.num_experts_per_tok);
        let w_ref = &self.w;
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

    /// Full forward: embeddings → layers → final norm → lm_head. Returns
    /// `(logits [T, vocab], hidden [T, hidden])`.
    pub fn forward(&self, token_ids: &[usize], caches: &mut [LayerCache]) -> (Vec<f32>, Vec<f32>) {
        self.forward_at(token_ids, caches, 0)
    }

    /// Position-aware forward: the new tokens start at `start_pos` (so a decode
    /// step after a prefill uses `start_pos = prefill_len`). The prefill case is
    /// `forward_at(ids, caches, 0)`.
    pub fn forward_at(
        &self,
        token_ids: &[usize],
        caches: &mut [LayerCache],
        start_pos: i64,
    ) -> (Vec<f32>, Vec<f32>) {
        let cfg = &self.cfg;
        let hid = cfg.hidden_size;
        let t = token_ids.len();
        let pos: Vec<i64> = (start_pos..start_pos + t as i64).collect();
        let mut h = embed(token_ids, &self.w["embed_tokens.weight"], cfg.vocab_size, hid);
        for layer in 0..cfg.num_hidden_layers {
            let x = rmsnorm(&h, &self.w[&w_name(layer, "input_layernorm.weight")], cfg.layernorm_epsilon);
            let a = self.attn_block(layer, &x, &pos, &mut caches[layer]);
            for i in 0..h.len() {
                h[i] += a[i];
            }
            let x = rmsnorm(&h, &self.w[&w_name(layer, "post_attention_layernorm.weight")], cfg.layernorm_epsilon);
            let f = self.ffn_block(layer, &x);
            for i in 0..h.len() {
                h[i] += f[i];
            }
        }
        let hidden = h.clone();
        let logits = linear(
            &rmsnorm(&h, &self.w["norm.weight"], cfg.layernorm_epsilon),
            &self.w["lm_head.weight"],
            hid,
            cfg.vocab_size,
        );
        (logits, hidden)
    }
}
