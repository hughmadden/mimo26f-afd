//! Coordinator weight loading (go-window step 1) — read the coordinator tensors
//! from the checkpoint shards and dequantize them to the `forward::Model` weight
//! map. The coordinator owns embeddings, attention, norms, the router, and the
//! dense layer-0 FFN (ARCHITECTURE §3); the routed experts live on the Sparks.
//!
//! Dtypes (verified against the live checkpoint, `spike/real_loader.py`):
//! embeddings / lm_head / norms / o_proj / sink / router gate are BF16; the
//! router bias is F32; the fused-QKV and the dense layer-0 gate/up/down are
//! `F8_E4M3` block-128 with a per-shard padded `*_scale_inv` grid. The fused-QKV
//! is TP4 shard-major (`[Q_c|K_c|V_c]` x4) and must be regrouped to `[Q|K|V]`
//! projection-major (the T1 word-salad trap) via `mimo26-load`'s golden-pinned
//! `reconstruct_layer_qkv`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mimo26_load::e4m3::decode_e4m3;
use mimo26_load::fused::reconstruct_layer_qkv;
use mimo26_load::Mat;
use mimo26_repack::safetensors::SafetensorsHeader;

use crate::config::{Config, LayerKind};
use crate::forward::w_name;

/// Dequantize a BF16 (little-endian u16) tensor to f32.
pub fn bf16_to_f32(raw: &[u8]) -> Vec<f32> {
    assert_eq!(raw.len() % 2, 0, "BF16 tensor must be u16-aligned");
    raw.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// Read a tensor's raw bytes + dtype + shape from the first shard that has it.
/// Checkpoint quirk: most backbone tensors are under `model.` but `lm_head`
/// is stored at the root.
fn read_raw(dir: &Path, name: &str) -> Option<(String, Vec<usize>, Vec<u8>)> {
    for i in 0..64usize {
        let path = dir.join(format!("model_pp0_ep{i}_shard0.safetensors"));
        if !path.exists() {
            continue;
        }
        let Ok(hdr) = SafetensorsHeader::read(&path) else { continue };
        for full in [format!("model.{name}"), name.to_string()] {
            let Some(entry) = hdr.tensors.get(&full) else { continue };
            let raw = hdr.read_tensor(&path, &full).ok()?;
            return Some((entry.dtype.clone(), entry.shape.clone(), raw));
        }
    }
    None
}

/// Read a BF16/F32 tensor as f32.
fn read_f32(dir: &Path, name: &str) -> Option<Vec<f32>> {
    let (dtype, _shape, raw) = read_raw(dir, name)?;
    Some(match dtype.as_str() {
        "BF16" => bf16_to_f32(&raw),
        "F32" => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!("unhandled dtype {other} for {name}"),
    })
}

/// Reconstruct + dequantize one layer's fused-QKV (TP4 shard-major → [Q|K|V]).
fn load_qkv(dir: &Path, layer: usize, kind: LayerKind) -> Vec<f32> {
    let (per, per_grid) = match kind {
        LayerKind::Ga => ((3072usize, 192, 128), 27usize),
        LayerKind::Swa => ((3072usize, 384, 256), 29usize),
    };
    let w_name_full = format!("model.{}", w_name(layer, "self_attn.qkv_proj.weight"));
    let s_name_full = format!("model.{}", w_name(layer, "self_attn.qkv_proj.weight_scale_inv"));
    let (_, _, wbytes) = read_raw(dir, &w_name(layer, "self_attn.qkv_proj.weight"))
        .unwrap_or_else(|| panic!("qkv missing: {w_name_full}"));
    let (_, _, sbytes) = read_raw(dir, &w_name(layer, "self_attn.qkv_proj.weight_scale_inv"))
        .unwrap_or_else(|| panic!("qkv scale missing: {s_name_full}"));

    let per_rows = per.0 + per.1 + per.2;
    let mut weights = Vec::with_capacity(4);
    let mut scales = Vec::with_capacity(4);
    for c in 0..4usize {
        let w = Mat::from_row_major(
            per_rows,
            crate::config::Config::real().hidden_size,
            wbytes[c * per_rows * 4096..(c + 1) * per_rows * 4096].to_vec(),
        )
        .expect("qkv shard weight");
        // The `*_scale_inv` grid is F32 (4 B each).
        let sdata: Vec<f32> = sbytes[c * per_grid * 32 * 4..(c + 1) * per_grid * 32 * 4]
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let s = Mat::from_row_major(per_grid, 32, sdata).expect("qkv shard scale");
        weights.push(w);
        scales.push(s);
    }
    reconstruct_layer_qkv(&weights, &scales, per, (128, 128), false)
        .expect("qkv reconstruct")
        .data
}

/// Dequantize one non-sharded FP8 block-128 matrix (the dense layer 0).
fn load_fp8_matrix(dir: &Path, name: &str, rows: usize, cols: usize) -> Vec<f32> {
    let (_, _, wbytes) = read_raw(dir, name).unwrap_or_else(|| panic!("fp8 missing: model.{name}"));
    let (_, _, sbytes) = read_raw(dir, &format!("{name}_scale_inv"))
        .unwrap_or_else(|| panic!("fp8 scale missing: model.{name}_scale_inv"));
    let scale_cols = cols / 128;
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let code = wbytes[r * cols + c];
            let si = ((r / 128) * scale_cols + (c / 128)) * 4;
            let scale = f32::from_le_bytes(sbytes[si..si + 4].try_into().unwrap());
            out[r * cols + c] = decode_e4m3(code) as f32 * scale;
        }
    }
    out
}

fn load_into(w: &mut HashMap<String, Vec<f32>>, dir: &Path, name: &str, want: usize) {
    let data = read_f32(dir, name)
        .unwrap_or_else(|| panic!("coordinator tensor missing: model.{name}"));
    assert_eq!(data.len(), want, "{name}: got {} elems, want {want}", data.len());
    w.insert(name.to_string(), data);
}

/// Load the full coordinator weight set into the `forward::Model` map.
pub fn load_coordinator_weights(dir: &Path, cfg: &Config) -> HashMap<String, Vec<f32>> {
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();

    load_into(&mut w, dir, "embed_tokens.weight", cfg.vocab_size * cfg.hidden_size);
    load_into(&mut w, dir, "lm_head.weight", cfg.vocab_size * cfg.hidden_size);
    load_into(&mut w, dir, "norm.weight", cfg.hidden_size);
    for layer in 0..cfg.num_hidden_layers {
        let kind = cfg.layer_kind(layer);
        let (q, k, v, o_in) = cfg.attn_dims(kind);
        load_into(&mut w, dir, &w_name(layer, "input_layernorm.weight"), cfg.hidden_size);
        load_into(&mut w, dir, &w_name(layer, "post_attention_layernorm.weight"), cfg.hidden_size);
        load_into(&mut w, dir, &w_name(layer, "self_attn.o_proj.weight"), cfg.hidden_size * o_in);
        if kind == LayerKind::Swa && cfg.add_swa_attention_sink_bias {
            load_into(&mut w, dir, &w_name(layer, "self_attn.attention_sink_bias"), cfg.num_attention_heads);
        }
        // Fused-QKV (FP8, TP4 shard-major) → [Q|K|V] f32.
        let qkv = load_qkv(dir, layer, kind);
        assert_eq!(qkv.len(), (q + k + v) * cfg.hidden_size, "qkv len");
        w.insert(w_name(layer, "self_attn.qkv_proj.weight"), qkv);
        if cfg.is_moe_layer(layer) {
            load_into(&mut w, dir, &w_name(layer, "mlp.gate.weight"), cfg.n_routed_experts * cfg.hidden_size);
            load_into(&mut w, dir, &w_name(layer, "mlp.gate.e_score_correction_bias"), cfg.n_routed_experts);
        } else {
            // Dense layer 0 (FP8 block-128, not TP-sharded).
            for (part, rows, cols) in [
                ("gate_proj", cfg.intermediate_size, cfg.hidden_size),
                ("up_proj", cfg.intermediate_size, cfg.hidden_size),
                ("down_proj", cfg.hidden_size, cfg.intermediate_size),
            ] {
                let name = w_name(layer, &format!("mlp.{part}.weight"));
                let m = load_fp8_matrix(dir, &name, rows, cols);
                w.insert(name, m);
            }
        }
    }
    w
}

/// The shard directory for the coordinator weights (env override for tests).
pub fn weights_dir() -> PathBuf {
    std::env::var("MIMO26_WEIGHTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join("models/XiaomiMiMo/MiMo-V2.6-Flash-RL")
        })
}
