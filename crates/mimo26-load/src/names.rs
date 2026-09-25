//! Checkpoint tensor names — canonicalisation + fail-loud name audit.
//!
//! Citations: `spike/loader.py:30-43` (`canonical_name`, `is_backbone_weight`;
//! mirrors `mimo26/loader.py:28/:43`), `spike/model.py:39-56` (required-weight
//! and router-bias audits; mirrors `mimo26/model.py:34-42/:85-88`),
//! `spike/shard_index.py:20-74` (classification audit).

use crate::LoadError;
use std::collections::{BTreeMap, BTreeSet};

/// `spike/model.py:39` — missing any of these raises (never silently defaults).
pub const REQUIRED_WEIGHTS: [&str; 3] = ["embed.weight", "norm.weight", "lm_head.weight"];

/// `spike/shard_index.py:22-24` — expected non-backbone prefixes (dropped from
/// the backbone dict; counted by `classify`, never "UNCLASSIFIED").
pub const NON_BACKBONE_PREFIXES: [&str; 6] = [
    "audio_encoder.",
    "model.audio_encoder.",
    "visual.",
    "model.visual.",
    "speech_embeddings.",
    "model.speech_embeddings.",
];

/// `spike/loader.py:30-36` — MTP prefix FIRST (order matters), then the
/// generic `model.` strip. A reversed order loses the `mtp.` root and turns
/// `model.mtp.layers.0.*` into `layers.0.*` — silent MTP/backbone conflation.
pub fn canonical_name(raw: &str) -> String {
    if let Some(rest) = raw.strip_prefix("model.mtp.") {
        format!("mtp.{rest}")
    } else if let Some(rest) = raw.strip_prefix("model.") {
        rest.to_string()
    } else {
        raw.to_string()
    }
}

/// `spike/loader.py:39-43` — backbone only; drops mtp/dflash/audio/visual.
pub fn is_backbone_weight(raw: &str) -> bool {
    let c = canonical_name(raw);
    c.starts_with("layers.")
        || matches!(c.as_str(), "embed_tokens.weight" | "norm.weight" | "lm_head.weight")
}

/// `spike/model.py:54` — the per-layer router bias name (sigmoid router's
/// `e_score_correction_bias`).
pub fn router_bias_name(layer: usize) -> String {
    format!("layers.{layer}.mlp.gate.e_score_correction_bias")
}

/// `spike/model.py:49-52` — fail-loud required-weight audit.
pub fn audit_required_weights<'a, I: IntoIterator<Item = &'a str>>(names: I) -> Result<(), LoadError> {
    let have: BTreeSet<&str> = names.into_iter().collect();
    let missing: Vec<String> = REQUIRED_WEIGHTS
        .iter()
        .filter(|r| !have.contains(*r))
        .map(|r| (*r).to_string())
        .collect();
    if !missing.is_empty() {
        return Err(LoadError::MissingRequiredWeights { missing });
    }
    Ok(())
}

/// `spike/model.py:53-56` — fail-loud per-layer router-bias audit.
pub fn audit_router_bias<'a, I: IntoIterator<Item = &'a str>>(
    names: I,
    n_layers: usize,
) -> Result<(), LoadError> {
    let have: BTreeSet<&str> = names.into_iter().collect();
    for layer in 0..n_layers {
        let key = router_bias_name(layer);
        if !have.contains(key.as_str()) {
            return Err(LoadError::MissingRouterBias { key });
        }
    }
    Ok(())
}

/// `spike/shard_index.py:43-52` — kind classification of a raw tensor name.
pub fn classify(raw: &str) -> &'static str {
    if is_expert_name(raw) {
        return "expert";
    }
    if raw.starts_with("model.mtp.") {
        return "mtp";
    }
    if raw.contains("dflash") || raw.contains("draft") {
        return "dflash";
    }
    if NON_BACKBONE_PREFIXES.iter().any(|p| raw.starts_with(p)) {
        return "audio/visual/speech"; // expected non-backbone (dropped by is_backbone_weight)
    }
    if is_backbone_weight(raw) {
        "backbone"
    } else {
        "UNCLASSIFIED"
    }
}

/// Result of a clean shard-index audit (`spike/shard_index.py:73-74`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditReport {
    pub tensors: usize,
    pub kinds: BTreeMap<&'static str, usize>,
    pub qkv: usize,
}

/// `spike/shard_index.py:55-74` — fail-loud audit of a weight map: anything
/// UNCLASSIFIED raises, an `mtp` name whose canonical form loses the `mtp.`
/// root raises, and the fused-QKV count must equal `expected_qkv_layers`
/// (48 on the live index — parameterised here).
pub fn audit_weight_map<'a, I: IntoIterator<Item = &'a str>>(
    names: I,
    expected_qkv_layers: usize,
) -> Result<AuditReport, LoadError> {
    let mut kinds: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut problems: Vec<String> = Vec::new();
    let mut tensors = 0usize;
    let mut qkv = 0usize;
    for raw in names {
        tensors += 1;
        let k = classify(raw);
        *kinds.entry(k).or_insert(0) += 1;
        if k == "UNCLASSIFIED" {
            problems.push(raw.to_string());
        }
        if k == "mtp" && !canonical_name(raw).starts_with("mtp.") {
            problems.push(format!("{raw}: mtp prefix lost by canonicalisation"));
        }
        if is_qkv_proj_name(raw) {
            qkv += 1;
        }
    }
    if qkv != expected_qkv_layers {
        problems.push(format!("fused-QKV count {qkv} != {expected_qkv_layers}"));
    }
    if !problems.is_empty() {
        return Err(LoadError::NameAudit { problems });
    }
    Ok(AuditReport { tensors, kinds, qkv })
}

/// `spike/shard_index.py:67-68` — `model.layers.{i}.self_attn.qkv_proj.weight`.
fn is_qkv_proj_name(raw: &str) -> bool {
    match raw.strip_prefix("model.layers.") {
        Some(rest) => match take_uint(rest) {
            Some((_, tail)) => tail == ".self_attn.qkv_proj.weight",
            None => false,
        },
        None => false,
    }
}

/// `spike/shard_index.py:20-21` — EXPERT_RE:
/// `^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight(_scale)?$`
pub fn is_expert_name(raw: &str) -> bool {
    expert_name_parts(raw).is_some()
}

fn expert_name_parts(raw: &str) -> Option<(usize, usize)> {
    let rest = raw.strip_prefix("model.layers.")?;
    let (layer, rest) = take_uint(rest)?;
    let rest = rest.strip_prefix(".mlp.experts.")?;
    let (expert, rest) = take_uint(rest)?;
    let rest = rest.strip_prefix('.')?;
    for proj in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
        if rest == proj || rest == format!("{proj}_scale").as_str() {
            return Some((layer, expert));
        }
    }
    None
}

fn take_uint(s: &str) -> Option<(usize, &str)> {
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    Some((digits.parse().ok()?, &s[digits.len()..]))
}
