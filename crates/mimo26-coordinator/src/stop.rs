//! Stop/length semantics + T28 output checks (A2 F1; the X1a needle gate is
//! the engine-side proof). The coordinator owns "stop/length semantics".
//!
//! X1a (ADVISOR-I4 §3.0): the chat path honours the FULL EOS set
//! `[151643, 151645, 151672]` from `generation_config.json` — greedy stops at
//! the first sampled EOS and records it; nothing is emitted after EOS. T28
//! (ADVISOR-I4 §3.5) adds the output checks: no `REWARD:` (RL-rollout artifact)
//! and no role-less assistant marker in the decoded text.

use crate::token::{is_eos, IM_START};

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Sampled a token in the served EOS set.
    Eos(u32),
    /// Reached the output length cap (T23: default `max_output_tokens`).
    MaxTokens,
    /// Decoded text ends with a configured stop string.
    StopString(String),
}

/// Stop/length configuration. `Default` is the served EOS set + the T23
/// default output cap (65,536 — never the checkpoint's 2048).
#[derive(Debug, Clone)]
pub struct StopConfig {
    pub eos: Vec<u32>,
    pub max_tokens: usize,
    pub stop_strings: Vec<String>,
}

impl Default for StopConfig {
    fn default() -> Self {
        StopConfig {
            eos: crate::token::EOS_TOKEN_IDS.to_vec(),
            max_tokens: 65_536,
            stop_strings: Vec::new(),
        }
    }
}

impl StopConfig {
    /// Decision after one sampled token: EOS first, then length, then stop
    /// strings. `generated` is the number of tokens produced so far (after the
    /// current token is appended).
    pub fn check(&self, token: u32, generated: usize, decoded: &str) -> Option<StopReason> {
        if is_eos(token) || self.eos.contains(&token) {
            return Some(StopReason::Eos(token));
        }
        if generated >= self.max_tokens {
            return Some(StopReason::MaxTokens);
        }
        for s in &self.stop_strings {
            if !s.is_empty() && decoded.ends_with(s) {
                return Some(StopReason::StopString(s.clone()));
            }
        }
        None
    }
}

/// T28 (ADVISOR-I4 §3.5) chat-path output checks: no `REWARD:` (RL-rollout
/// artifact) and no role-less assistant marker (raw-mode leak). "Nothing after
/// EOS" holds by construction — the loop stops at the EOS set. Returns the
/// violations (empty = clean).
pub fn check_output(out: &str) -> Vec<String> {
    let mut bad = Vec::new();
    if out.contains("REWARD:") {
        bad.push("REWARD: (RL-rollout artifact — T28)".to_string());
    }
    if out.contains(IM_START) {
        bad.push("role-less assistant marker in output (T28)".to_string());
    }
    bad
}
