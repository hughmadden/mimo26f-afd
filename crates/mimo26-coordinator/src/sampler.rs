//! Sampler — temperature / top-p / top-k / greedy + repetition penalty, with
//! the ADVISOR-I3 §10.2 defaults (T23). The coordinator owns sampling
//! (ARCHITECTURE.md §3).

/// Per-token seen set for repetition penalty: a 152,576-bit (19 KB) bitmap per
/// request, built at prefill and extended with each sampled token (T23).
pub struct SeenSet {
    bits: Vec<u64>,
    vocab: usize,
}

impl SeenSet {
    pub fn new(vocab: usize) -> Self {
        SeenSet { bits: vec![0u64; (vocab + 63) / 64], vocab }
    }

    pub fn mark(&mut self, id: usize) {
        assert!(id < self.vocab, "token id {id} out of vocab {}", self.vocab);
        self.bits[id / 64] |= 1u64 << (id % 64);
    }

    pub fn contains(&self, id: usize) -> bool {
        assert!(id < self.vocab, "token id {id} out of vocab {}", self.vocab);
        self.bits[id / 64] & (1u64 << (id % 64)) != 0
    }
}

/// Sampling parameters. `Default` is the ADVISOR-I3 §10.2 / T23 server default:
/// temperature 1.0, top_p 0.95, repetition_penalty 1.05, top_k off, and
/// `max_tokens = max_output_tokens` (65,536) — never the checkpoint's 2048.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// 0.0 selects greedily (argmax).
    pub temperature: f32,
    /// Nucleus cutoff in (0, 1]; `>= 1.0` disables.
    pub top_p: f32,
    /// Keep only the top-k logits; `0` disables.
    pub top_k: usize,
    /// HF/vLLM semantics: positive logits / penalty, negative logits * penalty.
    pub repetition_penalty: f32,
    /// Default output cap (never 2048 — T23).
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 0,
            repetition_penalty: 1.05,
            max_tokens: 65_536,
        }
    }
}

/// Divide positive logits by `penalty` and multiply negative ones by it, only
/// for seen tokens (HF/vLLM `repetition_penalty` semantics, T23).
pub fn apply_repetition_penalty(logits: &mut [f32], seen: &SeenSet, penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for (i, l) in logits.iter_mut().enumerate() {
        if seen.contains(i) {
            *l = if *l < 0.0 { *l * penalty } else { *l / penalty };
        }
    }
}

/// Scale logits by `1/temperature`. `temperature == 0` is greedy and handled by
/// the caller (this function is a no-op then).
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature == 0.0 {
        return;
    }
    for l in logits.iter_mut() {
        *l /= temperature;
    }
}

/// Keep only the top-k logits; everything strictly below the k-th largest is
/// set to `-inf` (ties at the boundary are kept).
pub fn apply_top_k(logits: &mut [f32], top_k: usize) {
    if top_k == 0 || top_k >= logits.len() {
        return;
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).expect("finite logits"));
    let cutoff = logits[idx[top_k - 1]];
    for l in logits.iter_mut() {
        if *l < cutoff {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Nucleus filter: keep the smallest set of most-probable tokens whose
/// cumulative softmax mass reaches `top_p`; the rest are set to `-inf`.
pub fn apply_top_p(logits: &mut [f32], top_p: f32) {
    if top_p >= 1.0 {
        return;
    }
    let probs = softmax(logits);
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).expect("finite logits"));
    let mut cum = 0.0f32;
    let mut keep_above = f32::NEG_INFINITY;
    for &i in &idx {
        cum += probs[i];
        keep_above = logits[i];
        if cum >= top_p {
            break;
        }
    }
    for l in logits.iter_mut() {
        if *l < keep_above {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Numerically stable softmax.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Greedy argmax; ties resolve to the lowest token id.
pub fn greedy(logits: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = i;
        }
    }
    best
}

/// Full sample: repetition penalty → temperature (0 == greedy) → top-k →
/// top-p → softmax → multinomial with a caller-supplied uniform `u` in [0, 1).
/// `u` is injected so the draw is deterministic and testable.
pub fn sample(logits: &[f32], params: &SamplingParams, seen: &SeenSet, u: f32) -> usize {
    let mut l = logits.to_vec();
    apply_repetition_penalty(&mut l, seen, params.repetition_penalty);
    if params.temperature == 0.0 {
        return greedy(&l);
    }
    apply_temperature(&mut l, params.temperature);
    apply_top_k(&mut l, params.top_k);
    apply_top_p(&mut l, params.top_p);
    let p = softmax(&l);
    let mut cum = 0.0f32;
    for (i, &pi) in p.iter().enumerate() {
        cum += pi;
        if u < cum {
            return i;
        }
    }
    // Float tail: fall back to the argmax of the (already filtered) logits.
    greedy(&l)
}
