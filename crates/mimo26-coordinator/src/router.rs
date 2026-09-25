//! MoE router — FP32 logits, FP64 sigmoid/selection (T22), twin of
//! `oracle/mimo26/nn/layers.py::router`.
//!
//! T22 (ADVISOR-I3 §10.4, COHERENCE-TRAPS §6.1): `e_score_correction_bias` is
//! F32 `[256]` and the reference selects top-k on FP32 biased scores. A BF16
//! bias flips near-ties (vLLM's open quality item). We compute FP32 logits
//! (`F.linear(h.float(), w.float())`), FP64 sigmoid and FP64 selection, so a
//! near-tie that flips under BF16 does not flip here.
//!
//! Selection uses `sigmoid(logits) + bias`; the returned weights use the
//! **unbiased** sigmoid scores, normalized across the top-k when `norm_topk`.

/// Route one batch. Returns `(indices, weights)` each flattened to
/// `[rows, top_k]` (indices as `usize` expert ids, weights FP32).
///
/// `x` is `[rows, in_dim]`; `gate` is `[n_experts, in_dim]` (BF16 checkpoint
/// weights already dequantized to FP32); `bias` is `[n_experts]` FP32
/// (`e_score_correction_bias`). `top_k` is the router's `num_experts_per_tok`.
pub fn router(
    x: &[f32],
    gate: &[f32],
    bias: &[f32],
    in_dim: usize,
    n_experts: usize,
    top_k: usize,
) -> (Vec<usize>, Vec<f32>) {
    assert_eq!(gate.len(), n_experts * in_dim, "gate shape must be [n_experts, in_dim]");
    assert_eq!(bias.len(), n_experts, "bias length must equal n_experts");
    assert!(top_k <= n_experts, "top_k exceeds n_experts");
    assert_eq!(x.len() % in_dim, 0, "x length must be a multiple of in_dim");
    let rows = x.len() / in_dim;

    // FP32 logits (oracle: `linear(...).astype(float64)` — logits in FP32
    // first, then promoted).
    let logits = crate::linear::linear(x, gate, in_dim, n_experts);
    router_from_logits(&logits, bias, rows, n_experts, top_k)
}

/// The FP64 sigmoid + top-k selection over already-computed FP32 logits
/// `[rows, n_experts]`. Split out so the GPU dense path computes the logits via
/// cuBLAS and reuses this exact selection body (T22 stays FP64-sigmoid).
pub fn router_from_logits(
    logits: &[f32],
    bias: &[f32],
    rows: usize,
    n_experts: usize,
    top_k: usize,
) -> (Vec<usize>, Vec<f32>) {
    assert_eq!(logits.len(), rows * n_experts, "logits length");
    assert_eq!(bias.len(), n_experts, "bias length must equal n_experts");
    assert!(top_k <= n_experts, "top_k exceeds n_experts");
    let mut indices = Vec::with_capacity(rows * top_k);
    let mut weights = Vec::with_capacity(rows * top_k);
    for r in 0..rows {
        // FP64 sigmoid scores.
        let mut scores = vec![0.0f64; n_experts];
        for e in 0..n_experts {
            let s = logits[r * n_experts + e] as f64;
            scores[e] = 1.0 / (1.0 + (-s).exp());
        }
        // Descending order by biased selection score; ties break on the lower
        // expert id (stable sort over ascending indices).
        let mut order: Vec<usize> = (0..n_experts).collect();
        order.sort_by(|&a, &b| {
            let sa = scores[a] + bias[a] as f64;
            let sb = scores[b] + bias[b] as f64;
            sb.partial_cmp(&sa).expect("sigmoid scores are finite")
        });
        // Weights use the unbiased sigmoid scores, normalized over the top-k.
        let mut wsum = 0.0f64;
        for &e in &order[..top_k] {
            wsum += scores[e];
        }
        for &e in &order[..top_k] {
            indices.push(e);
            weights.push((scores[e] / wsum) as f32);
        }
    }
    (indices, weights)
}
