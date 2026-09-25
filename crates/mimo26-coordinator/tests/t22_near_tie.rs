//! T22 — router bias precision negative (ADVISOR-I3 §10.4, COHERENCE-TRAPS
//! §6.1). `e_score_correction_bias` is F32 [256]; the reference selects top-k
//! on FP32 biased scores. A BF16 bias flips a near-tie (vLLM's open quality
//! item). The coordinator router must therefore hold the bias in FP32 — this
//! test fails if a BF16-rounded bias is used for selection.

use mimo26_coordinator::router;

/// Two experts are near-tied after sigmoid; the FP32 bias selects expert 0,
/// but the BF16-rounded bias flips selection to expert 1.
#[test]
fn t22_near_tie_does_not_flip_under_fp32_bias() {
    // in_dim = 1, x = [1.0] so logits are exactly the gate rows.
    let x = [1.0f32];
    // Logits chosen for sigmoid ≈ 0.45 / ≈ 0.55 / ≈ 0.
    let gate = [
        -0.20067068934440613f32, // sigmoid ≈ 0.4500000015
        0.20067068934440613f32,  // sigmoid ≈ 0.5499999985
        -20.0f32,                // sigmoid ≈ 0
    ];
    // FP32 bias: 0.45+0.998 = 1.448 > 0.55+0.897 = 1.447 → expert 0 wins.
    let bias_fp32 = [0.998f32, 0.897, -10.0];
    // BF16-rounded bias (RNE): 0.998 → 0.996094, 0.897 → 0.898438.
    // 0.45+0.996094 = 1.446094 < 0.55+0.898438 = 1.448438 → expert 1 wins (flip).
    let bias_bf16 = [0.996094f32, 0.898438, -10.0];

    let (idx_fp, _) = router(&x, &gate, &bias_fp32, 1, 3, 2);
    assert_eq!(
        idx_fp,
        vec![0usize, 1],
        "FP32 bias must select expert 0 first (near-tie must not flip)"
    );

    let (idx_bf, _) = router(&x, &gate, &bias_bf16, 1, 3, 2);
    assert_eq!(
        idx_bf,
        vec![1usize, 0],
        "fixture must flip under a BF16 bias (otherwise the negative is vacuous)"
    );
}
