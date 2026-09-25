//! Coordinator weight loading (go-window step 1) — BF16 dequant + shape pins.
//! The fast unit test runs in the merge gate; the full load is `#[ignore]`d
//! (weights-host dependent, ~2.5 GB read + dequant).

use mimo26_coordinator::bf16_to_f32;

#[test]
fn bf16_dequant_is_exact_for_common_values() {
    let raw = [
        0x80u8, 0x3f, // 1.0 (0x3F80)
        0x00, 0x40, // 2.0 (0x4000)
        0x00, 0xc0, // -2.0 (0xC000)
    ];
    let v = bf16_to_f32(&raw);
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], -2.0);
}

/// Full coordinator weight load: shape pins against the real checkpoint.
#[test]
#[ignore = "weights-host checkpoint (~2.5 GB); run with MIMO26_WEIGHTS_DIR set"]
fn load_coordinator_weights_shapes() {
    let cfg = mimo26_coordinator::Config::real();
    let dir = mimo26_coordinator::weights_dir();
    assert!(dir.join("model_pp0_ep0_shard0.safetensors").exists(), "weights dir: {}", dir.display());
    let w = mimo26_coordinator::load_coordinator_weights(&dir, &cfg);
    assert_eq!(w["embed_tokens.weight"].len(), cfg.vocab_size * cfg.hidden_size);
    assert_eq!(w["lm_head.weight"].len(), cfg.vocab_size * cfg.hidden_size);
    assert_eq!(w["norm.weight"].len(), cfg.hidden_size);
    // Spot check that the dequant is real (finite, non-zero) — the checkpoint's
    // final-norm weight is not ~1.0 (measured ~3.5), so no exact-value pin here.
    let norm = &w["norm.weight"];
    assert!(norm.iter().all(|v| v.is_finite()), "norm.weight must be finite, got {:?}", &norm[..4]);
    assert!(norm.iter().any(|v| *v != 0.0), "norm.weight must be non-trivial");
    // Router bias is F32 (T22: not BF16) and the gate is [256, 4096].
    assert_eq!(w["layers.1.mlp.gate.e_score_correction_bias"].len(), cfg.n_routed_experts);
    assert_eq!(w["layers.1.mlp.gate.weight"].len(), cfg.n_routed_experts * cfg.hidden_size);

    // FP8 fused-QKV + dense layer-0 dequant match the spike reference (golden
    // from `spike/real_loader.py` + `spike/quant.py` on the real checkpoint).
    let qkv = &w["layers.0.self_attn.qkv_proj.weight"];
    let qkv_head = [
        0.03211374208331108, -0.010035544633865356, 0.03010663390159607, 0.00802843552082777,
        -0.004014217760413885, 0.009031989611685276, 0.040142178535461426, -0.0005644993507303298,
    ];
    for (i, want) in qkv_head.iter().enumerate() {
        assert!((qkv[i] - want).abs() < 1e-5, "qkv[{i}] got {} want {want}", qkv[i]);
    }
    let qkv_sum: f32 = qkv.iter().sum();
    assert!((qkv_sum - (-397.9837951660156)).abs() < 1.0, "qkv sum got {qkv_sum}");

    let gate = &w["layers.0.mlp.gate_proj.weight"];
    let gate_head = [
        -0.006018007639795542, -0.02036864124238491, -0.006018007639795542, -0.012036015279591084,
        -0.008332625962793827, 0.004629236645996571, 0.014813557267189026, 0.0010415782453492284,
    ];
    for (i, want) in gate_head.iter().enumerate() {
        assert!((gate[i] - want).abs() < 1e-5, "gate[{i}] got {} want {want}", gate[i]);
    }
    let gate_sum: f32 = gate.iter().sum();
    assert!((gate_sum - 7175.9541015625).abs() < 20.0, "gate sum got {gate_sum}");
}
