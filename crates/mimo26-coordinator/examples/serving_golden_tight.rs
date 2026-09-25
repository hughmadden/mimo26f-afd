//! Tightened serving golden (builder follow-up before L5): give the CPU twin the
//! SAME FP8 unit-scale KV round-trip (`StoreMode::Fp8Unit`) so the diff drops to
//! the A-f32q accumulation-order class — a subtle RoPE-offset / sink / SWA
//! off-by-one / value_scale bug can no longer hide inside the quantization bar.
//!
//! Covers: T>=300 prefill (the SWA ring wraps past its window, both GA and SWA
//! layers present) plus ONE T==1 decode step after the prefill. Run on the coordinator:
//! `cargo run --features cuda --example serving_golden_tight`.
//!
//! Bars (recorded here): prefill logits and decode logits each <= 1e-3 max-abs
//! vs the FP8-KV CPU twin (the A-f32q accumulation-order class through 4 layers).

use mimo26_attn::cache::StoreMode;
use mimo26_coordinator::config::Config;
use mimo26_coordinator::forward::{gen_weights, LayerCache, Model};
use mimo26_coordinator::serving::{Fp8KvCache, ServingModel};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()))
}

fn main() {
    let cfg = Config::tiny(); // 4 layers: layer 0 GA+dense, layers 1-3 SWA+MoE; window 4
    let w = gen_weights(&cfg, 42, 0.02);
    let cpu = Model::new(cfg.clone(), w.clone());
    let gpu = ServingModel::new(cfg.clone(), w).expect("ServingModel::new");

    let prefill: Vec<usize> = (0..300).map(|i| i % cfg.vocab_size).collect();
    let mut cpu_caches: Vec<LayerCache> = (0..cfg.num_hidden_layers)
        .map(|l| LayerCache::for_layer_mode(&cfg, l, StoreMode::Fp8Unit)).collect();
    let mut gpu_caches: Vec<Fp8KvCache> = (0..cfg.num_hidden_layers)
        .map(|l| Fp8KvCache::for_layer(&cfg, l)).collect();

    let (cpu_pre, _) = cpu.forward_at(&prefill, &mut cpu_caches, 0);
    let (gpu_pre, _) = gpu.forward(&prefill, &mut gpu_caches, None);
    // Chunked prefill returns only the last chunk's rows (logits over the final
    // chunk). Compare those against the tail rows of the full CPU twin.
    let tail = cpu_pre.len() - gpu_pre.len();
    let prefill_diff = max_abs_diff(&cpu_pre[tail..], &gpu_pre);

    // One decode step at position 300 (position continuity through the cache).
    let tok = 42usize;
    let (cpu_dec, _) = cpu.forward_at(&[tok], &mut cpu_caches, 300);
    let (gpu_dec, _) = gpu.forward(&[tok], &mut gpu_caches, None);
    let decode_diff = max_abs_diff(&cpu_dec, &gpu_dec);

    let tol = 1e-3f32; // A-f32q accumulation-order class through 4 layers
    println!(
        "SERVING_GOLDEN_TIGHT layers={} window={} prefill_T=300 decode_T=1 prefill_diff={prefill_diff:e} decode_diff={decode_diff:e} tol={tol:e}",
        cfg.num_hidden_layers, cfg.sliding_window
    );
    let ok = prefill_diff < tol && decode_diff < tol
        && prefill_diff.is_finite() && decode_diff.is_finite();
    if ok {
        println!("RESULT: PASS tightened serving golden (FP8-KV CPU twin, T=300 + decode)");
    } else {
        eprintln!("RESULT: FAIL prefill_diff={prefill_diff:e} decode_diff={decode_diff:e}");
        std::process::exit(1);
    }
}
