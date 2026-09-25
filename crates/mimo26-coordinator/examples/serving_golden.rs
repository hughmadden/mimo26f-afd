//! Golden check: the serving forward (GPU attention + GPU dense) vs the CPU
//! `forward::Model`, prefill only (T=3, positions 0..2). Run on the coordinator (sm_120):
//! `cargo run --features cuda --example serving_golden`.
//!
//! The GPU attention uses FP8 unit-scale KV while the CPU twin uses f32 KV, so
//! the logits differ by the E4M3 KV quantization (a few percent) — the pass bar
//! is loose enough to absorb that but tight enough to catch a wiring bug (NaN or
//! an O(1) divergence).

use mimo26_coordinator::config::Config;
use mimo26_coordinator::forward::{gen_weights, LayerCache, Model};
use mimo26_coordinator::serving::{Fp8KvCache, ServingModel};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()))
}

fn main() {
    let cfg = Config::tiny();
    let w = gen_weights(&cfg, 42, 0.02);
    let cpu = Model::new(cfg.clone(), w.clone());
    let gpu = ServingModel::new(cfg.clone(), w).expect("ServingModel::new");

    let ids = [5usize, 13, 77];
    let mut cpu_caches: Vec<LayerCache> =
        (0..cfg.num_hidden_layers).map(|l| LayerCache::for_layer(&cfg, l)).collect();
    let mut gpu_caches: Vec<Fp8KvCache> =
        (0..cfg.num_hidden_layers).map(|l| Fp8KvCache::for_layer(&cfg, l)).collect();

    let (cpu_logits, cpu_hidden) = cpu.forward(&ids, &mut cpu_caches);
    let (gpu_logits, gpu_hidden) = gpu.forward(&ids, &mut gpu_caches, None);

    let ld = max_abs_diff(&cpu_logits, &gpu_logits);
    let hd = max_abs_diff(&cpu_hidden, &gpu_hidden);
    let tol = 0.2f32; // FP8 KV quantization class (E4M3 ~6% per value, through layers)
    println!("SERVING_GOLDEN layers={} logits_max_abs={ld:e} hidden_max_abs={hd:e} tol={tol:e}",
        cfg.num_hidden_layers);
    if ld < tol && hd < tol && ld.is_finite() && hd.is_finite() {
        println!("RESULT: PASS serving forward (GPU attention + dense) == CPU Model within FP8-KV quantization");
    } else {
        eprintln!("RESULT: FAIL serving forward diverged from CPU (logits {ld:e} hidden {hd:e})");
        std::process::exit(1);
    }
}
