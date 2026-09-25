//! Golden check: the GPU dense path (`sgemm_nt`) vs the CPU `linear` twin.
//! Run on the coordinator (sm_120): `cargo run --features cuda --example dense_golden`.
//! Not part of the CPU merge gate (requires `--features cuda` + a GPU).

use mimo26_coordinator::gpu_dense::DenseDevice;
use mimo26_coordinator::linear;

fn main() {
    let mut dev = DenseDevice::new().expect("DenseDevice::new");
    // Several shapes, including the real dims (hid 4096, vocab 152576, inter 16384,
    // QKV 2048+... ) are covered by the small shapes; here we pin the formula.
    for &(m, k, n) in &[(7usize, 13usize, 5usize), (1usize, 64usize, 64usize), (4usize, 128usize, 192usize)] {
        let mut a = vec![0.0f32; m * k];
        let mut w = vec![0.0f32; n * k];
        for i in 0..m * k {
            a[i] = ((i as f32) * 0.37).sin() * 1.5;
        }
        for i in 0..n * k {
            w[i] = ((i as f32) * 0.13).cos() * 2.0;
        }
        dev.upload("w", &w, n, k).expect("upload");
        let gpu = dev.linear(&a, "w", m).expect("gpu linear");
        let cpu = linear(&a, &w, k, n);
        assert_eq!(gpu.len(), cpu.len());
        let mut max_abs = 0.0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            max_abs = max_abs.max((g - c).abs());
        }
        // FP32 SGEMM (TF32 off) vs a scalar FP32 dot: equal up to the different
        // accumulation order (the phase-contract "accumulation order only" class).
        let tol = 1e-4f32;
        if max_abs >= tol {
            eprintln!("DENSE_GOLDEN FAIL m={m} k={k} n={n} max_abs_diff={max_abs:e}");
            std::process::exit(1);
        }
        println!("DENSE_GOLDEN m={m} k={k} n={n} max_abs_diff={max_abs:e} PASS");
    }
    println!("RESULT: PASS dense SGEMM golden (TF32 off)");
}
