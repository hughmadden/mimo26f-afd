//! Dense-path line probe (builder step 4): time the cuBLAS SGEMM in pedantic
//! (TF32-off) vs default (TF32) mode for the prefill shapes, to locate the
//! prefill-throughput bottleneck. Run on the coordinator:
//! `cargo run --features cuda --example dense_bench`.

use mimo26_coordinator::cublas::Handle;
use mimo26_attn::device::DeviceBuffer;

fn f32_bytes(x: &[f32]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) }
}

fn time_sgemm(m: usize, n: usize, k: usize, iters: usize) {
    let a = vec![0.1f32; m * k];
    let b = vec![0.2f32; n * k];
    let mut c = vec![0.0f32; m * n];
    let da = DeviceBuffer::alloc(a.len() * 4).unwrap();
    let db = DeviceBuffer::alloc(b.len() * 4).unwrap();
    let dc = DeviceBuffer::alloc(c.len() * 4).unwrap();
    da.upload(f32_bytes(&a)).unwrap();
    db.upload(f32_bytes(&b)).unwrap();

    for (label, pedantic) in [("pedantic(TF32-off)", true), ("default(TF32)", false)] {
        // build a handle in the requested mode (pedantic already, or default via re-create)
        let handle = if pedantic {
            Handle::new().unwrap()
        } else {
            // default mode: create + set DEFAULT_MATH (0)
            unsafe {
                let mut h: mimo26_coordinator::cublas::CublasHandle = core::ptr::null_mut();
                mimo26_coordinator::cublas::cublasCreate_v2(&mut h);
                mimo26_coordinator::cublas::cublasSetMathMode(h, 0); // CUBLAS_DEFAULT_MATH
                Handle::from_raw(h)
            }
        };
        // warmup
        for _ in 0..3 {
            unsafe {
                mimo26_coordinator::cublas::sgemm_nt(&handle, da.as_ptr() as *const f32, db.as_ptr() as *const f32, m, n, k, dc.as_ptr() as *mut f32);
            }
        }
        unsafe { mimo26_attn::cuda::cudaDeviceSynchronize(); }
        let start = std::time::Instant::now();
        for _ in 0..iters {
            unsafe {
                mimo26_coordinator::cublas::sgemm_nt(&handle, da.as_ptr() as *const f32, db.as_ptr() as *const f32, m, n, k, dc.as_ptr() as *mut f32);
            }
        }
        unsafe { mimo26_attn::cuda::cudaDeviceSynchronize(); }
        let elapsed = start.elapsed();
        let per = elapsed.as_secs_f64() / iters as f64;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        println!("SGEMM {label} m={m} n={n} k={k} per_ms={:.3} tflops={:.2}",
            per * 1e3, flops / per / 1e12);
    }
}

fn main() {
    // QKV (GA): [4027, 13568] x [13568, 4096]
    time_sgemm(4027, 13568, 4096, 5);
    // lm_head: [4027, 152576] x [152576, 4096]
    time_sgemm(4027, 152576, 4096, 3);
    // o_proj: [4027, 4096] x [4096, 8192]
    time_sgemm(4027, 8192, 4096, 5);
}
