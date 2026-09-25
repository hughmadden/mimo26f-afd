//! Feature-gated cuBLAS link for the coordinator's GPU dense path (I5-R8a), plus
//! the device-forward glue kernels (`kernels/glue.cu`, perf reset R1).
//!
//! The CPU merge gate (`cargo test --workspace`, default features) must not
//! require CUDA, so the cuBLAS symbols and the nvcc compile only happen under the
//! `cuda` feature (which also enables `mimo26-attn/cuda` for the attention kernels).
//! `cublasSgemm_v2` resolves against `libcublas` (a system library — the
//! REUSE row is added with the dense-path commit). The bake arch comes from
//! `MIMO26F_CUDA_ARCH` (default sm_89), as in `mimo26-attn/build.rs`.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/glue.cu");
    println!("cargo:rerun-if-changed=kernels/dflash.cu");
    println!("cargo:rerun-if-changed=kernels/lm8.cu");
    println!("cargo:rerun-if-changed=../mimo26-attn/kernels/include/mimo26_attn_device.cuh");
    for key in ["MIMO26F_CUDA_ARCH", "MIMO26F_NVCC"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let nvcc = env::var("MIMO26F_NVCC").unwrap_or_else(|_| "nvcc".into());
    let arch = env::var("MIMO26F_CUDA_ARCH").unwrap_or_else(|_| "sm_89".into());
    let mut objs = Vec::new();
    for name in ["glue", "dflash", "lm8"] {
        let src = manifest.join(format!("kernels/{name}.cu"));
        let obj = out.join(format!("{name}.o"));
        let status = Command::new(&nvcc)
            .args(["-O3", "-std=c++17", "-lineinfo", "--ftz=false"])
            .arg(format!("-arch={arch}"))
            .arg("-Xcompiler=-fPIC")
            .arg(format!("-I{}", manifest.join("../mimo26-attn/kernels/include").display()))
            .arg("-c")
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .status()
            .unwrap_or_else(|e| panic!("failed to run {nvcc}: {e}"));
        assert!(status.success(), "nvcc compile of {} failed", src.display());
        objs.push(obj);
    }
    let lib = out.join("libmimo26_coord_kernels.a");
    let status = Command::new("ar").arg("rcs").arg(&lib).args(&objs).status().expect("failed to run ar");
    assert!(status.success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=mimo26_coord_kernels");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
}
