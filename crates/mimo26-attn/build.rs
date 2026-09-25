//! Feature-gated Rust→CUDA FFI link for the attention kernels.
//!
//! The CPU merge gate (`cargo test --workspace`, default features) must not
//! require nvcc, so the whole compile+link is skipped unless the `cuda` feature
//! is on. When it is, this compiles the six kernel TUs (`attn_decode_splitkv`,
//! `attn_decode_tc` — the TC/pipe/c1/prefill-tc entry points — `attn_reduce`,
//! `attn_prefill_chunk`, `rope`, `kv_cache_fp8`) into a relocatable archive and
//! emits the link directives so the serving binary resolves `m26_attn_*` plus
//! `cudart`.
//!
//! The bake arch comes from `MIMO26F_CUDA_ARCH` (default sm_89, the dev-host 4090 dev
//! proxy); the 5090 sm_120 and GB10 sm_121 bakes are set by the caller.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for key in ["MIMO26F_CUDA_ARCH", "MIMO26F_NVCC", "MIMO26F_CUDA_LIB"] {
        println!("cargo:rerun-if-env-changed={key}");
    }

    if !cfg!(feature = "cuda") {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace = manifest_dir.parent().unwrap().parent().unwrap().to_path_buf();
    let kernels = [
        "attn_decode_splitkv.cu",
        "attn_decode_tc.cu",
        "attn_prefill_fa.cu",
        "attn_reduce.cu",
        "attn_prefill_chunk.cu",
        "rope.cu",
        "kv_cache_fp8.cu",
    ];
    let include = workspace.join("crates/mimo26-attn/kernels/include");
    for k in &kernels {
        println!("cargo:rerun-if-changed={}", workspace.join("crates/mimo26-attn/kernels").join(k).display());
    }
    println!("cargo:rerun-if-changed={}", include.display());

    let nvcc = env::var("MIMO26F_NVCC").unwrap_or_else(|_| "nvcc".into());
    let arch = env::var("MIMO26F_CUDA_ARCH").unwrap_or_else(|_| "sm_89".into());
    let cuda_lib = env::var("MIMO26F_CUDA_LIB").unwrap_or_else(|_| "/usr/local/cuda/lib64".into());

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lib = out.join("libmimo26_attn_kernels.a");

    let mut objs = Vec::new();
    for k in &kernels {
        let src = workspace.join("crates/mimo26-attn/kernels").join(k);
        let obj = out.join(format!("{}.o", src.file_stem().unwrap().to_string_lossy()));
        let status = Command::new(&nvcc)
            .arg("-O3")
            .arg("-std=c++17")
            .arg("-lineinfo")
            .arg("--ftz=false")
            .arg(format!("-arch={arch}"))
            .arg("-Xcompiler=-fPIC")
            .arg(format!("-I{}", include.display()))
            .arg("-c")
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .status()
            .unwrap_or_else(|e| panic!("failed to run {nvcc}: {e}"));
        assert!(status.success(), "nvcc compile of {} failed", src.display());
        objs.push(obj);
    }

    let status = Command::new("ar").arg("rcs").arg(&lib).args(&objs).status().expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=mimo26_attn_kernels");
    println!("cargo:rustc-link-search=native={cuda_lib}");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
