//! Feature-gated Rust→CUDA FFI link for the expert kernels.
//!
//! The CPU merge gate (`cargo test --workspace`, default features) must not
//! require nvcc, so the whole compile+link is skipped unless the `cuda` feature
//! is on. When it is, this script compiles `expert_gemm.cu` (the frozen layout-v2
//! per-M FFN) into a relocatable object, archives it, and emits the link
//! directives so the serving binary resolves `m26x_*` plus `cudart`.
//!
//! Machine-local bake pins come from the environment (defaults are the dev-host 4090
//! dev proxy — sm_89 / baked arch 89 / 128 SMs / class 2048); the Spark
//! sm_121a bake is set by the caller for the remote build.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for key in [
        "MIMO26F_CUDA_ARCH",
        "MIMO26F_BAKED_ARCH",
        "MIMO26F_BAKED_SMS",
        "MIMO26F_CAPACITY_CLASS",
        "MIMO26F_NVCC",
        "MIMO26F_CUDA_LIB",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }

    // Emit the baked arch/SM/capacity as compile-time constants so the runtime
    // manifest identity (read via env!()) always matches what nvcc bakes — no
    // runtime env fallback that could drift to the 4090 defaults.
    let baked_arch = env::var("MIMO26F_BAKED_ARCH").unwrap_or_else(|_| "89".into());
    let baked_sms = env::var("MIMO26F_BAKED_SMS").unwrap_or_else(|_| "128".into());
    let capacity = env::var("MIMO26F_CAPACITY_CLASS").unwrap_or_else(|_| "2048".into());
    println!("cargo:rustc-env=MIMO26F_BAKED_ARCH={baked_arch}");
    println!("cargo:rustc-env=MIMO26F_BAKED_SMS={baked_sms}");
    println!("cargo:rustc-env=MIMO26F_CAPACITY_CLASS={capacity}");

    if !cfg!(feature = "cuda") {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // crates/mimo26-spark -> workspace root.
    let workspace = manifest_dir.parent().unwrap().parent().unwrap().to_path_buf();
    // The frozen per-M FFN plus the R18c mixed-M decode dispatch, plus the
    // device-side route reduce (collapses padded rows -> BF16 return rows).
    let kernels = [
        workspace.join("crates/mimo26-expert/kernels/expert_gemm.cu"),
        workspace.join("crates/mimo26-expert/kernels/expert_gemm_mixed.cu"),
        workspace.join("crates/mimo26-expert/kernels/route_reduce.cu"),
    ];
    let include = workspace.join("crates/mimo26-expert/kernels/include");
    for k in &kernels {
        println!("cargo:rerun-if-changed={}", k.display());
    }
    println!("cargo:rerun-if-changed={}", workspace.join("crates/mimo26-expert/kernels/mixed_dispatch.cuh").display());

    let nvcc = env::var("MIMO26F_NVCC").unwrap_or_else(|_| "nvcc".into());
    let arch = env::var("MIMO26F_CUDA_ARCH").unwrap_or_else(|_| "sm_89".into());
    let cuda_lib = env::var("MIMO26F_CUDA_LIB").unwrap_or_else(|_| "/usr/local/cuda/lib64".into());

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lib = out.join("libmimo26_expert_kernels.a");

    // 1. Compile each kernel TU to a relocatable object (PIC so the Rust PIE
    //    binary can link it; no -rdc needed — all device code is in headers).
    let mut objs = Vec::new();
    for k in &kernels {
        let obj = out.join(format!(
            "{}.o",
            k.file_stem().unwrap().to_string_lossy()
        ));
        let status = Command::new(&nvcc)
            .arg("-O3")
            .arg("-std=c++17")
            .arg("-lineinfo")
            .arg("--ftz=false")
            .arg(format!("-arch={arch}"))
            .arg("-Xcompiler=-fPIC")
            .arg(format!("-DM26X_BAKED_ARCH={baked_arch}"))
            .arg(format!("-DM26X_BAKED_SMS={baked_sms}"))
            .arg(format!("-DM26X_CAPACITY_CLASS={capacity}"))
            .arg("-DM26X_EXACT_M=1")
            .arg(format!("-I{}", include.display()))
            .arg("-c")
            .arg(k)
            .arg("-o")
            .arg(&obj)
            .status()
            .unwrap_or_else(|e| panic!("failed to run {nvcc}: {e}"));
        assert!(status.success(), "nvcc compile of {} failed", k.display());
        objs.push(obj);
    }

    // 1b. The B1 serving path (perf reset R3): kernels/b1_serve.cu (planner +
    //     FC1/quantizer/FC2/reduce chain over the B1 compute core) and the B1
    //     prepare/quantizer/reducer TUs. Object names carry a b1_ prefix so B1's
    //     route_reduce.cu does not collide with the B2 one above.
    let b1_dir = workspace.join("crates/mimo26-expert/kernels/b1");
    let b1_sources = [
        manifest_dir.join("kernels/b1_serve.cu"),
        b1_dir.join("prepare.cu"),
        b1_dir.join("quant_v1.cu"),
        b1_dir.join("route_reduce.cu"),
    ];
    println!("cargo:rerun-if-changed={}", b1_dir.display());
    // The Spark-side kernel headers (b1_fc1_m64.cuh) are included by b1_serve.cu:
    // watch the whole directory, or a header-only edit leaves a stale object.
    println!("cargo:rerun-if-changed={}", manifest_dir.join("kernels").display());
    for k in &b1_sources {
        println!("cargo:rerun-if-changed={}", k.display());
        let obj = out.join(format!("b1_{}.o", k.file_stem().unwrap().to_string_lossy()));
        // The block-scaled MMA needs the arch-specific target: name the virtual
        // (compute_XXXa) and real (sm_XXXa) architectures explicitly, as B1's own
        // build does; a bare -arch=sm_121a emits plain compute_121 PTX on nvcc 13.
        let (virt, real) = match arch.strip_prefix("sm_") {
            Some(n) if n.ends_with('a') => (format!("compute_{n}"), arch.clone()),
            _ => (arch.replace("sm_", "compute_"), arch.clone()),
        };
        let status = Command::new(&nvcc)
            .arg("-O3")
            .arg("-std=c++17")
            .arg("-lineinfo")
            .arg("--ftz=false")
            .arg(format!("--gpu-architecture={virt}"))
            .arg(format!("--gpu-code={real}"))
            .arg("-Xcompiler=-fPIC")
            .arg(format!("-DM26B1_TARGET_ARCH={baked_arch}"))
            .arg(format!("-I{}", b1_dir.display()))
            .arg(format!("-I{}", workspace.join("crates/mimo26-expert/kernels").display()))
            .arg(format!("-I{}", include.display()))
            .arg("-c")
            .arg(k)
            .arg("-o")
            .arg(&obj)
            .status()
            .unwrap_or_else(|e| panic!("failed to run {nvcc}: {e}"));
        assert!(status.success(), "nvcc compile of {} failed", k.display());
        objs.push(obj);
    }

    // 2. Archive the objects into a static library.
    let status = Command::new("ar")
        .arg("rcs")
        .arg(&lib)
        .args(&objs)
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed");

    // 3. Link directives: the kernel archive + the CUDA runtime + the C++ runtime
    //    (the nvcc-compiled .cu objects reference __cxa_guard_acquire and friends,
    //    and Rust links with -nodefaultlibs so libstdc++ must be named explicitly).
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=mimo26_expert_kernels");
    println!("cargo:rustc-link-search=native={cuda_lib}");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
