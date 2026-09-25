//! Rust FFI surface for the layout-v2 expert CUDA ABI.
//!
//! Mirrors `crates/mimo26-expert/kernels/include/mimo26_expert_kernels.h` (and
//! the constants in `mimo26_slice_layout.h`). No CUDA header is imported:
//! `cudaError_t` is an `i32` (`cudaSuccess == 0`) and `cudaStream_t` is an
//! opaque pointer, so these declarations type-check and link on the CPU merge
//! gate with **no nvcc**. The CUDA symbols are only resolved when the serving
//! binary links the compiled kernel object (a separate feature-gated build).
//!
//! # Safety contract (caller-owned, mirroring the header comments)
//!
//! * Every pointer in [`M26xPlan`] and every buffer argument must stay valid
//!   and immutable until all work on the stream completes.
//! * `fault` must be zero-initialised before launch and checked after; a
//!   non-zero fault means malformed metadata was detected and **no weights were
//!   read** — it must fail loud, never be ignored.
//! * Padded groups have no metadata entry and must return before any load.
//! * Buffer sizes are **bytes**, never dtype-dependent element counts.
//! * `m26x_check_aot` must be called against the resident manifest arch/SM/
//!   capacity before any FFN launch (the AOT bake refuses the wrong SM).
//!
//! These are declarations, not safe wrappers: safe wrappers that enforce the
//! contract land with the daemon.

use core::ffi::{c_int, c_void};

/// `cudaError_t` — an enum; `cudaSuccess == 0`.
pub type CudaError = c_int;

/// `cudaStream_t` (`CUstream`) — an opaque pointer on every supported build.
pub type CudaStream = *mut c_void;

/// Projection selectors (`M26X_PROJ_GATE/UP/DOWN`).
pub const PROJ_GATE: c_int = 0;
pub const PROJ_UP: c_int = 1;
pub const PROJ_DOWN: c_int = 2;

/// Output dtypes (`M26X_OUT_F32` / `M26X_OUT_BF16`).
pub const OUT_F32: c_int = 0;
pub const OUT_BF16: c_int = 1;

/// `m26x_plan` — layout + capacity + grouping metadata.
///
/// Field order and types are the C ABI (`mimo26_expert_kernels.h`); the layout
/// test below pins `size_of == 96` and every field offset, so a drift from the
/// header is a test failure.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct M26xPlan {
    pub layout_version: u32,
    pub manifest_arch: c_int,
    pub manifest_sms: c_int,
    pub capacity_class: c_int,
    pub resident_experts: c_int,
    pub n_groups: c_int,
    pub padded_groups: c_int,
    pub total_tokens: c_int,
    pub max_m: c_int,
    pub grouped_bytes: u64,
    pub x_bytes: u64,
    pub out_bytes: u64,
    pub scratch_bytes: u64,
    pub expert_ids: *const c_int,
    pub group_offsets: *const c_int,
    pub fault: *mut u32,
}

unsafe extern "C" {
    /// Pure host metadata check, no GPU access. Returns zero on success.
    pub fn m26x_validate_host_plan(
        p: *const M26xPlan,
        host_ids: *const c_int,
        host_offsets: *const c_int,
    ) -> c_int;

    /// AOT gate: refuses a device/manifest whose arch/SM/capacity does not match
    /// the bake. `naive` is the trap-bitfield (`0` = correct).
    pub fn m26x_check_aot(arch: c_int, sms: c_int, capacity: c_int, naive: u32) -> CudaError;

    /// Device identity readback (arch, SM count).
    pub fn m26x_device_identity(arch: *mut c_int, sm_count: *mut c_int) -> CudaError;

    /// Grouped GEMM, one projection (`proj`) over every group.
    pub fn m26x_grouped_gemm_v2(
        p: *const M26xPlan,
        grouped: *const u8,
        x: *const f32,
        proj: c_int,
        out_dtype: c_int,
        naive: u32,
        out: *mut c_void,
        stream: CudaStream,
    ) -> CudaError;

    /// Full expert FFN (gate + up + SiLU + down) for one plan. `scratch` holds
    /// TWO `[T,512]` f32 arrays. Route weights are not applied here; rank
    /// partials are summed outside this primitive.
    pub fn m26x_expert_ffn_v2(
        p: *const M26xPlan,
        grouped: *const u8,
        x: *const f32,
        out_dtype: c_int,
        naive: u32,
        scratch: *mut f32,
        out: *mut c_void,
        stream: CudaStream,
    ) -> CudaError;

    /// R18c phase-wise mixed-M decode dispatch (production). Bitwise-identical
    /// to the per-M launch for decode widths 1..8.
    pub fn m26x_expert_ffn_mixed_v2(
        p: *const M26xPlan,
        grouped: *const u8,
        x: *const f32,
        out_dtype: c_int,
        naive: u32,
        scratch: *mut f32,
        out: *mut c_void,
        stream: CudaStream,
    ) -> CudaError;

    /// Device gather: build the padded routed x rows from the token-major hidden
    /// rows (`src`/`dst` are the group-order `row_token`/`row_padded` maps; `x`
    /// is zeroed by the caller so padding rows read as 0).
    pub fn m26x_gather_x(
        hidden: *const f32,
        src: *const c_int,
        dst: *const c_int,
        rows: c_int,
        hidden_dim: c_int,
        x: *mut f32,
        stream: CudaStream,
    ) -> CudaError;

    /// Device-side route reduce: collapse the padded FFN rows into per-token
    /// BF16 return rows (8,192 B/token). `padded`/`weight`/`token_off` are the
    /// token-major (stable-sorted) route arrays from `RoutePlan`.
    pub fn m26x_route_reduce(
        ffn_out: *const f32,
        padded: *const c_int,
        weight: *const f32,
        token_off: *const c_int,
        tokens: c_int,
        hidden: c_int,
        routes: c_int,
        bf16: *mut u16,
        stream: CudaStream,
    ) -> CudaError;

    /// Full-checkpoint unpack, independent of TP slices (checked dimensions).
    pub fn m26x_unpack_matrix(
        payload: *const u8,
        payload_bytes: u64,
        scales: *const u8,
        scale_bytes: u64,
        rows: c_int,
        cols: c_int,
        naive: u32,
        out: *mut f32,
        out_bytes: u64,
        stream: CudaStream,
    ) -> CudaError;
}

/// Device smoke — only compiled under the `cuda` feature (which also links the
/// kernels). Runs on a GPU: verifies the FFI resolves `m26x_device_identity`,
/// that its arch/SM readback matches the bake the build.rs compiled for, and
/// that `m26x_check_aot` accepts the matching device while refusing a wrong SM.
#[cfg(all(test, feature = "cuda"))]
mod cuda_smoke {
    use super::*;
    use std::env;

    fn baked() -> (i32, i32, i32) {
        let arch: i32 = env::var("MIMO26F_BAKED_ARCH").unwrap_or_else(|_| "89".into()).parse().unwrap();
        let sms: i32 = env::var("MIMO26F_BAKED_SMS").unwrap_or_else(|_| "128".into()).parse().unwrap();
        let capacity: i32 = env::var("MIMO26F_CAPACITY_CLASS").unwrap_or_else(|_| "2048".into()).parse().unwrap();
        (arch, sms, capacity)
    }

    #[test]
    fn device_identity_matches_bake() {
        let (want_arch, want_sms, capacity) = baked();
        let mut arch = 0i32;
        let mut sms = 0i32;
        // SAFETY: arch/sms are valid out-pointers; the kernel writes them.
        let rc = unsafe { m26x_device_identity(&mut arch, &mut sms) };
        assert_eq!(rc, 0, "m26x_device_identity failed (rc {rc})");
        assert_eq!(arch, want_arch, "device arch {arch} != bake {want_arch}");
        assert_eq!(sms, want_sms, "device SMs {sms} != bake {want_sms}");

        // AOT gate accepts the matching device...
        let ok = unsafe { m26x_check_aot(want_arch, want_sms, capacity, 0) };
        assert_eq!(ok, 0, "AOT gate refused the matching device");
        // ...and refuses a wrong arch.
        let bad = unsafe { m26x_check_aot(want_arch + 1, want_sms, capacity, 0) };
        assert_ne!(bad, 0, "AOT gate accepted a wrong arch");
    }

    #[test]
    fn mixed_dispatch_resolves() {
        // A zeroed plan fails host validation (no GPU work), which proves the
        // R18c mixed-M entry is linked and its validation runs.
        let plan = M26xPlan {
            layout_version: 0,
            manifest_arch: 0,
            manifest_sms: 0,
            capacity_class: 0,
            resident_experts: 0,
            n_groups: 0,
            padded_groups: 0,
            total_tokens: 0,
            max_m: 0,
            grouped_bytes: 0,
            x_bytes: 0,
            out_bytes: 0,
            scratch_bytes: 0,
            expert_ids: core::ptr::null(),
            group_offsets: core::ptr::null(),
            fault: core::ptr::null_mut(),
        };
        // SAFETY: all pointers are null and the plan is invalid; the call must
        // return an error without dereferencing any buffer.
        let rc = unsafe {
            m26x_expert_ffn_mixed_v2(
                &plan,
                core::ptr::null(),
                core::ptr::null(),
                0,
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        };
        assert_ne!(rc, 0, "mixed dispatch accepted an invalid plan");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};
    use mimo26_expert::slice;

    /// Pin the `m26x_plan` C ABI: size and every field offset.
    #[test]
    fn plan_matches_c_abi() {
        assert_eq!(size_of::<M26xPlan>(), 96);
        assert_eq!(offset_of!(M26xPlan, layout_version), 0);
        assert_eq!(offset_of!(M26xPlan, manifest_arch), 4);
        assert_eq!(offset_of!(M26xPlan, manifest_sms), 8);
        assert_eq!(offset_of!(M26xPlan, capacity_class), 12);
        assert_eq!(offset_of!(M26xPlan, resident_experts), 16);
        assert_eq!(offset_of!(M26xPlan, n_groups), 20);
        assert_eq!(offset_of!(M26xPlan, padded_groups), 24);
        assert_eq!(offset_of!(M26xPlan, total_tokens), 28);
        assert_eq!(offset_of!(M26xPlan, max_m), 32);
        assert_eq!(offset_of!(M26xPlan, grouped_bytes), 40);
        assert_eq!(offset_of!(M26xPlan, x_bytes), 48);
        assert_eq!(offset_of!(M26xPlan, out_bytes), 56);
        assert_eq!(offset_of!(M26xPlan, scratch_bytes), 64);
        assert_eq!(offset_of!(M26xPlan, expert_ids), 72);
        assert_eq!(offset_of!(M26xPlan, group_offsets), 80);
        assert_eq!(offset_of!(M26xPlan, fault), 88);
    }

    /// Pin the ABI constants against the Rust twin of `mimo26_slice_layout.h`
    /// (the sole writer of the layout is `mimo26-repack`; `slice` re-states it).
    #[test]
    fn constants_match_slice_twin() {
        assert_eq!(PROJ_GATE, 0);
        assert_eq!(PROJ_UP, 1);
        assert_eq!(PROJ_DOWN, 2);
        assert_eq!(OUT_F32, 0);
        assert_eq!(OUT_BF16, 1);
        assert_eq!(slice::LAYOUT_VERSION, 2);
        assert_eq!(slice::QUARTER_SLICE_BYTES, 3_342_336);
        assert_eq!(slice::HIDDEN, 4096);
        assert_eq!(slice::INTERMEDIATE, 2048);
    }
}
