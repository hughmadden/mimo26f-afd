//! Minimal cuBLAS bindings for the coordinator GPU dense path (I5-R8a: QKV,
//! O, dense layer 0, router gate, lm_head). TF32 is OFF — the handle is created
//! in `CUBLAS_PEDANTIC_MATH` (IEEE FP32 SGEMM, no TF32 tensor cores) so the
//! dense activations stay FP32-semantic (Lattice-v1 §5; the builder's "TF32 off").
//!
//! Declarations only — they resolve against `libcublas` (linked by `build.rs`
//! under the `cuda` feature). The CPU merge gate never references these symbols.

use core::ffi::{c_int, c_void};

/// `CUBLAS_STATUS_SUCCESS`.
pub const SUCCESS: c_int = 0;
/// `CUBLAS_OP_N` (no transpose).
pub const OP_N: c_int = 0;
/// `CUBLAS_OP_T` (transpose).
pub const OP_T: c_int = 1;
/// `CUBLAS_PEDANTIC_MATH` — strict IEEE FP32, TF32 disabled (the "TF32 off" mode).
pub const PEDANTIC_MATH: c_int = 2;
/// `CUBLAS_DEFAULT_MATH`: tensor cores allowed (the BF16 path, perf reset R1b).
pub const DEFAULT_MATH: c_int = 0;
/// `cudaDataType_t` values.
pub const CUDA_R_32F: c_int = 0;
pub const CUDA_R_16BF: c_int = 14;
/// `CUBLAS_COMPUTE_32F`.
pub const COMPUTE_32F: c_int = 68;
/// `CUBLAS_GEMM_DEFAULT`.
pub const GEMM_DEFAULT: c_int = -1;

pub type CublasHandle = *mut c_void;

unsafe extern "C" {
    pub fn cublasCreate_v2(handle: *mut CublasHandle) -> c_int;
    pub fn cublasDestroy_v2(handle: CublasHandle) -> c_int;
    pub fn cublasSetMathMode(handle: CublasHandle, mode: c_int) -> c_int;
    #[allow(non_snake_case, clippy::too_many_arguments)]
    pub fn cublasGemmEx(
        handle: CublasHandle,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: c_int,
        lda: c_int,
        b: *const c_void,
        b_type: c_int,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        c_type: c_int,
        ldc: c_int,
        compute_type: c_int,
        algo: c_int,
    ) -> c_int;
    #[allow(non_snake_case)]
    pub fn cublasSgemm_v2(
        handle: CublasHandle,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: *const f32,
        c: *mut f32,
        ldc: c_int,
    ) -> c_int;
}

/// RAII cuBLAS handle, created in pedantic (TF32-off) math mode.
pub struct Handle {
    raw: CublasHandle,
}

impl Handle {
    pub fn new() -> Result<Self, String> {
        let mut raw: CublasHandle = core::ptr::null_mut();
        // SAFETY: cublasCreate_v2 writes a valid handle into `raw`; the handle
        // is owned by this struct and destroyed on drop.
        let st = unsafe { cublasCreate_v2(&mut raw) };
        if st != SUCCESS {
            return Err(format!("cublasCreate: status {st}"));
        }
        // SAFETY: `raw` is a valid handle; PEDANTIC_MATH disables TF32.
        let st = unsafe { cublasSetMathMode(raw, PEDANTIC_MATH) };
        if st != SUCCESS {
            // SAFETY: `raw` is still a valid handle to release on failure.
            unsafe { cublasDestroy_v2(raw) };
            return Err(format!("cublasSetMathMode: status {st}"));
        }
        Ok(Self { raw })
    }

    /// A handle in `mode` (`DEFAULT_MATH` for the tensor-core BF16 path).
    pub fn new_with_math(mode: c_int) -> Result<Self, String> {
        let mut raw: CublasHandle = core::ptr::null_mut();
        // SAFETY: cublasCreate_v2 writes a valid handle into `raw`.
        let st = unsafe { cublasCreate_v2(&mut raw) };
        if st != SUCCESS {
            return Err(format!("cublasCreate: status {st}"));
        }
        // SAFETY: `raw` is a live handle.
        let st = unsafe { cublasSetMathMode(raw, mode) };
        if st != SUCCESS {
            unsafe { cublasDestroy_v2(raw) };
            return Err(format!("cublasSetMathMode: status {st}"));
        }
        Ok(Self { raw })
    }

    pub fn raw(&self) -> CublasHandle {
        self.raw
    }

    /// Wrap an already-created handle (owns it; destroyed on drop). Used by the
    /// bench to compare math modes against the pedantic default.
    pub fn from_raw(raw: CublasHandle) -> Self {
        Self { raw }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: `raw` is the handle created in `new`, destroyed exactly once.
        unsafe { cublasDestroy_v2(self.raw) };
    }
}

// SAFETY: a cuBLAS handle serializes its own use internally, so it is safe to
// share across threads; ownership (create/destroy) is unchanged and single.
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

/// `C = A @ B^T` in row-major over **device** pointers: `A` is `[m, k]`, `B` is
/// `[n, k]`, `C` is `[m, n]`. This is the `linear(x, w)` shape (`x` `[T, hid]`,
/// weight `w` `[out, hid]`, result `[T, out]`).
///
/// Row-major GEMM through the column-major cuBLAS API: compute
/// `C^T = B @ A^T` by passing `B` as column-major `[k, n]` (transa=OP_T to
/// un-transpose it) and `A` as column-major `[k, m]`; the result lands in the
/// row-major `C` buffer as its column-major `[n, m]` twin. `alpha=1`, `beta=0`.
///
/// # Safety
/// `a`, `b`, `c` must be valid device pointers to `m*k`, `n*k`, `m*n` f32.
pub unsafe fn sgemm_nt(
    handle: &Handle,
    a: *const f32,
    b: *const f32,
    m: usize,
    n: usize,
    k: usize,
    c: *mut f32,
) {
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // SAFETY: upheld by the caller (valid device pointers + live handle).
    let st = unsafe {
        cublasSgemm_v2(
            handle.raw(),
            OP_T,       // B (column-major [k,n]) -> B^T = [n,k] = the weight
            OP_N,       // A (column-major [k,m]) = A^T, used as-is
            n as c_int, // rows of C^T
            m as c_int, // cols of C^T
            k as c_int,
            &alpha,
            b,
            k as c_int, // ldb: column-major [k, n]
            a,
            k as c_int, // lda: column-major [k, m]
            &beta,
            c,
            n as c_int, // ldc: column-major [n, m]
        )
    };
    assert_eq!(st, SUCCESS, "cublasSgemm status {st}");
}

/// `C = A @ B^T` over **device** pointers with BF16 `A` `[m, k]` (activations)
/// and BF16 `B` `[n, k]` (the weight), FP32 accumulate, FP32 `C` `[m, n]` — the
/// row-major `linear` shape through the column-major API, as [`sgemm_nt`].
///
/// # Safety
/// `a`, `b`, `c` must be valid device pointers to `m*k` / `n*k` BF16 and `m*n` f32.
pub unsafe fn gemm_bf16_nt(handle: &Handle, a: *const u16, b: *const u16, m: usize, n: usize, k: usize, c: *mut f32) {
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // SAFETY: upheld by the caller.
    let st = unsafe {
        cublasGemmEx(
            handle.raw(),
            OP_T,
            OP_N,
            n as c_int,
            m as c_int,
            k as c_int,
            &alpha as *const f32 as *const c_void,
            b as *const c_void,
            CUDA_R_16BF,
            k as c_int,
            a as *const c_void,
            CUDA_R_16BF,
            k as c_int,
            &beta as *const f32 as *const c_void,
            c as *mut c_void,
            CUDA_R_32F,
            n as c_int,
            COMPUTE_32F,
            GEMM_DEFAULT,
        )
    };
    assert_eq!(st, SUCCESS, "cublasGemmEx status {st}");
}
