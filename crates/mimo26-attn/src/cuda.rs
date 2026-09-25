//! Minimal CUDA runtime bindings for the attention serving path's device-memory
//! layer (mirrors `mimo26-spark::cuda`; the CUDA runtime API is a standard
//! surface, not project logic — the two copies are the same `cudart` symbols).
//!
//! Only the calls the serving binary needs: allocation, H2D/D2H copy, memset,
//! stream create/sync/destroy, and error-string readback. Declarations only —
//! they resolve against `libcudart` (linked by `build.rs` under the `cuda`
//! feature). On the CPU merge gate these symbols are never referenced, so no
//! nvcc or cudart is required there.

use core::ffi::{c_char, c_int, c_void};

use crate::ffi::{CudaError, CudaStream};

/// `cudaSuccess`.
pub const SUCCESS: CudaError = 0;
/// `cudaMemcpyHostToDevice`.
pub const H2D: c_int = 1;
/// `cudaMemcpyDeviceToHost`.
pub const D2H: c_int = 2;
/// `cudaMemcpyDeviceToDevice`.
pub const D2D: c_int = 3;

/// `cudaEvent_t` (opaque runtime handle).
pub type CudaEvent = *mut c_void;

unsafe extern "C" {
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> CudaError;
    pub fn cudaFree(ptr: *mut c_void) -> CudaError;
    /// Page-lock an existing host range for fast async copies (perf reset R2:
    /// the RDMA receive rings the return planes land in).
    pub fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: u32) -> CudaError;
    pub fn cudaHostUnregister(ptr: *mut c_void) -> CudaError;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> CudaError;
    pub fn cudaMemcpy2D(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: c_int,
    ) -> CudaError;
    pub fn cudaMemset(ptr: *mut c_void, value: c_int, count: usize) -> CudaError;
    pub fn cudaDeviceSynchronize() -> CudaError;
    pub fn cudaStreamCreate(stream: *mut CudaStream) -> CudaError;
    pub fn cudaStreamSynchronize(stream: CudaStream) -> CudaError;
    pub fn cudaStreamDestroy(stream: CudaStream) -> CudaError;
    pub fn cudaEventCreate(event: *mut CudaEvent) -> CudaError;
    pub fn cudaEventDestroy(event: CudaEvent) -> CudaError;
    pub fn cudaEventRecord(event: CudaEvent, stream: CudaStream) -> CudaError;
    pub fn cudaEventSynchronize(event: CudaEvent) -> CudaError;
    pub fn cudaEventElapsedTime(ms: *mut f32, start: CudaEvent, end: CudaEvent) -> CudaError;
    pub fn cudaGetLastError() -> CudaError;
    pub fn cudaGetErrorString(err: CudaError) -> *const c_char;
}

/// The runtime's description of a CUDA error, or a fallback string on a null.
pub fn error_string(err: CudaError) -> String {
    if err == SUCCESS {
        return "cudaSuccess".to_string();
    }
    // SAFETY: cudaGetErrorString returns a pointer to a static NUL-terminated
    // string owned by the runtime; it is only read, never freed.
    let p = unsafe { cudaGetErrorString(err) };
    if p.is_null() {
        return format!("cuda error {err}");
    }
    // SAFETY: p points to a static NUL-terminated C string.
    let bytes = unsafe { core::ffi::CStr::from_ptr(p) }.to_bytes();
    String::from_utf8_lossy(bytes).into_owned()
}
