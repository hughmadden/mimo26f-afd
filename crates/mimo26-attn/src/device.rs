//! A minimal safe device-memory buffer: `cudaMalloc`/`cudaFree` with RAII
//! (mirrors `mimo26-spark::device`). A buffer owns exactly one `cudaMalloc`
//! allocation and frees it once on drop. It is `Send`/`Sync` — a device
//! allocation is addressable from any host thread in the owning process, and the
//! single-owner free is unchanged.

use core::ffi::c_void;

use crate::cuda;
use crate::ffi::CudaError;

/// One device allocation, freed on drop.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

impl DeviceBuffer {
    /// Allocate `bytes` on the current device.
    pub fn alloc(bytes: usize) -> Result<Self, CudaError> {
        if bytes == 0 {
            return Ok(Self { ptr: core::ptr::null_mut(), bytes: 0 });
        }
        let mut ptr: *mut c_void = core::ptr::null_mut();
        // SAFETY: &mut ptr is a valid out-pointer; cudaMalloc writes the device
        // pointer into it.
        let rc = unsafe { cuda::cudaMalloc(&mut ptr, bytes) };
        if rc != cuda::SUCCESS {
            return Err(rc);
        }
        Ok(Self { ptr, bytes })
    }

    /// The device pointer (for passing to the kernels).
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Byte length.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Copy `src` (exactly `bytes()` long) to the device.
    pub fn upload(&self, src: &[u8]) -> Result<(), CudaError> {
        assert_eq!(src.len(), self.bytes, "upload size mismatch");
        if self.bytes == 0 {
            return Ok(());
        }
        // SAFETY: self.ptr is a valid cudaMalloc allocation of self.bytes; src
        // is a host slice of the same length; H2D copy is within bounds.
        let rc =
            unsafe { cuda::cudaMemcpy(self.ptr, src.as_ptr() as *const c_void, src.len(), cuda::H2D) };
        if rc != cuda::SUCCESS {
            return Err(rc);
        }
        Ok(())
    }

    /// Copy `bytes()` bytes from the device into `dst`.
    pub fn download(&self, dst: &mut [u8]) -> Result<(), CudaError> {
        assert_eq!(dst.len(), self.bytes, "download size mismatch");
        if self.bytes == 0 {
            return Ok(());
        }
        // SAFETY: self.ptr is a valid cudaMalloc allocation of self.bytes; dst
        // is a host slice of the same length; D2H copy is within bounds.
        let rc =
            unsafe { cuda::cudaMemcpy(dst.as_mut_ptr() as *mut c_void, self.ptr, dst.len(), cuda::D2H) };
        if rc != cuda::SUCCESS {
            return Err(rc);
        }
        Ok(())
    }

    /// Copy the first `src.len()` (≤ `bytes()`) host bytes into the device buffer
    /// (for a reused, right-sized scratch buffer). The trailing device bytes are
    /// left untouched.
    pub fn upload_prefix(&self, src: &[u8]) -> Result<(), CudaError> {
        assert!(src.len() <= self.bytes, "upload_prefix size overflow");
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: self.ptr is a valid cudaMalloc allocation of self.bytes ≥ src.len().
        let rc = unsafe { cuda::cudaMemcpy(self.ptr, src.as_ptr() as *const c_void, src.len(), cuda::H2D) };
        if rc != cuda::SUCCESS {
            return Err(rc);
        }
        Ok(())
    }

    /// Copy the first `dst.len()` (≤ `bytes()`) device bytes into the host slice.
    pub fn download_prefix(&self, dst: &mut [u8]) -> Result<(), CudaError> {
        assert!(dst.len() <= self.bytes, "download_prefix size overflow");
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: self.ptr is a valid cudaMalloc allocation of self.bytes ≥ dst.len().
        let rc =
            unsafe { cuda::cudaMemcpy(dst.as_mut_ptr() as *mut c_void, self.ptr, dst.len(), cuda::D2H) };
        if rc != cuda::SUCCESS {
            return Err(rc);
        }
        Ok(())
    }

    /// Zero the buffer on the device.
    pub fn zero(&self) -> Result<(), CudaError> {
        if self.bytes == 0 {
            return Ok(());
        }
        // SAFETY: self.ptr is a valid cudaMalloc allocation of self.bytes.
        let rc = unsafe { cuda::cudaMemset(self.ptr, 0, self.bytes) };
        if rc != cuda::SUCCESS {
            return Err(rc);
        }
        Ok(())
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr was allocated by cudaMalloc and is owned exclusively
            // by this buffer; freeing exactly once at drop.
            unsafe { cuda::cudaFree(self.ptr) };
        }
    }
}

// SAFETY: a cudaMalloc allocation is addressable from any host thread in the
// owning process; single-owner free on drop is unchanged.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}
