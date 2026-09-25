//! GPU dense path (I5-R8a): FP32-resident weights + cuBLAS SGEMM (TF32 off).
//! The dense layers — fused QKV, o_proj, dense layer 0, router gate, lm_head —
//! run on the 5090 via `cublasSgemm_v2` in pedantic (IEEE FP32) math; the CPU
//! `forward::Model` stays the golden bisect reference. Only compiled under the
//! `cuda` feature (the CPU merge gate never references these symbols).

use std::collections::HashMap;
use std::sync::Mutex;

use mimo26_attn::cuda;
use mimo26_attn::device::DeviceBuffer;

use crate::cublas::{self, Handle};

fn f32_bytes(x: &[f32]) -> &[u8] {
    // SAFETY: a &[f32] is a valid &[u8] of 4x the length (little-endian host).
    unsafe { core::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) }
}

fn f32_bytes_mut(x: &mut [f32]) -> &mut [u8] {
    // SAFETY: a &mut [f32] is a valid &mut [u8] of 4x the length.
    unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut u8, x.len() * 4) }
}

/// Per-call scratch (input `dx` + output `dc`) reused across `linear` calls so a
/// prefill does not cudaMalloc on every layer (the I5-R15 dense step: the
/// per-request allocs were the Spark's ~360 ms prefill cost; same class here).
struct Scratch {
    dx: DeviceBuffer,
    dx_cap: usize,
    dc: DeviceBuffer,
    dc_cap: usize,
}

impl Scratch {
    fn new() -> Result<Self, String> {
        Ok(Self {
            dx: DeviceBuffer::alloc(4).map_err(cuda::error_string)?,
            dx_cap: 4,
            dc: DeviceBuffer::alloc(4).map_err(cuda::error_string)?,
            dc_cap: 4,
        })
    }
}

/// FP32-resident dense weights on the device, keyed by the oracle `w_name`.
pub struct DenseDevice {
    handle: Handle,
    /// Tensor-core handle (default math) for the BF16 weights (perf reset R1b).
    handle_bf16: Handle,
    weights: HashMap<String, DeviceBuffer>,
    shapes: HashMap<String, (usize, usize)>, // (out, in)
    /// Weights stored as BF16 (RNE from the FP32 dequant) instead of FP32.
    bf16: std::collections::HashSet<String>,
    scratch: Mutex<Scratch>,
}

/// FP32 -> BF16 round-to-nearest-even (finite inputs; the checkpoint dequant
/// never produces NaN/Inf).
pub fn f32_to_bf16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

impl DenseDevice {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            handle: Handle::new()?,
            handle_bf16: Handle::new_with_math(cublas::DEFAULT_MATH)?,
            weights: HashMap::new(),
            shapes: HashMap::new(),
            bf16: std::collections::HashSet::new(),
            scratch: Mutex::new(Scratch::new()?),
        })
    }

    /// Upload one weight `[out, in]` row-major f32 (stored as-is; `sgemm_nt`
    /// applies the transpose inside the GEMM).
    pub fn upload(&mut self, name: &str, w: &[f32], out: usize, inp: usize) -> Result<(), String> {
        if w.len() != out * inp {
            return Err(format!(
                "dense {name}: {} elems != {out} x {inp}",
                w.len()
            ));
        }
        let buf = DeviceBuffer::alloc(w.len() * 4).map_err(cuda::error_string)?;
        buf.upload(f32_bytes(w)).map_err(cuda::error_string)?;
        self.shapes.insert(name.to_string(), (out, inp));
        self.weights.insert(name.to_string(), buf);
        Ok(())
    }

    /// Upload one weight `[out, in]` as BF16 (RNE on the host, so the device never
    /// holds the FP32 copy). Consumed by [`DenseDevice::gemm_bf16_dev`].
    pub fn upload_bf16(&mut self, name: &str, w: &[f32], out: usize, inp: usize) -> Result<(), String> {
        if w.len() != out * inp {
            return Err(format!("dense {name}: {} elems != {out} x {inp}", w.len()));
        }
        let h: Vec<u16> = w.iter().map(|&x| f32_to_bf16_rne(x)).collect();
        let buf = DeviceBuffer::alloc(h.len() * 2).map_err(cuda::error_string)?;
        // SAFETY: a &[u16] is a valid &[u8] of twice the length.
        let bytes = unsafe { core::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * 2) };
        buf.upload(bytes).map_err(cuda::error_string)?;
        self.shapes.insert(name.to_string(), (out, inp));
        self.weights.insert(name.to_string(), buf);
        self.bf16.insert(name.to_string());
        Ok(())
    }

    /// Upload one weight `[out, in]` already in BF16 (little-endian `u16` bytes, as
    /// the checkpoint stores it), e.g. the DFlash drafter.
    pub fn upload_bf16_raw(&mut self, name: &str, raw: &[u8], out: usize, inp: usize) -> Result<(), String> {
        if raw.len() != out * inp * 2 {
            return Err(format!("dense {name}: {} bytes != {out} x {inp} x 2", raw.len()));
        }
        let buf = DeviceBuffer::alloc(raw.len()).map_err(cuda::error_string)?;
        buf.upload(raw).map_err(cuda::error_string)?;
        self.shapes.insert(name.to_string(), (out, inp));
        self.weights.insert(name.to_string(), buf);
        self.bf16.insert(name.to_string());
        Ok(())
    }

    /// Device pointer of a resident weight (read-only use by custom kernels).
    pub fn weight_ptr(&self, wname: &str) -> Result<*const u8, String> {
        self.weights.get(wname).map(|w| w.as_ptr() as *const u8).ok_or_else(|| format!("dense: unknown weight {wname}"))
    }

    /// Whether a resident weight is stored as BF16.
    pub fn is_bf16(&self, wname: &str) -> bool {
        self.bf16.contains(wname)
    }

    /// `c = x @ w.T` on tensor cores: `x` `[m, in]` BF16, `w` BF16, `c` `[m, out]`
    /// FP32 (FP32 accumulate). Device pointers, no copies, default stream.
    ///
    /// # Safety
    /// `x` and `c` must be valid device pointers to `m*in` BF16 and `m*out` f32.
    pub unsafe fn gemm_bf16_dev(&self, x: *const u16, wname: &str, m: usize, c: *mut f32) -> Result<(), String> {
        let &(out, inp) = self.shapes.get(wname).ok_or_else(|| format!("dense: unknown weight {wname}"))?;
        if !self.bf16.contains(wname) {
            return Err(format!("dense {wname}: not a BF16 weight"));
        }
        let w = self.weights.get(wname).ok_or_else(|| format!("dense: unknown weight {wname}"))?;
        // SAFETY: upheld by the caller for x/c; w is a live resident BF16 weight.
        unsafe { cublas::gemm_bf16_nt(&self.handle_bf16, x, w.as_ptr() as *const u16, m, out, inp, c) };
        Ok(())
    }

    /// `(out, in)` of a resident weight.
    pub fn shape(&self, wname: &str) -> Result<(usize, usize), String> {
        self.shapes
            .get(wname)
            .copied()
            .ok_or_else(|| format!("dense: unknown weight {wname}"))
    }

    /// `c = x @ w.T` entirely on the device (perf reset R1): `x` is `[m, in]`
    /// f32 and `c` `[m, out]` f32, both caller-owned device memory. No copies,
    /// no allocation; ordered on the default stream like every other launch.
    ///
    /// # Safety
    /// `x` and `c` must be valid device pointers to `m*in` and `m*out` f32.
    pub unsafe fn gemm_dev(&self, x: *const f32, wname: &str, m: usize, c: *mut f32) -> Result<(), String> {
        let &(out, inp) = self
            .shapes
            .get(wname)
            .ok_or_else(|| format!("dense: unknown weight {wname}"))?;
        let w = self
            .weights
            .get(wname)
            .ok_or_else(|| format!("dense: unknown weight {wname}"))?;
        // SAFETY: upheld by the caller for x/c; w is a live resident weight.
        unsafe { cublas::sgemm_nt(&self.handle, x, w.as_ptr() as *const f32, m, out, inp, c) };
        Ok(())
    }

    /// `linear(x, w) = x @ w.T`: `x` is `[m, inp]` host f32, result `[m, out]`.
    pub fn linear(&self, x: &[f32], wname: &str, m: usize) -> Result<Vec<f32>, String> {
        let &(out, inp) = self
            .shapes
            .get(wname)
            .ok_or_else(|| format!("dense: unknown weight {wname}"))?;
        if x.len() != m * inp {
            return Err(format!(
                "dense {wname}: x {} elems != {m} x {inp}",
                x.len()
            ));
        }
        let w = self
            .weights
            .get(wname)
            .ok_or_else(|| format!("dense: unknown weight {wname}"))?;
        let mut sc = self.scratch.lock().map_err(|_| "dense scratch lock".to_string())?;
        // Grow the scratch buffers to the largest seen shape (no per-call cudaMalloc).
        let need_dx = x.len() * 4;
        if sc.dx_cap < need_dx {
            sc.dx = DeviceBuffer::alloc(need_dx).map_err(cuda::error_string)?;
            sc.dx_cap = need_dx;
        }
        sc.dx.upload_prefix(f32_bytes(x)).map_err(cuda::error_string)?;
        let need_dc = m * out * 4;
        if sc.dc_cap < need_dc {
            sc.dc = DeviceBuffer::alloc(need_dc).map_err(cuda::error_string)?;
            sc.dc_cap = need_dc;
        }
        // SAFETY: sc.dx, w, sc.dc are valid device allocations of the asserted
        // sizes; the handle is live and every buffer outlives the synchronous call.
        unsafe {
            cublas::sgemm_nt(
                &self.handle,
                sc.dx.as_ptr() as *const f32,
                w.as_ptr() as *const f32,
                m,
                out,
                inp,
                sc.dc.as_ptr() as *mut f32,
            );
        }
        let mut outv = vec![0.0f32; m * out];
        sc.dc.download_prefix(f32_bytes_mut(&mut outv))
            .map_err(cuda::error_string)?;
        Ok(outv)
    }

    /// `linear_from_device(dx, w) = dx @ w.T`: `dx` is already on the device
    /// (`[m, inp]` f32). Skips the input upload (Q/O-on-device, I5-R17 step 2);
    /// only the `[m, out]` result is downloaded.
    pub fn linear_from_device(
        &self,
        dx: &DeviceBuffer,
        wname: &str,
        m: usize,
    ) -> Result<Vec<f32>, String> {
        let &(out, inp) = self
            .shapes
            .get(wname)
            .ok_or_else(|| format!("dense: unknown weight {wname}"))?;
        if dx.bytes() < m * inp * 4 {
            return Err(format!(
                "dense {wname}: device x {} bytes < {m} x {inp} x 4",
                dx.bytes()
            ));
        }
        let w = self
            .weights
            .get(wname)
            .ok_or_else(|| format!("dense: unknown weight {wname}"))?;
        let mut sc = self.scratch.lock().map_err(|_| "dense scratch lock".to_string())?;
        let need_dc = m * out * 4;
        if sc.dc_cap < need_dc {
            sc.dc = DeviceBuffer::alloc(need_dc).map_err(cuda::error_string)?;
            sc.dc_cap = need_dc;
        }
        // SAFETY: dx (caller), w, sc.dc are valid device allocations of the
        // asserted sizes; the handle is live and every buffer outlives the call.
        unsafe {
            cublas::sgemm_nt(
                &self.handle,
                dx.as_ptr() as *const f32,
                w.as_ptr() as *const f32,
                m,
                out,
                inp,
                sc.dc.as_ptr() as *mut f32,
            );
        }
        let mut outv = vec![0.0f32; m * out];
        sc.dc.download_prefix(f32_bytes_mut(&mut outv))
            .map_err(cuda::error_string)?;
        Ok(outv)
    }

    /// `linear_to_device(x, w) = x @ w.T`: like [`linear`] but returns the device
    /// output (a fresh allocation) instead of downloading it — the Q-on-device
    /// device-residency path (I5-R17).
    pub fn linear_to_device(&self, x: &[f32], wname: &str, m: usize) -> Result<DeviceBuffer, String> {
        let &(out, inp) = self
            .shapes
            .get(wname)
            .ok_or_else(|| format!("dense: unknown weight {wname}"))?;
        if x.len() != m * inp {
            return Err(format!(
                "dense {wname}: x {} elems != {m} x {inp}",
                x.len()
            ));
        }
        let w = self
            .weights
            .get(wname)
            .ok_or_else(|| format!("dense: unknown weight {wname}"))?;
        let mut sc = self.scratch.lock().map_err(|_| "dense scratch lock".to_string())?;
        let need_dx = x.len() * 4;
        if sc.dx_cap < need_dx {
            sc.dx = DeviceBuffer::alloc(need_dx).map_err(cuda::error_string)?;
            sc.dx_cap = need_dx;
        }
        sc.dx.upload_prefix(f32_bytes(x)).map_err(cuda::error_string)?;
        let d_out = DeviceBuffer::alloc(m * out * 4).map_err(cuda::error_string)?;
        // SAFETY: sc.dx, w, d_out are valid device allocations of the asserted
        // sizes; the handle is live and every buffer outlives the synchronous call.
        unsafe {
            cublas::sgemm_nt(
                &self.handle,
                sc.dx.as_ptr() as *const f32,
                w.as_ptr() as *const f32,
                m,
                out,
                inp,
                d_out.as_ptr() as *mut f32,
            );
        }
        Ok(d_out)
    }
}
