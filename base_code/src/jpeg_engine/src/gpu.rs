/// GPU acceleration — trait-based interface for GPU kernels.

use half::f16;

pub trait GpuKernel: Send + Sync {
    fn batch_idct_2d(&self, blocks: &mut [[f16; 64]]) -> Result<(), GpuError>;
    fn batch_dct_2d(&self, blocks: &mut [[f16; 64]]) -> Result<(), GpuError>;
    fn batch_ycbcr_to_rgb(
        &self, y: &[f16], cb: &[f16], cr: &[f16],
        r: &mut [u8], g: &mut [u8], b: &mut [u8],
    ) -> Result<(), GpuError>;
    fn device_name(&self) -> &str;
}

#[derive(Debug)]
pub enum GpuError {
    NotAvailable,
    MemoryError,
    KernelError(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NotAvailable => write!(f, "GPU not available"),
            GpuError::MemoryError => write!(f, "GPU memory error"),
            GpuError::KernelError(msg) => write!(f, "GPU kernel error: {msg}"),
        }
    }
}

impl std::error::Error for GpuError {}

pub struct CpuKernel;

impl GpuKernel for CpuKernel {
    fn batch_idct_2d(&self, blocks: &mut [[f16; 64]]) -> Result<(), GpuError> {
        let len = blocks.len();
        if len == 0 { return Ok(()); }
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            blocks.par_iter_mut().for_each(|block| {
                crate::idct::idct_2d(block);
            });
        }
        #[cfg(not(feature = "rayon"))]
        {
            let ptr = blocks.as_mut_ptr();
            for i in 0..len {
                unsafe { crate::idct::idct_2d(&mut *ptr.add(i)); }
            }
        }
        Ok(())
    }

    fn batch_dct_2d(&self, blocks: &mut [[f16; 64]]) -> Result<(), GpuError> {
        let len = blocks.len();
        if len == 0 { return Ok(()); }
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            blocks.par_iter_mut().for_each(|block| {
                crate::dct::fdct_2d(block);
            });
        }
        #[cfg(not(feature = "rayon"))]
        {
            let ptr = blocks.as_mut_ptr();
            for i in 0..len {
                unsafe { crate::dct::fdct_2d(&mut *ptr.add(i)); }
            }
        }
        Ok(())
    }

    fn batch_ycbcr_to_rgb(
        &self, y: &[f16], cb: &[f16], cr: &[f16],
        r: &mut [u8], g: &mut [u8], b: &mut [u8],
    ) -> Result<(), GpuError> {
        let n = y.len();
        let mut i = 0;
        while i + 8 <= n {
            let (r8, g8, b8) = crate::scaling::ycbcr_to_rgb_8(
                &y[i..i+8], &cb[i..i+8], &cr[i..i+8],
            );
            r[i..i+8].copy_from_slice(&r8);
            g[i..i+8].copy_from_slice(&g8);
            b[i..i+8].copy_from_slice(&b8);
            i += 8;
        }
        while i < n {
            let (rp, gp, bp) = crate::scaling::ycbcr_to_rgb(y[i], cb[i], cr[i]);
            r[i] = rp; g[i] = gp; b[i] = bp;
            i += 1;
        }
        Ok(())
    }

    fn device_name(&self) -> &str { "CPU" }
}

// ──────────────────────────────────────────────────────
// f32 ↔ f16 helpers (for FFI boundary only)
// ──────────────────────────────────────────────────────

pub fn f32_to_f16_slice(src: &[f32], dst: &mut [f16]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f16::from_f32(*s);
    }
}

pub fn f16_to_f32_slice(src: &[f16], dst: &mut [f32]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = s.to_f32();
    }
}

/// Transmute &[f16] to &[u16] for OpenCL buffer I/O.
/// half::f16 is #[repr(transparent)] over u16.
pub fn f16_as_u16_slice(slice: &[f16]) -> &[u16] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u16, slice.len()) }
}

pub fn u16_as_f16_slice_mut(slice: &mut [u16]) -> &mut [f16] {
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut f16, slice.len()) }
}

pub fn f16_as_u16_slice_mut(slice: &mut [f16]) -> &mut [u16] {
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u16, slice.len()) }
}

// ──────────────────────────────────────────────────────
// OpenCL GPU kernel
// ──────────────────────────────────────────────────────

pub fn gpu_available() -> bool {
    #[cfg(feature = "opencl")]
    { opencl_backend::probe_device() }
    #[cfg(not(feature = "opencl"))]
    { false }
}

pub fn create_kernel() -> Box<dyn GpuKernel> {
    #[cfg(feature = "opencl")]
    {
        match opencl_backend::OpenClKernel::new() {
            Ok(k) => {
                eprintln!("[jpeg_engine] Using GPU: {}", k.device_name());
                return Box::new(k);
            }
            Err(e) => {
                eprintln!("[jpeg_engine] GPU unavailable ({}), falling back to CPU", e);
            }
        }
    }
    Box::new(CpuKernel)
}

#[cfg(feature = "opencl")]
mod opencl_backend {
    use super::*;
    use ocl::{ProQue, Buffer, Kernel, MemFlags};

    const IDCT_KERNEL_SRC: &str = r#"
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable

    __constant half IDCT_MAT[64] = {
        0.3535h, 0.4904h, 0.4619h, 0.4157h,
        0.3535h, 0.2778h, 0.1913h, 0.0975h,
        0.3535h, 0.4157h, 0.1913h,-0.0975h,
       -0.3535h,-0.4904h,-0.4619h,-0.2778h,
        0.3535h, 0.2778h,-0.1913h,-0.4904h,
       -0.3535h, 0.0975h, 0.4619h, 0.4157h,
        0.3535h, 0.0975h,-0.4619h,-0.2778h,
        0.3535h, 0.4157h,-0.1913h,-0.4904h,
        0.3535h,-0.0975h,-0.4619h, 0.2778h,
        0.3535h,-0.4157h,-0.1913h, 0.4904h,
        0.3535h,-0.2778h,-0.1913h, 0.4904h,
       -0.3535h,-0.0975h, 0.4619h,-0.4157h,
        0.3535h,-0.4157h, 0.1913h, 0.0975h,
       -0.3535h, 0.4904h,-0.4619h, 0.2778h,
        0.3535h,-0.4904h, 0.4619h,-0.4157h,
        0.3535h,-0.2778h, 0.1913h,-0.0975h
    };

    inline void idct_1d(__constant half* mat, half* row, half* dst) {
        half s0 = row[0], s1 = row[1], s2 = row[2], s3 = row[3];
        half s4 = row[4], s5 = row[5], s6 = row[6], s7 = row[7];
        dst[0] = s0*mat[0]  + s1*mat[1]  + s2*mat[2]  + s3*mat[3]
               + s4*mat[4]  + s5*mat[5]  + s6*mat[6]  + s7*mat[7];
        dst[1] = s0*mat[8]  + s1*mat[9]  + s2*mat[10] + s3*mat[11]
               + s4*mat[12] + s5*mat[13] + s6*mat[14] + s7*mat[15];
        dst[2] = s0*mat[16] + s1*mat[17] + s2*mat[18] + s3*mat[19]
               + s4*mat[20] + s5*mat[21] + s6*mat[22] + s7*mat[23];
        dst[3] = s0*mat[24] + s1*mat[25] + s2*mat[26] + s3*mat[27]
               + s4*mat[28] + s5*mat[29] + s6*mat[30] + s7*mat[31];
        dst[4] = s0*mat[32] + s1*mat[33] + s2*mat[34] + s3*mat[35]
               + s4*mat[36] + s5*mat[37] + s6*mat[38] + s7*mat[39];
        dst[5] = s0*mat[40] + s1*mat[41] + s2*mat[42] + s3*mat[43]
               + s4*mat[44] + s5*mat[45] + s6*mat[46] + s7*mat[47];
        dst[6] = s0*mat[48] + s1*mat[49] + s2*mat[50] + s3*mat[51]
               + s4*mat[52] + s5*mat[53] + s6*mat[54] + s7*mat[55];
        dst[7] = s0*mat[56] + s1*mat[57] + s2*mat[58] + s3*mat[59]
               + s4*mat[60] + s5*mat[61] + s6*mat[62] + s7*mat[63];
    }

    __kernel void batch_idct(__global half* blocks, int num_blocks) {
        int gid = get_global_id(0);
        if (gid >= num_blocks) return;
        __global half* block = blocks + gid * 64;
        half tmp[64];
        for (int y = 0; y < 8; y++) {
            int off = y * 8;
            half row[8];
            row[0] = block[off];   row[1] = block[off+1];
            row[2] = block[off+2]; row[3] = block[off+3];
            row[4] = block[off+4]; row[5] = block[off+5];
            row[6] = block[off+6]; row[7] = block[off+7];
            idct_1d(IDCT_MAT, row, tmp + off);
        }
        for (int x = 0; x < 8; x++) {
            half col[8];
            col[0] = tmp[x];      col[1] = tmp[8+x];
            col[2] = tmp[16+x];   col[3] = tmp[24+x];
            col[4] = tmp[32+x];   col[5] = tmp[40+x];
            col[6] = tmp[48+x];   col[7] = tmp[56+x];
            half d[8];
            idct_1d(IDCT_MAT, col, d);
            block[x]     = d[0];  block[8+x]   = d[1];
            block[16+x]  = d[2];  block[24+x]  = d[3];
            block[32+x]  = d[4];  block[40+x]  = d[5];
            block[48+x]  = d[6];  block[56+x]  = d[7];
        }
    }
    "#;

    pub struct OpenClKernel {
        pro_que: ProQue,
        idct_kernel: Kernel,
        device: String,
        buf: std::sync::Mutex<Option<Buffer<u16>>>,
    }

    unsafe impl Sync for OpenClKernel {}

    impl OpenClKernel {
        pub fn new() -> Result<Self, GpuError> {
            let pro_que = ProQue::builder()
                .src(IDCT_KERNEL_SRC)
                .dims(1)
                .build()
                .map_err(|e| GpuError::KernelError(format!("OpenCL init: {e}")))?;

            let device_name = pro_que.device().name()
                .map_err(|e| GpuError::KernelError(format!("device name: {e}")))?;

            let idct_kernel = ocl::Kernel::builder()
                .program(&pro_que.program())
                .name("batch_idct")
                .queue(pro_que.queue().clone())
                .arg(None::<&ocl::Buffer<u16>>)
                .arg(&0i32)
                .build()
                .map_err(|e| GpuError::KernelError(format!("kernel build: {e}")))?;

            Ok(OpenClKernel { pro_que, idct_kernel, device: device_name, buf: std::sync::Mutex::new(None) })
        }
    }

    impl GpuKernel for OpenClKernel {
        fn batch_idct_2d(&self, blocks: &mut [[f16; 64]]) -> Result<(), GpuError> {
            let n = blocks.len();
            if n == 0 { return Ok(()); }
            let total = n * 64;

            let mut buf_guard = self.buf.lock().unwrap();
            if buf_guard.as_ref().map_or(true, |b| b.len() < total) {
                *buf_guard = Some(Buffer::builder()
                    .queue(self.pro_que.queue().clone())
                    .flags(MemFlags::new().read_write())
                    .len(total)
                    .build()
                    .map_err(|e| GpuError::KernelError(format!("buffer alloc: {e}")))?);
            }
            let buf = buf_guard.as_ref().unwrap();

            // f16 transmutes to u16 — zero copy
            let flat: &[f16] = unsafe {
                std::slice::from_raw_parts(blocks.as_ptr() as *const f16, total)
            };
            buf.write(f16_as_u16_slice(flat))
                .enq().map_err(|e| GpuError::KernelError(format!("write: {e}")))?;

            self.idct_kernel.set_arg(0, buf)
                .map_err(|e| GpuError::KernelError(format!("arg 0: {e}")))?;
            self.idct_kernel.set_arg(1, &(n as i32))
                .map_err(|e| GpuError::KernelError(format!("arg 1: {e}")))?;

            unsafe {
                let local_size = if n < 64 { n as u64 } else { 64 };
                self.idct_kernel.cmd()
                    .global_work_size(n)
                    .local_work_size((local_size,))
                    .enq()
                    .map_err(|e| GpuError::KernelError(format!("enqueue: {e}")))?;
            }

            let mut raw = vec![0u16; total];
            buf.read(&mut raw)
                .enq().map_err(|e| GpuError::KernelError(format!("readback: {e}")))?;
            let flat_mut: &mut [f16] = unsafe {
                std::slice::from_raw_parts_mut(blocks.as_mut_ptr() as *mut f16, total)
            };
            flat_mut.copy_from_slice(u16_as_f16_slice_mut(&mut raw));

            Ok(())
        }

        fn batch_dct_2d(&self, _blocks: &mut [[f16; 64]]) -> Result<(), GpuError> {
            Err(GpuError::NotAvailable)
        }

        fn batch_ycbcr_to_rgb(
            &self, _y: &[f16], _cb: &[f16], _cr: &[f16],
            _r: &mut [u8], _g: &mut [u8], _b: &mut [u8],
        ) -> Result<(), GpuError> {
            Err(GpuError::NotAvailable)
        }

        fn device_name(&self) -> &str { &self.device }
    }

    pub fn probe_device() -> bool {
        OpenClKernel::new().is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_kernel_batch_idct() {
        let kernel = CpuKernel;
        let mut blocks = vec![[f16::ZERO; 64]; 10];
        blocks[0][0] = f16::from_f32(8.0);
        kernel.batch_idct_2d(&mut blocks).unwrap();
        let diff = (blocks[0][0].to_f32() - 1.0).abs();
        assert!(diff < 0.01, "expected ~1.0, got {}", blocks[0][0]);
    }

    #[test]
    fn test_create_kernel_fallback() {
        let kernel = create_kernel();
        assert!(!kernel.device_name().is_empty());
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_opencl_kernel_creation() {
        match opencl_backend::OpenClKernel::new() {
            Ok(k) => println!("OpenCL device: {}", k.device_name()),
            Err(e) => println!("OpenCL not available: {e}"),
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_fp16_idct_correctness() {
        let n = 10usize;
        let mut blocks: Vec<[f16; 64]> = (0..n).map(|i| {
            let mut b = [f16::ZERO; 64];
            b[0] = f16::from_f32((i as f32 + 1.0) * 16.0);
            b
        }).collect();

        // CPU reference
        let mut expected = blocks.clone();
        CpuKernel.batch_idct_2d(&mut expected).unwrap();

        // GPU FP16
        let gpu = opencl_backend::OpenClKernel::new().expect("GPU init");
        gpu.batch_idct_2d(&mut blocks).unwrap();

        let epsilon = f16::from_f32(1.0);
        for i in 0..n {
            for j in 0..64 {
                let diff = if blocks[i][j] > expected[i][j] {
                    blocks[i][j] - expected[i][j]
                } else {
                    expected[i][j] - blocks[i][j]
                };
                assert!(diff < epsilon,
                    "FP16 mismatch block {i} coeff {j}: GPU={:.3?} CPU={:.3?}",
                    blocks[i][j], expected[i][j]);
            }
        }
        println!("FP16 IDCT: all {n} blocks passed");
    }
}
