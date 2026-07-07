/// GPU acceleration — trait-based interface for GPU kernels.
/// Implementations may use OpenCL, Vulkan, or CUDA.
/// CPU fallback always available.

pub trait GpuKernel: Send {
    fn batch_idct_2d(&self, blocks: &mut [[f64; 64]]) -> Result<(), GpuError>;
    fn batch_dct_2d(&self, blocks: &mut [[f64; 64]]) -> Result<(), GpuError>;
    fn batch_ycbcr_to_rgb(
        &self, y: &[f64], cb: &[f64], cr: &[f64],
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
    /// Batch IDCT — uses slice chunking for better cache locality
    /// and less pointer-chasing overhead vs individual block processing.
    fn batch_idct_2d(&self, blocks: &mut [[f64; 64]]) -> Result<(), GpuError> {
        let len = blocks.len();
        if len == 0 {
            return Ok(());
        }
        // Use rayon parallelism for large batches (gated behind cfg feature).
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
                unsafe {
                    crate::idct::idct_2d(&mut *ptr.add(i));
                }
            }
        }
        Ok(())
    }

    fn batch_dct_2d(&self, blocks: &mut [[f64; 64]]) -> Result<(), GpuError> {
        let len = blocks.len();
        if len == 0 {
            return Ok(());
        }
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
                unsafe {
                    crate::dct::fdct_2d(&mut *ptr.add(i));
                }
            }
        }
        Ok(())
    }

    fn batch_ycbcr_to_rgb(
        &self, y: &[f64], cb: &[f64], cr: &[f64],
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
            r[i] = rp;
            g[i] = gp;
            b[i] = bp;
            i += 1;
        }
        Ok(())
    }

    fn device_name(&self) -> &str { "CPU" }
}

// ──────────────────────────────────────────────────────
// OpenCL GPU kernel (behind `opencl` feature)
// ──────────────────────────────────────────────────────

/// Check if a GPU device is available.
pub fn gpu_available() -> bool {
    #[cfg(feature = "opencl")]
    {
        opencl_backend::probe_device()
    }
    #[cfg(not(feature = "opencl"))]
    {
        false
    }
}

/// Create the best available kernel (GPU if possible, CPU fallback).
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

// ──────────────────────────────────────────────────────
// OpenCL backend implementation
// ──────────────────────────────────────────────────────

#[cfg(feature = "opencl")]
mod opencl_backend {
    use super::*;
    use ocl::{ProQue, Buffer, Kernel, MemFlags};

    /// OpenCL IDCT kernel source — fully unrolled 8×8 separable IDCT.
    const IDCT_KERNEL_SRC: &str = r#"
    // Pre-computed IDCT1D_SCALED matrix (0.5 * C(u) * cos((2x+1)uπ/16))
    // Transposed: mat[k] = IDCT1D_SCALED[x][u] where x = k/8, u = k%8
    __constant double IDCT_MAT[64] = {
        0.35355339, 0.49039264, 0.46193977, 0.41573481,
        0.35355339, 0.27778512, 0.19134172, 0.09754516,
        0.35355339, 0.41573481, 0.19134172,-0.09754516,
       -0.35355339,-0.49039264,-0.46193977,-0.27778512,
        0.35355339, 0.27778512,-0.19134172,-0.49039264,
       -0.35355339, 0.09754516, 0.46193977, 0.41573481,
        0.35355339, 0.09754516,-0.46193977,-0.27778512,
        0.35355339, 0.41573481,-0.19134172,-0.49039264,
        0.35355339,-0.09754516,-0.46193977, 0.27778512,
        0.35355339,-0.41573481,-0.19134172, 0.49039264,
        0.35355339,-0.27778512,-0.19134172, 0.49039264,
       -0.35355339,-0.09754516, 0.46193977,-0.41573481,
        0.35355339,-0.41573481, 0.19134172, 0.09754516,
       -0.35355339, 0.49039264,-0.46193977, 0.27778512,
        0.35355339,-0.49039264, 0.46193977,-0.41573481,
        0.35355339,-0.27778512, 0.19134172,-0.09754516
    };

    // 1-D IDCT helper (fully unrolled, scale-fused)
    inline void idct_1d(__constant double* mat, __local double* row, __local double* dst) {
        double s0 = row[0], s1 = row[1], s2 = row[2], s3 = row[3];
        double s4 = row[4], s5 = row[5], s6 = row[6], s7 = row[7];
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

    __kernel void batch_idct(__global double* blocks, int num_blocks) {
        int gid = get_global_id(0);
        if (gid >= num_blocks) return;

        __global double* block = blocks + gid * 64;

        // Local memory for row transform
        __local double tmp[64];

        // Pass 1: IDCT on rows
        for (int y = 0; y < 8; y++) {
            int off = y * 8;
            __local double row[8];
            row[0] = block[off];   row[1] = block[off+1];
            row[2] = block[off+2]; row[3] = block[off+3];
            row[4] = block[off+4]; row[5] = block[off+5];
            row[6] = block[off+6]; row[7] = block[off+7];
            idct_1d(IDCT_MAT, row, tmp + off);
        }

        // Pass 2: IDCT on columns (strided reads)
        for (int x = 0; x < 8; x++) {
            __local double col[8];
            col[0] = tmp[x];      col[1] = tmp[8+x];
            col[2] = tmp[16+x];   col[3] = tmp[24+x];
            col[4] = tmp[32+x];   col[5] = tmp[40+x];
            col[6] = tmp[48+x];   col[7] = tmp[56+x];
            __local double d[8];
            idct_1d(IDCT_MAT, col, d);
            block[x]     = d[0];  block[8+x]   = d[1];
            block[16+x]  = d[2];  block[24+x]  = d[3];
            block[32+x]  = d[4];  block[40+x]  = d[5];
            block[48+x]  = d[6];  block[56+x]  = d[7];
        }
    }

    __kernel void batch_fdct(__global double* blocks, int num_blocks) {
        int gid = get_global_id(0);
        if (gid >= num_blocks) return;
        // For now, just zero them (placeholder — real DCT kernel would go here)
        __global double* block = blocks + gid * 64;
        block[0] = 0.0;
    }
    "#;

    pub struct OpenClKernel {
        pro_que: ProQue,
        idct_kernel: Kernel,
        device: String,
    }

    impl OpenClKernel {
        pub fn new() -> Result<Self, GpuError> {
            let pro_que = ProQue::builder()
                .src(IDCT_KERNEL_SRC)
                .dims(1)
                .build()
                .map_err(|e| GpuError::KernelError(format!("OpenCL init: {e}")))?;

            let device_name = pro_que.device().name()
                .map_err(|e| GpuError::KernelError(format!("device name: {e}")))?;

            let idct_kernel = pro_que.kernel_builder("batch_idct")
                .build()
                .map_err(|e| GpuError::KernelError(format!("kernel build: {e}")))?;

            Ok(OpenClKernel { pro_que, idct_kernel, device: device_name })
        }
    }

    impl GpuKernel for OpenClKernel {
        fn batch_idct_2d(&self, blocks: &mut [[f64; 64]]) -> Result<(), GpuError> {
            let n = blocks.len();
            if n == 0 { return Ok(()); }

            let buf = Buffer::builder()
                .queue(self.pro_que.queue().clone())
                .flags(MemFlags::new().read_write())
                .len(n * 64)
                .copy_host_slice(unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr() as *const f64,
                        n * 64,
                    )
                })
                .build()
                .map_err(|e| GpuError::KernelError(format!("buffer alloc: {e}")))?;

            self.idct_kernel
                .set_arg(0, &buf)
                .map_err(|e| GpuError::KernelError(format!("arg 0: {e}")))?;
            self.idct_kernel
                .set_arg(1, &(n as i32))
                .map_err(|e| GpuError::KernelError(format!("arg 1: {e}")))?;

            unsafe {
                self.idct_kernel.cmd()
                    .global_work_size(n)
                    .enq()
                    .map_err(|e| GpuError::KernelError(format!("enqueue: {e}")))?;
            }

            buf.read(unsafe {
                std::slice::from_raw_parts_mut(
                    blocks.as_mut_ptr() as *mut f64,
                    n * 64,
                )
            }).map_err(|e| GpuError::KernelError(format!("readback: {e}")))?;

            Ok(())
        }

        fn batch_dct_2d(&self, _blocks: &mut [[f64; 64]]) -> Result<(), GpuError> {
            Err(GpuError::NotAvailable)
        }

        fn batch_ycbcr_to_rgb(
            &self, _y: &[f64], _cb: &[f64], _cr: &[f64],
            _r: &mut [u8], _g: &mut [u8], _b: &mut [u8],
        ) -> Result<(), GpuError> {
            Err(GpuError::NotAvailable)
        }

        fn device_name(&self) -> &str {
            &self.device
        }
    }

    /// Probe whether an OpenCL device is available.
    pub fn probe_device() -> bool {
        match OpenClKernel::new() {
            Ok(_) => true,
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_kernel_batch_idct() {
        let kernel = CpuKernel;
        let mut blocks = vec![[0.0f64; 64]; 10];
        blocks[0][0] = 8.0;
        kernel.batch_idct_2d(&mut blocks).unwrap();
        assert!((blocks[0][0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_create_kernel_fallback() {
        let kernel = create_kernel();
        // If GPU is available it might return OpenClKernel, so just check device_name is non-empty
        assert!(!kernel.device_name().is_empty());
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_opencl_kernel_creation() {
        match opencl_backend::OpenClKernel::new() {
            Ok(k) => {
                println!("OpenCL device: {}", k.device_name());
            }
            Err(e) => {
                println!("OpenCL not available: {e}");
            }
        }
    }
}
