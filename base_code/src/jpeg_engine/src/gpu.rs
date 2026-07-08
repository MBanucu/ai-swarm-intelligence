/// GPU acceleration — trait-based interface for GPU kernels.
/// Implementations may use OpenCL, Vulkan, or CUDA.
/// CPU fallback always available.

pub trait GpuKernel: Send + Sync {
    fn batch_idct_2d(&self, blocks: &mut [[f32; 64]]) -> Result<(), GpuError>;
    fn batch_dct_2d(&self, blocks: &mut [[f32; 64]]) -> Result<(), GpuError>;
    fn batch_ycbcr_to_rgb(
        &self, y: &[f32], cb: &[f32], cr: &[f32],
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
    fn batch_idct_2d(&self, blocks: &mut [[f32; 64]]) -> Result<(), GpuError> {
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

    fn batch_dct_2d(&self, blocks: &mut [[f32; 64]]) -> Result<(), GpuError> {
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
        &self, y: &[f32], cb: &[f32], cr: &[f32],
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

    /// OpenCL IDCT kernel source — AAN butterfly 1D IDCT + transposed private store.
    ///
    /// Uses the same even-part butterfly and direct odd-part as the CPU path,
    /// with a transposed private-memory layout (same trick as CPU idct.rs)
    /// to make column-pass reads unit-stride (coalesced).
    const IDCT_KERNEL_SRC: &str = r#"
    // Even-part constants (same as CPU idct.rs)
    __constant float E_A = 0.35355339059327376f;  // 1/(2√2)
    __constant float E_B = 0.46193976625564337f;  // cos(π/8)/2
    __constant float E_C = 0.19134171618254492f;  // cos(3π/8)/2

    // Odd-part constants (same O matrix)
    __constant float O_B = 0.49039264020161522f;  // cos(π/16)/2
    __constant float O_C = 0.41573480615127262f;  // cos(3π/16)/2
    __constant float O_D = 0.27778511650980114f;  // cos(5π/16)/2
    __constant float O_E = 0.09754516100806417f;  // cos(7π/16)/2

    // AAN 1D IDCT — butterfly even + direct odd (22 mul, ~32 add)
    inline void idct_1d_aan(float* restrict src, float* restrict dst) {
        float s0 = src[0], s1 = src[1], s2 = src[2], s3 = src[3];
        float s4 = src[4], s5 = src[5], s6 = src[6], s7 = src[7];

        // ── Even part (butterfly, 6 mul) ──
        float sum_even  = s0 + s4;
        float diff_even = s0 - s4;

        float b_s2 = E_B * s2;
        float c_s6 = E_C * s6;
        float c_s2 = E_C * s2;
        float b_s6 = E_B * s6;

        float cross_plus  = b_s2 + c_s6;
        float cross_minus = c_s2 - b_s6;

        float a_sum  = E_A * sum_even;
        float a_diff = E_A * diff_even;

        float e0 = a_sum + cross_plus;
        float e1 = a_diff + cross_minus;
        float e2 = a_diff - cross_minus;
        float e3 = a_sum - cross_plus;

        // ── Odd part (direct 4×4, 16 mul) ──
        float o0 = O_B*s1 + O_C*s3 + O_D*s5 + O_E*s7;
        float o1 = O_C*s1 - O_E*s3 - O_B*s5 - O_D*s7;
        float o2 = O_D*s1 - O_B*s3 + O_E*s5 + O_C*s7;
        float o3 = O_E*s1 - O_D*s3 + O_C*s5 - O_B*s7;

        // Combine
        dst[0] = e0 + o0;  dst[7] = e0 - o0;
        dst[1] = e1 + o1;  dst[6] = e1 - o1;
        dst[2] = e2 + o2;  dst[5] = e2 - o2;
        dst[3] = e3 + o3;  dst[4] = e3 - o3;
    }

    __kernel void batch_idct(__global float* restrict blocks, int num_blocks) {
        int gid = get_global_id(0);
        if (gid >= num_blocks) return;

        __global float* restrict block = blocks + gid * 64;

        // Private memory for transposed row results (same trick as CPU path:
        // store rows in transposed order so column reads are unit-stride)
        float tmp[64];

        // ── Pass 1: IDCT on rows, store TRANSPOSED into tmp ──
        // tmp[k*8 + y] = result[y][k] (column-major temp)
        for (int y = 0; y < 8; y++) {
            int off = y * 8;
            float row0 = block[off];    float row1 = block[off+1];
            float row2 = block[off+2];  float row3 = block[off+3];
            float row4 = block[off+4];  float row5 = block[off+5];
            float row6 = block[off+6];  float row7 = block[off+7];
            float row[8] = {row0, row1, row2, row3, row4, row5, row6, row7};
            float d[8];
            idct_1d_aan(row, d);
            // Transposed store: d[k] -> tmp[k*8 + y]
            tmp[y]       = d[0];  tmp[8  + y] = d[1];
            tmp[16 + y]  = d[2];  tmp[24 + y] = d[3];
            tmp[32 + y]  = d[4];  tmp[40 + y] = d[5];
            tmp[48 + y]  = d[6];  tmp[56 + y] = d[7];
        }

        // ── Pass 2: IDCT on columns — read from tmp (unit stride) ──
        // tmp[x*8 .. x*8+7] contains column x as 8 consecutive floats
        for (int x = 0; x < 8; x++) {
            int off = x * 8;
            float col0 = tmp[off];    float col1 = tmp[off+1];
            float col2 = tmp[off+2];  float col3 = tmp[off+3];
            float col4 = tmp[off+4];  float col5 = tmp[off+5];
            float col6 = tmp[off+6];  float col7 = tmp[off+7];
            float col[8] = {col0, col1, col2, col3, col4, col5, col6, col7};
            float d[8];
            idct_1d_aan(col, d);
            // Store to column x of block (standard row-major output)
            block[x]      = d[0];  block[8+x]    = d[1];
            block[16+x]   = d[2];  block[24+x]   = d[3];
            block[32+x]   = d[4];  block[40+x]   = d[5];
            block[48+x]   = d[6];  block[56+x]   = d[7];
        }
    }

    __kernel void batch_fdct(__global float* blocks, int num_blocks) {
        int gid = get_global_id(0);
        if (gid >= num_blocks) return;
        // Placeholder — real DCT kernel would go here
        __global float* block = blocks + gid * 64;
        block[0] = 0.0;
    }

    __kernel void batch_ycbcr_to_rgb(
        __global const float* y, __global const float* cb, __global const float* cr,
        __global uchar* r, __global uchar* g, __global uchar* b,
        int num_pixels)
    {
        int gid = get_global_id(0);
        if (gid >= num_pixels) return;

        float cb_off = cb[gid] - 128.0f;
        float cr_off = cr[gid] - 128.0f;

        float rv = y[gid] + 1.402f    * cr_off;
        float gv = y[gid] - 0.344136f * cb_off - 0.714136f * cr_off;
        float bv = y[gid] + 1.772f    * cb_off;

        // Clamp and round
        r[gid] = (uchar)(clamp(rv + 0.5f, 0.0f, 255.0f));
        g[gid] = (uchar)(clamp(gv + 0.5f, 0.0f, 255.0f));
        b[gid] = (uchar)(clamp(bv + 0.5f, 0.0f, 255.0f));
    }
    "#;

    pub struct OpenClKernel {
        pro_que: ProQue,
        idct_kernel: Kernel,
        device: String,
        buf: std::sync::Mutex<Option<Buffer<f32>>>,
    }

    // Safety: the OpenClKernel is only ever used from a single thread at a time
    // (gated behind OnceLock + &self method calls).  The ocl Kernel and ProQue
    // internally contain `*mut c_void` which is !Sync, but our usage pattern
    // (serialized through the trait methods) is safe.
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

            // Build kernel with explicit program reference and queue.
            // In ocl 0.19, kernel arguments must be declared at build time;
            // we pass placeholder `None` args that will be set later via set_arg().
            let idct_kernel = ocl::Kernel::builder()
                .program(&pro_que.program())
                .name("batch_idct")
                .queue(pro_que.queue().clone())
                .arg(None::<&ocl::Buffer<f32>>)
                .arg(&0i32)
                .build()
                .map_err(|e| GpuError::KernelError(format!("kernel build: {e}")))?;

            Ok(OpenClKernel { pro_que, idct_kernel, device: device_name, buf: std::sync::Mutex::new(None) })
        }
    }

    impl GpuKernel for OpenClKernel {
        fn batch_idct_2d(&self, blocks: &mut [[f32; 64]]) -> Result<(), GpuError> {
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

            // Write host data to device buffer
            buf.write(unsafe {
                std::slice::from_raw_parts(blocks.as_ptr() as *const f32, total)
            }).enq().map_err(|e| GpuError::KernelError(format!("write: {e}")))?;

            self.idct_kernel
                .set_arg(0, buf)
                .map_err(|e| GpuError::KernelError(format!("arg 0: {e}")))?;
            self.idct_kernel
                .set_arg(1, &(n as i32))
                .map_err(|e| GpuError::KernelError(format!("arg 1: {e}")))?;

            unsafe {
                let local_size = if n < 64 { n as u64 } else { 64 };
                self.idct_kernel.cmd()
                    .global_work_size(n)
                    .local_work_size((local_size,))
                    .enq()
                    .map_err(|e| GpuError::KernelError(format!("enqueue: {e}")))?;
            }

            buf.read(unsafe {
                std::slice::from_raw_parts_mut(
                    blocks.as_mut_ptr() as *mut f32,
                    total,
                )
            }).enq().map_err(|e| GpuError::KernelError(format!("readback: {e}")))?;

            Ok(())
        }

        fn batch_dct_2d(&self, _blocks: &mut [[f32; 64]]) -> Result<(), GpuError> {
            Err(GpuError::NotAvailable)
        }

        fn batch_ycbcr_to_rgb(
            &self, y: &[f32], cb: &[f32], cr: &[f32],
            r: &mut [u8], g: &mut [u8], b: &mut [u8],
        ) -> Result<(), GpuError> {
            let n = y.len();
            if n == 0 { return Ok(()); }

            let total = n;

            // Allocate fresh device buffers for each color plane
            // (the Mutex in self.buf is for the IDCT reuse pool, not used here)
            let buf_y = Buffer::builder()
                .queue(self.pro_que.queue().clone())
                .flags(MemFlags::new().read_only())
                .len(total)
                .build()
                .map_err(|e| GpuError::KernelError(format!("buf_y alloc: {e}")))?;
            let buf_cb = Buffer::builder()
                .queue(self.pro_que.queue().clone())
                .flags(MemFlags::new().read_only())
                .len(total)
                .build()
                .map_err(|e| GpuError::KernelError(format!("buf_cb alloc: {e}")))?;
            let buf_cr = Buffer::builder()
                .queue(self.pro_que.queue().clone())
                .flags(MemFlags::new().read_only())
                .len(total)
                .build()
                .map_err(|e| GpuError::KernelError(format!("buf_cr alloc: {e}")))?;
            let buf_r = Buffer::builder()
                .queue(self.pro_que.queue().clone())
                .flags(MemFlags::new().write_only())
                .len(total)
                .build()
                .map_err(|e| GpuError::KernelError(format!("buf_r alloc: {e}")))?;
            let buf_g = Buffer::builder()
                .queue(self.pro_que.queue().clone())
                .flags(MemFlags::new().write_only())
                .len(total)
                .build()
                .map_err(|e| GpuError::KernelError(format!("buf_g alloc: {e}")))?;
            let buf_b = Buffer::builder()
                .queue(self.pro_que.queue().clone())
                .flags(MemFlags::new().write_only())
                .len(total)
                .build()
                .map_err(|e| GpuError::KernelError(format!("buf_b alloc: {e}")))?;

            // Write inputs
            buf_y.write(unsafe {
                std::slice::from_raw_parts(y.as_ptr() as *const f32, total)
            }).enq().map_err(|e| GpuError::KernelError(format!("write y: {e}")))?;
            buf_cb.write(cb).enq().map_err(|e| GpuError::KernelError(format!("write cb: {e}")))?;
            buf_cr.write(cr).enq().map_err(|e| GpuError::KernelError(format!("write cr: {e}")))?;

            // Build and launch kernel — rebuild each time since arguments change
            let kernel = ocl::Kernel::builder()
                .program(&self.pro_que.program())
                .name("batch_ycbcr_to_rgb")
                .queue(self.pro_que.queue().clone())
                .arg(&buf_y)
                .arg(&buf_cb)
                .arg(&buf_cr)
                .arg(&buf_r)
                .arg(&buf_g)
                .arg(&buf_b)
                .arg(&(n as i32))
                .build()
                .map_err(|e| GpuError::KernelError(format!("kernel build ycbcr: {e}")))?;

            unsafe {
                let local_size = if n < 64 { n as u64 } else { 64 };
                kernel.cmd()
                    .global_work_size(n)
                    .local_work_size((local_size,))
                    .enq()
                    .map_err(|e| GpuError::KernelError(format!("enqueue ycbcr: {e}")))?;
            }

            // Read outputs
            buf_r.read(r).enq().map_err(|e| GpuError::KernelError(format!("read r: {e}")))?;
            buf_g.read(g).enq().map_err(|e| GpuError::KernelError(format!("read g: {e}")))?;
            buf_b.read(b).enq().map_err(|e| GpuError::KernelError(format!("read b: {e}")))?;

            Ok(())
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
        let mut blocks = vec![[0.0f32; 64]; 10];
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
