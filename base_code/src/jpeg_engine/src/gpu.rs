/// GPU acceleration — trait-based interface for GPU kernels.
/// Implementations may use OpenCL, Vulkan, or CUDA.
/// CPU fallback always available.

pub trait GpuKernel: Send + Sync {
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
    fn batch_idct_2d(&self, blocks: &mut [[f64; 64]]) -> Result<(), GpuError> {
        for block in blocks.iter_mut() {
            crate::idct::idct_2d(block);
        }
        Ok(())
    }

    fn batch_dct_2d(&self, blocks: &mut [[f64; 64]]) -> Result<(), GpuError> {
        for block in blocks.iter_mut() {
            crate::dct::fdct_2d(block);
        }
        Ok(())
    }

    fn batch_ycbcr_to_rgb(
        &self, y: &[f64], cb: &[f64], cr: &[f64],
        r: &mut [u8], g: &mut [u8], b: &mut [u8],
    ) -> Result<(), GpuError> {
        for i in 0..y.len() {
            let (rp, gp, bp) = crate::scaling::ycbcr_to_rgb(y[i], cb[i], cr[i]);
            r[i] = rp;
            g[i] = gp;
            b[i] = bp;
        }
        Ok(())
    }

    fn device_name(&self) -> &str { "CPU" }
}

pub fn gpu_available() -> bool {
    false
}

pub fn create_kernel() -> Box<dyn GpuKernel> {
    Box::new(CpuKernel)
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
        assert_eq!(kernel.device_name(), "CPU");
    }
}
