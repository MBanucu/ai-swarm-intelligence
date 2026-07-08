pub mod header;
pub mod huffman;
pub mod dct;
pub mod idct;
pub mod scaling;
pub mod gpu;

use half::f16;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct JpegInfo {
    pub width: u16,
    pub height: u16,
    pub components: u8,
    pub sampling_h: [u8; 3],
    pub sampling_v: [u8; 3],
    pub qtables: [[u16; 64]; 4],
    pub huff_dc: Option<huffman::HuffmanTable>,
    pub huff_ac: Option<huffman::HuffmanTable>,
}

#[derive(Debug, Clone)]
pub struct MCU {
    pub blocks: Vec<[[f16; 64]; 3]>,
}

#[no_mangle]
pub extern "C" fn jpeg_decode_header(data: *const u8, len: usize) -> *mut JpegInfo {
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    match header::parse_header(slice) {
        Ok(info) => Box::into_raw(Box::new(info)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn jpeg_free_info(info: *mut JpegInfo) {
    if !info.is_null() {
        unsafe { drop(Box::from_raw(info)) };
    }
}

#[no_mangle]
pub extern "C" fn dct_2d(blocks: *mut u16) {
    let slice: &mut [f16] = unsafe {
        std::slice::from_raw_parts_mut(blocks as *mut f16, 64)
    };
    let arr: &mut [f16; 64] = slice.try_into().unwrap();
    dct::fdct_2d(arr);
}

#[no_mangle]
pub extern "C" fn idct_2d(blocks: *mut u16) {
    let slice: &mut [f16] = unsafe {
        std::slice::from_raw_parts_mut(blocks as *mut f16, 64)
    };
    let arr: &mut [f16; 64] = slice.try_into().unwrap();
    idct::idct_2d(arr);
}

// GPU auto-dispatch threshold
const GPU_THRESHOLD: usize = 500_000;

fn gpu_kernel() -> &'static Option<Box<dyn gpu::GpuKernel>> {
    static KERNEL: OnceLock<Option<Box<dyn gpu::GpuKernel>>> = OnceLock::new();
    KERNEL.get_or_init(|| {
        #[cfg(feature = "gpu")]
        {
            let k = gpu::create_kernel();
            if k.device_name() != "CPU" {
                Some(k)
            } else {
                None
            }
        }
        #[cfg(not(feature = "gpu"))]
        None
    })
}

#[inline(always)]
pub fn idct_2d_batch_const<const N: usize>(blocks: &mut [[f16; 64]]) {
    debug_assert!(N == 0 || blocks.len() == N,
        "idct_2d_batch_const: N={} but blocks.len()={}", N, blocks.len());

    let use_gpu = if N > 0 { N >= GPU_THRESHOLD } else { blocks.len() >= GPU_THRESHOLD };

    if use_gpu {
        if let Some(kernel) = gpu_kernel() {
            if kernel.batch_idct_2d(blocks).is_ok() {
                return;
            }
        }
    }

    idct::batch_idct_2d(blocks);
}

#[no_mangle]
pub extern "C" fn idct_2d_batch(blocks: *mut u16, count: u32) {
    let n = count as usize;
    let slice: &mut [f16] = unsafe {
        std::slice::from_raw_parts_mut(blocks as *mut f16, n * 64)
    };
    let blocks: &mut [[f16; 64]] = unsafe {
        std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut [f16; 64], n)
    };
    idct_2d_batch_const::<0>(blocks);
}

#[no_mangle]
pub extern "C" fn scale_upsample(
    input: *const u8, in_w: u16, in_h: u16,
    output: *mut u8, out_w: u16, out_h: u16,
) {
    let src = unsafe { std::slice::from_raw_parts(input, in_w as usize * in_h as usize) };
    let dst = unsafe { std::slice::from_raw_parts_mut(output, out_w as usize * out_h as usize) };
    scaling::bilinear_upsample(src, in_w as usize, in_h as usize, dst, out_w as usize, out_h as usize);
}

#[no_mangle]
pub extern "C" fn scale_downsample(
    input: *const u8, in_w: u16, in_h: u16,
    output: *mut u8, out_w: u16, out_h: u16,
) {
    let src = unsafe { std::slice::from_raw_parts(input, in_w as usize * in_h as usize) };
    let dst = unsafe { std::slice::from_raw_parts_mut(output, out_w as usize * out_h as usize) };
    scaling::bilinear_downsample(src, in_w as usize, in_h as usize, dst, out_w as usize, out_h as usize);
}

#[no_mangle]
pub extern "C" fn ycbcr_to_rgb(y: u16, cb: u16, cr: u16) -> u32 {
    let yf = f16::from_bits(y);
    let cbf = f16::from_bits(cb);
    let crf = f16::from_bits(cr);
    let (r, g, b) = scaling::ycbcr_to_rgb(yf, cbf, crf);
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

#[no_mangle]
pub extern "C" fn huffman_decode(
    data: *const u8, data_len: usize, bit_offset: usize,
    table: *const huffman::HuffmanTable,
) -> i32 {
    let slice = unsafe { std::slice::from_raw_parts(data, data_len) };
    let table = unsafe { &*table };
    huffman::decode_symbol(slice, bit_offset, table)
}

#[no_mangle]
pub extern "C" fn huffman_table_create() -> *mut huffman::HuffmanTable {
    Box::into_raw(Box::new(huffman::HuffmanTable::default()))
}

#[no_mangle]
pub extern "C" fn huffman_table_free(table: *mut huffman::HuffmanTable) {
    if !table.is_null() {
        unsafe { drop(Box::from_raw(table)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_idct_roundtrip() {
        let mut block = [f16::ZERO; 64];
        block[0] = f16::from_f32(128.0);
        block[1] = f16::from_f32(16.0);
        let orig = block;
        dct_2d(block.as_mut_ptr() as *mut u16);
        idct_2d(block.as_mut_ptr() as *mut u16);
        for i in 0..64 {
            let diff = if block[i] > orig[i] { block[i] - orig[i] } else { orig[i] - block[i] };
            assert!(diff < f16::from_f32(2.0), "roundtrip error at {i}: {:?} vs {:?}", block[i], orig[i]);
        }
    }

    #[test]
    fn test_ycbcr_to_rgb_black() {
        let rgb = ycbcr_to_rgb(f16::ZERO.to_bits(), f16::from_f32(128.0).to_bits(), f16::from_f32(128.0).to_bits());
        assert_eq!(rgb, 0x000000);
    }

    #[test]
    fn test_benchmark_batch_idct() {
        let mut blocks: Vec<[f16; 64]> = (0..1000).map(|i| {
            let mut b = [f16::ZERO; 64];
            for j in 0..64 {
                b[j] = f16::from_f32((i as f32 * j as f32 % 256.0) - 128.0);
            }
            b
        }).collect();
        let start = std::time::Instant::now();
        for block in &mut blocks {
            idct_2d(block.as_mut_ptr() as *mut u16);
        }
        let elapsed = start.elapsed();
        let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / 1000.0;
        println!("Batch IDCT: {:.6} ms/iter", ms_per_iter);
        assert!(ms_per_iter < 20.0, "IDCT too slow: {:.6}ms/iter", ms_per_iter);
    }
}
