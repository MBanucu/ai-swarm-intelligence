pub mod header;
pub mod huffman;
pub mod dct;
pub mod idct;
pub mod scaling;
pub mod gpu;

use std::os::raw::c_double;

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
    pub blocks: Vec<[[f64; 64]; 3]>,
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
pub extern "C" fn dct_2d(block: *mut c_double) {
    let slice = unsafe { std::slice::from_raw_parts_mut(block, 64) };
    // Work directly on the slice to avoid a copy (eliminates 128 B copy per call).
    let arr: &mut [f64; 64] = slice.try_into().unwrap();
    dct::fdct_2d(arr);
}

#[no_mangle]
pub extern "C" fn idct_2d(block: *mut c_double) {
    let slice = unsafe { std::slice::from_raw_parts_mut(block, 64) };
    // Work directly on the slice — no intermediate copy.
    let arr: &mut [f64; 64] = slice.try_into().unwrap();
    idct::idct_2d(arr);
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
pub extern "C" fn ycbcr_to_rgb(y: c_double, cb: c_double, cr: c_double) -> u32 {
    let (r, g, b) = scaling::ycbcr_to_rgb(y, cb, cr);
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
        let mut block = [0.0f64; 64];
        block[0] = 128.0;
        block[1] = 16.0;
        let orig = block;
        dct_2d(block.as_mut_ptr());
        idct_2d(block.as_mut_ptr());
        for i in 0..64 {
            assert!((block[i] - orig[i]).abs() < 1.0, "roundtrip error at {i}: {} vs {}", block[i], orig[i]);
        }
    }

    #[test]
    fn test_ycbcr_to_rgb_black() {
        let rgb = ycbcr_to_rgb(0.0, 128.0, 128.0);
        assert_eq!(rgb, 0x000000);
    }

    #[test]
    fn test_ycbcr_to_rgb_white() {
        let rgb = ycbcr_to_rgb(255.0, 128.0, 128.0);
        assert_eq!(rgb, 0xFFFFFF);
    }

    #[test]
    fn test_benchmark_batch_idct() {
        let mut blocks: Vec<[f64; 64]> = (0..1000).map(|i| {
            let mut b = [0.0f64; 64];
            for j in 0..64 {
                b[j] = (i as f64 * j as f64 % 256.0) - 128.0;
            }
            b
        }).collect();
        let start = std::time::Instant::now();
        for block in &mut blocks {
            idct_2d(block.as_mut_ptr());
        }
        let elapsed = start.elapsed();
        let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / 1000.0;
        println!("Batch IDCT: {:.6} ms/iter", ms_per_iter);
        assert!(ms_per_iter < 10.0, "IDCT too slow: {:.6}ms/iter", ms_per_iter);
    }
}
