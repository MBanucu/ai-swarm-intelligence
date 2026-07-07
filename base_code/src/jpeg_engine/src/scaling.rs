pub fn ycbcr_to_rgb(y: f64, cb: f64, cr: f64) -> (u8, u8, u8) {
    let r = y + 1.402 * (cr - 128.0);
    let g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0);
    let b = y + 1.772 * (cb - 128.0);
    let r = r.clamp(0.0, 255.0).round() as u8;
    let g = g.clamp(0.0, 255.0).round() as u8;
    let b = b.clamp(0.0, 255.0).round() as u8;
    (r, g, b)
}

pub fn bilinear_upsample(
    src: &[u8], src_w: usize, src_h: usize,
    dst: &mut [u8], dst_w: usize, dst_h: usize,
) {
    let x_ratio = if dst_w > 1 { (src_w - 1) as f64 / (dst_w - 1) as f64 } else { 0.0 };
    let y_ratio = if dst_h > 1 { (src_h - 1) as f64 / (dst_h - 1) as f64 } else { 0.0 };

    for y in 0..dst_h {
        let sy = y as f64 * y_ratio;
        let sy_int = sy as usize;
        let sy_frac = sy - sy_int as f64;
        let sy2 = (sy_int + 1).min(src_h - 1);

        for x in 0..dst_w {
            let sx = x as f64 * x_ratio;
            let sx_int = sx as usize;
            let sx_frac = sx - sx_int as f64;
            let sx2 = (sx_int + 1).min(src_w - 1);

            let a = src[sy_int * src_w + sx_int] as f64;
            let b = src[sy_int * src_w + sx2] as f64;
            let c = src[sy2 * src_w + sx_int] as f64;
            let d = src[sy2 * src_w + sx2] as f64;

            let val = a * (1.0 - sx_frac) * (1.0 - sy_frac)
                + b * sx_frac * (1.0 - sy_frac)
                + c * (1.0 - sx_frac) * sy_frac
                + d * sx_frac * sy_frac;

            dst[y * dst_w + x] = val.clamp(0.0, 255.0).round() as u8;
        }
    }
}

pub fn bilinear_downsample(
    src: &[u8], src_w: usize, src_h: usize,
    dst: &mut [u8], dst_w: usize, dst_h: usize,
) {
    let x_ratio = src_w as f64 / dst_w as f64;
    let y_ratio = src_h as f64 / dst_h as f64;

    for y in 0..dst_h {
        let sy = (y as f64 * y_ratio) as usize;
        for x in 0..dst_w {
            let sx = (x as f64 * x_ratio) as usize;
            dst[y * dst_w + x] = src[sy * src_w + sx];
        }
    }
}

pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let r = r as f64;
    let g = g as f64;
    let b = b as f64;
    let y  =  0.299 * r + 0.587 * g + 0.114 * b;
    let cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    let cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    (y, cb, cr)
}

/// Convert 8x8 pixel block to YCbCr component planes
pub fn block_to_ycbcr(pixels: &[(u8, u8, u8); 64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut y_block = vec![0.0f64; 64];
    let mut cb_block = vec![0.0f64; 64];
    let mut cr_block = vec![0.0f64; 64];

    for i in 0..64 {
        let (y, cb, cr) = rgb_to_ycbcr(pixels[i].0, pixels[i].1, pixels[i].2);
        y_block[i] = y;
        cb_block[i] = cb;
        cr_block[i] = cr;
    }
    (y_block, cb_block, cr_block)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ycbcr_black() {
        let (r, g, b) = ycbcr_to_rgb(0.0, 128.0, 128.0);
        assert_eq!((r, g, b), (0, 0, 0));
    }

    #[test]
    fn test_ycbcr_white() {
        let (r, g, b) = ycbcr_to_rgb(255.0, 128.0, 128.0);
        assert_eq!((r, g, b), (255, 255, 255));
    }

    #[test]
    fn test_upsample_2x() {
        let src = vec![10u8, 20, 30, 40];
        let mut dst = vec![0u8; 16];
        bilinear_upsample(&src, 2, 2, &mut dst, 4, 4);
        assert_eq!(dst[0], 10);
        assert_eq!(dst[15], 40);
    }

    #[test]
    fn test_downsample_2x() {
        let src: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let mut dst = vec![0u8; 16];
        bilinear_downsample(&src, 8, 8, &mut dst, 4, 4);
        assert_eq!(dst[0], 0);
    }
}
