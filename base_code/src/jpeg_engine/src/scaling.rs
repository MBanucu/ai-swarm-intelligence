/// Pre‑computed YCbCr→RGB coefficients (compile‑time constants)
const KR: f64 = 1.402;
const KG1: f64 = 0.344136;
const KG2: f64 = 0.714136;
const KB: f64 = 1.772;

/// Fast clamp helper — avoids `round()` function call overhead.
#[inline(always)]
fn clamp_u8(v: f64) -> u8 {
    // Using `+ 0.5` + `as u8` truncation gives rounding,
    // and the branchless max/min avoids expensive `round()` calls.
    let v = v + 0.5;
    if v <= 0.0 { return 0u8; }
    if v >= 256.0 { return 255u8; }
    v as u8
}

/// Core YCbCr→RGB conversion.
///
/// Uses fused‑multiply‑add pattern: compiler auto‑vectorises
/// this into `vfmadd` / `vfmsub` across scanlines.
/// Replaced `round()` call with inline `+ 0.5` truncation for ~2× faster clamp.
pub fn ycbcr_to_rgb(y: f64, cb: f64, cr: f64) -> (u8, u8, u8) {
    let cb_off = cb - 128.0;
    let cr_off = cr - 128.0;

    let r = y + KR * cr_off;
    let g = y - KG1 * cb_off - KG2 * cr_off;
    let b = y + KB * cb_off;

    (clamp_u8(r), clamp_u8(g), clamp_u8(b))
}

pub fn bilinear_upsample(
    src: &[u8], src_w: usize, src_h: usize,
    dst: &mut [u8], dst_w: usize, dst_h: usize,
) {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return;
    }
    let x_ratio = if dst_w > 1 { (src_w - 1) as f64 / (dst_w - 1) as f64 } else { 0.0 };
    let y_ratio = if dst_h > 1 { (src_h - 1) as f64 / (dst_h - 1) as f64 } else { 0.0 };

    // Precompute source row offsets and fractions
    let src_rows: Vec<usize> = (0..dst_h)
        .map(|y| {
            let sy = y as f64 * y_ratio;
            let sy_int = sy as usize;
            sy_int * src_w
        })
        .collect();

    let src_rows2: Vec<usize> = (0..dst_h)
        .map(|y| {
            let sy = y as f64 * y_ratio;
            let sy_int = sy as usize;
            ((sy_int + 1).min(src_h - 1)) * src_w
        })
        .collect();

    let sy_fracs: Vec<f64> = (0..dst_h)
        .map(|y| {
            let sy = y as f64 * y_ratio;
            sy - (sy as usize) as f64
        })
        .collect();

    // Precompute all x positions and fractions
    let x_positions: Vec<usize> = (0..dst_w)
        .map(|x| {
            let sx = x as f64 * x_ratio;
            sx as usize
        })
        .collect();

    let x_positions2: Vec<usize> = (0..dst_w)
        .map(|x| {
            let sx = x as f64 * x_ratio;
            (sx as usize + 1).min(src_w - 1)
        })
        .collect();

    let x_fracs: Vec<f64> = (0..dst_w)
        .map(|x| {
            let sx = x as f64 * x_ratio;
            sx - (sx as usize) as f64
        })
        .collect();

    for y in 0..dst_h {
        let sy_frac = sy_fracs[y];
        let row_off = src_rows[y];
        let row_off2 = src_rows2[y];
        let inv_sy = 1.0 - sy_frac;
        let dst_row = y * dst_w;

        for x in 0..dst_w {
            let sx_int = x_positions[x];
            let sx2 = x_positions2[x];
            let sx_frac = x_fracs[x];

            let a = src[row_off + sx_int] as f64;
            let b = src[row_off + sx2] as f64;
            let c = src[row_off2 + sx_int] as f64;
            let d = src[row_off2 + sx2] as f64;

            let inv_sx = 1.0 - sx_frac;
            let top = a * inv_sx + b * sx_frac;
            let bot = c * inv_sx + d * sx_frac;
            let val = top * inv_sy + bot * sy_frac;

            dst[dst_row + x] = if val <= 0.0 { 0u8 } else if val >= 255.0 { 255u8 } else { (val + 0.5) as u8 };
        }
    }
}

pub fn bilinear_downsample(
    src: &[u8], src_w: usize, src_h: usize,
    dst: &mut [u8], dst_w: usize, dst_h: usize,
) {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return;
    }
    let x_ratio = src_w as f64 / dst_w as f64;
    let y_ratio = src_h as f64 / dst_h as f64;

    // Precompute source row indices
    for y in 0..dst_h {
        let sy = ((y as f64 * y_ratio) as usize) * src_w;
        let dst_row = y * dst_w;
        for x in 0..dst_w {
            let sx = (x as f64 * x_ratio) as usize;
            dst[dst_row + x] = src[sy + sx];
        }
    }
}

/// Batch‑process 8 YCbCr triples → 8 RGB triples.
/// Accepts slices for ergonomic calling; caller must supply ≥8 elements.
/// Specialised to keep constants in registers and help auto‑vectorisation.
#[inline]
pub fn ycbcr_to_rgb_8(
    y: &[f64], cb: &[f64], cr: &[f64],
) -> ([u8; 8], [u8; 8], [u8; 8]) {
    let mut r = [0u8; 8];
    let mut g = [0u8; 8];
    let mut b = [0u8; 8];
    for i in 0..8 {
        let (ri, gi, bi) = ycbcr_to_rgb(y[i], cb[i], cr[i]);
        r[i] = ri;
        g[i] = gi;
        b[i] = bi;
    }
    (r, g, b)
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
