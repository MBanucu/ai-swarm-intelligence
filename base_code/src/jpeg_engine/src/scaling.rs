use half::f16;

const KR: f16 = f16::from_f32_const(1.402);
const KG1: f16 = f16::from_f32_const(0.344136);
const KG2: f16 = f16::from_f32_const(0.714136);
const KB: f16 = f16::from_f32_const(1.772);

#[inline(always)]
fn clamp_u8(v: f16) -> u8 {
    let v = v + f16::from_f32(0.5);
    if v <= f16::ZERO { return 0u8; }
    if v >= f16::from_f32(256.0) { return 255u8; }
    v.to_f32() as u8
}

pub fn ycbcr_to_rgb(y: f16, cb: f16, cr: f16) -> (u8, u8, u8) {
    let cb_off = cb - f16::from_f32(128.0);
    let cr_off = cr - f16::from_f32(128.0);

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
    let x_ratio = if dst_w > 1 { f16::from_f32((src_w - 1) as f32) / f16::from_f32((dst_w - 1) as f32) } else { f16::ZERO };
    let y_ratio = if dst_h > 1 { f16::from_f32((src_h - 1) as f32) / f16::from_f32((dst_h - 1) as f32) } else { f16::ZERO };

    let src_rows: Vec<usize> = (0..dst_h)
        .map(|y| {
            let sy = f16::from_f32(y as f32) * y_ratio;
            let sy_int = sy.to_f32() as usize;
            sy_int * src_w
        })
        .collect();

    let src_rows2: Vec<usize> = (0..dst_h)
        .map(|y| {
            let sy = f16::from_f32(y as f32) * y_ratio;
            let sy_int = sy.to_f32() as usize;
            ((sy_int + 1).min(src_h - 1)) * src_w
        })
        .collect();

    let sy_fracs: Vec<f16> = (0..dst_h)
        .map(|y| {
            let sy = f16::from_f32(y as f32) * y_ratio;
            sy - f16::from_f32(sy.to_f32() as usize as f32)
        })
        .collect();

    let x_positions: Vec<usize> = (0..dst_w)
        .map(|x| {
            let sx = f16::from_f32(x as f32) * x_ratio;
            sx.to_f32() as usize
        })
        .collect();

    let x_positions2: Vec<usize> = (0..dst_w)
        .map(|x| {
            let sx = f16::from_f32(x as f32) * x_ratio;
            (sx.to_f32() as usize + 1).min(src_w - 1)
        })
        .collect();

    let x_fracs: Vec<f16> = (0..dst_w)
        .map(|x| {
            let sx = f16::from_f32(x as f32) * x_ratio;
            sx - f16::from_f32(sx.to_f32() as usize as f32)
        })
        .collect();

    for y in 0..dst_h {
        let sy_frac = sy_fracs[y];
        let row_off = src_rows[y];
        let row_off2 = src_rows2[y];
        let inv_sy = f16::from_f32(1.0) - sy_frac;
        let dst_row = y * dst_w;

        for x in 0..dst_w {
            let sx_int = x_positions[x];
            let sx2 = x_positions2[x];
            let sx_frac = x_fracs[x];

            let a = f16::from_f32(src[row_off + sx_int] as f32);
            let b = f16::from_f32(src[row_off + sx2] as f32);
            let c = f16::from_f32(src[row_off2 + sx_int] as f32);
            let d = f16::from_f32(src[row_off2 + sx2] as f32);

            let inv_sx = f16::from_f32(1.0) - sx_frac;
            let top = a * inv_sx + b * sx_frac;
            let bot = c * inv_sx + d * sx_frac;
            let val = top * inv_sy + bot * sy_frac;

            dst[dst_row + x] = if val <= f16::ZERO { 0u8 } else if val >= f16::from_f32(255.0) { 255u8 } else { (val + f16::from_f32(0.5)).to_f32() as u8 };
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
    let x_ratio = f16::from_f32(src_w as f32) / f16::from_f32(dst_w as f32);
    let y_ratio = f16::from_f32(src_h as f32) / f16::from_f32(dst_h as f32);

    for y in 0..dst_h {
        let sy = (f16::from_f32(y as f32) * y_ratio).to_f32() as usize * src_w;
        let dst_row = y * dst_w;
        for x in 0..dst_w {
            let sx = (f16::from_f32(x as f32) * x_ratio).to_f32() as usize;
            dst[dst_row + x] = src[sy + sx];
        }
    }
}

#[inline]
pub fn ycbcr_to_rgb_8(
    y: &[f16], cb: &[f16], cr: &[f16],
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

pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (f16, f16, f16) {
    let r = f16::from_f32(r as f32);
    let g = f16::from_f32(g as f32);
    let b = f16::from_f32(b as f32);
    let y  =  f16::from_f32(0.299) * r + f16::from_f32(0.587) * g + f16::from_f32(0.114) * b;
    let cb = f16::from_f32(128.0) - f16::from_f32(0.168736) * r - f16::from_f32(0.331264) * g + f16::from_f32(0.5) * b;
    let cr = f16::from_f32(128.0) + f16::from_f32(0.5) * r - f16::from_f32(0.418688) * g - f16::from_f32(0.081312) * b;
    (y, cb, cr)
}

pub fn block_to_ycbcr(pixels: &[(u8, u8, u8); 64]) -> (Vec<f16>, Vec<f16>, Vec<f16>) {
    let mut y_block = vec![f16::ZERO; 64];
    let mut cb_block = vec![f16::ZERO; 64];
    let mut cr_block = vec![f16::ZERO; 64];

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
        let (r, g, b) = ycbcr_to_rgb(f16::ZERO, f16::from_f32(128.0), f16::from_f32(128.0));
        assert_eq!((r, g, b), (0, 0, 0));
    }

    #[test]
    fn test_ycbcr_white() {
        let (r, g, b) = ycbcr_to_rgb(f16::from_f32(255.0), f16::from_f32(128.0), f16::from_f32(128.0));
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
