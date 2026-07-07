use std::f64::consts::PI;

#[inline]
fn cos(x: f64) -> f64 {
    (x * PI / 16.0).cos()
}

pub fn idct_2d(block: &mut [f64; 64]) {
    let mut tmp = [0.0f64; 64];

    for x in 0..8 {
        for y in 0..8 {
            let mut sum = 0.0;
            for u in 0..8 {
                for v in 0..8 {
                    let cu = if u == 0 { 1.0 / 2.0f64.sqrt() } else { 1.0 };
                    let cv = if v == 0 { 1.0 / 2.0f64.sqrt() } else { 1.0 };
                    sum += cu * cv * block[u * 8 + v]
                         * cos((2.0 * x as f64 + 1.0) * u as f64)
                         * cos((2.0 * y as f64 + 1.0) * v as f64);
                }
            }
            tmp[x * 8 + y] = 0.25 * sum;
        }
    }
    *block = tmp;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idct_zero_block() {
        let mut block = [0.0f64; 64];
        idct_2d(&mut block);
        for &val in &block {
            assert!(val.abs() < 0.001, "expected near-zero, got {}", val);
        }
    }

    #[test]
    fn test_idct_dc_only() {
        let mut block = [0.0f64; 64];
        block[0] = 8.0;
        idct_2d(&mut block);
        for &val in &block {
            assert!((val - 1.0).abs() < 0.001, "DC-only IDCT: expected 1.0, got {}", val);
        }
    }
}
