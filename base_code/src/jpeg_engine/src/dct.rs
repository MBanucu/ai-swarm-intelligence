use std::f64::consts::PI;

fn cos(x: f64) -> f64 {
    (x * PI / 16.0).cos()
}

pub fn fdct_2d(block: &mut [f64; 64]) {
    let mut temp = [0.0f64; 64];

    for u in 0..8 {
        for v in 0..8 {
            let mut sum = 0.0;
            for x in 0..8 {
                for y in 0..8 {
                    sum += block[x * 8 + y] * cos((2.0 * x as f64 + 1.0) * u as f64)
                         * cos((2.0 * y as f64 + 1.0) * v as f64);
                }
            }
            let cu = if u == 0 { 1.0 / 2.0f64.sqrt() } else { 1.0 };
            let cv = if v == 0 { 1.0 / 2.0f64.sqrt() } else { 1.0 };
            temp[u * 8 + v] = 0.25 * cu * cv * sum;
        }
    }
    *block = temp;
}

pub fn idct_2d_naive(block: &mut [f64; 64]) {
    let mut temp = [0.0f64; 64];

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
            temp[x * 8 + y] = 0.25 * sum;
        }
    }
    *block = temp;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fdct_idct_roundtrip() {
        let orig: [f64; 64] = core::array::from_fn(|i| {
            let row = i / 8;
            let col = i % 8;
            (row as f64 * col as f64 % 256.0) - 128.0
        });
        let mut block = orig;
        fdct_2d(&mut block);
        idct_2d_naive(&mut block);
        for i in 0..64 {
            assert!((block[i] - orig[i]).abs() < 0.1,
                "roundtrip error at {i}: got {}, expected {}", block[i], orig[i]);
        }
    }

    #[test]
    fn test_fdct_identity() {
        let mut block = [1.0f64; 64];
        fdct_2d(&mut block);
        assert!((block[0] - 8.0).abs() < 0.01, "DC value should be 8.0, got {}", block[0]);
        for i in 1..64 {
            assert!(block[i].abs() < 0.01, "AC value at {i} should be 0, got {}", block[i]);
        }
    }
}
