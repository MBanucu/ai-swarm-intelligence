use std::sync::OnceLock;

// ──────────────────────────────────────────────────────
// Pre‑computed 1‑D DCT/IDCT transformation matrices
// Lazy‑initialised once.  The tables are tiny (64 f64).
// ──────────────────────────────────────────────────────

/// 1‑D DCT coefficient matrix: C[u][x] = cos((2·x+1)·u·π/16)
fn c_matrix() -> &'static [[f64; 8]; 8] {
    static TAB: OnceLock<[[f64; 8]; 8]> = OnceLock::new();
    TAB.get_or_init(|| {
        let mut m = [[0.0f64; 8]; 8];
        for u in 0..8 {
            for x in 0..8 {
                let arg = ((2 * x + 1) * u) as f64 * core::f64::consts::PI / 16.0;
                m[u][x] = arg.cos();
            }
        }
        m
    })
}

/// Normalised 1‑D DCT matrix:
///   DCT1D[u][x] = cu · cos((2·x+1)·u·π/16)
///   cu = 1/√2 for u==0, 1 otherwise.
/// The 0.25 total 2‑D scaling is applied in the column pass.
fn dct1d_matrix() -> &'static [[f64; 8]; 8] {
    static TAB: OnceLock<[[f64; 8]; 8]> = OnceLock::new();
    TAB.get_or_init(|| {
        let inv_sqrt2 = 0.7071067811865475f64;
        let mut m = [[0.0f64; 8]; 8];
        for u in 0..8 {
            let cu = if u == 0 { inv_sqrt2 } else { 1.0 };
            for x in 0..8 {
                let arg = ((2 * x + 1) * u) as f64 * core::f64::consts::PI / 16.0;
                m[u][x] = cu * arg.cos();
            }
        }
        m
    })
}

// ──────────────────────────────────────────────────────
// 1‑D transform helper
// ──────────────────────────────────────────────────────

/// Apply 1‑D DCT to an 8‑element vector.
#[inline(always)]
fn fdct_1d(src: &[f64; 8]) -> [f64; 8] {
    let mat = dct1d_matrix();
    let mut dst = [0.0f64; 8];
    for u in 0..8 {
        let row = &mat[u];
        dst[u] = src[0] * row[0] + src[1] * row[1] + src[2] * row[2] + src[3] * row[3]
               + src[4] * row[4] + src[5] * row[5] + src[6] * row[6] + src[7] * row[7];
    }
    dst
}

// ──────────────────────────────────────────────────────
// 2‑D transforms
// ──────────────────────────────────────────────────────

/// Forward DCT — separable row‑column implementation
/// Complexity: 8·64 + 8·64 = 1024 multiply‑adds  (vs 4096 for naive)
pub fn fdct_2d(block: &mut [f64; 64]) {
    let mut tmp = [0.0f64; 64];

    // 1) 1‑D DCT on every row
    for y in 0..8 {
        let mut row = [0.0f64; 8];
        let off = y * 8;
        row[0] = block[off];
        row[1] = block[off + 1];
        row[2] = block[off + 2];
        row[3] = block[off + 3];
        row[4] = block[off + 4];
        row[5] = block[off + 5];
        row[6] = block[off + 6];
        row[7] = block[off + 7];
        let d = fdct_1d(&row);
        tmp[off]     = d[0];
        tmp[off + 1] = d[1];
        tmp[off + 2] = d[2];
        tmp[off + 3] = d[3];
        tmp[off + 4] = d[4];
        tmp[off + 5] = d[5];
        tmp[off + 6] = d[6];
        tmp[off + 7] = d[7];
    }

    // 2) 1‑D DCT on every column, with 0.25 total scaling.
    // (Row pass: cu·cos, column pass: 0.25·cv·cos → total 0.25·cu·cv·cos·cos)
    for u in 0..8 {
        let mut col = [0.0f64; 8];
        col[0] = tmp[u];
        col[1] = tmp[8 + u];
        col[2] = tmp[16 + u];
        col[3] = tmp[24 + u];
        col[4] = tmp[32 + u];
        col[5] = tmp[40 + u];
        col[6] = tmp[48 + u];
        col[7] = tmp[56 + u];
        let d = fdct_1d(&col);
        block[u]      = 0.25 * d[0];
        block[8 + u]  = 0.25 * d[1];
        block[16 + u] = 0.25 * d[2];
        block[24 + u] = 0.25 * d[3];
        block[32 + u] = 0.25 * d[4];
        block[40 + u] = 0.25 * d[5];
        block[48 + u] = 0.25 * d[6];
        block[56 + u] = 0.25 * d[7];
    }
}

/// Naïve 2‑D IDCT (used only in tests for roundtrip verification)
pub fn idct_2d_naive(block: &mut [f64; 64]) {
    let mat = c_matrix();
    let mut temp = [0.0f64; 64];

    for x in 0..8 {
        for y in 0..8 {
            let mut sum = 0.0;
            for u in 0..8 {
                for v in 0..8 {
                    let cu = if u == 0 { 1.0 / 2.0f64.sqrt() } else { 1.0 };
                    let cv = if v == 0 { 1.0 / 2.0f64.sqrt() } else { 1.0 };
                    sum += cu * cv * block[u * 8 + v]
                         * mat[u][x]  // cos((2x+1)uπ/16)
                         * mat[v][y]; // cos((2y+1)vπ/16)
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
