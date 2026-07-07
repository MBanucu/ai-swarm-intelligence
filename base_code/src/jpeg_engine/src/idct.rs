use std::sync::OnceLock;

// ──────────────────────────────────────────────────────
// Pre‑computed 1‑D IDCT matrix  (lazy‑initialised)
// ──────────────────────────────────────────────────────

/// 1‑D IDCT matrix:
///   IDCT1D[x][u] = cu · cos((2·x+1)·u·π/16)
/// where cu = 1/√2 for u==0, 1 otherwise.
fn idct1d_matrix() -> &'static [[f64; 8]; 8] {
    static TAB: OnceLock<[[f64; 8]; 8]> = OnceLock::new();
    TAB.get_or_init(|| {
        let inv_sqrt2 = 0.7071067811865475f64;
        let mut m = [[0.0f64; 8]; 8];
        for x in 0..8 {
            for u in 0..8 {
                let cu = if u == 0 { inv_sqrt2 } else { 1.0 };
                let arg = ((2 * x + 1) * u) as f64 * core::f64::consts::PI / 16.0;
                m[x][u] = cu * arg.cos();
            }
        }
        m
    })
}

/// 1‑D IDCT helper — fully unrolled inner loop for max throughput.
/// Computes `dst[x] = sum_u IDCT1D[x][u] · src[u]`.
#[inline(always)]
fn idct_1d(src: &[f64; 8]) -> [f64; 8] {
    let mat = idct1d_matrix();
    let mut dst = [0.0f64; 8];
    for x in 0..8 {
        let row = &mat[x];
        dst[x] = src[0] * row[0] + src[1] * row[1] + src[2] * row[2] + src[3] * row[3]
               + src[4] * row[4] + src[5] * row[5] + src[6] * row[6] + src[7] * row[7];
    }
    dst
}

// ──────────────────────────────────────────────────────
// 2‑D IDCT — separable row‑column implementation
// ──────────────────────────────────────────────────────

/// Inverse DCT — separable row‑column implementation.
///
/// 1) 1‑D IDCT on every row.
/// 2) 1‑D IDCT on every column, with final 0.25 scaling
///    distributed as 0.5 in each pass.
///
/// Total: 8·64 + 8·64 = 1024 multiply‑adds  (vs 4096 for naive).
pub fn idct_2d(block: &mut [f64; 64]) {
    let mut tmp = [0.0f64; 64];

    // ── Pass 1: IDCT on rows ──
    for y in 0..8 {
        let off = y * 8;
        let mut row = [0.0f64; 8];
        row[0] = block[off];
        row[1] = block[off + 1];
        row[2] = block[off + 2];
        row[3] = block[off + 3];
        row[4] = block[off + 4];
        row[5] = block[off + 5];
        row[6] = block[off + 6];
        row[7] = block[off + 7];
        // Apply 1‑D IDCT, then scale by 0.5
        let d = idct_1d(&row);
        tmp[off]     = 0.5 * d[0];
        tmp[off + 1] = 0.5 * d[1];
        tmp[off + 2] = 0.5 * d[2];
        tmp[off + 3] = 0.5 * d[3];
        tmp[off + 4] = 0.5 * d[4];
        tmp[off + 5] = 0.5 * d[5];
        tmp[off + 6] = 0.5 * d[6];
        tmp[off + 7] = 0.5 * d[7];
    }

    // ── Pass 2: IDCT on columns ──
    for x in 0..8 {
        let mut col = [0.0f64; 8];
        col[0] = tmp[x];
        col[1] = tmp[8 + x];
        col[2] = tmp[16 + x];
        col[3] = tmp[24 + x];
        col[4] = tmp[32 + x];
        col[5] = tmp[40 + x];
        col[6] = tmp[48 + x];
        col[7] = tmp[56 + x];
        // Apply 1‑D IDCT, then scale by 0.5 (total 0.25)
        let d = idct_1d(&col);
        block[x]      = 0.5 * d[0];
        block[8 + x]  = 0.5 * d[1];
        block[16 + x] = 0.5 * d[2];
        block[24 + x] = 0.5 * d[3];
        block[32 + x] = 0.5 * d[4];
        block[40 + x] = 0.5 * d[5];
        block[48 + x] = 0.5 * d[6];
        block[56 + x] = 0.5 * d[7];
    }
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
