use half::f16;

// ──────────────────────────────────────────────────────
// Optimised 1‑D DCT — even/odd decomposition
//
// The forward DCT decomposes into:
//   Even outputs (0,2,4,6): 4-point DCT on (y[n] + y[7-n])
//   Odd outputs (1,3,5,7):  4-point DST on (y[n] - y[7-n])
//
// Arithmetic: 32 mul + 40 add per 1‑D transform.
// ──────────────────────────────────────────────────────

/// Row‑pass even submatrix: Y[2k] = Σ_n E_row[k][n] · (y[n] + y[7-n])
/// For k=0..3, n=0..3.  Contains C(2k)·cos(π·k·(2n+1)/8).
const E_ROW: [[f16; 4]; 4] = [
    [f16::from_f32_const(0.70710678), f16::from_f32_const(0.70710678), f16::from_f32_const(0.70710678), f16::from_f32_const(0.70710678)],
    [f16::from_f32_const(0.92387953), f16::from_f32_const(0.38268343), f16::from_f32_const(-0.38268343), f16::from_f32_const(-0.92387953)],
    [f16::from_f32_const(0.70710678), f16::from_f32_const(-0.70710678), f16::from_f32_const(-0.70710678), f16::from_f32_const(0.70710678)],
    [f16::from_f32_const(0.38268343), f16::from_f32_const(-0.92387953), f16::from_f32_const(0.92387953), f16::from_f32_const(-0.38268343)],
];

/// Row‑pass odd submatrix: Y[2k+1] = Σ_n O_row[k][n] · (y[n] - y[7-n])
/// Contains cos(π·(2k+1)·(2n+1)/16).
const O_ROW: [[f16; 4]; 4] = [
    [f16::from_f32_const(0.98078528), f16::from_f32_const(0.83146961), f16::from_f32_const(0.55557023), f16::from_f32_const(0.19509032)],
    [f16::from_f32_const(0.83146961), f16::from_f32_const(-0.19509032), f16::from_f32_const(-0.98078528), f16::from_f32_const(-0.55557023)],
    [f16::from_f32_const(0.55557023), f16::from_f32_const(-0.98078528), f16::from_f32_const(0.19509032), f16::from_f32_const(0.83146961)],
    [f16::from_f32_const(0.19509032), f16::from_f32_const(-0.55557023), f16::from_f32_const(0.83146961), f16::from_f32_const(-0.98078528)],
];

/// Column‑pass even submatrix (0.25 scaling fused).
const E_COL: [[f16; 4]; 4] = [
    [f16::from_f32_const(0.17677670), f16::from_f32_const(0.17677670), f16::from_f32_const(0.17677670), f16::from_f32_const(0.17677670)],
    [f16::from_f32_const(0.23096988), f16::from_f32_const(0.09567086), f16::from_f32_const(-0.09567086), f16::from_f32_const(-0.23096988)],
    [f16::from_f32_const(0.17677670), f16::from_f32_const(-0.17677670), f16::from_f32_const(-0.17677670), f16::from_f32_const(0.17677670)],
    [f16::from_f32_const(0.09567086), f16::from_f32_const(-0.23096988), f16::from_f32_const(0.23096988), f16::from_f32_const(-0.09567086)],
];

/// Column‑pass odd submatrix (0.25 scaling fused).
const O_COL: [[f16; 4]; 4] = [
    [f16::from_f32_const(0.24519632), f16::from_f32_const(0.20786740), f16::from_f32_const(0.13889256), f16::from_f32_const(0.04877258)],
    [f16::from_f32_const(0.20786740), f16::from_f32_const(-0.04877258), f16::from_f32_const(-0.24519632), f16::from_f32_const(-0.13889256)],
    [f16::from_f32_const(0.13889256), f16::from_f32_const(-0.24519632), f16::from_f32_const(0.04877258), f16::from_f32_const(0.20786740)],
    [f16::from_f32_const(0.04877258), f16::from_f32_const(-0.13889256), f16::from_f32_const(0.20786740), f16::from_f32_const(-0.24519632)],
];

/// 1‑D row‑pass DCT — even/odd decomposition.
#[inline(always)]
fn fdct_1d(src: &[f16; 8]) -> [f16; 8] {
    let s0 = src[0]; let s1 = src[1]; let s2 = src[2]; let s3 = src[3];
    let s4 = src[4]; let s5 = src[5]; let s6 = src[6]; let s7 = src[7];

    // Even-part inputs: sums
    let e0 = s0 + s7; let e1 = s1 + s6; let e2 = s2 + s5; let e3 = s3 + s4;
    // Odd-part inputs: differences
    let o0 = s0 - s7; let o1 = s1 - s6; let o2 = s2 - s5; let o3 = s3 - s4;

    // Even outputs (indices 0,2,4,6) — 4-point DCT on sums
    // E_ROW[0] → DCT[0] (k=0), E_ROW[1] → DCT[2] (k=1),
    // E_ROW[2] → DCT[4] (k=2), E_ROW[3] → DCT[6] (k=3)
    let y0 = E_ROW[0][0]*e0 + E_ROW[0][1]*e1 + E_ROW[0][2]*e2 + E_ROW[0][3]*e3;
    let y2 = E_ROW[1][0]*e0 + E_ROW[1][1]*e1 + E_ROW[1][2]*e2 + E_ROW[1][3]*e3;
    let y4 = E_ROW[2][0]*e0 + E_ROW[2][1]*e1 + E_ROW[2][2]*e2 + E_ROW[2][3]*e3;
    let y6 = E_ROW[3][0]*e0 + E_ROW[3][1]*e1 + E_ROW[3][2]*e2 + E_ROW[3][3]*e3;

    // Odd outputs (indices 1,3,5,7) — 4-point DST on diffs
    let y1 = O_ROW[0][0]*o0 + O_ROW[0][1]*o1 + O_ROW[0][2]*o2 + O_ROW[0][3]*o3;
    let y3 = O_ROW[1][0]*o0 + O_ROW[1][1]*o1 + O_ROW[1][2]*o2 + O_ROW[1][3]*o3;
    let y5 = O_ROW[2][0]*o0 + O_ROW[2][1]*o1 + O_ROW[2][2]*o2 + O_ROW[2][3]*o3;
    let y7 = O_ROW[3][0]*o0 + O_ROW[3][1]*o1 + O_ROW[3][2]*o2 + O_ROW[3][3]*o3;

    [y0, y1, y2, y3, y4, y5, y6, y7]
}

/// 1‑D column‑pass DCT (0.25 scaling fused) — even/odd decomposition.
#[inline(always)]
fn fdct_1d_col(src: &[f16; 8]) -> [f16; 8] {
    let s0 = src[0]; let s1 = src[1]; let s2 = src[2]; let s3 = src[3];
    let s4 = src[4]; let s5 = src[5]; let s6 = src[6]; let s7 = src[7];

    let e0 = s0 + s7; let e1 = s1 + s6; let e2 = s2 + s5; let e3 = s3 + s4;
    let o0 = s0 - s7; let o1 = s1 - s6; let o2 = s2 - s5; let o3 = s3 - s4;

    // Even outputs with 0.25 scaling
    // E_COL[0] → DCT[0], E_COL[1] → DCT[2],
    // E_COL[2] → DCT[4], E_COL[3] → DCT[6]
    let y0 = E_COL[0][0]*e0 + E_COL[0][1]*e1 + E_COL[0][2]*e2 + E_COL[0][3]*e3;
    let y2 = E_COL[1][0]*e0 + E_COL[1][1]*e1 + E_COL[1][2]*e2 + E_COL[1][3]*e3;
    let y4 = E_COL[2][0]*e0 + E_COL[2][1]*e1 + E_COL[2][2]*e2 + E_COL[2][3]*e3;
    let y6 = E_COL[3][0]*e0 + E_COL[3][1]*e1 + E_COL[3][2]*e2 + E_COL[3][3]*e3;

    // Odd outputs with 0.25 scaling
    let y1 = O_COL[0][0]*o0 + O_COL[0][1]*o1 + O_COL[0][2]*o2 + O_COL[0][3]*o3;
    let y3 = O_COL[1][0]*o0 + O_COL[1][1]*o1 + O_COL[1][2]*o2 + O_COL[1][3]*o3;
    let y5 = O_COL[2][0]*o0 + O_COL[2][1]*o1 + O_COL[2][2]*o2 + O_COL[2][3]*o3;
    let y7 = O_COL[3][0]*o0 + O_COL[3][1]*o1 + O_COL[3][2]*o2 + O_COL[3][3]*o3;

    [y0, y1, y2, y3, y4, y5, y6, y7]
}

// ──────────────────────────────────────────────────────
// LLM DCT (behind cfg feature "llm_aan")
// ──────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────
// 1‑D DCT coefficient matrix (kept for test verification)
// ──────────────────────────────────────────────────────

/// 1‑D DCT coefficient matrix: C[u][x] = cos((2·x+1)·u·π/16)
const C_MATRIX: [[f16; 8]; 8] = [
    [f16::from_f32_const(1.0), f16::from_f32_const(1.0), f16::from_f32_const(1.0), f16::from_f32_const(1.0), f16::from_f32_const(1.0), f16::from_f32_const(1.0), f16::from_f32_const(1.0), f16::from_f32_const(1.0)],
    [f16::from_f32_const(0.98078528), f16::from_f32_const(0.83146961), f16::from_f32_const(0.55557023), f16::from_f32_const(0.19509032), f16::from_f32_const(-0.19509032), f16::from_f32_const(-0.55557023), f16::from_f32_const(-0.83146961), f16::from_f32_const(-0.98078528)],
    [f16::from_f32_const(0.92387953), f16::from_f32_const(0.38268343), f16::from_f32_const(-0.38268343), f16::from_f32_const(-0.92387953), f16::from_f32_const(-0.92387953), f16::from_f32_const(-0.38268343), f16::from_f32_const(0.38268343), f16::from_f32_const(0.92387953)],
    [f16::from_f32_const(0.83146961), f16::from_f32_const(-0.19509032), f16::from_f32_const(-0.98078528), f16::from_f32_const(-0.55557023), f16::from_f32_const(0.55557023), f16::from_f32_const(0.98078528), f16::from_f32_const(0.19509032), f16::from_f32_const(-0.83146961)],
    [f16::from_f32_const(0.70710678), f16::from_f32_const(-0.70710678), f16::from_f32_const(-0.70710678), f16::from_f32_const(0.70710678), f16::from_f32_const(0.70710678), f16::from_f32_const(-0.70710678), f16::from_f32_const(-0.70710678), f16::from_f32_const(0.70710678)],
    [f16::from_f32_const(0.55557023), f16::from_f32_const(-0.98078528), f16::from_f32_const(0.19509032), f16::from_f32_const(0.83146961), f16::from_f32_const(-0.83146961), f16::from_f32_const(-0.19509032), f16::from_f32_const(0.98078528), f16::from_f32_const(-0.55557023)],
    [f16::from_f32_const(0.38268343), f16::from_f32_const(-0.92387953), f16::from_f32_const(0.92387953), f16::from_f32_const(-0.38268343), f16::from_f32_const(-0.38268343), f16::from_f32_const(0.92387953), f16::from_f32_const(-0.92387953), f16::from_f32_const(0.38268343)],
    [f16::from_f32_const(0.19509032), f16::from_f32_const(-0.55557023), f16::from_f32_const(0.83146961), f16::from_f32_const(-0.98078528), f16::from_f32_const(0.98078528), f16::from_f32_const(-0.83146961), f16::from_f32_const(0.55557023), f16::from_f32_const(-0.19509032)],
];

// ──────────────────────────────────────────────────────
// 2‑D transforms
// ──────────────────────────────────────────────────────

/// Forward DCT — separable row‑column implementation
/// Complexity: 8·64 + 8·64 = 1024 multiply‑adds  (vs 4096 for naive)
pub fn fdct_2d(block: &mut [f16; 64]) {
    let mut tmp = [f16::ZERO; 64];

    // 1) 1‑D DCT on every row
    let mut off = 0;
    while off < 64 {
        let row = [
            block[off], block[off+1], block[off+2], block[off+3],
            block[off+4], block[off+5], block[off+6], block[off+7],
        ];
        let d = fdct_1d(&row);
        tmp[off]     = d[0];
        tmp[off+1]   = d[1];
        tmp[off+2]   = d[2];
        tmp[off+3]   = d[3];
        tmp[off+4]   = d[4];
        tmp[off+5]   = d[5];
        tmp[off+6]   = d[6];
        tmp[off+7]   = d[7];
        off += 8;
    }

    // 2) 1‑D DCT on every column, using column‑pass matrix
    //    which includes the 0.25 total scaling (fused — no extra multiply).
    for u in 0..8 {
        let col = [
            tmp[u], tmp[8 + u], tmp[16 + u], tmp[24 + u],
            tmp[32 + u], tmp[40 + u], tmp[48 + u], tmp[56 + u],
        ];
        let d = fdct_1d_col(&col);
        block[u]      = d[0];
        block[8 + u]  = d[1];
        block[16 + u] = d[2];
        block[24 + u] = d[3];
        block[32 + u] = d[4];
        block[40 + u] = d[5];
        block[48 + u] = d[6];
        block[56 + u] = d[7];
    }
}

/// Naïve 2‑D IDCT (used only in tests for roundtrip verification)
pub fn idct_2d_naive(block: &mut [f16; 64]) {
    let mut temp = [f16::ZERO; 64];

    for x in 0..8 {
        for y in 0..8 {
            let mut sum = f16::ZERO;
            for u in 0..8 {
                for v in 0..8 {
                    let cu = if u == 0 { f16::from_f32(0.70710678) } else { f16::ONE };
                    let cv = if v == 0 { f16::from_f32(0.70710678) } else { f16::ONE };
                    sum += cu * cv * block[u * 8 + v]
                         * C_MATRIX[u][x]  // cos((2x+1)uπ/16)
                         * C_MATRIX[v][y]; // cos((2y+1)vπ/16)
                }
            }
            temp[x * 8 + y] = f16::from_f32(0.25) * sum;
        }
    }
    *block = temp;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_abs(x: f16) -> f16 {
        f16::from_bits(x.to_bits() & 0x7FFF)
    }

    #[test]
    fn test_fdct_idct_roundtrip() {
        let orig: [f16; 64] = core::array::from_fn(|i| {
            let row = i / 8;
            let col = i % 8;
            f16::from_f32(row as f32 * col as f32 % 256.0 - 128.0)
        });
        let mut block = orig;
        fdct_2d(&mut block);
        idct_2d_naive(&mut block);
        for i in 0..64 {
            assert!(f16_abs(block[i] - orig[i]) < f16::from_f32(2.0),
                "roundtrip error at {i}: got {}, expected {}", block[i], orig[i]);
        }
    }

    #[test]
    fn test_fdct_identity() {
        let mut block = [f16::ONE; 64];
        fdct_2d(&mut block);
        assert!(f16_abs(block[0] - f16::from_f32(8.0)) < f16::from_f32(0.01), "DC value should be 8.0, got {}", block[0]);
        for i in 1..64 {
            assert!(f16_abs(block[i]) < f16::from_f32(0.01), "AC value at {i} should be 0, got {}", block[i]);
        }
    }
}
