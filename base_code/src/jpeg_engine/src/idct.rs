// ──────────────────────────────────────────────────────
// Gen‑5 mutation: Even/odd decomposition for 1‑D IDCT
//
// Instead of a full 8×8 matrix‑vector multiply (64 mul+56 add),
// exploit cosine symmetry: cos((2x+1)(8-u)π/16) = ±cos((2x+1)uπ/16).
// This lets us compute f[x] and f[7-x] from a single even+odd pair,
// cutting operations ≈47% (32 mul + 28 add + 4 sub per 1‑D pass).
//
// The tmp buffer is declared with 64‑byte alignment so AVX loads/stores
// never cross cache‑line boundaries.
// ──────────────────────────────────────────────────────

use crate::AlignedBlock;

// ──────────────────────────────────────────────────────
// Pre‑computed 4×4 even/odd sub‑matrices
//
// Extracted from the full 8×8 IDCT1D matrix row‑major layout:
//   IDCT1D[x][u] = 0.5 · C[u] · cos((2x+1)uπ/16)
// with C[0]=1/√2, C[u≠0]=1.
//
//   EVEN[x][k] = IDCT1D[x][2k]   for x=0..3, k=0..3
//   ODD[x][k]  = IDCT1D[x][2k+1]
// ──────────────────────────────────────────────────────

const EVEN: [[f64; 4]; 4] = [
    [3.53553390593273730858e-01, 4.61939766255643369242e-01, 3.53553390593273786369e-01, 1.91341716182544918645e-01],
    [3.53553390593273730858e-01, 1.91341716182544918645e-01, -3.53553390593273730858e-01, -4.61939766255643424753e-01],
    [3.53553390593273730858e-01, -1.91341716182544863134e-01, -3.53553390593273841880e-01, 4.61939766255643258219e-01],
    [3.53553390593273730858e-01, -4.61939766255643369242e-01, 3.53553390593273675346e-01, -1.91341716182544946401e-01],
];

const ODD: [[f64; 4]; 4] = [
    [4.90392640201615215290e-01, 4.15734806151272617836e-01, 2.77785116509801144336e-01, 9.75451610080641656753e-02],
    [4.15734806151272617836e-01, -9.75451610080640962863e-02, -4.90392640201615215290e-01, -2.77785116509801088824e-01],
    [2.77785116509801144336e-01, -4.90392640201615215290e-01, 9.75451610080641517975e-02, 4.15734806151272728858e-01],
    [9.75451610080641656753e-02, -2.77785116509801088824e-01, 4.15734806151272728858e-01, -4.90392640201615326312e-01],
];

// ──────────────────────────────────────────────────────
// 1‑D IDCT — even/odd decomposition
//
// For each x in 0..3 we compute:
//   even(x) = Σ_{k=0..3} F[2k]   · EVEN[x][k]
//   odd(x)  = Σ_{k=0..3} F[2k+1] · ODD[x][k]
//   f[x]    = even(x) + odd(x)
//   f[7-x]  = even(x) - odd(x)
//
// Unrolled: 32 mul + 28 add + 4 sub (vs 64 mul + 56 add).
// ──────────────────────────────────────────────────────
#[inline(always)]
fn idct_1d(src: &[f64; 8]) -> [f64; 8] {
    let s0 = src[0]; let s1 = src[1]; let s2 = src[2]; let s3 = src[3];
    let s4 = src[4]; let s5 = src[5]; let s6 = src[6]; let s7 = src[7];

    // x = 0 → pair (0, 7)
    let e0 = s0*EVEN[0][0] + s2*EVEN[0][1] + s4*EVEN[0][2] + s6*EVEN[0][3];
    let o0 = s1*ODD[0][0]  + s3*ODD[0][1]  + s5*ODD[0][2]  + s7*ODD[0][3];

    // x = 1 → pair (1, 6)
    let e1 = s0*EVEN[1][0] + s2*EVEN[1][1] + s4*EVEN[1][2] + s6*EVEN[1][3];
    let o1 = s1*ODD[1][0]  + s3*ODD[1][1]  + s5*ODD[1][2]  + s7*ODD[1][3];

    // x = 2 → pair (2, 5)
    let e2 = s0*EVEN[2][0] + s2*EVEN[2][1] + s4*EVEN[2][2] + s6*EVEN[2][3];
    let o2 = s1*ODD[2][0]  + s3*ODD[2][1]  + s5*ODD[2][2]  + s7*ODD[2][3];

    // x = 3 → pair (3, 4)
    let e3 = s0*EVEN[3][0] + s2*EVEN[3][1] + s4*EVEN[3][2] + s6*EVEN[3][3];
    let o3 = s1*ODD[3][0]  + s3*ODD[3][1]  + s5*ODD[3][2]  + s7*ODD[3][3];

    [
        e0 + o0,  // f[0]
        e1 + o1,  // f[1]
        e2 + o2,  // f[2]
        e3 + o3,  // f[3]
        e3 - o3,  // f[4]
        e2 - o2,  // f[5]
        e1 - o1,  // f[6]
        e0 - o0,  // f[7]
    ]
}

// ──────────────────────────────────────────────────────
// 2‑D IDCT — separable row‑column with transposed temp
// ──────────────────────────────────────────────────────

/// Inverse DCT — row‑column separable.
///
/// Row pass → transposed store into aligned tmp (unit‑stride
/// column reads in pass 2), then column pass → row‑major output.
pub fn idct_2d(block: &mut [f64; 64]) {
    let mut storage = AlignedBlock([0.0f64; 64]);
    let tmp: &mut [f64; 64] = &mut storage.0;

    // ── Pass 1: 1‑D IDCT on rows, store transposed ──
    // tmp[k*8 + y] = G[y][k] (column‑major layout)
    for y in 0..8 {
        let row = [
            block[y*8], block[y*8 + 1], block[y*8 + 2], block[y*8 + 3],
            block[y*8 + 4], block[y*8 + 5], block[y*8 + 6], block[y*8 + 7],
        ];
        let d = idct_1d(&row);
        // Transposed store — 1 cache‑line per column slice
        tmp[y]     = d[0];
        tmp[8 + y]   = d[1];
        tmp[16 + y]  = d[2];
        tmp[24 + y]  = d[3];
        tmp[32 + y]  = d[4];
        tmp[40 + y]  = d[5];
        tmp[48 + y]  = d[6];
        tmp[56 + y]  = d[7];
    }

    // ── Pass 2: 1‑D IDCT on columns (unit‑stride reads) ──
    for x in 0..8 {
        let off = x << 3;  // x * 8
        let col = [
            tmp[off], tmp[off+1], tmp[off+2], tmp[off+3],
            tmp[off+4], tmp[off+5], tmp[off+6], tmp[off+7],
        ];
        let d = idct_1d(&col);
        // Store to column x (standard row‑major)
        block[x]      = d[0];
        block[8 + x]  = d[1];
        block[16 + x] = d[2];
        block[24 + x] = d[3];
        block[32 + x] = d[4];
        block[40 + x] = d[5];
        block[48 + x] = d[6];
        block[56 + x] = d[7];
    }
}

/// Multi‑block batched IDCT — processes blocks via raw pointer.
pub fn batch_idct_2d(blocks: &mut [[f64; 64]]) {
    let n = blocks.len();
    if n == 0 {
        return;
    }
    let ptr = blocks.as_mut_ptr();
    for i in 0..n {
        unsafe { idct_2d(&mut *ptr.add(i)); }
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
