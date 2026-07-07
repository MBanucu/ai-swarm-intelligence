// ──────────────────────────────────────────────────────
// Pre‑computed 1‑D IDCT matrix (scale‑fused)
//
//   IDCT1D[x][u] = 0.5 * C(u) * cos((2x+1)uπ/16)
//   where C(0) = 1/√2, C(u≠0) = 1
// ──────────────────────────────────────────────────────

/// IDCT matrix stored row-major: IDCT1D[x][u].
/// Constant data — lives in .rodata, fully visible to LLVM.
const IDCT1D: [[f64; 8]; 8] = [
    [3.53553390593273730858e-01, 4.90392640201615215290e-01, 4.61939766255643369242e-01, 4.15734806151272617836e-01, 3.53553390593273786369e-01, 2.77785116509801144336e-01, 1.91341716182544918645e-01, 9.75451610080641656753e-02],
    [3.53553390593273730858e-01, 4.15734806151272617836e-01, 1.91341716182544918645e-01, -9.75451610080640962863e-02, -3.53553390593273730858e-01, -4.90392640201615215290e-01, -4.61939766255643424753e-01, -2.77785116509801088824e-01],
    [3.53553390593273730858e-01, 2.77785116509801144336e-01, -1.91341716182544863134e-01, -4.90392640201615215290e-01, -3.53553390593273841880e-01, 9.75451610080641517975e-02, 4.61939766255643258219e-01, 4.15734806151272728858e-01],
    [3.53553390593273730858e-01, 9.75451610080641656753e-02, -4.61939766255643369242e-01, -2.77785116509801088824e-01, 3.53553390593273675346e-01, 4.15734806151272728858e-01, -1.91341716182544946401e-01, -4.90392640201615326312e-01],
    [3.53553390593273730858e-01, -9.75451610080640962863e-02, -4.61939766255643424753e-01, 2.77785116509800922291e-01, 3.53553390593273841880e-01, -4.15734806151272562325e-01, -1.91341716182545279468e-01, 4.90392640201615215290e-01],
    [3.53553390593273730858e-01, -2.77785116509800977802e-01, -1.91341716182545168445e-01, 4.90392640201615215290e-01, -3.53553390593273342279e-01, -9.75451610080640130196e-02, 4.61939766255643369242e-01, -4.15734806151272506813e-01],
    [3.53553390593273730858e-01, -4.15734806151272673347e-01, 1.91341716182545001912e-01, 9.75451610080643877199e-02, -3.53553390593273564324e-01, 4.90392640201615326312e-01, -4.61939766255643202708e-01, 2.77785116509800755757e-01],
    [3.53553390593273730858e-01, -4.90392640201615215290e-01, 4.61939766255643258219e-01, -4.15734806151272562325e-01, 3.53553390593273286768e-01, -2.77785116509800755757e-01, 1.91341716182544779867e-01, -9.75451610080642905753e-02],
];

/// 1‑D IDCT — fully unrolled matrix-vector multiply, return by value.
#[inline(always)]
fn idct_1d(src: &[f64; 8]) -> [f64; 8] {
    let r0 = &IDCT1D[0]; let r1 = &IDCT1D[1]; let r2 = &IDCT1D[2]; let r3 = &IDCT1D[3];
    let r4 = &IDCT1D[4]; let r5 = &IDCT1D[5]; let r6 = &IDCT1D[6]; let r7 = &IDCT1D[7];
    let s0 = src[0]; let s1 = src[1]; let s2 = src[2]; let s3 = src[3];
    let s4 = src[4]; let s5 = src[5]; let s6 = src[6]; let s7 = src[7];
    [
        s0*r0[0] + s1*r0[1] + s2*r0[2] + s3*r0[3] + s4*r0[4] + s5*r0[5] + s6*r0[6] + s7*r0[7],
        s0*r1[0] + s1*r1[1] + s2*r1[2] + s3*r1[3] + s4*r1[4] + s5*r1[5] + s6*r1[6] + s7*r1[7],
        s0*r2[0] + s1*r2[1] + s2*r2[2] + s3*r2[3] + s4*r2[4] + s5*r2[5] + s6*r2[6] + s7*r2[7],
        s0*r3[0] + s1*r3[1] + s2*r3[2] + s3*r3[3] + s4*r3[4] + s5*r3[5] + s6*r3[6] + s7*r3[7],
        s0*r4[0] + s1*r4[1] + s2*r4[2] + s3*r4[3] + s4*r4[4] + s5*r4[5] + s6*r4[6] + s7*r4[7],
        s0*r5[0] + s1*r5[1] + s2*r5[2] + s3*r5[3] + s4*r5[4] + s5*r5[5] + s6*r5[6] + s7*r5[7],
        s0*r6[0] + s1*r6[1] + s2*r6[2] + s3*r6[3] + s4*r6[4] + s5*r6[5] + s6*r6[6] + s7*r6[7],
        s0*r7[0] + s1*r7[1] + s2*r7[2] + s3*r7[3] + s4*r7[4] + s5*r7[5] + s6*r7[6] + s7*r7[7],
    ]
}

// ──────────────────────────────────────────────────────
// 2‑D IDCT — separable row‑column implementation
// ──────────────────────────────────────────────────────

/// Inverse DCT — separable row‑column implementation.
///
/// # Gen‑4 Mutation: Transposed temp buffer layout
///
/// The row pass writes its result to `tmp` in **transposed** order
/// (i.e. `tmp[k*8 + y] = G[y][k]`).  This makes the column-pass
/// reads unit‑stride — all 8 elements of a column live in
/// `tmp[x*8 .. x*8+8]`, i.e. one or two contiguous 64‑B cache lines
/// instead of 8 strided lines.
///
/// The column pass still writes to `block` in the standard row‑major
/// layout (same as before).
pub fn idct_2d(block: &mut [f64; 64]) {
    let mut tmp = [0.0f64; 64];

    // ── Pass 1: IDCT on rows, store TRANSPOSED into tmp ──
    // tmp[k*8 + y] = G[y][k] (column‑major temp)
    for y in 0..8 {
        let row = [
            block[y*8], block[y*8 + 1], block[y*8 + 2], block[y*8 + 3],
            block[y*8 + 4], block[y*8 + 5], block[y*8 + 6], block[y*8 + 7],
        ];
        let d = idct_1d(&row);
        // Transposed store: d[k] → tmp[k*8 + y]
        tmp[y]       = d[0];
        tmp[8 + y]   = d[1];
        tmp[16 + y]  = d[2];
        tmp[24 + y]  = d[3];
        tmp[32 + y]  = d[4];
        tmp[40 + y]  = d[5];
        tmp[48 + y]  = d[6];
        tmp[56 + y]  = d[7];
    }

    // ── Pass 2: IDCT on columns — unit‑stride reads from tmp ──
    // tmp[x*8 + u] = G[u][x], so we read 8 consecutive f64s
    for x in 0..8 {
        let off = x << 3;  // x * 8
        let col = [
            tmp[off], tmp[off+1], tmp[off+2], tmp[off+3],
            tmp[off+4], tmp[off+5], tmp[off+6], tmp[off+7],
        ];
        let d = idct_1d(&col);
        // Store to column x of block (standard row‑major output)
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
