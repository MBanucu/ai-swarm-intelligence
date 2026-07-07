// ──────────────────────────────────────────────────────
// Optimised 1‑D IDCT — even/odd decomposition
//
// The 8×8 IDCT matrix is decomposed into two 4×4 sub‑matrices
// for the even‑indexed (0,2,4,6) and odd‑indexed (1,3,5,7) columns,
// exploiting the output symmetry of the DCT‑II basis.
//
//   Even outputs:  e[0..3] = E_4×4 × src[0,2,4,6]
//   Odd outputs:   o[0..3] = O_4×4 × src[1,3,5,7]
//   Final:  out[x] = e[x] + o[x],  out[7−x] = e[x] − o[x]
//
// Arithmetic: 32 mul + 40 add per 1‑D transform
// (vs 64 mul + 56 add for dense 8×8 mat‑vec).
// ──────────────────────────────────────────────────────

/// Even‑column 4×4 sub‑matrix of the IDCT (columns 0,2,4,6).
const E: [[f64; 4]; 4] = [
    [0.35355339059327376, 0.46193976625564337, 0.35355339059327376, 0.19134171618254492],
    [0.35355339059327376, 0.19134171618254492,-0.35355339059327376,-0.46193976625564337],
    [0.35355339059327376,-0.19134171618254492,-0.35355339059327376, 0.46193976625564337],
    [0.35355339059327376,-0.46193976625564337, 0.35355339059327376,-0.19134171618254492],
];

/// Odd‑column 4×4 sub‑matrix (columns 1,3,5,7).
const O: [[f64; 4]; 4] = [
    [0.49039264020161522, 0.41573480615127262, 0.27778511650980114, 0.09754516100806417],
    [0.41573480615127262,-0.09754516100806417,-0.49039264020161522,-0.27778511650980114],
    [0.27778511650980114,-0.49039264020161522, 0.09754516100806417, 0.41573480615127262],
    [0.09754516100806417,-0.27778511650980114, 0.41573480615127262,-0.49039264020161522],
];

/// 1‑D IDCT — even/odd decomposition.
///
/// The 4×4 mat-vec is written as plain scalar loops so LLVM's auto-vectorizer
/// can generate optimal SIMD (AVX/FMA when target features are enabled via
/// `.cargo/config.toml`). Manual intrinsics would block other compiler
/// optimisations like loop unrolling, register renaming, and instruction
/// scheduling.
///
/// When AVX2 + FMA are detected at runtime (important for CI / pre-built
/// binaries), an explicit SIMD path via `_mm256_fmadd_pd` is used instead.
#[inline(always)]
fn idct_1d(src: &[f64; 8]) -> [f64; 8] {
    // Runtime feature detection — catches systems where the binary was
    // compiled for generic x86-64 but the host supports AVX2+FMA.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return idct_1d_avx(src);
        }
    }
    idct_1d_scalar(src)
}

/// Scalar 1‑D IDCT reference (always available).
#[inline(always)]
fn idct_1d_scalar(src: &[f64; 8]) -> [f64; 8] {
    let s0 = src[0]; let s1 = src[1]; let s2 = src[2]; let s3 = src[3];
    let s4 = src[4]; let s5 = src[5]; let s6 = src[6]; let s7 = src[7];

    // Even part:  4×4 mat‑vec on (s0, s2, s4, s6)
    let e0 = E[0][0]*s0 + E[0][1]*s2 + E[0][2]*s4 + E[0][3]*s6;
    let e1 = E[1][0]*s0 + E[1][1]*s2 + E[1][2]*s4 + E[1][3]*s6;
    let e2 = E[2][0]*s0 + E[2][1]*s2 + E[2][2]*s4 + E[2][3]*s6;
    let e3 = E[3][0]*s0 + E[3][1]*s2 + E[3][2]*s4 + E[3][3]*s6;

    // Odd part:  4×4 mat‑vec on (s1, s3, s5, s7)
    let o0 = O[0][0]*s1 + O[0][1]*s3 + O[0][2]*s5 + O[0][3]*s7;
    let o1 = O[1][0]*s1 + O[1][1]*s3 + O[1][2]*s5 + O[1][3]*s7;
    let o2 = O[2][0]*s1 + O[2][1]*s3 + O[2][2]*s5 + O[2][3]*s7;
    let o3 = O[3][0]*s1 + O[3][1]*s3 + O[3][2]*s5 + O[3][3]*s7;

    // Combine using symmetry: out[x] = e[x]+o[x], out[7-x] = e[x]-o[x]
    [
        e0 + o0, e1 + o1, e2 + o2, e3 + o3,
        e3 - o3, e2 - o2, e1 - o1, e0 - o0,
    ]
}

/// AVX2 + FMA accelerated 1‑D IDCT using explicit `_mm256_fmadd_pd`.
///
/// The 4×4 even and odd sub‑matrix multiplies are each computed with
/// 4 FMAs (fused multiply‑add) — accumulating one column of the sub‑matrix
/// per FMA.  This reduces instruction count from 32 mul + 40 add to
/// 8 FMA + 4 add + 4 sub, nearly 2× arithmetic density vs scalar.
///
/// The result is stored as two 256‑bit writes (8 f64s = 64 bytes).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn idct_1d_avx(src: &[f64; 8]) -> [f64; 8] {
    use core::arch::x86_64::*;

    unsafe {
        // ── Even part: accumulate E columns × even-indexed inputs ──
        // e accumulates [e0, e1, e2, e3] across 4 FMA operations.
        // Column 0 of E multiplied by s0 = src[0]
        let col0 = _mm256_set_pd(E[3][0], E[2][0], E[1][0], E[0][0]);
        let mut e = _mm256_mul_pd(col0, _mm256_set1_pd(src[0]));

        // Column 1 of E multiplied by s2 = src[2]
        let col1 = _mm256_set_pd(E[3][1], E[2][1], E[1][1], E[0][1]);
        e = _mm256_fmadd_pd(col1, _mm256_set1_pd(src[2]), e);

        // Column 2 of E multiplied by s4 = src[4]
        let col2 = _mm256_set_pd(E[3][2], E[2][2], E[1][2], E[0][2]);
        e = _mm256_fmadd_pd(col2, _mm256_set1_pd(src[4]), e);

        // Column 3 of E multiplied by s6 = src[6]
        let col3 = _mm256_set_pd(E[3][3], E[2][3], E[1][3], E[0][3]);
        e = _mm256_fmadd_pd(col3, _mm256_set1_pd(src[6]), e);

        // ── Odd part: accumulate O columns × odd-indexed inputs ──
        // o accumulates [o0, o1, o2, o3]
        let ocol0 = _mm256_set_pd(O[3][0], O[2][0], O[1][0], O[0][0]);
        let mut o = _mm256_mul_pd(ocol0, _mm256_set1_pd(src[1]));

        let ocol1 = _mm256_set_pd(O[3][1], O[2][1], O[1][1], O[0][1]);
        o = _mm256_fmadd_pd(ocol1, _mm256_set1_pd(src[3]), o);

        let ocol2 = _mm256_set_pd(O[3][2], O[2][2], O[1][2], O[0][2]);
        o = _mm256_fmadd_pd(ocol2, _mm256_set1_pd(src[5]), o);

        let ocol3 = _mm256_set_pd(O[3][3], O[2][3], O[1][3], O[0][3]);
        o = _mm256_fmadd_pd(ocol3, _mm256_set1_pd(src[7]), o);

        // Combine: out[x] = e[x]+o[x], out[7-x] = e[x]-o[x]
        // e[0..3] = [e0, e1, e2, e3], o[0..3] = [o0, o1, o2, o3]
        let plus  = _mm256_add_pd(e, o);  // [e0+o0, e1+o1, e2+o2, e3+o3]
        let minus = _mm256_sub_pd(e, o);  // [e0-o0, e1-o1, e2-o2, e3-o3]
        // Reverse lanes: [e3-o3, e2-o2, e1-o1, e0-o0]
        let rev_minus = _mm256_permute4x64_pd::<0b00011011>(minus);

        // Store: out[0..4] = plus,  out[4..8] = rev_minus
        let mut out = [0.0f64; 8];
        _mm256_storeu_pd(out.as_mut_ptr(), plus);
        _mm256_storeu_pd(out.as_mut_ptr().add(4), rev_minus);
        out
    }
}

// ──────────────────────────────────────────────────────
// 2‑D IDCT — separable row‑column implementation
// ──────────────────────────────────────────────────────

/// 32‑byte aligned buffer — AVX loads/stores require 32‑byte alignment
/// for best performance (`vmovapd` / `vmovaps`).  Even on CPUs that handle
/// unaligned access efficiently, aligned loads avoid the µop penalty in
/// certain micro‑architectures (Sandy/Ivy Bridge, early Haswell).
#[repr(align(32))]
struct AlignedF64<const N: usize>([f64; N]);

/// Inverse DCT — separable row‑column implementation.
///
/// # Gen‑4 Mutation (retained): Transposed temp buffer layout
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
    // 32‑byte aligned temporary buffer for transposed row results.
    // Alignment guarantees AVX loads can use `vmovapd` instead of `vmovupd`,
    // which avoids a ~50% penalty on some micro‑architectures.
    let mut tmp = AlignedF64::<64>([0.0f64; 64]);
    let tmp = &mut tmp.0;

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

/// Multi‑block batched IDCT — processes blocks with optional parallelism.
pub fn batch_idct_2d(blocks: &mut [[f64; 64]]) {
    let n = blocks.len();
    if n == 0 {
        return;
    }
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        // Process 4 blocks per worker chunk for good cache utilization
        blocks.par_chunks_mut(4).for_each(|chunk| {
            for block in chunk {
                idct_2d(block);
            }
        });
    }
    #[cfg(not(feature = "rayon"))]
    {
        let ptr = blocks.as_mut_ptr();
        for i in 0..n {
            unsafe { idct_2d(&mut *ptr.add(i)); }
        }
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
