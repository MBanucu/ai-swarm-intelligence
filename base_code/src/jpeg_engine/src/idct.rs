// ──────────────────────────────────────────────────────
// Optimised 1‑D IDCT — butterfly-even / direct-odd
//
// The 8×8 IDCT matrix is decomposed into two 4×4 sub‑matrices
// for the even‑indexed (0,2,4,6) and odd‑indexed (1,3,5,7) columns,
// exploiting the output symmetry of the DCT‑II basis.
//
//   Even outputs:  e[0..3] = E_4×4 × src[0,2,4,6]
//   Odd outputs:   o[0..3] = O_4×4 × src[1,3,5,7]
//   Final:  out[x] = e[x] + o[x],  out[7−x] = e[x] − o[x]
//
// The EVEN 4‑point sub‑matrix uses a butterfly structure (6 mults)
// instead of a full 4×4 mat‑vec (16 mults):
//   E = [a, b, a, c]   where a = 1/(2√2), b = cos(π/8)/2, c = cos(3π/8)/2
//       [a, c,-a,-b]
//       [a,-c,-a, b]
//       [a,-b, a,-c]
//
// The ODD sub‑matrix is a full 4×4 mat‑vec (16 mults) — this matrix
// does not factor into simple pairwise rotations, so we keep the
// direct formulation for numerical accuracy.
//
// Arithmetic: 22 mul + ~32 add per 1‑D transform
// (vs 64 mul + 56 add for dense 8×8 mat‑vec, or 32 mul + 40 add
//  for old even/odd with full 4×4 in both sub‑matrices).
// ──────────────────────────────────────────────────────

/// Even‑column 4×4 sub‑matrix constants — butterfly factors.
/// a = 1/(2√2) ≈ 0.35355339
/// b = cos(π/8)/2 ≈ 0.46193977
/// c = cos(3π/8)/2 ≈ 0.19134172
const E_A: f32 = 0.35355339059327376;
const E_B: f32 = 0.46193976625564337;
const E_C: f32 = 0.19134171618254492;

/// Odd‑column 4×4 sub‑matrix (columns 1,3,5,7).
const O: [[f32; 4]; 4] = [
    [0.49039264020161522, 0.41573480615127262, 0.27778511650980114, 0.09754516100806417],
    [0.41573480615127262,-0.09754516100806417,-0.49039264020161522,-0.27778511650980114],
    [0.27778511650980114,-0.49039264020161522, 0.09754516100806417, 0.41573480615127262],
    [0.09754516100806417,-0.27778511650980114, 0.41573480615127262,-0.49039264020161522],
];

/// 1‑D IDCT — butterfly-even, direct-odd decomposition.
///
/// Even part uses 6 mults (vs 16 for direct 4×4 mat-vec).
/// Odd part uses full 4×4 mat-vec (16 mults).
/// Total: 22 mults + ~32 adds per 1‑D transform.
#[inline(always)]
fn idct_1d(src: &[f32; 8]) -> [f32; 8] {
    let s0 = src[0]; let s1 = src[1]; let s2 = src[2]; let s3 = src[3];
    let s4 = src[4]; let s5 = src[5]; let s6 = src[6]; let s7 = src[7];

    // ── Even part: butterfly 4‑point IDCT (6 mults + 8 adds) ──
    let sum_even  = s0 + s4;    // F0 + F4
    let diff_even = s0 - s4;    // F0 − F4

    // 4 multiplications for cross-terms (b·F2, c·F6, c·F2, b·F6)
    let b_s2 = E_B * s2;
    let c_s6 = E_C * s6;
    let c_s2 = E_C * s2;
    let b_s6 = E_B * s6;

    let even_cross_plus  = b_s2 + c_s6;   // b·F2 + c·F6
    let even_cross_minus = c_s2 - b_s6;   // c·F2 − b·F6

    // 2 multiplications for DC-like terms
    let a_sum  = E_A * sum_even;    // a·(F0+F4)
    let a_diff = E_A * diff_even;   // a·(F0−F4)

    let e0 = a_sum + even_cross_plus;
    let e1 = a_diff + even_cross_minus;
    let e2 = a_diff - even_cross_minus;
    let e3 = a_sum - even_cross_plus;

    // ── Odd part: full 4×4 mat‑vec on (s1, s3, s5, s7) ──
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

// ──────────────────────────────────────────────────────
// 2‑D IDCT — separable row‑column implementation
// ──────────────────────────────────────────────────────

/// 32‑byte aligned buffer — AVX loads/stores require 32‑byte alignment
/// for best performance (`vmovapd` / `vmovaps`).  Even on CPUs that handle
/// unaligned access efficiently, aligned loads avoid the µop penalty in
/// certain micro‑architectures (Sandy/Ivy Bridge, early Haswell).
#[repr(align(32))]
struct AlignedF64<const N: usize>([f32; N]);

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
pub fn idct_2d(block: &mut [f32; 64]) {
    // 32‑byte aligned temporary buffer for transposed row results.
    // Alignment guarantees AVX loads can use `vmovapd` instead of `vmovupd`,
    // which avoids a ~50% penalty on some micro‑architectures.
    let mut tmp = AlignedF64::<64>([0.0f32; 64]);
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
    // tmp[x*8 + u] = G[u][x], so we read 8 consecutive f32s
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
pub fn batch_idct_2d(blocks: &mut [[f32; 64]]) {
    let n = blocks.len();
    if n == 0 {
        return;
    }
    // For small-medium batches, sequential processing avoids rayon's
    // thread-pool wake-up overhead (~20-100 µs).  Only use parallel
    // dispatch for batches where the speedup outweighs the fixed cost.
    if n < 4096 {
        let ptr = blocks.as_mut_ptr();
        for i in 0..n {
            unsafe { idct_2d(&mut *ptr.add(i)); }
        }
        return;
    }
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        // Adaptive chunking: target ~4 chunks per CPU core so tasks stay
        // large enough for L1 cache reuse but numerous enough to balance
        // load.  Minimum chunk size of 128 prevents excessive task overhead.
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let chunk_size = (n / (num_threads * 4)).max(128);
        blocks.par_chunks_mut(chunk_size).for_each(|chunk| {
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
        let mut block = [0.0f32; 64];
        idct_2d(&mut block);
        for &val in &block {
            assert!(val.abs() < 0.001, "expected near-zero, got {}", val);
        }
    }

    #[test]
    fn test_idct_dc_only() {
        let mut block = [0.0f32; 64];
        block[0] = 8.0;
        idct_2d(&mut block);
        for &val in &block {
            assert!((val - 1.0).abs() < 0.001, "DC-only IDCT: expected 1.0, got {}", val);
        }
    }
}
