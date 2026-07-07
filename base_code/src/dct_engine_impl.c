/* dct_engine_impl.c -- 8x8 2D IDCT with AVX2 intrinsics
 *
 * Standard separable approach: row transform -> transpose -> row transform
 * -> transpose.  Compile-time constant coefficients, AVX2 FMA, fully
 * unrolled for maximum throughput.  AVX2 in-register transpose eliminates
 * scalar transpose overhead.
 *
 * Coefficients: COEFFS[k*8+p] = 0.5 * alpha(k) * cos((2p+1)*k*pi/16)
 *   alpha(0) = 1/sqrt(2), alpha(k!=0) = 1
 *
 * Build:
 *   gcc -O3 -mavx2 -mfma -ffast-math -fno-math-errno \
 *       -ftree-vectorize -funroll-loops -flto -fPIC -shared \
 *       -o libdct_engine.so dct_engine_impl.c -lm
 */

#include <immintrin.h>
#include <math.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define S2I 0.707106781186547524400844362104849039  /* 1/sqrt(2) */

/* ---- Row 0: k=0, alpha=S2I, cos((2p+1)*0*pi/16) = cos(0) = 1 ---- */
#define C00 (0.5*S2I)
#define C01 (0.5*S2I)
#define C02 (0.5*S2I)
#define C03 (0.5*S2I)
#define C04 (0.5*S2I)
#define C05 (0.5*S2I)
#define C06 (0.5*S2I)
#define C07 (0.5*S2I)

/* ---- Row 1: k=1, alpha=1 ---- */
#define C10 (0.5*cos( 1.*M_PI/16.))
#define C11 (0.5*cos( 3.*M_PI/16.))
#define C12 (0.5*cos( 5.*M_PI/16.))
#define C13 (0.5*cos( 7.*M_PI/16.))
#define C14 (0.5*cos( 9.*M_PI/16.))
#define C15 (0.5*cos(11.*M_PI/16.))
#define C16 (0.5*cos(13.*M_PI/16.))
#define C17 (0.5*cos(15.*M_PI/16.))

/* ---- Row 2: k=2, alpha=1, cos((2p+1)*pi/8) ---- */
#define C20 (0.5*cos( 1.*M_PI/ 8.))
#define C21 (0.5*cos( 3.*M_PI/ 8.))
#define C22 (0.5*cos( 5.*M_PI/ 8.))
#define C23 (0.5*cos( 7.*M_PI/ 8.))
#define C24 (0.5*cos( 9.*M_PI/ 8.))
#define C25 (0.5*cos(11.*M_PI/ 8.))
#define C26 (0.5*cos(13.*M_PI/ 8.))
#define C27 (0.5*cos(15.*M_PI/ 8.))

/* ---- Row 3: k=3, alpha=1 ---- */
#define C30 (0.5*cos( 3.*M_PI/16.))
#define C31 (0.5*cos( 9.*M_PI/16.))
#define C32 (0.5*cos(15.*M_PI/16.))
#define C33 (0.5*cos(21.*M_PI/16.))
#define C34 (0.5*cos(27.*M_PI/16.))
#define C35 (0.5*cos(33.*M_PI/16.))
#define C36 (0.5*cos(39.*M_PI/16.))
#define C37 (0.5*cos(45.*M_PI/16.))

/* ---- Row 4: k=4, alpha=1, cos((2p+1)*pi/4) ---- */
#define C40 (0.5*cos( 1.*M_PI/ 4.))
#define C41 (0.5*cos( 3.*M_PI/ 4.))
#define C42 (0.5*cos( 5.*M_PI/ 4.))
#define C43 (0.5*cos( 7.*M_PI/ 4.))
#define C44 (0.5*cos( 9.*M_PI/ 4.))
#define C45 (0.5*cos(11.*M_PI/ 4.))
#define C46 (0.5*cos(13.*M_PI/ 4.))
#define C47 (0.5*cos(15.*M_PI/ 4.))

/* ---- Row 5: k=5, alpha=1 ---- */
#define C50 (0.5*cos( 5.*M_PI/16.))
#define C51 (0.5*cos(15.*M_PI/16.))
#define C52 (0.5*cos(25.*M_PI/16.))
#define C53 (0.5*cos(35.*M_PI/16.))
#define C54 (0.5*cos(45.*M_PI/16.))
#define C55 (0.5*cos(55.*M_PI/16.))
#define C56 (0.5*cos(65.*M_PI/16.))
#define C57 (0.5*cos(75.*M_PI/16.))

/* ---- Row 6: k=6, alpha=1, cos((2p+1)*3*pi/8) ---- */
#define C60 (0.5*cos( 3.*M_PI/ 8.))
#define C61 (0.5*cos( 9.*M_PI/ 8.))
#define C62 (0.5*cos(15.*M_PI/ 8.))
#define C63 (0.5*cos(21.*M_PI/ 8.))
#define C64 (0.5*cos(27.*M_PI/ 8.))
#define C65 (0.5*cos(33.*M_PI/ 8.))
#define C66 (0.5*cos(39.*M_PI/ 8.))
#define C67 (0.5*cos(45.*M_PI/ 8.))

/* ---- Row 7: k=7, alpha=1 ---- */
#define C70 (0.5*cos( 7.*M_PI/16.))
#define C71 (0.5*cos(21.*M_PI/16.))
#define C72 (0.5*cos(35.*M_PI/16.))
#define C73 (0.5*cos(49.*M_PI/16.))
#define C74 (0.5*cos(63.*M_PI/16.))
#define C75 (0.5*cos(77.*M_PI/16.))
#define C76 (0.5*cos(91.*M_PI/16.))
#define C77 (0.5*cos(105.*M_PI/16.))

/* 64 coefficients, row-major, 32-byte aligned */
static const double COEFFS[64] __attribute__((aligned(32))) = {
    C00,C01,C02,C03,C04,C05,C06,C07,
    C10,C11,C12,C13,C14,C15,C16,C17,
    C20,C21,C22,C23,C24,C25,C26,C27,
    C30,C31,C32,C33,C34,C35,C36,C37,
    C40,C41,C42,C43,C44,C45,C46,C47,
    C50,C51,C52,C53,C54,C55,C56,C57,
    C60,C61,C62,C63,C64,C65,C66,C67,
    C70,C71,C72,C73,C74,C75,C76,C77,
};

/* ---- 1D 8-point IDCT on a contiguous row ---- */
static inline void idct_1d(const double *__restrict__ in,
                            double *__restrict__ out) {
    __m256d acc0 = _mm256_setzero_pd();  /* p = 0..3 */
    __m256d acc1 = _mm256_setzero_pd();  /* p = 4..7 */

    #define STEP(k) do {                                                \
        __m256d inb = _mm256_set1_pd(in[(k)]);                          \
        acc0 = _mm256_fmadd_pd(inb, _mm256_load_pd(&COEFFS[(k)*8]), acc0); \
        acc1 = _mm256_fmadd_pd(inb, _mm256_load_pd(&COEFFS[(k)*8+4]), acc1); \
    } while (0)

    STEP(0); STEP(1); STEP(2); STEP(3);
    STEP(4); STEP(5); STEP(6); STEP(7);

    _mm256_store_pd(&out[0], acc0);
    _mm256_store_pd(&out[4], acc1);
}

/* ---- AVX2 in-register 8x8 transpose: src -> a (no aliasing issues) ---- */
static inline void transpose8x8(double *__restrict__ a,
                                 const double *__restrict__ src) {
    /* Load all 16 halves (8 rows × 2 halves) */
    __m256d r0  = _mm256_load_pd(&src[0]);
    __m256d r1  = _mm256_load_pd(&src[4]);
    __m256d r2  = _mm256_load_pd(&src[8]);
    __m256d r3  = _mm256_load_pd(&src[12]);
    __m256d r4  = _mm256_load_pd(&src[16]);
    __m256d r5  = _mm256_load_pd(&src[20]);
    __m256d r6  = _mm256_load_pd(&src[24]);
    __m256d r7  = _mm256_load_pd(&src[28]);
    __m256d r8  = _mm256_load_pd(&src[32]);
    __m256d r9  = _mm256_load_pd(&src[36]);
    __m256d r10 = _mm256_load_pd(&src[40]);
    __m256d r11 = _mm256_load_pd(&src[44]);
    __m256d r12 = _mm256_load_pd(&src[48]);
    __m256d r13 = _mm256_load_pd(&src[52]);
    __m256d r14 = _mm256_load_pd(&src[56]);
    __m256d r15 = _mm256_load_pd(&src[60]);

    /* Transpose four 4x4 quadrants */

    /* Quadrant Q00: rows 0-3, cols 0-3 (r0,r2,r4,r6 => columns 0-3, first 4 rows) */
    __m256d u00 = _mm256_unpacklo_pd(r0, r2);  /* {a00,a10, a01,a11} */
    __m256d u01 = _mm256_unpackhi_pd(r0, r2);  /* {a02,a12, a03,a13} */
    __m256d u02 = _mm256_unpacklo_pd(r4, r6);  /* {a20,a30, a21,a31} */
    __m256d u03 = _mm256_unpackhi_pd(r4, r6);  /* {a22,a32, a23,a33} */
    /* s0..s3 = first 4 elements of cols 0-3 of the transposed matrix */
    __m256d s0  = _mm256_permute2f128_pd(u00, u02, 0x20); /* {a00,a10,a20,a30} */
    __m256d s2  = _mm256_permute2f128_pd(u00, u02, 0x31); /* {a01,a11,a21,a31} */
    __m256d s1  = _mm256_permute2f128_pd(u01, u03, 0x20); /* {a02,a12,a22,a32} */
    __m256d s3  = _mm256_permute2f128_pd(u01, u03, 0x31); /* {a03,a13,a23,a33} */

    /* Quadrant Q01: rows 4-7, cols 0-3 (r8,r10,r12,r14 => last 4 elements of cols 0-3) */
    u00 = _mm256_unpacklo_pd(r8,  r10);  /* {a40,a50, a41,a51} */
    u01 = _mm256_unpackhi_pd(r8,  r10);  /* {a42,a52, a43,a53} */
    u02 = _mm256_unpacklo_pd(r12, r14);  /* {a60,a70, a61,a71} */
    u03 = _mm256_unpackhi_pd(r12, r14);  /* {a62,a72, a63,a73} */
    __m256d s4  = _mm256_permute2f128_pd(u00, u02, 0x20); /* {a40,a50,a60,a70} */
    __m256d s6  = _mm256_permute2f128_pd(u00, u02, 0x31); /* {a41,a51,a61,a71} */
    __m256d s5  = _mm256_permute2f128_pd(u01, u03, 0x20); /* {a42,a52,a62,a72} */
    __m256d s7  = _mm256_permute2f128_pd(u01, u03, 0x31); /* {a43,a53,a63,a73} */

    /* Quadrant Q10: rows 0-3, cols 4-7 (r1,r3,r5,r7 => first 4 elements of cols 4-7) */
    u00 = _mm256_unpacklo_pd(r1, r3);  /* {a04,a14, a05,a15} */
    u01 = _mm256_unpackhi_pd(r1, r3);  /* {a06,a16, a07,a17} */
    u02 = _mm256_unpacklo_pd(r5, r7);  /* {a24,a34, a25,a35} */
    u03 = _mm256_unpackhi_pd(r5, r7);  /* {a26,a36, a27,a37} */
    __m256d s8  = _mm256_permute2f128_pd(u00, u02, 0x20); /* {a04,a14,a24,a34} */
    __m256d s10 = _mm256_permute2f128_pd(u00, u02, 0x31); /* {a05,a15,a25,a35} */
    __m256d s9  = _mm256_permute2f128_pd(u01, u03, 0x20); /* {a06,a16,a26,a36} */
    __m256d s11 = _mm256_permute2f128_pd(u01, u03, 0x31); /* {a07,a17,a27,a37} */

    /* Quadrant Q11: rows 4-7, cols 4-7 (r9,r11,r13,r15 => last 4 elements of cols 4-7) */
    u00 = _mm256_unpacklo_pd(r9,  r11);  /* {a44,a54, a45,a55} */
    u01 = _mm256_unpackhi_pd(r9,  r11);  /* {a46,a56, a47,a57} */
    u02 = _mm256_unpacklo_pd(r13, r15);  /* {a64,a74, a65,a75} */
    u03 = _mm256_unpackhi_pd(r13, r15);  /* {a66,a76, a67,a77} */
    __m256d s12 = _mm256_permute2f128_pd(u00, u02, 0x20); /* {a44,a54,a64,a74} */
    __m256d s14 = _mm256_permute2f128_pd(u00, u02, 0x31); /* {a45,a55,a65,a75} */
    __m256d s13 = _mm256_permute2f128_pd(u01, u03, 0x20); /* {a46,a56,a66,a76} */
    __m256d s15 = _mm256_permute2f128_pd(u01, u03, 0x31); /* {a47,a57,a67,a77} */

    /* Store results in row-major order.
     * Row i of output = first 4 + last 4 elements of column i of input.
     * s0,s4 = col 0 {a00,a10,a20,a30} + {a40,a50,a60,a70} = row 0 of output
     * s2,s6 = col 1 {a01,a11,a21,a31} + {a41,a51,a61,a71} = row 1
     * s1,s5 = col 2 {a02,a12,a22,a32} + {a42,a52,a62,a72} = row 2
     * s3,s7 = col 3 {a03,a13,a23,a33} + {a43,a53,a63,a73} = row 3
     * s8,s12 = col 4 {a04,a14,a24,a34} + {a44,a54,a64,a74} = row 4
     * s10,s14 = col 5 {a05,a15,a25,a35} + {a45,a55,a65,a75} = row 5
     * s9,s13 = col 6 {a06,a16,a26,a36} + {a46,a56,a66,a76} = row 6
     * s11,s15 = col 7 {a07,a17,a27,a37} + {a47,a57,a67,a77} = row 7
     */
    _mm256_store_pd(&a[0],  s0);   /* row 0, cols 0-3 */
    _mm256_store_pd(&a[4],  s4);   /* row 0, cols 4-7 */
    _mm256_store_pd(&a[8],  s2);   /* row 1, cols 0-3 */
    _mm256_store_pd(&a[12], s6);   /* row 1, cols 4-7 */
    _mm256_store_pd(&a[16], s1);   /* row 2, cols 0-3 */
    _mm256_store_pd(&a[20], s5);   /* row 2, cols 4-7 */
    _mm256_store_pd(&a[24], s3);   /* row 3, cols 0-3 */
    _mm256_store_pd(&a[28], s7);   /* row 3, cols 4-7 */
    _mm256_store_pd(&a[32], s8);   /* row 4, cols 0-3 */
    _mm256_store_pd(&a[36], s12);  /* row 4, cols 4-7 */
    _mm256_store_pd(&a[40], s10);  /* row 5, cols 0-3 */
    _mm256_store_pd(&a[44], s14);  /* row 5, cols 4-7 */
    _mm256_store_pd(&a[48], s9);   /* row 6, cols 0-3 */
    _mm256_store_pd(&a[52], s13);  /* row 6, cols 4-7 */
    _mm256_store_pd(&a[56], s11);  /* row 7, cols 0-3 */
    _mm256_store_pd(&a[60], s15);  /* row 7, cols 4-7 */
}

/* ---- Main 2D IDCT entry point (in-place on 64 contiguous doubles) ---- */
void idct_2d(double *block) {
    double tmp[64] __attribute__((aligned(32)));

    /* Pass 1: row transforms (block -> tmp) */
    idct_1d(&block[0],  &tmp[0]);
    idct_1d(&block[8],  &tmp[8]);
    idct_1d(&block[16], &tmp[16]);
    idct_1d(&block[24], &tmp[24]);
    idct_1d(&block[32], &tmp[32]);
    idct_1d(&block[40], &tmp[40]);
    idct_1d(&block[48], &tmp[48]);
    idct_1d(&block[56], &tmp[56]);

    /* Transpose (tmp -> block): columns become rows */
    transpose8x8(block, tmp);

    /* Pass 2: row transforms on transposed data (= columns of original) */
    idct_1d(&block[0],  &tmp[0]);
    idct_1d(&block[8],  &tmp[8]);
    idct_1d(&block[16], &tmp[16]);
    idct_1d(&block[24], &tmp[24]);
    idct_1d(&block[32], &tmp[32]);
    idct_1d(&block[40], &tmp[40]);
    idct_1d(&block[48], &tmp[48]);
    idct_1d(&block[56], &tmp[56]);

    /* Transpose back (tmp -> block) */
    transpose8x8(block, tmp);
}
