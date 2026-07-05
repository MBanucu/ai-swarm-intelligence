/* dct_engine_impl.c -- 8x8 2D IDCT with GCC vector_size(32) SIMD
 *
 * Implements separable row-column 2D IDCT using 256-bit (4×double) vectors.
 * Row transform -> transpose -> row transform -> transpose keeps all memory
 * access contiguous and SIMD-friendly.
 *
 * Build: gcc -O3 -mavx2 -mfma -ffast-math -fPIC -shared -o libdct_engine.so
 *        dct_engine_impl.c -lm
 */

#include <math.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* 256-bit vector of 4 doubles (requires AVX) */
typedef double double4 __attribute__((vector_size(32)));

/* Precomputed 1D IDCT coefficients: 8×8 row-major.
 * coeffs[k*8+p] = 0.5 * alpha(k) * cos((2*p+1) * k * pi / 16)
 * alpha(0)=1/sqrt2, alpha(k!=0)=1
 */
static double coeffs[64] __attribute__((aligned(32)));

__attribute__((constructor))
static void init_coeffs(void) {
    static const double sqrt2inv = 0.707106781186547524400844362104849039;
    for (int k = 0; k < 8; k++) {
        double alpha = (k == 0) ? sqrt2inv : 1.0;
        for (int p = 0; p < 8; p++) {
            coeffs[k * 8 + p] = 0.5 * alpha *
                cos((2.0 * p + 1.0) * (double)k * M_PI / 16.0);
        }
    }
}

/* 1D 8-point IDCT using double4 SIMD.
 * Processes 4 output lanes (p=0..3 and p=4..7) in parallel.
 * Fully unrolled for maximum throughput.
 */
static inline void idct_1d(const double *__restrict__ in, double *__restrict__ out) {
    double4 acc0 = {0.0, 0.0, 0.0, 0.0};
    double4 acc1 = {0.0, 0.0, 0.0, 0.0};

    /* Manually unrolled: for k in 0..7, broadcast in[k] and multiply-add
     * the two coefficient groups (columns 0-3 and 4-7). */
#define IDCT_STEP(k, idx) do {                         \
    double4 inb = {in[idx], in[idx], in[idx], in[idx]}; \
    acc0 += inb * *(const double4 *)&coeffs[k*8];       \
    acc1 += inb * *(const double4 *)&coeffs[k*8 + 4];   \
} while (0)

    IDCT_STEP(0, 0); IDCT_STEP(1, 1);
    IDCT_STEP(2, 2); IDCT_STEP(3, 3);
    IDCT_STEP(4, 4); IDCT_STEP(5, 5);
    IDCT_STEP(6, 6); IDCT_STEP(7, 7);

    *(double4 *)&out[0] = acc0;
    *(double4 *)&out[4] = acc1;
}

/* 8x8 in-place transpose on a 64-element flat row-major array. */
static inline void transpose8x8(double *__restrict__ a) {
    double t[64] __attribute__((aligned(32)));
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            t[j * 8 + i] = a[i * 8 + j];
    for (int i = 0; i < 64; i++)
        a[i] = t[i];
}

/* Main 2D IDCT entry point.
 * block: pointer to 64 contiguous doubles in row-major order.
 * Transform is performed in-place.
 */
void idct_2d(double *block) {
    double tmp[64] __attribute__((aligned(32)));

    /* ---- Pass 1: row transforms ---- */
    idct_1d(&block[0],  &tmp[0]);
    idct_1d(&block[8],  &tmp[8]);
    idct_1d(&block[16], &tmp[16]);
    idct_1d(&block[24], &tmp[24]);
    idct_1d(&block[32], &tmp[32]);
    idct_1d(&block[40], &tmp[40]);
    idct_1d(&block[48], &tmp[48]);
    idct_1d(&block[56], &tmp[56]);

    /* ---- Transpose tmp -> block ---- */
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            block[j * 8 + i] = tmp[i * 8 + j];

    /* ---- Pass 2: row transforms on transposed data ---- */
    idct_1d(&block[0],  &tmp[0]);
    idct_1d(&block[8],  &tmp[8]);
    idct_1d(&block[16], &tmp[16]);
    idct_1d(&block[24], &tmp[24]);
    idct_1d(&block[32], &tmp[32]);
    idct_1d(&block[40], &tmp[40]);
    idct_1d(&block[48], &tmp[48]);
    idct_1d(&block[56], &tmp[56]);

    /* ---- Transpose back tmp -> block ---- */
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            block[j * 8 + i] = tmp[i * 8 + j];
}
