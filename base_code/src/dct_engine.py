import math

# Precompute 1D IDCT coefficients:
#   coeffs[freq][pos] = 0.5 * alpha(freq) * cos((2*pos+1) * freq * pi / 16)
# This absorbs the alpha scaling and the 0.5 factor, saving one multiply per term.
_C = [[0.0] * 8 for _ in range(8)]
for f in range(8):
    af = 1.0 / math.sqrt(2.0) if f == 0 else 1.0
    for p in range(8):
        _C[f][p] = 0.5 * af * math.cos((2.0 * p + 1.0) * f * math.pi / 16.0)


def _idct_1d(s):
    """Fully unrolled 8-point 1D IDCT.

    Eliminates all loop overhead by computing each of the 8 output
    samples as a single dot product of the input vector with the
    appropriate pre‑computed coefficient row.
    """
    c0, c1, c2, c3, c4, c5, c6, c7 = _C
    s0, s1, s2, s3, s4, s5, s6, s7 = s
    return [
        s0 * c0[0] + s1 * c1[0] + s2 * c2[0] + s3 * c3[0]
        + s4 * c4[0] + s5 * c5[0] + s6 * c6[0] + s7 * c7[0],
        s0 * c0[1] + s1 * c1[1] + s2 * c2[1] + s3 * c3[1]
        + s4 * c4[1] + s5 * c5[1] + s6 * c6[1] + s7 * c7[1],
        s0 * c0[2] + s1 * c1[2] + s2 * c2[2] + s3 * c3[2]
        + s4 * c4[2] + s5 * c5[2] + s6 * c6[2] + s7 * c7[2],
        s0 * c0[3] + s1 * c1[3] + s2 * c2[3] + s3 * c3[3]
        + s4 * c4[3] + s5 * c5[3] + s6 * c6[3] + s7 * c7[3],
        s0 * c0[4] + s1 * c1[4] + s2 * c2[4] + s3 * c3[4]
        + s4 * c4[4] + s5 * c5[4] + s6 * c6[4] + s7 * c7[4],
        s0 * c0[5] + s1 * c1[5] + s2 * c2[5] + s3 * c3[5]
        + s4 * c4[5] + s5 * c5[5] + s6 * c6[5] + s7 * c7[5],
        s0 * c0[6] + s1 * c1[6] + s2 * c2[6] + s3 * c3[6]
        + s4 * c4[6] + s5 * c5[6] + s6 * c6[6] + s7 * c7[6],
        s0 * c0[7] + s1 * c1[7] + s2 * c2[7] + s3 * c3[7]
        + s4 * c4[7] + s5 * c5[7] + s6 * c6[7] + s7 * c7[7],
    ]


def idct_2d(matrix):
    """Row‑column 2D IDCT.

    Row pass: apply the unrolled 1D IDCT to each of the 8 rows.
    Column pass: accumulate the column transform *without* an explicit
    transpose, avoiding the zip(*…) + list() overhead of the naive version.
    """
    # Row transform
    t0 = _idct_1d(matrix[0])
    t1 = _idct_1d(matrix[1])
    t2 = _idct_1d(matrix[2])
    t3 = _idct_1d(matrix[3])
    t4 = _idct_1d(matrix[4])
    t5 = _idct_1d(matrix[5])
    t6 = _idct_1d(matrix[6])
    t7 = _idct_1d(matrix[7])
    temp = [t0, t1, t2, t3, t4, t5, t6, t7]

    # Column transform:  result[i][j] = sum_k _C[k][i] * temp[k][j]
    result = [[0.0] * 8 for _ in range(8)]
    for k in range(8):
        ck = _C[k]
        tk = temp[k]
        r0, r1, r2, r3 = result[0], result[1], result[2], result[3]
        r4, r5, r6, r7 = result[4], result[5], result[6], result[7]
        ck0, ck1, ck2, ck3 = ck[0], ck[1], ck[2], ck[3]
        ck4, ck5, ck6, ck7 = ck[4], ck[5], ck[6], ck[7]
        tk0, tk1, tk2, tk3 = tk[0], tk[1], tk[2], tk[3]
        tk4, tk5, tk6, tk7 = tk[4], tk[5], tk[6], tk[7]

        r0[0] += ck0 * tk0
        r0[1] += ck0 * tk1
        r0[2] += ck0 * tk2
        r0[3] += ck0 * tk3
        r0[4] += ck0 * tk4
        r0[5] += ck0 * tk5
        r0[6] += ck0 * tk6
        r0[7] += ck0 * tk7

        r1[0] += ck1 * tk0
        r1[1] += ck1 * tk1
        r1[2] += ck1 * tk2
        r1[3] += ck1 * tk3
        r1[4] += ck1 * tk4
        r1[5] += ck1 * tk5
        r1[6] += ck1 * tk6
        r1[7] += ck1 * tk7

        r2[0] += ck2 * tk0
        r2[1] += ck2 * tk1
        r2[2] += ck2 * tk2
        r2[3] += ck2 * tk3
        r2[4] += ck2 * tk4
        r2[5] += ck2 * tk5
        r2[6] += ck2 * tk6
        r2[7] += ck2 * tk7

        r3[0] += ck3 * tk0
        r3[1] += ck3 * tk1
        r3[2] += ck3 * tk2
        r3[3] += ck3 * tk3
        r3[4] += ck3 * tk4
        r3[5] += ck3 * tk5
        r3[6] += ck3 * tk6
        r3[7] += ck3 * tk7

        r4[0] += ck4 * tk0
        r4[1] += ck4 * tk1
        r4[2] += ck4 * tk2
        r4[3] += ck4 * tk3
        r4[4] += ck4 * tk4
        r4[5] += ck4 * tk5
        r4[6] += ck4 * tk6
        r4[7] += ck4 * tk7

        r5[0] += ck5 * tk0
        r5[1] += ck5 * tk1
        r5[2] += ck5 * tk2
        r5[3] += ck5 * tk3
        r5[4] += ck5 * tk4
        r5[5] += ck5 * tk5
        r5[6] += ck5 * tk6
        r5[7] += ck5 * tk7

        r6[0] += ck6 * tk0
        r6[1] += ck6 * tk1
        r6[2] += ck6 * tk2
        r6[3] += ck6 * tk3
        r6[4] += ck6 * tk4
        r6[5] += ck6 * tk5
        r6[6] += ck6 * tk6
        r6[7] += ck6 * tk7

        r7[0] += ck7 * tk0
        r7[1] += ck7 * tk1
        r7[2] += ck7 * tk2
        r7[3] += ck7 * tk3
        r7[4] += ck7 * tk4
        r7[5] += ck7 * tk5
        r7[6] += ck7 * tk6
        r7[7] += ck7 * tk7

    return result


def ycbcr_to_rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    r = max(0, min(255, round(r)))
    g = max(0, min(255, round(g)))
    b = max(0, min(255, round(b)))
    return (r, g, b)


def decode_mcu(y_block, cb_block, cr_block):
    y_pixels = idct_2d(y_block)
    cb_pixels = idct_2d(cb_block)
    cr_pixels = idct_2d(cr_block)
    pixels = []
    for row_idx in range(8):
        pixel_row = []
        for col_idx in range(8):
            pixel_row.append(ycbcr_to_rgb(
                y_pixels[row_idx][col_idx] + 128,
                cb_pixels[row_idx][col_idx] + 128,
                cr_pixels[row_idx][col_idx] + 128,
            ))
        pixels.append(pixel_row)
    return pixels
