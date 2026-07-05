import math

# Precompute 1D IDCT coefficients as tuples for fast read access.
#   coeffs[freq][pos] = 0.5 * alpha(freq) * cos((2*pos+1) * freq * pi / 16)
#   alpha(0) = 1/sqrt(2), alpha(f!=0) = 1
def _build_coeffs():
    rows = []
    for f in range(8):
        af = 1.0 / math.sqrt(2.0) if f == 0 else 1.0
        rows.append(tuple(
            0.5 * af * math.cos((2.0 * p + 1.0) * f * math.pi / 16.0)
            for p in range(8)
        ))
    return tuple(rows)

_C = _build_coeffs()
# Pre-extract all 64 individual coefficients as module-level constants
# to eliminate tuple lookups in the inner loops.
(_C00, _C01, _C02, _C03, _C04, _C05, _C06, _C07) = _C[0]
(_C10, _C11, _C12, _C13, _C14, _C15, _C16, _C17) = _C[1]
(_C20, _C21, _C22, _C23, _C24, _C25, _C26, _C27) = _C[2]
(_C30, _C31, _C32, _C33, _C34, _C35, _C36, _C37) = _C[3]
(_C40, _C41, _C42, _C43, _C44, _C45, _C46, _C47) = _C[4]
(_C50, _C51, _C52, _C53, _C54, _C55, _C56, _C57) = _C[5]
(_C60, _C61, _C62, _C63, _C64, _C65, _C66, _C67) = _C[6]
(_C70, _C71, _C72, _C73, _C74, _C75, _C76, _C77) = _C[7]


def idct_2d(matrix):
    """Row-column 2D IDCT using even/odd symmetry decomposition.

    Exploits C[k][j] symmetry: even frequencies are symmetric
    (C[k][j] == C[k][7-j]), odd frequencies are anti-symmetric
    (C[k][j] == -C[k][7-j]) in the spatial index j.

    This halves multiplications vs the fully-unrolled naive approach:
    ~512 mults instead of ~1024 per 8x8 block.
    """
    # Bind all 64 coefficients to local variables for zero-overhead access.
    c00, c01, c02, c03, c04, c05, c06, c07 = _C00, _C01, _C02, _C03, _C04, _C05, _C06, _C07
    c10, c11, c12, c13, c14, c15, c16, c17 = _C10, _C11, _C12, _C13, _C14, _C15, _C16, _C17
    c20, c21, c22, c23, c24, c25, c26, c27 = _C20, _C21, _C22, _C23, _C24, _C25, _C26, _C27
    c30, c31, c32, c33, c34, c35, c36, c37 = _C30, _C31, _C32, _C33, _C34, _C35, _C36, _C37
    c40, c41, c42, c43, c44, c45, c46, c47 = _C40, _C41, _C42, _C43, _C44, _C45, _C46, _C47
    c50, c51, c52, c53, c54, c55, c56, c57 = _C50, _C51, _C52, _C53, _C54, _C55, _C56, _C57
    c60, c61, c62, c63, c64, c65, c66, c67 = _C60, _C61, _C62, _C63, _C64, _C65, _C66, _C67
    c70, c71, c72, c73, c74, c75, c76, c77 = _C70, _C71, _C72, _C73, _C74, _C75, _C76, _C77

    # ---- Row transform ---------------------------------------------------
    # For each row: t_{i,j} = sum_k S[k] * C[k][j]
    # Using even/odd symmetry we compute pairs (j, 7-j) together.

    # Helper for one row: given 8 input samples s[0..7], compute 8 t-values.
    # Even-part coefficients (k=0,2,4,6) at positions j=0,1,2,3
    # Odd-part coefficients (k=1,3,5,7) at positions j=0,1,2,3

    # --- Row 0 ---
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[0]
    # Even sums (freqs 0,2,4,6) at j=0,1,2,3
    e0 = s0*c00 + s2*c20 + s4*c40 + s6*c60
    e1 = s0*c01 + s2*c21 + s4*c41 + s6*c61
    e2 = s0*c02 + s2*c22 + s4*c42 + s6*c62
    e3 = s0*c03 + s2*c23 + s4*c43 + s6*c63
    # Odd sums (freqs 1,3,5,7) at j=0,1,2,3
    o0 = s1*c10 + s3*c30 + s5*c50 + s7*c70
    o1 = s1*c11 + s3*c31 + s5*c51 + s7*c71
    o2 = s1*c12 + s3*c32 + s5*c52 + s7*c72
    o3 = s1*c13 + s3*c33 + s5*c53 + s7*c73
    # Combine
    t00 = e0 + o0; t07 = e0 - o0
    t01 = e1 + o1; t06 = e1 - o1
    t02 = e2 + o2; t05 = e2 - o2
    t03 = e3 + o3; t04 = e3 - o3

    # --- Row 1 ---
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[1]
    e0 = s0*c00 + s2*c20 + s4*c40 + s6*c60
    e1 = s0*c01 + s2*c21 + s4*c41 + s6*c61
    e2 = s0*c02 + s2*c22 + s4*c42 + s6*c62
    e3 = s0*c03 + s2*c23 + s4*c43 + s6*c63
    o0 = s1*c10 + s3*c30 + s5*c50 + s7*c70
    o1 = s1*c11 + s3*c31 + s5*c51 + s7*c71
    o2 = s1*c12 + s3*c32 + s5*c52 + s7*c72
    o3 = s1*c13 + s3*c33 + s5*c53 + s7*c73
    t10 = e0 + o0; t17 = e0 - o0
    t11 = e1 + o1; t16 = e1 - o1
    t12 = e2 + o2; t15 = e2 - o2
    t13 = e3 + o3; t14 = e3 - o3

    # --- Row 2 ---
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[2]
    e0 = s0*c00 + s2*c20 + s4*c40 + s6*c60
    e1 = s0*c01 + s2*c21 + s4*c41 + s6*c61
    e2 = s0*c02 + s2*c22 + s4*c42 + s6*c62
    e3 = s0*c03 + s2*c23 + s4*c43 + s6*c63
    o0 = s1*c10 + s3*c30 + s5*c50 + s7*c70
    o1 = s1*c11 + s3*c31 + s5*c51 + s7*c71
    o2 = s1*c12 + s3*c32 + s5*c52 + s7*c72
    o3 = s1*c13 + s3*c33 + s5*c53 + s7*c73
    t20 = e0 + o0; t27 = e0 - o0
    t21 = e1 + o1; t26 = e1 - o1
    t22 = e2 + o2; t25 = e2 - o2
    t23 = e3 + o3; t24 = e3 - o3

    # --- Row 3 ---
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[3]
    e0 = s0*c00 + s2*c20 + s4*c40 + s6*c60
    e1 = s0*c01 + s2*c21 + s4*c41 + s6*c61
    e2 = s0*c02 + s2*c22 + s4*c42 + s6*c62
    e3 = s0*c03 + s2*c23 + s4*c43 + s6*c63
    o0 = s1*c10 + s3*c30 + s5*c50 + s7*c70
    o1 = s1*c11 + s3*c31 + s5*c51 + s7*c71
    o2 = s1*c12 + s3*c32 + s5*c52 + s7*c72
    o3 = s1*c13 + s3*c33 + s5*c53 + s7*c73
    t30 = e0 + o0; t37 = e0 - o0
    t31 = e1 + o1; t36 = e1 - o1
    t32 = e2 + o2; t35 = e2 - o2
    t33 = e3 + o3; t34 = e3 - o3

    # --- Row 4 ---
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[4]
    e0 = s0*c00 + s2*c20 + s4*c40 + s6*c60
    e1 = s0*c01 + s2*c21 + s4*c41 + s6*c61
    e2 = s0*c02 + s2*c22 + s4*c42 + s6*c62
    e3 = s0*c03 + s2*c23 + s4*c43 + s6*c63
    o0 = s1*c10 + s3*c30 + s5*c50 + s7*c70
    o1 = s1*c11 + s3*c31 + s5*c51 + s7*c71
    o2 = s1*c12 + s3*c32 + s5*c52 + s7*c72
    o3 = s1*c13 + s3*c33 + s5*c53 + s7*c73
    t40 = e0 + o0; t47 = e0 - o0
    t41 = e1 + o1; t46 = e1 - o1
    t42 = e2 + o2; t45 = e2 - o2
    t43 = e3 + o3; t44 = e3 - o3

    # --- Row 5 ---
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[5]
    e0 = s0*c00 + s2*c20 + s4*c40 + s6*c60
    e1 = s0*c01 + s2*c21 + s4*c41 + s6*c61
    e2 = s0*c02 + s2*c22 + s4*c42 + s6*c62
    e3 = s0*c03 + s2*c23 + s4*c43 + s6*c63
    o0 = s1*c10 + s3*c30 + s5*c50 + s7*c70
    o1 = s1*c11 + s3*c31 + s5*c51 + s7*c71
    o2 = s1*c12 + s3*c32 + s5*c52 + s7*c72
    o3 = s1*c13 + s3*c33 + s5*c53 + s7*c73
    t50 = e0 + o0; t57 = e0 - o0
    t51 = e1 + o1; t56 = e1 - o1
    t52 = e2 + o2; t55 = e2 - o2
    t53 = e3 + o3; t54 = e3 - o3

    # --- Row 6 ---
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[6]
    e0 = s0*c00 + s2*c20 + s4*c40 + s6*c60
    e1 = s0*c01 + s2*c21 + s4*c41 + s6*c61
    e2 = s0*c02 + s2*c22 + s4*c42 + s6*c62
    e3 = s0*c03 + s2*c23 + s4*c43 + s6*c63
    o0 = s1*c10 + s3*c30 + s5*c50 + s7*c70
    o1 = s1*c11 + s3*c31 + s5*c51 + s7*c71
    o2 = s1*c12 + s3*c32 + s5*c52 + s7*c72
    o3 = s1*c13 + s3*c33 + s5*c53 + s7*c73
    t60 = e0 + o0; t67 = e0 - o0
    t61 = e1 + o1; t66 = e1 - o1
    t62 = e2 + o2; t65 = e2 - o2
    t63 = e3 + o3; t64 = e3 - o3

    # --- Row 7 ---
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[7]
    e0 = s0*c00 + s2*c20 + s4*c40 + s6*c60
    e1 = s0*c01 + s2*c21 + s4*c41 + s6*c61
    e2 = s0*c02 + s2*c22 + s4*c42 + s6*c62
    e3 = s0*c03 + s2*c23 + s4*c43 + s6*c63
    o0 = s1*c10 + s3*c30 + s5*c50 + s7*c70
    o1 = s1*c11 + s3*c31 + s5*c51 + s7*c71
    o2 = s1*c12 + s3*c32 + s5*c52 + s7*c72
    o3 = s1*c13 + s3*c33 + s5*c53 + s7*c73
    t70 = e0 + o0; t77 = e0 - o0
    t71 = e1 + o1; t76 = e1 - o1
    t72 = e2 + o2; t75 = e2 - o2
    t73 = e3 + o3; t74 = e3 - o3

    # ---- Column transform ------------------------------------------------
    # result[i][j] = sum_{k=0}^{7} C[k][i] * t[k][j]
    # Same even/odd symmetry applied along the frequency axis k.
    # For each column j we compute pairs (i, 7-i) together.
    # We use the same coefficient pattern as the row pass:
    #   even: C[0][i], C[2][i], C[4][i], C[6][i]
    #   odd:  C[1][i], C[3][i], C[5][i], C[7][i]

    result = [[0.0]*8 for _ in range(8)]

    # Column 0: t[k][0] values
    # E_i = t00*C[0][i] + t20*C[2][i] + t40*C[4][i] + t60*C[6][i]
    # O_i = t10*C[1][i] + t30*C[3][i] + t50*C[5][i] + t70*C[7][i]
    # i=0:
    e0 = t00*c00 + t20*c20 + t40*c40 + t60*c60
    o0 = t10*c10 + t30*c30 + t50*c50 + t70*c70
    result[0][0] = e0 + o0; result[7][0] = e0 - o0
    # i=1:
    e1 = t00*c01 + t20*c21 + t40*c41 + t60*c61
    o1 = t10*c11 + t30*c31 + t50*c51 + t70*c71
    result[1][0] = e1 + o1; result[6][0] = e1 - o1
    # i=2:
    e2 = t00*c02 + t20*c22 + t40*c42 + t60*c62
    o2 = t10*c12 + t30*c32 + t50*c52 + t70*c72
    result[2][0] = e2 + o2; result[5][0] = e2 - o2
    # i=3:
    e3 = t00*c03 + t20*c23 + t40*c43 + t60*c63
    o3 = t10*c13 + t30*c33 + t50*c53 + t70*c73
    result[3][0] = e3 + o3; result[4][0] = e3 - o3

    # Column 1: t[k][1]
    e0 = t01*c00 + t21*c20 + t41*c40 + t61*c60
    o0 = t11*c10 + t31*c30 + t51*c50 + t71*c70
    result[0][1] = e0 + o0; result[7][1] = e0 - o0
    e1 = t01*c01 + t21*c21 + t41*c41 + t61*c61
    o1 = t11*c11 + t31*c31 + t51*c51 + t71*c71
    result[1][1] = e1 + o1; result[6][1] = e1 - o1
    e2 = t01*c02 + t21*c22 + t41*c42 + t61*c62
    o2 = t11*c12 + t31*c32 + t51*c52 + t71*c72
    result[2][1] = e2 + o2; result[5][1] = e2 - o2
    e3 = t01*c03 + t21*c23 + t41*c43 + t61*c63
    o3 = t11*c13 + t31*c33 + t51*c53 + t71*c73
    result[3][1] = e3 + o3; result[4][1] = e3 - o3

    # Column 2: t[k][2]
    e0 = t02*c00 + t22*c20 + t42*c40 + t62*c60
    o0 = t12*c10 + t32*c30 + t52*c50 + t72*c70
    result[0][2] = e0 + o0; result[7][2] = e0 - o0
    e1 = t02*c01 + t22*c21 + t42*c41 + t62*c61
    o1 = t12*c11 + t32*c31 + t52*c51 + t72*c71
    result[1][2] = e1 + o1; result[6][2] = e1 - o1
    e2 = t02*c02 + t22*c22 + t42*c42 + t62*c62
    o2 = t12*c12 + t32*c32 + t52*c52 + t72*c72
    result[2][2] = e2 + o2; result[5][2] = e2 - o2
    e3 = t02*c03 + t22*c23 + t42*c43 + t62*c63
    o3 = t12*c13 + t32*c33 + t52*c53 + t72*c73
    result[3][2] = e3 + o3; result[4][2] = e3 - o3

    # Column 3: t[k][3]
    e0 = t03*c00 + t23*c20 + t43*c40 + t63*c60
    o0 = t13*c10 + t33*c30 + t53*c50 + t73*c70
    result[0][3] = e0 + o0; result[7][3] = e0 - o0
    e1 = t03*c01 + t23*c21 + t43*c41 + t63*c61
    o1 = t13*c11 + t33*c31 + t53*c51 + t73*c71
    result[1][3] = e1 + o1; result[6][3] = e1 - o1
    e2 = t03*c02 + t23*c22 + t43*c42 + t63*c62
    o2 = t13*c12 + t33*c32 + t53*c52 + t73*c72
    result[2][3] = e2 + o2; result[5][3] = e2 - o2
    e3 = t03*c03 + t23*c23 + t43*c43 + t63*c63
    o3 = t13*c13 + t33*c33 + t53*c53 + t73*c73
    result[3][3] = e3 + o3; result[4][3] = e3 - o3

    # Column 4: t[k][4]
    e0 = t04*c00 + t24*c20 + t44*c40 + t64*c60
    o0 = t14*c10 + t34*c30 + t54*c50 + t74*c70
    result[0][4] = e0 + o0; result[7][4] = e0 - o0
    e1 = t04*c01 + t24*c21 + t44*c41 + t64*c61
    o1 = t14*c11 + t34*c31 + t54*c51 + t74*c71
    result[1][4] = e1 + o1; result[6][4] = e1 - o1
    e2 = t04*c02 + t24*c22 + t44*c42 + t64*c62
    o2 = t14*c12 + t34*c32 + t54*c52 + t74*c72
    result[2][4] = e2 + o2; result[5][4] = e2 - o2
    e3 = t04*c03 + t24*c23 + t44*c43 + t64*c63
    o3 = t14*c13 + t34*c33 + t54*c53 + t74*c73
    result[3][4] = e3 + o3; result[4][4] = e3 - o3

    # Column 5: t[k][5]
    e0 = t05*c00 + t25*c20 + t45*c40 + t65*c60
    o0 = t15*c10 + t35*c30 + t55*c50 + t75*c70
    result[0][5] = e0 + o0; result[7][5] = e0 - o0
    e1 = t05*c01 + t25*c21 + t45*c41 + t65*c61
    o1 = t15*c11 + t35*c31 + t55*c51 + t75*c71
    result[1][5] = e1 + o1; result[6][5] = e1 - o1
    e2 = t05*c02 + t25*c22 + t45*c42 + t65*c62
    o2 = t15*c12 + t35*c32 + t55*c52 + t75*c72
    result[2][5] = e2 + o2; result[5][5] = e2 - o2
    e3 = t05*c03 + t25*c23 + t45*c43 + t65*c63
    o3 = t15*c13 + t35*c33 + t55*c53 + t75*c73
    result[3][5] = e3 + o3; result[4][5] = e3 - o3

    # Column 6: t[k][6]
    e0 = t06*c00 + t26*c20 + t46*c40 + t66*c60
    o0 = t16*c10 + t36*c30 + t56*c50 + t76*c70
    result[0][6] = e0 + o0; result[7][6] = e0 - o0
    e1 = t06*c01 + t26*c21 + t46*c41 + t66*c61
    o1 = t16*c11 + t36*c31 + t56*c51 + t76*c71
    result[1][6] = e1 + o1; result[6][6] = e1 - o1
    e2 = t06*c02 + t26*c22 + t46*c42 + t66*c62
    o2 = t16*c12 + t36*c32 + t56*c52 + t76*c72
    result[2][6] = e2 + o2; result[5][6] = e2 - o2
    e3 = t06*c03 + t26*c23 + t46*c43 + t66*c63
    o3 = t16*c13 + t36*c33 + t56*c53 + t76*c73
    result[3][6] = e3 + o3; result[4][6] = e3 - o3

    # Column 7: t[k][7]
    e0 = t07*c00 + t27*c20 + t47*c40 + t67*c60
    o0 = t17*c10 + t37*c30 + t57*c50 + t77*c70
    result[0][7] = e0 + o0; result[7][7] = e0 - o0
    e1 = t07*c01 + t27*c21 + t47*c41 + t67*c61
    o1 = t17*c11 + t37*c31 + t57*c51 + t77*c71
    result[1][7] = e1 + o1; result[6][7] = e1 - o1
    e2 = t07*c02 + t27*c22 + t47*c42 + t67*c62
    o2 = t17*c12 + t37*c32 + t57*c52 + t77*c72
    result[2][7] = e2 + o2; result[5][7] = e2 - o2
    e3 = t07*c03 + t27*c23 + t47*c43 + t67*c63
    o3 = t17*c13 + t37*c33 + t57*c53 + t77*c73
    result[3][7] = e3 + o3; result[4][7] = e3 - o3

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
