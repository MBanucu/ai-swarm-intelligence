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
    """Row-column 2D IDCT with fully unrolled row and column passes.

    Row transforms are inlined to eliminate function-call overhead and
    intermediate list creation.  The column pass uses the same fully
    unrolled k-loop with local coefficient bindings.
    All 64 coefficients are pre-extracted as local constants to eliminate
    tuple-indexing overhead.
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

    # ---- Row transforms (inlined, no intermediate list allocations) ----
    # Row 0
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[0]
    t00 = s0*c00 + s1*c10 + s2*c20 + s3*c30 + s4*c40 + s5*c50 + s6*c60 + s7*c70
    t01 = s0*c01 + s1*c11 + s2*c21 + s3*c31 + s4*c41 + s5*c51 + s6*c61 + s7*c71
    t02 = s0*c02 + s1*c12 + s2*c22 + s3*c32 + s4*c42 + s5*c52 + s6*c62 + s7*c72
    t03 = s0*c03 + s1*c13 + s2*c23 + s3*c33 + s4*c43 + s5*c53 + s6*c63 + s7*c73
    t04 = s0*c04 + s1*c14 + s2*c24 + s3*c34 + s4*c44 + s5*c54 + s6*c64 + s7*c74
    t05 = s0*c05 + s1*c15 + s2*c25 + s3*c35 + s4*c45 + s5*c55 + s6*c65 + s7*c75
    t06 = s0*c06 + s1*c16 + s2*c26 + s3*c36 + s4*c46 + s5*c56 + s6*c66 + s7*c76
    t07 = s0*c07 + s1*c17 + s2*c27 + s3*c37 + s4*c47 + s5*c57 + s6*c67 + s7*c77

    # Row 1
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[1]
    t10 = s0*c00 + s1*c10 + s2*c20 + s3*c30 + s4*c40 + s5*c50 + s6*c60 + s7*c70
    t11 = s0*c01 + s1*c11 + s2*c21 + s3*c31 + s4*c41 + s5*c51 + s6*c61 + s7*c71
    t12 = s0*c02 + s1*c12 + s2*c22 + s3*c32 + s4*c42 + s5*c52 + s6*c62 + s7*c72
    t13 = s0*c03 + s1*c13 + s2*c23 + s3*c33 + s4*c43 + s5*c53 + s6*c63 + s7*c73
    t14 = s0*c04 + s1*c14 + s2*c24 + s3*c34 + s4*c44 + s5*c54 + s6*c64 + s7*c74
    t15 = s0*c05 + s1*c15 + s2*c25 + s3*c35 + s4*c45 + s5*c55 + s6*c65 + s7*c75
    t16 = s0*c06 + s1*c16 + s2*c26 + s3*c36 + s4*c46 + s5*c56 + s6*c66 + s7*c76
    t17 = s0*c07 + s1*c17 + s2*c27 + s3*c37 + s4*c47 + s5*c57 + s6*c67 + s7*c77

    # Row 2
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[2]
    t20 = s0*c00 + s1*c10 + s2*c20 + s3*c30 + s4*c40 + s5*c50 + s6*c60 + s7*c70
    t21 = s0*c01 + s1*c11 + s2*c21 + s3*c31 + s4*c41 + s5*c51 + s6*c61 + s7*c71
    t22 = s0*c02 + s1*c12 + s2*c22 + s3*c32 + s4*c42 + s5*c52 + s6*c62 + s7*c72
    t23 = s0*c03 + s1*c13 + s2*c23 + s3*c33 + s4*c43 + s5*c53 + s6*c63 + s7*c73
    t24 = s0*c04 + s1*c14 + s2*c24 + s3*c34 + s4*c44 + s5*c54 + s6*c64 + s7*c74
    t25 = s0*c05 + s1*c15 + s2*c25 + s3*c35 + s4*c45 + s5*c55 + s6*c65 + s7*c75
    t26 = s0*c06 + s1*c16 + s2*c26 + s3*c36 + s4*c46 + s5*c56 + s6*c66 + s7*c76
    t27 = s0*c07 + s1*c17 + s2*c27 + s3*c37 + s4*c47 + s5*c57 + s6*c67 + s7*c77

    # Row 3
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[3]
    t30 = s0*c00 + s1*c10 + s2*c20 + s3*c30 + s4*c40 + s5*c50 + s6*c60 + s7*c70
    t31 = s0*c01 + s1*c11 + s2*c21 + s3*c31 + s4*c41 + s5*c51 + s6*c61 + s7*c71
    t32 = s0*c02 + s1*c12 + s2*c22 + s3*c32 + s4*c42 + s5*c52 + s6*c62 + s7*c72
    t33 = s0*c03 + s1*c13 + s2*c23 + s3*c33 + s4*c43 + s5*c53 + s6*c63 + s7*c73
    t34 = s0*c04 + s1*c14 + s2*c24 + s3*c34 + s4*c44 + s5*c54 + s6*c64 + s7*c74
    t35 = s0*c05 + s1*c15 + s2*c25 + s3*c35 + s4*c45 + s5*c55 + s6*c65 + s7*c75
    t36 = s0*c06 + s1*c16 + s2*c26 + s3*c36 + s4*c46 + s5*c56 + s6*c66 + s7*c76
    t37 = s0*c07 + s1*c17 + s2*c27 + s3*c37 + s4*c47 + s5*c57 + s6*c67 + s7*c77

    # Row 4
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[4]
    t40 = s0*c00 + s1*c10 + s2*c20 + s3*c30 + s4*c40 + s5*c50 + s6*c60 + s7*c70
    t41 = s0*c01 + s1*c11 + s2*c21 + s3*c31 + s4*c41 + s5*c51 + s6*c61 + s7*c71
    t42 = s0*c02 + s1*c12 + s2*c22 + s3*c32 + s4*c42 + s5*c52 + s6*c62 + s7*c72
    t43 = s0*c03 + s1*c13 + s2*c23 + s3*c33 + s4*c43 + s5*c53 + s6*c63 + s7*c73
    t44 = s0*c04 + s1*c14 + s2*c24 + s3*c34 + s4*c44 + s5*c54 + s6*c64 + s7*c74
    t45 = s0*c05 + s1*c15 + s2*c25 + s3*c35 + s4*c45 + s5*c55 + s6*c65 + s7*c75
    t46 = s0*c06 + s1*c16 + s2*c26 + s3*c36 + s4*c46 + s5*c56 + s6*c66 + s7*c76
    t47 = s0*c07 + s1*c17 + s2*c27 + s3*c37 + s4*c47 + s5*c57 + s6*c67 + s7*c77

    # Row 5
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[5]
    t50 = s0*c00 + s1*c10 + s2*c20 + s3*c30 + s4*c40 + s5*c50 + s6*c60 + s7*c70
    t51 = s0*c01 + s1*c11 + s2*c21 + s3*c31 + s4*c41 + s5*c51 + s6*c61 + s7*c71
    t52 = s0*c02 + s1*c12 + s2*c22 + s3*c32 + s4*c42 + s5*c52 + s6*c62 + s7*c72
    t53 = s0*c03 + s1*c13 + s2*c23 + s3*c33 + s4*c43 + s5*c53 + s6*c63 + s7*c73
    t54 = s0*c04 + s1*c14 + s2*c24 + s3*c34 + s4*c44 + s5*c54 + s6*c64 + s7*c74
    t55 = s0*c05 + s1*c15 + s2*c25 + s3*c35 + s4*c45 + s5*c55 + s6*c65 + s7*c75
    t56 = s0*c06 + s1*c16 + s2*c26 + s3*c36 + s4*c46 + s5*c56 + s6*c66 + s7*c76
    t57 = s0*c07 + s1*c17 + s2*c27 + s3*c37 + s4*c47 + s5*c57 + s6*c67 + s7*c77

    # Row 6
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[6]
    t60 = s0*c00 + s1*c10 + s2*c20 + s3*c30 + s4*c40 + s5*c50 + s6*c60 + s7*c70
    t61 = s0*c01 + s1*c11 + s2*c21 + s3*c31 + s4*c41 + s5*c51 + s6*c61 + s7*c71
    t62 = s0*c02 + s1*c12 + s2*c22 + s3*c32 + s4*c42 + s5*c52 + s6*c62 + s7*c72
    t63 = s0*c03 + s1*c13 + s2*c23 + s3*c33 + s4*c43 + s5*c53 + s6*c63 + s7*c73
    t64 = s0*c04 + s1*c14 + s2*c24 + s3*c34 + s4*c44 + s5*c54 + s6*c64 + s7*c74
    t65 = s0*c05 + s1*c15 + s2*c25 + s3*c35 + s4*c45 + s5*c55 + s6*c65 + s7*c75
    t66 = s0*c06 + s1*c16 + s2*c26 + s3*c36 + s4*c46 + s5*c56 + s6*c66 + s7*c76
    t67 = s0*c07 + s1*c17 + s2*c27 + s3*c37 + s4*c47 + s5*c57 + s6*c67 + s7*c77

    # Row 7
    s0, s1, s2, s3, s4, s5, s6, s7 = matrix[7]
    t70 = s0*c00 + s1*c10 + s2*c20 + s3*c30 + s4*c40 + s5*c50 + s6*c60 + s7*c70
    t71 = s0*c01 + s1*c11 + s2*c21 + s3*c31 + s4*c41 + s5*c51 + s6*c61 + s7*c71
    t72 = s0*c02 + s1*c12 + s2*c22 + s3*c32 + s4*c42 + s5*c52 + s6*c62 + s7*c72
    t73 = s0*c03 + s1*c13 + s2*c23 + s3*c33 + s4*c43 + s5*c53 + s6*c63 + s7*c73
    t74 = s0*c04 + s1*c14 + s2*c24 + s3*c34 + s4*c44 + s5*c54 + s6*c64 + s7*c74
    t75 = s0*c05 + s1*c15 + s2*c25 + s3*c35 + s4*c45 + s5*c55 + s6*c65 + s7*c75
    t76 = s0*c06 + s1*c16 + s2*c26 + s3*c36 + s4*c46 + s5*c56 + s6*c66 + s7*c76
    t77 = s0*c07 + s1*c17 + s2*c27 + s3*c37 + s4*c47 + s5*c57 + s6*c67 + s7*c77

    # ---- Column transform (direct computation per cell) --------------------
    # result[i][j] = sum_{k=0}^{7} C[k][i] * t[k][j]
    # Each cell computed in one expression — no intermediate loads/stores.
    result = [
        [c00*t00 + c10*t10 + c20*t20 + c30*t30 + c40*t40 + c50*t50 + c60*t60 + c70*t70,
         c00*t01 + c10*t11 + c20*t21 + c30*t31 + c40*t41 + c50*t51 + c60*t61 + c70*t71,
         c00*t02 + c10*t12 + c20*t22 + c30*t32 + c40*t42 + c50*t52 + c60*t62 + c70*t72,
         c00*t03 + c10*t13 + c20*t23 + c30*t33 + c40*t43 + c50*t53 + c60*t63 + c70*t73,
         c00*t04 + c10*t14 + c20*t24 + c30*t34 + c40*t44 + c50*t54 + c60*t64 + c70*t74,
         c00*t05 + c10*t15 + c20*t25 + c30*t35 + c40*t45 + c50*t55 + c60*t65 + c70*t75,
         c00*t06 + c10*t16 + c20*t26 + c30*t36 + c40*t46 + c50*t56 + c60*t66 + c70*t76,
         c00*t07 + c10*t17 + c20*t27 + c30*t37 + c40*t47 + c50*t57 + c60*t67 + c70*t77],

        [c01*t00 + c11*t10 + c21*t20 + c31*t30 + c41*t40 + c51*t50 + c61*t60 + c71*t70,
         c01*t01 + c11*t11 + c21*t21 + c31*t31 + c41*t41 + c51*t51 + c61*t61 + c71*t71,
         c01*t02 + c11*t12 + c21*t22 + c31*t32 + c41*t42 + c51*t52 + c61*t62 + c71*t72,
         c01*t03 + c11*t13 + c21*t23 + c31*t33 + c41*t43 + c51*t53 + c61*t63 + c71*t73,
         c01*t04 + c11*t14 + c21*t24 + c31*t34 + c41*t44 + c51*t54 + c61*t64 + c71*t74,
         c01*t05 + c11*t15 + c21*t25 + c31*t35 + c41*t45 + c51*t55 + c61*t65 + c71*t75,
         c01*t06 + c11*t16 + c21*t26 + c31*t36 + c41*t46 + c51*t56 + c61*t66 + c71*t76,
         c01*t07 + c11*t17 + c21*t27 + c31*t37 + c41*t47 + c51*t57 + c61*t67 + c71*t77],

        [c02*t00 + c12*t10 + c22*t20 + c32*t30 + c42*t40 + c52*t50 + c62*t60 + c72*t70,
         c02*t01 + c12*t11 + c22*t21 + c32*t31 + c42*t41 + c52*t51 + c62*t61 + c72*t71,
         c02*t02 + c12*t12 + c22*t22 + c32*t32 + c42*t42 + c52*t52 + c62*t62 + c72*t72,
         c02*t03 + c12*t13 + c22*t23 + c32*t33 + c42*t43 + c52*t53 + c62*t63 + c72*t73,
         c02*t04 + c12*t14 + c22*t24 + c32*t34 + c42*t44 + c52*t54 + c62*t64 + c72*t74,
         c02*t05 + c12*t15 + c22*t25 + c32*t35 + c42*t45 + c52*t55 + c62*t65 + c72*t75,
         c02*t06 + c12*t16 + c22*t26 + c32*t36 + c42*t46 + c52*t56 + c62*t66 + c72*t76,
         c02*t07 + c12*t17 + c22*t27 + c32*t37 + c42*t47 + c52*t57 + c62*t67 + c72*t77],

        [c03*t00 + c13*t10 + c23*t20 + c33*t30 + c43*t40 + c53*t50 + c63*t60 + c73*t70,
         c03*t01 + c13*t11 + c23*t21 + c33*t31 + c43*t41 + c53*t51 + c63*t61 + c73*t71,
         c03*t02 + c13*t12 + c23*t22 + c33*t32 + c43*t42 + c53*t52 + c63*t62 + c73*t72,
         c03*t03 + c13*t13 + c23*t23 + c33*t33 + c43*t43 + c53*t53 + c63*t63 + c73*t73,
         c03*t04 + c13*t14 + c23*t24 + c33*t34 + c43*t44 + c53*t54 + c63*t64 + c73*t74,
         c03*t05 + c13*t15 + c23*t25 + c33*t35 + c43*t45 + c53*t55 + c63*t65 + c73*t75,
         c03*t06 + c13*t16 + c23*t26 + c33*t36 + c43*t46 + c53*t56 + c63*t66 + c73*t76,
         c03*t07 + c13*t17 + c23*t27 + c33*t37 + c43*t47 + c53*t57 + c63*t67 + c73*t77],

        [c04*t00 + c14*t10 + c24*t20 + c34*t30 + c44*t40 + c54*t50 + c64*t60 + c74*t70,
         c04*t01 + c14*t11 + c24*t21 + c34*t31 + c44*t41 + c54*t51 + c64*t61 + c74*t71,
         c04*t02 + c14*t12 + c24*t22 + c34*t32 + c44*t42 + c54*t52 + c64*t62 + c74*t72,
         c04*t03 + c14*t13 + c24*t23 + c34*t33 + c44*t43 + c54*t53 + c64*t63 + c74*t73,
         c04*t04 + c14*t14 + c24*t24 + c34*t34 + c44*t44 + c54*t54 + c64*t64 + c74*t74,
         c04*t05 + c14*t15 + c24*t25 + c34*t35 + c44*t45 + c54*t55 + c64*t65 + c74*t75,
         c04*t06 + c14*t16 + c24*t26 + c34*t36 + c44*t46 + c54*t56 + c64*t66 + c74*t76,
         c04*t07 + c14*t17 + c24*t27 + c34*t37 + c44*t47 + c54*t57 + c64*t67 + c74*t77],

        [c05*t00 + c15*t10 + c25*t20 + c35*t30 + c45*t40 + c55*t50 + c65*t60 + c75*t70,
         c05*t01 + c15*t11 + c25*t21 + c35*t31 + c45*t41 + c55*t51 + c65*t61 + c75*t71,
         c05*t02 + c15*t12 + c25*t22 + c35*t32 + c45*t42 + c55*t52 + c65*t62 + c75*t72,
         c05*t03 + c15*t13 + c25*t23 + c35*t33 + c45*t43 + c55*t53 + c65*t63 + c75*t73,
         c05*t04 + c15*t14 + c25*t24 + c35*t34 + c45*t44 + c55*t54 + c65*t64 + c75*t74,
         c05*t05 + c15*t15 + c25*t25 + c35*t35 + c45*t45 + c55*t55 + c65*t65 + c75*t75,
         c05*t06 + c15*t16 + c25*t26 + c35*t36 + c45*t46 + c55*t56 + c65*t66 + c75*t76,
         c05*t07 + c15*t17 + c25*t27 + c35*t37 + c45*t47 + c55*t57 + c65*t67 + c75*t77],

        [c06*t00 + c16*t10 + c26*t20 + c36*t30 + c46*t40 + c56*t50 + c66*t60 + c76*t70,
         c06*t01 + c16*t11 + c26*t21 + c36*t31 + c46*t41 + c56*t51 + c66*t61 + c76*t71,
         c06*t02 + c16*t12 + c26*t22 + c36*t32 + c46*t42 + c56*t52 + c66*t62 + c76*t72,
         c06*t03 + c16*t13 + c26*t23 + c36*t33 + c46*t43 + c56*t53 + c66*t63 + c76*t73,
         c06*t04 + c16*t14 + c26*t24 + c36*t34 + c46*t44 + c56*t54 + c66*t64 + c76*t74,
         c06*t05 + c16*t15 + c26*t25 + c36*t35 + c46*t45 + c56*t55 + c66*t65 + c76*t75,
         c06*t06 + c16*t16 + c26*t26 + c36*t36 + c46*t46 + c56*t56 + c66*t66 + c76*t76,
         c06*t07 + c16*t17 + c26*t27 + c36*t37 + c46*t47 + c56*t57 + c66*t67 + c76*t77],

        [c07*t00 + c17*t10 + c27*t20 + c37*t30 + c47*t40 + c57*t50 + c67*t60 + c77*t70,
         c07*t01 + c17*t11 + c27*t21 + c37*t31 + c47*t41 + c57*t51 + c67*t61 + c77*t71,
         c07*t02 + c17*t12 + c27*t22 + c37*t32 + c47*t42 + c57*t52 + c67*t62 + c77*t72,
         c07*t03 + c17*t13 + c27*t23 + c37*t33 + c47*t43 + c57*t53 + c67*t63 + c77*t73,
         c07*t04 + c17*t14 + c27*t24 + c37*t34 + c47*t44 + c57*t54 + c67*t64 + c77*t74,
         c07*t05 + c17*t15 + c27*t25 + c37*t35 + c47*t45 + c57*t55 + c67*t65 + c77*t75,
         c07*t06 + c17*t16 + c27*t26 + c37*t36 + c47*t46 + c57*t56 + c67*t66 + c77*t76,
         c07*t07 + c17*t17 + c27*t27 + c37*t37 + c47*t47 + c57*t57 + c67*t67 + c77*t77],
    ]

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
