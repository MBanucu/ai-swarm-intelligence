import ctypes
import os
import math

# Load the C shared library
_lib_path = os.path.join(os.path.dirname(__file__), 'libdct_engine.so')
_lib = ctypes.CDLL(_lib_path)
_lib.idct_2d.argtypes = [ctypes.c_void_p]
_lib.idct_2d.restype = None

# Pre-allocate a reusable 64-double ctypes buffer (not thread-safe, but single-threaded test)
_buf = (ctypes.c_double * 64)()


def idct_2d(matrix):
    """8x8 2D IDCT via C implementation with SIMD vector extensions.

    Flattens the 8x8 Python list-of-lists into a 64-element contiguous
    double array, passes it to the C idct_2d(), then reconstructs the
    list-of-lists result.
    """
    # Flatten into pre-allocated ctypes buffer
    for i in range(8):
        row = matrix[i]
        base = i * 8
        _buf[base]     = row[0]
        _buf[base + 1] = row[1]
        _buf[base + 2] = row[2]
        _buf[base + 3] = row[3]
        _buf[base + 4] = row[4]
        _buf[base + 5] = row[5]
        _buf[base + 6] = row[6]
        _buf[base + 7] = row[7]

    _lib.idct_2d(ctypes.byref(_buf))

    # Unflatten result
    result0 = [_buf[0],  _buf[1],  _buf[2],  _buf[3],  _buf[4],  _buf[5],  _buf[6],  _buf[7]]
    result1 = [_buf[8],  _buf[9],  _buf[10], _buf[11], _buf[12], _buf[13], _buf[14], _buf[15]]
    result2 = [_buf[16], _buf[17], _buf[18], _buf[19], _buf[20], _buf[21], _buf[22], _buf[23]]
    result3 = [_buf[24], _buf[25], _buf[26], _buf[27], _buf[28], _buf[29], _buf[30], _buf[31]]
    result4 = [_buf[32], _buf[33], _buf[34], _buf[35], _buf[36], _buf[37], _buf[38], _buf[39]]
    result5 = [_buf[40], _buf[41], _buf[42], _buf[43], _buf[44], _buf[45], _buf[46], _buf[47]]
    result6 = [_buf[48], _buf[49], _buf[50], _buf[51], _buf[52], _buf[53], _buf[54], _buf[55]]
    result7 = [_buf[56], _buf[57], _buf[58], _buf[59], _buf[60], _buf[61], _buf[62], _buf[63]]
    return [result0, result1, result2, result3, result4, result5, result6, result7]


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
