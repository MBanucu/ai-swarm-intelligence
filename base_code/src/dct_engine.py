import ctypes
import os
import math

# Load the C shared library
_lib_path = os.path.join(os.path.dirname(__file__), 'libdct_engine.so')
_lib = ctypes.CDLL(_lib_path)
_lib.idct_2d.argtypes = [ctypes.c_void_p]
_lib.idct_2d.restype = None

# Allocate a 32-byte aligned 64-double buffer using posix_memalign
# to satisfy AVX aligned load/store requirements.
_aligned_buf_ptr = ctypes.c_void_p()
_lib_posix = ctypes.CDLL(None)
_lib_posix.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
_lib_posix.posix_memalign.restype = ctypes.c_int

_buf_size = 64 * 8  # 64 doubles × 8 bytes each
ret = _lib_posix.posix_memalign(ctypes.byref(_aligned_buf_ptr), 32, _buf_size)
if ret != 0:
    raise MemoryError("posix_memalign failed")
_buf = (ctypes.c_double * 64).from_address(_aligned_buf_ptr.value)

# Pre-compute row slice indices for fast flattening
_row_slices = [slice(i * 8, (i + 1) * 8) for i in range(8)]


def idct_2d(matrix):
    """8x8 2D IDCT via C implementation with SIMD vector extensions.

    Flattens the 8x8 Python list-of-lists into a 64-element contiguous
    double array using fast slice assignment, passes it to the C idct_2d(),
    then reconstructs the list-of-lists result with a list comprehension.
    """
    # Fast flatten using ctypes slice assignment (one Python-level loop per row)
    for i in range(8):
        _buf[_row_slices[i]] = matrix[i]

    _lib.idct_2d(_aligned_buf_ptr)

    # Fast unflatten using list-of-rows comprehension
    return [
        [_buf[i], _buf[i+1], _buf[i+2], _buf[i+3],
         _buf[i+4], _buf[i+5], _buf[i+6], _buf[i+7]]
        for i in range(0, 64, 8)
    ]


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
