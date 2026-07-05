import ctypes
import math
import os
import time
import unittest

try:
    from src.dct_engine import idct_2d
except (ImportError, AttributeError):
    _so_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "libdct_engine.so"
    )
    _lib = ctypes.CDLL(_so_path)
    _lib.idct_2d.argtypes = [ctypes.c_void_p]
    _lib.idct_2d.restype = None
    _buf = (ctypes.c_double * 64)()

    def idct_2d(matrix):
        for i in range(8):
            row = matrix[i]
            base = i * 8
            for j in range(8):
                _buf[base + j] = row[j]
        _lib.idct_2d(ctypes.byref(_buf))
        return [[_buf[i * 8 + j] for j in range(8)] for i in range(8)]

from src.dct_engine import ycbcr_to_rgb, decode_mcu


class TestDCTEngine(unittest.TestCase):
    def test_idct_2d_zero_block(self):
        zero_block = [[0.0] * 8 for _ in range(8)]
        result = idct_2d(zero_block)
        self.assertEqual(len(result), 8)
        self.assertTrue(all(len(row) == 8 for row in result))
        for row in result:
            for val in row:
                self.assertTrue(abs(val) < 0.001, f"Expected near-zero, got {val}")

    def test_idct_2d_identity(self):
        identity_block = [[0.0] * 8 for _ in range(8)]
        identity_block[0][0] = 8.0
        result = idct_2d(identity_block)
        for row in result:
            for val in row:
                self.assertTrue(abs(val - 1.0) < 0.001, f"Expected near 1.0, got {val}")

    def test_ycbcr_to_rgb_black(self):
        r, g, b = ycbcr_to_rgb(0, 128, 128)
        self.assertEqual((r, g, b), (0, 0, 0))

    def test_ycbcr_to_rgb_white(self):
        r, g, b = ycbcr_to_rgb(255, 128, 128)
        self.assertEqual((r, g, b), (255, 255, 255))

    def test_ycbcr_to_rgb_clipping(self):
        r, g, b = ycbcr_to_rgb(-50, 0, 300)
        self.assertTrue(0 <= r <= 255)
        self.assertTrue(0 <= g <= 255)
        self.assertTrue(0 <= b <= 255)

    def test_decode_mcu_smoke(self):
        y_block = [[0.0] * 8 for _ in range(8)]
        cb_block = [[0.0] * 8 for _ in range(8)]
        cr_block = [[0.0] * 8 for _ in range(8)]
        pixels = decode_mcu(y_block, cb_block, cr_block)
        self.assertEqual(len(pixels), 8)
        self.assertTrue(all(len(row) == 8 for row in pixels))
        for row in pixels:
            for r, g, b in row:
                self.assertTrue(0 <= r <= 255)
                self.assertTrue(0 <= g <= 255)
                self.assertTrue(0 <= b <= 255)

    def test_idct_2d_performance(self):
        blocks = [[[float(i * j % 256 - 128) for j in range(8)] for i in range(8)]]
        start = time.perf_counter()
        for _ in range(1000):
            for block in blocks:
                idct_2d(block)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 5.0, f"IDCT too slow: {elapsed:.3f}s for 1000 iterations")


if __name__ == "__main__":
    unittest.main(verbosity=2)
