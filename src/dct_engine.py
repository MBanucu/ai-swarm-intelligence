import math
import struct

ALPHA = [1.0 / math.sqrt(2) if u == 0 else 1.0 for u in range(8)]

COSINE_TABLE = [
    [math.cos((2 * pos + 1) * freq * math.pi / 16) for pos in range(8)]
    for freq in range(8)
]


def idct_1d(block):
    result = [0.0] * 8
    for pos in range(8):
        s = 0.0
        for freq in range(8):
            s += ALPHA[freq] * block[freq] * COSINE_TABLE[freq][pos]
        result[pos] = s * 0.5
    return result


def idct_2d(matrix):
    temp = [idct_1d(row) for row in matrix]
    temp_t = list(zip(*temp))
    result_t = [idct_1d(list(col)) for col in temp_t]
    return [list(row) for row in zip(*result_t)]


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
