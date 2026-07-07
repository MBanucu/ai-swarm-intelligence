#!/usr/bin/env python3
"""Quick benchmark: 1000 iterations of idct_2d, output ms/iter."""
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.dct_engine import idct_2d

# Build a test block
block = [[float(i * j % 256 - 128) for j in range(8)] for i in range(8)]

# Warmup
for _ in range(100):
    idct_2d(block)

# Benchmark
start = time.perf_counter()
for _ in range(1000):
    idct_2d(block)
elapsed = time.perf_counter() - start

ms_per_iter = (elapsed / 1000) * 1000  # convert to ms
print(f"{ms_per_iter:.6f}")
