#!/bin/bash
set -e
SRC="$(dirname "$0")/dct_engine_impl.c"
OUT="$(dirname "$0")/libdct_engine.so"
gcc -O3 -mavx2 -mfma -ffast-math -ftree-vectorize -fPIC -shared -o "$OUT" "$SRC" -lm
echo "Built $OUT"
