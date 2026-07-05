# Benchmark History

| Gen | Child | Timestamp | ms/iter | Notes |
|-----|-------|-----------|---------|-------|
| Gen 0 | baseline | - | 0.086 | Original naive fully-unrolled row-column IDCT |
| Gen 1 | Child 1 | 2026-07-05 | 0.045 | Symmetry-based even/odd decomposition (halves multiplications) |
| Gen 1 | Child 1 | 0.042274ms | 2026-07-05T17:49:05+02:00 |
