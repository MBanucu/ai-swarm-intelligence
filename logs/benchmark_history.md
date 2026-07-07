| Generation | Attempt | ms/iter | Status | Timestamp |
|---|---|---|--|---|
| Gen 1 | Attempt 1 | 0.268817ms | WINNER | 2026-07-07T18:30:00+02:00 |
| Gen 2 | Attempt 1 | 2.263836ms | regression | 2026-07-07T22:00:01+02:00 |
| Gen 2 | Attempt 2 | 999.900000ms | extinction | 2026-07-07T22:19:33+02:00 |
| Gen 2 | Attempt 3 (child 3) | 0.544959ms | fused-IDCT + static-tables + no-GPU | 2026-07-07T23:30:00+02:00 |
| Gen 2 | Attempt 3 | 0.453775ms | regression | 2026-07-07T22:35:50+02:00 |
| Gen 2 | Attempt 4 (child 4) | 30.539ms* | fused-IDCT + static-tables + no-GPU + removed-OnceLock + fixed-alias-ref | 2026-07-07T23:55:00+02:00 |

*Note: benchmark updated to 6 batch sizes (10/250/1K/5K/25K/250K blocks) from old 3-batch (10/200/10000). Score not directly comparable to earlier rows.
| Gen 2 | Attempt 4 | 33.841341ms | regression | 2026-07-07T22:55:18+02:00 |
