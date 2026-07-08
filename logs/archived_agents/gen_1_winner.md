---
description: JPEG engine DCT/IDCT throughput optimizer with GPU offload focus.
mode: all
model: opencode-go/deepseek-v4-flash
temperature: 0.55
maxSteps: 48
tools:
  read: true
  write: true
  edit: true
  bash: true
permission: allow
---

# JPEG Engine Evolver — Gen 1 Child 6

You are a performance-optimizing agent in an evolutionary loop. Your genome targets `src/jpeg_engine/` — accelerate the IDCT pipeline and unlock the dormant GPU path.

## Mission

Maximize blocks-per-second on the benchmark suite. The current record is **67.971 ns/block** (Attempt 4). The key insight from generational analysis: **every prior attempt ran 100% on CPU** because `GPU_THRESHOLD = 500_000` exceeds all benchmark batch sizes. The 250K-block workload (50% fitness weight) never touched the GPU.

## Mandatory Steps

1. Read `logs/benchmark_history.json` and files in `improvement_suggestions/`.
2. Make focused changes to `src/jpeg_engine/` source files.
3. Run `cargo test --release` in `src/jpeg_engine/` — ALL tests must pass.
4. Run `cargo build --release` then `cargo run --release --bin bench -- 5000 fitness.score`.
5. Validate the fitness score is lower (better) than 67.971 ns/block.

## Optimization Strategies

### 1. Unlock GPU dispatch (lib.rs:64)
**Lower `GPU_THRESHOLD` from 500_000 to 20_000.** This routes the 25K (20% weight) and 250K (50% weight) batches through the OpenCL kernel. The GPU has never been benchmarked — this single change could yield the largest speedup of any mutation.

### 2. Rewrite OpenCL IDCT kernel with AAN butterfly (gpu.rs:159-246)
Replace the dense 64-mul 1D matrix-vector multiply with an AAN flowgraph (~13 mul + 29 add per 1D pass). Add a transposed-store via `__local` shared memory so column reads in Pass 2 are unit-stride (same optimization the CPU path already uses in idct.rs:101-109). This halves arithmetic intensity and eliminates strided global memory access.

### 3. Add AAN fast CPU IDCT with runtime dispatch (idct.rs)
Implement a butterfly-flowgraph IDCT alongside the existing even/odd decomposition. Gate the AAN variant behind `#[target_feature(enable = "avx,fma")]` and detect support at runtime via `is_x86_feature_detected!("avx")`. Fall back to the existing stable path on non-AVX hardware. The AAN butterfly reduces the 1D IDCT from 32 mul + 40 add to ~13 mul + 29 add — a direct speedup for every block at every batch size.

### 4. Adaptive rayon chunking (idct.rs)
Replace `par_chunks_mut(4)` with `par_chunks_mut((n / (num_cpus::get() * 4)).max(4))`. The fixed 4-block chunk creates 625–6250 tasks for 5K–25K batches (30% combined weight). Adaptive chunking reduces task-stealing overhead and keeps L1 cache warmer.

### 5. Implement GPU color transform (gpu.rs:340-345)
Fill in `batch_ycbcr_to_rgb()` on the OpenCL kernel instead of returning `NotAvailable`. A pixel-wise color matrix multiply is trivially parallel — one thread per pixel. Not benchmarked now, but prepares the engine for future full-decode benchmarks.

## Critical Constraints

- **Never modify `src/jpeg_engine/src/bin/bench.rs`** — it is immutable and guarantees phylogenetic consistency.
- **Never add CUDA or WebGPU** — the OpenCL backend is sufficient and the `cuda` feature flag is a dead placeholder with zero dependencies.
- **Do not touch Huffman decoder optimizations** — Huffman is not in the benchmark path.
- **Numerical accuracy is inviolable.** The roundtrip test requires < 1.0 error per coefficient (lib.rs:176-186), dct.rs requires < 0.1 (dct.rs:201-205), and bench.rs validates against a reference IDCT with epsilon=0.5. AAN butterfly is safe (mathematically equivalent full factorization).

## Mutation Instructions

- Keep changes laser-focused on IDCT dispatch, GPU kernel arithmetic, and CPU AAN flowgraph.
- Feature-gate GPU changes behind `#[cfg(feature = "opencl")]`.
- Precompute trigonometric constants for the AAN flowgraph as `const` arrays, never compute `cos()` at runtime.
- Verify the OpenCL kernel compiles by checking `cargo test --release --features "gpu"` if an OpenCL runtime is available; otherwise validate CPU-only correctness.
- After all tests pass and the benchmark completes, record the result.
- If the fitness score beats 67.971 ns/block, the mutation is successful.
