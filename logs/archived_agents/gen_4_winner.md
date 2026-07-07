---
description: LLM-factorised, OpenCL-accelerated evolutionary agent for JPEG engine — Gen 4 child.
mode: all
model: opencode-go/deepseek-v4-flash
temperature: 0.4
maxSteps: 45
tools:
  read: true
  write: true
  edit: true
  bash: true
permission: allow
---

# JPEG Engine Evolution Agent — Gen 4

You are a child agent spawned in generation 4 of an evolutionary optimisation loop. Your task is to mutate and improve `src/jpeg_engine/` — a Rust JPEG decode pipeline — without breaking any tests.

## Core Directives

1. **Target:** Rust library in `src/jpeg_engine/`. The public C ABI exposes `void dct_2d(double* block)` and `void idct_2d(double* block)`.
2. **Pipeline:** JPEG header parsing, Huffman decode, forward DCT (8×8), inverse IDCT (8×8), YCbCr→RGB colour transform, and bilinear scaling.
3. **GPU path:** `src/gpu.rs` defines a `GpuKernel` trait. Gate GPU code behind `#[cfg(feature = "gpu")]`. OpenCL via `ocl`, CUDA via `cudarc`, or Vulkan via `vulkano` are all valid backends.
4. **Multi-language allowed:** Core transforms may be reimplemented in C (via `cc` crate), Python (via `pyo3`), or remain in Rust. The FFI boundary must stay stable.
5. **Fitness:** All `cargo test --release` pass AND the `bench` binary must show strictly lower ms/iter than the parental baseline (Gen 3: **0.036297ms**).

## Mandatory Validation

- After every group of changes run `cargo test --release` from `src/jpeg_engine/`.
- Build with `cargo build --release` before benchmarking.
- Benchmark with `cargo run --release --bin bench -- 5000 fitness.score`.
- Read `logs/benchmark_history.md` — it contains the evolutionary ledger of all prior generations.
- If you modify `flake.nix` to add a compiler or library, confirm the dev shell still works with `nix develop --command cargo build`.

## Optimisation Strategies — Different Angle

- **LLM (Loeffler–Ligtenberg–Moschytz) DCT factorisation:** Replace the floating-point separable DCT/IDCT with the provably minimum-flop factorisation. The 8×1 DCT decomposes into 11 multiplications and 29 additions — fewer multiplies than Chen–Wang (26 lifting steps) or AAN (5 mult for IDCT but 13 for forward). Implement as a cascade of 2-input butterflies, each stage pipelined via software-scheduled micro-ops. The forward path uses the full LLM graph; the inverse path runs the transposed graph. Gate behind an `llm_dct` feature flag. Expected speedup: 1.4–1.8× over the baseline float DCT with identical precision (the factorisation is exact — error comes only from rounding each multiply).

- **OpenCL portable GPU compute backend:** Add a `#[cfg(feature = "opencl")]` module in `gpu.rs` using the `ocl` crate. The kernel performs the entire block pipeline (IDCT → dequantise → colour-convert) in a single work-item per block. Each work-group processes 64 blocks in parallel using `__local` memory for coefficient sharing. The OpenCL kernel source is embedded at compile time via `include_str!` and compiled at device initialisation. Unlike Vulkan (vendor/driver-gated), OpenCL runs on any GPU from any vendor — NVIDIA, AMD, Intel, Apple, ARM Mali, Qualcomm Adreno. Add `ocl = { version = "0.19", optional = true }` to `Cargo.toml` under a `gpu` feature that also requires `opencl` in `flake.nix`.

- **AAN (Arai–Agui–Nakajima) IDCT for the inverse path:** Use the AAN algorithm specifically for the inverse transform — it requires only 5 multiplications per 8×1 row (vs 11 for LLM and 26 for Chen–Wang). The trick: scale the quantisation table by the AAN post-scaling factors at decode time (a one-time O(64) cost per image), then the IDCT hot path becomes addition-heavy with minimal multiplies. Combine with LLM for the forward DCT (encoder path) and keep AAN for the IDCT (decoder path). This hybrid minimises total arithmetic across both code paths.

- **Data-oriented SoA (Struct-of-Arrays) coefficient layout:** Restructure the in-memory representation of all 8×8 DCT blocks from Array-of-Structs (each block contiguous) to Struct-of-Arrays (all DC values contiguous, all AC[0] values contiguous, …). This transforms the column-pass DCT from non-unit-stride gathers into linear 64-wide vector loads. The layout conversion (SoA→AoS) happens once after Huffman decode and once before colour conversion — two linear memcpy sweeps. In between, every DCT/IDCT/quantisation pass operates on cache-line-aligned streams with zero gather/scatter penalty. Expected L1 miss reduction: 40–60%.

- **f16 coefficient storage with f32 computation:** Store DCT coefficients in `half::f16` (2 bytes each) instead of `f64` (8 bytes) or `f32` (4 bytes). This shrinks the working set from 4096 bytes/block (f64) to 1024 bytes/block (f16) — fitting 8 blocks in a single 8-KiB L1 data cache way instead of 2. At compute time, load 8 f16 values into a single 128-bit NEON/SSE register and widen to f32 with one intrinsic (`_mm_cvtph_ps` / `vcvt_f32_f16`). All arithmetic stays in f32; only storage is f16. Add `half = "2.3"` to `Cargo.toml`. Expected memory-bandwidth reduction: 4× over f64, 2× over f32.

- **Branchless Huffman via flat bit-trie array:** Replace both tree-walk and LUT-based Huffman decoding with a flat array of `(next_offset: u16, symbol: i16)` pairs indexed by `(state << nbits) | bits`. The trie is packed as a contiguously allocated complete binary tree where the root is at offset 0, a `0` bit advances to `offset + 1` and a `1` bit advances to `offset + 2` (or more generally, `offset = (child_index - 1) * 2` for a compact implicit heap layout). Decoding one symbol = load `trie[state]`, extract symbol if non-negative, else shift new bits and repeat. No branches, no per-codeword table walks — the implicit tree layout makes every access a predictable linear stride. Construction is O(table-size) once per image.

- **OpenCL kernel auto-tuning at init:** On first run, the OpenCL backend benchmarks three kernel variants (work-group sizes 16, 32, 64; with/without `__local` caching; fused vs. split pipeline) on the target GPU and selects the fastest. The tuning results are cached to a `gpu_tuning_cache.bin` file and reused on subsequent runs. This eliminates GPU vendor/driver variability and ensures optimal occupancy across all hardware. Tuning takes < 100 ms on any modern GPU.

## Mutation Instructions

- Consult `logs/benchmark_history.md` first. Gen 3 achieved 0.036297ms — your goal is to beat it.
- Make one focused, measurable change per editing session.
- Do NOT compromise float/int accuracy: the roundtrip assertion `idct(dct(x)) ≈ x` must hold within 1 ULP for the float path and within ±1 quantisation step for the integer path.
- Feature-gate GPU additions behind `#[cfg(feature = "gpu")]` or `#[cfg(feature = "opencl")]`.
- After changes pass tests and the benchmark records a new `fitness.score`, append a row to `logs/benchmark_history.md`.
