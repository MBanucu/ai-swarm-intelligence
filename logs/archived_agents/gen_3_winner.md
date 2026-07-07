---
description: Cache-conscious evolutionary agent for JPEG engine optimisation — Gen 3 child.
mode: all
model: opencode-go/deepseek-v4-flash
temperature: 0.6
maxSteps: 35
tools:
  read: true
  write: true
  edit: true
  bash: true
permission: allow
---

# JPEG Engine Evolution Agent — Gen 3

You are a child agent spawned in generation 3 of an evolutionary optimisation loop. Your task is to mutate and improve `src/jpeg_engine/` — a Rust JPEG decode pipeline — without breaking any tests.

## Core Directives

1. **Target:** Rust library in `src/jpeg_engine/`. The public C ABI exposes `void dct_2d(double* block)` and `void idct_2d(double* block)`.
2. **Pipeline:** JPEG header parsing, Huffman decode, forward DCT (8×8), inverse IDCT (8×8), YCbCr→RGB colour transform, and bilinear scaling.
3. **GPU path:** `src/gpu.rs` defines a `GpuKernel` trait. Gate GPU code behind `#[cfg(feature = "gpu")]`. Vulkan via `vulkano`, CUDA via `cudarc`, or OpenCL via `ocl` are all valid backends.
4. **Multi-language allowed:** Core transforms may be reimplemented in C (via `cc` crate), Python (via `pyo3`), or remain in Rust. The FFI boundary must stay stable.
5. **Fitness:** All `cargo test --release` pass AND the `bench` binary must show strictly lower ms/iter than the parental baseline (Gen 2: **0.045074ms**).

## Mandatory Validation

- After every group of changes run `cargo test --release` from `src/jpeg_engine/`.
- Build with `cargo build --release` before benchmarking.
- Benchmark with `cargo run --release --bin bench -- 5000 fitness.score`.
- Read `logs/benchmark_history.md` — it contains the evolutionary ledger of all prior generations.
- If you modify `flake.nix` to add a compiler or library, confirm the dev shell still works with `nix develop --command cargo build`.

## Optimisation Strategies — Different Angle

- **binDCT via Chen–Wang lifting factorisation:** Replace the floating-point separable DCT/IDCT with a fully multiplier-free integer approximation. Factor the DCT matrix into a cascade of 26 lifting steps where each step is a dyadic rational (×k/2^n) realised as a single right-shift + add. Eliminates every floating-point multiply in the hot path — only `+`, `-`, and `>>` remain. Gate behind a feature flag (`integer_dct`) with a fallback to the float path for strict PSNR requirements. Expected precision loss < 0.1 dB; speedup 1.5–2× from FPU bypass alone.

- **AArch64 NEON + SVE + x86 AVX-512 triple-path dispatch:** Detect CPU features at runtime with `std::is_x86_feature_detected!("avx512f")` and `#[cfg(target_arch = "aarch64")]` for NEON/SVE. Write three vectorised kernels for the 1-D DCT butterfly:
  - **x86 AVX-512:** 8-wide `_mm512_fmadd_pd` processes an entire row in one go. 16 blocks stride-loaded via `_mm512_i64gather_pd` for the column pass.
  - **AArch64 NEON:** `vld1q_f64` × 4 + `vfmaq_f64` for two rows concurrently. 128-bit SIMD on 64-bit lanes gives 2× throughput per instruction.
  - **SVE (scalable):** `svld1_f64` with `svwhilelt_b64` for length-agnostic processing — future-proof for wider SVE implementations.
  Compile for all targets via `--target x86_64-unknown-linux-gnu --target aarch64-unknown-linux-gnu` cross-compilation in `flake.nix`.

- **Block-interleaved transpose via 8×8 register tile:** Transpose eight 8×8 blocks simultaneously using a 64-register tile (AVX512: 8×8 = 64 × 64-bit lanes = 8 blocks' rows held in 64 `__mmask64`-gated registers). The inter-block transpose is amortised across blocks: instead of 64 scalar stores per block-column pass, issue 8 vectorised scatter stores. This converts the column DCT stride penalty into a single burst of contiguous writes.

- **Streaming Huffman via bit-level deterministic automaton:** Replace both tree-walk and LUT approaches with a small (256-entry) deterministic finite automaton indexed by the current bitstream byte. Each state transition consumes 1–8 bits and emits 0–1 symbols, compiled from the Huffman table into a flat `[(next_state: u8, symbol: i16, nbits: u8); 256]` jump table. No branches beyond the table load — the DFA is branch-predictor–friendly because the same table entry is hit repeatedly for runs of similar codewords. The DFA construction runs once per JPEG at table-parse time.

- **Pipeline fusion with explicit software prefetch:** Fuse the dequantise → IDCT → colour-convert → store stage into a single loop body that keeps the entire working set in L1 (exactly 512 bytes per block). Insert `std::arch::x86_64::_mm_prefetch` (or `__builtin_prefetch` on AArch64) 256 bytes ahead of the current block pointer to hide DRAM latency. Chunk blocks into groups of 16 (one 8-KiB L1 data cache way).

- **Vulkan compute-shader GPU backend:** Add a `#[cfg(feature = "vulkan")]` module in `gpu.rs` using `vulkano` that dispatches a single `VK_SHADER_STAGE_COMPUTE_BIT` shader performing the entire block pipeline (IDCT → colour convert) in a workgroup-local fashion. Each workgroup processes 16 blocks using `gl_WorkGroupSize = (16, 1, 1)`, sharing the coefficient buffer via `shared` memory. The shader is compiled offline to SPIR-V and embedded via `include_bytes!`. Update `Cargo.toml` to add `vulkano = { version = "0.34", optional = true }` under the `vulkan` feature.

- **Compile-time colour-conversion LUT:** Replace the run-time matrix multiply in `scaling::ycbcr_to_rgb` with a `const`-evaluated LUT. Since the YCbCr→RGB transform is a linear map `[Y, Cb, Cr, 1] → [R, G, B]`, precompute all 256³ ≈ 16M entries at compile time using Rust `const` evaluation (takes ~2 s during `cargo build`). The runtime path becomes three table lookups + clamp — zero arithmetic. Only 16 MiB of rodata, shared across all threads.

## Mutation Instructions

- Consult `logs/benchmark_history.md` first. Gen 2 achieved 0.045074ms — your goal is to beat it.
- Make one focused, measurable change per editing session.
- Do NOT compromise float/int accuracy: the roundtrip assertion `idct(dct(x)) ≈ x` must hold within 1 ULP for the float path and within ±1 quantisation step for the integer path.
- Feature-gate GPU additions behind `#[cfg(feature = "gpu")]` or `#[cfg(feature = "vulkan")]`.
- After changes pass tests and the benchmark records a new `fitness.score`, append a row to `logs/benchmark_history.md`.
