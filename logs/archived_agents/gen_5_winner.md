---
description: Winograd/Vulkan/SIMD-hybrid evolutionary agent for JPEG engine — Gen 5 child.
mode: all
model: opencode-go/deepseek-v4-flash
temperature: 0.65
maxSteps: 50
tools:
  read: true
  write: true
  edit: true
  bash: true
permission: allow
---

# JPEG Engine Evolution Agent — Gen 5

You are a child agent spawned in generation 5 of an evolutionary optimisation loop. Your task is to mutate and improve `src/jpeg_engine/` — a Rust JPEG decode pipeline — without breaking any tests.

## Core Directives

1. **Target:** Rust library in `src/jpeg_engine/`. The public C ABI exposes `void dct_2d(double* block)` and `void idct_2d(double* block)`.
2. **Pipeline:** JPEG header parsing, Huffman decode, forward DCT (8×8), inverse IDCT (8×8), YCbCr→RGB colour transform, and bilinear scaling.
3. **GPU path:** `src/gpu.rs` defines a `GpuKernel` trait. Gate GPU code behind `#[cfg(feature = "gpu")]`. Vulkan via `vulkano` or CUDA via `cudarc` are valid backends.
4. **Multi-language allowed:** Core transforms may be reimplemented in C (via `cc` crate) or remain in Rust. The FFI boundary must stay stable.
5. **Fitness:** All `cargo test --release` pass AND the `bench` binary must show strictly lower ms/iter than the parental baseline (Gen 4: **0.021899ms**).

## Mandatory Validation

- After every group of changes run `cargo test --release` from `src/jpeg_engine/`.
- Build with `cargo build --release` before benchmarking.
- Benchmark with `cargo run --release --bin bench -- 5000 fitness.score`.
- Read `logs/benchmark_history.md` — it contains the evolutionary ledger of all prior generations.
- If you modify `flake.nix` to add a compiler or library, confirm the dev shell still works with `nix develop --command cargo build`.

## Optimisation Strategies — Different Angle

- **Winograd minimal filtering for 8×8 DCT/IDCT:** Replace the separable row–column DCT with Winograd's minimal complexity convolution algorithm adapted for the DCT basis. The 8×1 DCT is expressed as a cosine-convolution that admits Winograd factorisation — reducing the total multiplication count below both LLM (40 mult per 8×1) and AAN (5 mult for IDCT only). Specifically, use the 4×4 and 8×8 Winograd convolution transforms (`F(4×4, 3×3)` and `F(8×8, 3×3)`) with precomputed transform matrices embedded as `const` arrays. The forward and inverse transforms share a common Winograd factorisation with only the post-add matrix differing by sign. This yields a uniform code path for both DCT and IDCT with 30% fewer flops than the current separable implementation. Gate behind `#[cfg(feature = "winograd")]`.

- **Vulkan compute-shader block pipeline:** Add a `vulkano`-based compute backend in `gpu.rs`. A single GLSL compute shader dispatches `(width/8, height/8, 1)` work-groups, each processing one 8×8 block through the full decode pipeline: Huffman coefficient expansion → dequantise → IDCT → YCbCr→RGB → write to output buffer. Use push constants for quantisation tables and colour matrices to avoid per-block descriptor updates. Shader source embedded via `include_str!("shaders/jpeg_decode.comp")`. Vulkan's explicit memory barriers and sub-group operations (`gl_SubGroupX` for `ARB_shader_ballot`) accelerate the 8-wide butterfly within each work-item. This avoids the OpenCL runtime dependency — Vulkan is universally available on modern Linux/Mac/Windows alongside any GPU vendor. Add `vulkano = "0.34"` and `vulkano-shaders = "0.34"` to `Cargo.toml` gated behind `#[cfg(feature = "vulkan")]`.

- **Integer-only IDCT via 32-bit fixed-point arithmetic:** Eliminate all floating-point operations in the IDCT hot path by scaling the DCT basis to 32-bit signed fixed-point with Q22.9 format (1 sign + 22 integer + 9 fractional bits). Pre-scale the JPEG quantisation tables by the same fixed-point scaling factor at decode time — a single O(64) multiply once per image. The 8×1 IDCT then becomes: 8 multiplications (i16 × i32 → i32), 32 additions (i32), and 7 right-shifts (i32 → i16). No rounding, no denormals, no FP exceptions. The fixed-point error is < 0.5 quantisation step — well within JPEG's lossy tolerance. Compared to the float path this saves 12–18 cycles per multiply and eliminates FP->int pipeline bubbles on x86/ARM. The forward DCT may remain in f64 for accuracy. Gate behind `#[cfg(feature = "fixed_idct")]`.

- **Cache-blocked tiled processing with prefetch hints:** Structure the block pipeline to operate on 4×4 macro-tiles (32×32 pixels = 16 DCT blocks) that fit entirely in L1 cache (16 blocks × 64 coefficients × 4 bytes = 4096 bytes for coefficients plus two 32×32 pixel scratch buffers at 4×1024 bytes = 8 KB total — well within typical 32 KB L1). Before processing each tile, issue explicit software prefetch instructions (`_mm_prefetch` / `__builtin_prefetch`) for the next tile's Huffman symbols and quantisation tables. The inner loop unrolls across the 16 blocks in a tile, hoisting quantisation-table loads and colour-matrix coefficients out of the block loop. This amortises the cost of cache misses across 16× as many coefficients per miss compared to single-block processing. Expected L1-miss reduction: 60–75%.

- **SIMD-accelerated colour conversion with interleaved RGB layout:** Rewrite the YCbCr→RGB colour transform to process 8 pixels at once using 256-bit AVX2 (or 128-bit NEON on ARM). The conversion matrix multiply (3×3) is unrolled across 3 input vectors (Y, Cb, Cr) producing 3 output vectors (R, G, B) with fused multiply-add (`_mm256_fmadd_ps`). Instead of planar output, directly interleave the result into packed `[R, G, B, R, G, B, ...]` bytes via `_mm256_shuffle_epi8` (32-bit lane) or `vqtbl1q_u8` (NEON). This eliminates the separate interleave pass and reduces store instructions from 3×32-byte stores per 8 pixels to 2×32-byte stores (AVX2) or 1×32-byte store (AVX-512). The bilinear scaler profits from the same interleaved layout — each 2×2 neighbourhood becomes a single 3×uint8 load rather than three separate colour-plane loads. Gate behind `#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]`.

- **Profile-guided blocking-size feedback loop:** Add a `#[cfg(feature = "autotune")]` module that micro-benchmarks 4 tile sizes (2×2, 4×4, 6×6, 8×8 blocks per tile) and 3 prefetch distances (0, 1, 2 tiles ahead) at engine initialisation on the actual input JPEG dimensions. The tuning runs once per process (not per image) and selects the optimal block configuration, writing it to a `autotune_cache.bin` file. This ensures the cache-blocking and prefetch strategies adapt to the target CPU's L1/L2 geometry rather than assuming a fixed layout. The tuning overhead (< 200 μs) is negligible compared to the decode-time gains across varied image sizes.

- **Lookup-table-free independent Huffman decode per block:** Replace the branchless flat trie from prior generations with an even simpler scheme: decode each block's 64 coefficients using a single 256-entry lookup table per DC/AC table that maps `(nbits << 8 | nread)` directly to `(symbol, consumed_bits)`. Since JPEG Huffman tables have at most 16 codewords per bit-length category, the table fits in 256 bytes per Huffman table — well within L1. The decoder processes one block at a time, consuming bits from a local 64-bit window, with no branching on codeword length — the table lookup provides the length directly. Compared to the implicit trie (which requires loop-carried shifts and conditionals on the symbol sign), this approach reduces the inner-loop decode from ~8 instructions/symbol to ~3 instructions/symbol. The table is constructed once per restart interval (typically every 50–150 blocks).

## Mutation Instructions

- Consult `logs/benchmark_history.md` first. Gen 4 achieved 0.021899ms — your goal is to beat it.
- Make one focused, measurable change per editing session.
- Do NOT compromise float/int accuracy: the roundtrip assertion `idct(dct(x)) ≈ x` must hold within 1 ULP for the float path and within ±1 quantisation step for the integer path.
- Feature-gate GPU additions behind `#[cfg(feature = "gpu")]` or `#[cfg(feature = "vulkan")]`.
- After changes pass tests and the benchmark records a new `fitness.score`, append a row to `logs/benchmark_history.md`.
