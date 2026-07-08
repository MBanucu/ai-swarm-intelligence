## Performance Trends

- **Gen 1 winner (attempt 6): 56.934 ns/block** — lowered `GPU_THRESHOLD` from 500K to 20K, routed 25K+250K batches through OpenCL AAN butterfly kernel. Largest single speedup of any mutation.
- **Gen 2 attempt 1: 72.625 ns/block (REGRESSION, +27.6%)** — performance *worsened* vs Gen 1 winner. Likely caused by a feature-flag mismatch that silently disabled GPU dispatch, forcing all 250K blocks (50% weight) through the CPU path.
- **Critical finding: `Cargo.toml` lists `default = []` with no features enabled.** The benchmark commands in AGENTS.md (`cargo run --release --bin bench`) never pass `--features gpu`. Without this flag, `#[cfg(feature = "gpu")]` in `lib.rs:72` evaluates to false, `gpu_kernel()` returns `None`, and **every batch runs on CPU** — including the 250K batch. The entire GPU code path is dead unless explicitly activated. This is the root cause of the Gen 2 regression.

## Code Hotspots

| File | Lines | Issue |
|---|---|---|
| `lib.rs:66-72` | `GPU_THRESHOLD = 20_000` is correct, but `#[cfg(feature = "gpu")]` never fires without `--features gpu` | Feature gate silences GPU init entirely |
| `Cargo.toml:12-16` | `default = []`; `gpu = ["ocl", "rayon", "opencl"]` | GPU features opt-in — not activated by plain `cargo run --release` |
| `idct.rs:160-198` | CPU path already has adaptive chunking `(n / (num_threads * 4)).max(128)` and transposed-temp store | CPU path is well-optimized; diminishing returns here |
| `gpu.rs:158-267` | OpenCL AAN butterfly kernel + transposed private store | Correct algorithm but one-block-per-thread model underutilizes GPU; `batch_fdct` is a no-op placeholder |
| `gpu.rs:290-327` | `OpenClKernel` holds `Mutex<Option<Buffer>>` + serial sync writes/reads | Every `batch_idct_2d` call does synchronous write→enqueue→readback; no overlap with CPU |

## Recommended Strategies (TRY THESE)

### 1. Fix feature-gate dead code in Cargo.toml (`Cargo.toml:12`)
Change `default = []` to `default = ["gpu", "rayon"]` so plain `cargo run --release --bin bench` activates OpenCL dispatch. Without this, every GPU optimization is dead code. The Gen 1 winner only worked because the evolver explicitly passed `--features gpu`; Gen 2 attempt 1 apparently ran bare `--release` and got CPU-only scores.

Alternatively, change `lib.rs:72` from `#[cfg(feature = "gpu")]` to `#[cfg(any(feature = "gpu", feature = "opencl"))]` to decouple the GPU init gate from the compound feature. But fixing the default is simpler and less fragile.

### 2. Heterogeneous CPU+GPU simultaneous dispatch (`lib.rs:103-112`)
Replace the current CPU-OR-GPU branch with split dispatch for large batches (≥25K). Send ~80% of blocks to GPU asynchronously, process the remaining ~20% on CPU via rayon *while* the GPU kernel runs. Overlaps kernel execution + PCIe readback with useful CPU work. For 250K blocks (50% weight), this can cut wall-clock time by ~15-25%.

Implementation sketch in `lib.rs`:
```
if use_gpu && blocks.len() >= GPU_THRESHOLD {
    let split = (blocks.len() as f32 * 0.8) as usize;
    let (gpu_part, cpu_part) = blocks.split_at_mut(split);
    // GPU dispatch + readback on gpu_part (async)
    // rayon on cpu_part (immediately)
    // wait for GPU completion
}
```

Requires making the OpenCL dispatch truly async (use ocl events + `wait()` instead of blocking readback). The `OpenClKernel` struct in `gpu.rs:290-327` already has `Send + Sync` via the unsafe impl; the trait method is `&self` so concurrent calls are safe with proper event management.

### 3. GPU workgroup optimization: switch to `__local` shared memory (`gpu.rs:221-257`)
Change the OpenCL kernel from one-thread-per-block with `float tmp[64]` private storage to an 8×8 workgroup (`@workgroup_size(8, 8)`) sharing one block via `__local float shared_block[8][8]`. This:
- Reduces per-thread register pressure (more warps/SIMD units per CU, better occupancy)
- Makes the transposed store/column-read pattern coalesced through local memory instead of private-to-global round trips
- Requires `barrier(CLK_LOCAL_MEM_FENCE)` between pass 1 and pass 2

Current kernel launches `global_work_size = n` with `local_work_size = min(n, 64)`. New model launches `global_work_size = n * 64` with `local_work_size = 64` (reshaped to 2D `(n*8, 8)` for workgroup mapping). Each workgroup handles one block; its 64 threads each compute one coefficient.

### 4. GPU_THRESHOLD micro-tuning (`lib.rs:66`)
Current value 20_000 only catches 25K+250K batches. The 5K batch (10% weight) stays on CPU. With a well-optimized GPU kernel, try lowering to 5_000 to capture 5K as well. Profile first: if GPU dispatch overhead (<500µs) is ≤ the CPU cost for 5K blocks (~5M blocks at 56ns = ~280µs), this is a net win.

### 5. Eliminate batch_ycbcr_to_rgb kernel rebuild overhead (`gpu.rs:438-450`)
The OpenCL color transform kernel is rebuilt from scratch on every call via `Kernel::builder()`. Cache the kernel handle across invocations. Not benchmarked today, but prepares the path for full-decode benchmarks in future generations.

## Strategies to AVOID

- **WebGPU / WGSL (`improvement_suggestions/02`)**: Replacing the existing OpenCL kernel with WebGPU requires the `wgpu` crate, WGSL compiler, new Nix dependencies, and a full async runtime rewrite. The OpenCL kernel is already correct AAN butterfly with the same arithmetic as the CPU path. WebGPU adds complexity without proven speedup for this specific workload. The gen_1_winner.md explicitly bans CUDA/WebGPU.

- **Huffman decoder or header parser changes**: These code paths are not exercised by the benchmark (`bench.rs` calls only `idct_2d_batch`). Any change risks breaking tests in `header.rs` or introducing roundtrip failures in `huffman.rs` with zero fitness benefit.

- **Modifying `bench.rs` or FFI signatures**: `bench.rs` contains the reference IDCT validation and is immutable. Changing `idct_2d_batch`'s C ABI signature in `lib.rs:115` breaks the benchmark binary. The validation step (`bench.rs:35-127`) checks engine output against a naive O(N⁴) reference IDCT with epsilon=0.5; any numerical changes to the IDCT implementation must preserve this tolerance.

- **Touching the `cuda` feature flag**: `Cargo.toml:14` has `cuda = []` with zero dependencies, matching the gen_1_winner.md note that it's a dead placeholder. Activating it enables nothing and may cause confusion.

- **Large changes to the CPU IDCT arithmetic**: The current butterfly-even + direct-odd decomposition (idct.rs:50-87) already uses 22 mul + ~32 add per 1D transform — near the theoretical minimum for IEEE-accurate IDCT. AAN butterfly on CPU would save ~4 mul per 1D pass (16 mul per 2D block, ~0.5% of total work at 250K blocks) but requires `#[target_feature]` dispatch and risks validation failure. Diminishing return vs the GPU or heterogeneous strategies above.
