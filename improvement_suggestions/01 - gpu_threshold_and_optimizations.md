
Based on the codebase structure and implementation details, there is a major architectural mismatch between the benchmark configuration and the dispatch logic that completely prevents the GPU acceleration from ever being used.

The following improvements can be made:

### 1. Fix the `GPU_THRESHOLD` Bottleneck (Critical Performance Fix)

In `src/jpeg_engine/src/lib.rs`, the code defines an auto-dispatch threshold for routing batches to the GPU:

```rust
// ── GPU auto-dispatch threshold ── 
// Batches ≥ this size are routed to GPU (when available)...
const GPU_THRESHOLD: usize = 500_000;

```

However, in `src/jpeg_engine/src/bin/bench.rs`, the largest workload configured in the benchmark suite is only **250,000 blocks**:

```rust
Batch { size: 250000, label: "250K",   weight: 0.50 },

```

Because `250,000 < 500,000`, **the GPU implementation is never triggered during the benchmarks**, leaving it completely dormant while the CPU processes the largest workload (which holds a dominant 50% fitness weight).

* **Improvement:** Lower `GPU_THRESHOLD` to a value like `20,000` or `25,000` so that the `25K` and `250K` batch sizes automatically cross over to the accelerated GPU pipelines as intended.

---

### 2. Introduce WebGPU (WGSL) Compute Shaders

The project currently relies on OpenCL (`ocl`) for GPU acceleration. While robust, OpenCL setup and memory management in Rust can have significant runtime overhead and lack cross-platform uniformity across modern desktop ecosystems (Vulkan/Metal/DX12).

* **Improvement:** Implement the `GpuKernel` interface using WebGPU shaders via the `wgpu` crate. By leveraging an $8 \times 8$ workgroup size (`@workgroup_size(8, 8)`), each thread maps to a single frequency coefficient of a JPEG block.
* **Shared Memory Matrix Transposition:** Since the 2D-IDCT is mathematically separable, you can load data into local GPU workgroup shared memory (`var<workgroup> shared_block: array<array<f32, 8>, 8>`), run a fast 1D row IDCT, synchronize with a `workgroupBarrier()`, and then process the columns. This avoids expensive global device memory reads and writes.

---

### 3. Replace Naive Loops with Fast IDCT Algorithms

The baseline implementation uses explicit loops for the 1D passes, which runs in $O(N^2)$ time per vector.

* **Improvement:** Implement optimized 1D flowgraph algorithms like **AAN (Arai, Agui, and Nakajima)** or **Chen's fast IDCT algorithm**. Unrolling these algorithms entirely into explicit butterfly additions, subtractions, and constant bit-shifts removes loop overhead and minimizes floating-point multiplication steps.

---

### 4. Enable Rayon Chunk Size Tuning

In `src/jpeg_engine/src/idct.rs`, the CPU parallel processing splits work using a fixed chunk size of 4 blocks:

```rust
blocks.par_chunks_mut(4).for_each(|chunk| { ... })

```

For medium-to-large CPU batches (like 5,000 or 25,000 blocks), a chunk size of 4 creates excessive scheduling overhead for the thread pool work-stealer.

* **Improvement:** Make the chunk size adaptive based on the input length (e.g., `(n / (num_cpus * 4)).max(4)`) to maximize cache locality and keep worker threads continuously saturated without task-switching saturation.
