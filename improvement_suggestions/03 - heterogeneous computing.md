In high-performance computing, this is called **heterogeneous computing** or **co-processing**. Instead of choosing between the CPU and GPU, you split the workload so that both processors operate in parallel.

For your specific JPEG engine evolutionary sandbox, utilizing both at the same time is a highly effective strategy to maximize throughput and achieve the lowest possible `ns/block` score.

Here is how you can achieve this simultaneously, along with the structural patterns to implement it in your Rust/OpenCL engine.

---

### 1. Workload Partitioning (Dynamic Splitting)

Because you have a vector of blocks (`Vec<[f32; 64]>`) being evaluated in a batch, you don't have to send all of them to one device. You can partition the batch:

* **The CPU** takes a small chunk (e.g., 20% of the blocks) and processes them using multi-threaded SIMD via `rayon`.
* **The GPU** takes the remaining massive chunk (e.g., 80% of the blocks) via an asynchronous OpenCL command queue.

### 2. Implementation Strategy in Rust

To do this efficiently without blocking the CPU while the GPU is processing, you must leverage **asynchronous execution** and **non-blocking OpenCL event callbacks**.

The architecture usually follows this pipeline:

1. **Split the Slice:** Divide your input buffer into two slices based on an optimal performance ratio (e.g., `let (gpu_part, cpu_part) = blocks.split_at_mut(gpu_share);`).
2. **Spawn the GPU Task Asynchronously:** Write the `gpu_part` buffer to the OpenCL device buffer, enqueue the kernel execution, and enqueue the read-back. Crucially, use an **out-of-order or non-blocking command queue** so your host thread doesn't sleep.
3. **Run the CPU Task Immediately:** While the GPU is computing in hardware, the main Rust thread uses Rayon (`par_chunks_mut`) to crunch through `cpu_part`.
4. **Synchronize at the End:** Once the CPU finishes, block or wait on the OpenCL completion event (`clWaitForEvents` or `.wait()` in Rust OpenCL bindings) to ensure the GPU data has copied back.

Here is a simplified structural layout of how this maps to your `lib.rs`:

```rust
pub fn idct_2d_batch_simultaneous(blocks: &mut [Block]) {
    let total_blocks = blocks.len();
  
    // Choose a sweet spot ratio (e.g., 80% GPU, 20% CPU)
    // Small batches might drop to 0% GPU to avoid context overhead
    let gpu_ratio = 0.80; 
    let gpu_count = (total_blocks as f32 * gpu_ratio) as usize;

    let (gpu_blocks, cpu_blocks) = blocks.split_at_mut(gpu_count);

    // 1. Kick off GPU Processing Asynchronously
    // Inside this function, enqueue writing, execution, and reading without blocking the host thread
    let gpu_event = gpu::enqueue_idct_async(gpu_blocks);

    // 2. Compute CPU portion concurrently using Rayon
    // The CPU is fully utilized while the GPU works independently on the PCIe bus
    cpu_blocks.par_chunks_mut(32).for_each(|chunk| {
        idct::idct_2d_cpu_simd(chunk);
    });

    // 3. Block until the GPU finishes copying data back
    if let Some(event) = gpu_event {
        event.wait().unwrap(); 
    }
}
```

---

### Key Bottlenecks to Watch Out For

* **The PCIe Bus BottleNeck:** Passing data to the GPU takes time. If the GPU portion finishes too quickly, the CPU might spend more time waiting for the memory to transfer back than it saved. You have to find the "sweet spot" batch size where simultaneous processing makes sense.
* **OpenCL Thread Safety:** Ensure that your OpenCL context and command queue handles are safely shared across threads (`Send` + `Sync`), or keep the GPU operations constrained to the main dispatching thread.
* **The "Straggler" Problem:** If you give the CPU too much work, the GPU will sit idle waiting for the CPU to finish its slice, or vice-versa. Because your evolutionary algorithm (`evolver.py`) runs benchmarks across various sizes (`1k`, `25k`, `250k`), your engine should ideally calculate the splitting ratio dynamically depending on the size of the incoming batch.
