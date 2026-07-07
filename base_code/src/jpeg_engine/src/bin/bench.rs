use jpeg_engine::{idct_2d, gpu};
use std::env;
use std::fs;
use std::time::Instant;

fn create_blocks(count: usize) -> Vec<[f64; 64]> {
    (0..count).map(|i| {
        let mut b = [0.0f64; 64];
        for j in 0..64 {
            b[j] = (i as f64 * j as f64 % 256.0) - 128.0;
        }
        b
    }).collect()
}

fn bench_individual(blocks: &mut Vec<[f64; 64]>, iter_count: usize, label: &str) -> f64 {
    for block in blocks.iter_mut() {
        idct_2d(block.as_mut_ptr());
    }

    let rounds = 10;
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..iter_count {
            for block in blocks.iter_mut() {
                idct_2d(block.as_mut_ptr());
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        samples.push((elapsed / iter_count as f64) * 1000.0);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    println!("  {:<10} {:>6} blocks  {:.6} ms/iter  [cpu individual]", label, blocks.len(), median);
    median
}

fn bench_batch(kernel: &dyn gpu::GpuKernel, blocks: &mut Vec<[f64; 64]>, iter_count: usize, label: &str) -> f64 {
    kernel.batch_idct_2d(blocks).ok();

    let rounds = 10;
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..iter_count {
            kernel.batch_idct_2d(blocks).ok();
        }
        let elapsed = start.elapsed().as_secs_f64();
        samples.push((elapsed / iter_count as f64) * 1000.0);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    println!("  {:<10} {:>6} blocks  {:.6} ms/iter  [gpu/batch]", label, blocks.len(), median);
    median
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let iter_count: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5000);

    let kernel = gpu::create_kernel();

    let mut low = create_blocks(10);
    let mut medium = create_blocks(200);
    let mut high = create_blocks(10000);

    println!("Benchmark — {} iterations per round:", iter_count);
    let low_ms = bench_individual(&mut low, iter_count, "Low");
    let med_ms = bench_individual(&mut medium, iter_count, "Medium");
    let high_ms = bench_batch(&*kernel, &mut high, iter_count, "High");

    let fitness = high_ms * 0.5 + med_ms * 0.3 + low_ms * 0.2;

    let score_path = args.get(2).cloned().unwrap_or_else(|| "fitness.score".to_string());
    fs::write(&score_path, format!("{:.6}", fitness)).unwrap();
    println!(
        "\nFitness (0.5*high + 0.3*mid + 0.2*low): {:.6} ms/iter -> {}",
        fitness, score_path
    );
}
