use jpeg_engine::idct_2d_batch;
use jpeg_engine::idct_2d_batch_cpu;
use jpeg_engine::idct_2d_batch_gpu;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::env;
use std::fs;
use std::f32::consts::PI;
use std::time::Instant;

fn c(k: usize) -> f32 {
    if k == 0 { 1.0 / (2.0f32).sqrt() } else { 1.0 }
}

/// Naive O(N^4) reference IDCT — mathematically correct implementation
/// of the 2D IDCT formula. Used to validate the engine's output.
fn reference_idct_2d(block: &[f32; 64]) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            let mut sum = 0.0;
            for v in 0..8 {
                for u in 0..8 {
                    sum += c(u) * c(v)
                        * block[v * 8 + u] as f32
                        * ((2 * x + 1) as f32 * u as f32 * PI / 16.0).cos()
                        * ((2 * y + 1) as f32 * v as f32 * PI / 16.0).cos();
                }
            }
            out[y * 8 + x] = 0.25 * sum;
        }
    }
    out
}

fn validate() {
    let mut rng = StdRng::seed_from_u64(0xbad_f00d_cafe_1337);
    let epsilon = 0.5;

    let mut test_cases: Vec<[f32; 64]> = Vec::new();

    // DC-only block
    let mut dc_block = [0.0f32; 64];
    dc_block[0] = 128.0;
    test_cases.push(dc_block);

    // Single AC coefficient at various positions
    for k in 1..5 {
        let mut ac_block = [0.0f32; 64];
        ac_block[k] = 64.0;
        test_cases.push(ac_block);
    }

    // Random blocks with realistic magnitude
    for _ in 0..20 {
        let mut b = [0.0f32; 64];
        b[0] = (rng.gen::<f32>() - 0.5) * 2048.0;
        for j in 1..64 {
            let scale = 256.0 / (j as f32).sqrt();
            b[j] = (rng.gen::<f32>() - 0.5) * 2.0 * scale;
        }
        test_cases.push(b);
    }

    println!("Validation: {} test cases against reference IDCT...", test_cases.len());

    for (i, block) in test_cases.iter().enumerate() {
        let expected = reference_idct_2d(block);
        let mut engine_block = *block;
        idct_2d_batch(engine_block.as_mut_ptr() as *mut f32, 1);

        for j in 0..64 {
            let diff = (engine_block[j] - expected[j]).abs();
            if diff > epsilon {
                eprintln!(
                    "VALIDATION FAILED: test case {} coefficient {}: engine={:.6} reference={:.6} diff={:.6}",
                    i, j, engine_block[j], expected[j], diff
                );
                std::process::exit(1);
            }
        }
    }
    println!("Validation: all {} test cases passed.", test_cases.len());

    // Batch validation — full reference for small sizes, spot-check for large
    let batch_sizes: &[(usize, usize)] = &[
        (10, 0),        // 0 = full comparison
        (1000, 0),
        (25000, 0),
        (250000, 20),   // spot-check first 20 blocks only
    ];

    for &(batch_size, check_n) in batch_sizes {
        let check_count = if check_n == 0 { batch_size } else { check_n };

        let mut blocks: Vec<[f32; 64]> = (0..batch_size).map(|i| {
            let mut b = [0.0f32; 64];
            b[0] = (i as f32 - 0.5) * 64.0;
            for j in 1..64 {
                b[j] = ((i * j) as f32 % 128.0) - 64.0;
            }
            b
        }).collect();

        let expected: Vec<[f32; 64]> = blocks[..check_count]
            .iter().map(|b| reference_idct_2d(b)).collect();

        let ptr = blocks.as_mut_ptr() as *mut f32;
        idct_2d_batch(ptr, blocks.len() as u32);

        for (k, (engine_block, ref_block)) in blocks[..check_count]
            .iter().zip(expected.iter()).enumerate()
        {
            for j in 0..64 {
                let diff = (engine_block[j] - ref_block[j]).abs();
                if diff > epsilon {
                    eprintln!(
                        "BATCH VALIDATION FAILED: batch_size={} block={} coeff={}: engine={:.6} reference={:.6} diff={:.6}",
                        batch_size, k, j, engine_block[j], ref_block[j], diff
                    );
                    std::process::exit(1);
                }
            }
        }
        println!("Validation: batch_size={} — {} blocks checked, all passed.",
                 batch_size, check_count);
    }
}

fn create_blocks(count: usize) -> Vec<[f32; 64]> {
    let mut rng = StdRng::seed_from_u64(0xdead_beef_cafe_babe);
    (0..count).map(|_| {
        let mut b = [0.0f32; 64];
        let dc = (rng.gen::<f32>() - 0.5) * 2048.0;
        let dc_q = 8.0;
        b[0] = (dc / dc_q).round() * dc_q;
        for j in 1..64 {
            let scale = 256.0 / (j as f32).sqrt();
            let val = (rng.gen::<f32>() - 0.5) * 2.0 * scale;
            let q = (j as f32 / 8.0 + 1.0) * 4.0;
            b[j] = (val / q).round() * q;
        }
        b
    }).collect()
}

/// Scale iteration count inversely with batch size so total work stays
/// bounded. Large batches get fewer iterations per round to avoid
/// multi-minute benchmarks on 250K+ blocks.
fn adaptive_iters(batch_size: usize, max_iters: usize) -> usize {
    (5_000_000 / batch_size).max(1).min(max_iters)
}

fn benchmark(blocks: &mut Vec<[f32; 64]>, iter_count: usize, mode: &str) -> f64 {
    let batch_size = blocks.len();
    let run = |b: &mut Vec<[f32; 64]>| match mode {
        "cpu" => idct_2d_batch_cpu(b.as_mut_ptr() as *mut f32, batch_size as u32),
        "gpu" => idct_2d_batch_gpu(b.as_mut_ptr() as *mut f32, batch_size as u32),
        _ => idct_2d_batch(b.as_mut_ptr() as *mut f32, batch_size as u32),
    };
    run(blocks); // warm-up

    let rounds = 10;
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..iter_count {
            run(blocks);
        }
        let elapsed = start.elapsed().as_secs_f64();
        samples.push(elapsed / iter_count as f64 / batch_size as f64 * 1_000_000_000.0);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    median
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut max_iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5000);
    let mut mode = "auto";
    let mut score_path = args.get(2).cloned().unwrap_or_else(|| "fitness.score".to_string());

    for i in 1..args.len() {
        if args[i] == "--mode" {
            mode = args.get(i + 1).map(|s| s.as_str()).unwrap_or("auto");
        } else if args[i] == "--iters" {
            max_iters = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(5000);
        } else if args[i] == "-o" {
            score_path = args.get(i + 1).cloned().unwrap_or_else(|| "fitness.score".to_string());
        }
    }

    validate();

    // Batch sizes spanning realistic JPEG workloads:
    //    10       — single block (20% old weight)
    //    250      — small thumbnail (~2 KP)
    //    1K       — medium tile / web image (~8 KP)
    //    5K       — large tile (~40 KP)
    //    25K      — HD frame (~200 KP, baseline GPU crossover)
    //    250K     — ~5 MP image (typical 5 MB JPEG, 50% weight)
    struct Batch {
        size: usize,
        label: &'static str,
        weight: f32,
    }
    let batches = [
        Batch { size: 10,     label: "10",     weight: 0.03 },
        Batch { size: 250,    label: "250",    weight: 0.07 },
        Batch { size: 1000,   label: "1K",     weight: 0.10 },
        Batch { size: 5000,   label: "5K",     weight: 0.10 },
        Batch { size: 25000,  label: "25K",    weight: 0.20 },
        Batch { size: 250000, label: "250K",   weight: 0.50 },
    ];

    println!("\nBenchmark (max {} iters/round, adaptive, mode={}):", max_iters, mode);
    println!("  {:>7} {:>7} {:>10} {:>8}", "Batch", "Blocks", "ns/block", "Weight");
    println!("  {:->7} {:->7} {:->10} {:->8}", "", "", "", "");

    let mut total: f32 = 0.0;
    let mut total_weight: f32 = 0.0;

    for batch in &batches {
        let iters = adaptive_iters(batch.size, max_iters);
        let mut blocks = create_blocks(batch.size);
        let ns = benchmark(&mut blocks, iters, mode);
        println!("  {:>7} {:>7} {:>10.3} {:>7.0}%", batch.label, batch.size, ns, batch.weight * 100.0);
        total += ns as f32 * batch.weight;
        total_weight += batch.weight;
    }

    assert!((total_weight - 1.0).abs() < 0.001, "weights must sum to 1.0");

    fs::write(&score_path, format!("{:.3}", total)).unwrap();
    println!("\nFitness (weighted avg): {:.3} ns/block -> {}", total, score_path);
}
