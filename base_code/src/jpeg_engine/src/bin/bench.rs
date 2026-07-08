use half::f16;
use jpeg_engine::idct_2d_batch;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::env;
use std::fs;
use std::time::Instant;

fn c(k: usize) -> f64 {
    if k == 0 { 1.0 / (2.0f64).sqrt() } else { 1.0 }
}

/// Reference IDCT using f64 precision, returns f16.
fn reference_idct_2d(block: &[f16; 64]) -> [f16; 64] {
    let mut out = [f16::ZERO; 64];
    for y in 0..8 {
        for x in 0..8 {
            let mut sum = 0.0f64;
            for v in 0..8 {
                for u in 0..8 {
                    sum += c(u) * c(v)
                        * block[v * 8 + u].to_f64()
                        * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos()
                        * ((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI / 16.0).cos();
                }
            }
            out[y * 8 + x] = f16::from_f64(0.25 * sum);
        }
    }
    out
}

fn validate() {
    let mut rng = StdRng::seed_from_u64(0xbad_f00d_cafe_1337);
    let epsilon = f16::from_f32(50.0);

    let mut test_cases: Vec<[f16; 64]> = Vec::new();

    let mut dc_block = [f16::ZERO; 64];
    dc_block[0] = f16::from_f32(128.0);
    test_cases.push(dc_block);

    for k in 1..5 {
        let mut ac_block = [f16::ZERO; 64];
        ac_block[k] = f16::from_f32(64.0);
        test_cases.push(ac_block);
    }

    for _ in 0..20 {
        let mut b = [f16::ZERO; 64];
        b[0] = f16::from_f64((rng.gen::<f64>() - 0.5) * 2048.0);
        for j in 1..64 {
            let scale = 256.0 / (j as f64).sqrt();
            b[j] = f16::from_f64((rng.gen::<f64>() - 0.5) * 2.0 * scale);
        }
        test_cases.push(b);
    }

    println!("Validation: {} test cases against reference IDCT...", test_cases.len());

    for (i, block) in test_cases.iter().enumerate() {
        let expected = reference_idct_2d(block);
        let mut engine_block = *block;
        idct_2d_batch(engine_block.as_mut_ptr() as *mut u16, 1);

        for j in 0..64 {
            let diff = if engine_block[j] > expected[j] {
                engine_block[j] - expected[j]
            } else {
                expected[j] - engine_block[j]
            };
            if diff > epsilon {
                eprintln!(
                    "VALIDATION FAILED: test case {} coefficient {}: engine={:.3?} reference={:.3?} diff={:.3?}",
                    i, j, engine_block[j], expected[j], diff
                );
                std::process::exit(1);
            }
        }
    }
    println!("Validation: all {} test cases passed.", test_cases.len());

    let batch_sizes: &[(usize, usize)] = &[
        (10, 0), (1000, 0), (25000, 0), (250000, 20),
    ];

    for &(batch_size, check_n) in batch_sizes {
        let check_count = if check_n == 0 { batch_size } else { check_n };

        let mut blocks: Vec<[f16; 64]> = (0..batch_size).map(|i| {
            let mut b = [f16::ZERO; 64];
            b[0] = f16::from_f64((i as f64 - 0.5) * 64.0);
            for j in 1..64 {
                b[j] = f16::from_f64(((i * j) as f64 % 128.0) - 64.0);
            }
            b
        }).collect();

        let expected: Vec<[f16; 64]> = blocks[..check_count]
            .iter().map(|b| reference_idct_2d(b)).collect();

        let ptr = blocks.as_mut_ptr() as *mut u16;
        idct_2d_batch(ptr, blocks.len() as u32);

        for (k, (engine_block, ref_block)) in blocks[..check_count]
            .iter().zip(expected.iter()).enumerate()
        {
            for j in 0..64 {
                let diff = if engine_block[j] > ref_block[j] {
                    engine_block[j] - ref_block[j]
                } else {
                    ref_block[j] - engine_block[j]
                };
                if diff > epsilon {
                    eprintln!(
                        "BATCH VALIDATION FAILED: batch_size={} block={} coeff={}: engine={:.3?} reference={:.3?} diff={:.3?}",
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

fn create_blocks(count: usize) -> Vec<[f16; 64]> {
    let mut rng = StdRng::seed_from_u64(0xdead_beef_cafe_babe);
    (0..count).map(|_| {
        let mut b = [f16::ZERO; 64];
        let dc = f16::from_f64((rng.gen::<f64>() - 0.5) * 2048.0);
        let dc_q = f16::from_f32(8.0);
        b[0] = f16::from_f64((dc.to_f64() / dc_q.to_f64()).round() * dc_q.to_f64());
        for j in 1..64 {
            let scale = 256.0 / (j as f64).sqrt();
            let val = f16::from_f64((rng.gen::<f64>() - 0.5) * 2.0 * scale);
            let q = f16::from_f64((j as f64 / 8.0 + 1.0) * 4.0);
            b[j] = f16::from_f64((val.to_f64() / q.to_f64()).round() * q.to_f64());
        }
        b
    }).collect()
}

fn adaptive_iters(batch_size: usize, max_iters: usize) -> usize {
    (5_000_000 / batch_size).max(1).min(max_iters)
}

fn benchmark(blocks: &mut Vec<[f16; 64]>, iter_count: usize, label: &str) -> f64 {
    idct_2d_batch(blocks.as_mut_ptr() as *mut u16, blocks.len() as u32);

    let rounds = 10;
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..iter_count {
            idct_2d_batch(blocks.as_mut_ptr() as *mut u16, blocks.len() as u32);
        }
        let elapsed = start.elapsed().as_secs_f64();
        samples.push((elapsed / iter_count as f64) * 1000.0);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    println!("  {:<10} {:>7} blocks  {:>9.6} ms/iter", label, blocks.len(), median);
    median
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let max_iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5000);

    validate();

    struct Batch { size: usize, label: &'static str, weight: f16 }
    let batches = [
        Batch { size: 10,     label: "10",     weight: f16::from_f32(0.03) },
        Batch { size: 250,    label: "250",    weight: f16::from_f32(0.07) },
        Batch { size: 1000,   label: "1K",     weight: f16::from_f32(0.10) },
        Batch { size: 5000,   label: "5K",     weight: f16::from_f32(0.10) },
        Batch { size: 25000,  label: "25K",    weight: f16::from_f32(0.20) },
        Batch { size: 250000, label: "250K",   weight: f16::from_f32(0.50) },
    ];

    println!("\nBenchmark (max {} iters/round, adaptive):", max_iters);
    println!("  {:>10} {:>7} {:>11} {:>7}", "Batch", "Blocks", "ms/iter", "Weight");
    println!("  ---------- ------- ----------- -------");

    let mut total = f16::ZERO;
    let mut total_weight = f16::ZERO;

    for batch in &batches {
        let iters = adaptive_iters(batch.size, max_iters);
        let mut blocks = create_blocks(batch.size);
        let ms = benchmark(&mut blocks, iters, batch.label);
        println!("  {:>10} {:>7} {:>9.6} {:>6.0}%", batch.label, batch.size, ms, batch.weight.to_f64() * 100.0);
        total += f16::from_f64(ms) * batch.weight;
        total_weight += batch.weight;
    }

    let score_path = args.get(2).cloned().unwrap_or_else(|| "fitness.score".to_string());
    fs::write(&score_path, format!("{:.6}", total.to_f64())).unwrap();
    println!("\nFitness (weighted avg): {:.6} ms/iter -> {}", total.to_f64(), score_path);
}
