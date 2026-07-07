use jpeg_engine::idct_2d_batch;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::env;
use std::fs;
use std::f64::consts::PI;
use std::time::Instant;

fn c(k: usize) -> f64 {
    if k == 0 { 1.0 / (2.0f64).sqrt() } else { 1.0 }
}

/// Naive O(N^4) reference IDCT — mathematically correct implementation
/// of the 2D IDCT formula. Used to validate the engine's output.
fn reference_idct_2d(block: &[f64; 64]) -> [f64; 64] {
    let mut out = [0.0f64; 64];
    for y in 0..8 {
        for x in 0..8 {
            let mut sum = 0.0;
            for v in 0..8 {
                for u in 0..8 {
                    sum += c(u) * c(v)
                        * block[v * 8 + u] as f64
                        * ((2 * x + 1) as f64 * u as f64 * PI / 16.0).cos()
                        * ((2 * y + 1) as f64 * v as f64 * PI / 16.0).cos();
                }
            }
            out[y * 8 + x] = 0.25 * sum;
        }
    }
    out
}

fn validate() {
    let mut rng = StdRng::seed_from_u64(0xbad_f00d_cafe_1337);
    let epsilon = 0.5; // generous tolerance for f64 rounding differences

    // Test pattern: single non-zero coefficient
    let mut test_cases: Vec<[f64; 64]> = Vec::new();

    // DC-only block
    let mut dc_block = [0.0f64; 64];
    dc_block[0] = 128.0;
    test_cases.push(dc_block);

    // Single AC coefficient at various positions
    for k in 1..5 {
        let mut ac_block = [0.0f64; 64];
        ac_block[k] = 64.0;
        test_cases.push(ac_block);
    }

    // Random blocks with realistic magnitude
    for _ in 0..20 {
        let mut b = [0.0f64; 64];
        b[0] = (rng.gen::<f64>() - 0.5) * 2048.0;
        for j in 1..64 {
            let scale = 256.0 / (j as f64).sqrt();
            b[j] = (rng.gen::<f64>() - 0.5) * 2.0 * scale;
        }
        test_cases.push(b);
    }

    println!("Validation: {} test cases against reference IDCT...", test_cases.len());

    for (i, block) in test_cases.iter().enumerate() {
        let expected = reference_idct_2d(block);
        let mut engine_block = *block;
        idct_2d_batch(engine_block.as_mut_ptr() as *mut f64, 1);

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

    // Validate batch processing at all benchmark sizes
    for &batch_size in &[10, 200, 10000] {
        let mut blocks: Vec<[f64; 64]> = (0..batch_size).map(|i| {
            let mut b = [0.0f64; 64];
            b[0] = (i as f64 - 0.5) * 64.0;
            for j in 1..64 {
                b[j] = ((i * j) as f64 % 128.0) - 64.0;
            }
            b
        }).collect();

        let expected: Vec<[f64; 64]> = blocks.iter().map(|b| reference_idct_2d(b)).collect();

        let ptr = blocks.as_mut_ptr() as *mut f64;
        idct_2d_batch(ptr, blocks.len() as u32);

        for (k, (engine_block, ref_block)) in blocks.iter().zip(expected.iter()).enumerate() {
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
        println!("Validation: batch_size={} — all {} blocks passed.", batch_size, blocks.len());
    }
}

fn create_blocks(count: usize) -> Vec<[f64; 64]> {
    let mut rng = StdRng::seed_from_u64(0xdead_beef_cafe_babe);
    (0..count).map(|_| {
        let mut b = [0.0f64; 64];
        let dc = (rng.gen::<f64>() - 0.5) * 2048.0;
        let dc_q = 8.0;
        b[0] = (dc / dc_q).round() * dc_q;
        for j in 1..64 {
            let scale = 256.0 / (j as f64).sqrt();
            let val = (rng.gen::<f64>() - 0.5) * 2.0 * scale;
            let q = (j as f64 / 8.0 + 1.0) * 4.0;
            b[j] = (val / q).round() * q;
        }
        b
    }).collect()
}

fn benchmark(blocks: &mut Vec<[f64; 64]>, iter_count: usize, label: &str) -> f64 {
    idct_2d_batch(blocks.as_mut_ptr() as *mut f64, blocks.len() as u32);

    let rounds = 10;
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..iter_count {
            idct_2d_batch(blocks.as_mut_ptr() as *mut f64, blocks.len() as u32);
        }
        let elapsed = start.elapsed().as_secs_f64();
        samples.push((elapsed / iter_count as f64) * 1000.0);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    println!("  {:<10} {:>6} blocks  {:.6} ms/iter", label, blocks.len(), median);
    median
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let iter_count: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5000);

    validate();

    let mut low = create_blocks(10);
    let mut medium = create_blocks(200);
    let mut high = create_blocks(10000);

    println!("\nBenchmark — {} iterations per round:", iter_count);
    let low_ms = benchmark(&mut low, iter_count, "Low");
    let med_ms = benchmark(&mut medium, iter_count, "Medium");
    let high_ms = benchmark(&mut high, iter_count, "High");

    let fitness = high_ms * 0.5 + med_ms * 0.3 + low_ms * 0.2;

    let score_path = args.get(2).cloned().unwrap_or_else(|| "fitness.score".to_string());
    fs::write(&score_path, format!("{:.6}", fitness)).unwrap();
    println!(
        "\nFitness (0.5*high + 0.3*mid + 0.2*low): {:.6} ms/iter -> {}",
        fitness, score_path
    );
}
