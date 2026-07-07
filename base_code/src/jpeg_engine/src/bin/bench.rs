use jpeg_engine::idct_2d;
use std::env;
use std::fs;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let iter_count: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5000);

    let mut blocks: Vec<[f64; 64]> = (0..200).map(|i| {
        let mut b = [0.0f64; 64];
        for j in 0..64 {
            b[j] = (i as f64 * j as f64 % 256.0) - 128.0;
        }
        b
    }).collect();

    for block in &mut blocks {
        idct_2d(block.as_mut_ptr());
    }

    let rounds = 10;
    let mut samples = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..iter_count {
            for block in &mut blocks {
                idct_2d(block.as_mut_ptr());
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        let ms_per_iter = (elapsed / iter_count as f64) * 1000.0;
        samples.push(ms_per_iter);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];

    let score_path = args.get(2).cloned().unwrap_or_else(|| "fitness.score".to_string());
    fs::write(&score_path, format!("{:.6}", median)).unwrap();
    println!("{:.6} ms/iter -> {}", median, score_path);
}
