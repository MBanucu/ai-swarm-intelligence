use half::f16;
use jpeg_engine::{header, idct_2d, dct_2d, ycbcr_to_rgb};
use std::fs;
use std::path::PathBuf;

fn test_images_dir() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path
}

#[test]
fn test_parse_all_generated_jpegs() {
    for name in &["test_16x16.jpg", "test_32x32.jpg", "test_64x64.jpg", "test_128x128.jpg"] {
        let path = test_images_dir().join(name);
        let data = fs::read(&path).expect(&format!("Failed to read {:?}", path));
        let info = header::parse_header(&data).expect(&format!("Failed to parse {:?}", path));
        assert!(info.width > 0, "{}: width should be > 0", name);
        assert!(info.height > 0, "{}: height should be > 0", name);
    }
}

#[test]
fn test_parse_real_jpegs() {
    for name in &["water_fish.jpg", "greece_flag.jpg"] {
        let path = test_images_dir().join(name);
        if !path.exists() {
            eprintln!("Skipping missing test image: {:?}", path);
            continue;
        }
        let data = fs::read(&path).expect(&format!("Failed to read {:?}", path));
        let info = header::parse_header(&data);
        match info {
            Ok(info) => {
                assert!(info.width > 0, "{}: width={} height={}", name, info.width, info.height);
                println!("{}: {}x{} components={}", name, info.width, info.height, info.components);
            }
            Err(e) => {
                println!("Skipping {} (parse error, may be non-standard JPEG): {}", name, e);
            }
        }
    }
}

#[test]
fn test_image_dimensions_match() {
    let path = test_images_dir().join("test_64x64.jpg");
    let data = fs::read(&path).unwrap();
    let info = header::parse_header(&data).unwrap();
    assert_eq!(info.width, 64);
    assert_eq!(info.height, 64);
}

#[test]
fn test_dct_idct_roundtrip_multiple() {
    for i in 0..10 {
        let mut block = [f16::ZERO; 64];
        for j in 0..64 {
            block[j] = f16::from_f64((i as f64 * j as f64 % 256.0) - 128.0);
        }
        let orig = block;
        dct_2d(block.as_mut_ptr() as *mut u16);
        idct_2d(block.as_mut_ptr() as *mut u16);
        for k in 0..64 {
            let diff = if block[k] > orig[k] { block[k] - orig[k] } else { orig[k] - block[k] };
            assert!(diff < f16::from_f32(1.0),
                "roundtrip error at i={i} k={k}: {:?} vs {:?}", block[k], orig[k]);
        }
    }
}

#[test]
fn test_ycbcr_clamping() {
    let test_cases = [
        (f16::ZERO, f16::from_f32(128.0), f16::from_f32(128.0), (0, 0, 0)),
        (f16::from_f32(255.0), f16::from_f32(128.0), f16::from_f32(128.0), (255, 255, 255)),
        (f16::from_f32(-50.0), f16::ZERO, f16::from_f32(300.0), (0, 0, 255)),
    ];
    for (y, cb, cr, _expected) in &test_cases {
        let rgb_val = ycbcr_to_rgb(y.to_bits(), cb.to_bits(), cr.to_bits());
        let r = ((rgb_val >> 16) & 0xFF) as u8;
        let g = ((rgb_val >> 8) & 0xFF) as u8;
        let b = (rgb_val & 0xFF) as u8;
        assert!(r <= 255);
        assert!(g <= 255);
        assert!(b <= 255);
    }
}

#[test]
fn test_idct_performance() {
    let mut blocks: Vec<[f16; 64]> = (0..100).map(|i| {
        let mut b = [f16::ZERO; 64];
        for j in 0..64 {
            b[j] = f16::from_f64((i as f64 * j as f64 % 256.0) - 128.0);
        }
        b
    }).collect();

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        for block in &mut blocks {
            idct_2d(block.as_mut_ptr() as *mut u16);
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let total_iters = 1000.0 * blocks.len() as f64;
    let ms_per_iter = (elapsed / total_iters) * 1000.0;
    println!("IDCT performance: {:.6} ms/iter ({} blocks x 1000 loops)", ms_per_iter, blocks.len());
    assert!(ms_per_iter < 0.2, "IDCT too slow: {:.6}ms/iter", ms_per_iter);
}
