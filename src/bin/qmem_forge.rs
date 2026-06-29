// src/bin/qmem_forge.rs
// ============================================================================
// QMEM FORGE — Dataset Crystallization CLI
// ============================================================================
//
// Usage:
//   cargo run --bin qmem_forge -- <safetensors_dir> <output_dir>
//
// Example:
//   cargo run --bin qmem_forge -- . ./qmem_crystals

use std::path::Path;
use std::time::Instant;
use sodals_neuro::engine::qmem_compiler::QmemFactory;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let (input_dir, output_dir) = if args.len() >= 3 {
        (args[1].as_str(), args[2].as_str())
    } else {
        // Default: current directory -> ./qmem_crystals
        (".", "./qmem_crystals")
    };

    eprintln!("┌─────────────────────────────────────────────────────────────┐");
    eprintln!("│            QMEM FORGE — Crystal Compiler v1.0              │");
    eprintln!("├─────────────────────────────────────────────────────────────┤");
    eprintln!("│ Input  : {:50}│", input_dir);
    eprintln!("│ Output : {:50}│", output_dir);
    eprintln!("└─────────────────────────────────────────────────────────────┘");

    let start = Instant::now();

    match QmemFactory::compile_dataset(Path::new(input_dir), Path::new(output_dir)) {
        Ok(total) => {
            let elapsed = start.elapsed();
            let throughput = if elapsed.as_secs_f64() > 0.0 {
                total as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };

            eprintln!("┌─────────────────────────────────────────────────────────────┐");
            eprintln!("│                  CRYSTALLIZATION COMPLETE                   │");
            eprintln!("├─────────────────────────────────────────────────────────────┤");
            eprintln!("│ Crystals  : {:>10}                                     │", total);
            eprintln!("│ Time      : {:>10.2}s                                    │", elapsed.as_secs_f64());
            eprintln!("│ Throughput: {:>10.1} crystals/sec                        │", throughput);
            eprintln!("└─────────────────────────────────────────────────────────────┘");
        }
        Err(e) => {
            eprintln!("[QMEM FORGE] FATAL: {}", e);
            std::process::exit(1);
        }
    }
}
