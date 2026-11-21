// src/crates/sodals_neuro/src/bin/build_qmem_cartridges.rs
// ============================================================================
// SODALS QMEM Cartridge Builder Binary
//
// Kullanım:
//   cargo run --bin build_qmem_cartridges -- [output_dir]
//
// Varsayılan output_dir: "." (çalışma dizini)
//
// Çıktı dosyaları:
//   primordial_cognition.qmem   — Töz bilgi (evrimsel substrat)
//   industrial_topology.qmem    — AssetOps / ITBench domain
//   sodals_core_merged.qmem     — Her ikisinin birleşimi (önerilen)
// ============================================================================

use std::env;
use sodals_neuro::qmem_builder::{build_all_cartridges, build_merged_cartridge};

fn main() {
    let args: Vec<String> = env::args().collect();
    let output_dir = args.get(1).map(|s| s.as_str()).unwrap_or(".");

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         SODALS QMEM Cartridge Builder                    ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Töz Bilgi    → primordial_cognition.qmem                ║");
    println!("║  Domain       → industrial_topology.qmem                 ║");
    println!("║  Math (D1)    → mathematics_analysis_topology.qmem       ║");
    println!("║  Algebra (D2) → abstract_algebra.qmem                    ║");
    println!("║  ArXiv        → arxiv_physics.qmem                       ║");
    println!("║  Birleşik     → sodals_core_merged.qmem                  ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("Çıktı dizini: {}", output_dir);
    println!();

    // Ayrı kartridjler
    match build_all_cartridges(output_dir) {
        Ok(()) => println!("[OK] Ayrı kartridjler oluşturuldu"),
        Err(e) => {
            eprintln!("[HATA] Ayrı kartridj oluşturulamadı: {}", e);
            std::process::exit(1);
        }
    }

    // Birleşik kartridj (önerilen: LlmEngine tek net_g ile her iki bilgiyi taşır)
    match build_merged_cartridge(output_dir, "sodals_core_merged.qmem") {
        Ok(()) => println!("[OK] Birleşik kartridj oluşturuldu"),
        Err(e) => {
            eprintln!("[HATA] Birleşik kartridj oluşturulamadı: {}", e);
            std::process::exit(1);
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("Sonraki adım: LlmEngine'e yükle");
    println!("  engine.load_qmem_cartridge(\"sodals_core_merged.qmem\")");
    println!("═══════════════════════════════════════════════════════════");
}
