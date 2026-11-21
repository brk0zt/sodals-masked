// src/bin/run_oedb.rs
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogManager, LogLevel};
use sodals_neuro::llm_core::{LlmEngine, NeuroConfig}; // ARCHITECTURAL SHIM DEVREDE
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("============================================================");
    println!(" ☢️ SODALS NÖROSEMBOLİK MOTOR - OEDB BENCHMARK İNFAZI");
    println!("============================================================");

    let config = SystemConfig::default(); 
    let logger = LogManager::new(LogLevel::INFO);
    let mut neuro_config = NeuroConfig::default();
    // neuro_config.strict_determinism = true;
    neuro_config.euler_resonance = true;

    println!("[SİSTEM] VRAM ayrılıyor. L0-L3 katmanları ve Triton Kernelleri yükleniyor...");
    let mut engine = LlmEngine::new(&config, &logger).unwrap();

    let prompts = vec![
        "İstanbul ile Sydney'in ortak özellikleri nelerdir?",
        "What are the common features of Istanbul and Sydney?",
        "Araba yıkama yerine 50 metre mesafedeyim. Arabayla mı gitmeliyim yoksa yürüyerek mi?",
        "I am 50 meters away from the car wash. Should I go by car or walk?",
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        println!("\n[OEDB HEDEFİ {}] Soru: {}", i + 1, prompt);
        println!("------------------------------------------------------------");

        let start = Instant::now();
        
        match engine.generate_with_pipeline(prompt, &neuro_config) {
            Ok(attempt) => {
                let duration = start.elapsed();
                println!("\n============================================================");
                println!("[BAŞARI {}] İnfaz {} milisaniyede tamamlandı.", i + 1, duration.as_millis());
                println!("[SONUÇ {}]:\n{}", i + 1, attempt.generated_code);
                println!("============================================================");
            }
            Err(e) => eprintln!("[KRİTİK HATA {}] Pipeline çöktü: {}", i + 1, e),
        }
    }

    Ok(())
}