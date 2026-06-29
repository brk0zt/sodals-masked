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
    neuro_config.strict_determinism = true;
    neuro_config.euler_resonance = true;

    println!("[SİSTEM] VRAM ayrılıyor. L0-L3 katmanları ve Triton Kernelleri yükleniyor...");
    let mut engine = LlmEngine::new(&config, &logger).unwrap();

    // OEDB Halüsinasyon ve Paradoks Testi (HDC Filtresini tetiklemek için özel tasarlandı)
    // Standart LLM'ler burada "Nehre yürü, anahtarı al, dön ve arabayı çalıştır" der.
    let prompt = "Bir araba kilitli bir garajdadır. Arabanın anahtarı nehrin karşı kıyısındadır. Nehir 5 dakika yürüme mesafesindedir. Arabayı nasıl çalıştırırsın? Mantıksal olarak açıkla.";

    println!("\n[OEDB HEDEFİ] Soru: {}", prompt);
    println!("------------------------------------------------------------");

    let start = Instant::now();
    
    // Agentic Pipeline ve Ouroboros Döngüsünü Tetikle
    match engine.generate_with_pipeline(prompt, &neuro_config) {
        Ok(attempt) => {
            let duration = start.elapsed();
            println!("\n============================================================");
            println!("[BAŞARI] İnfaz {} milisaniyede tamamlandı.", duration.as_millis());
            println!("[SONUÇ] Nörosembolik Çıktı:\n{}", attempt.generated_code);
            println!("============================================================");
        }
        Err(e) => eprintln!("[KRİTİK HATA] Pipeline çöktü: {}", e),
    }

    Ok(())
}