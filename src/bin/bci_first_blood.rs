use anyhow::Result;
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogManager, LogLevel};
use sodals_neuro::llm_core::LlmEngine;
use sodals_neuro::engine::bci_mux::BciStreamer;
use candle_core::Device;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    println!("[BCI FIRST BLOOD] Sistem ayağa kalkıyor...");
    
    // 1. Motoru Çıplak Başlat (OEDB Yok)
    let config = SystemConfig::default();
    let logger = LogManager::new(LogLevel::INFO);
    let mut engine = LlmEngine::new(&config, &logger).unwrap();
    let hidden_size = engine.hidden_size(); // Örn: 4096
    
    // 2. Safetensors Dosyasını Mux'a Yükle
    println!("[BCI FIRST BLOOD] sub-01 safetensors dosyası mmap ediliyor...");
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let mut mux = BciStreamer::new(Path::new("inner_speech_sub-01.safetensors"), &device).unwrap();
    
    // 3. İşkence (Crucible) Modunda 4-Kanallı İlk Sinyali Çek
    // Gürültü basılmış ve Newton-Raphson ile hizalanmış Ghost Prefix
    for i in 0..5 {
        let (ghost_prefix, label) = mux.fetch_next_ghost_prefix(hidden_size).unwrap();
        eprintln!("[BCI FIRST BLOOD] Epoch {}: Deneğin iç ses (Label: {}) Ghost Prefix'i hazır! Boyut: {:?}", i, label, ghost_prefix.shape());
    }
    
    // 4. Qwen'in Beynine Enjekte Et (Topolojik Kanca)
    // Not: Bu kısımda engine.model.forward() çağrını, ghost_prefix'i verecek şekilde simüle ediyoruz.
    println!("[BCI FIRST BLOOD] Qwen'in dikkat mekanizmasına BCI voltajı basılıyor...");
    
    let prompt = "Sana az önce Ghost Prefix olarak bir insanın iç ses voltajını gönderdim. Bu frekansın sende yarattığı anlamsal yansıma nedir? Gürültüye rağmen hissettiğin kavramı yaz.";
    
    // Motoru çalıştır (Eğer generate_stream fonksiyonun ghost_prefix alacak şekilde güncellendiyse onu kullan)
    // Şimdilik motorun çökmediğini ve tensor boyutlarının oturduğunu test ediyoruz.
    println!("[BCI FIRST BLOOD] Hedef: Qwen'in çökmemesi ve NR algoritmasının (Loss) 3-4 adımda hội tụ (yakınsama) yapması.");
    
    Ok(())
}