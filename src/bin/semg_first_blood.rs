// src/bin/semg_first_blood.rs
// ============================================================================
// sEMG FIRST BLOOD — Subvocal Speech Pipeline Validation
// ============================================================================

use anyhow::{Result, Context};
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogManager, LogLevel};
use sodals_neuro::llm_core::LlmEngine;
use sodals_neuro::engine::semg_phoneme_hunter::SemgStreamer;
use candle_core::{Device, Tensor};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    println!("[sEMG FIRST BLOOD] Subvocal Speech Pipeline ayağa kalkıyor...");
    
    // 1. Motoru Başlat
    let config = SystemConfig::default();
    let logger = LogManager::new(LogLevel::INFO);
    let engine = LlmEngine::new(&config, &logger).unwrap();
    let hidden_size = engine.hidden_size();
    
    // 2. sEMG Streamer'ı Yükle (Dummy veya gerçek veri)
    println!("[sEMG FIRST BLOOD] real_subvocal_semg.safetensors mmap ediliyor...");
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    
    // sEMG verisi genelde 256Hz veya 1000Hz olur. Nieto veri seti 256Hz.
    let sample_rate = 256.0;
    let mut streamer = SemgStreamer::new(
        Path::new("real_subvocal_semg.safetensors"), 
        &device, 
        sample_rate
    ).context("sEMG Streamer başlatılamadı. Dosya mevcut mu?")?;
    
    // 3. DSP Pipeline üzerinden "Phonetic Ghost Prefix" çıkarımı
    println!("[sEMG FIRST BLOOD] DSP Pipeline (HPF -> Rectify -> RMS) çalıştırılıyor...");
    
    for i in 0..streamer.num_samples {
        // Fetch raw 2048-D phonetic features (no zero-padding)
        let (ghost_prefix, label) = streamer.fetch_next_phonetic_prefix()?;
        
        println!(
            "[sEMG FIRST BLOOD] Epoch {}: Etiket: {}, Raw Features Shape: {:?}", 
            i, label, ghost_prefix.shape()
        );
        
        // İlk 5 epoch yeterli
        if i >= 4 { break; }
    }
    
    println!("[sEMG FIRST BLOOD] Başarı! sEMG kanalları 2048-D raw özellik vektörüne çıkarıldı.");
    println!("[sEMG FIRST BLOOD] NOT: Bu ham vektörler SemgAdapter MLP ile {}-D LLM uzayına projekte edilmeli.", hidden_size);
    println!("[sEMG FIRST BLOOD] Artık Qwen'in dikkat mekanizmasına bu latent'ları basabiliriz.");
    
    Ok(())
}
