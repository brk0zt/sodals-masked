use std::path::PathBuf;
use std::sync::Arc;
use clap::Parser;
use tokio::signal;
use tracing::{info, warn, error};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

// DOĞRU BAĞIMLILIKLAR (Derleyici loglarındaki gerçek crate yolları)
use sodals_neuro::engine::llm_engine::LlmEngine;
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogManager, LogLevel};
use sodals_kernel::oedb_runner::EodbRunner;

#[derive(Parser, Debug)]
#[command(author = "Burak Öztürk", version = "2.0", about = "SODALS Neuro-Symbolic Engine")]
struct Args {
    #[arg(short, long, default_value = "oedb_scenarios.json")]
    scenarios: PathBuf,

    #[arg(short, long, default_value = "info")]
    log_level: String,

    #[arg(long, default_value_t = true)]
    with_engine: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Gelişmiş Loglama Sistemini Kur (Gereksiz token spam'ini susturur)
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info,sodals_neuro=info"))?;

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false))
        .with(filter)
        .init();

    let args = Args::parse();
    info!("SODALS Neuro-Engine başlatılıyor...");

    // 2. Motor Konfigürasyonlarını Hazırla
    let sys_config = SystemConfig::load().unwrap_or_else(|_| SystemConfig::default());
    let log_manager = LogManager::new(LogLevel::INFO); // ÇÖZÜM: LogLevel eklendi
    
    // 3. LLM Engine'i (Qwen Core) Yükle
    info!("Model ağırlıkları VRAM'e taşınıyor (Asymmetric DMA)...");
    
    let engine_instance = LlmEngine::new(&sys_config, &log_manager)?;
    
    // 4. OEDB Test Koşucusunu Hazırla
    // ÇÖZÜM: EodbRunner'ın gerçek API'sine göre başlatılması
    let json_path = args.scenarios.to_str().unwrap_or("oedb_scenarios.json");
    let mut runner = EodbRunner::from_file(json_path)?;
    runner.attach_engine(engine_instance);

    info!("╔═══════════════════════════════════════════════════════════════════╗");
    info!("║   SODALS v2.0 — Symplectic ODE-Driven Asymmetric Latent System    ║");
    info!("╚═══════════════════════════════════════════════════════════════════╝");

    // 5. Arka Planda Testleri Başlat
    let runner_handle = tokio::spawn(async move {
        if let Err(e) = runner.run_all().await {
            error!("Benchmark sırasında kritik hata: {:?}", e);
        }
    });

    // Sinyal bekletme ve zarif çıkış (Graceful Shutdown)
    tokio::select! {
        _ = runner_handle => {
            info!("Bütün test senaryoları tamamlandı.");
        }
        _ = signal::ctrl_c() => {
            warn!("\n[SİNYAL] Kullanıcı tarafından durduruldu. Çıkış yapılıyor...");
        }
    }

    info!("SODALS motoru güvenli bir şekilde kapatıldı.");
    Ok(())
}