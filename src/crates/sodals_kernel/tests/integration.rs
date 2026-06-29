use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogManager, LogLevel};
use sodals_kernel::oedb_runner::EodbRunner;
use sodals_neuro::llm_core::LlmEngine;

#[tokio::test]
async fn test_oedb_runner_with_engine() {
    // We'll load default system configuration and adjust paths for tests
    let mut config = SystemConfig::default();
    
    let cwd = std::env::current_dir().unwrap();
    println!("CWD: {:?}", cwd);
    config.model_path = cwd.join("../../../models/model.gguf").to_string_lossy().to_string();
    config.tokenizer_path = cwd.join("../../../models/tokenizer.json").to_string_lossy().to_string();
    let json_path = cwd.join("../../../oedb_scenarios.json").to_string_lossy().to_string();
    
    let logger = LogManager::new(LogLevel::INFO);
    
    println!("[INTEGRATION] Yükleniyor... LlmEngine");
    let engine = LlmEngine::new(&config, &logger).expect("LlmEngine yüklenemedi");
    
    println!("[INTEGRATION] Yükleniyor... EodbRunner");
    let mut runner = EodbRunner::from_file(&json_path).expect("EodbRunner could not load JSON");
    
    println!("[INTEGRATION] Motor Teste Bağlanıyor...");
    runner.attach_engine(engine);
    
    println!("[INTEGRATION] OEDB Başlatılıyor...");
    let report = runner.run_all().await.expect("OEDB failed to run");
    runner.print_report_card(&report);
    
    // Ensure that there are no generic or completely failed tests
    assert!(report.total_scenarios > 0, "No scenarios loaded");
}
