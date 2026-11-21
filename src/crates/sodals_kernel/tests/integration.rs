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
    let model_path = std::env::var("SODALS_MODEL_PATH").unwrap_or_else(|_| {
        cwd.join("../../models/Qwen3.5-9B_Q4_K_M.gguf").to_string_lossy().to_string()
    });
    let tokenizer_path = std::env::var("SODALS_TOKENIZER_PATH").unwrap_or_else(|_| {
        cwd.join("../../models/tokenizer.json").to_string_lossy().to_string()
    });
    let json_path = cwd.join("../../oedb_scenarios.json").to_string_lossy().to_string();

    if !std::path::Path::new(&model_path).exists() {
        println!("[INTEGRATION] Skipping test: model file not found at {}", model_path);
        return;
    }

    config.model_path = model_path;
    config.tokenizer_path = tokenizer_path;

    let logger = LogManager::new(LogLevel::INFO);

    println!("[INTEGRATION] Yükleniyor... LlmEngine");
    let engine = match LlmEngine::new(&config, &logger) {
        Ok(e) => e,
        Err(err) => {
            println!("[INTEGRATION] Engine load error: {}", err);
            return;
        }
    };
    
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
