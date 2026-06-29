use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogManager, LogLevel};
use sodals_neuro::llm_core::LlmEngine;

#[tokio::main]
async fn main() {
    let mut config = SystemConfig::default();
    config.model_path = r"C:\Users\Burak\Desktop\sodals_core\models\Qwen3.5-9B-Q4_K_M.gguf".to_string();
    config.tokenizer_path = r"C:\Users\Burak\Desktop\sodals_core\models\tokenizer.json".to_string();
    
    let logger = LogManager::new(LogLevel::INFO);
    println!("Loading engine...");
    let mut engine = LlmEngine::new(&config, &logger).unwrap();
    
    let neuro_config = sodals_neuro::llm_core::NeuroConfig::engineer_mode();
    println!("Generating...");
    match sodals_neuro::engine::generation_loop::generate_stream(&mut engine, &"1+1=".to_string(), None, None, |t| { print!("{}", t); std::io::Write::flush(&mut std::io::stdout()).unwrap(); }, &neuro_config, 0) {
        Ok(res) => println!("\nSUCCESS: {}", res),
        Err(e) => println!("\nERROR: {:?}", e),
    }
}
