use sodals_neuro::llm_core::LlmEngine;
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogManager, LogLevel};

fn main() -> anyhow::Result<()> {
    println!("Testing LlmEngine initialization...");
    let config = SystemConfig::default();
    let logger = LogManager::new(LogLevel::INFO);
    
    // This will likely fail if no model exists, but we want to see if it even STARTS
    match LlmEngine::new(&config, &logger) {
        Ok(_) => println!("✅ Success!"),
        Err(e) => println!("❌ Failed (as expected if no weights): {}", e),
    }
    
    Ok(())
}
