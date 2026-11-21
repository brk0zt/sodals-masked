use anyhow::Result;
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogManager, LogLevel};
use sodals_neuro::engine::llm_engine::LlmEngine;

fn main() -> Result<()> {
    let sys_config = SystemConfig::load().unwrap_or_else(|_| SystemConfig::default());
    let log_manager = LogManager::new(LogLevel::INFO);
    let _engine = LlmEngine::new(&sys_config, &log_manager)?;
    println!("LlmEngine check completed.");
    Ok(())
}

