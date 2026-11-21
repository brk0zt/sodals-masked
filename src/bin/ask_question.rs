use anyhow::Result;
use sodals_neuro::engine::generation_loop::generate_stream;
use sodals_neuro::engine::llm_engine::LlmEngine;
use sodals_neuro::llm_core::NeuroConfig;
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogLevel, LogManager};

fn main() -> Result<()> {
    std::env::set_var("SODALS_PURE_INFERENCE", "1");
    tracing_subscriber::fmt::init();

    let sys_config = SystemConfig::load().unwrap_or_else(|_| SystemConfig::default());
    let log_manager = LogManager::new(LogLevel::INFO);
    let mut engine = LlmEngine::new(&sys_config, &log_manager)?;

    let mut config = NeuroConfig::engineer_mode();
    config.euler_resonance = false;
    config.physics_enabled = false;
    config.top_p = 0.9;
    config.min_p = 0.02;
    config.repeat_penalty = 1.15;
    config.temperature = 0.75;
    config.top_k = 40;

    let prompt = "<|im_start|>user\nİstanbul ile Sydney'in ortak özellikleri nelerdir?<|im_end|>\n<|im_start|>assistant\n";
    println!("Prompt: {}", prompt);
    let output = generate_stream(&mut engine, prompt, None, None, |t| { print!("{}", t); }, &config, 128)?;
    println!("\n\n=== OUTPUT COMPLETE ===");
    println!("Final output: {:?}", output);

    Ok(())
}
