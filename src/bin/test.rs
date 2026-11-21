use sodals_neuro::engine::llm_engine::LlmEngine;
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogManager, LogLevel};
use sodals_neuro::engine::generation_loop::generate_stream;
use sodals_neuro::llm_core::NeuroConfig;

fn main() -> anyhow::Result<()> {
    std::env::set_var("SODALS_VERBOSE", "1");
    tracing_subscriber::fmt::init();

    let sys_config = SystemConfig::load().unwrap_or_else(|_| SystemConfig::default());
    let log_manager = LogManager::new(LogLevel::INFO);
    let mut engine = LlmEngine::new(&sys_config, &log_manager)?;

    let mut config = NeuroConfig::default();
    config.temperature = 0.7;
    config.strict_determinism = false;

    println!("=== ENGLISH PROMPT ===");
    let prompt_en = "<|im_start|>user\nWhat are the common features of Sydney and Istanbul?<|im_end|>\n<|im_start|>assistant\n";
    let output_en = generate_stream(&mut engine, prompt_en, None, None, |t| { print!("{}", t); }, &config, 42)?;
    println!("\n\n=== ENGLISH OUTPUT COMPLETE ===\n\n");

    engine.reset_internal_state();

    println!("=== TURKISH PROMPT ===");
    let prompt_tr = "<|im_start|>user\nSydney ile İstanbul'un ortak özellikleri nelerdir?<|im_end|>\n<|im_start|>assistant\n";
    let output_tr = generate_stream(&mut engine, prompt_tr, None, None, |t| { print!("{}", t); }, &config, 42)?;
    println!("\n\n=== TURKISH OUTPUT COMPLETE ===\n\n");

    Ok(())
}
