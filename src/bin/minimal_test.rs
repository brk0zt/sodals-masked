use sodals_neuro::engine::llm_engine::LlmEngine;
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogManager, LogLevel};
use sodals_neuro::engine::generation_loop::generate_stream;
use sodals_neuro::llm_core::NeuroConfig;

fn main() -> anyhow::Result<()> {
    std::env::set_var("SODALS_VERBOSE", "1");
    std::env::set_var("SODALS_PURE_INFERENCE", "1");
    tracing_subscriber::fmt::init();

    let sys_config = SystemConfig::load().unwrap_or_else(|_| SystemConfig::default());
    let log_manager = LogManager::new(LogLevel::INFO);
    let mut engine = LlmEngine::new(&sys_config, &log_manager)?;

    // Debug: tokenizer vocab alignment
    println!("=== TOKENIZER DEBUG ===");
    let vocab_size = engine.tokenizer.get_vocab_size(true);
    println!("Vocab size (with added): {}", vocab_size);
    let base_vocab_size = engine.tokenizer.get_vocab_size(false);
    println!("Vocab size (base): {}", base_vocab_size);
    
    let simple_prompt = "The capital of France is";
    let encoded = engine.tokenizer.encode(simple_prompt, false).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let ids = encoded.get_ids();
    println!("Simple prompt tokens ({}): {:?}", ids.len(), ids);
    
    // Check specific token IDs
    let paris_encoded = engine.tokenizer.encode(" Paris", false).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let paris_ids = paris_encoded.get_ids();
    println!("\" Paris\" tokens: {:?}", paris_ids);
    for &id in paris_ids {
        let decoded = engine.tokenizer.decode(&[id], false).unwrap_or_default();
        println!("  id={} decoded={:?}", id, decoded);
    }
    
    // Check some random IDs to confirm
    for test_id in [0, 1, 10, 270, 271, 1000, 10000, 24000, 248044, 248045, 248046] {
        let decoded = engine.tokenizer.decode(&[test_id], false).unwrap_or_default();
        println!("  id={} decoded={:?}", test_id, decoded);
    }

    // Pure deterministic mode: no sampling, no steering, no physics, no euler
    let mut config = NeuroConfig::engineer_mode();
    config.euler_resonance = false;
    config.physics_enabled = false;
    config.top_p = 0.9;
    config.min_p = 0.02;
    config.repeat_penalty = 1.15;
    config.temperature = 0.75;
    config.top_k = 40;

    // Test 1: Simple completion prompt (no ChatML)
    println!("\n=== TEST 1: Simple completion prompt ===");
    println!("Prompt: {:?}", simple_prompt);
    let output = generate_stream(&mut engine, simple_prompt, None, None, |t| { print!("{}", t); }, &config, 0, 64)?;
    println!("\n\n=== OUTPUT COMPLETE ===\nFinal output: {:?}\n\n", output);

    engine.reset_internal_state();

    // Test 2: English ChatML prompt (original test case) with pure inference
    println!("\n=== TEST 2: English ChatML prompt (pure inference) ===");
    let prompt_en = "<|im_start|>user\nWhat are the common features of Sydney and Istanbul?<|im_end|>\n<|im_start|>assistant\n";
    println!("Prompt: {:?}", prompt_en);
    let output2 = generate_stream(&mut engine, prompt_en, None, None, |t| { print!("{}", t); }, &config, 0, 64)?;
    println!("\n\n=== OUTPUT COMPLETE ===\nFinal output: {:?}\n\n", output2);

    Ok(())
}
