use candle_core::{DType, Tensor};
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogLevel, LogManager};
use sodals_neuro::llm_core::LlmEngine;

fn main() -> anyhow::Result<()> {
    std::env::set_var("SODALS_PURE_INFERENCE", "1");
    std::env::set_var("SODALS_VERBOSE", "1");
    std::env::set_var("SODALS_DUMP_LAYERS", "1");

    let mut config = SystemConfig::default();
    config.model_path =
        r"C:\Users\Burak\Desktop\sodals_core\models\Qwen3.5-9B_Q4_K_M.gguf".to_string();
    config.tokenizer_path =
        r"C:\Users\Burak\Desktop\sodals_core\models\tokenizer.json".to_string();

    let logger = LogManager::new(LogLevel::INFO);
    eprintln!("[layer_dump] Loading model...");
    let engine = LlmEngine::new(&config, &logger)?;

    // Use a single-token input: token 9906 = "Hello"
    let token = 9906u32;
    let input = Tensor::new(&[token], &engine.device)?.unsqueeze(0)?;

    eprintln!("\n===== FORWARD PASS (1 token, prefill) =====");
    let _logits = engine.model.read().unwrap().forward(
        &input, 0, None, None, None, None, None, None, None, None, None
    )?;

    eprintln!("\n===== FORWARD PASS (1 token, decode) =====");
    let token2 = 44072u32; // a common token after "Hello"
    let input2 = Tensor::new(&[token2], &engine.device)?.unsqueeze(0)?;
    let _logits2 = engine.model.read().unwrap().forward(
        &input2, 1, None, None, None, None, None, None, None, None, None
    )?;

    eprintln!("\n[layer_dump] Done");
    Ok(())
}
