use candle_core::{DType, Tensor};
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogLevel, LogManager};
use sodals_neuro::llm_core::LlmEngine;

fn main() -> anyhow::Result<()> {
    std::env::set_var("SODALS_PURE_INFERENCE", "1");
    std::env::set_var("SODALS_VERBOSE", "1");

    let mut config = SystemConfig::default();
    config.model_path =
        r"C:\Users\Burak\Desktop\sodals_core\models\Qwen3.5-9B_Q4_K_M.gguf".to_string();
    config.tokenizer_path =
        r"C:\Users\Burak\Desktop\sodals_core\models\tokenizer.json".to_string();

    let logger = LogManager::new(LogLevel::INFO);
    eprintln!("[ssm_debug] Loading model...");
    let engine = LlmEngine::new(&config, &logger)?;

    // Single token forward: "Hello" = 9419
    let input = Tensor::new(&[9419u32], &engine.device)?.unsqueeze(0)?;
    eprintln!("\n[FORWARD] Single token id=9419 'Hello'");
    let logits = engine.model.read().unwrap().forward(
        &input, 0, None, None, None, None, None, None, None, None, None
    )?;

    // Get top-10 logits
    let last = logits.squeeze(1)?.to_dtype(DType::F32)?;
    let vec: Vec<f32> = last.flatten_all()?.to_vec1()?;
    let mut rank: Vec<(u32, f32)> = vec.iter().enumerate()
        .map(|(i, &v)| (i as u32, v)).collect();
    rank.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("\n[LOGITS] Top-10 after single token 'Hello' (id=9419):");
    for (i, (id, logit)) in rank.iter().take(10).enumerate() {
        let t = engine.tokenizer.decode(&[*id], false).unwrap_or_default();
        let display = if t.trim().is_empty() { format!("<0x{:02X}>", id) } else { t.trim().to_string() };
        eprintln!("  top{}: id={:>6} logit={:>8.2} tok='{}'", i+1, id, logit, display);
    }

    eprintln!("\n[ssm_debug] Done");
    Ok(())
}
