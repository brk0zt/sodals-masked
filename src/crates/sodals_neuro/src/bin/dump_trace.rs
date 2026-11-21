//! Layer-by-layer divergence trace dumps (attn_norm_in, Q/K proj, RoPE, layer_out).

use candle_core::Tensor;
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogLevel, LogManager};
use sodals_neuro::llm_core::LlmEngine;

fn build_chatml_prompt(user: &str) -> String {
    format!(
        "<|im_start|>system\n\
You are a precise expert analyst. Answer the user's question based strictly on the \
provided context. Do not perform mathematical computations unless explicitly asked.\n\
<|im_end|>\n\
<|im_start|>user\n\
{user}\n\
<|im_end|>\n\
<|im_start|>assistant\n"
    )
}

fn main() -> anyhow::Result<()> {
    std::env::set_var("SODALS_PURE_INFERENCE", "1");
    std::env::set_var("SODALS_TRACE_DUMP", "1");
    std::env::set_var("SODALS_TRACE_LAYER", "3");

    let mut config = SystemConfig::default();
    config.model_path =
        r"C:\Users\Burak\Desktop\sodals_core\models\Qwen3.5-9B_Q4_K_M.gguf".to_string();
    config.tokenizer_path =
        r"C:\Users\Burak\Desktop\sodals_core\models\tokenizer.json".to_string();

    let logger = LogManager::new(LogLevel::INFO);
    eprintln!("[dump_trace] Loading model...");
    let engine = LlmEngine::new(&config, &logger)?;

    let grounded = build_chatml_prompt("Senin ismin ne?");
    let encoded = engine
        .tokenizer
        .encode(grounded, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let ids: Vec<u32> = encoded.get_ids().to_vec();
    let last = ids.len().saturating_sub(1);
    eprintln!("[dump_trace] {} tokens, trace token = {last}", ids.len());

    std::fs::write(
        "rust_prompt_tokens.bin",
        ids.iter().flat_map(|id| id.to_le_bytes()).collect::<Vec<_>>(),
    )?;
    std::env::set_var("SODALS_TRACE_TOKEN", last.to_string());

    let input = Tensor::new(ids.as_slice(), &engine.device)?.unsqueeze(0)?;
    let _ = engine
        .model
        .read()
        .unwrap()
        .forward_hidden_states(&input, 0)?;

    eprintln!("[dump_trace] Done — rust_*_layer3_token{last}.bin files written");
    Ok(())
}
