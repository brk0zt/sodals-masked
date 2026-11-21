//! Dump Q/K tensors after Q/K-norm (layer 3) during prefill.
//! Uses the same ChatML prompt as generation_loop.

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
    std::env::set_var("SODALS_DEBUG_QK_NORM", "1");
    std::env::set_var("SODALS_DEBUG_LAYER", "3");

    let mut config = SystemConfig::default();
    config.model_path =
        r"C:\Users\Burak\Desktop\sodals_core\models\Qwen3.5-9B_Q4_K_M.gguf".to_string();
    config.tokenizer_path =
        r"C:\Users\Burak\Desktop\sodals_core\models\tokenizer.json".to_string();

    let logger = LogManager::new(LogLevel::INFO);
    eprintln!("[dump_qk_norm] Loading model...");
    let engine = LlmEngine::new(&config, &logger)?;

    let grounded = build_chatml_prompt("Senin ismin ne?");
    let encoded = engine
        .tokenizer
        .encode(grounded.clone(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let ids: Vec<u32> = encoded.get_ids().to_vec();
    eprintln!("[dump_qk_norm] {} tokens, last idx = {}", ids.len(), ids.len() - 1);
    eprintln!("[dump_qk_norm] first 5 IDs: {:?}", &ids[..5.min(ids.len())]);

    // Save token IDs for llama.cpp reference run (same prefill input)
    {
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        std::fs::write("rust_prompt_tokens.bin", &bytes)?;
        std::fs::write(
            "rust_prompt_tokens.txt",
            ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(","),
        )?;
        eprintln!("[dump_qk_norm] wrote rust_prompt_tokens.bin ({} u32 tokens)", ids.len());
    }

    std::env::set_var("SODALS_DEBUG_TOKEN", ids.len().saturating_sub(1).to_string());

    let input = Tensor::new(ids.as_slice(), &engine.device)?.unsqueeze(0)?;
    eprintln!("[dump_qk_norm] Running forward_hidden_states...");
    let _ = engine
        .model
        .read()
        .unwrap()
        .forward_hidden_states(&input, 0)?;

    eprintln!("[dump_qk_norm] Done. Check rust_q_normed_layer3_token*.bin and rust_k_normed_layer3_token*.bin");
    Ok(())
}
