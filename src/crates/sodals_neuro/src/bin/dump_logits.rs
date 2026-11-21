use candle_core::{Tensor};
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogLevel, LogManager};
use sodals_neuro::llm_core::LlmEngine;
use std::fs;

fn main() -> anyhow::Result<()> {
    std::env::set_var("SODALS_PURE_INFERENCE", "1");
    std::env::set_var("SODALS_DUMP_LAYERS", "1");
    std::env::set_var("SODALS_DUMP_LAYERS_BIN", "1");
    std::env::set_var("SODALS_DUMP_ATTN_RESIDUAL", "1");
    std::env::set_var("SODALS_DUMP_SSM", "1");

    let mut config = SystemConfig::default();
    config.model_path =
        r"C:\Users\Burak\Desktop\sodals_core\models\Qwen3.5-9B_Q4_K_M.gguf".to_string();
    config.tokenizer_path =
        r"C:\Users\Burak\Desktop\sodals_core\models\tokenizer.json".to_string();

    let logger = LogManager::new(LogLevel::INFO);
    eprintln!("[compare_layers] Loading model...");
    let engine = LlmEngine::new(&config, &logger)?;

    // Read the same 48 tokens used for the llama reference dumps
    let token_path = r"C:\Users\Burak\Desktop\sodals_core\rust_prompt_tokens.bin";
    let token_data = fs::read(token_path)?;
    let tokens: Vec<u32> = token_data.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    eprintln!("[compare_layers] Read {} tokens from {}", tokens.len(), token_path);

    let input = Tensor::from_slice(&tokens, tokens.len(), &engine.device)?.unsqueeze(0)?;

    eprintln!("[compare_layers] Forward pass with {} tokens...", tokens.len());
    let _logits = engine.model.read().unwrap().forward(
        &input, 0, None, None, None, None, None, None, None, None, None
    )?;
    eprintln!("[compare_layers] Forward pass complete.");

    // Now compare our rust_l_out-*.bin files with llama_l_out-*.bin files
    eprintln!("\n[compare_layers] Comparing per-layer outputs with llama reference:");
    for layer_idx in 0..32 {
        let rust_path = format!("rust_l_out-{}.bin", layer_idx);
        let llama_path = format!(r"C:\Users\Burak\Desktop\sodals_core\llama_l_out-{}.bin", layer_idx);

        let rust_data = match fs::read(&rust_path) {
            Ok(d) => d,
            Err(e) => { eprintln!("  Layer {}: no rust dump ({})", layer_idx, e); continue; }
        };
        let llama_data = match fs::read(&llama_path) {
            Ok(d) => d,
            Err(e) => { eprintln!("  Layer {}: no llama ref ({})", layer_idx, e); continue; }
        };

        if rust_data.len() != llama_data.len() {
            eprintln!("  Layer {}: SIZE MISMATCH rust={} llama={}", layer_idx, rust_data.len(), llama_data.len());
            continue;
        }

        let rust_vals: Vec<f32> = rust_data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let llama_vals: Vec<f32> = llama_data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let n = rust_vals.len();
        let mut max_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f64;
        let mut large_count = 0;
        for i in 0..n {
            let diff = (rust_vals[i] - llama_vals[i]).abs();
            sum_abs_diff += diff as f64;
            if diff > max_diff { max_diff = diff; }
            if diff > 0.01 { large_count += 1; }
        }
        let avg_diff = sum_abs_diff / n as f64;

        // Per-dimension stats
        let rust_rms = (rust_vals.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
        let llama_rms = (llama_vals.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
        let rust_mean = rust_vals.iter().sum::<f32>() / n as f32;
        let llama_mean = llama_vals.iter().sum::<f32>() / n as f32;
        let rust_min = rust_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let llama_min = llama_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let rust_max = rust_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let llama_max = llama_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let flag = if max_diff < 0.01 { "OK  " } else if max_diff < 0.1 { "WARN" } else if max_diff < 1.0 { "BIG " } else { "FAIL" };
        eprintln!("  [{}] Layer {:2}: max_diff={:.4e} avg_diff={:.4e} large(>0.01)={:>5}/{:<5} | rust: RMS={:.4} mean={:.4} min={:.4} max={:.4} | llama: RMS={:.4} mean={:.4} min={:.4} max={:.4}",
            flag, layer_idx, max_diff, avg_diff, large_count, n, rust_rms, rust_mean, rust_min, rust_max, llama_rms, llama_mean, llama_min, llama_max);

        // Also print the 10 largest differences for FAIL layers
        if max_diff > 1.0 {
            let mut diffs: Vec<(usize, f32)> = (0..n.min(1000)).map(|i| (i, (rust_vals[i] - llama_vals[i]).abs())).collect();
            diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("    Top-5 worst diffs:");
            for i in 0..5.min(diffs.len()) {
                let (idx, d) = diffs[i];
                eprintln!("      [{}] idx={:>6} rust={:+.4} llama={:+.4} diff={:.4}", i, idx, rust_vals[idx], llama_vals[idx], d);
            }
        }
    }

    Ok(())
}
