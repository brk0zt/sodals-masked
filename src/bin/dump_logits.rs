use anyhow::Result;
use candle_core::Tensor;
use candle_core::Module;
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogLevel, LogManager};
use sodals_neuro::engine::llm_engine::LlmEngine;

fn main() -> Result<()> {
    std::env::set_var("SODALS_PURE_INFERENCE", "1");

    let sys_config = SystemConfig::load().unwrap_or_else(|_| SystemConfig::default());
    let log_manager = LogManager::new(LogLevel::INFO);
    let engine = LlmEngine::new(&sys_config, &log_manager)?;

    // Fixed input: 48 tokens, identical to what llama-debug consumes via LLAMA_TOKENS_FILE.
    let raw = std::fs::read("rust_prompt_tokens.bin")?;
    let n = raw.len() / 4;
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let b: [u8; 4] = [raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]];
        ids.push(u32::from_le_bytes(b));
    }
    println!("[DUMP_LOGITS] loaded {} tokens", n);

    let ids_t = Tensor::new(&ids[..], &engine.device)?.unsqueeze(0)?; // [1, n]

    let guard = engine.model.read().unwrap();

    // J-Lens diagnostics in LlmEngine::new runs a dummy-prompt forward that populates
    // every SSM layer's state_cache. Clear all KV/SSM caches so this dump starts from a
    // true zero initial state, matching llama.cpp's fresh prefill.
    guard.model.clear_kv_cache();

    // Full prefill, all neuromorphic scaffolding disabled (phase=None, no ghost prefix, no qmem).
    let hidden = guard.model.forward(
        &ids_t,
        0,
        None,                        // poles_ptr
        None,                        // impedance_ptr
        None,                        // flags_ptr
        None,                        // n_nodes
        None,                        // phase
        None,                        // ghost_prefix
        None,                        // token_concept_map_ptr
        Some(&guard.model.qmem_parameters),
        None,                        // qmem_loader
        None,                        // qmem_cartridge
        None,                        // gnn_graph_ptrs
    )?;
    eprintln!("[DUMP_LOGITS] hidden shape: {:?}", hidden.shape());

    // Apply LM head to every position -> [1, seq, vocab]
    let logits_all = guard.lm_head.forward(&hidden)?.to_dtype(candle_core::DType::F32)?;
    eprintln!("[DUMP_LOGITS] logits_all shape: {:?}", logits_all.shape());
    drop(guard);

    let logits_all = logits_all.squeeze(0)?; // [seq, vocab]
    let seq = logits_all.dim(0)?;
    let vocab = logits_all.dim(1)?;

    let flat: Vec<f32> = logits_all.flatten_all()?.to_vec1()?;
    let mut bytes = Vec::with_capacity(flat.len() * 4);
    for v in &flat {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write("rust_logits_all.bin", &bytes)?;
    println!(
        "[DUMP_LOGITS] wrote rust_logits_all.bin: seq={} vocab={} ({:.1} MB, seq-major f32)",
        seq,
        vocab,
        bytes.len() as f64 / 1e6
    );

    // Last-token top-10 for a quick sanity print
    let last = &flat[(seq - 1) * vocab..seq * vocab];
    let mut ranked: Vec<(usize, f32)> = last.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("[DUMP_LOGITS] last-token top-10:");
    for (i, (tid, lg)) in ranked.iter().take(10).enumerate() {
        println!("  top{}: id={} logit={:.4}", i + 1, tid, lg);
    }

    Ok(())
}
