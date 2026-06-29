// src/bin/bci_calibrator.rs
// ============================================================================
// CYBORG CALIBRATION BRIDGE — 5-Second Few-Shot Calibrator
// ============================================================================

use anyhow::{Result, Context};
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::{Conv1d, Conv1dConfig, Linear, VarBuilder, VarMap, Module, Optimizer};
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogManager, LogLevel};
use sodals_neuro::llm_core::LlmEngine;
use sodals_neuro::engine::semg_phoneme_hunter::SemgStreamer;
use std::path::Path;
use std::collections::HashSet;

const MAX_PHONEMES: usize = 3;

struct SemgAdapter {
    conv1: candle_nn::Conv1d,
    conv2: candle_nn::Conv1d,
    fc_out: Linear,
}

impl SemgAdapter {
    fn new(out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = candle_nn::Conv1dConfig { stride: 2, padding: 2, ..Default::default() };
        let conv1 = candle_nn::conv1d(8, 32, 5, cfg, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv1d(32, 64, 5, cfg, vb.pp("conv2"))?;
        let fc_out = candle_nn::linear(1024, out_dim, vb.pp("fc_out"))?;
        Ok(Self { conv1, conv2, fc_out })
    }
}

impl Module for SemgAdapter {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // Step 1: Restore physical hardware dimensions [batch, 8 channels, 64 timesteps]
        let xs = xs.reshape((xs.dim(0)?, 8, 64))?;
        // Step 2: Temporal convolution 1 -> [batch, 32, 32]
        let xs = self.conv1.forward(&xs)?.gelu()?;
        // Step 3: Temporal convolution 2 -> [batch, 64, 16]
        let xs = self.conv2.forward(&xs)?.gelu()?;
        // Step 4: Flatten -> [batch, 1024]
        let xs = xs.flatten_from(1)?;
        // Step 5: Project to latent space
        self.fc_out.forward(&xs)
    }
}

fn get_word_for_label(label: u32) -> &'static str {
    match label {
        31 | 61 => "up",
        32 | 62 => "down",
        33 | 63 => "right",
        34 | 64 => "left",
        _ => "unknown",
    }
}

fn mse_loss(pred: &Tensor, target: &Tensor) -> candle_core::Result<Tensor> {
    let diff = pred.sub(target)?;
    diff.sqr()?.mean_all()
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("============================================================");
    println!(" CYBORG CALIBRATION BRIDGE — 5-Second Few-Shot Calibrator");
    println!("============================================================");

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

    println!("\n[1] Booting LLM Engine to fetch embedding space...");
    let config = SystemConfig::default();
    let logger = LogManager::new(LogLevel::INFO);
    let engine = LlmEngine::new(&config, &logger).unwrap();
    
    let embed_matrix = {
        let model_guard = engine.model.read().unwrap();
        model_guard.model().embed_tokens.embeddings().clone()
    };
    let hidden_size = embed_matrix.dim(1)?;

    let semg_path = Path::new("real_subvocal_semg_sub02.safetensors");
    println!("\n[2] Loading Target Subject Data from {:?} ...", semg_path);
    let sample_rate = 256.0;
    
    // =========================================================================
    // PASS 1: Scan ALL epochs to compute GLOBAL Subject Baseline (Mean/Std)
    // This is the critical fix: Z-Score must use the full population statistics,
    // not the 4-sample subset which destroys phonetic amplitude peaks.
    // =========================================================================
    println!("    [PASS 1] Computing Global Subject Baseline from ALL epochs...");
    let mut streamer = SemgStreamer::new(semg_path, &device, sample_rate)
        .context("Failed to load sEMG dataset")?;
    let total_epochs = streamer.num_samples;
    
    let mut all_raw_features = Vec::new();
    for _ in 0..total_epochs {
        if let Ok((ghost_prefix, _label)) = streamer.fetch_next_phonetic_prefix() {
            all_raw_features.push(ghost_prefix.flatten_all()?);
        }
    }
    
    let all_stacked = Tensor::stack(&all_raw_features, 0)?;
    let global_mean = all_stacked.mean_all()?.to_scalar::<f32>()? as f64;
    let global_diff = all_stacked.affine(1.0, -global_mean)?;
    let global_var = global_diff.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
    let global_std = global_var.sqrt();
    
    println!("    [PASS 1] Global Baseline from {} epochs -> Mean: {:.6}, Std: {:.6}",
             all_raw_features.len(), global_mean, global_std);
    drop(all_raw_features);
    drop(all_stacked);
    
    // =========================================================================
    // PASS 2: Extract exactly 4 few-shot samples (one per cardinal direction)
    // and normalize them using the GLOBAL baseline from Pass 1.
    // =========================================================================
    println!("    [PASS 2] Extracting 4-shot samples with Global Baseline normalization...");
    let mut streamer2 = SemgStreamer::new(semg_path, &device, sample_rate)
        .context("Failed to reload sEMG dataset for Pass 2")?;
    
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut collected_words = HashSet::new();
    let target_words = ["up", "down", "left", "right"];
    
    for _ in 0..total_epochs {
        if collected_words.len() == 4 {
            break;
        }
        
        if let Ok((ghost_prefix, label)) = streamer2.fetch_next_phonetic_prefix() {
            let label_val = label as u32;
            let word = get_word_for_label(label_val);
            
            if word == "unknown" || !target_words.contains(&word) || collected_words.contains(word) {
                continue;
            }
            
            let spaced_word = format!(" {}", word);
            let encoding = engine.tokenizer.encode(spaced_word.as_str(), false)
                .unwrap_or_else(|_| engine.tokenizer.encode(word, false).unwrap());
            let ids = encoding.get_ids().to_vec();

            if !ids.is_empty() {
                let num_tokens = ids.len().min(MAX_PHONEMES);
                let mut sequence_embs = Vec::new();
                
                for i in 0..num_tokens {
                    let emb = embed_matrix.i((ids[i] as usize, ..))?;
                    sequence_embs.push(emb);
                }
                
                for _ in num_tokens..MAX_PHONEMES {
                    let eos_emb = embed_matrix.i((151643usize, ..))?;
                    sequence_embs.push(eos_emb);
                }
                
                let target_seq = Tensor::cat(&sequence_embs, 0)?;
                
                inputs.push(ghost_prefix.flatten_all()?);
                targets.push(target_seq);
                collected_words.insert(word.to_string());
                println!("    -> Extracted few-shot sample for: '{}'", word);
            }
        }
    }
    
    if collected_words.len() < 4 {
        anyhow::bail!("Could not find 4 distinct samples (up, down, left, right)!");
    }
    
    // --- Apply GLOBAL Subject Baseline Z-Score Normalization ---
    let stacked_inputs = Tensor::stack(&inputs, 0)?;
    let mul_val = 1.0 / (global_std + 1e-8);
    let add_val = -global_mean * mul_val;
    let normalized_inputs = stacked_inputs.affine(mul_val, add_val)?;
    
    println!("    [*] Applied GLOBAL Z-Score Normalization to 4-shot samples (Global Mean: {:.6}, Global Std: {:.6})", global_mean, global_std);

    let targets_tensor = Tensor::stack(&targets, 0)?;

    drop(engine);
    drop(embed_matrix);

    println!("\n[3] Loading Foundation Model (semg_foundation_adapter.safetensors)...");
    let mut varmap = VarMap::new();
    
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let input_dim = 512; 
    let adapter = SemgAdapter::new(hidden_size * MAX_PHONEMES, vb)?;
    
    varmap.load("semg_foundation_adapter.safetensors")?;
    println!("    -> Foundation weights loaded.");
    
    println!("\n[4] Initializing Transfer Learning Loop...");
    let mut trainable_vars = Vec::new();
    {
        let data = varmap.data().lock().unwrap();
        for (name, var) in data.iter() {
            if name.contains("conv") {
                trainable_vars.push(var.clone());
            }
        }
    }
    
    println!("    -> Found {} trainable variable(s) for 'conv1/conv2'. 'fc_out' is strictly frozen.", trainable_vars.len());
    
    let mut opt = candle_nn::optim::AdamW::new_lr(trainable_vars, 1e-4)?;
    
    let epochs = 200;
    println!("\n[5] Executing Gentle Transfer Learning ({} epochs, lr=1e-4) ...", epochs);
    
    for epoch in 1..=epochs {
        let pred = adapter.forward(&normalized_inputs)?;
        let loss = mse_loss(&pred, &targets_tensor)?;
        
        opt.backward_step(&loss)?;
        
        if epoch % 25 == 0 || epoch == 1 {
            println!("    -> Epoch {:3}/{}: Loss = {:.6}", epoch, epochs, loss.to_scalar::<f32>()?);
        }
    }
    
    println!("\n[6] Exporting Fine-Tuned Artifacts...");
    varmap.save("semg_calibrated_adapter.safetensors")?;
    println!("    -> Successfully saved calibrated weights to semg_calibrated_adapter.safetensors");
    
    println!("\n[CYBORG CALIBRATION] 5-Second Transfer Learning Complete.");
    Ok(())
}
