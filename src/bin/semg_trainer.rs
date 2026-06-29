// src/bin/semg_trainer.rs
// ============================================================================
// CYBORG CALIBRATION BRIDGE — sEMG 1D-CNN Adapter Trainer
// ============================================================================

use anyhow::{Result, Context};
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::{Conv1d, Conv1dConfig, Linear, VarBuilder, VarMap, Module, Optimizer};
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogManager, LogLevel};
use sodals_neuro::llm_core::LlmEngine;
use sodals_neuro::engine::semg_phoneme_hunter::SemgStreamer;
use std::path::Path;

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

// Nieto Dataset trigger code mapping
// Source: Inner Speech EEG dataset event markers
fn get_word_for_label(label: u32) -> &'static str {
    match label {
        // Protocol markers (no words)
        1 => "protocol_start",
        12 => "protocol_end",
        13 => "baseline_start",
        14 => "baseline_end",
        15 => "run_start",
        16 => "run_end",
        17 => "cognitive_control_question",
        21 => "pronounced_speech_start",
        22 => "inner_speech_start",
        23 => "visualized_condition_start",

        // Trial cues - Direction words
        31 => "up",       // "Arriba/Up" trial - start of cue interval
        32 => "down",     // "Abajo/Down" trial - start of cue interval
        33 => "right",    // "Derecha/Right" trial - start of cue interval
        34 => "left",     // "Izquierda/Left" trial - start of cue interval

        // Interval markers (no words)
        42 => "concentration_start",
        44 => "action_start",
        45 => "relax_start",
        46 => "rest_start",
        51 => "inter_run_rest_start",

        // Cognitive control answers - Direction words
        61 => "up",       // Answer: "Arriba/Up"
        62 => "down",     // Answer: "Abajo/Down"
        63 => "right",    // Answer: "Derecha/Right"
        64 => "left",     // Answer: "Izquierda/Left"

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
    println!(" CYBORG CALIBRATION BRIDGE — sEMG Adapter Trainer");
    println!("============================================================");

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

    // 1. Boot LLM Engine to get embedding matrix and tokenizer
    println!("\n[1] Booting LLM Engine to fetch embedding space...");
    let config = SystemConfig::default();
    let logger = LogManager::new(LogLevel::INFO);
    let engine = LlmEngine::new(&config, &logger).unwrap();
    
    let embed_matrix = {
        let model_guard = engine.model.read().unwrap();
        model_guard.model().embed_tokens.embeddings().clone()
    };
    let hidden_size = embed_matrix.dim(1)?;
    println!("    -> Extracted embed_matrix: {:?}", embed_matrix.shape());

    // 2. Load Multi-Subject sEMG Data
    println!("\n[2] Scanning for Multi-Subject sEMG Data...");
    let sample_rate = 256.0;
    
    let mut global_inputs = Vec::new();
    let mut global_targets = Vec::new();
    
    let mut subject_files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(".") {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("real_subvocal_semg_sub") && name.ends_with(".safetensors") {
                    subject_files.push(path);
                }
            }
        }
    }
    
    if subject_files.is_empty() {
        let fallback = Path::new("real_subvocal_semg.safetensors");
        if fallback.exists() {
            subject_files.push(fallback.to_path_buf());
        }
    }
    
    if subject_files.is_empty() {
        anyhow::bail!("No subject datasets found in the current directory!");
    }
    
    for semg_path in &subject_files {
        println!("    -> Processing Subject Dataset: {:?}", semg_path);
        
        let mut streamer = SemgStreamer::new(semg_path, &device, sample_rate)
            .context(format!("Failed to load {:?}", semg_path))?;
        let total_epochs = streamer.num_samples;
        
        let mut subject_inputs = Vec::new();
        let mut subject_targets = Vec::new();
        
        for _ in 0..total_epochs {
            if let Ok((ghost_prefix, label)) = streamer.fetch_next_phonetic_prefix() {
                let label_val = label as u32;
                let word = get_word_for_label(label_val);
                
                if word == "unknown" {
                    continue;
                }
                
                let spaced_word = format!(" {}", word);
                let encoding = engine.tokenizer.encode(spaced_word.as_str(), false).map_err(|e| anyhow::anyhow!(e))?;
                let mut ids = encoding.get_ids().to_vec();
                
                if ids.is_empty() {
                    let encoding = engine.tokenizer.encode(word, false).map_err(|e| anyhow::anyhow!(e))?;
                    ids = encoding.get_ids().to_vec();
                }

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
                    
                    subject_inputs.push(ghost_prefix.flatten_all()?);
                    subject_targets.push(target_seq);
                }
            }
        }
        
        if subject_inputs.is_empty() {
            println!("       [!] No valid mapped samples for this subject.");
            continue;
        }
        
        // --- Subject-Wise Z-Score Normalization ---
        let stacked_inputs = Tensor::stack(&subject_inputs, 0)?;
        
        let mean_val = stacked_inputs.mean_all()?.to_scalar::<f32>()? as f64;
        let diff = stacked_inputs.affine(1.0, -mean_val)?;
        let var_val = diff.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
        let std_val = var_val.sqrt();
        
        let mul_val = 1.0 / (std_val + 1e-8);
        let add_val = -mean_val * mul_val;
        let normalized_inputs = stacked_inputs.affine(mul_val, add_val)?;
        
        println!("       [*] Normalized - Mean: {:.6}, Std: {:.6}", mean_val, std_val);
        
        global_inputs.push(normalized_inputs);
        let stacked_targets = Tensor::stack(&subject_targets, 0)?;
        global_targets.push(stacked_targets);
    }
    
    if global_inputs.is_empty() {
        anyhow::bail!("No valid samples mapped across any subjects!");
    }
    
    let inputs_tensor = Tensor::cat(&global_inputs, 0)?;
    let targets_tensor = Tensor::cat(&global_targets, 0)?;
    let num_samples = inputs_tensor.dim(0)?;
    
    println!("    -> Prepared {} total foundation samples for training.", num_samples);

    // 3. Initialize Adapter Network
    println!("\n[3] Initializing SemgFoundationAdapter (1D-CNN Temporal Extractor) ...");
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    
    let input_dim = 512; // 8 channels * 64 timesteps
    let adapter = SemgAdapter::new(hidden_size * MAX_PHONEMES, vb)?;
    
    let initial_lr = 1e-4;
    let mut opt = candle_nn::optim::AdamW::new_lr(varmap.all_vars(), initial_lr)?;
    
    let epochs = 100;
    
    // Gradient explosion guard threshold:
    // MSE loss on 12288-dim output can produce enormous gradients.
    // When batch loss > this threshold, we scale the loss down before backward
    // so the gradient magnitude stays bounded. This is equivalent to adaptive
    // gradient norm clipping implemented via loss scaling.
    let grad_clip_threshold = 1.0f64;
    
    println!("\n[4] Starting Foundation Training Loop ({} epochs, initial LR={}, grad_clip={}) ...", epochs, initial_lr, grad_clip_threshold);

    // Free the LLM Engine to recover 14+ GB of VRAM for AdamW and gradients
    drop(engine);
    // Also drop embed_matrix to free the 1.2 GB embedding table
    drop(embed_matrix);

    let batch_size = 64;
    let num_batches = (num_samples + batch_size - 1) / batch_size;
    let mut best_loss = f32::MAX;

    for epoch in 1..=epochs {
        // --- Step Learning Rate Decay ---
        // Aggressive early decay to catch the danger zone before momentum explodes.
        // Phase 1 (1-25):   LR = 1e-4 (controlled descent)
        // Phase 2 (26-50):  LR = 1e-5 (fine-tuning)
        // Phase 3 (51-100): LR = 5e-6 (stabilization lock)
        if epoch == 26 {
            let new_lr = 1e-5;
            opt.set_learning_rate(new_lr);
            println!("    [LR DECAY] Epoch {}: Learning Rate reduced to {}", epoch, new_lr);
        } else if epoch == 51 {
            let new_lr = 5e-6;
            opt.set_learning_rate(new_lr);
            println!("    [LR DECAY] Epoch {}: Learning Rate reduced to {}", epoch, new_lr);
        }

        let mut total_loss = 0.0f32;
        let mut clipped_batches = 0u32;
        
        for i in 0..num_batches {
            let start = i * batch_size;
            let end = (start + batch_size).min(num_samples);
            let current_batch_size = end - start;
            
            let b_inputs = inputs_tensor.narrow(0, start, current_batch_size)?;
            let b_targets = targets_tensor.narrow(0, start, current_batch_size)?;
            
            let pred = adapter.forward(&b_inputs)?;
            let loss = mse_loss(&pred, &b_targets)?;
            let loss_val = loss.to_scalar::<f32>()? as f64;
            
            // --- Gradient Explosion Guard (Loss-Scaling Clip) ---
            // When loss > threshold, scale it down so backward() receives bounded
            // gradients. This prevents the AdamW momentum from accumulating in
            // one catastrophic direction and blowing weights out of Qwen's manifold.
            let effective_loss = if loss_val > grad_clip_threshold {
                clipped_batches += 1;
                let scale = grad_clip_threshold / loss_val;
                loss.affine(scale, 0.0)?
            } else {
                loss
            };
            
            opt.backward_step(&effective_loss)?;
            total_loss += loss_val as f32;
        }
        
        let avg_loss = total_loss / num_batches as f32;
        
        // --- Best-Model Checkpoint ---
        // Saves the best weights so even if later epochs destabilize,
        // we never lose the optimal convergence point.
        if avg_loss < best_loss {
            best_loss = avg_loss;
            varmap.save("semg_foundation_adapter.safetensors")?;
        }
        
        if epoch % 5 == 0 || epoch == 1 {
            println!("    -> Epoch {:3}/{}: Avg Loss = {:.6} (best: {:.6}) [clipped: {}/{}]", 
                     epoch, epochs, avg_loss, best_loss, clipped_batches, num_batches);
        }
    }
    
    // Final save is the best model (already saved via checkpoint above)
    println!("\n[5] Exporting Artifacts...");
    println!("    -> Best model already saved to semg_foundation_adapter.safetensors (best loss: {:.6})", best_loss);
    
    println!("\n[CYBORG CALIBRATION] Pipeline Complete. Foundation Adapter ready for online inference.");
    Ok(())
}
