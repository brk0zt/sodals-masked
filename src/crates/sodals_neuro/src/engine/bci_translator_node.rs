// src/crates/sodals_neuro/src/engine/bci_translator_node.rs
// ============================================================================
// SODALS I/O NODE — Tier-1 BCI-to-Text Translator
// ============================================================================

use anyhow::{Result, Context};
use candle_core::{Tensor, Device, DType, IndexOp};
use candle_nn::{Conv1dConfig, Linear, VarBuilder, Module};
use crate::engine::llm_engine::LlmEngine;
use crate::engine::bci_logits_processor::{BciLogitsProcessor, create_default_bci_processor};
use std::path::Path;

/// Neural bridge projecting raw sEMG features into LLM Semantic Space.
pub struct SemgAdapter {
    conv1: candle_nn::Conv1d,
    conv2: candle_nn::Conv1d,
    fc_out: Linear,
}

impl SemgAdapter {
    pub fn from_safetensors(path: &Path, device: &Device) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, device)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        
        // 1D-CNN Temporal Feature Extractor
        // Input: [batch, 8, 256] -> conv1 -> [batch, 32, 128] -> conv2 -> [batch, 64, 64] -> flatten -> [batch, 4096] -> fc_out -> [batch, 12288]
        let cfg = candle_nn::Conv1dConfig { stride: 2, padding: 2, ..Default::default() };
        let conv1 = candle_nn::conv1d(8, 32, 5, cfg, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv1d(32, 64, 5, cfg, vb.pp("conv2"))?;
        let fc_out = candle_nn::linear(1024, 12288, vb.pp("fc_out"))?;
        
        Ok(Self { conv1, conv2, fc_out })
    }

    pub fn new_dummy(device: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, device);
        let cfg = candle_nn::Conv1dConfig { stride: 2, padding: 2, ..Default::default() };
        let conv1 = candle_nn::conv1d(8, 32, 5, cfg, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv1d(32, 64, 5, cfg, vb.pp("conv2"))?;
        let fc_out = candle_nn::linear(1024, 12288, vb.pp("fc_out"))?;
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

pub struct BciTranslator {
    pub adapter: SemgAdapter,
}

impl BciTranslator {
    pub fn new(adapter_path: &Path, device: &Device) -> Result<Self> {
        let adapter = if adapter_path.exists() {
            SemgAdapter::from_safetensors(adapter_path, device)?
        } else {
            eprintln!("[BCI NODE] WARNING: Adapter weights not found, using dummy initialization.");
            SemgAdapter::new_dummy(device)?
        };
        Ok(Self { adapter })
    }

    /// Performs a single-token forced articulation pass:
    /// It temporarily applies the phonetic boost/mask, forces the model to 
    /// generate EXACTLY ONE word, and immediately flushes state.
    /// 
    /// Uses GLOBAL Z-Score normalization (computed from dataset baseline)
    /// to preserve relative amplitude information across epochs.
    pub fn decode_thought(
        &mut self, 
        engine: &mut LlmEngine, 
        semg_raw: &Tensor, 
        qmem_domain: &str,
        global_mean: f64,
        global_std: f64,
    ) -> Result<String> {
        // 1. Project sEMG [1, 1, 2048] -> [1, 4096]
        let flat_semg = semg_raw.flatten_from(1)?;

        // 1b. Apply GLOBAL Z-Score Normalization (matches training pipeline)
        // CRITICAL: Using global statistics preserves relative amplitude
        // across epochs. Local normalization destroys signal relationships.
        let mul = 1.0 / (global_std + 1e-8);
        let add = -global_mean * mul;
        let normalized_semg = flat_semg.affine(mul, add)?;

        let adapted_latent = self.adapter.forward(&normalized_semg)?;

        // DEBUG: Check if adapter produces different outputs for different inputs
        let norm_mean = normalized_semg.mean_all()?.to_scalar::<f32>()?;
        let norm_std = (normalized_semg.affine(1.0, -norm_mean as f64)?.sqr()?.mean_all()?.to_scalar::<f32>()? as f64).sqrt() as f32;
        let adapted_mean = adapted_latent.mean_all()?.to_scalar::<f32>()?;
        let adapted_std = (adapted_latent.affine(1.0, -adapted_mean as f64)?.sqr()?.mean_all()?.to_scalar::<f32>()? as f64).sqrt() as f32;
        eprintln!("[BCI DEBUG] Input norm: mean={:.4}, std={:.4} | Adapted latent: mean={:.4}, std={:.4}", 
                  norm_mean, norm_std, adapted_mean, adapted_std);

        // 2. Setup isolated BCI Processor
        let vocab_size = {
            let model_guard = engine.model.read().unwrap();
            model_guard.model().embed_tokens.embeddings().dim(0)?
        };
        let mut bci_proc = create_default_bci_processor(vocab_size, &engine.device, &engine.tokenizer);
        // Tier-1 is a PURE phonetic sub-word synthesizer.
        // Semantic domain intelligence is injected at Tier-2 (QYPTOS) level,
        // not here. Hard vocabulary masking destroys the phonetic engine.
        bci_proc.semantic_masking_enabled = false;
        // Initialize Tier-1 phonetic quarantine: only 1-4 char ASCII alphabetic tokens allowed.
        bci_proc.initialize_phonetic_quarantine(&engine.tokenizer)?;
        bci_proc.set_semg_latent(adapted_latent);

        // 3. Isolated Forward Pass
        let prompt = format!(
            "<|im_start|>system\nDecode the subvocal thought into a single word from the domain: {}.\n<|im_end|>\n<|im_start|>user\nThought input detected. Decode:\n<|im_end|>\n<|im_start|>assistant\n",
            qmem_domain
        );
        
        let encoding = engine.tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!(e))?;
        let mut input_ids_vec = encoding.get_ids().to_vec();
        
        let mut final_word = String::new();
        let mut seq_len = 0;
        
        for step_idx in 0..3 {
            let input_ids = Tensor::new(input_ids_vec.as_slice(), &engine.device)?.unsqueeze(0)?;
            
            let logits = {
                let model_guard = engine.model.read().unwrap();
                model_guard.forward(&input_ids, seq_len, None, None, None, None, None, None, None)?
            };
            
            // Extract last token logits
            let last_logits = logits.i((0, logits.dim(1)? - 1, ..))?;

            // 4. Apply BCI Steer (Phonetic Boost + Semantic Mask)
            let embed_matrix = {
                let model_guard = engine.model.read().unwrap();
                model_guard.model().embed_tokens.embeddings().clone()
            };
            
            // process() handles phonetic boosting and EOS ban
            let steered_logits = bci_proc.process(&last_logits, Some(&embed_matrix), step_idx)?;

            // 4b. Hard-ban reasoning tokens (<think>, </think>) to prevent
            // the model from entering internal reasoning mode instead of decoding.
            let mut logit_vec: Vec<f32> = steered_logits.to_vec1()?;
            if logit_vec.len() > 151668 {
                logit_vec[151667] = f32::NEG_INFINITY; // <think>
                logit_vec[151668] = f32::NEG_INFINITY; // </think>
            }
            let final_logits = Tensor::from_vec(logit_vec, steered_logits.dims(), &engine.device)?;

            // 5. Sample & Decode
            let token_id = final_logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;
            
            if token_id == 151643 || token_id == 151645 {
                break;
            }

            let decoded_token = engine.tokenizer.decode(&[token_id], true).map_err(|e| anyhow::anyhow!(e))?;
            
            final_word.push_str(&decoded_token);
            
            // Prepare for next step
            seq_len += input_ids_vec.len();
            input_ids_vec = vec![token_id];
        }

        // Clean up partial sub-word artifacts
        final_word = final_word.replace("Ġ", "").replace("_", "").trim().to_string();

        // 6. State Sterilization
        engine.reset_internal_state();

        Ok(final_word)
    }
}
