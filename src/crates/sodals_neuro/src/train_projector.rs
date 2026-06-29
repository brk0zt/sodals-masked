// src/train_projector.rs - Teacher-Student Alignment Training

use candle_core::{Tensor, Device, DType};
use candle_nn::{Optimizer, VarBuilder, VarMap, AdamW};
use crate::interface_layer::HolographicProjector;
use crate::fast_math::SemanticWave;
use crate::physics_neuron::PhysicsNeuron;
use std::path::Path;
use tracing::{info, debug};

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub lambda_magnitude: f64, // Weight for magnitude loss
    pub device: Device,
    pub checkpoint_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 32,
            num_epochs: 1000,
            lambda_magnitude: 0.5,
            device: Device::Cpu,
            checkpoint_interval: 100,
        }
    }
}

/// Training dataset: word -> (QYPTOS wave, Qwen embedding)
pub struct AlignmentDataset {
    /// Input: Complex-valued semantic waves from QYPTOS
    waves: Vec<SemanticWave>,
    
    /// Target: Real-valued embeddings from frozen Qwen
    embeddings: Vec<Vec<f64>>,
    
    /// Vocabulary: words corresponding to each sample
    words: Vec<String>,
}

impl AlignmentDataset {
    /// Create dataset from vocabulary
    /// 
    /// For each word:
    /// 1. Generate QYPTOS wave (physics neuron simulation)
    /// 2. Get Qwen embedding (frozen teacher)
    pub fn from_vocabulary(
        words: &[String],
        wave_dim: usize,
        embedding_dim: usize,
    ) -> anyhow::Result<Self> {
        info!("Creating alignment dataset for {} words", words.len());
        
        let mut waves = Vec::new();
        let mut embeddings = Vec::new();

        for word in words {
            // Generate QYPTOS wave for word
            let wave = Self::generate_sodals_wave(word, wave_dim)?;
            waves.push(wave);

            // Get Qwen embedding (would call frozen model in production)
            let embedding = Self::get_qwen_embedding(word, embedding_dim)?;
            embeddings.push(embedding);
        }

        Ok(Self {
            waves,
            embeddings,
            words: words.to_vec(),
        })
    }

    /// Generate semantic wave using QYPTOS physics
    /// 
    /// In production: Feed word through trained QYPTOS network
    /// For now: Simulate with physics neuron
    fn generate_sodals_wave(word: &str, dim: usize) -> anyhow::Result<SemanticWave> {
        // Simple heuristic: Hash word to get initial phase/frequency
        let hash = Self::hash_string(word);
        let base_freq = 1.0 + (hash % 10) as f64 * 0.1;
        let base_phase = (hash % 360) as f64 * std::f64::consts::PI / 180.0;

        // Initialize neurons
        let mut neurons = Vec::new();
        for i in 0..dim {
            let freq_offset = (i as f64 * 0.1) % 1.0;
            let _phase_offset = (i as f64 * 0.2) % (2.0 * std::f64::consts::PI);
            
            let mut neuron = PhysicsNeuron::new(0.1);
            neuron.param_a = base_freq + freq_offset;
            neurons.push(neuron);
        }

        // Simulate for a few steps to reach steady state
        let input_signal = num_complex::Complex64::new(
            base_phase.cos(),
            base_phase.sin(),
        );

        let mut states = Vec::new();
        for neuron in &mut neurons {
            // Drive with input
            for _ in 0..10 {
                neuron.forward(input_signal);
            }
            states.push(neuron.get_state());
        }

        Ok(SemanticWave::from_vec(states))
    }

    /// Get embedding from frozen Qwen model
    /// 
    /// In production: Call actual Qwen tokenizer + embedding layer
    /// For now: Deterministic hash-based embedding
    fn get_qwen_embedding(word: &str, dim: usize) -> anyhow::Result<Vec<f64>> {
        let hash = Self::hash_string(word);
        let mut embedding = Vec::with_capacity(dim);
        
        // Generate deterministic embedding based on word
        for i in 0..dim {
            let seed = hash.wrapping_add(i as u64);
            let value = (seed as f64).sin() * 0.1; // Small values like real embeddings
            embedding.push(value);
        }

        Ok(embedding)
    }

    /// Simple string hash for reproducibility
    fn hash_string(s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    /// Get batch for training
    pub fn get_batch(
        &self,
        batch_idx: usize,
        batch_size: usize,
    ) -> Option<Batch> {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(self.waves.len());
        
        if start >= self.waves.len() {
            return None;
        }

        Some(Batch {
            waves: self.waves[start..end].to_vec(),
            embeddings: self.embeddings[start..end].to_vec(),
            words: self.words[start..end].to_vec(),
        })
    }

    pub fn num_batches(&self, batch_size: usize) -> usize {
        self.waves.len().div_ceil(batch_size)
    }
}

/// Single training batch
pub struct Batch {
    pub waves: Vec<SemanticWave>,
    pub embeddings: Vec<Vec<f64>>,
    pub words: Vec<String>,
}

/// Train holographic projector (Whitepaper II4.1)
/// 
/// Teacher-Student setup:
/// - Teacher: Frozen Qwen embeddings (ground truth)
/// - Student: SODALS projector (learnable)
/// - Loss: Cosine similarity + magnitude matching
pub fn train_alignment_layer(
    config: TrainingConfig,
    dataset: AlignmentDataset,
    checkpoint_path: Option<&Path>,
) -> anyhow::Result<HolographicProjector> {
    info!("Starting holographic alignment training");
    info!("Config: {:?}", config);

    let device = &config.device;
    
    // Setup model and optimizer
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F64, device);

    let wave_dim = dataset.waves[0].len();
    let embedding_dim = dataset.embeddings[0].len();

    let projector = HolographicProjector::new(vb.clone(), wave_dim, embedding_dim)?;
    
    // Use AdamW (better than SGD for this task)
    let mut opt = AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: config.learning_rate,
            ..Default::default()
        },
    )?;

    info!("Model: {}D wave -> {}D embedding", wave_dim, embedding_dim);
    info!("Dataset: {} samples", dataset.waves.len());

    // Training loop
    let mut best_loss = f64::INFINITY;
    
    for epoch in 0..config.num_epochs {
        let mut epoch_loss = 0.0;
        let num_batches = dataset.num_batches(config.batch_size);

        for batch_idx in 0..num_batches {
            let batch = dataset.get_batch(batch_idx, config.batch_size)
                .expect("Batch index out of bounds");

            // Forward pass
            let loss = compute_batch_loss(
                &projector,
                &batch,
                config.lambda_magnitude,
                device,
            )?;

            // Backward pass
            opt.backward_step(&loss)?;

            epoch_loss += loss.to_scalar::<f64>()?;
        }

        epoch_loss /= num_batches as f64;

        // Logging
        if epoch % 10 == 0 {
            debug!("Epoch {}: loss={:.6}", epoch, epoch_loss);
        }

        if epoch % config.checkpoint_interval == 0 {
            info!("Epoch {}: loss={:.6}", epoch, epoch_loss);
            
            // Save best model
            if epoch_loss < best_loss {
                best_loss = epoch_loss;
                if let Some(path) = checkpoint_path {
                    save_checkpoint(&varmap, path)?;
                    info!("Saved checkpoint (loss={:.6})", best_loss);
                }
            }
        }
    }

    info!("Training complete. Best loss: {:.6}", best_loss);
    Ok(projector)
}

/// Compute loss for a batch (Whitepaper II4.2)
/// 
/// L_total = L_cos + IIIIL_mag
fn compute_batch_loss(
    projector: &HolographicProjector,
    batch: &Batch,
    lambda: f64,
    device: &Device,
) -> anyhow::Result<Tensor> {
    // Project waves to embeddings
    let pred = projector.project_batch(&batch.waves)?;
    
    // Stack target embeddings
    let target = stack_embeddings(&batch.embeddings, device)?;

    // Cosine similarity loss
    let pred_norm = pred.sqr()?.sum_keepdim(1)?.sqrt()?;
    let target_norm = target.sqr()?.sum_keepdim(1)?.sqrt()?;
    
    let pred_normalized = pred.broadcast_div(&pred_norm)?;
    let target_normalized = target.broadcast_div(&target_norm)?;
    
    let cosine_sim = (pred_normalized * target_normalized)?.sum(1)?;
    let l_cos = (1.0 - cosine_sim)?.mean_all()?;

    // Magnitude loss
    let mag_diff = (pred_norm - target_norm)?;
    let l_mag = mag_diff.sqr()?.mean_all()?;

    // Combined loss
    let loss = (l_cos + (lambda * l_mag)?)?;
    
    Ok(loss)
}

/// Stack embeddings into batch tensor
fn stack_embeddings(embeddings: &[Vec<f64>], device: &Device) -> anyhow::Result<Tensor> {
    let batch_size = embeddings.len();
    let dim = embeddings[0].len();
    
    let flat: Vec<f64> = embeddings.iter()
        .flat_map(|e| e.iter().copied())
        .collect();
    
    Tensor::from_vec(flat, (batch_size, dim), device).map_err(anyhow::Error::from)
}

/// Save checkpoint
fn save_checkpoint(varmap: &VarMap, path: &Path) -> anyhow::Result<()> {
    varmap.save(path)?;
    Ok(())
}

/// Load checkpoint
pub fn load_checkpoint(path: &Path, _device: &Device) -> anyhow::Result<VarMap> {
    let mut varmap = VarMap::new();
    varmap.load(path)?;
    Ok(varmap)
}

/// Create sample vocabulary for testing
pub fn create_sample_vocabulary() -> Vec<String> {
    vec![
        // Basic nouns
        "cat", "dog", "sun", "moon", "tree", "river", "mountain", "ocean",
        // Verbs
        "run", "walk", "think", "create", "destroy", "build", "dance", "sing",
        // Adjectives
        "happy", "sad", "fast", "slow", "bright", "dark", "hot", "cold",
        // Polysemy examples (Whitepaper II4.4)
        "bank", // finance vs. river
        "bat",  // animal vs. sports
        "spring", // season vs. water source
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_creation() {
        let vocab = create_sample_vocabulary();
        let dataset = AlignmentDataset::from_vocabulary(&vocab, 10, 768).unwrap();
        
        assert_eq!(dataset.waves.len(), vocab.len());
        assert_eq!(dataset.embeddings.len(), vocab.len());
        assert_eq!(dataset.waves[0].len(), 10);
        assert_eq!(dataset.embeddings[0].len(), 768);
    }

    #[test]
    fn test_batch_iteration() {
        let vocab = create_sample_vocabulary();
        let dataset = AlignmentDataset::from_vocabulary(&vocab, 10, 768).unwrap();
        
        let batch_size = 4;
        let num_batches = dataset.num_batches(batch_size);
        
        for i in 0..num_batches {
            let batch = dataset.get_batch(i, batch_size).unwrap();
            assert!(batch.waves.len() <= batch_size);
        }
    }

    #[test]
    fn test_deterministic_embeddings() {
        // Same word should give same embedding
        let emb1 = AlignmentDataset::get_qwen_embedding("cat", 768).unwrap();
        let emb2 = AlignmentDataset::get_qwen_embedding("cat", 768).unwrap();
        
        assert_eq!(emb1, emb2);
    }

    #[test]
    fn test_training_single_epoch() {
        let vocab = vec!["test".to_string()];
        let dataset = AlignmentDataset::from_vocabulary(&vocab, 5, 128).unwrap();
        
        let config = TrainingConfig {
            num_epochs: 1,
            batch_size: 1,
            ..Default::default()
        };
        
        let result = train_alignment_layer(config, dataset, None);
        if let Err(ref e) = result {
            panic!("Training failed with error: {:?}", e);
        }
        assert!(result.is_ok());
    }
}













