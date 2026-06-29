// src/engine/logits_fusion.rs

use candle_core::{Tensor, Device};
use candle_transformers::generation::LogitsProcessor;
use anyhow::{Result, Error};
use tracing::{debug, trace, info, warn};
use crate::engine::chimera_router::NeuroConfig;

pub struct RawToken {
    pub text: String,
    pub logits: Vec<f32>,
}

/// YieldedToken: Rich token output with semantic context
/// Enables NeuroKernel to make steering decisions with full token information
#[derive(Clone, Debug)]
pub struct YieldedToken {
    pub text: String,
    pub token_id: u32,
    pub logits: Option<Vec<f32>>,        // Raw logits before sampling (for analysis)
    pub hidden_state: Option<Vec<f32>>,  // Hidden layer activations (for steering feedback)
    pub confidence: f32,                  // Model's confidence in this token (0.0-1.0)
}

// ============================================================================
// LOGIT-LEVEL MIXTURE OF EXPERTS (MoE) ARCHITECTURE
// ============================================================================

pub trait LogitPerspective: Send + Sync {
    fn compute_bias_mask(&self, vocab_size: usize, device: &Device) -> Result<Tensor>;
    fn is_geometer(&self) -> bool { false }
}

/// 1. Syntax Perspective: AST ve Kod kurallarini denetler
pub struct SyntaxPerspective {
    pub is_code_block: bool,
}

impl LogitPerspective for SyntaxPerspective {
    fn compute_bias_mask(&self, vocab_size: usize, device: &Device) -> Result<Tensor> {
        let bias_vec = vec![0.0f32; vocab_size];
        Tensor::from_vec(bias_vec, (vocab_size,), device).map_err(|e| anyhow::anyhow!(e.to_string()))
    }
}

/// 2. Causal Perspective: Semantic Divergence ve Enerji stabilitesini denetler
pub struct CausalPerspective {
    pub energy: f32,
    pub is_pink_zone: bool,
}

impl LogitPerspective for CausalPerspective {
    fn compute_bias_mask(&self, vocab_size: usize, device: &Device) -> Result<Tensor> {
        Ok(Tensor::zeros((vocab_size,), candle_core::DType::F32, device)?)
    }
}

/// 3. Geometer Perspective: Topolojik Ripple Effect ve Keystone degerlerini denetler
pub struct GeometerPerspective {
    pub regret_magnitude: f32,
}

impl LogitPerspective for GeometerPerspective {
    fn compute_bias_mask(&self, vocab_size: usize, device: &Device) -> Result<Tensor> {
        Ok(Tensor::zeros((vocab_size,), candle_core::DType::F32, device)?)
    }
    fn is_geometer(&self) -> bool { true }
}

pub struct PhysicsGuidedLogitsProcessor {
    base_processor: LogitsProcessor,
    experts: Vec<Box<dyn LogitPerspective>>,
    pub blind_experts: bool,
    pub(crate) fused_bias_tensor: Option<Tensor>,
}

impl PhysicsGuidedLogitsProcessor {
    /// Inject the Oracle's dynamically pre-computed bias into the expert fused mask.
    pub fn update_oracle_bias(&mut self, oracle_bias: candle_core::Tensor) {
        self.fused_bias_tensor = match self.fused_bias_tensor.take() {
            Some(existing) => existing.broadcast_add(&oracle_bias).ok(),
            None           => Some(oracle_bias),
        };
    }

    pub fn new(
        seed: u64,
        temperature: Option<f64>,
        top_p: Option<f64>,
        experts: Vec<Box<dyn LogitPerspective>>,
    ) -> Self {
        Self {
            base_processor: LogitsProcessor::new(seed, temperature, top_p),
            experts,
            blind_experts: false,
            fused_bias_tensor: None,
        }
    }

    pub fn set_experts(&mut self, experts: Vec<Box<dyn LogitPerspective>>) {
        self.experts = experts;
        self.fused_bias_tensor = None;
    }

    /// Check if any expert is a Geometer perspective
    pub fn has_geometer(&self) -> bool {
        self.experts.iter().any(|expert| expert.is_geometer())
    }

    pub fn sample(&mut self, logits: &Tensor, config: &NeuroConfig) -> Result<u32> {
        if !config.euler_resonance || self.blind_experts {
            if self.blind_experts {
                tracing::debug!("[PHASE 3.5] Experts Blinded: Sampling with RAW causal logits.");
            }
            return self.base_processor.sample(logits).map_err(Error::msg);
        }

        if self.fused_bias_tensor.is_none() {
            let vocab_size = logits.dim(candle_core::D::Minus1)?;
            let device = logits.device();

            let mut combined_bias = Tensor::zeros((vocab_size,), candle_core::DType::F32, device)?;

            for expert in &self.experts {
                let expert_mask = expert.compute_bias_mask(vocab_size, device)?;
                combined_bias = combined_bias.broadcast_add(&expert_mask)?;
            }

            combined_bias = combined_bias.clamp(-100000.0f32, 100000.0f32)?;
            self.fused_bias_tensor = Some(combined_bias);
        }

        let modified_logits = if let Some(ref bias) = self.fused_bias_tensor {
            logits.broadcast_add(bias)?
        } else {
            logits.clone()
        };

        self.base_processor.sample(&modified_logits).map_err(Error::msg)
    }
}
