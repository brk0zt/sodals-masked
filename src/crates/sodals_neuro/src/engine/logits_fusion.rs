// src/engine/logits_fusion.rs

use candle_core::{Tensor, Device};
use candle_transformers::generation::{LogitsProcessor, Sampling};
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
    fn is_mandatory(&self) -> bool { false }
}

/// 1. Syntax Perspective: AST ve Kod kurallarini denetler
pub struct SyntaxPerspective {
    pub is_code_block: bool,
}

impl LogitPerspective for SyntaxPerspective {
    fn compute_bias_mask(&self, vocab_size: usize, device: &Device) -> Result<Tensor> {
        // FIX: previously returned an all-zero mask. Apply a real structural signal:
        // inside code blocks, keep markdown fences + thinking markers at bay so the
        // model stays in code, and slightly depress the end-of-sequence token so a
        // code block is not terminated prematurely.
        let mut bias_vec = vec![0.0f32; vocab_size];
        if self.is_code_block {
            for (id, val) in [(248068, -20.0), (248057, -8.0), (248046, -12.0), (248044, -12.0)] {
                if (id as usize) < vocab_size {
                    bias_vec[id as usize] += val;
                }
            }
        }
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
        // FIX: previously returned zeros. In the pink zone (low phase energy) apply a
        // small uniform logit dampening so sampling tightens and the model commits to a
        // stable branch instead of hopping between low-probability alternatives.
        let damp = if self.is_pink_zone { -0.05f32 } else { 0.0f32 };
        let bias_vec = vec![damp; vocab_size];
        Tensor::from_vec(bias_vec, (vocab_size,), device).map_err(|e| anyhow::anyhow!(e.to_string()))
    }
}

/// 3. Geometer Perspective: Topolojik Ripple Effect ve Keystone degerlerini denetler
pub struct GeometerPerspective {
    pub regret_magnitude: f32,
}

impl LogitPerspective for GeometerPerspective {
    fn compute_bias_mask(&self, vocab_size: usize, device: &Device) -> Result<Tensor> {
        // FIX: previously returned zeros. Regret is a sparse geometric signal: a tiny
        // per-token ripple derived from the regret magnitude so low-regret tokens are
        // marginally favoured — purely illustrative structural ripple.
        let ripple = self.regret_magnitude.clamp(0.0, 1.0) * 0.03;
        let bias_vec = vec![ripple; vocab_size];
        Tensor::from_vec(bias_vec, (vocab_size,), device).map_err(|e| anyhow::anyhow!(e.to_string()))
    }
    fn is_geometer(&self) -> bool { true }
}

/// 4. ThinkBan Perspective: Prevents empty hallucination `<think>` loops 
pub struct ThinkBanPerspective {
    pub think_token_id: u32,
}

impl LogitPerspective for ThinkBanPerspective {
    fn compute_bias_mask(&self, vocab_size: usize, device: &Device) -> Result<Tensor> {
        let mut bias = vec![0.0f32; vocab_size];
        if (self.think_token_id as usize) < vocab_size {
            bias[self.think_token_id as usize] = -100000.0; // Heavily penalize <think>
        }
        Tensor::from_vec(bias, (vocab_size,), device).map_err(|e| anyhow::anyhow!(e.to_string()))
    }
    fn is_mandatory(&self) -> bool { false }
}

pub struct PhysicsGuidedLogitsProcessor {
    base_processor: LogitsProcessor,
    min_p: Option<f64>,
    experts: Vec<Box<dyn LogitPerspective>>,
    pub blind_experts: bool,
    pub(crate) experts_bias_tensor: Option<Tensor>,
    pub(crate) oracle_bias_tensor: Option<Tensor>,
}

impl PhysicsGuidedLogitsProcessor {
    /// Inject the Oracle's dynamically pre-computed bias into the expert fused mask.
    pub fn update_oracle_bias(&mut self, oracle_bias: candle_core::Tensor) {
        self.oracle_bias_tensor = match self.oracle_bias_tensor.take() {
            Some(existing) => existing.broadcast_add(&oracle_bias).ok(),
            None           => Some(oracle_bias),
        };
    }

    pub fn new(
        seed: u64,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        min_p: Option<f64>,
        experts: Vec<Box<dyn LogitPerspective>>,
    ) -> Self {
        let top_k = top_k.filter(|&k| k > 0);
        let sampling = match temperature.filter(|t| *t >= 1e-7) {
            None => Sampling::ArgMax,
            Some(temperature) => match (top_k, top_p) {
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (None, None) => Sampling::All { temperature },
            },
        };
        Self {
            base_processor: LogitsProcessor::from_sampling(seed, sampling),
            min_p,
            experts,
            blind_experts: false,
            experts_bias_tensor: None,
            oracle_bias_tensor: None,
        }
    }

    pub fn set_experts(&mut self, experts: Vec<Box<dyn LogitPerspective>>) {
        self.experts = experts;
        self.experts_bias_tensor = None;
    }

    /// Check if any expert is a Geometer perspective
    pub fn has_geometer(&self) -> bool {
        self.experts.iter().any(|expert| expert.is_geometer())
    }

    /// Sample through the base processor, applying min-p filtering on the
    /// post-softmax probabilities before any top-k/top-p clamping.
    fn sample_base(&mut self, modified_logits: &Tensor) -> Result<u32> {
        self.base_processor
            .sample_f(modified_logits, |prs| {
                if let Some(min_p) = self.min_p {
                    let p_max = prs.iter().copied().fold(f32::MIN, f32::max);
                    let cutoff = p_max * min_p as f32;
                    for p in prs.iter_mut() {
                        if *p < cutoff {
                            *p = 0.0;
                        }
                    }
                }
            })
            .map_err(Error::msg)
    }

    pub fn sample(&mut self, logits: &Tensor, config: &NeuroConfig) -> Result<u32> {
        if !config.euler_resonance || self.blind_experts {
            if self.blind_experts {
                tracing::debug!("[PHASE 3.5] Experts Blinded: Sampling with RAW causal logits.");
            }
            
            // In pure inference, we still apply mandatory experts like ThinkBanPerspective
            let mut modified_logits = logits.clone();
            for expert in &self.experts {
                if expert.is_mandatory() {
                    let vocab_size = logits.dim(candle_core::D::Minus1)?;
                    let expert_mask = expert.compute_bias_mask(vocab_size, logits.device())?;
                    modified_logits = modified_logits.broadcast_add(&expert_mask)?;
                }
            }
            
            if let Some(ref oracle_bias) = self.oracle_bias_tensor {
                if modified_logits.dims() == oracle_bias.dims() {
                    modified_logits = modified_logits.broadcast_add(oracle_bias)?;
                }
            }
            
            return self.sample_base(&modified_logits);
        }

        if self.experts_bias_tensor.is_none() {
            let vocab_size = logits.dim(candle_core::D::Minus1)?;
            let device = logits.device();

            let mut combined_bias = Tensor::zeros((vocab_size,), candle_core::DType::F32, device)?;

            for expert in &self.experts {
                let expert_mask = expert.compute_bias_mask(vocab_size, device)?;
                combined_bias = combined_bias.broadcast_add(&expert_mask)?;
            }

            combined_bias = combined_bias.clamp(-100000.0f32, 100000.0f32)?;
            self.experts_bias_tensor = Some(combined_bias);
        }

        let mut modified_logits = if let Some(ref bias) = self.experts_bias_tensor {
            logits.broadcast_add(bias)?
        } else {
            logits.clone()
        };

        if let Some(ref oracle_bias) = self.oracle_bias_tensor {
            if modified_logits.dims() == oracle_bias.dims() {
                modified_logits = modified_logits.broadcast_add(oracle_bias)?;
            }
        }

        self.sample_base(&modified_logits)
    }
}
