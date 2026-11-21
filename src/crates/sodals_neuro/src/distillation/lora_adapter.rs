// src/distillation/lora_adapter.rs
//
// SplitComplex & Real LoRA Adapter for Selective Distillation.
//
// Rank decomposition: W_new = W_frozen + (alpha / rank) * (A * B)
//   - A: Frozen matrix [rank, in_dim]
//   - B: Trainable matrix [out_dim, rank]
//
// This ensures that only B is modified during training, restricting catastrophic forgetting.

use candle_core::{Tensor, Device, Result};
use candle_nn::{VarBuilder, VarMap};

#[derive(Debug, Clone)]
pub struct LoRaAdapterConfig {
    pub in_dim: usize,
    pub out_dim: usize,
    pub rank: usize,
    pub alpha: f64,
}

impl Default for LoRaAdapterConfig {
    fn default() -> Self {
        Self {
            in_dim: 4096,
            out_dim: 4096,
            rank: 8,
            alpha: 16.0,
        }
    }
}

pub struct SplitComplexLoRaAdapter {
    pub config: LoRaAdapterConfig,
    /// Frozen projection matrix A [rank, in_dim]
    pub a_frozen: Tensor,
    /// Trainable projection matrix B [out_dim, rank]
    pub b_trainable: Tensor,
    pub scaling: f64,
}

impl SplitComplexLoRaAdapter {
    pub fn new(vb: VarBuilder, config: LoRaAdapterConfig) -> Result<Self> {
        let scaling = config.alpha / config.rank as f64;
        
        // A is frozen (initialized with Gaussian / spectral weights)
        let a_frozen = Tensor::randn(0.0f32, 1.0f32, (config.rank, config.in_dim), vb.device())?;
        
        // B is trainable (initialized to zeros so adapter starts as identity / zero offset)
        let b_trainable = vb.get_with_hints(
            (config.out_dim, config.rank),
            "b_trainable",
            candle_nn::init::ZERO,
        )?;

        Ok(Self {
            config,
            a_frozen,
            b_trainable,
            scaling,
        })
    }

    /// Forward pass through the adapter: delta_x = scaling * (x * A^T * B^T)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, seq_len, in_dim]
        let a_t = self.a_frozen.t()?; // [in_dim, rank]
        let h = x.broadcast_matmul(&a_t)?; // [batch, seq_len, rank]
        let b_t = self.b_trainable.t()?; // [rank, out_dim]
        let delta = h.broadcast_matmul(&b_t)?; // [batch, seq_len, out_dim]
        
        (&delta * self.scaling)
    }

    /// Compute L2 norm of the trainable weights for drift regularization: ||B||^2
    pub fn l2_regularization_loss(&self) -> Result<Tensor> {
        self.b_trainable.sqr()?.sum_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_lora_adapter_zero_init() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = LoRaAdapterConfig {
            in_dim: 64,
            out_dim: 64,
            rank: 4,
            alpha: 8.0,
        };

        let adapter = SplitComplexLoRaAdapter::new(vb, config)?;
        let input = Tensor::ones((1, 10, 64), DType::F32, &device)?;
        let output = adapter.forward(&input)?;

        // Zero init for B means adapter initial output must be zero tensor
        let max_val: f32 = output.abs()?.max_all()?.to_scalar()?;
        assert!(max_val < 1e-6, "Initial output should be 0");
        Ok(())
    }
}
