use candle_core::{Device, Result, Tensor, Module};
use candle_nn::{Linear, VarBuilder};
use std::collections::HashMap;
use std::sync::Arc;
use crate::models::qwen_custom::{UnifiedLinear, QRmsNorm};

/// J-Lens Readout integration
/// Projects the real-valued residual stream h_l into vocabulary space logits
pub struct JLensReadout {
    jacobian_matrices: HashMap<usize, Linear>,
    workspace_layers: std::ops::Range<usize>,
    lm_head: Arc<UnifiedLinear>,
    ln_f: Arc<QRmsNorm>,
    /// When true, only the LAST token of the chunk is projected through lm_head.
    /// Runtime generation only consumes the last-token readout, and projecting the
    /// full sequence would allocate [1, seq, vocab] logits per workspace layer,
    /// OOMing the 8GB VRAM during CDCL re-prefills at growing seq_len.
    /// The diagnosis path sets this to false (it needs per-position top-k accuracy).
    last_token_only: bool,
}

impl JLensReadout {
    pub fn new(
        workspace_layers: std::ops::Range<usize>,
        lm_head: Arc<UnifiedLinear>,
        ln_f: Arc<QRmsNorm>,
        device: &Device,
    ) -> Self {
        Self::new_with_mode(workspace_layers, lm_head, ln_f, device, true)
    }

    pub fn new_with_mode(
        workspace_layers: std::ops::Range<usize>,
        lm_head: Arc<UnifiedLinear>,
        ln_f: Arc<QRmsNorm>,
        device: &Device,
        last_token_only: bool,
    ) -> Self {
        let mut jacobian_matrices = HashMap::new();
        let file_path = "jlens_weights.safetensors";
        
        // Attempt to load trained Jacobian matrices from safetensors
        if std::path::Path::new(file_path).exists() {
            if let Ok(vb) = unsafe { VarBuilder::from_mmaped_safetensors(&[file_path], candle_core::DType::F32, device) } {
                let hidden_size = ln_f.weight().dim(0).unwrap_or(4096);
                
                for layer_idx in workspace_layers.clone() {
                    let pp = vb.pp(format!("layer_{}", layer_idx));
                    if let Ok(linear_layer) = candle_nn::linear(hidden_size, hidden_size, pp) {
                        // Acceptance Criteria Gate: Ensure non-zero weight norm and non-NaN
                        let weight_norm = linear_layer.weight().sqr().unwrap().sum_all().unwrap().sqrt().unwrap().to_scalar::<f32>().unwrap_or(0.0);
                        if weight_norm > 1e-6 {
                            jacobian_matrices.insert(layer_idx, linear_layer);
                        }
                    }
                }
                if !jacobian_matrices.is_empty() {
                    eprintln!("[J-LENS] Acceptance Criteria Gate: PASSED (Loaded {} verified tuned probes)", jacobian_matrices.len());
                }
            }
        }

        Self {
            jacobian_matrices,
            workspace_layers,
            lm_head,
            ln_f,
            last_token_only,
        }
    }

    /// Identity-mode readout: never loads `jlens_weights.safetensors`, so every
    /// layer uses the raw hidden state (runtime fallback behavior). Used by the
    /// diagnosis to establish the baseline the trained readout must beat.
    pub fn new_identity(
        workspace_layers: std::ops::Range<usize>,
        lm_head: Arc<UnifiedLinear>,
        ln_f: Arc<QRmsNorm>,
        device: &Device,
        last_token_only: bool,
    ) -> Self {
        Self {
            jacobian_matrices: HashMap::new(),
            workspace_layers,
            lm_head,
            ln_f,
            last_token_only,
        }
    }

    pub fn is_workspace_layer(&self, layer_idx: usize) -> bool {
        self.workspace_layers.contains(&layer_idx)
    }

    /// Returns the contiguous span of layers that have trained `layer_{i}.weight`
    /// tensors in `jlens_weights.safetensors` (if the file exists). Used by the
    /// runtime to replace the boot-prompt diagnosis when Tuned-Lens weights exist.
    pub fn trained_layers(device: &Device) -> Option<std::ops::Range<usize>> {
        let file_path = "jlens_weights.safetensors";
        if !std::path::Path::new(file_path).exists() {
            return None;
        }
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[file_path], candle_core::DType::F32, device)
        }
        .ok()?;
        let found: Vec<usize> = (0..128usize)
            .filter(|i| vb.contains_tensor(&format!("layer_{}.weight", i)))
            .collect();
        if found.is_empty() {
            return None;
        }
        Some(*found.first()?..(*found.last()? + 1))
    }

    /// Computes lens(h_l) = lm_head(ln_f(J_l * h_l + b_l))
    /// When `last_token_only` is true and the chunk spans multiple positions, only
    /// the final position is projected. This keeps the runtime allocation at
    /// [1, 1, vocab] per layer instead of [1, seq, vocab], which OOMs the 8GB VRAM
    /// during CDCL re-prefills at growing seq_len.
    pub fn compute_readout(&self, h_l: &Tensor, layer_idx: usize) -> Result<Tensor> {
        let h_l = if self.last_token_only && h_l.dim(1)? > 1 {
            h_l.narrow(1, h_l.dim(1)? - 1, 1)?
        } else {
            h_l.clone()
        };

        let h_proj = if let Some(j_l) = self.jacobian_matrices.get(&layer_idx) {
            // Apply Affine Jacobian Projection (J_l * h_l + b_l)
            j_l.forward(&h_l)?
        } else {
            // Fallback: No trained matrix available, just map directly
            h_l
        };
        
        // Project to vocabulary space logits!
        let normalized = self.ln_f.forward(&h_proj)?;
        let logits = self.lm_head.forward(&normalized)?;
        Ok(logits)
    }
}
