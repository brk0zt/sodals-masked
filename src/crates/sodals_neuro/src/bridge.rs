// src/bridge.rs - Complex-to-Real Bridge with Wirtinger Calculus

use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};
use num_complex::Complex64;
use crate::fast_math::SemanticWave;

/// ModReLU Activation: f(z) = ReLU(|z| + b) * z / |z|
/// Preserves phase, modulates magnitude non-linearly.
fn mod_relu(z: Complex64, bias: f64) -> Complex64 {
    let norm = z.norm();
    if norm < 1e-10 {
        return Complex64::new(0.0, 0.0);
    }
    let scale = (norm + bias).max(0.0) / norm;
    z.scale(scale)
}

/// Bridge between QYPTOS complex physics and LLM real-valued embeddings
/// 
/// Implements Teacher-Student training (Whitepaper II4):
/// - Forward: II(II?) III V(II?) via learned linear projection
/// - Backward: Wirtinger derivatives for complex-valued gradients
pub struct SodalsBridge {
    /// Learned projection: flattened [Re, Im] III LLM embedding
    adapter_layer: Linear,
    mod_relu_bias: Tensor,
    output_dim: usize,
    device: Device,
}

impl SodalsBridge {
    /// Create new bridge
    /// 
    /// # Arguments
    /// * `input_complex_dim` - Number of QYPTOS neurons (e.g., 100)
    /// * `output_dim` - LLM embedding dimension (e.g., 4096 for Qwen)
    pub fn new(
        vb: VarBuilder,
        input_complex_dim: usize,
        output_dim: usize,
    ) -> anyhow::Result<Self> {
        // Each complex number = 2 real components (Re, Im)
        let flattened_dim = input_complex_dim * 2;
        let adapter_layer = linear(flattened_dim, output_dim, vb.pp("bridge"))?;
        let mod_relu_bias = vb.pp("mod_relu").get(input_complex_dim, "bias")?;
        let device = vb.device().clone();
        
        Ok(Self {
            adapter_layer,
            mod_relu_bias,
            output_dim,
            device,
        })
    }

    /// Forward pass: Complex wave III Real embedding
    /// 
    /// Physics proof metrics (Whitepaper II1.1, II1.2):
    /// - Mean phase: Average semantic orientation
    /// - Resonance energy: Total wave magnitude
    pub fn forward(&self, wave: &SemanticWave) -> anyhow::Result<Tensor> {
        let amplitudes = wave.as_slice();
        self.forward_raw(amplitudes)
    }

    /// Forward pass from raw complex array
    pub fn forward_raw(&self, amplitudes: &[Complex64]) -> anyhow::Result<Tensor> {
        // Physics diagnostics (keep for debugging, remove in production)
        if cfg!(debug_assertions) && !amplitudes.is_empty() {
            let mean_phase: f64 = amplitudes.iter()
                .map(|c| c.arg())
                .sum::<f64>() / amplitudes.len() as f64;
            
            let resonance_energy: f64 = amplitudes.iter()
                .map(|c| c.norm_sqr())
                .sum::<f64>() / amplitudes.len() as f64;
            
            tracing::debug!(
                "Physics: mean_phase={:.2} rad, resonance_energy={:.4}",
                mean_phase,
                resonance_energy
            );
        }

        // Apply ModReLU activation
        let biases = self.mod_relu_bias.to_vec1::<f64>()?;
        if biases.len() != amplitudes.len() {
            anyhow::bail!("ModReLU bias dimension mismatch");
        }

        // Flatten to [Re1, Im1, Re2, Im2, ...]
        // Use f64 to preserve precision (matches Complex64)
        let flattened: Vec<f64> = amplitudes.iter()
            .zip(biases.iter())
            .map(|(&z, &b)| mod_relu(z, b))
            .flat_map(|c| [c.re, c.im])
            .collect();

        // Convert to tensor (batch_size = 1)
        let input_tensor = Tensor::from_vec(
            flattened,
            (1, amplitudes.len() * 2),
            &self.device,
        )?;

        // Learned linear projection
        let output = self.adapter_layer.forward(&input_tensor)?;
        Ok(output)
    }

    /// Backward pass: Compute Wirtinger derivatives (Whitepaper II6)
    /// 
    /// Given:
    /// - dL/dEmbedding: Gradient from LLM loss
    /// - Complex state z
    /// 
    /// Returns:
    /// - IIIL/IIIz and IIIL/IIIzII (Wirtinger derivatives)
    /// 
    /// Wirtinger calculus treats z and zII as independent variables:
    /// IIIf/IIIz = (1/2)(IIIf/IIIRe - iIIIIIf/IIIIm)
    /// IIIf/IIIzII = (1/2)(IIIf/IIIRe + iIIIIIf/IIIIm)
    pub fn backward_wirtinger(
        &self,
        amplitudes: &[Complex64],
        grad_output: &Tensor,
    ) -> anyhow::Result<WirtingerGradients> {
        // 1. Compute gradients w.r.t. flattened input [Re, Im, ...]
        let grad_input = self.compute_input_gradients(grad_output)?;
        
        // 2. Reshape to (Re, Im) pairs
        let grad_vec = grad_input.to_vec1::<f64>()?;
        
        if grad_vec.len() != amplitudes.len() * 2 {
            anyhow::bail!(
                "Gradient size mismatch: expected {}, got {}",
                amplitudes.len() * 2,
                grad_vec.len()
            );
        }

        // 3. Convert to Wirtinger form
        let mut grad_z = Vec::with_capacity(amplitudes.len());
        let mut grad_z_conj = Vec::with_capacity(amplitudes.len());

        for i in 0..amplitudes.len() {
            let grad_re = grad_vec[2 * i];
            let grad_im = grad_vec[2 * i + 1];

            // Wirtinger derivatives (Whitepaper II6)
            // IIIL/IIIz = (1/2)(IIIL/IIIRe - iIIIIIL/IIIIm)
            let dz = Complex64::new(
                0.5 * grad_re,
                -0.5 * grad_im,
            );

            // IIIL/IIIzII = (1/2)(IIIL/IIIRe + iIIIIIL/IIIIm)
            let dz_conj = Complex64::new(
                0.5 * grad_re,
                0.5 * grad_im,
            );

            grad_z.push(dz);
            grad_z_conj.push(dz_conj);
        }

        Ok(WirtingerGradients {
            grad_z,
            grad_z_conj,
        })
    }

    /// Compute gradients w.r.t. input using Candle's autodiff
    /// 
    /// Note: This is a simplified version. Full implementation would:
    /// 1. Wrap forward in a differentiable function
    /// 2. Use Candle's backward() to get gradients
    /// 3. Handle batching properly
    fn compute_input_gradients(&self, _grad_output: &Tensor) -> anyhow::Result<Tensor> {
        // TODO: Implement proper autodiff when Candle supports it
        // For now, return placeholder
        
        // In production, this would be:
        // 1. Create Variable from input
        // 2. Forward pass with grad tracking
        // 3. Call backward() on loss
        // 4. Extract input.grad()
        
        anyhow::bail!("Autodiff not yet implemented - requires Candle Variable support")
    }

    /// Apply Wirtinger gradients to update complex parameters
    /// 
    /// Update rule (Whitepaper II6):
    /// z_new = z_old - IIIIIIIL/IIIzII
    /// 
    /// Note: We use IIIL/IIIzII (not IIIL/IIIz) for gradient descent
    pub fn apply_gradients(
        amplitudes: &mut [Complex64],
        gradients: &WirtingerGradients,
        learning_rate: f64,
    ) -> anyhow::Result<()> {
        if amplitudes.len() != gradients.grad_z_conj.len() {
            anyhow::bail!("Size mismatch: {} vs {}", 
                amplitudes.len(), 
                gradients.grad_z_conj.len()
            );
        }

        for (amp, grad) in amplitudes.iter_mut().zip(&gradients.grad_z_conj) {
            // Gradient descent: z -= IIIIIIIL/IIIzII
            *amp -= grad * learning_rate;
        }

        Ok(())
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Phase 24: Compress peripheral anomaly into Ghost Token tensor
    /// 
    /// Takes an anomalous SemanticWave (from detect_peripheral_anomalies)
    /// and compresses it into a dense Tensor that can be injected as ghost_prefix.
    /// 
    /// The ghost tensor mathematically alters attention trajectory without
    /// consuming extra sequence length in the KV-Cache.
    pub fn compress_anomaly_to_ghost(&self, anomaly_wave: &SemanticWave) -> anyhow::Result<Tensor> {
        // Forward the anomaly wave through the bridge to get real-valued embedding
        let ghost_embedding = self.forward(anomaly_wave)?;
        
        // Reshape to [1, 1, hidden_dim] for broadcast addition to BOS token
        // This creates the "gravitational pull" on the LLM's attention
        let ghost_tensor = ghost_embedding.reshape((1, 1, self.output_dim))?;
        
        eprintln!(
            "[GHOST COMPRESSOR] Anomaly wave ({} nodes) compressed to ghost tensor [1, 1, {}]",
            anomaly_wave.amplitudes.len(),
            self.output_dim
        );
        
        Ok(ghost_tensor)
    }
}

/// Wirtinger gradient representation
/// 
/// For a complex variable z, we compute:
/// - IIIL/IIIz: Gradient w.r.t. z
/// - IIIL/IIIzII: Gradient w.r.t. conjugate zII
/// 
/// These allow optimization in complex space without
/// breaking phase information (Whitepaper II6)
#[derive(Debug, Clone)]
pub struct WirtingerGradients {
    /// IIIL/IIIz (Holomorphic derivative)
    pub grad_z: Vec<Complex64>,
    
    /// IIIL/IIIzII (Antiholomorphic derivative)
    /// Used for gradient descent: z_new = z - IIIIIIIL/IIIzII
    pub grad_z_conj: Vec<Complex64>,
}

impl WirtingerGradients {
    /// Create from Cartesian gradients
    /// 
    /// Given IIIL/IIIRe and IIIL/IIIIm, compute Wirtinger form
    pub fn from_cartesian(grad_re: &[f64], grad_im: &[f64]) -> anyhow::Result<Self> {
        if grad_re.len() != grad_im.len() {
            anyhow::bail!("Gradient dimension mismatch");
        }

        let mut grad_z = Vec::with_capacity(grad_re.len());
        let mut grad_z_conj = Vec::with_capacity(grad_re.len());

        for (&gre, &gim) in grad_re.iter().zip(grad_im.iter()) {
            grad_z.push(Complex64::new(0.5 * gre, -0.5 * gim));
            grad_z_conj.push(Complex64::new(0.5 * gre, 0.5 * gim));
        }

        Ok(Self { grad_z, grad_z_conj })
    }

    /// Norm of gradient (for monitoring convergence)
    pub fn norm(&self) -> f64 {
        self.grad_z_conj.iter()
            .map(|g| g.norm_sqr())
            .sum::<f64>()
            .sqrt()
    }
}

/// Training utilities
pub mod training {
    use super::*;

    /// Cosine Phase Loss (Whitepaper II4.2)
    /// 
    /// L_total = L_cos + IIIIL_mag
    /// 
    /// where:
    /// - L_cos: 1 - cos_similarity (direction alignment)
    /// - L_mag: ||pred|| - ||target||)?I (magnitude match)
    pub fn cosine_phase_loss(
        pred: &Tensor,
        target: &Tensor,
        lambda: f64,
    ) -> anyhow::Result<f64> {
        // Cosine similarity: dot(a,b) / (||a||II||b||)
        let dot = (pred * target)?.sum_all()?.to_vec0::<f64>()?;
        
        let pred_norm = pred.sqr()?.sum_all()?.to_vec0::<f64>()?.sqrt();
        let target_norm = target.sqr()?.sum_all()?.to_vec0::<f64>()?.sqrt();
        
        let cos_sim = dot / (pred_norm * target_norm + 1e-8);
        let l_cos = 1.0 - cos_sim;
        
        // Magnitude loss
        let l_mag = (pred_norm - target_norm).powi(2);
        
        Ok(l_cos + lambda * l_mag)
    }

    /// Phase rotation for polysemy (Whitepaper II4.4)
    /// 
    /// Example: "Bank" context shift
    /// - Finance: II = 0II
    /// - River: II = 90II (rotate by context)
    pub fn rotate_phase(wave: &mut SemanticWave, delta_theta: f64) {
        wave.rotate_phase_mut(delta_theta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;
    use std::f64::consts::PI;

    #[test]
    fn test_wirtinger_symmetry() {
        // Wirtinger derivatives should satisfy:
        // IIIf/IIIz + IIIf/IIIzII* = IIIf/IIIRe
        let grad_re = vec![2.0, 4.0];
        let grad_im = vec![1.0, -1.0];
        
        let grads = WirtingerGradients::from_cartesian(&grad_re, &grad_im).unwrap();
        
        // Check Wirtinger identity
        for i in 0..grad_re.len() {
            let sum = grads.grad_z[i] + grads.grad_z_conj[i];
            assert!((sum.re - grad_re[i]).abs() < 1e-10);
            assert!(sum.im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_gradient_application() {
        let mut amplitudes = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
        ];
        
        let gradients = WirtingerGradients {
            grad_z: vec![Complex64::new(0.1, 0.0); 2],
            grad_z_conj: vec![Complex64::new(0.1, 0.0); 2],
        };
        
        let original = amplitudes.clone();
        SodalsBridge::apply_gradients(&mut amplitudes, &gradients, 0.1).unwrap();
        
        // Should have moved in gradient direction
        assert_ne!(amplitudes[0], original[0]);
    }

    #[test]
    fn test_forward_pass() -> anyhow::Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F64, &Device::Cpu);
        
        let bridge = SodalsBridge::new(vb, 10, 768)?;
        
        let wave = SemanticWave::from_phases(&vec![0.0; 10]);
        let embedding = bridge.forward(&wave)?;
        
        assert_eq!(embedding.dims()[1], 768);
        Ok(())
    }

    #[test]
    fn test_cosine_phase_loss() -> anyhow::Result<()> {
        let pred = Tensor::new(&[1.0f64, 0.0, 0.0], &Device::Cpu)?;
        let target = Tensor::new(&[0.0f64, 1.0, 0.0], &Device::Cpu)?;
        
        let loss = training::cosine_phase_loss(&pred, &target, 0.5)?;
        
        // Orthogonal vectors should have high loss
        assert!(loss > 0.5);
        Ok(())
    }
}













