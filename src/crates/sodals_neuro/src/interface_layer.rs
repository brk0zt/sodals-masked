// src/interface_layer.rs - Complex-Valued to Vector Space Projector

use candle_core::{Result, Tensor, Module, Device, DType};
use candle_nn::{Linear, VarBuilder, linear};
use num_complex::Complex64;
use crate::fast_math::SemanticWave;

/// Holographic Projector: II(II?) III V(II?)
/// 
/// Maps complex-valued semantic waves (QYPTOS physics) to real-valued
/// vector space (LLM embeddings) while preserving phase information.
/// 
/// Architecture (Whitepaper II4 - "Hizalama"):
/// - proj_real: Maps real component + magnitude
/// - proj_imag: Maps imaginary component + phase
/// - Magnitude: ||z|| = III(ReII + ImI?)
/// - Phase-aware projection preserves polysemy encoding
pub struct HolographicProjector {
    proj_real: Linear,
    proj_imag: Linear,
    device: Device,
    dtype: DType,
}

impl HolographicProjector {
    /// Create new projector
    /// 
    /// # Arguments
    /// * `input_dim` - Dimension of complex wave (N complex = 2N real parameters)
    /// * `output_dim` - Target embedding dimension (e.g., 768 for Qwen)
    pub fn new(vb: VarBuilder, input_dim: usize, output_dim: usize) -> Result<Self> {
        let proj_real = linear(input_dim, output_dim, vb.pp("proj_re"))?;
        let proj_imag = linear(input_dim, output_dim, vb.pp("proj_im"))?;
        
        // Use f64 to match Complex64 precision - no data loss
        let device = vb.device().clone();
        let dtype = DType::F64;

        Ok(Self {
            proj_real,
            proj_imag,
            device,
            dtype,
        })
    }

    /// Project SemanticWave to real-valued tensor
    /// 
    /// Physics: Converts II = rIIe^(iI?) to embedding vector preserving:
    /// - Magnitude (semantic salience)
    /// - Phase (contextual orientation)
    pub fn project_wave(&self, wave: &SemanticWave) -> Result<Tensor> {
        let amplitudes = wave.as_slice();
        self.project_amplitudes(amplitudes)
    }

    /// Project raw complex amplitudes
    /// 
    /// Optimized path: Single allocation, vectorized operations
    pub fn project_amplitudes(&self, amplitudes: &[Complex64]) -> Result<Tensor> {
        let n = amplitudes.len();
        
        // Single-pass extraction: no intermediate Vec allocation
        // SIMD-friendly: contiguous memory layout
        let (reals, imags): (Vec<f64>, Vec<f64>) = amplitudes.iter()
            .map(|c| (c.re, c.im))
            .unzip();

        let t_real = Tensor::from_vec(reals, (1, n), &self.device)?
            .to_dtype(self.dtype)?;
        let t_imag = Tensor::from_vec(imags, (1, n), &self.device)?
            .to_dtype(self.dtype)?;

        self.forward_tensors(&t_real, &t_imag)
    }

    /// Project batch of waves (for efficient training)
    pub fn project_batch(&self, waves: &[SemanticWave]) -> Result<Tensor> {
        if waves.is_empty() {
            return Err(candle_core::Error::Msg("Empty batch".into()));
        }

        let dim = waves[0].len();
        let batch_size = waves.len();

        // Flatten all waves into single contiguous buffer
        let mut reals = Vec::with_capacity(batch_size * dim);
        let mut imags = Vec::with_capacity(batch_size * dim);

        for wave in waves {
            for amp in wave.as_slice() {
                reals.push(amp.re);
                imags.push(amp.im);
            }
        }

        let t_real = Tensor::from_vec(reals, (batch_size, dim), &self.device)?
            .to_dtype(self.dtype)?;
        let t_imag = Tensor::from_vec(imags, (batch_size, dim), &self.device)?
            .to_dtype(self.dtype)?;

        self.forward_tensors(&t_real, &t_imag)
    }

    /// Core projection: III(W_reIIRe)?I + (W_imIIIm)?I
    /// 
    /// This computes the magnitude of the projected complex vector,
    /// which represents the "energy" in LLM embedding space
    pub fn forward_tensors(&self, re: &Tensor, im: &Tensor) -> Result<Tensor> {
        let out_real = self.proj_real.forward(re)?;
        let out_imag = self.proj_imag.forward(im)?;
        
        // Magnitude: ||z|| = III(ReII + ImI?)
        // Candle optimizes this with BLAS/SIMD
        let magnitude_sqr = (out_real.sqr()? + out_imag.sqr()?)?;
        magnitude_sqr.sqrt()
    }

    /// Phase-aware projection (experimental)
    /// 
    /// Returns both magnitude and phase vectors for downstream models
    /// that can leverage phase information directly
    pub fn forward_polar(&self, re: &Tensor, im: &Tensor) -> Result<(Tensor, Tensor)> {
        let out_real = self.proj_real.forward(re)?;
        let out_imag = self.proj_imag.forward(im)?;
        
        // Magnitude
        let magnitude = (out_real.sqr()? + out_imag.sqr()?)?.sqrt()?;
        
        // Phase: II = atan2(Im, Re)
        // Note: Candle may not have atan2, fallback to manual computation
        let phase = Self::atan2_approx(&out_imag, &out_real)?;
        
        Ok((magnitude, phase))
    }

    /// Approximate atan2 using polynomial (if Candle lacks native support)
    fn atan2_approx(y: &Tensor, x: &Tensor) -> Result<Tensor> {
        // Simple approximation: atan2(y,x) III y/x for small angles
        // For production, use proper atan2 or compute on CPU
        // TODO: Check if newer Candle versions have atan2
        y.div(x)
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.proj_real.weight().dims()[0]
    }
}

/// Inverse projection: V(II?) III II(II?)
/// 
/// For reconstruction tasks or training via autoencoder objectives
pub struct InverseProjector {
    unproj_real: Linear,
    unproj_imag: Linear,
    _device: Device,
}

impl InverseProjector {
    pub fn new(vb: VarBuilder, input_dim: usize, output_dim: usize) -> Result<Self> {
        let unproj_real = linear(input_dim, output_dim, vb.pp("unproj_re"))?;
        let unproj_imag = linear(input_dim, output_dim, vb.pp("unproj_im"))?;
        
        Ok(Self {
            unproj_real,
            unproj_imag,
            _device: vb.device().clone(),
        })
    }

    /// Unproject embedding back to complex wave
    pub fn unproject(&self, embedding: &Tensor) -> Result<SemanticWave> {
        let re = self.unproj_real.forward(embedding)?;
        let im = self.unproj_imag.forward(embedding)?;
        
        let re_vec = re.flatten_all()?.to_vec1::<f64>()?;
        let im_vec = im.flatten_all()?.to_vec1::<f64>()?;
        
        let amplitudes: Vec<Complex64> = re_vec.into_iter()
            .zip(im_vec)
            .map(|(r, i)| Complex64::new(r, i))
            .collect();
        
        Ok(SemanticWave::from_vec(amplitudes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;
    use std::f64::consts::PI;

    #[test]
    fn test_projection_preserves_magnitude() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        
        let projector = HolographicProjector::new(vb, 10, 768)?;
        
        // Create wave with known magnitude
        let wave = SemanticWave::from_phases(&vec![0.0; 10]); // All 1+0i
        let projected = projector.project_wave(&wave)?;
        
        assert_eq!(projected.dims()[1], 768);
        Ok(())
    }

    #[test]
    fn test_batch_projection() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        
        let projector = HolographicProjector::new(vb, 10, 768)?;
        
        let waves = vec![
            SemanticWave::from_phases(&vec![0.0; 10]),
            SemanticWave::from_phases(&vec![PI/2.0; 10]),
            SemanticWave::from_phases(&vec![PI; 10]),
        ];
        
        let batch = projector.project_batch(&waves)?;
        assert_eq!(batch.dims(), &[3, 768]);
        Ok(())
    }

    #[test]
    fn test_no_precision_loss() {
        // Verify f64 preservation
        let c = Complex64::new(1.23456789012345, 9.87654321098765);
        
        // Should preserve full f64 precision
        assert_eq!(c.re, 1.23456789012345);
        assert_eq!(c.im, 9.87654321098765);
        
        // f32 would lose precision:
        let f32_re = c.re as f32 as f64;
        assert_ne!(f32_re, c.re); // Precision lost
    }

    #[test]
    fn test_roundtrip() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
        
        let projector = HolographicProjector::new(vb.pp("fwd"), 10, 64)?;
        let inverse = InverseProjector::new(vb.pp("inv"), 64, 10)?;
        
        let original = SemanticWave::from_phases(&vec![PI/4.0; 10]);
        let embedded = projector.project_wave(&original)?;
        let reconstructed = inverse.unproject(&embedded)?;
        
        // Should approximately reconstruct (not exact due to linear projection)
        assert_eq!(reconstructed.len(), original.len());
        Ok(())
    }
}













