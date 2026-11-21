use candle_core::{Tensor, Result};

/// Complex State: standard representation for agnostic sensor ingestion.
/// Manages real and imaginary components as parallel tensors.
#[derive(Clone)]
pub struct ComplexState {
    pub real: Tensor, // Semantic Identity (Pixels, I-Wave, Token Embeddings)
    pub imag: Tensor, // Momentum/Phase (Optical Flow, Q-Wave, Attention Phase)
}

impl ComplexState {
    /// Maps In-Phase to real and Quadrature to imag from Radar signals.
    pub fn from_radar(i_signal: &Tensor, q_signal: &Tensor) -> Result<Self> {
        if i_signal.shape() != q_signal.shape() {
            return Err(candle_core::Error::ShapeMismatchBinaryOp {
                lhs: i_signal.shape().clone(),
                rhs: q_signal.shape().clone(),
                op: "from_radar",
            });
        }
        Ok(Self {
            real: i_signal.clone(),
            imag: q_signal.clone(),
        })
    }

    /// Maps static pixel data to real and frame-delta (movement) to imag from Vision signals.
    pub fn from_vision(pixels: &Tensor, optical_flow: &Tensor) -> Result<Self> {
        if pixels.shape() != optical_flow.shape() {
            return Err(candle_core::Error::ShapeMismatchBinaryOp {
                lhs: pixels.shape().clone(),
                rhs: optical_flow.shape().clone(),
                op: "from_vision",
            });
        }
        Ok(Self {
            real: pixels.clone(),
            imag: optical_flow.clone(),
        })
    }

    /// Maps NLP embeddings to real and positional/semantic phase to imag from Text signals.
    pub fn from_text(embeddings: &Tensor, phase_weights: &Tensor) -> Result<Self> {
        if embeddings.shape() != phase_weights.shape() {
            return Err(candle_core::Error::ShapeMismatchBinaryOp {
                lhs: embeddings.shape().clone(),
                rhs: phase_weights.shape().clone(),
                op: "from_text",
            });
        }
        Ok(Self {
            real: embeddings.clone(),
            imag: phase_weights.clone(),
        })
    }

    /// Computes sqrt(real^2 + imag^2)
    pub fn magnitude(&self) -> Result<Tensor> {
        let r2 = (&self.real * &self.real)?;
        let i2 = (&self.imag * &self.imag)?;
        (r2 + i2)?.sqrt()
    }

    /// Computes arctan2(imag, real)
    pub fn phase(&self) -> Result<Tensor> {
        let device = self.real.device();
        let orig_dtype = self.real.dtype();
        
        // Cast to F32 for CUDA kernel safety (prevent garbage reads on f16/bf16 tensors)
        let imag_contig = self.imag.to_dtype(candle_core::DType::F32)?.contiguous()?;
        let real_contig = self.real.to_dtype(candle_core::DType::F32)?.contiguous()?;

        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            use crate::gpu_ops::SodalsCudaAccess;
            use crate::gpu_ops::apply_atan2_cuda;
            
            let out = Tensor::zeros_like(&real_contig)?;
            let y_ptr = imag_contig.get_physical_address()?;
            let x_ptr = real_contig.get_physical_address()?;
            let out_ptr = out.get_physical_address()?;
            
            unsafe {
                apply_atan2_cuda(
                    std::ptr::null_mut(),
                    out_ptr as *mut _,
                    y_ptr as *const _,
                    x_ptr as *const _,
                    real_contig.elem_count() as i32
                );
            }
            return out.to_dtype(orig_dtype);
        }

        // CPU Fallback
        let y_vec = imag_contig.flatten_all()?.to_vec1::<f32>()?;
        let x_vec = real_contig.flatten_all()?.to_vec1::<f32>()?;
        
        let mut out_vec = Vec::with_capacity(y_vec.len());
        for i in 0..y_vec.len() {
            out_vec.push(y_vec[i].atan2(x_vec[i]));
        }
        
        Tensor::from_vec(out_vec, self.real.shape(), device)?.to_dtype(orig_dtype)
    }

    // ========================================================================
    // Fundamental Complex Arithmetic
    // ========================================================================

    /// Complex addition: (a + bi) + (c + di) = (a+c) + (b+d)i
    pub fn add(&self, other: &ComplexState) -> Result<ComplexState> {
        Ok(ComplexState {
            real: (&self.real + &other.real)?,
            imag: (&self.imag + &other.imag)?,
        })
    }

    /// Complex subtraction: (a + bi) - (c + di) = (a-c) + (b-d)i
    pub fn sub(&self, other: &ComplexState) -> Result<ComplexState> {
        Ok(ComplexState {
            real: (&self.real - &other.real)?,
            imag: (&self.imag - &other.imag)?,
        })
    }

    /// Elementwise complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    /// This is NOT matrix multiplication — it's Hadamard (elementwise) product in ℂ.
    pub fn complex_mul(&self, other: &ComplexState) -> Result<ComplexState> {
        let ac = (&self.real * &other.real)?;
        let bd = (&self.imag * &other.imag)?;
        let ad = (&self.real * &other.imag)?;
        let bc = (&self.imag * &other.real)?;
        Ok(ComplexState {
            real: (ac - bd)?,
            imag: (ad + bc)?,
        })
    }

    /// Complex conjugate: (a + bi)* = a - bi
    /// Conjugation flips the sign of the imaginary component.
    /// Essential for computing inner products ⟨z₁, z₂⟩ = z₁ · z₂*
    pub fn conjugate(&self) -> Result<ComplexState> {
        Ok(ComplexState {
            real: self.real.clone(),
            imag: self.imag.neg()?,
        })
    }

    /// Scalar multiplication: α · (a + bi) = αa + αbi
    pub fn scale(&self, alpha: f64) -> Result<ComplexState> {
        Ok(ComplexState {
            real: (&self.real * alpha)?,
            imag: (&self.imag * alpha)?,
        })
    }

    /// Squared magnitude: |z|² = Re² + Im² (avoids sqrt for efficiency)
    pub fn magnitude_squared(&self) -> Result<Tensor> {
        let r2 = (&self.real * &self.real)?;
        let i2 = (&self.imag * &self.imag)?;
        (r2 + i2)
    }

    /// Complex inner product: ⟨self, other⟩ = Σ self_i · other_i*
    /// Returns a ComplexState with the real and imaginary sums.
    /// Real part: Σ(a_i·c_i + b_i·d_i), Imag part: Σ(b_i·c_i - a_i·d_i)
    pub fn inner_product(&self, other: &ComplexState) -> Result<ComplexState> {
        let conj = other.conjugate()?;
        let product = self.complex_mul(&conj)?;
        // Sum over all dimensions except batch
        Ok(ComplexState {
            real: product.real.sum_all()?,
            imag: product.imag.sum_all()?,
        })
    }

    /// Create a zero ComplexState with the same shape and device
    pub fn zeros_like(&self) -> Result<ComplexState> {
        Ok(ComplexState {
            real: Tensor::zeros_like(&self.real)?,
            imag: Tensor::zeros_like(&self.imag)?,
        })
    }

    /// Create a ComplexState from raw real and imaginary tensors
    pub fn new(real: Tensor, imag: Tensor) -> Self {
        Self { real, imag }
    }
}
