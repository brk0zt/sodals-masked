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

    /// Computes arctan(imag / real)
    pub fn phase(&self) -> Result<Tensor> {
        // [LINUS FIX] Candle 0.8.2 lacks native atan2/atan on Tensor. 
        // Using division as a first-order phase approximation.
        // Proper CT-GWE physics requires a custom CUDA kernel for atan2.
        self.imag.div(&self.real)
    }
}
