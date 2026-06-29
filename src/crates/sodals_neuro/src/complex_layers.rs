use candle_core::{Tensor, Result};
use candle_nn::{Linear, Module};
use crate::complex_tensor::ComplexState;

/// Complex-Valued Linear Layer (Phase-Modulated Attention Projection)
/// Implements Euler-accurate cross-multiplication for complex projections.
pub struct ComplexLinear {
    pub real_weight: Linear, // A matrix (Identity Projection)
    pub imag_weight: Linear, // B matrix (Momentum/Phase Projection)
}

impl ComplexLinear {
    pub fn new(real_weight: Linear, imag_weight: Linear) -> Self {
        Self {
            real_weight,
            imag_weight,
        }
    }

    /// Precise Complex Matrix Projection:
    /// Z_out = (X * A^T - Y * B^T + bias_real) + i(Y * A^T + X * B^T + bias_imag)
    /// This formulation ensures destructive interference in the real component.
    pub fn forward(&self, input: &ComplexState) -> Result<ComplexState> {
        let x = &input.real;
        let y = &input.imag;

        // Extract raw weights for cross-multiplication
        let a_w = self.real_weight.weight();
        let b_w = self.imag_weight.weight();

        // Real component: (X * A^T + bias_real) - (Y * B^T)
        // We use the full linear.forward(x) to include the real bias correctly.
        let x_a_bias = self.real_weight.forward(x)?;
        let y_b = y.matmul(&b_w.t()?)?;
        let real_out = (x_a_bias - y_b)?;

        // Imaginary component: (Y * A^T) + (X * B^T + bias_imag)
        // We use the full linear.forward(x) on the imaginary weight to include the imag bias.
        let y_a = y.matmul(&a_w.t()?)?;
        let x_b_bias = self.imag_weight.forward(x)?;
        let imag_out = (y_a + x_b_bias)?;

        Ok(ComplexState {
            real: real_out,
            imag: imag_out,
        })
    }
}

/// Holographic Attention Layer
/// Replaces standard dot-product attention with Complex-Valued Wave Resonance.
/// Attention is only granted when semantic identity (Real) and phase momentum (Imag) are aligned.
pub struct HolographicAttention {
    pub q_proj: ComplexLinear,
    pub k_proj: ComplexLinear,
    pub v_proj: ComplexLinear,
    pub o_proj: ComplexLinear,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl HolographicAttention {
    /// Executes the Wave Resonance forward pass.
    /// Logits are computed as the Real component of the complex conjugate dot product.
    pub fn forward(&self, input: &ComplexState, physics_mask: Option<&Tensor>) -> Result<ComplexState> {
        let (b_size, seq_len, _) = input.real.dims3()?;
        let scale = (self.head_dim as f64).sqrt();

        // 1. Wave Projection: Project complex inputs into Q, K, V space.
        let q = self.q_proj.forward(input)?;
        let k = self.k_proj.forward(input)?;
        let v = self.v_proj.forward(input)?;

        // 2. Multi-Head Reshape: (B, S, H, D) and Transpose to (B, H, S, D)
        let q_r = q.real.reshape((b_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let q_i = q.imag.reshape((b_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k_r = k.real.reshape((b_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k_i = k.imag.reshape((b_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v_r = v.real.reshape((b_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v_i = v.imag.reshape((b_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        // 3. Holographic Resonance (Logits): Real(Q * K*)
        // Resonance = Q_real * K_real^T + Q_imag * K_imag^T
        let r1 = q_r.matmul(&k_r.transpose(2, 3)?)?;
        let r2 = q_i.matmul(&k_i.transpose(2, 3)?)?;
        let mut logits = ((r1 + r2)? / scale)?;

        // 4. Physics Injection: Apply external Wronskian/GPR impedance mask.
        if let Some(mask) = physics_mask {
            logits = logits.broadcast_add(mask)?;
        }

        // Softmax normalization over sequence dimension
        let attn_weights = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;

        // 5. Apply to Value Wave: (Weights * V_real) + i(Weights * V_imag)
        let out_r = attn_weights.matmul(&v_r)?;
        let out_i = attn_weights.matmul(&v_i)?;

        // 6. Output Projection: Reshape back to (B, S, Hidden) and pass through O-Proj.
        let out_r = out_r.transpose(1, 2)?.reshape((b_size, seq_len, self.num_heads * self.head_dim))?;
        let out_i = out_i.transpose(1, 2)?.reshape((b_size, seq_len, self.num_heads * self.head_dim))?;

        let projected_state = ComplexState { real: out_r, imag: out_i };
        self.o_proj.forward(&projected_state)
    }
}
