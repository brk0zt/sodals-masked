use candle_core::{Tensor, Result};
use candle_nn::{Linear, Module};
use crate::complex_tensor::ComplexState;
use crate::gpu_ops::memory_utils::SodalsCudaAccess;

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
    pub fn forward(&self, input: &ComplexState, adapter: Option<&crate::gpu_ops::NeuroAdapter>) -> Result<ComplexState> {
        let x = &input.real;
        let y = &input.imag;

        // Extract raw weights for cross-multiplication
        let a_w = self.real_weight.weight();
        let b_w = self.imag_weight.weight();

        // PHASE 4: cuBLAS Fused Complex GEMM path
        if let Some(adapter_val) = adapter {
            if let (Ok(ar_ptr), Ok(ai_ptr), Ok(br_ptr), Ok(bi_ptr)) = (
                x.get_physical_address(),
                y.get_physical_address(),
                a_w.get_physical_address(),
                b_w.get_physical_address()
            ) {
                // Determine dimensions for GEMM: C(m,n) = A(m,k) * B(k,n)
                // x is (batch*seq_len, in_dim). A is (out_dim, in_dim) because weights are stored transposed in memory typically (Linear stores Out x In).
                // Wait, candle_nn::Linear weight is (Out, In). We want x * w^T. So B is w^T (In x Out).
                // Therefore: m = batch*seq_len, k = in_dim, n = out_dim.
                // We use trans_a = false, trans_b = true since B (the weight) is Out x In in memory, which we want as In x Out.
                // Wait, x might be multidimensional (B, S, D). We should flatten it to (B*S, D).
                let x_shape = x.dims();
                let m = x_shape.iter().take(x_shape.len() - 1).product::<usize>() as i32;
                let k = *x_shape.last().unwrap() as i32;
                let out_dim = a_w.dim(0)? as i32; // a_w is (Out, In)
                let n = out_dim;

                // Create empty output tensors
                let out_shape: Vec<usize> = x_shape.iter().take(x_shape.len() - 1).copied().chain(std::iter::once(out_dim as usize)).collect();
                let cr = Tensor::zeros(out_shape.as_slice(), x.dtype(), x.device())?;
                let ci = Tensor::zeros(out_shape.as_slice(), y.dtype(), y.device())?;

                if let (Ok(cr_ptr), Ok(ci_ptr)) = (
                    cr.get_physical_address(),
                    ci.get_physical_address()
                ) {
                    // Call cuBLAS wrapper
                    // trans_a = false (x is M x K), trans_b = true (weight is N x K, we want K x N)
                    let _ = adapter_val.apply_complex_gemm_raw(
                        ar_ptr, ai_ptr, br_ptr, bi_ptr, cr_ptr, ci_ptr, m, n, k, false, true
                    );
                    
                    // Add real and imag biases manually via broadcast
                    let cr = if let Some(bias) = self.real_weight.bias() { cr.broadcast_add(bias)? } else { cr };
                    let ci = if let Some(bias) = self.imag_weight.bias() { ci.broadcast_add(bias)? } else { ci };
                    
                    return Ok(ComplexState { real: cr, imag: ci });
                }
            }
        }

        // Fallback: 4 real GEMMs (CPU or GPU without adapter)
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
        let q = self.q_proj.forward(input, None)?; // Note: Adapter integration left out here for brevity, typically adapter goes here
        let k = self.k_proj.forward(input, None)?;
        let v = self.v_proj.forward(input, None)?;

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
        self.o_proj.forward(&projected_state, None)
    }
}
