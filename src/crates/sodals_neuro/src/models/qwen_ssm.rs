// src/models/qwen_ssm.rs
// Qwen3.5 SSM (State Space Model) Implementation
// Based on Gated DeltaNet linear state space models

use std::sync::RwLock;
use candle_core::{Device, Result, Tensor, Module};
use crate::models::qwen_custom::{UnifiedLinear, QRmsNorm};

/// SSM (State Space Model) Block for Qwen3.5
/// Implements selective state space model similar to Gated DeltaNet
pub struct QwenSsmBlock {
    /// QKV projection
    ssm_in_proj: UnifiedLinear,
    /// Gate Z projection
    attn_gate: UnifiedLinear,
    /// Conv1D for state space
    conv1d: Tensor,
    /// SSM output projection
    ssm_out_proj: UnifiedLinear,
    /// SSM normalization
    ssm_norm: Option<QRmsNorm>,
    /// SSM parameters (A, dt, B, C)
    ssm_a: Tensor,
    /// Time-step bias
    ssm_dt: Tensor,
    /// Beta weight
    ssm_beta: UnifiedLinear,
    /// Alpha weight
    ssm_alpha: UnifiedLinear,
    /// State cache: (ssm_state, conv_state)
    state_cache: RwLock<Option<(Tensor, Tensor)>>,
    /// Device
    device: Device,
}

impl QwenSsmBlock {
    pub fn new(
        ssm_in_proj: UnifiedLinear,
        attn_gate: UnifiedLinear,
        conv1d: Tensor,
        ssm_out_proj: UnifiedLinear,
        ssm_norm: Option<QRmsNorm>,
        ssm_a: Tensor,
        ssm_dt: Tensor,
        ssm_beta: UnifiedLinear,
        ssm_alpha: UnifiedLinear,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            ssm_in_proj,
            attn_gate,
            conv1d,
            ssm_out_proj,
            ssm_norm,
            ssm_a,
            ssm_dt,
            ssm_beta,
            ssm_alpha,
            state_cache: RwLock::new(None),
            device: device.clone(),
        })
    }

    pub fn clear_cache(&self) {
        if let Ok(mut guard) = self.state_cache.write() {
            *guard = None;
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let _ = (b_sz, seq_len);

        // 1. Input projection
        // x_proj shape: [b, seq_len, 8192]
        let x_proj = self.ssm_in_proj.forward(x)?;

        // 2. Convolution step
        // conv_out shape: [b, seq_len, 8192]
        let conv_out = self.apply_conv1d(&x_proj, seq_len)?;

        let q = conv_out.narrow(candle_core::D::Minus1, 0, 2048)?;
        let k = conv_out.narrow(candle_core::D::Minus1, 2048, 2048)?;
        let v = conv_out.narrow(candle_core::D::Minus1, 2048 + 2048, 4096)?;

        let q = candle_nn::ops::silu(&q)?;
        let k = candle_nn::ops::silu(&k)?;
        let v = candle_nn::ops::silu(&v)?;

        // Apply L2 Norm on q and k per head (head size 128)
        let q = self.l2_norm(&q, 128)?;
        let k = self.l2_norm(&k, 128)?;

        let gate_z = self.attn_gate.forward(x)?;
        let alpha_proj = self.ssm_alpha.forward(x)?;
        let beta_proj = self.ssm_beta.forward(x)?;

        let beta = candle_nn::ops::sigmoid(&beta_proj)?;
        let gate_z = candle_nn::ops::silu(&gate_z)?; // SiLU gate

        // 5. Compute SSM decay parameter: decay_gate = exp(-exp(ssm_a) * softplus(alpha_proj + ssm_dt))
        let a_exp = self.ssm_a.exp()?; // exp(ssm_a)
        let dt_val = alpha_proj.broadcast_add(&self.ssm_dt)?;
        let dt_softplus = self.softplus(&dt_val)?;
        let g = (a_exp.broadcast_mul(&dt_softplus)? * -1.0)?;
        let decay_gate = g.exp()?;

        // 6. Recurrent State Update (Qwen3.5 DeltaNet head structure)
        // FIX: Use dynamic head count instead of hardcoded 16/32
        // q: [b, seq_len, ssm_group_count heads, head_dim]
        // k: [b, seq_len, ssm_group_count heads, head_dim]
        // v: [b, seq_len, 2*ssm_group_count heads, head_dim]
        let q_dim = q.dims3()?.2;
        let k_dim = k.dims3()?.2;
        let v_dim = v.dims3()?.2;
        let head_dim = 128; // Qwen 3.5 head_dim from metadata
        let ssm_group_count = q_dim / head_dim; // Should be 16 for Qwen 3.5 9B
        let v_head_count = v_dim / head_dim; // Should be 32 for Qwen 3.5 9B

        let q_heads = q.reshape((b_sz, seq_len, ssm_group_count, head_dim))?;
        let k_heads = k.reshape((b_sz, seq_len, ssm_group_count, head_dim))?;
        let v_heads = v.reshape((b_sz, seq_len, v_head_count, head_dim))?;

        // Run recurrent scan to compute output y
        // y_heads shape: [b, seq_len, 32, 128]
        let y_heads = self.recurrent_scan(
            &q_heads, &k_heads, &v_heads,
            &decay_gate, &beta,
            seq_len, b_sz
        )?;

        // 7. Output Normalization and Gating
        // Apply ssm_norm to y_heads: [b, seq_len, 32, 128] -> [b, seq_len, 32, 128]
        let y_norm = if let Some(ref norm) = self.ssm_norm {
            let y_reshaped = y_heads.reshape((b_sz * seq_len * 32, 128))?;
            let y_normed = norm.forward(&y_reshaped)?;
            y_normed.reshape((b_sz, seq_len, 32, 128))?
        } else {
            y_heads
        };

        // Flatten to [b, seq_len, 4096]
        let y_flat = y_norm.reshape((b_sz, seq_len, 4096))?;

        // Apply Gate Z: y_gated = y_flat * gate_z
        let y_gated = (y_flat * gate_z)?;

        // 8. Output projection
        let ssm_out = self.ssm_out_proj.forward(&y_gated)?.to_dtype(candle_core::DType::F32)?;
        Ok(ssm_out)
    }

    /// Simplified forward pass for long sequences (prefill)
    /// Uses standard linear attention instead of recurrent DeltaNet
    fn simplified_forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden_size) = x.dims3()?;
        
        // 1. Input projection
        let x_proj = self.ssm_in_proj.forward(x)?;
        
        // 2. Gate projection
        let gate_z = self.attn_gate.forward(x)?;
        
        // 3. Split Q, K, V
        let q = x_proj.narrow(candle_core::D::Minus1, 0, 2048)?;
        let k = x_proj.narrow(candle_core::D::Minus1, 2048, 2048)?;
        let v = x_proj.narrow(candle_core::D::Minus1, 2048 + 2048, 4096)?;
        
        // 4. Apply SiLU activation
        let q = candle_nn::ops::silu(&q)?;
        let k = candle_nn::ops::silu(&k)?;
        let v = candle_nn::ops::silu(&v)?;
        
        // 5. Reshape for attention
        // FIX: Use dynamic head count instead of hardcoded 16/32
        let q_dim = q.dims3()?.2;
        let k_dim = k.dims3()?.2;
        let v_dim = v.dims3()?.2;
        let head_dim = 128; // Qwen 3.5 head_dim from metadata
        let ssm_group_count = q_dim / head_dim; // Should be 16 for Qwen 3.5 9B
        let v_head_count = v_dim / head_dim; // Should be 32 for Qwen 3.5 9B

        let q = q.reshape((b_sz, seq_len, ssm_group_count, head_dim))?.transpose(1, 2)?; // [b, ssm_group_count, seq, head_dim]
        let k = k.reshape((b_sz, seq_len, ssm_group_count, head_dim))?.transpose(1, 2)?; // [b, ssm_group_count, seq, head_dim]
        let v = v.reshape((b_sz, seq_len, v_head_count, head_dim))?.transpose(1, 2)?; // [b, v_head_count, seq, head_dim]

        // 6. Expand Q and K from ssm_group_count to v_head_count (2:1 GQA-style)
        let expansion_ratio = v_head_count / ssm_group_count; // Should be 2 for Qwen 3.5 9B
        let q = q.unsqueeze(2)?.expand((b_sz, ssm_group_count, expansion_ratio, seq_len, head_dim))?.contiguous()?.reshape((b_sz, v_head_count, seq_len, head_dim))?;
        let k = k.unsqueeze(2)?.expand((b_sz, ssm_group_count, expansion_ratio, seq_len, head_dim))?.contiguous()?.reshape((b_sz, v_head_count, seq_len, head_dim))?;

        // 7. Standard attention (no recurrence)
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?; // [b, v_head_count, seq, seq]
        let scores = (scores / (head_dim as f64).sqrt())?;

        

        // 8. Causal mask

        let mask = Tensor::tril2(seq_len, candle_core::DType::F32, &self.device)?;

        let mask = mask.unsqueeze(0)?.unsqueeze(0)?;

        let mask_neg = ((Tensor::ones_like(&mask)? - &mask)? * -1e9)?;

        let scores = scores.broadcast_add(&mask_neg)?;

        

        // 9. Softmax

        let attn_weights = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;

        

        // 10. Apply to values

        let attn_output = attn_weights.matmul(&v.contiguous()?)?; // [b, 32, seq, 128]

        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, 4096))?;

        

        // 11. Apply gate and output projection

        let gate_z = candle_nn::ops::silu(&gate_z)?;

        let y_gated = (attn_output * gate_z)?;

        self.ssm_out_proj.forward(&y_gated)
    }

    fn apply_conv1d(&self, x_proj: &Tensor, seq_len: usize) -> Result<Tensor> {



        let (b_sz, _, d_inner) = x_proj.dims3()?;

        let mut cache_guard = self.state_cache.write().unwrap();

        

        // conv1d after dequantize: [8192, 4] (GGUF transposes on load from [4, 8192])
        // narrow on dim 1 (kernel axis) to get per-step weights: each is [8192, 1]
        let w3 = self.conv1d.narrow(1, 3, 1)?; // [8192, 1] — current input weight
        let w2 = self.conv1d.narrow(1, 2, 1)?; // [8192, 1]
        let w1 = self.conv1d.narrow(1, 1, 1)?; // [8192, 1]
        let w0 = self.conv1d.narrow(1, 0, 1)?; // [8192, 1]

        if seq_len == 1 {
            let (ssm_state, conv_state) = if let Some((ref s_state, ref c_state)) = *cache_guard {
                (s_state.clone(), c_state.clone())
            } else {
                let s_state = Tensor::zeros((b_sz, 32, 128, 128), x_proj.dtype(), &self.device)?;
                let c_state = Tensor::zeros((b_sz, d_inner, 3), x_proj.dtype(), &self.device)?;
                (s_state, c_state)
            };

            let x_t = x_proj.transpose(1, 2)?; // [b, 8192, 1]
            
            let c0 = x_t.clone(); 
            let c1 = conv_state.narrow(2, 0, 1)?; 
            let c2 = conv_state.narrow(2, 1, 1)?; 
            let c3 = conv_state.narrow(2, 2, 1)?; 

            let out = (
                (c0.broadcast_mul(&w3)?) +
                (c1.broadcast_mul(&w2)?) +
                (c2.broadcast_mul(&w1)?) +
                (c3.broadcast_mul(&w0)?)
            )?; 

            let new_conv_state = Tensor::cat(&[&c0, &c1, &c2], 2)?;
            // Preserve existing ssm_state - only update conv_state
            *cache_guard = Some((ssm_state, new_conv_state));

            out.transpose(1, 2)
        } else {
            let x_t = x_proj.transpose(1, 2)?; 

            let zeros = Tensor::zeros((b_sz, d_inner, 3), x_proj.dtype(), &self.device)?;
            let padded = Tensor::cat(&[&zeros, &x_t], 2)?; 

            let s0 = padded.narrow(2, 3, seq_len)?;
            let s1 = padded.narrow(2, 2, seq_len)?;
            let s2 = padded.narrow(2, 1, seq_len)?;
            let s3 = padded.narrow(2, 0, seq_len)?;

            let out = (
                (s0.broadcast_mul(&w3)?) +
                (s1.broadcast_mul(&w2)?) +
                (s2.broadcast_mul(&w1)?) +
                (s3.broadcast_mul(&w0)?)
            )?; 

            let c0 = padded.narrow(2, seq_len + 2, 1)?; 

            let c1 = padded.narrow(2, seq_len + 1, 1)?; 

            let c2 = padded.narrow(2, seq_len, 1)?; 

            let new_conv_state = Tensor::cat(&[&c0, &c1, &c2], 2)?; 
            // Preserve existing ssm_state (will be set by recurrent_scan); use zeros for fresh prefill
            let existing_ssm = if let Some((ref s, _)) = *cache_guard {
                s.clone()
            } else {
                Tensor::zeros((b_sz, 32, 128, 128), x_proj.dtype(), &self.device)?
            };
            *cache_guard = Some((existing_ssm, new_conv_state));

            out.transpose(1, 2)
        }
    }

    fn l2_norm(&self, x: &Tensor, head_dim: usize) -> Result<Tensor> {
        let (b_sz, seq_len, total_dim) = x.dims3()?;
        let num_heads = total_dim / head_dim;
        let x_reshaped = x.reshape((b_sz, seq_len, num_heads, head_dim))?;
        
        let sq = (&x_reshaped * &x_reshaped)?;
        let sum = sq.sum_keepdim(candle_core::D::Minus1)?;
        let norm = (sum + 1e-6)?.sqrt()?;
        let out = x_reshaped.broadcast_div(&norm)?;
        
        out.reshape((b_sz, seq_len, total_dim))
    }

    fn softplus(&self, x: &Tensor) -> Result<Tensor> {
        (x.exp()? + 1.0)?.log()
    }

    fn recurrent_scan(
        &self,
        q_heads: &Tensor, 
        k_heads: &Tensor, 
        v_heads: &Tensor, 
        decay_gate: &Tensor,   
        beta: &Tensor,    
        seq_len: usize,
        b_sz: usize,
    ) -> Result<Tensor> {
        let mut cache_guard = self.state_cache.write().unwrap();
        
        let (ssm_state, conv_state) = if let Some((ref s_state, ref c_state)) = *cache_guard {
            (s_state.clone(), c_state.clone())
        } else {
            let s_state = Tensor::zeros((b_sz, 32, 128, 128), q_heads.dtype(), &self.device)?;
            let c_state = Tensor::zeros((b_sz, 8192, 3), q_heads.dtype(), &self.device)?;
            (s_state, c_state)
        };

        if seq_len == 1 {
            let q_t = q_heads.squeeze(1)?; // [b_sz, 16, 128]
            let k_t = k_heads.squeeze(1)?; // [b_sz, 16, 128]
            let v_t = v_heads.squeeze(1)?; // [b_sz, 32, 128]
            let decay_gate_t = decay_gate.squeeze(1)?; // [b_sz, 32]
            let beta_t = beta.squeeze(1)?; // [b_sz, 32]

            // Qwen3.5 DeltaNet: S_t = decayed_s + delta_t ⊗ K_t
            // State: [b_sz, 32, 128, 128] (32 heads, 128 key_dim, 128 value_dim)

            // k_t: [b_sz, 16, 128] -> [b_sz, 32, 128] (2:1 expand)
            let k_t_expanded = k_t.unsqueeze(2)?.expand((b_sz, 16, 2, 128))?.contiguous()?.reshape((b_sz, 32, 128))?;

            // 1. Decay the recurrent state: S_t = S_{t-1} * decay_gate_t
            let decayed_s = ssm_state.broadcast_mul(&decay_gate_t.unsqueeze(2)?.unsqueeze(3)?)?;

            // 2. Vectorized S @ K computation using decayed state (k^T @ S)
            let sk = k_t_expanded.unsqueeze(2)?.matmul(&decayed_s)?.squeeze(2)?; // [b_sz, 32, 128]

            // 3. Compute error e_t and apply write gate beta_t
            let e = v_t.sub(&sk)?; // [b_sz, 32, 128]
            let delta = e.broadcast_mul(&beta_t.unsqueeze(2)?)?;

            // 4. Update the state with outer product k_t_expanded ⊗ delta
            let outer = k_t_expanded.unsqueeze(3)?.matmul(&delta.unsqueeze(2)?)?; // [b_sz, 32, 128, 128]
            let new_s = (&decayed_s + &outer)?; // [b_sz, 32, 128, 128]

            // Expand Q from 16 to 32 heads
            let q_t_expanded = q_t.unsqueeze(2)?.expand((b_sz, 16, 2, 128))?.contiguous()?.reshape((b_sz, 32, 128))?;

            // Scale query
            let scale = 1.0 / (128.0f64.sqrt());
            let q_scaled = (&q_t_expanded * scale)?;

            // Vectorized output: Q_t^T @ S
            let o_t = q_scaled.unsqueeze(2)?.matmul(&new_s)?.squeeze(2)?; // [b_sz, 32, 128]

            let final_conv = conv_state.clone();
            *cache_guard = Some((new_s.clone(), final_conv));

            o_t.unsqueeze(1) 

        } else {

            // Prefill processing
            // q_heads: [b_sz, seq_len, 16, 128]
            // k_heads: [b_sz, seq_len, 16, 128]
            // v_heads: [b_sz, seq_len, 32, 128]

            // Expand K and Q from 16 to 32 heads
            let k_heads_expanded = k_heads.unsqueeze(3)?.expand((b_sz, seq_len, 16, 2, 128))?.contiguous()?.reshape((b_sz, seq_len, 32, 128))?;
            let q_heads_expanded = q_heads.unsqueeze(3)?.expand((b_sz, seq_len, 16, 2, 128))?.contiguous()?.reshape((b_sz, seq_len, 32, 128))?;

            let scale = 1.0 / (128.0f64.sqrt());
            let q_heads_scaled = (&q_heads_expanded * scale)?;

            let mut s = ssm_state;
            let mut outputs = Vec::with_capacity(seq_len);

            for t in 0..seq_len {
                let k_t = k_heads_expanded.narrow(1, t, 1)?.squeeze(1)?; // [b_sz, 32, 128]
                let q_t = q_heads_scaled.narrow(1, t, 1)?.squeeze(1)?; // [b_sz, 32, 128]
                let v_t = v_heads.narrow(1, t, 1)?.squeeze(1)?; // [b_sz, 32, 128]
                let decay_gate_t = decay_gate.narrow(1, t, 1)?.squeeze(1)?; // [b_sz, 32]
                let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?; // [b_sz, 32]

                // 1. Decay the state
                let decayed_s = s.broadcast_mul(&decay_gate_t.unsqueeze(2)?.unsqueeze(3)?)?;

                // 2. k^T @ S
                let sk = k_t.unsqueeze(2)?.matmul(&decayed_s)?.squeeze(2)?; // [b_sz, 32, 128]
                let e = v_t.sub(&sk)?;
                let delta = e.broadcast_mul(&beta_t.unsqueeze(2)?)?;

                // 3. Outer product update (k_t ⊗ delta)
                let outer = k_t.unsqueeze(3)?.matmul(&delta.unsqueeze(2)?)?; // [b_sz, 32, 128, 128]
                s = (&decayed_s + &outer)?; 

                // 4. Output: Q_t^T @ S
                let o_t = q_t.unsqueeze(2)?.matmul(&s)?.squeeze(2)?; // [b_sz, 32, 128]
                outputs.push(o_t.unsqueeze(1)?); 
            }

            let final_conv = conv_state.clone();
            *cache_guard = Some((s.clone(), final_conv));

            Tensor::cat(&outputs, 1)
        }
    }
}
