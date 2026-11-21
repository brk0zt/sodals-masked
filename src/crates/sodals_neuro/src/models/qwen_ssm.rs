// src/models/qwen_ssm.rs
// Qwen3.5 SSM (State Space Model) Implementation
// Based on Gated DeltaNet linear state space models

use std::sync::{OnceLock, RwLock};
use candle_core::{DType, Device, Result, Tensor, Module};
use crate::models::qwen_custom::{UnifiedLinear, QRmsNorm};

fn ssm_verbose() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("SODALS_VERBOSE").is_some())
}

fn ssm_dump() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("SODALS_DUMP_SSM").is_some())
}

fn ssm_debug_decay() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("SODALS_DEBUG_DECAY").is_some())
}

fn ssm_debug_qk_norm() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("SODALS_DEBUG_QK_NORM").is_some())
}

/// Cached SSM recurrent & convolution state with explicit initialization flags.
pub struct SsmLayerState {
    pub ssm_state: Option<Tensor>,
    pub conv_state: Option<Tensor>,
}

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
    /// State cache: independent Option<ssm_state> (always F32) and Option<conv_state>
    pub state_cache: RwLock<SsmLayerState>,
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
        let block = Self {
            ssm_in_proj,
            attn_gate,
            conv1d,
            ssm_out_proj,
            ssm_norm,
            ssm_a,
            ssm_dt,
            ssm_beta,
            ssm_alpha,
            state_cache: RwLock::new(SsmLayerState {
                ssm_state: None,
                conv_state: None,
            }),
            device: device.clone(),
        };

        if ssm_verbose() {
            let ssm_a_vals = block.ssm_a.flatten_all()?.to_vec1::<f32>().unwrap_or_default();
            if ssm_a_vals.len() >= 4 {
                eprintln!("[SSM INIT] ssm_a first 4: {:?} (shape: {:?})", &ssm_a_vals[..4], block.ssm_a.dims());
                let a_min = ssm_a_vals.iter().cloned().fold(f32::INFINITY, f32::min);
                let a_max = ssm_a_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let a_mean = ssm_a_vals.iter().sum::<f32>() / ssm_a_vals.len() as f32;
                eprintln!("[SSM INIT] ssm_a stats: mean={:.6} min={:.6} max={:.6}", a_mean, a_min, a_max);
            }
            let dt_vals = block.ssm_dt.flatten_all()?.to_vec1::<f32>().unwrap_or_default();
            if dt_vals.len() >= 4 {
                eprintln!("[SSM INIT] ssm_dt first 4: {:?} (shape: {:?})", &dt_vals[..4], block.ssm_dt.dims());
                let dt_min = dt_vals.iter().cloned().fold(f32::INFINITY, f32::min);
                let dt_max = dt_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let dt_mean = dt_vals.iter().sum::<f32>() / dt_vals.len() as f32;
                eprintln!("[SSM INIT] ssm_dt stats: mean={:.6} min={:.6} max={:.6}", dt_mean, dt_min, dt_max);
            }
        }

        Ok(block)
    }

    pub fn clear_cache(&self) {
        if let Ok(mut guard) = self.state_cache.write() {
            guard.ssm_state = None;
            guard.conv_state = None;
        }
    }

    fn dump_intermediate(&self, name: &str, tensor: &Tensor, layer_idx: usize) {
        if !ssm_dump() { return; }
        let fname = format!("rust_layer{}_{}.bin", layer_idx, name);
        if let Ok(data) = tensor.to_dtype(candle_core::DType::F32).and_then(|t| t.flatten_all().and_then(|t| t.to_vec1::<f32>())) {
            let n = data.len();
            let rms = (data.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
            let mean = data.iter().sum::<f32>() / n as f32;
            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut bytes = Vec::with_capacity(n * 4);
            for &v in &data { bytes.extend_from_slice(&v.to_le_bytes()); }
            let _ = std::fs::write(&fname, &bytes);
            eprintln!("[SSM_DUMP] {}: shape={:?} n={} RMS={:.4} mean={:.4} min={:.4} max={:.4}", fname, tensor.dims(), n, rms, mean, min, max);
        }
    }

    pub fn forward(&self, x: &Tensor, layer_idx: usize, seqlen_offset: usize, qmem_cartridge: Option<(&Tensor, f32)>) -> Result<Tensor> {
        // Reset recurrent & conv states cleanly on every new prompt (seqlen_offset == 0)
        if seqlen_offset == 0 {
            self.clear_cache();
        }
        let (b_sz, seq_len, _) = x.dims3()?;
        let dump_ssm = ssm_dump();
        let verbose = ssm_verbose();

        // 1. Input projection
        // x_proj shape: [b, seq_len, 8192]
        let x_proj = self.ssm_in_proj.forward(x)?;
        if dump_ssm { self.dump_intermediate("xproj", &x_proj, layer_idx); }

        // 2. Convolution step
        // conv_out shape: [b, seq_len, 8192]
        let conv_out_raw = self.apply_conv1d(&x_proj, seq_len)?;
        if dump_ssm { self.dump_intermediate("conv_out", &conv_out_raw, layer_idx); }

        // Reference: modeling_qwen3_5.py line 499: mixed_qkv = F.silu(conv1d(mixed_qkv))
        // SiLU is applied to the ENTIRE conv output (q, k, AND v), not just v.
        let conv_out = candle_nn::ops::silu(&conv_out_raw)?;
        if dump_ssm { self.dump_intermediate("conv_out_silu", &conv_out, layer_idx); }

        // Dynamically compute dimensions based on weights
        let in_proj_dim = conv_out.dims3()?.2;
        let v_dim = in_proj_dim / 2; // For Qwen 3.5, v_dim is half of in_proj_dim (e.g. 4096 out of 8192)
        let q_dim = (in_proj_dim - v_dim) / 2;
        let k_dim = (in_proj_dim - v_dim) / 2;
        
        let v_head_count = self.ssm_a.dims()[0];
        let head_dim = v_dim / v_head_count;

        let q = conv_out.narrow(candle_core::D::Minus1, 0, q_dim)?;
        let k = conv_out.narrow(candle_core::D::Minus1, q_dim, k_dim)?;
        let v = conv_out.narrow(candle_core::D::Minus1, q_dim + k_dim, v_dim)?;

        if dump_ssm { self.dump_intermediate("q", &q, layer_idx); }
        if dump_ssm { self.dump_intermediate("k", &k, layer_idx); }
        if dump_ssm { self.dump_intermediate("v", &v, layer_idx); }

        let alpha_proj = self.ssm_alpha.forward(x)?;
        if dump_ssm { self.dump_intermediate("alpha_proj", &alpha_proj, layer_idx); }

        let beta_proj = self.ssm_beta.forward(x)?;
        let beta = candle_nn::ops::sigmoid(&beta_proj)?;
        if dump_ssm { self.dump_intermediate("beta", &beta, layer_idx); }

        // 5. Compute SSM decay parameter: decay_gate = exp(ssm_a * softplus(alpha_proj + ssm_dt))
        // Python reference: g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias) = -A * softplus(...)
        // llama.cpp qwen35: gate = softplus(alpha + dt) * ssm_a, then decay = exp(gate).
        // The GGUF stores ssm_a = -exp(A_log) = -A directly (verified: all values negative),
        // so NO additional exp()/neg() must be applied here.
        // g = -A * softplus(...)  (always negative), decay_gate = exp(g) in (0, 1]
        let dt_val = alpha_proj.broadcast_add(&self.ssm_dt)?;
        let dt_softplus = self.softplus(&dt_val)?;
        let g = dt_softplus.broadcast_mul(&self.ssm_a)?;
        let decay_gate = g.exp()?;
        
        // Log the raw values as requested before any potential clipping
        if ssm_debug_decay() {
            let ssm_a_f32 = self.ssm_a.to_dtype(candle_core::DType::F32)?;
            let dt_f32 = dt_softplus.to_dtype(candle_core::DType::F32)?;
            let dg_f32 = decay_gate.to_dtype(candle_core::DType::F32)?;
            
            let a_mean = ssm_a_f32.mean_all()?.to_scalar::<f32>()?;
            let a_min = ssm_a_f32.min_all()?.to_scalar::<f32>()?;
            let a_max = ssm_a_f32.max_all()?.to_scalar::<f32>()?;
            
            let dt_mean = dt_f32.mean_all()?.to_scalar::<f32>()?;
            let dt_min = dt_f32.min_all()?.to_scalar::<f32>()?;
            let dt_max = dt_f32.max_all()?.to_scalar::<f32>()?;
            
            let dg_mean = dg_f32.mean_all()?.to_scalar::<f32>()?;
            let dg_min = dg_f32.min_all()?.to_scalar::<f32>()?;
            let dg_max = dg_f32.max_all()?.to_scalar::<f32>()?;
            
            eprintln!("[DEBUG DECAY] Layer {} | ssm_a(raw): mean={:.6}, min={:.6}, max={:.6}", layer_idx, a_mean, a_min, a_max);
            eprintln!("[DEBUG DECAY] Layer {} | dt_softplus: mean={:.6}, min={:.6}, max={:.6}", layer_idx, dt_mean, dt_min, dt_max);
            eprintln!("[DEBUG DECAY] Layer {} | decay_gate(raw): mean={:.6}, min={:.6}, max={:.6}", layer_idx, dg_mean, dg_min, dg_max);
        }
        
        if dump_ssm { self.dump_intermediate("decay_gate", &decay_gate, layer_idx); }

        // 6. Reshape for attention
        let q_head_count = q_dim / head_dim;
        let k_head_count = k_dim / head_dim;

        let q_heads = q.reshape((b_sz, seq_len, q_head_count, head_dim))?;
        let k_heads = k.reshape((b_sz, seq_len, k_head_count, head_dim))?;
        let v_heads = v.reshape((b_sz, seq_len, v_head_count, head_dim))?;

        let q_heads = self.l2_norm(&q_heads, head_dim)?;
        let k_heads = self.l2_norm(&k_heads, head_dim)?;
        if dump_ssm { self.dump_intermediate("q_l2normed", &q_heads, layer_idx); }
        if dump_ssm { self.dump_intermediate("k_l2normed", &k_heads, layer_idx); }

        let orig_dtype = q.dtype();
        let q_heads = q_heads.to_dtype(candle_core::DType::F32)?;
        let k_heads = k_heads.to_dtype(candle_core::DType::F32)?;
        let v_heads = v_heads.to_dtype(candle_core::DType::F32)?;
        let decay_gate = decay_gate.to_dtype(candle_core::DType::F32)?;
        let beta = beta.to_dtype(candle_core::DType::F32)?;

        // Reference: torch_recurrent_gated_delta_rule line 365:
        //   scale = 1 / (query.shape[-1] ** 0.5)
        //   query = query * scale
        // This scale is ALWAYS applied regardless of L2 norm.
        let q_scale = 1.0 / (head_dim as f64).sqrt();
        let q_heads = (q_heads * q_scale)?;

        // Gate all 6 GPU->CPU syncs strictly behind cached SODALS_VERBOSE (eliminates 144 syncs/token!)
        if verbose {
            let dg_mean = decay_gate.mean_all()?.to_scalar::<f32>()?;
            let dg_min = decay_gate.min_all()?.to_scalar::<f32>()?;
            let dg_max = decay_gate.max_all()?.to_scalar::<f32>()?;
            let b_mean = beta.mean_all()?.to_scalar::<f32>()?;
            let b_min = beta.min_all()?.to_scalar::<f32>()?;
            let b_max = beta.max_all()?.to_scalar::<f32>()?;
            eprintln!("[SSM PARAMS] decay_gate: mean={:.6} min={:.6} max={:.6} | beta: mean={:.6} min={:.6} max={:.6}",
                dg_mean, dg_min, dg_max, b_mean, b_min, b_max);
            if dg_max > 0.99 {
                eprintln!("[SSM WARNING] decay_gate max={:.6} - insufficient decay may cause state accumulation", dg_max);
            }
            if dg_min < 0.01 {
                eprintln!("[SSM WARNING] decay_gate min={:.6} - excessive decay may cause state loss", dg_min);
            }
        }

        let y_heads = self.recurrent_scan(
            &q_heads, &k_heads, &v_heads,
            &decay_gate, &beta,
            seq_len, b_sz,
            qmem_cartridge
        )?;

        if dump_ssm { self.dump_intermediate("y_heads", &y_heads, layer_idx); }
        
        let y_heads = y_heads.to_dtype(orig_dtype)?;

        if verbose {
            let y_heads_f32 = y_heads.to_dtype(candle_core::DType::F32)?;
            let y_scan_rms = y_heads_f32.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            eprintln!("[SSM SCAN] y_heads RMS={:.6}", y_scan_rms);
        }

        let y_norm = if let Some(ref norm) = self.ssm_norm {
            let y_reshaped = y_heads.reshape((b_sz * seq_len * v_head_count, head_dim))?;
            let y_normed = norm.forward(&y_reshaped)?;
            y_normed.reshape((b_sz, seq_len, v_head_count, head_dim))?
        } else {
            y_heads
        };

        let gate_z = self.attn_gate.forward(x)?;
        let gate_z_silu = candle_nn::ops::silu(&gate_z)?;
        if dump_ssm { self.dump_intermediate("gate_z_post_silu", &gate_z_silu, layer_idx); }

        if verbose {
            let gate_z_silu_f32 = gate_z_silu.to_dtype(candle_core::DType::F32)?;
            let gate_z_mean = gate_z_silu_f32.mean_all()?.to_scalar::<f32>()?;
            eprintln!("[SSM GATE] z_proj: post_silu_mean={:.6}", gate_z_mean);
        }
        let gate_z_reshaped = gate_z_silu.reshape((b_sz, seq_len, v_head_count, head_dim))?;
        let y_gated = (y_norm * gate_z_reshaped)?;

        if dump_ssm { self.dump_intermediate("y_gated", &y_gated, layer_idx); }

        let y_final = y_gated.reshape((b_sz, seq_len, v_dim))?;

        let y_final = self.ssm_out_proj.forward(&y_final)?;
        if dump_ssm { self.dump_intermediate("ssm_out", &y_final, layer_idx); }

        Ok(y_final)
    }

    fn apply_conv1d(&self, x_proj: &Tensor, seq_len: usize) -> Result<Tensor> {
        let (b_sz, _, d_inner) = x_proj.dims3()?;
        let mut cache_guard = self.state_cache.write().unwrap();

        let w0 = self.conv1d.narrow(1, 0, 1)?; // [8192, 1] -> Oldest input weight
        let w1 = self.conv1d.narrow(1, 1, 1)?; // [8192, 1]
        let w2 = self.conv1d.narrow(1, 2, 1)?; // [8192, 1]
        let w3 = self.conv1d.narrow(1, 3, 1)?; // [8192, 1] -> Current input weight

        if seq_len == 1 {
            let conv_state = if let Some(ref c_state) = cache_guard.conv_state {
                c_state.clone()
            } else {
                Tensor::zeros((b_sz, d_inner, 3), x_proj.dtype(), &self.device)?
            };

            let x_t = x_proj.transpose(1, 2)?; // [b, 8192, 1]

            let c_past_0 = conv_state.narrow(2, 0, 1)?; // x[t-3]
            let c_past_1 = conv_state.narrow(2, 1, 1)?; // x[t-2]
            let c_past_2 = conv_state.narrow(2, 2, 1)?; // x[t-1]

            let out = (
                (x_t.broadcast_mul(&w3)?) +
                (c_past_2.broadcast_mul(&w2)?) +
                (c_past_1.broadcast_mul(&w1)?) +
                (c_past_0.broadcast_mul(&w0)?)
            )?;

            let new_conv_state = Tensor::cat(&[&c_past_1, &c_past_2, &x_t], 2)?;
            cache_guard.conv_state = Some(new_conv_state);

            out.transpose(1, 2)
        } else {
            let x_t = x_proj.transpose(1, 2)?;

            let past_conv = if let Some(ref c) = cache_guard.conv_state {
                c.clone()
            } else {
                Tensor::zeros((b_sz, d_inner, 3), x_proj.dtype(), &self.device)?
            };
            let padded = Tensor::cat(&[&past_conv, &x_t], 2)?;

            let s0 = padded.narrow(2, 3, seq_len)?; // x[t]
            let s1 = padded.narrow(2, 2, seq_len)?; // x[t-1]
            let s2 = padded.narrow(2, 1, seq_len)?; // x[t-2]
            let s3 = padded.narrow(2, 0, seq_len)?; // x[t-3]

            let out = (
                (s0.broadcast_mul(&w3)?) +
                (s1.broadcast_mul(&w2)?) +
                (s2.broadcast_mul(&w1)?) +
                (s3.broadcast_mul(&w0)?)
            )?;

            let c0 = padded.narrow(2, seq_len, 1)?;
            let c1 = padded.narrow(2, seq_len + 1, 1)?;
            let c2 = padded.narrow(2, seq_len + 2, 1)?;

            let new_conv_state = Tensor::cat(&[&c0, &c1, &c2], 2)?;
            cache_guard.conv_state = Some(new_conv_state);

            out.transpose(1, 2)
        }
    }

    fn l2_norm(&self, x: &Tensor, head_dim: usize) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() == 4 {
            // Already 4D: [b, seq, heads, head_dim] - normalize across last dimension
            let sq = (x * x)?;
            let sum = sq.sum_keepdim(candle_core::D::Minus1)?;
            let norm = (sum + 1e-6)?.sqrt()?;
            x.broadcast_div(&norm)
        } else if dims.len() == 3 {
            // 3D: [b, seq, total_dim] - reshape to 4D first
            let (b_sz, seq_len, total_dim) = x.dims3()?;
            let num_heads = total_dim / head_dim;
            let x_reshaped = x.reshape((b_sz, seq_len, num_heads, head_dim))?;
            
            let sq = (&x_reshaped * &x_reshaped)?;
            let sum = sq.sum_keepdim(candle_core::D::Minus1)?;
            let norm = (sum + 1e-6)?.sqrt()?;
            let out = x_reshaped.broadcast_div(&norm)?;
            
            out.reshape((b_sz, seq_len, total_dim))
        } else {
            Err(candle_core::Error::Msg(format!("l2_norm: unexpected tensor rank {}, expected 3 or 4", dims.len())))
        }
    }

    fn softplus(&self, x: &Tensor) -> Result<Tensor> {
        // Numerically stable F32 softplus: clamp input before exp() so exp(x) never overflows to inf/NaN!
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let threshold = x_f32.gt(20.0f32)?;
        let x_clamped = x_f32.clamp(-40.0f32, 20.0f32)?;
        let stable = (x_clamped.exp()? + 1.0)?.log()?;
        let out = threshold.where_cond(&x_f32, &stable)?;
        out.to_dtype(orig_dtype)
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
        qmem_cartridge: Option<(&Tensor, f32)>,
    ) -> Result<Tensor> {
        let mut cache_guard = self.state_cache.write().unwrap();

        let (mut base_ssm_state, is_initial_state) = if let Some(ref s_state) = cache_guard.ssm_state {
            // Ensure F32 dtype even if externally populated
            (s_state.to_dtype(DType::F32)?, false)
        } else {
            let v_head_count = v_heads.dim(2)?;
            let head_dim = v_heads.dim(3)?;
            (Tensor::zeros((b_sz, v_head_count, head_dim, head_dim), DType::F32, &self.device)?, true)
        };

        // Native Delta Gate Memory Write: S_0 = S_0 + outer_cartridge
        if is_initial_state {
            if let Some((cart_tensor, similarity)) = qmem_cartridge {
                let (cb, cs) = cart_tensor.dims2().unwrap_or((1usize, 4096usize));
                // cart_tensor is [1, 4096], reshape into [1, 32, 128]
                if cs == 4096 {
                    let reshaped = cart_tensor.reshape((cb, 32, 128))?;
                    // For k_cartridge and v_cartridge, we can just use variations of the same reshape for now
                    let k_cart = reshaped.clone().contiguous()?;
                    let v_cart = reshaped.contiguous()?; // Not mathematically perfect, but acts as a robust semantic prior
                    
                    let beta_cart = similarity as f64; // Use true holographic similarity as gate

                    let sk_cart = k_cart.unsqueeze(2)?.matmul(&base_ssm_state)?.squeeze(2)?; // [b_sz, 32, 128]
                    let e_cart = v_cart.sub(&sk_cart)?; // [b_sz, 32, 128]
                    let delta_cart = (e_cart * beta_cart)?;
                    let outer_cart = k_cart.unsqueeze(3)?.matmul(&delta_cart.unsqueeze(2)?)?; // [b_sz, 32, 128, 128]
                    
                    base_ssm_state = (&base_ssm_state + &outer_cart)?;
                }
            }
        }

        if seq_len == 1 {
            let q_t = q_heads.squeeze(1)?.contiguous()?; // [b_sz, 16, 128]
            let k_t = k_heads.squeeze(1)?.contiguous()?; // [b_sz, 16, 128]
            let v_t = v_heads.squeeze(1)?.contiguous()?; // [b_sz, 32, 128]
            let decay_gate_t = decay_gate.squeeze(1)?.contiguous()?; // [b_sz, 32]
            let beta_t = beta.squeeze(1)?.contiguous()?; // [b_sz, 32]

            // Qwen3.5 DeltaNet: S_t = decayed_s + delta_t ⊗ K_t
            // State: [b_sz, 32, 128, 128] (32 heads, 128 key_dim, 128 value_dim)

            // llama.cpp qwen35 expands q/k from num_k_heads (16) to num_v_heads (32)
            // via ggml_repeat_4d, which TILES: [K_0, K_1, ..., K_15, K_0, K_1, ..., K_15].
            // Heads h and h+16 share the same q/k; beta/v stay per-v-head (32).
            let k_t_expanded = Tensor::cat(&[&k_t, &k_t], 1)?;

            // 1. Decay the recurrent state: S_t = S_{t-1} * decay_gate_t
            // Reference: state = exp(g) * state where g is log-space decay
            let decayed_s = base_ssm_state.broadcast_mul(&decay_gate_t.unsqueeze(2)?.unsqueeze(3)?)?;

            // 2. Vectorized S @ K computation using decayed state (k^T @ S)
            let sk = k_t_expanded.unsqueeze(2)?.matmul(&decayed_s)?.squeeze(2)?; // [b_sz, 32, 128]

            // 3. Compute error e_t and apply write gate beta_t
            let e = v_t.sub(&sk)?; // [b_sz, 32, 128]
            let delta = e.broadcast_mul(&beta_t.unsqueeze(2)?)?;

            // 4. Update the state with outer product k_t_expanded ⊗ delta
            let outer = k_t_expanded.unsqueeze(3)?.matmul(&delta.unsqueeze(2)?)?; // [b_sz, 32, 128, 128]
            let new_s = (&decayed_s + &outer)?; // [b_sz, 32, 128, 128]
            
            // FIX: Removed state clipping - it was causing quality degradation
            // Keep only decay gate clipping for numerical stability
            let q_t_expanded = Tensor::cat(&[&q_t, &q_t], 1)?;

            // Vectorized output: Q_t^T @ S (Queries are L2-normalized; no double q_scale)
            let o_t = q_t_expanded.unsqueeze(2)?.matmul(&new_s)?.squeeze(2)?; // [b_sz, 32, 128]

            cache_guard.ssm_state = Some(new_s);

            o_t.unsqueeze(1)
        } else {

            // Prefill processing
            // q_heads: [b_sz, seq_len, 16, 128]
            // k_heads: [b_sz, seq_len, 16, 128]
            // v_heads: [b_sz, seq_len, 32, 128]

            // Expand K and Q from 16 to 32 heads (TILE, matching llama ggml_repeat_4d:
            // [K_0..K_15, K_0..K_15] so heads h and h+16 share q/k)
            let k_heads_expanded = Tensor::cat(&[&k_heads, &k_heads], 2)?;
            let q_heads_expanded = Tensor::cat(&[&q_heads, &q_heads], 2)?;

            let mut s = base_ssm_state;
            let mut outputs = Vec::with_capacity(seq_len);

            for t in 0..seq_len {
                let k_t = k_heads_expanded.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // [b_sz, 32, 128]
                let q_t = q_heads_expanded.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // [b_sz, 32, 128]
                let v_t = v_heads.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // [b_sz, 32, 128]
                let decay_gate_t = decay_gate.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // [b_sz, 32]
                let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?.contiguous()?; // [b_sz, 32]

                // 1. Decay the state
                let decayed_s = s.broadcast_mul(&decay_gate_t.unsqueeze(2)?.unsqueeze(3)?)?;

                // 2. k^T @ S
                let sk = k_t.unsqueeze(2)?.matmul(&decayed_s)?.squeeze(2)?; // [b_sz, 32, 128]
                let e = v_t.sub(&sk)?;
                let delta = e.broadcast_mul(&beta_t.unsqueeze(2)?)?;

                // 3. Outer product update (k_t ⊗ delta)
                let outer = k_t.unsqueeze(3)?.matmul(&delta.unsqueeze(2)?)?; // [b_sz, 32, 128, 128]
                s = (&decayed_s + &outer)?;
                
                // Cached flag: zero env syscalls per step
                if ssm_verbose() && t == 0 {
                    let s_rms = s.flatten_all()?.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                    let s_max = s.flatten_all()?.abs()?.max_all()?.to_scalar::<f32>()?;
                    eprintln!("[SSM STATE] prefill t=0: RMS={:.6} max={:.6}", s_rms, s_max);
                }

                // 4. Output: Q_t^T @ S
                let o_t = q_t.unsqueeze(2)?.matmul(&s)?.squeeze(2)?; // [b_sz, 32, 128]
                if ssm_debug_qk_norm() && t == 0 {
                    let s_flat = s.flatten_all()?.to_vec1::<f32>().unwrap_or_default();
                    let o_flat = o_t.flatten_all()?.to_vec1::<f32>().unwrap_or_default();
                    if s_flat.len() >= 10 {
                        eprintln!("[DEBUG SSM PREFILL] t=0 S first 10: {:?}", &s_flat[..10]);
                    }
                    if o_flat.len() >= 10 {
                        eprintln!("[DEBUG SSM PREFILL] t=0 o_t first 10: {:?}", &o_flat[..10]);
                    }
                }
                outputs.push(o_t.unsqueeze(1)?);
            }

            cache_guard.ssm_state = Some(s);

            Tensor::cat(&outputs, 1)
        }
    }
}
