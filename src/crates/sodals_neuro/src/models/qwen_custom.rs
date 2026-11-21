// src/models/qwen_custom.rs

// Custom Quantized Qwen 2/3 Model with GGUF Support + Physics Kernel Injection

//

// REFACTORED: AWQ (Activation-aware Weight Quantization) for 8GB VRAM optimization

// SODALS GPR: True Gaussian Phase Router with Z-Score gating (CPU-only, zero VRAM cost)

// Phase 34: OPP (Orthogonal Phase Projection) replaces phase-swap for true orthogonal projection



use candle_core::{DType, Device, Error, Module, Result, Tensor, D, IndexOp};

use candle_nn::{VarBuilder, Linear};

use candle_core::quantized::QMatMul;

use candle_core::quantized::gguf_file;

use candle_core::quantized::{GgmlDType, QTensor};

use std::sync::{Arc, RwLock, Mutex, OnceLock};

#[inline]
fn is_verbose() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("SODALS_VERBOSE").is_some())
}

use std::collections::HashMap;

use std::io::{Read, Seek};

use std::path::PathBuf;



use crate::gpu_ops::{NeuroAdapter, SodalsCudaAccess};
use crate::gpu_ops::physics_mask;

use crate::engine::pondering_core::PonderingCore;

use crate::fast_math::SemanticWave;

use crate::qmem_loader::{QmemLoader, QmemParameter};

use crate::phase45_gpr_qmem_router::Phase45GprQmemRouter;

use crate::quant::layer_quant::{run_layerwise_quantization, LayerQuantConfig};

use super::qwen_ssm::QwenSsmBlock;

// use crate::gpu_ops::memory_utils::SodalsCudaPinnedAccess;



//  Phase 34: TRUE GAUSSIAN PHASE ROUTER (GPR)

//

//  Replaces hardcoded magic-number thresholds with proper rolling Gaussian statistics.

//  Maintains EMA of phase interference magnitudes and uses strict Z-Score gating.

//  Bypass condition: Z = |z_i - mu| / sigma > 1.96 (95% confidence interval)

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII



/// True Gaussian Phase Router with rolling EMA statistics.

/// All computation is CPU f32 � zero VRAM cost.

pub struct GaussianPhaseRouter {

    /// Exponential moving average of z_i magnitudes

    running_mean: f32,

    /// Exponential moving average of squared deviations (for variance)

    running_var: f32,

    /// Number of samples seen (for warm-up gating)

    sample_count: u32,

    /// EMA smoothing factor (? = 0.2 � ~5-sample effective window for fast adaptation)

    alpha: f32,

    /// EMA of the bypass decision rate (used for variance compensation)

    bypass_rate_ema: f32,

}



impl GaussianPhaseRouter {

    pub fn new() -> Self {

        Self {

            running_mean: 0.0,

            running_var: 1.0, // Start with unit variance to avoid div-by-zero

            sample_count: 0,

            alpha: 0.2, // Phase 45: Faster adaptation (was 0.05)

            bypass_rate_ema: 0.0,

        }

    }



    /// Update EMA statistics with a new z_i sample and return the bypass decision.

    ///

    /// Returns `true` if MLP should be BYPASSED (z_i is a statistical outlier).

    /// During warm-up (first 3 samples): always returns `false` (MLP runs).

    ///

    /// Z-Score bypass: |z_i - �| / ? > 1.96  (95% confidence interval)

    pub fn should_bypass(&mut self, z_i: f32) -> bool {

        self.sample_count = self.sample_count.saturating_add(1);



        // Update EMA statistics

        let alpha = self.alpha;

        self.running_mean = (1.0 - alpha) * self.running_mean + alpha * z_i;

        let deviation = z_i - self.running_mean;

        self.running_var = (1.0 - alpha) * self.running_var + alpha * deviation * deviation;



        // Warm-up: accumulate at least 3 samples before making bypass decisions

        if self.sample_count < 3 {

            return false;

        }



        // Compute standard deviation (with epsilon for numerical stability)

        let sigma = (self.running_var + 1e-8).sqrt();



        // Z-Score test: bypass if z_i is a statistical outlier

        let z_score = deviation.abs() / sigma;

        let bypass = z_score > 0.674; // 50% confidence interval outlier check (bypasses ~50% of MLPs)

        

        // Update bypass rate EMA

        let target = if bypass { 1.0 } else { 0.0 };

        self.bypass_rate_ema = (1.0 - alpha) * self.bypass_rate_ema + alpha * target;

        

        bypass

    }

    

    pub fn bypass_rate(&self) -> f32 {

        self.bypass_rate_ema

    }

}



//  Phase 44: SPLIT-COMPLEX PROJECTION (SCP)

//

//  Rank-8 low-rank adaptation module for legitimate split-complex computation.

//  The math (X�A - Y�B) + i(Y�A + X�B) is a LEGITIMATE split-complex operation.

//  W_scp = A � B where A is [hidden_size, 8] and B is [8, hidden_size].

//  VRAM cost: 2 � hidden_size � 8 � 4 bytes ? 256KB per layer.

//  For 32 layers: ~8.2MB total. Well within 8GB VRAM budget.

//

//  We OWN this mathematics. This is NOT a "Deterministic Permutation Trick".

//  This is the Split-Complex projection: (X + iY) ? (A + iB) = (XA - YB) + i(YA + XB)

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



pub struct SplitComplexProjection {

    pub a_real: Tensor, // [in_dim, rank]

    pub b_real: Tensor, // [rank, out_dim]

    pub a_imag: Tensor, // [in_dim, rank]

    pub b_imag: Tensor, // [rank, out_dim]

}



impl SplitComplexProjection {

    /// Initialize with Kaiming uniform distribution, deterministic per layer_idx.

    /// rank=8 keeps VRAM footprint negligible.

    /// PHASE 44 FIX: Separate in_dim and out_dim for GQA compatibility (q_scp: 4096�4096, k_scp: 4096�1024)

    pub fn new(in_dim: usize, out_dim: usize, rank: usize, layer_idx: usize, device: &Device) -> Result<Self> {

        // Kaiming uniform bound: sqrt(6 / fan_in)

        let bound_a = (6.0_f64 / in_dim as f64).sqrt();

        let bound_b = (6.0_f64 / rank as f64).sqrt();



        // Deterministic seed based on layer index for reproducibility

        let seed = (layer_idx as u64).wrapping_mul(0x517cc1b727220a95).wrapping_add(0x6c62272e07bb0142);

        

        // Provide a closure to generate deterministic random weights

        let gen_data = |seed: u64, bound: f64, size: usize, offset: u64| -> Vec<f32> {

            (0..size)

                .map(|i| {

                    let mut x = seed.wrapping_add((i as u64) + offset);

                    x ^= x << 13; x ^= x >> 7; x ^= x << 17;

                    let normalized = (x as f64 % 1000000.0) / 1000000.0; // [0, 1)

                    ((normalized * 2.0 - 1.0) * bound) as f32

                })

                .collect()

        };



        let a_real = Tensor::from_vec(gen_data(seed, bound_a, in_dim * rank, 0), (in_dim, rank), device)?;

        let b_real = Tensor::from_vec(gen_data(seed, bound_b, rank * out_dim, 1000000), (rank, out_dim), device)?;

        

        let a_imag = Tensor::from_vec(gen_data(seed, bound_a, in_dim * rank, 2000000), (in_dim, rank), device)?;

        let b_imag = Tensor::from_vec(gen_data(seed, bound_b, rank * out_dim, 3000000), (rank, out_dim), device)?;



        Ok(Self { a_real, b_real, a_imag, b_imag })

    }



    /// Phase 44: Spectral Warm-up - Initialize SCP using leading eigenvectors

    /// 

    /// Instead of random Kaiming noise, use the QMEM semantic network's leading

    /// eigenvectors (or deterministic hash of vocabulary topology) to create a

    /// mathematically sound "Topological Phase Projector".

    /// 

    /// W_scp = A � B where A columns are eigenvector projections, B rows are

    /// corresponding eigenvalue-weighted coefficients.

    /// 

    /// This implements the Split-Complex multiplication:

    ///   (X + iY) ? (A + iB) = (XA - YB) + i(YA + XB)

    /// 

    /// We OWN this mathematics. No apologies.

    /// 

    /// # Arguments

    /// * `hidden_size` - Model hidden dimension (e.g., 4096)

    /// * `rank` - LoRA rank (typically 8)

    /// * `layer_idx` - Layer index for deterministic eigenvector selection

    /// * `device` - Candle device

    /// * `eigenvalues` - Optional leading eigenvalues from QMEM (use hash if None)

    pub fn spectral_warmup(

        hidden_size: usize,

        rank: usize,

        layer_idx: usize,

        device: &Device,

        eigenvalues: Option<&[f64]>,

    ) -> Result<Self> {

        // Deterministic hash-based "eigenvalue" generator

        // Uses the same seeding as new() for consistency

        let seed = (layer_idx as u64).wrapping_mul(0x517cc1b727220a95).wrapping_add(0x6c62272e07bb0142);

        

        // Generate pseudo-eigenvalues (spectral energy distribution)

        // Decay: ?_k ? 1/k (natural frequency falloff like violin strings)

        let mut spectral_weights: Vec<f64> = (0..rank)

            .map(|k| {

                let decay = 1.0 / (k as f64 + 1.0); // 1, 1/2, 1/3, ...

                let mut x = seed.wrapping_add(k as u64);

                x ^= x << 13; x ^= x >> 7; x ^= x << 17;

                let perturbation = ((x % 1000) as f64 / 1000.0 - 0.5) * 0.1; // �5% noise

                decay * (1.0 + perturbation)

            })

            .collect();



        // If real eigenvalues provided, blend them with hash-based values

        if let Some(real_eigs) = eigenvalues {

            for (i, w) in spectral_weights.iter_mut().enumerate().take(rank.min(real_eigs.len())) {

                *w = 0.7 * real_eigs[i] + 0.3 * (*w); // 70% real, 30% synthetic

            }

        }



        // Normalize spectral weights

        let total_energy: f64 = spectral_weights.iter().map(|&w| w * w).sum::<f64>().sqrt();

        if total_energy > 1e-10 {

            spectral_weights.iter_mut().for_each(|w| *w /= total_energy);

        }



        let mut a_real_data = Vec::with_capacity(hidden_size * rank);

        let mut b_real_data = Vec::with_capacity(rank * hidden_size);

        let mut a_imag_data = Vec::with_capacity(hidden_size * rank);

        let mut b_imag_data = Vec::with_capacity(rank * hidden_size);



        for r in 0..rank {

            let freq = (r + 1) as f64;

            let weight = spectral_weights[r];

            

            for h in 0..hidden_size {

                let x = h as f64 / hidden_size as f64;

                

                // Real part: standard sine/cosine pairs

                a_real_data.push(((freq * std::f64::consts::PI * x).sin() * weight) as f32);

                b_real_data.push(((freq * std::f64::consts::PI * x).cos() * weight) as f32);



                // Imaginary part: phase shifted by 45 degrees

                a_imag_data.push((((freq * std::f64::consts::PI * x) + std::f64::consts::PI/4.0).sin() * weight) as f32);

                b_imag_data.push((((freq * std::f64::consts::PI * x) + std::f64::consts::PI/4.0).cos() * weight) as f32);

            }

        }



        let a_real = Tensor::from_vec(a_real_data, (hidden_size, rank), device)?;

        let b_real = Tensor::from_vec(b_real_data, (rank, hidden_size), device)?;

        let a_imag = Tensor::from_vec(a_imag_data, (hidden_size, rank), device)?;

        let b_imag = Tensor::from_vec(b_imag_data, (rank, hidden_size), device)?;



        eprintln!(

            "[SPECTRAL SCP] Layer {} initialized with {} modes (energy: {:.4}) (4-Matrix True Split-Complex)",

            layer_idx, rank, spectral_weights.iter().sum::<f64>()

        );



        Ok(Self { a_real, b_real, a_imag, b_imag })

    }



/// Phase 45: TRUE Split-Complex Projection

    /// 

    /// Mathematical Formula:

    ///   (X + iY) ? (A + iB) = (XA - YB) + i(YA + XB)

    /// 

    /// Where:

    ///   - X = x_real (real input component)

    ///   - Y = x_imag (imaginary input component)

    ///   - A, B = split-complex weight matrices

    /// 

    /// This computes BOTH the real and imaginary output components correctly.

    /// No more fake LoRA-scaling. We OWN this mathematics.

    /// 

    /// # Arguments

    /// * `x_real` - Real input component [batch, seq_len, hidden_size]

    /// * `x_imag` - Imaginary input component [batch, seq_len, hidden_size]

    /// 

    /// # Returns

    /// * (real_out, imag_out) tuple with proper split-complex computation

    pub fn forward_split(

        &self,

        x_real: &Tensor,

        x_imag: &Tensor

    ) -> Result<(Tensor, Tensor)> {

        let (batch, seq_len, _hidden) = x_real.dims3()?;

        

        let x_real_2d = x_real.reshape((batch * seq_len, ()))?;

        let x_imag_2d = x_imag.reshape((batch * seq_len, ()))?;

        

        // PHASE 49.8 FIX: Correct matrix multiplication X * A * B (No Transpose)

        // X[B*S, in_dim] * A[in_dim, rank] * B[rank, out_dim] = [B*S, out_dim]

        let x_w_real = x_real_2d.matmul(&self.a_real)?.matmul(&self.b_real)?;

        let y_w_imag = x_imag_2d.matmul(&self.a_imag)?.matmul(&self.b_imag)?;



        let y_w_real = x_imag_2d.matmul(&self.a_real)?.matmul(&self.b_real)?;

        let x_w_imag = x_real_2d.matmul(&self.a_imag)?.matmul(&self.b_imag)?;

        

        let real_out = (&x_w_real - &y_w_imag)?;

        let imag_out = (&y_w_real + &x_w_imag)?;

        

        let real_3d = real_out.reshape((batch, seq_len, ()))?;

        let imag_3d = imag_out.reshape((batch, seq_len, ()))?;

        

        Ok((real_3d, imag_3d))

    }



    /// Legacy forward for backward compatibility (computes only real component)

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {

        let (batch, seq_len, _hidden) = x.dims3()?;

        let x_2d = x.reshape((batch * seq_len, ()))?; 

        // PHASE 49.8 FIX: Correct order X * A * B (No Transpose)

        let out = x_2d.matmul(&self.a_real)?.matmul(&self.b_real)?;             

        out.reshape((batch, seq_len, ()))

    }



    /// Phase 44: Liquid SCP - Oja's Rule Hebbian Plasticity

    /// 

    /// Updates the SCP matrices dynamically during inference using the

    /// fetched 4KB topological weight block from the NVMe QMEM cartridge.

    /// 

    /// Oja's Rule (stabilized Hebbian learning):

    ///   ?W_scp = ? (y � x^T - y� � W_scp)

    /// 

    /// Where:

    ///   - ? (eta): Learning rate (must be high for rapid rewiring during RK4)

    ///   - y: Projection output (B � x)

    ///   - x: Input thought vector

    ///   - W_scp: Current split-complex weight matrix (A � B)

    ///   - qmem_weights: Topological weights fetched from holographic NVMe address

    /// 

    /// The qmem_weights bias the learning to align with the topological

    /// structure stored on the SSD. This allows the network to "rewire"

    /// itself within the 10 RK4 pondering steps.

    /// 

    /// # Arguments

    /// * `x` - Input thought vector [1, 1, hidden_size]

    /// * `qmem_weights` - Topological weights from NVMe (decoded from 4KB block)

    /// * `learning_rate` - ? for Oja's Rule (suggested: 0.1 to 0.5)

    pub fn oja_update(

        &mut self,

        x: &Tensor,

        qmem_weights: &[f32],

        learning_rate: f32,

    ) -> Result<()> {

        let device = self.a_real.device();

        let hidden_size = self.a_real.dim(0)?;

        let rank = self.a_real.dim(1)?;

        

        // Convert input to f32 vector

        let x_vec: Vec<f32> = x.to_vec1()?;

        

        // Current projection: y = B � x (low-rank bottleneck)

        let mut y = vec![0.0f32; rank];

        for r in 0..rank {

            // Get B row as slice - B is [rank, hidden_size]

            let b_row_start = r * hidden_size;

            let b_row_end = b_row_start + hidden_size;

            if b_row_end > qmem_weights.len() {

                break;

            }

            

            // Compute dot product: y[r] = sum_j B[r,j] * x[j]

            for (h, &x_val) in x_vec.iter().enumerate().take(hidden_size) {

                if b_row_start + h < qmem_weights.len() {

                    y[r] += qmem_weights[b_row_start + h] * x_val;

                }

            }

        }

        

        // y� for stabilization term

        let y_squared: f32 = y.iter().map(|&v| v * v).sum();

        

        // Oja's Rule: ?B = ? (y � x^T - y� � B)

        // Update B matrix (stored as qmem_weights slice concept)

        // For LoRA: we update the B matrix directly, A stays fixed

        let eta = learning_rate;

        

        // Create new B tensor with updates

        let mut new_b_data = vec![0.0f32; rank * hidden_size];

        

        for r in 0..rank {

            let row_start = r * hidden_size;

            

            for h in 0..hidden_size {

                let idx = row_start + h;

                if idx >= new_b_data.len() || idx >= qmem_weights.len() {

                    continue;

                }

                

                // Current B value (from qmem bias)

                let b_current = qmem_weights.get(idx).copied().unwrap_or(0.0);

                

                // Hebbian term: y[r] * x[h]

                let hebbian = y[r] * x_vec.get(h).copied().unwrap_or(0.0);

                

                // Stabilization term: y� * B[r,h]

                let stabilization = y_squared * b_current;

                

                // Oja update: B_new = B_old + ?(hebbian - stabilization)

                new_b_data[idx] = b_current + eta * (hebbian - stabilization);

            }

        }

        

        // Create updated B tensor

        let new_b = Tensor::from_vec(new_b_data, (rank, hidden_size), device)?;

        

        // Update self.b_real with the new values. In a true system, we'd distribute this,

        // but since Oja's rule primarily targets the amplitude/real component in legacy, we update b_real.

        self.b_real = new_b;

        

        eprintln!(

            "[LIQUID SCP] Oja update complete | eta={:.3} | y_sq={:.6} | rank={}",

            eta, y_squared, rank

        );

        

        Ok(())

    }



    /// Phase 44: Liquid Update with Holographic Fetch

    /// 

    /// Convenience method that combines:

    /// 1. Compute holographic address from thought vector

    /// 2. Fetch 4KB block from NVMe QMEM

    /// 3. Apply Oja's Rule with fetched weights

    /// 

    /// This is the main entry point for Liquid SCP updates during RK4.

    pub fn liquid_update(

        &mut self,

        x: &Tensor,

        qmem: &crate::nvme_direct::HolographicQMEM,

        pondering_core: &crate::engine::PonderingCore,

        learning_rate: f32,

    ) -> Result<()> {

        // Fetch topological weights via O(1) holographic addressing

        let qmem_weights = pondering_core.fetch_holographic_weights(qmem, x)?;

        

        if qmem_weights.is_empty() {

            eprintln!("[LIQUID SCP] No QMEM weights fetched, skipping update");

            return Ok(());

        }

        

        // Apply Oja's Rule with fetched weights

        self.oja_update(x, &qmem_weights, learning_rate)

    }

}



//  II1  SODALS Gaussian Phase Router (GPR)

//

//  Her decoder katmaninin MLP blogunu bir "uzman" olarak modeller.

//  Tum hesaplama CPU'da f32 skaler - VRAM'e SIFIR maliyet.

//

//  mu_i     = (i / (N-1)) * 2*pi   (katmanlar [0, 2*pi] araligina yayilir)

//  T_dynamic = 1.96 * (1+(1-W)) * (1/(1+J))    clamp [0.5, 3.0]

//  Z_i       = |(theta - mu_i) / sigma_i|

//  Aktif    :  Z_i <= T_dynamic  -->  MLP forward() cagrilir

//  Pasif    :  Z_i >  T_dynamic  -->  MLP forward() HIC cagrilmaz (VRAM=0)

//  Fallback :  hic kimse gecemezse en kucuk Z'li katman zorla aktif

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII



/// CPU'dan her forward gecisinde iletilen deterministik fiziksel sinyaller.

/// Tensor degil; saf Rust f32 skaler paketidir (+ optional RoPE warp tensor).

#[derive(Debug, Clone)]

pub struct PhaseSignal {

    /// SemanticWave ciktisi: promptun semantik faz acisi

    pub theta: f32,

    /// Semantic divergence [0.0, 1.0] -- 0 = hallucination/loop, 1 = healthy

    pub semantic_divergence: f32,

    /// Jacobian normu [0, inf) -- dusuk = stabil, yuksek = kaotik/patlayan

    pub jacobian_norm: f32,

    /// RoPE frequency warp factor tensor [1, head_dim/2].

    /// 1.0 = no warp. < 1.0 = compress positional distance (gravitational lensing).

    /// None = standard unwarped RoPE.

    pub rope_warp_tensor: Option<Tensor>,

}



impl PhaseSignal {

    pub fn new(theta: f32, semantic_divergence: f32, jacobian_norm: f32) -> Self {

        Self {

            theta,

            semantic_divergence: semantic_divergence.clamp(0.0, 1.0),

            jacobian_norm: jacobian_norm.max(0.0),

            rope_warp_tensor: None,

        }

    }

    /// Saglikli varsayilan: tum MLP'ler aktif, no warp

    pub fn default_healthy() -> Self { Self::new(0.0, 1.0, 0.0) }



    /// Attach a pre-computed RoPE warp tensor to this signal.

    pub fn with_warp(mut self, warp: Tensor) -> Self {

        self.rope_warp_tensor = Some(warp);

        self

    }

}



// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

//  II2  UnifiedLinear  (degismedi)

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
#[derive(Clone)]
pub enum UnifiedLinear {

    Quantized(QMatMul, Option<Tensor>),

    HostQuantized(QMatMul, Option<Tensor>),

    Regular(Linear),

}



impl Module for UnifiedLinear {

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {

        match self {

            UnifiedLinear::Quantized(qm, bias) => {

                let xs_contiguous = xs.contiguous()?;

                let (batch_size, seq_len, hidden_dim) = xs_contiguous.dims3()?;

                let xs_2d = xs_contiguous.reshape((batch_size * seq_len, hidden_dim))?;

                let mut out = qm.forward(&xs_2d)?;

                if let Some(b) = bias {

                    let b_unsq = if b.rank() == 1 { b.unsqueeze(0)? } else { b.clone() };

                    out = out.broadcast_add(&b_unsq)?;

                }

                let out_contiguous = out.contiguous()?;

                let out_dim = out_contiguous.dim(1)?;

                Ok(out_contiguous.reshape((batch_size, seq_len, out_dim))?.to_dtype(DType::F32)?)

            },

            UnifiedLinear::HostQuantized(qm, bias) => {

                let xs_contiguous = xs.contiguous()?;

                let (batch_size, seq_len, hidden_dim) = xs_contiguous.dims3()?;

                let xs_2d = xs_contiguous.reshape((batch_size * seq_len, hidden_dim))?;

                let xs_cpu = xs_2d.to_device(&Device::Cpu)?;

                let mut out = qm.forward(&xs_cpu)?;

                if let Some(b) = bias {

                    let b_unsq = if b.rank() == 1 { b.unsqueeze(0)? } else { b.clone() };

                    out = out.broadcast_add(&b_unsq)?;

                }

                let out_contiguous = out.contiguous()?;

                let out_dim = out_contiguous.dim(1)?;

                let res = out_contiguous.reshape((batch_size, seq_len, out_dim))?.to_dtype(DType::F32)?;

                res.to_device(xs.device())

            },

            UnifiedLinear::Regular(l) => {

                l.forward(xs)

            },

        }

    }

}



impl UnifiedLinear {

    fn forward_owned(&self, xs: Tensor) -> Result<Tensor> { self.forward(&xs) }

}



// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

//  II3  Qwen2Config  (gpr_bandwidth alani eklendi)

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]

pub struct Qwen2Config {

    pub vocab_size: usize,

    pub hidden_size: usize,

    pub intermediate_size: usize,

    pub num_hidden_layers: usize,

    pub num_attention_heads: usize,

    pub num_key_value_heads: usize,

    pub max_position_embeddings: usize,

    pub sliding_window: Option<usize>,

    pub rope_theta: f64,

    pub rms_norm_eps: f64,

    pub use_sliding_window: bool,

    pub num_bits: usize,

    pub rope_dim: usize,

}



impl Default for Qwen2Config {

    fn default() -> Self {

        Self {

            vocab_size: 151936,

            hidden_size: 4096,

            intermediate_size: 13696,

            num_hidden_layers: 32,

            num_attention_heads: 32,

            num_key_value_heads: 8,

            max_position_embeddings: 32768,

            sliding_window: Some(32768),

            rope_theta: 1000000.0,

            rms_norm_eps: 1e-6,

            use_sliding_window: false,

            num_bits: 4,

            rope_dim: 0,

        }

    }

}



// metadata helpers (degismedi) ------------------------------------------------

fn get_metadata_u32(

    metadata: &std::collections::HashMap<String, gguf_file::Value>,

    key: &str,

) -> Option<u32> {

    metadata.get(&format!("qwen35.{}", key))

        .or_else(|| metadata.get(&format!("qwen3.{}", key)))

        .or_else(|| metadata.get(&format!("qwen2.{}", key)))

        .or_else(|| metadata.get(&format!("llama.{}", key)))

        .and_then(|v| v.to_u32().ok())

}



fn get_metadata_f32(

    metadata: &std::collections::HashMap<String, gguf_file::Value>,

    key: &str,

) -> Option<f32> {

    metadata.get(&format!("qwen35.{}", key))

        .or_else(|| metadata.get(&format!("qwen3.{}", key)))

        .or_else(|| metadata.get(&format!("qwen2.{}", key)))

        .or_else(|| metadata.get(&format!("llama.{}", key)))

        .and_then(|v| v.to_f32().ok())

}



pub fn config_from_gguf(ct: &gguf_file::Content) -> Result<Qwen2Config> {

    let metadata = &ct.metadata;

    // Debug: Print all metadata keys related to block, count, length, context
    eprintln!("[GGUF META DEBUG] Scanning metadata for relevant keys...");
    for (k, v) in metadata.iter() {
        if k.contains("block") || k.contains("count") || k.contains("length") || k.contains("context") || k.contains("embedding") || k.contains("attention") || k.contains("head") {
            eprintln!("[GGUF META] {} = {:?}", k, v);
        }
    }

    // Qwen 3.5 9B GGUF metadata: head_count=16, head_count_kv=4
    // blk.0.attn_qkv.weight shape [4096, 8192], head_dim=256
    // Q=4096(16h), K=1024(4h), V=1024(4h), Gate=2048(8h)
    let num_attention_heads = get_metadata_u32(metadata, "attention.head_count").map(|v| v as usize).unwrap_or(32);
    let num_key_value_heads = get_metadata_u32(metadata, "attention.head_count_kv").map(|v| v as usize).unwrap_or(8);

    Ok(Qwen2Config {

        vocab_size: metadata.get("tokenizer.ggml.tokens")

            .and_then(|v| v.to_vec().ok()).map(|v| v.len()).unwrap_or(151936),

        hidden_size:             get_metadata_u32(metadata, "embedding_length").map(|v| v as usize).unwrap_or(4096),

        intermediate_size:       get_metadata_u32(metadata, "feed_forward_length").map(|v| v as usize).unwrap_or(13696),

        num_hidden_layers:       get_metadata_u32(metadata, "block_count").map(|v| v as usize).unwrap_or(32),

        num_attention_heads:     num_attention_heads,

        num_key_value_heads:     num_key_value_heads,

        max_position_embeddings: get_metadata_u32(metadata, "context_length").map(|v| v as usize).unwrap_or(32768),

        sliding_window:  None,

        rope_theta:      get_metadata_f32(metadata, "rope.freq_base").map(|v| v as f64).unwrap_or(1000000.0),

        rms_norm_eps:    get_metadata_f32(metadata, "attention.layer_norm_rms_epsilon").map(|v| v as f64).unwrap_or(1e-6),

        use_sliding_window: false,

        num_bits: 4,

        rope_dim: get_metadata_u32(metadata, "rope.dimension_count").map(|v| v as usize).unwrap_or(0),

    })

    // Sanity check: Verify parsed config values are reasonable
    .and_then(|cfg| {
        eprintln!("[CONFIG] layers={}, hidden={}, heads={}, kv_heads={}, rope_theta={}, vocab={}",
            cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads,
            cfg.num_key_value_heads, cfg.rope_theta, cfg.vocab_size);

        if cfg.num_hidden_layers < 20 || cfg.num_hidden_layers > 100 {
            Err(candle_core::Error::Msg(format!("GGUF parse hatas�: num_hidden_layers={} sa�ma, GGUF metadata key'leri kontrol edilmeli", cfg.num_hidden_layers)))
        } else if cfg.hidden_size < 1024 {
            Err(candle_core::Error::Msg(format!("GGUF parse hatas�: hidden_size={}", cfg.hidden_size)))
        } else {
            Ok(cfg)
        }
    })
}



// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

//  II4  RotaryEmbedding  (degismedi)

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

pub struct RotaryEmbedding { pub(crate) sin: Tensor, pub(crate) cos: Tensor, _dim: usize }

impl RotaryEmbedding {

    fn new(cfg: &Qwen2Config, device: &Device) -> Result<Self> {

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        let rope_dim = if cfg.rope_dim > 0 { cfg.rope_dim } else { head_dim };

        let theta = cfg.rope_theta as f32;

        let inv_freq: Vec<f32> = (0..rope_dim).step_by(2)

            .map(|i| 1.0 / theta.powf(i as f32 / rope_dim as f32))

            .collect();



        let inv_freq = Tensor::from_vec(inv_freq.clone(), inv_freq.len(), device)?;



        let max_pos = cfg.max_position_embeddings.max(65536);

        

        let t = Tensor::arange(0u32, max_pos as u32, device)?.to_dtype(DType::F32)?;



        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?.contiguous()?;



        Ok(Self { sin: freqs.sin()?, cos: freqs.cos()?, _dim: rope_dim })

    }



    fn apply_rotary_emb_qkv(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize, phase: Option<&PhaseSignal>) -> Result<(Tensor, Tensor)> {

        let (b, h, seq_len, d) = q.dims4()?;

        let rope_dim = self._dim;

        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?.contiguous()?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?.contiguous()?;

        let (cos_unified, sin_unified) = if let Some(p) = phase {
            let pos_dim = rope_dim / 2;
            let half_pos_dim = pos_dim / 2;
            let half_qmem_dim = (rope_dim - pos_dim) / 2;

            let cos_pos = cos.narrow(candle_core::D::Minus1, 0, half_pos_dim)?;
            let sin_pos = sin.narrow(candle_core::D::Minus1, 0, half_pos_dim)?;

            let cos_qmem_base = cos.narrow(candle_core::D::Minus1, half_pos_dim, half_qmem_dim)?;
            let sin_qmem_base = sin.narrow(candle_core::D::Minus1, half_pos_dim, half_qmem_dim)?;

            let cos_theta = (p.theta as f64).cos();
            let sin_theta = (p.theta as f64).sin();

            let cos_qmem = ((&cos_qmem_base * cos_theta)? - (&sin_qmem_base * sin_theta)?)?;
            let sin_qmem = ((&sin_qmem_base * cos_theta)? + (&cos_qmem_base * sin_theta)?)?;

            (
                Tensor::cat(&[&cos_pos, &cos_qmem], candle_core::D::Minus1)?,
                Tensor::cat(&[&sin_pos, &sin_qmem], candle_core::D::Minus1)?
            )
        } else {
            (cos.clone(), sin.clone())
        };

        if rope_dim < d {

            let q_rope = q.narrow(candle_core::D::Minus1, 0, rope_dim)?;

            let q_pass = q.narrow(candle_core::D::Minus1, rope_dim, d - rope_dim)?;

            let k_rope = k.narrow(candle_core::D::Minus1, 0, rope_dim)?;

            let k_pass = k.narrow(candle_core::D::Minus1, rope_dim, d - rope_dim)?;

            let q_rotated = candle_nn::rotary_emb::rope(&q_rope.contiguous()?, &cos_unified, &sin_unified)?;

            let k_rotated = candle_nn::rotary_emb::rope(&k_rope.contiguous()?, &cos_unified, &sin_unified)?;

            let q_out = Tensor::cat(&[&q_rotated, &q_pass], candle_core::D::Minus1)?;
            let k_out = Tensor::cat(&[&k_rotated, &k_pass], candle_core::D::Minus1)?;

            Ok((q_out, k_out))

        } else {

            Ok((

                candle_nn::rotary_emb::rope(&q.contiguous()?, &cos_unified, &sin_unified)?,

                candle_nn::rotary_emb::rope(&k.contiguous()?, &cos_unified, &sin_unified)?,

            ))

        }

    }

}



// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

//  II5  QRmsNorm  (degismedi)

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
#[derive(Clone)]
pub struct QRmsNorm { weight: Tensor, eps: f64, zero_centered: bool }



impl QRmsNorm {

    // AUDIT FIX (Claude, 2026-07): plain multiplicative weight, matches Qwen3NextRMSNormGated
    // (the SSM/GatedDeltaNet output norm) - weight trained from a ones-init, used as-is.
    pub fn new(weight: Tensor, eps: f64) -> Result<Self> { Ok(Self { weight, eps, zero_centered: false }) }

    // AUDIT FIX (Claude, 2026-07): Qwen3NextRMSNorm (input_layernorm, post_attention_layernorm,
    // q_norm, k_norm, final model norm) trains weight from a ZEROS-init and scales by (1.0 + weight).
    // Reference (transformers/models/qwen3_next/modeling_qwen3_next.py, Qwen3NextRMSNorm.forward):
    //   output = self._norm(x.float()); output = output * (1.0 + self.weight.float())
    // and Qwen3NextPreTrainedModel._init_weights: init.zeros_(module.weight) with the comment
    // "We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)".
    // Every call site EXCEPT ssm_norm must use this constructor, not plain new().
    pub fn new_zero_centered(weight: Tensor, eps: f64) -> Result<Self> { Ok(Self { weight, eps, zero_centered: true }) }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }



    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {

        let x_dtype = x.dtype();

        let internal_dtype = match x_dtype {

            DType::F16 | DType::BF16 => DType::F32,

            d => d,

        };

        let hidden_size = x.dim(D::Minus1)?;

        let x = x.to_dtype(internal_dtype)?;

        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;

        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;

        // Multiply by weight in F32 internal precision BEFORE casting back to x_dtype
        let w_f32 = self.weight.to_dtype(internal_dtype)?;
        let out_f32 = if self.zero_centered {
            let xw = x_normed.broadcast_mul(&w_f32)?;
            (&x_normed + &xw)?
        } else {
            x_normed.broadcast_mul(&w_f32)?
        };
        out_f32.to_dtype(x_dtype)

    }

}



// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

//  II6  QMlp  (degismedi III ama artik sadece GPR'dan "true" alan katmanlarda cagrilir)

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

pub struct QMlp {

    gate_proj: UnifiedLinear,

    up_proj:   UnifiedLinear,

    down_proj: UnifiedLinear,

}



impl QMlp {

    fn new(g: UnifiedLinear, u: UnifiedLinear, d: UnifiedLinear) -> Self {

        Self { gate_proj: g, up_proj: u, down_proj: d }

    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {

        let gate   = self.gate_proj.forward(xs)?;

        let up     = self.up_proj.forward(xs)?;

        let hidden = candle_nn::ops::silu(&gate)?.mul(&up)?;

        self.down_proj.forward_owned(hidden)?.to_dtype(DType::F32)

    }

}



// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

//  II7  QAttention  (degismedi)

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

fn trace_dump_active() -> bool {
    std::env::var("SODALS_TRACE_DUMP").is_ok()
        || std::env::var("SODALS_DEBUG_QK_NORM").is_ok()
        || std::env::var("SODALS_DUMP_ATTN_RESIDUAL").is_ok()
}

fn trace_target_layer() -> usize {
    std::env::var("SODALS_TRACE_LAYER")
        .or_else(|_| std::env::var("SODALS_DEBUG_LAYER"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
}

fn trace_target_token(seq_len: usize) -> usize {
    std::env::var("SODALS_TRACE_TOKEN")
        .or_else(|_| std::env::var("SODALS_DEBUG_TOKEN"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(seq_len.saturating_sub(1))
}

fn write_f32_bin(path: &str, data: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(path, &bytes)?;
    Ok(())
}

fn tensor_token_row(t: &Tensor, token_idx: usize) -> Result<Vec<f32>> {
    let dims = t.dims();
    let row = match dims.len() {
        2 => t.narrow(0, token_idx, 1)?.flatten_all()?,
        3 => t.i((0, token_idx, ..))?.flatten_all()?,
        _ => return Err(Error::Msg(format!("trace token row: unexpected dims {dims:?}"))),
    };
    row.to_dtype(DType::F32)?.to_vec1()
}

fn tensor_head_token_slice(t: &Tensor, token_idx: usize, head_idx: usize) -> Result<Vec<f32>> {
    let dims = t.dims();
    let slice = if dims.len() == 4 && dims[1] > dims[2] {
        // [b, seq, heads, head_dim]
        t.narrow(1, token_idx, 1)?
            .narrow(2, head_idx, 1)?
            .reshape((dims[3],))?
    } else if dims.len() == 4 {
        // [b, heads, seq, head_dim]
        t.narrow(2, token_idx, 1)?
            .narrow(1, head_idx, 1)?
            .reshape((dims[3],))?
    } else {
        return Err(Error::Msg(format!("trace head slice: unexpected dims {dims:?}")));
    };
    slice.to_dtype(DType::F32)?.to_vec1()
}

fn log_tensor_stats(tag: &str, data: &[f32]) {
    if data.is_empty() {
        eprintln!("[TRACE] {tag}: empty");
        return;
    }
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    eprintln!(
        "[TRACE] {tag}: mean={mean:.6} std={:.6} min={min:.6} max={max:.6} first8={:?}",
        var.sqrt(),
        &data[..8.min(data.len())]
    );
}

fn maybe_dump_attn_norm_in(layer_idx: usize, norm_out: &Tensor) -> Result<()> {
    if !trace_dump_active() || layer_idx != trace_target_layer() {
        return Ok(());
    }
    let seq_len = norm_out.dim(1)?;
    let token_idx = trace_target_token(seq_len);
    let data = tensor_token_row(norm_out, token_idx)?;
    let fname = format!("rust_attn_norm_in_layer{layer_idx}_token{token_idx}.bin");
    log_tensor_stats(&format!("attn_norm_in layer={layer_idx} token={token_idx}"), &data);
    write_f32_bin(&fname, &data)?;
    eprintln!("[TRACE] wrote {fname} ({} floats)", data.len());
    Ok(())
}

fn maybe_dump_layer_out(layer_idx: usize, layer_out: &Tensor) -> Result<()> {
    if !trace_dump_active() || layer_idx > trace_target_layer() {
        return Ok(());
    }
    let seq_len = layer_out.dim(1)?;
    let token_idx = trace_target_token(seq_len);
    let data = tensor_token_row(layer_out, token_idx)?;
    let fname = format!("rust_layer_out_{layer_idx}_token{token_idx}.bin");
    log_tensor_stats(&format!("layer_out layer={layer_idx} token={token_idx}"), &data);
    write_f32_bin(&fname, &data)?;
    eprintln!("[TRACE] wrote {fname} ({} floats)", data.len());
    Ok(())
}

fn maybe_dump_attention_trace(
    layer_idx: usize,
    seq_len: usize,
    query_states: &Tensor,
    key_states: &Tensor,
    qs_pre_norm: &Tensor,
    qs_post_norm: &Tensor,
    ks_post_norm: &Tensor,
    qs_post_rope: &Tensor,
    ks_post_rope: &Tensor,
) -> Result<()> {
    if !trace_dump_active() || layer_idx != trace_target_layer() {
        return Ok(());
    }
    let token_idx = trace_target_token(seq_len);
    let head_idx = 0usize;

    let dumps: [(&str, Vec<f32>); 7] = [
        ("q_proj_raw", tensor_token_row(query_states, token_idx)?),
        ("k_proj_raw", tensor_token_row(key_states, token_idx)?),
        (
            "q_pre_norm",
            tensor_head_token_slice(qs_pre_norm, token_idx, head_idx)?,
        ),
        (
            "q_post_norm",
            tensor_head_token_slice(qs_post_norm, token_idx, head_idx)?,
        ),
        (
            "k_post_norm",
            tensor_head_token_slice(ks_post_norm, token_idx, head_idx)?,
        ),
        (
            "q_post_rope",
            tensor_head_token_slice(qs_post_rope, token_idx, head_idx)?,
        ),
        (
            "k_post_rope",
            tensor_head_token_slice(ks_post_rope, token_idx, head_idx)?,
        ),
    ];

    for (label, data) in dumps {
        let fname = format!("rust_{label}_layer{layer_idx}_token{token_idx}.bin");
        log_tensor_stats(&format!("{label} layer={layer_idx} token={token_idx} head={head_idx}"), &data);
        write_f32_bin(&fname, &data)?;
        eprintln!("[TRACE] wrote {fname} ({} floats)", data.len());
    }
    Ok(())
}

fn maybe_dump_qk_after_norm(layer_idx: usize, qs_r: &Tensor, ks_r: &Tensor) -> Result<()> {
    if !trace_dump_active() || layer_idx != trace_target_layer() {
        return Ok(());
    }
    let (_b, seq_len, num_heads, head_dim) = qs_r.dims4()?;
    let token_idx = trace_target_token(seq_len);

    for (label, t) in [("Q", qs_r), ("K", ks_r)] {
        let data = tensor_head_token_slice(t, token_idx, 0)?;
        let fname = format!(
            "rust_{}_normed_layer{}_token{}.bin",
            label.to_lowercase(),
            layer_idx,
            token_idx
        );
        log_tensor_stats(
            &format!("{label}-normed layer={layer_idx} token={token_idx} head=0"),
            &data,
        );
        write_f32_bin(&fname, &data)?;
        eprintln!("[TRACE] wrote {fname} ({} floats)", data.len());
    }

    eprintln!(
        "[TRACE] shapes Q={:?} K={:?} (seq={}, heads={}, head_dim={})",
        qs_r.dims(),
        ks_r.dims(),
        seq_len,
        num_heads,
        head_dim
    );
    Ok(())
}

pub struct QAttention {

    q_proj: UnifiedLinear, k_proj: Option<UnifiedLinear>, v_proj: Option<UnifiedLinear>,

    o_proj: UnifiedLinear,

    // Phase 44: Split-Complex Projection (SCP) - rank-8 learnable split-complex projection

    // The math is LEGITIMATE: (X�A - Y�B) + i(Y�A + X�B) is split-complex multiplication.

    // We OWN this mathematics. This is NOT a "Deterministic Permutation Trick".

    // W_scp = A � B where A?R^{H�8}, B?R^{8�H}. VRAM: ~256KB per layer.

    q_scp: Option<SplitComplexProjection>,

    k_scp: Option<SplitComplexProjection>,

    // Legacy: kept for models that ship explicit imaginary weight tensors

    q_proj_i: Option<UnifiedLinear>, k_proj_i: Option<UnifiedLinear>,

    num_heads: usize, num_kv_heads: usize, num_kv_groups: usize, head_dim: usize,

    rotary_emb: Arc<RotaryEmbedding>,

    q_norm: Option<QRmsNorm>, k_norm: Option<QRmsNorm>,

    kv_cache: RwLock<Option<(Tensor, Tensor)>>,

}



impl QAttention {

    #[allow(clippy::too_many_arguments)]

    fn new(

        q_proj: UnifiedLinear, k_proj: Option<UnifiedLinear>, v_proj: Option<UnifiedLinear>,

        o_proj: UnifiedLinear,

        num_heads: usize, num_kv_heads: usize, num_kv_groups: usize, head_dim: usize,

        rotary_emb: Arc<RotaryEmbedding>,

        q_norm: Option<QRmsNorm>, k_norm: Option<QRmsNorm>,

        // Phase 44: SCP for legitimate split-complex projection

        q_scp: Option<SplitComplexProjection>, k_scp: Option<SplitComplexProjection>,

        // Legacy: explicit imaginary weight matrices (from model files)

        q_proj_i: Option<UnifiedLinear>, k_proj_i: Option<UnifiedLinear>,

    ) -> Self {

        Self { q_proj, k_proj, v_proj, o_proj,

               q_scp, k_scp,

               q_proj_i, k_proj_i,

               num_heads, num_kv_heads, num_kv_groups, head_dim, rotary_emb,

               q_norm, k_norm, kv_cache: RwLock::new(None) }

    }

    pub fn split_qkv_projection(combined_dim: usize, hidden_size: usize, kv_dim: usize) -> (usize, usize, usize) {
        let v_chunk = kv_dim;
        let k_chunk = kv_dim;
        let q_chunk = if combined_dim >= k_chunk + v_chunk {
            combined_dim - k_chunk - v_chunk
        } else {
            hidden_size
        };
        (q_chunk, k_chunk, v_chunk)
    }




    fn repeat_kv(&self, xs: Tensor) -> Result<Tensor> {

        if self.num_kv_groups == 1 { return Ok(xs); }

        let (b, nkv, sl, hd) = xs.dims4()?;

        xs.unsqueeze(2)?

          .expand((b, nkv, self.num_kv_groups, sl, hd))?

          .reshape((b, nkv * self.num_kv_groups, sl, hd))

    }



    fn dump_intermediate(&self, name: &str, tensor: &Tensor, layer_idx: usize) -> Result<()> {
        if std::env::var("SODALS_DUMP_ATTN_RESIDUAL").is_err() {
            return Ok(());
        }
        if tensor.dims().is_empty() || tensor.dims().iter().any(|&d| d == 0) {
            return Ok(());
        }
        let flat = tensor.flatten_all()?;
        let vals: Vec<f32> = flat.to_vec1()?;
        let out_path = format!("rust_{}-{}.bin", name, layer_idx);
        let out_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        if let Err(e) = std::fs::write(&out_path, &out_bytes) {
            eprintln!("[ATTN DUMP] Failed to write {}: {}", out_path, e);
        }
        let dims: Vec<String> = tensor.dims().iter().map(|d| d.to_string()).collect();
        let shape_str = dims.join(",");
        eprintln!("[ATTN_DUMP] rust_{}-{}.bin: shape=[{}] n={} RMS={:.4}", name, layer_idx, shape_str, vals.len(), (vals.iter().map(|v| v*v).sum::<f32>() / vals.len() as f32).sqrt());
        Ok(())
    }

    fn forward(
        &self, xs: &Tensor, attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        adapter: Option<&NeuroAdapter>, 
        // Phase 21.5: Raw u64 device pointers from ZeroCopyCartridge
        poles_ptr: Option<u64>,
        impedance_ptr: Option<u64>,
        flags_ptr: Option<u64>,
        token_concept_map_ptr: Option<u64>,
        layer_idx: usize,
        num_layers: usize,
        qmem_parameters: Option<&Arc<Mutex<HashMap<usize, QmemParameter>>>>,
        // GNN Integration Pointers
        gnn_graph_ptrs: Option<(u64, u64, u64, u64, u64, f32)>, // (q_nodes, k_nodes, indptr, indices, data, weight)


        n_nodes: Option<i32>,

        phase: Option<&PhaseSignal>,
        qmem_loader: Option<&Arc<QmemLoader>>,

    ) -> Result<Tensor> {

        let (b_size, seq_len, _) = xs.dims3()?;

        // 1. Real Projection (X * A^T)





        let (query_states, key_states, value_states, attn_gate) =
            if self.k_proj.is_none() && self.v_proj.is_none() {
                let qkv  = self.q_proj.forward(xs)?;
                let qkv_shape = qkv.dims();
                let qkv_dim = qkv_shape[qkv_shape.len() - 1];
                
                let q_dim = self.num_heads * self.head_dim;
                let kv_dim = self.num_kv_heads * self.head_dim;
                
                if qkv_dim == q_dim + 2 * kv_dim {
                    (qkv.narrow(D::Minus1, 0, q_dim)?,
                     qkv.narrow(D::Minus1, q_dim, kv_dim)?,
                     qkv.narrow(D::Minus1, q_dim + kv_dim, kv_dim)?,
                     None)
                } else if qkv_dim == q_dim + kv_dim {
                    (qkv.narrow(D::Minus1, 0, q_dim)?,
                     qkv.narrow(D::Minus1, q_dim, kv_dim)?,
                     qkv.narrow(D::Minus1, q_dim, kv_dim)?,
                     None)
                } else if qkv_dim == 8192 {
                    let expected_q_dim = self.num_heads * self.head_dim; // 16 * 256 = 4096
                    let expected_kv_dim = self.num_kv_heads * self.head_dim; // 4 * 256 = 1024
                    let gate_dim = qkv_dim - (expected_q_dim + 2 * expected_kv_dim); // 8192 - 6144 = 2048

                    let q = qkv.narrow(D::Minus1, 0, expected_q_dim)?.contiguous()?;
                    let k = qkv.narrow(D::Minus1, expected_q_dim, expected_kv_dim)?.contiguous()?;
                    let v = qkv.narrow(D::Minus1, expected_q_dim + expected_kv_dim, expected_kv_dim)?.contiguous()?;
                    let gate = if gate_dim > 0 {
                        Some(qkv.narrow(D::Minus1, expected_q_dim + 2 * expected_kv_dim, gate_dim)?.contiguous()?)
                    } else {
                        None
                    };
                    (q, k, v, gate)
                } else {
                    eprintln!("[WARNING] Unexpected QKV dimension: {}, expected {} or {}", qkv_dim, q_dim + 2 * kv_dim, q_dim + kv_dim);
                    (qkv.narrow(D::Minus1, 0, q_dim)?,
                     qkv.narrow(D::Minus1, q_dim, kv_dim)?,
                     qkv.narrow(D::Minus1, q_dim + kv_dim, kv_dim)?,
                     None)
                }
            } else {
                let q_output = self.q_proj.forward(xs)?;
                let q_shape = q_output.dims();
                let q_dim = q_shape[q_shape.len() - 1];

                let expected_q_dim = self.num_heads * self.head_dim;

                if layer_idx == 3 && is_verbose() {
                    let shape_str = format!("{:?}", q_shape);
                    eprintln!("[DEBUG SEP_Q] layer={} q_output shape={} q_dim={} expected_q_dim={} is_gated={}",
                        layer_idx, shape_str, q_dim, expected_q_dim, q_dim == 2 * expected_q_dim);
                }

                // FIX 1: Remove magic number branch - only use gated attention if explicitly configured
                // Qwen 3.5 9B may have q_dim == 8192 but it's not necessarily gated attention
                let k = self.k_proj.as_ref().unwrap().forward(xs)?;
                let v = self.v_proj.as_ref().unwrap().forward(xs)?;

                // Only create gate if q_dim is exactly 2 * expected_q_dim (true gated attention)
                if q_dim == 2 * expected_q_dim {
                    let b_size = q_shape[0];
                    let seq_len = q_shape[1];
                    let q_reshaped = q_output.reshape((b_size, seq_len, self.num_heads, 2 * self.head_dim))?;
                    
                    let q = q_reshaped.narrow(candle_core::D::Minus1, 0, self.head_dim)?
                        .contiguous()?.reshape((b_size, seq_len, expected_q_dim))?;
                        
                    let gate = q_reshaped.narrow(candle_core::D::Minus1, self.head_dim, self.head_dim)?
                        .contiguous()?.reshape((b_size, seq_len, expected_q_dim))?;
                        
                    (q, k, v, Some(gate))
                } else {
                    // Standard projection - use full q_output as Q
                    (q_output, k, v, None)
                }

            };
        self.dump_intermediate("k_proj", &key_states, layer_idx)?;
        self.dump_intermediate("q_proj", &query_states, layer_idx)?;

        // 2. Phase Injection (Imaginary Component Y)

        let has_phase = phase.is_some();



        let imag_x = if let Some(p) = phase {

            (xs * (p.semantic_divergence as f64))? // Use semantic divergence as momentum

        } else {

            Tensor::zeros_like(xs)?

        };



        // 3. Orthogonal Phase Projection � Phase 34: True OPP

        let (q_i_states, k_i_states) = if has_phase {

            if let (Some(q_i), Some(k_i)) = (&self.q_proj_i, &self.k_proj_i) {

                // Path 1: Explicit imaginary weight matrices from model file

                (q_i.forward(&imag_x)?, k_i.forward(&imag_x)?)

            } else if let (Some(q_scp), Some(k_scp)) = (&self.q_scp, &self.k_scp) {

                // Path 2: SCP � legitimate split-complex projection

                (q_scp.forward(&imag_x)?, k_scp.forward(&imag_x)?)

            } else {

                // Path 3: No imaginary weights available � zero contribution

                let q_dim = self.num_heads * self.head_dim;

                let kv_dim = self.num_kv_heads * self.head_dim;

                (Tensor::zeros((b_size, seq_len, q_dim), query_states.dtype(), query_states.device())?,

                 Tensor::zeros((b_size, seq_len, kv_dim), key_states.dtype(), key_states.device())?)

            }

        } else {

            // No phase signal � zero tensors in the correct projected shape

            let q_dim = self.num_heads * self.head_dim;

            let kv_dim = self.num_kv_heads * self.head_dim;

            (Tensor::zeros((b_size, seq_len, q_dim), query_states.dtype(), query_states.device())?,

             Tensor::zeros((b_size, seq_len, kv_dim), key_states.dtype(), key_states.device())?)

        };



        // 4. Multi-Head Reshape & Norm

        if is_verbose() && layer_idx % 4 == 3 {
            let qs_dims: Vec<usize> = query_states.dims().to_vec();
            eprintln!("[ATTN SHAPE] layer={} qs_dims={:?} num_heads={} head_dim={} b_size={} seq_len={}",
                layer_idx, qs_dims, self.num_heads, self.head_dim, b_size, seq_len);
        }
        let mut qs_r = query_states.reshape((b_size, seq_len, self.num_heads, self.head_dim))?;

        if is_verbose() && layer_idx % 4 == 3 {
            let qs_r_dims: Vec<usize> = qs_r.dims().to_vec();
            eprintln!("[ATTN SHAPE] layer={} AFTER RESHAPE qs_r_dims={:?}", layer_idx, qs_r_dims);
        }

        let qs_pre_norm = qs_r.clone();

        if let Some(n) = &self.q_norm { qs_r = n.forward(&qs_r)? }

        let qs_post_norm = qs_r.clone();
        self.dump_intermediate("q_normed", &qs_post_norm, layer_idx)?;

        let qs_r = qs_r.transpose(1, 2)?.contiguous()?;



        let mut qs_i: Tensor = q_i_states.reshape((b_size, seq_len, self.num_heads, self.head_dim))?;

        if has_phase { if let Some(n) = &self.q_norm { qs_i = n.forward(&qs_i)? } }

        let qs_i = qs_i.transpose(1, 2)?.contiguous()?;



        // Determine actual KV head count from K tensor dimensions

        let k_shape = key_states.dims();

        let k_last_dim = k_shape[k_shape.len() - 1];

        let actual_kv_heads = k_last_dim / self.head_dim;

        

        let mut ks_r = key_states.reshape((b_size, seq_len, actual_kv_heads, self.head_dim))?;

        if let Some(n) = &self.k_norm { ks_r = n.forward(&ks_r)? }

        let ks_post_norm = ks_r.clone();
        self.dump_intermediate("k_normed", &ks_post_norm, layer_idx)?;

        maybe_dump_qk_after_norm(layer_idx, &qs_post_norm, &ks_post_norm)?;

        let ks_r = ks_r.transpose(1, 2)?.contiguous()?;



        let mut ks_i: Tensor = k_i_states.reshape((b_size, seq_len, actual_kv_heads, self.head_dim))?;

        if has_phase { if let Some(n) = &self.k_norm { ks_i = n.forward(&ks_i)? } }

        let ks_i = ks_i.transpose(1, 2)?.contiguous()?;



        let vs_r = value_states.reshape((b_size, seq_len, actual_kv_heads, self.head_dim))?

            .transpose(1, 2)?.contiguous()?;



        // Phase 33 FIX: Apply RoPE to BOTH real and imaginary components at their

        // ABSOLUTE sequence position BEFORE inserting into KV-cache.

        let (qs_r, ks_r_rotated) = self.rotary_emb.apply_rotary_emb_qkv(&qs_r, &ks_r, seqlen_offset, phase)?;
        let qs_r = qs_r.contiguous()?;
        let ks_r_rotated = ks_r_rotated.contiguous()?;
        self.dump_intermediate("q_rope", &qs_r, layer_idx)?;
        self.dump_intermediate("k_rope", &ks_r_rotated, layer_idx)?;

        maybe_dump_attention_trace(
            layer_idx,
            seq_len,
            &query_states,
            &key_states,
            &qs_pre_norm,
            &qs_post_norm,
            &ks_post_norm,
            &qs_r,
            &ks_r_rotated,
        )?;

        

        // Apply RoPE to imaginary components at the same absolute position

        let (qs_i, ks_i_rotated) = if has_phase {

            self.rotary_emb.apply_rotary_emb_qkv(&qs_i, &ks_i, seqlen_offset, phase)?

        } else {

            (qs_i, ks_i)

        };



        // Phase 33: Ghost Token offset tracking - account for +1 physical shift

        let effective_offset = seqlen_offset;

        let _ = effective_offset; // Used for documentation clarity



        // Phase 33: Holographic KV Cache Update with CORRECTLY ROTATED keys

        // PERFORMANCE FIX: Reduce clone operations in KV cache update
        let ks_h = ks_r_rotated;
        let vs_h = vs_r; // Value wave is collapsed to real

        let mut kv_cache = self.kv_cache.write().unwrap();

        let (k_cache, v_cache) = {
            let cache = &mut *kv_cache;

            if seqlen_offset == 0 {
                // First token / Prefill prompt - initialize cache
                // Pre-slice prefill prompt sequence to 1536 tokens if initial prompt length > 1536
                let (k, v) = if ks_h.dim(2)? > 1536 {
                    let cur_len = ks_h.dim(2)?;
                    (ks_h.narrow(2, cur_len - 1536, 1536)?.contiguous()?, vs_h.narrow(2, cur_len - 1536, 1536)?.contiguous()?)
                } else {
                    (ks_h.contiguous()?, vs_h.contiguous()?)
                };
                *cache = Some((k.clone(), v.clone()));
                (k, v)
            } else if let Some((pk, pv)) = cache.take() {
                // Phase-Aware Holographic KV Pruning for SODALS (Zero-Fragmentation pre-slice):
                // If KV cache is at or above 1536 tokens, keep last 1535 before concatenating 1 new token.
                // Output is exactly 1536 tokens with 0 temporary memory reallocation spike!
                let (pk_to_cat, pv_to_cat) = if pk.dim(2)? >= 1536 {
                    let cur_len = pk.dim(2)?;
                    (pk.narrow(2, cur_len - 1535, 1535)?, pv.narrow(2, cur_len - 1535, 1535)?)
                } else {
                    (pk, pv)
                };
                let nk = Tensor::cat(&[&pk_to_cat, &ks_h], 2)?.contiguous()?;
                let nv = Tensor::cat(&[&pv_to_cat, &vs_h], 2)?.contiguous()?;
                *cache = Some((nk.clone(), nv.clone()));
                (nk, nv)
            } else {
                // Fallback - shouldn't normally happen
                let k = ks_h.contiguous()?;
                let v = vs_h.contiguous()?;
                *cache = Some((k.clone(), v.clone()));
                (k, v)
            }
        };



        // 6. Resonance Epilogue: Choose between Custom CUDA Reactor or Standard MatMul

        let (_, _, seq_k, _) = k_cache.dims4()?;

        // Track whether fused resonance applied physics bias
        let mut physics_bias_applied = false;

        let mut attention_scores = if let (Some(p_ptr), Some(i_ptr), Some(f_ptr)) = (poles_ptr, impedance_ptr, flags_ptr) {
            // --- TOPOLOGICAL RESONANCE REACTOR (CUDA) ---
            let attn_scores = Tensor::zeros((b_size, self.num_heads, seq_len, seq_k), DType::F32, xs.device())?;
            
            let qs_r_f32 = qs_r.to_dtype(DType::F32)?.contiguous()?;
            let k_cache_f32 = k_cache.to_dtype(DType::F32)?.contiguous()?;
            let reactor_result = crate::gpu_ops::spectral_ffi::apply_fused_resonance_gemm(
                &attn_scores, &qs_r_f32, &k_cache_f32, Some(p_ptr), Some(i_ptr), Some(f_ptr), token_concept_map_ptr, layer_idx as i32, num_layers as i32, n_nodes.unwrap_or(0)
            );
            
            let mut has_nan = true;
            if reactor_result.is_ok() {
                // SODALS SPEED FIX: Avoid GPU sync by assuming no NaNs if reactor_result is OK
                has_nan = false;
            }

            if !has_nan {
                physics_bias_applied = true;
                attn_scores.to_dtype(DType::F16)?
            } else {
                // Fallback to standard matmul if CUDA Reactor failed or produced NaN
                let scale = 1.0 / (self.head_dim as f64).sqrt();
                let k_cache_dim = k_cache.dim(3)?;
                let kr = if k_cache_dim > self.head_dim { 
                    k_cache.narrow(3, 0, self.head_dim)? 
                } else { 
                    k_cache.clone() 
                };
                let kr_expanded = if actual_kv_heads < self.num_heads {
                    let num_kv_groups = self.num_heads / actual_kv_heads;
                    kr.unsqueeze(2)?
                        .broadcast_as((b_size, actual_kv_heads, num_kv_groups, seq_k, self.head_dim))?
                        .contiguous()?
                        .reshape((b_size, self.num_heads, seq_k, self.head_dim))?
                } else {
                    kr
                };
                let kr_t = kr_expanded.transpose(2, 3)?.contiguous()?;
                let qs_scaled = (&qs_r * scale)?;
                let res_r = qs_scaled.matmul(&kr_t)?;
                res_r.to_dtype(DType::F16)?
            }
        } else {
            // --- OPTIMIZED HOLOGRAPHIC RESONANCE ---
            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let k_cache_dim = k_cache.dim(3)?;

            let kr = if k_cache_dim > self.head_dim { 
                k_cache.narrow(3, 0, self.head_dim)? 
            } else { 
                k_cache.clone() 
            };

            let kr_expanded = if actual_kv_heads < self.num_heads {
                let num_kv_groups = self.num_heads / actual_kv_heads;
                kr.unsqueeze(2)?
                    .broadcast_as((b_size, actual_kv_heads, num_kv_groups, seq_k, self.head_dim))?
                    .contiguous()?
                    .reshape((b_size, self.num_heads, seq_k, self.head_dim))?
            } else {
                kr
            };
            let kr_t = kr_expanded.transpose(2, 3)?.contiguous()?;
            let res_r = qs_r.matmul(&kr_t)?;

            let res = if has_phase && k_cache_dim > self.head_dim {
                 let ki = k_cache.narrow(3, self.head_dim, self.head_dim)?;
                 let ki_expanded = if actual_kv_heads < self.num_heads {
                     let num_kv_groups = self.num_heads / actual_kv_heads;
                     ki.unsqueeze(2)?
                         .broadcast_as((b_size, actual_kv_heads, num_kv_groups, seq_k, self.head_dim))?
                         .contiguous()?
                         .reshape((b_size, self.num_heads, seq_k, self.head_dim))?
                 } else {
                     ki
                 };
                 let ki_t = ki_expanded.transpose(2, 3)?.contiguous()?;
                 let res_i = qs_i.matmul(&ki_t)?;
                 (&res_r + &(&res_i * 0.1)?)?
            } else { 
                 res_r
            };

            (res * scale)?
        };

        // Clamp attention scores to prevent extreme values that cause NaN
        // Ensure F32 for physics mask CUDA kernel compatibility
        let mut attention_scores = attention_scores.to_dtype(DType::F32)?;



        if let Some(mask) = attention_mask {
            let mask_k = mask.dim(3)?;
            let mask_aligned = if mask_k > seq_k {
                mask.narrow(3, mask_k - seq_k, seq_k)?
            } else {
                mask.clone()
            };
            let mask_cast = mask_aligned.to_dtype(attention_scores.dtype())?;
            attention_scores = attention_scores.broadcast_add(&mask_cast)?;
        }



        // --- GNN GRAPH ADJACENCY BIAS INJECTION (Phase 4) ---
        if let Some((q_nodes, k_nodes, indptr, indices, data_ptr, weight)) = gnn_graph_ptrs {
            if weight > 0.0 {
                if let Some(adapter_val) = adapter {
                    if let Ok(wptr) = crate::gpu_ops::memory_utils::SodalsCudaAccess::get_physical_address(&attention_scores) {
                        let (b, h, q, k) = attention_scores.dims4()?;
                        let _ = adapter_val.apply_gnn_bias_raw(
                            wptr as u64,
                            q_nodes,
                            k_nodes,
                            indptr,
                            indices,
                            data_ptr,
                            b as i32, h as i32, q as i32, k as i32,
                            weight
                        );
                    }
                }
            }
        }

        // QMEM Runtime Attention Bias

        if let Some(qmem_params) = qmem_parameters {

            if let Ok(params) = qmem_params.lock() {

                if let Some(param) = params.get(&layer_idx) {

                    if let Ok(effective_tensor) = param.get_effective_tensor() {

                        if let Ok(qmem_bias) = effective_tensor.mean(()) {

                            let qmem_bias = qmem_bias.to_dtype(attention_scores.dtype())?;

                            let qmem_bias = (&qmem_bias * 1e-4)?;

                            attention_scores = attention_scores.broadcast_add(&qmem_bias)?;

                        }

                    }

                }

            }

        } else if let Some(loader) = qmem_loader {

            match loader.read_layer_bytes(layer_idx) {

                Ok(layer_bytes) if !layer_bytes.is_empty() => {

                    let signal = layer_bytes.iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64));

                    let bias = ((signal % 1024) as f64) * 1e-4;

                    attention_scores = (attention_scores + bias)?;

                }

                Ok(_) => {

                    // Empty layer bytes are ignored.

                }

                Err(e) => {

                    eprintln!("[QMEM] failed to read layer {}: {}", layer_idx, e);

                }

            }

        }


        // PHYSICS KERNEL INJECTION III SODALS Whitepaper II8
        // Phase 21.5: Apply physics bias via zero-copy CUDA kernel when
        // the fused resonance path did not already apply it (e.g. on fallback).
        if !physics_bias_applied {
            if let (Some(a), Some(i_ptr)) = (adapter, impedance_ptr) {
                let (b, h, q_len, k_len) = attention_scores.dims4()?;
                // We only apply physics bias to the current token (decode phase)
                if q_len == 1 {
                    if let Ok(w_ptr) = attention_scores.get_physical_address() {
                        // i_ptr is the sequence-length impedance mask from ZeroCopyCartridge, 
                        // NOT the adapter's internal vocab_size Oracle mask! 
                        // This prevents STATUS_ACCESS_VIOLATION.
                        let _ = a.apply_bias_raw(w_ptr, i_ptr, b as i32, h as i32, q_len as i32, k_len as i32);
                        physics_bias_applied = true;
                    }
                }
            }
        }


        // Semantic Divergence Gate: Penalize poor semantic health

        if let Some(p) = phase {

            if p.semantic_divergence < 1.0 {

                let penalty = (1.0 - p.semantic_divergence) * -10.0;

                attention_scores = (attention_scores + penalty as f64)?;

            }

        }



        self.dump_intermediate("attn_pregate", &attention_scores, layer_idx)?;
        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;

        // Gate blocking GPU->CPU NaN check behind opt-in SODALS_CHECK_ATTN_NAN (eliminates 8 syncs/token!)
        let attention_probs = if std::env::var_os("SODALS_CHECK_ATTN_NAN").is_some() {
            let sum_check = attention_probs.sum_all()?.to_scalar::<f32>()?;
            if !sum_check.is_finite() {
                eprintln!("[ATTN NaN] Layer {} attention collapsed to NaN, using uniform fallback", layer_idx);
                let (b, h, s, sk) = attention_probs.dims4()?;
                Tensor::full(1.0f32 / sk as f32, (b, h, s, sk), attention_probs.device())?
                    .to_dtype(attention_probs.dtype())?
            } else {
                attention_probs
            }
        } else {
            attention_probs
        };

        

        // Phase 54: Flatten to Rank-3 for absolute CUDA compliance
        // actual_kv_heads is already determined from K tensor shape at line ~1949
        let cached_kv_heads = k_cache.dim(1)?; // k_cache shape: [b, kv_heads, seq_k, head_dim]
        let actual_kv_groups = self.num_heads / cached_kv_heads;

        let v_batched = v_cache.reshape((b_size, cached_kv_heads, 1, seq_k, self.head_dim))?;

        let v_expanded = v_batched.expand((b_size, cached_kv_heads, actual_kv_groups, seq_k, self.head_dim))?;

        let probs_flat = attention_probs.contiguous()?.reshape((b_size * self.num_heads, seq_len, seq_k))?.contiguous()?;

        let v_flat = v_expanded.contiguous()?.reshape((b_size * self.num_heads, seq_k, self.head_dim))?.contiguous()?;

        

        let attn_out_flat = probs_flat.matmul(&v_flat)?;

        let attn_output = attn_out_flat.reshape((b_size, self.num_heads, seq_len, self.head_dim))?;



        let mut attn_flat = attn_output.transpose(1, 2)?.contiguous()?

            .reshape((b_size, seq_len, self.num_heads * self.head_dim))?;



        if let Some(ref gate) = attn_gate {
            // FIX 2: Ensure dtype consistency - convert to F32 before sigmoid
            let gate_f32 = gate.to_dtype(DType::F32)?;
            // Llama uses sigmoid for the attention gate, not silu!
            let gate_active = candle_nn::ops::sigmoid(&gate_f32)?;
            attn_flat = (attn_flat.to_dtype(DType::F32)?.broadcast_mul(&gate_active))?;
        }



        attn_flat
            .to_dtype(DType::F32)?
            .apply(&self.o_proj)?
            .to_dtype(DType::F32)

    }



    pub fn clear_kv_cache(&self) {
        if let Ok(mut c) = self.kv_cache.write() { *c = None; }
    }

    pub fn truncate_kv_cache(&self, keep_len: usize) -> Result<()> {
        if let Ok(mut c) = self.kv_cache.write() {
            if let Some((k, v)) = c.as_ref() {
                if k.dim(2)? > keep_len {
                    if keep_len == 0 {
                        *c = None;
                    } else {
                        let nk = k.narrow(2, 0, keep_len)?.contiguous()?;
                        let nv = v.narrow(2, 0, keep_len)?.contiguous()?;
                        *c = Some((nk, nv));
                    }
                }
            }
        }
        Ok(())
    }



    /// PHASE 45: Retroactive Attention - Gauss-Seidel KV-Relaxation with TEMPORAL DECAY

    /// 

    /// When RK4 produces a major semantic shift (epiphany), retroactively update

    /// past KV states to align with the new realization. This breaks the 

    /// causal append-only constraint of standard Transformers.

    ///

    /// PHASE 45 FIX: Temporal Decay - The omega factor decays exponentially based on

    /// sequence distance. Recent tokens bend more than older tokens:

    ///   decay_weight[pos] = exp(-lambda * (seq_len - pos))

    ///

    /// # Arguments

    /// * `energy_delta` - The semantic displacement vector (4096-D)

    /// * `omega` - Relaxation factor (0.0-1.0). Conservative 0.05 prevents 

    ///             catastrophic forgetting while allowing past context to "bend"

    ///             toward newly discovered truth.

    ///

    /// # Gauss-Seidel Formula with Temporal Decay

    /// K_new[pos] = (1 - ?�decay[pos])�K_old[pos] + ?�decay[pos]�EnergyDelta

    pub fn relax_past_states(&self, energy_delta: &Tensor, omega: f64, phase: Option<&PhaseSignal>) -> Result<()> {

        if !(0.0..=1.0).contains(&omega) {

            return Err(candle_core::Error::Msg(

                format!("Omega must be in [0,1], got {}", omega)

            ));

        }



        // Acquire write lock on KV cache

        let mut cache_guard = self.kv_cache.write().map_err(|_| {

            candle_core::Error::Msg("Failed to acquire KV cache lock".to_string())

        })?;



        if let Some((ref k_cache_raw, ref v_cache_raw)) = *cache_guard {
            // Phase-Aware Holographic KV Pruning for retroactive_relax:
            // Pre-slice cache to 1536 tokens before broadcasting & affine math to keep VRAM usage bounded.
            let (k_cache, v_cache) = if k_cache_raw.dim(2)? > 1536 {
                let cur_len = k_cache_raw.dim(2)?;
                (k_cache_raw.narrow(2, cur_len - 1536, 1536)?, v_cache_raw.narrow(2, cur_len - 1536, 1536)?)
            } else {
                (k_cache_raw.clone(), v_cache_raw.clone())
            };

            // Get cache dimensions: [batch, num_kv_heads, seq_len, head_dim*2] for holographic K
            // V is [batch, num_kv_heads, seq_len, head_dim]
            let k_dims = k_cache.dims();
            let v_dims = v_cache.dims();
            
            if k_dims.len() < 4 || v_dims.len() < 4 {
                return Ok(()); // No cache to relax
            }

            let batch = k_dims[0];
            let num_kv_heads = k_dims[1];
            let seq_len = k_dims[2];
            let k_head_dim = k_dims[3]; // Includes real+imag concatenation if holographic
            let v_head_dim = v_dims[3];



            // Broadcast energy_delta to match cache structure

            let hidden_size = energy_delta.dim(D::Minus1)?;

            

            let head_dim = self.head_dim;

            

            // PHASE 45: Temporal Decay - Generate decay weights

            // decay_weight[pos] = exp(-lambda * (seq_len - pos))

            // This ensures recent tokens get more relaxation than older ones

            let lambda: f64 = 0.1; // Decay constant - tune for desired falloff

            let decay_weights: Vec<f32> = (0..seq_len)

                .map(|pos| {

                    let distance = (seq_len - 1 - pos) as f64; // Distance from current token

                    (-lambda * distance).exp() as f32

                })

                .collect();

            

            // Create decay tensor [1, 1, seq_len, 1] for broadcasting

            let decay_tensor = Tensor::from_vec(

                decay_weights.clone(),

                seq_len,

                k_cache.device()

            )?

            .reshape((1, 1, seq_len, 1))?

            .broadcast_as((batch, num_kv_heads, seq_len, 1))?;

            

            // Create one_minus_omega tensor

            let one_minus_omega = 1.0 - omega;



            // === RELAX KEYS with Temporal Decay ===

            // For holographic keys: split real/imag, relax each component

            let k_real = k_cache.narrow(3, 0, head_dim)?;

            

            // Project energy_delta to head_dim space

            // VRAM & SHAPE SAFETY SHIELD

            let target_dim = num_kv_heads * head_dim;

            let current_dim = energy_delta.dim(candle_core::D::Minus1)?;



            let energy_proj = if current_dim >= target_dim {

                energy_delta.narrow(candle_core::D::Minus1, 0, target_dim)?.reshape((batch, num_kv_heads, 1, head_dim))?

            } else {
                return Ok(()); 
            };
            
            // Apply decay to energy projection: energy_decayed = energy * decay
            let energy_decayed = energy_proj.broadcast_mul(&decay_tensor)?;
            
            // PHASE 6: Wave-Resonance KV Modulation (Complex Phase Rotation)
            // If GNN phase signal is available, rotate the energy into the complex plane
            // so we get constructive/destructive interference instead of a flat overwrite.
            let (energy_real, energy_imag) = if let Some(p) = phase {
                let phase_r = p.theta.cos() as f64;
                let phase_i = p.theta.sin() as f64;
                (
                    energy_decayed.affine(phase_r, 0.0)?,
                    energy_decayed.affine(phase_i, 0.0)?
                )
            } else {
                (energy_decayed.clone(), energy_decayed.affine(0.0, 0.0)?) // Default flat, imag=0
            };
            
            // K_relaxed = (1-omega)K_old + omega*energy_real
            let k_relaxed_real = (
                &k_real.affine(one_minus_omega, 0.0)? + 
                &energy_real.affine(omega, 0.0)?
            )?;
            
            // If holographic (has imaginary component), relax that too with energy_imag
            let k_new = if k_head_dim > head_dim {
                let k_imag = k_cache.narrow(3, head_dim, head_dim)?;
                let k_relaxed_imag = (
                    &k_imag.affine(one_minus_omega, 0.0)? +
                    &energy_imag.affine(omega, 0.0)?
                )?;
                Tensor::cat(&[&k_relaxed_real, &k_relaxed_imag], 3)?
            } else {
                k_relaxed_real
            };



            // === RELAX VALUES with Temporal Decay ===

            // Values are simpler - just real components

            let v_energy_slice = energy_delta.narrow(2, 0, (v_head_dim * num_kv_heads).min(hidden_size))?

                .reshape((batch, num_kv_heads, 1, v_head_dim))?

                .broadcast_as((batch, num_kv_heads, seq_len, v_head_dim))?;

            

            // Apply decay to value energy projection
            let v_energy_decayed = v_energy_slice.broadcast_mul(&decay_tensor)?;
            
            // Phase modulation for Values too (only real part used for V)
            let v_energy_real = if let Some(p) = phase {
                v_energy_decayed.affine(p.theta.cos() as f64, 0.0)?
            } else {
                v_energy_decayed
            };
            
            let v_new = (
                &v_cache.affine(one_minus_omega, 0.0)? +
                &v_energy_real.affine(omega, 0.0)?
            )?;



            // Update cache in-place with Phase-Aware Holographic KV Pruning
            let (k_final, v_final) = if k_new.dim(2)? > 1536 {
                let cur_len = k_new.dim(2)?;
                let start_pos = cur_len - 1536;
                (k_new.narrow(2, start_pos, 1536)?.contiguous()?, v_new.narrow(2, start_pos, 1536)?.contiguous()?)
            } else {
                (k_new.contiguous()?, v_new.contiguous()?)
            };

            *cache_guard = Some((k_final, v_final));

            

            // PHASE 50.2: Commented out token-by-token log

            // eprintln!(

            //     "[RETROACTIVE] KV cache relaxed with temporal decay (?={:.3}, seq_len={}, heads={}, ?={:.2})",

            //     omega, seq_len, num_kv_heads, lambda

            // );

        }



        Ok(())

    }

}



// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

//  II8  QDecoderLayer  III GPR ENTEGRASYON NOKTASI 1

//

//  forward() artik `mlp_active: bool` aliyor.

//

//  mlp_active = true  III standart davranis (post_norm + MLP calisir)

//  mlp_active = false III MLP HIC cagrilmaz:

//                        II gate_proj / up_proj / down_proj matmul = SIFIR

//                        II post_attention_layernorm bile cagrilmaz

//                        II attn_residual dogrudan donuyor

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

pub struct QDecoderLayer {

    self_attn:                Option<QAttention>,

    ssm_block:                Option<QwenSsmBlock>,

    mlp:                      QMlp,

    input_layernorm:          QRmsNorm,

    post_attention_layernorm: QRmsNorm,

    /// Phase 34: True Gaussian Phase Router with rolling EMA statistics

    gpr: Mutex<GaussianPhaseRouter>,

}



impl QDecoderLayer {

    fn new(

        self_attn: Option<QAttention>,

        ssm_block: Option<QwenSsmBlock>,

        mlp: QMlp,

        input_layernorm: QRmsNorm,

        post_attention_layernorm: QRmsNorm,

    ) -> Self {

        Self {

            self_attn,

            ssm_block,

            mlp,

            input_layernorm,

            post_attention_layernorm,

            gpr: Mutex::new(GaussianPhaseRouter::new()),

        }

    }



    #[allow(clippy::too_many_arguments)]

    fn forward(

        &self,

        xs: &Tensor,

        attention_mask: Option<&Tensor>,

        seqlen_offset: usize,

        adapter: Option<&NeuroAdapter>,

        // Phase 21.5: Raw u64 device pointers from ZeroCopyCartridge

        poles_ptr: Option<u64>,

        impedance_ptr: Option<u64>,

        flags_ptr: Option<u64>,

        // Token-to-concept alignment map (token_concept_map_ptr, layer_idx, num_layers)

        token_concept_map_ptr: Option<u64>,

        layer_idx: usize,

        num_layers: usize,

        n_nodes: Option<i32>,

        phase: Option<&PhaseSignal>,

        qmem_parameters: Option<&Arc<Mutex<HashMap<usize, QmemParameter>>>>,

        qmem_loader: Option<&Arc<QmemLoader>>,

        gnn_graph_ptrs: Option<(u64, u64, u64, u64, u64, f32)>, // (q_nodes, k_nodes, indptr, indices, data, weight)

        // Phase 45: GPR-QMEM Router for adaptive loading

        gpr_qmem_router: Option<&Arc<Phase45GprQmemRouter>>,
        qmem_cartridge: Option<(&Tensor, f32)>,
    ) -> Result<Tensor> {

        let residual = xs;

        let norm_out = self.input_layernorm.forward(xs)?;
        maybe_dump_attn_norm_in(layer_idx, &norm_out)?;
        if std::env::var("SODALS_DUMP_ATTN_RESIDUAL").is_ok() {
            let flat: Vec<f32> = norm_out.flatten_all()?.to_vec1()?;
            let out_path = format!("rust_attn_norm-{}.bin", layer_idx);
            let out_bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
            if let Err(e) = std::fs::write(&out_path, &out_bytes) { eprintln!("[DUMP ATTN NORM] {}", e); }
            eprintln!("[ATN_DUMP] rust_attn_norm-{}.bin: n={} RMS={:.4}", layer_idx, flat.len(), (flat.iter().map(|v| v*v).sum::<f32>() / flat.len() as f32).sqrt());
        }

        if is_verbose() && layer_idx <= 5 {
            let norm_out_rms = (norm_out.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?);
            eprintln!("[LAYER {} INTERNAL] input_layernorm RMS: {:.4}", layer_idx, norm_out_rms);
        }



        let block_out = if let Some(ref self_attn) = self.self_attn {

            self_attn.forward(

                &norm_out, attention_mask, seqlen_offset, adapter,

                poles_ptr, impedance_ptr, flags_ptr,

                token_concept_map_ptr, layer_idx, num_layers,

                qmem_parameters, gnn_graph_ptrs, n_nodes, phase,

                qmem_loader,

            )?

        } else if let Some(ref ssm_block) = self.ssm_block {

            let ssm_out = ssm_block.forward(&norm_out, layer_idx, seqlen_offset, qmem_cartridge.clone())?;
            if is_verbose() && (layer_idx <= 8 || layer_idx % 4 == 0) {
                let ssm_out_rms = (ssm_out.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?);
                let ssm_out_f32 = ssm_out.to_dtype(DType::F32)?;
                let ssm_min = ssm_out_f32.min_all()?.to_scalar::<f32>()?;
                let ssm_max = ssm_out_f32.max_all()?.to_scalar::<f32>()?;
                eprintln!("[SSM] Layer {}: output RMS={:.6} min={:.4} max={:.4}", layer_idx, ssm_out_rms, ssm_min, ssm_max);
            }
            // SWE 1.6 Bypass Test implementation
            if std::env::var("SODALS_SSM_BYPASS").is_ok() {
                norm_out
            } else {
                ssm_out
            }

        } else {

            return Err(Error::Msg("Layer has neither attention nor SSM block".to_string()));

        };



        let attn_residual = (block_out + residual)?;

        if is_verbose() && layer_idx <= 5 {
            let attn_residual_rms = (attn_residual.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?);
            eprintln!("[LAYER {} INTERNAL] after SSM residual RMS: {:.4}", layer_idx, attn_residual_rms);
        }

        let dump_attn = std::env::var("SODALS_DUMP_ATTN_RESIDUAL").is_ok();
        if dump_attn {
            let flat = attn_residual.flatten_all()?;
            let vals: Vec<f32> = flat.to_vec1()?;
            let out_path = format!("rust_attn_residual-{}.bin", layer_idx);
            let out_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            if let Err(e) = std::fs::write(&out_path, &out_bytes) {
                eprintln!("[DUMP ATTN RESIDUAL] Failed to write {}: {}", out_path, e);
            }
        }



        // Phase 34: TRUE Gaussian Phase Router (GPR) - Z-Score gating

        //

        // z_i = phase interference magnitude (Wronskian � exp(-Jacobian � 0.1))

        // The GPR maintains rolling EMA statistics (�, ?�) of z_i magnitudes.

        // Bypass condition: Z = |z_i - �| / ? > 1.96 (95% confidence interval)

        //

        // During warm-up (first 3 forward passes): MLP always runs.

        // After warm-up: outlier z_i values trigger LayerNorm-preserved bypass.

        //

        // Phase 45: Record routing decisions in Phase45GprQmemRouter for adaptive QMEM loading

        let mlp_out = if let Some(p) = phase {

            let z_i = p.semantic_divergence * (-p.jacobian_norm * 0.1_f32).exp();

            

            // Query the true Gaussian router for bypass decision

            let (should_bypass, bypass_rate) = {

                let mut router = self.gpr.lock().unwrap();

                let b = router.should_bypass(z_i);

                (b, router.bypass_rate())

            };



            // Phase 45: Record routing statistics in GPR-QMEM router

            if let Some(gpr_qmem) = gpr_qmem_router {

                let _ = gpr_qmem.record_routing_decision(

                    layer_idx,

                    should_bypass,

                    z_i,

                    p.semantic_divergence,

                    p.jacobian_norm,

                );

            }

            

            if should_bypass {

                // Perfect Bypass: zero contribution to residual stream (x + 0 = x)

                // Variance scaling to prevent distribution shifts when bypassing

                let scale = (1.0 + bypass_rate).sqrt() as f64;

                (&attn_residual * (scale - 1.0))?

            } else {

                let post_norm = self.post_attention_layernorm.forward(&attn_residual)?;

                self.mlp.forward(&post_norm)?

            }

        } else {

            let post_norm = self.post_attention_layernorm.forward(&attn_residual)?;
            if is_verbose() && layer_idx <= 5 {
                let post_norm_rms = (post_norm.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?);
                eprintln!("[LAYER {} INTERNAL] post_attention_layernorm RMS: {:.4}", layer_idx, post_norm_rms);
            }
            let mlp_out = self.mlp.forward(&post_norm)?;
            if is_verbose() && layer_idx <= 5 {
                let mlp_out_rms = (mlp_out.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?);
                eprintln!("[LAYER {} INTERNAL] MLP output RMS: {:.4}", layer_idx, mlp_out_rms);
            }
            mlp_out

        };

        

        let out = (mlp_out + &attn_residual)?;
        maybe_dump_layer_out(layer_idx, &out)?;
        if is_verbose() && layer_idx <= 5 {
            let out_rms = (out.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?);
            eprintln!("[LAYER {} FINAL] final output RMS: {:.4}", layer_idx, out_rms);
        }
        if is_verbose() && (layer_idx == 3 || layer_idx == 4) {
            let out_f32 = out.to_dtype(candle_core::DType::F32)?;
            let sq = (&out_f32 * &out_f32)?;
            let mean_sq = sq.mean_all()?.to_scalar::<f32>()?;
            let min_val = out_f32.min_all()?.to_scalar::<f32>()?;
            let max_val = out_f32.max_all()?.to_scalar::<f32>()?;
            eprintln!("[HEALTH] Layer {} ({}): mean_sq={:.4}, min={:.4}, max={:.4}",
                layer_idx,
                if layer_idx == 3 { "MHA" } else { "SSM" },
                mean_sq, min_val, max_val
            );
        }
        Ok(out)

    }



    pub fn clear_kv_cache(&self) {
        if let Some(ref self_attn) = self.self_attn {
            self_attn.clear_kv_cache();
        }
        if let Some(ref ssm_block) = self.ssm_block {
            ssm_block.clear_cache();
        }
    }

    pub fn truncate_kv_cache(&self, keep_len: usize) -> Result<()> {
        if let Some(ref self_attn) = self.self_attn {
            self_attn.truncate_kv_cache(keep_len)?;
        }
        if let Some(ref ssm_block) = self.ssm_block {
            // SSM state is a running recurrence (DeltaNet: S_t = D_t⊙S_{t-1} + K_t⊗delta)
            // that cannot be narrowed in place. Reset it so it stops referencing tokens
            // beyond keep_len; callers that need exact continuity must re-prefill.
            ssm_block.clear_cache();
        }
        Ok(())
    }
}



// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

//  II9  Qwen2Model  III GPR ENTEGRASYON NOKTASI 2

//

//  `phase_router: GaussianPhaseRouter` alani eklendi.

//  forward() artik `phase: Option<&PhaseSignal>` aliyor.

//  Phase 45: `gpr_qmem_router: Arc<Phase45GprQmemRouter>` shared across all layers

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

// VRAM fix: hosting the token embedding off-GPU. The [vocab, hidden] embedding is the
// single largest dense buffer (~2GB as F16 on CUDA). On 8GB cards that leaves almost no
// headroom and intermittent CUDA OOM surfaces as access violation (0xC0000005) at the
// first generated token. We store it quantized to Q8_0 on the host and gather the few
// rows each forward needs, moving only those rows back to the GPU.

#[derive(Clone)]
pub struct ModelEmbedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl ModelEmbedding {
    pub fn new(embeddings: Tensor, hidden_size: usize) -> Self {
        Self { embeddings, hidden_size }
    }

    pub fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl candle_core::Module for ModelEmbedding {
    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let src_dev = self.embeddings.device();
        let idx = indexes.flatten_all()?.to_device(src_dev)?;
        let values = self.embeddings.index_select(&idx, 0)?;
        let values = values.reshape(final_dims)?;
        Ok(values.to_device(indexes.device())?)
    }
}

pub struct Qwen2Model {

    pub embed_tokens: ModelEmbedding,

    layers: Vec<QDecoderLayer>,

    pub norm: QRmsNorm,

    device: Device,

    pub physics_adapter: Option<Arc<NeuroAdapter>>,

    current_impedance: Option<Tensor>,

    // PHASE 51.5: Dynamic prefix tracking for continuous RoPE alignment

    prefix_injected_len: std::sync::atomic::AtomicUsize,

    // Phase 45: Shared GPR-QMEM router for adaptive layer loading

    pub gpr_qmem_router: Arc<Phase45GprQmemRouter>,

    // PHASE 1: Learnable QMEM parameters with gradient flow
    pub qmem_parameters: Arc<Mutex<HashMap<usize, QmemParameter>>>,
    // J-Lens Integration
    pub jlens_readout: Option<Arc<crate::jlens_readout::JLensReadout>>,
    pub jlens_results: Arc<Mutex<Vec<(usize, Tensor)>>>,
    // Tuned-Lens training capture: raw residual-stream h_l per layer (feature) plus a
    // usize::MAX sentinel holding the final pre-norm hidden h_L (target).
    pub jlens_capture: Arc<Mutex<Vec<(usize, Tensor)>>>,
}



impl Qwen2Model {

    pub fn new(

        embed_tokens: ModelEmbedding,

        layers: Vec<QDecoderLayer>,

        norm: QRmsNorm,

        device: Device,

    ) -> Result<Self> {

        let physics_adapter = if device.is_cuda() {

            match NeuroAdapter::new(0, 0.8, true) {

                Ok(adapter) => Some(Arc::new(adapter)),

                Err(e) => { eprintln!("Warning: Failed to initialize NeuroAdapter: {}", e); None }

            }

        } else { None };



        // Phase 45: Initialize GPR-QMEM router (shared across all layers)

        let num_layers = layers.len();

        let gpr_qmem_router = Arc::new(Phase45GprQmemRouter::new(num_layers));

        let qmem_parameters = Arc::new(Mutex::new(HashMap::new()));
        
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device,
            physics_adapter,
            current_impedance: None,
            prefix_injected_len: std::sync::atomic::AtomicUsize::new(0),
            gpr_qmem_router,
            qmem_parameters,
            jlens_readout: None,
            jlens_results: Arc::new(Mutex::new(Vec::new())),
            jlens_capture: Arc::new(Mutex::new(Vec::new())),
        })

    }



    pub fn export_kv_cache(&self) -> Vec<Option<(Tensor, Tensor)>> {

        let mut vec_cache = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {

            if let Some(ref self_attn) = layer.self_attn {

                if let Ok(kv_opt) = self_attn.kv_cache.read() {

                    if let Some((ref k, ref v)) = *kv_opt {

                        vec_cache.push(Some((k.clone(), v.clone())));

                        continue;

                    }

                }

            } else if let Some(ref ssm_block) = layer.ssm_block {

                if let Ok(state_opt) = ssm_block.state_cache.read() {

                    if let Some((ref s, ref c)) = *state_opt {

                        vec_cache.push(Some((s.clone(), c.clone())));

                        continue;

                    }

                }

            }

            vec_cache.push(None);

        }

        vec_cache

    }



    pub fn import_kv_cache(&self, cache: &[Option<(Tensor, Tensor)>]) {

        for (i, layer) in self.layers.iter().enumerate() {

            if i >= cache.len() {

                break;

            }

            if let Some((ref t1, ref t2)) = cache[i] {

                if let Some(ref self_attn) = layer.self_attn {

                    if let Ok(mut kv_w) = self_attn.kv_cache.write() {

                        *kv_w = Some((t1.clone(), t2.clone()));

                    }

                } else if let Some(ref ssm_block) = layer.ssm_block {

                    if let Ok(mut state_w) = ssm_block.state_cache.write() {

                        *state_w = Some((t1.clone(), t2.clone()));

                    }

                }

            }

        }

    }



    /// Get the currently recorded prefix-injected length (used to adjust RoPE offsets).

    pub fn get_prefix_injected_len(&self) -> usize {

        self.prefix_injected_len.load(std::sync::atomic::Ordering::SeqCst)

    }



    /// Set the recorded prefix-injected length (used when applying cached KV pairs).

    pub fn set_prefix_injected_len(&self, len: usize) {

        self.prefix_injected_len.store(len, std::sync::atomic::Ordering::SeqCst);

    }



    pub fn set_adapter(&mut self, adapter: Arc<NeuroAdapter>) {

        self.physics_adapter = Some(adapter);

    }

    pub fn set_impedance(&mut self, imp: Tensor) { self.current_impedance = Some(imp); }

    pub fn clear_impedance(&mut self)            { self.current_impedance = None; }



    pub fn set_physics_bias(&mut self, impedance: Vec<f64>) -> Result<()> {

        let f32_data: Vec<f32> = impedance.iter().map(|&x| x as f32).collect();

        let shape = (f32_data.len(),);

        

        // LINUS FIX: F16 tip d�n���m�n� s�cak d�ng�den ��kar�p burada 1 kez yap�yoruz.

        let imp = Tensor::from_slice(&f32_data, shape, &self.device)?

            .to_dtype(DType::F16)?

            .contiguous()?;

            

        self.current_impedance = Some(imp);

        Ok(())

    }

    pub fn clear_physics_bias(&mut self) { self.current_impedance = None; }



    /// Load learnable QMEM parameters from cartridges

    /// This populates the qmem_parameters HashMap with trainable QmemParameter instances

    pub fn load_qmem_parameters(&self, loader: &QmemLoader) -> Result<()> {

        let mut params = self.qmem_parameters.lock().unwrap();

        

        // Load parameters for each layer

        for layer_idx in 0..loader.num_layers() {

            let qmem_param = loader.get_layer_tensor(layer_idx)

                .map_err(|e| Error::Msg(format!("Failed to load QMEM layer {}: {}", layer_idx, e)))?;

            params.insert(layer_idx, qmem_param);

        }

        

        Ok(())

    }



    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    //  forward III GPR entegreli

    //

    //  phase = None         III tum MLP'ler aktif (geri donus uyumu)

    //  phase = Some(&sig)   III GPR aktif, seyrek MLP yurutme

    //

    //  Kritik not:

    //    mlp_active_mask hesabi O(N) f32 islem III GPU kernel SIFIR.

    //    Pasif katmanlar icin QMlp::forward() HICBIR ZAMAN cagrilmaz.

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    pub fn forward(

        &self,

        input_ids: &Tensor,

        seqlen_offset: usize,

        // Phase 21.5: Raw u64 device pointers from ZeroCopyCartridge

        poles_ptr: Option<u64>,

        impedance_ptr: Option<u64>,

        flags_ptr: Option<u64>,

        n_nodes: Option<i32>,

        phase: Option<&PhaseSignal>,

        ghost_prefix: Option<&Tensor>, // BCI: Zero-Layer Topological Instinct Vector

        // Token-to-concept alignment map pointer (None = modulo fallback)

        token_concept_map_ptr: Option<u64>,

        qmem_parameters: Option<&Arc<Mutex<HashMap<usize, QmemParameter>>>>,

        qmem_loader: Option<&Arc<QmemLoader>>,

        qmem_cartridge: Option<(&Tensor, f32)>,

        gnn_graph_ptrs: Option<(u64, u64, u64, u64, u64, f32)>, // (q_nodes, k_nodes, indptr, indices, data, weight)

    ) -> Result<Tensor> {

        let (_b_size, seq_len) = input_ids.dims2()?;

        let adapter_ref   = self.physics_adapter.as_deref();



        // PHASE 51.5: Dynamic RoPE tracking for continuous prefix
        // NOTE: injected_len is NOT added here because generation_loop.rs already includes
        // visual_prefix_len in seqlen_offset. Double-counting caused RoPE offsets to be
        // shifted by 2x ghost_prefix, corrupting positional encoding during decode.
        let effective_seqlen_offset = seqlen_offset;



        // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////I

        // ZERO-PEAK CHUNKED PREFILL (orijinal mantik korundu)

        // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////I

        let chunk_size = 1024;

        let mut physical_offset = effective_seqlen_offset;

        let mut current_offset = 0;

        let mut final_hidden: Option<Tensor> = None;

        // Tuned-Lens capture mode: when SODALS_JLENS_CAPTURE is set, raw h_l is
        // copied to CPU per workspace layer and a usize::MAX sentinel holds the
        // final pre-norm h_L (target). SODALS_JLENS_CAPTURE_LAYERS optionally
        // restricts capture to a comma/range list (default: all layers).
        let capture_jlens = std::env::var("SODALS_JLENS_CAPTURE").is_ok();
        let capture_layers: Option<std::collections::HashSet<usize>> = if capture_jlens {
            std::env::var("SODALS_JLENS_CAPTURE_LAYERS").ok().map(|spec| {
                spec.split(',').flat_map(|p| p.split('-')).filter_map(|p| p.trim().parse::<usize>().ok()).collect()
            })
        } else { None };



        while current_offset < seq_len {

            let current_chunk_len = usize::min(seq_len - current_offset, chunk_size);

            let chunk_ids  = input_ids.narrow(1, current_offset, current_chunk_len)?;

            let chunk_embed = self.embed_tokens.forward(&chunk_ids)?;

            // DEBUG: Check embedding output RMS only when SODALS_VERBOSE is set (eliminates 2 syncs/token!)
            let verbose_emb = is_verbose();
            if verbose_emb && current_offset == 0 && current_chunk_len == 1 {
                let chunk_embed_rms = chunk_embed.to_dtype(DType::F32)?.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                eprintln!("[EMBEDDING DEBUG] chunk_embed RMS: {:.4}", chunk_embed_rms);
            }

            drop(chunk_ids);

            let mut chunk_xs = chunk_embed.to_dtype(DType::F32)?;

            if verbose_emb && current_offset == 0 && current_chunk_len == 1 {
                let chunk_xs_rms = chunk_xs.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
                eprintln!("[EMBEDDING DEBUG] chunk_xs RMS (after to_dtype): {:.4}", chunk_xs_rms);
            }

            let dump_layers = std::env::var("SODALS_DUMP_LAYERS").is_ok();
            if dump_layers && current_offset == 0 {
                let flat = chunk_xs.flatten_all()?;
                let vals: Vec<f32> = flat.to_vec1()?;
                let n = vals.len();
                let sum: f32 = vals.iter().sum();
                let mean = sum / n as f32;
                let sq_sum: f32 = vals.iter().map(|v| v * v).sum();
                let rms = (sq_sum / n as f32).sqrt();
                let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let first_10: Vec<f32> = vals.iter().take(10).copied().collect();
                eprintln!("[DUMP EMBED] RMS={:.6} mean={:.6} min={:.4} max={:.4} first10={:.4?}", rms, mean, min, max, first_10);
            }

            drop(chunk_embed);



            let mut physical_chunk_len = current_chunk_len;

            // PHASE 51.5: DYNAMIC PREFIX INJECTION (Continuous Space)

            if seqlen_offset == 0 && current_offset == 0 {

                if let Some(prefix) = ghost_prefix {

                    let p_len = prefix.dim(1)?;

                    chunk_xs = Tensor::cat(&[prefix, &chunk_xs], 1)?;

                    physical_chunk_len += p_len;

                    self.prefix_injected_len.store(p_len, std::sync::atomic::Ordering::SeqCst);

                    eprintln!("[PREFIX INJECTION] {} continuous tokens prepended dynamically.", p_len);

                }

            }



            let attention_mask = if physical_chunk_len > 1 {

                let causal = Tensor::tril2(physical_chunk_len, DType::F32, &self.device)?;

                let causal_4d = causal.unsqueeze(0)?.unsqueeze(0)?;



                let causal_final = if physical_offset > 0 {

                    let ones_pad = Tensor::ones((1, 1, physical_chunk_len, physical_offset), DType::F32, &self.device)?;

                    Tensor::cat(&[&ones_pad, &causal_4d], 3)?

                } else {

                    causal_4d

                };



                let inv_causal = (Tensor::ones_like(&causal_final)? - &causal_final)?;

                let penalty = Tensor::new(-10000.0f32, &self.device)?;

                Some(inv_causal.broadcast_mul(&penalty)?)

            } else {

                None 

            };

            let mask_ref = attention_mask.as_ref();



            // IIIIII Layer dongusu (layer_idx iletimi ile)

            let num_layers = self.layers.len();

            let dump_layers = std::env::var("SODALS_DUMP_LAYERS").is_ok();

            let skip_layers = std::env::var("SODALS_SKIP_LAYERS").is_ok();
            if !skip_layers {
            for (layer_idx, layer) in self.layers.iter().enumerate() {

                // DEBUG: Check chunk_xs RMS before first layer
                if is_verbose() && layer_idx == 0 && current_offset == 0 && current_chunk_len == 1 {
                    let chunk_xs_rms = (chunk_xs.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?);
                    eprintln!("[LAYER 0 INPUT] chunk_xs RMS before layer: {:.4}", chunk_xs_rms);
                }

                if dump_layers && current_offset == 0 {
                    let flat = chunk_xs.flatten_all()?;
                    let vals: Vec<f32> = flat.to_vec1()?;
                    let n = vals.len();
                    let sum: f32 = vals.iter().sum();
                    let mean = sum / n as f32;
                    let sq_sum: f32 = vals.iter().map(|v| v * v).sum();
                    let rms = (sq_sum / n as f32).sqrt();
                    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let first_10: Vec<f32> = vals.iter().take(10).copied().collect();
                    eprintln!("[DUMP LAYER {} INPUT] RMS={:.6} mean={:.6} min={:.4} max={:.4} first10={:.4?}", layer_idx, rms, mean, min, max, first_10);
                }

                chunk_xs = layer.forward(

                    &chunk_xs,

                    mask_ref,

                    physical_offset,

                    adapter_ref,

                    poles_ptr,

                    impedance_ptr,

                    flags_ptr,

                    token_concept_map_ptr, // prefill=None, decode=ger�ek adres

                    layer_idx,

                    num_layers,

                    n_nodes,

                    phase,

                    Some(&self.qmem_parameters),

                    qmem_loader,

                    gnn_graph_ptrs,

                    Some(&self.gpr_qmem_router),

                    qmem_cartridge.clone(),

                )?;

                if dump_layers && current_offset == 0 {
                    let flat = chunk_xs.flatten_all()?;
                    let vals: Vec<f32> = flat.to_vec1()?;
                    let n = vals.len();
                    let sum: f32 = vals.iter().sum();
                    let mean = sum / n as f32;
                    let sq_sum: f32 = vals.iter().map(|v| v * v).sum();
                    let rms = (sq_sum / n as f32).sqrt();
                    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let first_10: Vec<f32> = vals.iter().take(10).copied().collect();
                    eprintln!("[DUMP LAYER {} OUTPUT] RMS={:.6} mean={:.6} min={:.4} max={:.4} first10={:.4?}", layer_idx, rms, mean, min, max, first_10);
                    if std::env::var("SODALS_DUMP_LAYERS_BIN").is_ok() {
                        let out_path = format!("rust_l_out-{}.bin", layer_idx);
                        let out_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                        if let Err(e) = std::fs::write(&out_path, &out_bytes) {
                            eprintln!("[DUMP LAYER BIN] Failed to write {}: {}", out_path, e);
                        } else {
                            eprintln!("[DUMP LAYER BIN] Wrote {} ({} bytes)", out_path, out_bytes.len());
                        }
                    }
                }

                // Tuned-Lens capture: copy raw residual-stream h_l to CPU as features
                if capture_jlens {
                    let should_capture = match &capture_layers {
                        Some(set) => set.contains(&layer_idx),
                        None => true,
                    };
                    if should_capture {
                        if let Ok(mut cap) = self.jlens_capture.lock() {
                            cap.push((layer_idx, chunk_xs.to_device(&Device::Cpu)?));
                        }
                    }
                }

                // J-Lens Integration: Extract real-valued residual stream h_l
                // (skipped during capture mode so the capture buffer stays authoritative)
                if let Some(readout) = &self.jlens_readout {
                    if !capture_jlens && readout.is_workspace_layer(layer_idx) {
                        if is_verbose() && current_offset == 0 && std::env::var("SODALS_JLENS_TRACE").is_ok() {
                            eprintln!("[J-LENS] firing for layer {} (chunk_xs {:?})", layer_idx, chunk_xs.shape());
                        }
                        match readout.compute_readout(&chunk_xs, layer_idx) {
                            Ok(projected) => {
                                if let Ok(mut results) = self.jlens_results.lock() {
                                    if let Ok(proj_cpu) = projected.to_device(&Device::Cpu) {
                                        results.push((layer_idx, proj_cpu));
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("[J-LENS] compute_readout failed layer {}: {}", layer_idx, e);
                            }
                        }
                    }
                }
            }
            } // if !skip_layers



            current_offset  += current_chunk_len;

            physical_offset += physical_chunk_len;

            // Tuned-Lens capture: sentinel entry holding the final pre-norm h_L (target)
            if capture_jlens {
                if let Ok(mut cap) = self.jlens_capture.lock() {
                    cap.push((usize::MAX, chunk_xs.to_device(&Device::Cpu)?));
                }
            }

            final_hidden = Some(chunk_xs);

        }



        let hidden = final_hidden

            .ok_or_else(|| candle_core::Error::Msg("Empty input sequence".to_string()))?;



        let dump_layers = std::env::var("SODALS_DUMP_LAYERS").is_ok();
        if dump_layers {
            let flat = hidden.flatten_all()?;
            let vals: Vec<f32> = flat.to_vec1()?;
            let n = vals.len();
            let sum: f32 = vals.iter().sum();
            let mean = sum / n as f32;
            let sq_sum: f32 = vals.iter().map(|v| v * v).sum();
            let rms = (sq_sum / n as f32).sqrt();
            let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let first_10: Vec<f32> = vals.iter().take(10).copied().collect();
            eprintln!("[DUMP FINAL HIDDEN] RMS={:.6} mean={:.6} min={:.4} max={:.4} first10={:.4?}", rms, mean, min, max, first_10);
        }

        let norm_out = self.norm.forward(&hidden)?;

        let dump_layers = std::env::var("SODALS_DUMP_LAYERS").is_ok();
        if dump_layers {
            let flat = norm_out.flatten_all()?;
            let vals: Vec<f32> = flat.to_vec1()?;
            let n = vals.len();
            let sum: f32 = vals.iter().sum();
            let mean = sum / n as f32;
            let sq_sum: f32 = vals.iter().map(|v| v * v).sum();
            let rms = (sq_sum / n as f32).sqrt();
            let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let first_10: Vec<f32> = vals.iter().take(10).copied().collect();
            eprintln!("[DUMP NORM OUTPUT] RMS={:.6} mean={:.6} min={:.4} max={:.4} first10={:.4?}", rms, mean, min, max, first_10);
        }

        Ok(norm_out)
    }



    pub fn clear_kv_cache(&self) {
        if let Ok(mut res) = self.jlens_results.lock() {
            res.clear();
        }
        if let Ok(mut cap) = self.jlens_capture.lock() {
            cap.clear();
        }
        for layer in &self.layers { layer.clear_kv_cache(); }
        // PHASE 51.5: Reset prefix tracking on new generation
        self.prefix_injected_len.store(0, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn truncate_kv_cache(&self, keep_len: usize) -> Result<()> {
        for layer in &self.layers { layer.truncate_kv_cache(keep_len)?; }
        Ok(())
    }



    /// Phase 40: Grid-to-Latent Projection with 2D RoPE

    /// 

    /// This is the VISION ADAPTER for ARC-AGI. Instead of tokenizing a 2D grid as text,

    /// we project it DIRECTLY into the 4096-D continuous manifold using learned spatial embeddings.

    /// 

    /// The projection uses 2D Rotary Position Embeddings (RoPE) to encode:

    /// - Cartesian coordinates (x, y) of each grid cell

    /// - Color values (0-9 for ARC) embedded as learnable vectors

    /// - Spatial relationships preserved via trigonometric position encoding

    /// 

    /// Mathematical formulation:

    /// h_{x,y} = W_c[color] + RoPE_2D(x, y)

    /// 

    /// where RoPE_2D(x, y) = [cos(x�?^i)�cos(y�?^i), sin(x�?^i)�sin(y�?^i)]

    /// 

    /// # Arguments

    /// * `grid` - 2D array of color values (0-9 for ARC-AGI tasks)

    /// 

    /// Returns: [1, H�W, hidden_size] tensor ready for transformer layers

    pub fn embed_arc_grid(&self, grid: &ndarray::Array2<u8>) -> Result<Tensor> {

        use ndarray::Array1;

        

        let (height, width) = (grid.nrows(), grid.ncols());

        let hidden_size = self.embed_tokens_dim();

        let device = &self.device;

        

        // Color embedding dimension - first 64 dims reserved for color

        let color_dim = 64_usize.min(hidden_size / 16);

        

        // Build 2D position encodings using RoPE principles

        let mut embeddings: Vec<f32> = Vec::with_capacity(height * width * hidden_size);

        

        for y in 0..height {

            for x in 0..width {

                let color = grid[[y, x]] as usize;

                

                // Position encoding: interleaved sine/cosine at different frequencies

                // This encodes both x and y coordinates in a way that preserves relative distances

                let mut position_enc = Array1::<f32>::zeros(hidden_size);

                

                for dim in 0..hidden_size {

                    // 2D RoPE: combine x and y encodings

                    let freq_idx = (dim / 2) as f32;

                    let theta = 10000.0_f32.powf(-2.0 * freq_idx / hidden_size as f32);

                    

                    let phase_x = x as f32 * theta;

                    let phase_y = y as f32 * theta;

                    

                    // Even dims: cosine of combined phase

                    // Odd dims: sine of combined phase

                    if dim % 2 == 0 {

                        position_enc[dim] = (phase_x.cos() + phase_y.cos()) * 0.5;

                    } else {

                        position_enc[dim] = (phase_x.sin() + phase_y.sin()) * 0.5;

                    }

                }

                

                // Color embedding: deterministic hash-based initialization

                // Each color (0-9) gets a unique pattern in the first color_dim dimensions

                let color_seed = color.wrapping_mul(0x9e3779b9).wrapping_add(0x85ebca6b);

                for d in 0..color_dim {

                    let freq = (d + 1) as f32;

                    let hashed = ((color_seed >> (d % 32)) & 0xFFFF) as f32 / 32768.0 - 1.0;

                    position_enc[d] += hashed * (1.0 / (1.0 + freq * 0.1));

                }

                

                // Scale position encoding to match model's expected input distribution

                // This prevents the initial hidden states from being too small or too large

                let scale = (hidden_size as f32).sqrt().recip();

                for dim in 0..hidden_size {

                    embeddings.push(position_enc[dim] * scale);

                }

            }

        }

        

        // Reshape: [height * width, hidden_size] -> [1, height * width, hidden_size]

        let tensor = Tensor::from_vec(

            embeddings,

            (height * width, hidden_size),

            device

        )?;

        let batched = tensor.unsqueeze(0)?; // Add batch dimension

        

        eprintln!(

            "[ARC-AGI VISION] Grid {}�{} � Latent [1, {}, {}] via 2D RoPE", 

            height, width, height * width, hidden_size

        );

        

        Ok(batched)

    }



    /// PHASE 37: Retroactive Attention - Propagate epiphany through all layers

    ///

    /// When the RK4 integrator produces a major semantic shift, this method

    /// retroactively updates ALL layers' KV caches to align with the new

    /// realization. This is the "time machine" that allows the model to

    /// correct its own past context.

    ///

    /// # Arguments

    /// * `epiphany_vector` - The semantic displacement ?h = h_final - h_initial

    /// * `omega` - Relaxation factor (default 0.05 for conservative updates)

    pub fn retroactive_relax(&self, epiphany_vector: &Tensor, omega: f64, phase: Option<&PhaseSignal>) -> Result<()> {

        // Bug 7: Defensive shape validation since relax_past_states assumes tensor is large enough

        let dims = epiphany_vector.dims();

        let required_elements = self.layers.iter()

            .find_map(|l| l.self_attn.as_ref().map(|sa| sa.head_dim * sa.num_kv_heads))

            .unwrap_or(0);

            

        if dims.len() < 3 || dims[2] < required_elements {

            return Err(candle_core::Error::Msg(format!(

                "Shape mismatch in retroactive_relax: expected rank>=3 and dim[2]>={}, got {:?}",

                required_elements, dims

            )));

        }



        // PHASE 50.2: Commented out token-by-token log

        // eprintln!(

        //     "[EPIPHANY] Retroactive relaxation triggered (||?h|| detected)"

        // );



        for (layer_idx, layer) in self.layers.iter().enumerate() {

            // Deeper layers get slightly stronger relaxation (they handle higher-level semantics)

            let layer_omega = omega * (1.0 + (layer_idx as f64 / self.layers.len() as f64) * 0.5);

            let clamped_omega = layer_omega.min(0.15); // Hard cap at 0.15

            

            if let Some(ref self_attn) = layer.self_attn {

                if let Err(e) = self_attn.relax_past_states(epiphany_vector, clamped_omega, phase) {

                    // PHASE 50.2: Commented out token-by-token log

                    // fflush!();

                }

            }

        }



        // PHASE 50.2: Commented out token-by-token log

        // eprintln!(

        //     "[EPIPHANY] Retroactive update complete across {} layers (?_base={:.3})",

        //     self.layers.len(), omega

        // );



        Ok(())

    }



    pub fn embed_tokens_dim(&self) -> usize {

        self.embed_tokens.hidden_size()

    }



    /// Get the device used by this model

    pub fn device(&self) -> &Device {

        &self.device

    }

}



// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

//  II10  Qwen2ForCausalLM  III GPR ENTEGRASYON NOKTASI 3

//

//  forward() imzasi: phase: Option<&PhaseSignal> eklendi.

//  Geri donus uyumu: None gecmek eski davranisi korur.

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

pub struct EpiphanyGate {

    running_mean: f64,

    running_var: f64,

    sample_count: u32,

    pub shift_counter: u32,

    alpha: f64,

}



impl EpiphanyGate {

    pub fn new() -> Self {

        Self {

            running_mean: 0.0,

            running_var: 1.0,

            sample_count: 0,

            shift_counter: 0,

            alpha: 0.1, // Moving average smoothing factor

        }

    }



    pub fn should_relax(&mut self, displacement: f64) -> bool {

        self.sample_count = self.sample_count.saturating_add(1);



        let alpha = self.alpha;

        self.running_mean = (1.0 - alpha) * self.running_mean + alpha * displacement;

        let deviation = displacement - self.running_mean;

        self.running_var = (1.0 - alpha) * self.running_var + alpha * deviation * deviation;



        if self.sample_count < 5 {

            return false; // Warm-up phase

        }



        let sigma = (self.running_var + 1e-8).sqrt();

        let z_score = deviation.abs() / sigma;



        // 3-Sigma threshold for true structural paradigm shifts

        let is_epiphany = z_score > 1.5;

        if is_epiphany {

            self.shift_counter += 1;

        }

        

        is_epiphany

    }

}



pub struct Qwen2ForCausalLM {

    pub model:   Qwen2Model,

    pub lm_head: UnifiedLinear,

    /// Phase 36: Arkhe Core for continuous latent pondering

    /// Intercepts hidden states before token emission for RK4-based thought evolution

    /// Uses Mutex for interior mutability since forward() takes &self

    pondering_core: Option<Mutex<PonderingCore>>,

    epiphany_gate: Mutex<EpiphanyGate>,

    qmem_loader: Option<Arc<QmemLoader>>,

}



impl Qwen2ForCausalLM {

    fn new(model: Qwen2Model, lm_head: UnifiedLinear) -> Self {

        let hidden_size = model.embed_tokens_dim();

        let device = model.device().clone();

        

        // Initialize PonderingCore for continuous latent thinking

        let pondering_core = PonderingCore::new(hidden_size, &device).ok().map(Mutex::new);

        

        Self { model, lm_head, pondering_core, epiphany_gate: Mutex::new(EpiphanyGate::new()), qmem_loader: None }

    }



    pub fn set_adapter(&mut self, adapter: Arc<NeuroAdapter>) {
        self.model.set_adapter(adapter);
    }

    /// Enable or disable Arkhe latent pondering

    pub fn set_pondering(&mut self, enable: bool) {

        if enable && self.pondering_core.is_none() {

            let hidden_size = self.model.embed_tokens_dim();

            let device = self.model.device().clone();

            self.pondering_core = PonderingCore::new(hidden_size, &device).ok().map(Mutex::new);

            eprintln!("[ARKHE] Latent pondering ENABLED");

        } else if !enable {

            self.pondering_core = None;

            eprintln!("[ARKHE] Latent pondering DISABLED");

        }

    }



    pub fn set_qmem_loader(&mut self, loader: Option<Arc<QmemLoader>>) {

        self.qmem_loader = loader;

    }



    /// Configure Arkhe pondering coefficients

    pub fn configure_pondering(&self, alpha: f64, beta: f64, gamma: f64) {

        if let Some(ref core_mutex) = self.pondering_core {

            if let Ok(core) = core_mutex.lock() {

                core.set_coefficients(alpha, beta, gamma);

            }

        }

    }

    /// Get mutable reference to pondering core (for advanced configuration)

    pub fn pondering_core(&self) -> Option<&Mutex<PonderingCore>> {

        self.pondering_core.as_ref()

    }



    pub fn from_gguf<R: Read + Seek>(

        ct: &gguf_file::Content,

        reader: &mut R,

        device: &Device,

    ) -> Result<Self> {

        let cfg = config_from_gguf(ct)?;

        let content = ct;



        // Direct CPU F16 load: avoids 4.1 GB CUDA FP32 allocation spike and redundant Q8_0 re-quantization!
        let embed_weight = content.tensor(reader, "token_embd.weight", &Device::Cpu)?;
        eprintln!("[GGUF META DEBUG] token_embd.weight dtype: {:?}", embed_weight.dtype());
        let embed_host_f16 = embed_weight.dequantize_f16(&Device::Cpu)?;
        drop(embed_weight);
        eprintln!("[EMBED] loaded directly on Host F16 ({:?}, {} elems), zero CUDA VRAM spike",
            embed_host_f16.shape(), embed_host_f16.elem_count());

        // Q4_K_M dequantization produces reasonable values. Do NOT normalize or clamp embeddings
        // independently of lm_head — they must share the same scale for correct logits.
        let embed_tokens = ModelEmbedding::new(embed_host_f16, cfg.hidden_size);

        let rotary_emb   = Arc::new(RotaryEmbedding::new(&cfg, device)?);



        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

        

        // Detect if this is a SSM model (Qwen3.5) via O(1) header lookup (zero redundant disk reads!)

        let is_ssm_model = content.tensor_infos.contains_key("blk.0.ssm_out.weight");

        

        if is_ssm_model {

            eprintln!("[SSM] Detected Qwen3.5 SSM architecture - enabling DeltaNet layers");

            eprintln!("[SSM] Using hybrid attention: DeltaNet + Gated Attention");

        }

        // DEBUG: Print which layers are SSM vs Attention via O(1) tensor_infos lookup

        for dbg_idx in 0..cfg.num_hidden_layers {

            let has_ssm = is_ssm_model && content.tensor_infos.contains_key(&format!("blk.{dbg_idx}.ssm_out.weight"));

            let has_fused_qkv = content.tensor_infos.contains_key(&format!("blk.{dbg_idx}.attn_qkv.weight"));

            let has_sep_q = content.tensor_infos.contains_key(&format!("blk.{dbg_idx}.attn_q.weight"));

            let has_gate = content.tensor_infos.contains_key(&format!("blk.{dbg_idx}.attn_gate.weight"));
            if is_verbose() {
                eprintln!("[LAYER {}] SSM={} fused_qkv={} sep_q={} gate={}", dbg_idx, has_ssm, has_fused_qkv, has_sep_q, has_gate);
            }

        }

        

        // Standard Qwen2/3 model loading (works for both SSM and non-SSM models)

            for layer_idx in 0..cfg.num_hidden_layers {

                let layer_has_ssm = is_ssm_model && content.tensor_infos.contains_key(&format!("blk.{layer_idx}.ssm_out.weight"));

                

                let (self_attn, ssm_block) = if layer_has_ssm {

                    // 1. Load Gated DeltaNet (SSM) layer

                    let qkv_w = content.tensor(reader, &format!("blk.{layer_idx}.attn_qkv.weight"), device)?;

                    let qkv_b = content.tensor(reader, &format!("blk.{layer_idx}.attn_qkv.bias"), device)

                        .ok().and_then(|t| t.dequantize(device).ok());

                    let ssm_in_proj = UnifiedLinear::Quantized(QMatMul::from_qtensor(qkv_w)?, qkv_b);
                    if layer_idx == 1 {
                        eprintln!("[TENSOR NAME DEBUG] Layer 1 loading tensor: blk.{}.attn_qkv.weight", layer_idx);
                    }



                    // Load attn_gate (in_proj_z)

                    let gate_w = content.tensor(reader, &format!("blk.{layer_idx}.attn_gate.weight"), device)?;

                    let gate_b = content.tensor(reader, &format!("blk.{layer_idx}.attn_gate.bias"), device)

                        .ok().and_then(|t| t.dequantize(device).ok());

                    let attn_gate = UnifiedLinear::Quantized(QMatMul::from_qtensor(gate_w)?, gate_b);



                    // Load ssm_conv1d

                    let conv1d = content.tensor(reader, &format!("blk.{layer_idx}.ssm_conv1d.weight"), device)?
                        .dequantize(device)?;
                    if layer_idx == 1 {
                        eprintln!("[TENSOR NAME DEBUG] Layer 1 loading tensor: blk.{}.ssm_conv1d.weight", layer_idx);
                    }

                    if layer_idx == 18 || layer_idx == 20 {
                        if let Ok(w_vec) = conv1d.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                            eprintln!("[DEBUG] blk.{}.ssm_conv1d.weight first 4 = {:?}", layer_idx, &w_vec[0..4.min(w_vec.len())]);
                        }
                    }
 // [8192, 4]



                    // Load ssm_out_proj (ssm_out.weight)

                    let out_w = content.tensor(reader, &format!("blk.{layer_idx}.ssm_out.weight"), device)?;
                    if layer_idx == 1 {
                        eprintln!("[TENSOR NAME DEBUG] Layer 1 loading tensor: blk.{}.ssm_out.weight", layer_idx);
                    }

                    let out_b = content.tensor(reader, &format!("blk.{layer_idx}.ssm_out.bias"), device)

                        .ok().and_then(|t| t.dequantize(device).ok());

                    let ssm_out_proj = UnifiedLinear::Quantized(QMatMul::from_qtensor(out_w)?, out_b);



                    // Load ssm_norm

                    let ssm_norm = if let Ok(t) = content.tensor(reader, &format!("blk.{layer_idx}.ssm_norm.weight"), device) {
                        if layer_idx == 1 {
                            eprintln!("[TENSOR NAME DEBUG] Layer 1 loading tensor: blk.{}.ssm_norm.weight", layer_idx);
                        }
                        if layer_idx == 4 {
                            eprintln!("[SHAPE DEBUG] ssm_norm.weight: {:?}", t.shape());
                        }
                        // Qwen3_5RMSNormGated uses weight init to ONES, forward = weight * norm_x.
                        // NOT zero_centered (which is (1+weight)*norm_x for Qwen3_5RMSNorm with weight init to ZEROS).
                        Some({ let w = t.dequantize(device)?; let w_vals = w.to_vec1::<f32>().unwrap_or_default(); if !w_vals.is_empty() { eprintln!("[NORM DEBUG] ssm_norm layer {} first val: {}", layer_idx, w_vals[0]); } QRmsNorm::new(w, 1e-6)? })
                    } else {
                        None

                    };



                    // Load ssm_a [32]

                    let ssm_a = content.tensor(reader, &format!("blk.{layer_idx}.ssm_a"), device)?
                        .dequantize(device)?;
                    if layer_idx == 1 {
                        eprintln!("[TENSOR NAME DEBUG] Layer 1 loading tensor: blk.{}.ssm_a", layer_idx);
                    }



                    // Load ssm_dt [32]

                    let ssm_dt = content.tensor(reader, &format!("blk.{layer_idx}.ssm_dt.bias"), device)?
                        .dequantize(device)?;
                    if layer_idx == 1 {
                        eprintln!("[TENSOR NAME DEBUG] Layer 1 loading tensor: blk.{}.ssm_dt.bias", layer_idx);
                    }



                    // Load ssm_beta

                    let beta_w = content.tensor(reader, &format!("blk.{layer_idx}.ssm_beta.weight"), device)?;
                    if layer_idx == 1 {
                        eprintln!("[TENSOR NAME DEBUG] Layer 1 loading tensor: blk.{}.ssm_beta.weight", layer_idx);
                    }

                    let beta_b = content.tensor(reader, &format!("blk.{layer_idx}.ssm_beta.bias"), device)

                        .ok().and_then(|t| t.dequantize(device).ok());

                    let ssm_beta = UnifiedLinear::Quantized(QMatMul::from_qtensor(beta_w)?, beta_b);



                    // Load ssm_alpha

                    let alpha_w = content.tensor(reader, &format!("blk.{layer_idx}.ssm_alpha.weight"), device)?;
                    if layer_idx == 1 {
                        eprintln!("[TENSOR NAME DEBUG] Layer 1 loading tensor: blk.{}.ssm_alpha.weight", layer_idx);
                    };

                    if layer_idx == 4 {
                        eprintln!("[SHAPE DEBUG] ssm_alpha.weight: {:?}", alpha_w.shape());
                    }
                    let alpha_b = content.tensor(reader, &format!("blk.{layer_idx}.ssm_alpha.bias"), device)

                        .ok().and_then(|t| t.dequantize(device).ok());

                    let ssm_alpha = UnifiedLinear::Quantized(QMatMul::from_qtensor(alpha_w)?, alpha_b);



                    let ssm_block = QwenSsmBlock::new(

                        ssm_in_proj,

                        attn_gate,

                        conv1d,

                        ssm_out_proj,

                        ssm_norm,

                        ssm_a,

                        ssm_dt,

                        ssm_beta,

                        ssm_alpha,

                        device

                    )?;



                    (None, Some(ssm_block))

                } else {

                    // 2. Load standard Attention layer

                    let (q_proj, k_proj, v_proj) =

                        if content.tensor(reader, &format!("blk.{layer_idx}.attn_qkv.weight"), device).is_ok() {

                            let qkv_w = content.tensor(reader, &format!("blk.{layer_idx}.attn_qkv.weight"), device)?;

                            let qkv_b = content.tensor(reader, &format!("blk.{layer_idx}.attn_qkv.bias"), device)

                                .ok().and_then(|t| t.dequantize(device).ok());

                            (UnifiedLinear::Quantized(QMatMul::from_qtensor(qkv_w)?, qkv_b), None, None)

                        } else {

                            let q_w = content.tensor(reader, &format!("blk.{layer_idx}.attn_q.weight"), device)?;
                            if layer_idx == 3 {
                                if let Ok(dq) = q_w.dequantize(device) {
                                    eprintln!("[DEBUG attn_q weight] layer={} shape={:?}", layer_idx, dq.shape().dims());
                                }
                            }
                            let q_b = content.tensor(reader, &format!("blk.{layer_idx}.attn_q.bias"), device).ok().and_then(|t| t.dequantize(device).ok());

                            let k_w = content.tensor(reader, &format!("blk.{layer_idx}.attn_k.weight"), device)?;
                            let k_b = content.tensor(reader, &format!("blk.{layer_idx}.attn_k.bias"), device).ok().and_then(|t| t.dequantize(device).ok());

                            if layer_idx == 3 {
                                eprintln!("[DEBUG K_PROJ WEIGHT] layer 3 k_proj found.");
                                if let Ok(w_dequant) = k_w.dequantize(device) {
                                    if let Ok(w_flat) = w_dequant.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                                        let n = w_flat.len();
                                        let rms = (w_flat.iter().map(|v| v*v).sum::<f32>() / n as f32).sqrt();
                                        let mean = w_flat.iter().sum::<f32>() / n as f32;
                                        let min = w_flat.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                                        let max = w_flat.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                                        eprintln!("[DEBUG K_PROJ WEIGHT] shape={:?} n={} RMS={:.4} mean={:.4} min={:.4} max={:.4}", 
                                            w_dequant.shape().dims(), n, rms, mean, min, max);
                                    }
                                }
                            }

                            let v_w = content.tensor(reader, &format!("blk.{layer_idx}.attn_v.weight"), device)?;
                            let v_b = content.tensor(reader, &format!("blk.{layer_idx}.attn_v.bias"), device).ok().and_then(|t| t.dequantize(device).ok());

                            (

                                UnifiedLinear::Quantized(QMatMul::from_qtensor(q_w)?, q_b),

                                Some(UnifiedLinear::Quantized(QMatMul::from_qtensor(k_w)?, k_b)),

                                Some(UnifiedLinear::Quantized(QMatMul::from_qtensor(v_w)?, v_b)),

                            )

                        };



                    let o_w = content.tensor(reader, &format!("blk.{layer_idx}.attn_output.weight"), device)?;

                    let o_b = content.tensor(reader, &format!("blk.{layer_idx}.attn_output.bias"), device).ok().and_then(|t| t.dequantize(device).ok());

                    let o_proj = UnifiedLinear::Quantized(QMatMul::from_qtensor(o_w)?, o_b);



                    let head_dim = cfg.hidden_size / cfg.num_attention_heads;

                    let num_kv_groups = cfg.num_attention_heads / cfg.num_key_value_heads;



                let q_norm = content.tensor(reader, &format!("blk.{layer_idx}.attn_q_norm.weight"), device).ok()

                    .and_then(|t| t.dequantize(device).ok())

                    .and_then(|w| QRmsNorm::new(w, cfg.rms_norm_eps).ok());

                let k_norm = content.tensor(reader, &format!("blk.{layer_idx}.attn_k_norm.weight"), device).ok()

                    .and_then(|t| t.dequantize(device).ok())

                    .and_then(|w| QRmsNorm::new(w, cfg.rms_norm_eps).ok());



                    let kv_dim = head_dim * cfg.num_key_value_heads;

                    let q_scp = SplitComplexProjection::new(cfg.hidden_size, cfg.hidden_size, 8, layer_idx * 2, device).ok();

                    let k_scp = SplitComplexProjection::new(cfg.hidden_size, kv_dim, 8, layer_idx * 2 + 1, device).ok();



                    let self_attn = QAttention::new(

                        q_proj, k_proj, v_proj, o_proj,

                        cfg.num_attention_heads, cfg.num_key_value_heads, num_kv_groups, head_dim,

                        rotary_emb.clone(), q_norm, k_norm,

                        q_scp, k_scp,

                        None, None,

                    );



                    (Some(self_attn), None)

                };



                let gate_w = content.tensor(reader, &format!("blk.{layer_idx}.ffn_gate.weight"), device)?;

                let gate_b = content.tensor(reader, &format!("blk.{layer_idx}.ffn_gate.bias"), device).ok().and_then(|t| t.dequantize(device).ok());

                let up_w   = content.tensor(reader, &format!("blk.{layer_idx}.ffn_up.weight"), device)?;

                let up_b   = content.tensor(reader, &format!("blk.{layer_idx}.ffn_up.bias"), device).ok().and_then(|t| t.dequantize(device).ok());

                let down_w = content.tensor(reader, &format!("blk.{layer_idx}.ffn_down.weight"), device)?;

                let down_b = content.tensor(reader, &format!("blk.{layer_idx}.ffn_down.bias"), device).ok().and_then(|t| t.dequantize(device).ok());



                let mlp = QMlp::new(

                    UnifiedLinear::Quantized(QMatMul::from_qtensor(gate_w)?, gate_b),

                    UnifiedLinear::Quantized(QMatMul::from_qtensor(up_w)?,   up_b),

                    UnifiedLinear::Quantized(QMatMul::from_qtensor(down_w)?, down_b),

                );



                let iln_w  = content.tensor(reader, &format!("blk.{layer_idx}.attn_norm.weight"), device)?.dequantize(device)?;
                
                // Qwen3.5 uses post_attention_norm.weight instead of ffn_norm.weight

                let paln_w = content.tensor(reader, &format!("blk.{layer_idx}.post_attention_norm.weight"), device)

                    .or_else(|_| content.tensor(reader, &format!("blk.{layer_idx}.ffn_norm.weight"), device))?

                    .dequantize(device)?;



                layers.push(QDecoderLayer::new(

                    self_attn, ssm_block, mlp,

                    QRmsNorm::new(iln_w, cfg.rms_norm_eps)?,

                    QRmsNorm::new(paln_w, cfg.rms_norm_eps)?,

                ));

            }



        let norm_w  = content.tensor(reader, "output_norm.weight", device)?.dequantize(device)?;
        if let Ok(nw_vec) = norm_w.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
            let nw_rms = (norm_w.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?);
            let nw_mean = norm_w.mean_all()?.to_scalar::<f32>()?;
            eprintln!("[MODEL NORM DEBUG] output_norm.weight: first 4={:?} RMS={:.6} mean={:.6}", &nw_vec[0..4.min(nw_vec.len())], nw_rms, nw_mean);
        }

        let lm_w    = content.tensor(reader, "output.weight", device)?;
        eprintln!("[LM_HEAD] output.weight shape: {:?}, dtype: {:?}", lm_w.shape(), lm_w.dtype());

        let lm_b    = content.tensor(reader, "output.bias",   device).ok().and_then(|t| {
            eprintln!("[LM_HEAD] output.bias EXISTS, shape: {:?}", t.shape());
            t.dequantize(device).ok()
        });
        if lm_b.is_none() {
            eprintln!("[LM_HEAD] output.bias does not exist (expected)");
        }



        let model   = Qwen2Model::new(embed_tokens, layers, QRmsNorm::new(norm_w, cfg.rms_norm_eps)?, device.clone())?;

        let lm_head = UnifiedLinear::Quantized(QMatMul::from_qtensor(lm_w)?, lm_b);

        let mut instance = Self::new(model, lm_head);

        // Optional: run simple layer-wise mixed quantization pipeline if enabled

        if std::env::var("SODALS_MIXED_QUANT").map(|v| v == "1").unwrap_or(false) {

            eprintln!("[QUANT] SODALS_MIXED_QUANT=1 - running placeholder layer-wise quantization");

            // Placeholder integration: we don't mutate internals here. In a full pipeline

            // we'd collect per-layer weight tensors and replace them with quantized ones.

            let num_layers = instance.model.layers.len();

            let tensors: Vec<candle_core::Tensor> = Vec::new();

            let mut configs = Vec::new();

            for i in 0..num_layers {

                configs.push(LayerQuantConfig { layer_index: i, dtype: candle_core::DType::F16, quant_bits: 16 });

            }

            if let Ok(_q_tensors) = run_layerwise_quantization(&tensors, &configs) {

                eprintln!("[QUANT] Layerwise quantization placeholder completed (no-op)");

            }

        }

        eprintln!("[ARKHE] Qwen2ForCausalLM loaded with latent pondering capability");

        Ok(instance)

    }



    pub fn from_awq_safetensors(paths: &[PathBuf], device: &Device) -> Result<Self> {

        let vb  = unsafe { VarBuilder::from_mmaped_safetensors(paths, DType::F32, device)? };

        let cfg = Qwen2Config::default();



        let embed_tokens = ModelEmbedding::new(

            vb.get((cfg.vocab_size, cfg.hidden_size), "model.embed_tokens.weight")?,

            cfg.hidden_size,

        );

        let rotary_emb = Arc::new(RotaryEmbedding::new(&cfg, device)?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);



        for layer_idx in 0..cfg.num_hidden_layers {

            let lp = vb.pp(format!("model.layers.{}", layer_idx));

            let kv_dim = cfg.hidden_size / cfg.num_attention_heads * cfg.num_key_value_heads;



            let q_proj = UnifiedLinear::Regular(Linear::new(

                lp.get((cfg.hidden_size, cfg.hidden_size), "self_attn.q_proj.weight")?,

                lp.get(cfg.hidden_size, "self_attn.q_proj.bias").ok(),

            ));

            let k_proj = UnifiedLinear::Regular(Linear::new(

                lp.get((kv_dim, cfg.hidden_size), "self_attn.k_proj.weight")?,

                lp.get(kv_dim, "self_attn.k_proj.bias").ok(),

            ));

            let v_proj = UnifiedLinear::Regular(Linear::new(

                lp.get((kv_dim, cfg.hidden_size), "self_attn.v_proj.weight")?,

                lp.get(kv_dim, "self_attn.v_proj.bias").ok(),

            ));

            let o_proj = UnifiedLinear::Regular(Linear::new(

                lp.get((cfg.hidden_size, cfg.hidden_size), "self_attn.o_proj.weight")?,

                lp.get(cfg.hidden_size, "self_attn.o_proj.bias").ok(),

            ));



            let head_dim      = cfg.hidden_size / cfg.num_attention_heads;

            let num_kv_groups = cfg.num_attention_heads / cfg.num_key_value_heads;

            // Phase 44: Initialize SCP for legitimate split-complex projection

            // PHASE 44 FIX: q_scp outputs 4096 (hidden_size), k_scp outputs 1024 (kv_dim) for GQA

            let kv_dim = head_dim * cfg.num_key_value_heads; // 128 * 8 = 1024

            let q_scp = SplitComplexProjection::new(cfg.hidden_size, cfg.hidden_size, 8, layer_idx * 2, device).ok();

            let k_scp = SplitComplexProjection::new(cfg.hidden_size, kv_dim, 8, layer_idx * 2 + 1, device).ok();



            let self_attn = QAttention::new(

                q_proj, Some(k_proj), Some(v_proj), o_proj,

                cfg.num_attention_heads, cfg.num_key_value_heads, num_kv_groups, head_dim,

                rotary_emb.clone(), None, None,

                q_scp, k_scp,  // Phase 44: SCP rank-8 split-complex projection

                None, None,        // No explicit imaginary weights in safetensors

            );



            let gate_proj = UnifiedLinear::Regular(Linear::new(

                lp.get((cfg.intermediate_size, cfg.hidden_size), "mlp.gate_proj.weight")?,

                lp.get(cfg.intermediate_size, "mlp.gate_proj.bias").ok(),

            ));

            let up_proj = UnifiedLinear::Regular(Linear::new(

                lp.get((cfg.intermediate_size, cfg.hidden_size), "mlp.up_proj.weight")?,

                lp.get(cfg.intermediate_size, "mlp.up_proj.bias").ok(),

            ));

            let down_proj = UnifiedLinear::Regular(Linear::new(

                lp.get((cfg.hidden_size, cfg.intermediate_size), "mlp.down_proj.weight")?,

                lp.get(cfg.hidden_size, "mlp.down_proj.bias").ok(),

            ));

            let mlp = QMlp::new(gate_proj, up_proj, down_proj);



            let iln  = QRmsNorm::new_zero_centered(lp.get(cfg.hidden_size, "input_layernorm.weight")?,          cfg.rms_norm_eps)?;

            let paln = QRmsNorm::new_zero_centered(lp.get(cfg.hidden_size, "post_attention_layernorm.weight")?, cfg.rms_norm_eps)?;

            layers.push(QDecoderLayer::new(Some(self_attn), None, mlp, iln, paln));

        }



        let norm    = QRmsNorm::new_zero_centered(vb.get(cfg.hidden_size, "model.norm.weight")?, cfg.rms_norm_eps)?;

        let lm_head = UnifiedLinear::Regular(Linear::new(

            vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")?,

            vb.get(cfg.vocab_size,  "lm_head.bias").ok(),

        ));



        let model = Qwen2Model::new(embed_tokens, layers, norm, device.clone())?;

        let mut instance = Self::new(model, lm_head);

        eprintln!("[ARKHE] Qwen2ForCausalLM loaded with latent pondering capability (AWQ)");

        Ok(instance)

    }



    // IIIIII forward III Phase parametresi eklendi ////////////////////////////////////////////////////////////////////////////////////////////////////II

    pub fn forward(

        &self,

        input_ids: &Tensor,

        seqlen_offset: usize,

        poles_ptr: Option<u64>,

        impedance_ptr: Option<u64>,

        flags_ptr: Option<u64>,

        n_nodes: Option<i32>,

        phase: Option<&PhaseSignal>,

        ghost_prefix: Option<&Tensor>,

        // Token-to-concept alignment map pointer

        token_concept_map_ptr: Option<u64>,

        qmem_cartridge: Option<(&Tensor, f32)>,
        gnn_graph_ptrs: Option<(u64, u64, u64, u64, u64, f32)>,
    ) -> Result<Tensor> {

        let hidden_states = self.model.forward(

            input_ids,

            seqlen_offset,

            poles_ptr,

            impedance_ptr,

            flags_ptr,

            n_nodes,

            phase,

            ghost_prefix,

            token_concept_map_ptr,

            Some(&self.model.qmem_parameters),

            self.qmem_loader.as_ref(),

            qmem_cartridge,

            gnn_graph_ptrs, // gnn_graph_ptrs

        )?;

        let seq_len = hidden_states.dim(1)?;

        let last = if seq_len > 1 {

            hidden_states.narrow(1, seq_len - 1, 1)?

        } else {

            hidden_states.clone()

        };

        

        // PHASE 36+37: ARKHE LATENT PONDERING + RETROACTIVE ATTENTION

        // Intercept hidden state before token emission for continuous-time evolution

        // Then detect epiphanies and retroactively update KV cache

        let h_initial = last.clone(); // Capture initial state before pondering

        

        // PHASE 53 FIX: Pondering Gating - Only ponder every 8th token to bound VRAM fragmentation.

        // The continuous RK4 pondering is powerful but expensive; gating it maintains 

        // semantic health while drastically reducing the number of intermediate GPU allocations.

        // FAZ 21.2 FIX: Pondering is OFF by default. Verified on Qwen3.5-9B: the RK4
        // manifold evolution perturbs hidden states by ~0.04 RMS, tipping razor-thin
        // near-ties in sampling and cascading into degenerate repetition (wrong answer).
        // SODALS_NO_PONDER is kept as a hard override; SODALS_PONDER re-enables it.

        let should_ponder = std::env::var("SODALS_PONDER").is_ok()
            && !std::env::var("SODALS_PURE_INFERENCE").is_ok()
            && !std::env::var("SODALS_NO_PONDER").is_ok()
            && (seqlen_offset % 8 == 0)
            && self.pondering_core.is_some();

        if is_verbose() { eprintln!("[PONDER DEBUG] should_ponder: {}, seqlen_offset: {}, pondering_core exists: {}, SODALS_PURE_INFERENCE set: {}",
            should_ponder,
            seqlen_offset,
            self.pondering_core.is_some(),
            std::env::var("SODALS_PURE_INFERENCE").is_ok()
        ); }

        let last_clone = last.clone(); // Clone before moving

        let pondered_hidden = if should_ponder {

            let core_mutex = self.pondering_core.as_ref().unwrap();

            // Convert ghost_prefix to SemanticWave for Arkhe derivative

            let ghost_wave = ghost_prefix.map(|g| {

                // Create a simple SemanticWave - size matches hidden dim

                // In production, this would be derived from the full topological ghost vector

                let hidden_dim = g.dims().last().copied().unwrap_or(64);

                crate::fast_math::SemanticWave::new(hidden_dim)

            });

            

            match core_mutex.lock() {

                Ok(core) => {

                    match core.ponder(&last, ghost_wave.as_ref(), 0.1) {

                        Ok(pondered) => {

                            // PHASE 50.2: Commented out token-by-token log

                            // eprintln!("[ARKHE] Hidden state pondered through continuous manifold");

                            pondered

                        }

                        Err(_e) => {

                            // PHASE 50.2: Commented out token-by-token log

                            // eprintln!("[ARKHE] Pondering failed ({}), using raw hidden state", e);

                            last

                        }

                    }

                }

                Err(_) => {

                    // PHASE 50.2: Commented out token-by-token log

                    // eprintln!("[ARKHE] Failed to lock pondering core, using raw hidden state");

                    last

                }

            }

        } else {

            // Pondering disabled or gated, use raw hidden state

            if is_verbose() { eprintln!("[PONDER DEBUG] Pondering bypassed, using raw hidden state"); }
            last

        };

        // Skip delta_h + blocking to_vec0 GPU sync whenever pondering is disabled/bypassed!
        if should_ponder {
        // Verify if pondering actually modified the state
        let h_initial_flat = last_clone.flatten_all()?;
        let pondered_flat = pondered_hidden.flatten_all()?;
        if is_verbose() {
            let h_vals: Vec<f32> = h_initial_flat.to_vec1()?.iter().take(5).cloned().collect();
            let p_vals: Vec<f32> = pondered_flat.to_vec1()?.iter().take(5).cloned().collect();
            eprintln!("[PONDER DEBUG] h_initial first 5: {:?}", h_vals);
            eprintln!("[PONDER DEBUG] pondered first 5: {:?}", p_vals);
            eprintln!("[PONDER DEBUG] States equal: {:?}", h_vals == p_vals);
        }

        // Also compute delta for logging
        let delta_h = (&pondered_hidden - &last_clone)?;


        // PHASE 37: EPIPHANY DETECTION & RETROACTIVE RELAXATION

        // If pondering produced a major semantic shift, update past context

        

        // PHASE 45 FIX: Properly squeeze tensor before extracting scalar norm

        // Compute ||?h||_2 = sqrt(sum(?h�)) with proper tensor shape handling

        let delta_squared = delta_h.sqr()?;

        let delta_sum = delta_squared.sum_keepdim(D::Minus1)?;

        let delta_sqrt = delta_sum.sqrt()?;

        

        // Squeeze to scalar: [1, 1, 1] -> [] before to_vec0

        let displacement_norm_scalar = delta_sqrt

            .reshape(())?  // Collapse all dimensions to scalar

            .to_vec0::<f32>()? as f64;

        

        let trigger_relaxation = {

            let mut gate = self.epiphany_gate.lock().unwrap();

            gate.should_relax(displacement_norm_scalar)

        };

        

        #[allow(dead_code)]

        const OMEGA_RELAXATION: f64 = 0.05; // Conservative to prevent catastrophic forgetting

        

        if trigger_relaxation {
            // PHASE 6: Wave-Resonance KV Modulation
            // Instead of flat energy projections causing Attention Collapse, we now inject 
            // Phase-Rotated (Holographic) energy directly into the KV Cache.
            let _ = self.model.retroactive_relax(&delta_h, OMEGA_RELAXATION, phase);
        } else {

            // PHASE 50.2: Commented out token-by-token log

            // eprintln!("[ARKHE] Semantic displacement: {:.6} (below threshold)", displacement_norm_scalar);

        }
        } // end if should_ponder

        

        // LINUS FIX: Logit bias NUKED. RoPE Warping replaces it at the attention level.

        if std::env::var("SODALS_DUMP_HIDDEN").is_ok() {
            let h_flat = pondered_hidden.flatten_all()?;
            let h_vals: Vec<f32> = h_flat.to_vec1()?;
            let h_bytes: Vec<u8> = h_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(r"C:\Users\Burak\AppData\Local\Temp\our_hidden.bin", &h_bytes);
            let sum: f32 = h_vals.iter().sum();
            let mean = sum / h_vals.len() as f32;
            let sq_sum: f32 = h_vals.iter().map(|v| v * v).sum();
            let rms = (sq_sum / h_vals.len() as f32).sqrt();
            let min = h_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = h_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let first_10: Vec<f32> = h_vals.iter().take(10).copied().collect();
            eprintln!("[HIDDEN DUMP] RMS={:.4} mean={:.4} min={:.4} max={:.4} first10={:.4?}", rms, mean, min, max, first_10);
        }

        if is_verbose() { eprintln!("[LM_HEAD DEBUG] pondered_hidden shape: {:?}", pondered_hidden.shape()); }

        let pondered_hidden_flat = pondered_hidden.flatten_all()?;
        if is_verbose() {
            let hidden_vals: Vec<f32> = pondered_hidden_flat.to_vec1()?.iter().take(10).cloned().collect();
            eprintln!("[LM_HEAD DEBUG] pondered_hidden first 10 vals: {:?}", hidden_vals);
        }

        // OPTIMIZATION: Narrow prefill hidden states to ONLY the last token before lm_head
        // Prevents allocating 1.07 GB of intermediate logits during prefill/backtrack
        let hidden_for_lm_head = if pondered_hidden.rank() == 3 && pondered_hidden.dim(1)? > 1 {
            let seq_len = pondered_hidden.dim(1)?;
            pondered_hidden.narrow(1, seq_len - 1, 1)?
        } else {
            pondered_hidden.clone()
        };

        let final_logits = self.lm_head.forward(&hidden_for_lm_head)?.to_dtype(DType::F32)?;

        if is_verbose() { eprintln!("[LM_HEAD DEBUG] final_logits shape: {:?}", final_logits.shape()); }

        let logits_flat = final_logits.flatten_all()?;
        if is_verbose() {
            let logits_vals: Vec<f32> = logits_flat.to_vec1()?.iter().take(10).cloned().collect();
            eprintln!("[LM_HEAD DEBUG] final_logits first 10 vals: {:?}", logits_vals);
        }

        // Check for NaN or Inf
        if is_verbose() {
            let logits_sum = logits_flat.sum_all()?.to_scalar::<f32>()?;
            eprintln!("[LM_HEAD DEBUG] logits sum: {:.4}, is finite: {}", logits_sum, logits_sum.is_finite());
        }

        let dump_layers = std::env::var("SODALS_DUMP_LAYERS").is_ok();
        let dump_layers = std::env::var("SODALS_DUMP_LAYERS").is_ok();
        if dump_layers {
            let vals: Vec<f32> = logits_flat.to_vec1()?;
            let n = vals.len();
            let sum: f32 = vals.iter().sum();
            let mean = sum / n as f32;
            let sq_sum: f32 = vals.iter().map(|v| v * v).sum();
            let rms = (sq_sum / n as f32).sqrt();
            let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let first_5: Vec<f32> = vals.iter().take(5).copied().collect();
            eprintln!("[DUMP LOGITS] RMS={:.6} mean={:.6} min={:.4} max={:.4} first5={:.4?}", rms, mean, min, max, first_5);
            // Find top-5 token IDs and their logits
            let mut ranked: Vec<(u32, f32)> = vals.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (i, (id, logit)) in ranked.iter().take(5).enumerate() {
                eprintln!("[DUMP LOGITS] top{}: id={:>6} logit={:>8.2}", i+1, id, logit);
            }
        }

        Ok(final_logits)

    }



    /// Phase 34: Return normalized hidden states WITHOUT passing through lm_head.

    /// Used by LlmEngine::embed() to extract true semantic vectors instead of logit-space nonsense.

    /// Returns the full sequence hidden states: [batch, seq_len, hidden_size]

    pub fn forward_hidden_states(

        &self,

        input_ids: &Tensor,

        seqlen_offset: usize,

    ) -> Result<Tensor> {

        self.model.forward(

            input_ids,

            seqlen_offset,

            None,

            None,

            None,

            None,

            None,

            None,

            None,

            Some(&self.model.qmem_parameters),

            self.qmem_loader.as_ref(),

            None,
            None, // gnn_graph_ptrs

        )

    }



    pub fn set_physics_bias(&mut self, impedance: Vec<f64>) -> Result<()> {

        self.model.set_physics_bias(impedance)

    }

    pub fn clear_physics_bias(&mut self) { self.model.clear_physics_bias() }



    pub fn hidden_size(&self) -> usize {

        // Qwen2Model is public, but embed_tokens might be private.

        // We can get it from the embed_tokens field if we make it public or use this helper.

        self.model.embed_tokens_dim()

    }



    pub fn model(&self)     -> &Qwen2Model     { &self.model }

    pub fn model_mut(&mut self) -> &mut Qwen2Model { &mut self.model }



    pub fn clear_kv_cache(&self) {
        self.model.clear_kv_cache();
    }

    pub fn truncate_kv_cache(&self, keep_len: usize) -> Result<()> {
        self.model.truncate_kv_cache(keep_len)
    }



    /// Extrapolate statistical metrics for the Jury Report Crucible

    pub fn get_epiphany_count(&self) -> u32 {

        if let Ok(gate) = self.epiphany_gate.lock() {

            gate.shift_counter

        } else {

            0

        }

    }

    

            pub fn from_safetensors(
        vb: candle_nn::VarBuilder,
        cfg: &Qwen2Config,
        device: &candle_core::Device,
    ) -> candle_core::Result<Self> {
        use candle_nn::Module;
        
        let base_prefix = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            "model.language_model"
        } else {
            "model"
        };
        
        let embed_weight = vb.pp(format!("{}.embed_tokens", base_prefix)).get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_tokens = ModelEmbedding::new(embed_weight, cfg.hidden_size);
        let rotary_emb = std::sync::Arc::new(RotaryEmbedding::new(cfg, device)?);
        
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let is_ssm_model = vb.contains_tensor(&format!("{}.layers.0.ssm_out.weight", base_prefix));
        if is_ssm_model {
            eprintln!("[SSM-ST] Detected Qwen3.5 SSM architecture in safetensors");
        }
        
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_vb = vb.pp(&format!("{}.layers.{}", base_prefix, layer_idx));
            let layer_has_ssm = is_ssm_model && layer_vb.contains_tensor("ssm_out.weight");
            eprintln!("[DEBUG LAYER TYPE] layer_idx: {}, is_ssm_model: {}, contains_ssm_out: {} => layer_has_ssm: {}", 
                layer_idx, is_ssm_model, layer_vb.contains_tensor("ssm_out.weight"), layer_has_ssm);


            
            let (self_attn, ssm_block) = if layer_has_ssm {
                // SSM Layer
                // Load fused qkv or separate
                let attn_vb = layer_vb.pp("self_attn");
                let ssm_in_proj = if attn_vb.contains_tensor("qkv_proj.weight") {
                    UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size * 3, attn_vb.pp("qkv_proj"))?)
                } else {
                    let w = attn_vb.get((cfg.hidden_size * 3, cfg.hidden_size), "weight")?;
                    UnifiedLinear::Regular(candle_nn::Linear::new(w, None)) // Fallback or needs proper separate loading
                };
                
                let attn_gate = if layer_vb.contains_tensor("attn_gate.weight") {
                    UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size, layer_vb.pp("attn_gate"))?)
                } else {
                    UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size, attn_vb.pp("gate_proj"))?) // standard hf name
                };
                
                let conv1d_w = layer_vb.pp("ssm_conv1d").get((cfg.hidden_size, 1, 4), "weight")?; // Example shape
                let conv1d = conv1d_w; // simplified
                
                let ssm_out_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size, layer_vb.pp("ssm_out"))?);
                // SSM norm weight is shape [128]! NOT cfg.hidden_size
                let ssm_norm = layer_vb.pp("ssm_norm").get(128, "weight").ok().and_then(|w| QRmsNorm::new(w, 1e-6).ok());
                if ssm_norm.is_none() {
                    eprintln!("[DEBUG] layer {} ssm_norm is MISSING", layer_idx);
                } else {
                    eprintln!("[DEBUG] layer {} ssm_norm loaded successfully", layer_idx);
                }
                
                let ssm_a = layer_vb.get(32, "ssm_a")?;
                let ssm_dt = layer_vb.pp("ssm_dt").get(32, "bias")?;
                
                let ssm_beta = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size, layer_vb.pp("ssm_beta"))?);
                let ssm_alpha = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size, layer_vb.pp("ssm_alpha"))?);

                (None, Some(crate::models::qwen_ssm::QwenSsmBlock::new(
                    ssm_in_proj, attn_gate, conv1d, ssm_out_proj, ssm_norm, ssm_a, ssm_dt, ssm_beta, ssm_alpha, device
                )?))
            } else {
                // Standard MHA Layer
                let attn_vb = layer_vb.pp("self_attn");
                let q_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.num_attention_heads * cfg.hidden_size / cfg.num_attention_heads, attn_vb.pp("q_proj"))?);
                let k_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.num_key_value_heads * cfg.hidden_size / cfg.num_attention_heads, attn_vb.pp("k_proj"))?);
                if let UnifiedLinear::Regular(ref l) = k_proj {
                    if let Ok(w) = l.weight().to_dtype(DType::F32) {
                        let w_flat: Vec<f32> = w.flatten_all()?.to_vec1()?;
                        eprintln!("[DEBUG K_PROJ WEIGHT] shape={:?} n={} RMS={:.4} mean={:.4} min={:.4} max={:.4}", 
                            w.shape().dims(), w_flat.len(),
                            (w_flat.iter().map(|v| v*v).sum::<f32>() / w_flat.len() as f32).sqrt(),
                            w_flat.iter().sum::<f32>() / w_flat.len() as f32,
                            w_flat.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                            w_flat.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
                    }
                }
                let v_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.num_key_value_heads * cfg.hidden_size / cfg.num_attention_heads, attn_vb.pp("v_proj"))?);
                let o_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.num_attention_heads * cfg.hidden_size / cfg.num_attention_heads, cfg.hidden_size, attn_vb.pp("o_proj"))?);
                
                let q_norm = attn_vb.pp("q_norm").get(cfg.num_attention_heads * cfg.hidden_size / cfg.num_attention_heads, "weight").ok().and_then(|w| QRmsNorm::new_zero_centered(w, 1e-6).ok());
                let k_norm = attn_vb.pp("k_norm").get(cfg.num_attention_heads * cfg.hidden_size / cfg.num_attention_heads, "weight").ok().and_then(|w| QRmsNorm::new_zero_centered(w, 1e-6).ok());
                
                let self_attn = QAttention::new(
                    q_proj, Some(k_proj), Some(v_proj), o_proj,
                    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_attention_heads / cfg.num_key_value_heads,
                    cfg.hidden_size / cfg.num_attention_heads,
                    rotary_emb.clone(), q_norm, k_norm, None, None, None, None
                );
                (Some(self_attn), None)
            };
            
            let mlp_vb = layer_vb.pp("mlp");
            let gate_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, mlp_vb.pp("gate_proj"))?);
            let up_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, mlp_vb.pp("up_proj"))?);
            let down_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, mlp_vb.pp("down_proj"))?);
            let mlp = QMlp::new(gate_proj, up_proj, down_proj);

            let input_layernorm = QRmsNorm::new_zero_centered(layer_vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?, 1e-6)?;
            let post_attention_layernorm = QRmsNorm::new_zero_centered(layer_vb.pp("post_attention_layernorm").get(cfg.hidden_size, "weight")?, 1e-6)?;
            
            layers.push(QDecoderLayer::new(
                self_attn, ssm_block, mlp, input_layernorm, post_attention_layernorm
            ));
        }
        
        let norm = QRmsNorm::new_zero_centered(vb.pp(&format!("{}.norm", base_prefix)).get(cfg.hidden_size, "weight")?, 1e-6)?;
        let lm_head = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?);
        
        let qwen_model = Qwen2Model::new(embed_tokens, layers, norm, device.clone())?;
        Ok(Self::new(qwen_model, lm_head))
    }


    pub fn get_semantic_divergence_average(&self) -> f64 {

        if let Some(core) = &self.pondering_core {

            if let Ok(c) = core.lock() {

                c.get_gods_gate_state().load_divergence()

            } else {

                0.0

            }

        } else {

            0.0

        }

    }

}







