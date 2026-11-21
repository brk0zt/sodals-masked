use candle_core::{Tensor, Result, DType};
use crate::complex_tensor::ComplexState;

// ============================================================================
// CVNN Activation Functions
// ============================================================================
// Reference: Trabelsi et al., "Deep Complex Networks" (ICLR 2018)
// These activations preserve complex-valued gradient flow while introducing
// the non-linearity required for universal approximation in ℂ.
// ============================================================================

/// CReLU: Applies ReLU independently to real and imaginary parts.
/// CReLU(z) = ReLU(Re(z)) + i·ReLU(Im(z))
///
/// Properties:
/// - Simplest complex activation, maintains component independence
/// - Phase information is partially lost in quadrants II, III, IV
/// - Gradient: ∂CReLU/∂z = H(Re(z)) + i·H(Im(z)) where H is Heaviside
pub fn crelu(z: &ComplexState) -> Result<ComplexState> {
    let real_out = z.real.relu()?;
    let imag_out = z.imag.relu()?;
    Ok(ComplexState {
        real: real_out,
        imag: imag_out,
    })
}

/// zReLU: Phase-gated activation that preserves the full complex value
/// only when the phase angle is in the first quadrant [0, π/2].
/// zReLU(z) = z  if arg(z) ∈ [0, π/2]
///          = 0  otherwise
///
/// Properties:
/// - Preserves magnitude and phase within the first quadrant
/// - Sharper gating than CReLU — acts as a phase-selective filter
/// - Useful for enforcing constructive interference patterns
pub fn zrelu(z: &ComplexState) -> Result<ComplexState> {
    // Compute phase angle θ = atan2(imag, real)
    let phase = z.phase()?;
    
    // Build mask: 1.0 where θ ∈ [0, π/2], else 0.0
    // Condition: real >= 0 AND imag >= 0  ⟺  first quadrant
    let zeros = Tensor::zeros_like(&z.real)?;
    let real_positive = z.real.ge(&zeros)?;       // Re(z) >= 0
    let imag_positive = z.imag.ge(&zeros)?;       // Im(z) >= 0
    let mask = (&real_positive * &imag_positive)?; // Logical AND via multiplication
    let mask_f = mask.to_dtype(z.real.dtype())?;
    
    let real_out = (&z.real * &mask_f)?;
    let imag_out = (&z.imag * &mask_f)?;
    
    Ok(ComplexState {
        real: real_out,
        imag: imag_out,
    })
}

/// modReLU: Magnitude-gated activation with learnable bias.
/// modReLU(z) = ReLU(|z| + b) · z/|z|  = ReLU(|z| + b) · e^{iθ}
///
/// When |z| + b > 0: output has same phase as input, magnitude = |z| + b
/// When |z| + b ≤ 0: output is zero (dead zone near origin)
///
/// Properties:
/// - Preserves phase information perfectly (only magnitude is gated)
/// - Learnable bias `b` controls the dead zone radius
/// - Recommended default: b ∈ [-0.5, 0.0] (slight shrinkage)
/// - This is the most theoretically motivated CVNN activation
pub fn mod_relu(z: &ComplexState, bias: f64) -> Result<ComplexState> {
    // |z| = sqrt(Re² + Im²) + ε for numerical stability
    let magnitude = z.magnitude()?;
    let eps = 1e-8;
    let magnitude_safe = (magnitude.clone() + eps)?;
    
    // ReLU(|z| + b)
    let shifted = (magnitude.clone() + bias)?;
    let activated_magnitude = shifted.relu()?;
    
    // Scale factor: ReLU(|z| + b) / |z|
    let scale = (&activated_magnitude / &magnitude_safe)?;
    
    let real_out = (&z.real * &scale)?;
    let imag_out = (&z.imag * &scale)?;
    
    Ok(ComplexState {
        real: real_out,
        imag: imag_out,
    })
}

/// Complex Tanh: Applies tanh independently to real and imaginary parts.
/// CTanh(z) = tanh(Re(z)) + i·tanh(Im(z))
///
/// Properties:
/// - Bounded output in the complex unit square [-1,1] × [-1,1]i
/// - Smooth, differentiable everywhere
/// - Good for stabilizing deep CVNN layers (prevents gradient explosion)
pub fn complex_tanh(z: &ComplexState) -> Result<ComplexState> {
    let real_out = z.real.tanh()?;
    let imag_out = z.imag.tanh()?;
    Ok(ComplexState {
        real: real_out,
        imag: imag_out,
    })
}

/// Complex Sigmoid: Applies sigmoid independently to real and imaginary parts.
/// CSigmoid(z) = σ(Re(z)) + i·σ(Im(z))
///
/// Properties:
/// - Bounded output in [0,1] × [0,1]i
/// - Useful for gating mechanisms in complex-valued architectures
pub fn complex_sigmoid(z: &ComplexState) -> Result<ComplexState> {
    // sigmoid(x) = 1 / (1 + exp(-x))
    let neg_real = z.real.neg()?;
    let neg_imag = z.imag.neg()?;
    let ones_r = Tensor::ones_like(&z.real)?;
    let ones_i = Tensor::ones_like(&z.imag)?;
    
    let real_out = (&ones_r / (ones_r.clone() + neg_real.exp()?)?)?;
    let imag_out = (&ones_i / (ones_i.clone() + neg_imag.exp()?)?)?;
    
    Ok(ComplexState {
        real: real_out,
        imag: imag_out,
    })
}

/// Phase-Amplitude Decoupled Activation (PADA):
/// Applies separate non-linearities to magnitude and phase.
/// PADA(z) = f_mag(|z|) · e^{i · f_phase(θ)}
///
/// Default: f_mag = tanh, f_phase = identity (phase-preserving)
/// This activation decouples "what" (magnitude/energy) from "where" (phase/angle).
///
/// Properties:
/// - Ideal for SODALS where phase carries semantic routing information
/// - Magnitude is bounded by tanh, preventing energy explosion
/// - Phase is preserved, maintaining concept identity through layers
pub fn phase_amplitude_activation(z: &ComplexState) -> Result<ComplexState> {
    let magnitude = z.magnitude()?;
    let eps = 1e-8;
    let magnitude_safe = (magnitude.clone() + eps)?;
    
    // f_mag(|z|) = tanh(|z|) — bounded, smooth
    let activated_mag = magnitude.tanh()?;
    
    // Scale = tanh(|z|) / |z| — preserves phase, applies magnitude non-linearity
    let scale = (&activated_mag / &magnitude_safe)?;
    
    let real_out = (&z.real * &scale)?;
    let imag_out = (&z.imag * &scale)?;
    
    Ok(ComplexState {
        real: real_out,
        imag: imag_out,
    })
}

// ============================================================================
// Complex Weight Initialization Utilities
// ============================================================================

/// Generate Rayleigh-distributed magnitudes + Uniform phases for complex weight init.
/// |W| ~ Rayleigh(σ), θ ~ U(-π, π)
/// W = |W| · e^{iθ} = |W|cos(θ) + i|W|sin(θ)
///
/// σ is typically set to 1/√(fan_in) for variance-preserving initialization.
pub fn complex_glorot_init(
    fan_in: usize,
    fan_out: usize,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let sigma = 1.0 / (fan_in as f64).sqrt();
    let n = fan_in * fan_out;
    
    // Rayleigh magnitude: |W| = σ * sqrt(-2 * ln(U))  where U ~ Uniform(0,1)
    // Approximate with normal distribution: |W| ~ |N(0, σ)|
    let mag_data: Vec<f32> = (0..n)
        .map(|i| {
            // Simple deterministic pseudo-random for reproducibility
            let u = ((i as f64 * 0.6180339887).fract() + 0.001).min(0.999);
            (sigma * (-2.0 * u.ln()).sqrt()) as f32
        })
        .collect();
    
    // Uniform phase: θ ~ U(-π, π)
    let phase_data: Vec<f32> = (0..n)
        .map(|i| {
            let u = ((i as f64 * 0.7548776662).fract());
            (u * 2.0 * std::f64::consts::PI - std::f64::consts::PI) as f32
        })
        .collect();
    
    let mag = Tensor::from_vec(mag_data, (fan_out, fan_in), device)?;
    let phase = Tensor::from_vec(phase_data, (fan_out, fan_in), device)?;
    
    // W_real = |W| * cos(θ),  W_imag = |W| * sin(θ)
    let cos_phase = phase.cos()?;
    let sin_phase = phase.sin()?;
    
    let w_real = (&mag * &cos_phase)?.to_dtype(dtype)?;
    let w_imag = (&mag * &sin_phase)?.to_dtype(dtype)?;
    
    Ok((w_real, w_imag))
}
