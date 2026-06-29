use ndarray::{Array1, Array2, Zip};
use num_complex::Complex64;
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign};
use anyhow::Result;

// ============================================================================
// RESIDUAL VECTOR FOR HIERARCHICAL QUANTIZATION
// ============================================================================

/// Residual vector for hierarchical quantization
/// Stores the "2% nuance" that F4 quantization loses
#[derive(Debug, Clone, Copy)]
pub struct ResidualVector {
    /// Phase residual (radians)
    pub phase_delta: f32,
    
    /// Magnitude residual (normalized)
    pub magnitude_delta: f32,
}

impl ResidualVector {
    pub fn zero() -> Self {
        Self {
            phase_delta: 0.0,
            magnitude_delta: 0.0,
        }
    }
}

// Implement Add trait (required by Zero trait)
impl Add for ResidualVector {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        ResidualVector {
            phase_delta: self.phase_delta + other.phase_delta,
            magnitude_delta: self.magnitude_delta + other.magnitude_delta,
        }
    }
}

// Implement Zero trait to support Array1::zeros<ResidualVector>()
use rustfft::num_traits::Zero;

impl Zero for ResidualVector {
    fn zero() -> Self {
        ResidualVector {
            phase_delta: 0.0,
            magnitude_delta: 0.0,
        }
    }
    
    fn is_zero(&self) -> bool {
        self.phase_delta == 0.0 && self.magnitude_delta == 0.0
    }
}

// ============================================================================
// SEMANTIC WAVE WITH HIGH-PRECISION MODE
// ============================================================================

/// SemanticWave: SIMD-vectorized quantum state with optional residual storage
#[derive(Debug, Clone)]
pub struct SemanticWave {
    /// Wave function amplitudes: complex-valued quantum states (II, III, III^N notation)
    pub amplitudes: Array1<Complex64>,
    
    /// High-precision residuals (optional)
    /// Stores the difference between exact and quantized values
    pub(crate) residuals: Option<Array1<ResidualVector>>,

    /// Angular velocity (omega = dtheta/dt)
    pub angular_velocity: f32,
}

impl SemanticWave {
    #[inline]
    pub fn new(size: usize) -> Self {
        Self {
            amplitudes: Array1::zeros(size),
            residuals: None,
            angular_velocity: 0.0,
        }
    }

    #[inline]
    pub fn from_vec(data: Vec<Complex64>) -> Self {
        Self {
            amplitudes: Array1::from_vec(data),
            residuals: None,
            angular_velocity: 0.0,
        }
    }

    /// Initialize from real values (zero imaginary part)
    #[inline]
    pub fn from_reals(reals: Vec<f64>) -> Self {
        Self {
            amplitudes: Array1::from_vec(
                reals.into_iter().map(|r| Complex64::new(r, 0.0)).collect()
            ),
            residuals: None,
            angular_velocity: 0.0,
        }
    }

    /// Enable high-precision mode (allocate residual storage)
    pub fn enable_high_precision(&mut self) {
        if self.residuals.is_none() {
            let size = self.amplitudes.len();
            self.residuals = Some(Array1::from_elem(size, ResidualVector::zero()));
        }
    }

    /// Check if high-precision mode is active
    pub fn has_high_precision(&self) -> bool {
        self.residuals.is_some()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.amplitudes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.amplitudes.is_empty()
    }

    #[inline]
    pub fn zero(size: usize) -> Self {
        Self::new(size)
    }

    pub fn from_phases(phases: &[f64]) -> Self {
        let amplitudes = Array1::from_vec(
            phases.iter()
                .map(|&theta| Complex64::new(theta.cos(), theta.sin()))
                .collect()
        );
        Self { amplitudes, residuals: None, angular_velocity: 0.0 }
    }

    // VECTORIZED OPERATIONS

    #[inline]
    pub fn scale_mut(&mut self, factor: f64) {
        self.amplitudes.mapv_inplace(|z| z * factor);
        // Residuals scale linearly too
        if let Some(ref mut res) = self.residuals {
            res.mapv_inplace(|mut r| {
                r.magnitude_delta *= factor as f32;
                r
            });
        }
    }

    #[inline]
    pub fn scaled(&self, factor: f64) -> Self {
        let mut result = self.clone();
        result.scale_mut(factor);
        result
    }

    #[inline]
    pub fn hadamard_mul_mut(&mut self, other: &SemanticWave) {
        Zip::from(&mut self.amplitudes)
            .and(&other.amplitudes)
            .for_each(|a, &b| *a *= b);
    }

    #[inline]
    pub fn hadamard_mul(&self, other: &SemanticWave) -> Self {
        Self {
            amplitudes: &self.amplitudes * &other.amplitudes,
            residuals: None, // Residuals invalidated by nonlinear op
            angular_velocity: 0.0,
        }
    }

    #[inline]
    pub fn inner_product(&self, other: &SemanticWave) -> Complex64 {
        Zip::from(&self.amplitudes)
            .and(&other.amplitudes)
            .fold(Complex64::new(0.0, 0.0), |acc, &a, &b| acc + a.conj() * b)
    }

    #[inline]
    pub fn norm_sqr(&self) -> f64 {
        self.amplitudes.iter()
            .map(|z| z.norm_sqr())
            .sum()
    }

    #[inline]
    pub fn norm(&self) -> f64 {
        self.norm_sqr().sqrt()
    }

    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > 1e-12 {
            self.scale_mut(1.0 / norm);
        }
    }

    pub fn normalized(&self) -> Self {
        let norm = self.norm();
        if norm > 1e-12 {
            self.scaled(1.0 / norm)
        } else {
            self.clone()
        }
    }

    pub fn resonance_norm(&self, weights: &Array1<f64>) -> f64 {
        assert_eq!(self.amplitudes.len(), weights.len());
        
        Zip::from(&self.amplitudes)
            .and(weights)
            .fold(0.0, |acc, &z, &w| acc + w * z.norm_sqr())
    }

    pub fn phases(&self) -> Array1<f64> {
        self.amplitudes.mapv(|z| z.arg())
    }

    pub fn conjugate_mut(&mut self) {
        self.amplitudes.mapv_inplace(|z| z.conj());
    }

    pub fn conjugate(&self) -> Self {
        Self {
            amplitudes: self.amplitudes.mapv(|z| z.conj()),
            residuals: self.residuals.clone(),
            angular_velocity: self.angular_velocity,
        }
    }

    pub fn rotate_phase_mut(&mut self, theta: f64) {
        let rotation = Complex64::new(theta.cos(), theta.sin());
        self.amplitudes.mapv_inplace(|z| z * rotation);
        
        // Residuals also rotate
        if let Some(ref mut res) = self.residuals {
            res.mapv_inplace(|mut r| {
                r.phase_delta += theta as f32;
                r
            });
        }
    }

    pub fn rotated_phase(&self, theta: f64) -> Self {
        let mut result = self.clone();
        result.rotate_phase_mut(theta);
        result
    }

    pub fn weighted_sum(states: &[SemanticWave], weights: &[f64]) -> SemanticWave {
        assert_eq!(states.len(), weights.len());
        assert!(!states.is_empty());

        let size = states[0].len();
        let mut result = Array1::zeros(size);

        for (state, &weight) in states.iter().zip(weights.iter()) {
            result.scaled_add(Complex64::new(weight, 0.0), &state.amplitudes);
        }

        SemanticWave { amplitudes: result, residuals: None, angular_velocity: 0.0 }
    }

    #[inline]
    pub fn add_scaled(&mut self, other: &SemanticWave, scale: f64) {
        self.amplitudes.scaled_add(Complex64::new(scale, 0.0), &other.amplitudes);
    }

    #[inline]
    pub fn sub_scaled(&mut self, other: &SemanticWave, scale: f64) {
        self.amplitudes.scaled_add(Complex64::new(-scale, 0.0), &other.amplitudes);
    }

    #[inline]
    pub fn as_slice(&self) -> &[Complex64] {
        self.amplitudes.as_slice().unwrap()
    }

    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [Complex64] {
        self.amplitudes.as_slice_mut().unwrap()
    }

    /// Compute cross-network Wronskian of two waves.
    /// W(f,g) = sum_j (f_j * g'_j - f'_j * g_j)
    pub fn compute_cross_network_wronskian(wave_f: &SemanticWave, wave_g: &SemanticWave) -> f64 {
        let len = wave_f.len().min(wave_g.len());
        if len == 0 {
            return 0.0;
        }
        
        let mut sum = Complex64::new(0.0, 0.0);
        for i in 0..len {
            let f = wave_f.amplitudes[i];
            let g = wave_g.amplitudes[i];
            
            // Integrate global angular velocity and a node-specific default frequency to ensure linear independence
            let w_f = Complex64::new(0.0, wave_f.angular_velocity as f64 + (i as f64 + 1.0) * 0.05);
            let w_g = Complex64::new(0.0, wave_g.angular_velocity as f64 + (i as f64 + 1.0) * 0.08);
            
            let f_prime = w_f * f;
            let g_prime = w_g * g;
            sum += f * g_prime - f_prime * g;
        }
        sum.norm()
    }
}

// OPERATOR OVERLOADING

impl Add for SemanticWave {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            amplitudes: self.amplitudes + rhs.amplitudes,
            residuals: None,
            angular_velocity: 0.0,
        }
    }
}

impl Sub for SemanticWave {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            amplitudes: self.amplitudes - rhs.amplitudes,
            residuals: None,
            angular_velocity: 0.0,
        }
    }
}

impl Mul<f64> for SemanticWave {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        self.scaled(rhs)
    }
}

impl AddAssign for SemanticWave {
    fn add_assign(&mut self, rhs: Self) {
        self.amplitudes += &rhs.amplitudes;
        self.residuals = None; // Invalidate residuals
    }
}

impl SubAssign for SemanticWave {
    fn sub_assign(&mut self, rhs: Self) {
        self.amplitudes -= &rhs.amplitudes;
        self.residuals = None;
    }
}

// HIERARCHICAL RESIDUAL QUANTIZATION (F4 + RESIDUALS)

/// F4 Galois Field element
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum F4 {
    Zero = 0,
    One = 1,
    W = 2,
    W2 = 3,
}

impl F4 {
    /// UPGRADED: from_complex with residual calculation
    /// 
    /// Returns (F4 center, phase_residual, magnitude_residual)
    /// Residuals = actual - center
    pub fn from_complex_with_residual(c: Complex64) -> (Self, f32, f32) {
        let magnitude = c.norm();
        let phase = c.arg();
        
        // Magnitude threshold for Zero
        if magnitude < 0.1 {
            return (F4::Zero, phase as f32, magnitude as f32);
        }
        
        // CRITICAL FIX: Quantize to nearest F4 center, compute residual
        // 
        // F4 centers in phase space:
        // - One:  0II    (0 rad)
        // - W:    90II   (II/2 rad)
        // - W2:   180II  (II rad)
        // - Zero: 270II  (-II/2 rad) [but we use magnitude check]
        
        let centers = [
            (F4::One, 0.0),                              // 0II
            (F4::W, std::f64::consts::FRAC_PI_2),       // 90II
            (F4::W2, std::f64::consts::PI),             // 180II
            (F4::Zero, -std::f64::consts::FRAC_PI_2),   // 270II
        ];
        
        // Find nearest center
        let (selected_f4, center_phase) = centers.iter()
            .min_by(|(_, p1), (_, p2)| {
                let dist1 = (phase - p1).abs();
                let dist2 = (phase - p2).abs();
                dist1.partial_cmp(&dist2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap();
        
        // Calculate residual
        let phase_delta = (phase - center_phase) as f32;
        let magnitude_delta = (magnitude - 1.0) as f32; // Centers assume unit magnitude
        
        (selected_f4, phase_delta, magnitude_delta)
    }

    /// Legacy quantization (no residuals) - DEPRECATED
    pub fn from_complex(c: Complex64) -> Self {
        let (f4, _, _) = Self::from_complex_with_residual(c);
        f4
    }

    /// Reconstruct complex from F4 + residual
    /// 
    /// CRITICAL: This preserves the "2% nuance"!
    pub fn to_complex_with_residual(&self, residual: ResidualVector) -> Complex64 {
        // Get F4 center
        let base = self.to_ideal_complex();
        
        // Apply residual corrections
        let magnitude = base.norm() + residual.magnitude_delta as f64;
        let phase = base.arg() + residual.phase_delta as f64;
        
        Complex64::from_polar(magnitude, phase)
    }

    /// F4 center (without residual)
    pub fn to_ideal_complex(&self) -> Complex64 {
        match self {
            F4::Zero => Complex64::new(0.0, 0.0),
            F4::One  => Complex64::new(1.0, 0.0),
            F4::W    => Complex64::new(0.0, 1.0),
            F4::W2   => Complex64::new(-1.0, 0.0),
        }
    }
}

impl std::ops::Add for F4 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, other: Self) -> Self {
        let val = (self as u8) ^ (other as u8);
        unsafe { std::mem::transmute(val) }
    }
}

impl std::ops::Mul for F4 {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        const MUL_TABLE: [[u8; 4]; 4] = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
        ];
        let val = MUL_TABLE[self as usize][other as usize];
        unsafe { std::mem::transmute(val) }
    }
}

pub const PARITY_CHECK_MATRIX: [[F4; 7]; 3] = [
    [F4::One, F4::One, F4::One, F4::Zero, F4::One, F4::Zero, F4::Zero],
    [F4::One, F4::W,   F4::W2,  F4::One,  F4::Zero, F4::One, F4::Zero],
    [F4::One, F4::W2,  F4::W,   F4::One,  F4::Zero, F4::Zero, F4::One],
];

impl SemanticWave {
    /// Convert to Skew Code with HIGH PRECISION
    /// 
    /// UPGRADED: Stores both F4 quantization + residuals
    pub fn to_skew_code_high_precision(&mut self) -> Vec<(F4, ResidualVector)> {
        // Enable residual storage if not already
        self.enable_high_precision();
        
        self.amplitudes.iter()
            .enumerate()
            .map(|(i, &c)| {
                let (f4, phase_delta, magnitude_delta) = F4::from_complex_with_residual(c);
                let residual = ResidualVector {
                    phase_delta,
                    magnitude_delta,
                };
                
                // Store residual
                if let Some(ref mut res) = self.residuals {
                    res[i] = residual;
                }
                
                (f4, residual)
            })
            .collect()
    }

    /// Legacy low-precision conversion
    pub fn to_skew_code(&self) -> Vec<F4> {
        self.amplitudes.iter()
            .map(|&c| F4::from_complex(c))
            .collect()
    }

    /// Heal from Skew Code with HIGH PRECISION
    /// 
    /// CRITICAL: Reconstructs using residuals to preserve nuance
    pub fn heal_from_skew_code_high_precision(&mut self, code_with_residuals: &[(F4, ResidualVector)]) {
        for (i, &(symbol, residual)) in code_with_residuals.iter().enumerate() {
            if i < self.amplitudes.len() {
                // Reconstruct with residual
                let current_norm = self.amplitudes[i].norm();
                let reconstructed = symbol.to_complex_with_residual(residual);
                
                // Preserve ORIGINAL magnitude if possible
                let new_norm = if current_norm < 0.01 { 
                    reconstructed.norm() 
                } else { 
                    current_norm 
                };
                
                let phase = reconstructed.arg();
                self.amplitudes[i] = Complex64::from_polar(new_norm, phase);
                
                // Store residual
                if let Some(ref mut res) = self.residuals {
                    res[i] = residual;
                }
            }
        }
    }

    /// Legacy healing (low precision)
    pub fn heal_from_skew_code(&mut self, code: &[F4]) {
        for (i, &symbol) in code.iter().enumerate() {
            if i < self.amplitudes.len() {
                let current_norm = self.amplitudes[i].norm();
                let ideal_dir = symbol.to_ideal_complex();
                
                let new_norm = if current_norm < 0.01 { 1.0 } else { current_norm };
                self.amplitudes[i] = ideal_dir * new_norm;
            }
        }
    }

    pub fn check_parity(&self) -> bool {
        let code = self.to_skew_code();
        let mut sum = F4::Zero;
        
        for sym in code {
            sum = sum + sym;
        }

        sum == F4::Zero
    }

    pub fn calculate_syndrome(&self) -> Vec<F4> {
        let code = self.to_skew_code();
        let n = code.len().min(7);
        
        let mut syndrome = vec![F4::Zero; 3];

        for i in 0..3 {
            let mut sum = F4::Zero;
            for j in 0..n {
                let h_val = PARITY_CHECK_MATRIX[i][j];
                let prod = code[j] * h_val;
                sum = sum + prod;
            }
            syndrome[i] = sum;
        }
        
        syndrome
    }
}

// ERROR CORRECTION WITH NUANCE PRESERVATION

#[derive(Debug, Clone, Copy)]
pub enum SkewCodeError {
    Uncorrectable,
    InvalidLength,
}

impl std::fmt::Display for SkewCodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SkewCodeError::Uncorrectable => write!(f, "Error uncorrectable"),
            SkewCodeError::InvalidLength => write!(f, "Invalid length"),
        }
    }
}

impl std::error::Error for SkewCodeError {}

pub fn encode_skew(data: &[F4; 4]) -> [F4; 7] {
    const GENERATOR: [[F4; 7]; 4] = [
        [F4::One, F4::Zero, F4::Zero, F4::Zero, F4::One, F4::One,  F4::One],
        [F4::Zero, F4::One, F4::Zero, F4::Zero, F4::One, F4::W,    F4::W2],
        [F4::Zero, F4::Zero, F4::One, F4::Zero, F4::One, F4::W2,   F4::W],
        [F4::Zero, F4::Zero, F4::Zero, F4::One, F4::Zero, F4::One, F4::One],
    ];
    
    let mut codeword = [F4::Zero; 7];
    
    for i in 0..4 {
        for j in 0..7 {
            let term = data[i] * GENERATOR[i][j];
            codeword[j] = codeword[j] + term;
        }
    }
    
    codeword
}

pub fn decode_skew(received: &[F4; 7]) -> Result<[F4; 4], SkewCodeError> {
    let mut syndrome = [F4::Zero; 3];
    
    for i in 0..3 {
        let mut sum = F4::Zero;
        for j in 0..7 {
            let term = received[j] * PARITY_CHECK_MATRIX[i][j];
            sum = sum + term;
        }
        syndrome[i] = sum;
    }
    
    if syndrome.iter().all(|&s| s == F4::Zero) {
        return Ok([received[0], received[1], received[2], received[3]]);
    }
    
    let error_location = match (syndrome[0] as u8, syndrome[1] as u8, syndrome[2] as u8) {
        (1, 1, 1) => Some(0),
        (0, 1, 0) => Some(1),
        (0, 0, 1) => Some(2),
        (0, 1, 1) => Some(3),
        (1, 1, 0) => Some(4),
        (1, 2, 3) => Some(5),
        (1, 3, 2) => Some(6),
        _ => None,
    };
    
    if let Some(loc) = error_location {
        let mut corrected = *received;
        corrected[loc] = corrected[loc] + F4::One;
        Ok([corrected[0], corrected[1], corrected[2], corrected[3]])
    } else {
        Err(SkewCodeError::Uncorrectable)
    }
}

// PHASE 23: CLIFFORD GEOMETRIC PRODUCT - DEEPCUBE AUDIT DEFENSE

/// GeometricProduct trait: implements true Cl(3,0) or Cl(2,0) Clifford algebra
/// Geometric Product: u v = u · v + u ∧ v (scalar + bivector)
pub trait GeometricProduct {
    /// Returns (inner_product_scalar, outer_product_bivector_magnitude)
    /// Inner product: symmetric part u · v = 0.5(uv + vu) -> scalar
    /// Outer product: anti-symmetric part u ∧ v = 0.5(uv - vu) -> bivector
    fn geometric_product(&self, other: &Self) -> (f64, f64);
}

/// Cl3Multivector: 3D Clifford multivector with scalar + vector + bivector + trivector components
/// Grade 0: scalar (w)
/// Grade 1: vector (x, y, z)
/// Grade 2: bivector (xy, xz, yz)
/// Grade 3: trivector (xyz)
#[derive(Debug, Clone, Copy)]
pub struct Cl3Multivector {
    pub scalar: f64,      // Grade 0: w
    pub x: f64,         // Grade 1: e1
    pub y: f64,         // Grade 1: e2
    pub z: f64,         // Grade 1: e3
    pub xy: f64,        // Grade 2: e1∧e2
    pub xz: f64,        // Grade 2: e1∧e3
    pub yz: f64,        // Grade 2: e2∧e3
    pub xyz: f64,       // Grade 3: e1∧e2∧e3 (pseudoscalar)
}

impl Cl3Multivector {
    /// Create a pure vector (grade 1)
    pub fn vector(x: f64, y: f64, z: f64) -> Self {
        Self {
            scalar: 0.0, x, y, z, xy: 0.0, xz: 0.0, yz: 0.0, xyz: 0.0,
        }
    }

    /// Create a scalar (grade 0)
    pub fn scalar(s: f64) -> Self {
        Self {
            scalar: s, x: 0.0, y: 0.0, z: 0.0, xy: 0.0, xz: 0.0, yz: 0.0, xyz: 0.0,
        }
    }

    /// Extract bivector magnitude for use in semantic wave repair
    pub fn bivector_magnitude(&self) -> f64 {
        (self.xy * self.xy + self.xz * self.xz + self.yz * self.yz).sqrt()
    }

    /// Extract the scalar (inner product) component
    pub fn inner_scalar(&self) -> f64 {
        self.scalar
    }
}

impl GeometricProduct for Cl3Multivector {
    /// Geometric Product: u v = u · v + u ∧ v
    /// Explicit separation:
    /// - Inner product (scalar): u · v = 0.5(uv + vu)
    /// - Outer product (bivector): u ∧ v = 0.5(uv - vu)
    fn geometric_product(&self, other: &Self) -> (f64, f64) {
        // Full geometric product: uv = u*v (all grades interact)
        let uv = self.multiply_full(other);
        let vu = other.multiply_full(self);
        
        // Inner product: symmetric part (scalar)
        // u · v = 0.5(uv + vu) -> scalar component only
        let inner = 0.5 * (uv.scalar + vu.scalar);
        
        // Outer product: anti-symmetric part (bivector)
        // u ∧ v = 0.5(uv - vu) -> bivector components only
        let bivector_xy = 0.5 * (uv.xy - vu.xy);
        let bivector_xz = 0.5 * (uv.xz - vu.xz);
        let bivector_yz = 0.5 * (uv.yz - vu.yz);
        let outer_magnitude = (bivector_xy * bivector_xy + 
                               bivector_xz * bivector_xz + 
                               bivector_yz * bivector_yz).sqrt();
        
        (inner, outer_magnitude)
    }
}

impl Cl3Multivector {
    /// Full Clifford multiplication (all grade interactions)
    /// This is the fundamental operation: uv = grade_decompose(sum of all products)
    fn multiply_full(&self, other: &Self) -> Self {
        // Grade 0 (scalar) * anything
        let scalar = self.scalar * other.scalar + self.x * other.x + self.y * other.y + self.z * other.z
                   - self.xy * other.xy - self.xz * other.xz - self.yz * other.yz - self.xyz * other.xyz;
        
        // Grade 1 (vector) components
        let x = self.scalar * other.x + self.x * other.scalar - self.y * other.xy - self.z * other.xz
              + self.xy * other.y + self.xz * other.z - self.yz * other.xyz + self.xyz * other.yz;
        
        let y = self.scalar * other.y + self.y * other.scalar + self.x * other.xy - self.z * other.yz
              - self.xy * other.x + self.yz * other.z + self.xz * other.xyz + self.xyz * other.xz;
        
        let z = self.scalar * other.z + self.z * other.scalar + self.x * other.xz + self.y * other.yz
              - self.xz * other.x - self.yz * other.y - self.xy * other.xyz + self.xyz * other.xy;
        
        // Grade 2 (bivector) components
        let xy = self.scalar * other.xy + self.xy * other.scalar + self.x * other.y - self.y * other.x
               + self.z * other.xyz - self.xyz * other.z + self.xz * other.yz - self.yz * other.xz;
        
        let xz = self.scalar * other.xz + self.xz * other.scalar + self.x * other.z - self.z * other.x
               - self.y * other.xyz + self.xyz * other.y + self.xy * other.yz - self.yz * other.xy;
        
        let yz = self.scalar * other.yz + self.yz * other.scalar + self.y * other.z - self.z * other.y
               + self.x * other.xyz - self.xyz * other.x - self.xy * other.xz + self.xz * other.xy;
        
        // Grade 3 (trivector/pseudoscalar)
        let xyz = self.scalar * other.xyz + self.xyz * other.scalar
                + self.x * other.yz - self.y * other.xz + self.z * other.xy
                + self.xy * other.z - self.xz * other.y + self.yz * other.x;
        
        Self { scalar, x, y, z, xy, xz, yz, xyz }
    }
}

/// Extend ResidualVector with Clifford geometric product for semantic wave repair
impl GeometricProduct for ResidualVector {
    /// Treat residuals as 2D Clifford vectors (simplified Cl(2,0))
    /// phase_delta -> x component, magnitude_delta -> y component
    fn geometric_product(&self, other: &Self) -> (f64, f64) {
        // Convert to Cl(2,0) multivectors - using _ prefix for intentionally unused components
        let _u_scalar = 0.0_f64;
        let u_x = self.phase_delta as f64;
        let u_y = self.magnitude_delta as f64;
        let _u_xy = 0.0_f64; // no initial bivector
        
        let _v_scalar = 0.0_f64;
        let v_x = other.phase_delta as f64;
        let v_y = other.magnitude_delta as f64;
        let _v_xy = 0.0_f64;
        
        // Geometric product in Cl(2,0):
        // uv = (u·v) + (u∧v)
        // Inner product (scalar): u·v = u_x*v_x + u_y*v_y
        let inner = u_x * v_x + u_y * v_y;
        
        // Outer product (bivector): u∧v = (u_x*v_y - u_y*v_x) e12
        let outer = (u_x * v_y - u_y * v_x).abs();
        
        (inner, outer)
    }
}

/// Apply Clifford geometric product to semantic wave repair
/// Uses bivector magnitude from geometric product as a resonance metric
pub fn apply_clifford_resonance(wave: &SemanticWave) -> f64 {
    if wave.amplitudes.len() < 2 {
        return 0.0;
    }
    
    let mut total_bivector = 0.0;
    let mut total_inner = 0.0;
    
    // Compute geometric product between adjacent amplitudes
    for i in 0..wave.amplitudes.len() - 1 {
        let z1 = wave.amplitudes[i];
        let z2 = wave.amplitudes[i + 1];
        
        // Convert complex to 3D vector (embedding in Cl(3,0))
        let u = Cl3Multivector::vector(z1.re, z1.im, 0.0);
        let v = Cl3Multivector::vector(z2.re, z2.im, 0.0);
        
        let (inner, outer) = u.geometric_product(&v);
        total_inner += inner.abs();
        total_bivector += outer;
    }
    
    // Return bivector magnitude as resonance metric
    // High bivector = high rotational independence (healthy)
    // Low bivector = collinear vectors (potential loop/hallucination)
    let _avg_inner = total_inner / (wave.amplitudes.len() as f64 - 1.0);
    total_bivector / (wave.amplitudes.len() as f64 - 1.0)
}

/// PHASE 47.5: STRUCTURAL ORTHOGONALITY ENFORCEMENT
/// 
/// Generates a strictly orthogonal 4096-D rotation matrix for R90.
/// R^T R = I, det(R) = 1, R^4 = I.
/// 
/// Uses a block-diagonal structure of 2x2 rotation matrices:
/// [[0,  1],
///  [-1, 0]]
/// 
/// This preserves the L2 norm of the thought vector during rotation and
/// ensures deterministic 4-step cycles without precision drift.
pub fn generate_orthogonal_r90(dim: usize) -> Array2<f64> {
    let mut matrix = Array2::<f64>::zeros((dim, dim));
    
    // Fill block-diagonal with 90-degree Givens rotations
    for i in (0..dim).step_by(2) {
        if i + 1 < dim {
            // [[cos 90, sin 90], [-sin 90, cos 90]] = [[0, 1], [-1, 0]]
            matrix[[i, i + 1]] = 1.0;
            matrix[[i + 1, i]] = -1.0;
        } else {
            // Handle odd dimension (rare in 4096-D but for safety)
            matrix[[i, i]] = 1.0;
        }
    }
    
    eprintln!("[PHYSICS] Generated strictly orthogonal R90 matrix (dim={})", dim);
    matrix
}

/// CRITICAL: Repair with HIGH PRECISION (preserves nuance)
/// 
/// Strategy:
/// 1. Convert current wave to (F4, Residual) pairs
/// 2. Process in 7-symbol chunks (Skew code blocks)
/// 3. Decode each block to correct errors in F4 layer
/// 4. Re-encode with corrected F4s
/// 5. Preserve ORIGINAL residuals from high-precision storage
/// 6. Reconstruct Complex64 from (CorrectedF4 + ORIGINALResidual)
pub fn repair_semantic_wave(wave: &SemanticWave) -> Result<SemanticWave> {
    let mut result = wave.clone();
    result.enable_high_precision();
    
    // Step 1: Extract (F4, Residual) pairs with high precision
    let code_with_residuals = result.to_skew_code_high_precision();
    let mut repaired_code_with_residuals = Vec::new();
    
    // Step 2-5: Process in 7-symbol chunks
    for chunk in code_with_residuals.chunks(7) {
        if chunk.len() == 7 {
            // Extract F4 blocks and residuals separately
            let mut block: [F4; 7] = [F4::Zero; 7];
            let mut residuals: [ResidualVector; 7] = [ResidualVector::zero(); 7];
            
            for (i, &(f4, res)) in chunk.iter().enumerate() {
                block[i] = f4;
                residuals[i] = res;
            }
            
            // Decode F4 layer with error correction
            match decode_skew(&block) {
                Ok(corrected_data) => {
                    // Re-encode with corrected data
                    let corrected_block = encode_skew(&corrected_data);
                    
                    // CRITICAL: Re-attach ORIGINAL residuals to corrected F4s
                    // This preserves the semantic nuance while fixing errors
                    for (i, &f4) in corrected_block.iter().enumerate() {
                        repaired_code_with_residuals.push((f4, residuals[i]));
                    }
                }
                Err(_) => {
                    // Uncorrectable error in this block
                    // Fall back to ORIGINAL F4s with residuals
                    for (i, &f4) in block.iter().enumerate() {
                        repaired_code_with_residuals.push((f4, residuals[i]));
                    }
                }
            }
        } else {
            // Partial block (end of data) - no repair, preserve as-is
            repaired_code_with_residuals.extend_from_slice(chunk);
        }
    }
    
    // Step 6: Reconstruct amplitudes with high precision
    result.heal_from_skew_code_high_precision(&repaired_code_with_residuals);
    
    Ok(result)
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual_vector_creation() {
        let res = ResidualVector::zero();
        assert_eq!(res.phase_delta, 0.0);
        assert_eq!(res.magnitude_delta, 0.0);
    }

    #[test]
    fn test_semantic_wave_high_precision() {
        let mut wave = SemanticWave::new(10);
        assert!(!wave.has_high_precision());
        
        wave.enable_high_precision();
        assert!(wave.has_high_precision());
    }

    #[test]
    fn test_f4_quantization_with_residual() {
        let running = Complex64::from_polar(1.0, 1.570);  // "Running"
        let fleeing = Complex64::from_polar(1.0, 1.601);  // "Fleeing"
        
        let (f4_r, phase_r, mag_r) = F4::from_complex_with_residual(running);
        let (f4_f, phase_f, mag_f) = F4::from_complex_with_residual(fleeing);
        
        // Both should map to W (imaginary unit / 90° phase)
        assert_eq!(f4_r, F4::W);
        assert_eq!(f4_f, F4::W);
        
        // But residuals should preserve difference
        let res_r = ResidualVector {
            phase_delta: phase_r,
            magnitude_delta: mag_r,
        };
        let res_f = ResidualVector {
            phase_delta: phase_f,
            magnitude_delta: mag_f,
        };
        
        let reconstructed_r = f4_r.to_complex_with_residual(res_r);
        let reconstructed_f = f4_f.to_complex_with_residual(res_f);
        
        let phase_diff = (reconstructed_r.arg() - reconstructed_f.arg()).abs();
        
        // Should preserve ~2% nuance
        assert!(phase_diff > 0.025, "Phase difference should be > 0.025 rad, got {}", phase_diff);
        assert!(phase_diff < 0.035, "Phase difference should be < 0.035 rad, got {}", phase_diff);
    }

    #[test]
    fn test_skew_code_encode_decode() {
        let data = [F4::One, F4::W, F4::W2, F4::Zero];
        let encoded = encode_skew(&data);
        
        let decoded = decode_skew(&encoded).expect("Decoding failed");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_skew_code_single_error_correction() {
        let data = [F4::One, F4::W, F4::W2, F4::Zero];
        let mut encoded = encode_skew(&data);
        
        // Introduce single-bit error at position 0
        encoded[0] = encoded[0].add(F4::One);
        
        let decoded = decode_skew(&encoded).expect("Decoding failed");
        assert_eq!(decoded, data, "Should correct single error");
    }

    #[test]
    fn test_repair_semantic_wave_high_precision() {
        // Create wave with data
        let mut wave = SemanticWave::from_reals(vec![
            0.5, 0.6, 0.7, 0.8, 0.3, 0.4, 0.5,
            0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        ]);
        
        // Enable high precision and capture ORIGINAL with residuals
        wave.enable_high_precision();
        let original_with_residuals = wave.to_skew_code_high_precision();
        
        // Verify residuals are non-zero (nuance captured)
        let has_nonzero_residuals = original_with_residuals.iter()
            .any(|(_, res)| res.phase_delta.abs() > 1e-6 || res.magnitude_delta.abs() > 1e-6);
        assert!(has_nonzero_residuals, "Should capture semantic nuance in residuals");
        
        // Repair should work
        let repaired = repair_semantic_wave(&wave).expect("Repair failed");
        assert_eq!(repaired.len(), wave.len());
        
        // Should have residuals preserved
        assert!(repaired.has_high_precision());
    }

    #[test]
    fn test_semantic_wave_operations_preserve_residuals() {
        let mut wave = SemanticWave::from_reals(vec![0.1, 0.2, 0.3]);
        wave.enable_high_precision();
        
        // Scaling should preserve residual structure
        wave.scale_mut(2.0);
        assert!(wave.has_high_precision());
        
        // Phase rotation should update residuals
        let phi = std::f64::consts::PI / 4.0;
        wave.rotate_phase_mut(phi);
        assert!(wave.has_high_precision());
    }
}














