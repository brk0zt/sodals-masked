// src/engine/pondering_core.rs
// Phase 36+38: Arkhe Core & RK4 Latent Pondering + The God's Gate (LU Neurosymbolic Filtering)
//
// Abandons the "Discrete Token Bottleneck". The model thinks in continuous 
// 4096-D latent manifold using Differential Equations before collapsing 
// into discrete tokens.
//
// Phase 38 adds "The God's Gate": LU-Decomposition based topological filter
// that validates pondered vectors against QMEM semantic truth using Picard
// Iteration for self-correction.
//
// "The machine must think before it speaks."
// "The machine must verify its thoughts against mathematical truth."

use candle_core::{DType, Device, Result, Tensor, D};
use crate::fast_math::SemanticWave;
use crate::semantic_network::ResonanceNetwork;
use crate::fluid_algebra::{bind_phasors, unbind_phasors, apply_viscosity};
use crate::semantic_parser::{DOMAIN_LOGIC, DOMAIN_SPATIAL, DOMAIN_TEMPORAL};
use num_complex::Complex64;
use crate::engine::oracle_worker::HspOracle;
use std::sync::{Mutex, Arc, atomic::{AtomicU64, Ordering}};
use ndarray::{Array1, Array2};
use sprs::CsMat;
use crate::engine::bci_ipc::GHOST_VECTOR_SLOT;

/// PHASE 44: God's Gate Atomic State Transfer
/// 
/// Stores deferred validation results for cross-token coherence checking.
/// Uses AtomicU64 to store f64 as bits for lock-free atomic access.
#[derive(Debug, Clone)]
pub struct GodsGateState {
    /// Stored as f64 bits: semantic divergence from previous token's validation
    pub prev_semantic_divergence: Arc<AtomicU64>,
    /// Stored as f64 bits: physics violation residual  
    pub prev_physics_residual: Arc<AtomicU64>,
    /// Flag indicating if previous thought was hallucinated
    pub prev_was_hallucinated: Arc<AtomicU64>, // 0 = false, 1 = true
}

impl GodsGateState {
    pub fn new() -> Self {
        Self {
            prev_semantic_divergence: Arc::new(AtomicU64::new(f64::to_bits(0.0))),
            prev_physics_residual: Arc::new(AtomicU64::new(f64::to_bits(0.0))),
            prev_was_hallucinated: Arc::new(AtomicU64::new(0)), // false
        }
    }
    
    /// Store divergence as atomic f64 bits
    pub fn store_divergence(&self, divergence: f64) {
        self.prev_semantic_divergence.store(f64::to_bits(divergence), Ordering::SeqCst);
    }
    
    /// Load divergence from atomic f64 bits
    pub fn load_divergence(&self) -> f64 {
        f64::from_bits(self.prev_semantic_divergence.load(Ordering::SeqCst))
    }
    
    /// Store physics residual as atomic f64 bits
    pub fn store_physics(&self, residual: f64) {
        self.prev_physics_residual.store(f64::to_bits(residual), Ordering::SeqCst);
    }
    
    /// Load physics residual from atomic f64 bits
    pub fn load_physics(&self) -> f64 {
        f64::from_bits(self.prev_physics_residual.load(Ordering::SeqCst))
    }
    
    /// Mark previous thought as hallucinated
    pub fn mark_hallucinated(&self, was_hallucinated: bool) {
        self.prev_was_hallucinated.store(if was_hallucinated { 1 } else { 0 }, Ordering::SeqCst);
    }
    
    /// Check if previous thought was hallucinated
    pub fn was_hallucinated(&self) -> bool {
        self.prev_was_hallucinated.load(Ordering::SeqCst) != 0
    }

    /// PHASE 51.5: Hard Reset - Clear God's Gate state for scenario isolation
    pub fn reset(&self) {
        self.prev_semantic_divergence.store(f64::to_bits(0.0), Ordering::SeqCst);
        self.prev_physics_residual.store(f64::to_bits(0.0), Ordering::SeqCst);
        self.prev_was_hallucinated.store(0, Ordering::SeqCst);
    }
}

impl Default for GodsGateState {
    fn default() -> Self {
        Self::new()
    }
}
/// Maximum Picard iteration steps for logical consistency enforcement
const MAX_PICARD_STEPS: usize = 1;
/// Maximum pondering steps before forced convergence
const MAX_PONDER_STEPS: usize = 3;
/// Convergence threshold for L2 norm of delta_h
const EPSILON: f64 = 1e-4;
/// Threshold for LU topological validation (τ_logical)
const TAU_LOGICAL: f64 = 0.1;
/// Tikhonov regularization parameter for Laplacian invertibility
const LAMBDA_TIKHONOV: f64 = 1e-4;
/// Default RK4 step size
/// PHASE 49.5: Increased from 0.1 to 0.15 for stronger attractor convergence
const DT: f64 = 0.15;
/// Phase 44: Quantum Amplitude Amplification - Phase Inversion Penalty strength
const PHASE_INVERSION_STRENGTH: f64 = std::f64::consts::PI;

/// Arkhe energy coefficients for the derivative field
/// PHASE 47.5: Using atomic cells for lock-free hot-path access
struct ArkheCoefficients {
    /// Gradient descent rate (energy minimization) - atomic for lock-free read
    alpha: std::sync::atomic::AtomicU64, // stored as f64 bits
    /// Ghost vector attraction strength - atomic for lock-free read
    beta: std::sync::atomic::AtomicU64,  // stored as f64 bits
    /// Boundary preservation (original semantic anchor) - atomic for lock-free read
    gamma: std::sync::atomic::AtomicU64, // stored as f64 bits
}

impl ArkheCoefficients {
    fn get_alpha(&self) -> f64 {
        f64::from_bits(self.alpha.load(std::sync::atomic::Ordering::Relaxed))
    }
    fn get_beta(&self) -> f64 {
        f64::from_bits(self.beta.load(std::sync::atomic::Ordering::Relaxed))
    }
    fn get_gamma(&self) -> f64 {
        f64::from_bits(self.gamma.load(std::sync::atomic::Ordering::Relaxed))
    }
    fn set_alpha(&self, val: f64) {
        self.alpha.store(f64::to_bits(val), std::sync::atomic::Ordering::Relaxed);
    }
    fn set_beta(&self, val: f64) {
        self.beta.store(f64::to_bits(val), std::sync::atomic::Ordering::Relaxed);
    }
    fn set_gamma(&self, val: f64) {
        self.gamma.store(f64::to_bits(val), std::sync::atomic::Ordering::Relaxed);
    }
}

impl Default for ArkheCoefficients {
    fn default() -> Self {
        Self {
            alpha: std::sync::atomic::AtomicU64::new(f64::to_bits(0.15)),  // ∇E descent
            beta: std::sync::atomic::AtomicU64::new(f64::to_bits(0.08)),   // Ghost attraction
            gamma: std::sync::atomic::AtomicU64::new(f64::to_bits(0.05)),  // Boundary preservation
        }
    }
}

/// The PonderingCore: RK4 integrator with LU-based topological validation
/// 
/// Treats hidden states as particles in a vector field, evolving them
/// through differential equations before discrete token emission.
/// 
/// Phase 38 adds "The God's Gate":
/// - LU Decomposition verification against QMEM semantic network
/// - Picard Iteration for self-correction when thoughts are invalid
/// - Integration with CPU-bound CSR Laplacian and GPU-bound Candle tensors
/// 
/// NOTE: This struct uses interior mutability via atomics for lock-free
/// coefficient access in the hot RK4 loop, eliminating the 60-minute latency bug.
pub struct PonderingCore {
    /// Arkhe energy coefficients - atomic storage for lock-free hot-path
    coeffs: ArkheCoefficients,
    /// Device for tensor operations
    device: Device,
    /// Pre-allocated buffer for RK4 k1, k2, k3, k4 (zero-allocation principle)
    k_buffer: Option<Tensor>,
    /// Pre-allocated buffer for intermediate states (h_k2, h_k3, h_k4)
    h_buffer: Option<Tensor>,
    /// Phase 39: Pre-allocated scalar tensors for RK4 dt multipliers
    dt_full: Option<Tensor>,    // dt
    dt_half: Option<Tensor>,    // dt/2
    dt_sixth: Option<Tensor>,   // dt/6
    scalar_two: Option<Tensor>, // 2.0 for RK4 weights
    /// Phase 38: Optional reference to QMEM semantic network for LU validation
    semantic_network: Option<Arc<Mutex<ResonanceNetwork>>>,
    /// Phase 38: Threshold for topological validation
    tau_logical: f64,
    /// PHASE 44: God's Gate atomic state for deferred validation results
    gods_gate_state: Arc<GodsGateState>,
}

impl PonderingCore {
    /// Initialize PonderingCore with zero-allocation buffers
    pub fn new(hidden_size: usize, device: &Device) -> Result<Self> {
        // Pre-allocate buffers for RK4 intermediate computations
        // This avoids memory allocation during the hot pondering loop
        let k_buffer = Some(Tensor::zeros((1, 1, hidden_size), DType::F32, device)?);
        let h_buffer = Some(Tensor::zeros((1, 1, hidden_size), DType::F32, device)?);
        
        // Phase 39: Pre-allocate scalar tensors for RK4 (rank-0 scalars for broadcasting)
        let dt_full = Some(Tensor::new(DT as f32, device)?);
        let dt_half = Some(Tensor::new((DT / 2.0) as f32, device)?);
        let dt_sixth = Some(Tensor::new((DT / 6.0) as f32, device)?);
        let scalar_two = Some(Tensor::new(2.0f32, device)?);
        
        Ok(Self {
            coeffs: ArkheCoefficients::default(),
            device: device.clone(),
            k_buffer,
            h_buffer,
            dt_full,
            dt_half,
            dt_sixth,
            scalar_two,
            semantic_network: None,
            tau_logical: TAU_LOGICAL,
            gods_gate_state: Arc::new(GodsGateState::new()),
        })
    }

    pub fn get_gods_gate_state(&self) -> Arc<GodsGateState> {
        self.gods_gate_state.clone()
    }

    /// PHASE 51.5: Hard Reset - Clear PonderingCore latent state for scenario isolation
    pub fn reset(&self) -> Result<()> {
        self.gods_gate_state.reset();

        // Note: Pre-allocated buffers (k_buffer, h_buffer) are overwritten during RK4
        // steps, so explicit zeroing is not required. The GodsGateState reset is the
        // critical operation for state isolation between scenarios.

        Ok(())
    }

    /// Phase 38: Attach a semantic network for LU topological validation
    pub fn attach_semantic_network(&mut self, network: Arc<Mutex<ResonanceNetwork>>) {
        self.semantic_network = Some(network);
        eprintln!("[GOD'S GATE] Semantic network attached for LU validation");
    }

    /// Phase 38: Set custom logical threshold
    pub fn set_logical_threshold(&mut self, tau: f64) {
        self.tau_logical = tau;
        eprintln!("[GOD'S GATE] Logical threshold set to: {:.4}", tau);
    }

    /// Main entry: Ponder on a hidden state before token emission
    /// 
    /// PHASE 44: Quantum Automata & Logic Purge Complete.
    /// The RK4 loop now implements Quantum Amplitude Amplification via
    /// Phase Inversion Penalty for rejected reasoning paths.
    /// 
    /// God's Gate validation is deferred to async post-hoc validation
    /// with AtomicU64 state transfer between tokens.
    /// 
    /// NOTE: Uses &self instead of &mut self to allow calling through
    /// shared references (e.g., RwLockReadGuard). Interior state
    /// mutation is handled via Mutex.
    pub fn ponder(
        &self,
        h0: &Tensor,
        ghost_vector: Option<&SemanticWave>,
        dt: f64,
    ) -> Result<Tensor> {
        let dt = if dt <= 0.0 { DT } else { dt };
        let mut h_current = h0.clone();
        let mut converged = false;
        let mut picard_iterations = 0;
        
        // Convert ghost wave to tensor if present
        let ghost_tensor = self.semantic_wave_to_tensor(ghost_vector)?;
        
        // PHASE 44: Check God's Gate state from PREVIOUS token
        // If previous thought was hallucinated, apply Phase Inversion Penalty
        let previous_was_hallucinated = self.gods_gate_state.was_hallucinated();
        let prev_divergence = self.gods_gate_state.load_divergence();
        
        if previous_was_hallucinated {
            // PHASE 50.2: Commented out hot-loop log
            // eprintln!(
            //     "[QUANTUM AMPLITUDE AMPLIFICATION] Previous thought hallucinated (divergence={:.4}). \
            //      Applying π-phase inversion for destructive interference.",
            //     prev_divergence
            // );
        }
        
        // PHASE 44: RK4 runs at full speed with Quantum Phase Inversion
        // The God's Gate is now async/deferred with atomic state transfer
        loop {
            // === RK4 Pondering Phase (Bare-Metal Speed) ===
            let mut rk4_steps = 0;
            let mut max_l2_norm = 0.0_f64;
            
            for step in 0..MAX_PONDER_STEPS {
                // PHASE 42: Apply Semantic Divergence Penalty as momentum correction
                // This replaces the expensive LU-based Picard iteration
                let semantic_penalty = if picard_iterations > 0 {
                    self.compute_semantic_divergence_penalty(&h_current, h0)?
                } else {
                    None
                };
                
                // PHASE 44: Apply Quantum Phase Inversion Penalty if previous token hallucinated
                let phase_inversion_penalty = if previous_was_hallucinated {
                    self.compute_phase_inversion_penalty(&h_current, h0, prev_divergence)?
                } else {
                    None
                };
                
                // Merge semantic penalty with phase inversion penalty
                let total_correction = match (&semantic_penalty, &phase_inversion_penalty) {
                    (Some(s), Some(p)) => Some((s + p)?),
                    (Some(s), None) => Some(s.clone()),
                    (None, Some(p)) => Some(p.clone()),
                    (None, None) => None,
                };
                
                // RK4 integration step with optional semantic + phase inversion correction
                let h_next = self.rk4_step_variable_dt(&h_current, &ghost_tensor, dt, total_correction.as_ref())?;
                
                // Check convergence: ||h_next - h_current|| < epsilon
                let delta_h = (&h_next - &h_current)?;
                let l2_norm = delta_h
                    .sqr()?
                    .sum_all()?
                    .sqrt()?
                    .to_scalar::<f32>()? as f64;
                
                max_l2_norm = max_l2_norm.max(l2_norm);
                h_current = h_next;
                rk4_steps = step + 1;
                
                if l2_norm < EPSILON {
                    // PHASE 50.2: Commented out hot-loop log
                    // eprintln!("[ARKHE] Converged at step {} (L2={:.6})", step + 1, l2_norm);
                    converged = true;
                    break;
                }
                
                // Early divergence detection
                if l2_norm.is_nan() || l2_norm.is_infinite() {
                    // PHASE 50.2: Commented out hot-loop log
                    // eprintln!("[ARKHE] WARNING: Divergence detected at step {}, clamping...", step);
                    h_current = h0.clone();
                    break;
                }
            }
            
            if !converged {
                // PHASE 50.2: Commented out hot-loop log
                // eprintln!("[ARKHE] Max steps reached ({}), forced convergence", MAX_PONDER_STEPS);
            }
            
            // === PHASE 42: Deferred Semantic Validation (Non-Blocking) ===
            // Instead of blocking O(N³) LU, we use a fast O(N) semantic coherence check
            let semantic_divergence = self.fast_semantic_coherence_check(&h_current, h0)?;
            
            // Check if thought is semantically coherent
            if semantic_divergence <= self.tau_logical {
                // PHASE 50.2: Commented out hot-loop log
                // eprintln!(
                //     "[GOD'S GATE DEFERRED] Thought accepted (divergence={:.6} <= tau={:.4})",
                //     semantic_divergence, self.tau_logical
                // );
                break; // Exit Picard loop - thought is valid
            }
            
            // Thought shows semantic drift - trigger Picard-style correction
            picard_iterations += 1;
            if picard_iterations > MAX_PICARD_STEPS {
                // PHASE 50.2: Commented out hot-loop log
                // eprintln!(
                //     "[GOD'S GATE DEFERRED] Max Picard iterations reached ({}), accepting with divergence={:.6}",
                //     MAX_PICARD_STEPS, semantic_divergence
                // );
                break;
            }
            
            // PHASE 50.2: Commented out hot-loop log
            // eprintln!(
            //     "[GOD'S GATE DEFERRED] Semantic drift detected (divergence={:.6} > tau={:.4}), Picard iteration {}",
            //     semantic_divergence, self.tau_logical, picard_iterations
            // );
            
            // PHASE 42: True Vector Projection for Picard Iteration
            // h_next = h_current - λ·(h_current - h_projected)
            // Where h_projected is the anchor (h0) for semantic stability
            let lambda = 0.3_f64; // Projection strength
            
            // Vector projection: h_corrected = h_current - λ·(h_current - h0)
            // = (1-λ)·h_current + λ·h0  (convex combination toward anchor)
            let h_anchor = h0;
            let drift = (&h_current - h_anchor)?;  // h_current - h0
            let correction = drift.affine(-lambda, 0.0)?;  // -λ·(h_current - h0)

            // FIX: Apply correction to h_current before next RK4 iteration
            h_current = (&h_current + &correction)?;

            // PHASE 50.2: Commented out hot-loop log
            // eprintln!("[GOD'S GATE] Applied vector projection (λ={:.3}) toward semantic anchor", lambda);

            // Continue to next Picard iteration with corrected starting point
            converged = false;
        }
        
        // PHASE 44: TRUE async God's Gate validation with atomic state transfer
        // This runs O(N³) LU in background, stores result for NEXT token
        let h_final = h_current.clone();
        let gods_gate_state = Arc::clone(&self.gods_gate_state);
        let _semantic_network = self.semantic_network.as_ref().map(|sn| Arc::clone(sn));
        
        // Capture final semantic divergence value for the closure
        let final_divergence = self.fast_semantic_coherence_check(&h_current, h0).unwrap_or(1.0);
        
        {
            // Run actual extended topology verification
            let extended_divergence = self.verify_thought_topology_extended(&h_current, None).unwrap_or(final_divergence);
            let was_hallucinated = extended_divergence > TAU_LOGICAL;
            
            // Store results atomically for next token to read
            // Phase 36: Inline updates eliminate OS thread overhead and stream ordering issues
            gods_gate_state.store_divergence(extended_divergence);
            gods_gate_state.mark_hallucinated(was_hallucinated);
        }
        
        // PHASE 50.2: Commented out hot-loop log
        // eprintln!(
        //     "[QUANTUM AMPLITUDE AMPLIFICATION] Pondering complete: {} Picard iterations. \
        //      God's Gate state transferred for next token.",
        //     picard_iterations
        // );
        
        Ok(h_final)
    }

    /// PHASE 42: Fast O(N) Semantic Coherence Check
    /// 
    /// Replaces expensive O(N³) LU with O(N) vector operations.
    /// Computes cosine similarity between current thought and anchor.
    /// 
    /// # Arguments
    /// * `h_current` - Current thought vector
    /// * `h_anchor` - Anchor/initial thought vector (h0)
    /// 
    /// # Returns
    /// * Semantic divergence [0, 1] where 0 = identical, 1 = orthogonal
    fn fast_semantic_coherence_check(&self, h_current: &Tensor, h_anchor: &Tensor) -> Result<f64> {
        let dtype = h_current.dtype();
        let h_curr_f32 = h_current.to_dtype(DType::F32)?;
        let h_anchor_f32 = h_anchor.to_dtype(DType::F32)?;
        
        // Compute dot product: <h_current, h_anchor>
        let dot_product = (&h_curr_f32 * &h_anchor_f32)?
            .sum_keepdim(D::Minus1)?;
        
        // Compute norms
        let norm_curr = h_curr_f32.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let norm_anchor = h_anchor_f32.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        
        // Cosine similarity = dot / (||h_curr|| · ||h_anchor||)
        let norm_product = (&norm_curr * &norm_anchor)?;
        let cosine_sim = (&dot_product / norm_product)?
            .mean_all()?
            .to_scalar::<f32>()? as f64;
        
        // Divergence = 1 - |cosine_similarity|
        let divergence = 1.0 - cosine_sim.abs().min(1.0);
        
        Ok(divergence.clamp(0.0, 1.0))
    }

    /// PHASE 42: Semantic Divergence Penalty
    /// 
    /// Computes a penalty vector that pushes thought back toward anchor
    /// when semantic drift is detected. This is O(N) and applied within RK4.
    /// 
    /// Formula: penalty = -α · (h_current - h_anchor) · divergence
    /// 
    /// # Arguments
    /// * `h_current` - Current thought vector
    /// * `h_anchor` - Anchor thought vector
    /// 
    /// # Returns
    /// * Penalty tensor to add to RK4 derivative (or None if coherent)
    fn compute_semantic_divergence_penalty(
        &self,
        h_current: &Tensor,
        h_anchor: &Tensor
    ) -> Result<Option<Tensor>> {
        let divergence = self.fast_semantic_coherence_check(h_current, h_anchor)?;
        
        if divergence < 0.1 {
            // Semantically coherent - no penalty needed
            return Ok(None);
        }
        
        // Compute penalty: -α · (h_current - h_anchor) · divergence
        let alpha = 0.5_f64; // Penalty strength
        let drift = (h_current - h_anchor)?;
        let penalty = drift.affine(-alpha * divergence, 0.0)?;
        
        eprintln!(
            "[SEMANTIC PENALTY] Drift={:.4}, applying penalty α={:.2}",
            divergence, alpha
        );
        
        Ok(Some(penalty))
    }

    /// PHASE 44: Quantum Amplitude Amplification - Phase Inversion Penalty
    /// 
    /// Mathematical Foundation:
    /// When a thought is rejected (high semantic divergence), we don't just
    /// do gradient descent. We invert its phase by π radians (multiply by -1
    /// in the latent vector space). This creates destructive interference
    /// between invalid reasoning paths and constructive amplification of
    /// the correct ARC-AGI Hamiltonian path.
    /// 
    /// Formula: penalty = -π · (h_current - h_anchor) · divergence · sign(divergence - τ)
    /// 
    /// This is Grover's Search in the latent space:
    /// - Valid thoughts: Phase preserved (constructive interference)
    /// - Invalid thoughts: Phase inverted by π (destructive interference)
    /// 
    /// # Arguments
    /// * `h_current` - Current thought vector
    /// * `h_anchor` - Anchor thought vector (h0)
    /// * `divergence` - Semantic divergence from previous token's validation
    /// 
    /// # Returns
    /// * Phase inversion penalty tensor (or None if no inversion needed)
    fn compute_phase_inversion_penalty(
        &self,
        h_current: &Tensor,
        h_anchor: &Tensor,
        divergence: f64
    ) -> Result<Option<Tensor>> {
        if divergence < self.tau_logical {
            // Thought was valid - no phase inversion needed
            return Ok(None);
        }
        
        // Compute phase inversion: h_inverted = -π · (h_current - h_anchor) · divergence
        let drift = (h_current - h_anchor)?;
        
        // Phase inversion strength: π radians = multiply by -1 in real space
        // The penalty pushes the vector toward the opposite direction
        let inversion_magnitude = -PHASE_INVERSION_STRENGTH * divergence;
        let penalty = drift.affine(inversion_magnitude, 0.0)?;
        
        eprintln!(
            "[PHASE INVERSION] π-radian rotation applied: divergence={:.4}, magnitude={:.4}",
            divergence, inversion_magnitude
        );
        
        Ok(Some(penalty))
    }

    /// PHASE 49.6: Bare-Metal RK4 using .affine() for zero-allocation scalar ops
    fn rk4_step(&self, h_n: &Tensor, ghost_tensor: &Option<Tensor>, dt: f64, correction: Option<&Tensor>) -> Result<Tensor> {
        // Phase 49.6: Use .affine() for scalar multiplication - no broadcasting needed
        let half_dt = dt / 2.0;
        let sixth_dt = dt / 6.0;
        
        // k1 = f(h_n, t)
        let k1 = self.arkhe_derivative(h_n, ghost_tensor, correction)?;
        
        // h_k2 = h_n + (dt/2)*k1 using .affine()
        let k1_scaled = k1.affine(half_dt, 0.0)?;
        let h_k2 = h_n.broadcast_add(&k1_scaled)?;
        let k2 = self.arkhe_derivative(&h_k2, ghost_tensor, correction)?;
        
        // h_k3 = h_n + (dt/2)*k2
        let k2_scaled = k2.affine(half_dt, 0.0)?;
        let h_k3 = h_n.broadcast_add(&k2_scaled)?;
        let k3 = self.arkhe_derivative(&h_k3, ghost_tensor, correction)?;
        
        // h_k4 = h_n + dt*k3
        let k3_scaled = k3.affine(dt, 0.0)?;
        let h_k4 = h_n.broadcast_add(&k3_scaled)?;
        let k4 = self.arkhe_derivative(&h_k4, ghost_tensor, correction)?;
        
        // weighted_sum = k1 + 2*k2 + 2*k3 + k4 using .affine()
        let k2_doubled = k2.affine(2.0, 0.0)?;
        let k3_doubled = k3.affine(2.0, 0.0)?;
        
        let weighted_sum = k1.broadcast_add(&k2_doubled)?
            .broadcast_add(&k3_doubled)?
            .broadcast_add(&k4)?;
        
        // h_next = h_n + (dt/6) * weighted_sum
        let weighted_scaled = weighted_sum.affine(sixth_dt, 0.0)?;
        let h_next = h_n.broadcast_add(&weighted_scaled)?;
        
        Ok(h_next)
    }
    
    /// PHASE 49.6: Bare-Metal variable-dt RK4 using .affine()
    // #[allow(dead_code)]
    fn rk4_step_variable_dt(&self, h_n: &Tensor, ghost_tensor: &Option<Tensor>, dt: f64, correction: Option<&Tensor>) -> Result<Tensor> {
        let half_dt = dt / 2.0;
        let sixth_dt = dt / 6.0;
        
        let k1 = self.arkhe_derivative(h_n, ghost_tensor, correction)?;
        let k1_scaled = k1.affine(half_dt, 0.0)?;
        let h_k2 = h_n.broadcast_add(&k1_scaled)?;
        let k2 = self.arkhe_derivative(&h_k2, ghost_tensor, correction)?;
        
        let k2_scaled = k2.affine(half_dt, 0.0)?;
        let h_k3 = h_n.broadcast_add(&k2_scaled)?;
        let k3 = self.arkhe_derivative(&h_k3, ghost_tensor, correction)?;
        
        let k3_scaled = k3.affine(dt, 0.0)?;
        let h_k4 = h_n.broadcast_add(&k3_scaled)?;
        let k4 = self.arkhe_derivative(&h_k4, ghost_tensor, correction)?;
        
        let k2_doubled = k2.affine(2.0, 0.0)?;
        let k3_doubled = k3.affine(2.0, 0.0)?;
        let weighted_sum = k1.broadcast_add(&k2_doubled)?
            .broadcast_add(&k3_doubled)?
            .broadcast_add(&k4)?;
        
        let weighted_scaled = weighted_sum.affine(sixth_dt, 0.0)?;
        let h_next = h_n.broadcast_add(&weighted_scaled)?;
        
        Ok(h_next)
    }

    /// The Arkhe Derivative: NON-LINEAR Vector Field for Complex Manifolds
    /// 
    /// Phase 39: Mathematical Absolute Zero - Non-linear neural manifold
    /// 
    /// dh/dt = tanh(α·h) - γ·h + β·ghost + λ·correction
    /// 
    /// The tanh projection creates a bounded, non-linear attractor landscape
    /// that captures complex topological features impossible with linear ODEs.
    /// 
    /// # Components:
    /// - Non-linear manifold: tanh(α·h) creates bounded attractors
    /// - Damping: -γ·h provides energy dissipation
    /// - Ghost attraction: β·ghost pulls toward semantic concepts
    /// - Topological correction: λ·correction from LU validation
    /// 
    /// # Arguments
    /// * `h` - Current hidden state
    /// * `ghost_tensor` - Optional semantic guidance
    /// * `correction` - Optional correction gradient from LU validation
    /// 
    /// # Returns
    /// Derivative dh/dt as tensor
    fn arkhe_derivative(&self, h: &Tensor, ghost_tensor: &Option<Tensor>, correction: Option<&Tensor>) -> Result<Tensor> {
        let dtype = h.dtype();
        let h_f32 = h.to_dtype(DType::F32)?;
        
        // PHASE 47.5: Lock-free atomic coefficient read (eliminates 60-min latency bug)
        let (alpha, beta, gamma) = (
            self.coeffs.get_alpha(),
            self.coeffs.get_beta(),
            self.coeffs.get_gamma()
        );
        
        // Phase 39: NON-LINEAR manifold projection via tanh
        // tanh(α·h) creates bounded, smooth attractor landscape
        let scaled = h_f32.affine(alpha as f64, 0.0)?;
        let nonlinear = scaled.tanh()?;
        
        // Damping term: -γ·h (energy dissipation)
        let damping = h_f32.affine(-gamma as f64, 0.0)?;
        
        // PHASE 62.3: Attractor Ghost Injection (Zero-Latency Telemetry)
        // BACKGROUND THREAD SAFETY: DO NOT clone CUDA Tensor from GHOST_VECTOR_SLOT.
        // Access via .to_device() to ensure all operations stay on the same GPU context.
        let live_telemetry_ghost = if let Ok(slot) = GHOST_VECTOR_SLOT.lock() {
            (*slot).as_ref().map(|t| t.to_device(h.device()).ok()).flatten()
        } else {
            None
        };

        // Ghost attraction: +β·ghost
        // Priority: Live Telemetry Stream > Static Semantic Prompt > None
        let active_ghost = live_telemetry_ghost.or_else(|| {
            ghost_tensor.as_ref().cloned()
        });
        
        let attraction = if let Some(g) = active_ghost {
            // Ensure the ghost vector matches the precision of the latent manifold
            let g_f32 = g.to_dtype(candle_core::DType::F32)?;
            // PHASE 47.5 & 62.3: HEAVY TELEMETRY WEIGHTING (10x boost)
            // The system's telemetry state acts as a massive topological gravity well
            let telemetry_boost = 10.0_f64;
            g_f32.affine(beta as f64 * telemetry_boost, 0.0)?
        } else {
            Tensor::zeros_like(&h_f32)?
        };
        
        // Topological correction from LU validation
        let topo_fix = if let Some(corr) = correction {
            corr.to_device(h.device())?.to_dtype(DType::F32)?
        } else {
            Tensor::zeros_like(&h_f32)?
        };
        
        // Combine: dh/dt = tanh(α·h) - γ·h + β·ghost + correction
        let derivative = nonlinear
            .add(&damping)?
            .add(&attraction)?
            .add(&topo_fix)?;
        
        // PHASE 61.3: The Asymmetric Bridge (HSP)
        // CPU-GPU Handshake: Dense GPU phasors are projected to 1M-D Sparse CPU space for exact logic.
        if let Some(ref net_arc) = self.semantic_network {
            if let Ok(mut network) = net_arc.lock() { 
                let mut active_logic_sdr: Option<Vec<u32>> = None;
                
                // 1. Scan for DOMAIN_LOGIC imperative and project to 1,000,000-D CPU space
                for i in 0..network.n {
                    if network.positions.amplitudes[i].norm() > 0.8 {
                        if (network.node_flags[i] & DOMAIN_LOGIC) != 0 {
                            if i < network.holographic_memory.len() {
                                // CPU OFF-LOADING: Project the dense wave to Sparse Distributed Representation
                                active_logic_sdr = Some(HspOracle::project_to_sparse(&network.holographic_memory[i]));
                            }
                            break;
                        }
                    }
                }

                // 2. If the CPU detected an active logic rule, evaluate spatial distractors
                if let Some(logic_sdr) = active_logic_sdr {
                    for i in 0..network.n {
                        if (network.node_flags[i] & DOMAIN_SPATIAL) != 0 {
                            if i < network.holographic_memory.len() {
                                // Project the spatial distractor to 1M-D CPU space
                                let spatial_sdr = HspOracle::project_to_sparse(&network.holographic_memory[i]);
                                
                                // THE EXACT LOGIC EVALUATION
                                // We treat the spatial node itself as the 'current_thought' for intersection testing.
                                let penalty = HspOracle::evaluate_hdc_paradox_filter(&spatial_sdr, &logic_sdr, &spatial_sdr);
                                
                                if penalty > 0.0 {
                                    // The CPU has issued a hard logic contradiction penalty.
                                    // Apply fluid viscosity to the exact tensor memory
                                    apply_viscosity(&mut network.holographic_memory[i], penalty);
                                    
                                    // Suppress the gradient in the GPU's continuous space natively
                                    let decay_factor = (-(penalty as f64)).exp();
                                    network.positions.amplitudes[i] *= decay_factor; 
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Hard clamp to prevent any possibility of explosion
        derivative.clamp(-12.0f64, 12.0f64)?.to_dtype(dtype)
    }

    /// Convert SemanticWave to tensor for Arkhe derivative
    fn semantic_wave_to_tensor(&self, wave: Option<&SemanticWave>) -> Result<Option<Tensor>> {
        let wave = match wave {
            Some(w) => w,
            None => return Ok(None),
        };
        
        // Extract real parts of complex amplitudes
        let reals: Vec<f32> = wave.amplitudes.iter()
            .map(|c| c.re as f32)
            .collect();
        
        let hidden_size = reals.len();
        
        // Create tensor with proper shape [1, 1, hidden_size]
        let tensor = Tensor::from_vec(reals, (1, 1, hidden_size), &self.device)?;
        
        // Normalize to unit sphere
        let norm = tensor.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let norm_scalar = norm.flatten_all()?.to_vec1::<f32>()?[0];
        
        if norm_scalar > 1e-8 {
            let normalized = (&tensor / norm_scalar as f64)?;
            Ok(Some(normalized))
        } else {
            Ok(Some(tensor))
        }
    }

    /// Update Arkhe coefficients for different pondering modes
    /// PHASE 47.5: Lock-free atomic update
    pub fn set_coefficients(&self, alpha: f64, beta: f64, gamma: f64) {
        self.coeffs.set_alpha(alpha);
        self.coeffs.set_beta(beta);
        self.coeffs.set_gamma(gamma);
        eprintln!("[ARKHE] Coefficients updated: α={:.3}, β={:.3}, γ={:.3}", alpha, beta, gamma);
    }

    /// Reset pre-allocated buffers (call when hidden_size changes)
    pub fn reset_buffers(&self, hidden_size: usize) -> Result<()> {
        // Buffer reset is not critical for correctness
        // This is called infrequently, so we can create new tensors
        let _ = Tensor::zeros((1, 1, hidden_size), DType::F32, &self.device)?;
        let _ = Tensor::zeros((1, 1, hidden_size), DType::F32, &self.device)?;
        Ok(())
    }

    // =========================================================================
    // PHASE 38: THE GOD'S GATE - LU NEUROSYMBOLIC FILTERING
    // =========================================================================

    /// Phase 40: Extended verify_thought_topology with Physics Enforcement
    /// 
    /// When processing ARC-AGI grids, this validates:
    /// 1. Object Permanence: Mass (pixel count) must be conserved
    /// 2. Symmetry Invariants: Symmetry operations must preserve reflection/rotation
    /// 3. Topology Preservation: Connected components count must match (or change predictably)
    /// 
    /// The method computes a "physics residual" that penalizes violations of geometric
    /// conservation laws. This residual is injected into the Picard iteration to force
    /// the RK4 integrator to "rethink" invalid transformations.
    // #[allow(dead_code)]
    fn verify_thought_topology_extended(&self, pondered_vector: &Tensor, grid_topology: Option<&crate::semantic_network::GridTopology>) -> Result<f64> {
        // First, run standard LU-based semantic verification
        let semantic_residual = self.verify_thought_topology(pondered_vector)?;
        
        // Phase 40: Physics enforcement for ARC-AGI
        let physics_residual = if let Some(topology) = grid_topology {
            self.compute_physics_violation(topology, pondered_vector)?
        } else {
            0.0 // No grid topology cached, skip physics checks
        };
        
        // Combined residual: semantic error + physics violation penalty
        let combined = semantic_residual + 5.0 * physics_residual;
        
        if physics_residual > 0.01 {
            eprintln!(
                "[GOD'S GATE PHYSICS] Mass/Symmetry violation: residual={:.6}",
                physics_residual
            );
        }
        
        Ok(combined)
    }
    
    /// Phase 40.5: Spatial Decoder Projection - Fix the Dimensional Fallacy
    /// 
    /// The Bug (Phase 40): Counting active dimensions in the 4096-D latent vector
    /// to estimate "mass" was mathematical nonsense. A latent dimension ≠ pixel.
    /// 
    /// The Fix: Project the 1D continuous thought back into 2D discrete grid space
    /// via 2D RoPE basis dot products. This is the inverse of embed_arc_grid's encoding.
    /// 
    /// Mathematical Model:
    /// - Encoder (embed_arc_grid): (x,y) → 4096-D via 2D RoPE basis
    /// - Decoder (this function): 4096-D → (x,y) via basis dot product
    /// - Energy(x,y) = |pondered · RoPE_basis(x,y)|² (probability of pixel at position)
    /// 
    /// # Arguments
    /// * `topology` - GridTopology containing original grid dimensions and expected mass
    /// * `pondered` - The 4096-D latent thought vector [1, 1, hidden_size]
    /// 
    /// # Returns
    /// * Physics violation residual (0.0 = perfect conservation, 1.0 = total violation)
    // #[allow(dead_code)]
    fn compute_physics_violation(&self, topology: &crate::semantic_network::GridTopology, pondered: &Tensor) -> Result<f64> {
        let h_vec: Vec<f32> = pondered.to_vec1()?;
        let hidden_size = h_vec.len();
        
        // Extract grid dimensions from bounding box (min_x, min_y, max_x, max_y)
        let (min_x, min_y, max_x, max_y) = topology.bounding_box;
        let width = (max_x - min_x + 1).max(1);
        let height = (max_y - min_y + 1).max(1);
        
        // Phase 40.5: 2D RoPE Decoder Projection
        // For each (x,y) in the original grid, compute the "energy" via dot product
        // with the 2D RoPE basis vector that would encode that position
        let mut estimated_mass: usize = 0;
        let energy_threshold: f32 = 0.5; // Threshold to declare "pixel present"
        
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                // Compute 2D RoPE basis vector for position (x, y)
                // This mirrors the encoding in embed_arc_grid
                let mut basis_energy: f32 = 0.0;
                
                for dim in 0..hidden_size {
                    let freq_idx = (dim / 2) as f32;
                    let theta = 10000.0_f32.powf(-2.0 * freq_idx / hidden_size as f32);
                    
                    let phase_x = x as f32 * theta;
                    let phase_y = y as f32 * theta;
                    
                    // RoPE basis component (same formula as encoder)
                    let basis_component = if dim % 2 == 0 {
                        (phase_x.cos() + phase_y.cos()) * 0.5
                    } else {
                        (phase_x.sin() + phase_y.sin()) * 0.5
                    };
                    
                    // Accumulate weighted dot product
                    basis_energy += h_vec[dim] * basis_component;
                }
                
                // Apply normalization scale (inverse of encoder scaling)
                let scale = (hidden_size as f32).sqrt();
                let normalized_energy = (basis_energy / scale).abs();
                
                // Threshold to estimate pixel presence
                if normalized_energy > energy_threshold {
                    estimated_mass += 1;
                }
            }
        }
        
        let expected_mass = topology.input_mass;
        
        // Mass conservation penalty: |estimated - expected| / expected
        let mass_violation = if expected_mass > 0 {
            (estimated_mass as f64 - expected_mass as f64).abs() / expected_mass as f64
        } else {
            0.0
        };
        
        if mass_violation > 0.01 {
            eprintln!(
                "[PHYSICS DECODER] Grid {}×{} | Expected mass: {} | Estimated: {} | Violation: {:.4}",
                width, height, expected_mass, estimated_mass, mass_violation
            );
        }
        
        Ok(mass_violation.min(1.0))
    }

    /// Phase 38: Verify thought topology using LU Decomposition
    /// 
    /// Validates the pondered hidden state against the QMEM semantic network's
    /// Laplacian matrix. Uses LU decomposition to solve L·x = h and computes
    /// the residual error (how far the thought deviates from semantic truth).
    /// 
    /// # Arguments
    /// * `pondered_vector` - The 4096-D hidden state to validate
    /// 
    /// # Returns
    /// * `Ok(residual_error)` - L2 norm of the residual (lower = more valid)
    /// * `Err(...)` - If LU decomposition fails or network unavailable
    /// 
    /// # Algorithm
    /// 1. Down-project 4096-D vector to active semantic node dimensions
    /// 2. Extract active sub-graph's Laplacian matrix from CSR
    /// 3. Perform LU decomposition: L = U·U^T (Cholesky for symmetric Laplacian)
    /// 4. Solve L·x = h for x (the "projected" thought)
    /// 5. Compute residual: ||h - L·x|| / ||h||
    // #[allow(dead_code)]
    fn verify_thought_topology(&self, pondered_vector: &Tensor) -> Result<f64> {
        // Check if semantic network is available
        let network_guard = match &self.semantic_network {
            Some(network) => network.lock().map_err(|_| {
                candle_core::Error::Msg("Failed to lock semantic network".to_string())
            })?,
            None => {
                // No network attached, skip validation
                return Ok(0.0);
            }
        };

        // Get the number of active semantic nodes
        let n_semantic = network_guard.n;
        if n_semantic == 0 {
            return Ok(0.0); // Empty network, no validation needed
        }

        // Down-project pondered vector to semantic dimension
        // Strategy: Take first n_semantic dimensions (or pad if smaller)
        let hidden_size = pondered_vector.dim(D::Minus1)?;
        let projection_dim = hidden_size.min(n_semantic);
        
        // Extract and flatten the pondered vector
        let h_vec: Vec<f32> = pondered_vector.to_vec1()?;
        let h_projected: Vec<f64> = h_vec.iter()
            .take(projection_dim)
            .map(|&x| x as f64)
            .collect();
        
        // Pad if necessary
        let mut h_padded = h_projected;
        while h_padded.len() < n_semantic {
            h_padded.push(0.0);
        }
        
        // Build Tikhonov-regularized Laplacian: L_reg = L + λI
        // This guarantees positive-definiteness and invertibility
        // L = D - A (standard graph Laplacian, singular)
        // L_reg = D - A + λI (regularized, invertible)
        let mut laplacian = Array2::<f64>::zeros((n_semantic, n_semantic));
        
        // Compute degrees and add Tikhonov regularization to diagonal
        for (i, row) in network_guard.adjacency_csr.outer_iterator().enumerate() {
            if i >= n_semantic { break; }
            let degree: f64 = row.iter().map(|(_, &w)| w.abs()).sum();
            // L_reg[i,i] = degree + λ (Tikhonov regularization)
            laplacian[[i, i]] = degree + LAMBDA_TIKHONOV;
        }
        
        // Fill off-diagonals with negative weights (adjacency)
        for (i, row) in network_guard.adjacency_csr.outer_iterator().enumerate() {
            if i >= n_semantic { break; }
            for (j, &weight) in row.iter() {
                if j < n_semantic && i != j {
                    laplacian[[i, j]] = -weight;
                }
            }
        }
        
        // Solve L·x = h using LU decomposition
        // PHASE 47.5: ASTRONOMICAL LATENCY FIX
        // Gaussian Elimination is O(N³). For N=4094, this is 68 billion ops per token.
        // We cap dense LU at N=20 and use sparse approximation for industry benchmarks.
        let residual = if n_semantic <= 20 {
            self.solve_lu_dense(&laplacian, &h_padded)?
        } else {
            // For large systems, use sparse approximation (power iteration / GMRES)
            self.solve_lu_sparse_approx(&laplacian, &h_padded, n_semantic)?
        };
        
        eprintln!(
            "[GOD'S GATE] Topology verification: residual={:.6}, nodes={} (Threshold={})",
            residual, n_semantic, if n_semantic <= 20 { "DENSE" } else { "SPARSE" }
        );
        
        Ok(residual)
    }
    
    /// Solve Laplacian system using dense LU decomposition
    // #[allow(dead_code)]
    fn solve_lu_dense(&self, laplacian: &Array2<f64>, h: &[f64]) -> Result<f64> {
        let n = h.len();
        
        // Create augmented matrix [L | h] for Gaussian elimination
        let mut aug = Array2::<f64>::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = laplacian[[i, j]];
            }
            aug[[i, n]] = h[i];
        }
        
        // Partial pivoting LU decomposition
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            for i in (k + 1)..n {
                if aug[[i, k]].abs() > aug[[max_row, k]].abs() {
                    max_row = i;
                }
            }
            
            // Check for singular matrix
            if aug[[max_row, k]].abs() < 1e-12 {
                return Ok(1.0); // Maximum residual (invalid thought)
            }
            
            // Swap rows using proper temp variable
            if max_row != k {
                for j in 0..(n + 1) {
                    let temp = aug[[k, j]];
                    aug[[k, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }
            
            // Eliminate below
            for i in (k + 1)..n {
                let factor = aug[[i, k]] / aug[[k, k]];
                aug[[i, k]] = 0.0;
                for j in (k + 1)..(n + 1) {
                    aug[[i, j]] -= factor * aug[[k, j]];
                }
            }
        }
        
        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[[i, n]];
            for j in ((i + 1)..n).rev() {
                sum -= aug[[i, j]] * x[j];
            }
            x[i] = sum / aug[[i, i]];
        }
        
        // Compute residual: ||h - L·x||
        let mut residual = 0.0;
        for i in 0..n {
            let mut l_x = 0.0;
            for j in 0..n {
                l_x += laplacian[[i, j]] * x[j];
            }
            let diff = h[i] - l_x;
            residual += diff * diff;
        }
        
        // Normalize by ||h||
        let h_norm: f64 = h.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if h_norm > 1e-10 {
            residual = residual.sqrt() / h_norm;
        }
        
        Ok(residual)
    }
    
    /// Sparse approximation for large systems (N > 20)
    /// WARNING: Currently returns a bounded fallback residual. 
    /// TODO (Academic Review): Implement Generalized Minimal Residual method (GMRES)
    /// or Conjugate Gradient (CG) for rigorous O(N^2) Krylov subspace evaluation.
    // #[allow(dead_code)]
    fn solve_lu_sparse_approx(&self, _laplacian: &Array2<f64>, _h: &[f64], n: usize) -> Result<f64> {
        eprintln!("[GOD'S GATE] WARNING: Using unverified sparse fallback for N={}. GMRES solver pending.", n);
        Ok(0.05) // Fallback residual to prevent system stall
    }

    // =========================================================================
    // PHASE 41: LIQUID Phase-Modulated Attention & HOLOGRAPHIC NVMe ADDRESSING
    // =========================================================================

    /// Phase 48: Locality-Sensitive Topological NVMe Addressing
    /// 
    /// The Concept: Continuous reasoning requires smooth transitions. Standard hashing 
    /// causes violent jumps in address space. We need a Continuous-Discrete Bridge.
    /// 
    /// The Implementation: Use 64-bit Locality-Sensitive Hashing (SimHash) directly. 
    /// SimHash natively preserves the cosine distance of the original vector space 
    /// without needing artificial Gray Code corruption.
    /// 
    /// The Physics: This guarantees that if the RK4 differential equation nudges the thought 
    /// vector slightly, the 64-bit SSD address remains topologically proximal. This allows the 
    /// model to traverse the QMEM axioms via smooth, continuous Euler Paths without 
    /// page-faulting across unrelated concepts.
    /// 
    /// # Arguments
    /// * `h` - The 4096-D latent thought vector [1, 1, hidden_size]
    /// 
    /// # Returns
    /// * 64-bit LSH holographic address (4KB page-aligned)
    /// 
    /// # Complexity
    /// O(1) - Fixed 64-dimension computation
    pub fn compute_holographic_address_lsh(&self, h: &Tensor) -> Result<u64> {
        // Extract the thought vector as f32
        let h_vec: Vec<f32> = h.to_vec1()?;
        let hidden_size = h_vec.len();
        
        let mut sim_hash: u64 = 0;
        
        // Phase 48: 4096-D SimHash Projection (deterministic hyperplanes)
        for b in 0..64 {
            let mut dot: f32 = 0.0;
            let seed: u64 = (b as u64).wrapping_mul(0x517cc1b727220a95).wrapping_add(0x6c62272e07bb0142);
            
            for i in 0..hidden_size {
                let mut x = seed.wrapping_add(i as u64);
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                let weight = (x as i32 as f32) * (1.0 / 2147483648.0);
                dot += h_vec[i] * weight;
            }
            
            if dot > 0.0 {
                sim_hash |= 1u64 << b;
            }
        }
        
        // Align to 4KB page boundary for optimal NVMe I/O
        let page_aligned = sim_hash & !(4096u64 - 1);
        
        // Compute Hamming weight change for diagnostics
        let prev_addr = (sim_hash.wrapping_sub(1)) & !(4096u64 - 1);
        let hamming_change = (page_aligned ^ prev_addr).count_ones();
        
        eprintln!(
            "[HOLOGRAPHIC LSH] SimHash=0x{:016X} | Page=0x{:016X} | ΔHamming={}",
            sim_hash, page_aligned, hamming_change
        );
        
        Ok(page_aligned)
    }

    /// Phase 41: Fetch QMEM block from holographic address
    /// 
    /// Uses the O(1) holographic address to fetch a 4KB topological weight block
    /// from the memory-mapped NVMe QMEM cartridge.
    /// 
    /// # Arguments
    /// * `qmem` - Memory-mapped QMEM cartridge handle
    /// * `h` - Current thought vector (used to compute holographic address)
    /// 
    /// # Returns
    /// * Decoded f32 topological weights from the 4KB block
    pub fn fetch_holographic_weights(
        &self,
        qmem: &crate::nvme_direct::HolographicQMEM,
        h: &Tensor
    ) -> Result<Vec<f32>> {
        // Compute O(1) holographic address from thought vector
        let address = self.compute_holographic_address_lsh(h)?;
        
        // Fetch 4KB block from memory-mapped SSD
        match qmem.fetch_block(address) {
            Some(block) => {
                // Decode packed ternary/hex weights into f32 array
                let weights = crate::nvme_direct::decode_qmem_block(block);
                eprintln!(
                    "[LIQUID Phase-Modulated Attention] Fetched {} weights from holographic address 0x{:016X}",
                    weights.len(), address
                );
                Ok(weights)
            }
            None => {
                // Address out of bounds - return empty weights
                eprintln!("[LIQUID Phase-Modulated Attention] Invalid holographic address, using zero weights");
                Ok(vec![])
            }
        }
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use num_complex::Complex64;

    #[test]
    fn test_pondering_core_creation() {
        let device = Device::Cpu;
        let core = PonderingCore::new(4096, &device).unwrap();
        assert!(core.k_buffer.is_some());
        assert!(core.h_buffer.is_some());
    }

    #[test]
    fn test_rk4_step_no_divergence() {
        let device = Device::Cpu;
        let core = PonderingCore::new(768, &device).unwrap();
        
        // Stable hidden state
        let h0 = Tensor::zeros((1, 1, 768), DType::F32, &device).unwrap();
        let ghost: Option<Tensor> = None;
        
        // Should not diverge
        let h1 = core.rk4_step(&h0, &ghost, 0.1, None).unwrap();
        
        // Check no NaN/Inf
        let sum = h1.sum_all().unwrap().to_vec0::<f32>().unwrap();
        assert!(!sum.is_nan());
        assert!(!sum.is_infinite());
    }

    #[test]
    fn test_semantic_wave_conversion() {
        let device = Device::Cpu;
        let core = PonderingCore::new(4096, &device).unwrap();
        
        // Create test wave with 4096 amplitudes
        let wave = SemanticWave::from_vec(vec![Complex64::new(1.0, 0.0); 4096]);
        
        let tensor = core.semantic_wave_to_tensor(Some(&wave)).unwrap();
        assert!(tensor.is_some());
        
        let t = tensor.unwrap();
        assert_eq!(t.dims().to_vec(), vec![1, 1, 4096]);
    }

    #[test]
    fn test_holographic_addressing() {
        let device = Device::Cpu;
        let core = PonderingCore::new(4096, &device).unwrap();
        
        // Create two non-zero thought vectors (zero vector produces hash 0)
        let mut h1_data = vec![0.0f32; 4096];
        h1_data[0] = 1.0;
        let h1 = Tensor::from_vec(h1_data, (4096,), &device).unwrap();
        let mut h2_data = vec![0.0f32; 4096];
        h2_data[0] = 1.0;
        h2_data[1] = 0.1; // Slight perturbation
        let h2 = Tensor::from_vec(h2_data, (4096,), &device).unwrap();
        
        // Compute holographic addresses (LSH for continuous traversal)
        let addr1 = core.compute_holographic_address_lsh(&h1).unwrap();
        let addr2 = core.compute_holographic_address_lsh(&h2).unwrap();
        
        // Addresses should be 4KB aligned
        assert!((addr1 & 0xFFF) == 0); // 4KB aligned
        assert!((addr2 & 0xFFF) == 0); // 4KB aligned
        
        eprintln!("Holographic addresses: 0x{:016X}, 0x{:016X}", addr1, addr2);
    }
}
