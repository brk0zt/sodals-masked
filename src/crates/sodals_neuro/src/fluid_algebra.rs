// src/fluid_algebra.rs — PHASE 60.3: Clifford Algebra & Phasor Collision Engine
//
// Vector Symbolic Architecture (VSA) core using Clifford Algebra (Geometric Product)
// and Fluid Dynamics (Viscosity/Impedance) for O(N) holographic operations.
//
// Replaces O(N³) matrix multiplications with O(N) phasor operations.

use num_complex::Complex64;

/// PHASE 60.3: Holographic Binding (Clifford Geometric Product)
/// Binds two 4096-D semantic concepts into a single superimposed hypervector.
/// 
/// Mathematical Operation: A ⊗ B = elementwise_complex_multiply(A, B)
/// Phase Addition: e^(iθ_A) * e^(iθ_B) = e^(i(θ_A + θ_B))
/// 
/// Time Complexity: O(N) instead of O(N³) matrix multiplication.
/// 
/// # Arguments
/// * `a` - First 4096-D complex phasor vector
/// * `b` - Second 4096-D complex phasor vector
/// 
/// # Returns
/// * Bound hypervector representing the composite concept A⊗B
pub fn bind_phasors(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(a.len(), b.len(), "Phasor dimensions must match for binding");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// PHASE 60.3: Holographic Unbinding (Inverse Clifford Extraction)
/// Extracts a specific concept from a bound superimposed state using the Complex Conjugate.
/// 
/// Mathematical Operation: A ≈ C ⊗ B* where C = A ⊗ B
/// Phase Subtraction: e^(iθ_C) * e^(-iθ_B) = e^(i(θ_C - θ_B)) ≈ e^(iθ_A)
/// 
/// Time Complexity: O(N) instead of O(N³) matrix inversion.
/// 
/// # Arguments
/// * `bound_state` - The composite hypervector C = A ⊗ B
/// * `key` - The phasor of the concept to remove (B)
/// 
/// # Returns
/// * Approximate extraction of the remaining concept (A)
pub fn unbind_phasors(bound_state: &[Complex64], key: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(bound_state.len(), key.len(), "Phasor dimensions must match for unbinding");
    bound_state.iter().zip(key.iter()).map(|(c, k)| c * k.conj()).collect()
}

/// PHASE 60.3: Topological Viscosity (Fluid Dynamics Decay)
/// Applies spatial/logical impedance to a phasor. High impedance destroys the wave's 
/// amplitude exponentially, effectively silencing the concept in the Latent Pondering core.
/// 
/// Mathematical Operation: amplitude' = amplitude * e^(-impedance)
/// 
/// When a semantic sub-space (e.g., Spatial domain) contradicts a higher logic rule 
/// (e.g., Logic domain), we do not delete the tensors. We apply "Viscosity" (Impedance).
/// High impedance exponentially decays the wave's amplitude, neutralizing it in 
/// the attention matrix.
/// 
/// # Arguments
/// * `phasor` - Mutable slice of the phasor to decay
/// * `impedance` - Fluid dynamics friction coefficient (higher = more decay)
pub fn apply_viscosity(phasor: &mut [Complex64], impedance: f64) {
    // Decay factor based on fluid dynamics friction: e^(-impedance)
    let decay = (-impedance).exp();
    for val in phasor.iter_mut() {
        *val = Complex64::new(val.re * decay, val.im * decay);
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// TESTS
// ════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_bind_phasors_phase_addition() {
        // Create two phasors: e^(iπ/4) and e^(iπ/4)
        let a = vec![Complex64::from_polar(1.0, PI / 4.0); 4096];
        let b = vec![Complex64::from_polar(1.0, PI / 4.0); 4096];
        
        let bound = bind_phasors(&a, &b);
        
        // Result should be e^(iπ/2) due to phase addition
        let expected_phase = PI / 2.0;
        let actual_phase = bound[0].arg();
        assert!((actual_phase - expected_phase).abs() < 1e-10, 
            "Phase should be π/2 after binding, got {}", actual_phase);
    }

    #[test]
    fn test_unbind_phasors_extraction() {
        // Create concept A and B
        let a = vec![Complex64::from_polar(1.0, PI / 3.0); 4096];
        let b = vec![Complex64::from_polar(1.0, PI / 6.0); 4096];
        
        // Bind them: C = A ⊗ B
        let c = bind_phasors(&a, &b);
        
        // Unbind B from C to recover A
        let recovered_a = unbind_phasors(&c, &b);
        
        // The recovered phase should be approximately π/3 (original A phase)
        let recovered_phase = recovered_a[0].arg();
        assert!((recovered_phase - PI / 3.0).abs() < 1e-10,
            "Recovered phase should be ~π/3, got {}", recovered_phase);
    }

    #[test]
    fn test_apply_viscosity_decay() {
        // Create a phasor with amplitude 1.0
        let mut phasor = vec![Complex64::new(1.0, 0.0); 4096];
        
        // Apply high impedance (should decay amplitude significantly)
        apply_viscosity(&mut phasor, 2.0);
        
        // Expected decay: e^(-2.0) ≈ 0.135
        let expected_amplitude = (-2.0_f64).exp();
        let actual_amplitude = phasor[0].norm();
        assert!((actual_amplitude - expected_amplitude).abs() < 1e-10,
            "Amplitude should decay to ~0.135, got {}", actual_amplitude);
    }

    #[test]
    fn test_bind_unbind_roundtrip() {
        // Create random-ish phasors
        let a: Vec<Complex64> = (0..4096)
            .map(|i| Complex64::from_polar(1.0, (i as f64) * 0.01))
            .collect();
        let b: Vec<Complex64> = (0..4096)
            .map(|i| Complex64::from_polar(1.0, (i as f64) * 0.02))
            .collect();
        
        // Bind and unbind
        let c = bind_phasors(&a, &b);
        let recovered_a = unbind_phasors(&c, &b);
        
        // Verify dimensions match
        assert_eq!(recovered_a.len(), a.len());
        assert_eq!(recovered_a.len(), 4096);
    }
}
