// src/physics_neuron.rs - SYMPLECTIC EULER INTEGRATOR with Soft Injection

use num_complex::Complex64 as FastComplex;
use rustfft::num_traits::Zero;

/// Physics-based neuron with dynamic damping for soft injection
#[derive(Debug, Clone)]
pub struct PhysicsNeuron {
    pub state_y: FastComplex,    // Position (Amplitude + Phase)
    pub velocity_v: FastComplex, // Velocity
    pub param_a: f64,            // Spring constant (k) - determines resonance frequency
    pub param_b: f64,            // Damping coefficient (b) - determines memory decay
    pub dt: f64,                 // Time step
    
    // Soft injection state
    soft_injection_mode: bool,
    original_damping: f64,       // Store original damping for restoration
}

impl PhysicsNeuron {
    pub fn new(dt: f64) -> Self {
        Self {
            state_y: FastComplex::zero(),
            velocity_v: FastComplex::zero(),
            param_a: 1.0,
            param_b: 0.1,
            dt,
            soft_injection_mode: false,
            original_damping: 0.1,
        }
    }

    /// Enable soft injection mode
    /// 
    /// Transitions damping from current value to critically damped (II = 1.0)
    /// This minimizes quantization error when injecting repaired data
    pub fn enable_soft_injection(&mut self) {
        self.original_damping = self.param_b;
        
        // Calculate critical damping: b_crit = 2IIIk
        let critical_damping = 2.0 * self.param_a.sqrt();
        
        // Apply softplus to ensure stability: II > 0
        self.param_b = softplus(critical_damping, 1.0);
        
        self.soft_injection_mode = true;
    }

    /// Disable soft injection mode and restore original damping
    pub fn disable_soft_injection(&mut self) {
        if self.soft_injection_mode {
            self.param_b = self.original_damping;
            self.soft_injection_mode = false;
        }
    }

    /// Get current damping ratio II
    /// 
    /// II = b / (2IIIk)
    /// - II < 1: Under-damped (oscillatory)
    /// - II = 1: Critically damped (optimal response)
    /// - II > 1: Over-damped (slow)
    pub fn damping_ratio(&self) -> f64 {
        if self.param_a > 0.0 {
            self.param_b / (2.0 * self.param_a.sqrt())
        } else {
            0.0
        }
    }

    /// Calculate acceleration: a = F_net / m
    /// 
    /// F_net = F_input - k*y - b*v
    /// Assumes m = 1
    #[inline(always)]
    fn acceleration(&self, y: FastComplex, v: FastComplex, input_force: FastComplex) -> FastComplex {
        let spring = y * (-self.param_a);  // -k * y (restoring force)
        let damping = v * (-self.param_b); // -b * v (friction)
        
        input_force + spring + damping
    }

    /// Semi-Implicit Euler (Symplectic Euler) Integration
    /// 
    /// Energy-conserving (symplectic) and computationally efficient
    /// 
    /// Update scheme:
    /// 1. v_{t+1} = v_t + a(x_t) * dt
    /// 2. x_{t+1} = x_t + v_{t+1} * dt  (uses new velocity)
    pub fn forward(&mut self, input_force: FastComplex) -> FastComplex {
        let dt = self.dt;

        // Update velocity using current position
        let accel = self.acceleration(self.state_y, self.velocity_v, input_force);
        self.velocity_v += accel * dt;

        // Update position using NEW velocity (symplectic trick)
        self.state_y += self.velocity_v * dt;
        
        self.state_y
    }

    /// Inject repaired state with soft transition
    /// 
    /// This method should be called when repaired data needs to be integrated
    /// It temporarily enables soft injection mode, applies the input, then restores
    pub fn inject_repaired(&mut self, repaired_state: FastComplex) {
        self.enable_soft_injection();
        
        // Calculate force needed to reach repaired state, accounting for restoring force
        let delta = repaired_state - self.state_y;
        let correction_force = delta * self.param_a + self.state_y * self.param_a;
        
        self.forward(correction_force);
        
        self.disable_soft_injection();
    }

    /// Get current state
    pub fn get_state(&self) -> FastComplex {
        self.state_y
    }

    /// Reset neuron to initial state
    pub fn reset(&mut self) {
        self.state_y = FastComplex::zero();
        self.velocity_v = FastComplex::zero();
        self.soft_injection_mode = false;
    }

    /// Check stability (Laplace criterion)
    /// 
    /// For stability, all poles must have negative real parts
    /// Pole locations: s = -IIII II IIIII(IIII - 1)
    /// 
    /// Returns true if system is stable
    pub fn is_stable(&self) -> bool {
        let zeta = self.damping_ratio();
        let omega = self.param_a.sqrt();
        
        // For stability: II > 0 and II > 0
        zeta > 0.0 && omega > 0.0
    }

    /// Get system energy (Hamiltonian)
    /// 
    /// H = (1/2)k|y|II + (1/2)|v|II
    pub fn energy(&self) -> f64 {
        let kinetic = 0.5 * self.velocity_v.norm_sqr();
        let potential = 0.5 * self.param_a * self.state_y.norm_sqr();
        kinetic + potential
    }
}

/// Softplus activation: ensures positive output
/// 
/// softplus(x) = ln(1 + e^x)
/// 
/// Properties:
/// - Always positive
/// - Smooth (differentiable)
/// - Approximates ReLU for large |x|
fn softplus(x: f64, beta: f64) -> f64 {
    let scaled = beta * x;
    if scaled > 20.0 {
        // Avoid overflow for large x
        scaled / beta
    } else {
        (1.0 + scaled.exp()).ln() / beta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_movement() {
        let mut neuron = PhysicsNeuron::new(0.1);
        
        // Should start at zero
        assert_eq!(neuron.state_y.re, 0.0);
        
        // Apply impulse
        let input = FastComplex::new(1.0, 0.0);
        neuron.forward(input);
        
        // Should move
        assert!(neuron.state_y.re > 0.0, "Neuron didn't respond to force!");
        assert!(neuron.velocity_v.re > 0.0, "Neuron didn't accelerate!");
    }

    #[test]
    fn test_stability() {
        let mut neuron = PhysicsNeuron::new(0.1);
        neuron.param_a = 4.0;
        neuron.param_b = 1.2;
        
        assert!(neuron.is_stable(), "System should be stable");
        
        // Apply large impulse
        neuron.forward(FastComplex::new(10.0, 0.0));
        
        // Run for 100 steps without input
        for _ in 0..100 {
            neuron.forward(FastComplex::zero());
        }
        
        // Should decay (damping)
        assert!(neuron.velocity_v.re.abs() < 1.0, "System unstable - velocity not decaying");
        assert!(!neuron.state_y.re.is_nan(), "System diverged - NaN detected");
    }

    #[test]
    fn test_soft_injection() {
        let mut neuron = PhysicsNeuron::new(0.1);
        neuron.param_a = 4.0;
        neuron.param_b = 1.2;
        
        let original_damping = neuron.param_b;
        
        // Enable soft injection
        neuron.enable_soft_injection();
        
        // Damping should be critical
        let zeta = neuron.damping_ratio();
        assert!((zeta - 1.0).abs() < 0.1, "Should be critically damped, got II={}", zeta);
        
        // Disable
        neuron.disable_soft_injection();
        
        // Should restore original
        assert_eq!(neuron.param_b, original_damping, "Damping not restored");
    }

    #[test]
    fn test_inject_repaired() {
        let mut neuron = PhysicsNeuron::new(0.1);
        neuron.param_a = 4.0;
        neuron.param_b = 1.2;
        
        // Set initial state
        neuron.state_y = FastComplex::new(1.0, 0.0);
        
        // Inject repaired state
        let repaired = FastComplex::new(1.5, 0.2);
        neuron.inject_repaired(repaired);
        
        // Should move toward repaired state
        assert!(neuron.state_y.re > 1.0, "Didn't inject repaired state");
        assert!(neuron.state_y.im > 0.0, "Phase not updated");
    }

    #[test]
    fn test_energy_conservation() {
        let mut neuron = PhysicsNeuron::new(0.01);
        neuron.param_a = 4.0;
        neuron.param_b = 0.0; // No damping for energy test
        
        // Initial impulse
        neuron.forward(FastComplex::new(1.0, 0.0));
        
        let initial_energy = neuron.energy();
        
        // Run for several steps
        for _ in 0..100 {
            neuron.forward(FastComplex::zero());
        }
        
        let final_energy = neuron.energy();
        
        // Energy should be conserved (symplectic integrator)
        let energy_change = (final_energy - initial_energy).abs() / initial_energy;
        assert!(energy_change < 0.1, "Energy not conserved: II = {:.2}%", energy_change * 100.0);
    }

    #[test]
    fn test_damping_ratio() {
        let mut neuron = PhysicsNeuron::new(0.1);
        
        // Under-damped
        neuron.param_a = 4.0;
        neuron.param_b = 1.0;
        assert!(neuron.damping_ratio() < 1.0, "Should be under-damped");
        
        // Critically damped
        neuron.param_b = 2.0 * neuron.param_a.sqrt();
        assert!((neuron.damping_ratio() - 1.0).abs() < 1e-6, "Should be critically damped");
        
        // Over-damped
        neuron.param_b = 5.0;
        assert!(neuron.damping_ratio() > 1.0, "Should be over-damped");
    }

    #[test]
    fn test_softplus() {
        // Should always be positive
        assert!(softplus(-10.0, 1.0) > 0.0);
        assert!(softplus(0.0, 1.0) > 0.0);
        assert!(softplus(10.0, 1.0) > 0.0);
        
        // Should approximate ReLU for large x
        assert!((softplus(10.0, 1.0) - 10.0).abs() < 0.1);
    }
}













