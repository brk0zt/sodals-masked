// src/engine/autonomy_gate.rs
// DIVERGENCE-GATED AUTONOMY BRIDGE
//
// Replaces human-in-the-loop "Press Y/N" approval loops with physics.
// The Core's Semantic Divergence Z-Score mathematically gates whether an
// Actuator (Skill) may execute immediately, must be deferred back to the
// Ouroboros PonderingCore for Picard Iteration, or must HALT and throw
// an interrupt to the TUI for human intervention.
//
// This is NOT a static approval system. It is a continuous physics gate:
//   - Z < 0.1  → The Core's ODE flow is coherent. Execute immediately.
//   - Z < 0.5  → Partial coherence. Defer to Picard self-correction.
//   - Z ≥ 0.5  → Constructive Interference Failure. Halt and prompt.

use std::fmt;

/// The three autonomy levels determined by the physics gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AutonomyLevel {
    /// Z-Score < 0.1: Full coherence. Execute immediately without UI blocking.
    Full,
    /// 0.1 ≤ Z-Score < 0.5: Partial coherence. Send back to Ouroboros
    /// PonderingCore for Picard Iteration (self-correction loop).
    Deferred,
    /// Z-Score ≥ 0.5: Constructive Interference Failure. Throw an interrupt
    /// to the TUI, requesting human intervention.
    HaltAndPrompt,
}

impl fmt::Display for AutonomyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutonomyLevel::Full => write!(f, "FULL"),
            AutonomyLevel::Deferred => write!(f, "DEFERRED"),
            AutonomyLevel::HaltAndPrompt => write!(f, "HALT_AND_PROMPT"),
        }
    }
}

/// The decision record returned by the gate for a single actuator evaluation.
#[derive(Debug, Clone)]
pub struct AutonomyVerdict {
    /// The determined autonomy level
    pub level: AutonomyLevel,
    /// The Z-Score that was evaluated
    pub z_score: f32,
    /// Human-readable reason string
    pub reason: String,
}

// ============================================================================
// AUTONOMY GATE — The physics-based decision engine
// ============================================================================

pub struct AutonomyGate {
    /// Threshold below which Full autonomy is granted
    pub full_threshold: f32,
    /// Threshold at which HaltAndPrompt is triggered
    pub halt_threshold: f32,
    /// Number of Picard iterations allowed for Deferred verdicts
    /// before escalating to HaltAndPrompt
    pub max_picard_retries: u32,
    /// Current retry counter for deferred executions
    pub deferred_retry_count: u32,
}

impl AutonomyGate {
    /// Create a gate with the canonical physics thresholds:
    ///   Full: Z < 0.1, Halt: Z ≥ 0.5
    pub fn new() -> Self {
        Self {
            full_threshold: 0.1,
            halt_threshold: 0.5,
            max_picard_retries: 3,
            deferred_retry_count: 0,
        }
    }

    /// Create a gate with custom thresholds (for tuning the physics envelope).
    pub fn with_thresholds(full: f32, halt: f32, max_retries: u32) -> Self {
        Self {
            full_threshold: full,
            halt_threshold: halt,
            max_picard_retries: max_retries,
            deferred_retry_count: 0,
        }
    }

    /// Evaluate whether an Actuator may execute based on the Core's
    /// current Semantic Divergence Z-Score.
    ///
    /// This is the primary decision point that replaces human-in-the-loop.
    /// The Z-Score is computed by the Oracle from the RK4 ODE flow state.
    pub fn can_execute_actuator(&self, divergence_z_score: f32) -> AutonomyVerdict {
        if divergence_z_score < self.full_threshold {
            AutonomyVerdict {
                level: AutonomyLevel::Full,
                z_score: divergence_z_score,
                reason: format!(
                    "Z={:.4} < {:.4}: ODE flow coherent. Immediate execution.",
                    divergence_z_score, self.full_threshold
                ),
            }
        } else if divergence_z_score < self.halt_threshold {
            AutonomyVerdict {
                level: AutonomyLevel::Deferred,
                z_score: divergence_z_score,
                reason: format!(
                    "Z={:.4} ∈ [{:.4}, {:.4}): Partial coherence. Deferring to Picard iteration.",
                    divergence_z_score, self.full_threshold, self.halt_threshold
                ),
            }
        } else {
            AutonomyVerdict {
                level: AutonomyLevel::HaltAndPrompt,
                z_score: divergence_z_score,
                reason: format!(
                    "Z={:.4} ≥ {:.4}: CONSTRUCTIVE INTERFERENCE FAILURE. Human intervention required.",
                    divergence_z_score, self.halt_threshold
                ),
            }
        }
    }

    /// Evaluate with escalation: if the actuator has been deferred too many
    /// times, escalate to HaltAndPrompt regardless of Z-Score.
    pub fn can_execute_with_escalation(&mut self, divergence_z_score: f32) -> AutonomyVerdict {
        let mut verdict = self.can_execute_actuator(divergence_z_score);

        if verdict.level == AutonomyLevel::Deferred {
            self.deferred_retry_count += 1;
            if self.deferred_retry_count >= self.max_picard_retries {
                verdict.level = AutonomyLevel::HaltAndPrompt;
                verdict.reason = format!(
                    "Z={:.4}: Picard retries exhausted ({}/{}). Escalating to HALT.",
                    divergence_z_score,
                    self.deferred_retry_count,
                    self.max_picard_retries
                );
            }
        } else if verdict.level == AutonomyLevel::Full {
            // Successful execution resets the retry counter
            self.deferred_retry_count = 0;
        }

        verdict
    }

    /// Reset the deferred retry counter (e.g., after a successful Full execution
    /// or after human intervention resolves a HaltAndPrompt).
    pub fn reset_retries(&mut self) {
        self.deferred_retry_count = 0;
    }
}

impl Default for AutonomyGate {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TUI INTERRUPT — Signal from the gate to the NeuroTerminal
// ============================================================================

/// Interrupt signal sent from the AutonomyGate to the TUI when
/// HaltAndPrompt is triggered. This is the ONLY time the TUI
/// requests human input — it is NOT a per-command approval loop.
#[derive(Debug, Clone)]
pub struct HaltInterrupt {
    /// The Z-Score that triggered the halt
    pub z_score: f32,
    /// The actuator that was attempting execution
    pub actuator_name: String,
    /// The action that was blocked
    pub blocked_action: String,
    /// The reason string from the AutonomyVerdict
    pub reason: String,
}

impl HaltInterrupt {
    pub fn new(
        z_score: f32,
        actuator_name: impl Into<String>,
        blocked_action: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            z_score,
            actuator_name: actuator_name.into(),
            blocked_action: blocked_action.into(),
            reason: reason.into(),
        }
    }
}
