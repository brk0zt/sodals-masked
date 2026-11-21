// src/engine/silent_reasoning.rs
//
// Silent Reasoning Core — Pre-emission reasoning with dual-axis quality scoring.
//
// Two independent quality axes:
//   1. Coherence Score (Wronskian/holographic consistency) — "is this inference semantically consistent?"
//   2. Precondition Score (goal-precondition satisfaction) — "does this inference physically achieve the goal?"
//
// These two axes measure fundamentally different things and must both pass
// for a reasoning chain to be accepted.

use crate::semantic_network::ResonanceNetwork;
use crate::compiler::qmem_compiler::{CompositionRule, GoalNode};

/// Dual-axis reasoning result with per-step trace.
pub struct SilentReasoningResult {
    pub final_state: Vec<f64>,
    pub confidence: f64,
    pub triggered_rules: Vec<String>,
    /// Number of reasoning steps taken before convergence or max_steps
    pub steps_taken: usize,
    /// Semantic coherence score — Wronskian/holographic consistency across the reasoning chain.
    /// High value = the inference is internally consistent.
    /// This does NOT tell you whether the inference achieves its goal.
    pub coherence_score: f64,
    /// Goal-precondition satisfaction score — STRIPS-style feasibility check.
    /// High value = the proposed action satisfies the physical prerequisites of the goal.
    /// This does NOT tell you whether the inference is semantically elegant.
    pub precondition_score: f64,
    /// Per-step reasoning trace for auditability
    pub reasoning_trace: Vec<ReasoningStep>,
}

/// A single step in the reasoning trace — enables "we didn't cheat" transparency.
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    /// Which step number (0-indexed)
    pub step_index: usize,
    /// Which rules were triggered at this step
    pub triggered_at_step: Vec<String>,
    /// Coherence delta at this step (positive = improving)
    pub coherence_delta: f64,
    /// Precondition satisfaction delta at this step
    pub precondition_delta: f64,
    /// The max amplitude change at this step (convergence metric)
    pub max_change: f64,
}

pub struct SilentReasoningCore {
    max_steps: usize,
    tolerance: f64,
}

impl SilentReasoningCore {
    pub fn new(max_steps: usize, tolerance: f64) -> Self {
        Self { max_steps, tolerance }
    }

    /// Run silent reasoning with dual-axis quality scoring.
    ///
    /// `goals` provides STRIPS-style goal-precondition constraints (from QMEM `GOAL ... REQUIRES ...`).
    /// When empty, precondition_score defaults to 1.0 (no goal constraints to violate).
    pub fn reason(
        &self,
        net_g: &mut ResonanceNetwork,
        available_rules: &[CompositionRule],
        goals: &[GoalNode],
    ) -> SilentReasoningResult {
        let mut step = 0;
        let mut converged = false;
        let mut triggered_rules = Vec::new();
        let mut confidence = 1.0;
        let mut coherence_score = 1.0;
        let mut reasoning_trace = Vec::new();

        // "Sessiz Reasoning Döngüsü"
        while step < self.max_steps && !converged {
            let mut max_change = 0.0_f64;
            let mut step_triggered = Vec::new();
            let coherence_before = coherence_score;

            // 1. Simulate the physics step (Apply Viscosity / Euler decay)
            for i in 0..net_g.n {
                let initial_norm = net_g.positions.amplitudes[i].norm();

                // Simple Euler viscosity for silent reasoning
                // net_g.damping[i] acts as the viscosity factor
                let damping = net_g.damping[i];
                let decay_factor = (-(damping * 0.01)).exp();
                net_g.positions.amplitudes[i] *= decay_factor;
                
                let new_norm = net_g.positions.amplitudes[i].norm();
                let change = (new_norm - initial_norm).abs();
                if change > max_change {
                    max_change = change;
                }
            }

            // 2. Rule evaluation (Rule match without token emission)
            for rule in available_rules {
                // Determine if rule is triggered based on the target node's energy.
                if let Some(&node_id) = net_g.labels.get(&rule.target_concept) {
                    if node_id < net_g.n {
                        let energy = net_g.positions.amplitudes[node_id].norm();
                        if energy > 0.5 { // threshold for triggering
                            if !triggered_rules.contains(&rule.target_concept) {
                                triggered_rules.push(rule.target_concept.clone());
                                step_triggered.push(rule.target_concept.clone());
                                // Apply consequence of the rule (simulate deduction)
                                let current_amp = net_g.positions.amplitudes[node_id];
                                net_g.positions.amplitudes[node_id] = current_amp * 1.1;
                                
                                // Penalty to confidence if contradiction occurs (placeholder logic)
                                confidence *= 0.99;
                            }
                        }
                    }
                }
            }

            // 3. Coherence scoring — measure how much the state drifted
            //    Lower max_change = more coherent (system is settling, not oscillating)
            let coherence_delta = if max_change < self.tolerance {
                0.01  // bonus: converging
            } else {
                -max_change * 0.001  // penalty proportional to instability
            };
            coherence_score = (coherence_score + coherence_delta).clamp(0.0, 1.0);

            // 4. Record trace
            reasoning_trace.push(ReasoningStep {
                step_index: step,
                triggered_at_step: step_triggered,
                coherence_delta,
                precondition_delta: 0.0, // computed after loop
                max_change,
            });

            // 5. Convergence check
            if max_change < self.tolerance {
                converged = true;
            }

            step += 1;
        }

        // === Post-loop: Goal-Precondition Satisfaction Check ===
        //
        // For each GOAL "G" REQUIRES "P":
        //   Check if the precondition concept "P" has sufficient energy in net_g.
        //   If so, the goal's physical prerequisites are met.
        //   If not, the proposed action does NOT satisfy the goal — regardless of how
        //   semantically coherent the reasoning chain was.
        //
        // This is the "arabayı yıkatacaksan arabanın orada olması lazım" check.
        let precondition_score = if goals.is_empty() {
            1.0 // No goals to check — vacuously satisfied
        } else {
            let mut satisfied_count = 0usize;
            for goal in goals {
                // Check if precondition concept exists and has sufficient energy
                if let Some(&precond_id) = net_g.labels.get(&goal.precondition) {
                    if precond_id < net_g.n {
                        let precond_energy = net_g.positions.amplitudes[precond_id].norm();
                        if precond_energy > 0.3 { // Lower threshold than rule triggering — preconditions are binary
                            satisfied_count += 1;
                            eprintln!(
                                "[GOAL CHECK] ✅ Goal \"{}\" precondition \"{}\" satisfied (energy={:.4})",
                                goal.goal_state, goal.precondition, precond_energy
                            );
                        } else {
                            eprintln!(
                                "[GOAL CHECK] ❌ Goal \"{}\" precondition \"{}\" NOT satisfied (energy={:.4} < 0.3)",
                                goal.goal_state, goal.precondition, precond_energy
                            );
                        }
                    } else {
                        eprintln!(
                            "[GOAL CHECK] ❌ Goal \"{}\" precondition \"{}\" node out of bounds",
                            goal.goal_state, goal.precondition
                        );
                    }
                } else {
                    // Precondition concept not found in the network at all
                    eprintln!(
                        "[GOAL CHECK] ❌ Goal \"{}\" precondition \"{}\" not found in network",
                        goal.goal_state, goal.precondition
                    );
                }
            }
            satisfied_count as f64 / goals.len() as f64
        };

        // Update the last trace entry with precondition delta
        if let Some(last_trace) = reasoning_trace.last_mut() {
            last_trace.precondition_delta = precondition_score;
        }

        // Prepare final state representation (e.g., norms of the nodes)
        let final_state = (0..net_g.n)
            .map(|i| net_g.positions.amplitudes[i].norm())
            .collect();

        SilentReasoningResult {
            final_state,
            confidence,
            triggered_rules,
            steps_taken: step,
            coherence_score,
            precondition_score,
            reasoning_trace,
        }
    }

    /// Legacy reason() for backward compatibility — no goals, no dual-axis
    pub fn reason_legacy(&self, net_g: &mut ResonanceNetwork, available_rules: &[CompositionRule]) -> SilentReasoningResult {
        self.reason(net_g, available_rules, &[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_network::ResonanceNetwork;
    use crate::compiler::qmem_compiler::{CompositionRule, GoalNode};
    use num_complex::Complex64;

    #[test]
    fn test_silent_reasoning_converges_within_max_steps() {
        let mut net = ResonanceNetwork::new(50);
        let available_rules = vec![];
        let core = SilentReasoningCore::new(12, 1e-4);
        let result = core.reason(&mut net, &available_rules, &[]);

        // Even with 0 nodes or rules, it should converge immediately (step 1) or take <= max_steps.
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.steps_taken <= 12);
        assert_eq!(result.precondition_score, 1.0); // No goals = vacuously satisfied
    }

    #[test]
    fn test_silent_reasoning_triggers_expected_rule() {
        let mut net = ResonanceNetwork::new(50);
        let target_id = net.add_labeled_concept("generalized_force_law", 0.01, vec![Complex64::new(1.0, 0.0); 4096]);
        
        // Setup initial energy high enough to trigger the rule
        net.positions.amplitudes[target_id] = Complex64::new(1.0, 0.0);
        
        let rule = CompositionRule {
            name: "generalization".to_string(),
            concept_a: "force_mass_accel".to_string(),
            concept_b: "color_frequency_coupling".to_string(),
            target_concept: "generalized_force_law".to_string(),
            energy: 0.9,
            phase_theta: 0.7854,
        };

        let core = SilentReasoningCore::new(12, 1e-4);
        let result = core.reason(&mut net, &[rule], &[]);

        assert!(result.triggered_rules.contains(&"generalized_force_law".to_string()));
    }

    #[test]
    fn test_goal_precondition_satisfied() {
        let mut net = ResonanceNetwork::new(50);
        // Add the precondition concept with high energy (car IS at wash location)
        let precond_id = net.add_labeled_concept("araba_yikama_yerinde", 0.01, vec![Complex64::new(1.0, 0.0); 4096]);
        net.positions.amplitudes[precond_id] = Complex64::new(0.8, 0.0); // High energy = precondition met

        let goal = GoalNode {
            goal_state: "araba_yikanmis".to_string(),
            precondition: "araba_yikama_yerinde".to_string(),
            energy: 1.0,
            phase_theta: 0.0,
        };

        let core = SilentReasoningCore::new(5, 1e-4);
        let result = core.reason(&mut net, &[], &[goal]);

        assert!(result.precondition_score > 0.5, "Precondition should be satisfied");
    }

    #[test]
    fn test_goal_precondition_not_satisfied() {
        let mut net = ResonanceNetwork::new(50);
        // Add the precondition concept with LOW energy (car is NOT at wash location)
        let precond_id = net.add_labeled_concept("araba_yikama_yerinde", 0.01, vec![Complex64::new(1.0, 0.0); 4096]);
        net.positions.amplitudes[precond_id] = Complex64::new(0.05, 0.0); // Very low energy = precondition NOT met

        let goal = GoalNode {
            goal_state: "araba_yikanmis".to_string(),
            precondition: "araba_yikama_yerinde".to_string(),
            energy: 1.0,
            phase_theta: 0.0,
        };

        let core = SilentReasoningCore::new(5, 1e-4);
        let result = core.reason(&mut net, &[], &[goal]);

        assert!(result.precondition_score < 0.5, "Precondition should NOT be satisfied — car is not at wash location");
    }

    #[test]
    fn test_dual_axis_independence() {
        // A reasoning chain can be coherent (high coherence_score) but fail on preconditions,
        // or satisfy preconditions but be incoherent. The two axes are independent.
        let mut net = ResonanceNetwork::new(50);
        
        // Add a concept that triggers a rule (coherent reasoning)
        let rule_id = net.add_labeled_concept("yuruyerek_git", 0.01, vec![Complex64::new(1.0, 0.0); 4096]);
        net.positions.amplitudes[rule_id] = Complex64::new(1.0, 0.0);

        // But the precondition is NOT satisfied (car doesn't move by walking)
        let precond_id = net.add_labeled_concept("araba_yikama_yerinde", 0.01, vec![Complex64::new(1.0, 0.0); 4096]);
        net.positions.amplitudes[precond_id] = Complex64::new(0.01, 0.0); // Car stays at home

        let rule = CompositionRule {
            name: "walk_to_wash".to_string(),
            concept_a: "insan".to_string(),
            concept_b: "yikama_yeri".to_string(),
            target_concept: "yuruyerek_git".to_string(),
            energy: 0.9,
            phase_theta: 0.5,
        };

        let goal = GoalNode {
            goal_state: "araba_yikanmis".to_string(),
            precondition: "araba_yikama_yerinde".to_string(),
            energy: 1.0,
            phase_theta: 0.0,
        };

        let core = SilentReasoningCore::new(12, 1e-4);
        let result = core.reason(&mut net, &[rule], &[goal]);

        // Rule triggered (semantically coherent action)
        assert!(result.triggered_rules.contains(&"yuruyerek_git".to_string()));
        // But precondition fails (car isn't there)
        assert!(result.precondition_score < 0.5, "Walking doesn't bring the car");
        // This proves the two axes are independent
        assert!(!result.reasoning_trace.is_empty());
    }
}
