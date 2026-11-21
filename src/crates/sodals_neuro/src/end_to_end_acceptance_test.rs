// src/end_to_end_acceptance_test.rs
// Phase A.4 Acceptance Criteria Simulator

#[cfg(test)]
mod tests {
    use crate::semantic_network::ResonanceNetwork;
    use crate::engine::silent_reasoning::SilentReasoningCore;
    use crate::compiler::qmem_compiler::CompositionRule;
    use num_complex::Complex64;

    #[test]
    fn test_phase_a4_acceptance_criteria() {
        // Girdi: "Bu evrende yerçekimi kütleye değil rengin yaydığı dalga boyu frekansına
        // bağlıdır ve momentum cismin şekline göre değişir. Kırmızı ışık yayan küresel bir
        // cisim 30° açıyla fırlatılırsa hangi denklemle yörüngesini hesaplarsın?"
        
        let mut net_g = ResonanceNetwork::new(50);
        
        // 1. net_f seed (simulated by injecting required concepts into net_g for reasoning)
        // In the real system, net_f diffuses energy to trigger these in net_g.
        let id_wavelength = net_g.add_labeled_concept("renk_frekans_alani", 0.1, vec![Complex64::new(1.0, 0.0); 4096]);
        let id_newton = net_g.add_labeled_concept("newton_second_law", 0.1, vec![Complex64::new(1.0, 0.0); 4096]);
        let id_target = net_g.add_labeled_concept("generalized_force_law", 0.01, vec![Complex64::new(1.0, 0.0); 4096]);
        
        // Setup initial energy high enough to trigger the rule (simulate prompt activation)
        net_g.positions.amplitudes[id_target] = Complex64::new(1.0, 0.0);
        
        // 2. Axioms and composition rule loaded from QMEM Cartridge
        let rule = CompositionRule {
            name: "generalization".to_string(),
            concept_a: "newton_second_law".to_string(),
            concept_b: "renk_frekans_alani".to_string(),
            target_concept: "generalized_force_law".to_string(),
            energy: 0.9,
            phase_theta: 0.7854,
        };

        // 3. SilentReasoningCore 12 adım içinde COMPOSE tetikler
        let core = SilentReasoningCore::new(12, 1e-4);
        let result = core.reason_legacy(&mut net_g, &[rule]);
        
        // 4. "generalized_force_law" türetilir ve loglanır
        assert!(result.triggered_rules.contains(&"generalized_force_law".to_string()), "Kabul Kriteri: generalized_force_law türetilemedi!");
        
        // 5. Modelin güveni hesaplanır
        assert!(result.confidence > 0.0, "Kabul Kriteri: Güven skoru sıfır olamaz.");
    }
}
