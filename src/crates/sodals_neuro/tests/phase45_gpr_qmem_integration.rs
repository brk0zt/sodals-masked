#[cfg(test)]
mod phase45_integration_tests {
    use sodals_neuro::phase45_gpr_qmem_router::Phase45GprQmemRouter;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// Integration test: Simulate a 32-layer model with varying bypass patterns
    /// and verify that the GPR-QMEM router produces sensible loading strategies
    #[test]
    fn test_gpr_qmem_full_integration() {
        let num_layers = 32;
        let router = Arc::new(Phase45GprQmemRouter::new(num_layers));

        // Simulate inference pass with realistic bypass patterns:
        // - Early layers: low bypass rate (frequent computation)
        // - Middle layers: mixed bypass rate
        // - Late layers: high bypass rate (often bypass, contain semantic residues)
        
        for seq_pos in 0..100 {
            for layer_idx in 0..num_layers {
                // Compute bypass probability based on layer position
                let early_bias = if layer_idx < 8 { 0.1 } else { 0.0 };
                let late_bias = if layer_idx >= 24 { 0.5 } else { 0.0 };
                let middle_bias = if layer_idx >= 8 && layer_idx < 24 { 0.2 } else { 0.0 };
                
                let bypass_prob = (early_bias + middle_bias + late_bias) + (seq_pos as f32 * 0.001);
                let bypass_prob = bypass_prob.min(1.0).max(0.0);
                
                let should_bypass = (seq_pos as f32 * 7.19) % 1.0 < bypass_prob;
                
                // Simulate phase signal
                let z_i = 0.5 + (layer_idx as f32 * 0.01);
                let semantic_divergence = 0.7 - (layer_idx as f32 * 0.01);
                let jacobian_norm = 0.3 + (if should_bypass { 0.2 } else { 0.0 });
                
                let _ = router.record_routing_decision(
                    layer_idx,
                    should_bypass,
                    z_i,
                    semantic_divergence,
                    jacobian_norm,
                );
            }
        }

        // Verify that early layers have high loading priority, late layers low
        let priorities = router.get_sorted_loading_priorities().unwrap();
        let early_layer_priority = priorities.iter()
            .find(|(idx, _)| *idx < 8)
            .map(|(_, p)| *p)
            .unwrap_or(0.0);
        let late_layer_priority = priorities.iter()
            .find(|(idx, _)| *idx >= 24)
            .map(|(_, p)| *p)
            .unwrap_or(1.0);

        assert!(
            early_layer_priority > late_layer_priority,
            "Early layers should have higher loading priority than late layers"
        );
    }

    /// Test that QMEM loading strategy respects layer priorities and memory budget
    #[test]
    fn test_qmem_loading_strategy_with_budget() {
        let num_layers = 16;
        let router = Phase45GprQmemRouter::new(num_layers);

        // Setup: layers 0-7 will have low bypass rates (high priority)
        //        layers 8-15 will have high bypass rates (low priority)
        for pass in 0..20 {
            for layer_idx in 0..num_layers {
                let should_bypass = layer_idx >= 8;
                let z_i = 0.5;
                let semantic_div = if layer_idx < 8 { 0.8 } else { 0.3 };
                let jac_norm = if layer_idx < 8 { 0.1 } else { 0.5 };

                let _ = router.record_routing_decision(layer_idx, should_bypass, z_i, semantic_div, jac_norm);
            }
        }

        // Define layer sizes (all 1MB each)
        let mut layer_sizes = HashMap::new();
        for layer_idx in 0..num_layers {
            layer_sizes.insert(layer_idx, 1024 * 1024);
        }

        // Budget: 10MB RAM (10 layers)
        let max_ram = 10 * 1024 * 1024;

        // Compute strategy
        let strategy = router.compute_optimal_loading_strategy(&layer_sizes, max_ram).unwrap();

        // Verify: high-priority layers (0-7) should be loaded to RAM
        let ram_layers: Vec<usize> = strategy
            .iter()
            .filter(|(_, use_ram)| *use_ram)
            .map(|(idx, _)| *idx)
            .collect();

        for layer_idx in 0..8 {
            assert!(
                ram_layers.contains(&layer_idx),
                "High-priority layer {} should be in RAM",
                layer_idx
            );
        }

        // Verify: we don't exceed RAM budget
        let ram_usage: usize = ram_layers.iter().map(|idx| layer_sizes[idx]).sum();
        assert!(ram_usage <= max_ram, "RAM usage {} exceeds budget {}", ram_usage, max_ram);
    }

    /// Test that warmup period allows all layers to establish baseline statistics
    #[test]
    fn test_warmup_period_protection() {
        let router = Phase45GprQmemRouter::new(4);

        // During warmup (first 5 passes per layer)
        for pass in 0..5 {
            for layer_idx in 0..4 {
                router.record_routing_decision(layer_idx, true, 0.5, 0.3, 0.2).unwrap();
            }
        }

        // All layers should still be in warmup
        for layer_idx in 0..4 {
            assert!(router.is_in_warmup(layer_idx, 10).unwrap());
        }

        // After more passes, should exit warmup
        for pass in 0..5 {
            for layer_idx in 0..4 {
                router.record_routing_decision(layer_idx, false, 0.5, 0.3, 0.2).unwrap();
            }
        }

        // Now all layers should have exited warmup
        for layer_idx in 0..4 {
            assert!(!router.is_in_warmup(layer_idx, 10).unwrap());
        }
    }

    /// Test statistics snapshot for monitoring
    #[test]
    fn test_statistics_snapshot() {
        let router = Phase45GprQmemRouter::new(2);

        // Record some decisions
        router.record_routing_decision(0, true, 0.5, 0.8, 0.2).unwrap();
        router.record_routing_decision(1, false, 0.3, 0.6, 0.4).unwrap();

        // Get snapshot
        let snapshot = router.get_statistics_snapshot().unwrap();
        assert_eq!(snapshot.len(), 2);

        // Layer 0 should have bypass_rate_ema > 0
        let layer0 = &snapshot[0];
        assert!(layer0.bypass_rate_ema > 0.0);

        // Layer 1 should have bypass_rate_ema = 0
        let layer1 = &snapshot[1];
        assert_eq!(layer1.bypass_rate_ema, 0.0);
    }

    /// Test that semantic divergence affects loading decisions
    #[test]
    fn test_semantic_divergence_correlation() {
        let router = Phase45GprQmemRouter::new(2);

        // Layer 0: healthy semantic divergence
        for _ in 0..10 {
            router.record_routing_decision(0, false, 0.5, 0.9, 0.1).unwrap();
        }

        // Layer 1: degraded semantic divergence
        for _ in 0..10 {
            router.record_routing_decision(1, true, 0.5, 0.2, 0.5).unwrap();
        }

        // Layer 0 should have higher loading priority
        let p0 = router.get_loading_priority(0).unwrap();
        let p1 = router.get_loading_priority(1).unwrap();
        assert!(p0 > p1, "Healthy layer should have higher priority");
    }

    /// Test Jacobian norm correlation with computational complexity
    #[test]
    fn test_jacobian_norm_computational_signal() {
        let router = Phase45GprQmemRouter::new(3);

        // Layer 0: stable (low Jacobian)
        for _ in 0..5 {
            router.record_routing_decision(0, false, 0.5, 0.7, 0.1).unwrap();
        }

        // Layer 1: mixed stability
        for i in 0..5 {
            let jac = if i % 2 == 0 { 0.3 } else { 0.6 };
            router.record_routing_decision(1, i % 2 == 1, 0.5, 0.5, jac).unwrap();
        }

        // Layer 2: unstable (high Jacobian)
        for _ in 0..5 {
            router.record_routing_decision(2, true, 0.5, 0.2, 0.8).unwrap();
        }

        // Verify priorities reflect stability
        let priorities = router.get_sorted_loading_priorities().unwrap();
        let p0 = priorities.iter().find(|(idx, _)| *idx == 0).unwrap().1;
        let p2 = priorities.iter().find(|(idx, _)| *idx == 2).unwrap().1;
        assert!(p0 > p2, "Stable layer should have higher priority than unstable");
    }
}
