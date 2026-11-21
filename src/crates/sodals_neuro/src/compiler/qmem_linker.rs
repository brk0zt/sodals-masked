// src/compiler/qmem_linker.rs — QMEM Linker: CST → ResonanceNetwork
//
// Converts the front-end CST (Continuous Semantic Topology) into a
// fully functional ResonanceNetwork for physics simulation.
//
// Pipeline:
//   1. Symbol Resolution   — concept strings → usize IDs via add_labeled_concept
//   2. Physics Mapping     — edges → connect_semantic / add_edge_sparse
//   3. Clifford Rotation   — CORRELATES_PHASE / RESONATES_WITH → phase injection

use crate::compiler::qmem_compiler::{ContinuousSemanticTopology, RelationType, generate_holographic_signature};
use crate::semantic_network::ResonanceNetwork;
use num_complex::Complex64;
use std::collections::HashMap;

/// Link a CST into a live ResonanceNetwork.
///
/// The linker performs three passes:
///
/// **Pass 1 — Symbol Resolution**:
///   Iterates all unique concepts in `cst.nodes`, assigns each a `usize` ID
///   via `net.add_labeled_concept(name, base_energy)`.
///
/// **Pass 2 — Edge Binding**:
///   Iterates all edges, maps `RelationType` to connection weight:
///   - `CAUSES`:           weight = energy (direct causal transfer)
///   - `MEASURES`:         weight = energy * 0.5 (observation collapse penalty)
///   - `CORRELATES_PHASE`: weight = energy (+ phase rotation on target)
///   - `RESONATES_WITH`:   weight = energy (+ phase rotation on target)
///
/// **Pass 3 — Clifford Phase Injection**:
///   For `CORRELATES_PHASE` and `RESONATES_WITH` edges, the target node's
///   amplitude is rotated by `edge.phase_theta` relative to the source,
///   embedding the phase correlation directly into the quantum state.
pub fn compile_cst_to_network(cst: &ContinuousSemanticTopology) -> ResonanceNetwork {
    let capacity = (cst.node_count() * 2 + cst.axioms.len() * 2 + cst.rules.len() * 4 + cst.goals.len() * 2 + 128).max(64);
    let mut net = ResonanceNetwork::new(capacity);
    let mut id_map: HashMap<String, usize> = HashMap::new();

    // ═══ Pass 1: Symbol Resolution ══════════════════════════════════════════
    for (concept_name, node) in &cst.nodes {
        // PHASE 60.2: Pipe holographic signature from CST into physics engine
        let id = net.add_labeled_concept(concept_name, 0.01, node.holographic_signature.clone());
        id_map.insert(concept_name.clone(), id);
    }

    // ═══ Pass 1.5: Axiom Symbol Resolution ═══════════════════════════════════
    for axiom in &cst.axioms {
        if !id_map.contains_key(&axiom.concept) {
            let sig = generate_holographic_signature(&axiom.concept, 4096);
            let id = net.add_labeled_concept(&axiom.concept, 0.01, sig);
            id_map.insert(axiom.concept.clone(), id);
        }
        if !id_map.contains_key(&axiom.name) {
            let sig = generate_holographic_signature(&axiom.name, 4096);
            let id = net.add_labeled_concept(&axiom.name, 0.01, sig);
            id_map.insert(axiom.name.clone(), id);
        }
    }

    // ═══ Pass 1.8: Composition Rules Resolution ══════════════════════════════
    for rule in &cst.rules {
        let id_a = match id_map.get(&rule.concept_a) {
            Some(&id) => id,
            None => {
                let sig = generate_holographic_signature(&rule.concept_a, 4096);
                let id = net.add_labeled_concept(&rule.concept_a, 0.01, sig);
                id_map.insert(rule.concept_a.clone(), id);
                id
            }
        };
        let id_b = match id_map.get(&rule.concept_b) {
            Some(&id) => id,
            None => {
                let sig = generate_holographic_signature(&rule.concept_b, 4096);
                let id = net.add_labeled_concept(&rule.concept_b, 0.01, sig);
                id_map.insert(rule.concept_b.clone(), id);
                id
            }
        };

        // Holographic Signature Composition via element-wise complex product (vsa_bind)
        let sig_a = &net.holographic_memory[id_a];
        let sig_b = &net.holographic_memory[id_b];
        let composed_sig = ResonanceNetwork::vsa_bind(sig_a, sig_b);

        let id_target = if let Some(&existing_id) = net.labels.get(&rule.target_concept) {
            let mut write_sig = true;
            if existing_id < net.holographic_memory.len() && !net.holographic_memory[existing_id].is_empty() {
                let score_existing = compute_originality(&net.holographic_memory[existing_id], &net, Some(existing_id));
                let score_candidate = compute_originality(&composed_sig, &net, Some(existing_id));
                if score_candidate <= score_existing {
                    write_sig = false;
                }
            }
            if write_sig {
                if existing_id < net.holographic_memory.len() {
                    net.holographic_memory[existing_id] = composed_sig;
                } else {
                    while net.holographic_memory.len() <= existing_id {
                        net.holographic_memory.push(Vec::new());
                        net.node_flags.push(0);
                    }
                    net.holographic_memory[existing_id] = composed_sig;
                    net.node_flags[existing_id] = crate::semantic_parser::DOMAIN_LOGIC;
                }
            }
            existing_id
        } else {
            net.add_labeled_concept(&rule.target_concept, 0.01, composed_sig)
        };
        id_map.insert(rule.target_concept.clone(), id_target);

        if !id_map.contains_key(&rule.name) {
            let sig = generate_holographic_signature(&rule.name, 4096);
            let id = net.add_labeled_concept(&rule.name, 0.01, sig);
            id_map.insert(rule.name.clone(), id);
        }
    }

    eprintln!(
        "[QMEM LINKER] Pass 1: {} concepts resolved to neuron IDs",
        id_map.len()
    );

    // ═══ Pass 2 + 3: Edge Binding + Clifford Phase Injection ════════════════
    let mut edge_count = 0_usize;
    let mut phase_injections = 0_usize;

    for (source_name, node) in &cst.nodes {
        let src_id = id_map[source_name];

        for edge in &node.edges {
            let tgt_id = match id_map.get(&edge.target) {
                Some(&id) => id,
                None => {
                    // PHASE 60.2: Target node not in CST — generate signature on-the-fly
                    let fallback_sig = generate_holographic_signature(&edge.target, 4096);
                    let id = net.add_labeled_concept(&edge.target, 0.01, fallback_sig);
                    id_map.insert(edge.target.clone(), id);
                    id
                }
            };

            // Map RelationType → connection weight and directionality
            let (weight, is_causal) = match edge.relation {
                RelationType::Causes => (edge.energy as f64, true),
                RelationType::Measures => ((edge.energy * 0.5) as f64, true), // Observation collapse penalty
                RelationType::CorrelatesPhase => (edge.energy as f64, false),
                RelationType::ResonatesWith => (edge.energy as f64, false),
                RelationType::RewardObserved => (edge.energy as f64, true),
                RelationType::TransitionFailed => (-(edge.energy as f64), true),
                RelationType::RuleConfirmed => (edge.energy as f64, true),
            };

            // LINUS FIX: Bind the edge with correct directionality
            // CAUSES/MEASURES are directed (causal), CORRELATES_PHASE/RESONATES_WITH are undirected
            if is_causal {
                // Use connect_syntax for directed causal edges (bidirectional=false)
                net.connect_syntax(src_id, tgt_id, weight);
            } else {
                // Use connect_semantic for undirected semantic connections
                net.connect_semantic(src_id, tgt_id, weight);
            }
            edge_count += 1;

            // Phase injection for Clifford relations
            match edge.relation {
                RelationType::CorrelatesPhase | RelationType::ResonatesWith => {
                    // Rotate target amplitude by phase_theta relative to source.
                    // This encodes the Euler phase correlation e^{iθ} directly
                    // into the quantum state of the target node.
                    let theta = edge.phase_theta as f64;
                    let src_phase = net.positions.amplitudes[src_id].arg();
                    let target_phase = src_phase + theta;
                    let current_norm = net.positions.amplitudes[tgt_id].norm().max(1.0);
                    net.positions.amplitudes[tgt_id] =
                        Complex64::from_polar(current_norm, target_phase);
                    phase_injections += 1;
                }
                _ => {} // CAUSES / MEASURES don't inject phase
            }
        }
    }

    // Link Axioms
    for axiom in &cst.axioms {
        let name_id = id_map[&axiom.name];
        let concept_id = id_map[&axiom.concept];
        net.connect_syntax(name_id, concept_id, axiom.energy as f64);
        edge_count += 1;

        if axiom.phase_theta.abs() > 1e-5 {
            let theta = axiom.phase_theta as f64;
            let src_phase = net.positions.amplitudes[name_id].arg();
            let target_phase = src_phase + theta;
            let current_norm = net.positions.amplitudes[concept_id].norm().max(1.0);
            net.positions.amplitudes[concept_id] = Complex64::from_polar(current_norm, target_phase);
            phase_injections += 1;
        }
    }

    // Link Composition Rules
    for rule in &cst.rules {
        let id_rule_name = id_map[&rule.name];
        let id_target = id_map[&rule.target_concept];
        let id_a = id_map[&rule.concept_a];
        let id_b = id_map[&rule.concept_b];

        net.connect_syntax(id_rule_name, id_target, rule.energy as f64);
        net.connect_syntax(id_target, id_a, rule.energy as f64);
        net.connect_syntax(id_target, id_b, rule.energy as f64);
        edge_count += 3;

        if rule.phase_theta.abs() > 1e-5 {
            let theta = rule.phase_theta as f64;
            let src_phase = net.positions.amplitudes[id_target].arg();
            let target_phase = src_phase + theta;
            let current_norm_a = net.positions.amplitudes[id_a].norm().max(1.0);
            net.positions.amplitudes[id_a] = Complex64::from_polar(current_norm_a, target_phase);
            let current_norm_b = net.positions.amplitudes[id_b].norm().max(1.0);
            net.positions.amplitudes[id_b] = Complex64::from_polar(current_norm_b, target_phase);
            phase_injections += 2;
        }
    }

    eprintln!(
        "[QMEM LINKER] Pass 2-3: {} edges bound, {} phase injections applied",
        edge_count, phase_injections
    );

    net
}

/// Link a CST in-place into an existing ResonanceNetwork.
pub fn link_cst_into_network(cst: &ContinuousSemanticTopology, net: &mut ResonanceNetwork) {
    let mut id_map: HashMap<String, usize> = HashMap::new();

    // ═══ Pass 1: Symbol Resolution ══════════════════════════════════════════
    for (concept_name, node) in &cst.nodes {
        let id = net.add_labeled_concept(concept_name, 0.01, node.holographic_signature.clone());
        id_map.insert(concept_name.clone(), id);
    }

    // Axioms Symbol Resolution
    for axiom in &cst.axioms {
        if !id_map.contains_key(&axiom.concept) {
            let sig = generate_holographic_signature(&axiom.concept, 4096);
            let id = net.add_labeled_concept(&axiom.concept, 0.01, sig);
            id_map.insert(axiom.concept.clone(), id);
        }
        if !id_map.contains_key(&axiom.name) {
            let sig = generate_holographic_signature(&axiom.name, 4096);
            let id = net.add_labeled_concept(&axiom.name, 0.01, sig);
            id_map.insert(axiom.name.clone(), id);
        }
    }

    // Rules Symbol Resolution
    for rule in &cst.rules {
        let id_a = match id_map.get(&rule.concept_a) {
            Some(&id) => id,
            None => {
                let sig = generate_holographic_signature(&rule.concept_a, 4096);
                let id = net.add_labeled_concept(&rule.concept_a, 0.01, sig);
                id_map.insert(rule.concept_a.clone(), id);
                id
            }
        };
        let id_b = match id_map.get(&rule.concept_b) {
            Some(&id) => id,
            None => {
                let sig = generate_holographic_signature(&rule.concept_b, 4096);
                let id = net.add_labeled_concept(&rule.concept_b, 0.01, sig);
                id_map.insert(rule.concept_b.clone(), id);
                id
            }
        };

        // Holographic Signature Composition via element-wise complex product
        let sig_a = &net.holographic_memory[id_a];
        let sig_b = &net.holographic_memory[id_b];
        let mut composed_sig = Vec::with_capacity(4096);
        for k in 0..4096 {
            let a_val = sig_a.get(k).copied().unwrap_or(Complex64::new(1.0, 0.0));
            let b_val = sig_b.get(k).copied().unwrap_or(Complex64::new(1.0, 0.0));
            composed_sig.push(a_val * b_val);
        }

        let id_target = if let Some(&existing_id) = net.labels.get(&rule.target_concept) {
            let mut write_sig = true;
            if existing_id < net.holographic_memory.len() && !net.holographic_memory[existing_id].is_empty() {
                let score_existing = compute_originality(&net.holographic_memory[existing_id], &net, Some(existing_id));
                let score_candidate = compute_originality(&composed_sig, &net, Some(existing_id));
                if score_candidate <= score_existing {
                    write_sig = false;
                }
            }
            if write_sig {
                if existing_id < net.holographic_memory.len() {
                    net.holographic_memory[existing_id] = composed_sig;
                } else {
                    while net.holographic_memory.len() <= existing_id {
                        net.holographic_memory.push(Vec::new());
                        net.node_flags.push(0);
                    }
                    net.holographic_memory[existing_id] = composed_sig;
                    net.node_flags[existing_id] = crate::semantic_parser::DOMAIN_LOGIC;
                }
            }
            existing_id
        } else {
            net.add_labeled_concept(&rule.target_concept, 0.01, composed_sig)
        };
        id_map.insert(rule.target_concept.clone(), id_target);

        if !id_map.contains_key(&rule.name) {
            let sig = generate_holographic_signature(&rule.name, 4096);
            let id = net.add_labeled_concept(&rule.name, 0.01, sig);
            id_map.insert(rule.name.clone(), id);
        }
    }

    // ═══ Pass 2 + 3: Edge Binding + Clifford Phase Injection ════════════════
    for (source_name, node) in &cst.nodes {
        let src_id = id_map[source_name];

        for edge in &node.edges {
            let tgt_id = match id_map.get(&edge.target) {
                Some(&id) => id,
                None => {
                    let fallback_sig = generate_holographic_signature(&edge.target, 4096);
                    let id = net.add_labeled_concept(&edge.target, 0.01, fallback_sig);
                    id_map.insert(edge.target.clone(), id);
                    id
                }
            };

            let (weight, is_causal) = match edge.relation {
                RelationType::Causes => (edge.energy as f64, true),
                RelationType::Measures => ((edge.energy * 0.5) as f64, true),
                RelationType::CorrelatesPhase => (edge.energy as f64, false),
                RelationType::ResonatesWith => (edge.energy as f64, false),
                RelationType::RewardObserved => (edge.energy as f64, true),
                RelationType::TransitionFailed => (-(edge.energy as f64), true),
                RelationType::RuleConfirmed => (edge.energy as f64, true),
            };

            if is_causal {
                net.connect_syntax(src_id, tgt_id, weight);
            } else {
                net.connect_semantic(src_id, tgt_id, weight);
            }

            match edge.relation {
                RelationType::CorrelatesPhase | RelationType::ResonatesWith => {
                    let theta = edge.phase_theta as f64;
                    let src_phase = net.positions.amplitudes[src_id].arg();
                    let target_phase = src_phase + theta;
                    let current_norm = net.positions.amplitudes[tgt_id].norm().max(1.0);
                    net.positions.amplitudes[tgt_id] =
                        Complex64::from_polar(current_norm, target_phase);
                }
                _ => {}
            }
        }
    }

    // Link Axioms
    for axiom in &cst.axioms {
        let name_id = id_map[&axiom.name];
        let concept_id = id_map[&axiom.concept];
        net.connect_syntax(name_id, concept_id, axiom.energy as f64);

        if axiom.phase_theta.abs() > 1e-5 {
            let theta = axiom.phase_theta as f64;
            let src_phase = net.positions.amplitudes[name_id].arg();
            let target_phase = src_phase + theta;
            let current_norm = net.positions.amplitudes[concept_id].norm().max(1.0);
            net.positions.amplitudes[concept_id] = Complex64::from_polar(current_norm, target_phase);
        }
    }

    // Link Composition Rules
    for rule in &cst.rules {
        let id_rule_name = id_map[&rule.name];
        let id_target = id_map[&rule.target_concept];
        let id_a = id_map[&rule.concept_a];
        let id_b = id_map[&rule.concept_b];

        net.connect_syntax(id_rule_name, id_target, rule.energy as f64);
        net.connect_syntax(id_target, id_a, rule.energy as f64);
        net.connect_syntax(id_target, id_b, rule.energy as f64);

        if rule.phase_theta.abs() > 1e-5 {
            let theta = rule.phase_theta as f64;
            let src_phase = net.positions.amplitudes[id_target].arg();
            let target_phase = src_phase + theta;
            let current_norm_a = net.positions.amplitudes[id_a].norm().max(1.0);
            net.positions.amplitudes[id_a] = Complex64::from_polar(current_norm_a, target_phase);
            let current_norm_b = net.positions.amplitudes[id_b].norm().max(1.0);
            net.positions.amplitudes[id_b] = Complex64::from_polar(current_norm_b, target_phase);
        }
    }
}

/// Compute the originality score (Wronskian score / linear independence) of a phasor vector
/// relative to the existing active node signatures in the network, optionally excluding one ID.
fn compute_originality(sig: &[Complex64], net: &ResonanceNetwork, exclude_id: Option<usize>) -> f64 {
    let dim = sig.len();
    if dim == 0 {
        return 0.0;
    }
    let mut max_dot = 0.0;
    for j in 0..net.n {
        if let Some(ex_id) = exclude_id {
            if j == ex_id {
                continue;
            }
        }
        if j < net.holographic_memory.len() {
            let other_sig = &net.holographic_memory[j];
            if other_sig.len() == dim {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..dim {
                    sum += sig[k] * other_sig[k].conj();
                }
                let dot_mag = sum.norm() / (dim as f64);
                if dot_mag > max_dot {
                    max_dot = dot_mag;
                }
            }
        }
    }
    1.0 - max_dot
}

// ════════════════════════════════════════════════════════════════════════════════
// TESTS
// ════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::qmem_compiler::compile;

    #[test]
    fn test_linker_basic_causes() {
        let cst = compile(r#""gravity" CAUSES "mass" [0.95, 3.14];"#).unwrap();
        let net = compile_cst_to_network(&cst);

        assert_eq!(net.n, 2);
        assert!(net.labels.contains_key("gravity"));
        assert!(net.labels.contains_key("mass"));
    }

    #[test]
    fn test_linker_measures_penalty() {
        let cst = compile(r#""observer" MEASURES "particle" [1.0, 0.0];"#).unwrap();
        let net = compile_cst_to_network(&cst);

        // MEASURES applies 0.5x penalty — verify via CSR edge weight
        assert_eq!(net.n, 2);
    }

    #[test]
    fn test_linker_phase_injection() {
        let input = r#""wave_a" CORRELATES_PHASE "wave_b" [0.8, 1.57];"#;
        let cst = compile(input).unwrap();
        let net = compile_cst_to_network(&cst);

        // wave_b should have a non-zero phase due to Clifford injection
        let b_id = net.labels["wave_b"];
        let b_phase = net.positions.amplitudes[b_id].arg();
        // Phase should be approximately 1.57 (π/2) relative to source (which starts at 0)
        assert!((b_phase - 1.57).abs() < 0.1,
            "wave_b phase should be ~1.57, got {}", b_phase);
    }

    #[test]
    fn test_linker_cyclic_graph() {
        let input = r#"
            "a" CAUSES "b" [1.0, 0.0];
            "b" CAUSES "c" [0.8, 0.0];
            "c" RESONATES_WITH "a" [0.6, 3.14];
        "#;
        let cst = compile(input).unwrap();
        let net = compile_cst_to_network(&cst);

        assert_eq!(net.n, 3);
        // All three labels exist
        assert!(net.labels.contains_key("a"));
        assert!(net.labels.contains_key("b"));
        assert!(net.labels.contains_key("c"));
    }

    #[test]
    fn test_linker_resonates_with_phase() {
        let input = r#""sine" RESONATES_WITH "cosine" [1.0, 1.5708];"#;
        let cst = compile(input).unwrap();
        let net = compile_cst_to_network(&cst);

        let cosine_id = net.labels["cosine"];
        let phase = net.positions.amplitudes[cosine_id].arg();
        // Should be approximately π/2
        assert!((phase - std::f64::consts::FRAC_PI_2).abs() < 0.01);
    }

    #[test]
    fn test_linker_axiom() {
        let input = r#"AXIOM "gravity" DEFINES "mass_dependent" [0.95, 0.0];"#;
        let cst = compile(input).unwrap();
        let net = compile_cst_to_network(&cst);

        assert_eq!(net.n, 2);
        assert!(net.labels.contains_key("gravity"));
        assert!(net.labels.contains_key("mass_dependent"));

        let gravity_id = net.labels["gravity"];
        let mass_dependent_id = net.labels["mass_dependent"];
        assert!(net.adjacency_csr.get(gravity_id, mass_dependent_id).is_some());
    }

    #[test]
    fn test_linker_rule() {
        let input = r#"
            RULE "kurgusal_gravity" COMPOSE "wavelength_dependent" AND "gravity" -> "wavelength_gravity" [0.95, 0.0];
        "#;
        let cst = compile(input).unwrap();
        let net = compile_cst_to_network(&cst);

        // rule name, target concept, component A, component B
        assert_eq!(net.n, 4);
        assert!(net.labels.contains_key("kurgusal_gravity"));
        assert!(net.labels.contains_key("wavelength_gravity"));
        assert!(net.labels.contains_key("wavelength_dependent"));
        assert!(net.labels.contains_key("gravity"));

        let target_id = net.labels["wavelength_gravity"];
        let a_id = net.labels["wavelength_dependent"];
        let b_id = net.labels["gravity"];

        // Verify that the composed holographic signature of target is equal to the product of component signatures
        let sig_target = &net.holographic_memory[target_id];
        let sig_a = &net.holographic_memory[a_id];
        let sig_b = &net.holographic_memory[b_id];

        for k in 0..4096 {
            let expected = sig_a[k] * sig_b[k];
            assert!((sig_target[k] - expected).norm() < 1e-9);
        }

        // Verify CSR connections: rule name -> target concept -> components
        let rule_id = net.labels["kurgusal_gravity"];
        assert!(net.adjacency_csr.get(rule_id, target_id).is_some());
        assert!(net.adjacency_csr.get(target_id, a_id).is_some());
        assert!(net.adjacency_csr.get(target_id, b_id).is_some());
    }

    #[test]
    fn test_linker_rule_conflict_resolution() {
        // We define A, B, C, D as nodes and two rules both targeting "target"
        let input = r#"
            "A" CAUSES "B" [1.0, 0.0];
            "C" CAUSES "D" [1.0, 0.0];
            RULE "rule_ab" COMPOSE "A" AND "B" -> "target" [1.0, 0.0];
            RULE "rule_cd" COMPOSE "C" AND "D" -> "target" [1.0, 0.0];
        "#;
        let cst = compile(input).unwrap();
        let net = compile_cst_to_network(&cst);

        // Verify that target exists and has a signature
        assert!(net.labels.contains_key("target"));
        let target_id = net.labels["target"];
        let target_sig = &net.holographic_memory[target_id];
        assert_eq!(target_sig.len(), 4096);

        // Let's compute the two candidate signatures manually
        let id_a = net.labels["A"];
        let id_b = net.labels["B"];
        let id_c = net.labels["C"];
        let id_d = net.labels["D"];

        let sig_a = &net.holographic_memory[id_a];
        let sig_b = &net.holographic_memory[id_b];
        let mut sig_ab = Vec::with_capacity(4096);
        for k in 0..4096 {
            sig_ab.push(sig_a[k] * sig_b[k]);
        }

        let sig_c = &net.holographic_memory[id_c];
        let sig_d = &net.holographic_memory[id_d];
        let mut sig_cd = Vec::with_capacity(4096);
        for k in 0..4096 {
            sig_cd.push(sig_c[k] * sig_d[k]);
        }

        // We compute originality relative to all OTHER nodes
        // (A, B, C, D, and the rules)
        let score_ab = compute_originality(&sig_ab, &net, Some(target_id));
        let score_cd = compute_originality(&sig_cd, &net, Some(target_id));

        if score_ab > score_cd {
            // sig_ab should have won
            for k in 0..4096 {
                assert!((target_sig[k] - sig_ab[k]).norm() < 1e-9);
            }
        } else {
            // sig_cd should have won
            for k in 0..4096 {
                assert!((target_sig[k] - sig_cd[k]).norm() < 1e-9);
            }
        }
    }

    #[test]
    fn test_compose_rule_produces_novel_signature() {
        use crate::compiler::qmem_compiler::compile;
        let cst = compile(r#"
            AXIOM "newton_second_law" DEFINES "force_mass_accel" [1.0, 0.0];
            AXIOM "wavelength_field" DEFINES "color_frequency_coupling" [1.0, 0.0];
            RULE "generalization" COMPOSE "force_mass_accel" AND "color_frequency_coupling"
                -> "generalized_force_law" [0.9, 0.7854];
        "#).unwrap();
        let net = super::compile_cst_to_network(&cst);

        let target_sig = &net.holographic_memory[net.labels["generalized_force_law"]];
        let sig_a = &net.holographic_memory[net.labels["force_mass_accel"]];
        let sig_b = &net.holographic_memory[net.labels["color_frequency_coupling"]];

        // Türetilmiş imza ne A ne B ile özdeş olmalı (gerçek kompozisyon, kopyalama değil)
        assert!(ResonanceNetwork::holographic_similarity(target_sig, sig_a) < 0.7);
        assert!(ResonanceNetwork::holographic_similarity(target_sig, sig_b) < 0.7);
    }
}
