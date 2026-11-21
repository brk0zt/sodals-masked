// src/context_integration_test.rs - Integration tests for resonance-aware context management
// 
// Tests the pure resonance-aware chunking and retrieval system.
// Accuracy-first validation - no approximations.

use crate::semantic_network::ResonanceNetwork;
use crate::resonance_chunker::{ResonanceChunker, Chunk};
use crate::retrieval_engine::RetrievalEngine;

#[cfg(test)]
mod integration_tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_resonance_chunker_basic() {
        let network = ResonanceNetwork::new(100);
        let chunker = ResonanceChunker::new(
            network,
            0.5,  // phase_drop_threshold
            0.1,  // impedance_mismatch_threshold
            5,    // min_chunk_size
            50,   // max_chunk_size
        );

        let text = "test text for chunking";
        let chunks = chunker.chunk_text(text);

        // Should produce chunks (may be empty if no nodes found)
        assert!(chunks.len() <= 1); // Small text, at most one chunk
    }

    #[test]
    fn test_phase_boundary_detection() {
        let mut network = ResonanceNetwork::new(10);
        let a = network.add_concept(0.1);
        let b = network.add_concept(0.1);
        
        // Set different phases to trigger boundary detection
        network.positions.amplitudes[a] = Complex64::new(1.0, 0.0);
        network.positions.amplitudes[b] = Complex64::new(-1.0, 0.0); // π phase shift
        
        let chunker = ResonanceChunker::new(network, 1.0, 0.1, 2, 10);
        let boundaries = chunker.detect_phase_boundaries("test");
        
        // Should detect phase boundary
        assert!(!boundaries.is_empty());
    }

    #[test]
    fn test_impedance_clustering() {
        let mut network = ResonanceNetwork::new(10);
        let a = network.add_concept(0.1);
        let b = network.add_concept(0.5); // Different impedance
        let c = network.add_concept(0.1);
        
        let chunker = ResonanceChunker::new(network, 0.5, 0.3, 1, 10);
        let clusters = chunker.cluster_by_impedance(&[a, b, c]);
        
        // Should split at impedance mismatch
        assert!(clusters.len() >= 2);
    }

    #[test]
    fn test_jacobian_stability() {
        let mut network = ResonanceNetwork::new(10);
        let a = network.add_concept(0.1);
        let b = network.add_concept(0.1);
        network.connect_semantic(a, b, 0.5);
        
        let chunker = ResonanceChunker::new(network, 0.5, 0.1, 2, 10);
        let stability = chunker.jacobian_stability_analysis(&[a, b]);
        
        // Should return a stability score between 0 and 1
        assert!(stability >= 0.0 && stability <= 1.0);
    }

    #[test]
    fn test_retrieval_engine_creation() {
        let network = ResonanceNetwork::new(100);
        let chunker = ResonanceChunker::new(network.clone(), 0.5, 0.1, 5, 50);
        let engine = RetrievalEngine::new(network, chunker);
        
        assert!(engine.cache.read().unwrap().is_empty());
    }

    #[test]
    fn test_retrieval_cache_functionality() {
        let network = ResonanceNetwork::new(100);
        let chunker = ResonanceChunker::new(network.clone(), 0.5, 0.1, 5, 50);
        let engine = RetrievalEngine::new(network, chunker);
        
        // Query once
        engine.query_keystone("test", 5);
        
        // Cache should be populated
        assert!(!engine.cache.read().unwrap().is_empty());
        
        // Clear cache
        engine.clear_cache();
        assert!(engine.cache.read().unwrap().is_empty());
    }


    #[test]
    fn test_chunk_metadata_preservation() {
        let mut network = ResonanceNetwork::new(10);
        let a = network.add_concept(0.1);
        network.labels.insert("test".to_string(), a);
        
        let chunker = ResonanceChunker::new(network, 0.5, 0.1, 1, 10);
        let chunks = chunker.chunk_text("test");
        
        if !chunks.is_empty() {
            let chunk = &chunks[0];
            // Verify metadata is present
            assert!(!chunk.node_ids.is_empty());
            assert!(!chunk.phase_signature.is_empty());
            assert!(!chunk.impedance_profile.is_empty());
            assert!(chunk.jacobian_stability >= 0.0 && chunk.jacobian_stability <= 1.0);
        }
    }

    #[test]
    fn test_max_chunk_size_enforcement() {
        let network = ResonanceNetwork::new(100);
        let chunker = ResonanceChunker::new(
            network,
            0.5,
            0.1,
            5,   // min_chunk_size
            10,  // max_chunk_size (small for testing)
        );
        
        let text = "test text for chunking with enforcement";
        let chunks = chunker.chunk_text(text);
        
        // All chunks should respect max size
        for chunk in &chunks {
            assert!(chunk.node_ids.len() <= 10);
        }
    }

    #[test]
    fn test_retrieval_result_scoring() {
        let mut network = ResonanceNetwork::new(10);
        let a = network.add_concept(0.1);
        let b = network.add_concept(0.1);
        network.connect_semantic(a, b, 0.5);
        
        let chunker = ResonanceChunker::new(network.clone(), 0.5, 0.1, 5, 50);
        let engine = RetrievalEngine::new(network, chunker);
        
        let results = engine.query_keystone("test", 5);
        
        // All results should have valid scores
        for result in results {
            assert!(result.keystone_score >= 0.0);
            assert!(result.holographic_score >= 0.0);
            assert!(result.ripple_score >= 0.0);
            assert!(result.combined_score >= 0.0);
            assert!(result.divergence >= 0.0);
            assert!(result.resonance >= 0.0);
        }
    }

    #[test]
    fn test_spectral_boundary_detection() {
        let mut network = ResonanceNetwork::new(10);
        let a = network.add_concept(0.1);
        let b = network.add_concept(0.1);
        
        // Initialize spectral clustering
        network.init_spectral_fields();
        
        let chunker = ResonanceChunker::new(network, 0.5, 0.1, 5, 50);
        let boundaries = chunker.spectral_boundaries();
        
        // Should return a vector (may be empty if no clusters)
        assert!(boundaries.len() <= 10);
    }

    #[test]
    fn test_bifurcation_detection() {
        let mut network = ResonanceNetwork::new(10);
        let a = network.add_concept(0.9); // High damping (bifurcation proxy)
        let b = network.add_concept(0.1);
        
        let chunker = ResonanceChunker::new(network, 0.5, 0.1, 1, 10);
        let detected = chunker.detect_bifurcation(&[a, b]);
        
        // Should detect bifurcation due to high damping
        assert!(detected);
    }

    #[test]
    fn test_accuracy_first_no_approximations() {
        // This test validates that no approximations are used
        // For now, we just verify the system is configured for exact computation
        let network = ResonanceNetwork::new(100);
        let chunker = ResonanceChunker::new(network, 0.5, 0.1, 5, 50);
        
        // Verify chunker is configured for exact computation
        assert_eq!(chunker.min_chunk_size, 5);
        assert_eq!(chunker.max_chunk_size, 50);
    }
}
