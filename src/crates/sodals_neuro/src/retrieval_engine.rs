// src/retrieval_engine.rs - Pure Resonance-Aware Retrieval Engine for SODALS
// 
// Implements keystone-first, holographic similarity, and propagation-based retrieval.
// NO fallback to embedding-based retrieval or vector DB - pure VSA and wave physics.
// Accuracy-first approach: exact computation only, no approximate nearest neighbor.

use crate::semantic_network::ResonanceNetwork;
use crate::resonance_chunker::{ResonanceChunker, Chunk};
use num_complex::Complex64;
use std::collections::HashMap;
use std::sync::RwLock;

/// Pure resonance-aware retrieval engine
/// 
/// Uses physics-inspired retrieval methods:
/// - Keystone identification + ripple effect propagation
/// - Holographic similarity for VSA signature matching
/// - Spectral clustering for zone-based retrieval
/// - Propagation-based multi-hop reasoning
pub struct RetrievalEngine {
    network: ResonanceNetwork,
    chunker: ResonanceChunker,
    pub cache: RwLock<HashMap<String, Vec<RetrievalResult>>>,
}

/// Retrieval result with resonance-aware scoring
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub chunk_id: usize,
    pub text: String,
    pub node_ids: Vec<usize>,
    pub keystone_score: f64,
    pub holographic_score: f64,
    pub ripple_score: f64,
    pub combined_score: f64,
    pub divergence: f64,
    pub resonance: f64,
}

impl RetrievalEngine {
    /// Create new retrieval engine
    /// 
    /// # Arguments
    /// * `network` - ResonanceNetwork for wave propagation and topology
    /// * `chunker` - ResonanceChunker for text segmentation
    pub fn new(network: ResonanceNetwork, chunker: ResonanceChunker) -> Self {
        Self {
            network,
            chunker,
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Query using keystone-first strategy
    /// 
    /// Identifies top-k keystone nodes and propagates ripple effect to find relevant chunks.
    /// Exact computation - no approximate nearest neighbor.
    pub fn query_keystone(&self, query: &str, top_k: usize) -> Vec<RetrievalResult> {
        // Check cache first
        let cache_key = format!("keystone_{}_{}", query, top_k);
        {
            let cache = self.cache.read().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                return cached.clone();
            }
        }

        // Extract query nodes
        let query_nodes = self.chunker.extract_node_sequence(query);
        if query_nodes.is_empty() {
            let mut cache = self.cache.write().unwrap();
            cache.insert(cache_key, Vec::new());
            return Vec::new();
        }

        // Identify keystones from network
        let keystones = self.network.identify_keystones(top_k * 2); // Get more for filtering

        // Compute ripple scores for each keystone
        let mut results: Vec<RetrievalResult> = Vec::new();
        for (idx, &keystone_id) in keystones.iter().enumerate() {
            if keystone_id >= self.network.n {
                continue;
            }

            // Propagate change from keystone
            let ripple = self.network.propagate_change(keystone_id);

            // Score based on overlap with query nodes
            let overlap_score = self.compute_overlap_score(&query_nodes, &ripple);

            if overlap_score > 0.0 {
                // Create retrieval result
                let chunk_text = self.get_chunk_text(keystone_id);
                results.push(RetrievalResult {
                    chunk_id: idx,
                    text: chunk_text,
                    node_ids: vec![keystone_id],
                    keystone_score: overlap_score,
                    holographic_score: 0.0,
                    ripple_score: overlap_score,
                    combined_score: overlap_score,
                    divergence: 0.0,
                    resonance: 0.0,
                });
            }
        }

        // Sort by keystone score and take top-k
        results.sort_by(|a, b| b.keystone_score.partial_cmp(&a.keystone_score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        // Cache results
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(cache_key, results.clone());
        }

        results
    }

    /// Query using holographic similarity
    /// 
    /// Uses VSA signature matching for retrieval.
    /// Exact computation - no approximate cosine similarity.
    pub fn query_holographic(&self, query_sig: &[Complex64], top_k: usize) -> Vec<RetrievalResult> {
        // Check cache first
        let cache_key = format!("holographic_{}_{}", query_sig.len(), top_k);
        {
            let cache = self.cache.read().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                return cached.clone();
            }
        }

        let mut results: Vec<RetrievalResult> = Vec::new();

        // Compare query signature with all holographic memories
        for node_id in 0..self.network.n {
            if node_id >= self.network.holographic_memory.len() {
                continue;
            }

            let node_sig = &self.network.holographic_memory[node_id];
            if node_sig.is_empty() {
                continue;
            }

            // Compute holographic similarity (exact VSA cosine)
            let similarity = ResonanceNetwork::holographic_similarity(query_sig, node_sig);

            if similarity > 0.1 { // Minimum similarity threshold
                let chunk_text = self.get_chunk_text(node_id);
                results.push(RetrievalResult {
                    chunk_id: node_id,
                    text: chunk_text,
                    node_ids: vec![node_id],
                    keystone_score: 0.0,
                    holographic_score: similarity,
                    ripple_score: 0.0,
                    combined_score: similarity,
                    divergence: 0.0,
                    resonance: 0.0,
                });
            }
        }

        // Sort by holographic score and take top-k
        results.sort_by(|a, b| b.holographic_score.partial_cmp(&a.holographic_score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        // Cache results
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(cache_key, results.clone());
        }

        results
    }

    /// Hybrid query (keystone + holographic)
    /// 
    /// Combines keystone-first and holographic similarity for comprehensive retrieval.
    /// Exact computation - no fallback to embedding-based retrieval.
    pub fn query_hybrid(&self, query: &str, top_k: usize) -> Vec<RetrievalResult> {
        // Check cache first
        let cache_key = format!("hybrid_{}_{}", query, top_k);
        {
            let cache = self.cache.read().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                return cached.clone();
            }
        }

        // Get keystone results
        let keystone_results = self.query_keystone(query, top_k * 2);

        // Generate query signature for holographic comparison
        let query_nodes = self.chunker.extract_node_sequence(query);
        let query_sig = self.generate_query_signature(&query_nodes);

        // Get holographic results
        let holographic_results = if !query_sig.is_empty() {
            self.query_holographic(&query_sig, top_k * 2)
        } else {
            Vec::new()
        };

        // Combine and deduplicate results
        let mut combined: HashMap<usize, RetrievalResult> = HashMap::new();

        for result in keystone_results {
            combined.entry(result.chunk_id).or_insert(result);
        }

        for result in holographic_results {
            combined.entry(result.chunk_id)
                .and_modify(|existing| {
                    existing.holographic_score = result.holographic_score;
                    existing.combined_score = 0.6 * existing.keystone_score + 0.4 * existing.holographic_score;
                })
                .or_insert(result);
        }

        // Compute combined scores
        for result in combined.values_mut() {
            result.combined_score = 0.6 * result.keystone_score + 0.4 * result.holographic_score;
            
            // Compute divergence and resonance for cognitive health gating
            let (divergence, resonance) = self.compute_divergence_resonance(&result.node_ids);
            result.divergence = divergence;
            result.resonance = resonance;
        }

        // Sort by combined score and take top-k
        let mut results: Vec<_> = combined.into_values().collect();
        results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        // Cache results
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(cache_key, results.clone());
        }

        results
    }

    /// Propagation-based retrieval
    /// 
    /// Uses ripple effect simulation from seed node for multi-hop reasoning.
    /// Exact computation - full depth propagation.
    pub fn query_propagation(&self, seed_node: usize, depth: usize, top_k: usize) -> Vec<RetrievalResult> {
        if seed_node >= self.network.n {
            return Vec::new();
        }

        // Check cache first
        let cache_key = format!("propagation_{}_{}_{}", seed_node, depth, top_k);
        {
            let cache = self.cache.read().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                return cached.clone();
            }
        }

        // Propagate change from seed node
        let ripple = self.network.propagate_change(seed_node);

        // Convert ripple results to retrieval results
        let mut results: Vec<RetrievalResult> = Vec::new();
        for (node_id, ripple_score) in ripple.iter().take(top_k) {
            if *node_id >= self.network.n {
                continue;
            }

            let chunk_text = self.get_chunk_text(*node_id);
            let (divergence, resonance) = self.compute_divergence_resonance(&vec![*node_id]);

            results.push(RetrievalResult {
                chunk_id: *node_id,
                text: chunk_text,
                node_ids: vec![*node_id],
                keystone_score: 0.0,
                holographic_score: 0.0,
                ripple_score: *ripple_score,
                combined_score: *ripple_score,
                divergence,
                resonance,
            });
        }

        // Sort by ripple score
        results.sort_by(|a, b| b.ripple_score.partial_cmp(&a.ripple_score).unwrap_or(std::cmp::Ordering::Equal));

        // Cache results
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(cache_key, results.clone());
        }

        results
    }

    /// Spectral clustering-based retrieval
    /// 
    /// Uses pre-computed spectral clusters for zone-based retrieval.
    pub fn query_spectral(&self, cluster_id: usize, top_k: usize) -> Vec<RetrievalResult> {
        // Check cache first
        let cache_key = format!("spectral_{}_{}", cluster_id, top_k);
        {
            let cache = self.cache.read().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                return cached.clone();
            }
        }

        let mut results: Vec<RetrievalResult> = Vec::new();

        // Get cluster from network
        if let Some(cluster) = self.network.spectral_clusters.get(&cluster_id) {
            for (idx, &node_id) in cluster.nodes.iter().enumerate().take(top_k) {
                if node_id >= self.network.n {
                    continue;
                }

                let chunk_text = self.get_chunk_text(node_id);
                let (divergence, resonance) = self.compute_divergence_resonance(&vec![node_id]);

                results.push(RetrievalResult {
                    chunk_id: node_id,
                    text: chunk_text,
                    node_ids: vec![node_id],
                    keystone_score: cluster.centroid.norm(), // Use centroid magnitude as centrality proxy
                    holographic_score: 0.0,
                    ripple_score: 0.0,
                    combined_score: cluster.centroid.norm(),
                    divergence,
                    resonance,
                });
            }
        }

        // Cache results
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(cache_key, results.clone());
        }

        results
    }

    /// Clear retrieval cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    /// Compute overlap score between query nodes and ripple results
    fn compute_overlap_score(&self, query_nodes: &[usize], ripple: &[(usize, f64)]) -> f64 {
        let mut total_score = 0.0;
        let mut count = 0;

        for &(node_id, ripple_score) in ripple {
            if query_nodes.contains(&node_id) {
                total_score += ripple_score;
                count += 1;
            }
        }

        if count > 0 {
            total_score / count as f64
        } else {
            0.0
        }
    }

    /// Generate query signature from node IDs
    fn generate_query_signature(&self, node_ids: &[usize]) -> Vec<Complex64> {
        let mut signature = Vec::new();
        let dim = 4096; // Standard holographic signature dimension

        for i in 0..dim {
            let mut sum = Complex64::new(0.0, 0.0);
            for &node_id in node_ids {
                if node_id < self.network.n && node_id < self.network.holographic_memory.len() {
                    let node_sig = &self.network.holographic_memory[node_id];
                    if i < node_sig.len() {
                        sum += node_sig[i];
                    }
                }
            }
            signature.push(sum);
        }

        // Normalize
        let norm: f64 = signature.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for sig in signature.iter_mut() {
                *sig /= norm;
            }
        }

        signature
    }

    /// Get chunk text for a node
    fn get_chunk_text(&self, node_id: usize) -> String {
        // In practice, this would retrieve the actual text chunk
        // For now, use label as placeholder
        if let Some(label) = self.network.labels.iter().find(|(_, &id)| id == node_id).map(|(l, _)| l) {
            label.clone()
        } else {
            format!("Node {}", node_id)
        }
    }

    /// Compute divergence and resonance for cognitive health gating
    fn compute_divergence_resonance(&self, node_ids: &[usize]) -> (f64, f64) {
        if node_ids.is_empty() {
            return (0.0, 0.0);
        }

        // Simplified divergence computation (linear independence)
        let mut divergence = 0.0;
        for i in 0..node_ids.len() {
            for j in (i + 1)..node_ids.len() {
                let id1 = node_ids[i];
                let id2 = node_ids[j];
                
                if id1 < self.network.n && id2 < self.network.n {
                    let pos1 = self.network.positions.amplitudes[id1];
                    let pos2 = self.network.positions.amplitudes[id2];
                    let diff = (pos1 - pos2).norm();
                    divergence += diff;
                }
            }
        }
        
        let n = node_ids.len() as f64;
        divergence /= if n > 1.0 { n * (n - 1.0) / 2.0 } else { 1.0 };

        // Simplified resonance computation (semantic coherence)
        let mut resonance = 0.0;
        if node_ids.len() > 1 {
            for i in 0..node_ids.len() - 1 {
                let id1 = node_ids[i];
                let id2 = node_ids[i + 1];
                
                if id1 < self.network.n && id2 < self.network.n {
                    let pos1 = self.network.positions.amplitudes[id1];
                    let pos2 = self.network.positions.amplitudes[id2];
                    let inner = (pos1 * pos2.conj()).norm();
                    resonance += inner;
                }
            }
            resonance /= (node_ids.len() - 1) as f64;
        }

        (divergence, resonance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieval_engine_creation() {
        let network = ResonanceNetwork::new(100);
        let chunker = ResonanceChunker::new(network.clone(), 0.5, 0.1, 5, 50);
        let engine = RetrievalEngine::new(network, chunker);
        
        assert!(engine.cache.read().unwrap().is_empty());
    }

    #[test]
    fn test_query_keystone() {
        let mut network = ResonanceNetwork::new(100);
        let a = network.add_concept(0.1);
        let b = network.add_concept(0.1);
        network.connect_semantic(a, b, 0.5);
        let chunker = ResonanceChunker::new(network.clone(), 0.5, 0.1, 5, 50);
        let mut engine = RetrievalEngine::new(network, chunker);
        
        let results = engine.query_keystone("test", 5);
        assert!(!results.is_empty() || engine.cache.read().unwrap().is_empty());
    }

    #[test]
    fn test_cache_functionality() {
        let network = ResonanceNetwork::new(100);
        let chunker = ResonanceChunker::new(network.clone(), 0.5, 0.1, 5, 50);
        let mut engine = RetrievalEngine::new(network, chunker);
        
        // Query once
        engine.query_keystone("test", 5);
        
        // Cache should be populated
        assert!(!engine.cache.read().unwrap().is_empty());
        
        // Clear cache
        engine.clear_cache();
        assert!(engine.cache.read().unwrap().is_empty());
    }
}
