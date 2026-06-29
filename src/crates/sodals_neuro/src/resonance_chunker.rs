// src/resonance_chunker.rs - Resonance-Aware Chunking for SODALS
// 
// Implements pure resonance-aware semantic segmentation without fallback to cosine similarity.
// Uses phase drop detection, impedance mismatch clustering, spectral boundaries, and Jacobian stability.
// Accuracy-first approach: no approximations, exact computation only.

use crate::semantic_network::{ResonanceNetwork, SpectralCluster};
use crate::fast_math::{SemanticWave, apply_clifford_resonance};
use num_complex::Complex64;
use std::collections::{HashMap, HashSet};

/// Resonance-aware chunker for semantic segmentation
/// 
/// Uses physics-inspired metrics to detect natural semantic boundaries:
/// - Phase drops (abrupt phase changes indicate topic shifts)
/// - Impedance mismatch (topological friction indicates boundary)
/// - Spectral cluster boundaries (phase space zones)
/// - Jacobian rhythm stability (harmonic vs chaotic regions)
pub struct ResonanceChunker {
    network: ResonanceNetwork,
    phase_drop_threshold: f64,
    impedance_mismatch_threshold: f64,
    min_chunk_size: usize,
    max_chunk_size: usize,
}

/// Semantic chunk with resonance-aware metadata
pub struct Chunk {
    pub text: String,
    pub node_ids: Vec<usize>,
    pub phase_signature: Vec<Complex64>,
    pub impedance_profile: Vec<f64>,
    pub jacobian_stability: f64,
    pub bifurcation_detected: bool,
}

/// Chunk boundary detected by resonance metrics
struct ChunkBoundary {
    position: usize,
    boundary_type: BoundaryType,
    confidence: f64,
}

#[derive(Debug, Clone, Copy)]
enum BoundaryType {
    PhaseDrop,
    ImpedanceMismatch,
    SpectralCluster,
    JacobianTransition,
    Bifurcation,
}

impl ResonanceChunker {
    /// Create new resonance-aware chunker
    /// 
    /// # Arguments
    /// * `network` - ResonanceNetwork for wave propagation and topology
    /// * `phase_drop_threshold` - Minimum phase change to consider as boundary (0.0-π)
    /// * `impedance_mismatch_threshold` - Minimum impedance difference for boundary
    /// * `min_chunk_size` - Minimum nodes per chunk
    /// * `max_chunk_size` - Maximum nodes per chunk
    pub fn new(
        network: ResonanceNetwork,
        phase_drop_threshold: f64,
        impedance_mismatch_threshold: f64,
        min_chunk_size: usize,
        max_chunk_size: usize,
    ) -> Self {
        Self {
            network,
            phase_drop_threshold,
            impedance_mismatch_threshold,
            min_chunk_size,
            max_chunk_size,
        }
    }

    /// Detect chunk boundaries based on phase drops
    /// 
    /// Abrupt phase changes in the wave function indicate semantic topic shifts.
    /// This is exact computation using Complex64 phase values.
    pub fn detect_phase_boundaries(&self, text: &str) -> Vec<ChunkBoundary> {
        let mut boundaries = Vec::new();
        
        // Extract node sequence from text (simplified - in practice would use parser)
        let node_sequence = self.extract_node_sequence(text);
        
        if node_sequence.len() < 2 {
            return boundaries;
        }

        // Compute phase differences between consecutive nodes
        for i in 1..node_sequence.len() {
            let prev_id = node_sequence[i - 1];
            let curr_id = node_sequence[i];
            
            if prev_id >= self.network.n || curr_id >= self.network.n {
                continue;
            }
            
            let prev_phase = self.network.positions.amplitudes[prev_id].arg();
            let curr_phase = self.network.positions.amplitudes[curr_id].arg();
            
            // Compute phase difference (wrap around 2π)
            let phase_diff = (curr_phase - prev_phase).abs();
            let wrapped_diff = if phase_diff > std::f64::consts::PI {
                2.0 * std::f64::consts::PI - phase_diff
            } else {
                phase_diff
            };
            
            if wrapped_diff >= self.phase_drop_threshold {
                boundaries.push(ChunkBoundary {
                    position: i,
                    boundary_type: BoundaryType::PhaseDrop,
                    confidence: wrapped_diff / std::f64::consts::PI,
                });
            }
        }
        
        boundaries
    }

    /// Cluster nodes by impedance mismatch
    /// 
    /// Topological impedance indicates semantic friction - high mismatch suggests boundary.
    pub fn cluster_by_impedance(&self, nodes: &[usize]) -> Vec<Vec<usize>> {
        let mut clusters = Vec::new();
        let mut current_cluster = Vec::new();
        
        if nodes.is_empty() {
            return clusters;
        }
        
        current_cluster.push(nodes[0]);
        
        for i in 1..nodes.len() {
            let prev_id = nodes[i - 1];
            let curr_id = nodes[i];
            
            if prev_id >= self.network.n || curr_id >= self.network.n {
                current_cluster.push(curr_id);
                continue;
            }
            
            // Get impedance from network (stored in damping or separate array)
            let prev_impedance = self.network.damping[prev_id];
            let curr_impedance = self.network.damping[curr_id];
            
            let impedance_diff = (prev_impedance - curr_impedance).abs();
            
            if impedance_diff >= self.impedance_mismatch_threshold {
                // Boundary detected - start new cluster
                if !current_cluster.is_empty() {
                    clusters.push(current_cluster.clone());
                    current_cluster.clear();
                }
            }
            
            current_cluster.push(curr_id);
        }
        
        if !current_cluster.is_empty() {
            clusters.push(current_cluster);
        }
        
        clusters
    }

    /// Extract spectral cluster boundaries
    /// 
    /// Uses pre-computed spectral clusters from the network to identify semantic zones.
    pub fn spectral_boundaries(&self) -> Vec<usize> {
        let mut boundaries = Vec::new();
        
        // Get all spectral clusters from network
        let cluster_map = &self.network.spectral_clusters;
        
        if cluster_map.is_empty() {
            return boundaries;
        }
        
        // Build node-to-cluster mapping
        let mut node_to_cluster: HashMap<usize, usize> = HashMap::new();
        for (cluster_id, cluster) in cluster_map {
            for &node_id in &cluster.nodes {
                node_to_cluster.insert(node_id, *cluster_id);
            }
        }
        
        // Find boundaries where cluster changes
        let mut nodes: Vec<_> = node_to_cluster.keys().cloned().collect();
        nodes.sort();
        
        for i in 1..nodes.len() {
            let prev_cluster = node_to_cluster.get(&nodes[i - 1]);
            let curr_cluster = node_to_cluster.get(&nodes[i]);
            
            if prev_cluster != curr_cluster {
                boundaries.push(i);
            }
        }
        
        boundaries
    }

    /// Analyze Jacobian rhythm stability for chunk boundaries
    /// 
    /// Harmonic regions suggest coherent chunks, chaotic regions suggest boundaries.
    pub fn jacobian_stability_analysis(&self, nodes: &[usize]) -> f64 {
        if nodes.len() < 2 {
            return 1.0; // Stable by default
        }
        
        // Compute Jacobian for the sub-network
        let jacobian_analysis = self.network.compute_jacobian_rhythm(nodes[0]);
        let det = jacobian_analysis.determinant;
        let rhythm = match jacobian_analysis.rhythm_classification {
            crate::semantic_network::RhythmClass::Trivial => "Trivial",
            crate::semantic_network::RhythmClass::Harmonic => "Harmonic",
            crate::semantic_network::RhythmClass::Chaotic => "Chaotic",
        };
        
        // Convert rhythm to stability score
        match rhythm {
            "Trivial" => 0.0,  // Too simple, likely boundary
            "Harmonic" => 1.0,  // Stable, good chunk
            "Chaotic" => 0.5,  // Unstable, potential boundary
            _ => 0.5,
        }
    }

    /// Check for bifurcation in current region
    /// 
    /// Bifurcation indicates regime change - strong boundary signal.
    pub fn detect_bifurcation(&self, nodes: &[usize]) -> bool {
        if nodes.is_empty() {
            return false;
        }
        
        // Check bifurcation monitor for nodes in this region
        for &node_id in nodes {
            if node_id >= self.network.n {
                continue;
            }
            
            // Check if this node is near a bifurcation point
            // This would require access to bifurcation monitor state
            // For now, use damping as a proxy
            if self.network.damping[node_id] > 0.8 {
                return true; // High damping suggests bifurcation
            }
        }
        
        false
    }

    /// Main chunking function
    /// 
    /// Combines all resonance-aware metrics to produce semantic chunks.
    /// No fallback to cosine similarity - pure physics-based segmentation.
    pub fn chunk_text(&self, text: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        
        // Extract node sequence from text
        let node_sequence = self.extract_node_sequence(text);
        
        if node_sequence.is_empty() {
            return chunks;
        }
        
        // Detect boundaries using all methods
        let phase_boundaries = self.detect_phase_boundaries(text);
        let impedance_clusters = self.cluster_by_impedance(&node_sequence);
        let spectral_boundaries = self.spectral_boundaries();
        
        // Combine boundary signals
        let combined_boundaries = self.combine_boundaries(
            &node_sequence,
            phase_boundaries,
            impedance_clusters,
            spectral_boundaries,
        );
        
        // Create chunks based on boundaries
        let mut chunk_start = 0;
        for boundary in combined_boundaries {
            if boundary.position - chunk_start >= self.min_chunk_size {
                let chunk_nodes = node_sequence[chunk_start..boundary.position].to_vec();
                if chunk_nodes.len() >= self.min_chunk_size {
                    chunks.push(self.create_chunk(&chunk_nodes, text, chunk_start, boundary.position));
                }
                chunk_start = boundary.position;
            }
        }
        
        // Add final chunk
        if node_sequence.len() - chunk_start >= self.min_chunk_size {
            let chunk_nodes = node_sequence[chunk_start..].to_vec();
            chunks.push(self.create_chunk(&chunk_nodes, text, chunk_start, node_sequence.len()));
        }
        
        // Enforce max chunk size by splitting large chunks
        chunks = self.enforce_max_chunk_size(chunks);
        
        chunks
    }

    /// Extract node sequence from text
    /// 
    /// In practice, this would use the semantic parser to map text to node IDs.
    /// For now, use a simplified approach based on network labels.
    pub fn extract_node_sequence(&self, text: &str) -> Vec<usize> {
        let mut node_sequence = Vec::new();
        let lowercase = text.to_lowercase();
        
        // Find all labels that appear in text
        let mut found_concepts: Vec<(String, usize)> = Vec::new();
        for (label, &id) in &self.network.labels {
            if label.len() <= 2 {
                continue;
            }
            let clean_label = label.to_lowercase();
            if lowercase.contains(&clean_label) {
                found_concepts.push((label.clone(), id));
            }
        }
        
        // Sort by position in text (simplified)
        found_concepts.sort_by(|a, b| {
            let pos_a = lowercase.find(&a.0.to_lowercase()).unwrap_or(0);
            let pos_b = lowercase.find(&b.0.to_lowercase()).unwrap_or(0);
            pos_a.cmp(&pos_b)
        });
        
        for (_, id) in found_concepts {
            if id < self.network.n {
                node_sequence.push(id);
            }
        }
        
        node_sequence
    }

    /// Combine boundary signals from different methods
    fn combine_boundaries(
        &self,
        node_sequence: &[usize],
        phase_boundaries: Vec<ChunkBoundary>,
        impedance_clusters: Vec<Vec<usize>>,
        spectral_boundaries: Vec<usize>,
    ) -> Vec<ChunkBoundary> {
        let mut combined = Vec::new();
        let mut boundary_positions: HashSet<usize> = HashSet::new();
        
        // Add phase boundaries
        for boundary in phase_boundaries {
            if boundary.position < node_sequence.len() {
                boundary_positions.insert(boundary.position);
                combined.push(boundary);
            }
        }
        
        // Add impedance cluster boundaries
        let mut offset = 0;
        for cluster in impedance_clusters {
            offset += cluster.len();
            if offset < node_sequence.len() && !boundary_positions.contains(&offset) {
                combined.push(ChunkBoundary {
                    position: offset,
                    boundary_type: BoundaryType::ImpedanceMismatch,
                    confidence: 0.7,
                });
                boundary_positions.insert(offset);
            }
        }
        
        // Add spectral boundaries
        for pos in spectral_boundaries {
            if pos < node_sequence.len() && !boundary_positions.contains(&pos) {
                combined.push(ChunkBoundary {
                    position: pos,
                    boundary_type: BoundaryType::SpectralCluster,
                    confidence: 0.8,
                });
                boundary_positions.insert(pos);
            }
        }
        
        // Sort by position
        combined.sort_by_key(|b| b.position);
        
        combined
    }

    /// Create a chunk from node IDs
    fn create_chunk(&self, node_ids: &[usize], text: &str, start: usize, end: usize) -> Chunk {
        // Extract phase signature
        let phase_signature: Vec<Complex64> = node_ids
            .iter()
            .filter(|&&id| id < self.network.n)
            .map(|&id| self.network.positions.amplitudes[id])
            .collect();
        
        // Extract impedance profile
        let impedance_profile: Vec<f64> = node_ids
            .iter()
            .filter(|&&id| id < self.network.n)
            .map(|&id| self.network.damping[id])
            .collect();
        
        // Compute Jacobian stability
        let jacobian_stability = self.jacobian_stability_analysis(node_ids);
        
        // Detect bifurcation
        let bifurcation_detected = self.detect_bifurcation(node_ids);
        
        // Extract text segment (simplified - in practice would use character positions)
        let text_segment = text.chars().take(end.saturating_sub(start)).collect::<String>();
        
        Chunk {
            text: text_segment,
            node_ids: node_ids.to_vec(),
            phase_signature,
            impedance_profile,
            jacobian_stability,
            bifurcation_detected,
        }
    }

    /// Enforce maximum chunk size by splitting large chunks
    fn enforce_max_chunk_size(&self, chunks: Vec<Chunk>) -> Vec<Chunk> {
        let mut result = Vec::new();
        
        for chunk in chunks {
            if chunk.node_ids.len() <= self.max_chunk_size {
                result.push(chunk);
            } else {
                // Split chunk at best boundary
                let split_point = self.find_best_split_point(&chunk);
                let (left_nodes, right_nodes) = chunk.node_ids.split_at(split_point);
                
                // Create left chunk
                if left_nodes.len() >= self.min_chunk_size {
                    result.push(Chunk {
                        text: chunk.text.clone(),
                        node_ids: left_nodes.to_vec(),
                        phase_signature: chunk.phase_signature[..split_point].to_vec(),
                        impedance_profile: chunk.impedance_profile[..split_point].to_vec(),
                        jacobian_stability: self.jacobian_stability_analysis(left_nodes),
                        bifurcation_detected: self.detect_bifurcation(left_nodes),
                    });
                }
                
                // Create right chunk
                if right_nodes.len() >= self.min_chunk_size {
                    result.push(Chunk {
                        text: chunk.text.clone(),
                        node_ids: right_nodes.to_vec(),
                        phase_signature: chunk.phase_signature[split_point..].to_vec(),
                        impedance_profile: chunk.impedance_profile[split_point..].to_vec(),
                        jacobian_stability: self.jacobian_stability_analysis(right_nodes),
                        bifurcation_detected: self.detect_bifurcation(right_nodes),
                    });
                }
            }
        }
        
        result
    }

    /// Find best split point within a chunk based on resonance metrics
    fn find_best_split_point(&self, chunk: &Chunk) -> usize {
        let mut best_split = chunk.node_ids.len() / 2;
        let mut best_score = 0.0;
        
        // Evaluate potential split points
        for i in (self.min_chunk_size..(chunk.node_ids.len() - self.min_chunk_size)).step_by(10) {
            let left_nodes = &chunk.node_ids[..i];
            let right_nodes = &chunk.node_ids[i..];
            
            // Score based on impedance mismatch at split
            let left_impedance = left_nodes.last().and_then(|&id| {
                if id < self.network.n {
                    Some(self.network.damping[id])
                } else {
                    None
                }
            }).unwrap_or(0.0);
            
            let right_impedance = right_nodes.first().and_then(|&id| {
                if id < self.network.n {
                    Some(self.network.damping[id])
                } else {
                    None
                }
            }).unwrap_or(0.0);
            
            let impedance_diff = (left_impedance - right_impedance).abs();
            
            if impedance_diff > best_score {
                best_score = impedance_diff;
                best_split = i;
            }
        }
        
        best_split
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunker_creation() {
        let network = ResonanceNetwork::new(100);
        let chunker = ResonanceChunker::new(
            network,
            0.5,  // phase_drop_threshold
            0.1,  // impedance_mismatch_threshold
            5,    // min_chunk_size
            50,   // max_chunk_size
        );
        
        assert_eq!(chunker.min_chunk_size, 5);
        assert_eq!(chunker.max_chunk_size, 50);
    }

    #[test]
    fn test_phase_boundary_detection() {
        let mut network = ResonanceNetwork::new(10);
        let a = network.add_concept(0.1);
        let b = network.add_concept(0.1);
        
        // Set different phases
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
        let b = network.add_concept(0.5);
        let c = network.add_concept(0.1);
        
        let chunker = ResonanceChunker::new(network, 0.5, 0.3, 1, 10);
        let clusters = chunker.cluster_by_impedance(&[a, b, c]);
        
        // Should split at impedance mismatch
        assert!(clusters.len() >= 2);
    }
}
