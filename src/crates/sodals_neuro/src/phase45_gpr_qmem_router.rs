/// Phase 45: Hierarchical QMEM + Gaussian Phase Router (GPR) Integration
///
/// Adaptive layer loading strategy that uses GaussianPhaseRouter statistics to
/// guide QMEM loading decisions. Layers with high bypass rates are loaded to
/// mmap (disk-backed), while frequently-used layers are kept in RAM.
///
/// Mathematical Foundation:
/// - GPR tracks z_i = semantic_divergence × exp(-jacobian_norm × 0.1)
/// - Running EMA: μ_k = (1-α)μ_{k-1} + α·z_i, σ²_k = (1-α)σ²_{k-1} + α(z_i - μ)²
/// - Bypass rate: P(bypass|layer_l) = EMA of bypasses in layer_l
/// - Loading priority: RAM_priority[l] = 1 - P(bypass|layer_l)

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Per-layer routing statistics for GPR-informed QMEM loading
#[derive(Debug, Clone)]
pub struct LayerRoutingStats {
    /// Layer index (0 to num_layers-1)
    pub layer_idx: usize,
    /// EMA of bypass decisions (0.0 = always runs, 1.0 = always bypassed)
    pub bypass_rate_ema: f32,
    /// Number of forward passes through this layer
    pub forward_count: u32,
    /// EMA of z_i magnitudes (phase interference)
    pub z_i_ema: f32,
    /// EMA of semantic divergence
    pub semantic_divergence_ema: f32,
    /// EMA of Jacobian norm
    pub jacobian_norm_ema: f32,
    /// Loading priority for QMEM (0.0 = prefer mmap, 1.0 = prefer RAM)
    pub ram_loading_priority: f32,
    /// Estimated memory savings from bypassing (in bytes)
    pub estimated_memory_saved: u64,
}

impl LayerRoutingStats {
    pub fn new(layer_idx: usize) -> Self {
        Self {
            layer_idx,
            bypass_rate_ema: 0.0,
            forward_count: 0,
            z_i_ema: 0.0,
            semantic_divergence_ema: 0.0,
            jacobian_norm_ema: 0.0,
            ram_loading_priority: 1.0, // Start with high priority (prefer RAM)
            estimated_memory_saved: 0,
        }
    }

    /// Update routing statistics from phase signal information
    pub fn update(&mut self, bypass_decision: bool, z_i: f32, semantic_divergence: f32, jacobian_norm: f32, alpha: f32) {
        self.forward_count = self.forward_count.saturating_add(1);

        // Update bypass rate EMA
        let target = if bypass_decision { 1.0 } else { 0.0 };
        self.bypass_rate_ema = (1.0 - alpha) * self.bypass_rate_ema + alpha * target;

        // Update z_i EMA
        self.z_i_ema = (1.0 - alpha) * self.z_i_ema + alpha * z_i;

        // Update semantic divergence EMA
        self.semantic_divergence_ema = (1.0 - alpha) * self.semantic_divergence_ema + alpha * semantic_divergence;

        // Update Jacobian norm EMA
        self.jacobian_norm_ema = (1.0 - alpha) * self.jacobian_norm_ema + alpha * jacobian_norm;

        // Recompute RAM loading priority based on bypass rate
        // Layers with high bypass rates get lower priority for RAM loading
        self.ram_loading_priority = 1.0 - self.bypass_rate_ema;
    }

    /// Estimate memory that could be saved by using mmap for this layer
    /// Assumes tensor_size is the size in bytes of the layer weights
    pub fn update_memory_savings(&mut self, tensor_size_bytes: u64) {
        // Memory saved if we use mmap instead of RAM
        // Savings = tensor_size × (1 - bypass_rate_ema)
        // But we want to keep frequently-used layers in RAM
        let savings_ratio = self.bypass_rate_ema;
        self.estimated_memory_saved = (tensor_size_bytes as f32 * savings_ratio) as u64;
    }
}

/// Phase 45 GPR-QMEM Router: Coordinates routing and loading decisions
pub struct Phase45GprQmemRouter {
    /// Per-layer routing and loading statistics
    pub layer_stats: Arc<Mutex<HashMap<usize, LayerRoutingStats>>>,
    /// Total layers in model
    pub num_layers: usize,
    /// EMA smoothing factor (matches GaussianPhaseRouter alpha)
    pub alpha: f32,
    /// Threshold for considering a layer "frequently bypassed"
    pub bypass_threshold: f32,
}

impl Phase45GprQmemRouter {
    pub fn new(num_layers: usize) -> Self {
        let mut layer_stats = HashMap::new();
        for layer_idx in 0..num_layers {
            layer_stats.insert(layer_idx, LayerRoutingStats::new(layer_idx));
        }

        Self {
            layer_stats: Arc::new(Mutex::new(layer_stats)),
            num_layers,
            alpha: 0.2, // Match GaussianPhaseRouter's alpha
            bypass_threshold: 0.5, // Layers with bypass_rate > 0.5 are "hot-bypassed"
        }
    }

    /// Record a routing decision for a layer
    pub fn record_routing_decision(
        &self,
        layer_idx: usize,
        bypass_decision: bool,
        z_i: f32,
        semantic_divergence: f32,
        jacobian_norm: f32,
    ) -> Result<(), String> {
        if layer_idx >= self.num_layers {
            return Err(format!("Invalid layer_idx: {}", layer_idx));
        }

        let mut stats = self.layer_stats.lock().unwrap();
        if let Some(layer_stat) = stats.get_mut(&layer_idx) {
            layer_stat.update(bypass_decision, z_i, semantic_divergence, jacobian_norm, self.alpha);
            Ok(())
        } else {
            Err(format!("Layer {} not found in routing stats", layer_idx))
        }
    }

    /// Get QMEM loading priority for a layer (0.0 = mmap only, 1.0 = prefer RAM)
    pub fn get_loading_priority(&self, layer_idx: usize) -> Result<f32, String> {
        if layer_idx >= self.num_layers {
            return Err(format!("Invalid layer_idx: {}", layer_idx));
        }

        let stats = self.layer_stats.lock().unwrap();
        if let Some(layer_stat) = stats.get(&layer_idx) {
            Ok(layer_stat.ram_loading_priority)
        } else {
            Err(format!("Layer {} not found in routing stats", layer_idx))
        }
    }

    /// Get all layer loading priorities sorted by priority (highest first)
    pub fn get_sorted_loading_priorities(&self) -> Result<Vec<(usize, f32)>, String> {
        let stats = self.layer_stats.lock().unwrap();
        let mut priorities: Vec<(usize, f32)> = stats
            .iter()
            .map(|(idx, stat)| (*idx, stat.ram_loading_priority))
            .collect();

        priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(priorities)
    }

    /// Determine optimal QMEM loading strategy for available RAM
    ///
    /// Algorithm:
    /// 1. Sort layers by RAM loading priority (highest first)
    /// 2. Greedily assign layers to RAM up to max_ram_bytes
    /// 3. Remaining layers use mmap loading
    ///
    /// Returns Vec of (layer_idx, use_ram: bool) tuples
    pub fn compute_optimal_loading_strategy(
        &self,
        layer_sizes: &HashMap<usize, usize>,
        max_ram_bytes: usize,
    ) -> Result<Vec<(usize, bool)>, String> {
        let priorities = self.get_sorted_loading_priorities()?;
        let mut result = Vec::new();
        let mut used_ram = 0usize;

        for (layer_idx, _priority) in priorities {
            let layer_size = layer_sizes.get(&layer_idx).copied().unwrap_or(0);
            let use_ram = (used_ram + layer_size) <= max_ram_bytes;

            if use_ram {
                used_ram += layer_size;
            }

            result.push((layer_idx, use_ram));
        }

        Ok(result)
    }

    /// Get statistics snapshot for monitoring/debugging
    pub fn get_statistics_snapshot(&self) -> Result<Vec<LayerRoutingStats>, String> {
        let stats = self.layer_stats.lock().unwrap();
        let mut snapshot: Vec<LayerRoutingStats> = stats.values().cloned().collect();
        snapshot.sort_by_key(|s| s.layer_idx);
        Ok(snapshot)
    }

    /// Reset statistics for a specific layer
    pub fn reset_layer_stats(&self, layer_idx: usize) -> Result<(), String> {
        if layer_idx >= self.num_layers {
            return Err(format!("Invalid layer_idx: {}", layer_idx));
        }

        let mut stats = self.layer_stats.lock().unwrap();
        if let Some(layer_stat) = stats.get_mut(&layer_idx) {
            *layer_stat = LayerRoutingStats::new(layer_idx);
            Ok(())
        } else {
            Err(format!("Layer {} not found in routing stats", layer_idx))
        }
    }

    /// Warm-up period management: layers during warm-up get uniform priority
    /// Returns true if we're still in warm-up (first N passes per layer)
    pub fn is_in_warmup(&self, layer_idx: usize, warmup_threshold: u32) -> Result<bool, String> {
        let stats = self.layer_stats.lock().unwrap();
        if let Some(layer_stat) = stats.get(&layer_idx) {
            Ok(layer_stat.forward_count < warmup_threshold)
        } else {
            Err(format!("Layer {} not found in routing stats", layer_idx))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_routing_stats_creation() {
        let stats = LayerRoutingStats::new(0);
        assert_eq!(stats.layer_idx, 0);
        assert_eq!(stats.bypass_rate_ema, 0.0);
        assert_eq!(stats.forward_count, 0);
        assert_eq!(stats.ram_loading_priority, 1.0);
    }

    #[test]
    fn test_layer_routing_stats_update() {
        let mut stats = LayerRoutingStats::new(0);
        
        // Record bypass decision
        stats.update(true, 0.5, 0.3, 0.2, 0.2);
        assert!(stats.bypass_rate_ema > 0.0);
        assert_eq!(stats.forward_count, 1);
        
        // Verify priority decreases with bypass rate
        let priority_after_bypass = stats.ram_loading_priority;
        assert!(priority_after_bypass < 1.0);
    }

    #[test]
    fn test_phase45_router_creation() {
        let router = Phase45GprQmemRouter::new(32);
        assert_eq!(router.num_layers, 32);
        let stats = router.layer_stats.lock().unwrap();
        assert_eq!(stats.len(), 32);
    }

    #[test]
    fn test_record_routing_decision() {
        let router = Phase45GprQmemRouter::new(4);
        
        // Record several bypass decisions
        for i in 0..4 {
            let bypass = i % 2 == 0;
            let result = router.record_routing_decision(i as usize, bypass, 0.5, 0.3, 0.2);
            assert!(result.is_ok());
        }

        // Verify stats were updated
        let stats = router.layer_stats.lock().unwrap();
        assert_eq!(stats.get(&0).unwrap().forward_count, 1);
    }

    #[test]
    fn test_loading_priority_ordering() {
        let router = Phase45GprQmemRouter::new(3);

        // Layer 0: always runs (bypass_rate = 0.0) → high priority
        for _ in 0..10 {
            router.record_routing_decision(0, false, 0.5, 0.3, 0.2).unwrap();
        }

        // Layer 1: mixed (bypass_rate ≈ 0.5) → medium priority
        for i in 0..10 {
            let bypass = i % 2 == 0;
            router.record_routing_decision(1, bypass, 0.5, 0.3, 0.2).unwrap();
        }

        // Layer 2: always bypassed (bypass_rate ≈ 1.0) → low priority
        for _ in 0..10 {
            router.record_routing_decision(2, true, 0.5, 0.3, 0.2).unwrap();
        }

        // Check ordering
        let priorities = router.get_sorted_loading_priorities().unwrap();
        assert_eq!(priorities[0].0, 0); // Layer 0 has highest priority
        assert_eq!(priorities[2].0, 2); // Layer 2 has lowest priority
    }

    #[test]
    fn test_compute_optimal_loading_strategy() {
        let router = Phase45GprQmemRouter::new(4);

        // Record statistics to create priority differences
        for _ in 0..5 {
            router.record_routing_decision(0, false, 0.5, 0.3, 0.2).unwrap();
            router.record_routing_decision(1, true, 0.5, 0.3, 0.2).unwrap();
            router.record_routing_decision(2, true, 0.5, 0.3, 0.2).unwrap();
            router.record_routing_decision(3, false, 0.5, 0.3, 0.2).unwrap();
        }

        // Define layer sizes
        let mut layer_sizes = HashMap::new();
        layer_sizes.insert(0, 1024); // 1KB
        layer_sizes.insert(1, 1024);
        layer_sizes.insert(2, 1024);
        layer_sizes.insert(3, 1024);

        // Compute strategy with 3KB RAM budget
        let strategy = router.compute_optimal_loading_strategy(&layer_sizes, 3072).unwrap();

        // Should load high-priority layers (0, 3) to RAM
        let ram_layers: Vec<usize> = strategy
            .iter()
            .filter(|(_, use_ram)| *use_ram)
            .map(|(idx, _)| *idx)
            .collect();

        assert!(ram_layers.contains(&0));
        assert!(ram_layers.contains(&3));
    }

    #[test]
    fn test_memory_savings_estimation() {
        let mut stats = LayerRoutingStats::new(0);
        stats.update(true, 0.5, 0.3, 0.2, 0.2);
        stats.update(true, 0.5, 0.3, 0.2, 0.2);
        stats.update(false, 0.5, 0.3, 0.2, 0.2);

        // Update memory savings (assuming 1MB layer)
        stats.update_memory_savings(1024 * 1024);

        // Savings should be proportional to bypass_rate × layer_size
        assert!(stats.estimated_memory_saved > 0);
    }

    #[test]
    fn test_warmup_period() {
        let router = Phase45GprQmemRouter::new(2);

        // During warmup, is_in_warmup should return true
        assert!(router.is_in_warmup(0, 5).unwrap());

        // Record 3 routing decisions (warmup threshold = 5)
        for _ in 0..3 {
            router.record_routing_decision(0, false, 0.5, 0.3, 0.2).unwrap();
        }

        assert!(router.is_in_warmup(0, 5).unwrap());

        // After 5 passes, should exit warmup
        for _ in 0..2 {
            router.record_routing_decision(0, false, 0.5, 0.3, 0.2).unwrap();
        }

        assert!(!router.is_in_warmup(0, 5).unwrap());
    }

    #[test]
    fn test_reset_layer_stats() {
        let router = Phase45GprQmemRouter::new(2);

        // Record some statistics
        router.record_routing_decision(0, true, 0.5, 0.3, 0.2).unwrap();
        assert_eq!(router.layer_stats.lock().unwrap().get(&0).unwrap().forward_count, 1);

        // Reset
        router.reset_layer_stats(0).unwrap();
        assert_eq!(router.layer_stats.lock().unwrap().get(&0).unwrap().forward_count, 0);
        assert_eq!(router.layer_stats.lock().unwrap().get(&0).unwrap().bypass_rate_ema, 0.0);
    }
}
