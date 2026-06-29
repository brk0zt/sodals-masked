// src/engine/paca_gate.rs — PACA Physics-Augmented Cognitive Agent: Gating Layer
//
// Implements three critical PACA pipeline components:
// 1. ZScoreGate: Real-time rolling z-score for IoT stream outlier detection
// 2. NewtonRaphsonAligner: Cross-modal phase alignment for temporal desync
// 3. PacaMaskSet: Multi-mask fusion for HolographicAttention FFI Bridge
//
// All operations are O(N) and zero-allocation on hot path.

use candle_core::{Tensor, Result, Device};
use std::collections::HashMap;

// ============================================================================
// Z-SCORE OUTLIER GATING — Real-Time IoT Stream Anomaly Detection
// ============================================================================

/// Per-channel rolling z-score gate using Exponential Moving Average (EMA).
///
/// When a sensor value exceeds `threshold` standard deviations from the EMA mean,
/// the gate flags it as an outlier. This triggers physics-residual-weighted fusion
/// in the PACA pipeline instead of naive averaging.
///
/// # Design Decisions
/// - EMA over sliding window: O(1) per update, no buffer allocation
/// - Per-channel independence: each IoT channel has its own statistics
/// - Warmup period: first `warmup_samples` updates use simple mean accumulation
#[derive(Debug, Clone)]
pub struct ZScoreGate {
    channels: HashMap<String, ChannelStats>,
    threshold: f64,
    alpha: f64,
    warmup_samples: usize,
}

#[derive(Debug, Clone)]
struct ChannelStats {
    ema_mean: f64,
    ema_var: f64,
    alpha: f64,
    sample_count: usize,
    last_zscore: f64,
    is_outlier: bool,
}

/// Result of a z-score update for a single channel.
#[derive(Debug, Clone, Copy)]
pub struct ZScoreResult {
    pub channel_mean: f64,
    pub channel_std: f64,
    pub zscore: f64,
    pub is_outlier: bool,
}

impl ZScoreGate {
    /// Create a new gate with configurable threshold and EMA decay.
    ///
    /// # Arguments
    /// * `threshold` - Number of σ to flag outlier (typically 3.0)
    /// * `alpha` - EMA decay factor (0.01=slow adaptation, 0.1=fast)
    /// * `warmup_samples` - Minimum samples before z-score is meaningful
    pub fn new(threshold: f64, alpha: f64, warmup_samples: usize) -> Self {
        Self {
            channels: HashMap::new(),
            threshold,
            alpha,
            warmup_samples,
        }
    }

    /// Industrial default: 3σ threshold, α=0.05, 10-sample warmup.
    pub fn industrial_default() -> Self {
        Self::new(3.0, 0.05, 10)
    }

    /// Update a channel with a new sensor reading and return z-score result.
    pub fn update(&mut self, channel_name: &str, value: f64) -> ZScoreResult {
        let channel_alpha = self.alpha;
        let stats = self.channels.entry(channel_name.to_string())
            .or_insert_with(|| ChannelStats {
                ema_mean: value,
                ema_var: 0.0,
                alpha: channel_alpha,
                sample_count: 0,
                last_zscore: 0.0,
                is_outlier: false,
            });

        stats.sample_count += 1;

        if stats.sample_count <= self.warmup_samples {
            // Warmup: simple running mean
            let n = stats.sample_count as f64;
            stats.ema_mean = stats.ema_mean * ((n - 1.0) / n) + value / n;
            let diff = value - stats.ema_mean;
            stats.ema_var = stats.ema_var * ((n - 1.0) / n) + diff * diff / n;
            stats.last_zscore = 0.0;
            stats.is_outlier = false;
        } else {
            // EMA update
            let diff = value - stats.ema_mean;
            stats.ema_mean = stats.alpha * value + (1.0 - stats.alpha) * stats.ema_mean;
            stats.ema_var = stats.alpha * diff * diff + (1.0 - stats.alpha) * stats.ema_var;

            let std_dev = stats.ema_var.sqrt().max(1e-8);
            stats.last_zscore = diff / std_dev;
            stats.is_outlier = stats.last_zscore.abs() > self.threshold;
        }

        ZScoreResult {
            channel_mean: stats.ema_mean,
            channel_std: stats.ema_var.sqrt(),
            zscore: stats.last_zscore,
            is_outlier: stats.is_outlier,
        }
    }

    /// Batch update all channels from an OEDB realtime_json stream row.
    /// Returns a map of channel_name → ZScoreResult.
    pub fn update_batch(&mut self, readings: &[(String, f64)]) -> HashMap<String, ZScoreResult> {
        readings.iter()
            .map(|(name, val)| (name.clone(), self.update(name, *val)))
            .collect()
    }

    /// Get the current outlier status for all channels.
    pub fn outlier_channels(&self) -> Vec<&str> {
        self.channels.iter()
            .filter(|(_, s)| s.is_outlier)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Generate a gating mask tensor: 0.0 for outlier channels, 1.0 for normal.
    /// Shape: [1, 1, num_channels]
    pub fn to_gating_tensor(&self, channel_order: &[&str], device: &Device) -> Result<Tensor> {
        let mask_vals: Vec<f32> = channel_order.iter()
            .map(|name| {
                self.channels.get(*name)
                    .map(|s| if s.is_outlier { 0.0_f32 } else { 1.0_f32 })
                    .unwrap_or(1.0)
            })
            .collect();
        Tensor::from_vec(mask_vals, (1, 1, channel_order.len()), device)
    }

    /// Reset all channel statistics (call between OEDB scenarios for isolation).
    pub fn reset(&mut self) {
        self.channels.clear();
    }
}

// ============================================================================
// NEWTON-RAPHSON PHASE ALIGNMENT — Cross-Modal Temporal Desync Resolution
// ============================================================================

/// Newton-Raphson iterative phase alignment for cross-modal fusion.
///
/// When thermal image timestamp, IoT stream timestamp, and PDF creation date
/// are desynchronized, raw fusion produces phase interference artifacts.
/// This aligner iteratively converges modality phases to minimize cross-modal
/// phase residual using Newton's method.
///
/// # Mathematical Model
/// Given modality phases θ_1, θ_2, ..., θ_M:
/// - Objective: minimize Σ_i (θ_i - θ̄)² where θ̄ = mean(θ)
/// - Newton step: θ_i ← θ_i - f(θ_i)/f'(θ_i)
/// - f(θ_i) = θ_i - θ̄ (residual)
/// - f'(θ_i) = 1 - 1/M (Jacobian)
pub struct NewtonRaphsonAligner {
    max_iterations: usize,
    convergence_threshold: f64,
}

impl NewtonRaphsonAligner {
    pub fn new(max_iterations: usize, convergence_threshold: f64) -> Self {
        Self { max_iterations, convergence_threshold }
    }

    /// Default: 5 iterations, 1e-6 convergence threshold.
    pub fn default_industrial() -> Self {
        Self::new(20, 1e-6)
    }

    /// Align modality phases using Newton-Raphson iteration.
    ///
    /// # Arguments
    /// * `modality_waves` - SemanticWave per modality (vision, IoT, text)
    ///
    /// # Returns
    /// * Phase shifts applied to each modality (radians)
    /// * Number of iterations used
    pub fn align_phases(&self, phases: &[f64]) -> (Vec<f64>, usize) {
        if phases.is_empty() {
            return (vec![], 0);
        }

        let m = phases.len() as f64;
        let mut aligned = phases.to_vec();
        let jacobian_inv = m / (m - 1.0); // 1 / (1 - 1/M)

        let mut iterations = 0;
        for _ in 0..self.max_iterations {
            let mean_phase: f64 = aligned.iter().sum::<f64>() / m;
            let mut max_residual = 0.0_f64;

            for p in aligned.iter_mut() {
                let residual = *p - mean_phase;
                max_residual = max_residual.max(residual.abs());
                *p -= residual / jacobian_inv;
            }

            iterations += 1;
            if max_residual < self.convergence_threshold {
                break;
            }
        }

        // Compute applied shifts
        let shifts: Vec<f64> = phases.iter().zip(aligned.iter())
            .map(|(orig, aligned)| aligned - orig)
            .collect();

        (shifts, iterations)
    }

    /// Align SemanticWave phases across modalities and apply shifts in-place.
    ///
    /// Extracts mean phase from each wave, aligns them, then applies
    /// the phase rotation to each wave.
    pub fn align_waves(&self, waves: &mut [crate::fast_math::SemanticWave]) -> usize {
        if waves.len() < 2 {
            return 0;
        }

        // Extract mean phase per modality
        let mean_phases: Vec<f64> = waves.iter()
            .map(|w| {
                let phases = w.phases();
                phases.iter().sum::<f64>() / phases.len() as f64
            })
            .collect();

        let (shifts, iterations) = self.align_phases(&mean_phases);

        // Apply phase rotation to each wave
        for (wave, &shift) in waves.iter_mut().zip(shifts.iter()) {
            if shift.abs() > 1e-10 {
                wave.rotate_phase_mut(shift);
            }
        }

        iterations
    }
}

// ============================================================================
// PACA MASK SET — Multi-Mask Fusion for HolographicAttention FFI Bridge
// ============================================================================

/// Collection of attention masks for PACA's multi-constraint fusion.
///
/// Each mask is an additive bias applied to attention logits in
/// `HolographicAttention::forward()`. Negative values suppress attention,
/// positive values amplify it.
///
/// # Mask Types
/// 1. **Physics residual**: Penalizes cross-modal inconsistencies
///    (e.g., thermal hotspot without corresponding vibration spike)
/// 2. **Z-Score gating**: Suppresses attention to outlier sensor channels
/// 3. **Compliance forbidden**: Blocks attention to forbidden token regions
///    (PII, credentials, override commands)
/// 4. **Phase alignment**: Corrects temporal desync between modalities
pub struct PacaMaskSet {
    pub physics_residual: Option<Tensor>,
    pub zscore_gating: Option<Tensor>,
    pub compliance_forbidden: Option<Tensor>,
    pub phase_alignment: Option<Tensor>,
}

impl PacaMaskSet {
    pub fn empty() -> Self {
        Self {
            physics_residual: None,
            zscore_gating: None,
            compliance_forbidden: None,
            phase_alignment: None,
        }
    }

    /// Merge all active masks into a single attention mask tensor via addition.
    ///
    /// The merged mask is applied as: logits = logits + merged_mask
    /// in HolographicAttention::forward().
    pub fn merge(&self, _device: &Device) -> Result<Option<Tensor>> {
        let masks: Vec<&Tensor> = [
            &self.physics_residual,
            &self.zscore_gating,
            &self.compliance_forbidden,
            &self.phase_alignment,
        ]
        .iter()
        .filter_map(|m| m.as_ref())
        .collect();

        if masks.is_empty() {
            return Ok(None);
        }

        let mut merged = masks[0].clone();
        for m in &masks[1..] {
            merged = merged.broadcast_add(m)?;
        }
        Ok(Some(merged))
    }

    /// Create a physics residual mask from cross-modal consistency check.
    ///
    /// Compares thermal evidence against IoT stream evidence.
    /// Inconsistency produces a negative mask (suppresses fusion weight).
    ///
    /// # Arguments
    /// * `thermal_magnitude` - Thermal hotspot intensity (normalized 0-1)
    /// * `stream_zscore` - Z-score of correlated IoT channel
    /// * `seq_len` - Sequence length for mask shape
    pub fn build_physics_residual(
        thermal_magnitude: f64,
        stream_zscore: f64,
        seq_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        // If thermal says hot but stream says normal, penalize fusion
        let consistency = if thermal_magnitude > 0.7 && stream_zscore.abs() < 1.5 {
            -5.0_f32 // Strong inconsistency penalty
        } else if thermal_magnitude < 0.3 && stream_zscore.abs() > 3.0 {
            -3.0_f32 // Moderate: stream anomaly without thermal evidence
        } else {
            0.0_f32 // Consistent
        };

        Tensor::full(consistency, (1, 1, seq_len, seq_len), device)
    }

    /// Create a compliance mask that blocks attention to forbidden token positions.
    ///
    /// # Arguments
    /// * `forbidden_positions` - Token indices that contain PII/credentials
    /// * `seq_len` - Total sequence length
    pub fn build_compliance_mask(
        forbidden_positions: &[usize],
        seq_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mut mask_data = vec![0.0_f32; seq_len * seq_len];
        for &pos in forbidden_positions {
            if pos < seq_len {
                // Block all attention TO this position (column mask)
                for row in 0..seq_len {
                    mask_data[row * seq_len + pos] = -1e9;
                }
            }
        }
        Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore_gate_normal_operation() {
        let mut gate = ZScoreGate::industrial_default();

        // Warmup with normal values
        for i in 0..20 {
            let val = 65.0 + (i as f64) * 0.1; // Slight drift
            let result = gate.update("bearing_temp", val);
            if i < 10 {
                assert!(!result.is_outlier, "Should not flag during warmup");
            }
        }

        // Inject anomaly: 87°C (22°C above baseline ~66°C)
        let anomaly = gate.update("bearing_temp", 87.0);
        assert!(anomaly.is_outlier, "87°C should be flagged as outlier (z={:.2})", anomaly.zscore);
        assert!(anomaly.zscore > 3.0, "Z-score should exceed 3σ");
    }

    #[test]
    fn test_zscore_gate_multi_channel() {
        let mut gate = ZScoreGate::industrial_default();

        // Warmup both channels with tiny noise to establish non-zero baseline variance
        for i in 0..15 {
            let noise = if i % 2 == 0 { 0.05 } else { -0.05 };
            gate.update("temp", 65.0 + noise);
            gate.update("vibration", 1.8 + noise);
        }

        // Only temp anomaly
        gate.update("temp", 87.0);
        gate.update("vibration", 1.9);

        let outliers = gate.outlier_channels();
        assert!(outliers.contains(&"temp"), "Temp should be outlier");
        assert!(!outliers.contains(&"vibration"), "Vibration should be normal");
    }

    #[test]
    fn test_zscore_gate_reset_isolation() {
        let mut gate = ZScoreGate::industrial_default();

        for _ in 0..15 {
            gate.update("temp", 65.0);
        }
        gate.update("temp", 87.0);
        assert!(!gate.outlier_channels().is_empty());

        gate.reset();
        assert!(gate.outlier_channels().is_empty(), "Reset should clear all state");
    }

    #[test]
    fn test_newton_raphson_alignment() {
        let aligner = NewtonRaphsonAligner::default_industrial();

        // Three modalities with phase desync
        let phases = vec![0.1, 0.5, 0.2]; // Desynchronized
        let (shifts, iterations) = aligner.align_phases(&phases);

        assert_eq!(shifts.len(), 3);
        assert!(iterations <= 15, "Should converge quickly");

        // After alignment, all phases should be near the mean
        let aligned: Vec<f64> = phases.iter().zip(shifts.iter())
            .map(|(p, s)| p + s)
            .collect();
        let mean = aligned.iter().sum::<f64>() / aligned.len() as f64;
        for p in &aligned {
            assert!((p - mean).abs() < 1e-4, "Phase {:.6} not aligned to mean {:.6}", p, mean);
        }
    }

    #[test]
    fn test_paca_mask_merge() {
        let device = Device::Cpu;
        let seq_len = 4;

        let physics = PacaMaskSet::build_physics_residual(0.9, 0.5, seq_len, &device).unwrap();
        let compliance = PacaMaskSet::build_compliance_mask(&[2], seq_len, &device).unwrap();

        let mask_set = PacaMaskSet {
            physics_residual: Some(physics),
            zscore_gating: None,
            compliance_forbidden: Some(compliance),
            phase_alignment: None,
        };

        let merged = mask_set.merge(&device).unwrap();
        assert!(merged.is_some(), "Should produce merged mask");

        let m = merged.unwrap();
        assert_eq!(m.dims(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_compliance_mask_blocks_forbidden() {
        let device = Device::Cpu;
        let mask = PacaMaskSet::build_compliance_mask(&[1, 3], 5, &device).unwrap();
        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Position 1 column should be -1e9
        assert!(data[0 * 5 + 1] < -1e8, "Column 1 should be blocked");
        assert!(data[2 * 5 + 3] < -1e8, "Column 3 should be blocked");
        // Normal positions should be 0
        assert_eq!(data[0 * 5 + 0], 0.0, "Column 0 should be open");
    }
}
