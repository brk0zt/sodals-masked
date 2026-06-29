// src/engine/semg_phoneme_hunter.rs
// ============================================================================
// sEMG PHONEME HUNTER — Subvocal Speech DSP Pipeline
// ============================================================================
//
// This module implements a real-time Digital Signal Processing pipeline for
// extracting phonetic intent from surface Electromyography (sEMG) signals
// captured from jaw/throat muscles during subvocal speech.
//
// Unlike EEG (which captures cortical field potentials at ~µV scale), sEMG
// captures Motor Unit Action Potentials (MUAPs) at ~mV scale with frequency
// content from 20-500Hz. The phonetic signal is encoded in the *envelope*
// of the muscle activation, not the raw waveform.
//
// Pipeline:
//   Raw sEMG → High-Pass IIR (20Hz) → Rectification → RMS Envelope (5Hz LPF)
//            → Multi-Channel Latent Projection → Ghost Token [1, 1, hidden_size]
//
// All filters are implemented as 1st-order IIR (Infinite Impulse Response)
// with zero heap allocation in the streaming path.

use anyhow::{Result, bail, Context};
use candle_core::{Tensor, DType, Device};
use std::path::Path;

// ============================================================================
// HARDWARE CONSTANTS — Nieto ds003626 EXG Specification
// ============================================================================

/// Number of EXG channels in the Nieto dataset (jaw/throat sEMG + EOG)
pub const SEMG_CHANNEL_COUNT: usize = 8;
/// Time steps per epoch at 256Hz sampling
pub const SEMG_TIME_STEPS: usize = 256;
/// Stage 3: Fixed Attention Window for adaptive peak cropping (250ms)
pub const SEMG_ATTENTION_WINDOW: usize = 64;
/// Cropped feature dimensionality: 8 channels × 64 timesteps = 512
pub const SEMG_FEATURE_DIM: usize = SEMG_CHANNEL_COUNT * SEMG_ATTENTION_WINDOW;

// ============================================================================
// 1. IIR HIGH-PASS FILTER — Baseline Wander Removal
// ============================================================================
//
// 1st-order IIR High-Pass Filter (Butterworth approximation):
//
//   y[n] = α · (y[n-1] + x[n] - x[n-1])
//
// where α = RC / (RC + dt), RC = 1 / (2π·fc), dt = 1/fs
//
// This removes DC offset, slow respiratory artifacts, and EEG/EOG crosstalk
// that bleeds into the EXG channels.

/// 1st-order IIR High-Pass Filter state (per-channel).
#[derive(Debug, Clone)]
pub struct HighPassFilter {
    /// Filter coefficient α = RC / (RC + dt)
    alpha: f64,
    /// Previous input sample x[n-1]
    x_prev: f64,
    /// Previous output sample y[n-1]
    y_prev: f64,
}

impl HighPassFilter {
    /// Create a new high-pass filter.
    ///
    /// * `cutoff_hz` — -3dB cutoff frequency (typically 20Hz for sEMG)
    /// * `sample_rate` — Sampling frequency in Hz (typically 256Hz)
    pub fn new(cutoff_hz: f64, sample_rate: f64) -> Self {
        let dt = 1.0 / sample_rate;
        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz);
        let alpha = rc / (rc + dt);

        Self {
            alpha,
            x_prev: 0.0,
            y_prev: 0.0,
        }
    }

    /// Reset filter state (call between non-contiguous epochs).
    #[inline]
    pub fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }

    /// Process a single sample through the high-pass filter.
    ///
    /// y[n] = α · (y[n-1] + x[n] - x[n-1])
    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.alpha * (self.y_prev + x - self.x_prev);
        self.x_prev = x;
        self.y_prev = y;
        y
    }

    /// Process an entire buffer in-place. Zero allocation.
    #[inline]
    pub fn process_buffer(&mut self, buffer: &mut [f64]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}

// ============================================================================
// 2. RMS ENVELOPE EXTRACTOR — Muscle Activation Envelope
// ============================================================================
//
// After high-pass filtering and rectification, the "phonetic intent" is
// encoded in the *amplitude envelope* of the sEMG burst.
//
// We extract this using a sliding-window RMS (Root Mean Square):
//
//   RMS[n] = sqrt( (1/W) · Σ_{k=0}^{W-1} x[n-k]² )
//
// For real-time streaming, we use a circular buffer with incremental
// sum-of-squares update (O(1) per sample, not O(W)).

/// Sliding-window RMS envelope extractor with O(1) amortized updates.
#[derive(Debug, Clone)]
pub struct RmsEnvelope {
    /// Circular buffer of squared samples
    buffer: Vec<f64>,
    /// Current write position in the circular buffer
    pos: usize,
    /// Running sum of squares (Σ x²)
    sum_sq: f64,
    /// Window size
    window_size: usize,
    /// True once the buffer has been fully filled at least once
    warm: bool,
}

impl RmsEnvelope {
    /// Create a new RMS envelope extractor.
    ///
    /// * `window_size` — Number of samples in the sliding window.
    ///   For 256Hz sampling and ~5Hz envelope: window ≈ 256/5 ≈ 50 samples.
    pub fn new(window_size: usize) -> Self {
        Self {
            buffer: vec![0.0; window_size],
            pos: 0,
            sum_sq: 0.0,
            window_size,
            warm: false,
        }
    }

    /// Reset envelope state.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.pos = 0;
        self.sum_sq = 0.0;
        self.warm = false;
    }

    /// Push a single (already rectified) sample and return the current RMS.
    ///
    /// O(1) amortized: subtracts the oldest squared sample, adds the new one.
    #[inline]
    pub fn push(&mut self, rectified_sample: f64) -> f64 {
        let sq = rectified_sample * rectified_sample;

        // Subtract oldest, add newest
        self.sum_sq -= self.buffer[self.pos];
        self.sum_sq += sq;

        // Guard against floating-point drift below zero
        if self.sum_sq < 0.0 {
            self.sum_sq = 0.0;
        }

        self.buffer[self.pos] = sq;
        self.pos = (self.pos + 1) % self.window_size;

        if self.pos == 0 && !self.warm {
            self.warm = true;
        }

        // RMS = sqrt(mean(x²))
        let divisor = if self.warm {
            self.window_size as f64
        } else {
            // During warm-up, only average over samples we've actually seen
            (self.pos.max(1)) as f64
        };

        (self.sum_sq / divisor).sqrt()
    }

    /// Process an entire buffer, returning the RMS envelope.
    pub fn process_buffer(&mut self, rectified: &[f64]) -> Vec<f64> {
        rectified.iter().map(|&s| self.push(s)).collect()
    }
}

// ============================================================================
// 3. sEMG PROCESSOR — Full Per-Channel DSP Chain
// ============================================================================

/// Complete sEMG DSP chain for a single channel.
///
/// Pipeline: High-Pass → |x| (Rectify) → RMS Envelope
#[derive(Debug, Clone)]
pub struct SemgChannelProcessor {
    pub hp_filter: HighPassFilter,
    pub rms_extractor: RmsEnvelope,
}

impl SemgChannelProcessor {
    /// Create a new per-channel sEMG processor.
    ///
    /// * `hp_cutoff` — High-pass cutoff in Hz (default: 20.0)
    /// * `sample_rate` — Sampling rate in Hz (default: 256.0)
    /// * `rms_window` — RMS window size in samples (default: 50 ≈ 5Hz envelope)
    pub fn new(hp_cutoff: f64, sample_rate: f64, rms_window: usize) -> Self {
        Self {
            hp_filter: HighPassFilter::new(hp_cutoff, sample_rate),
            rms_extractor: RmsEnvelope::new(rms_window),
        }
    }

    /// Default configuration for 256Hz sEMG: 20Hz HPF, 50-sample RMS window.
    pub fn default_256hz() -> Self {
        Self::new(20.0, 256.0, 50)
    }

    /// Reset all internal state (call between non-contiguous epochs).
    pub fn reset(&mut self) {
        self.hp_filter.reset();
        self.rms_extractor.reset();
    }

    /// Process a raw sEMG channel buffer through the full pipeline.
    ///
    /// Returns the muscle activation envelope (RMS values).
    /// Zero allocation beyond the output vector.
    pub fn process(&mut self, raw_channel: &[f64]) -> Vec<f64> {
        let mut envelope = Vec::with_capacity(raw_channel.len());

        for &sample in raw_channel {
            // Step 1: High-pass filter (remove DC + baseline wander)
            let hp_out = self.hp_filter.process(sample);

            // Step 2: Full-wave rectification (|x|)
            let rectified = hp_out.abs();

            // Step 3: RMS envelope extraction
            let rms = self.rms_extractor.push(rectified);

            envelope.push(rms);
        }

        envelope
    }
}

// ============================================================================
// 4. MULTI-CHANNEL sEMG PROCESSOR
// ============================================================================

/// Multi-channel sEMG processor with per-channel DSP chains.
///
/// Manages N independent channel processors and handles the final
/// projection into the LLM latent space.
pub struct SemgProcessor {
    /// Per-channel DSP chains
    channels: Vec<SemgChannelProcessor>,
    /// Number of EXG channels
    num_channels: usize,
    /// Sampling rate
    sample_rate: f64,
}

impl SemgProcessor {
    /// Create a processor for `num_channels` sEMG channels.
    pub fn new(num_channels: usize, sample_rate: f64) -> Self {
        let channels = (0..num_channels)
            .map(|_| SemgChannelProcessor::new(20.0, sample_rate, 50))
            .collect();

        Self {
            channels,
            num_channels,
            sample_rate,
        }
    }

    /// Reset all channel processors.
    pub fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.reset();
        }
    }

    /// Process a multi-channel sEMG epoch.
    ///
    /// Input: `raw_epoch` — flattened [num_channels × time_steps] in f64
    /// Output: Multi-channel RMS envelope [num_channels][time_steps]
    pub fn process_epoch(
        &mut self,
        raw_epoch: &[f64],
        time_steps: usize,
    ) -> Vec<Vec<f64>> {
        assert_eq!(
            raw_epoch.len(),
            self.num_channels * time_steps,
            "sEMG: epoch length mismatch"
        );

        self.channels
            .iter_mut()
            .enumerate()
            .map(|(ch_idx, processor)| {
                processor.reset(); // Reset between epochs
                let start = ch_idx * time_steps;
                let end = start + time_steps;
                processor.process(&raw_epoch[start..end])
            })
            .collect()
    }
}

// ============================================================================
// 5. PHONETIC LATENT PROJECTION
// ============================================================================
//
// Projects the 8-channel RMS envelope into the LLM's hidden_size
// dimension space, creating "Ghost Tokens" that encode subvocal phonetic
// intent.
//
// Math:
//   8 EXG channels × 256 timesteps = 2048 features (SEMG_FEATURE_DIM)
//   For Qwen hidden_size = 4096: zero-pad the remaining 2048 dimensions.
//   For Qwen hidden_size = 2048: exact fit, no padding needed.
//
// Strategy:
//   1. Interleave all 8 channel envelopes into a 2048-D vector
//   2. L2-normalize to prevent magnitude explosion in attention
//   3. Zero-pad to hidden_size (4096) for Qwen embedding alignment
//   4. Reshape to [1, 1, hidden_size] for attention injection

/// Extract phonetic latent vectors from raw sEMG safetensors data.
///
/// Takes a raw sEMG tensor [batch, channels, time_steps] and produces
/// raw phonetic features [batch, 1, SEMG_FEATURE_DIM] (2048-D).
/// 
/// NOTE: This outputs the true physical dimension (2048-D) which must be
/// projected to LLM hidden_size (e.g., 4096) via the SemgAdapter MLP.
pub fn extract_phonetic_latents(
    raw_semg: &Tensor,
    sample_rate: f64,
) -> Result<Tensor> {
    let dims = raw_semg.dims();
    if dims.len() != 3 {
        bail!(
            "sEMG: Expected 3D tensor [batch, channels, timesteps], got {:?}",
            dims
        );
    }

    let batch_size = dims[0];
    let num_channels = dims[1];
    let time_steps = dims[2];
    let device = raw_semg.device().clone();

    // HARDWARE VALIDATION: Enforce 8 EXG channels
    if num_channels != SEMG_CHANNEL_COUNT {
        bail!(
            "sEMG: Expected exactly {} EXG channels, got {}. \
             Verify the safetensors contains ONLY EXG data (channels 128..136).",
            SEMG_CHANNEL_COUNT, num_channels
        );
    }

    let feature_dim = SEMG_FEATURE_DIM;

    let mut ghost_tokens: Vec<f32> = Vec::with_capacity(batch_size * feature_dim);

    let mut processor = SemgProcessor::new(num_channels, sample_rate);

    for epoch_idx in 0..batch_size {
        // Extract single epoch [1, channels, timesteps]
        let epoch_tensor = raw_semg.narrow(0, epoch_idx, 1)?;
        let flat_f32: Vec<f32> = epoch_tensor.flatten_all()?.to_vec1()?;

        // Scale to µV
        let flat_f64: Vec<f64> = flat_f32.iter()
            .map(|&v| (v as f64) * 1_000_000.0)
            .collect();

        // Run DSP pipeline: HPF → Rectify → RMS Envelope
        let envelopes = processor.process_epoch(&flat_f64, time_steps);

        // Directive 2: Compute Mean Activation to find Global Peak
        let mut max_mean = -1.0f64;
        let mut t_peak = 0;
        for t in 0..time_steps {
            let mut sum = 0.0;
            for ch in 0..num_channels {
                sum += envelopes[ch][t];
            }
            let mean = sum / (num_channels as f64);
            if mean > max_mean {
                max_mean = mean;
                t_peak = t;
            }
        }

        // Directive 3: Define 64-Sample Attention Window
        // Place start roughly 60ms (16 samples) before peak onset
        let mut start_idx = t_peak.saturating_sub(16);
        
        // Ensure the window of 64 perfectly fits within current epoch bounds [0..time_steps]
        if start_idx + SEMG_ATTENTION_WINDOW > time_steps {
            start_idx = time_steps.saturating_sub(SEMG_ATTENTION_WINDOW);
        }
        let end_idx = start_idx + SEMG_ATTENTION_WINDOW;

        // Directive 4: Crop and Project
        // ONLY interleave samples falling within the high-attention temporal window
        let mut latent = Vec::with_capacity(feature_dim);
        for t in start_idx..end_idx {
            for ch in 0..num_channels {
                latent.push(envelopes[ch][t] as f32);
            }
        }

        // L2 normalize the cropped latent vector
        let norm: f32 = latent.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in latent.iter_mut() {
                *v /= norm;
            }
        }

        ghost_tokens.extend_from_slice(&latent);
    }

    // Reshape to [batch_size, 1, 512]
    let tensor = Tensor::from_vec(
        ghost_tokens,
        (batch_size, 1, feature_dim),
        &device,
    )?;

    Ok(tensor)
}

// ============================================================================
// 6. sEMG STREAMER — Safetensors Loader + Pipeline Integration
// ============================================================================

/// Streaming sEMG loader analogous to BciStreamer but for EXG channels.
pub struct SemgStreamer {
    /// Raw sEMG epochs: [num_samples, num_exg_channels, time_steps]
    pub semg_waves: Tensor,
    /// Labels per epoch
    pub labels: Tensor,
    /// Current read index
    pub current_idx: usize,
    /// Total epochs
    pub num_samples: usize,
    /// Per-channel DSP processor
    pub processor: SemgProcessor,
    /// Number of EXG channels
    pub num_channels: usize,
    /// Time steps per epoch
    pub time_steps: usize,
    /// Sampling rate
    pub sample_rate: f64,
}

impl SemgStreamer {
    /// Load sEMG data from a safetensors file.
    pub fn new(path: &Path, device: &Device, sample_rate: f64) -> Result<Self> {
        let safetensors = candle_core::safetensors::load(path, device)?;

        let semg_waves = safetensors.get("semg_waves")
            .or_else(|| safetensors.get("data"))
            .or_else(|| safetensors.get("exg_data"))
            .ok_or_else(|| anyhow::anyhow!(
                "No semg_waves/data/exg_data tensor in {:?}", path
            ))?
            .clone();

        let labels = safetensors.get("labels")
            .ok_or_else(|| anyhow::anyhow!("No labels tensor in {:?}", path))?
            .clone();

        let num_samples = semg_waves.dim(0)?;
        let num_channels = semg_waves.dim(1)?;
        let time_steps = semg_waves.dim(2)?;

        eprintln!(
            "[sEMG STREAMER] Loaded: {} epochs × {} channels × {} timesteps",
            num_samples, num_channels, time_steps
        );

        Ok(Self {
            semg_waves,
            labels,
            current_idx: 0,
            num_samples,
            processor: SemgProcessor::new(num_channels, sample_rate),
            num_channels,
            time_steps,
            sample_rate,
        })
    }

    /// Fetch the next raw phonetic features.
    ///
    /// Returns (raw_features [1, 1, SEMG_FEATURE_DIM], label).
    /// NOTE: The caller must project 2048-D features to LLM hidden_size via SemgAdapter.
    pub fn fetch_next_phonetic_prefix(
        &mut self,
    ) -> Result<(Tensor, i32)> {
        if self.current_idx >= self.num_samples {
            self.current_idx = 0;
        }

        let epoch_tensor = self.semg_waves.narrow(0, self.current_idx, 1)?;

        let label = self.labels.to_vec1::<u32>()
            .or_else(|_| self.labels.to_vec1::<i64>().map(|v| v.into_iter().map(|x| x as u32).collect()))
            .with_context(|| "Failed to decode sEMG labels")?[self.current_idx] as i32;

        // Run the full DSP pipeline on this single epoch
        // Output: raw 2048-D features (must be adapted before LLM injection)
        let ghost_prefix = extract_phonetic_latents(
            &epoch_tensor,
            self.sample_rate,
        )?;

        self.current_idx += 1;
        Ok((ghost_prefix, label))
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highpass_removes_dc() {
        let mut hp = HighPassFilter::new(20.0, 256.0);

        // Feed constant DC signal — output should decay to ~0
        let dc_signal: Vec<f64> = vec![1.0; 256];
        let mut output = Vec::new();
        for &s in &dc_signal {
            output.push(hp.process(s));
        }

        // After 256 samples at 256Hz (1 second), the DC component
        // should be heavily attenuated. Last sample should be near zero.
        let last = output.last().unwrap().abs();
        assert!(
            last < 0.01,
            "HPF failed to attenuate DC: last sample = {:.6}",
            last
        );
    }

    #[test]
    fn test_highpass_passes_high_frequency() {
        let mut hp = HighPassFilter::new(20.0, 256.0);

        // Feed a 100Hz sine wave (well above 20Hz cutoff)
        let freq = 100.0;
        let fs = 256.0;
        let samples: Vec<f64> = (0..512)
            .map(|n| (2.0 * std::f64::consts::PI * freq * n as f64 / fs).sin())
            .collect();

        let mut output = Vec::new();
        for &s in &samples {
            output.push(hp.process(s));
        }

        // After warm-up (~50 samples), the output should have significant amplitude
        let rms_out: f64 = output[100..].iter().map(|x| x * x).sum::<f64>()
            / (output.len() - 100) as f64;
        let rms_out = rms_out.sqrt();

        assert!(
            rms_out > 0.5,
            "HPF attenuated 100Hz signal too much: RMS = {:.6}",
            rms_out
        );
    }

    #[test]
    fn test_rms_envelope_constant_signal() {
        let mut rms = RmsEnvelope::new(10);

        // Feed constant amplitude 2.0
        let mut last_rms = 0.0;
        for _ in 0..100 {
            last_rms = rms.push(2.0);
        }

        // RMS of constant 2.0 = 2.0
        assert!(
            (last_rms - 2.0).abs() < 1e-10,
            "RMS of constant 2.0 should be 2.0, got {}",
            last_rms
        );
    }

    #[test]
    fn test_full_channel_pipeline() {
        let mut proc = SemgChannelProcessor::default_256hz();

        // Simulate a 256-sample sEMG burst:
        // First 100 samples: silence (near zero)
        // Samples 100-200: muscle activation (50Hz oscillation)
        // Last 56 samples: silence again
        let mut signal = vec![0.001; 256]; // tiny baseline
        for i in 100..200 {
            signal[i] = 0.5 * (2.0 * std::f64::consts::PI * 50.0 * i as f64 / 256.0).sin();
        }

        let envelope = proc.process(&signal);

        // The envelope should be near zero in the silent regions
        // and elevated during the burst
        let silent_rms: f64 = envelope[0..50].iter().sum::<f64>() / 50.0;
        let burst_rms: f64 = envelope[150..190].iter().sum::<f64>() / 40.0;

        assert!(
            burst_rms > silent_rms * 2.0,
            "Burst envelope ({:.6}) should be significantly larger than silence ({:.6})",
            burst_rms,
            silent_rms
        );
    }
}
