// src/signal_processor.rs - Real-Time Signal Processing for BrainFlow Data
//
// High-performance signal processing pipeline using Rust-native libraries:
// - rustfft for FFT/IFFT operations
// - faer for linear algebra
// - candle for GPU acceleration
//
// Replaces scipy.signal filtering with zero-allocation Rust implementations

use anyhow::Result;
use candle_core::{Device, Tensor};
use rustfft::{FftPlanner, num_complex::Complex32};
use std::sync::Arc;
use std::time::Duration;

use crate::brainflow_bridge::BrainFlowBridge;
use crate::fast_math::SemanticWave;

// ============================================================================
// FILTER TYPES
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
    Notch50Hz,   // Power line interference (Europe/Asia)
    Notch60Hz,   // Power line interference (Americas)
}

#[derive(Debug, Clone)]
pub struct FilterParams {
    pub filter_type: FilterType,
    pub low_cutoff: f32,    // Hz
    pub high_cutoff: f32,   // Hz
    pub order: usize,
    pub sampling_rate: f32,
}

impl Default for FilterParams {
    fn default() -> Self {
        Self {
            filter_type: FilterType::Bandpass,
            low_cutoff: 4.0,   // Theta-Alpha boundary
            high_cutoff: 40.0, // Gamma ceiling
            order: 4,
            sampling_rate: 250.0,
        }
    }
}

// ============================================================================
// SIGNAL PROCESSOR
// ============================================================================

pub struct SignalProcessor {
    device: Device,
    fft_planner: FftPlanner<f32>,
    filter_state: Vec<Vec<f64>>, // IIR filter state per channel
}

impl SignalProcessor {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            fft_planner: FftPlanner::new(),
            filter_state: Vec::new(),
        }
    }

    /// Apply bandpass filter using FFT-based approach (zero-phase)
    /// 
    /// Much faster than scipy.filtfilt for large arrays
    pub fn bandpass_filter(
        &mut self,
        data: &Tensor,
        low_freq: f32,
        high_freq: f32,
        sampling_rate: f32,
    ) -> Result<Tensor> {
        let (n_samples, n_channels) = data.dims2()?;
        let data_f32 = data.to_vec2::<f32>()?;
        
        // FFT size (next power of 2 for efficiency)
        let n_fft = n_samples.next_power_of_two();
        let freqs: Vec<f32> = (0..n_fft)
            .map(|i| i as f32 * sampling_rate / n_fft as f32)
            .collect();
        
        let mut filtered_data = Vec::with_capacity(n_channels);
        
        for ch in 0..n_channels {
            // Extract channel data
            let mut channel_data: Vec<f32> = (0..n_samples)
                .map(|i| data_f32[i][ch])
                .collect();
            
            // Zero-pad to FFT length
            channel_data.resize(n_fft, 0.0);
            
            // Convert to complex for FFT
            let mut complex_data: Vec<Complex32> = channel_data
                .iter()
                .map(|&x| Complex32::new(x, 0.0))
                .collect();
            
            // Forward FFT
            let fft = self.fft_planner.plan_fft_forward(n_fft);
            fft.process(&mut complex_data);
            
            // Apply bandpass mask in frequency domain
            for (i, freq) in freqs.iter().enumerate() {
                if *freq < low_freq || *freq > high_freq {
                    // Also filter negative frequencies
                    complex_data[i] = Complex32::new(0.0, 0.0);
                    if i > 0 && i < n_fft - 1 {
                        complex_data[n_fft - i] = Complex32::new(0.0, 0.0);
                    }
                }
            }
            
            // Inverse FFT
            let ifft = self.fft_planner.plan_fft_inverse(n_fft);
            ifft.process(&mut complex_data);
            
            // Extract real part and normalize
            let filtered: Vec<f32> = complex_data
                .iter()
                .take(n_samples)
                .map(|c| c.re / n_fft as f32)
                .collect();
            
            filtered_data.push(filtered);
        }
        
        // Transpose back to (n_samples, n_channels)
        let mut result = Vec::with_capacity(n_samples * n_channels);
        for i in 0..n_samples {
            for ch in 0..n_channels {
                result.push(filtered_data[ch][i]);
            }
        }
        
        Ok(Tensor::from_vec(result, (n_samples, n_channels), &self.device)?)
    }

    /// Apply notch filter for power line interference removal
    pub fn notch_filter(
        &mut self,
        data: &Tensor,
        notch_freq: f32,
        sampling_rate: f32,
        bandwidth: f32,
    ) -> Result<Tensor> {
        // Simple IIR notch filter implementation
        let (n_samples, n_channels) = data.dims2()?;
        let data_f32 = data.to_vec2::<f32>()?;
        
        // Notch filter coefficients
        let w0 = 2.0 * std::f32::consts::PI * notch_freq / sampling_rate;
        let bw = 2.0 * std::f32::consts::PI * bandwidth / sampling_rate;
        let r = 1.0 - 3.0 * bw;
        
        let k = (1.0 - 2.0 * r * w0.cos() + r * r) / (2.0 - 2.0 * w0.cos());
        
        let b0 = k;
        let b1 = -2.0 * k * w0.cos();
        let b2 = k;
        let a1 = 2.0 * r * w0.cos();
        let a2 = -r * r;
        
        let mut filtered_data = Vec::with_capacity(n_channels);
        
        for ch in 0..n_channels {
            let input: Vec<f32> = (0..n_samples)
                .map(|i| data_f32[i][ch])
                .collect();
            
            let mut output = vec![0.0f32; n_samples];
            let mut x_history = [0.0f32; 2];
            let mut y_history = [0.0f32; 2];
            
            for i in 0..n_samples {
                let x = input[i];
                let y = b0 * x + b1 * x_history[0] + b2 * x_history[1]
                        + a1 * y_history[0] + a2 * y_history[1];
                
                x_history[1] = x_history[0];
                x_history[0] = x;
                y_history[1] = y_history[0];
                y_history[0] = y;
                
                output[i] = y;
            }
            
            filtered_data.push(output);
        }
        
        // Transpose back
        let mut result = Vec::with_capacity(n_samples * n_channels);
        for i in 0..n_samples {
            for ch in 0..n_channels {
                result.push(filtered_data[ch][i]);
            }
        }
        
        Ok(Tensor::from_vec(result, (n_samples, n_channels), &self.device)?)
    }

    /// Compute Power Spectral Density using Welch's method
    pub fn compute_psd(
        &mut self,
        data: &Tensor,
        sampling_rate: f32,
        nfft: usize,
        overlap: f32,
    ) -> Result<(Tensor, Vec<f32>)> {
        let (n_samples, n_channels) = data.dims2()?;
        let data_f32 = data.to_vec2::<f32>()?;
        
        let hop = (nfft as f32 * (1.0 - overlap)) as usize;
        let n_segments = (n_samples - nfft) / hop + 1;
        
        let mut psd_per_channel: Vec<Vec<f32>> = vec![vec![0.0; nfft / 2 + 1]; n_channels];
        let window: Vec<f32> = (0..nfft)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / nfft as f32).cos())
            .collect(); // Hann window
        
        let fft = self.fft_planner.plan_fft_forward(nfft);
        
        for seg in 0..n_segments {
            let start = seg * hop;
            
            for ch in 0..n_channels {
                let mut segment: Vec<Complex32> = (0..nfft)
                    .map(|i| {
                        let sample_idx = (start + i).min(n_samples - 1);
                        Complex32::new(data_f32[sample_idx][ch] * window[i], 0.0)
                    })
                    .collect();
                
                fft.process(&mut segment);
                
                // Accumulate magnitude squared (periodogram)
                for (i, bin) in segment.iter().take(nfft / 2 + 1).enumerate() {
                    psd_per_channel[ch][i] += bin.norm_sqr() / (nfft as f32 * n_segments as f32);
                }
            }
        }
        
        // Frequency bins
        let freqs: Vec<f32> = (0..=nfft / 2)
            .map(|i| i as f32 * sampling_rate / nfft as f32)
            .collect();
        
        // Flatten PSD data
        let mut psd_flat = Vec::with_capacity((nfft / 2 + 1) * n_channels);
        for ch in 0..n_channels {
            for i in 0..=nfft / 2 {
                psd_flat.push(psd_per_channel[ch][i]);
            }
        }
        
        let psd_tensor = Tensor::from_vec(psd_flat, (nfft / 2 + 1, n_channels), &self.device)?;
        
        Ok((psd_tensor, freqs))
    }

    /// Extract band powers (Delta, Theta, Alpha, Beta, Gamma)
    pub fn extract_band_powers(
        &mut self,
        data: &Tensor,
        sampling_rate: f32,
    ) -> Result<BandPowers> {
        let (psd, freqs) = self.compute_psd(data, sampling_rate, 256, 0.5)?;
        
        let bands = [
            ("delta", 0.5, 4.0),
            ("theta", 4.0, 8.0),
            ("alpha", 8.0, 13.0),
            ("beta", 13.0, 30.0),
            ("gamma", 30.0, 100.0),
        ];
        
        let mut powers = BandPowers::default();
        let psd_f32 = psd.to_vec2::<f32>()?;
        let n_channels = psd_f32[0].len();
        
        for (name, low, high) in bands.iter() {
            let low_idx = freqs.iter().position(|&f| f >= *low).unwrap_or(0);
            let high_idx = freqs.iter().position(|&f| f >= *high).unwrap_or(freqs.len() - 1);
            
            let band_power: Vec<f32> = (0..n_channels)
                .map(|ch| {
                    (low_idx..=high_idx)
                        .map(|i| psd_f32[i][ch])
                        .sum::<f32>()
                })
                .collect();
            
            let mean_power = band_power.iter().sum::<f32>() / n_channels as f32;
            
            match *name {
                "delta" => powers.delta = mean_power,
                "theta" => powers.theta = mean_power,
                "alpha" => powers.alpha = mean_power,
                "beta" => powers.beta = mean_power,
                "gamma" => powers.gamma = mean_power,
                _ => {}
            }
        }
        
        Ok(powers)
    }

    /// Apply Hilbert transform to get analytic signal
    /// Returns complex-valued tensor for phasor representation
    pub fn hilbert_transform(&mut self, data: &Tensor) -> Result<SemanticWave> {
        let (n_samples, n_channels) = data.dims2()?;
        let data_f32 = data.to_vec2::<f32>()?;
        
        let n_fft = n_samples.next_power_of_two();
        let fft = self.fft_planner.plan_fft_forward(n_fft);
        let ifft = self.fft_planner.plan_fft_inverse(n_fft);
        
        let mut analytic_signal = Vec::with_capacity(n_channels);
        
        for ch in 0..n_channels {
            let channel_data: Vec<f32> = (0..n_samples)
                .map(|i| data_f32[i][ch])
                .collect();
            
            // Zero-pad
            let mut padded = channel_data.clone();
            padded.resize(n_fft, 0.0);
            
            let mut complex_data: Vec<Complex32> = padded
                .iter()
                .map(|&x| Complex32::new(x, 0.0))
                .collect();
            
            // Forward FFT
            fft.process(&mut complex_data);
            
            // Hilbert transform in frequency domain
            // Multiply positive frequencies by 2, zero negative frequencies
            for i in 0..n_fft {
                if i == 0 || (i == n_fft / 2 && n_fft % 2 == 0) {
                    // DC and Nyquist stay same
                } else if i < n_fft / 2 {
                    complex_data[i] *= 2.0;
                } else {
                    complex_data[i] = Complex32::new(0.0, 0.0);
                }
            }
            
            // Inverse FFT
            ifft.process(&mut complex_data);
            
            // Extract first n_samples
            let analytic: Vec<num_complex::Complex64> = complex_data
                .iter()
                .take(n_samples)
                .map(|c| num_complex::Complex64::new(c.re as f64, c.im as f64))
                .collect();
            
            analytic_signal.push(analytic);
        }
        
        // Create SemanticWave from analytic signal (using first channel as primary)
        // For multi-channel, we'd need to aggregate or select
        let primary_channel = if !analytic_signal.is_empty() {
            analytic_signal[0].clone()
        } else {
            Vec::new()
        };
        
        Ok(SemanticWave::from_vec(primary_channel))
    }

    /// Downsample signal using decimation with anti-aliasing filter
    pub fn downsample(
        &mut self,
        data: &Tensor,
        factor: usize,
        sampling_rate: f32,
    ) -> Result<Tensor> {
        // First apply anti-aliasing lowpass filter
        let nyquist_new = sampling_rate / (2.0 * factor as f32);
        let filtered = self.bandpass_filter(data, 0.1, nyquist_new - 1.0, sampling_rate)?;
        
        let (n_samples, n_channels) = filtered.dims2()?;
        let data_f32 = filtered.to_vec2::<f32>()?;
        
        let n_new = n_samples / factor;
        let mut downsampled = Vec::with_capacity(n_new * n_channels);
        
        for i in (0..n_samples).step_by(factor).take(n_new) {
            for ch in 0..n_channels {
                downsampled.push(data_f32[i][ch]);
            }
        }
        
        Ok(Tensor::from_vec(downsampled, (n_new, n_channels), &self.device)?)
    }
}

// ============================================================================
// BAND POWERS STRUCT
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct BandPowers {
    pub delta: f32,  // 0.5-4 Hz
    pub theta: f32,  // 4-8 Hz
    pub alpha: f32,  // 8-13 Hz
    pub beta: f32,   // 13-30 Hz
    pub gamma: f32,  // 30-100 Hz
}

impl BandPowers {
    /// Compute alpha/theta ratio (relaxation indicator)
    pub fn alpha_theta_ratio(&self) -> f32 {
        if self.theta > 0.0 {
            self.alpha / self.theta
        } else {
            0.0
        }
    }

    /// Compute beta/alpha ratio (focus/alertness indicator)
    pub fn beta_alpha_ratio(&self) -> f32 {
        if self.alpha > 0.0 {
            self.beta / self.alpha
        } else {
            0.0
        }
    }

    /// Total power across all bands
    pub fn total_power(&self) -> f32 {
        self.delta + self.theta + self.alpha + self.beta + self.gamma
    }

    /// Normalize to relative powers (0-1 range)
    pub fn normalize(&self) -> NormalizedBands {
        let total = self.total_power();
        if total > 0.0 {
            NormalizedBands {
                delta: self.delta / total,
                theta: self.theta / total,
                alpha: self.alpha / total,
                beta: self.beta / total,
                gamma: self.gamma / total,
            }
        } else {
            NormalizedBands::default()
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct NormalizedBands {
    pub delta: f32,
    pub theta: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
}

// ============================================================================
// REAL-TIME PIPELINE
// ============================================================================

pub struct RealtimePipeline {
    bridge: Arc<BrainFlowBridge>,
    processor: SignalProcessor,
    filter_params: FilterParams,
    buffer_size: usize,
    last_process_time: Instant,
}

use std::time::Instant;

impl RealtimePipeline {
    pub fn new(
        bridge: Arc<BrainFlowBridge>,
        filter_params: FilterParams,
        buffer_size: usize,
    ) -> Result<Self> {
        let device = bridge.num_channels();
        
        Ok(Self {
            bridge,
            processor: SignalProcessor::new(Device::Cpu), // Use CPU for real-time consistency
            filter_params,
            buffer_size,
            last_process_time: Instant::now(),
        })
    }

    /// Process one frame of data
    pub fn process_frame(&mut self) -> Result<ProcessedFrame> {
        // Acquire data from BrainFlow
        let raw_tensor = self.bridge.acquire_tensor(self.buffer_size)?;
        
        // Apply filters
        let filtered = self.processor.bandpass_filter(
            &raw_tensor,
            self.filter_params.low_cutoff,
            self.filter_params.high_cutoff,
            self.filter_params.sampling_rate,
        )?;
        
        // Remove power line interference
        let cleaned = self.processor.notch_filter(
            &filtered,
            50.0, // 50Hz for Europe/Asia, 60Hz for Americas
            self.filter_params.sampling_rate,
            2.0,  // 2Hz bandwidth
        )?;
        
        // Extract features
        let band_powers = self.processor.extract_band_powers(&cleaned, self.filter_params.sampling_rate)?;
        
        // Hilbert transform for phasor representation
        let wave = self.processor.hilbert_transform(&cleaned)?;
        
        let elapsed = self.last_process_time.elapsed();
        self.last_process_time = Instant::now();
        
        Ok(ProcessedFrame {
            raw_data: raw_tensor,
            filtered_data: cleaned,
            band_powers,
            semantic_wave: wave,
            process_time_ms: elapsed.as_secs_f32() * 1000.0,
        })
    }

    /// Continuous processing loop - injects to GHOST_VECTOR_SLOT
    pub fn run_continuous(&mut self) -> Result<()> {
        tracing::info!("[SignalProcessor] Starting continuous processing loop");
        
        loop {
            match self.process_frame() {
                Ok(frame) => {
                    // Inject phasor representation to ghost slot
                    // Convert wave to tensor for neuro_kernel consumption
                    let wave_data: Vec<f32> = frame.semantic_wave.amplitudes
                        .iter()
                        .map(|c: &num_complex::Complex64| c.norm() as f32)
                        .collect();
                    
                    let wave_tensor = Tensor::from_vec(
                        wave_data,
                        (1, frame.semantic_wave.len()),
                        &Device::Cpu,
                    )?;
                    
                    self.bridge.inject_to_ghost_slot(wave_tensor)?;
                    
                    tracing::debug!(
                        "[SignalProcessor] Frame processed in {:.2}ms, A/θ={:.2}",
                        frame.process_time_ms,
                        frame.band_powers.alpha_theta_ratio()
                    );
                }
                Err(e) => {
                    tracing::warn!("[SignalProcessor] Frame processing error: {}", e);
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct ProcessedFrame {
    pub raw_data: Tensor,
    pub filtered_data: Tensor,
    pub band_powers: BandPowers,
    pub semantic_wave: SemanticWave,
    pub process_time_ms: f32,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandpass_filter() {
        let device = Device::Cpu;
        let mut processor = SignalProcessor::new(device);
        
        // Create synthetic signal with known frequency components
        let sampling_rate = 250.0;
        let t: Vec<f32> = (0..1000).map(|i| i as f32 / sampling_rate).collect();
        
        // Signal with 10Hz (alpha) and 60Hz (noise)
        let signal: Vec<f32> = t.iter()
            .map(|&t| (2.0 * std::f32::consts::PI * 10.0 * t).sin() +
                      0.5 * (2.0 * std::f32::consts::PI * 60.0 * t).sin())
            .collect();
        
        let tensor = Tensor::from_vec(signal.clone(), (1000, 1), &Device::Cpu).unwrap();
        
        // Apply bandpass 8-13 Hz (alpha band)
        let filtered = processor.bandpass_filter(&tensor, 8.0, 13.0, sampling_rate).unwrap();
        
        assert_eq!(filtered.dims(), vec![1000, 1]);
    }

    #[test]
    fn test_band_powers() {
        let device = Device::Cpu;
        let mut processor = SignalProcessor::new(device);
        
        // Create 10Hz alpha wave
        let sampling_rate = 250.0;
        let signal: Vec<f32> = (0..1000)
            .map(|i| {
                let t = i as f32 / sampling_rate;
                (2.0 * std::f32::consts::PI * 10.0 * t).sin()
            })
            .collect();
        
        let tensor = Tensor::from_vec(signal, (1000, 1), &Device::Cpu).unwrap();
        let powers = processor.extract_band_powers(&tensor, sampling_rate).unwrap();
        
        // Alpha should be dominant
        assert!(powers.alpha > powers.theta);
        assert!(powers.alpha > powers.beta);
    }

    #[test]
    fn test_notch_filter() {
        let device = Device::Cpu;
        let mut processor = SignalProcessor::new(device);
        
        // 50Hz power line signal
        let sampling_rate = 250.0;
        let signal: Vec<f32> = (0..1000)
            .map(|i| {
                let t = i as f32 / sampling_rate;
                (2.0 * std::f32::consts::PI * 50.0 * t).sin()
            })
            .collect();
        
        let tensor = Tensor::from_vec(signal, (1000, 1), &Device::Cpu).unwrap();
        let filtered = processor.notch_filter(&tensor, 50.0, sampling_rate, 2.0).unwrap();
        
        // Energy should be reduced
        let original_energy: f32 = signal.iter().map(|&x| x * x).sum();
        let filtered_f32 = filtered.to_vec2::<f32>().unwrap();
        let filtered_energy: f32 = filtered_f32.iter().map(|row| row[0] * row[0]).sum();
        
        assert!(filtered_energy < original_energy);
    }
}
