// src/spectral_memory.rs - Spectral Memory with Holographic Signature Verification

use rustfft::{FftPlanner, num_complex::Complex};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::RwLock;

// Thread-local FFT planner - LOCK-FREE, one planner per thread
thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f32>> = RefCell::new(FftPlanner::new());
}

/// Spectral Memory Compressor with Holographic Verification
/// 
/// Implements "Holographic" or "Spectral" compression for episodic memory
/// with integrity verification via phase-domain signatures
/// 
/// # Concurrency
/// - FFT planning uses thread-local storage: LOCK-FREE, no contention
/// - Signature store uses Mutex: only locked when storing/retrieving metadata
pub struct SpectralCompressor {
    signature_store: RwLock<HashMap<String, HolographicSignature>>,
}

/// Holographic signature for semantic integrity verification
#[derive(Debug, Clone)]
pub struct HolographicSignature {
    /// Dominant frequency components (top-K)
    pub dominant_frequencies: Vec<(usize, f32)>,
    
    /// Phase signature (averaged phase of dominant components)
    pub phase_signature: f32,
    
    /// Spectral entropy (measure of information content)
    pub spectral_entropy: f32,
    
    /// Timestamp of signature creation
    pub timestamp: std::time::SystemTime,
}

impl Default for SpectralCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl SpectralCompressor {
    pub fn new() -> Self {
        Self {
            signature_store: RwLock::new(HashMap::new()),
        }
    }

    /// Compress dense vector into sparse spectral components asynchronously
    /// Moves the heavy O(N log N) FFT math into tokio's dedicated blocking pool
    pub async fn compress_async(&self, vector: Vec<f32>, keep_ratio: f32) -> std::io::Result<Vec<(usize, Complex<f32>)>> {
        tokio::task::spawn_blocking(move || {
            let len = vector.len();
            
            // Get FFT from thread-local storage: no lock, ~nanosecond access
            let fft = FFT_PLANNER.with(|planner| {
                planner.borrow_mut().plan_fft_forward(len)
            });

            let mut buffer: Vec<Complex<f32>> = vector.iter()
                .map(|&v| Complex::new(v, 0.0))
                .collect();

            fft.process(&mut buffer);

            let k = (len as f32 * keep_ratio).ceil() as usize;
            
            let mut indexed_coeffs: Vec<(usize, Complex<f32>)> = buffer.into_iter()
                .enumerate()
                .collect();

            indexed_coeffs.sort_unstable_by(|a, b| {
                b.1.norm_sqr().partial_cmp(&a.1.norm_sqr()).unwrap_or(std::cmp::Ordering::Equal)
            });

            indexed_coeffs.into_iter().take(k).collect()
        })
        .await
        .map_err(std::io::Error::other)
    }

    /// Decompress sparse spectral components back to dense vector asynchronously
    /// Moves the heavy O(N log N) Inverse FFT math into tokio's dedicated blocking pool
    pub async fn decompress_async(&self, sparse: Vec<(usize, Complex<f32>)>, original_len: usize) -> std::io::Result<Vec<f32>> {
        tokio::task::spawn_blocking(move || {
            // Get FFT from thread-local storage: no lock, ~nanosecond access
            let fft = FFT_PLANNER.with(|planner| {
                planner.borrow_mut().plan_fft_inverse(original_len)
            });

            let mut buffer = vec![Complex::new(0.0, 0.0); original_len];
            
            for &(idx, val) in &sparse {
                if idx < original_len {
                    buffer[idx] = val;
                }
            }

            fft.process(&mut buffer);

            let scale = 1.0 / original_len as f32;
            buffer.iter().map(|c| c.re * scale).collect()
        })
        .await
        .map_err(std::io::Error::other)
    }

    /// Create holographic signature for a signal
    /// 
    /// Stores frequency-domain fingerprint for later verification
    pub fn create_signature(&self, signal: &[f32], signature_id: String) -> HolographicSignature {
        let len = signal.len();
        
        // Get FFT from thread-local storage: no lock, ~nanosecond access
        let fft = FFT_PLANNER.with(|planner| {
            planner.borrow_mut().plan_fft_forward(len)
        });

        let mut buffer: Vec<Complex<f32>> = signal.iter()
            .map(|&v| Complex::new(v, 0.0))
            .collect();

        fft.process(&mut buffer);

        // Extract dominant frequencies (top 10)
        let mut indexed_coeffs: Vec<(usize, Complex<f32>)> = buffer.iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();

        indexed_coeffs.sort_unstable_by(|a, b| {
            b.1.norm_sqr().partial_cmp(&a.1.norm_sqr()).unwrap_or(std::cmp::Ordering::Equal)
        });

        let dominant_frequencies: Vec<(usize, f32)> = indexed_coeffs
            .iter()
            .take(10)
            .map(|(idx, c)| (*idx, c.norm()))
            .collect();

        // Calculate average phase of dominant components
        let phase_signature = indexed_coeffs
            .iter()
            .take(10)
            .map(|(_, c)| c.arg())
            .sum::<f32>() / 10.0;

        // Calculate spectral entropy
        let spectral_entropy = Self::calculate_spectral_entropy(&buffer);

        let signature = HolographicSignature {
            dominant_frequencies,
            phase_signature,
            spectral_entropy,
            timestamp: std::time::SystemTime::now(),
        };

        let mut store = self.signature_store.write().unwrap();
        store.insert(signature_id, signature.clone());

        signature
    }

    /// Verify holographic signature of a repaired signal
    /// 
    /// Returns Ok(()) if signature matches, Err with divergence measure otherwise
    pub fn verify_holographic_signature(
        &self,
        signal: &[f32],
        signature_id: &str,
        phase_tolerance: f32,
    ) -> Result<(), SignatureVerificationError> {
        let stored_signature = {
            let store = self.signature_store.read().unwrap();
            store.get(signature_id).cloned()
                .ok_or(SignatureVerificationError::SignatureNotFound)?
        };

        // Compute current signature
        let current_signature = self.create_signature_without_storing(signal);

        // Check phase divergence
        let phase_divergence = (current_signature.phase_signature - stored_signature.phase_signature).abs();
        
        if phase_divergence > phase_tolerance {
            return Err(SignatureVerificationError::PhaseDivergence {
                expected: stored_signature.phase_signature,
                observed: current_signature.phase_signature,
                divergence: phase_divergence,
            });
        }

        // Check frequency alignment
        let frequency_match = Self::compare_frequency_signatures(
            &stored_signature.dominant_frequencies,
            &current_signature.dominant_frequencies,
        );

        if frequency_match < 0.7 {
            return Err(SignatureVerificationError::FrequencyMismatch {
                similarity: frequency_match,
            });
        }

        // Check entropy consistency (within 20%)
        let entropy_ratio = current_signature.spectral_entropy / stored_signature.spectral_entropy;
        if !(0.8..=1.2).contains(&entropy_ratio) {
            return Err(SignatureVerificationError::EntropyAnomaly {
                expected: stored_signature.spectral_entropy,
                observed: current_signature.spectral_entropy,
            });
        }

        Ok(())
    }

    /// Create signature without storing (for verification)
    fn create_signature_without_storing(&self, signal: &[f32]) -> HolographicSignature {
        let len = signal.len();
        
        // Get FFT from thread-local storage: no lock, ~nanosecond access
        let fft = FFT_PLANNER.with(|planner| {
            planner.borrow_mut().plan_fft_forward(len)
        });

        let mut buffer: Vec<Complex<f32>> = signal.iter()
            .map(|&v| Complex::new(v, 0.0))
            .collect();

        fft.process(&mut buffer);

        let mut indexed_coeffs: Vec<(usize, Complex<f32>)> = buffer.iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();

        indexed_coeffs.sort_unstable_by(|a, b| {
            b.1.norm_sqr().partial_cmp(&a.1.norm_sqr()).unwrap_or(std::cmp::Ordering::Equal)
        });

        let dominant_frequencies: Vec<(usize, f32)> = indexed_coeffs
            .iter()
            .take(10)
            .map(|(idx, c)| (*idx, c.norm()))
            .collect();

        let phase_signature = indexed_coeffs
            .iter()
            .take(10)
            .map(|(_, c)| c.arg())
            .sum::<f32>() / 10.0;

        let spectral_entropy = Self::calculate_spectral_entropy(&buffer);

        HolographicSignature {
            dominant_frequencies,
            phase_signature,
            spectral_entropy,
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Calculate spectral entropy
    /// 
    /// H = -II p_i log(p_i)
    /// where p_i = |X_i|II / II|X_j|II
    fn calculate_spectral_entropy(spectrum: &[Complex<f32>]) -> f32 {
        let total_power: f32 = spectrum.iter().map(|c| c.norm_sqr()).sum();
        
        if total_power == 0.0 {
            return 0.0;
        }

        let entropy: f32 = spectrum.iter()
            .map(|c| {
                let p = c.norm_sqr() / total_power;
                if p > 1e-10 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();

        entropy
    }

    /// Compare two frequency signatures
    /// 
    /// Returns similarity score [0, 1]
    fn compare_frequency_signatures(
        sig1: &[(usize, f32)],
        sig2: &[(usize, f32)],
    ) -> f32 {
        let freq_set1: Vec<usize> = sig1.iter().map(|(idx, _)| *idx).collect();
        let freq_set2: Vec<usize> = sig2.iter().map(|(idx, _)| *idx).collect();

        // Count common frequencies
        let common_count = freq_set1.iter()
            .filter(|f| freq_set2.contains(f))
            .count();

        // Jaccard similarity
        let union_count = freq_set1.len() + freq_set2.len() - common_count;
        
        if union_count == 0 {
            1.0
        } else {
            common_count as f32 / union_count as f32
        }
    }

    /// Clear old signatures (garbage collection)
    pub fn clear_old_signatures(&self, max_age_secs: u64) {
        let mut store = self.signature_store.write().unwrap();
        let now = std::time::SystemTime::now();

        store.retain(|_, sig| {
            if let Ok(duration) = now.duration_since(sig.timestamp) {
                duration.as_secs() < max_age_secs
            } else {
                true
            }
        });
    }
}

/// Errors that can occur during signature verification
#[derive(Debug, Clone)]
pub enum SignatureVerificationError {
    SignatureNotFound,
    PhaseDivergence {
        expected: f32,
        observed: f32,
        divergence: f32,
    },
    FrequencyMismatch {
        similarity: f32,
    },
    EntropyAnomaly {
        expected: f32,
        observed: f32,
    },
}

impl std::fmt::Display for SignatureVerificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignatureVerificationError::SignatureNotFound => {
                write!(f, "Holographic signature not found in store")
            }
            SignatureVerificationError::PhaseDivergence { expected, observed, divergence } => {
                write!(
                    f,
                    "Phase divergence detected: IIII = {:.4} (expected={:.4}, observed={:.4})",
                    divergence, expected, observed
                )
            }
            SignatureVerificationError::FrequencyMismatch { similarity } => {
                write!(f, "Frequency signature mismatch: similarity = {:.2}", similarity)
            }
            SignatureVerificationError::EntropyAnomaly { expected, observed } => {
                write!(
                    f,
                    "Spectral entropy anomaly: expected={:.2}, observed={:.2}",
                    expected, observed
                )
            }
        }
    }
}

impl std::error::Error for SignatureVerificationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_compression_decompression() {
        let compressor = SpectralCompressor::new();
        
        // Create test signal
        let signal: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        
        // Compress
        let compressed = compressor.compress_async(signal.clone(), 0.1).await.unwrap();
        
        // Decompress
        let reconstructed = compressor.decompress_async(compressed, signal.len()).await.unwrap();
        
        assert_eq!(reconstructed.len(), signal.len());
    }

    #[test]
    fn test_signature_creation() {
        let compressor = SpectralCompressor::new();
        
        let signal: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        
        let signature = compressor.create_signature(&signal, "test_signal".to_string());
        
        assert_eq!(signature.dominant_frequencies.len(), 10);
        assert!(!signature.phase_signature.is_nan());
        assert!(signature.spectral_entropy > 0.0);
    }

    #[test]
    fn test_signature_verification() {
        let compressor = SpectralCompressor::new();
        
        let signal: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        
        // Create signature
        compressor.create_signature(&signal, "test_signal".to_string());
        
        // Verify with same signal (should pass)
        let result = compressor.verify_holographic_signature(&signal, "test_signal", 0.1);
        assert!(result.is_ok(), "Signature verification should pass for identical signal");
    }

    #[test]
    fn test_signature_verification_fails_on_corruption() {
        let compressor = SpectralCompressor::new();
        
        let signal: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        
        // Create signature
        compressor.create_signature(&signal, "test_signal".to_string());
        
        // Corrupt signal
        let mut corrupted = signal.clone();
        corrupted[50] += 10.0; // Large corruption
        
        // Verify with corrupted signal (should fail)
        let result = compressor.verify_holographic_signature(&corrupted, "test_signal", 0.1);
        assert!(result.is_err(), "Signature verification should fail for corrupted signal");
    }

    #[test]
    fn test_spectral_entropy() {
        // Pure sine wave (low entropy)
        let pure_sine: Vec<Complex<f32>> = (0..128)
            .map(|i| Complex::new((i as f32 * 0.1).sin(), 0.0))
            .collect();
        
        let entropy_sine = SpectralCompressor::calculate_spectral_entropy(&pure_sine);
        
        // Random signal (high entropy)
        let random: Vec<Complex<f32>> = (0..128)
            .map(|i| Complex::new((i as f32 * 123.456).sin(), 0.0))
            .collect();
        
        let entropy_random = SpectralCompressor::calculate_spectral_entropy(&random);
        
        // Random should have higher entropy
        assert!(entropy_random > entropy_sine, "Random signal should have higher entropy");
    }
}













