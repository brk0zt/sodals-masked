// src/engine/bci_mux.rs
// ============================================================================
// BCI HARDWARE CRUCIBLE & TOPOLOGICAL ALIGNMENT BRIDGE
// ============================================================================
//
// Three mathematical components for 4-channel → 64-channel reconstruction:
//
//   1. GAUSSIAN NOISE CRUCIBLE: Synthetic degradation (thermal + 50Hz AC)
//   2. NEWTON-RAPHSON ALIGNMENT: Phase-angle root-finding for manifold snap
//   3. Z-SCORE OUTLIER GATING: EMA-based EMG artifact rejection (3σ gate)
//
// All computation is CPU f64 — zero VRAM cost. Zero-allocation in hot loops.

use candle_core::{DType, Device, Tensor};
use anyhow::Result;
use num_complex::Complex64;
use crate::fast_math::SemanticWave;
use std::path::Path;

// ============================================================================
// 1. GAUSSIAN NOISE CRUCIBLE — Synthetic Degradation Engine
// ============================================================================

/// Injects physiologically-realistic noise into a pristine 4-channel EEG subset.
///
/// Two noise sources:
///   - Gaussian White Noise (thermal sensor noise): N(0, σ²)
///   - 50Hz AC Power-Line Interference: A·sin(2π·50·t + φ)
///
/// Uses Box-Muller transform for Gaussian sampling (zero external deps).
pub struct GaussianNoiseCrucible {
    /// Thermal noise standard deviation (µV). Typical in-ear: 2.0–5.0 µV.
    pub thermal_sigma: f64,
    /// 50Hz interference amplitude (µV). Typical unshielded: 10.0–30.0 µV.
    pub ac_amplitude: f64,
    /// AC phase offset per channel (radians). Models spatial electrode variance.
    pub ac_phase_offsets: [f64; 4],
    /// Sampling rate (Hz). Default: 256 Hz for in-ear BCI.
    pub sampling_rate: f64,
    /// Deterministic PRNG state (xorshift64) for reproducible noise.
    rng_state: u64,
}

impl GaussianNoiseCrucible {
    pub fn new(thermal_sigma: f64, ac_amplitude: f64, sampling_rate: f64) -> Self {
        Self {
            thermal_sigma,
            ac_amplitude,
            ac_phase_offsets: [0.0, 0.7854, 1.5708, 2.3562], // 0, π/4, π/2, 3π/4
            sampling_rate,
            rng_state: 0xDEAD_BEEF_CAFE_BABEu64,
        }
    }

    /// Default crucible calibrated for AirPods-style in-ear EEG.
    pub fn airpods_default() -> Self {
        Self::new(3.5, 15.0, 256.0)
    }

    /// Inline xorshift64 PRNG — zero-allocation, deterministic.
    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Box-Muller transform: two uniform → one Gaussian sample.
    #[inline]
    fn sample_gaussian(&mut self) -> f64 {
        let u1 = (self.next_u64() as f64) / (u64::MAX as f64);
        let u2 = (self.next_u64() as f64) / (u64::MAX as f64);
        let u1_clamped = u1.max(1e-15); // Prevent ln(0)
        (-2.0 * u1_clamped.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Degrade a pristine 4-channel chunk in-place.
    ///
    /// `wave`: SemanticWave with 4 × time_steps complex amplitudes.
    /// `epoch_offset`: sample index offset for continuous AC phase.
    ///
    /// Returns the degraded wave (original is consumed).
    pub fn degrade(&mut self, wave: &mut SemanticWave, epoch_offset: usize) {
        let n = wave.amplitudes.len();
        let channels = 4usize;
        let time_steps = n / channels;

        for ch in 0..channels.min(n) {
            for t in 0..time_steps {
                let idx = ch * time_steps + t;
                if idx >= n { break; }

                // Thermal noise: N(0, σ²)
                let thermal = self.sample_gaussian() * self.thermal_sigma;

                // 50Hz AC interference: A·sin(2π·50·t_sec + φ_ch)
                let t_sec = (epoch_offset + t) as f64 / self.sampling_rate;
                let ac = self.ac_amplitude
                    * (std::f64::consts::TAU * 50.0 * t_sec + self.ac_phase_offsets[ch]).sin();

                // Inject into real component (EEG is real-valued)
                wave.amplitudes[idx] += Complex64::new(thermal + ac, 0.0);
            }
        }
    }

    /// Degrade and return a new wave (non-mutating convenience).
    pub fn degrade_copy(&mut self, wave: &SemanticWave, epoch_offset: usize) -> SemanticWave {
        let mut degraded = wave.clone();
        self.degrade(&mut degraded, epoch_offset);
        degraded
    }
}

// ============================================================================
// 2. NEWTON-RAPHSON PHASE ALIGNMENT (CIRCULAR MSE)
// ============================================================================
//
// Given:
//   - noisy 4-channel SemanticWave (degraded)
//   - golden 64-channel QMEM attractor (reference manifold)
//
// Find θ* such that the Phase Alignment Loss L(θ) between the phase-rotated
// noisy wave and the golden attractor is minimized.
//
//   L(θ) = Σ_i |z_noisy_i · e^{iθ} - z_golden_i|²
//
// Newton-Raphson iteration:
//   θ_{n+1} = θ_n - L'(θ_n) / L''(θ_n)

pub struct NewtonRaphsonAligner {
    pub max_iterations: usize,
    /// RELATİF Convergence threshold for |L'(θ)| / L(θ).
    pub epsilon: f64,
    pub damping: f64,
}

impl NewtonRaphsonAligner {
    pub fn new() -> Self {
        Self {
            max_iterations: 15,
            // BAŞ MİMAR & CLAUDE DÜZELTMESİ: Epsilon tekrar 1e-4'e çekildi.
            // Artık mutlak değere değil, hatanın kendisine oranına bakacağız.
            epsilon: 1e-4, 
            damping: 0.60,
        }
    }

    /// Compute Circular MSE loss and its first two derivatives w.r.t. θ.
    #[inline]
    fn phase_loss_derivatives(
        &self,
        noisy: &[Complex64],
        golden: &[Complex64],
        theta: f64,
    ) -> (f64, f64, f64) {
        let rot = Complex64::new(theta.cos(), theta.sin());
        let mut loss = 0.0f64;
        let mut loss_prime = 0.0f64;
        let mut loss_double_prime = 0.0f64;

        let len = noisy.len().min(golden.len());
        for i in 0..len {
            let z_rot = noisy[i] * rot;        // z_i · e^{iθ}
            let diff = z_rot - golden[i];       // error vector

            // L(θ) = Σ|diff|²
            loss += diff.norm_sqr();

            // L'(θ) = -2·Σ Im(conj(diff) · z_rot)
            let cross = diff.conj() * z_rot;
            loss_prime += -2.0 * cross.im;

            // L''(θ) ≈ 2·Σ [|z_i|² - Re(conj(g_i)·z_rot)]
            let g_cross = golden[i].conj() * z_rot;
            loss_double_prime += 2.0 * (noisy[i].norm_sqr() - g_cross.re);
        }

        (loss, loss_prime, loss_double_prime)
    }

    pub fn align(
        &self,
        noisy: &SemanticWave,
        golden: &SemanticWave,
    ) -> (SemanticWave, f64, usize, f64) {
        let n_slice = noisy.amplitudes.as_slice().unwrap();
        let g_slice = golden.amplitudes.as_slice().unwrap();

        // BAŞ MİMAR İNFAZI: ANALİTİK PROCRUSTES ÇÖZÜMÜ (EXACT ANALYTICAL SOLUTION)
        // Newton-Raphson kosinüs dalgasında kaosa giriyordu. Döngüyü tamamen sildik.
        // İki sinyalin çapraz-korelasyonunun açısı (argument), bizi anında global minimuma götürür.
        
        let mut cross_corr = num_complex::Complex64::new(0.0, 0.0);
        let len = n_slice.len().min(g_slice.len());
        
        for i in 0..len {
            // Çapraz-Korelasyon: Noisy * Conj(Golden)
            cross_corr += n_slice[i] * g_slice[i].conj();
        }

        // Analitik olarak en mükemmel faz açısını TIK diye buluyoruz (0 iterasyon):
        let theta = -cross_corr.arg(); 

        // Telemetri için sadece son kaybı hesaplayalım
        let rot = num_complex::Complex64::new(theta.cos(), theta.sin());
        let mut final_error = 0.0f64;
        for i in 0..len {
            let diff = (n_slice[i] * rot) - g_slice[i];
            final_error += diff.norm_sqr();
        }

        eprintln!("┌─────────────────────────────────────────────────────────────────────────────┐");
        eprintln!("│ [EXACT-ALIGN] NR Döngüsü İptal! Analitik Procrustes Çözümü Uygulandı.       │");
        eprintln!("│ Optimal θ* : {:>13.6} rad                                           │", theta);
        eprintln!("│ Final Loss  : {:>13.4e}                                              │", final_error);
        eprintln!("└─────────────────────────────────────────────────────────────────────────────┘");

        // Dalgayı kusursuz açıya döndür
        let aligned = noisy.rotated_phase(theta);
        
        // iters = 0, çünkü algoritma artık tahmin etmiyor, doğrudan BİLİYOR.
        (aligned, theta, 0, final_error)
    }
}

impl Default for NewtonRaphsonAligner {
    fn default() -> Self { Self::new() }
}

// ============================================================================
// 3. Z-SCORE OUTLIER GATING — Real-Time EMG Artifact Rejection
// ============================================================================
//
// Rolling EMA of amplitude magnitudes with online variance tracking.
// If Z = |x - μ| / σ > 3.0, the sample is classified as EMG artifact
// (jaw clench, blink, head movement) and bypassed.

/// Per-channel Z-Score gate with EMA statistics.
#[derive(Clone)]
pub struct ZScoreGate {
    /// EMA of amplitude magnitude.
    running_mean: f64,
    /// EMA of squared deviation (for variance).
    running_var: f64,
    /// EMA smoothing factor (α). Smaller = slower adaptation.
    alpha: f64,
    /// Samples seen (for warm-up gating).
    sample_count: u64,
    /// Z-Score rejection threshold (default: 3.0 = 3σ).
    pub threshold: f64,
    /// Count of rejected artifacts.
    pub artifact_count: u64,
}

impl ZScoreGate {
    pub fn new(alpha: f64, threshold: f64) -> Self {
        Self {
            running_mean: 0.0,
            running_var: 1.0,
            alpha,
            sample_count: 0,
            threshold,
            artifact_count: 0,
        }
    }

    /// Default gate: α=0.05 (20-sample window), 3σ threshold.
    pub fn default_3sigma() -> Self {
        Self::new(0.05, 3.0)
    }

    /// Update EMA statistics and return (is_artifact, z_score).
    ///
    /// Returns `true` if the sample should be REJECTED (EMG artifact).
    /// During warm-up (first 10 samples): always returns `false`.
    #[inline]
    pub fn gate(&mut self, amplitude: f64) -> (bool, f64) {
        self.sample_count = self.sample_count.saturating_add(1);

        // Update EMA
        let a = self.alpha;
        self.running_mean = (1.0 - a) * self.running_mean + a * amplitude;
        let deviation = amplitude - self.running_mean;
        self.running_var = (1.0 - a) * self.running_var + a * deviation * deviation;

        // Warm-up: accumulate baseline before gating
        if self.sample_count < 10 {
            return (false, 0.0);
        }

        let sigma = (self.running_var + 1e-12).sqrt();
        let z_score = deviation.abs() / sigma;

        let is_artifact = z_score > self.threshold;
        if is_artifact {
            self.artifact_count += 1;
        }

        (is_artifact, z_score)
    }

    /// Reset gate statistics (e.g., between sessions).
    pub fn reset(&mut self) {
        self.running_mean = 0.0;
        self.running_var = 1.0;
        self.sample_count = 0;
        self.artifact_count = 0;
    }
}

/// Multi-channel Z-Score gate array (one gate per EEG channel).
pub struct MultiChannelGate {
    pub gates: Vec<ZScoreGate>,
}

impl MultiChannelGate {
    pub fn new(num_channels: usize) -> Self {
        Self {
            gates: vec![ZScoreGate::default_3sigma(); num_channels],
        }
    }

    /// Gate an entire chunk. Returns `true` if ANY channel exceeds 3σ.
    pub fn gate_chunk(&mut self, amplitudes: &[f64]) -> bool {
        let mut any_artifact = false;
        for (ch, &amp) in amplitudes.iter().enumerate() {
            if ch < self.gates.len() {
                let (is_artifact, _z) = self.gates[ch].gate(amp);
                if is_artifact {
                    any_artifact = true;
                }
            }
        }
        any_artifact
    }

    pub fn reset_all(&mut self) {
        for g in &mut self.gates { g.reset(); }
    }

    pub fn total_artifacts(&self) -> u64 {
        self.gates.iter().map(|g| g.artifact_count).sum()
    }
}

// ============================================================================
// TAKENS PHASE-SPACE EMBEDDING
// ============================================================================
//
// Lifts a purely real discrete signal x[t] into C^N via the 1st-order
// derivative embedding:
//
//   z[t] = x[t]  +  i · (x[t+1] - x[t])
//
// This is a discrete approximation of the delay-coordinate embedding theorem
// (Takens, 1981). It guarantees that:
//
//   1. The imaginary part is non-zero whenever adjacent samples differ.
//   2. The Newton-Raphson Wronskian W'(θ) = -2·Σ Im[conj(diff)·z_rot]
//      is no longer identically zero on the zero-phase initial guess.
//   3. Phase alignment now reflects genuine topological differences between
//      the noisy AirPods signal and the clean golden reference.
//
// The function is O(N) with a single pass and zero heap allocations beyond
// the output SemanticWave.

/// Build a Takens phase-space embedding from a real SemanticWave.
///
/// For each sample t:
///   Re(z[t]) = x[t]
///   Im(z[t]) = x[t+1] - x[t]     (last element: Im = 0, boundary pad)
///
/// Input wave must be purely real (Im == 0); the crucible may have already
/// degraded the Re part — that is fine, we operate on the degraded values.
#[inline]
pub fn takens_embed(wave: &SemanticWave) -> SemanticWave {
    use num_complex::Complex64;

    let n = wave.amplitudes.len();
    let mut out = Vec::with_capacity(n);

    // SAFETY: Array1 created from from_reals is always contiguous.
    let slice = wave.amplitudes.as_slice().expect("non-contiguous SemanticWave");

    for t in 0..n {
        let x_t   = slice[t].re;
        // Derivative approximation — zero-pad at boundary
        let dx_dt = if t + 1 < n { slice[t + 1].re - x_t } else { 0.0 };
        out.push(Complex64::new(x_t, dx_dt));
    }

    SemanticWave::from_vec(out)
}

// ============================================================================
// BCI STREAMER — Unified Pipeline (existing API preserved)
// ============================================================================

pub struct BciStreamer {
    /// Raw 64-channel EEG epochs: [num_samples, 64, 256]
    pub eeg_waves: Tensor,
    /// Labels per epoch
    pub labels: Tensor,
    /// Current read index (circular buffer)
    pub current_idx: usize,
    /// Total number of samples
    pub num_samples: usize,
    /// Hardware Crucible for noise injection
    pub crucible: GaussianNoiseCrucible,
    /// Newton-Raphson phase aligner
    pub aligner: NewtonRaphsonAligner,
    /// Multi-channel Z-Score artifact gate
    pub artifact_gate: MultiChannelGate,
}

impl BciStreamer {
    pub fn new(path: &Path, device: &Device) -> Result<Self> {
        let safetensors = candle_core::safetensors::load(path, device)?;

        let eeg_waves = safetensors.get("eeg_data")
            .or_else(|| safetensors.get("data"))
            .or_else(|| safetensors.get("eeg_waves"))
            .ok_or_else(|| anyhow::anyhow!("No eeg_data, data, or eeg_waves tensor in safetensors"))?
            .clone();

        let labels = safetensors.get("labels")
            .ok_or_else(|| anyhow::anyhow!("No labels tensor in safetensors"))?
            .clone();

        let num_samples = eeg_waves.dim(0)?;

        Ok(Self {
            eeg_waves,
            labels,
            current_idx: 0,
            num_samples,
            crucible: GaussianNoiseCrucible::airpods_default(),
            aligner: NewtonRaphsonAligner::new(),
            artifact_gate: MultiChannelGate::new(4),
        })
    }

    pub fn len(&self) -> usize { self.num_samples }

    /// Fetch next ghost prefix with full Hardware Crucible pipeline:
    ///   1. Extract 4-channel AirPods simulation from 64-channel lab data
    ///   2. Z-Score gate → reject EMG artifacts
    ///   3. Gaussian Crucible → inject synthetic noise
    ///   4. Newton-Raphson → align phase to golden 64-ch attractor
    ///   5. Project to hidden_size-D ghost prefix
    pub fn fetch_next_ghost_prefix(&mut self, hidden_size: usize) -> Result<(Tensor, i32)> {
        if self.current_idx >= self.num_samples {
            self.current_idx = 0;
        }

        // [1, 136, 256] — single thought epoch (actual channel count from safetensors)
        let raw_thought = self.eeg_waves.narrow(0, self.current_idx, 1)?;
        let label = self.labels.to_vec1::<i64>()?[self.current_idx] as i32;

        // === STEP 1: AirPods 4-Channel Simulation (channels 15, 16, 50, 51) ===
        // narrow(dim, start, len): ch 15-16 = T7/P7 (left), ch 50-51 = T8/P8 (right)
        let left_ear  = raw_thought.narrow(1, 15, 2)?;  // [1, 2, 256]
        let right_ear = raw_thought.narrow(1, 50, 2)?;  // [1, 2, 256]
        let simulated_airpods = Tensor::cat(&[&left_ear, &right_ear], 1)?; // [1, 4, 256]

        // === STEP 2: Z-Score Artifact Gating ===
        let flat_4ch: Vec<f32> = simulated_airpods.flatten_all()?.to_vec1()?;
        let amplitudes_per_ch: Vec<f64> = (0..4)
            .map(|ch| {
                let start = ch * 256;
                let end = (start + 256).min(flat_4ch.len());
                flat_4ch[start..end].iter().map(|&v| (v as f64).abs()).sum::<f64>() / 256.0
            })
            .collect();

        let is_artifact = self.artifact_gate.gate_chunk(&amplitudes_per_ch);

        if is_artifact {
            eprintln!(
                "[BCI Z-GATE] EMG artifact detected at epoch {} (total: {}). Bypassing alignment.",
                self.current_idx, self.artifact_gate.total_artifacts()
            );
            let ghost_prefix = Tensor::zeros(
                (1, 1, hidden_size), DType::F32, simulated_airpods.device()
            )?;
            self.current_idx += 1;
            return Ok((ghost_prefix, label));
        }

        // === STEP 3: Gaussian Crucible — Inject Realistic Noise ===
        // BAŞ MİMAR FİX: UNIT MISMATCH (Birim Uyuşmazlığı)!
        // Ham veri Volt (1e-6) cinsinden. Crucible gürültüsü ise MikroVolt (15.0) cinsinden.
        // Veriyi 1,000,000 ile çarparak fiziksel donanım ölçeğine eşitliyoruz!
        let raw_clean: Vec<f64> = flat_4ch.iter().map(|&v| (v as f64) * 1_000_000.0).collect();
        let mut noisy_wave_real = SemanticWave::from_reals(raw_clean.clone());
        self.crucible.degrade(&mut noisy_wave_real, self.current_idx * 256);

        // === TAKENS PHASE-SPACE EMBEDDING ===
        // KANIT TESTİ: İşkence edilmiş sinyali bilerek 0.8 radyan fazından kaydırıyoruz.
        let noisy_embedded = takens_embed(&noisy_wave_real).rotated_phase(0.8);

    // Golden Reference (Temiz referans - o da artık MikroVolt ölçeğinde)
        let golden_wave_real = SemanticWave::from_reals(raw_clean);
        let golden_embedded  = takens_embed(&golden_wave_real);
        // === STEP 5: Newton-Raphson Phase Alignment on the Embedded Manifold ===
        let (aligned, theta, iters, error) = self.aligner.align(&noisy_embedded, &golden_embedded);

        eprintln!(
            "[BCI NR-ALIGN] theta*={:.6} rad | iters={} | Loss={:.4e} | epoch={}",
            theta, iters, error, self.current_idx
        );

        // === STEP 6: Topological Projection to hidden_size ===
        // Interleave Re and Im parts so both halves of the phase-space manifold
        // are encoded into the ghost prefix token.
        let aligned_reals: Vec<f32> = aligned.amplitudes.iter()
            .flat_map(|c| [c.re as f32, c.im as f32])
            .collect();
        let current_dim = aligned_reals.len();

        let flattened = Tensor::from_vec(
            aligned_reals, (1, current_dim), simulated_airpods.device()
        )?;

        let ghost_vector = if current_dim < hidden_size {
            let padding = Tensor::zeros(
                (1, hidden_size - current_dim), DType::F32, flattened.device()
            )?;
            Tensor::cat(&[&flattened, &padding], 1)?
        } else {
            flattened.narrow(1, 0, hidden_size)?
        };

        let ghost_prefix = ghost_vector.reshape((1, 1, hidden_size))?;

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
    fn test_gaussian_crucible_deterministic() {
        let mut crucible = GaussianNoiseCrucible::airpods_default();
        let reals = vec![0.0; 1024]; // 4 channels × 256 time steps
        let mut wave = SemanticWave::from_reals(reals);

        crucible.degrade(&mut wave, 0);

        // Noise injected: amplitudes should no longer be zero
        let energy: f64 = wave.amplitudes.iter().map(|c| c.norm_sqr()).sum();
        assert!(energy > 0.0, "Crucible must inject non-zero noise");
    }

    #[test]
    fn test_newton_raphson_convergence() {
        // Create a known signal and rotate it by a known angle
        let golden_data: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let golden = SemanticWave::from_reals(golden_data);

        let known_theta = 0.5; // radians
        let noisy = golden.rotated_phase(known_theta);

        let aligner = NewtonRaphsonAligner::new();
        let (_aligned, recovered_theta, iters, _error) = aligner.align(&noisy, &golden);

        // Newton-Raphson should recover θ within tolerance
        let theta_error = (recovered_theta + known_theta).abs(); // Should cancel
        assert!(theta_error < 0.01 || (theta_error - std::f64::consts::TAU).abs() < 0.01,
            "NR failed: recovered={:.4}, expected≈{:.4}, err={:.4}, iters={}",
            recovered_theta, -known_theta, theta_error, iters);
    }

    #[test]
    fn test_zscore_gate_artifact_detection() {
        let mut gate = ZScoreGate::default_3sigma();

        // Feed 20 normal samples to establish baseline
        for i in 0..20 {
            let val = 1.0 + 0.1 * (i as f64 * 0.3).sin();
            let (rejected, _z) = gate.gate(val);
            assert!(!rejected, "Normal sample should not be rejected during baseline");
        }

        // Feed a massive spike (jaw clench simulation)
        let (rejected, z) = gate.gate(100.0);
        assert!(rejected, "100µV spike must be rejected (z={:.2})", z);
        assert!(z > 3.0, "Z-score must exceed 3σ threshold");
        assert_eq!(gate.artifact_count, 1);
    }

    #[test]
    fn test_multichannel_gate() {
        let mut gate = MultiChannelGate::new(4);

        // Establish baseline with 15 normal chunks
        for _ in 0..15 {
            let normal = [1.0, 1.1, 0.9, 1.05];
            assert!(!gate.gate_chunk(&normal));
        }

        // One channel spikes
        let spike = [1.0, 1.1, 500.0, 1.05];
        assert!(gate.gate_chunk(&spike), "Spike on channel 3 must trigger gate");
    }
}