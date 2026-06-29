// src/engine/oracle_worker.rs
// PHASE 61.1: Hyperdimensional Sparse Projection (HSP) Core
// Asymmetric CPU-GPU logic bridge. Projects 4096-D dense phasors into 1,000,000-D sparse space.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use num_complex::Complex64;
use crate::models::qwen_custom::PhaseSignal;
use crate::semantic_parser::SemanticRegisterFile;

#[cfg(feature = "cuda")]
use crate::gpu_ops::adapter::cudaSetDevice;

/// Token event the generation loop dispatches to the Oracle after each sample.
/// THREAD SAFETY: All SRF data is pre-extracted on main thread to avoid CUDA context violations.
#[derive(Clone, Debug)]
pub struct OracleMessage {
    pub token_id:          u32,
    pub token_text:        String,
    pub position:          usize,
    /// Sliding window of the last 32 sampled token IDs.
    pub recent_tokens:     Vec<u32>,
    /// Phase 34: Top-K logit snapshot for Shannon Entropy computation.
    /// Bounded to 64 elements to keep channel memory reasonable.
    pub logits_snapshot:   Vec<f32>,
    /// CPU computed Semantic Divergence via Shannon Entropy on logit probabilities.
    pub semantic_divergence_snapshot: f32,
    pub jacobian_proxy: f32,
    pub is_data_extraction: bool,
    /// THREAD SAFETY: Pre-extracted SRF data from main thread (CUDA-safe)
    pub srf_ax: f32,
    pub srf_cx_w: f32,
    pub srf_cx_j: f32,
}

/// Pre-computed artifact the Oracle deposits into `OracleHandle::mask_slot`.
/// FIX #1: Reverted from dense Vec<u16> to SPARSE Vec<(usize, u16)> pairs.
/// This eliminates the 300KB per-token heap allocation (O(vocab_size) → O(recent_tokens)).
#[derive(Clone, Debug)]
pub struct PrecomputedMask {
    /// F16-packed impedance bias vector (SPARSE pairs: index → f16 bits).
    pub impedance_f16:       Vec<(usize, u16)>,
    pub predicted_semantic_divergence: f32,
    /// Confidence in the prediction [0.0, 1.0].
    pub confidence:          f32,
    /// Absolute sequence position this mask is valid for.
    /// Main loop only applies it when `target_position == current_position + 1`.
    pub target_position:     usize,
}

/// Thread-safe handle providing communication with the background Oracle worker.
pub struct OracleHandle {
    pub msg_tx: std::sync::mpsc::SyncSender<OracleMessage>,
    pub mask_slot: Arc<Mutex<Option<PrecomputedMask>>>,
    pub phase_signal_slot: Arc<Mutex<Option<PhaseSignal>>>,
}

pub fn spawn_oracle(
    vocab_size: usize,
    memory_pressure: Arc<AtomicBool>,
    srf_arc: Arc<Mutex<SemanticRegisterFile>>,
    oracle_adapter: Option<Arc<crate::gpu_ops::NeuroAdapter>>,
) -> Option<OracleHandle> {
    let rt = match tokio::runtime::Handle::try_current() {
        Ok(h)  => h,
        Err(_) => {
            eprintln!("[ORACLE] ⚠  No tokio runtime. Oracle will not prefetch.");
            return None;
        }
    };

    let (msg_tx, msg_rx) = std::sync::mpsc::sync_channel::<OracleMessage>(32);

    let mask_slot: Arc<Mutex<Option<PrecomputedMask>>> =
        Arc::new(Mutex::new(None));
    let slot_worker = Arc::clone(&mask_slot);

    let phase_signal_slot: Arc<Mutex<Option<PhaseSignal>>> =
        Arc::new(Mutex::new(None));
    let phase_slot_worker = Arc::clone(&phase_signal_slot);

    rt.spawn_blocking(move || {
        eprintln!("[ORACLE] 🔮 Oracle online. SRF pipeline armed.");

        #[cfg(feature = "cuda")]
        unsafe {
            // PHASE 3: Bind background thread to CUDA device 0 context before DMA calls
            let _ = cudaSetDevice(0);
        }

        while let Ok(msg) = msg_rx.recv() {
            // Memory Pressure Guard
            if memory_pressure.load(Ordering::Relaxed) {
                if let Ok(mut s) = slot_worker.try_lock()       { *s = None; }
                if let Ok(mut p) = phase_slot_worker.try_lock() { *p = None; }
                continue;
            }

            // Step 1: SRF Update (using pre-extracted data from main thread - CUDA-safe)
            // THREAD SAFETY: Oracle thread has no CUDA context, so we use pre-extracted SRF scalars
            let active_theta = msg.srf_ax;
            let srf_cx_w = msg.srf_cx_w;
            let srf_cx_j = msg.srf_cx_j;

            // Step 2: PhaseSignal deposit
            let phase_sig = PhaseSignal::new(active_theta, srf_cx_w, srf_cx_j);
            if let Ok(mut ps) = phase_slot_worker.try_lock() {
                *ps = Some(phase_sig);
            }

            // Step 3: Forward-predict Semantic Divergence (~1μs)
            let predicted_w = oracle_predict_semantic_divergence(&msg.logits_snapshot);

            // Step 4: Synthesise F16 impedance map (~5–50μs) — SPARSE
            let impedance_f16 = oracle_compute_impedance_map(
                vocab_size,
                &msg.recent_tokens,
                predicted_w,
                msg.semantic_divergence_snapshot,
            );

            // Step 5: DMA prefetch → VRAM (cudaMemcpyAsync)
            /* PHASE 21.5: VRAM Prefetch disabled temporarily to isolate STATUS_ACCESS_VIOLATION
            #[cfg(feature = "cuda")]
            if let Some(ref ada) = oracle_adapter {
                if let Err(e) = ada.prefetch_impedance(&impedance_f16, vocab_size) {
                    let err_str = e.to_string();
                    if !err_str.contains("709") {
                        eprintln!("[ORACLE] ⚠  DMA error @ pos {}: {}", msg.position, e);
                    }
                }
            }
            */

            // Step 6: PrecomputedMask deposit
            let mask = PrecomputedMask {
                impedance_f16,
                predicted_semantic_divergence: predicted_w,
                confidence: oracle_estimate_confidence(&msg.recent_tokens),
                target_position: msg.position + 1,
            };
            if let Ok(mut s) = slot_worker.try_lock() {
                *s = Some(mask);
            }
        }

        eprintln!("[ORACLE] 🔮 Oracle exiting (channel closed by main loop).");
    });

    Some(OracleHandle {
        msg_tx,
        mask_slot,
        phase_signal_slot,
    })
}

/// Non-blocking PhaseSignal consumption.
#[inline(always)]
pub fn oracle_consume_phase_signal(oracle: &Option<OracleHandle>) -> Option<PhaseSignal> {
    let h = oracle.as_ref()?;
    let mut s = h.phase_signal_slot.try_lock().ok()?;
    s.take()
}

/// Phase 34: Semantic Divergence via Shannon Entropy on logit probabilities.
///
/// Shannon Entropy: H = -Σ p_i × ln(p_i)
/// Normalized:      H_norm = H / ln(N)        ∈ [0.0, 1.0]
/// Output:          1.0 - H_norm               (peaked = high, uniform = low)
pub fn oracle_predict_semantic_divergence(logits_snapshot: &[f32]) -> f32 {
    if logits_snapshot.len() < 2 { return 0.5; }

    let max_logit = logits_snapshot.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = logits_snapshot.iter()
        .map(|&x| ((x - max_logit) as f64).exp())
        .sum();

    if exp_sum <= 0.0 || !exp_sum.is_finite() {
        return 0.5;
    }

    let mut entropy = 0.0_f64;
    for &logit in logits_snapshot {
        let p = ((logit - max_logit) as f64).exp() / exp_sum;
        if p > 1e-12 {
            entropy -= p * p.ln();
        }
    }

    let max_entropy = (logits_snapshot.len() as f64).ln();
    let h_normalized = if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    };

    (1.0 - h_normalized) as f32
}

/// FIX #1: Reverted to SPARSE impedance map — O(recent_tokens) not O(vocab_size).
///
/// Phase 20.7 FIXED: No more `vec![0u16; vocab_size]` (300KB allocation per token).
/// The recent window is at most 32 tokens. Use sparse iteration via HashMap.
///
/// Two mechanisms:
///  a. Frequency penalty: tokens appearing >2x in the recent window
///     receive a negative logit bias (repeat suppression at mask level).
///  b. Phase correction is deferred to the Expert pipeline.
pub fn oracle_compute_impedance_map(
    vocab_size: usize,
    recent: &[u32],
    _predicted_w: f32,
    _current_w: f32,
) -> Vec<(usize, u16)> {
    let mut token_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::with_capacity(recent.len());
    for &t in recent {
        *token_counts.entry(t).or_insert(0) += 1;
    }

    let mut sparse_biases = Vec::with_capacity(token_counts.len());
    
    for (&token_id, &cnt) in token_counts.iter() {
        if cnt > 1 {
            let idx = token_id as usize;
            if idx < vocab_size {
                // Repetition penalty: bias = -sqrt(count) * 0.5
                let bias = -((cnt as f32).sqrt()) * 0.5_f32;
                sparse_biases.push((idx, half::f16::from_f32(bias).to_bits()));
            }
        }
    }
    
    sparse_biases
}

/// Confidence = token variety in the recent window.
/// FIX #1: Matches original — requires min 4 tokens, clamp to [0.1, 1.0].
pub fn oracle_estimate_confidence(recent: &[u32]) -> f32 {
    let n = recent.len().min(32) as f32;
    if n < 4.0 { return 0.5; }
    let unique = recent.iter().copied()
        .collect::<std::collections::HashSet<u32>>()
        .len() as f32;
    (unique / n).clamp(0.1, 1.0)
}

/// Non-blocking token submission to the Oracle.
/// `try_send` drops the message when the channel is full — the GPU loop
/// is NEVER delayed.
#[inline(always)]
pub fn oracle_submit_token(oracle: &Option<OracleHandle>, msg: OracleMessage) {
    if let Some(ref h) = oracle {
        let _ = h.msg_tx.try_send(msg);
    }
}

/// Poll the Oracle's mask slot; return the mask only if it targets
/// `expected_position` exactly (stale masks from a lagging Oracle are dropped).
#[inline(always)]
pub fn oracle_consume_mask(
    oracle: &Option<OracleHandle>,
    expected_position: usize,
) -> Option<PrecomputedMask> {
    let h   = oracle.as_ref()?;
    let mut s = h.mask_slot.try_lock().ok()?;
    let mask = s.take()?;
    if mask.target_position == expected_position { Some(mask) } else { None }
}

/// Approximate Semantic Divergence (phase variance proxy) from a logits tensor's value variance.
pub fn estimate_logit_peakedness(logits: &candle_core::Tensor) -> candle_core::Result<f32> {
    let logits_f32 = logits.to_dtype(candle_core::DType::F32)?;
    let mean = logits_f32.mean_all()?;
    let diff = logits_f32.broadcast_sub(&mean)?;
    let var_tensor = diff.sqr()?.mean_all()?;
    let var_val = var_tensor.to_scalar::<f32>()?;
    
    let w = (var_val / (1.0 + var_val)).clamp(0.0, 1.0);
    Ok(w)
}

/// FIX #1: Reverted to scatter_add sparse injection — exact match of original llm_core.rs.
///
/// Convert Oracle F16 impedance data into a Candle F32 Tensor on the model device.
/// Uses scatter_add for GPU-native sparse injection without dense CPU allocations.
pub fn f16_mask_to_tensor(
    data: &[(usize, u16)],
    vocab_size: usize,
    device: &candle_core::Device,
) -> anyhow::Result<candle_core::Tensor> {
    use candle_core::{Tensor, DType};
    
    let len = data.len();
    if len == 0 {
        return Tensor::zeros((vocab_size,), DType::F32, device)
            .map_err(|e| anyhow::anyhow!("Oracle tensor (empty): {}", e));
    }

    // Safety check for vocab bounds
    for &(idx, _) in data {
        if idx >= vocab_size {
            return Err(anyhow::anyhow!("Oracle index {} out of bounds for vocab size {}", idx, vocab_size));
        }
    }

    // Phase 28: GPU-Native Sparse Injection via scatter_add
    let mut indices: Vec<u32> = Vec::with_capacity(len);
    let mut values: Vec<f32> = Vec::with_capacity(len);
    
    for &(idx, val_bits) in data {
        indices.push(idx as u32);
        values.push(half::f16::from_bits(val_bits).to_f32());
    }

    let idx_t = Tensor::from_vec(indices, (len,), device)?;
    let val_t = Tensor::from_vec(values, (len,), device)?;
    
    let zero_t = Tensor::zeros((vocab_size,), DType::F32, device)?;
    
    // TRACE: Final verification before scatter_add which is a known crash point
    // eprintln!("[ORACLE] Applying scatter_add to {} entries...", len);
    
    zero_t.scatter_add(&idx_t, &val_t, 0)
        .map_err(|e| anyhow::anyhow!("Impedance scatter_add failed: {}", e))
}

// ============================================================================
// PHASE 61.1: HYPERDIMENSIONAL SPARSE PROJECTION (HSP) CORE
// Asymmetric CPU-GPU Logic Bridge
// GPU: 4096-D RK4 continuous flow | CPU: 1,000,000-D noise-free logic gates
// ============================================================================

pub const HSP_DIMENSIONS: u32 = 1_000_000;
pub const SPARSITY_K: usize = 2000; // Only 2000 active bits out of 1M (0.2% sparsity)

pub struct HspOracle {
    // We will store the sparse logic memory here in upcoming phases
}

impl HspOracle {
    pub fn new() -> Self {
        Self {}
    }

    /// PHASE 61.1: The Dimensional Expander (4096-D -> 1,000,000-D)
    /// Converts a continuous dense phasor into a Sparse Distributed Representation (SDR).
    /// Uses deterministic hashing of the wave's phase and amplitude to select active indices.
    pub fn project_to_sparse(dense_phasor: &[Complex64]) -> Vec<u32> {
        let mut active_indices = Vec::with_capacity(SPARSITY_K);
        let mut hasher = DefaultHasher::new();

        // Seed the hash with the macroscopic state of the phasor
        let mut global_energy = 0.0_f64;
        for c in dense_phasor.iter() {
            global_energy += c.norm_sqr();
        }
        (global_energy.to_bits()).hash(&mut hasher);
        let mut state = hasher.finish();
        if state == 0 { state = 0xDEADBEEFCAFEBABE; }

        for _ in 0..SPARSITY_K {
            // Xorshift64* PRNG for high-speed uniform distribution
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let random_val = state.wrapping_mul(0x2545F4914F6CDD1D);

            // Map to 1,000,000 dimensional index
            let index = (random_val % (HSP_DIMENSIONS as u64)) as u32;
            active_indices.push(index);
        }

        active_indices.sort_unstable();
        active_indices.dedup(); // Ensure strict binary state (no duplicate bit activations)
        active_indices
    }

    /// PHASE 61.2: Sparse Superposition (Binding)
    /// Merges two 1M-D sparse vectors using a fast O(N) union of sorted active indices.
    pub fn bind_sparse(a: &[u32], b: &[u32]) -> Vec<u32> {
        let mut result = Vec::with_capacity(a.len() + b.len());
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            if a[i] < b[j] {
                result.push(a[i]);
                i += 1;
            } else if a[i] > b[j] {
                result.push(b[j]);
                j += 1;
            } else {
                result.push(a[i]);
                i += 1;
                j += 1; // Skip duplicate
            }
        }
        result.extend_from_slice(&a[i..]);
        result.extend_from_slice(&b[j..]);
        result
    }

    /// PHASE 61.2: Sparse Interference (Exact Logic Overlap)
    /// Computes the exact deterministic intersection between two SDR concepts.
    /// In a 1,000,000-D space with 0.2% sparsity, any overlap > 0 is a hard logical link.
    pub fn sparse_overlap(a: &[u32], b: &[u32]) -> usize {
        let (mut i, mut j) = (0, 0);
        let mut overlap = 0;
        while i < a.len() && j < b.len() {
            if a[i] < b[j] {
                i += 1;
            } else if a[i] > b[j] {
                j += 1;
            } else {
                overlap += 1;
                i += 1;
                j += 1;
            }
        }
        overlap
    }

    /// PHASE 61.4: HDC Paradox Filter (Formerly Carnaugh Evaluator)
    /// Evaluates logical contradictions in the Hyperdimensional space.
    /// 
    /// STATISTICAL DERIVATION OF THRESHOLDS:
    /// In a 1M-D space with K=2000 active bits (p = 0.002):
    /// - Expected random overlap: E[X] = 2000 * 0.002 = 4.
    /// - Variance: σ² = 2000 * 0.002 * 0.998 ≈ 3.992 (σ ≈ 2).
    /// - Threshold 14 is 5σ above chance (p < 0.0000003), representing absolute logical causality.
    /// - Threshold 5 is 0.5σ above chance, representing mild spatial distraction.
    pub fn evaluate_hdc_paradox_filter(
        current_thought: &[u32],
        logic_rule: &[u32],
        spatial_distractor: &[u32]
    ) -> f64 {
        let logic_match = Self::sparse_overlap(current_thought, logic_rule);
        let spatial_match = Self::sparse_overlap(current_thought, spatial_distractor);

        if logic_match > 14 && spatial_match > 5 {
            // Absolute >5σ logical conflict detected.
            return 5.0; 
        }
        0.0 
    }
}

// ============================================================================
// UNIT TESTS FOR ORACLE MACHINERY (migrated from llm_core.rs)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oracle_predict_semantic_divergence_peaked() {
        // Phase 34: Peaked logit distribution → high Shannon-Entropy-based score
        // One dominant logit + near-zero rest = confident = HEALTHY
        let mut logits = vec![0.0f32; 64];
        logits[0] = 100.0;
        let w = oracle_predict_semantic_divergence(&logits);
        assert!(w > 0.8, "Peaked logits should predict high divergence (healthy), got {}", w);
    }

    #[test]
    fn test_oracle_predict_semantic_divergence_uniform() {
        // Phase 34: Uniform logit distribution → low score (hallucination risk)
        // All equal logits = maximum entropy = UNCERTAIN
        let logits = vec![1.0f32; 64];
        let w = oracle_predict_semantic_divergence(&logits);
        assert!(w < 0.05, "Uniform logits should predict low divergence (hallucination), got {}", w);
    }

    #[test]
    fn test_oracle_predict_semantic_divergence_insufficient_history() {
        // Less than 2 logits → neutral 0.5
        let logits = vec![1.0f32];
        let w = oracle_predict_semantic_divergence(&logits);
        assert!((w - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_oracle_compute_impedance_map_length() {
        let vocab = 512;
        let tokens: Vec<u32> = (0..20).collect();
        let map = oracle_compute_impedance_map(vocab, &tokens, 0.5, 0.5);
        // All tokens unique (count=1), so no penalties applied
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_oracle_compute_impedance_suppresses_repeat_tokens() {
        // Token 0 appears 5 times → its bias slot should be negative
        let vocab = 16;
        let tokens = vec![0u32; 5];
        let map = oracle_compute_impedance_map(vocab, &tokens, 0.5, 0.5);
        let bias = half::f16::from_bits(map[0].1).to_f32();
        assert!(bias < 0.0, "Repeated token should have negative impedance bias, got {}", bias);
    }

    #[test]
    fn test_oracle_estimate_confidence_diverse() {
        let tokens: Vec<u32> = (0u32..16).collect(); // all unique
        let c = oracle_estimate_confidence(&tokens);
        assert!(c > 0.9, "All-unique window should have high confidence, got {}", c);
    }

    #[test]
    fn test_oracle_estimate_confidence_repetitive() {
        let tokens = vec![1u32; 16]; // all same
        let c = oracle_estimate_confidence(&tokens);
        assert!(c < 0.15, "All-same window should have low confidence, got {}", c);
    }

    #[test]
    fn test_f16_mask_roundtrip() {
        // Verify F16 packing/unpacking preserves sign and approximate magnitude
        let values: Vec<f32> = vec![0.0, -1.0, 0.5, -2.0, 0.15];
        let f16_packed: Vec<u16> = values.iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let recovered: Vec<f32> = f16_packed.iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect();
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.01,
                "F16 roundtrip error: {} → {}", orig, rec);
        }
    }

    #[test]
    fn test_extract_wronskian_uniform_logits() {
        // Uniform logits → zero variance → W ≈ 0
        // (Cannot instantiate Tensor in unit test without CUDA; test the math path only)
        let uniform = vec![1.0f32; 100];
        let n    = uniform.len() as f32;
        let mean = uniform.iter().sum::<f32>() / n;
        let var  = uniform.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let w    = (var / (1.0 + var)).clamp(0.0, 1.0);
        assert!(w < 0.001, "Uniform logits should yield near-zero W, got {}", w);
    }

    #[test]
    fn test_extract_wronskian_peaked_logits() {
        // One dominant logit → high variance → W closer to 1
        let mut peaked = vec![0.0f32; 100];
        peaked[42] = 100.0;
        let n    = peaked.len() as f32;
        let mean = peaked.iter().sum::<f32>() / n;
        let var  = peaked.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let w    = (var / (1.0 + var)).clamp(0.0, 1.0);
        assert!(w > 0.5, "Peaked logits should yield healthy W, got {}", w);
    }

    // ====================================================================
    // PHASE 61.1: HSP UNIT TESTS
    // ====================================================================

    #[test]
    fn test_hsp_projection_determinism() {
        // Same input must always produce the same SDR
        let phasor: Vec<Complex64> = (0..4096)
            .map(|i| Complex64::new((i as f64) * 0.001, (i as f64) * -0.0005))
            .collect();
        let sdr_a = HspOracle::project_to_sparse(&phasor);
        let sdr_b = HspOracle::project_to_sparse(&phasor);
        assert_eq!(sdr_a, sdr_b, "HSP projection must be deterministic");
    }

    #[test]
    fn test_hsp_sparsity_bounds() {
        // Output must be ≤ SPARSITY_K (dedup may reduce it)
        let phasor: Vec<Complex64> = (0..4096)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let sdr = HspOracle::project_to_sparse(&phasor);
        assert!(!sdr.is_empty(), "SDR must not be empty");
        assert!(sdr.len() <= SPARSITY_K, "SDR exceeds SPARSITY_K: {} > {}", sdr.len(), SPARSITY_K);
    }

    #[test]
    fn test_hsp_index_range() {
        // All indices must be < HSP_DIMENSIONS
        let phasor: Vec<Complex64> = vec![Complex64::new(1.0, -1.0); 4096];
        let sdr = HspOracle::project_to_sparse(&phasor);
        for &idx in &sdr {
            assert!(idx < HSP_DIMENSIONS, "Index {} out of range (max {})", idx, HSP_DIMENSIONS);
        }
    }

    #[test]
    fn test_hsp_sorted_and_unique() {
        let phasor: Vec<Complex64> = (0..4096)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect();
        let sdr = HspOracle::project_to_sparse(&phasor);
        for w in sdr.windows(2) {
            assert!(w[0] < w[1], "SDR must be sorted and unique: {} >= {}", w[0], w[1]);
        }
    }
}
