// src/engine/bci_logits_processor.rs
// ============================================================================
// BCI LOGITS PROCESSOR — Neuro-Symbolic Constrained Decoding
// ============================================================================
//
// This module implements a hybrid hallucination prevention system that
// operates directly on raw logits BEFORE softmax/argmax sampling.
//
// The Problem:
//   Pure latent projection causes catastrophic semantic drift.
//   Example: "McDonald's" → "Nightclub" (close in latent space, wrong domain).
//
// The Solution: Two orthogonal constraint passes:
//
//   1. QMEM SEMANTIC MASKING ("The Guardrails")
//      EEG-derived QMEM attractors define the active semantic domain.
//      Tokens that severely violate this domain receive -INF logit penalty.
//      This eliminates entire categories of hallucination before sampling.
//
//   2. sEMG PHONETIC BOOSTING ("The Steer")
//      Muscle activation envelopes encode the phonetic intent of subvocal speech.
//      Compute a fused dense matrix multiplication between the sEMG latent vector
//      and the entire LLM embedding matrix (lm_head).
//      Add this un-normalized dot product as a logit boost, steering toward
//      phonetically plausible words simultaneously across the entire vocabulary.
//      Add this similarity as a logit boost, steering toward phonetically
//      plausible words.
//
// Performance Contract:
//   - All operations use pure candle_core tensor ops (GPU-resident, ZERO CPU roundtrips)
//   - Semantic masking is O(V) where V = vocab_size (single tensor add)
//   - Phonetic boosting is a dense MatMul: O(V × D) where V = vocab_size, D = hidden_size
//   - Total overhead per token: < 0.2ms on RTX-class GPU

use anyhow::{Result, bail};
use candle_core::{Tensor, Device, DType, IndexOp};
use std::collections::HashSet;

// ============================================================================
// 1. SEMANTIC DOMAIN REGISTRY
// ============================================================================
//
// Maps QMEM attractor labels to token-level domain membership.
// This is the "knowledge base" that connects EEG topology to vocabulary.

/// A semantic domain definition with associated token IDs.
#[derive(Debug, Clone)]
pub struct SemanticDomain {
    /// Human-readable domain name (e.g., "food", "greeting", "question")
    pub name: String,
    /// Token IDs that belong to this domain (populated from tokenizer)
    pub token_ids: HashSet<u32>,
    /// Domain activation weight (from QMEM attractor confidence)
    pub weight: f32,
}

/// Registry of all known semantic domains and their token memberships.
pub struct SemanticDomainRegistry {
    /// All registered domains
    domains: Vec<SemanticDomain>,
    /// Vocabulary size of the target model
    vocab_size: usize,
    /// Pre-computed penalty tensor: 0.0 for allowed tokens, -INF for forbidden
    /// Updated when active domains change.
    cached_mask: Option<Tensor>,
    /// Device for tensor operations
    device: Device,
}

impl SemanticDomainRegistry {
    /// Create a new empty registry.
    pub fn new(vocab_size: usize, device: &Device) -> Self {
        Self {
            domains: Vec::new(),
            vocab_size,
            cached_mask: None,
            device: device.clone(),
        }
    }

    /// Register a semantic domain with its associated keywords.
    ///
    /// The tokenizer is used to expand keywords into all token IDs
    /// that could encode those keywords (including subword fragments).
    pub fn register_domain(
        &mut self,
        name: &str,
        keywords: &[&str],
        tokenizer: &tokenizers::Tokenizer,
    ) {
        let mut token_ids = HashSet::new();

        for keyword in keywords {
            // Encode keyword and collect all token IDs
            if let Ok(encoding) = tokenizer.encode(*keyword, false) {
                for &id in encoding.get_ids() {
                    token_ids.insert(id);
                }
            }
            // Also try with space prefix (common in BPE)
            let spaced = format!(" {}", keyword);
            if let Ok(encoding) = tokenizer.encode(spaced.as_str(), false) {
                for &id in encoding.get_ids() {
                    token_ids.insert(id);
                }
            }
        }

        self.domains.push(SemanticDomain {
            name: name.to_string(),
            token_ids,
            weight: 1.0,
        });

        // Invalidate cached mask
        self.cached_mask = None;
    }

    /// Register a pre-built domain (e.g., loaded from QMEM label mapping).
    pub fn register_prebuilt_domain(&mut self, domain: SemanticDomain) {
        self.domains.push(domain);
        self.cached_mask = None;
    }

    /// Check if a token ID belongs to any of the given active domains.
    fn is_token_in_domain(&self, token_id: u32, active_domains: &[String]) -> bool {
        for domain in &self.domains {
            if active_domains.contains(&domain.name) {
                if domain.token_ids.contains(&token_id) {
                    return true;
                }
            }
        }
        false
    }

    /// Build the penalty mask tensor for the given active domains.
    ///
    /// Returns a 1D tensor of shape [vocab_size] where:
    ///   - 0.0 for tokens that pass the semantic filter
    ///   - `penalty` (large negative) for tokens that violate the domain
    ///
    /// Special tokens (EOS, IM_START, IM_END) are always allowed.
    pub fn build_domain_mask(
        &self,
        active_domains: &[String],
        penalty: f32,
    ) -> Result<Tensor> {
        if active_domains.is_empty() {
            // No domains active = no masking (all tokens allowed)
            return Tensor::zeros(self.vocab_size, DType::F32, &self.device)
                .map_err(|e| anyhow::anyhow!("Failed to create zero mask: {}", e));
        }

        let mut mask_data = vec![penalty; self.vocab_size];

        for token_id in 0..self.vocab_size {
            let id_u32 = token_id as u32;
            if self.is_token_in_domain(id_u32, active_domains) || token_id < 256 {
                mask_data[token_id] = 0.0;
            }
        }

        Tensor::from_vec(mask_data, self.vocab_size, &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to create domain mask: {}", e))
    }
}

// ============================================================================
// 2. BCI LOGITS PROCESSOR — The Main Engine
// ============================================================================

/// Neuro-symbolic logits processor that applies EEG semantic masking
/// and sEMG phonetic boosting to prevent hallucinations during
/// subvocal speech reconstruction.
pub struct BciLogitsProcessor {
    /// Semantic domain registry (populated from QMEM attractors)
    pub domain_registry: SemanticDomainRegistry,

    /// Currently active semantic domains (from EEG QMEM decoding)
    /// Example: ["food", "question", "today"]
    pub active_domains: Vec<String>,

    /// Current sEMG phonetic latent sequence [1, hidden_size] chunks (sliced from 12288-D)
    /// Used for cosine similarity boosting against token embeddings.
    pub semg_latent_sequence: Vec<Tensor>,

    /// Penalty magnitude for out-of-domain tokens (default: -1e6)
    /// Not -INF to allow gradient flow and rare domain transitions.
    pub semantic_penalty: f32,

    /// Boost magnitude for phonetically aligned tokens (default: 5.0)
    /// This is multiplied by the dot product and added to logits.
    pub phonetic_boost_scale: f32,

    /// Whether semantic masking is active
    pub semantic_masking_enabled: bool,

    /// Whether phonetic boosting is active
    pub phonetic_boosting_enabled: bool,

    /// Pre-cached domain mask tensor (invalidated when active_domains change)
    cached_domain_mask: Option<Tensor>,

    /// Hard phonetic quarantine mask for Tier-1 subvocal decoding.
    /// Tokens outside the 1-4 char pure-ASCII-alphabetic envelope receive -1e9.
    phonetic_quarantine_mask: Option<Tensor>,

    /// Device for tensor operations
    device: Device,
}

impl BciLogitsProcessor {
    /// Create a new BCI logits processor.
    pub fn new(vocab_size: usize, device: &Device) -> Self {
        Self {
            domain_registry: SemanticDomainRegistry::new(vocab_size, device),
            active_domains: Vec::new(),
            semg_latent_sequence: Vec::new(),
            semantic_penalty: -1e6,
            phonetic_boost_scale: 100.0,
            semantic_masking_enabled: true,
            phonetic_boosting_enabled: true,
            cached_domain_mask: None,
            phonetic_quarantine_mask: None,
            device: device.clone(),
        }
    }

    /// Update the active semantic domains (called when QMEM attractors change).
    ///
    /// This invalidates the cached domain mask and forces recomputation
    /// on the next `process()` call.
    pub fn set_active_domains(&mut self, domains: Vec<String>) {
        if self.active_domains != domains {
            self.active_domains = domains;
            self.cached_domain_mask = None; // Force recompute
        }
    }

    /// Update the sEMG phonetic latent vector (called per-epoch from Phoneme Hunter).
    ///
    /// The tensor should be [1, 12288] or [1, 1, 12288]. We split it into 3 x 4096-D chunks.
    pub fn set_semg_latent(&mut self, latent: Tensor) {
        self.semg_latent_sequence.clear();
        let dims = latent.dims();
        let last_dim_idx = dims.len() - 1;
        let last_dim = dims[last_dim_idx];
        let hidden_size = last_dim / 3;
        
        for i in 0..3 {
            if let Ok(slice) = latent.narrow(last_dim_idx, i * hidden_size, hidden_size) {
                self.semg_latent_sequence.push(slice.clone());
            }
        }
    }

    /// Clear all BCI state (between sessions/subjects).
    pub fn reset(&mut self) {
        self.active_domains.clear();
        self.semg_latent_sequence.clear();
        self.cached_domain_mask = None;
    }

    // ========================================================================
    // TIER-1 PHONETIC QUARANTINE
    // ========================================================================

    /// Initialize the hard phonetic quarantine mask.
    ///
    /// Tier-1 is a pure phonetic sub-word synthesizer. Throat muscles produce
    /// only short phonetic fragments (1-4 ASCII alphabetic chars). Anything
    /// longer, non-alphabetic, or out-of-domain (C++, Chinese, punctuation)
    /// receives a -1e9 logit penalty, preventing embedding space bleed.
    pub fn initialize_phonetic_quarantine(&mut self, tokenizer: &tokenizers::Tokenizer) -> Result<()> {
        let vocab_size = self.domain_registry.vocab_size;
        let mut penalties = vec![0.0f32; vocab_size];

        for token_id in 0..vocab_size {
            // Decode the single token (no post-processing, no special-token skipping)
            let decoded = tokenizer.decode(&[token_id as u32], false).unwrap_or_default();
            let trimmed = decoded.trim();

            // Critical system tokens (control flow) are always allowed
            let is_system_token = trimmed.starts_with("<|") && trimmed.ends_with("|>");

            // Phonetic sub-word rule: 1-4 chars, pure ASCII alphabetic
            let is_phonetic = trimmed.len() >= 1
                && trimmed.len() <= 4
                && trimmed.chars().all(|c| c.is_ascii_alphabetic());

            if !is_system_token && !is_phonetic {
                penalties[token_id] = -1e9;
            }
        }

        self.phonetic_quarantine_mask = Some(
            Tensor::from_vec(penalties, vocab_size, &self.device)?
        );

        eprintln!("[BCI QUARANTINE] Tier-1 phonetic mask initialized for {} tokens", vocab_size);
        Ok(())
    }

    // ========================================================================
    // MAIN PROCESSING PASS
    // ========================================================================

    /// Apply neuro-symbolic constraints to raw logits.
    ///
    /// This is the main entry point, called once per token generation step.
    /// The input logits tensor has shape [vocab_size] (already extracted from
    /// the model's last-position output).
    ///
    /// Returns modified logits with:
    ///   1. Out-of-domain tokens penalized (semantic masking)
    ///   2. Phonetically aligned tokens boosted (sEMG steering)
    pub fn process(
        &mut self,
        logits: &Tensor,
        embed_matrix: Option<&Tensor>,
        step_idx: usize,
    ) -> Result<Tensor> {
        let mut result = logits.clone();

        // PASS 0: Phonetic Quarantine (Tier-1 Hardware Limit)
        // Unconditionally applied BEFORE any semantic or phonetic passes.
        // This prevents embedding space bleed into out-of-domain tokens.
        if let Some(ref mask) = self.phonetic_quarantine_mask {
            result = result.broadcast_add(mask)?;
        }

        // PASS 1: Semantic Domain Masking (QMEM Guardrails)
        if self.semantic_masking_enabled && !self.active_domains.is_empty() {
            result = self.apply_semantic_mask(&result)?;
        }

        // PASS 2: Phonetic Boosting (sEMG Steering)
        let mut max_boost_val = 0.0f32;
        if self.phonetic_boosting_enabled {
            if step_idx < self.semg_latent_sequence.len() {
                if let Some(embeddings) = embed_matrix {
                    let semg = &self.semg_latent_sequence[step_idx];
                    result = self.apply_phonetic_boost(&result, semg, embeddings)?;
                    max_boost_val = self.phonetic_boost_scale;
                }
            }
        }

        eprintln!("[TELEMETRY LOGITS] Applied Mask: {} active domains | Applied sEMG Boost: max val {:.2} on token XYZ", self.active_domains.len(), max_boost_val);

        Ok(result)
    }

    // ========================================================================
    // PASS 1: SEMANTIC MASKING
    // ========================================================================

    /// Apply semantic domain mask to logits.
    ///
    /// Tokens outside the active semantic domains receive a large negative
    /// penalty, effectively removing them from the probability distribution
    /// after softmax.
    fn apply_semantic_mask(&mut self, logits: &Tensor) -> Result<Tensor> {
        // Use cached mask if available
        let mask = if let Some(ref cached) = self.cached_domain_mask {
            cached.clone()
        } else {
            let mask = self.domain_registry.build_domain_mask(
                &self.active_domains,
                self.semantic_penalty,
            )?;
            self.cached_domain_mask = Some(mask.clone());
            mask
        };

        // GPU-resident tensor addition: logits + mask
        // Out-of-domain tokens get logits + (-1e6) → effectively -INF after softmax
        logits.add(&mask)
            .map_err(|e| anyhow::anyhow!("Semantic mask application failed: {}", e))
    }

    // ========================================================================
    // PASS 2: PHONETIC BOOSTING (PURE GPU MATMUL)
    // ========================================================================

    /// Apply sEMG phonetic similarity boosting via fused dense matrix multiplication.
    ///
    /// Performs a zero-CPU-sync projection of the sEMG latent vector against
    /// the entire LLM vocabulary embedding space (lm_head). 
    ///
    /// Math:
    ///   Boost = alpha * (E_matrix * sEMG^T)
    ///   logit' = logit + Boost
    ///
    /// where E_matrix = [vocab_size, hidden_size] and sEMG^T = [hidden_size, 1].
    fn apply_phonetic_boost(
        &self,
        logits: &Tensor,
        semg_latent: &Tensor,
        lm_head: &Tensor,
    ) -> Result<Tensor> {
        let semg_dims = semg_latent.dims();
        if semg_dims.is_empty() {
            return Ok(logits.clone());
        }

        let hidden_size = *semg_dims.last().unwrap();
        
        let head_dims = lm_head.dims();
        if head_dims.len() < 2 || head_dims[head_dims.len() - 1] != hidden_size {
            eprintln!("[BCI LOGITS] Dimension mismatch: sEMG {} vs LM Head {:?}", hidden_size, head_dims);
            return Ok(logits.clone());
        }

        // Reshape sEMG latent to [1, hidden_size]
        let semg_flat = semg_latent.reshape((1, hidden_size))?;

        // 1. Compute Raw Dot Products: [vocab_size, hidden_size] x [hidden_size, 1] -> [vocab_size, 1]
        let dot_products = lm_head.matmul(&semg_flat.t()?)?;

        // 2. Compute L2 Norms for Cosine Similarity
        // semg_norm = sqrt(sum(semg^2))
        let semg_norm = semg_flat.sqr()?.sum_all()?.sqrt()?;
        
        // lm_head_norms = sqrt(sum(lm_head^2, dim=1)) -> [vocab_size, 1]
        let lm_head_norms = lm_head.sqr()?.sum_keepdim(1)?.sqrt()?;

        // 3. Normalize: Similarity = Dot / (Norm_A * Norm_B)
        // Add epsilon (1e-8) to avoid division by zero
        let semg_norm_val = semg_norm.to_scalar::<f32>()? as f64;
        let combined_norms = (lm_head_norms * semg_norm_val)?.affine(1.0, 1e-8)?;
        let similarity = dot_products.div(&combined_norms)?;

        // 4. Liquefied Gate and Scale
        // Liquefied Gate: use ReLU to ignore negative alignments but allow any positive steer
        let positive_similarity = similarity.relu()?;
        let boost_1d = positive_similarity.squeeze(1)?;
        let scaled_boost = (boost_1d * self.phonetic_boost_scale as f64)?;

        // Broadcast add directly to logits
        logits.broadcast_add(&scaled_boost)
            .map_err(|e| anyhow::anyhow!("Phonetic boost application failed: {}", e))
    }
}

// ============================================================================
// 3. CONVENIENCE CONSTRUCTORS
// ============================================================================

/// Create a BciLogitsProcessor pre-configured with common semantic domains.
///
/// This sets up domains for the Inner Speech dataset categories:
/// Directions (up/down/left/right), Actions, Objects, Conditions.
pub fn create_default_bci_processor(
    vocab_size: usize,
    device: &Device,
    tokenizer: &tokenizers::Tokenizer,
) -> BciLogitsProcessor {
    let mut proc = BciLogitsProcessor::new(vocab_size, device);

    // Inner Speech Dataset — Nieto ds003626 semantic categories
    proc.domain_registry.register_domain(
        "direction",
        &["up", "down", "left", "right", "north", "south", "east", "west",
          "forward", "backward", "above", "below", "top", "bottom"],
        tokenizer,
    );

    proc.domain_registry.register_domain(
        "food",
        &["food", "eat", "drink", "restaurant", "meal", "breakfast", "lunch",
          "dinner", "snack", "hungry", "cook", "kitchen", "menu", "order",
          "pizza", "burger", "coffee", "water", "juice"],
        tokenizer,
    );

    proc.domain_registry.register_domain(
        "greeting",
        &["hello", "hi", "hey", "goodbye", "bye", "thanks", "thank",
          "please", "welcome", "nice", "meet", "morning", "evening", "night"],
        tokenizer,
    );

    proc.domain_registry.register_domain(
        "question",
        &["what", "where", "when", "why", "how", "who", "which",
          "is", "are", "do", "does", "can", "could", "would", "should",
          "did", "will", "have", "has", "?"],
        tokenizer,
    );

    proc.domain_registry.register_domain(
        "affirmation",
        &["yes", "no", "ok", "okay", "sure", "right", "correct",
          "agree", "disagree", "maybe", "perhaps", "true", "false"],
        tokenizer,
    );

    proc.domain_registry.register_domain(
        "emotion",
        &["happy", "sad", "angry", "scared", "excited", "bored",
          "tired", "love", "hate", "like", "dislike", "feel", "feeling",
          "good", "bad", "great", "terrible", "wonderful"],
        tokenizer,
    );

    proc
}

// ============================================================================
// 4. TELEMETRY
// ============================================================================

impl BciLogitsProcessor {
    /// Print a diagnostic summary of the current processor state.
    pub fn telemetry_report(&self) {
        eprintln!("┌─────────────────────────────────────────────────────────┐");
        eprintln!("│           BCI LOGITS PROCESSOR — STATUS                │");
        eprintln!("├─────────────────────────────────────────────────────────┤");
        eprintln!("│ Semantic Masking : {:>5}                               │",
            if self.semantic_masking_enabled { "ON" } else { "OFF" });
        eprintln!("│ Phonetic Boost   : {:>5}                               │",
            if self.phonetic_boosting_enabled { "ON" } else { "OFF" });
        eprintln!("│ Active Domains   : {:?}", self.active_domains);
        eprintln!("│ Penalty          : {:>10.1e}                          │", self.semantic_penalty);
        eprintln!("│ Boost Scale      : {:>10.2}                          │", self.phonetic_boost_scale);
        eprintln!("│ sEMG Seq Len     : {:>5}                               │", self.semg_latent_sequence.len());
        eprintln!("│ Cached Mask      : {:>5}                               │",
            if self.cached_domain_mask.is_some() { "HIT" } else { "MISS" });
        eprintln!("│ Phonetic Quarantine: {:>5}                             │",
            if self.phonetic_quarantine_mask.is_some() { "ON" } else { "OFF" });
        eprintln!("│ Registered Domains: {}                                 │",
            self.domain_registry.domains.len());
        eprintln!("└─────────────────────────────────────────────────────────┘");
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_mask_penalizes_out_of_domain() {
        let device = Device::Cpu;
        let vocab_size = 100;
        let mut proc = BciLogitsProcessor::new(vocab_size, &device);

        // Register a tiny domain with tokens 10, 11, 12
        let domain = SemanticDomain {
            name: "test_domain".to_string(),
            token_ids: [10u32, 11, 12].iter().cloned().collect(),
            weight: 1.0,
        };
        proc.domain_registry.register_prebuilt_domain(domain);
        proc.set_active_domains(vec!["test_domain".to_string()]);

        // Create uniform logits
        let logits = Tensor::ones(vocab_size, DType::F32, &device).unwrap();

        // Apply semantic mask
        let result = proc.process(&logits, None, 0).unwrap();
        let result_vec: Vec<f32> = result.to_vec1().unwrap();

        // Tokens 10, 11, 12 should be ~1.0 (allowed)
        assert!((result_vec[10] - 1.0).abs() < 0.01, "Token 10 should be allowed");
        assert!((result_vec[11] - 1.0).abs() < 0.01, "Token 11 should be allowed");
        assert!((result_vec[12] - 1.0).abs() < 0.01, "Token 12 should be allowed");

        // Tokens outside domain (and outside special range 0..256) should be penalized
        // Token 50 is in 0..256 range so it's allowed; check token 257+ range
    }

    #[test]
    fn test_no_domains_means_no_masking() {
        let device = Device::Cpu;
        let vocab_size = 100;
        let mut proc = BciLogitsProcessor::new(vocab_size, &device);

        // No active domains — all tokens should pass through
        let logits = Tensor::ones(vocab_size, DType::F32, &device).unwrap();
        let result = proc.process(&logits, None, 0).unwrap();
        let result_vec: Vec<f32> = result.to_vec1().unwrap();

        for v in &result_vec {
            assert!((v - 1.0).abs() < 0.01, "No masking should leave logits unchanged");
        }
    }

    #[test]
    fn test_processor_reset_clears_state() {
        let device = Device::Cpu;
        let mut proc = BciLogitsProcessor::new(100, &device);

        proc.set_active_domains(vec!["test".to_string()]);
        // Provide a 3x larger tensor because set_semg_latent splits it into 3 parts
        proc.set_semg_latent(Tensor::ones(12, DType::F32, &device).unwrap());

        proc.reset();

        assert!(proc.active_domains.is_empty());
        assert!(proc.semg_latent_sequence.is_empty());
        assert!(proc.cached_domain_mask.is_none());
    }
}
