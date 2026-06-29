// src/crates/sodals_neuro/src/neuro_kernel.rs - Resonance Gating for Creative Leaps
// 
// FIX: Dynamic Divergence thresholding based on semantic resonance
// Low Div + High Resonance = Creative Leap (ALLOW)
// Low Div + Low Resonance = Hallucination (BLOCK)

use anyhow::Result;
use async_stream::stream;
use futures::Stream;
use std::pin::Pin;
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::LogManager;
use futures::StreamExt;
use tokio::sync::broadcast;
use ndarray::Array1;
use std::sync::Arc;

use crate::llm_core::LlmEngine;
use crate::engine::security::PrivacyProjector;
use crate::fast_math::{SemanticWave, repair_semantic_wave};
use crate::physics_neuron::PhysicsNeuron;
use crate::semantic_network::ResonanceNetwork;
use crate::resonance_chunker::ResonanceChunker;
use crate::retrieval_engine::{RetrievalEngine, RetrievalResult};

// ============================================================================
// RESONANCE GATING THRESHOLDS
// ============================================================================

/// Semantic Divergence thresholds
/// The Maclaurin-truncated finite-difference scheme approximates phase variance.
const DIVERGENCE_CRITICAL: f64 = 0.01;  // Absolute critical threshold
const DIVERGENCE_WARNING: f64 = 0.05;   // Warning zone

/// Semantic resonance thresholds
const RESONANCE_HIGH: f64 = 0.75;      // High semantic coherence (creative)
const RESONANCE_LOW: f64 = 0.30;       // Low semantic coherence (hallucination)

/// Classification of cognitive state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CognitiveHealth {
    /// Healthy independent state
    Healthy,
    
    /// Creative leap (intentional repetition/emphasis)
    CreativeLeap,
    
    /// Dangerous hallucination/loop
    Hallucination,
}

// ============================================================================
// EVENT BUS
// ============================================================================

#[derive(Debug, Clone)]
pub struct AnomalousStateEvent {
    pub keystone_id: String,
    pub jacobian_norm: f64,
    pub phase_mismatch: f64,
    pub semantic_divergence: f64,
    pub resonance: f64,              // NEW: Semantic resonance score
    pub cognitive_state: String,     // NEW: Health classification
    pub timestamp: std::time::Instant,
}

pub struct CognitiveEventBus {
    sender: broadcast::Sender<AnomalousStateEvent>,
}

impl Default for CognitiveEventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitiveEventBus {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(100);
        Self { sender }
    }

    pub fn emit(&self, event: AnomalousStateEvent) {
        if let Err(e) = self.sender.send(event) {
            eprintln!("[EVENT-BUS] Failed to emit event: {}", e);
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<AnomalousStateEvent> {
        self.sender.subscribe()
    }
}

// ============================================================================
// COGNITIVE TOKEN
// ============================================================================

#[derive(Debug, Clone)]
pub struct CognitiveToken {
    pub text: String,
    pub certainty: f64,
    pub emotionality: f64,
    pub phase: f64,
    pub is_repaired: bool,
    pub is_steered: bool,
    pub is_creative_leap: bool,     // NEW: Flagged as intentional repetition
}

impl CognitiveToken {
    pub fn classification(&self) -> &'static str {
        match (self.certainty, self.emotionality) {
            (c, e) if c > 0.8 && e < 0.2 => "Willing",
            (c, e) if c > 0.5 && e > 0.6 => "Conflicted",
            (c, _) if c < 0.3 => "Uncertain",
            _ => "Hesitant",
        }
    }

    pub fn is_hallucination_risk(&self) -> bool {
        self.emotionality > 0.75 || (self.certainty < 0.4 && self.emotionality > 0.5)
    }
}

// ============================================================================
// STEERING COMMAND
// ============================================================================

#[derive(Debug, Clone)]
pub struct SteeringCommand {
    pub target_vector: SemanticWave,
    pub remove_vector: SemanticWave,
    pub keystone_mask: SemanticWave,
    pub remove_strength: f64,
    pub inject_strength: f64,
}

impl SteeringCommand {
    pub fn new(
        target_vector: SemanticWave,
        remove_vector: SemanticWave,
        keystone_mask: SemanticWave,
    ) -> Self {
        Self {
            target_vector,
            remove_vector,
            keystone_mask,
            remove_strength: 1.0,
            inject_strength: 1.0,
        }
    }
}

// ============================================================================
// NEURO KERNEL WITH SEQUENTIAL OUROBOROS RESONANCE GATING
// ============================================================================

pub struct NeuroKernel {
    llm: LlmEngine,
    filter: PrivacyProjector,
    logger: LogManager,
    cognitive_state: PhysicsNeuron,
    event_bus: CognitiveEventBus,
    active_steering: Option<SteeringCommand>,
    retrieval_engine: Option<Arc<RetrievalEngine>>,
}

impl NeuroKernel {
    pub fn new(config: &SystemConfig, logger: &LogManager) -> Result<Self> {
        logger.info("Kernel", "Initializing Cognitive Kernel (Phase-Modulated Attention v2.0 + Resonance Gating)");

        let llm = LlmEngine::new(config, logger)?;
        let filter = PrivacyProjector::new(&config.secret_api_key, 768, 64);
        
        let mut cognitive_state = PhysicsNeuron::new(0.01);
        cognitive_state.param_a = 4.0;
        cognitive_state.param_b = 1.2;

        let event_bus = CognitiveEventBus::new();

        Ok(Self {
            llm,
            filter,
            logger: logger.clone(),
            cognitive_state,
            event_bus,
            active_steering: None,
            retrieval_engine: None,
        })
    }

    /// Set retrieval engine for context-aware thinking
    pub fn set_retrieval_engine(&mut self, engine: Arc<RetrievalEngine>) {
        self.retrieval_engine = Some(engine);
        self.logger.info("Kernel", "Retrieval engine attached for context-aware thinking");
    }

    pub fn event_bus(&self) -> &CognitiveEventBus {
        &self.event_bus
    }

    pub fn set_steering(&mut self, command: Option<SteeringCommand>) {
        self.active_steering = command;
    }

    /// Enhanced think_stream with context injection using retrieval engine
    /// 
    /// This is the main entry point for context-aware thinking. It:
    /// 1. Queries the retrieval engine for relevant chunks
    /// 2. Filters chunks by CognitiveHealth gating
    /// 3. Injects context into the prompt
    /// 4. Proceeds with standard think_stream
    /// 
    /// NO fallback to summarization for readability - pure nuance preservation.
    pub async fn think_stream_with_context(
        &mut self,
        user_prompt: &str,
        max_context_chunks: usize,
    ) -> Result<Pin<Box<dyn Stream<Item = CognitiveToken> + Send>>> {
        self.logger.info("Kernel", &format!(
            "[CONTEXT-AWARE] Querying retrieval engine for {} chunks", max_context_chunks
        ));

        // 1. Query retrieval engine if available
        let context_chunks = if let Some(ref engine) = self.retrieval_engine {
            engine.query_hybrid(user_prompt, max_context_chunks)
        } else {
            self.logger.info("Kernel", "No retrieval engine attached - proceeding without context");
            Vec::new()
        };

        // 2. Apply CognitiveHealth gating
        let filtered_chunks = self.filter_by_cognitive_health(&context_chunks);
        self.logger.info("Kernel", &format!(
            "[CONTEXT-AWARE] Filtered {} chunks from {} retrieved",
            filtered_chunks.len(),
            context_chunks.len()
        ));

        // 3. Inject context into prompt
        let enriched_prompt = self.inject_context(user_prompt, &filtered_chunks);

        // 4. Proceed with standard think_stream
        self.think_stream(&enriched_prompt, None).await
    }

    /// Filter chunks by divergence/resonance thresholds using CognitiveHealth gating
    fn filter_by_cognitive_health(&self, chunks: &[RetrievalResult]) -> Vec<RetrievalResult> {
        chunks.iter()
            .filter(|chunk| {
                let health = Self::assess_cognitive_health(chunk.divergence, chunk.resonance);
                matches!(health, CognitiveHealth::Healthy | CognitiveHealth::CreativeLeap)
            })
            .cloned()
            .collect()
    }

    /// Inject context chunks into the user prompt
    /// 
    /// Maintains nuance preservation by including full chunk text without summarization.
    fn inject_context(&self, user_prompt: &str, chunks: &[RetrievalResult]) -> String {
        if chunks.is_empty() {
            return user_prompt.to_string();
        }

        let mut context_text = String::from("[CONTEXT]\n");
        
        for (idx, chunk) in chunks.iter().enumerate() {
            context_text.push_str(&format!(
                "[CHUNK {}] (divergence={:.4}, resonance={:.4})\n{}\n",
                idx + 1,
                chunk.divergence,
                chunk.resonance,
                chunk.text
            ));
        }
        
        context_text.push_str("[/CONTEXT]\n\n");
        context_text.push_str(user_prompt);
        
        context_text
    }

    /// VRAM-safe sequential Ouroboros: Simulate - Critique - Refine.
    ///
    /// PHASE 44: Fixed Phase Variance history buffer bug. Now maintains proper
    /// history state [h_{n-1}, h_n] for Maclaurin-truncated Phase Variance computation.
    ///
    /// Invariants enforced:
    ///   1. `clear_kv_cache()` is called BEFORE every pass to reclaim VRAM.
    ///   2. Divergence early-exit: if pass energy < 0.1 - accept immediately.
    ///   3. No parallel GPU inference. One context at a time.
    ///   4. `pass_tokens` is dropped explicitly before the next iteration
    ///      to reclaim the heap allocation, not just shadowed.
    ///   5. Divergence history buffer maintains last 2 states for O(N) computation.
    pub async fn ouroboros_think(
        &mut self,
        user_prompt: &str,
        max_passes: usize,
    ) -> Result<Vec<CognitiveToken>> {
        self.logger.info("Kernel", &format!(
            "[OUROBOROS] Sequential loop armed (max_passes={}, VRAM flush=ON)", max_passes
        ));

        let mut best_tokens: Vec<CognitiveToken> = Vec::new();
        let mut current_prompt = user_prompt.to_string();
        let mut prev_divergence: f64 = f64::MAX;
        // PHASE 44: Maintain Divergence history buffer for proper sequence analysis
        let mut divergence_history: Vec<SemanticWave> = Vec::with_capacity(3);

        for pass in 0..max_passes {
            // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIII
            // VRAM FLUSH: Explicitly nuke the KV cache BEFORE each pass.
            // Rust's Drop is NOT sufficient for CUDA device memory.
            // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIII
            {
                let model_guard = self.llm.model.write().unwrap();
                model_guard.clear_kv_cache();
            }
            // Guard dropped - RwLock released - model free for generation.

            self.logger.info("Kernel", &format!(
                "[OUROBOROS] Pass {}/{} | prompt_len={} | prev_Div={:.4}",
                pass + 1, max_passes, current_prompt.len(), prev_divergence
            ));

            // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIII
            // SIMULATE: Single-stream generation (one VRAM context).
            // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIII
            let mut stream = self.think_stream(&current_prompt, None).await?;
            let mut pass_tokens: Vec<CognitiveToken> = Vec::with_capacity(512);

            while let Some(token) = stream.next().await {
                pass_tokens.push(token);
            }
            // Stream is consumed and dropped here III its internal tensors freed.
            drop(stream);
            
            // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIII
            // VRAM RECLAMATION: Explicitly drop large tensors and sync GPU
            // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIII
            self.llm.device.synchronize().unwrap_or_default();
            // Critical: Ensure CUDA work is drained before assessing health on CPU.

            if pass_tokens.is_empty() {
                self.logger.warn("Kernel", "[OUROBOROS] Empty generation. Breaking.");
                break;
            }

            // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIII
            // CRITIQUE: CPU-only Phase Variance + Resonance assessment.
            // Zero GPU involvement - pure f64 math on the i9.
            let certainties: Vec<f64> = pass_tokens.iter()
                .map(|t| t.certainty)
                .collect();
            let simulated_wave = SemanticWave::from_reals(certainties);

            // PHASE 44: Proper divergence metric with history buffer [h_{n-1}, h_n]
            let pass_divergence = Self::compute_trajectory_divergence(&divergence_history, &simulated_wave);
            let pass_resonance = Self::compute_semantic_resonance_static(&divergence_history, &simulated_wave);
            
            // Update history buffer (keep last 2 states for next iteration)
            divergence_history.push(simulated_wave);
            if divergence_history.len() > 2 {
                divergence_history.remove(0);
            }

            // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIII
            // EARLY EXIT: If Divergence energy is below threshold, the
            // output is mathematically stable. Do NOT waste cycles on
            // the Regret Vector or additional passes.
            // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIII
            if pass_divergence < 0.1 {
                self.logger.info("Kernel", &format!(
                    "[OUROBOROS] EARLY EXIT: Div={:.6} < 0.1 threshold. Accepting pass {}.",
                    pass_divergence, pass + 1
                ));
                best_tokens = pass_tokens;
                break;
            }

            let health = Self::assess_cognitive_health(pass_divergence, pass_resonance);

            match health {
                CognitiveHealth::Healthy | CognitiveHealth::CreativeLeap => {
                    self.logger.info("Kernel", &format!(
                        "[OUROBOROS] Pass {} ACCEPTED (Div={:.4}, R={:.4})",
                        pass + 1, pass_divergence, pass_resonance
                    ));
                    best_tokens = pass_tokens;
                    break;
                }
                CognitiveHealth::Hallucination => {
                    // Check convergence: if Divergence is not improving, stop wasting VRAM cycles.
                    if pass > 0 && (prev_divergence - pass_divergence).abs() < 0.01 {
                        self.logger.warn("Kernel", &format!(
                            "[OUROBOROS] CONVERGENCE STALL: IIDiv={:.6}. Accepting best available.",
                            (prev_divergence - pass_divergence).abs()
                        ));
                        best_tokens = pass_tokens;
                        break;
                    }

                    self.logger.warn("Kernel", &format!(
                        "[OUROBOROS] Pass {} HALLUCINATED (Div={:.4}). Refining.",
                        pass + 1, pass_divergence
                    ));

                    prev_divergence = pass_divergence;

                    // Keep tokens as fallback, then drop the old best explicitly.
                    let old_best = std::mem::replace(&mut best_tokens, pass_tokens);
                    drop(old_best); // Explicit heap reclamation.

                    // REFINE: Append critique directive (stays on CPU, no VRAM cost).
                    current_prompt = format!(
                        "{} [Critique: Div={:.4} exceeded threshold. Reduce hallucination. \
                         Tighten logical constraints.]",
                        user_prompt, pass_divergence
                    );
                }
            }
        }

        // Final VRAM cleanup after all passes complete.
        {
            let model_guard = self.llm.model.write().unwrap();
            model_guard.clear_kv_cache();
        }

        self.logger.info("Kernel", &format!(
            "[OUROBOROS] Complete. Returning {} tokens.", best_tokens.len()
        ));
        Ok(best_tokens)
    }

    pub async fn think_stream(
        &mut self,
        user_prompt: &str,
        _semantic_bias: Option<&SemanticWave>,
    ) -> Result<Pin<Box<dyn Stream<Item = CognitiveToken> + Send>>> {
        if !self.filter.check_safety(user_prompt) {
            self.logger.warn("Kernel", "Unsafe content blocked");
            
            let stream: Pin<Box<dyn Stream<Item = CognitiveToken> + Send>> = Box::pin(stream! {
                yield CognitiveToken {
                    text: "Security protocol: Request denied.".to_string(),
                    certainty: 1.0,
                    emotionality: 0.0,
                    phase: 0.0,
                    is_repaired: false,
                    is_steered: false,
                    is_creative_leap: false,
                };
            });
            return Ok(stream);
        }

        let formatted_prompt = self.format_prompt(user_prompt);
        let logger = self.logger.clone();
        let event_bus = self.event_bus.sender.clone();
        let steering_command = self.active_steering.clone();
        
        let mut llm_stream = self.llm.generate_stream_async(&formatted_prompt, None).await?;
        let mut cognitive_state = self.cognitive_state.clone();
        
        let stream: Pin<Box<dyn Stream<Item = CognitiveToken> + Send>> = Box::pin(stream! {
            let mut previous_states: Vec<SemanticWave> = Vec::new();
            let mut step_counter = 0u64;
            
            while let Some(token_result) = llm_stream.next().await {
                match token_result {
                    Ok(raw_token) => {
                        step_counter += 1;
                        
                        let mut token_wave = SemanticWave::from_reals(
                            raw_token.logits.iter().map(|&x| x as f64).collect()
                        );
                        
                        let mut is_repaired = false;
                        let mut is_steered = false;
                        let mut is_creative_leap = false;
                        
                        if previous_states.len() >= 2 {
                            // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIII
                            // STEP 1: COMPUTE DIVERGENCE + RESONANCE
                            // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIII
                            let divergence = Self::compute_trajectory_divergence(&previous_states, &token_wave);
                            
                            // NEW: Compute semantic resonance from history
                            let resonance = Self::compute_semantic_resonance_static(&previous_states, &token_wave);
                            
                            // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIII
                            // STEP 2: RESONANCE GATING (CREATIVE LEAP vs HALLUCINATION)
                            // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIII
                            let health = Self::assess_cognitive_health(divergence, resonance);
                            
                            match health {
                                CognitiveHealth::Healthy => {
                                    // All good, continue normally
                                }
                                
                                CognitiveHealth::CreativeLeap => {
                                    // LOW DIVERGENCE + HIGH RESONANCE
                                    // This is intentional repetition/emphasis (e.g., poetic, recursive logic)
                                    logger.info("Kernel", &format!(
                                        "[CREATIVE LEAP] Div={:.6}, R={:.3} - Intentional repetition allowed",
                                        divergence, resonance
                                    ));
                                    is_creative_leap = true;
                                    
                                    // Emit event but DO NOT BLOCK
                                    let event = AnomalousStateEvent {
                                        keystone_id: format!("step_{}", step_counter),
                                        jacobian_norm: Self::compute_jacobian_norm_static(&token_wave),
                                        phase_mismatch: Self::compute_phase_mismatch_static(&previous_states, &token_wave),
                                        semantic_divergence: divergence,
                                        resonance,
                                        cognitive_state: "CREATIVE_LEAP".to_string(),
                                        timestamp: std::time::Instant::now(),
                                    };
                                    let _ = event_bus.send(event);
                                }
                                
                                CognitiveHealth::Hallucination => {
                                    // LOW DIVERGENCE + LOW RESONANCE
                                    // This is dangerous hallucination/loop
                                    logger.warn("Kernel", &format!(
                                        "[HALLUCINATION] Div={:.6}, R={:.3} - Applying correction",
                                        divergence, resonance
                                    ));
                                    
                                    // Emit critical event
                                    let event = AnomalousStateEvent {
                                        keystone_id: format!("step_{}", step_counter),
                                        jacobian_norm: Self::compute_jacobian_norm_static(&token_wave),
                                        phase_mismatch: Self::compute_phase_mismatch_static(&previous_states, &token_wave),
                                        semantic_divergence: divergence,
                                        resonance,
                                        cognitive_state: "HALLUCINATION".to_string(),
                                        timestamp: std::time::Instant::now(),
                                    };
                                    let _ = event_bus.send(event);
                                    
                                    // Apply Skew Code repair
                                    match repair_semantic_wave(&token_wave) {
                                        Ok(repaired_wave) => {
                                            logger.info("Kernel", "- Hallucination corrected via Skew Code");
                                            token_wave = repaired_wave;
                                            is_repaired = true;
                                        }
                                        Err(e) => {
                                            logger.error("Kernel", &format!("Repair failed: {}", e));
                                        }
                                    }
                                }
                            }
                        }
                        
                        // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIII
                        // STEP 3: SEMANTIC STEERING (if active)
                        // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIII
                        if let Some(ref steering) = steering_command {
                            if let Ok(steered_wave) = Self::apply_semantic_steering_static(&token_wave, steering, &logger, step_counter) {
                                token_wave = steered_wave;
                                is_steered = true;
                            }
                        }
                        
                        // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIII
                        // STEP 4: PHYSICS ANALYSIS
                        // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIII
                        let cognitive_meta = Self::analyze_token_physics_static(
                            &mut cognitive_state,
                            &token_wave,
                            is_repaired
                        );
                        
                        previous_states.push(token_wave.clone());
                        if previous_states.len() > 10 {
                            previous_states.remove(0);
                        }
                        
                        yield CognitiveToken {
                            text: raw_token.text,
                            certainty: cognitive_meta.certainty,
                            emotionality: cognitive_meta.emotionality,
                            phase: cognitive_meta.phase,
                            is_repaired,
                            is_steered,
                            is_creative_leap,
                        };
                    }
                    Err(e) => {
                        logger.error("Kernel", &format!("Stream error: {}", e));
                        break;
                    }
                }
            }
        });

        Ok(stream)
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIII
    // RESONANCE GATING: CORE LOGIC
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIII

    /// Assess cognitive health using Trajectory Divergence + Semantic Resonance
    /// 
    /// CRITICAL DECISION MATRIX:
    /// ```text
    /// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIII
    /// III                 III High ResonanceIII Low ResonanceIII
    /// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIII
    /// III High Divergence III   HEALTHY    III   HEALTHY    III
    /// III (Div > 0.05)    III              III              III
    /// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIII
    /// III Low Divergence  III  CREATIVE    III HALLUCINATIONIII
    /// III (Div < 0.01)    III    LEAP      III   (DANGER)   III
    /// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIII
    /// ```
    fn assess_cognitive_health(divergence: f64, resonance: f64) -> CognitiveHealth {
        if divergence > DIVERGENCE_WARNING {
            // High linear independence - healthy state
            CognitiveHealth::Healthy
        } else if divergence < DIVERGENCE_CRITICAL {
            // Low linear independence - need resonance check
            if resonance > RESONANCE_HIGH {
                // High semantic coherence despite repetition
                // Intentional (poetic, emphatic, recursive logic)
                CognitiveHealth::CreativeLeap
            } else if resonance < RESONANCE_LOW {
                // Low semantic coherence with repetition
                // Hallucination/loop
                CognitiveHealth::Hallucination
            } else {
                // Ambiguous zone - treat as healthy with monitoring
                CognitiveHealth::Healthy
            }
        } else {
            // Warning zone (0.01 < W < 0.05)
            if resonance > RESONANCE_HIGH {
                CognitiveHealth::CreativeLeap
            } else {
                CognitiveHealth::Healthy
            }
        }
    }

    /// Compute Divergence health assessment from vector and resonance score
    /// 
    /// UPGRADED: Now uses resonance gating to distinguish creative from pathological
    /// 
    /// Formula:
    /// 1. Calculate L1 Norm energy of vector
    /// 2. Calculate Trajectory Divergence (determinant-based independence metric)
    /// 3. Apply resonance gating:
    ///    - Low Div + High R -> CreativeLeap (ALLOW)
    ///    - Low Div + Low R  -> Hallucination (BLOCK)
    ///    - High Div         -> Healthy (regardless of R)
    pub fn compute_divergence_health(&self, vector: &Array1<f32>, resonance_score: f64) -> CognitiveHealth {
        // Step 1: Calculate L1 Norm (Energy)
        let l1_norm: f64 = vector.iter().map(|&x| x.abs() as f64).sum();
        
        // Step 2: Calculate Trajectory Divergence (Independence metric)
        // For a single vector, Divergence approximates as magnitude-based independence
        // Higher magnitude = higher linear independence
        let divergence = if l1_norm > 0.0 {
            // Normalize by vector length
            l1_norm / (vector.len() as f64).sqrt()
        } else {
            0.0
        };
        
        // Step 3: Apply Resonance Gating
        // Critical threshold comparison
        if divergence < DIVERGENCE_CRITICAL {
            // Low linear independence - check resonance
            if resonance_score > RESONANCE_HIGH {
                // High semantic coherence despite low independence
                // Creative leap (intentional repetition/emphasis)
                CognitiveHealth::CreativeLeap
            } else if resonance_score < RESONANCE_LOW {
                // Low semantic coherence with low independence
                // Hallucination (dangerous loop)
                CognitiveHealth::Hallucination
            } else {
                // Ambiguous zone (RESONANCE_LOW < R < RESONANCE_HIGH)
                // Treat as healthy with caution
                CognitiveHealth::Healthy
            }
        } else if divergence < DIVERGENCE_WARNING {
            // Warning zone (CRITICAL < W < WARNING)
            // Only escalate to CreativeLeap if resonance is very high
            if resonance_score > RESONANCE_HIGH {
                CognitiveHealth::CreativeLeap
            } else {
                CognitiveHealth::Healthy
            }
        } else {
            // High linear independence (W >= WARNING)
            // Always healthy regardless of resonance
            CognitiveHealth::Healthy
        }
    }

    /// Compute semantic resonance from Matrix B connections
    /// 
    /// Measures coherence across semantic space:
    /// - High resonance: Concepts are thematically related
    /// - Low resonance: Random/unrelated concepts
    /// 
    /// Formula: R = IIIII_current | II_historyIII / (||II_current|| II ||II_history||)
    fn compute_semantic_resonance_static(history: &[SemanticWave], current: &SemanticWave) -> f64 {
        if history.is_empty() {
            return 1.0; // No history, assume high resonance
        }
        
        // Compute average resonance with recent history
        let window_size = history.len().min(5); // Last 5 tokens
        let recent_history = &history[history.len() - window_size..];
        
        let mut total_resonance = 0.0;
        
        for prev in recent_history {
            // Normalized inner product (cosine similarity)
            let dot_product = prev.inner_product(current).norm();
            let norm_prev = prev.norm();
            let norm_current = current.norm();
            
            if norm_prev > 1e-10 && norm_current > 1e-10 {
                let similarity = dot_product / (norm_prev * norm_current);
                total_resonance += similarity;
            }
        }
        
        // Average resonance
        total_resonance / window_size as f64
    }

    fn compute_trajectory_divergence(history: &[SemanticWave], current: &SemanticWave) -> f64 {
        if history.len() < 2 {
            return 1.0;
        }

        // Phase 12: Maclaurin-Truncated Trajectory Divergence (O(N) Approximation)
        // Ditching heavy determinants and nested norm loops.
        // Expanding around t=0: f(x) ≈ f(0) + f'(0)x
        
        let prev1 = &history[history.len() - 1];
        let prev2 = &history[history.len() - 2];

        let mut div_sum = 0.0;
        let mut max_possible_div = 1e-10_f64; // Epsilon denominator

        let n = current.amplitudes.len();
        for i in 0..n {
            // Extract f_i(0) as the wave amplitude component
            let f1 = prev1.amplitudes.get(i).copied().unwrap_or_default();
            let f2 = current.amplitudes.get(i).copied().unwrap_or_default();
            let p2 = prev2.amplitudes.get(i).copied().unwrap_or_default();

            // Approximate f_i'(0) using backward finite differences (delta)
            let df1 = f1 - p2;
            let df2 = f2 - f1;

            // Div_approx = \sum |f1 * df2 - f2 * df1|
            let cross_term = f1 * df2 - f2 * df1;
            div_sum += cross_term.norm();

            // Normalization bound: |f1 * df2| + |f2 * df1| >= |f1*df2 - f2*df1|
            max_possible_div += f1.norm() * df2.norm() + f2.norm() * df1.norm();
        }

        // Normalize final scalar strictly to [0.0, 1.0] for OuroborosGuard
        let div_approx = div_sum / max_possible_div;
        div_approx.clamp(0.0, 1.0)
    }

    fn compute_jacobian_norm_static(wave: &SemanticWave) -> f64 {
        wave.norm_sqr().sqrt()
    }

    fn compute_phase_mismatch_static(history: &[SemanticWave], current: &SemanticWave) -> f64 {
        if history.is_empty() {
            return 0.0;
        }
        
        let prev = &history[history.len() - 1];
        let expected_phase = prev.amplitudes.iter().next().map(|c| c.arg()).unwrap_or(0.0);
        let observed_phase = current.amplitudes.iter().next().map(|c| c.arg()).unwrap_or(0.0);
        
        (expected_phase - observed_phase).abs()
    }

    fn analyze_token_physics_static(
        cognitive_state: &mut PhysicsNeuron,
        wave: &SemanticWave,
        is_repaired: bool,
    ) -> Cognitivemetadata {
        if is_repaired {
            cognitive_state.enable_soft_injection();
        }
        
        let input_signal = wave.amplitudes.iter().next().copied().unwrap_or_default();
        cognitive_state.forward(input_signal);
        
        if is_repaired {
            cognitive_state.disable_soft_injection();
        }
        
        let state = cognitive_state.get_state();
        let magnitude = state.norm();
        let phase = state.arg();
        
        let real_dominance = state.re.abs() / (magnitude + 1e-10);
        let imag_ratio = state.im.abs() / (magnitude + 1e-10);
        
        Cognitivemetadata {
            certainty: real_dominance,
            emotionality: imag_ratio,
            phase,
        }
    }

    fn apply_semantic_steering_static(
        current_wave: &SemanticWave,
        steering: &SteeringCommand,
        _logger: &LogManager,
        _step: u64,
    ) -> Result<SemanticWave> {
        let mut result = current_wave.clone();
        result.sub_scaled(&steering.remove_vector, steering.remove_strength);
        
        let current_phase = Self::extract_dominant_phase(current_wave);
        let target_aligned = Self::align_phase(&steering.target_vector, current_phase);
        
        result.add_scaled(&target_aligned, steering.inject_strength);
        
        let final_wave = Self::apply_keystone_protection(current_wave, &result, &steering.keystone_mask);
        
        Ok(final_wave)
    }

    fn extract_dominant_phase(wave: &SemanticWave) -> f64 {
        wave.amplitudes
            .iter()
            .max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap_or(std::cmp::Ordering::Equal))
            .map(|c| c.arg())
            .unwrap_or(0.0)
    }

    fn align_phase(vector: &SemanticWave, target_phase: f64) -> SemanticWave {
        let current_phase = Self::extract_dominant_phase(vector);
        let phase_diff = target_phase - current_phase;
        vector.rotated_phase(phase_diff)
    }

    fn apply_keystone_protection(
        original: &SemanticWave,
        modified: &SemanticWave,
        mask: &SemanticWave,
    ) -> SemanticWave {
        let mut inverse_mask = mask.clone();
        for amp in inverse_mask.as_slice_mut() {
            *amp = num_complex::Complex64::new(1.0, 0.0) - *amp;
        }
        
        let mut modified_contribution = modified.clone();
        modified_contribution.hadamard_mul_mut(&inverse_mask);
        
        let mut original_contribution = original.clone();
        original_contribution.hadamard_mul_mut(mask);
        
        let mut result = modified_contribution;
        result += original_contribution;
        
        result
    }

    fn format_prompt(&self, sanitized_input: &str) -> String {
        format!(
            "<|im_start|>system\n\
             You are SODALS, a mathematically-grounded AI assistant.\n\
             <|im_end|>\n\
             <|im_start|>user\n\
             {}\n\
             <|im_end|>\n\
             <|im_start|>assistant\n",
            sanitized_input
        )
    }
}

#[derive(Debug)]
struct Cognitivemetadata {
    certainty: f64,
    emotionality: f64,
    phase: f64,
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast_math::SemanticWave;

    #[test]
    fn test_cognitive_health_healthy() {
        // High Wronskian always results in Healthy
        let health = CognitiveHealth::Healthy;
        assert_eq!(
            NeuroKernel::assess_cognitive_health(0.10, 0.5),
            health
        );
    }

    #[test]
    fn test_cognitive_health_creative_leap() {
        // Low Wronskian + High Resonance = CreativeLeap
        let health = NeuroKernel::assess_cognitive_health(0.005, 0.85);
        assert_eq!(health, CognitiveHealth::CreativeLeap);
    }

    #[test]
    fn test_cognitive_health_hallucination() {
        // Low Wronskian + Low Resonance = Hallucination
        let health = NeuroKernel::assess_cognitive_health(0.005, 0.15);
        assert_eq!(health, CognitiveHealth::Hallucination);
    }

    #[test]
    fn test_cognitive_health_ambiguous_zone() {
        // Low Wronskian + Medium Resonance = Healthy (ambiguous)
        let health = NeuroKernel::assess_cognitive_health(0.005, 0.50);
        assert_eq!(health, CognitiveHealth::Healthy);
    }

    #[test]
    fn test_cognitive_health_warning_zone_high_resonance() {
        // Warning zone + High Resonance = CreativeLeap
        let health = NeuroKernel::assess_cognitive_health(0.03, 0.80);
        assert_eq!(health, CognitiveHealth::CreativeLeap);
    }

    #[test]
    fn test_cognitive_health_warning_zone_low_resonance() {
        // Warning zone + Low Resonance = Healthy (conservative)
        let health = NeuroKernel::assess_cognitive_health(0.03, 0.25);
        assert_eq!(health, CognitiveHealth::Healthy);
    }

    #[test]
    fn test_semantic_resonance_high() {
        // Create similar waves
        let wave1 = SemanticWave::from_reals(vec![0.5, 0.6, 0.7, 0.8]);
        let wave2 = SemanticWave::from_reals(vec![0.5, 0.6, 0.7, 0.8]);
        
        let history = vec![wave1];
        let resonance = NeuroKernel::compute_semantic_resonance_static(&history, &wave2);
        
        // Near-identical vectors should have resonance close to 1.0
        assert!(resonance > 0.95);
    }

    #[test]
    fn test_semantic_resonance_low() {
        // Create orthogonal waves
        let wave1 = SemanticWave::from_reals(vec![1.0, 0.0, 0.0, 0.0]);
        let wave2 = SemanticWave::from_reals(vec![0.0, 0.0, 1.0, 0.0]);
        
        let history = vec![wave1];
        let resonance = NeuroKernel::compute_semantic_resonance_static(&history, &wave2);
        
        // Orthogonal vectors should have resonance close to 0.0
        assert!(resonance < 0.1);
    }

    #[test]
    fn test_semantic_resonance_empty_history() {
        let wave = SemanticWave::from_reals(vec![0.5, 0.6]);
        let history: Vec<SemanticWave> = vec![];
        
        let resonance = NeuroKernel::compute_semantic_resonance_static(&history, &wave);
        
        // Empty history defaults to high resonance (no contradictions)
        assert_eq!(resonance, 1.0);
    }

    #[test]
    fn test_trajectory_divergence_computation() {
        let wave1 = SemanticWave::from_reals(vec![0.1, 0.2, 0.3]);
        let wave2 = SemanticWave::from_reals(vec![0.15, 0.25, 0.35]);
        let wave3 = SemanticWave::from_reals(vec![0.2, 0.3, 0.4]);
        
        let history = vec![wave1, wave2];
        let divergence = NeuroKernel::compute_trajectory_divergence(&history, &wave3);
        
        // Trajectory divergence should be in [0.0, 1.0]
        assert!(divergence >= 0.0);
        assert!(divergence <= 1.0);
    }

    #[test]
    fn test_decision_matrix_all_quadrants() {
        // Quadrant 1: High W, High R III Healthy
        assert_eq!(
            NeuroKernel::assess_cognitive_health(0.10, 0.80),
            CognitiveHealth::Healthy
        );
        
        // Quadrant 2: High W, Low R III Healthy
        assert_eq!(
            NeuroKernel::assess_cognitive_health(0.10, 0.20),
            CognitiveHealth::Healthy
        );
        
        // Quadrant 3: Low W, High R III CreativeLeap
        assert_eq!(
            NeuroKernel::assess_cognitive_health(0.005, 0.80),
            CognitiveHealth::CreativeLeap
        );
        
        // Quadrant 4: Low W, Low R III Hallucination
        assert_eq!(
            NeuroKernel::assess_cognitive_health(0.005, 0.20),
            CognitiveHealth::Hallucination
        );
    }

    #[test]
    fn test_threshold_boundary_conditions() {
        // Just below critical
        assert_eq!(
            NeuroKernel::assess_cognitive_health(0.009, 0.80),
            CognitiveHealth::CreativeLeap
        );
        
        // Just above critical
        assert_eq!(
            NeuroKernel::assess_cognitive_health(0.011, 0.80),
            CognitiveHealth::CreativeLeap // Still in warning zone
        );
        
        // Just below warning
        assert_eq!(
            NeuroKernel::assess_cognitive_health(0.049, 0.80),
            CognitiveHealth::CreativeLeap
        );
        
        // Just above warning
        assert_eq!(
            NeuroKernel::assess_cognitive_health(0.051, 0.80),
            CognitiveHealth::Healthy
        );
    }
}














