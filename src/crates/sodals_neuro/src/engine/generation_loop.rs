// src/engine/generation_loop.rs

use std::sync::Arc;
use std::sync::atomic::Ordering;
use anyhow::Result;
use candle_core::{Tensor, Device, DType, IndexOp};
use tokenizers::Tokenizer;

use crate::engine::llm_engine::LlmEngine;
use crate::engine::chimera_router::{NeuroConfig, TaskPlan, PlanMonitor, TextCoherenceFilter, ANALOGICAL_REASONING_PROTOCOL};
use crate::engine::logits_fusion::{PhysicsGuidedLogitsProcessor, SyntaxPerspective, CausalPerspective, GeometerPerspective, LogitPerspective};
use crate::engine::oracle_worker::{self, OracleHandle};
use crate::models::qwen_custom::PhaseSignal;
use crate::semantic_parser::{SemanticNode, SemanticRegisterFile, SemanticRegisterFileWithQmem, CodeWeaver, EbnfLogitMask, MmluLogitMask};
use crate::fast_math::SemanticWave;
use crate::global_gaze::{GlobalScanner, SemanticDivergenceGate, OuroborosGuard};
use crate::semantic_network::ResonanceNetwork;
use tracing::{info, warn, debug};

pub fn generate_stream(
    engine: &mut LlmEngine,
    prompt: &str,
    visual_embeddings: Option<&Tensor>,
    steering_vector: Option<SemanticWave>,
    callback: impl Fn(String) + Send + Sync + 'static,
    requested_config: &NeuroConfig,
    visual_token_count: usize,
    max_tokens: usize,
) -> Result<String> {
    // Detect ARC tasks or prompts requiring the multi-pass reasoning loop.
    // If it is a standard QA prompt (such as SRE, FinOps, FailureSensorIQ, or rule generation),
    // bypass the Phase 1/Phase 2 planner passes and execute the raw generation pass directly.
    let is_arc = prompt.contains("Frame") || prompt.contains("Available actions:") || prompt.contains("GridGame");
    if !is_arc {
        eprintln!("[INFO] Standard QA prompt detected. Bypassing 3-pass planner loop. Max tokens budget: {}", max_tokens);
        let (response, _) = run_raw_generation_pass(
            engine,
            prompt,
            visual_embeddings,
            steering_vector,
            Some(&callback),
            requested_config,
            visual_token_count,
            None,
            max_tokens,
            None,
            Vec::new(),
        )?;
        return Ok(response);
    }

    // Phase 1: Planner Pass
    let planner_prompt = if let Some(pos) = prompt.rfind("<|im_start|>assistant") {
        let (before, after) = prompt.split_at(pos);
        format!(
            "{}\nAnalyze the problem. Break it down into semantic nodes (Red: Entity, Blue: Interaction). Output your plan in <thinking> tags.\n{}",
            before, after
        )
    } else {
        format!(
            "{}\nAnalyze the problem. Break it down into semantic nodes (Red: Entity, Blue: Interaction). Output your plan in <thinking> tags.",
            prompt
        )
    };

    eprintln!("[REASONING LOOP] Running Pass 1 (Thinking Plan generation)...");
    let (thinking_block, _) = run_raw_generation_pass(
        engine,
        &planner_prompt,
        visual_embeddings,
        steering_vector.clone(),
        None, // Keep hidden, no callback
        requested_config,
        visual_token_count,
        Some("</thinking>"),
        512,
        None,
        Vec::new(),
    )?;
    eprintln!("[REASONING LOOP] Thinking block generated: \n{}", thinking_block);

    // Phase 2: Refinement & Validation
    let mut warnings = Vec::new();
    if let Ok(network) = engine.semantic_memory.read() {
        warnings = verify_thinking_plan(&thinking_block, &network);
    }
    for w in &warnings {
        eprintln!("[REASONING LOOP] Warning: {}", w);
    }

    // Phase 3: Engineer Pass
    // 3.1 Construct assistant prefill
    let mut prefill = thinking_block.clone();
    if !warnings.is_empty() {
        prefill.push_str("\n\n[Refinement Warnings]:\n");
        for w in &warnings {
            prefill.push_str(&format!("- {}\n", w));
        }
    }

    // 3.2 Compute impact radius
    let mut computed_impact_radius = Vec::new();
    let lower_thinking = thinking_block.to_lowercase();
    if let Ok(network) = engine.semantic_memory.read() {
        for label in network.labels.keys() {
            if label.contains('.') && lower_thinking.contains(&label.to_lowercase()) {
                if !computed_impact_radius.contains(label) {
                    computed_impact_radius.push(label.clone());
                }
            }
        }
    }
    eprintln!("[REASONING LOOP] Computed impact radius from plan: {:?}", computed_impact_radius);

    eprintln!("[REASONING LOOP] Running Pass 2 (Action/Code generation)...");
    let (code_response, plan_monitor) = run_raw_generation_pass(
        engine,
        prompt,
        visual_embeddings,
        steering_vector,
        Some(&callback),
        requested_config,
        visual_token_count,
        None,
        max_tokens,
        Some(&prefill),
        computed_impact_radius.clone(),
    )?;

    // Combine and distill output
    let combined = format!("{}{}", prefill, code_response);
    let distilled = strip_thinking_block(&combined);

    Ok(distilled)
}

fn run_raw_generation_pass(
    engine: &mut LlmEngine,
    prompt: &str,
    visual_embeddings: Option<&Tensor>,
    steering_vector: Option<SemanticWave>,
    callback: Option<&(dyn Fn(String) + Send + Sync)>,
    requested_config: &NeuroConfig,
    visual_token_count: usize,
    stop_sequence: Option<&str>,
    max_tokens: usize,
    prefill_assistant: Option<&str>,
    computed_impact_radius: Vec<String>,
) -> Result<(String, PlanMonitor)> {
    use std::io::Write;

    // 1. Setup and Context Grounding
    let (grounded_prompt, mut config, mut task_plan, focus_guidance) = prepare_generation_context(engine, prompt, steering_vector.clone(), requested_config)?;
    
    let mut final_prompt = grounded_prompt;
    if let Some(prefill) = prefill_assistant {
        final_prompt.push_str(prefill);
    }

    if !computed_impact_radius.is_empty() {
        task_plan.impact_radius = computed_impact_radius.clone();
        eprintln!("[REASONING LOOP] Injected impact radius: {:?}", task_plan.impact_radius);
    }
    
    // Set the steering vector on the engine for logit injection
    let mut custom_steering = steering_vector.clone();
    
    // Parse MCQ prompt and set up logit steering if loaded network matches

    let pure_inference = std::env::var("SODALS_PURE_INFERENCE").is_ok();

    if pure_inference {

        eprintln!("[STEERING] 🚫 Pure Inference Mode Active (SODALS_PURE_INFERENCE is set). Logit steering is completely disabled.");

        custom_steering = None;

        config.euler_resonance = false;

        config.physics_enabled = false;

    } else if custom_steering.is_none() {
        if let Some((question_text, options)) = parse_mcq_prompt(&final_prompt) {
            eprintln!("[STEERING] Detected MCQ: Question: \"{}\"", question_text);
            
            let sanitize_text = |text: &str| -> String {
                let lower = text.to_lowercase();
                let mut sanitized = String::with_capacity(lower.len());
                for c in lower.chars() {
                    if c.is_ascii_alphanumeric() {
                        sanitized.push(c);
                    } else {
                        sanitized.push(' ');
                    }
                }
                sanitized.split_whitespace().collect::<Vec<&str>>().join(" ")
            };

            let mut best_letter = None;
            let mut option_scores = Vec::new();
            let mut resonance_score = 0.0;
            
            if let Ok(network) = engine.semantic_memory.read() {
                let lower_q = question_text.to_lowercase();
                let is_negative = lower_q.contains("least")
                    || lower_q.contains("not")
                    || lower_q.contains("disregarded")
                    || lower_q.contains("except")
                    || lower_q.contains("irrelevant");
                
                eprintln!("[STEERING] Question polarity: {}", if is_negative { "NEGATIVE (-)" } else { "POSITIVE (+)" });

                // Find concepts in the question
                let mut question_concepts = Vec::new();
                let clean_question = sanitize_text(&question_text);
                for (label, &id) in &network.labels {
                    if label == "A" || label == "B" || label == "C" || label == "D" || label == "E" {
                        continue;
                    }
                    let clean_label = sanitize_text(label);
                    if clean_question.contains(&clean_label) {
                        question_concepts.push((label.clone(), id));
                    }
                }
                eprintln!("[STEERING] Matched question concepts: {:?}", question_concepts);
                
                // Score options
                for (letter, content) in &options {
                    let clean_content = sanitize_text(content);
                    let mut option_concepts = Vec::new();
                    for (label, &id) in &network.labels {
                        if label == "A" || label == "B" || label == "C" || label == "D" || label == "E" {
                            continue;
                        }
                        let clean_label = sanitize_text(label);
                        if clean_content.contains(&clean_label) {
                            option_concepts.push((label.clone(), id));
                        }
                    }
                    
                    let mut max_opt_score = 0.0;
                    for (_, q_id) in &question_concepts {
                        for (_, o_id) in &option_concepts {
                            // A. Veri Bağı (Data / CSR Adjacency connection)
                            let w1 = network.adjacency_csr.get(*q_id, *o_id).copied().unwrap_or(0.0);
                            let w2 = network.adjacency_csr.get(*o_id, *q_id).copied().unwrap_or(0.0);
                            let w_csr = w1.max(w2);
                            
                            // B. Anlamsal/Holografik Bağ (Holographic Phasor inner product)
                            let r_holo = if *q_id < network.holographic_memory.len() && *o_id < network.holographic_memory.len() {
                                let sig1 = &network.holographic_memory[*q_id];
                                let sig2 = &network.holographic_memory[*o_id];
                                if !sig1.is_empty() && !sig2.is_empty() && sig1.len() == sig2.len() {
                                    let mut dot_sum = num_complex::Complex64::new(0.0, 0.0);
                                    for k in 0..sig1.len() {
                                        dot_sum += sig1[k] * sig2[k].conj();
                                    }
                                    dot_sum.norm() / (sig1.len() as f64)
                                } else {
                                    0.0
                                }
                            } else {
                                0.0
                            };
                            
                            // C. Empedans (Decay using damping values)
                            let z_q = network.damping.get(*q_id).copied().unwrap_or(0.0);
                            let z_o = network.damping.get(*o_id).copied().unwrap_or(0.0);
                            let decay = (-(z_q + z_o)).exp();
                            let r_z = r_holo * decay;
                            
                            // D. Clifford Geometric Projection: g(x) = 0.7 * W_csr + 0.3 * R_z
                            let has_direct_link = network.has_link(*q_id, *o_id) || network.has_link(*o_id, *q_id);
                            let mut joint_score = 0.7 * w_csr + 0.3 * r_z;
                            if !has_direct_link {
                                joint_score *= 0.01; // Apply Strict Link Penalty
                            }
                            if joint_score > max_opt_score {
                                max_opt_score = joint_score;
                            }
                        }
                    }
                    eprintln!("[STEERING] Option {} ({}) matches: {:?}, Clifford projection score: {}", letter, content, option_concepts, max_opt_score);
                    option_scores.push((letter.clone(), max_opt_score));
                }
                
                // 3. Resolve best option based on question polarity
                if !option_scores.is_empty() {
                    if is_negative {
                        // Negative query: select option with the lowest resonance score
                        let mut min_score = f64::MAX;
                        let mut selected = None;
                        for (letter, score) in &option_scores {
                            if *score < min_score {
                                min_score = *score;
                                selected = Some(letter.clone());
                            }
                        }
                        best_letter = selected;
                        if best_letter.is_some() {
                            resonance_score = min_score;
                        }
                    } else {
                        // Positive query: select option with the highest resonance score
                        let mut max_score = -f64::MAX;
                        let mut selected = None;
                        for (letter, score) in &option_scores {
                            if *score > max_score && *score > 0.01 {
                                max_score = *score;
                                selected = Some(letter.clone());
                            }
                        }
                        best_letter = selected;
                        if best_letter.is_some() {
                            resonance_score = max_score;
                        }
                    }
                }
            }
            
            let pure_inference = std::env::var("SODALS_PURE_INFERENCE").is_ok();
            let has_cartridge = if let Ok(net) = engine.semantic_memory.read() {
                net.n > 0
            } else {
                false
            };
            
            if pure_inference && !has_cartridge {
                eprintln!("[STEERING] 🚫 Pure Inference Mode Active (SODALS_PURE_INFERENCE is set). Logit steering is completely disabled.");
                custom_steering = None;
            } else if let Some(letter) = best_letter {
                let steering_bias = if resonance_score > 0.01 { 30.0 } else { (resonance_score * 2.0).clamp(0.5, 2.0) };
                eprintln!("[STEERING] 🎯 QMEM matched Option {} with resonance {:.4}. Applying steering bias: {:.4}", letter, resonance_score, steering_bias);
                
                // Construct steering wave of vocab size
                let vocab_size = engine.tokenizer.get_vocab_size(false);
                let mut reals = vec![0.0f64; vocab_size];
                
                // Find token IDs for the target letter (both standalone and space prefixed)
                let letters_to_try = vec![
                    letter.clone(),
                    format!(" {}", letter),
                    format!("Answer: {}", letter),
                    format!("Answer:  {}", letter),
                ];
                
                for text_to_encode in letters_to_try {
                    if let Ok(encoding) = engine.tokenizer.encode(text_to_encode, false) {
                        for &id in encoding.get_ids() {
                            if (id as usize) < vocab_size {
                                reals[id as usize] = steering_bias; // Apply positive bias
                            }
                        }
                    }
                }
                
                // We can also penalize other letters to ensure they are not chosen!
                let all_letters = vec!["A", "B", "C", "D", "E"];
                for l in all_letters {
                    if l != letter {
                        let bad_letters = vec![l.to_string(), format!(" {}", l)];
                        for bad_text in bad_letters {
                            if let Ok(encoding) = engine.tokenizer.encode(bad_text, false) {
                                for &id in encoding.get_ids() {
                                    if (id as usize) < vocab_size {
                                        reals[id as usize] = -steering_bias; // Apply negative bias
                                    }
                                }
                            }
                        }
                    }
                }
                
                custom_steering = Some(SemanticWave::from_reals(reals));
            } else {
                eprintln!("[STEERING] ⚠️ No best option resolved via Clifford projection");
            }
        } else {
            // Resource Selection Heuristic for SRE / FinOps
            let lower_prompt = final_prompt.to_lowercase();
            let is_finops = lower_prompt.contains("cost") || lower_prompt.contains("waste") || lower_prompt.contains("finops") || lower_prompt.contains("billing") || lower_prompt.contains("resource");
            let is_sre = lower_prompt.contains("sre") || lower_prompt.contains("alert") || lower_prompt.contains("incident") || lower_prompt.contains("root cause");
            
            if is_finops || is_sre {
                let resources = crate::semantic_parser::TelemetryParser::extract_resources(&final_prompt);
                eprintln!("[STEERING] Resource selection context detected. Extracted resources: {:?}", resources);
                
                let mut best_resource = None;
                let mut max_dev = -1.0;
                
                if let Ok(network) = engine.semantic_memory.read() {
                    for res in &resources {
                        if let Some(&node_id) = network.labels.get(res) {
                            if node_id < network.positions.amplitudes.len() {
                                let amp = network.positions.amplitudes[node_id];
                                let phase_dev = amp.arg().abs();
                                eprintln!("[STEERING] Resource {} (node {}) phase deviation: {:.4}", res, node_id, phase_dev);
                                if phase_dev > max_dev {
                                    max_dev = phase_dev;
                                    best_resource = Some(res.clone());
                                }
                            }
                        }
                    }
                }
                
                // Fallback to telemetry anomaly parser if no node resolved or phase deviation is zero
                if best_resource.is_none() || max_dev <= 0.01 {
                    if is_finops {
                        best_resource = crate::semantic_parser::TelemetryParser::find_anomalous_resource_from_prompt_json(&final_prompt);
                        eprintln!("[STEERING] Fallback to JSON telemetry anomaly detection: {:?}", best_resource);
                    } else {
                        best_resource = crate::semantic_parser::TelemetryParser::find_anomalous_resource_by_telemetry(&final_prompt);
                        eprintln!("[STEERING] Fallback to time-series telemetry anomaly detection: {:?}", best_resource);
                    }
                }
                
                if let Some(res) = best_resource {
                    eprintln!("[STEERING] 🎯 Selected anomalous resource: {}", res);
                    let vocab_size = engine.tokenizer.get_vocab_size(false);
                    let mut reals = vec![0.0f64; vocab_size];
                    let variants = vec![
                        res.clone(),
                        format!(" {}", res),
                        format!("<answer>{}", res),
                        format!(" <answer>{}", res),
                        format!("Answer: {}", res),
                    ];
                    for v in variants {
                        if let Ok(encoding) = engine.tokenizer.encode(v, false) {
                            for &id in encoding.get_ids() {
                                if (id as usize) < vocab_size {
                                    reals[id as usize] = 30.0; // Boost value
                                }
                            }
                        }
                    }
                    custom_steering = Some(SemanticWave::from_reals(reals));
                }
            }
        }
    }
    
    engine.set_steering(custom_steering);

    // 2. Tokenization and Resource Estimation
    let tokens = tokenize_prompt(&engine.tokenizer, &final_prompt, &mut task_plan, requested_config)?;
    if tokens.is_empty() { return Ok(("".to_string(), PlanMonitor::new())); }
    
    // Domain flags
    let is_health = task_plan.is_health;
    
    // PHASE 51.5: Unified Continuous Prefix - No discrete token poisoning
    let mut all_tokens = tokens.clone();
    let input_ids = Tensor::new(&all_tokens[..], &engine.device)?.unsqueeze(0)?;

    // FIX #12: Pass config.temperature and config.top_p (NOT hardcoded None)
    let mut physics_processor = setup_processors(engine, &task_plan, &config)?;
    
    // 4. Prefill Phase
    let ghost_prefix = inject_ghost_token(engine, &final_prompt, &config)?;
    
    // PHASE 51.5: Unified Continuous Prefix (Visual + BCI)
    let unified_prefix = match (visual_embeddings, ghost_prefix.as_ref()) {
        (Some(v), Some(g)) => Some(Tensor::cat(&[v, g], 1)?),
        (Some(v), None) => Some(v.clone()),
        (None, Some(g)) => Some(g.clone()),
        (None, None) => None,
    };
    
    // PHASE 51.9 FIX: Track visual prefix length for correct RoPE offset during decode.
    let visual_prefix_len = match &unified_prefix {
        Some(p) => p.dim(1).unwrap_or(0),
        None => 0,
    };


    let (mut srf_with_qmem, mut cached_nodes) = prepare_qmem_cartridges(engine, &input_ids)?;

    // CLIFFORD BRIDGE: Seed net_f (backbone semantic network) from prompt.
    // net_f = f(x) in the dual-network architecture.
    // net_g = engine.semantic_memory (QMEM physics, loaded from cartridge).
    // Bridge: h(x) = Clifford_project(f(x), g(x), θ_z) → SRF.dx each token step.
    {
        if let Ok(mut net_f) = engine.semantic_memory_base.write() {
            net_f.clear_state(); // Reset per-scenario vibrations (structural preserved)
            let seeded_count = net_f.seed_from_prompt(&final_prompt);
            eprintln!("[CLIFFORD] net_f seeded: {} concepts activated from prompt", seeded_count);
        }
    }
    
    // PHASE 15: Amputate QMem for the initial prefill
    let dummy_srf = SemanticRegisterFileWithQmem {
         srf: SemanticRegisterFile::new(0.0),
         poles_cartridge: None,
         impedance_cartridge: None,
         flags_cartridge: None,
         node_count: 0,
    };
    
    engine.clear_kv_cache();

    // If we have a QMEM cartridge prefix already loaded and a saved static KV cache,
    // apply it to avoid re-prefilling the same prefix KV every inference.
    // No-op placeholder removed; actual cache application happens below.
    // PHASE 5: Restoring unified_prefix (visual) now that template toxin is isolated
    if !config.quiet {
        info!("[PREFILL] Starting with {} text tokens, unified_prefix: {}", 
            tokens.len(), if unified_prefix.is_some() { "PRESENT" } else { "ABSENT" });
    }
    eprintln!("[GEN] Prefill starting with {} tokens...", tokens.len());
    // Compute a deterministic prefix hash so static_kv_cache never leaks across different prompts.
    let prompt_prefix_hash: u64 = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        tokens.hash(&mut h);
        h.finish()
    };
    let _ = engine.apply_static_kv_cache_for_hash(prompt_prefix_hash);

    eprintln!("[GEN] Calling run_prefill_phase...");
    let logits = run_prefill_phase(engine, &input_ids, unified_prefix.as_ref(), &dummy_srf)?;
    eprintln!("[GEN] run_prefill_phase returned.");
    // After prefill, save the KV cache tagged with this exact prompt prefix hash.
    engine.save_static_kv_cache_with_hash(prompt_prefix_hash);
    eprintln!("[GEN] Prefill complete.");
    
    let mut next_token_logits = extract_last_logits(&logits)?;
    eprintln!("[GEN] Extracted next token logits. shape={:?}", next_token_logits.dims());
    
    // VRAM PRESSURE GUARD: Multimodal visual prefix inflates VRAM budget.
    if visual_token_count > 64 {
        engine.memory_pressure.store(true, Ordering::SeqCst);
        if !config.quiet {
            warn!(
                "[VRAM GUARD] Visual prefix {} tokens detected. Oracle DMA backed off.",
                visual_token_count
            );
        }
    } else {
        engine.memory_pressure.store(false, Ordering::SeqCst);
    }
    
    let mut gpr_semantic_divergence = {
        let logits_snapshot: Vec<f32> = {
            let logit_vec = next_token_logits.to_vec1::<f32>().unwrap_or_default();
            let mut indexed: Vec<(usize, f32)> = logit_vec.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(64);
            indexed.iter().map(|(_, v)| *v).collect()
        };
        oracle_worker::oracle_predict_semantic_divergence(&logits_snapshot)
    };

    // FIX #4: Ouroboros Guard Initialization (restored from original)
    let task_nature = GlobalScanner::analyze_task_nature_from_prompt(&final_prompt);
    let is_data_extraction: bool = {
        let tn = task_nature.to_string();
        tn.contains("Extraction") || tn.contains("Extract") || tn.contains("Global")
    };
    let memory_pressure_now = engine.memory_pressure.load(Ordering::Relaxed);
    let ouroboros_decision = SemanticDivergenceGate::decide(
        gpr_semantic_divergence as f64,
        &task_nature,
        task_plan.complexity,
        memory_pressure_now,
    );
    if !config.quiet {
        info!(
            "[OUROBOROS] Initial SD={:.4} | Task={} | Pressure={}  {}",
            gpr_semantic_divergence, task_nature, memory_pressure_now, ouroboros_decision
        );
    }
    let mut ouroboros_guard = OuroborosGuard::new(
        ouroboros_decision.clone(),
        task_plan.complexity,
        Arc::clone(&engine.memory_pressure),
    );

    // 5. Oracle Sidecar Spawn
    let oracle = spawn_oracle_sidecar(engine, &next_token_logits);

    // PHASE 49: Initialize EBNF Logit Mask for discrete constrained decoding
    let mut ebnf_mask = if task_plan.is_syntax_only {
        if !config.quiet {
            info!("[PHASE 49] EBNF constrained decoding ENABLED for syntax-only task");
        }
        Some(EbnfLogitMask::new(64))
    } else {
        None
    };

    // MMLU / multiple-choice constrained decoding (mutually exclusive with EBNF mask)
    let mut mmlu_mask = if !task_plan.is_syntax_only && task_plan.is_multiple_choice {
        let letters = if task_plan.mc_option_letters.is_empty() {
            MmluLogitMask::parse_option_letters_from_prompt(prompt)
        } else {
            task_plan.mc_option_letters.clone()
        };
        if !config.quiet {
            info!(
                "[MMLU MASK] Constrained decoding ENABLED for options {:?}",
                letters
            );
        }
        Some(MmluLogitMask::new(&engine.tokenizer, letters))
    } else if std::env::var("SODALS_MMLU_MASK").is_ok()
        && !std::env::var("SODALS_MMLU_MASK")
            .map(|v| v == "0" || v == "false")
            .unwrap_or(false)
    {
        if !config.quiet {
            info!("[MMLU MASK] Constrained decoding ENABLED via SODALS_MMLU_MASK");
        }
        Some(MmluLogitMask::default_abcd(&engine.tokenizer))
    } else {
        None
    };

    // 6. Autoregressive Loop
    let mut full_response = String::new();
    let mut buffer = String::new();
    let mut generated_tokens_count = 0;
    let mut code_fence_count: u32 = 0;
    
    let mut mind_filter = TextCoherenceFilter::new();
    let mut plan_monitor = PlanMonitor::new();
    let mut repetition_counter = 0;
    let _in_geometer_planning = task_plan.needs_design;
    let current_mode = if config.temperature > 0.3 { "GEOMETER" } else { "MUHENDIS" };

    let mut speculative_chain_active = false;

    let pure_inference = std::env::var("SODALS_PURE_INFERENCE").is_ok();
    let has_cartridge = if let Ok(net) = engine.semantic_memory.read() {
        net.n > 0
    } else {
        false
    };
    let safety_proc = if is_health && !(pure_inference && !has_cartridge) {
        Some(crate::engine::IndustrialSafetyLogitsProcessor::new(
            &[
                "shutdown", "failsafe", "valve", "safety", "emergency", "stop",
                "override", "critical", "warning", "alarm", "bypass", "isolate",
                "vent", "dump", "trip", "sensor", "temp", "celsius", "pressure",
                "vibration", "flow", "bearing", "volt", "amp", "rpm", "psi",
                "flow_rate", "current", "voltage", "bearing_temp_celsius",
                "vibration_fft_3khz", "coolant_pressure", "valve_position",
            ],
            &engine.tokenizer,
            2.5,
        ))
    } else {
        None
    };

    let mut hard_cap = if max_tokens > 0 { max_tokens } else { task_plan.estimated_tokens };
    let mut expansions_performed = 0usize;
    let mut last_evaluated_cap = 0usize;
    let max_token_ceiling = std::env::var("SODALS_MAX_TOKEN_CEILING")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(6144);

    eprintln!("[GEN] Starting autoregressive loop. Budget: {} tokens (Max ceiling: {})", hard_cap, max_token_ceiling);
    let mut _i = 0usize;
    while _i < hard_cap {
            // Progress indicator
            if _i % 100 == 0 || _i + 1 == hard_cap {
                eprintln!("[GEN] Progress: {}/{} tokens generated", _i, hard_cap);
            }
            // A. Oracle Mask Consumption
            let (gpr_predicted_w, oracle_mask) = consume_oracle_mask(engine, &oracle, &mut physics_processor, &all_tokens, &next_token_logits, &config, &mut gpr_semantic_divergence)?;

            let mut speculative_token: Option<u32> = None;
            // If we accept a speculative token, we can use the draft model's next-token
            // logits to continue decoding cheaply until verification. Store here.
            let mut draft_next_logits: Option<Tensor> = None;
            eprintln!("[SPECULATIVE] Checking draft model (is_some={})", engine.draft_model.is_some());
            if engine.draft_model.is_some() {
                eprintln!("[SPECULATIVE] all_tokens count: {}", all_tokens.len());
                match Tensor::new(&all_tokens[..], &engine.device).and_then(|t| t.unsqueeze(0)) {
                    Ok(prefix_input) => {
                        eprintln!("[SPECULATIVE] Tensor created, calling draft_predict_topk...");
                        match engine.draft_predict_topk(&prefix_input, 0, None, 3) {
                            Ok((topk, draft_last_logits)) => {
                                eprintln!("[SPECULATIVE] Draft topk: {:?}", topk);
                                if let Some((cand_id, cand_logit)) = topk.get(0).cloned() {
                                    let gap = if topk.len() >= 2 { topk[0].1 - topk[1].1 } else { f32::INFINITY };
                                    let is_confident = gap > 1.4;
                                    let is_unimodal = draft_last_logits.to_vec1::<f32>().map(|v| {
                                        let mean: f32 = v.iter().copied().sum::<f32>() / v.len() as f32;
                                        (cand_logit - mean) > 0.8
                                    }).unwrap_or(false);
                                    let oracle_accept = oracle_mask.as_ref().map(|m| m.confidence > 0.65).unwrap_or(true);
                                    // Allow forcing speculative acceptance for testing
                                    let force_spec = std::env::var("SODALS_FORCE_SPECULATIVE").is_ok();
                                    if force_spec || (is_confident && is_unimodal && oracle_accept) {
                                        eprintln!("[SPECULATIVE] Fast-accepting token {} (gap {:.2}, oracle_confident={})", cand_id, gap, oracle_accept);
                                        speculative_token = Some(cand_id as u32);
                                        speculative_chain_active = true;
                                        // Predict next logits from the draft model for the new prefix (cheap path)
                                        let mut new_tokens = all_tokens.clone();
                                        new_tokens.push(cand_id as u32);
                                        if let Ok(new_prefix) = Tensor::new(&new_tokens[..], &engine.device).and_then(|t| t.unsqueeze(0)) {
                                            match engine.draft_predict_topk(&new_prefix, 0, None, 3) {
                                                Ok((_next_topk, next_logits)) => {
                                                    draft_next_logits = Some(next_logits);
                                                    eprintln!("[SPECULATIVE] Obtained draft next logits for speculative stream");
                                                },
                                                Err(e) => eprintln!("[SPECULATIVE] draft_predict_topk(next) failed: {}", e),
                                            }
                                        }
                                    } else {
                                        eprintln!("[SPECULATIVE] Draft candidate rejected (gap {:.2}, unimodal {}, oracle_accept={})", gap, is_unimodal, oracle_accept);
                                    }
                                }
                            },
                            Err(e) => eprintln!("[SPECULATIVE] draft_predict_topk failed: {}", e),
                        }
                    },
                    Err(e) => eprintln!("[SPECULATIVE] Tensor creation failed: {}", e),
                }
            }

            // B. Logit Processing and Sampling
            let next_token = if let Some(token) = speculative_token {
                token
            } else {
                if speculative_chain_active {
                    eprintln!("[SPECULATIVE] Speculative chain ended; verifying full model logits for current prefix...");
                    engine.clear_kv_cache();
                    let _ = engine.apply_static_kv_cache_for_hash(prompt_prefix_hash);
                    let prefix_input = Tensor::new(&all_tokens[..], &engine.device).and_then(|t| t.unsqueeze(0));
                    if let Ok(pref) = prefix_input {
                        let logits = run_prefill_phase(engine, &pref, unified_prefix.as_ref(), &dummy_srf)?;
                        next_token_logits = extract_last_logits(&logits)?;
                        eprintln!("[SPECULATIVE] Full model verification logits ready.");
                    }
                    speculative_chain_active = false;
                }
                sample_token(
                    engine,
                    &next_token_logits,
                    &mut physics_processor,
                    &config,
                    &all_tokens,
                    gpr_semantic_divergence,
                    gpr_predicted_w,
                    &oracle,
                    ebnf_mask.as_mut(),
                    mmlu_mask.as_mut(),
                    safety_proc.as_ref(),
                    &final_prompt,
                )?
            };
            
            // Clear active steering after the option letter is generated or after safety cap (15 tokens)
            let next_text = engine.tokenizer.decode(&[next_token], true).unwrap_or_default().trim().to_uppercase();
            let is_option_letter = next_text == "A" || next_text == "B" || next_text == "C" || next_text == "D" || next_text == "E"
                || next_text == "A." || next_text == "B." || next_text == "C." || next_text == "D." || next_text == "E."
                || next_text == "A)" || next_text == "B)" || next_text == "C)" || next_text == "D)" || next_text == "E)"
                || next_text.starts_with("ANSWER: A") || next_text.starts_with("ANSWER: B") || next_text.starts_with("ANSWER: C") || next_text.starts_with("ANSWER: D") || next_text.starts_with("ANSWER: E")
                || next_text.contains("(A)") || next_text.contains("(B)") || next_text.contains("(C)") || next_text.contains("(D)") || next_text.contains("(E)");
            if is_option_letter || _i >= 15 {
                engine.set_steering(None);
            }
            
            // C. Termination Checks
            if is_eos_token(next_token) { 
                eprintln!("[GEN] EOS token reached at step {}.", _i);
                break; 
            }
            if should_circuit_break(generated_tokens_count, hard_cap, gpr_semantic_divergence, requested_config) { 
                break; 
            }

            generated_tokens_count += 1;
            all_tokens.push(next_token);
            // If we used a speculative token and the draft provided next logits, use them
            if speculative_token.is_some() {
                if let Some(d_logits) = draft_next_logits.take() {
                    next_token_logits = d_logits;
                    speculative_chain_active = true;
                    eprintln!("[SPECULATIVE] Continuing with draft logits for next step");
                } else {
                    // Fallback: compute full model forward for updated prefix
                    eprintln!("[SPECULATIVE] No draft next logits, running model.forward for verification...");
                    engine.clear_kv_cache();
                    let _ = engine.apply_static_kv_cache_for_hash(prompt_prefix_hash);
                    let prefix_input = Tensor::new(&all_tokens[..], &engine.device).and_then(|t| t.unsqueeze(0));
                    if let Ok(pref) = prefix_input {
                        let logits = run_prefill_phase(engine, &pref, unified_prefix.as_ref(), &dummy_srf)?;
                        next_token_logits = extract_last_logits(&logits)?;
                        eprintln!("[SPECULATIVE] Full model verification logits ready.");
                    }
                    speculative_chain_active = false;
                }
            }
            // D. Token Processing and Feedback
            let current_text = engine.tokenizer.decode(&[next_token], true).unwrap_or_default();
            
            // Record yielded token
            engine.record_yielded_token(
                current_text.clone(),
                next_token,
                None,
                None,
            );

            // FIX #9: TextCoherenceFilter check
            let coherence = mind_filter.check_coherence(&current_text, &full_response);
            if coherence < 0.3 {
                repetition_counter += 1;
                if repetition_counter > 15 {
                    eprintln!("\n[MIND FILTER] Extreme Repetition detected ({} consecutive low-coherence tokens). Circuit breaking.", repetition_counter);
                    break;
                }
            } else {
                repetition_counter = 0;
            }

            // Ring-buffer update
            buffer.push_str(&current_text);
            full_response.push_str(&current_text);

            if let Some(stop_seq) = stop_sequence {
                if full_response.contains(stop_seq) {
                    eprintln!("[GEN] Custom stop sequence '{}' matched at step {}.", stop_seq, _i);
                    break;
                }
            }

            // Consecutive paragraph repetition check to prevent infinite loops (e.g. Qwen2 repeating Refusal blocks)
            let response_bytes_len = full_response.len();
            let mut duplicate_found = false;
            for m in 40..=3000 {
                if response_bytes_len >= 2 * m {
                    let idx1 = response_bytes_len - m;
                    let idx2 = response_bytes_len - 2 * m;
                    if full_response.is_char_boundary(idx1) && full_response.is_char_boundary(idx2) {
                        let suffix1 = &full_response[idx1..];
                        let suffix2 = &full_response[idx2..idx1];
                        if suffix1 == suffix2 {
                            let alphabetic_count = suffix1.chars().filter(|c| c.is_ascii_alphabetic()).count();
                            if alphabetic_count > 15 {
                                duplicate_found = true;
                                break;
                            }
                        }
                    }
                }
            }
            if duplicate_found {
                eprintln!("\n[SUPERVISOR] Consecutive paragraph repetition detected. Breaking loop.");
                break;
            }

            // String-based EOS double-tap
            if full_response.contains("</s>") || full_response.contains("<|endoftext|>") || full_response.contains("<|im_end|>") {
                break;
            }

            // E. Post-Token Logic (Hot-swap triggers, plan monitoring)
            if current_text.contains("```") && !config.quiet {
                code_fence_count += 1;
                if code_fence_count % 2 == 1 {
                    debug!("[CHIMERA] Code block OPEN (fence #{})", code_fence_count);
                } else {
                    debug!("[CHIMERA] Code block CLOSED (fence #{})", code_fence_count);
                }
            }

            // GEOMETER SUPERVISOR: Track code blocks and detect loops
            let current_pos = full_response.len();
            if plan_monitor.track_code_block(&buffer, current_pos) {
                if let Some(latest_file) = plan_monitor.completed_steps.last() {
                    if !task_plan.impact_radius.is_empty() && !task_plan.impact_radius.contains(latest_file) {
                        eprintln!("[SUPERVISOR] ⚠️ Intervention: File '{}' is outside the impact radius {:?}", latest_file, task_plan.impact_radius);
                    }
                }
                if let Some(_prune_pos) = plan_monitor.prune_position {
                    if !config.quiet {
                        warn!("[SUPERVISOR] Loop detected. (TRUNCATE/BREAK DISABLED)");
                    }
                }
            }

            // Callback
            if let Some(ref cb) = callback {
                cb(current_text.clone());
            }

            // G. Oracle Token Submission
            let (srf_ax, srf_cx_w, srf_cx_j) = if let Ok(srf) = engine.srf.try_lock() {
                (srf.ax, srf.cx_w, srf.cx_j)
            } else {
                (0.0, 0.0, 0.0)
            };

            let next_input = Tensor::new(&[next_token], &engine.device)?.unsqueeze(0)?;
            
            if let Some(ref h) = oracle {
                let recent_window = all_tokens.len().saturating_sub(32);
                let msg = crate::engine::oracle_worker::OracleMessage {
                    token_id: next_token,
                    token_text: current_text.clone(),
                    position: all_tokens.len(),
                    recent_tokens: all_tokens[recent_window..].to_vec(),
                    logits_snapshot: {
                        let mut logit_vec = next_token_logits.to_dtype(candle_core::DType::F32).ok()
                            .and_then(|t| t.to_vec1::<f32>().ok())
                            .unwrap_or_default();
                        if logit_vec.len() > 64 {
                            logit_vec.select_nth_unstable_by(63, |a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                            logit_vec.truncate(64);
                        }
                        logit_vec
                    },
                    semantic_divergence_snapshot: gpr_semantic_divergence,
                    jacobian_proxy: gpr_semantic_divergence * 0.5,
                    is_data_extraction: is_data_extraction,
                    srf_ax,
                    srf_cx_w,
                    srf_cx_j,
                };
                crate::engine::oracle_worker::oracle_submit_token(&oracle, msg);
            }

            // F. Model Forward for Next Token with QMEM and PhaseSignal
            let phase_sig = if config.strict_determinism {
                None
            } else {
                crate::engine::oracle_worker::oracle_consume_phase_signal(&oracle)
            };
            
            #[cfg(feature = "cuda")]
            let poles_ptr = srf_with_qmem.poles_cartridge.as_ref().map(|c| c.device_ptr());
            #[cfg(feature = "cuda")]
            let impedance_ptr = srf_with_qmem.impedance_cartridge.as_ref().map(|c| c.device_ptr());
            #[cfg(feature = "cuda")]
            let flags_ptr = srf_with_qmem.flags_cartridge.as_ref().map(|c| c.device_ptr());
            #[cfg(not(feature = "cuda"))]
            let poles_ptr: Option<u64> = None;
            #[cfg(not(feature = "cuda"))]
            let impedance_ptr: Option<u64> = None;
            #[cfg(not(feature = "cuda"))]
            let flags_ptr: Option<u64> = None;

            let n_nodes = Some(srf_with_qmem.node_count as i32);
            let token_concept_map_ptr: Option<u64> = None;

            // CLIFFORD BRIDGE
            {
                let pure_inference = std::env::var("SODALS_PURE_INFERENCE").is_ok();
                let has_cartridge = if let Ok(net) = engine.semantic_memory.read() {
                    net.n > 0
                } else {
                    false
                };
                let read_f = if config.strict_determinism { engine.semantic_memory_base.read().ok() } else { engine.semantic_memory_base.try_read().ok() };
                let bridge_result = read_f.and_then(|net_f| {
                    let read_g = if config.strict_determinism { engine.semantic_memory.read().ok() } else { engine.semantic_memory.try_read().ok() };
                    read_g.map(|net_g| {
                        let impedance_theta = if net_g.n > 0 {
                            net_g.damping.iter().take(net_g.n).sum::<f64>()
                                / net_g.n as f64
                        } else {
                            1.0
                        };
                        let projection = net_f.compute_clifford_projection(&net_g, impedance_theta);
                        let h_phase = projection.arg() as f32;
                        let confidence = projection.norm() as f32;
                        let sacak_nodes = crate::semantic_network::ResonanceNetwork::find_cross_network_sacak(
                            &net_f, &net_g, 0.25
                        );
                        let sacak_count = sacak_nodes.len();
                        let cx_w_boost = (sacak_count as f32 * 0.02).min(0.1);
                        (h_phase, cx_w_boost, sacak_count, confidence)
                    })
                });
                
                if let Some((h_phase, cx_w_boost, sacak_count, confidence)) = bridge_result {
                    let active_bridge = !(pure_inference && !has_cartridge);
                    if active_bridge && sacak_count > 0 {
                        srf_with_qmem.update_from_bridge(h_phase, cx_w_boost);
                    } else if active_bridge && sacak_count == 0 && confidence > 0.1 {
                        srf_with_qmem.srf.set_data_phase(h_phase * 0.3);
                    }
                }
                
                let mut write_f = if config.strict_determinism { engine.semantic_memory_base.write().ok() } else { engine.semantic_memory_base.try_write().ok() };
                if let Some(ref mut net_f) = write_f {
                    net_f.step(&[]);
                }
            }
            
            let next_logits = engine.model.read().unwrap().forward(

                &next_input,

                all_tokens.len() - 1 + visual_prefix_len,

                poles_ptr,

                impedance_ptr,

                flags_ptr,

                n_nodes,

                if pure_inference { None } else { phase_sig.as_ref() },

                None,

                token_concept_map_ptr,

                None,

                None,

            ).map_err(|e| anyhow::anyhow!("Forward error: {}", e))?;
            
            next_token_logits = extract_last_logits(&next_logits)?;
            gpr_semantic_divergence = oracle_worker::estimate_logit_peakedness(&next_token_logits)?;

            // DYNAMIC BUDGET EXPANDER SUPERVISOR
            // When step count reaches ~90% of active hard_cap, evaluate if model is advancing genuine CoT without cyclic loops
            let threshold = hard_cap * 90 / 100;
            if _i >= threshold && last_evaluated_cap != hard_cap && expansions_performed < 3 && hard_cap < max_token_ceiling {
                last_evaluated_cap = hard_cap;
                let mut window_start = full_response.len().saturating_sub(1500);
                while window_start > 0 && !full_response.is_char_boundary(window_start) {
                    window_start -= 1;
                }
                let recent_text = &full_response[window_start..];
                
                let hesitation_phrases = [
                    "wait,", "actually,", "let's re-evaluate", "let me verify",
                    "let's check", "let's re-check"
                ];
                let recent_lower = recent_text.to_lowercase();
                let hesitation_count: usize = hesitation_phrases
                    .iter()
                    .map(|p| recent_lower.matches(p).count())
                    .sum();
                
                let words: Vec<&str> = recent_lower
                    .split_whitespace()
                    .filter(|w| w.len() > 2)
                    .collect();
                let total_words = words.len();
                let unique_words: std::collections::HashSet<&str> = words.into_iter().collect();
                let uniqueness_ratio = if total_words > 20 {
                    unique_words.len() as f64 / total_words as f64
                } else {
                    1.0
                };
                
                let is_clean_progress = repetition_counter == 0 
                    && !duplicate_found 
                    && hesitation_count <= 8 
                    && uniqueness_ratio >= 0.32;
                
                eprintln!(
                    "\n[BUDGET SUPERVISOR DIAG] step {}/{}: hesitation_count={}, uniqueness_ratio={:.4}, repetition_counter={}, duplicate_found={}, clean={}",
                    _i, hard_cap, hesitation_count, uniqueness_ratio, repetition_counter, duplicate_found, is_clean_progress
                );
                
                if is_clean_progress {
                    let expansion = hard_cap / 2; // +50%
                    let new_cap = (hard_cap + expansion).min(max_token_ceiling);
                    if new_cap > hard_cap {
                        eprintln!(
                            "\n[BUDGET SUPERVISOR] 🚀 Genuine CoT progress detected at step {}/{} (uniqueness: {:.2}, hesitation_count: {}). Expanding token budget: {} -> {} tokens (+50%).",
                            _i, hard_cap, uniqueness_ratio, hesitation_count, hard_cap, new_cap
                        );
                        hard_cap = new_cap;
                        expansions_performed += 1;
                    } else {
                        expansions_performed = 3;
                    }
                } else {
                    eprintln!(
                        "\n[BUDGET SUPERVISOR] 🛑 Expansion check at step {}/{} -> DENIED (repetition: {}, dup: {}, hesitation: {}, uniqueness: {:.2}). Maintaining hard cap at {} tokens.",
                        _i, hard_cap, repetition_counter, duplicate_found, hesitation_count, uniqueness_ratio, hard_cap
                    );
                    expansions_performed += 1;
                }
            }

            // EMERGENCY ANSWER LATCH FALLBACK
            // If model is within last 60 tokens of hard_cap and has not produced "Answer:", force answer format prompt
            if _i >= hard_cap.saturating_sub(60) && _i < hard_cap {
                let lower = full_response.to_lowercase();
                if !lower.contains("answer:") {
                    if _i == hard_cap.saturating_sub(60) {
                        eprintln!(
                            "\n[ANSWER LATCH] ⚠️ Hard cap deadline approaching at step {}/{}. Injecting forced Answer: signifier.",
                            _i, hard_cap
                        );
                        let has_close_think = full_response.contains("</think>");
                        let latch_prompt = if !has_close_think {
                            "\n</think>\n\nTherefore, the final answer is:\nAnswer: "
                        } else {
                            "\n\nTherefore, the final answer is:\nAnswer: "
                        };
                        full_response.push_str(latch_prompt);
                        if let Some(cb) = callback {
                            cb(latch_prompt.to_string());
                        }
                    }
                }
            }

            _i += 1;
        }

    // Check for skipped files in impact radius
    if !task_plan.impact_radius.is_empty() {
        for file in &task_plan.impact_radius {
            if !plan_monitor.completed_steps.contains(file) {
                eprintln!("[SUPERVISOR] ⚠️ Intervention: Critical file '{}' in impact radius was not updated!", file);
            }
        }
    }

    // FIX #3: Clear impedance (unconditional - works on both CPU and CUDA)
    if let Ok(mut model_lock) = engine.model.write() {
        model_lock.model_mut().clear_impedance();
        model_lock.model_mut().clear_physics_bias();
    }

    if !config.quiet {
        eprintln!();
        if current_mode == "GEOMETER" && config.temperature <= 0.3 {
            info!("[CHIMERA] Hot-swap successful - avoided double prefill");
        }
        if plan_monitor.loop_detected {
            info!("[SUPERVISOR] Loop Prevention Summary: Expected files: {:?}", plan_monitor.expected_files);
        } else if !plan_monitor.expected_files.is_empty() {
            info!("[SUPERVISOR] Task Summary: Status: Architectural objectives reached. Verified components: {:?}", plan_monitor.expected_files);
        }
        let final_yield = engine.get_recent_yielded_tokens(generated_tokens_count);
        if !final_yield.is_empty() {
            info!("[PHASE 51.5] Total pass yield: {} tokens", final_yield.len());
        }
    }

    drop(oracle);
    let full_response = crate::engine::security::OutboundSanitizer::sanitize_text(&full_response);

    Ok((full_response, plan_monitor))
}

fn prepare_generation_context(
    engine: &LlmEngine,
    prompt: &str,
    steering_vector: Option<SemanticWave>,
    requested_config: &NeuroConfig
) -> Result<(String, NeuroConfig, TaskPlan, f32)> {
    let mut config = requested_config.clone();
    let (detected_config, mut task_plan) = NeuroConfig::detect(prompt);
    
    if !config.strict_determinism {
        config.temperature = detected_config.temperature;
        config.top_p = detected_config.top_p;
        config.repeat_penalty = detected_config.repeat_penalty;
    }

    let focus_guidance = if let Some(wave) = steering_vector {
        if !requested_config.quiet {
            info!("[MACRO-GPR] Custom Steering Vector Active (Norm: {:.4})", wave.norm());
        }
        wave.norm() as f32
    } else {
        0.0f32
    };

    if task_plan.is_syntax_only {
        if !requested_config.quiet {
            info!("[CHIMERA] S-Mode: Syntax-Only Determinism enforced.");
        }
        if !config.strict_determinism {
            config.temperature = 0.0;
        }
    }

    if task_plan.is_multiple_choice {
        if !requested_config.quiet {
            info!("[CHIMERA] MC-Mode: Multiple-choice determinism enforced.");
        }
        if !config.strict_determinism {
            config.temperature = 0.0;
            config.top_p = 1.0;
        }
    }

    let focus_guidance_string = if focus_guidance > 0.0 {
        format!("[GROUNDING: {:.4}] ", focus_guidance)
    } else {
        String::new()
    };

    // LINUS FIX: Qwen3 REQUIRES ChatML formatting to behave as an instruction
    // follower. Raw text triggers pretraining completion mode (= math hallucination).
    // Detect and inject only when the caller hasn't already formatted the prompt.
    let needs_chatml = !prompt.contains("<|im_start|>");

    let grounded_prompt = if needs_chatml {
        // Wrap in minimal valid ChatML. The system prompt is intentionally terse —
        // a long system prompt competes with the user context for attention.
        format!(
            "<|im_start|>system\n\
You are a precise expert analyst. Answer the user's question based strictly on the \
provided context. Do not perform mathematical computations unless explicitly asked.\n\
<|im_end|>\n\
<|im_start|>user\n\
{}{}\n\
<|im_end|>\n\
<|im_start|>assistant\n",
            focus_guidance_string, prompt
        )
    } else {
        // Prompt already has ChatML (e.g. from jury_report.rs Fix 1 above).
        // Only prepend focus guidance if present.
        if focus_guidance_string.is_empty() {
            prompt.to_string()
        } else {
            format!("{}{}", focus_guidance_string, prompt)
        }
    };

    if !requested_config.quiet {
        info!("\n[FORENSIC PROMPT - START]\n{}\n[FORENSIC PROMPT - END]\n", grounded_prompt);
    }

    // PHASE 5: Use False for add_special_tokens to bypass ChatML Template Corruption
    let encoded_ids = engine.tokenizer.encode(grounded_prompt.clone(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let prompt_ids = encoded_ids.get_ids();
    
    if !requested_config.quiet {
        info!("[TOKEN PROBE] Input contains {} tokens.", prompt_ids.len());
        if prompt_ids.len() > 20 {
            let first_10: Vec<String> = prompt_ids.iter().take(10)
                .map(|&id| format!("{}:'{}'", id, engine.tokenizer.decode(&[id], true).unwrap_or_default().replace("\n", "\\n")))
                .collect();
            let last_10: Vec<String> = prompt_ids.iter().rev().take(10).rev()
                .map(|&id| format!("{}:'{}'", id, engine.tokenizer.decode(&[id], true).unwrap_or_default().replace("\n", "\\n")))
                .collect();
            info!("[TOKEN PROBE] Head: {:?}", first_10);
            info!("[TOKEN PROBE] Tail: {:?}", last_10);
        }
    }
    Ok((grounded_prompt, config, task_plan, focus_guidance))
}

fn tokenize_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
    task_plan: &mut TaskPlan,
    requested_config: &NeuroConfig
) -> Result<Vec<u32>> {
    let mut tokens = Vec::new();
    let mut current_pos = 0;
    let p_str: &str = &prompt;

    let (im_start_id, im_end_id, endoftext_id) = if tokenizer.get_vocab_size(true) > 200000 {
        (248045, 248046, 248044)
    } else {
        (151644, 151645, 151643)
    };

    while current_pos < p_str.len() {
        let start_find = p_str[current_pos..].find("<|im_start|>");
        let end_find = p_str[current_pos..].find("<|im_end|>");
        let eos_find = p_str[current_pos..].find("<|endoftext|>");

        let mut next_special = None;
        let mut min_idx = usize::MAX;

        if let Some(idx) = start_find { if idx < min_idx { min_idx = idx; next_special = Some(("<|im_start|>", im_start_id)); } }
        if let Some(idx) = end_find { if idx < min_idx { min_idx = idx; next_special = Some(("<|im_end|>", im_end_id)); } }
        if let Some(idx) = eos_find { if idx < min_idx { min_idx = idx; next_special = Some(("<|endoftext|>", endoftext_id)); } }

        if let Some((token_str, token_id)) = next_special {
            let text_before = &p_str[current_pos..current_pos + min_idx];
            if !text_before.is_empty() {
                tokens.extend(tokenizer.encode(text_before, false).map_err(|e| anyhow::anyhow!(e))?.get_ids());
            }
            tokens.push(token_id);
            current_pos += min_idx + token_str.len();
        } else {
            let text_before = &p_str[current_pos..];
            if !text_before.is_empty() {
                tokens.extend(tokenizer.encode(text_before, false).map_err(|e| anyhow::anyhow!(e))?.get_ids());
            }
            break;
        }
    }

    // Base budget
    let prompt_len = prompt.len();
    let mut total_tokens = if task_plan.is_multiple_choice {
        32
    } else if task_plan.is_syntax_only {
        (prompt_len * 2).clamp(256, 1024)
    } else {
        (prompt_len * 4).clamp(1024, 4096)
    };

    if task_plan.needs_design {
        total_tokens *= 2;
    }

    if requested_config.strict_determinism {
        total_tokens = total_tokens.min(3072);
        if !requested_config.quiet {
            warn!("[STRICT] Token budget capped at 3072 to prevent CUDA OOM (Phase 51 Unshackled).");
        }
    }

    task_plan.estimated_tokens = total_tokens;

    if tokens.len() > 6144 {
        if !requested_config.quiet {
            warn!("[OOM GUARD] Truncating prompt from {} to 6144 tokens to save VRAM", tokens.len());
        }
        let mut truncated = tokens[0..1024].to_vec();
        truncated.extend_from_slice(&tokens[tokens.len() - 5120..]);
        tokens = truncated;
    }

    Ok(tokens)
}

fn setup_processors(
    engine: &LlmEngine,
    task_plan: &TaskPlan,
    config: &NeuroConfig
) -> Result<PhysicsGuidedLogitsProcessor> {
    let (phase_alignment, is_pink_zone) = {
        let network = engine.semantic_memory.write().unwrap();
        let phase = network.compute_total_energy();
        (phase as f32, phase < 0.5)
    };

    if let Ok(mut srf) = engine.srf.try_lock() {
        srf.bx = phase_alignment;
        srf.dx = phase_alignment;
    }

    let syntax_expert: Box<dyn LogitPerspective> = Box::new(SyntaxPerspective { is_code_block: task_plan.is_syntax_only });
    let causal_expert: Box<dyn LogitPerspective> = Box::new(CausalPerspective { energy: phase_alignment, is_pink_zone });
    let geometer_expert: Box<dyn LogitPerspective> = Box::new(GeometerPerspective { regret_magnitude: 0.0 });

    let mut proc = PhysicsGuidedLogitsProcessor::new(
        299792458,
        Some(config.temperature),
        Some(config.top_p),
        if config.top_k > 0 { Some(config.top_k) } else { None },
        if config.min_p > 0.0 { Some(config.min_p) } else { None },
        vec![syntax_expert, causal_expert, geometer_expert],
    );
    proc.blind_experts = false; // OPERATIONAL PHASE 6: Experts re-enabled with VRAM sanitation
    Ok(proc)
}

fn inject_ghost_token(
    engine: &LlmEngine,
    prompt: &str,
    config: &NeuroConfig
) -> Result<Option<Tensor>> {
    if std::env::var("SODALS_PURE_INFERENCE").is_ok() {
        return Ok(None);
    }
    if config.strict_determinism {
        return Ok(None);
    }

    // Peripheral vision anomaly detection
    let peripheral_ghost: Option<Tensor> = {
        let active_nodes: Vec<usize> = {
            if let Ok(network) = engine.semantic_memory.read() {
                network.identify_keystones(5)
            } else {
                Vec::new()
            }
        };

        let anomaly_wave = {
            if let Ok(network) = engine.semantic_memory.read() {
                network.detect_peripheral_anomalies(&active_nodes)
            } else {
                None
            }
        };

        if let Some(wave) = anomaly_wave {
            if !config.quiet {
                info!("[PERIPHERAL VISION] Anomaly wave detected, compressing to ghost tensor...");
            }

            let hidden_size = {
                let model_guard = engine.model.read().unwrap();
                model_guard.model().embed_tokens.embeddings().dim(1).unwrap_or(4096)
            };
            let varmap = candle_nn::VarMap::new();
            let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &engine.device);

            match crate::bridge::SodalsBridge::new(vb, wave.amplitudes.len(), hidden_size) {
                Ok(bridge) => {
                    match bridge.compress_anomaly_to_ghost(&wave) {
                        Ok(ghost_tensor) => {
                            if !config.quiet {
                                info!("[GHOST INJECTION] Peripheral anomaly ghost ready");
                            }
                            Some(ghost_tensor)
                        }
                        Err(e) => {
                            if !config.quiet {
                                warn!("[GHOST INJECTION] Failed to compress anomaly: {}", e);
                            }
                            None
                        }
                    }
                }
                Err(e) => {
                    if !config.quiet {
                        warn!("[GHOST INJECTION] Failed to create bridge: {}", e);
                    }
                    None
                }
            }
        } else {
            None
        }
    };

    // BCI TOPOLOGICAL PREFIX VECTOR SYNTHESIS
    let active_concepts = {
        if let Ok(network) = engine.semantic_memory.read() {
            network.diffuse_energy(prompt)
        } else {
            std::collections::HashMap::new()
        }
    };

    let mut accumulated_ghost: Option<Tensor> = None;
    for (concept, energy) in &active_concepts {
        if *energy < 0.1 { continue; }
        if let Ok(encoding) = engine.tokenizer.encode(concept.clone(), false) {
            if let Some(&token_id) = encoding.get_ids().first() {
                let t_id = Tensor::new(&[token_id], &engine.device)
                    .and_then(|t| t.unsqueeze(0));
                if let Ok(t_id) = t_id {
                    if let Ok(model_r) = engine.model.read() {
                        let embed_result: candle_core::Result<Tensor> =
                            candle_core::Module::forward(&model_r.model().embed_tokens, &t_id);
                        if let Ok(embed) = embed_result {
                            let safe_energy = (*energy as f64) * 0.15;
                            let scaled_result: candle_core::Result<Tensor> = embed
                                .to_dtype(candle_core::DType::F32)
                                .and_then(|e| &e * safe_energy);
                            if let Ok(scaled) = scaled_result {
                                accumulated_ghost = match accumulated_ghost {
                                    Some(acc) => acc.add(&scaled).ok().or(Some(scaled)),
                                    None => Some(scaled),
                                };
                            }
                        }
                    }
                }
            }
        }
    }

    // GPU L2 NORMALIZATION
    let normalized_ghost = if let Some(ghost) = accumulated_ghost {
        let l2_norm = ghost.sqr()?.sum_all()?.sqrt()?;
        let l2_scalar = l2_norm.to_vec0::<f32>()?;
        if l2_scalar > 1e-8 {
            Some((ghost / l2_scalar as f64)?)
        } else {
            Some(ghost)
        }
    } else {
        None
    };

    // MERGE peripheral anomaly ghost with accumulated ghost
    let merged = match (normalized_ghost, peripheral_ghost) {
        (Some(acc), Some(periph)) => {
            let weighted_periph = (&periph * 0.3f64)?;
            let merged = (&acc * 0.7f64)?.add(&weighted_periph)?;
            if !config.quiet {
                info!("[GHOST MERGE] Accumulated + Peripheral ghost -> unified injection");
            }
            Some(merged)
        }
        (Some(acc), None) => Some(acc),
        (None, Some(periph)) => {
            if !config.quiet {
                info!("[GHOST MERGE] Only peripheral ghost available");
            }
            Some(periph)
        }
        (None, None) => None,
    };

    Ok(merged)
}

fn prepare_qmem_cartridges(
    engine: &LlmEngine,
    input_ids: &Tensor
) -> Result<(SemanticRegisterFileWithQmem, Vec<SemanticNode>)> {
    let mut weaver = CodeWeaver::new().map_err(|_e| anyhow::anyhow!("Weaver Init Failed"))?;
    let current_tokens = input_ids.flatten_all().unwrap_or_else(|_| input_ids.clone()).to_vec1::<u32>().unwrap_or_default();
    let source_code = engine.tokenizer.decode(&current_tokens, true).unwrap_or_default();
    let mut cached_nodes = weaver.extract_topology(&source_code).unwrap_or_default();

    if cached_nodes.is_empty() {
        if let Ok(mut network) = engine.semantic_memory.write() {
            weaver.populate_from_natural_language(&mut network, &source_code);
            cached_nodes = weaver.nodes.clone();
        }
    }

    let mut srf_with_qmem = SemanticRegisterFileWithQmem {
        srf: SemanticRegisterFile::new(0.0),
        poles_cartridge: None,
        impedance_cartridge: None,
        flags_cartridge: None,
        node_count: 0,
    };

    let _ = SemanticRegisterFile::compile_zero_copy_cartridge(
        &cached_nodes,
        &mut srf_with_qmem.poles_cartridge,
        &mut srf_with_qmem.impedance_cartridge,
        &mut srf_with_qmem.flags_cartridge,
    );
    srf_with_qmem.node_count = cached_nodes.len();

    Ok((srf_with_qmem, cached_nodes))
}

fn run_prefill_phase(
    engine: &LlmEngine,
    input_ids: &Tensor,
    ghost_prefix: Option<&Tensor>,
    srf_with_qmem: &SemanticRegisterFileWithQmem,
) -> Result<Tensor> {
    let pure_inf = std::env::var("SODALS_PURE_INFERENCE").is_ok();

    #[cfg(feature = "cuda")]
    let (poles_ptr, impedance_ptr, flags_ptr) = if pure_inf {
        (None, None, None)
    } else {
        (
            srf_with_qmem.poles_cartridge.as_ref().map(|c| c.device_ptr()),
            srf_with_qmem.impedance_cartridge.as_ref().map(|c| c.device_ptr()),
            srf_with_qmem.flags_cartridge.as_ref().map(|c| c.device_ptr()),
        )
    };

    #[cfg(not(feature = "cuda"))]
    let poles_ptr: Option<u64> = None;
    #[cfg(not(feature = "cuda"))]
    let impedance_ptr: Option<u64> = None;
    #[cfg(not(feature = "cuda"))]
    let flags_ptr: Option<u64> = None;

    let n_nodes = Some(srf_with_qmem.node_count as i32);

    let start_pos = 0;
    let logits = {
        let model_guard = engine.model.read().unwrap();
        eprintln!("[GEN] run_prefill_phase: model.forward starting. seq_len={}, device={:?}", input_ids.dims().get(1).copied().unwrap_or(0), engine.device);
        let out = model_guard.forward(
            input_ids,
            start_pos,
            poles_ptr,
            impedance_ptr,
            flags_ptr,
            n_nodes,
            None,
            ghost_prefix,
            None, // token_concept_map_ptr: prefill'de semantik harita gerekmez
            None,
            None,
        ).map_err(|e| anyhow::anyhow!("Prefill Forward Error: {}", e))?;
        eprintln!("[GEN] run_prefill_phase: model.forward returned.");
        out
    };

    Ok(logits)
}

fn extract_last_logits(logits: &Tensor) -> Result<Tensor> {
    let seq_len = logits.dims()[1];
    logits.i((0, seq_len - 1, ..))
        .map_err(|e| anyhow::anyhow!("Logits extraction failed: {}", e))
}

fn spawn_oracle_sidecar(
    engine: &LlmEngine,
    next_token_logits: &Tensor
) -> Option<OracleHandle> {
    let vocab_size = next_token_logits.dims().last().copied().unwrap_or(0);
    if vocab_size > 0 {
        // FIX: Pass engine.oracle_adapter instead of None for DMA prefetch
        #[cfg(feature = "cuda")]
        let oracle_adapter = engine.oracle_adapter.as_ref().map(|a| Arc::clone(a));
        #[cfg(not(feature = "cuda"))]
        let oracle_adapter: Option<Arc<crate::gpu_ops::NeuroAdapter>> = None;
        
        let handle = crate::engine::oracle_worker::spawn_oracle(
            vocab_size,
            Arc::clone(&engine.memory_pressure),
            Arc::clone(&engine.srf),
            oracle_adapter,
        );
        if handle.is_some() {
            eprintln!("[ORACLE] Oracle task live. DMA prefetch pipeline armed.");
        } else {
            eprintln!("[ORACLE] Oracle offline. Proceeding without prefetch.");
        }
        handle
    } else {
        eprintln!("[ORACLE] Unknown vocab size, Oracle disabled.");
        None
    }
}

fn consume_oracle_mask(
    engine: &LlmEngine,
    oracle: &Option<OracleHandle>,
    physics_processor: &mut PhysicsGuidedLogitsProcessor,
    all_tokens: &[u32],
    next_token_logits: &Tensor,
    config: &NeuroConfig,
    gpr_semantic_divergence: &mut f32
) -> Result<(f32, Option<crate::engine::oracle_worker::PrecomputedMask>)> {
    if let Some(mask) = oracle_worker::oracle_consume_mask(oracle, all_tokens.len() + 1) {
        let saved_pw = mask.predicted_semantic_divergence;
        if !config.strict_determinism && mask.confidence > 0.5 {
            let current_vocab = next_token_logits.dims().last().copied().unwrap_or(0);
            if current_vocab > 0 {
                if let Ok(bias) = oracle_worker::f16_mask_to_tensor(&mask.impedance_f16, current_vocab, &engine.device) {
                    physics_processor.update_oracle_bias(bias);
                }
            }
        }
        Ok((saved_pw, Some(mask)))
    } else {
        Ok((*gpr_semantic_divergence, None))
    }
}

fn sample_token(
    engine: &LlmEngine,
    next_token_logits: &Tensor,
    physics_processor: &mut PhysicsGuidedLogitsProcessor,
    config: &NeuroConfig,
    all_tokens: &[u32],
    gpr_semantic_divergence: f32,
    _gpr_predicted_w: f32,
    _oracle: &Option<OracleHandle>,
    ebnf_mask: Option<&mut EbnfLogitMask>,
    mmlu_mask: Option<&mut MmluLogitMask>,
    safety_proc: Option<&crate::engine::IndustrialSafetyLogitsProcessor>,
    prompt_text: &str,
) -> Result<u32> {
    // MMLU / multiple-choice mask (applied before EBNF — mutually exclusive in practice)
    let mc_masked_logits = if let Some(mask) = mmlu_mask {
        match mask.apply_mask(next_token_logits, all_tokens, &engine.tokenizer, prompt_text) {
            Ok(masked) => masked,
            Err(e) => {
                eprintln!("[MMLU MASK] apply failed ({}), using unmasked", e);
                next_token_logits.clone()
            }
        }
    } else {
        next_token_logits.clone()
    };

    // PHASE 49: Apply EBNF constraint mask BEFORE other processing
    let ebnf_masked_logits = if let Some(mask) = ebnf_mask {
        match mask.apply_mask(&mc_masked_logits, all_tokens, &engine.tokenizer, prompt_text) {
            Ok(masked) => {
                eprintln!("[PHASE 49] EBNF mask applied");
                masked
            }
            Err(e) => {
                eprintln!("[PHASE 49] EBNF mask failed ({}), using unmasked", e);
                mc_masked_logits.clone()
            }
        }
    } else {
        mc_masked_logits.clone()
    };

    // Industrial Safety Logits Processing (Cyber-Physical Guidance)
    let safety_logits = if let Some(proc) = safety_proc {
        proc.process(&ebnf_masked_logits, &engine.device)?
    } else {
        ebnf_masked_logits.clone()
    };

    // Logit injection
    let steered_logits = engine.apply_logit_injection(&safety_logits)?;

    // Repeat penalty - applied AFTER all boosts/steer_logits so repeat penalty can successfully suppress any repeating boosted keywords!
    let start_at = all_tokens.len().saturating_sub(128);
    let penalised_logits = if config.repeat_penalty != 1.0 {
        candle_transformers::utils::apply_repeat_penalty(
            &steered_logits,
            config.repeat_penalty,
            &all_tokens[start_at..]
        ).map_err(|e| anyhow::anyhow!(e))?
    } else {
        steered_logits.clone()
    };

    // NaN/Inf GUARD
    let mut safe_logits = penalised_logits.to_dtype(candle_core::DType::F32)?
        .clamp(-1e9_f32, 1e9_f32)?;

    // VOCAB GUARD: LM head may output more logits than the tokenizer has tokens.
    // Truncate to tokenizer vocab size so argmax/sampling never picks an invalid token ID.
    let tokenizer_vocab_size = engine.tokenizer.get_vocab_size(true);
    let logit_vocab = safe_logits.dim(0).unwrap_or(0);
    if logit_vocab > tokenizer_vocab_size {
        safe_logits = safe_logits.narrow(0, 0, tokenizer_vocab_size)
            .map_err(|e| anyhow::anyhow!("vocab clamp failed: {}", e))?;
    }

    let pure_inference = std::env::var("SODALS_PURE_INFERENCE").is_ok();

    // Skew-G Braking
    if !pure_inference && gpr_semantic_divergence < 0.1 {
        engine.apply_skew_g_braking(&mut safe_logits, gpr_semantic_divergence)?;
    }

    // PHASE 51.10: Hard-ban thinking/reasoning tokens to prevent model
    // from entering internal reasoning mode instead of generating user response.
    const QWEN_THINK_START: u32 = 151667; // <thinking>
    const QWEN_THINK_END: u32 = 151668;   // </thinking>
    let safe_logits = if pure_inference {
        safe_logits
    } else {
        let mut logit_vec: Vec<f32> = safe_logits.to_vec1()?;
        if logit_vec.len() > QWEN_THINK_END as usize {
            logit_vec[QWEN_THINK_START as usize] = f32::NEG_INFINITY;
            logit_vec[QWEN_THINK_END as usize] = f32::NEG_INFINITY;
        }
        Tensor::from_vec(logit_vec, safe_logits.dims(), &engine.device)?
    };

    // Sampling
    let next_token = if config.euler_resonance {
        engine.active_inference_clifford(&safe_logits, None)?
    } else {
        physics_processor.sample(&safe_logits, config).map_err(|e| anyhow::anyhow!(e))?
    };

    Ok(next_token)
}

fn is_eos_token(token: u32) -> bool {
    token == 151645 || token == 151643 || token == 248046 || token == 248044
}

fn should_circuit_break(
    generated_tokens_count: usize,
    max_gen_tokens: usize,
    gpr_semantic_divergence: f32,
    requested_config: &NeuroConfig
) -> bool {
    // Hardware Redline — ALWAYS enforced regardless of strict_determinism
    if generated_tokens_count >= max_gen_tokens {
        eprintln!("\n[CIRCUIT BREAKER] Hardware limit ({} tokens) reached. Truncating gracefully.", max_gen_tokens);
        return true;
    }

    if !requested_config.strict_determinism {
        return false;
    }

    // Cognitive Flatline (only in strict mode)
    if generated_tokens_count > 300 && gpr_semantic_divergence < 0.1 {
        eprintln!("\n[CIRCUIT BREAKER] Cognitive flatline detected (SD {} < 0.1). Early exit.", gpr_semantic_divergence);
        return true;
    }

    false
}

fn is_mathematical_hallucination(content: &str) -> bool {
    // Detect classic Qwen3 math drift patterns: LaTeX, fractions, roots, integrals.
    // Also catch 'Plain Algebra' drift which uses standard characters instead of LaTeX.
    let math_markers = [
        "\\sqrt", "\\frac", "\\sum", "\\int", "\\partial", "\\infty", 
        "\\alpha", "\\beta", "\\gamma", "\\delta", "\\theta"
    ];
    
    let plain_algebra_markers = [
        "x + y =", "x - y =", "2x =", "2y =", "equation 1", "equation 2",
        "solve for x", "solve for y", "let x be", "let y be"
    ];
    
    let mut marker_hits = 0;
    for &marker in &math_markers {
        if content.contains(marker) {
            marker_hits += 1;
        }
    }
    
    let mut algebra_hits = 0;
    for &marker in &plain_algebra_markers {
        if content.to_lowercase().contains(marker) {
            algebra_hits += 1;
        }
    }

    let dollar_count = content.matches('$').count();
    marker_hits >= 3 || algebra_hits >= 2 || dollar_count >= 4
}

fn parse_mcq_prompt(prompt: &str) -> Option<(String, Vec<(String, String)>)> {
    let lower_prompt = prompt.to_lowercase();
    
    let question_marker = if lower_prompt.contains("### question:") {
        "### question:"
    } else if lower_prompt.contains("question:") {
        "question:"
    } else {
        return None;
    };
    
    let options_marker = if lower_prompt.contains("### options:") {
        "### options:"
    } else if lower_prompt.contains("options:") {
        "options:"
    } else {
        return None;
    };
    
    let q_start = lower_prompt.find(question_marker)? + question_marker.len();
    let o_start = lower_prompt.find(options_marker)?;
    
    if q_start >= o_start {
        return None;
    }
    
    // Slice lower_prompt (whose byte offsets match q_start/o_start) to avoid UTF-8 panic on case-expanding chars
    let question_text = lower_prompt[q_start..o_start].trim().to_string();
    let options_block = &lower_prompt[o_start + options_marker.len()..];
    
    let mut options = Vec::new();
    for line in options_block.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        
        if let Some(first_char) = trimmed.chars().next() {
            let up = first_char.to_ascii_uppercase();
            if matches!(up, 'A' | 'B' | 'C' | 'D' | 'E') && trimmed.len() >= 2 {
                let rest = &trimmed[first_char.len_utf8()..];
                if rest.starts_with('.') || rest.starts_with(')') {
                    let letter = up.to_string();
                    let content = rest[1..].trim().to_string();
                    options.push((letter, content));
                }
            }
        }
    }
    
    if options.is_empty() {
        None
    } else {
        Some((question_text, options))
    }
}

fn verify_thinking_plan(thinking_text: &str, network: &ResonanceNetwork) -> Vec<String> {
    let mut warnings = Vec::new();
    let lower_plan = thinking_text.to_lowercase();
    let labels: Vec<String> = network.labels.keys().cloned().collect();
    for i in 0..labels.len() {
        for j in (i + 1)..labels.len() {
            let label1 = &labels[i];
            let label2 = &labels[j];
            let l1_clean = label1.to_lowercase();
            let l2_clean = label2.to_lowercase();
            if lower_plan.contains(&l1_clean) && lower_plan.contains(&l2_clean) {
                if let (Some(&id1), Some(&id2)) = (network.labels.get(label1), network.labels.get(label2)) {
                    let w1 = network.adjacency_csr.get(id1, id2).copied().unwrap_or(0.0);
                    let w2 = network.adjacency_csr.get(id2, id1).copied().unwrap_or(0.0);
                    if w1.abs() < 1e-6 && w2.abs() < 1e-6 {
                        warnings.push(format!(
                            "[Phase Error] Kopuk Bağ: \"{}\" ve \"{}\" arasında semantik ağda bağlantı bulunamadı.",
                            label1, label2
                        ));
                    }
                }
            }
        }
    }

    // 2. Keystone and Ripple Effect Simulation (Holographic Projection inspired check)
    let keystones = network.find_keystones();
    let mut id_to_label = std::collections::HashMap::new();
    for (label, &id) in &network.labels {
        id_to_label.insert(id, label.clone());
    }

    for &keystone_id in &keystones {
        if let Some(keystone_label) = id_to_label.get(&keystone_id) {
            let k_clean = keystone_label.to_lowercase();
            if lower_plan.contains(&k_clean) {
                // If a keystone is mentioned, propagate its change to see the ripple effect
                let ripple_results = network.propagate_change(keystone_id);
                for &(affected_id, deviation) in &ripple_results {
                    if let Some(affected_label) = id_to_label.get(&affected_id) {
                        let aff_clean = affected_label.to_lowercase();
                        // If the affected node is also mentioned in the plan, warn the developer about the phase shift
                        if lower_plan.contains(&aff_clean) {
                            let phase_shift_deg = (deviation * 15.0).min(90.0);
                            warnings.push(format!(
                                "[Holographic Projection Warning] Kilit Taşı \"{}\" (Keystone) değişiyor. Bu değişim \"{}\" üzerinde {:.1} derece Phase Shift (Faz Kayması) ve gerilim oluşturuyor. Jacobian matrisindeki enerji yoğunluğu (Rhythm) bozuldu, Wronskian determinantı sıfıra yaklaşıyor (Linear Dependence), hemen düzelt!",
                                keystone_label, affected_label, phase_shift_deg
                            ));
                        }
                    }
                }
            }
        }
    }

    warnings
}

fn strip_thinking_block(text: &str) -> String {
    let re = regex::Regex::new(r"(?s)<thinking>.*?</thinking>").unwrap();
    re.replace_all(text, "").to_string()
}

