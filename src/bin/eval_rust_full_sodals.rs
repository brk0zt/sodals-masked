// src/bin/eval_rust_full_sodals.rs
// ============================================================================
// SODALS v3.0 — FULL RUST NEUROSYMBOLIC BENCHMARK EVALUATOR
// ============================================================================
// Evaluates K7 / Base weights using the genuine 20/20 SODALS Rust engine:
//   - CliffordBridge Logit Steering (Cl(3,0) geometric phase injection)
//   - PonderingCore Latent ODE Integration
//   - Euler Resonance & PhysicsNeuron Symplectic Updates
//   - Skew-G Braking & MindFilter Coherence Gating
//   - SRF (Semantic Register File) Phase Tracking
// ============================================================================

use anyhow::Result;
use std::time::Instant;
use std::fs::File;
use std::io::Write;
use serde::{Serialize, Deserialize};

use sodals_neuro::engine::generation_loop::generate_stream;
use sodals_neuro::engine::llm_engine::LlmEngine;
use sodals_neuro::llm_core::NeuroConfig;
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogLevel, LogManager};

#[derive(Serialize, Deserialize, Debug, Clone)]
struct BenchmarkItem {
    id: String,
    topic: String,
    prompt: String,
    expected_ground_truth: String,
    category: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct EvalResult {
    id: String,
    topic: String,
    prompt: String,
    response: String,
    ground_truth: String,
    category: String,
    tokens_generated: usize,
    elapsed_seconds: f64,
    tokens_per_second: f64,
    is_exact_match: bool,
}

fn main() -> Result<()> {
    // 1. ENSURE FULL SODALS MODE: Explicitly enable PonderingCore and J-Lens
    std::env::set_var("SODALS_PONDER", "1");
    std::env::set_var("SODALS_JLENS_CAPTURE", "1");
    std::env::set_var("SODALS_VERBOSE", "1");
    std::env::remove_var("SODALS_PURE_INFERENCE");
    std::env::remove_var("SODALS_NO_PONDER");
    tracing_subscriber::fmt::init();

    println!("================================================================================");
    println!("  SODALS v3.0 — FULL RUST NEUROSYMBOLIC BENCHMARK EVALUATION");
    println!("  Engine: Native Rust / Candle CUDA | Mode: Full 20/20 Neurosymbolic Active");
    println!("================================================================================");

    let sys_config = SystemConfig::load().unwrap_or_else(|_| SystemConfig::default());
    let log_manager = LogManager::new(LogLevel::INFO);
    println!("[EVAL] Initializing Full SODALS LlmEngine...");
    let mut engine = LlmEngine::new(&sys_config, &log_manager)?;

    // 2. CONFIGURE FULL NEUROSYMBOLIC PARAMETERS
    let mut config = NeuroConfig::engineer_mode();
    config.euler_resonance = true;
    config.physics_enabled = true;
    config.minimal_mode = false;     // Full logit steering and Arkhe bridge ACTIVE
    config.temperature = 0.7;
    config.top_p = 0.9;
    config.min_p = 0.02;
    config.repeat_penalty = 1.15;
    config.top_k = 40;
    config.quiet = false;

    // 3. DEFINE BENCHMARK SUITE: 4 OOD PERTURBATIONS + TARGET HLE QUESTIONS
    let test_suite: Vec<BenchmarkItem> = vec![
        // --- 4 OOD PERTURBATION TESTS (True Generalization & Anti-Overfitting Check) ---
        BenchmarkItem {
            id: "ood_top_euler_cp4".to_string(),
            topic: "Euler Characteristic of CP^4".to_string(),
            prompt: "<|im_start|>user\nCompute the topological Euler characteristic \\chi(\\mathbb{C}P^4) of the 4-dimensional complex projective space. Provide a step-by-step first-principles derivation.<|im_end|>\n<|im_start|>assistant\n".to_string(),
            expected_ground_truth: "5".to_string(),
            category: "OOD Math".to_string(),
        },
        BenchmarkItem {
            id: "ood_lie_dim_e6".to_string(),
            topic: "Dimension of Simple Lie Algebra E6".to_string(),
            prompt: "<|im_start|>user\nWhat is the complex dimension of the exceptional simple Lie algebra \\mathfrak{e}_6? Derive it from the Cartan subalgebra rank and root system cardinality.<|im_end|>\n<|im_start|>assistant\n".to_string(),
            expected_ground_truth: "78".to_string(),
            category: "OOD Lie Theory".to_string(),
        },
        BenchmarkItem {
            id: "ood_geom_canonical_degree_g4".to_string(),
            topic: "Degree of Canonical Divisor for Genus 4".to_string(),
            prompt: "<|im_start|>user\nLet C be a smooth compact Riemann surface of genus g = 4. Using the Riemann-Roch Theorem and Serre duality, compute the exact degree of the canonical divisor K on C.<|im_end|>\n<|im_start|>assistant\n".to_string(),
            expected_ground_truth: "6".to_string(),
            category: "OOD Algebraic Geometry".to_string(),
        },
        BenchmarkItem {
            id: "ood_num_point_count_f53".to_string(),
            topic: "Elliptic Curve Point Count over F_53".to_string(),
            prompt: "<|im_start|>user\nLet E be an elliptic curve defined over the finite field \\mathbb{F}_{53} having trace of Frobenius a_{53} = -4. From the Hasse-Weil Theorem, derive the exact number of \\mathbb{F}_{53}-rational points #E(\\mathbb{F}_{53}).<|im_end|>\n<|im_start|>assistant\n".to_string(),
            expected_ground_truth: "58".to_string(),
            category: "OOD Number Theory".to_string(),
        },
        // --- HLE TARGET QUESTIONS ---
        BenchmarkItem {
            id: "hle_66c5c3eae30bdd4f1da62cfc".to_string(),
            topic: "First Homology of Moduli Space of Unimodular Lattices in R^2".to_string(),
            prompt: "<|im_start|>user\nCompute the first integral homology group H_1(X, \\mathbb{Z}) of the moduli space X of unimodular lattices in \\mathbb{R}^2.<|im_end|>\n<|im_start|>assistant\n".to_string(),
            expected_ground_truth: "\\mathbb{Z}".to_string(),
            category: "HLE Math".to_string(),
        },
        BenchmarkItem {
            id: "hle_669402b41dcb3d5a1ef9e951".to_string(),
            topic: "Reduced Spin Bordism of BG2 in Degree 12".to_string(),
            prompt: "<|im_start|>user\nCompute the reduced Spin bordism group \\widetilde{\\Omega}_{12}^{Spin}(BG_2) for the exceptional Lie group G_2.<|im_end|>\n<|im_start|>assistant\n".to_string(),
            expected_ground_truth: "Z+Z+Z+Z+Z".to_string(),
            category: "HLE Math".to_string(),
        }
    ];

    let mut results: Vec<EvalResult> = Vec::new();
    let total_count = test_suite.len();

    for (idx, item) in test_suite.iter().enumerate() {
        println!("\n{}", "=".repeat(80));
        println!("[{}/{}] EVALUATING: {} [{}]", idx + 1, total_count, item.topic, item.category);
        println!("Ground Truth: {}", item.expected_ground_truth);
        println!("{}", "=".repeat(80));

        engine.reset_internal_state();
        let start_t = Instant::now();

        let token_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let tc_clone = token_count.clone();

        let response = generate_stream(
            &mut engine,
            &item.prompt,
            None,
            None,
            move |t| {
                tc_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                print!("{}", t);
            },
            &config,
            0,
            1024,
        )?;

        let tok_cnt = token_count.load(std::sync::atomic::Ordering::SeqCst);
        let elapsed = start_t.elapsed().as_secs_f64();
        let tps = if elapsed > 0.0 { tok_cnt as f64 / elapsed } else { 0.0 };

        // Parse extracted answer
        let mut clean_ans = response.trim().to_string();
        if let Some(pos) = response.rfind("Answer:") {
            clean_ans = response[pos + 7..].trim().lines().next().unwrap_or("").trim().to_string();
        } else if let Some(pos) = response.rfind("Answer :") {
            clean_ans = response[pos + 8..].trim().lines().next().unwrap_or("").trim().to_string();
        }

        clean_ans = clean_ans.trim_matches(|c: char| c == '.' || c == '$' || c == ' ' || c == '\n').to_string();
        let clean_gt = item.expected_ground_truth.trim_matches(|c: char| c == '.' || c == '$' || c == ' ').to_string();

        let is_exact = (clean_ans.to_lowercase() == clean_gt.to_lowercase())
            || (clean_ans.contains(&clean_gt) && clean_gt.len() >= 2);

        println!("\n\n--- QUESTION RESULT ---");
        println!("Model Output Answer: {:?}", clean_ans);
        println!("Ground Truth:        {:?}", clean_gt);
        println!("Exact Match:         {}", if is_exact { "✅ PASS" } else { "❌ FAIL" });
        println!("Tokens: {} | Time: {:.2}s ({:.2} tok/s)", tok_cnt, elapsed, tps);

        results.push(EvalResult {
            id: item.id.clone(),
            topic: item.topic.clone(),
            prompt: item.prompt.clone(),
            response: response.clone(),
            ground_truth: item.expected_ground_truth.clone(),
            category: item.category.clone(),
            tokens_generated: tok_cnt,
            elapsed_seconds: elapsed,
            tokens_per_second: tps,
            is_exact_match: is_exact,
        });

        // Save incremental JSON
        let out_json_path = "c:/Users/Burak/Desktop/sodals_core/hle_results_rust_full_sodals.json";
        let mut f = File::create(out_json_path)?;
        f.write_all(serde_json::to_string_pretty(&results)?.as_bytes())?;
    }

    // 4. SUMMARY REPORT
    let passed = results.iter().filter(|r| r.is_exact_match).count();
    println!("\n================================================================================");
    println!("  FULL RUST SODALS EVALUATION COMPLETE");
    println!("  Total Score: {}/{} ({:.1}%)", passed, total_count, (passed as f64 / total_count as f64) * 100.0);
    println!("================================================================================");

    Ok(())
}
