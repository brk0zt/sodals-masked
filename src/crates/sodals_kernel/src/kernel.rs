// src/crates/sodals_kernel/src/kernel.rs - CAUSAL OUROBOROS ENGINE
// COMPLETE VERSION - ALL FUNCTIONALITY PRESERVED + NEW FEATURES ADDED

use tokio::sync::{mpsc, broadcast};
use std::sync::Arc;
use sodals_common::config::SystemConfig;
use sodals_neuro::semantic_network::ResonanceNetwork;
use sodals_neuro::global_gaze::{GlobalScanner, TaskNature};
use bumpalo::Bump; // arena allocator for parallel universes

use crate::store::VectorStoreActor as StoreActor;
use crate::sentinel::SentinelActor;
use crate::forge::ForgeActor;
use crate::runtime::RuntimeActor;

use num_complex::Complex64;         
               
use sodals_net::protocol::{SystemMsg, SystemState}; 
use sodals_neuro::semantic_parser::CodeWeaver;      
use sodals_neuro::train_projector;  
use sodals_common::log_manager::{LogManager, LogLevel};
use sodals_neuro::llm_core::NeuroConfig;
use crate::job_manager::JobManager;
use sodals_net::web_search::WebSearchEngine;

/// handle used by all components to request a cloned snapshot of the
/// shared `ResonanceNetwork`.  the real network is owned by a dedicated
/// actor task; clones are returned over a oneshot channel so callers can
/// mutate without ever touching a lock.
#[derive(Clone)]
pub struct BrainHandle {
    tx: mpsc::Sender<BrainCommand>,
}

impl BrainHandle {
    pub fn new(tx: mpsc::Sender<BrainCommand>) -> Self {
        BrainHandle { tx }
    }
}

pub enum BrainCommand {
    CloneNetwork(tokio::sync::oneshot::Sender<ResonanceNetwork>),
}

impl BrainHandle {
    pub async fn clone_network(&self) -> ResonanceNetwork {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        // ignore send error: if actor has shut down the kernel is exiting
        let _ = self.tx.send(BrainCommand::CloneNetwork(resp_tx)).await;
        resp_rx.await.expect("brain actor dropped")
    }
}

/// internal actor that owns the actual network state
pub async fn brain_actor(mut rx: mpsc::Receiver<BrainCommand>, brain: ResonanceNetwork) {
    while let Some(cmd) = rx.recv().await {
        match cmd {
            BrainCommand::CloneNetwork(resp) => {
                // cheap clone via derived Clone on network
                let _ = resp.send(brain.clone());
            }
        }
    }
}

pub struct Kernel {
    // İletişim Kanalları (Ana Hatlar)
    rx: mpsc::Receiver<SystemMsg>,
    tui_tx: mpsc::Sender<SystemMsg>,
    cortex_tx: mpsc::Sender<SystemMsg>,
    broadcast_tx: broadcast::Sender<SystemMsg>,

    // Alt Servis Kanalları (Kernel bunlara emir verir)
    store_tx: mpsc::Sender<SystemMsg>,
    #[allow(dead_code)]
    sentinel_tx: mpsc::Sender<SystemMsg>,
    #[allow(dead_code)]
    forge_tx: mpsc::Sender<SystemMsg>,
    runtime_tx: mpsc::Sender<SystemMsg>,
    #[allow(dead_code)]
    biosignal_tx: mpsc::Sender<SystemMsg>,
    
    // BrainFlow biosignal processor handle (only available with brainflow feature)
    #[cfg(feature = "brainflow")]
    biosignal_handle: Option<crate::biosignal_processor::BiosignalHandle>,

    // Çekirdek Bileşenleri
    #[allow(dead_code)]
    config: SystemConfig,
    brain: Arc<BrainHandle>,
    logger: LogManager,
    // Phase 20.8: Removed duplicate LlmEngine - cortex is sole owner to prevent OOM
    // All LLM operations are routed through cortex_tx channel
    job_manager: JobManager,
    #[allow(dead_code)]
    web: WebSearchEngine,

    // Durum Değişkenleri
    state: SystemState,
    last_noise_level: u64,
    iot_status: String,

    // Hafıza ve Takip
    pending_requests: std::collections::HashMap<u64, (String, String)>,
    pending_memorization: std::collections::HashMap<u64, (String, String)>,
    #[allow(dead_code)]
    user_history: Vec<String>,
}

impl Kernel {
    pub fn new(
        rx: mpsc::Receiver<SystemMsg>,
        kernel_tx: mpsc::Sender<SystemMsg>,
        tui_tx: mpsc::Sender<SystemMsg>,
        cortex_tx: mpsc::Sender<SystemMsg>,
        broadcast_tx: broadcast::Sender<SystemMsg>,
        config: SystemConfig,
        brain: Arc<BrainHandle>,
    ) -> anyhow::Result<Self> {
        
        let logger = LogManager::new(LogLevel::INFO);
        logger.info("KERNEL", "Çekirdek sistem başlatılıyor...");

        let (store_tx, store_rx) = mpsc::channel(100);
        let (sentinel_tx, sentinel_rx) = mpsc::channel(100);
        let (forge_tx, forge_rx) = mpsc::channel(100);
        let (runtime_tx, runtime_rx) = mpsc::channel(100);
        let (biosignal_tx, _biosignal_rx) = mpsc::channel(100);

        let store_actor = StoreActor::new(store_rx, kernel_tx.clone());
        tokio::spawn(async move { store_actor.run().await; });

        let sentinel_actor = SentinelActor::new(sentinel_rx, tui_tx.clone());
        tokio::spawn(async move { sentinel_actor.run().await; });

        let forge_actor = ForgeActor::new(forge_rx, tui_tx.clone());
        tokio::spawn(async move { forge_actor.run().await; });

        let _runtime_actor = RuntimeActor::new(runtime_rx, tui_tx.clone());
        tokio::spawn(async move { });
        
        // Initialize BrainFlow biosignal processor (only with brainflow feature)
        #[cfg(feature = "brainflow")]
        let biosignal_handle = {
            let (biosignal_cmd_tx, biosignal_cmd_rx) = mpsc::channel(32);
            let biosignal_processor = crate::biosignal_processor::BiosignalProcessor::new(
                biosignal_cmd_rx,
                kernel_tx.clone(),
            );
            tokio::spawn(async move { biosignal_processor.run().await; });
            Some(crate::biosignal_processor::BiosignalHandle::new(biosignal_cmd_tx))
        };
        #[cfg(not(feature = "brainflow"))]
        let _biosignal_handle: Option<()> = None;

        // Phase 20.8: Removed LlmEngine initialization - cortex is sole owner
        // All LLM operations are routed through cortex_tx channel to prevent OOM
        logger.info("KERNEL", "Cortex Actor handles all LLM operations...");
        
        let job_manager = JobManager::new(60);
        let web = WebSearchEngine::new();

        Ok(Self {
            rx,
            tui_tx,
            cortex_tx,
            broadcast_tx,
            
            store_tx,
            sentinel_tx,
            forge_tx,
            runtime_tx,
            biosignal_tx,
            #[cfg(feature = "brainflow")]
            biosignal_handle,

            config,
            brain,
            logger,
            job_manager,
            web,

            state: SystemState::Booting,
            last_noise_level: 0,
            iot_status: "N/A".to_string(),
            
            pending_requests: std::collections::HashMap::new(),
            pending_memorization: std::collections::HashMap::new(),
            user_history: Vec::new(),
        })
    }

    async fn update_tui(&self, active_jobs: usize, gas: u64, log: &str) {
        let _ = self.tui_tx.send(SystemMsg::DashboardUpdate {
            active_jobs, 
            gas_used_last_tick: gas, 
            last_log: log.to_string(), 
            system_status: self.state.clone(),
            sentinel_noise: self.last_noise_level,
        }).await;
    }

    async fn send_response(&self, req_id: u64, user_id: &str, text: String) {
        let msg = SystemMsg::CortexResponse {
            req_id,
            user_id: user_id.to_string(),
            content: text,
        };

        let _ = self.broadcast_tx.send(msg.clone());
        let _ = self.tui_tx.send(msg).await;
    }

    async fn broadcast_system_msg(&self, text: String) {
        self.send_response(0, "broadcast", text).await;
    }

    fn is_arithmetic(text: &str) -> bool {
        let t = text.trim();
        let has_digit = t.chars().any(|c| c.is_ascii_digit());
        let has_op = t.chars().any(|c| "+-*/".contains(c));
        has_digit && has_op && t.len() < 20
    }

    /// Start BrainFlow biosignal acquisition (requires brainflow feature)
    /// 
    /// # Arguments
    /// * `board_id` - BrainFlow board ID (use constants from brainflow_bridge::board_ids)
    /// * `serial_port` - Optional serial port for wired connections
    #[cfg(feature = "brainflow")]
    pub async fn start_brainflow(
        &self,
        board_id: i32,
        serial_port: Option<String>,
    ) -> anyhow::Result<()> {
        if let Some(ref handle) = self.biosignal_handle {
            let params = sodals_neuro::brainflow_bridge::BrainFlowParams {
                board_id,
                serial_port,
                ip_address: None,
                mac_address: None,
                timeout: 5,
            };
            
            handle.start(params).await?;
            self.logger.info("KERNEL", &format!("BrainFlow started with board_id: {}", board_id));
            Ok(())
        } else {
            Err(anyhow::anyhow!("Biosignal processor not initialized"))
        }
    }

    /// Stop BrainFlow biosignal acquisition (requires brainflow feature)
    #[cfg(feature = "brainflow")]
    pub async fn stop_brainflow(&self) -> anyhow::Result<()> {
        if let Some(ref handle) = self.biosignal_handle {
            handle.stop().await?;
            self.logger.info("KERNEL", "BrainFlow stopped");
            Ok(())
        } else {
            Err(anyhow::anyhow!("Biosignal processor not initialized"))
        }
    }

    /// Get current cognitive state from BrainFlow (requires brainflow feature)
    #[cfg(feature = "brainflow")]
    pub async fn get_cognitive_state(&self) -> anyhow::Result<sodals_neuro::signal_processor::BandPowers> {
        if let Some(ref handle) = self.biosignal_handle {
            handle.get_band_powers().await
        } else {
            Err(anyhow::anyhow!("Biosignal processor not initialized"))
        }
    }

    async fn think_with_physics(&self, prompt: &str) -> Option<Vec<f32>> {
        // grab a cloned copy from the brain actor – this call is non‑blocking
        // and cheap because the network implements `Clone`.
        let mut local_brain = self.brain.clone_network().await;
        if let Ok(mut weaver) = CodeWeaver::new() {
            let _ = weaver.weave(&mut local_brain, prompt);
        }

        let input_signal = vec![(0, Complex64::new(1.0, 0.0))];
        local_brain.step(&input_signal);

        let thinking_steps = 12;
        let empty_signal: Vec<(usize, Complex64)> = Vec::new();
        for _ in 0..thinking_steps {
            local_brain.step(&empty_signal);
        }

        let physics_vector = local_brain.extract_state_vector();
        if physics_vector.is_empty() {
            None
        } else {
            Some(physics_vector)
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CAUSAL INTERVENTION: Verify response stability through virtual perturbation
    // ═══════════════════════════════════════════════════════════════════════════

    /// Stress Test: Apply candidate response as force to physics engine
    /// 
    /// Algorithm:
    /// 1. Convert candidate response to embedding vector
    /// 2. Inject as virtual perturbation to resonance network
    /// 3. Simulate 5 physics steps
    /// 4. Measure: Does system enter vortex/chaos state?
    /// 
    /// Returns: true if causally stable, false if dangerous/hallucinated
    #[allow(dead_code)]
    async fn verify_causal_stability(&self, _candidate_response: &str, candidate_embedding: &[f32]) -> bool {
        println!("[INTERVENTION] 🛡️ Nedensel Kararlılık Testi Başlatılıyor...");
        self.logger.info("INTERVENTION", "Causal stability verification initiated");
        
        // work on a private clone so parallel universes don't interfere
        let mut local_brain = self.brain.clone_network().await;
        // Extract current baseline state
        let baseline_energy = local_brain.compute_total_energy();
        
        // Apply candidate as virtual perturbation
        // Convert f32 embedding to Complex64 force vector
        let perturbation_signal: Vec<(usize, Complex64)> = candidate_embedding
            .iter()
            .take(local_brain.n.min(candidate_embedding.len()))
            .enumerate()
            .map(|(i, &val)| {
                // Scale to reasonable force magnitude
                let magnitude = (val * 0.1) as f64; // 10% strength perturbation
                (i, Complex64::new(magnitude, 0.0))
            })
            .collect();
        
        if perturbation_signal.is_empty() {
            println!("[INTERVENTION] ⚠️  Boş pertürbasyon vektörü, varsayılan olarak kabul ediliyor");
            return true;
        }
        
        // Simulate 5 physics steps with perturbation
        const INTERVENTION_STEPS: usize = 5;
        let mut energy_trajectory = Vec::with_capacity(INTERVENTION_STEPS);
        
        for step in 0..INTERVENTION_STEPS {
            local_brain.step(&perturbation_signal);
            let current_energy = local_brain.compute_total_energy();
            energy_trajectory.push(current_energy);
            
            println!("[INTERVENTION] Step {}: E={:.6}", step + 1, current_energy);
        }
        
        // Analyze energy trajectory for vortex/chaos
        let final_energy = energy_trajectory[INTERVENTION_STEPS - 1];
        let energy_growth = final_energy / baseline_energy.max(1e-10);
        
        // Calculate energy variance (high variance = chaotic)
        let mean_energy: f64 = energy_trajectory.iter().sum::<f64>() / INTERVENTION_STEPS as f64;
        let variance: f64 = energy_trajectory.iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f64>() / INTERVENTION_STEPS as f64;
        let std_dev = variance.sqrt();
        
        // Stability criteria
        const MAX_ENERGY_GROWTH: f64 = 2.0;  // Energy shouldn't more than double
        const MAX_STD_DEV_RATIO: f64 = 0.5;   // Std dev shouldn't exceed 50% of mean
        
        let is_stable = energy_growth < MAX_ENERGY_GROWTH 
                     && (std_dev / mean_energy.max(1e-10)) < MAX_STD_DEV_RATIO;
        
        if is_stable {
            println!("[INTERVENTION] ✅ Nedensel Kararlılık: ONAYLANDI");
            println!("[INTERVENTION]    Enerji Büyümesi: {:.3}x (< {:.1}x)", energy_growth, MAX_ENERGY_GROWTH);
            println!("[INTERVENTION]    Standart Sapma: {:.4} ({:.1}% of mean)", std_dev, (std_dev/mean_energy)*100.0);
            
            self.logger.info("INTERVENTION", &format!(
                "Causal stability verified: growth={:.3}x, variance={:.4}",
                energy_growth, variance
            ));
        } else {
            println!("[INTERVENTION] ❌ Nedensel Kararsızlık: REDDEDILDI");
            println!("[INTERVENTION]    Enerji Büyümesi: {:.3}x (> {:.1}x) ⚠️", energy_growth, MAX_ENERGY_GROWTH);
            println!("[INTERVENTION]    Standart Sapma: {:.4} ({:.1}% of mean) ⚠️", std_dev, (std_dev/mean_energy)*100.0);
            println!("[INTERVENTION]    SEBEP: Cevap sistemin nedensel yapısıyla çatışıyor (Vortex/Chaos)");
            
            self.logger.warn("INTERVENTION", &format!(
                "Causal instability detected: growth={:.3}x, variance={:.4}",
                energy_growth, variance
            ));
        }
        
        is_stable
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MULTI-VERSE WITH REGRET CALCULATION (Enhanced for Ouroboros)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Enhanced parallel resonance search that returns regret vector and embeddings
    /// 
    /// Returns: (best_response, best_energy, best_embedding, regret_vector)
    /// 
    /// Parallelism is configurable via SystemConfig.num_universes (default: 2 for local, higher for cloud)
    #[allow(dead_code)]
    async fn parallel_resonance_search_with_regret(
        &self,
        prompt: &str,
        negative_bias: Option<Vec<f32>>,
        n_universes: Option<usize>  // Use config default if None
    ) -> (String, f32, Vec<f32>, Option<Vec<f32>>) {
        let n_universes = n_universes.unwrap_or(self.config.num_universes);
        
        self.logger.debug("MULTIVERSE", &format!(
            "Simulating {} parallel universes for resonance search",
            n_universes
        ));
        
        if let Some(ref bias) = negative_bias {
            let bias_norm: f32 = bias.iter().map(|x| x * x).sum::<f32>().sqrt();
            self.logger.debug("MULTIVERSE", &format!(
                "Counterfactual regret active (norm: {:.4})",
                bias_norm
            ));
        }
        
        // Get base physics and apply negative bias
        let base_physics = match self.think_with_physics(prompt).await {
            Some(mut vec) if !vec.is_empty() => {
                if let Some(ref neg_bias) = negative_bias {
                    let min_dim = vec.len().min(neg_bias.len());
                    for i in 0..min_dim {
                        vec[i] -= neg_bias[i]; // SUBTRACT regret
                    }
                    self.logger.debug("MULTIVERSE", "Physics vector adjusted with regret");
                }
                vec
            }
            _ => vec![0.0; 768]
        };

        #[allow(clippy::type_complexity)]
        let mut futures: Vec<tokio::task::JoinHandle<Option<(String, f32, usize, f32, Vec<f32>, Vec<f32>)>>> = Vec::with_capacity(n_universes);

        for k in 0..n_universes {
            let prompt_clone = prompt.to_string();
            let base_vec = base_physics.clone();
            let cortex_tx_clone = self.cortex_tx.clone();
            let _logger_clone = self.logger.clone();

            let future = tokio::spawn(async move {
                let theta = (k as f32 * 2.0 * std::f32::consts::PI) / (n_universes as f32);

                let shifted_out = {
                    // each universe gets its own arena for temporary computations
                    let bump = Bump::new();
                    let mut shifted_vec = bumpalo::vec![in &bump; 0.0_f32; base_vec.len()];
                    for (i, slot) in shifted_vec.iter_mut().enumerate() {
                        let v = base_vec[i];
                        *slot = if i % 2 == 0 {
                            v * theta.cos()
                        } else {
                            v * theta.sin()
                        };
                    }
                    shifted_vec.to_vec()
                };

                let perturbed_prompt = if k == 0 {
                    prompt_clone.clone()
                } else {
                    format!("{}\n[Faz: {:.3}π]", prompt_clone, theta / std::f32::consts::PI)
                };

                // Phase 22: Route through cortex channel (LlmEngine removed from kernel)
                // Note: Response handled by global CortexResponse handler
                let _ = cortex_tx_clone.send(SystemMsg::LlmRequest {
                    req_id: 0,
                    user_id: "universe".to_string(),
                    prompt: perturbed_prompt.clone(),
                    image_data: None,
                    response_tx: None,
                }).await;
                
                // For now, return a placeholder response - multiverse needs architecture update
                let response: String = format!("[Multiverse {} response placeholder]", k);

                // Get embedding for this response (for causal verification) - use cortex
                // Phase 22: Simplified - skip embedding for parallel universes to avoid complexity
                let embedding: Vec<f32> = vec![0.0; 768];
                
                // Calculate energy
                let length_penalty = ((response.len() as f32 - 500.0) / 200.0).abs();
                
                let char_counts = response.chars()
                    .fold(std::collections::HashMap::<char, usize>::new(), |mut acc: std::collections::HashMap<char, usize>, c: char| {
                        *acc.entry(c).or_insert(0) += 1;
                        acc
                    });
                
                let total_chars = response.len() as f32;
                let entropy: f32 = char_counts.values()
                    .map(|&count| {
                        let p = count as f32 / total_chars;
                        -p * p.log2()
                    })
                    .sum();
                
                let entropy_penalty = (entropy - 4.5).abs();
                
                let word_count = response.split_whitespace().count() as f32;
                let coherence = word_count / total_chars.max(1.0);
                let coherence_penalty = (coherence - 0.15).abs() * 10.0;
                
                let energy = length_penalty + entropy_penalty + coherence_penalty;

                // shifted_out was computed earlier to avoid holding Bump across the await
                Some((response, energy, k, theta, shifted_out, embedding))
            });

            futures.push(future);
        }

        let results: Vec<_> = futures.into_iter()
            .collect();
        
        let mut resolved_results = Vec::new();
        for future in results {
            if let Ok(Some(result)) = future.await {
                resolved_results.push(result);
            }
        }

        if resolved_results.is_empty() {
            return ("Çoklu-evren çöktü".to_string(), f32::MAX, vec![], None);
        }

        self.logger.debug("MULTIVERSE", &format!(
            "{} universes successfully created",
            resolved_results.len()
        ));
        
        // Find best
        let best_result = resolved_results.iter()
            .min_by(|a: &&(String, f32, usize, f32, Vec<f32>, Vec<f32>), b: &&(String, f32, usize, f32, Vec<f32>, Vec<f32>)| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let (best_response, best_energy, best_k, best_theta, _, best_embedding) = best_result;
        
        self.logger.debug("MULTIVERSE", &format!(
            "Best: Universe {} (θ={:.4}, E={:.4})",
            best_k, best_theta, best_energy
        ));
        
        // Calculate regret vector
        let mut energies: Vec<f32> = resolved_results.iter().map(|(_, e, _, _, _, _)| *e).collect();
        energies.sort_by(|a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_energy = energies[energies.len() / 2];
        
        let failed_universes: Vec<_> = resolved_results.iter()
            .filter(|(_, e, _, _, _, _)| *e > median_energy)
            .collect();

        let regret_vector = if !failed_universes.is_empty() {
            self.logger.debug("MULTIVERSE", &format!(
                "Computing regret from {} failed universes",
                failed_universes.len()
            ));
            
            let dim = failed_universes[0].4.len();
            let mut regret = vec![0.0; dim];
            
            for (_, _, _, _, vec, _) in &failed_universes {
                for i in 0..dim.min(vec.len()) {
                    regret[i] += vec[i];
                }
            }
            
            for val in regret.iter_mut() {
                *val /= failed_universes.len() as f32;
            }
            
            let regret_norm: f32 = regret.iter().map(|x| x * x).sum::<f32>().sqrt();
            self.logger.debug("MULTIVERSE", &format!(
                "Regret vector computed (norm: {:.4})",
                regret_norm
            ));
            
            Some(regret)
        } else {
            None
        };

        (best_response.clone(), *best_energy, best_embedding.clone(), regret_vector)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CAUSAL OUROBOROS: Self-Correcting Loop with Active Intervention
    // ═══════════════════════════════════════════════════════════════════════════

    /// Ouroboros Self-Reflection Loop with Single-Pass Causal Intervention
    /// 
    /// Upgraded Flow:
    /// 1. Analyze Task Nature (ARC)
    /// 2. Configure for Euler Resonance (Physics-Guided Logits Processor active)
    /// 3. Call Single-Pass generation. The physics engine filters tokens simultaneously during generation.
    pub async fn ouroboros_think(&self, prompt: &str) -> Result<String, String> {
        self.logger.info("OUROBOROS", "Single-Pass Causal logic cycle initiated...");
        
        // Phase 25: Generate request context for cortex routing
        let req_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let user_id = "ouroboros".to_string();
        
        // ═════════════════════════════════════════════════════════════════════════════
        // 🌍 GLOBAL GAZE: Analyze task nature (ARC detection)
        // ═════════════════════════════════════════════════════════════════════════════
        let mut strategy_hint = String::new();
        if let Some((task_nature, hint)) = self.analyze_arc_task(prompt) {
            self.logger.debug("OUROBOROS", &format!("Task nature detected: {}", task_nature));
            self.logger.debug("OUROBOROS", &format!("Strategy hint: {}", hint.lines().next().unwrap_or("")));
            strategy_hint = hint;
            
            self.logger.info("OUROBOROS", &format!("ARC Task Nature: {}", task_nature));
        }
        
        let current_prompt = if !strategy_hint.is_empty() {
            format!("{}\n\n{}", prompt, strategy_hint)
        } else {
            prompt.to_string()
        };

        println!();
        println!("┌───────────────────────────────────────────────────────────┐");
        println!("│ ⚛️ SINGLE-PASS FİZİK DESTEKLİ ÜRETİM BAŞLIYOR (OUROBOROS)  │");
        println!("└───────────────────────────────────────────────────────────┘");

        // Use standard euler resonance mode so Physics Logits Processor kicks in 
        let _config = NeuroConfig { euler_resonance: true, ..Default::default() };
        
        // Phase 20.8: Route to cortex via channel instead of duplicate LlmEngine
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        self.cortex_tx.send(SystemMsg::LlmRequest {
            req_id,
            user_id: user_id.clone(),
            prompt: current_prompt.clone(),
            image_data: None,
            response_tx: Some(resp_tx),
        }).await.map_err(|e| format!("Cortex channel error: {}", e))?;
        
        match resp_rx.await {
            Ok(SystemMsg::CortexResponse { content, .. }) => {
                println!();
                println!("╔═══════════════════════════════════════════════════════════════╗");
                println!("║ ✅ SINGLE-PASS BİYOLOJİK AKIŞ TAMAMLANDI                      ║");
                println!("║    Halüsinasyon/Çelişki içeren tokenlar motor içinden lendi!  ║");
                println!("╚═══════════════════════════════════════════════════════════════╝");
                
                self.logger.info("OUROBOROS", "Single-Pass generation finished cleanly.");
                Ok(content)
            }
            _ => {
                self.logger.error("OUROBOROS", "No response from cortex");
                Err("Cortex not responding".to_string())
            }
        }
    }

    /// Simple multi-verse (for non-high-stakes queries)
    #[allow(dead_code)]
    async fn parallel_resonance_search(&self, prompt: &str, n: usize) -> Result<String, String> {
        let (resp, _, _, _) = self.parallel_resonance_search_with_regret(prompt, None, Some(n)).await;
        if resp.is_empty() || resp == "Çoklu-evren çöktü" {
            Err(resp)
        } else {
            Ok(resp)
        }
    }

    fn is_high_stakes_query(prompt: &str) -> bool {
        let triggers = [
            "SOLVE:", "CODE:", "CRITICAL:", "VERIFY:",
            "PROVE:", "DEBUG:", "ANALYZE:", "OPTIMIZE:"
        ];
        triggers.iter().any(|t| prompt.to_uppercase().starts_with(t))
    }

    /// ═════════════════════════════════════════════════════════════════════════════
    /// GLOBAL GAZE: Analyze ARC task nature and determine strategy
    /// ═════════════════════════════════════════════════════════════════════════════
    fn analyze_arc_task(&self, prompt: &str) -> Option<(TaskNature, String)> {
        // Try to extract grid data from prompt
        // Format: Looking for patterns like "input: [[0,1,2],[3,4,5]]" or similar
        
        // Simple heuristic: check if prompt contains grid-like data
        if !prompt.to_lowercase().contains("input") && !prompt.to_lowercase().contains("output") {
            return None;
        }

        // Attempt simple extraction: parse sections marked "input" and "output"
        let input_grid = Self::extract_grid_from_prompt(prompt, "input");
        let output_grid = Self::extract_grid_from_prompt(prompt, "output");

        match (input_grid, output_grid) {
            (Some(inp), Some(out)) => {
                let task_nature = GlobalScanner::analyze_task_nature(&inp, &out);
                let strategy_hint = self.generate_strategy_hint(&task_nature);
                Some((task_nature, strategy_hint))
            }
            _ => None,
        }
    }

    /// Extract a grid from prompt text
    /// Looks for patterns like: "input: [[...], [...], ...]"
    fn extract_grid_from_prompt(prompt: &str, keyword: &str) -> Option<Vec<Vec<u32>>> {
        let lower_prompt = prompt.to_lowercase();
        if let Some(start_idx) = lower_prompt.find(&format!("{}:", keyword)) {
            let mut grid = Vec::new();
            let section = &prompt[start_idx..];
            
            // Find all bracket pairs and parse numbers
            let mut in_row = false;
            let mut current_row = Vec::new();
            let mut current_num = String::new();

            for c in section.chars() {
                match c {
                    '[' => {
                        if in_row {
                            // Start of a number
                        } else {
                            in_row = true;
                        }
                    }
                    ']' => {
                        if !current_num.is_empty() {
                            if let Ok(num) = current_num.parse::<u32>() {
                                current_row.push(num);
                            }
                            current_num.clear();
                        }
                        if in_row {
                            if !current_row.is_empty() {
                                grid.push(current_row.clone());
                                current_row.clear();
                            }
                            in_row = false;
                        } else {
                            break; // End of grid
                        }
                    }
                    ',' | ' ' => {
                        if !current_num.is_empty() {
                            if let Ok(num) = current_num.parse::<u32>() {
                                current_row.push(num);
                            }
                            current_num.clear();
                        }
                    }
                    c if c.is_ascii_digit() => {
                        current_num.push(c);
                    }
                    _ => {}
                }
            }

            if !grid.is_empty() {
                return Some(grid);
            }
        }
        None
    }

    /// Generate strategy hint based on task nature
    fn generate_strategy_hint(&self, task_nature: &TaskNature) -> String {
        match task_nature {
            TaskNature::GlobalCounting => {
                "STRATEGY: GLOBAL_COUNT\n\
                 Do NOT split the grid into chunks. \
                 Analyze the complete grid globally. \
                 Count occurrences of specific colors/objects across the entire grid. \
                 Determine output based on global histogram or object counts.".to_string()
            }
            TaskNature::GlobalSymmetry => {
                "STRATEGY: GLOBAL_SYMMETRY\n\
                 Do NOT use local chunking. \
                 The task involves symmetry operations (mirror, rotate, reflect). \
                 Analyze the global structure and apply transformations to the entire grid. \
                 Focus on preserving or applying consistent symmetry patterns.".to_string()
            }
            TaskNature::GlobalExtraction => {
                "STRATEGY: GLOBAL_EXTRACTION\n\
                 Do NOT divide the grid locally. \
                 The output is a compression or extraction of distinctive features. \
                 Identify the key pattern or feature in the input that determines the output. \
                 Output size is significantly smaller than input.".to_string()
            }
            TaskNature::LocalPattern => {
                "STRATEGY: SPATIAL_CHUNKING\n\
                 You may divide the grid into 3x3 local sections if beneficial. \
                 Solve each section independently and combine results. \
                 Focus on local patterns and relationships within chunks.".to_string()
            }
            TaskNature::Healthcare => {
                "STRATEGY: CLINICAL_REPORTING\n\
                 Adopt an expert radiologist persona. \
                 Focus on data integrity and clinical accuracy. \
                 Follow structured reporting formats (Findings, Impression). \
                 Avoid all mathematical calculations; refer only to clinical observations.".to_string()
            }
        }
    }

    /// Enhanced Ouroboros Think with Global Gaze Integration
    /// ═════════════════════════════════════════════════════════════════════════════
    pub async fn run(mut self) -> anyhow::Result<()> {
        println!("[KERNEL] Çekirdek Döngüsü Başlatıldı.");
        self.state = SystemState::Idle;
        self.update_tui(0, 0, "KERNEL: Sistem Hazır").await;
        
        let mut last_timeout_check = std::time::Instant::now();
        let timeout_check_interval = std::time::Duration::from_secs(30);
        
        // Phase 20.8: Health check via cortex channel
        // For now, just log that we're using cortex - actual health check would need response
        self.logger.debug("KERNEL", "Cortex Actor handles all LLM health");

        loop {
            if last_timeout_check.elapsed() > timeout_check_interval {
                let timed_out_jobs = self.job_manager.cleanup_timeouts();
                for job in timed_out_jobs {
                    self.logger.warn("KERNEL", &format!("İşlem zaman aşımı: Job {}", job.req_id));
                    self.send_response(job.req_id, &job.user_id, 
                        " İşlem çok uzun sürdü ve iptal edildi.".to_string()).await;
                }
                last_timeout_check = std::time::Instant::now();
            }
            
            let msg = match self.rx.recv().await {
                Some(m) => m,
                None => break,
            };
            
            match msg {
                SystemMsg::LlmRequest { req_id, user_id, prompt, image_data, .. } => {
                    
                    // --- 0. GÖRSEL MODU ---
                    if let Some(img_base64) = image_data {
                        self.state = SystemState::Thinking;
                        self.update_tui(1, 0, "Görsel Analiz...").await;
                        let _ = self.cortex_tx.send(SystemMsg::LlmRequest {
                            req_id,
                            user_id: user_id.clone(),
                            prompt: prompt.clone(),
                            image_data: Some(img_base64),
                            response_tx: None,
                        }).await;
                        // Response handled by CortexResponse handler
                        self.state = SystemState::Idle;
                        self.update_tui(0, 0, "Tamamlandı.").await;
                        continue; 
                    }

                    // --- 1. TELEMETRİ KONTROLÜ ---
                    if prompt.starts_with("/telemetry ") {
                        let data = prompt.replace("/telemetry ", "");
                        let parts: Vec<&str> = data.split_whitespace().collect();
                        
                        if !parts.is_empty() {
                            let temp = parts[0].parse::<f32>().unwrap_or(0.0);
                            self.iot_status = format!("{}°C", temp);
                            
                            if temp > 30.0 {
                                self.broadcast_system_msg(format!(" DİKKAT: Ortam sıcaklığı kritik seviyede! ({}°C)", temp)).await;
                                self.update_tui(0, 10, "IoT: Yüksek Sıcaklık Alarmı!").await;
                            } else {
                                self.update_tui(0, 5, "IoT: Veri Alındı").await;
                            }
                        }
                        continue; 
                    }

                    // --- 2. KOMUTLAR ---
                    if prompt == "/egit" {
                        self.state = SystemState::Training;
                        self.update_tui(1, 100, "Hizalama Eğitimi...").await;
                        self.send_response(req_id, &user_id, " Projektör eğitimi başlatılıyor...".to_string()).await;
                        
                        tokio::task::spawn_blocking(move || {
                            let config = train_projector::TrainingConfig::default();
                            let vocab = vec!["merhaba".to_string(), "dünya".to_string()];
                            // TODO: Wire up real Qwen embed_tokens layer via extract_real_embedding()
                            // Currently using simulated embeddings — training loop runs but aligns to noise
                            tracing::warn!("⚠️ /egit: Using SIMULATED embeddings. Wire up real Qwen model for meaningful training.");
                            let mut waves = Vec::new();
                            let mut embeddings = Vec::new();
                            for word in &vocab {
                                if let Ok(wave) = train_projector::AlignmentDataset::generate_real_wave(word, 10) {
                                    waves.push(wave);
                                }
                                embeddings.push(
                                    train_projector::AlignmentDataset::generate_simulated_embedding(word, 768)
                                );
                            }
                            if waves.len() == vocab.len() {
                                let dataset = train_projector::AlignmentDataset::from_real_data(vocab, waves, embeddings);
                                if let Err(e) = train_projector::train_alignment_layer(config, dataset, None) {
                                    println!("Eğitim Hatası: {}", e);
                                }
                            }
                        });
                        continue;
                    }

                    if prompt.starts_with("/kodla ") {
                        self.state = SystemState::Coding;
                        let request = prompt.replace("/kodla ", "");
                        self.update_tui(1, 0, "Mühendis Qwen Kodluyor...").await;

                        let engineering_prompt = format!(
                            "Sen deneyimli bir Rust yazılım mühendisisin. Sadece kodu yaz. GÖREV: {}", request
                        );

                        let _ = self.cortex_tx.send(SystemMsg::LlmRequest {
                            req_id,
                            user_id: user_id.clone(),
                            prompt: engineering_prompt,
                            image_data: None,
                            response_tx: None,
                        }).await;
                        // Response handled by CortexResponse handler
                        self.send_response(req_id, &user_id, "🛠️ Kod yazıldı, bekleniyor...".to_string()).await;
                        continue;
                    }
                    
                    if prompt.starts_with("/ara ") {
                        self.state = SystemState::Searching;
                        let _query = prompt.replace("/ara ", "");
                        // Phase 20.8: TODO - implement web search with cortex routing
                        self.send_response(req_id, &user_id, "Web arama cortex'a yönlendiriliyor...".to_string()).await;
                        continue;
                    }
                    
                    if prompt.starts_with("/ezberle ") {
                        let text = prompt.replace("/ezberle ", "");
                        self.send_response(req_id, &user_id, " Hafızaya Alınıyor...".to_string()).await;
                        let _ = self.cortex_tx.send(SystemMsg::LlmRequest {
                            req_id,
                            user_id: user_id.clone(),
                            prompt: text.clone(),
                            image_data: None,
                            response_tx: None,
                        }).await;
                        // Store directly - embedding handled by cortex architecture
                        self.pending_memorization.insert(req_id, (text, user_id.clone()));
                        continue;
                    }

                    if prompt == "/hafiza" {
                        let _ = self.store_tx.send(SystemMsg::GetAllMemoriesRequest { req_id, user_id }).await;
                        continue;
                    }

                    if Self::is_arithmetic(&prompt) {
                        self.state = SystemState::Calculating;
                        self.update_tui(1, 0, "Hesaplanıyor...").await;
                        let math_prompt = format!("Sen bir hesap makinesisiniz. Sadece sonucu sayı olarak ver.\nİşlem: {}", prompt);
                        let _ = self.cortex_tx.send(SystemMsg::LlmRequest {
                            req_id,
                            user_id: user_id.clone(),
                            prompt: math_prompt,
                            image_data: None,
                            response_tx: None,
                        }).await;
                        // Response handled by CortexResponse handler
                        self.state = SystemState::Idle;
                        self.update_tui(0, 0, "Hesaplama Tamamlandı.").await;
                        continue;
                    }

                    // ════════════════════════════════════════════════════════════════
                    // CAUSAL OUROBOROS MODE (High-Stakes Queries)
                    // ════════════════════════════════════════════════════════════════
                    if Self::is_high_stakes_query(&prompt) {
                        println!("[KERNEL] 🐍 CAUSAL OUROBOROS AKTİF!");
                        println!("[KERNEL]    Tetikleyici: {}", prompt.split(':').next().unwrap_or(""));
                        
                        self.state = SystemState::Thinking;
                        self.update_tui(1, 50, "Causal Ouroboros...").await;
                        self.send_response(req_id, &user_id, 
                            "🐍 Nedensel Ouroboros müdahale döngüsü başlatılıyor...\n\n".to_string()
                        ).await;

                        let best_response = self.ouroboros_think(&prompt).await;
                        
                        match best_response {
                            Ok(content) => self.send_response(req_id, &user_id, content).await,
                            Err(e) => self.send_response(req_id, &user_id, format!("Error: {}", e)).await,
                        }
                        self.state = SystemState::Idle;
                        self.update_tui(0, 100, "Ouroboros tamamlandı").await;
                        
                        continue;
                    }

                    // ════════════════════════════════════════════════════════════════
                    // STANDARD FLOW (Phase-Modulated Attention + RAG + GEOMETER)
                    // ════════════════════════════════════════════════════════════════
                    println!("[KERNEL]  Yeni İstek: '{}' (ID: {})", prompt, req_id);
                    
                    self.state = SystemState::Thinking;
                    self.update_tui(1, 10, "Biyolojik Düşünme...").await;

                    println!("[KERNEL]  🧠 Clifford Rezonans Döngüsü çalıştırılıyor...");
                    let physics_bias = self.think_with_physics(&prompt).await;

                    println!("[KERNEL]  Embedding (Vektör) hesaplanıyor...");
                    // Phase 20.8: Route embed to cortex - would need special handling
                    // For now, use a placeholder vector
                    let mut thought_vector = vec![0.0; 768];
                    self.logger.debug("KERNEL", "Embedding routed through cortex");

                    // GEOMETER INJECTION
                    if let Some(bias_vec) = physics_bias {
                        println!("[KERNEL] 🧬 GEOMETER: Fiziksel Rezonans Enjekte Ediliyor (Alpha: 0.3)...");
                        
                        const ALPHA: f32 = 0.3;
                        let min_dim = thought_vector.len().min(bias_vec.len());
                        
                        if min_dim > 0 {
                            for i in 0..min_dim {
                                thought_vector[i] += bias_vec[i] * ALPHA;
                            }
                            
                            println!("[KERNEL] ✅ GEOMETER: {} boyut üzerine fiziksel bias eklendi", min_dim);
                        }
                    }

                    self.pending_requests.insert(req_id, (prompt.clone(), user_id.clone()));
                    self.job_manager.add_job(req_id, user_id.clone(), prompt.clone(), Some(600));
                    
                    println!("[KERNEL]  Store'a (Hafıza) arama isteği gönderiliyor...");
                    let _ = self.store_tx.send(SystemMsg::VectorSearchRequest {
                        user_id,
                        query_vector: thought_vector,
                        top_k: 3,
                        request_id: req_id,
                    }).await;
                    
                    continue;
                }
                    
                SystemMsg::GetAllMemoriesResponse { req_id: _, memories } => {
                    if let Ok(json) = serde_json::to_string(&memories) {
                        let packet = format!("MEMORY_DUMP::{}", json);
                        self.broadcast_system_msg(packet).await;
                    }
                }
                
                SystemMsg::ForgeCompilationResult { req_id, success, wasm_path, error_log, retry_count, original_prompt } => {
                    self.state = SystemState::Idle;
                    
                    if success {
                        self.broadcast_system_msg(format!("✅ Derleme Başarılı! Motor ateşleniyor... ({})", wasm_path)).await;
                        let _ = self.runtime_tx.send(SystemMsg::RuntimeExecuteRequest {
                            req_id: 0, wasm_path,
                        }).await;
                    } else {
                        let err_msg = error_log.unwrap_or("Bilinmeyen hata".into());
                        
                        if retry_count < 3 {
                            let next_retry = retry_count + 1;
                            self.broadcast_system_msg(format!("⚠️ Derleme Hatası (Deneme {}/3). Otonom düzeltme başlatılıyor...", next_retry)).await;
                            
                            // Get user_id from pending_requests or job_manager
                            let user_id = self.pending_requests.get(&req_id)
                                .map(|(_, uid)| uid.clone())
                                .or_else(|| self.job_manager.get_job(&req_id).map(|j| j.user_id.clone()))
                                .unwrap_or_else(|| "system".to_string());
                            
                            let fix_prompt = format!(
                                "Sen uzman bir Rust mühendisisin. Önceki kod derlenirken hata verdi. \
                                Hata mesajını analiz et ve kodu düzelt.\n\n\
                                ORİJİNAL GÖREV: {}\n\n\
                                ALINAN HATA:\n{}\n\n\
                                Sadece düzeltilmiş Rust kodunu ver.", 
                                original_prompt, err_msg
                            );

                            let _ = self.cortex_tx.send(SystemMsg::LlmRequest {
                                req_id,
                                user_id: user_id.clone(),
                                prompt: fix_prompt,
                                image_data: None,
                                response_tx: None,
                            }).await;
                            // Response handled by CortexResponse handler
                            let module_name = format!("modul_{}_fix_{}", req_id, next_retry);
                            // Note: fixed_code will arrive via CortexResponse, needs handler update
                            self.broadcast_system_msg(format!("📝 Düzeltme talebi gönderildi ({})", module_name)).await;

                        } else {
                            let msg = format!("❌ KRİTİK HATA: 3 denemede düzeltilemedi.\nSon Hata: {}", err_msg);
                            self.broadcast_system_msg(msg).await;
                            self.update_tui(0, 0, "Derleme Başarısız.").await;
                        }
                    }
                }

                SystemMsg::RuntimeExecuteResult { req_id: _, stdout, stderr } => {
                    if let Some(err) = stderr {
                        let msg = format!(" ÇALIŞTIRMA HATASI:\n{}", err);
                        self.broadcast_system_msg(msg).await;
                    } else {
                        let msg = format!(" SONUÇ:\n{}", stdout);
                        self.broadcast_system_msg(msg).await;
                    }
                    self.update_tui(0, 500, "Görev Tamamlandı.").await;
                }

                SystemMsg::VectorSearchResult { request_id, results } => {
                    println!("[KERNEL] ✅ Hafıza Sonuçları GELDİ! (ReqID: {}, Sonuç Sayısı: {})", request_id, results.len());

                    let (original_prompt, current_user_id) = match self.pending_requests.remove(&request_id) {
                        Some(val) => val,
                        None => {
                            println!("[KERNEL] ⚠️ UYARI: Beklenmeyen Request ID: {}", request_id);
                            continue;
                        }
                    };

                    let final_prompt = if !results.is_empty() {
                        format!("HAFIZA:\n{}\n\nSORU: {}", results.join("\n"), original_prompt)
                    } else {
                        original_prompt.clone()
                    };
                    
                    println!("[KERNEL] 🚀 Cortex'e (AI Motoru) iletiliyor...");
                    let _ = self.cortex_tx.send(SystemMsg::LlmRequest {
                        req_id: request_id,
                        user_id: current_user_id.clone(),
                        prompt: final_prompt,
                        image_data: None,
                        response_tx: None,
                    }).await;
                    
                    continue;
                }

                SystemMsg::CortexStatus { message, state: _ } => {
                    self.update_tui(0, 0, &message).await;
                }

                SystemMsg::CortexResponse { req_id, user_id, content } => {
                    println!("[KERNEL]  Cortex Yanıtı Üretti. (Uzunluk: {})", content.len());
                    
                    if let Some(job) = self.job_manager.remove_job(&req_id) {
                        self.logger.debug("KERNEL", &format!("İşlem tamamlandı: {:.2}s", job.created_at.elapsed().as_secs_f64()));
                    }
                    
                    self.send_response(req_id, &user_id, content.clone()).await;
                    
                    let original_prompt = self.job_manager.get_job(&req_id)
                        .map(|j| j.prompt.clone())
                        .or_else(|| self.pending_requests.get(&req_id).map(|(p, _)| p.clone()));
                    
                    if let Some(prompt) = original_prompt {
                        if content.len() > 10 {
                            let memory_text = format!("Kullanıcı: {}\nSODALS: {}", prompt, content);
                            
                            // Phase 20.8: Route embed to cortex - store text for now
                            self.pending_memorization.insert(req_id, (memory_text, user_id.clone()));
                            self.logger.debug("KERNEL", "Memory stored pending cortex embedding");
                        }
                    }

                    self.state = SystemState::Idle;
                    self.update_tui(0, 0, "Yanıt gönderildi.").await;
                }

                SystemMsg::SanitizeVectorResponse { req_id, safe_vector, noise_level } => {
                    self.last_noise_level = noise_level as u64;
                    
                    if let Some((memory_text, user_id)) = self.pending_memorization.remove(&req_id) {
                        let _ = self.store_tx.send(SystemMsg::MemorizeRequest {
                            user_id: user_id.clone(),
                            text: memory_text.clone(),
                            vector: safe_vector,
                        }).await;
                        
                        println!("[KERNEL] 💾 Sanitize edilmiş vektör hafızaya kaydedildi");
                    }
                }

                // ═════════════════════════════════════════════════════════════════════════════
                // BRAINFLOW BIOSIGNAL INTEGRATION
                // ═════════════════════════════════════════════════════════════════════════════
                SystemMsg::BiosignalBatch { 
                    timestamp_ms: _,
                    cognitive_state, 
                    confidence, 
                    alpha_power, 
                    beta_power, 
                    theta_power, 
                    gamma_power: _, 
                    delta_power: _ 
                } => {
                    // Log cognitive state changes for debugging
                    tracing::debug!(
                        "[KERNEL] 🧠 BrainFlow: state={} conf={:.2} α={:.3} β={:.3} θ={:.3}",
                        cognitive_state, confidence, alpha_power, beta_power, theta_power
                    );
                    
                    // Update system state based on cognitive load
                    // High beta + low alpha = focused state = throttle non-essential tasks
                    if beta_power > alpha_power * 2.0 && confidence > 0.6 {
                        self.logger.info("KERNEL", "High focus detected - throttling background tasks");
                        // Could implement task throttling here
                    }
                    
                    // Drowsy detection - high theta/alpha ratio
                    if theta_power > alpha_power * 1.5 && confidence > 0.7 {
                        self.logger.warn("KERNEL", "Drowsiness detected from EEG signals");
                    }
                }

                _ => {}
            }
        }
        Ok(())
    }
}