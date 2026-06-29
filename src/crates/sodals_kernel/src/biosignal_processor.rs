// src/biosignal_processor.rs - High-Level Biosignal API for sodals_kernel
//
// Integrates BrainFlow acquisition with the Causal Ouroboros engine.
// Provides cognitive state classification and real-time feature extraction.

use anyhow::Result;
use tokio::sync::{mpsc, oneshot};
use std::sync::Arc;
use tracing;

use sodals_neuro::brainflow_bridge::{BrainFlowBridge, BrainFlowParams, board_ids};
use sodals_neuro::signal_processor::{SignalProcessor, FilterParams, BandPowers, RealtimePipeline};
use sodals_neuro::neuro_kernel::{CognitiveToken, CognitiveEventBus, AnomalousStateEvent};
use sodals_net::protocol::SystemMsg;

// ============================================================================
// BIOSIGNAL DATA TYPES
// ============================================================================

#[derive(Debug, Clone)]
pub struct BiosignalBatch {
    pub timestamp: std::time::Instant,
    pub raw_data: Vec<Vec<f64>>, // (samples, channels)
    pub filtered_data: Vec<Vec<f64>>,
    pub band_powers: BandPowers,
    pub cognitive_state: CognitiveState,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CognitiveState {
    Idle,
    Focused,      // High beta, low alpha
    Relaxed,      // High alpha, low beta
    Drowsy,       // High theta, low beta
    Stressed,     // High beta, elevated gamma
    Creative,     // Elevated alpha-theta boundary
    Overloaded,   // Gamma saturation
}

impl std::fmt::Display for CognitiveState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CognitiveState::Idle => write!(f, "IDLE"),
            CognitiveState::Focused => write!(f, "FOCUSED"),
            CognitiveState::Relaxed => write!(f, "RELAXED"),
            CognitiveState::Drowsy => write!(f, "DROWSY"),
            CognitiveState::Stressed => write!(f, "STRESSED"),
            CognitiveState::Creative => write!(f, "CREATIVE"),
            CognitiveState::Overloaded => write!(f, "OVERLOADED"),
        }
    }
}

// ============================================================================
// STATE CLASSIFIER
// ============================================================================

pub struct StateClassifier {
    history: Vec<(CognitiveState, f32)>,
    history_size: usize,
    transition_threshold: f32,
}

impl StateClassifier {
    pub fn new(history_size: usize) -> Self {
        Self {
            history: Vec::with_capacity(history_size),
            history_size,
            transition_threshold: 0.3,
        }
    }

    /// Classify cognitive state from band powers
    pub fn classify(&mut self, powers: &BandPowers) -> (CognitiveState, f32) {
        let alpha_theta = powers.alpha_theta_ratio();
        let beta_alpha = powers.beta_alpha_ratio();
        let total = powers.total_power();
        
        // Simple rule-based classification (can be replaced with ML model)
        let (state, confidence) = if beta_alpha > 2.0 && powers.beta > powers.theta * 3.0 {
            (CognitiveState::Focused, (beta_alpha / 3.0).min(1.0))
        } else if alpha_theta > 1.5 && powers.alpha > 0.3 {
            (CognitiveState::Relaxed, (alpha_theta / 2.0).min(1.0))
        } else if powers.theta > powers.alpha && powers.theta > 0.4 {
            (CognitiveState::Drowsy, (powers.theta / 0.5).min(1.0))
        } else if beta_alpha > 3.0 && powers.gamma > 0.15 {
            (CognitiveState::Stressed, (powers.gamma / 0.2).min(1.0))
        } else if powers.alpha > 0.25 && powers.theta > 0.2 && powers.beta < 0.2 {
            (CognitiveState::Creative, 0.7)
        } else if total > 2.0 && powers.gamma > 0.2 {
            (CognitiveState::Overloaded, (total / 3.0).min(1.0))
        } else {
            (CognitiveState::Idle, 0.5)
        };

        // Update history for smoothing
        self.history.push((state, confidence));
        if self.history.len() > self.history_size {
            self.history.remove(0);
        }

        // Return smoothed state
        let smoothed = self.smoothed_state();
        (smoothed.0, smoothed.1)
    }

    fn smoothed_state(&self) -> (CognitiveState, f32) {
        if self.history.is_empty() {
            return (CognitiveState::Idle, 0.0);
        }

        // Simple majority vote with confidence averaging
        let mut counts = std::collections::HashMap::new();
        for (state, conf) in &self.history {
            let entry = counts.entry(*state).or_insert((0.0, 0usize));
            entry.0 += *conf;
            entry.1 += 1;
        }

        let mut best_state = CognitiveState::Idle;
        let mut best_score = 0.0;

        for (state, (total_conf, count)) in counts {
            let avg_conf = total_conf / count as f32;
            let score = avg_conf * (count as f32 / self.history.len() as f32);
            if score > best_score {
                best_score = score;
                best_state = state;
            }
        }

        (best_state, best_score)
    }

    /// Detect state transition
    pub fn detect_transition(&self) -> Option<CognitiveState> {
        if self.history.len() < 2 {
            return None;
        }

        let current = self.history.last()?.0;
        let previous = self.history[self.history.len() - 2].0;

        if current != previous {
            Some(current)
        } else {
            None
        }
    }
}

// ============================================================================
// BIOSIGNAL PROCESSOR ACTOR
// ============================================================================

pub struct BiosignalProcessor {
    bridge: Option<Arc<BrainFlowBridge>>,
    pipeline: Option<RealtimePipeline>,
    classifier: StateClassifier,
    event_bus: CognitiveEventBus,
    filter_params: FilterParams,
    
    // Channels
    cmd_rx: mpsc::Receiver<BiosignalCommand>,
    data_tx: mpsc::Sender<SystemMsg>,
    
    // State
    is_running: bool,
    buffer_size: usize,
}

#[derive(Debug)]
pub enum BiosignalCommand {
    Start(BrainFlowParams),
    Stop,
    SetFilterParams(FilterParams),
    GetState(oneshot::Sender<CognitiveState>),
    GetBandPowers(oneshot::Sender<BandPowers>),
}

impl BiosignalProcessor {
    pub fn new(
        cmd_rx: mpsc::Receiver<BiosignalCommand>,
        data_tx: mpsc::Sender<SystemMsg>,
    ) -> Self {
        Self {
            bridge: None,
            pipeline: None,
            classifier: StateClassifier::new(10),
            event_bus: CognitiveEventBus::new(),
            filter_params: FilterParams::default(),
            cmd_rx,
            data_tx,
            is_running: false,
            buffer_size: 256,
        }
    }

    /// Main actor loop
    pub async fn run(mut self) {
        tracing::info!("[BiosignalProcessor] Actor started");

        while let Some(cmd) = self.cmd_rx.recv().await {
            match cmd {
                BiosignalCommand::Start(params) => {
                    if let Err(e) = self.start(params).await {
                        tracing::error!("[BiosignalProcessor] Start failed: {}", e);
                    }
                }
                BiosignalCommand::Stop => {
                    self.stop().await;
                }
                BiosignalCommand::SetFilterParams(params) => {
                    self.filter_params = params;
                    tracing::info!("[BiosignalProcessor] Filter params updated");
                }
                BiosignalCommand::GetState(resp) => {
                    let (state, _) = self.classifier.smoothed_state();
                    let _ = resp.send(state);
                }
                BiosignalCommand::GetBandPowers(resp) => {
                    // Return last known band powers or empty
                    let _ = resp.send(BandPowers::default());
                }
            }
        }

        tracing::info!("[BiosignalProcessor] Actor stopped");
    }

    async fn start(&mut self, params: BrainFlowParams) -> Result<()> {
        if self.is_running {
            return Err(anyhow::anyhow!("Already running"));
        }

        tracing::info!("[BiosignalProcessor] Starting with board_id {}", params.board_id);

        // Create BrainFlow bridge
        let bridge = Arc::new(BrainFlowBridge::new(params)?);
        bridge.prepare()?;
        bridge.start_stream(self.buffer_size as i32)?;

        // Create pipeline
        let pipeline = RealtimePipeline::new(
            Arc::clone(&bridge),
            self.filter_params.clone(),
            self.buffer_size,
        )?;

        self.bridge = Some(bridge);
        self.pipeline = Some(pipeline);
        self.is_running = true;

        // Spawn processing task
        self.spawn_processing_task();

        tracing::info!("[BiosignalProcessor] Started successfully");
        Ok(())
    }

    fn spawn_processing_task(&mut self) {
        let data_tx = self.data_tx.clone();
        let mut classifier = StateClassifier::new(10);
        let bridge = self.bridge.clone();
        
        tokio::spawn(async move {
            let mut processor = SignalProcessor::new(candle_core::Device::Cpu);
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));

            loop {
                interval.tick().await;

                if let Some(ref b) = bridge {
                    // Acquire and process data
                    match b.acquire_tensor(256) {
                        Ok(tensor) => {
                            // Apply filters
                            let filtered = processor.bandpass_filter(
                                &tensor, 4.0, 40.0, 250.0
                            ).ok();

                            // Extract band powers
                            let powers = if let Some(ref f) = filtered {
                                processor.extract_band_powers(f, 250.0).ok()
                                    .unwrap_or_default()
                            } else {
                                BandPowers::default()
                            };

                            // Classify state
                            let (state, confidence) = classifier.classify(&powers);

                            // Detect transitions
                            if let Some(transition) = classifier.detect_transition() {
                                tracing::info!(
                                    "[BiosignalProcessor] State transition: {:?}",
                                    transition
                                );
                            }

                            // Send to kernel
                            let msg = SystemMsg::BiosignalBatch {
                                timestamp_ms: std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_millis() as u64,
                                cognitive_state: format!("{:?}", state),
                                confidence,
                                alpha_power: powers.alpha,
                                beta_power: powers.beta,
                                theta_power: powers.theta,
                                gamma_power: powers.gamma,
                                delta_power: powers.delta,
                            };

                            if let Err(e) = data_tx.send(msg).await {
                                tracing::warn!("[BiosignalProcessor] Failed to send data: {}", e);
                                break;
                            }

                            // Inject to ghost slot for neuro_kernel
                            let _ = b.inject_to_ghost_slot(tensor);
                        }
                        Err(e) => {
                            tracing::debug!("[BiosignalProcessor] Acquisition error: {}", e);
                        }
                    }
                }
            }
        });
    }

    async fn stop(&mut self) {
        if !self.is_running {
            return;
        }

        tracing::info!("[BiosignalProcessor] Stopping...");

        if let Some(ref bridge) = self.bridge {
            bridge.stop_stream().ok();
            bridge.release().ok();
        }

        self.bridge = None;
        self.pipeline = None;
        self.is_running = false;

        tracing::info!("[BiosignalProcessor] Stopped");
    }
}

// ============================================================================
// HANDLE FOR KERNEL INTEGRATION
// ============================================================================

#[derive(Clone)]
pub struct BiosignalHandle {
    cmd_tx: mpsc::Sender<BiosignalCommand>,
}

impl BiosignalHandle {
    pub fn new(cmd_tx: mpsc::Sender<BiosignalCommand>) -> Self {
        Self { cmd_tx }
    }

    pub async fn start(&self, params: BrainFlowParams) -> Result<()> {
        self.cmd_tx.send(BiosignalCommand::Start(params)).await
            .map_err(|_| anyhow::anyhow!("Command channel closed"))
    }

    pub async fn stop(&self) -> Result<()> {
        self.cmd_tx.send(BiosignalCommand::Stop).await
            .map_err(|_| anyhow::anyhow!("Command channel closed"))
    }

    pub async fn get_state(&self) -> Result<CognitiveState> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx.send(BiosignalCommand::GetState(tx)).await
            .map_err(|_| anyhow::anyhow!("Command channel closed"))?;
        
        rx.await.map_err(|_| anyhow::anyhow!("Response channel closed"))
    }

    pub async fn get_band_powers(&self) -> Result<BandPowers> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx.send(BiosignalCommand::GetBandPowers(tx)).await
            .map_err(|_| anyhow::anyhow!("Command channel closed"))?;
        
        rx.await.map_err(|_| anyhow::anyhow!("Response channel closed"))
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_classifier() {
        let mut classifier = StateClassifier::new(5);

        // Test focused state (high beta, low theta)
        let focused = BandPowers {
            delta: 0.1,
            theta: 0.1,
            alpha: 0.2,
            beta: 0.8,
            gamma: 0.1,
        };

        let (state, conf) = classifier.classify(&focused);
        assert_eq!(state, CognitiveState::Focused);
        assert!(conf > 0.5);

        // Test relaxed state (high alpha, low beta)
        let relaxed = BandPowers {
            delta: 0.1,
            theta: 0.15,
            alpha: 0.6,
            beta: 0.1,
            gamma: 0.05,
        };

        let (state, conf) = classifier.classify(&relaxed);
        assert_eq!(state, CognitiveState::Relaxed);
    }

    #[test]
    fn test_cognitive_state_display() {
        assert_eq!(format!("{}", CognitiveState::Focused), "FOCUSED");
        assert_eq!(format!("{}", CognitiveState::Relaxed), "RELAXED");
    }
}
