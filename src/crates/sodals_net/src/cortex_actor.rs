use tokio::sync::mpsc;
use crate::protocol::SystemMsg;
use crate::config::SystemConfig;
use crate::log_manager::LogManager;
use tokio::time::Duration;
use sodals_neuro::llm_core::{LlmEngine, NeuroConfig};
use sodals_neuro::engine::security::{OutboundSanitizer, StreamSanitizer};
use std::sync::{Arc, Mutex};
use crate::log_manager::LogLevel;

pub struct CortexActor {
    rx: mpsc::Receiver<SystemMsg>,
    kernel_tx: mpsc::Sender<SystemMsg>,
    model: Option<Arc<Mutex<LlmEngine>>>,
    config: SystemConfig,
    logger: LogManager,
    /// Receives () when memory crosses 85% of limit.
    /// Registered via sodals_alloc::register_pressure_hook at startup.
    mem_pressure_rx: Option<std::sync::mpsc::Receiver<()>>,
}

impl CortexActor {
    /// Lightweight constructor — does NOT load the model.
    /// Pass `mem_pressure_rx` from `std::sync::mpsc::sync_channel(1)`;
    /// register the corresponding Sender via `sodals_alloc::register_pressure_hook`.
    pub fn new(
        rx: mpsc::Receiver<SystemMsg>,
        kernel_tx: mpsc::Sender<SystemMsg>,
        config: SystemConfig,
        mem_pressure_rx: Option<std::sync::mpsc::Receiver<()>>,
    ) -> Self {
        let logger = LogManager::new(LogLevel::INFO);
        Self {
            rx,
            kernel_tx,
            model: None,
            config,
            logger,
            mem_pressure_rx,
        }
    }

    /// Main actor lifecycle - runs asynchronously
    pub async fn run(mut self) {
        // Immediately notify Kernel that we're loading
        let _ : Result<(), tokio::sync::mpsc::error::SendError<SystemMsg>> =
            self.kernel_tx
                .send(SystemMsg::CortexStatus {
                    message: "Model Yükleniyor...".to_string(),
                    state: "LOADING".to_string(),
                })
                .await;

        // Spawn blocking task to load the model (non-blocking for actor loop)
        let kernel_tx_clone = self.kernel_tx.clone();
        let config_clone = self.config.clone();
        let logger_clone = self.logger.clone();
        
        // Create the join handle for model loading - stored directly without Option
        let mut model_handle = tokio::task::spawn_blocking(move || {
            logger_clone.info("CORTEX", "Biological model loading (this may take 1-2 minutes)...");
            logger_clone.info("CORTEX", "Model loading started, please wait...");
            
            let model_load_start = std::time::Instant::now();
            let result = LlmEngine::new(&config_clone, &logger_clone);
            let elapsed = model_load_start.elapsed();
            
            match &result {
                Ok(_) => {
                    logger_clone.info("CORTEX", &format!("Biological model successfully loaded: {:.2} seconds", elapsed.as_secs_f64()));
                    if elapsed.as_secs() > 120 {
                        logger_clone.warn("CORTEX", &format!("Model loading took {} seconds - this is very long!", elapsed.as_secs()));
                    }
                }
                Err(e) => {
                    logger_clone.warn("CORTEX", &format!("Biological model failed to load (after {} seconds): {}", elapsed.as_secs_f64(), e));
                    logger_clone.warn("CORTEX", "Possible causes: Corrupted model file, insufficient RAM, disk I/O error");
                }
            }
            
            // wrap engine in a mutex so callers can borrow mutably
            (result.map(|eng| Arc::new(Mutex::new(eng))), kernel_tx_clone)
        });

        // Pending requests queue (for requests that arrive before model is ready)
        let mut pending_requests = Vec::new();

        // Track whether model loading is complete
        let mut model_loaded = false;

        // Actor message loop - continues while model is loading
        loop {
            tokio::select! {
                // Check if model loading is complete (only poll if not yet loaded)
                result = &mut model_handle, if !model_loaded => {
                    model_loaded = true;
                    match result {
                        Ok((model_result, tx)) => {
                            match model_result {
                                Ok(model) => {
                                    self.model = Some(model);
                                    // Notify Kernel that we're ready
                                    let _ : Result<(), tokio::sync::mpsc::error::SendError<SystemMsg>> =
                                        tx.send(SystemMsg::CortexStatus {
                                            message: "Nöral Motor Aktif".to_string(),
                                            state: "READY".to_string(),
                                        })
                                        .await;
                                    self.logger.info("CORTEX", "CortexActor ready and accepting messages");

                                    // ── Memory-pressure forwarder ─────────────────────
                                    // Spawns a tiny blocking task that polls the std channel
                                    // and forwards pressure signals into the tokio channel.
                                    if let Some(pressure_rx) = self.mem_pressure_rx.take() {
                                        let ktx = self.kernel_tx.clone();
                                        tokio::task::spawn_blocking(move || {
                                        while let Ok(()) = pressure_rx.recv() {
                                            let used_bytes =
                                                sodals_neuro::sodals_alloc::current_usage() as f64;
                                            let limit_bytes =
                                                sodals_neuro::sodals_alloc::memory_limit_bytes() as f64;
                                            let used_gb = used_bytes / 1_073_741_824.0;
                                            let limit_gb = limit_bytes / 1_073_741_824.0;
                                            let pct = if limit_bytes > 0.0 {
                                                (used_bytes / limit_bytes) * 100.0
                                            } else {
                                                0.0
                                            };

                                            // Lightweight log/status message for dashboards
                                            let _ = ktx.blocking_send(SystemMsg::CortexStatus {
                                                message: format!(
                                                    "[MEM] Memory pressure: {:.1} GB / {:.1} GB ({:.0}%)",
                                                    used_gb, limit_gb, pct
                                                ),
                                                state: "MEMORY_PRESSURE".to_string(),
                                            });

                                            // Structured memory-pressure signal for Kernel / other actors
                                            let _ = ktx.blocking_send(SystemMsg::MemoryPressure {
                                                used_gb,
                                                limit_gb,
                                                pct,
                                            });
                                        }
                                        });
                                    }
                                    
                                    // Process any pending requests
                                    for (req_id, user_id, prompt, physics_bias) in pending_requests.drain(..) {
                                        self.handle_llm_request(req_id, user_id, prompt, physics_bias).await;
                                    }
                                }
                                Err(e) => {
                                    self.logger.error("CORTEX", &format!("Model failed to load: {}", e));
                                    let _ : Result<(), tokio::sync::mpsc::error::SendError<SystemMsg>> =
                                        tx.send(SystemMsg::CortexStatus {
                                            message: format!("Model Yükleme Hatası: {}", e),
                                            state: "ERROR".to_string(),
                                        })
                                        .await;
                                }
                            }
                        }
                        Err(e) => {
                            self.logger.error("CORTEX", &format!("Model loading task error: {}", e));
                            let _ = self.kernel_tx.send(SystemMsg::CortexStatus {
                                message: "Model Yükleme Task Hatası".to_string(),
                                state: "ERROR".to_string(),
                            }).await;
                        }
                    }
                }

                // Handle incoming messages
                msg = self.rx.recv() => {
                    match msg {
                        Some(SystemMsg::LlmRequest { req_id, user_id, prompt, image_data, .. }) => {
                            // Ignore image requests (should be handled elsewhere)
                            if image_data.is_some() {
                                continue;
                            }

                            // For now, we don't have physics_bias in the request
                            // This will need to be passed from Kernel if needed
                            if self.model.is_some() {
                                self.handle_llm_request(req_id, user_id, prompt, None).await;
                            } else {
                                // Queue the request until model is ready
                                self.logger.info("CORTEX", &format!("Model not yet loaded, request queued (ReqID: {})", req_id));
                                pending_requests.push((req_id, user_id, prompt, None));
                            }
                        }
                        Some(SystemMsg::Shutdown) => {
                            self.logger.info("CORTEX", "Shutdown signal received, exiting...");
                            break;
                        }
                        Some(_) => {
                            // Ignore other messages
                        }
                        None => {
                            // Channel closed
                            self.logger.info("CORTEX", "Channel closed, exiting...");
                            break;
                        }
                    }
                }
            }
        }
    }

    async fn handle_llm_request(
        &self,
        req_id: u64,
        user_id: String,
        prompt: String,
        physics_bias: Option<candle_core::Tensor>,
    ) {
        let model: Arc<Mutex<LlmEngine>> = match &self.model {
            Some(m) => Arc::clone(m),
            None => {
                let _ = self.kernel_tx.send(SystemMsg::CortexResponse {
                    req_id,
                    user_id,
                    content: "⚠️ Model henüz yüklenmedi. Lütfen bekleyin.".to_string(),
                }).await;
                return;
            }
        };

        let kernel_tx = self.kernel_tx.clone();
        let prompt_clone = prompt.clone();
        let logger = self.logger.clone();
        
        // Channel for token streaming — carries ONLY the new chunk (zero-copy, O(1) per token)
        let (token_tx, mut token_rx) = mpsc::channel::<String>(100);
        let token_tx_blocking = token_tx.clone();

        // Spawn task to forward token updates to Kernel (PII-sanitized via StreamSanitizer)
        let kernel_tx_stream = kernel_tx.clone();
        tokio::spawn(async move {
            let mut sanitizer = StreamSanitizer::new();
            while let Some(chunk) = token_rx.recv().await {
                // Push chunk through sliding-window PII buffer
                if let Some(safe_text) = sanitizer.push(&chunk) {
                    let _ = kernel_tx_stream.send(SystemMsg::TokenStream {
                        token: safe_text,
                    }).await;
                }
            }
            // End-of-stream: flush remaining buffer
            if let Some(tail) = sanitizer.flush() {
                let _ = kernel_tx_stream.send(SystemMsg::TokenStream {
                    token: tail,
                }).await;
            }
        });

        // Spawn blocking task for model inference
        tokio::spawn(async move {
            let logger_blocking = logger.clone();

            let result: Result<Result<anyhow::Result<String>, _>, _> =
                tokio::time::timeout(
                    Duration::from_secs(1200), // 20 minutes timeout
                    tokio::task::spawn_blocking(move || {
                        let token_tx_clone = token_tx_blocking.clone();

                        // Callback for token streaming — sends ONLY the new chunk.
                        // NO accumulated String clone. O(1) per token.
                        let callback = move |chunk: String| {
                            let _ = token_tx_clone.try_send(chunk);
                        };

                        // model is Arc<Mutex<LlmEngine>> so lock it before calling
                        let mut engine = match model.lock() {
                            Ok(g) => g,
                            Err(e) => {
                                // poisoned, but continue with inner value
                                let guard: std::sync::MutexGuard<'_, LlmEngine> = e.into_inner();
                                guard
                            }
                        };

                        logger_blocking.info("CORTEX", "Bio engine başlatılıyor...");
                        logger_blocking.info(
                            "CORTEX",
                            &format!("Prompt uzunluğu: {} karakter", prompt_clone.len()),
                        );
                        let _start_time = std::time::Instant::now();

                        // perform generation and return result
                        sodals_neuro::engine::generation_loop::generate_stream(
                            &mut engine,
                            &prompt_clone,
                            physics_bias.as_ref(),
                            None,
                            callback,
                            &NeuroConfig::default(),
                            0,  // PHASE 49.6: No visual tokens for text-only generation
                            0,  // max_tokens (default 0)
                        )
                    }),
                )
                .await;

            match result {
                // successful result: engine generated a string
                Ok(Ok(Ok(response))) => {
                    logger.info("CORTEX", "Bio model completed successfully");
                    let sanitized = OutboundSanitizer::sanitize_text(&response);
                    let _ = kernel_tx.send(SystemMsg::CortexResponse {
                        req_id,
                        user_id,
                        content: sanitized,
                    })
                    .await;
                }
                Ok(Ok(Err(e))) => {
                    logger.error("CORTEX", &format!("Bio model hatası: {:?}", e));
                    let _ = kernel_tx.send(SystemMsg::CortexResponse {
                        req_id,
                        user_id,
                        content: format!("⚠️ Sistem hatası: {}. Lütfen tekrar deneyin.", e),
                    })
                    .await;
                }
                Ok(Err(e)) => {
                    let error_str = format!("{:?}", e);
                    let error_msg = if error_str.contains("panic") || error_str.contains("Panic") {
                        "⚠️ Model işlemi beklenmedik bir hata ile durdu. Model dosyası veya bellek kontrol edilmeli."
                    } else if error_str.contains("cancelled") || error_str.contains("Cancelled") {
                        "⚠️ Model işlemi iptal edildi. Lütfen tekrar deneyin."
                    } else {
                        "⚠️ Sistem işlem hatası. Model yanıt veremedi. Lütfen tekrar deneyin."
                    };
                    logger.error("CORTEX", &format!("Spawn blocking hatası: {}", error_str));
                    let _ = kernel_tx.send(SystemMsg::CortexResponse {
                        req_id,
                        user_id,
                        content: error_msg.to_string(),
                    }).await;
                }
                Err(_) => {
                    logger.error("CORTEX", "Timeout (20 dakika)");
                    let _ = kernel_tx.send(SystemMsg::CortexResponse {
                        req_id,
                        user_id,
                        content: "⚠️ İşlem zaman aşımına uğradı (20 dakika). Forward pass takılmış görünüyor.".to_string(),
                    }).await;
                }
            }
        });
    }
}