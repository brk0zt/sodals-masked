// src/protocol.rs - SECURE MESSAGING

use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

// [YENİ] Sistemin Ruh Halleri (State Machine)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SystemState {
    Booting,            // Açılıyor
    Idle,               // Boşta
    Thinking,           // Düşünüyor (LLM)
    Calculating,        // Hesaplıyor (Matematik)
    Searching,          // Arıyor (Web/Hafıza)
    Coding,             // Kodluyor
    Training,           // Eğitiliyor
    Secure,             // Güvenli Mod
    Alarm(String),      // Tehlike (Mesajlı)
    Panic(String),      // Sistem Hatası
}

// String dönüşümü (TUI için)
impl std::fmt::Display for SystemState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SystemState::Booting => write!(f, "BAŞLATILIYOR"),
            SystemState::Idle => write!(f, "HAZIR"),
            SystemState::Thinking => write!(f, "DÜŞÜNÜYOR"),
            SystemState::Calculating => write!(f, "HESAPLIYOR"),
            SystemState::Searching => write!(f, "ARIYOR"),
            SystemState::Coding => write!(f, "KODLUYOR"),
            SystemState::Training => write!(f, "EĞİTİLİYOR"),
            SystemState::Secure => write!(f, "GÜVENLİ MOD"),
            SystemState::Alarm(msg) => write!(f, "ALARM: {}", msg),
            SystemState::Panic(msg) => write!(f, "HATA: {}", msg),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: u64,
    pub user_id: String,
    pub text: String,
    pub vector: Vec<f32>,
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum SystemMsg {
    // --- ÇEKİRDEK İŞLEMLERİ ---
    
    /// Kullanıcıdan gelen RAG/LLM isteği
    LlmRequest {
        req_id: u64,
        user_id: String,
        prompt: String,
        image_data: Option<String>,
        /// Optional response channel for direct reply (fixes ouroboros deadlock)
        #[serde(skip)]
        response_tx: Option<oneshot::Sender<SystemMsg>>,
    },

    /// Ağ katmanına (Socket) gönderilen yanıt
    CortexResponse {
        req_id: u64,
        user_id: String,
        content: String,
    },

    // --- HAFIZA VEKTÖR İŞLEMLERİ ---

    /// Metni vektöre çevirip kaydetme emri
    MemorizeRequest {
        user_id: String,
        text: String,
        vector: Vec<f32>,
    },

    /// Vektör veritabanında arama
    VectorSearchRequest {
        request_id: u64,
        user_id: String,
        query_vector: Vec<f32>,
        top_k: usize,
    },

    /// Arama sonucu
    VectorSearchResult {
        request_id: u64,
        results: Vec<String>,
    },

    /// Tüm hafızayı listele (Debug/Dashboard için)
    GetAllMemoriesRequest {
        req_id: u64,
        user_id: String,
    },

    GetAllMemoriesResponse {
        req_id: u64,
        memories: Vec<MemoryRecord>,
    },
    
    /// Hafızayı temizle
    PruneMemoriesRequest {
        req_id: u64,
        retention_days: u64,
    },
    
    PruneMemoriesResult {
        req_id: u64,
        deleted_count: usize,
        details: String,
    },

    // --- GÜVENLİK (SENTINEL) ---

    /// Vektör verisini gürültü ekleyerek güvenli hale getir
    SanitizeVectorRequest {
        req_id: u64,
        raw_vector: Vec<f32>,
    },

    /// Güvenli vektör yanıtı
    SanitizeVectorResponse {
        req_id: u64,
        safe_vector: Vec<f32>,
        noise_level: f64,
    },

    // --- KOD ÜRETİMİ VE DERLEME (FORGE) ---

    ForgeCompileRequest {
        req_id: u64,
        source_code: String,
        module_name: String,
        // YENİ EKLENEN ALANLAR:
        retry_count: u32,
        original_prompt: String, 
    },

    ForgeCompilationResult {
        req_id: u64,
        success: bool,
        wasm_path: String,
        error_log: Option<String>,
        // YENİ EKLENEN ALANLAR:
        retry_count: u32,
        original_prompt: String,
    },

    // --- RUNTIME (WASM ÇALIŞTIRMA) ---
    
    RuntimeExecuteRequest {
        req_id: u64,
        wasm_path: String,
    },

    RuntimeExecuteResult {
        req_id: u64,
        stdout: String,
        stderr: Option<String>,
    },

    // --- SİSTEM YÖNETİMİ ---

    DashboardUpdate {
        active_jobs: usize,
        gas_used_last_tick: u64,
        sentinel_noise: u64,
        last_log: String,
        system_status: SystemState,
    },
    
    /// Token stream for TUI display (chunk-by-chunk generation)
    /// Only carries the NEW chunk — consumers accumulate locally.
    TokenStream {
        token: String,
    },
    
    // --- CORTEX ACTOR (Neural Model Handler) ---
    
    /// Status updates from CortexActor to Kernel
    CortexStatus {
        message: String,
        state: String,
    },

    /// Emitted by the memory monitor when RAM usage crosses 85% of the configured limit.
    /// CortexActor should log this and optionally throttle ongoing generation.
    MemoryPressure {
        used_gb:  f64,
        limit_gb: f64,
        pct:      f64,
    },

    // --- BRAINFLOW BIOSIGNAL INTEGRATION ---
    
    /// Real-time biosignal batch from BrainFlow hardware
    /// Contains EEG/EMG data with cognitive state classification
    BiosignalBatch {
        timestamp_ms: u64,  // Milliseconds since epoch for serde compatibility
        cognitive_state: String,
        confidence: f32,
        alpha_power: f32,
        beta_power: f32,
        theta_power: f32,
        gamma_power: f32,
        delta_power: f32,
    },

    Shutdown,
}

// Manual Clone implementation for SystemMsg (response_tx cannot be cloned)
impl Clone for SystemMsg {
    fn clone(&self) -> Self {
        match self {
            SystemMsg::LlmRequest { req_id, user_id, prompt, image_data, .. } => {
                SystemMsg::LlmRequest {
                    req_id: *req_id,
                    user_id: user_id.clone(),
                    prompt: prompt.clone(),
                    image_data: image_data.clone(),
                    response_tx: None, // Cannot clone oneshot::Sender
                }
            }
            SystemMsg::CortexResponse { req_id, user_id, content } => {
                SystemMsg::CortexResponse {
                    req_id: *req_id,
                    user_id: user_id.clone(),
                    content: content.clone(),
                }
            }
            SystemMsg::MemorizeRequest { user_id, text, vector } => {
                SystemMsg::MemorizeRequest {
                    user_id: user_id.clone(),
                    text: text.clone(),
                    vector: vector.clone(),
                }
            }
            SystemMsg::VectorSearchRequest { request_id, user_id, query_vector, top_k } => {
                SystemMsg::VectorSearchRequest {
                    request_id: *request_id,
                    user_id: user_id.clone(),
                    query_vector: query_vector.clone(),
                    top_k: *top_k,
                }
            }
            SystemMsg::VectorSearchResult { request_id, results } => {
                SystemMsg::VectorSearchResult {
                    request_id: *request_id,
                    results: results.clone(),
                }
            }
            SystemMsg::GetAllMemoriesRequest { req_id, user_id } => {
                SystemMsg::GetAllMemoriesRequest {
                    req_id: *req_id,
                    user_id: user_id.clone(),
                }
            }
            SystemMsg::GetAllMemoriesResponse { req_id, memories } => {
                SystemMsg::GetAllMemoriesResponse {
                    req_id: *req_id,
                    memories: memories.clone(),
                }
            }
            SystemMsg::PruneMemoriesRequest { req_id, retention_days } => {
                SystemMsg::PruneMemoriesRequest {
                    req_id: *req_id,
                    retention_days: *retention_days,
                }
            }
            SystemMsg::PruneMemoriesResult { req_id, deleted_count, details } => {
                SystemMsg::PruneMemoriesResult {
                    req_id: *req_id,
                    deleted_count: *deleted_count,
                    details: details.clone(),
                }
            }
            SystemMsg::SanitizeVectorRequest { req_id, raw_vector } => {
                SystemMsg::SanitizeVectorRequest {
                    req_id: *req_id,
                    raw_vector: raw_vector.clone(),
                }
            }
            SystemMsg::SanitizeVectorResponse { req_id, safe_vector, noise_level } => {
                SystemMsg::SanitizeVectorResponse {
                    req_id: *req_id,
                    safe_vector: safe_vector.clone(),
                    noise_level: *noise_level,
                }
            }
            SystemMsg::ForgeCompileRequest { req_id, source_code, module_name, retry_count, original_prompt } => {
                SystemMsg::ForgeCompileRequest {
                    req_id: *req_id,
                    source_code: source_code.clone(),
                    module_name: module_name.clone(),
                    retry_count: *retry_count,
                    original_prompt: original_prompt.clone(),
                }
            }
            SystemMsg::ForgeCompilationResult { req_id, success, wasm_path, error_log, retry_count, original_prompt } => {
                SystemMsg::ForgeCompilationResult {
                    req_id: *req_id,
                    success: *success,
                    wasm_path: wasm_path.clone(),
                    error_log: error_log.clone(),
                    retry_count: *retry_count,
                    original_prompt: original_prompt.clone(),
                }
            }
            SystemMsg::RuntimeExecuteRequest { req_id, wasm_path } => {
                SystemMsg::RuntimeExecuteRequest {
                    req_id: *req_id,
                    wasm_path: wasm_path.clone(),
                }
            }
            SystemMsg::RuntimeExecuteResult { req_id, stdout, stderr } => {
                SystemMsg::RuntimeExecuteResult {
                    req_id: *req_id,
                    stdout: stdout.clone(),
                    stderr: stderr.clone(),
                }
            }
            SystemMsg::DashboardUpdate { active_jobs, gas_used_last_tick, sentinel_noise, last_log, system_status } => {
                SystemMsg::DashboardUpdate {
                    active_jobs: *active_jobs,
                    gas_used_last_tick: *gas_used_last_tick,
                    sentinel_noise: *sentinel_noise,
                    last_log: last_log.clone(),
                    system_status: system_status.clone(),
                }
            }
            SystemMsg::TokenStream { token } => {
                SystemMsg::TokenStream {
                    token: token.clone(),
                }
            }
            SystemMsg::CortexStatus { message, state } => {
                SystemMsg::CortexStatus {
                    message: message.clone(),
                    state: state.clone(),
                }
            }
            SystemMsg::MemoryPressure { used_gb, limit_gb, pct } => {
                SystemMsg::MemoryPressure {
                    used_gb: *used_gb,
                    limit_gb: *limit_gb,
                    pct: *pct,
                }
            }
            SystemMsg::BiosignalBatch { timestamp_ms, cognitive_state, confidence, alpha_power, beta_power, theta_power, gamma_power, delta_power } => {
                SystemMsg::BiosignalBatch {
                    timestamp_ms: *timestamp_ms,
                    cognitive_state: cognitive_state.clone(),
                    confidence: *confidence,
                    alpha_power: *alpha_power,
                    beta_power: *beta_power,
                    theta_power: *theta_power,
                    gamma_power: *gamma_power,
                    delta_power: *delta_power,
                }
            }
            SystemMsg::Shutdown => SystemMsg::Shutdown,
        }
    }
}