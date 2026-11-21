// src/crates/sodals_net/src/bin/openai_server.rs
// OpenAI-compatible chat completions server backed by the SODALS LlmEngine.
//   GET  /v1/models            -> model list ("Burak 0.1")
//   POST /v1/chat/completions  -> chat completions (stream + non-stream)
//
// Run:
//   cargo run -p sodals_net --bin openai_server
// Then point any OpenAI client at http://127.0.0.1:PORT/v1 (e.g. Open WebUI,
// Python openai SDK with base_url, or curl).

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, IntoResponse, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::stream::{self, Stream};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogLevel, LogManager};
use sodals_neuro::engine::generation_loop::generate_stream;
use sodals_neuro::engine::llm_engine::LlmEngine;
use sodals_neuro::engine::chimera_router::NeuroConfig;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::Mutex;
use tokio::sync::mpsc;
use tracing::info;

const MODEL_ID: &str = "Burak 0.1";

struct AppState {
    engine: Arc<Mutex<LlmEngine>>,
}

// ============================================================================
// OpenAI request/response types
// ============================================================================

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
}

#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Serialize, Default)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ChatCompletion {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    index: usize,
    message: ChatMessageOut,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChatMessageOut {
    role: String,
    content: String,
}

// ============================================================================
// Handlers
// ============================================================================

async fn list_models() -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelEntry {
            id: MODEL_ID.to_string(),
            object: "model",
            created: 0,
            owned_by: "sodals",
        }],
    })
}

fn build_prompt(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        match m.role.as_str() {
            "system" => {
                out.push_str(&format!("<|im_start|>system\n{}\n<|im_end|>\n", m.content));
            }
            "assistant" => {
                out.push_str(&format!(
                    "<|im_start|>assistant\n{}\n<|im_end|>\n",
                    m.content
                ));
            }
            _ => {
                out.push_str(&format!("<|im_start|>user\n{}\n<|im_end|>\n", m.content));
            }
        }
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

fn build_config(req: &ChatCompletionRequest) -> NeuroConfig {
    let mut cfg = NeuroConfig::default();
    cfg.euler_resonance = true;
    cfg.physics_enabled = true;
    cfg.strict_determinism = false;
    cfg.quiet = true;
    if let Some(t) = req.temperature {
        cfg.temperature = t;
    }
    if let Some(p) = req.top_p {
        cfg.top_p = p;
    }
    if let Some(k) = req.top_k {
        cfg.top_k = k;
    }
    cfg
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let max_tokens = req.max_tokens.unwrap_or(256);
    if req.stream {
        let (tx, rx) = mpsc::channel::<String>(64);
        let engine = state.engine.clone();
        let cfg = build_config(&req);
        let prompt = build_prompt(&req.messages);
        std::thread::spawn(move || {
            let mut engine = engine.lock().unwrap();
            let _ = generate_stream(
                &mut engine,
                &prompt,
                None,
                None,
                move |t: String| {
                    let _ = tx.blocking_send(t);
                },
                &cfg,
                0,
                max_tokens,
            );
        });
        let stream = stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|chunk| {
                let json = serde_json::to_string(&ChatCompletionChunk {
                    id: "chatcmpl-sodals".to_string(),
                    object: "chat.completion.chunk",
                    created: 0,
                    model: MODEL_ID.to_string(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: Some("assistant".to_string()),
                            content: Some(chunk),
                        },
                        finish_reason: None,
                    }],
                })
                .unwrap_or_default();
                let ev = Event::default().data(json);
                (Ok::<Event, Infallible>(ev), rx)
            })
        });
        let done = Event::default().data("[DONE]");
        Sse::new(stream.chain(stream::once(async move { Ok::<Event, Infallible>(done) })))
            .into_response()
    } else {
        let engine = state.engine.clone();
        let cfg = build_config(&req);
        let prompt = build_prompt(&req.messages);
        let buf = Arc::new(Mutex::new(String::new()));
        let buf_closure = buf.clone();
        let full = tokio::task::spawn_blocking(move || {
            let mut engine = engine.lock().unwrap();
            generate_stream(
                &mut engine,
                &prompt,
                None,
                None,
                move |t: String| {
                    if let Ok(mut b) = buf_closure.lock() {
                        b.push_str(&t);
                    }
                },
                &cfg,
                0,
                max_tokens,
            )
            .map_err(|e| format!("{:?}", e))
        })
        .await
        .unwrap_or_else(|_| Err("generation task panicked".to_string()));
        if let Err(e) = full {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": e, "type": "sodals_error", "code": 500}})),
            )
                .into_response();
        }
        let content = buf.lock().unwrap().clone();
        let resp = ChatCompletion {
            id: "chatcmpl-sodals".to_string(),
            object: "chat.completion",
            created: 0,
            model: MODEL_ID.to_string(),
            choices: vec![CompletionChoice {
                index: 0,
                message: ChatMessageOut {
                    role: "assistant".to_string(),
                    content: content.clone(),
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        };
        Json(resp).into_response()
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let sys_config = SystemConfig::load().unwrap_or_else(|_| SystemConfig::default());
    let log_manager = LogManager::new(LogLevel::INFO);
    info!("[OPENAI] Loading LlmEngine (model={}) ...", sys_config.model_path);
    let mut engine = LlmEngine::new(&sys_config, &log_manager)?;
    info!("[OPENAI] Engine ready.");

    // Load QMEM cartridge library for latent-memory steering (best-effort).
    let cartridge_dir = std::env::var("SODALS_QMEM_DIR")
        .unwrap_or_else(|_| "qmem_crystals".to_string());
    if std::path::Path::new(&cartridge_dir).exists() {
        engine.init_cartridge_library(&cartridge_dir, 1.5);
        info!("[OPENAI] QMEM cartridge library loaded from {}", cartridge_dir);
    } else {
        info!("[OPENAI] QMEM cartridge dir not found: {}", cartridge_dir);
    }

    let state = Arc::new(AppState {
        engine: Arc::new(Mutex::new(engine)),
    });

    let port: u16 = std::env::var("SODALS_OPENAI_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(|| async { "ok" }))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state);

    let addr = format!("127.0.0.1:{}", port);
    info!("[OPENAI] Listening on http://{}", addr);
    info!("[OPENAI] Base URL for clients: http://{}/v1", addr);
    info!("[OPENAI] Model id: {}", MODEL_ID);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
