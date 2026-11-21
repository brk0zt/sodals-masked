// src/crates/sodals_net/src/server.rs
// REST API Gateway for SODALS Kernel (Edge AI Service)
// 
// Architecture:
// 1. Accepts JSON requests via HTTP
// 2. Forwards to Kernel via mpsc channel
// 3. Subscribes to broadcast channel for response
// 4. Filters by req_id and returns result within timeout
// 5. Validates requests via X-API-KEY header

use axum::{
    extract::{State, DefaultBodyLimit},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
// use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};
use tokio::time::timeout;
use tracing::{info, warn, error};
// use uuid::Uuid;
use rand::Rng;
use crate::protocol::SystemMsg;

// ============================================================================
// APPLICATION STATE
// ============================================================================

/// Shared application state holding communication channels to the Kernel
#[derive(Clone)]
pub struct AppState {
    /// Channel to send messages to the Kernel
    pub kernel_tx: mpsc::Sender<SystemMsg>,
    /// Broadcast sender to subscribe for Kernel responses
    pub broadcast_tx: broadcast::Sender<SystemMsg>,
    /// API key for authentication
    pub api_key: String,
    /// Optional cache for faster lookups (future enhancement)
    pub config_debug: bool,
}

// ============================================================================
// REQUEST/RESPONSE DTOs
// ============================================================================

/// Request to solve a prompt (main endpoint)
#[derive(Debug, Deserialize, Serialize)]
pub struct SolveRequest {
    /// The prompt/query to process
    pub prompt: String,
    /// If true, triggers special high-stakes inference (e.g., "SOLVE:" prefix)
    #[serde(default)]
    pub high_stakes: bool,
    /// Optional user ID for tracking
    #[serde(default)]
    pub user_id: Option<String>,
    /// Optional image data (base64 encoded)
    #[serde(default)]
    pub image_data: Option<String>,
}

/// Response from solve endpoint
#[derive(Debug, Serialize)]
pub struct SolveResponse {
    /// Unique request identifier
    pub id: String,
    /// The computed response
    pub response: String,
    /// Processing time in seconds
    pub processing_time_secs: f64,
    /// Energy used (estimate)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy: Option<f64>,
    /// Model used for inference
    #[serde(default)]
    pub model: String,
}

/// Request to memorize/store text in vector database
#[derive(Debug, Deserialize, Serialize)]
pub struct MemorizeRequest {
    /// Text to store
    pub text: String,
    /// Optional user ID for scoped storage
    #[serde(default)]
    pub user_id: Option<String>,
    /// Optional category/tag
    #[serde(default)]
    pub category: Option<String>,
}

/// Response from memorize endpoint
#[derive(Debug, Serialize)]
pub struct MemorizeResponse {
    /// Request ID
    pub id: String,
    /// Status message
    pub status: String,
    /// Vector dimension (for client awareness)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_dim: Option<usize>,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Service status
    pub status: String,
    /// Version
    pub version: String,
    /// Timestamp
    pub timestamp: String,
}

/// Error response wrapper
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    /// Error code (HTTP-like)
    pub code: u16,
    /// Error message
    pub message: String,
    /// Request ID if available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Timestamp
    pub timestamp: String,
}

// ============================================================================
// MIDDLEWARE & SECURITY
// ============================================================================

/// API Key validation middleware
pub async fn validate_api_key(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // Allow health check without authentication
    if request.uri().path() == "/health" {
        return Ok(next.run(request).await);
    }

    let api_key = headers
        .get("X-API-KEY")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");

    if api_key != state.api_key {
        warn!("Unauthorized API request from {:?}", request.headers().get("user-agent"));
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                code: 401,
                message: "Invalid or missing X-API-KEY header".to_string(),
                request_id: None,
                timestamp: chrono::Utc::now().to_rfc3339(),
            }),
        ));
    }

    Ok(next.run(request).await)
}

// ============================================================================
// HANDLERS
// ============================================================================

/// GET /health - Liveness probe for load balancers
pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "operational".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

/// POST /v1/solve - Main inference endpoint
pub async fn solve_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SolveRequest>,
) -> Result<Json<SolveResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();
    let req_id: u64 = rand::thread_rng().gen();
    let user_id = payload.user_id.unwrap_or_else(|| "anonymous".to_string());

    info!(
        "[SOLVE] 🎯 Request: req_id={}, prompt_len={}, high_stakes={}",
        req_id,
        payload.prompt.len(),
        payload.high_stakes
    );

    // Validate input
    if payload.prompt.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                code: 400,
                message: "Prompt cannot be empty".to_string(),
                request_id: Some(req_id.to_string()),
                timestamp: chrono::Utc::now().to_rfc3339(),
            }),
        ));
    }

    // Optionally prepend high-stakes prefix
    let prompt = if payload.high_stakes {
        format!("SOLVE: {}", payload.prompt)
    } else {
        payload.prompt
    };

    // Subscribe BEFORE sending to kernel_tx so fast Kernel responses can never race past us!
    let mut rx = state.broadcast_tx.subscribe();

    // Send request to Kernel
    let msg = SystemMsg::LlmRequest {
        req_id,
        user_id,
        prompt,
        image_data: payload.image_data,
        response_tx: None,
    };

    if let Err(e) = state.kernel_tx.send(msg).await {
        error!("[SOLVE] ❌ Failed to send message to Kernel: {}", e);
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                code: 503,
                message: "Kernel service unavailable".to_string(),
                request_id: Some(req_id.to_string()),
                timestamp: chrono::Utc::now().to_rfc3339(),
            }),
        ));
    }

    // Timeout: 30 seconds to receive response from Kernel
    match timeout(Duration::from_secs(30), wait_for_response(req_id, &mut rx)).await {
        Ok(Ok(response_content)) => {
            let elapsed = start.elapsed().as_secs_f64();
            info!("[SOLVE] ✅ Response: req_id={}, time={}s", req_id, elapsed);

            Ok(Json(SolveResponse {
                id: req_id.to_string(),
                response: response_content,
                processing_time_secs: elapsed,
                energy: Some(elapsed * 0.01), // Rough estimate
                model: "sodals-kernel".to_string(),
            }))
        }
        Ok(Err(e)) => {
            error!("[SOLVE] ❌ Error waiting for response: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    code: 500,
                    message: format!("Kernel error: {}", e),
                    request_id: Some(req_id.to_string()),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                }),
            ))
        }
        Err(_) => {
            error!("[SOLVE] ⏱️  Request timeout: req_id={}", req_id);
            Err((
                StatusCode::GATEWAY_TIMEOUT,
                Json(ErrorResponse {
                    code: 504,
                    message: "Kernel did not respond within 30 seconds".to_string(),
                    request_id: Some(req_id.to_string()),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                }),
            ))
        }
    }
}

/// POST /v1/memorize - Store text in vector database
pub async fn memorize_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<MemorizeRequest>,
) -> Result<Json<MemorizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let req_id: u64 = rand::thread_rng().gen();
    let user_id = payload.user_id.unwrap_or_else(|| "anonymous".to_string());

    info!(
        "[MEMORIZE] 💾 Request: req_id={}, text_len={}, user={}",
        req_id,
        payload.text.len(),
        user_id
    );

    // Validate input
    if payload.text.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                code: 400,
                message: "Text cannot be empty".to_string(),
                request_id: Some(req_id.to_string()),
                timestamp: chrono::Utc::now().to_rfc3339(),
            }),
        ));
    }

    // For now, use a zero vector (later: compute embedding)
    // In production, you'd call an embedding model here
    let vector = vec![0.0; 768]; // Standard embedding dimension

    // Create memorize request
    let msg = SystemMsg::MemorizeRequest {
        user_id,
        text: payload.text,
        vector,
    };

    // Send to Kernel
    if let Err(e) = state.kernel_tx.send(msg).await {
        error!("[MEMORIZE] ❌ Failed to send: {}", e);
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                code: 503,
                message: "Kernel service unavailable".to_string(),
                request_id: Some(req_id.to_string()),
                timestamp: chrono::Utc::now().to_rfc3339(),
            }),
        ));
    }

    info!("[MEMORIZE] ✅ Stored: req_id={}", req_id);

    Ok(Json(MemorizeResponse {
        id: req_id.to_string(),
        status: "stored".to_string(),
        vector_dim: Some(768),
    }))
}

// ============================================================================
// HELPERS
// ============================================================================

/// Wait for a response from the broadcast channel matching the given req_id
/// 
/// This function:
/// 1. Subscribes to the broadcast channel
/// 2. Loops through messages until it finds one with matching req_id
/// 3. Extracts the content and returns it
/// 4. Handles errors gracefully
async fn wait_for_response(
    req_id: u64,
    rx: &mut broadcast::Receiver<SystemMsg>,
) -> Result<String, String> {
    loop {
        match rx.recv().await {
            Ok(msg) => {
                match msg {
                    SystemMsg::CortexResponse {
                        req_id: msg_id,
                        content,
                        ..
                    } => {
                        if msg_id == req_id {
                            return Ok(content);
                        }
                    }
                    _ => {
                        // Ignore other message types, keep waiting
                    }
                }
            }
            Err(broadcast::error::RecvError::Lagged(_)) => {
                return Err("Receiver lagged, message queue overflow".to_string());
            }
            Err(broadcast::error::RecvError::Closed) => {
                return Err("Broadcast channel closed".to_string());
            }
        }
    }
}

// ============================================================================
// SERVER INITIALIZATION
// ============================================================================

/// Starts the REST API server
/// 
/// # Arguments
/// * `kernel_tx` - Sender channel to the Kernel actor
/// * `broadcast_tx` - Broadcast sender for Kernel responses
/// * `port` - Port to listen on (default: 8080)
pub async fn run_server(
    kernel_tx: mpsc::Sender<SystemMsg>,
    broadcast_tx: broadcast::Sender<SystemMsg>,
    port: u16,
) {
    // Load environment variables from .env file
    let _ = dotenvy::dotenv();
    
    // Load API key from environment
    // Priority: SODALS_API_KEY > SODALS_SECRET_KEY > error
    let api_key = std::env::var("SODALS_API_KEY")
        .or_else(|_| std::env::var("SODALS_SECRET_KEY"))
        .unwrap_or_else(|e| {
            error!("[SERVER] ❌ API key not configured: {}", e);
            error!("[SERVER] 💡 Set SODALS_API_KEY in .env file or environment");
            panic!("API key configuration required for server startup");
        });

    let state = Arc::new(AppState {
        kernel_tx,
        broadcast_tx,
        api_key,
        config_debug: false,
    });

    // Build router with middleware
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/solve", post(solve_handler))
        .route("/v1/memorize", post(memorize_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            validate_api_key,
        ))
        .layer(DefaultBodyLimit::max(1024 * 1024 * 10)) // 10MB max request size
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            error!("[SERVER] ❌ Failed to bind to {}: {}", addr, e);
            return;
        }
    };

    info!("[SERVER] 🚀 API Gateway listening on http://{}", addr);
    info!("[SERVER] 📋 Endpoints:");
    info!("[SERVER]   GET  /health           - Health check");
    info!("[SERVER]   POST /v1/solve         - Inference (requires X-API-KEY)");
    info!("[SERVER]   POST /v1/memorize      - Store memory (requires X-API-KEY)");

    if let Err(e) = axum::serve(listener, app).await {
        error!("[SERVER] ❌ Server error: {}", e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_request_deserialization() {
        let json = r#"{"prompt": "What is 2+2?", "high_stakes": false}"#;
        let req: SolveRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "What is 2+2?");
        assert!(!req.high_stakes);
    }

    #[test]
    fn test_solve_response_serialization() {
        let resp = SolveResponse {
            id: "test-id".to_string(),
            response: "The answer is 4".to_string(),
            processing_time_secs: 1.234,
            energy: Some(0.01234),
            model: "sodals-kernel".to_string(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("test-id"));
        assert!(json.contains("1.234"));
    }

    #[test]
    fn test_memorize_request_validation() {
        let json = r#"{"text": "Important memory"}"#;
        let req: MemorizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.text, "Important memory");
        assert_eq!(req.user_id, None);
    }
}
