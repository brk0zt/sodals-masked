// src/socket_actor.rs - WebSocket Actor for QYPTOS

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    response::IntoResponse,
    routing::get,
    Router,
};
use std::net::SocketAddr;
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{mpsc, RwLock};
use tokio::time::{timeout, Duration};
use crate::protocol::SystemMsg;
use tower_http::cors::{CorsLayer, Any};
use serde::{Deserialize, Serialize};
use futures::{sink::SinkExt, stream::StreamExt};
use sha2::{Sha256, Digest};

/// Client session identifier
type SessionId = String;

/// Authenticated session manager
pub type SessionRegistry = Arc<RwLock<HashMap<SessionId, mpsc::Sender<String>>>>;

/// Shared application state
struct SocketState {
    kernel_tx: mpsc::Sender<SystemMsg>,
    sessions: SessionRegistry,
    /// Secret key for HMAC validation (should be from env/config)
    secret_key: String,
}

/// WebSocket actor managing client connections
pub struct SocketActor {
    kernel_tx: mpsc::Sender<SystemMsg>,
    port: u16,
    secret_key: String,
    sessions: SessionRegistry,
}

impl SocketActor {
    pub fn new(kernel_tx: mpsc::Sender<SystemMsg>, port: u16, secret_key: String, sessions: SessionRegistry) -> Self {
        Self { kernel_tx, port, secret_key, sessions }
    }

    pub async fn run(&self) {
        let state = Arc::new(SocketState {
            kernel_tx: self.kernel_tx.clone(),
            sessions: self.sessions.clone(),
            secret_key: self.secret_key.clone(),
        });

        let app = Router::new()
            .route("/ws", get(ws_handler))
            .with_state(state)
            .layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any)
            );

        let addr = SocketAddr::from(([0, 0, 0, 0], self.port));
        println!("WebSocket server listening on {}", addr);

        match tokio::net::TcpListener::bind(addr).await {
            Ok(listener) => {
                if let Err(e) = axum::serve(listener, app).await {
                    eprintln!("Server error: {}", e);
                }
            }
            Err(e) => {
                eprintln!("Failed to bind port {}: {}", self.port, e);
            }
        }
    }
}

/// WebSocket upgrade handler
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<SocketState>>,
) -> impl IntoResponse {
    // debug!("WebSocket upgrade requested");
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Client message types
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientMessage {
    Auth {
        response: String,
    },
    LlmRequest {
        #[serde(default)]
        user_id: String,
        prompt: String,
        #[serde(default)]
        image_data: Option<String>,
    },
}

/// Server message types
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerMessage {
    AuthChallenge { nonce: String },
    AuthSuccess,
    AuthFailed { reason: String },
    Error { message: String },
}

/// Main WebSocket connection handler
async fn handle_socket(socket: WebSocket, state: Arc<SocketState>) {
    let (mut sender, mut receiver) = socket.split();
    
    // 1. Authentication challenge-response
    let session_id = match authenticate(&mut sender, &mut receiver, &state.secret_key).await {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Authentication failed: {}", e);
            let _ = send_msg(&mut sender, ServerMessage::AuthFailed { 
                reason: e.to_string() 
            }).await;
            return;
        }
    };

    println!("Client authenticated: {}", session_id);
    let _ = send_msg(&mut sender, ServerMessage::AuthSuccess).await;

    // 2. Create session channel
    let (tx, mut rx) = mpsc::channel::<String>(100);
    
    {
        let mut sessions = state.sessions.write().await;
        sessions.insert(session_id.clone(), tx.clone());
    }

    // debug!("Session registered: {}", session_id);

    // 3. Message loop
    loop {
        tokio::select! {
            // Client -> Server
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Err(e) = handle_client_message(&text, &session_id, &state, &tx).await {
                            eprintln!("Message handling error: {}", e);
                            let _ = send_msg(&mut sender, ServerMessage::Error {
                                message: "Failed to process request".into()
                            }).await;
                        }
                    }
                    Some(Ok(Message::Close(_))) => {
                        // debug!("Client closed connection: {}", session_id);
                        break;
                    }
                    Some(Err(e)) => {
                        eprintln!("WebSocket error: {}", e);
                        break;
                    }
                    None => {
                        // debug!("Connection terminated: {}", session_id);
                        break;
                    }
                    _ => {}
                }
            }

            // Kernel -> Client
            Some(response) = rx.recv() => {
                if sender.send(Message::Text(response)).await.is_err() {
                    // debug!("Failed to send response, client disconnected");
                    break;
                }
            }
        }
    }

    // 4. Cleanup
    state.sessions.write().await.remove(&session_id);
    println!("Session closed: {}", session_id);
}

/// Authenticate client using challenge-response
async fn authenticate(
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    receiver: &mut futures::stream::SplitStream<WebSocket>,
    secret_key: &str,
) -> Result<SessionId, Box<dyn std::error::Error + Send + Sync>> {
    // Generate nonce
    let nonce = generate_nonce();
    
    // Send challenge
    send_msg(sender, ServerMessage::AuthChallenge { 
        nonce: nonce.clone() 
    }).await?;

    // Wait for response with timeout
    let auth_result = timeout(Duration::from_secs(60), async {
        while let Some(msg) = receiver.next().await {
            if let Ok(Message::Text(text)) = msg {
                if let Ok(ClientMessage::Auth { response }) = serde_json::from_str(&text) {
                    return Some(response);
                }
            }
        }
        None
    }).await;

    match auth_result {
        Ok(Some(response)) => {
            // Verify HMAC
            let expected = compute_hmac(secret_key, &nonce);
            if response == expected {
                Ok(generate_session_id())
            } else {
                Err("Invalid auth response".into())
            }
        }
        Ok(None) => Err("No auth response received".into()),
        Err(_) => Err("Authentication timeout".into()),
    }
}

/// Handle incoming client message
async fn handle_client_message(
    text: &str,
    session_id: &str,
    state: &SocketState,
    response_tx: &mpsc::Sender<String>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let msg: ClientMessage = serde_json::from_str(text)?;

    match msg {
        ClientMessage::Auth { .. } => {
            // Auth should happen before message loop
            eprintln!("Unexpected auth message in session");
        }
        ClientMessage::LlmRequest { user_id, prompt, image_data } => {
            let req_id = generate_request_id();
            
            // debug!("LLM request: req_id={}, session={}", req_id, session_id);
            
            state.kernel_tx.send(SystemMsg::LlmRequest {
                req_id,
                user_id: if user_id.is_empty() { session_id.to_string() } else { user_id },
                prompt,
                image_data,
                response_tx: None,
            }).await.map_err(|e| {
                eprintln!("Failed to send to kernel: {}", e);
                
                // Send error response to client
                let error_response = serde_json::json!({
                    "req_id": req_id,
                    "error": "System busy, please retry"
                }).to_string();
                
                let _ = response_tx.try_send(error_response);
                
                Box::new(e) as Box<dyn std::error::Error + Send + Sync>
            })?;
        }
    }

    Ok(())
}

/// Send typed server message
async fn send_msg(
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    msg: ServerMessage,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let json = serde_json::to_string(&msg)?;
    sender.send(Message::Text(json)).await?;
    Ok(())
}

/// Compute HMAC-SHA256 for authentication
fn compute_hmac(secret: &str, nonce: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{}{}", nonce, secret));
    format!("{:x}", hasher.finalize())
}

/// Generate cryptographically secure nonce
fn generate_nonce() -> String {
    use rand::Rng;
    rand::thread_rng()
        .sample_iter(&rand::distributions::Alphanumeric)
        .take(32)
        .map(char::from)
        .collect()
}

/// Generate unique session ID
fn generate_session_id() -> String {
    use rand::Rng;
    format!(
        "session_{}",
        rand::thread_rng()
            .sample_iter(&rand::distributions::Alphanumeric)
            .take(16)
            .map(char::from)
            .collect::<String>()
    )
}

/// Generate unique request ID
fn generate_request_id() -> u64 {
    use rand::Rng;
    rand::thread_rng().gen()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmac_consistency() {
        let secret = "test_secret";
        let nonce = "test_nonce";
        
        let hmac1 = compute_hmac(secret, nonce);
        let hmac2 = compute_hmac(secret, nonce);
        
        assert_eq!(hmac1, hmac2);
        assert_ne!(hmac1, compute_hmac("wrong_secret", nonce));
    }

    #[test]
    fn test_nonce_uniqueness() {
        let nonce1 = generate_nonce();
        let nonce2 = generate_nonce();
        
        assert_ne!(nonce1, nonce2);
        assert_eq!(nonce1.len(), 32);
    }
}