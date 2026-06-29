// Example: Integration with SODALS Kernel
// File: examples/run_api_gateway.rs
// 
// This example demonstrates how to integrate the REST API server with the Kernel actor.
// Run with: cargo run --example run_api_gateway --features "full"

use sodals_net::protocol::SystemMsg;
use sodals_net::server::run_server;
use tokio::sync::{broadcast, mpsc};
use tracing::{error, info};

/// Mock Kernel Actor for demonstration
/// In production, this would be the actual Kernel implementation
async fn mock_kernel_actor(
    mut kernel_rx: mpsc::Receiver<SystemMsg>,
    broadcast_tx: broadcast::Sender<SystemMsg>,
) {
    info!("[KERNEL] 🧠 Mock Kernel started");

    while let Some(msg) = kernel_rx.recv().await {
        match msg {
            SystemMsg::LlmRequest {
                req_id,
                user_id,
                prompt,
                image_data: _,
                ..
            } => {
                info!(
                    "[KERNEL] 📩 LLM Request: req_id={}, user={}, prompt_len={}",
                    req_id,
                    user_id,
                    prompt.len()
                );

                // Simulate processing
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // Generate response
                let response = if prompt.contains("SOLVE:") {
                    "🔬 High-stakes inference result: The answer requires careful analysis."
                        .to_string()
                } else {
                    format!("Regular response to: {}", prompt)
                };

                // Send response via broadcast (all subscribers receive this)
                let response_msg = SystemMsg::CortexResponse {
                    req_id,
                    user_id,
                    content: response,
                };

                if let Err(e) = broadcast_tx.send(response_msg) {
                    error!("[KERNEL] ❌ Failed to send response: {}", e);
                }
            }

            SystemMsg::MemorizeRequest {
                user_id,
                text,
                vector: _,
            } => {
                info!(
                    "[KERNEL] 💾 Memorize Request: user={}, text_len={}",
                    user_id,
                    text.len()
                );

                // In production, this would store to vector DB
                info!("[KERNEL] ✅ Stored in memory database");
            }

            _ => {
                info!("[KERNEL] ⚠️  Received unhandled message type");
            }
        }
    }

    info!("[KERNEL] 🛑 Kernel actor stopped");
}

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_target(true)
        .with_thread_ids(true)
        .with_level(true)
        .init();

    info!("🚀 SODALS API Gateway Example");
    info!("==============================");

    // Create communication channels
    // mpsc: HTTP Handlers → Kernel
    let (kernel_tx, kernel_rx) = mpsc::channel::<SystemMsg>(100);

    // broadcast: Kernel → All subscribers (HTTP handlers)
    let (broadcast_tx, _broadcast_rx) = broadcast::channel::<SystemMsg>(1000);

    // Spawn the mock kernel actor
    let broadcast_tx_for_kernel = broadcast_tx.clone();
    let kernel_handle = tokio::spawn(async move {
        mock_kernel_actor(kernel_rx, broadcast_tx_for_kernel).await;
    });

    // Start the API server
    let server_handle = tokio::spawn(async move {
        run_server(kernel_tx, broadcast_tx, 8080).await;
    });

    info!("✨ Service initialized");
    info!("");
    info!("📝 Test the server with these commands:");
    info!("");
    info!("1. Health check (no auth required):");
    info!("   curl -X GET http://localhost:8080/health");
    info!("");
    info!("2. Solve endpoint (requires X-API-KEY header):");
    info!("   curl -X POST http://localhost:8080/v1/solve \\");
    info!("     -H 'X-API-KEY: sodals-secret-key' \\");
    info!("     -H 'Content-Type: application/json' \\");
    info!("     -d '{{");
    info!("       \"prompt\": \"What is 2+2?\",");
    info!("       \"high_stakes\": false");
    info!("     }}'");
    info!("");
    info!("3. High-stakes inference:");
    info!("   curl -X POST http://localhost:8080/v1/solve \\");
    info!("     -H 'X-API-KEY: sodals-secret-key' \\");
    info!("     -H 'Content-Type: application/json' \\");
    info!("     -d '{{");
    info!("       \"prompt\": \"Complex mathematical proof needed\",");
    info!("       \"high_stakes\": true");
    info!("     }}'");
    info!("");
    info!("4. Memorize endpoint:");
    info!("   curl -X POST http://localhost:8080/v1/memorize \\");
    info!("     -H 'X-API-KEY: sodals-secret-key' \\");
    info!("     -H 'Content-Type: application/json' \\");
    info!("     -d '{{");
    info!("       \"text\": \"Important fact to remember\",");
    info!("       \"user_id\": \"demo_user\"");
    info!("     }}'");
    info!("");
    info!("5. Test unauthorized access (should fail):");
    info!("   curl -X POST http://localhost:8080/v1/solve \\");
    info!("     -H 'Content-Type: application/json' \\");
    info!("     -d '{{ \"prompt\": \"test\" }}'");
    info!("");
    info!("Press Ctrl+C to stop");

    // Wait for both handles (they run indefinitely)
    let _ = tokio::join!(kernel_handle, server_handle);
}
