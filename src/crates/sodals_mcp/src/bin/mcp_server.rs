// src/crates/sodals_mcp/src/bin/mcp_server.rs
// ═══════════════════════════════════════════════════════════════════════
// SODALS MCP Server — standalone binary
// ═══════════════════════════════════════════════════════════════════════
//
// Usage:
//   sodals-mcp-server --transport stdio   (for Claude Desktop / Cursor)
//   sodals-mcp-server --transport sse --port 3001
//
// Claude Desktop config (mcp_config.json):
//   {
//     "mcpServers": {
//       "sodals": {
//         "command": "sodals-mcp-server",
//         "args": ["--transport", "stdio"]
//       }
//     }
//   }

use anyhow::Result;
use clap::Parser;
use tokio::sync::{broadcast, mpsc};
use tracing::info;
use tracing_subscriber::EnvFilter;

use sodals_mcp::server::McpServer;
use sodals_mcp::transport::{StdioTransport, SseTransportBridge, SseTransportHandle};
use sodals_mcp::types::JsonRpcMessage;

// ============================================================================
// CLI ARGUMENTS
// ============================================================================

#[derive(Parser, Debug)]
#[command(
    name = "sodals-mcp-server",
    about = "SODALS Model Context Protocol (MCP) Server",
    version
)]
struct Args {
    /// Transport mode: "stdio" for subprocess, "sse" for HTTP/SSE
    #[arg(short, long, default_value = "stdio")]
    transport: String,

    /// Port for SSE transport (ignored for stdio)
    #[arg(short, long, default_value = "3001")]
    port: u16,

    /// Kernel channel buffer size
    #[arg(long, default_value = "256")]
    buffer_size: usize,
}

// ============================================================================
// MAIN
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // --- Initialize tracing ---
    // For stdio transport, we log to stderr to keep stdout clean for JSON-RPC
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("sodals_mcp=info"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();

    info!("═══════════════════════════════════════════════════════");
    info!("  SODALS MCP Server v{}", env!("CARGO_PKG_VERSION"));
    info!("  Transport: {}", args.transport);
    info!("═══════════════════════════════════════════════════════");

    // --- Create kernel channels ---
    // In a real deployment, these would connect to the actual SODALS kernel.
    // For standalone mode, we create local channels.
    let (kernel_tx, mut kernel_rx) = mpsc::channel(args.buffer_size);
    let (broadcast_tx, _) = broadcast::channel(args.buffer_size);
    let broadcast_tx_clone = broadcast_tx.clone();

    // Spawn a simple kernel echo handler for standalone mode
    tokio::spawn(async move {
        use sodals_net::protocol::SystemMsg;

        while let Some(msg) = kernel_rx.recv().await {
            match msg {
                SystemMsg::LlmRequest {
                    req_id,
                    user_id,
                    prompt,
                    ..
                } => {
                    // Echo response for standalone mode
                    let response = SystemMsg::CortexResponse {
                        req_id,
                        user_id,
                        content: format!(
                            "[SODALS MCP Standalone] Received prompt ({} chars): {}",
                            prompt.len(),
                            if prompt.len() > 100 {
                                format!("{}...", &prompt[..100])
                            } else {
                                prompt
                            }
                        ),
                    };
                    let _ = broadcast_tx_clone.send(response);
                }
                SystemMsg::VectorSearchRequest {
                    request_id,
                    ..
                } => {
                    let response = SystemMsg::VectorSearchResult {
                        request_id,
                        results: vec![
                            "Standalone mode: no vector database connected.".to_string(),
                        ],
                    };
                    let _ = broadcast_tx_clone.send(response);
                }
                SystemMsg::ForgeCompileRequest {
                    req_id,
                    module_name,
                    ..
                } => {
                    let response = SystemMsg::ForgeCompilationResult {
                        req_id,
                        success: false,
                        wasm_path: String::new(),
                        error_log: Some(format!(
                            "Standalone mode: cannot compile module '{}'.",
                            module_name
                        )),
                        retry_count: 0,
                        original_prompt: String::new(),
                    };
                    let _ = broadcast_tx_clone.send(response);
                }
                SystemMsg::RuntimeExecuteRequest { req_id, .. } => {
                    let response = SystemMsg::RuntimeExecuteResult {
                        req_id,
                        stdout: "Standalone mode: no WASM runtime available.".to_string(),
                        stderr: None,
                    };
                    let _ = broadcast_tx_clone.send(response);
                }
                _ => {
                    // Other messages are fire-and-forget or not handled in standalone
                }
            }
        }
    });

    // --- Start MCP server ---
    let mut server = McpServer::new(kernel_tx, broadcast_tx);

    match args.transport.as_str() {
        "stdio" => {
            info!("[MCP] Starting stdio transport — reading JSON-RPC from stdin");
            let mut transport = StdioTransport::new();
            server.run(&mut transport).await?;
        }
        "sse" => {
            info!(
                "[MCP] Starting SSE transport on port {}",
                args.port
            );

            let (mut bridge, handle) = SseTransportBridge::new(args.buffer_size);

            // Start Axum server for SSE endpoints
            let sse_handle = handle.clone();
            let port = args.port;
            tokio::spawn(async move {
                if let Err(e) = run_sse_server(sse_handle, port).await {
                    tracing::error!("[MCP] SSE server error: {}", e);
                }
            });

            // Run MCP server over the SSE bridge
            server.run(&mut bridge).await?;
        }
        other => {
            return Err(anyhow::anyhow!(
                "Unknown transport: '{}'. Use 'stdio' or 'sse'.",
                other
            ));
        }
    }

    Ok(())
}

// ============================================================================
// SSE HTTP SERVER
// ============================================================================

async fn run_sse_server(
    handle: SseTransportHandle,
    port: u16,
) -> Result<()> {
    use axum::routing::{get, post};
    use axum::Router;
    use std::sync::Arc;

    // The Axum state holds the SSE transport handle
    let state = Arc::new(handle);

    let app = Router::new()
        .route("/mcp/sse", get(sse_handler))
        .route("/mcp/message", post(message_handler))
        .layer(
            tower_http::cors::CorsLayer::permissive(),
        )
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    info!("[MCP/SSE] HTTP server listening on http://{}", addr);
    info!("[MCP/SSE] Endpoints:");
    info!("[MCP/SSE]   GET  /mcp/sse     — SSE stream (server → client)");
    info!("[MCP/SSE]   POST /mcp/message — JSON-RPC messages (client → server)");

    axum::serve(listener, app).await?;
    Ok(())
}

/// SSE stream endpoint — server pushes JSON-RPC responses/notifications here
async fn sse_handler(
    axum::extract::State(state): axum::extract::State<
        std::sync::Arc<SseTransportHandle>,
    >,
) -> axum::response::sse::Sse<impl futures::stream::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>>
{
    use axum::response::sse::{Event, KeepAlive};

    let rx = state.outgoing_rx.clone();

    let stream = async_stream::stream! {
        let mut rx: tokio::sync::MutexGuard<'_, tokio::sync::mpsc::Receiver<JsonRpcMessage>> = rx.lock().await;
        loop {
            match rx.recv().await {
                Some(msg) => {
                    if let Ok(json) = serde_json::to_string(&msg) {
                        yield Ok(Event::default().data(json));
                    }
                }
                None => break,
            }
        }
    };

    axum::response::sse::Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("ping"),
    )
}

/// POST message endpoint — client sends JSON-RPC requests here
async fn message_handler(
    axum::extract::State(state): axum::extract::State<
        std::sync::Arc<SseTransportHandle>,
    >,
    axum::Json(msg): axum::Json<JsonRpcMessage>,
) -> axum::http::StatusCode {
    match state.incoming_tx.send(msg).await {
        Ok(_) => axum::http::StatusCode::ACCEPTED,
        Err(_) => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
    }
}
