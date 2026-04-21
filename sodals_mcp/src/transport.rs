// src/crates/sodals_mcp/src/transport.rs
// ═══════════════════════════════════════════════════════════════════════
// MCP Transport Layer — stdio and HTTP/SSE
// ═══════════════════════════════════════════════════════════════════════

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;
use tracing::{debug, trace};

use crate::types::JsonRpcMessage;

// ============================================================================
// TRANSPORT TRAIT
// ============================================================================

/// Abstraction over the wire transport used by MCP.
///
/// Implementations handle serialization framing but NOT protocol semantics.
#[async_trait]
pub trait McpTransport: Send + Sync + 'static {
    /// Read the next JSON-RPC message from the transport.
    /// Returns `None` when the transport is closed / EOF.
    async fn recv(&mut self) -> Result<Option<JsonRpcMessage>>;

    /// Write a JSON-RPC message to the transport.
    async fn send(&mut self, msg: &JsonRpcMessage) -> Result<()>;

    /// Gracefully close the transport.
    async fn close(&mut self) -> Result<()>;
}

// ============================================================================
// STDIO TRANSPORT
// ============================================================================

/// Newline-delimited JSON over stdin/stdout.
///
/// This is the canonical transport for local MCP servers launched as
/// subprocesses (e.g. from Claude Desktop, Cursor, VS Code).
///
/// Framing: one JSON object per line, terminated by `\n`.
/// Stderr is reserved for diagnostic logging.
pub struct StdioTransport {
    reader: BufReader<tokio::io::Stdin>,
    writer: tokio::io::Stdout,
    closed: bool,
}

impl StdioTransport {
    pub fn new() -> Self {
        Self {
            reader: BufReader::new(tokio::io::stdin()),
            writer: tokio::io::stdout(),
            closed: false,
        }
    }
}

impl Default for StdioTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn recv(&mut self) -> Result<Option<JsonRpcMessage>> {
        if self.closed {
            return Ok(None);
        }

        let mut line = String::new();
        let bytes_read = self
            .reader
            .read_line(&mut line)
            .await
            .context("Failed to read from stdin")?;

        if bytes_read == 0 {
            debug!("[MCP/stdio] EOF on stdin — transport closed");
            self.closed = true;
            return Ok(None);
        }

        let line = line.trim();
        if line.is_empty() {
            // Skip blank lines (keepalive / padding)
            return self.recv().await;
        }

        trace!("[MCP/stdio] ← {}", line);

        let msg: JsonRpcMessage =
            serde_json::from_str(line).context("Failed to parse JSON-RPC message from stdin")?;

        Ok(Some(msg))
    }

    async fn send(&mut self, msg: &JsonRpcMessage) -> Result<()> {
        if self.closed {
            return Err(anyhow::anyhow!("Transport is closed"));
        }

        let json = serde_json::to_string(msg).context("Failed to serialize JSON-RPC message")?;
        trace!("[MCP/stdio] → {}", json);

        self.writer
            .write_all(json.as_bytes())
            .await
            .context("Failed to write to stdout")?;
        self.writer
            .write_all(b"\n")
            .await
            .context("Failed to write newline")?;
        self.writer
            .flush()
            .await
            .context("Failed to flush stdout")?;

        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed = true;
        self.writer.flush().await?;
        Ok(())
    }
}

// ============================================================================
// CHANNEL TRANSPORT (for in-process / testing)
// ============================================================================

/// Channel-based transport for in-process communication and testing.
///
/// Uses tokio mpsc channels — one for each direction.
/// Useful when the MCP server runs inside the same process as the host.
pub struct ChannelTransport {
    rx: mpsc::Receiver<JsonRpcMessage>,
    tx: mpsc::Sender<JsonRpcMessage>,
    closed: bool,
}

impl ChannelTransport {
    /// Create a pair of channel transports connected to each other.
    /// Returns `(server_side, client_side)`.
    pub fn pair(buffer: usize) -> (Self, Self) {
        let (tx_a, rx_b) = mpsc::channel(buffer);
        let (tx_b, rx_a) = mpsc::channel(buffer);

        let server = ChannelTransport {
            rx: rx_a,
            tx: tx_a,
            closed: false,
        };
        let client = ChannelTransport {
            rx: rx_b,
            tx: tx_b,
            closed: false,
        };

        (server, client)
    }
}

#[async_trait]
impl McpTransport for ChannelTransport {
    async fn recv(&mut self) -> Result<Option<JsonRpcMessage>> {
        if self.closed {
            return Ok(None);
        }
        match self.rx.recv().await {
            Some(msg) => Ok(Some(msg)),
            None => {
                self.closed = true;
                Ok(None)
            }
        }
    }

    async fn send(&mut self, msg: &JsonRpcMessage) -> Result<()> {
        if self.closed {
            return Err(anyhow::anyhow!("Transport is closed"));
        }
        self.tx
            .send(msg.clone())
            .await
            .map_err(|_| anyhow::anyhow!("Channel closed"))
    }

    async fn close(&mut self) -> Result<()> {
        self.closed = true;
        Ok(())
    }
}

// ============================================================================
// SSE TRANSPORT (Server-Sent Events)
// ============================================================================

/// SSE Transport state for the Axum-based HTTP/SSE endpoint.
///
/// Architecture:
///   Client → POST /mcp/message → JSON-RPC request → server processes
///   Server → GET  /mcp/sse     → SSE stream ← JSON-RPC responses
///
/// This struct holds the channels that bridge the Axum handlers
/// to the MCP server's event loop.
pub struct SseTransportBridge {
    /// Incoming messages from HTTP POST requests
    pub incoming_rx: mpsc::Receiver<JsonRpcMessage>,
    /// Outgoing messages to be sent as SSE events
    pub outgoing_tx: mpsc::Sender<JsonRpcMessage>,
}

/// Handle held by Axum HTTP handlers to push/pull messages.
#[derive(Clone)]
pub struct SseTransportHandle {
    /// Send incoming POST request to the MCP server
    pub incoming_tx: mpsc::Sender<JsonRpcMessage>,
    /// Receive outgoing SSE events from the MCP server
    pub outgoing_rx: std::sync::Arc<tokio::sync::Mutex<mpsc::Receiver<JsonRpcMessage>>>,
}

impl SseTransportBridge {
    /// Create a new SSE transport bridge.
    /// Returns `(bridge_for_mcp_server, handle_for_axum)`.
    pub fn new(buffer: usize) -> (Self, SseTransportHandle) {
        let (incoming_tx, incoming_rx) = mpsc::channel(buffer);
        let (outgoing_tx, outgoing_rx) = mpsc::channel(buffer);

        let bridge = SseTransportBridge {
            incoming_rx,
            outgoing_tx,
        };

        let handle = SseTransportHandle {
            incoming_tx,
            outgoing_rx: std::sync::Arc::new(tokio::sync::Mutex::new(outgoing_rx)),
        };

        (bridge, handle)
    }
}

#[async_trait]
impl McpTransport for SseTransportBridge {
    async fn recv(&mut self) -> Result<Option<JsonRpcMessage>> {
        match self.incoming_rx.recv().await {
            Some(msg) => Ok(Some(msg)),
            None => Ok(None),
        }
    }

    async fn send(&mut self, msg: &JsonRpcMessage) -> Result<()> {
        self.outgoing_tx
            .send(msg.clone())
            .await
            .map_err(|e| anyhow::anyhow!("SSE outgoing channel closed: {}", e))
    }

    async fn close(&mut self) -> Result<()> {
        // Channels will be dropped when the bridge is dropped
        Ok(())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{JsonRpcRequest, JsonRpcId};

    #[tokio::test]
    async fn test_channel_transport_roundtrip() {
        let (mut server, mut client) = ChannelTransport::pair(16);

        // Client sends a request
        let req = JsonRpcMessage::Request(JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(1),
            method: "ping".to_string(),
            params: None,
        });

        client.send(&req).await.unwrap();

        // Server receives it
        let received = server.recv().await.unwrap().unwrap();
        match received {
            JsonRpcMessage::Request(r) => {
                assert_eq!(r.method, "ping");
                assert_eq!(r.id, JsonRpcId::Number(1));
            }
            _ => panic!("Expected Request"),
        }
    }

    #[tokio::test]
    async fn test_channel_transport_close() {
        let (mut server, mut client) = ChannelTransport::pair(16);

        // Drop the client (close sender side)
        drop(client);

        // Server should get None
        let result = server.recv().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_sse_bridge_roundtrip() {
        let (mut bridge, handle) = SseTransportBridge::new(16);

        // Simulate HTTP POST → MCP server
        let req = JsonRpcMessage::Request(JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(99),
            method: "tools/list".to_string(),
            params: None,
        });

        handle.incoming_tx.send(req).await.unwrap();

        let received = bridge.recv().await.unwrap().unwrap();
        match received {
            JsonRpcMessage::Request(r) => assert_eq!(r.method, "tools/list"),
            _ => panic!("Expected Request"),
        }

        // Simulate MCP server → SSE stream
        let resp = JsonRpcMessage::Response(crate::types::JsonRpcResponse::success(
            JsonRpcId::Number(99),
            serde_json::json!({"tools": []}),
        ));
        bridge.send(&resp).await.unwrap();

        let mut rx = handle.outgoing_rx.lock().await;
        let sent = rx.recv().await.unwrap();
        match sent {
            JsonRpcMessage::Response(r) => {
                assert_eq!(r.id, JsonRpcId::Number(99));
                assert!(r.result.is_some());
            }
            _ => panic!("Expected Response"),
        }
    }
}
