// src/crates/sodals_mcp/src/client.rs
// ═══════════════════════════════════════════════════════════════════════
// MCP Client — consume external MCP tool servers from SODALS
// ═══════════════════════════════════════════════════════════════════════

use anyhow::{Context, Result};
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;
use tracing::{debug, info, warn};

use crate::transport::McpTransport;
use crate::types::*;

// ============================================================================
// MCP CLIENT
// ============================================================================

/// MCP Client for SODALS to consume external tool servers.
///
/// Usage:
/// ```ignore
/// let mut client = McpClient::new();
/// client.connect(transport).await?;
///
/// let tools = client.list_tools().await?;
/// let result = client.call_tool("some_tool", json!({"key": "value"})).await?;
///
/// client.disconnect().await?;
/// ```
pub struct McpClient {
    /// Client identity
    info: ClientInfo,
    /// The transport layer
    transport: Option<Box<dyn McpTransport>>,
    /// Monotonically increasing request ID
    next_id: AtomicU64,
    /// Pending requests awaiting responses (reserved for concurrent request support)
    #[allow(dead_code)]
    pending: Arc<DashMap<u64, oneshot::Sender<JsonRpcResponse>>>,
    /// Server capabilities after initialization
    server_capabilities: Option<ServerCapabilities>,
    /// Server info after initialization
    server_info: Option<ServerInfo>,
    /// Whether connection is alive
    connected: bool,
}

impl McpClient {
    /// Create a new MCP client.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            info: ClientInfo {
                name: name.into(),
                version: version.into(),
            },
            transport: None,
            next_id: AtomicU64::new(1),
            pending: Arc::new(DashMap::new()),
            server_capabilities: None,
            server_info: None,
            connected: false,
        }
    }

    /// Create a default SODALS MCP client.
    pub fn sodals() -> Self {
        Self::new("sodals-mcp-client", env!("CARGO_PKG_VERSION"))
    }

    /// Connect to an MCP server via the given transport.
    /// Performs the `initialize` / `initialized` handshake.
    pub async fn connect(&mut self, transport: Box<dyn McpTransport>) -> Result<()> {
        self.transport = Some(transport);

        info!(
            "[MCP/client] Connecting as {} v{}",
            self.info.name, self.info.version
        );

        // Step 1: Send initialize request
        let init_params = InitializeParams {
            protocol_version: MCP_PROTOCOL_VERSION.to_string(),
            capabilities: ClientCapabilities {
                roots: None,
                sampling: None,
            },
            client_info: self.info.clone(),
        };

        let response = self
            .request(
                "initialize",
                Some(serde_json::to_value(init_params).context("Failed to serialize init params")?),
            )
            .await
            .context("Initialize handshake failed")?;

        // Parse the server's capabilities
        if let Some(result) = response.result {
            let init_result: InitializeResult = serde_json::from_value(result)
                .context("Failed to parse initialize result")?;

            info!(
                "[MCP/client] Connected to {} v{} (protocol: {})",
                init_result.server_info.name,
                init_result.server_info.version,
                init_result.protocol_version,
            );

            if let Some(ref instructions) = init_result.instructions {
                info!("[MCP/client] Server instructions: {}", instructions);
            }

            self.server_capabilities = Some(init_result.capabilities);
            self.server_info = Some(init_result.server_info);
        } else if let Some(err) = response.error {
            return Err(anyhow::anyhow!(
                "Server rejected initialization: {} (code: {})",
                err.message,
                err.code
            ));
        }

        // Step 2: Send initialized notification
        self.notify("initialized", None).await?;
        self.connected = true;

        Ok(())
    }

    /// Disconnect from the MCP server.
    pub async fn disconnect(&mut self) -> Result<()> {
        if let Some(ref mut transport) = self.transport {
            transport.close().await?;
        }
        self.connected = false;
        self.transport = None;
        info!("[MCP/client] Disconnected");
        Ok(())
    }

    /// Check if the client is connected.
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Get the server's capabilities (available after connect).
    pub fn server_capabilities(&self) -> Option<&ServerCapabilities> {
        self.server_capabilities.as_ref()
    }

    /// Get the server's info (available after connect).
    pub fn server_info(&self) -> Option<&ServerInfo> {
        self.server_info.as_ref()
    }

    // ========================================================================
    // TOOLS
    // ========================================================================

    /// List all tools available on the remote server.
    pub async fn list_tools(&mut self) -> Result<Vec<Tool>> {
        let response = self.request("tools/list", None).await?;

        if let Some(result) = response.result {
            let list: ListToolsResult = serde_json::from_value(result)
                .context("Failed to parse tools/list result")?;
            Ok(list.tools)
        } else {
            let err_msg = response
                .error
                .map(|e| e.message)
                .unwrap_or_else(|| "Unknown error".to_string());
            Err(anyhow::anyhow!("tools/list failed: {}", err_msg))
        }
    }

    /// Call a tool on the remote server.
    pub async fn call_tool(
        &mut self,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult> {
        let params = CallToolParams {
            name: name.into(),
            arguments,
        };

        let response = self
            .request(
                "tools/call",
                Some(serde_json::to_value(params).context("Failed to serialize call params")?),
            )
            .await?;

        if let Some(result) = response.result {
            let call_result: CallToolResult = serde_json::from_value(result)
                .context("Failed to parse tools/call result")?;
            Ok(call_result)
        } else {
            let err_msg = response
                .error
                .map(|e| e.message)
                .unwrap_or_else(|| "Unknown error".to_string());
            Err(anyhow::anyhow!("tools/call failed: {}", err_msg))
        }
    }

    // ========================================================================
    // RESOURCES
    // ========================================================================

    /// List all resources available on the remote server.
    pub async fn list_resources(&mut self) -> Result<Vec<Resource>> {
        let response = self.request("resources/list", None).await?;

        if let Some(result) = response.result {
            let list: ListResourcesResult = serde_json::from_value(result)
                .context("Failed to parse resources/list result")?;
            Ok(list.resources)
        } else {
            Ok(vec![])
        }
    }

    /// Read a specific resource by URI.
    pub async fn read_resource(&mut self, uri: impl Into<String>) -> Result<Vec<ResourceContent>> {
        let params = ReadResourceParams { uri: uri.into() };

        let response = self
            .request(
                "resources/read",
                Some(serde_json::to_value(params)?),
            )
            .await?;

        if let Some(result) = response.result {
            let read_result: ReadResourceResult = serde_json::from_value(result)
                .context("Failed to parse resources/read result")?;
            Ok(read_result.contents)
        } else {
            Ok(vec![])
        }
    }

    // ========================================================================
    // PROMPTS
    // ========================================================================

    /// List all prompts available on the remote server.
    pub async fn list_prompts(&mut self) -> Result<Vec<Prompt>> {
        let response = self.request("prompts/list", None).await?;

        if let Some(result) = response.result {
            let list: ListPromptsResult = serde_json::from_value(result)
                .context("Failed to parse prompts/list result")?;
            Ok(list.prompts)
        } else {
            Ok(vec![])
        }
    }

    // ========================================================================
    // PING
    // ========================================================================

    /// Send a ping to the server and wait for a response.
    pub async fn ping(&mut self) -> Result<Duration> {
        let start = std::time::Instant::now();
        let _response = self.request("ping", None).await?;
        Ok(start.elapsed())
    }

    // ========================================================================
    // LOW-LEVEL TRANSPORT
    // ========================================================================

    /// Send a JSON-RPC request and wait for the response.
    async fn request(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<JsonRpcResponse> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        let request = JsonRpcRequest::new(id, method, params);
        debug!("[MCP/client] → {} (id={})", method, id);

        let transport = self
            .transport
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Not connected"))?;

        transport
            .send(&JsonRpcMessage::Request(request))
            .await
            .context("Failed to send request")?;

        // Wait for the matching response
        // Simple synchronous approach: read messages until we get our response
        let timeout_duration = Duration::from_secs(30);
        let response = tokio::time::timeout(timeout_duration, async {
            loop {
                match transport.recv().await {
                    Ok(Some(JsonRpcMessage::Response(resp))) => {
                        if resp.id == JsonRpcId::Number(id) {
                            return Ok(resp);
                        } else {
                            debug!(
                                "[MCP/client] Received response for different id: {}",
                                resp.id
                            );
                        }
                    }
                    Ok(Some(JsonRpcMessage::Notification(notif))) => {
                        debug!(
                            "[MCP/client] Received notification: {}",
                            notif.method
                        );
                        // Process notifications but keep waiting for our response
                    }
                    Ok(Some(JsonRpcMessage::Request(req))) => {
                        warn!(
                            "[MCP/client] Received unexpected request from server: {}",
                            req.method
                        );
                    }
                    Ok(None) => {
                        return Err(anyhow::anyhow!("Transport closed while waiting for response"));
                    }
                    Err(e) => {
                        return Err(e.context("Transport error while waiting for response"));
                    }
                }
            }
        })
        .await
        .context("Request timeout (30s)")??;

        debug!("[MCP/client] ← response for id={}", id);
        Ok(response)
    }

    /// Send a JSON-RPC notification (no response expected).
    async fn notify(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<()> {
        let notification = JsonRpcNotification::new(method, params);

        let transport = self
            .transport
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Not connected"))?;

        transport
            .send(&JsonRpcMessage::Notification(notification))
            .await
            .context("Failed to send notification")
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::ChannelTransport;
    use crate::server::McpServer;
    use tokio::sync::{broadcast, mpsc};

    #[tokio::test]
    async fn test_client_connect_and_list_tools() {
        let (mut server_transport, client_transport) = ChannelTransport::pair(32);
        let (kernel_tx, _kernel_rx) = mpsc::channel(16);
        let (broadcast_tx, _broadcast_rx) = broadcast::channel(16);

        // Start server in background
        let mut server = McpServer::new(kernel_tx, broadcast_tx);
        let server_handle = tokio::spawn(async move {
            let _ = server.run(&mut server_transport).await;
        });

        // Connect client
        let mut client = McpClient::sodals();
        client.connect(Box::new(client_transport)).await.unwrap();

        assert!(client.is_connected());
        assert!(client.server_capabilities().is_some());
        assert!(client.server_info().is_some());
        assert_eq!(
            client.server_info().unwrap().name,
            "sodals-mcp-server"
        );

        // List tools
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 6);

        // Call health tool
        let result = client
            .call_tool("sodals_health", serde_json::json!({}))
            .await
            .unwrap();
        assert!(!result.is_error);

        // Ping
        let latency = client.ping().await.unwrap();
        assert!(latency.as_millis() < 1000);

        // Disconnect
        client.disconnect().await.unwrap();
        assert!(!client.is_connected());

        // Server should stop when transport closes
        let _ = server_handle.await;
    }

    #[tokio::test]
    async fn test_client_call_unknown_tool() {
        let (mut server_transport, client_transport) = ChannelTransport::pair(32);
        let (kernel_tx, _kernel_rx) = mpsc::channel(16);
        let (broadcast_tx, _broadcast_rx) = broadcast::channel(16);

        let mut server = McpServer::new(kernel_tx, broadcast_tx);
        let server_handle = tokio::spawn(async move {
            let _ = server.run(&mut server_transport).await;
        });

        let mut client = McpClient::sodals();
        client.connect(Box::new(client_transport)).await.unwrap();

        // Calling an unknown tool should return an error
        let result = client
            .call_tool("nonexistent_tool", serde_json::json!({}))
            .await;
        // The server returns an error response for unknown tools
        assert!(result.is_err());

        client.disconnect().await.unwrap();
        let _ = server_handle.await;
    }
}
