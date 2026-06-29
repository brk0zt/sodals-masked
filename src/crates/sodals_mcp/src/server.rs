// src/crates/sodals_mcp/src/server.rs
// ═══════════════════════════════════════════════════════════════════════
// MCP Server — handles the full MCP lifecycle over any transport
// ═══════════════════════════════════════════════════════════════════════

use anyhow::Result;
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, error, info, warn};

use sodals_net::protocol::SystemMsg;

use crate::registry::{PromptRegistry, ResourceRegistry, ToolRegistry};
use crate::tools::{build_tool_registry, KernelBridge};
use crate::transport::McpTransport;
use crate::types::*;

// ============================================================================
// MCP SERVER
// ============================================================================

/// The SODALS MCP Server.
///
/// Speaks the Model Context Protocol over any transport (stdio, SSE, channel).
/// Exposes SODALS kernel capabilities as MCP tools, resources, and prompts.
pub struct McpServer {
    /// Server identity
    info: ServerInfo,
    /// Advertised capabilities
    capabilities: ServerCapabilities,
    /// Registered tools
    tool_registry: ToolRegistry,
    /// Registered resources
    resource_registry: ResourceRegistry,
    /// Registered prompts
    prompt_registry: PromptRegistry,
    /// Whether the client has completed initialization
    initialized: bool,
}

impl McpServer {
    /// Create a new MCP server wired to the SODALS kernel.
    pub fn new(
        kernel_tx: mpsc::Sender<SystemMsg>,
        broadcast_tx: broadcast::Sender<SystemMsg>,
    ) -> Self {
        let bridge = KernelBridge::new(kernel_tx, broadcast_tx);

        Self {
            info: ServerInfo {
                name: "sodals-mcp-server".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability { list_changed: true }),
                resources: Some(ResourcesCapability {
                    subscribe: false,
                    list_changed: false,
                }),
                prompts: Some(PromptsCapability {
                    list_changed: false,
                }),
                logging: Some(LoggingCapability {}),
            },
            tool_registry: build_tool_registry(bridge),
            resource_registry: ResourceRegistry::new(),
            prompt_registry: PromptRegistry::new(),
            initialized: false,
        }
    }

    /// Run the MCP server event loop over the given transport.
    ///
    /// This method reads JSON-RPC messages from the transport, dispatches
    /// them according to the MCP specification, and writes responses back.
    /// It runs until the transport is closed or a shutdown is requested.
    pub async fn run(&mut self, transport: &mut dyn McpTransport) -> Result<()> {
        info!(
            "[MCP] Server starting — {} v{}",
            self.info.name, self.info.version
        );
        info!(
            "[MCP] {} tools registered",
            self.tool_registry.len()
        );

        loop {
            let msg = match transport.recv().await {
                Ok(Some(msg)) => msg,
                Ok(None) => {
                    info!("[MCP] Transport closed — shutting down");
                    break;
                }
                Err(e) => {
                    error!("[MCP] Transport error: {} — continuing", e);
                    continue;
                }
            };

            match msg {
                JsonRpcMessage::Request(request) => {
                    let response = self.handle_request(request).await;
                    if let Err(e) = transport.send(&JsonRpcMessage::Response(response)).await {
                        error!("[MCP] Failed to send response: {}", e);
                    }
                }
                JsonRpcMessage::Notification(notification) => {
                    self.handle_notification(notification).await;
                }
                JsonRpcMessage::Response(response) => {
                    // Servers don't normally receive responses, but log it
                    debug!("[MCP] Received unexpected response: id={}", response.id);
                }
            }
        }

        info!("[MCP] Server stopped");
        Ok(())
    }

    // ========================================================================
    // REQUEST DISPATCH
    // ========================================================================

    async fn handle_request(&mut self, request: JsonRpcRequest) -> JsonRpcResponse {
        let id = request.id.clone();
        let method = request.method.as_str();

        debug!("[MCP] ← {} (id={})", method, id);

        match method {
            // --- Lifecycle ---
            "initialize" => self.handle_initialize(id, request.params),
            "ping" => self.handle_ping(id),

            // --- Tools ---
            "tools/list" => self.handle_tools_list(id),
            "tools/call" => self.handle_tools_call(id, request.params).await,

            // --- Resources ---
            "resources/list" => self.handle_resources_list(id),
            "resources/read" => self.handle_resources_read(id, request.params),

            // --- Prompts ---
            "prompts/list" => self.handle_prompts_list(id),
            "prompts/get" => self.handle_prompts_get(id, request.params).await,

            // --- Unknown ---
            _ => {
                warn!("[MCP] Unknown method: {}", method);
                JsonRpcResponse::error(
                    id,
                    error_codes::METHOD_NOT_FOUND,
                    format!("Method not found: {}", method),
                )
            }
        }
    }

    // ========================================================================
    // NOTIFICATION DISPATCH
    // ========================================================================

    async fn handle_notification(&mut self, notification: JsonRpcNotification) {
        match notification.method.as_str() {
            "initialized" => {
                info!("[MCP] ✅ Client confirmed initialization — server is ready");
                self.initialized = true;
            }
            "notifications/cancelled" => {
                debug!("[MCP] Client cancelled a request");
                // In the future: cancel in-flight kernel operations
            }
            other => {
                debug!("[MCP] Received notification: {}", other);
            }
        }
    }

    // ========================================================================
    // HANDLERS: LIFECYCLE
    // ========================================================================

    fn handle_initialize(
        &self,
        id: JsonRpcId,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        // Parse client capabilities (best-effort)
        let client_info = params
            .as_ref()
            .and_then(|p| serde_json::from_value::<InitializeParams>(p.clone()).ok());

        if let Some(ref info) = client_info {
            info!(
                "[MCP] Initialize from {} v{} (protocol: {})",
                info.client_info.name,
                info.client_info.version,
                info.protocol_version
            );
        }

        let result = InitializeResult {
            protocol_version: MCP_PROTOCOL_VERSION.to_string(),
            capabilities: self.capabilities.clone(),
            server_info: self.info.clone(),
            instructions: Some(
                "SODALS MCP Server — Use tools/list to discover available capabilities. \
                 Available tools: sodals_solve, sodals_memorize, sodals_vector_search, \
                 sodals_forge_compile, sodals_runtime_exec, sodals_health."
                    .to_string(),
            ),
        };

        JsonRpcResponse::success(
            id,
            serde_json::to_value(result).unwrap_or(serde_json::json!({})),
        )
    }

    fn handle_ping(&self, id: JsonRpcId) -> JsonRpcResponse {
        JsonRpcResponse::success(id, serde_json::json!({}))
    }

    // ========================================================================
    // HANDLERS: TOOLS
    // ========================================================================

    fn handle_tools_list(&self, id: JsonRpcId) -> JsonRpcResponse {
        let result = ListToolsResult {
            tools: self.tool_registry.list(),
        };

        JsonRpcResponse::success(
            id,
            serde_json::to_value(result).unwrap_or(serde_json::json!({"tools": []})),
        )
    }

    async fn handle_tools_call(
        &self,
        id: JsonRpcId,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing params for tools/call",
                );
            }
        };

        let call_params: CallToolParams = match serde_json::from_value(params) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid params: {}", e),
                );
            }
        };

        let tool_entry = match self.tool_registry.get(&call_params.name) {
            Some(entry) => entry,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Unknown tool: {}", call_params.name),
                );
            }
        };

        info!("[MCP] Calling tool: {}", call_params.name);

        match tool_entry.handler.call(call_params.arguments).await {
            Ok(result) => JsonRpcResponse::success(
                id,
                serde_json::to_value(result).unwrap_or(serde_json::json!({})),
            ),
            Err(e) => {
                error!("[MCP] Tool {} error: {}", call_params.name, e);
                let error_result = CallToolResult::error(format!("Tool error: {}", e));
                JsonRpcResponse::success(
                    id,
                    serde_json::to_value(error_result).unwrap_or(serde_json::json!({})),
                )
            }
        }
    }

    // ========================================================================
    // HANDLERS: RESOURCES
    // ========================================================================

    fn handle_resources_list(&self, id: JsonRpcId) -> JsonRpcResponse {
        let result = ListResourcesResult {
            resources: self.resource_registry.list(),
        };

        JsonRpcResponse::success(
            id,
            serde_json::to_value(result).unwrap_or(serde_json::json!({"resources": []})),
        )
    }

    fn handle_resources_read(
        &self,
        id: JsonRpcId,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing params for resources/read",
                );
            }
        };

        let read_params: ReadResourceParams = match serde_json::from_value(params) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid params: {}", e),
                );
            }
        };

        match self.resource_registry.get(&read_params.uri) {
            Some(_entry) => {
                // For now, return empty — resource handlers can be wired later
                let result = ReadResourceResult { contents: vec![] };
                JsonRpcResponse::success(
                    id,
                    serde_json::to_value(result).unwrap_or(serde_json::json!({})),
                )
            }
            None => JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                format!("Resource not found: {}", read_params.uri),
            ),
        }
    }

    // ========================================================================
    // HANDLERS: PROMPTS
    // ========================================================================

    fn handle_prompts_list(&self, id: JsonRpcId) -> JsonRpcResponse {
        let result = ListPromptsResult {
            prompts: self.prompt_registry.list(),
        };

        JsonRpcResponse::success(
            id,
            serde_json::to_value(result).unwrap_or(serde_json::json!({"prompts": []})),
        )
    }

    async fn handle_prompts_get(
        &self,
        id: JsonRpcId,
        params: Option<serde_json::Value>,
    ) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    "Missing params for prompts/get",
                );
            }
        };

        let get_params: GetPromptParams = match serde_json::from_value(params) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    error_codes::INVALID_PARAMS,
                    format!("Invalid params: {}", e),
                );
            }
        };

        match self.prompt_registry.get(&get_params.name) {
            Some(entry) => match entry.handler.get(get_params.arguments).await {
                Ok(result) => JsonRpcResponse::success(
                    id,
                    serde_json::to_value(result).unwrap_or(serde_json::json!({})),
                ),
                Err(e) => JsonRpcResponse::error(
                    id,
                    error_codes::INTERNAL_ERROR,
                    format!("Prompt error: {}", e),
                ),
            },
            None => JsonRpcResponse::error(
                id,
                error_codes::INVALID_PARAMS,
                format!("Prompt not found: {}", get_params.name),
            ),
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::ChannelTransport;

    fn create_test_server() -> McpServer {
        let (kernel_tx, _kernel_rx) = mpsc::channel(16);
        let (broadcast_tx, _broadcast_rx) = broadcast::channel(16);
        McpServer::new(kernel_tx, broadcast_tx)
    }

    #[tokio::test]
    async fn test_initialize_handshake() {
        let mut server = create_test_server();

        let init_params = InitializeParams {
            protocol_version: MCP_PROTOCOL_VERSION.to_string(),
            capabilities: ClientCapabilities::default(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
        };

        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(1),
            method: "initialize".to_string(),
            params: Some(serde_json::to_value(init_params).unwrap()),
        }).await;

        assert!(response.result.is_some());
        assert!(response.error.is_none());

        let result: InitializeResult =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.protocol_version, MCP_PROTOCOL_VERSION);
        assert_eq!(result.server_info.name, "sodals-mcp-server");
        assert!(result.capabilities.tools.is_some());
    }

    #[tokio::test]
    async fn test_ping() {
        let mut server = create_test_server();

        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(2),
            method: "ping".to_string(),
            params: None,
        }).await;

        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[tokio::test]
    async fn test_tools_list() {
        let mut server = create_test_server();

        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(3),
            method: "tools/list".to_string(),
            params: None,
        }).await;

        assert!(response.result.is_some());
        let result: ListToolsResult =
            serde_json::from_value(response.result.unwrap()).unwrap();

        assert_eq!(result.tools.len(), 6);

        let tool_names: Vec<&str> = result.tools.iter().map(|t| t.name.as_str()).collect();
        assert!(tool_names.contains(&"sodals_solve"));
        assert!(tool_names.contains(&"sodals_memorize"));
        assert!(tool_names.contains(&"sodals_health"));
    }

    #[tokio::test]
    async fn test_tools_call_health() {
        let mut server = create_test_server();

        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(4),
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({
                "name": "sodals_health",
                "arguments": {}
            })),
        }).await;

        assert!(response.result.is_some());
        let result: CallToolResult =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn test_unknown_method() {
        let mut server = create_test_server();

        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(5),
            method: "nonexistent/method".to_string(),
            params: None,
        }).await;

        assert!(response.error.is_some());
        assert_eq!(
            response.error.unwrap().code,
            error_codes::METHOD_NOT_FOUND
        );
    }

    #[tokio::test]
    async fn test_tools_call_unknown_tool() {
        let mut server = create_test_server();

        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(6),
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({
                "name": "nonexistent_tool",
                "arguments": {}
            })),
        }).await;

        assert!(response.error.is_some());
    }

    #[tokio::test]
    async fn test_full_lifecycle_over_channel() {
        let (mut server_transport, mut client_transport) = ChannelTransport::pair(32);
        let (kernel_tx, _kernel_rx) = mpsc::channel(16);
        let (broadcast_tx, _broadcast_rx) = broadcast::channel(16);

        let mut server = McpServer::new(kernel_tx, broadcast_tx);

        // Run server in background
        let server_handle = tokio::spawn(async move {
            server.run(&mut server_transport).await
        });

        // Client: send initialize
        let init_req = JsonRpcMessage::Request(JsonRpcRequest::new(
            1,
            "initialize",
            Some(serde_json::to_value(InitializeParams {
                protocol_version: MCP_PROTOCOL_VERSION.to_string(),
                capabilities: ClientCapabilities::default(),
                client_info: ClientInfo {
                    name: "test".to_string(),
                    version: "0.1".to_string(),
                },
            }).unwrap()),
        ));
        client_transport.send(&init_req).await.unwrap();

        // Client: receive initialize response
        let resp = client_transport.recv().await.unwrap().unwrap();
        match resp {
            JsonRpcMessage::Response(r) => {
                assert!(r.result.is_some());
                assert_eq!(r.id, JsonRpcId::Number(1));
            }
            _ => panic!("Expected Response"),
        }

        // Client: send initialized notification
        let init_notif = JsonRpcMessage::Notification(JsonRpcNotification::new(
            "initialized",
            None,
        ));
        client_transport.send(&init_notif).await.unwrap();

        // Client: list tools
        let list_req = JsonRpcMessage::Request(JsonRpcRequest::new(
            2,
            "tools/list",
            None,
        ));
        client_transport.send(&list_req).await.unwrap();

        let resp = client_transport.recv().await.unwrap().unwrap();
        match resp {
            JsonRpcMessage::Response(r) => {
                let result: ListToolsResult =
                    serde_json::from_value(r.result.unwrap()).unwrap();
                assert_eq!(result.tools.len(), 6);
            }
            _ => panic!("Expected Response"),
        }

        // Close transport to stop the server
        drop(client_transport);
        let _ = server_handle.await;
    }
}
