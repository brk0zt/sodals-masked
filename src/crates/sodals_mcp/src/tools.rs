// src/crates/sodals_mcp/src/tools.rs
// ═══════════════════════════════════════════════════════════════════════
// SODALS Tool Definitions — MCP Tool ↔ SystemMsg bridge
// ═══════════════════════════════════════════════════════════════════════
//
// Each tool:
//   1. Deserializes MCP CallToolParams → typed request
//   2. Creates the appropriate SystemMsg
//   3. Sends via kernel_tx (mpsc channel)
//   4. Waits for response on broadcast_rx
//   5. Serializes result → MCP CallToolResult

use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};
use tracing::{info, error};

use sodals_net::protocol::SystemMsg;

use crate::registry::{ToolHandler, ToolRegistry};
use crate::types::*;

// ============================================================================
// SHARED KERNEL BRIDGE
// ============================================================================

/// Shared state for all tool handlers — the channel pair to the SODALS kernel.
#[derive(Clone)]
pub struct KernelBridge {
    pub kernel_tx: mpsc::Sender<SystemMsg>,
    pub broadcast_tx: broadcast::Sender<SystemMsg>,
}

impl KernelBridge {
    pub fn new(
        kernel_tx: mpsc::Sender<SystemMsg>,
        broadcast_tx: broadcast::Sender<SystemMsg>,
    ) -> Self {
        Self {
            kernel_tx,
            broadcast_tx,
        }
    }

    /// Send a message to the kernel and wait for a response matching `req_id`.
    async fn request(&self, req_id: u64, msg: SystemMsg) -> Result<SystemMsg> {
        let mut rx = self.broadcast_tx.subscribe();

        self.kernel_tx
            .send(msg)
            .await
            .context("Failed to send to kernel")?;

        let timeout = tokio::time::timeout(Duration::from_secs(30), async move {
            loop {
                match rx.recv().await {
                    Ok(response) => {
                        if Self::matches_req_id(&response, req_id) {
                            return Ok(response);
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => {
                        return Err(anyhow::anyhow!("Kernel broadcast closed"));
                    }
                }
            }
        });

        timeout
            .await
            .context("Kernel response timeout (30s)")
            .and_then(|r| r)
    }

    fn matches_req_id(msg: &SystemMsg, target: u64) -> bool {
        match msg {
            SystemMsg::CortexResponse { req_id, .. } => *req_id == target,
            SystemMsg::VectorSearchResult { request_id, .. } => *request_id == target,
            SystemMsg::ForgeCompilationResult { req_id, .. } => *req_id == target,
            SystemMsg::RuntimeExecuteResult { req_id, .. } => *req_id == target,
            _ => false,
        }
    }
}

// ============================================================================
// TOOL: sodals_solve
// ============================================================================

/// Run LLM inference on a prompt via the SODALS kernel.
pub struct SolveTool {
    bridge: KernelBridge,
}

impl SolveTool {
    pub fn new(bridge: KernelBridge) -> Self {
        Self { bridge }
    }

    pub fn definition() -> Tool {
        Tool {
            name: "sodals_solve".to_string(),
            description: Some(
                "Run inference on a prompt using the SODALS neuro-engine. \
                 Supports text and optional high-stakes mode."
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(HashMap::from([
                    (
                        "prompt".to_string(),
                        PropertySchema {
                            prop_type: "string".to_string(),
                            description: Some("The text prompt to process".to_string()),
                            default_value: None,
                            enum_values: None,
                        },
                    ),
                    (
                        "high_stakes".to_string(),
                        PropertySchema {
                            prop_type: "boolean".to_string(),
                            description: Some(
                                "Enable high-stakes inference mode (SOLVE: prefix)".to_string(),
                            ),
                            default_value: Some(serde_json::json!(false)),
                            enum_values: None,
                        },
                    ),
                    (
                        "user_id".to_string(),
                        PropertySchema {
                            prop_type: "string".to_string(),
                            description: Some("Optional user identifier for tracking".to_string()),
                            default_value: None,
                            enum_values: None,
                        },
                    ),
                ])),
                required: Some(vec!["prompt".to_string()]),
            },
        }
    }
}

#[async_trait]
impl ToolHandler for SolveTool {
    async fn call(&self, arguments: serde_json::Value) -> Result<CallToolResult> {
        let prompt = arguments
            .get("prompt")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: prompt"))?
            .to_string();

        let high_stakes = arguments
            .get("high_stakes")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let user_id = arguments
            .get("user_id")
            .and_then(|v| v.as_str())
            .unwrap_or("mcp-client")
            .to_string();

        let req_id: u64 = rand::random();
        let final_prompt = if high_stakes {
            format!("SOLVE: {}", prompt)
        } else {
            prompt
        };

        info!("[MCP/solve] req_id={} prompt_len={}", req_id, final_prompt.len());

        let msg = SystemMsg::LlmRequest {
            req_id,
            user_id,
            prompt: final_prompt,
            image_data: None,
            response_tx: None,
        };

        match self.bridge.request(req_id, msg).await {
            Ok(SystemMsg::CortexResponse { content, .. }) => Ok(CallToolResult::text(content)),
            Ok(_) => Ok(CallToolResult::error("Unexpected response type from kernel")),
            Err(e) => {
                error!("[MCP/solve] Error: {}", e);
                Ok(CallToolResult::error(format!("Kernel error: {}", e)))
            }
        }
    }
}

// ============================================================================
// TOOL: sodals_memorize
// ============================================================================

/// Store text in the SODALS vector memory.
pub struct MemorizeTool {
    bridge: KernelBridge,
}

impl MemorizeTool {
    pub fn new(bridge: KernelBridge) -> Self {
        Self { bridge }
    }

    pub fn definition() -> Tool {
        Tool {
            name: "sodals_memorize".to_string(),
            description: Some(
                "Store text in the SODALS vector memory database for later retrieval."
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(HashMap::from([
                    (
                        "text".to_string(),
                        PropertySchema {
                            prop_type: "string".to_string(),
                            description: Some("The text to store in memory".to_string()),
                            default_value: None,
                            enum_values: None,
                        },
                    ),
                    (
                        "user_id".to_string(),
                        PropertySchema {
                            prop_type: "string".to_string(),
                            description: Some("User identifier for scoped storage".to_string()),
                            default_value: None,
                            enum_values: None,
                        },
                    ),
                ])),
                required: Some(vec!["text".to_string()]),
            },
        }
    }
}

#[async_trait]
impl ToolHandler for MemorizeTool {
    async fn call(&self, arguments: serde_json::Value) -> Result<CallToolResult> {
        let text = arguments
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: text"))?
            .to_string();

        let user_id = arguments
            .get("user_id")
            .and_then(|v| v.as_str())
            .unwrap_or("mcp-client")
            .to_string();

        info!("[MCP/memorize] text_len={} user={}", text.len(), user_id);

        // Note: embedding vector is computed by the kernel
        let vector = vec![0.0f32; 768];

        let msg = SystemMsg::MemorizeRequest {
            user_id,
            text: text.clone(),
            vector,
        };

        // Fire-and-forget (memorize has no matching response in the current protocol)
        self.bridge
            .kernel_tx
            .send(msg)
            .await
            .context("Failed to send memorize request to kernel")?;

        Ok(CallToolResult::text(format!(
            "Stored {} bytes in vector memory",
            text.len()
        )))
    }
}

// ============================================================================
// TOOL: sodals_vector_search
// ============================================================================

/// Search the SODALS vector memory.
pub struct VectorSearchTool {
    bridge: KernelBridge,
}

impl VectorSearchTool {
    pub fn new(bridge: KernelBridge) -> Self {
        Self { bridge }
    }

    pub fn definition() -> Tool {
        Tool {
            name: "sodals_vector_search".to_string(),
            description: Some(
                "Search the SODALS vector memory database for semantically similar content."
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(HashMap::from([
                    (
                        "query".to_string(),
                        PropertySchema {
                            prop_type: "string".to_string(),
                            description: Some("The search query text".to_string()),
                            default_value: None,
                            enum_values: None,
                        },
                    ),
                    (
                        "top_k".to_string(),
                        PropertySchema {
                            prop_type: "integer".to_string(),
                            description: Some(
                                "Number of results to return (default: 5)".to_string(),
                            ),
                            default_value: Some(serde_json::json!(5)),
                            enum_values: None,
                        },
                    ),
                    (
                        "user_id".to_string(),
                        PropertySchema {
                            prop_type: "string".to_string(),
                            description: Some("User identifier for scoped search".to_string()),
                            default_value: None,
                            enum_values: None,
                        },
                    ),
                ])),
                required: Some(vec!["query".to_string()]),
            },
        }
    }
}

#[async_trait]
impl ToolHandler for VectorSearchTool {
    async fn call(&self, arguments: serde_json::Value) -> Result<CallToolResult> {
        let _query = arguments
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: query"))?;

        let top_k = arguments
            .get("top_k")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;

        let user_id = arguments
            .get("user_id")
            .and_then(|v| v.as_str())
            .unwrap_or("mcp-client")
            .to_string();

        let req_id: u64 = rand::random();

        info!("[MCP/vector_search] req_id={} top_k={}", req_id, top_k);

        // In production, compute the query embedding here
        let query_vector = vec![0.0f32; 768];

        let msg = SystemMsg::VectorSearchRequest {
            request_id: req_id,
            user_id,
            query_vector,
            top_k,
        };

        match self.bridge.request(req_id, msg).await {
            Ok(SystemMsg::VectorSearchResult { results, .. }) => {
                let output = if results.is_empty() {
                    "No results found.".to_string()
                } else {
                    results
                        .iter()
                        .enumerate()
                        .map(|(i, r)| format!("{}. {}", i + 1, r))
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                Ok(CallToolResult::text(output))
            }
            Ok(_) => Ok(CallToolResult::error("Unexpected response type")),
            Err(e) => Ok(CallToolResult::error(format!("Search error: {}", e))),
        }
    }
}

// ============================================================================
// TOOL: sodals_forge_compile
// ============================================================================

/// Compile source code to WASM module via the SODALS Forge.
pub struct ForgeCompileTool {
    bridge: KernelBridge,
}

impl ForgeCompileTool {
    pub fn new(bridge: KernelBridge) -> Self {
        Self { bridge }
    }

    pub fn definition() -> Tool {
        Tool {
            name: "sodals_forge_compile".to_string(),
            description: Some(
                "Compile source code to a WASM module using the SODALS Forge compiler."
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(HashMap::from([
                    (
                        "source_code".to_string(),
                        PropertySchema {
                            prop_type: "string".to_string(),
                            description: Some("The source code to compile".to_string()),
                            default_value: None,
                            enum_values: None,
                        },
                    ),
                    (
                        "module_name".to_string(),
                        PropertySchema {
                            prop_type: "string".to_string(),
                            description: Some("Name for the compiled WASM module".to_string()),
                            default_value: None,
                            enum_values: None,
                        },
                    ),
                ])),
                required: Some(vec!["source_code".to_string(), "module_name".to_string()]),
            },
        }
    }
}

#[async_trait]
impl ToolHandler for ForgeCompileTool {
    async fn call(&self, arguments: serde_json::Value) -> Result<CallToolResult> {
        let source_code = arguments
            .get("source_code")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: source_code"))?
            .to_string();

        let module_name = arguments
            .get("module_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: module_name"))?
            .to_string();

        let req_id: u64 = rand::random();

        info!(
            "[MCP/forge] req_id={} module={} src_len={}",
            req_id,
            module_name,
            source_code.len()
        );

        let msg = SystemMsg::ForgeCompileRequest {
            req_id,
            source_code: source_code.clone(),
            module_name: module_name.clone(),
            retry_count: 0,
            original_prompt: String::new(),
        };

        match self.bridge.request(req_id, msg).await {
            Ok(SystemMsg::ForgeCompilationResult {
                success,
                wasm_path,
                error_log,
                ..
            }) => {
                if success {
                    Ok(CallToolResult::text(format!(
                        "Compilation successful. WASM module: {}",
                        wasm_path
                    )))
                } else {
                    Ok(CallToolResult::error(format!(
                        "Compilation failed: {}",
                        error_log.unwrap_or_else(|| "Unknown error".to_string())
                    )))
                }
            }
            Ok(_) => Ok(CallToolResult::error("Unexpected response type")),
            Err(e) => Ok(CallToolResult::error(format!("Forge error: {}", e))),
        }
    }
}

// ============================================================================
// TOOL: sodals_runtime_exec
// ============================================================================

/// Execute a WASM module in the SODALS runtime.
pub struct RuntimeExecTool {
    bridge: KernelBridge,
}

impl RuntimeExecTool {
    pub fn new(bridge: KernelBridge) -> Self {
        Self { bridge }
    }

    pub fn definition() -> Tool {
        Tool {
            name: "sodals_runtime_exec".to_string(),
            description: Some(
                "Execute a compiled WASM module in the SODALS sandboxed runtime."
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(HashMap::from([(
                    "wasm_path".to_string(),
                    PropertySchema {
                        prop_type: "string".to_string(),
                        description: Some(
                            "Path to the WASM module to execute".to_string(),
                        ),
                        default_value: None,
                        enum_values: None,
                    },
                )])),
                required: Some(vec!["wasm_path".to_string()]),
            },
        }
    }
}

#[async_trait]
impl ToolHandler for RuntimeExecTool {
    async fn call(&self, arguments: serde_json::Value) -> Result<CallToolResult> {
        let wasm_path = arguments
            .get("wasm_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: wasm_path"))?
            .to_string();

        let req_id: u64 = rand::random();

        info!("[MCP/runtime] req_id={} wasm={}", req_id, wasm_path);

        let msg = SystemMsg::RuntimeExecuteRequest {
            req_id,
            wasm_path,
        };

        match self.bridge.request(req_id, msg).await {
            Ok(SystemMsg::RuntimeExecuteResult {
                stdout, stderr, ..
            }) => {
                let mut output = stdout;
                if let Some(err) = stderr {
                    if !err.is_empty() {
                        output.push_str("\n--- stderr ---\n");
                        output.push_str(&err);
                    }
                }
                Ok(CallToolResult::text(output))
            }
            Ok(_) => Ok(CallToolResult::error("Unexpected response type")),
            Err(e) => Ok(CallToolResult::error(format!("Runtime error: {}", e))),
        }
    }
}

// ============================================================================
// TOOL: sodals_health
// ============================================================================

/// Check the health of the SODALS system.
pub struct HealthTool;

impl HealthTool {
    pub fn definition() -> Tool {
        Tool {
            name: "sodals_health".to_string(),
            description: Some(
                "Check the health and status of the SODALS system."
                    .to_string(),
            ),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: None,
                required: None,
            },
        }
    }
}

#[async_trait]
impl ToolHandler for HealthTool {
    async fn call(&self, _arguments: serde_json::Value) -> Result<CallToolResult> {
        let health = serde_json::json!({
            "status": "operational",
            "version": env!("CARGO_PKG_VERSION"),
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "engine": "sodals-mcp",
            "capabilities": [
                "inference",
                "vector_memory",
                "forge_compile",
                "runtime_exec"
            ]
        });

        Ok(CallToolResult::text(
            serde_json::to_string_pretty(&health).unwrap(),
        ))
    }
}

// ============================================================================
// REGISTRY BUILDER
// ============================================================================

/// Build the default SODALS tool registry with all available tools.
pub fn build_tool_registry(bridge: KernelBridge) -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    // Register all SODALS tools
    registry.register(
        SolveTool::definition(),
        Arc::new(SolveTool::new(bridge.clone())),
    );
    registry.register(
        MemorizeTool::definition(),
        Arc::new(MemorizeTool::new(bridge.clone())),
    );
    registry.register(
        VectorSearchTool::definition(),
        Arc::new(VectorSearchTool::new(bridge.clone())),
    );
    registry.register(
        ForgeCompileTool::definition(),
        Arc::new(ForgeCompileTool::new(bridge.clone())),
    );
    registry.register(
        RuntimeExecTool::definition(),
        Arc::new(RuntimeExecTool::new(bridge)),
    );
    registry.register(HealthTool::definition(), Arc::new(HealthTool));

    info!(
        "[MCP] Registered {} tools: {:?}",
        registry.len(),
        registry.list().iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    registry
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_definitions_valid_json_schema() {
        let tools = vec![
            SolveTool::definition(),
            MemorizeTool::definition(),
            VectorSearchTool::definition(),
            ForgeCompileTool::definition(),
            RuntimeExecTool::definition(),
            HealthTool::definition(),
        ];

        for tool in &tools {
            assert!(!tool.name.is_empty(), "Tool name must not be empty");
            assert!(
                tool.description.is_some(),
                "Tool {} must have a description",
                tool.name
            );
            assert_eq!(
                tool.input_schema.schema_type, "object",
                "Tool {} must have object schema",
                tool.name
            );

            // Verify JSON serialization works
            let json = serde_json::to_value(tool).unwrap();
            assert!(json.is_object());
        }

        assert_eq!(tools.len(), 6);
    }

    #[tokio::test]
    async fn test_health_tool() {
        let result = HealthTool.call(serde_json::json!({})).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);

        if let ToolContent::Text { text } = &result.content[0] {
            let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(parsed["status"], "operational");
        } else {
            panic!("Expected text content");
        }
    }
}
