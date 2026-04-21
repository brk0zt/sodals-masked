// src/crates/sodals_mcp/src/types.rs
// ═══════════════════════════════════════════════════════════════════════
// MCP Protocol Types — JSON-RPC 2.0 + Model Context Protocol
// Specification: https://modelcontextprotocol.io
// ═══════════════════════════════════════════════════════════════════════

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// JSON-RPC 2.0 CORE
// ============================================================================

/// A JSON-RPC 2.0 message (request, response, or notification).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonRpcMessage {
    Request(JsonRpcRequest),
    Response(JsonRpcResponse),
    Notification(JsonRpcNotification),
}

/// JSON-RPC 2.0 Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: JsonRpcId,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

/// JSON-RPC 2.0 Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: JsonRpcId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 Notification (no id, no response expected)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

/// JSON-RPC 2.0 Error object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// Request ID — can be integer or string per spec
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum JsonRpcId {
    Number(u64),
    String(String),
}

impl std::fmt::Display for JsonRpcId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonRpcId::Number(n) => write!(f, "{}", n),
            JsonRpcId::String(s) => write!(f, "{}", s),
        }
    }
}

// ============================================================================
// JSON-RPC ERROR CODES
// ============================================================================

/// Standard JSON-RPC 2.0 error codes + MCP extensions
pub mod error_codes {
    pub const PARSE_ERROR: i64 = -32700;
    pub const INVALID_REQUEST: i64 = -32600;
    pub const METHOD_NOT_FOUND: i64 = -32601;
    pub const INVALID_PARAMS: i64 = -32602;
    pub const INTERNAL_ERROR: i64 = -32603;
}

// ============================================================================
// MCP PROTOCOL VERSION
// ============================================================================

/// The MCP protocol version we implement
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

// ============================================================================
// MCP LIFECYCLE — INITIALIZE
// ============================================================================

/// Parameters for the `initialize` request (client → server)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    /// The MCP protocol version the client supports
    pub protocol_version: String,
    /// Client capabilities
    pub capabilities: ClientCapabilities,
    /// Information about the client
    pub client_info: ClientInfo,
}

/// Result of the `initialize` request (server → client)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    /// The MCP protocol version the server chose
    pub protocol_version: String,
    /// Server capabilities
    pub capabilities: ServerCapabilities,
    /// Information about the server
    pub server_info: ServerInfo,
    /// Optional instructions for the client on how to use the server
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

// ============================================================================
// CAPABILITIES
// ============================================================================

/// What the client can do
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientCapabilities {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roots: Option<RootsCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingCapability>,
}

/// What the server can do
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logging: Option<LoggingCapability>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCapability {
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcesCapability {
    #[serde(default)]
    pub subscribe: bool,
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptsCapability {
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoggingCapability {}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RootsCapability {
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplingCapability {}

// ============================================================================
// SERVER / CLIENT INFO
// ============================================================================

/// Information about the MCP server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

/// Information about the MCP client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

// ============================================================================
// MCP TOOLS
// ============================================================================

/// An MCP Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    /// Unique tool name
    pub name: String,
    /// Human-readable description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema describing the tool's input parameters
    pub input_schema: ToolInputSchema,
}

/// JSON Schema for tool input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInputSchema {
    /// Always "object" for MCP tools
    #[serde(rename = "type")]
    pub schema_type: String,
    /// Property definitions
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, PropertySchema>>,
    /// Required property names
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

/// Schema for a single property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertySchema {
    #[serde(rename = "type")]
    pub prop_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "default", default, skip_serializing_if = "Option::is_none")]
    pub default_value: Option<serde_json::Value>,
    #[serde(rename = "enum", default, skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
}

/// Parameters for `tools/call`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallToolParams {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// Result of `tools/call`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CallToolResult {
    pub content: Vec<ToolContent>,
    #[serde(default)]
    pub is_error: bool,
}

/// Result of `tools/list`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListToolsResult {
    pub tools: Vec<Tool>,
}

// ============================================================================
// MCP RESOURCES
// ============================================================================

/// An MCP Resource
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Resource {
    /// URI identifying the resource
    pub uri: String,
    /// Human-readable name
    pub name: String,
    /// Description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// MIME type
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// A resource template with URI patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceTemplate {
    pub uri_template: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Parameters for `resources/read`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceParams {
    pub uri: String,
}

/// Result of `resources/read`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResourceResult {
    pub contents: Vec<ResourceContent>,
}

/// Content of a resource
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceContent {
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

/// Result of `resources/list`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListResourcesResult {
    pub resources: Vec<Resource>,
}

// ============================================================================
// MCP PROMPTS
// ============================================================================

/// An MCP Prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prompt {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Vec<PromptArgument>>,
}

/// A prompt argument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgument {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub required: bool,
}

/// Parameters for `prompts/get`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptParams {
    pub name: String,
    #[serde(default)]
    pub arguments: HashMap<String, String>,
}

/// Result of `prompts/get`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetPromptResult {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub messages: Vec<PromptMessage>,
}

/// A prompt message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMessage {
    pub role: String,
    pub content: ToolContent,
}

/// Result of `prompts/list`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListPromptsResult {
    pub prompts: Vec<Prompt>,
}

// ============================================================================
// CONTENT TYPES (shared across tools, resources, prompts)
// ============================================================================

/// Content block — text or image
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ToolContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    #[serde(rename = "resource")]
    Resource {
        resource: ResourceContent,
    },
}

// ============================================================================
// PING
// ============================================================================

/// Empty params for `ping`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingParams {}

// ============================================================================
// LOGGING
// ============================================================================

/// Log level for `notifications/message`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

/// Parameters for `notifications/message`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogMessageParams {
    pub level: LogLevel,
    pub logger: String,
    pub data: serde_json::Value,
}

// ============================================================================
// BUILDER HELPERS
// ============================================================================

impl JsonRpcRequest {
    /// Create a new JSON-RPC request
    pub fn new(id: u64, method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: JsonRpcId::Number(id),
            method: method.into(),
            params,
        }
    }
}

impl JsonRpcResponse {
    /// Create a success response
    pub fn success(id: JsonRpcId, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response
    pub fn error(id: JsonRpcId, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

impl JsonRpcNotification {
    /// Create a new notification
    pub fn new(method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: method.into(),
            params,
        }
    }
}

impl CallToolResult {
    /// Create a text result
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: vec![ToolContent::Text {
                text: content.into(),
            }],
            is_error: false,
        }
    }

    /// Create an error result
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![ToolContent::Text {
                text: message.into(),
            }],
            is_error: true,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsonrpc_request_serialization() {
        let req = JsonRpcRequest::new(1, "tools/list", None);
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("\"method\":\"tools/list\""));
    }

    #[test]
    fn test_jsonrpc_response_success() {
        let resp = JsonRpcResponse::success(
            JsonRpcId::Number(42),
            serde_json::json!({"tools": []}),
        );
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"id\":42"));
        assert!(json.contains("\"result\""));
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn test_jsonrpc_response_error() {
        let resp = JsonRpcResponse::error(
            JsonRpcId::Number(1),
            error_codes::METHOD_NOT_FOUND,
            "Method not found",
        );
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("-32601"));
    }

    #[test]
    fn test_tool_definition_serialization() {
        let tool = Tool {
            name: "sodals_solve".to_string(),
            description: Some("Run inference".to_string()),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: Some(HashMap::from([
                    ("prompt".to_string(), PropertySchema {
                        prop_type: "string".to_string(),
                        description: Some("The prompt".to_string()),
                        default_value: None,
                        enum_values: None,
                    }),
                ])),
                required: Some(vec!["prompt".to_string()]),
            },
        };
        let json = serde_json::to_string_pretty(&tool).unwrap();
        assert!(json.contains("sodals_solve"));
        assert!(json.contains("\"type\": \"object\""));
    }

    #[test]
    fn test_call_tool_result_text() {
        let result = CallToolResult::text("Answer is 42");
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("Answer is 42"));
        assert!(json.contains("\"isError\":false"));
    }

    #[test]
    fn test_jsonrpc_message_deserialization() {
        let req_json = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let msg: JsonRpcMessage = serde_json::from_str(req_json).unwrap();
        match msg {
            JsonRpcMessage::Request(req) => {
                assert_eq!(req.method, "initialize");
                assert_eq!(req.id, JsonRpcId::Number(1));
            }
            _ => panic!("Expected Request"),
        }
    }

    #[test]
    fn test_initialize_params_roundtrip() {
        let params = InitializeParams {
            protocol_version: MCP_PROTOCOL_VERSION.to_string(),
            capabilities: ClientCapabilities::default(),
            client_info: ClientInfo {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
            },
        };
        let json = serde_json::to_value(&params).unwrap();
        let decoded: InitializeParams = serde_json::from_value(json).unwrap();
        assert_eq!(decoded.protocol_version, MCP_PROTOCOL_VERSION);
        assert_eq!(decoded.client_info.name, "test-client");
    }
}
