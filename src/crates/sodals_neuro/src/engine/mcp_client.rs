use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

/// Standard MCP JSON-RPC 2.0 Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

/// Standard MCP JSON-RPC 2.0 Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    pub id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

/// MCP Tool Call Structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCall {
    pub name: String,
    pub arguments: Value,
}

/// MCP Tool Result Structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResult {
    pub content: Vec<Value>,
    #[serde(default)]
    pub is_error: bool,
}

/// Lightweight, deterministic Async MCP Client over STDIO
pub struct McpClient {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    request_id: u64,
}

impl McpClient {
    /// Spawn a native background process to act as the exact MCP JSON-RPC Server
    pub async fn new(command: &str, args: &[&str]) -> Result<Self> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = child.stdin.take().ok_or_else(|| anyhow::anyhow!("Failed to open native stdin pipe"))?;
        let stdout = BufReader::new(child.stdout.take().ok_or_else(|| anyhow::anyhow!("Failed to open native stdout pipe"))?);

        // Standard JSON-RPC init is expected immediately on connect for MCP implementations.
        // As a minimal bridge, we just prepare the pipes.
        
        Ok(Self { _child: child, stdin, stdout, request_id: 1 })
    }

    /// Dispatch JSON-RPC over stdout/stdin accurately
    pub async fn send_request(&mut self, method: &str, params: Option<Value>) -> Result<McpResponse> {
        let req = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: self.request_id,
            method: method.to_string(),
            params,
        };
        self.request_id += 1;

        let req_str = serde_json::to_string(&req)? + "\n";
        self.stdin.write_all(req_str.as_bytes()).await?;
        self.stdin.flush().await?;

        let mut response_str = String::new();
        self.stdout.read_line(&mut response_str).await?;
        
        let resp: McpResponse = serde_json::from_str(&response_str)?;
        Ok(resp)
    }

    /// Retrieve the catalog of active deterministic commands
    pub async fn list_tools(&mut self) -> Result<Value> {
        let resp = self.send_request("tools/list", None).await?;
        if let Some(err) = resp.error {
            anyhow::bail!("MCP Remote Error: {:?}", err);
        }
        resp.result.ok_or_else(|| anyhow::anyhow!("Malformed tools/list response: Missing 'result' block"))
    }

    /// Execute a tool precisely and return its data content. NO agentic loops here.
    pub async fn call_tool(&mut self, tool: McpToolCall) -> Result<McpResult> {
        let params = serde_json::json!({
            "name": tool.name,
            "arguments": tool.arguments
        });
        
        let resp = self.send_request("tools/call", Some(params)).await?;
        
        if let Some(err) = resp.error {
            anyhow::bail!("MCP Tool Context Error: {:?}", err);
        }
        
        let result = resp.result.ok_or_else(|| anyhow::anyhow!("Malformed tools/call response: Missing 'result' block"))?;
        
        let content = result.get("content")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();
            
        let is_error = result.get("isError")
            .and_then(|e| e.as_bool())
            .unwrap_or(false);

        Ok(McpResult { content, is_error })
    }
}

pub struct McpInterceptor;

impl McpInterceptor {
    /// Phase 33 FIX: Native async execution path - NO GLOBAL_RT.block_on deadlock hazard.
    /// This function is now async and must be awaited by callers in async context.
    /// 
    /// Scans a prompt string for precise tool extraction sequences, evaluates them structurally,
    /// and returns the contextualized payload explicitly devoid of "Agentic loop" race hazards.
    pub async fn evaluate_and_inject(prompt: &str) -> String {
        let start_tag = "<mcp_request>";
        let end_tag = "</mcp_request>";
        
        if let Some(start_idx) = prompt.find(start_tag) {
            if let Some(offset) = prompt[start_idx..].find(end_tag) {
                let end_idx = start_idx + offset;
                let absolute_end = end_idx + end_tag.len();
                let request_payload = &prompt[start_idx + start_tag.len()..end_idx];
                
                eprintln!("[MCP BRIDGE] II Intercepted Tool Sequence. Suspending Cognitive Loop.\n{}", request_payload);
                
                let tool_call: Result<McpToolCall, _> = serde_json::from_str(request_payload);
                match tool_call {
                    Ok(call) => {
                        // Dynamically resolve target command map via standard environment.
                        // Default fallback used in diagnostic execution contexts.
                        let mcp_cmd = std::env::var("SODALS_MCP_EXEC").unwrap_or_else(|_| "npx".to_string());
                        let mcp_args_raw = std::env::var("SODALS_MCP_ARGS").unwrap_or_else(|_| "-y @modelcontextprotocol/server-everything".to_string());
                        let mcp_args: Vec<&str> = mcp_args_raw.split_whitespace().collect();
                        
                        // Phase 33 FIX: Native await - no GLOBAL_RT.block_on deadlock
                        let mcp_output = async {
                            match McpClient::new(&mcp_cmd, &mcp_args).await {
                                Ok(mut client) => {
                                    eprintln!("[MCP BRIDGE] I Link established to {} {}", mcp_cmd, mcp_args_raw);
                                    match client.call_tool(call).await {
                                        Ok(res) => {
                                            if res.is_error {
                                                eprintln!("[MCP BRIDGE] X Tool encountered logical error during execution.");
                                            }
                                            serde_json::to_string_pretty(&res.content).unwrap_or_default()
                                        },
                                        Err(e) => format!("{{\"error\": \"Tool execution failed bounds: {}\"}}", e),
                                    }
                                },
                                Err(e) => format!("{{\"error\": \"Server execution pipeline failed: {}\"}}", e),
                            }
                        }.await;
                        
                        let mut resolved_prompt = String::with_capacity(prompt.len() + mcp_output.len() + 100);
                        resolved_prompt.push_str(&prompt[..start_idx]);
                        resolved_prompt.push_str("<mcp_result>\n");
                        resolved_prompt.push_str(&mcp_output);
                        resolved_prompt.push_str("\n</mcp_result>\n");
                        resolved_prompt.push_str(&prompt[absolute_end..]);
                        
                        eprintln!("[MCP BRIDGE] II Tool execution finalized. Context statically injected.");
                        return resolved_prompt;
                    },
                    Err(e) => {
                        eprintln!("[MCP BRIDGE] X Fatal JSON constraint violation in requested tool payload: {}", e);
                    }
                }
            }
        }
        
        prompt.to_string()
    }
    
    /// Phase 33: Bridge for sync contexts using block_in_place - ONLY use when truly necessary.
    /// Prefer `evaluate_and_inject().await` in async contexts to avoid thread blocking.
    pub fn evaluate_and_inject_sync(prompt: &str) -> String {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(Self::evaluate_and_inject(prompt))
        })
    }
}
