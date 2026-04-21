// sodals_mcp — Model Context Protocol server & client for SODALS
//
// Architecture:
//   Host (SODALS Kernel)
//     └─ McpServer  ← external AI agents connect here (stdio / SSE)
//     └─ McpClient  → consumes external MCP tool servers
//     └─ Workflows  → orchestrate tools via agentic patterns
//
// Wire format: JSON-RPC 2.0 over newline-delimited JSON (stdio)
//              or HTTP POST + Server-Sent Events (SSE)

pub mod types;
pub mod transport;
pub mod registry;
pub mod tools;
pub mod server;
pub mod client;
pub mod workflows;

pub use server::McpServer;
pub use client::McpClient;
pub use types::*;

