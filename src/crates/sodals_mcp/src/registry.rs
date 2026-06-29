// src/crates/sodals_mcp/src/registry.rs
// ═══════════════════════════════════════════════════════════════════════
// MCP Tool / Resource / Prompt Registries
// ═══════════════════════════════════════════════════════════════════════

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use crate::types::*;

// ============================================================================
// TOOL HANDLER TRAIT
// ============================================================================

/// A handler that can execute an MCP tool call.
///
/// Implementations bridge the MCP call into the SODALS kernel
/// by creating the appropriate `SystemMsg` and waiting for a response.
#[async_trait]
pub trait ToolHandler: Send + Sync {
    /// Execute the tool with the given arguments.
    async fn call(&self, arguments: serde_json::Value) -> Result<CallToolResult>;
}

// ============================================================================
// TOOL REGISTRY
// ============================================================================

/// A registered tool: metadata + handler.
pub struct ToolEntry {
    /// MCP tool definition (name, description, input schema)
    pub definition: Tool,
    /// The handler that executes the tool
    pub handler: Arc<dyn ToolHandler>,
}

/// Registry of all MCP tools available on this server.
pub struct ToolRegistry {
    tools: HashMap<String, ToolEntry>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool with its handler.
    pub fn register(&mut self, definition: Tool, handler: Arc<dyn ToolHandler>) {
        let name = definition.name.clone();
        self.tools.insert(
            name,
            ToolEntry {
                definition,
                handler,
            },
        );
    }

    /// List all registered tools.
    pub fn list(&self) -> Vec<Tool> {
        self.tools.values().map(|e| e.definition.clone()).collect()
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<&ToolEntry> {
        self.tools.get(name)
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// RESOURCE REGISTRY
// ============================================================================

/// A registered resource.
pub struct ResourceEntry {
    pub definition: Resource,
    pub handler: Arc<dyn ResourceHandler>,
}

/// Handler for reading a resource.
#[async_trait]
pub trait ResourceHandler: Send + Sync {
    /// Read the resource content.
    async fn read(&self, uri: &str) -> Result<Vec<ResourceContent>>;
}

/// Registry of all MCP resources available on this server.
pub struct ResourceRegistry {
    resources: HashMap<String, ResourceEntry>,
    templates: Vec<ResourceTemplate>,
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
            templates: Vec::new(),
        }
    }

    /// Register a static resource.
    pub fn register(&mut self, definition: Resource, handler: Arc<dyn ResourceHandler>) {
        let uri = definition.uri.clone();
        self.resources.insert(
            uri,
            ResourceEntry {
                definition,
                handler,
            },
        );
    }

    /// Register a resource template (dynamic URI pattern).
    pub fn register_template(&mut self, template: ResourceTemplate) {
        self.templates.push(template);
    }

    /// List all registered resources.
    pub fn list(&self) -> Vec<Resource> {
        self.resources
            .values()
            .map(|e| e.definition.clone())
            .collect()
    }

    /// List all resource templates.
    pub fn list_templates(&self) -> &[ResourceTemplate] {
        &self.templates
    }

    /// Look up a resource by URI.
    pub fn get(&self, uri: &str) -> Option<&ResourceEntry> {
        self.resources.get(uri)
    }

    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }
}

impl Default for ResourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PROMPT REGISTRY
// ============================================================================

/// Handler for resolving a prompt template.
#[async_trait]
pub trait PromptHandler: Send + Sync {
    /// Resolve the prompt with the given arguments.
    async fn get(&self, arguments: HashMap<String, String>) -> Result<GetPromptResult>;
}

/// A registered prompt.
pub struct PromptEntry {
    pub definition: Prompt,
    pub handler: Arc<dyn PromptHandler>,
}

/// Registry of all MCP prompts available on this server.
pub struct PromptRegistry {
    prompts: HashMap<String, PromptEntry>,
}

impl PromptRegistry {
    pub fn new() -> Self {
        Self {
            prompts: HashMap::new(),
        }
    }

    pub fn register(&mut self, definition: Prompt, handler: Arc<dyn PromptHandler>) {
        let name = definition.name.clone();
        self.prompts.insert(
            name,
            PromptEntry {
                definition,
                handler,
            },
        );
    }

    pub fn list(&self) -> Vec<Prompt> {
        self.prompts
            .values()
            .map(|e| e.definition.clone())
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<&PromptEntry> {
        self.prompts.get(name)
    }

    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }
}

impl Default for PromptRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct EchoHandler;

    #[async_trait]
    impl ToolHandler for EchoHandler {
        async fn call(&self, arguments: serde_json::Value) -> Result<CallToolResult> {
            Ok(CallToolResult::text(format!("Echo: {}", arguments)))
        }
    }

    #[test]
    fn test_tool_registry_register_and_list() {
        let mut registry = ToolRegistry::new();

        let tool = Tool {
            name: "echo".to_string(),
            description: Some("Echoes input".to_string()),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: None,
                required: None,
            },
        };

        registry.register(tool, Arc::new(EchoHandler));

        assert_eq!(registry.len(), 1);
        assert!(registry.get("echo").is_some());
        assert!(registry.get("nonexistent").is_none());

        let tools = registry.list();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
    }

    #[tokio::test]
    async fn test_tool_handler_call() {
        let handler = EchoHandler;
        let result = handler
            .call(serde_json::json!({"message": "hello"}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
    }
}
