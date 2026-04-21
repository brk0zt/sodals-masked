// src/crates/sodals_mcp/src/workflows/chain.rs
// ═══════════════════════════════════════════════════════════════════════
// Chaining Workflow — Sequential Pipeline
// ═══════════════════════════════════════════════════════════════════════
//
//   input → Step₁ → Step₂ → Step₃ → ... → Stepₙ → output
//
// Each step receives the output of the previous step as its input.
// An optional `transform` function between steps allows reshaping
// the output of step N into the arguments expected by step N+1.
//
// If any step fails (returns is_error=true or Err), the chain halts
// immediately and returns the error with full provenance.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info};

use crate::registry::ToolHandler;
use crate::types::*;

// ============================================================================
// CHAIN STEP
// ============================================================================

/// A single step in a chain pipeline.
///
/// Each step has a handler (any ToolHandler) and an optional transform
/// function that reshapes the previous step's output into arguments
/// for this step.
pub struct ChainStep {
    /// Human-readable name for tracing/debugging
    pub name: String,
    /// The tool handler to execute
    pub handler: Arc<dyn ToolHandler>,
    /// Optional transform: takes the text output of the previous step
    /// and produces the JSON arguments for this step.
    /// If `None`, the previous output is passed as `{"input": "<text>"}`.
    pub transform: Option<Arc<dyn InputTransform>>,
}

/// Transform the output of one step into the input of the next.
#[async_trait]
pub trait InputTransform: Send + Sync {
    /// Transform the previous step's text output into JSON arguments.
    async fn transform(&self, previous_output: &str) -> Result<serde_json::Value>;
}

/// Default transform: wraps text in `{"input": "<text>"}`.
pub struct DefaultTransform;

#[async_trait]
impl InputTransform for DefaultTransform {
    async fn transform(&self, previous_output: &str) -> Result<serde_json::Value> {
        Ok(serde_json::json!({ "input": previous_output }))
    }
}

/// JSON passthrough transform: parses the previous output as JSON
/// and passes it directly as arguments.
pub struct JsonPassthrough;

#[async_trait]
impl InputTransform for JsonPassthrough {
    async fn transform(&self, previous_output: &str) -> Result<serde_json::Value> {
        serde_json::from_str(previous_output)
            .context("Failed to parse previous step output as JSON")
    }
}

/// Custom key mapping transform.
pub struct KeyMapTransform {
    /// The key name to use for the previous output
    pub key: String,
}

#[async_trait]
impl InputTransform for KeyMapTransform {
    async fn transform(&self, previous_output: &str) -> Result<serde_json::Value> {
        Ok(serde_json::json!({ &self.key: previous_output }))
    }
}

// ============================================================================
// CHAIN RESULT
// ============================================================================

/// The result of executing a chain, including intermediate outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainResult {
    /// Final output text
    pub output: String,
    /// Whether the chain completed successfully
    pub success: bool,
    /// Outputs from each step (for debugging/tracing)
    pub step_outputs: Vec<StepOutput>,
    /// Total number of steps executed (may be less than chain length on error)
    pub steps_executed: usize,
}

/// Output of a single step in the chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutput {
    pub step_name: String,
    pub step_index: usize,
    pub output: String,
    pub is_error: bool,
}

// ============================================================================
// CHAIN ORCHESTRATOR
// ============================================================================

/// A sequential pipeline of tool calls.
///
/// # Example
/// ```ignore
/// let chain = Chain::new("solve-and-memorize")
///     .step("solve", solve_handler, None)
///     .step("memorize", memorize_handler, Some(Arc::new(KeyMapTransform { key: "text".into() })));
///
/// let result = chain.execute(json!({"prompt": "What is 2+2?"})).await?;
/// ```
pub struct Chain {
    /// Name of the chain (for logging)
    pub name: String,
    /// Ordered sequence of steps
    steps: Vec<ChainStep>,
    /// If true, execution halts on the first error
    halt_on_error: bool,
}

impl Chain {
    /// Create a new empty chain.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            steps: Vec::new(),
            halt_on_error: true,
        }
    }

    /// Add a step to the chain.
    pub fn step(
        mut self,
        name: impl Into<String>,
        handler: Arc<dyn ToolHandler>,
        transform: Option<Arc<dyn InputTransform>>,
    ) -> Self {
        self.steps.push(ChainStep {
            name: name.into(),
            handler,
            transform,
        });
        self
    }

    /// Set whether to halt on the first error (default: true).
    pub fn halt_on_error(mut self, halt: bool) -> Self {
        self.halt_on_error = halt;
        self
    }

    /// Number of steps in the chain.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Execute the chain with initial arguments.
    pub async fn execute(&self, initial_args: serde_json::Value) -> Result<ChainResult> {
        info!(
            "[Chain/{}] Starting {} steps",
            self.name,
            self.steps.len()
        );

        let mut current_args = initial_args;
        let mut step_outputs = Vec::with_capacity(self.steps.len());
        let mut last_output = String::new();

        for (i, step) in self.steps.iter().enumerate() {
            debug!(
                "[Chain/{}/step-{}] Executing '{}'",
                self.name, i, step.name
            );

            // Execute the step
            let result = step.handler.call(current_args.clone()).await
                .with_context(|| format!("Chain step '{}' (index {}) failed", step.name, i))?;

            // Extract text from the result
            let output_text = extract_text(&result);

            step_outputs.push(StepOutput {
                step_name: step.name.clone(),
                step_index: i,
                output: output_text.clone(),
                is_error: result.is_error,
            });

            // Check for errors
            if result.is_error {
                error!(
                    "[Chain/{}/step-{}] '{}' returned error: {}",
                    self.name, i, step.name, output_text
                );

                if self.halt_on_error {
                    return Ok(ChainResult {
                        output: output_text,
                        success: false,
                        step_outputs,
                        steps_executed: i + 1,
                    });
                }
            }

            last_output = output_text.clone();

            // Prepare arguments for the next step (if there is one)
            if i + 1 < self.steps.len() {
                let next_step = &self.steps[i + 1];
                current_args = match &next_step.transform {
                    Some(transform) => transform.transform(&output_text).await?,
                    None => serde_json::json!({ "input": output_text }),
                };
            }
        }

        info!(
            "[Chain/{}] Completed all {} steps",
            self.name,
            self.steps.len()
        );

        Ok(ChainResult {
            output: last_output,
            success: true,
            step_outputs,
            steps_executed: self.steps.len(),
        })
    }
}

/// Also make Chain usable as a ToolHandler itself (composable).
#[async_trait]
impl ToolHandler for Chain {
    async fn call(&self, arguments: serde_json::Value) -> Result<CallToolResult> {
        let result = self.execute(arguments).await?;
        if result.success {
            Ok(CallToolResult::text(result.output))
        } else {
            Ok(CallToolResult::error(result.output))
        }
    }
}

// ============================================================================
// HELPERS
// ============================================================================

/// Extract text content from a CallToolResult.
pub(crate) fn extract_text(result: &CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|c| match c {
            ToolContent::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    struct UppercaseHandler;

    #[async_trait]
    impl ToolHandler for UppercaseHandler {
        async fn call(&self, args: serde_json::Value) -> Result<CallToolResult> {
            let input = args
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            Ok(CallToolResult::text(input.to_uppercase()))
        }
    }

    struct PrefixHandler {
        prefix: String,
    }

    #[async_trait]
    impl ToolHandler for PrefixHandler {
        async fn call(&self, args: serde_json::Value) -> Result<CallToolResult> {
            let input = args
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            Ok(CallToolResult::text(format!("{}{}", self.prefix, input)))
        }
    }

    struct ErrorHandler;

    #[async_trait]
    impl ToolHandler for ErrorHandler {
        async fn call(&self, _args: serde_json::Value) -> Result<CallToolResult> {
            Ok(CallToolResult::error("Something went wrong"))
        }
    }

    #[tokio::test]
    async fn test_chain_two_steps() {
        let chain = Chain::new("test-chain")
            .step("uppercase", Arc::new(UppercaseHandler), None)
            .step(
                "prefix",
                Arc::new(PrefixHandler {
                    prefix: ">>".to_string(),
                }),
                None,
            );

        let result = chain
            .execute(serde_json::json!({"input": "hello world"}))
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.steps_executed, 2);
        assert_eq!(result.output, ">>HELLO WORLD");
        assert_eq!(result.step_outputs.len(), 2);
        assert_eq!(result.step_outputs[0].output, "HELLO WORLD");
        assert_eq!(result.step_outputs[1].output, ">>HELLO WORLD");
    }

    #[tokio::test]
    async fn test_chain_halt_on_error() {
        let chain = Chain::new("error-chain")
            .step("uppercase", Arc::new(UppercaseHandler), None)
            .step("fail", Arc::new(ErrorHandler), None)
            .step(
                "never-reached",
                Arc::new(PrefixHandler {
                    prefix: "!".to_string(),
                }),
                None,
            );

        let result = chain
            .execute(serde_json::json!({"input": "test"}))
            .await
            .unwrap();

        assert!(!result.success);
        assert_eq!(result.steps_executed, 2); // stopped at step 2
        assert_eq!(result.step_outputs.len(), 2);
    }

    #[tokio::test]
    async fn test_chain_as_tool_handler() {
        let chain = Chain::new("composable")
            .step("uppercase", Arc::new(UppercaseHandler), None);

        // Use Chain as a ToolHandler
        let result: CallToolResult = chain
            .call(serde_json::json!({"input": "compose me"}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(extract_text(&result), "COMPOSE ME");
    }

    #[tokio::test]
    async fn test_chain_with_custom_transform() {
        let chain = Chain::new("custom-transform")
            .step("uppercase", Arc::new(UppercaseHandler), None)
            .step(
                "prefix",
                Arc::new(PrefixHandler {
                    prefix: "PREFIX:".to_string(),
                }),
                Some(Arc::new(KeyMapTransform {
                    key: "input".to_string(),
                })),
            );

        let result = chain
            .execute(serde_json::json!({"input": "hello"}))
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.output, "PREFIX:HELLO");
    }

    #[tokio::test]
    async fn test_empty_chain() {
        let chain = Chain::new("empty");

        let result = chain
            .execute(serde_json::json!({}))
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.steps_executed, 0);
        assert!(result.output.is_empty());
    }
}
