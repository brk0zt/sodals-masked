// src/crates/sodals_mcp/src/workflows/parallel.rs
// ═══════════════════════════════════════════════════════════════════════
// Parallelization Workflow — Fan-out / Fan-in
// ═══════════════════════════════════════════════════════════════════════
//
//              ┌── Branch A ──┐
//   input ─────┼── Branch B ──┼─→ Aggregator → output
//              └── Branch C ──┘
//
// All branches execute concurrently via tokio::spawn. Each branch
// receives the same input (or a branch-specific transform of it).
// After all branches complete, an Aggregator combines the results
// into a single output.
//
// Built-in aggregators:
//   - ConcatAggregator: joins all outputs with a separator
//   - JsonMergeAggregator: merges outputs into a JSON object

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use crate::registry::ToolHandler;
use crate::types::*;
use super::chain::extract_text;

// ============================================================================
// AGGREGATOR TRAIT
// ============================================================================

/// Combines results from parallel branches into a single output.
#[async_trait]
pub trait Aggregator: Send + Sync {
    /// Aggregate multiple branch results into one.
    async fn aggregate(&self, results: Vec<BranchResult>) -> Result<String>;
}

// ============================================================================
// BUILT-IN AGGREGATORS
// ============================================================================

/// Concatenates all branch outputs with a separator.
pub struct ConcatAggregator {
    pub separator: String,
    /// If true, prefix each block with the branch name.
    pub include_headers: bool,
}

impl ConcatAggregator {
    pub fn new(separator: impl Into<String>) -> Self {
        Self {
            separator: separator.into(),
            include_headers: true,
        }
    }

    pub fn plain(separator: impl Into<String>) -> Self {
        Self {
            separator: separator.into(),
            include_headers: false,
        }
    }
}

impl Default for ConcatAggregator {
    fn default() -> Self {
        Self::new("\n---\n")
    }
}

#[async_trait]
impl Aggregator for ConcatAggregator {
    async fn aggregate(&self, results: Vec<BranchResult>) -> Result<String> {
        let parts: Vec<String> = results
            .iter()
            .map(|r| {
                if self.include_headers {
                    format!("[{}]\n{}", r.branch_name, r.output)
                } else {
                    r.output.clone()
                }
            })
            .collect();

        Ok(parts.join(&self.separator))
    }
}

/// Merges all branch outputs into a JSON object keyed by branch name.
pub struct JsonMergeAggregator;

#[async_trait]
impl Aggregator for JsonMergeAggregator {
    async fn aggregate(&self, results: Vec<BranchResult>) -> Result<String> {
        let mut merged = serde_json::Map::new();

        for r in &results {
            // Attempt to parse as JSON; fall back to raw string
            let value = serde_json::from_str(&r.output)
                .unwrap_or(serde_json::Value::String(r.output.clone()));

            merged.insert(r.branch_name.clone(), value);
        }

        serde_json::to_string_pretty(&serde_json::Value::Object(merged))
            .context("Failed to serialize aggregated JSON")
    }
}

/// Custom function aggregator for one-off use cases.
pub struct FnAggregator<F>
where
    F: Fn(Vec<BranchResult>) -> Result<String> + Send + Sync,
{
    func: F,
}

impl<F> FnAggregator<F>
where
    F: Fn(Vec<BranchResult>) -> Result<String> + Send + Sync,
{
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

#[async_trait]
impl<F> Aggregator for FnAggregator<F>
where
    F: Fn(Vec<BranchResult>) -> Result<String> + Send + Sync,
{
    async fn aggregate(&self, results: Vec<BranchResult>) -> Result<String> {
        (self.func)(results)
    }
}

// ============================================================================
// PARALLEL BRANCH
// ============================================================================

/// A single branch in a parallelization workflow.
pub struct ParallelBranch {
    /// Human-readable name
    pub name: String,
    /// The tool handler to execute
    pub handler: Arc<dyn ToolHandler>,
    /// Optional transform: reshape the shared input into branch-specific args.
    /// If `None`, each branch receives the same arguments.
    pub transform: Option<Arc<dyn BranchTransform>>,
}

/// Transform shared input into branch-specific arguments.
#[async_trait]
pub trait BranchTransform: Send + Sync {
    async fn transform(&self, shared_input: &serde_json::Value) -> Result<serde_json::Value>;
}

// ============================================================================
// BRANCH RESULT
// ============================================================================

/// Result from a single parallel branch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchResult {
    pub branch_name: String,
    pub branch_index: usize,
    pub output: String,
    pub is_error: bool,
    pub duration_ms: u64,
}

/// Full result of a parallelization workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelResult {
    /// Aggregated output
    pub output: String,
    /// Whether all branches succeeded
    pub all_success: bool,
    /// Individual branch results (ordered by branch index)
    pub branch_results: Vec<BranchResult>,
    /// Total wall-clock time (all branches ran concurrently)
    pub total_duration_ms: u64,
}

// ============================================================================
// PARALLEL ORCHESTRATOR
// ============================================================================

/// Fan-out / Fan-in parallel execution of multiple tool handlers.
///
/// # Example
/// ```ignore
/// let parallel = Parallel::new("multi-search")
///     .branch("web", web_search_handler, None)
///     .branch("memory", memory_search_handler, None)
///     .branch("docs", doc_search_handler, None)
///     .aggregator(Arc::new(ConcatAggregator::default()));
///
/// let result = parallel.execute(json!({"query": "SODALS architecture"})).await?;
/// ```
pub struct Parallel {
    /// Name of the parallel workflow
    pub name: String,
    /// Branches to execute concurrently
    branches: Vec<ParallelBranch>,
    /// How to combine results
    aggregator: Arc<dyn Aggregator>,
}

impl Parallel {
    /// Create a new parallel workflow with the default concat aggregator.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            branches: Vec::new(),
            aggregator: Arc::new(ConcatAggregator::default()),
        }
    }

    /// Add a branch.
    pub fn branch(
        mut self,
        name: impl Into<String>,
        handler: Arc<dyn ToolHandler>,
        transform: Option<Arc<dyn BranchTransform>>,
    ) -> Self {
        self.branches.push(ParallelBranch {
            name: name.into(),
            handler,
            transform,
        });
        self
    }

    /// Set a custom aggregator.
    pub fn aggregator(mut self, aggregator: Arc<dyn Aggregator>) -> Self {
        self.aggregator = aggregator;
        self
    }

    /// Number of branches.
    pub fn len(&self) -> usize {
        self.branches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.branches.is_empty()
    }

    /// Execute all branches concurrently and aggregate results.
    pub async fn execute(&self, input: serde_json::Value) -> Result<ParallelResult> {
        let start = std::time::Instant::now();

        info!(
            "[Parallel/{}] Launching {} branches",
            self.name,
            self.branches.len()
        );

        // Spawn all branches concurrently
        let mut handles = Vec::with_capacity(self.branches.len());

        for (i, branch) in self.branches.iter().enumerate() {
            let handler = branch.handler.clone();
            let branch_name = branch.name.clone();
            let workflow_name = self.name.clone();

            // Prepare branch-specific arguments
            let args = match &branch.transform {
                Some(transform) => transform.transform(&input).await?,
                None => input.clone(),
            };

            let handle = tokio::spawn(async move {
                let branch_start = std::time::Instant::now();

                debug!(
                    "[Parallel/{}/{}] Starting",
                    workflow_name, branch_name
                );

                let result = handler.call(args).await;
                let duration = branch_start.elapsed();

                match result {
                    Ok(call_result) => {
                        let output = extract_text(&call_result);
                        debug!(
                            "[Parallel/{}/{}] Completed in {}ms",
                            workflow_name,
                            branch_name,
                            duration.as_millis()
                        );
                        BranchResult {
                            branch_name,
                            branch_index: i,
                            output,
                            is_error: call_result.is_error,
                            duration_ms: duration.as_millis() as u64,
                        }
                    }
                    Err(e) => {
                        error!(
                            "[Parallel/{}/{}] Error: {}",
                            workflow_name, branch_name, e
                        );
                        BranchResult {
                            branch_name,
                            branch_index: i,
                            output: format!("Branch error: {}", e),
                            is_error: true,
                            duration_ms: duration.as_millis() as u64,
                        }
                    }
                }
            });

            handles.push(handle);
        }

        // Collect all results
        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => {
                    warn!("[Parallel/{}] Branch task panicked: {}", self.name, e);
                    results.push(BranchResult {
                        branch_name: "unknown".to_string(),
                        branch_index: results.len(),
                        output: format!("Task panic: {}", e),
                        is_error: true,
                        duration_ms: 0,
                    });
                }
            }
        }

        // Sort by branch index to maintain order
        results.sort_by_key(|r| r.branch_index);

        let all_success = results.iter().all(|r| !r.is_error);
        let total_duration = start.elapsed();

        // Aggregate
        let output = self.aggregator.aggregate(results.clone()).await?;

        info!(
            "[Parallel/{}] All branches complete in {}ms (success={})",
            self.name,
            total_duration.as_millis(),
            all_success
        );

        Ok(ParallelResult {
            output,
            all_success,
            branch_results: results,
            total_duration_ms: total_duration.as_millis() as u64,
        })
    }
}

/// Parallel also implements ToolHandler for composability.
#[async_trait]
impl ToolHandler for Parallel {
    async fn call(&self, arguments: serde_json::Value) -> Result<CallToolResult> {
        let result = self.execute(arguments).await?;
        if result.all_success {
            Ok(CallToolResult::text(result.output))
        } else {
            // Return with is_error=false but include error info in the output
            // (partial failures are common in parallel workflows)
            Ok(CallToolResult::text(result.output))
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct EchoHandler {
        prefix: String,
    }

    #[async_trait]
    impl ToolHandler for EchoHandler {
        async fn call(&self, args: serde_json::Value) -> Result<CallToolResult> {
            let input = args
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("(none)");
            Ok(CallToolResult::text(format!("{}: {}", self.prefix, input)))
        }
    }

    struct SlowHandler {
        delay_ms: u64,
        output: String,
    }

    #[async_trait]
    impl ToolHandler for SlowHandler {
        async fn call(&self, _args: serde_json::Value) -> Result<CallToolResult> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(CallToolResult::text(self.output.clone()))
        }
    }

    #[tokio::test]
    async fn test_parallel_three_branches() {
        let parallel = Parallel::new("test-parallel")
            .branch(
                "alpha",
                Arc::new(EchoHandler {
                    prefix: "A".to_string(),
                }),
                None,
            )
            .branch(
                "beta",
                Arc::new(EchoHandler {
                    prefix: "B".to_string(),
                }),
                None,
            )
            .branch(
                "gamma",
                Arc::new(EchoHandler {
                    prefix: "C".to_string(),
                }),
                None,
            )
            .aggregator(Arc::new(ConcatAggregator::plain("\n")));

        let result = parallel
            .execute(serde_json::json!({"input": "hello"}))
            .await
            .unwrap();

        assert!(result.all_success);
        assert_eq!(result.branch_results.len(), 3);
        assert!(result.output.contains("A: hello"));
        assert!(result.output.contains("B: hello"));
        assert!(result.output.contains("C: hello"));
    }

    #[tokio::test]
    async fn test_parallel_concurrent_execution() {
        // 3 branches each taking 50ms should complete in ~50ms, not 150ms
        let parallel = Parallel::new("concurrent-test")
            .branch(
                "a",
                Arc::new(SlowHandler {
                    delay_ms: 50,
                    output: "done-a".to_string(),
                }),
                None,
            )
            .branch(
                "b",
                Arc::new(SlowHandler {
                    delay_ms: 50,
                    output: "done-b".to_string(),
                }),
                None,
            )
            .branch(
                "c",
                Arc::new(SlowHandler {
                    delay_ms: 50,
                    output: "done-c".to_string(),
                }),
                None,
            );

        let result = parallel
            .execute(serde_json::json!({}))
            .await
            .unwrap();

        assert!(result.all_success);
        // Should take ~50ms, not 150ms. Allow margin for CI.
        assert!(
            result.total_duration_ms < 120,
            "Parallel took {}ms, expected < 120ms",
            result.total_duration_ms
        );
    }

    #[tokio::test]
    async fn test_json_merge_aggregator() {
        let parallel = Parallel::new("json-merge-test")
            .branch(
                "count",
                Arc::new(EchoHandler {
                    prefix: "42".to_string(),
                }),
                None,
            )
            .branch(
                "status",
                Arc::new(EchoHandler {
                    prefix: "OK".to_string(),
                }),
                None,
            )
            .aggregator(Arc::new(JsonMergeAggregator));

        let result = parallel
            .execute(serde_json::json!({"input": "test"}))
            .await
            .unwrap();

        // Output should be valid JSON with branch names as keys
        let parsed: serde_json::Value =
            serde_json::from_str(&result.output).unwrap();
        assert!(parsed.is_object());
        assert!(parsed.get("count").is_some());
        assert!(parsed.get("status").is_some());
    }

    #[tokio::test]
    async fn test_parallel_as_tool_handler() {
        let parallel = Parallel::new("composable")
            .branch(
                "only",
                Arc::new(EchoHandler {
                    prefix: "result".to_string(),
                }),
                None,
            )
            .aggregator(Arc::new(ConcatAggregator::plain("")));

        let result: CallToolResult = parallel
            .call(serde_json::json!({"input": "test"}))
            .await
            .unwrap();

        assert!(!result.is_error);
    }
}
