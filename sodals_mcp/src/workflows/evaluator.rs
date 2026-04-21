// src/crates/sodals_mcp/src/workflows/evaluator.rs
// ═══════════════════════════════════════════════════════════════════════
// Evaluator-Optimizer Pattern (Producer-Grader-Feedback-Iteration)
// ═══════════════════════════════════════════════════════════════════════
//
//   ┌──────────────────────────────────────────────────┐
//   │                                                  │
//   │  ┌───────────┐   ┌─────────┐   ┌────────────┐  │
//   │  │  Producer  │──→│ Grader  │──→│  Feedback   │──┘
//   │  │ (generate) │   │ (eval)  │   │ (if needed) │
//   │  └───────────┘   └─────────┘   └────────────┘
//   │        ↑                              │
//   │        └──────────────────────────────┘
//   │                  iterate
//   │
//   └──→ quality ≥ threshold OR max_iterations → DONE
//
// The Producer generates an output. The Grader evaluates it and returns
// a quality score + feedback. If the score is below the threshold,
// the feedback is fed back to the Producer for refinement. This loop
// continues until quality is met or max iterations are exhausted.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::registry::ToolHandler;
use crate::types::*;
use super::chain::extract_text;

// ============================================================================
// GRADER TRAIT
// ============================================================================

/// Evaluates an output and produces a quality score + feedback.
#[async_trait]
pub trait Grader: Send + Sync {
    /// Grade the output.
    ///
    /// Returns a `GradeResult` with a score [0.0, 1.0] and optional feedback.
    async fn grade(
        &self,
        original_input: &serde_json::Value,
        output: &str,
        iteration: usize,
    ) -> Result<GradeResult>;
}

/// Result of grading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradeResult {
    /// Quality score — 0.0 (terrible) to 1.0 (perfect)
    pub score: f64,
    /// Whether the output passes the quality bar
    pub passed: bool,
    /// Textual feedback for the producer to improve
    pub feedback: String,
    /// Structured critique (optional) — key-value pairs
    #[serde(default)]
    pub criteria: Vec<CriterionResult>,
}

/// Result for a single evaluation criterion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionResult {
    pub name: String,
    pub score: f64,
    pub comment: String,
}

// ============================================================================
// BUILT-IN GRADERS
// ============================================================================

/// Rule-based grader that checks multiple criteria.
pub struct RuleBasedGrader {
    criteria: Vec<GraderRule>,
}

/// A single grading rule.
pub struct GraderRule {
    pub name: String,
    pub check: Arc<dyn Fn(&str) -> (f64, String) + Send + Sync>,
    pub weight: f64,
}

impl RuleBasedGrader {
    pub fn new() -> Self {
        Self {
            criteria: Vec::new(),
        }
    }

    /// Add a grading rule.
    pub fn rule(
        mut self,
        name: impl Into<String>,
        weight: f64,
        check: impl Fn(&str) -> (f64, String) + Send + Sync + 'static,
    ) -> Self {
        self.criteria.push(GraderRule {
            name: name.into(),
            check: Arc::new(check),
            weight,
        });
        self
    }

    /// Add common rules for checking minimum length, keyword presence, etc.
    pub fn min_length(self, min: usize) -> Self {
        self.rule("min_length", 0.3, move |output| {
            if output.len() >= min {
                (1.0, format!("Length {} >= {} ✓", output.len(), min))
            } else {
                (
                    output.len() as f64 / min as f64,
                    format!("Too short: {} < {} characters", output.len(), min),
                )
            }
        })
    }

    pub fn must_contain(self, keyword: &str) -> Self {
        let keyword = keyword.to_string();
        let kw_display = keyword.clone();
        self.rule(
            format!("contains_{}", kw_display),
            0.2,
            move |output| {
                if output.to_lowercase().contains(&keyword.to_lowercase()) {
                    (1.0, format!("Contains '{}' ✓", keyword))
                } else {
                    (0.0, format!("Missing required keyword '{}'", keyword))
                }
            },
        )
    }

    pub fn no_error_markers(self) -> Self {
        self.rule("no_errors", 0.5, |output| {
            let error_markers = ["error", "failed", "exception", "panic", "invalid"];
            let lower = output.to_lowercase();
            let found: Vec<&str> = error_markers
                .iter()
                .filter(|m| lower.contains(**m))
                .copied()
                .collect();

            if found.is_empty() {
                (1.0, "No error markers found ✓".to_string())
            } else {
                (0.0, format!("Found error markers: {:?}", found))
            }
        })
    }
}

impl Default for RuleBasedGrader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Grader for RuleBasedGrader {
    async fn grade(
        &self,
        _original_input: &serde_json::Value,
        output: &str,
        _iteration: usize,
    ) -> Result<GradeResult> {
        let mut criteria_results = Vec::with_capacity(self.criteria.len());
        let mut total_weight = 0.0;
        let mut weighted_score = 0.0;
        let mut feedback_parts = Vec::new();

        for rule in &self.criteria {
            let (score, comment) = (rule.check)(output);

            criteria_results.push(CriterionResult {
                name: rule.name.clone(),
                score,
                comment: comment.clone(),
            });

            weighted_score += score * rule.weight;
            total_weight += rule.weight;

            if score < 1.0 {
                feedback_parts.push(format!("[{}] {}", rule.name, comment));
            }
        }

        let final_score = if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            1.0
        };

        let feedback = if feedback_parts.is_empty() {
            "All criteria passed.".to_string()
        } else {
            format!("Improvements needed:\n{}", feedback_parts.join("\n"))
        };

        Ok(GradeResult {
            score: final_score,
            passed: final_score >= 0.8, // Default pass threshold
            feedback,
            criteria: criteria_results,
        })
    }
}

/// LLM-based grader that uses a ToolHandler for evaluation.
pub struct LlmGrader {
    /// The LLM handler used for grading
    handler: Arc<dyn ToolHandler>,
    /// The grading prompt template
    grading_prompt: String,
    /// Quality threshold
    threshold: f64,
}

impl LlmGrader {
    pub fn new(handler: Arc<dyn ToolHandler>) -> Self {
        Self {
            handler,
            grading_prompt: "Evaluate the following output on a scale of 0.0 to 1.0 for quality, \
                correctness, and completeness. Respond with ONLY a JSON object: \
                {\"score\": <float>, \"feedback\": \"<improvement suggestions>\"}".to_string(),
            threshold: 0.8,
        }
    }

    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.grading_prompt = prompt.into();
        self
    }

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }
}

#[async_trait]
impl Grader for LlmGrader {
    async fn grade(
        &self,
        original_input: &serde_json::Value,
        output: &str,
        iteration: usize,
    ) -> Result<GradeResult> {
        let prompt = format!(
            "{}\n\n--- Original Input ---\n{}\n\n--- Output (Iteration {}) ---\n{}",
            self.grading_prompt,
            serde_json::to_string_pretty(original_input).unwrap_or_default(),
            iteration,
            output
        );

        let result = self
            .handler
            .call(serde_json::json!({ "prompt": prompt }))
            .await?;
        let response = extract_text(&result);

        // Try to parse as JSON
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response) {
            let score = parsed
                .get("score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);

            let feedback = parsed
                .get("feedback")
                .and_then(|v| v.as_str())
                .unwrap_or("No feedback provided")
                .to_string();

            Ok(GradeResult {
                score,
                passed: score >= self.threshold,
                feedback,
                criteria: vec![],
            })
        } else {
            // Fallback: couldn't parse response
            warn!("LLM grader response was not valid JSON: {}", response);
            Ok(GradeResult {
                score: 0.5,
                passed: false,
                feedback: format!("Raw grader response: {}", response),
                criteria: vec![],
            })
        }
    }
}

// ============================================================================
// FEEDBACK BUILDER
// ============================================================================

/// Constructs the feedback prompt for the next iteration.
#[async_trait]
pub trait FeedbackBuilder: Send + Sync {
    /// Build the arguments for the next producer call, incorporating grader feedback.
    async fn build_feedback(
        &self,
        original_input: &serde_json::Value,
        previous_output: &str,
        grade: &GradeResult,
        iteration: usize,
    ) -> Result<serde_json::Value>;
}

/// Default feedback builder: wraps everything into a refinement prompt.
pub struct DefaultFeedbackBuilder;

#[async_trait]
impl FeedbackBuilder for DefaultFeedbackBuilder {
    async fn build_feedback(
        &self,
        original_input: &serde_json::Value,
        previous_output: &str,
        grade: &GradeResult,
        iteration: usize,
    ) -> Result<serde_json::Value> {
        let original_prompt = original_input
            .get("prompt")
            .or_else(|| original_input.get("input"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let prompt = format!(
            "REVISION REQUEST (Iteration {})\n\n\
             Original task: {}\n\n\
             Previous attempt (score: {:.2}/1.0):\n{}\n\n\
             Feedback from evaluator:\n{}\n\n\
             Please produce an improved version addressing the feedback above.",
            iteration,
            original_prompt,
            grade.score,
            previous_output,
            grade.feedback
        );

        Ok(serde_json::json!({ "prompt": prompt }))
    }
}

// ============================================================================
// ITERATION RESULT
// ============================================================================

/// Result of a single iteration in the evaluator-optimizer loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationResult {
    pub iteration: usize,
    pub output: String,
    pub grade: GradeResult,
}

/// Full result of the evaluator-optimizer workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalOptResult {
    /// Final output (from the best-scoring iteration)
    pub output: String,
    /// Whether the quality threshold was met
    pub passed: bool,
    /// Score of the final output
    pub final_score: f64,
    /// Total iterations executed
    pub iterations: usize,
    /// History of all iterations
    pub history: Vec<IterationResult>,
}

// ============================================================================
// EVALUATOR-OPTIMIZER ORCHESTRATOR
// ============================================================================

/// The Evaluator-Optimizer pattern: produce → grade → feedback → iterate.
///
/// # Example
/// ```ignore
/// let eval_opt = EvaluatorOptimizer::new("code-refiner")
///     .producer(Arc::new(code_gen_handler))
///     .grader(Arc::new(
///         RuleBasedGrader::new()
///             .min_length(100)
///             .no_error_markers()
///     ))
///     .max_iterations(3)
///     .quality_threshold(0.85);
///
/// let result = eval_opt.execute(json!({"prompt": "Write a sorting function"})).await?;
/// ```
pub struct EvaluatorOptimizer {
    /// Name for logging
    pub name: String,
    /// The producer tool handler
    producer: Option<Arc<dyn ToolHandler>>,
    /// The grader
    grader: Option<Arc<dyn Grader>>,
    /// Feedback builder (constructs the refinement prompt)
    feedback_builder: Arc<dyn FeedbackBuilder>,
    /// Maximum number of iterations
    max_iterations: usize,
    /// Quality threshold [0.0, 1.0] — stop iterating when met
    quality_threshold: f64,
    /// If true, return the best output across all iterations even if threshold wasn't met
    return_best: bool,
}

impl EvaluatorOptimizer {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            producer: None,
            grader: None,
            feedback_builder: Arc::new(DefaultFeedbackBuilder),
            max_iterations: 3,
            quality_threshold: 0.8,
            return_best: true,
        }
    }

    /// Set the producer handler.
    pub fn producer(mut self, producer: Arc<dyn ToolHandler>) -> Self {
        self.producer = Some(producer);
        self
    }

    /// Set the grader.
    pub fn grader(mut self, grader: Arc<dyn Grader>) -> Self {
        self.grader = Some(grader);
        self
    }

    /// Set the feedback builder.
    pub fn feedback_builder(mut self, builder: Arc<dyn FeedbackBuilder>) -> Self {
        self.feedback_builder = builder;
        self
    }

    /// Set the maximum number of iterations (default: 3).
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set the quality threshold (default: 0.8).
    pub fn quality_threshold(mut self, threshold: f64) -> Self {
        self.quality_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Whether to return the best output even if threshold wasn't met (default: true).
    pub fn return_best(mut self, val: bool) -> Self {
        self.return_best = val;
        self
    }

    /// Execute the evaluator-optimizer loop.
    pub async fn execute(&self, input: serde_json::Value) -> Result<EvalOptResult> {
        let producer = self
            .producer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("EvaluatorOptimizer: no producer configured"))?;
        let grader = self
            .grader
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("EvaluatorOptimizer: no grader configured"))?;

        info!(
            "[EvalOpt/{}] Starting (max_iter={}, threshold={:.2})",
            self.name, self.max_iterations, self.quality_threshold
        );

        let mut history = Vec::with_capacity(self.max_iterations);
        let mut current_args = input.clone();
        let mut best_output = String::new();
        let mut best_score: f64 = -1.0;

        for iteration in 0..self.max_iterations {
            // Step 1: Produce
            info!(
                "[EvalOpt/{}/iter-{}] Producing...",
                self.name, iteration
            );

            let produce_result = producer.call(current_args.clone()).await
                .with_context(|| format!("Producer failed at iteration {}", iteration))?;
            let output = extract_text(&produce_result);

            if produce_result.is_error {
                warn!(
                    "[EvalOpt/{}/iter-{}] Producer returned error: {}",
                    self.name, iteration, output
                );
            }

            // Step 2: Grade
            debug!(
                "[EvalOpt/{}/iter-{}] Grading (output_len={})",
                self.name, iteration, output.len()
            );

            let grade = grader
                .grade(&input, &output, iteration)
                .await
                .with_context(|| format!("Grader failed at iteration {}", iteration))?;

            info!(
                "[EvalOpt/{}/iter-{}] Score: {:.2} (passed={}) — {}",
                self.name, iteration, grade.score, grade.passed,
                if grade.feedback.len() > 80 {
                    format!("{}...", &grade.feedback[..80])
                } else {
                    grade.feedback.clone()
                }
            );

            // Track best
            if grade.score > best_score {
                best_score = grade.score;
                best_output = output.clone();
            }

            history.push(IterationResult {
                iteration,
                output: output.clone(),
                grade: grade.clone(),
            });

            // Step 3: Check if we passed
            if grade.score >= self.quality_threshold {
                info!(
                    "[EvalOpt/{}] ✅ Quality threshold met at iteration {} (score={:.2})",
                    self.name, iteration, grade.score
                );

                return Ok(EvalOptResult {
                    output,
                    passed: true,
                    final_score: grade.score,
                    iterations: iteration + 1,
                    history,
                });
            }

            // Step 4: Build feedback for next iteration (if not last)
            if iteration + 1 < self.max_iterations {
                info!(
                    "[EvalOpt/{}/iter-{}] Below threshold ({:.2} < {:.2}), building feedback...",
                    self.name, iteration, grade.score, self.quality_threshold
                );

                current_args = self
                    .feedback_builder
                    .build_feedback(&input, &output, &grade, iteration + 1)
                    .await
                    .context("Feedback builder failed")?;
            }
        }

        // Exhausted iterations
        warn!(
            "[EvalOpt/{}] Max iterations ({}) exhausted. Best score: {:.2}",
            self.name, self.max_iterations, best_score
        );

        let final_output = if self.return_best {
            best_output
        } else {
            history
                .last()
                .map(|r| r.output.clone())
                .unwrap_or_default()
        };

        Ok(EvalOptResult {
            output: final_output,
            passed: false,
            final_score: best_score,
            iterations: self.max_iterations,
            history,
        })
    }
}

/// EvaluatorOptimizer also implements ToolHandler for composability.
#[async_trait]
impl ToolHandler for EvaluatorOptimizer {
    async fn call(&self, arguments: serde_json::Value) -> Result<CallToolResult> {
        let result = self.execute(arguments).await?;
        if result.passed {
            Ok(CallToolResult::text(result.output))
        } else {
            // Still return the best output — partial success is better than nothing
            Ok(CallToolResult::text(format!(
                "[Best of {} iterations, score={:.2}]\n{}",
                result.iterations, result.final_score, result.output
            )))
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A producer that gets better over iterations.
    struct ImprovingProducer {
        call_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ToolHandler for ImprovingProducer {
        async fn call(&self, args: serde_json::Value) -> Result<CallToolResult> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);

            // Each iteration produces a longer, better output
            let output = match count {
                0 => "short".to_string(),
                1 => "a medium-length answer that addresses the question".to_string(),
                _ => "a comprehensive, well-structured answer that thoroughly \
                      addresses the original question with detailed explanations \
                      and relevant examples to support the response"
                    .to_string(),
            };

            Ok(CallToolResult::text(output))
        }
    }

    /// A producer that always produces the same output.
    struct ConstantProducer {
        output: String,
    }

    #[async_trait]
    impl ToolHandler for ConstantProducer {
        async fn call(&self, _args: serde_json::Value) -> Result<CallToolResult> {
            Ok(CallToolResult::text(self.output.clone()))
        }
    }

    #[tokio::test]
    async fn test_eval_opt_passes_on_first_try() {
        let eval_opt = EvaluatorOptimizer::new("first-try")
            .producer(Arc::new(ConstantProducer {
                output: "This is a perfect, comprehensive answer.".to_string(),
            }))
            .grader(Arc::new(
                RuleBasedGrader::new()
                    .min_length(10)
                    .no_error_markers(),
            ))
            .max_iterations(3)
            .quality_threshold(0.8);

        let result = eval_opt
            .execute(serde_json::json!({"prompt": "test"}))
            .await
            .unwrap();

        assert!(result.passed);
        assert_eq!(result.iterations, 1);
        assert!(result.final_score >= 0.8);
    }

    #[tokio::test]
    async fn test_eval_opt_improves_over_iterations() {
        let counter = Arc::new(AtomicUsize::new(0));

        let eval_opt = EvaluatorOptimizer::new("improving")
            .producer(Arc::new(ImprovingProducer {
                call_count: counter.clone(),
            }))
            .grader(Arc::new(
                RuleBasedGrader::new().min_length(50),
            ))
            .max_iterations(5)
            .quality_threshold(0.8);

        let result = eval_opt
            .execute(serde_json::json!({"prompt": "answer the question"}))
            .await
            .unwrap();

        // Should pass on iteration 2 or 3 (medium or long answer)
        assert!(result.passed);
        assert!(result.iterations <= 3);
        assert!(result.history.len() >= 2);

        // Scores should improve across iterations
        if result.history.len() >= 2 {
            assert!(
                result.history[1].grade.score >= result.history[0].grade.score,
                "Score should improve: iter0={:.2} iter1={:.2}",
                result.history[0].grade.score,
                result.history[1].grade.score
            );
        }
    }

    #[tokio::test]
    async fn test_eval_opt_exhausts_iterations() {
        let eval_opt = EvaluatorOptimizer::new("unreachable")
            .producer(Arc::new(ConstantProducer {
                output: "short".to_string(),
            }))
            .grader(Arc::new(
                RuleBasedGrader::new().min_length(1000), // Will never pass
            ))
            .max_iterations(3)
            .quality_threshold(0.9);

        let result = eval_opt
            .execute(serde_json::json!({"prompt": "impossible"}))
            .await
            .unwrap();

        assert!(!result.passed);
        assert_eq!(result.iterations, 3);
        assert_eq!(result.history.len(), 3);
    }

    #[tokio::test]
    async fn test_rule_based_grader() {
        let grader = RuleBasedGrader::new()
            .min_length(20)
            .must_contain("answer")
            .no_error_markers();

        // Good output
        let good = grader
            .grade(&serde_json::json!({}), "Here is a detailed answer to your question with more content", 0)
            .await
            .unwrap();

        assert!(good.score > 0.8, "Good output scored: {:.2}", good.score);
        assert!(!good.criteria.is_empty());

        // Bad output
        let bad = grader
            .grade(&serde_json::json!({}), "error", 0)
            .await
            .unwrap();

        assert!(bad.score < 0.5, "Bad output scored: {:.2}", bad.score);
    }

    #[tokio::test]
    async fn test_eval_opt_as_tool_handler() {
        let eval_opt = EvaluatorOptimizer::new("composable")
            .producer(Arc::new(ConstantProducer {
                output: "A great comprehensive output that meets all criteria.".to_string(),
            }))
            .grader(Arc::new(RuleBasedGrader::new().min_length(10)))
            .max_iterations(2)
            .quality_threshold(0.8);

        let result: CallToolResult = eval_opt
            .call(serde_json::json!({"prompt": "test"}))
            .await
            .unwrap();

        assert!(!result.is_error);
        let text = extract_text(&result);
        assert!(!text.is_empty());
    }

    #[tokio::test]
    async fn test_feedback_builder_default() {
        let builder = DefaultFeedbackBuilder;

        let grade = GradeResult {
            score: 0.4,
            passed: false,
            feedback: "Needs more detail.".to_string(),
            criteria: vec![],
        };

        let feedback = builder
            .build_feedback(
                &serde_json::json!({"prompt": "Write a poem"}),
                "Roses are red",
                &grade,
                2,
            )
            .await
            .unwrap();

        let prompt = feedback
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap();

        assert!(prompt.contains("REVISION REQUEST"));
        assert!(prompt.contains("Iteration 2"));
        assert!(prompt.contains("Roses are red"));
        assert!(prompt.contains("Needs more detail"));
        assert!(prompt.contains("0.40"));
    }
}
