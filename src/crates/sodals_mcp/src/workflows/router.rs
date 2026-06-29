// src/crates/sodals_mcp/src/workflows/router.rs
// ═══════════════════════════════════════════════════════════════════════
// Routing Workflow — Input Classification + Dispatch
// ═══════════════════════════════════════════════════════════════════════
//
//                    ┌─→ Route "math"     → math_handler
//   input → classify ├─→ Route "code"     → code_handler
//                    ├─→ Route "creative"  → creative_handler
//                    └─→ fallback         → general_handler
//
// The Classifier inspects the input and returns a route key.
// The Router dispatches to the matching Route handler.
// If no route matches, the fallback is used.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::registry::ToolHandler;
use crate::types::*;
use super::chain::extract_text;

// ============================================================================
// CLASSIFIER TRAIT
// ============================================================================

/// Inspects input and produces a route key.
///
/// The route key is matched against the Router's registered routes.
/// Return a key string (e.g. "math", "code", "creative").
#[async_trait]
pub trait Classifier: Send + Sync {
    /// Classify the input and return a route key.
    /// Also returns a confidence score [0.0, 1.0] for tracing.
    async fn classify(&self, input: &serde_json::Value) -> Result<ClassifyResult>;
}

/// Result of classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyResult {
    /// The route key to dispatch to
    pub route_key: String,
    /// Confidence score [0.0, 1.0]
    pub confidence: f64,
    /// Optional reasoning for the classification
    pub reasoning: Option<String>,
}

// ============================================================================
// BUILT-IN CLASSIFIERS
// ============================================================================

/// Keyword-based classifier: scans input text for keyword matches.
///
/// Each route key has a list of trigger keywords. The route whose
/// keywords appear most frequently wins.
pub struct KeywordClassifier {
    /// route_key → list of trigger keywords (lowercase)
    keywords: HashMap<String, Vec<String>>,
    /// Which input field to scan (default: "prompt" or "input")
    input_fields: Vec<String>,
}

impl KeywordClassifier {
    pub fn new() -> Self {
        Self {
            keywords: HashMap::new(),
            input_fields: vec!["prompt".to_string(), "input".to_string(), "query".to_string()],
        }
    }

    /// Add keywords that trigger a specific route.
    pub fn route(mut self, key: impl Into<String>, keywords: Vec<&str>) -> Self {
        self.keywords.insert(
            key.into(),
            keywords.iter().map(|k| k.to_lowercase()).collect(),
        );
        self
    }

    /// Set which input fields to scan.
    pub fn input_fields(mut self, fields: Vec<&str>) -> Self {
        self.input_fields = fields.iter().map(|f| f.to_string()).collect();
        self
    }

    /// Extract text from input JSON by scanning known fields.
    fn extract_input_text(&self, input: &serde_json::Value) -> String {
        // Try each field
        for field in &self.input_fields {
            if let Some(text) = input.get(field).and_then(|v| v.as_str()) {
                return text.to_lowercase();
            }
        }
        // Fallback: serialize the whole input
        input.to_string().to_lowercase()
    }
}

impl Default for KeywordClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Classifier for KeywordClassifier {
    async fn classify(&self, input: &serde_json::Value) -> Result<ClassifyResult> {
        let text = self.extract_input_text(input);
        let mut best_key = String::new();
        let mut best_score = 0usize;

        for (route_key, keywords) in &self.keywords {
            let score: usize = keywords
                .iter()
                .filter(|kw| text.contains(kw.as_str()))
                .count();

            if score > best_score {
                best_score = score;
                best_key = route_key.clone();
            }
        }

        if best_score > 0 {
            let total_keywords = self.keywords.get(&best_key).map_or(1, |k| k.len());
            let confidence = best_score as f64 / total_keywords as f64;

            Ok(ClassifyResult {
                route_key: best_key,
                confidence: confidence.min(1.0),
                reasoning: Some(format!(
                    "Matched {}/{} keywords",
                    best_score, total_keywords
                )),
            })
        } else {
            Ok(ClassifyResult {
                route_key: String::new(),
                confidence: 0.0,
                reasoning: Some("No keywords matched — using fallback".to_string()),
            })
        }
    }
}

/// LLM-based classifier: uses a ToolHandler (e.g., sodals_solve) to classify.
///
/// Sends a classification prompt to the LLM and parses the route key
/// from the response.
pub struct LlmClassifier {
    /// The LLM handler to use for classification
    handler: Arc<dyn ToolHandler>,
    /// Available route keys the LLM should choose from
    route_keys: Vec<String>,
    /// System prompt prefix
    system_prompt: String,
}

impl LlmClassifier {
    pub fn new(handler: Arc<dyn ToolHandler>, route_keys: Vec<String>) -> Self {
        let keys_str = route_keys.join(", ");
        Self {
            handler,
            route_keys: route_keys.clone(),
            system_prompt: format!(
                "Classify the following input into exactly one of these categories: [{}]. \
                 Respond with ONLY the category name, nothing else.",
                keys_str
            ),
        }
    }
}

#[async_trait]
impl Classifier for LlmClassifier {
    async fn classify(&self, input: &serde_json::Value) -> Result<ClassifyResult> {
        let input_text = input
            .get("prompt")
            .or_else(|| input.get("input"))
            .and_then(|v| v.as_str())
            .unwrap_or(&input.to_string())
            .to_string();

        let args = serde_json::json!({
            "prompt": format!("{}\n\nInput: {}", self.system_prompt, input_text)
        });

        let result = self.handler.call(args).await?;
        let response = extract_text(&result).trim().to_lowercase();

        // Find the matching route key
        let matched = self
            .route_keys
            .iter()
            .find(|k| response.contains(&k.to_lowercase()))
            .cloned();

        match matched {
            Some(key) => Ok(ClassifyResult {
                route_key: key,
                confidence: 0.8,
                reasoning: Some(format!("LLM classified as: {}", response)),
            }),
            None => Ok(ClassifyResult {
                route_key: String::new(),
                confidence: 0.0,
                reasoning: Some(format!(
                    "LLM response '{}' did not match any route key",
                    response
                )),
            }),
        }
    }
}

// ============================================================================
// ROUTE
// ============================================================================

/// A single route: a handler with optional input preprocessing.
pub struct Route {
    /// Human-readable name
    pub name: String,
    /// The handler to execute for this route
    pub handler: Arc<dyn ToolHandler>,
    /// Optional preprocessing of input before dispatch.
    /// If None, the original input is passed unchanged.
    pub preprocess: Option<Arc<dyn RoutePreprocess>>,
}

/// Preprocess input before dispatching to a route.
#[async_trait]
pub trait RoutePreprocess: Send + Sync {
    async fn preprocess(&self, input: &serde_json::Value) -> Result<serde_json::Value>;
}

// ============================================================================
// ROUTER RESULT
// ============================================================================

/// Result of a routing decision + dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterResult {
    /// The route that was selected
    pub selected_route: String,
    /// Classification details
    pub classification: ClassifyResult,
    /// The output from the dispatched handler
    pub output: String,
    /// Whether the route succeeded
    pub success: bool,
}

// ============================================================================
// ROUTER ORCHESTRATOR
// ============================================================================

/// Input classification + dispatch to specialized handlers.
///
/// # Example
/// ```ignore
/// let router = Router::new("task-router")
///     .classifier(Arc::new(KeywordClassifier::new()
///         .route("math", vec!["calculate", "compute", "solve", "equation"])
///         .route("code", vec!["write", "function", "implement", "code"])
///     ))
///     .route("math", "Math Handler", math_handler, None)
///     .route("code", "Code Handler", code_handler, None)
///     .fallback("General", general_handler, None);
///
/// let result = router.execute(json!({"prompt": "calculate 2+2"})).await?;
/// assert_eq!(result.selected_route, "math");
/// ```
pub struct Router {
    /// Name for logging
    pub name: String,
    /// The classifier that determines which route to take
    classifier: Option<Arc<dyn Classifier>>,
    /// Map of route_key → Route
    routes: HashMap<String, Route>,
    /// Fallback route when no match is found
    fallback: Option<Route>,
    /// Minimum confidence required to dispatch (default: 0.0)
    min_confidence: f64,
}

impl Router {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            classifier: None,
            routes: HashMap::new(),
            fallback: None,
            min_confidence: 0.0,
        }
    }

    /// Set the classifier.
    pub fn classifier(mut self, classifier: Arc<dyn Classifier>) -> Self {
        self.classifier = Some(classifier);
        self
    }

    /// Add a route.
    pub fn route(
        mut self,
        key: impl Into<String>,
        name: impl Into<String>,
        handler: Arc<dyn ToolHandler>,
        preprocess: Option<Arc<dyn RoutePreprocess>>,
    ) -> Self {
        let key = key.into();
        self.routes.insert(
            key.clone(),
            Route {
                name: name.into(),
                handler,
                preprocess,
            },
        );
        self
    }

    /// Set the fallback route.
    pub fn fallback(
        mut self,
        name: impl Into<String>,
        handler: Arc<dyn ToolHandler>,
        preprocess: Option<Arc<dyn RoutePreprocess>>,
    ) -> Self {
        self.fallback = Some(Route {
            name: name.into(),
            handler,
            preprocess,
        });
        self
    }

    /// Set minimum confidence for dispatch (below this → fallback).
    pub fn min_confidence(mut self, min: f64) -> Self {
        self.min_confidence = min;
        self
    }

    /// Execute: classify input, then dispatch to the matching route.
    pub async fn execute(&self, input: serde_json::Value) -> Result<RouterResult> {
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Router has no classifier configured"))?;

        // Step 1: Classify
        let classification = classifier.classify(&input).await
            .context("Classification failed")?;

        info!(
            "[Router/{}] Classified as '{}' (confidence={:.2})",
            self.name, classification.route_key, classification.confidence
        );

        // Step 2: Find the route
        let use_fallback = classification.route_key.is_empty()
            || classification.confidence < self.min_confidence
            || !self.routes.contains_key(&classification.route_key);

        let route = if use_fallback {
            debug!(
                "[Router/{}] Using fallback (key='{}', conf={:.2}, min={:.2})",
                self.name, classification.route_key, classification.confidence, self.min_confidence
            );
            self.fallback.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "No matching route '{}' and no fallback configured",
                    classification.route_key
                )
            })?
        } else {
            self.routes.get(&classification.route_key).unwrap()
        };

        let route_name = route.name.clone();

        info!(
            "[Router/{}] Dispatching to route '{}'",
            self.name, route_name
        );

        // Step 3: Preprocess input if needed
        let args = match &route.preprocess {
            Some(preprocess) => preprocess.preprocess(&input).await?,
            None => input,
        };

        // Step 4: Execute
        let result = route.handler.call(args).await?;
        let output = extract_text(&result);

        Ok(RouterResult {
            selected_route: route_name,
            classification,
            output,
            success: !result.is_error,
        })
    }
}

/// Router also implements ToolHandler for composability.
#[async_trait]
impl ToolHandler for Router {
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
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    struct TypedHandler {
        response: String,
    }

    #[async_trait]
    impl ToolHandler for TypedHandler {
        async fn call(&self, _args: serde_json::Value) -> Result<CallToolResult> {
            Ok(CallToolResult::text(self.response.clone()))
        }
    }

    #[tokio::test]
    async fn test_keyword_router_math() {
        let router = Router::new("test-router")
            .classifier(Arc::new(
                KeywordClassifier::new()
                    .route("math", vec!["calculate", "compute", "solve", "equation"])
                    .route("code", vec!["write", "function", "implement", "code"]),
            ))
            .route(
                "math",
                "Math Handler",
                Arc::new(TypedHandler {
                    response: "MATH_RESULT".to_string(),
                }),
                None,
            )
            .route(
                "code",
                "Code Handler",
                Arc::new(TypedHandler {
                    response: "CODE_RESULT".to_string(),
                }),
                None,
            )
            .fallback(
                "General",
                Arc::new(TypedHandler {
                    response: "GENERAL_RESULT".to_string(),
                }),
                None,
            );

        let result = router
            .execute(serde_json::json!({"prompt": "calculate the equation 2+2"}))
            .await
            .unwrap();

        assert_eq!(result.selected_route, "Math Handler");
        assert_eq!(result.output, "MATH_RESULT");
        assert!(result.classification.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_keyword_router_code() {
        let router = Router::new("test-router")
            .classifier(Arc::new(
                KeywordClassifier::new()
                    .route("math", vec!["calculate", "compute"])
                    .route("code", vec!["write", "function", "implement"]),
            ))
            .route(
                "math",
                "Math",
                Arc::new(TypedHandler {
                    response: "MATH".to_string(),
                }),
                None,
            )
            .route(
                "code",
                "Code",
                Arc::new(TypedHandler {
                    response: "CODE".to_string(),
                }),
                None,
            )
            .fallback(
                "General",
                Arc::new(TypedHandler {
                    response: "GENERAL".to_string(),
                }),
                None,
            );

        let result = router
            .execute(serde_json::json!({"prompt": "write a function to implement sorting"}))
            .await
            .unwrap();

        assert_eq!(result.selected_route, "Code");
        assert_eq!(result.output, "CODE");
    }

    #[tokio::test]
    async fn test_keyword_router_fallback() {
        let router = Router::new("fallback-test")
            .classifier(Arc::new(
                KeywordClassifier::new()
                    .route("math", vec!["calculate"]),
            ))
            .route(
                "math",
                "Math",
                Arc::new(TypedHandler {
                    response: "MATH".to_string(),
                }),
                None,
            )
            .fallback(
                "General",
                Arc::new(TypedHandler {
                    response: "FALLBACK".to_string(),
                }),
                None,
            );

        let result = router
            .execute(serde_json::json!({"prompt": "tell me a joke"}))
            .await
            .unwrap();

        assert_eq!(result.selected_route, "General");
        assert_eq!(result.output, "FALLBACK");
    }

    #[tokio::test]
    async fn test_min_confidence_threshold() {
        let router = Router::new("confidence-test")
            .classifier(Arc::new(
                KeywordClassifier::new()
                    .route("specific", vec!["very", "specific", "unique", "rare", "keyword"]),
            ))
            .route(
                "specific",
                "Specific",
                Arc::new(TypedHandler {
                    response: "SPECIFIC".to_string(),
                }),
                None,
            )
            .fallback(
                "General",
                Arc::new(TypedHandler {
                    response: "GENERAL".to_string(),
                }),
                None,
            )
            .min_confidence(0.5);

        // Only matches 1/5 keywords → confidence = 0.2 < 0.5 → fallback
        let result = router
            .execute(serde_json::json!({"prompt": "a very normal request"}))
            .await
            .unwrap();

        assert_eq!(result.selected_route, "General");
    }

    #[tokio::test]
    async fn test_router_as_tool_handler() {
        let router = Router::new("composable")
            .classifier(Arc::new(
                KeywordClassifier::new().route("any", vec!["test"]),
            ))
            .route(
                "any",
                "Any",
                Arc::new(TypedHandler {
                    response: "ROUTED".to_string(),
                }),
                None,
            )
            .fallback(
                "Fallback",
                Arc::new(TypedHandler {
                    response: "FB".to_string(),
                }),
                None,
            );

        let result: CallToolResult = router
            .call(serde_json::json!({"input": "test input"}))
            .await
            .unwrap();

        assert!(!result.is_error);
    }
}
