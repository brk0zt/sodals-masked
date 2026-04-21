// src/crates/sodals_mcp/src/workflows/mod.rs
// ═══════════════════════════════════════════════════════════════════════
// Agentic Workflow Orchestration Patterns
// ═══════════════════════════════════════════════════════════════════════
//
// Four fundamental patterns for composing MCP tools into higher-order
// workflows. Each pattern is transport-agnostic and works with any
// ToolHandler implementation.
//
//   ┌─────────────┐
//   │   Chaining   │  A → B → C → D  (sequential pipeline)
//   ├─────────────┤
//   │  Parallel    │  A ──┬── B ──┐
//   │              │      ├── C ──┤  (fan-out / fan-in)
//   │              │      └── D ──┘
//   ├─────────────┤
//   │   Routing    │  classify(input) → route_A | route_B | fallback
//   ├─────────────┤
//   │  Evaluator-  │  produce → grade → feedback → iterate
//   │  Optimizer   │  (until quality ≥ threshold or max iterations)
//   └─────────────┘

pub mod chain;
pub mod parallel;
pub mod router;
pub mod evaluator;

pub use chain::{Chain, ChainStep};
pub use parallel::{Parallel, ParallelBranch, Aggregator, ConcatAggregator, JsonMergeAggregator};
pub use router::{Router, Route, Classifier, KeywordClassifier};
pub use evaluator::{EvaluatorOptimizer, GradeResult};
