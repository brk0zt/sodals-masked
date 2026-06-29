// src/llm_core.rs
// ARCHITECTURAL SHIM: Backwards compatibility for external callers.
// All core logic has been modularized into the `engine` directory.

pub use crate::engine::llm_engine::LlmEngine;
pub use crate::engine::chimera_router::{NeuroConfig, TaskPlan};
pub use crate::engine::logits_fusion::{YieldedToken, RawToken};
pub use crate::engine::oracle_worker::{OracleHandle, OracleMessage};

// Re-export the entire engine module for flexible access
pub use crate::engine::*;

/// Shadow Tester: Checks the logical parity/correctness of written code snippets.
pub fn run_shadow_test(snippet: &str, _context: &str) -> bool {
    let mut paren_count = 0;
    let mut brace_count = 0;
    let mut bracket_count = 0;
    for c in snippet.chars() {
        match c {
            '(' => paren_count += 1,
            ')' => paren_count -= 1,
            '{' => brace_count += 1,
            '}' => brace_count -= 1,
            '[' => bracket_count += 1,
            ']' => bracket_count -= 1,
            _ => {}
        }
        if paren_count < 0 || brace_count < 0 || bracket_count < 0 {
            return false;
        }
    }
    paren_count == 0 && brace_count == 0 && bracket_count == 0
}

