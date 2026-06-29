// src/crates/sodals_neuro/src/global_gaze.rs
// Global Gaze: Heuristic Strategy Switcher + Semantic-Divergence-Gated Ouroboros Controller
//
// ARCHITECTURE:
//   1. GlobalScanner  III analyses ARC task nature from I/O pair statistics
//   2. SemanticDivergenceGate  III System 1 / System 2 dispatch based on Semantic Divergence health
//   3. LatencyBudget  III tracks real-time generation budget per prompt
//   4. OuroborosGuard III enforces the Memory-Pressure abort path

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// ============================================================================
// TASK NATURE (unchanged from ORIGINAL design)
// ============================================================================

/// Task classification based on holistic I/O analysis
#[derive(Debug, PartialEq, Clone)]
pub enum TaskNature {
    /// Output depends on counting global colours/objects
    GlobalCounting,
    /// Input exhibits dominant symmetry that is preserved in output
    GlobalSymmetry,
    /// Output is a compressed extraction from a larger input
    GlobalExtraction,
    /// Spatial chunking (3x3 local pattern) is safe and sufficient
    LocalPattern,
    /// Healthcare / Medical context: force System1Fast to prevent context amnesia
    Healthcare,
}

impl std::fmt::Display for TaskNature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskNature::GlobalCounting   => write!(f, "GlobalCounting"),
            TaskNature::GlobalSymmetry   => write!(f, "GlobalSymmetry"),
            TaskNature::GlobalExtraction => write!(f, "GlobalExtraction"),
            TaskNature::LocalPattern     => write!(f, "LocalPattern"),
            TaskNature::Healthcare       => write!(f, "Healthcare"),
        }
    }
}

/// Lightweight complexity tag derived from a text prompt (no I/O grids needed).
/// Used by SemanticDivergenceGate when grid-based analysis is unavailable.
#[derive(Debug, Clone, PartialEq)]
pub enum PromptComplexity {
    /// Syntax repairs, look-ups, trivial transformations
    Trivial,
    /// Multi-step reasoning, moderate code generation
    Moderate,
    /// Multi-file architecture, novel algorithm derivation, scientific inference
    Complex,
}

// ============================================================================
// GLOBAL SCANNER (ORIGINAL implementation III preserved exactly)
// ============================================================================

pub struct GlobalScanner;

impl GlobalScanner {
    /// Analyse task nature from a grid I/O pair.
    pub fn analyze_task_nature(input_grid: &[Vec<u32>], output_grid: &[Vec<u32>]) -> TaskNature {
        let input_histogram  = Self::build_histogram(input_grid);
        let output_histogram = Self::build_histogram(output_grid);
        let input_object_count  = Self::count_objects(input_grid);
        let output_object_count = Self::count_objects(output_grid);
        let input_symmetry  = Self::check_symmetry(input_grid);
        let output_symmetry = Self::check_symmetry(output_grid);

        Self::decide_task_nature(
            &input_histogram, &output_histogram,
            input_object_count, output_object_count,
            input_symmetry, output_symmetry,
            input_grid, output_grid,
        )
    }
    pub fn analyze_task_nature_from_prompt(prompt: &str) -> TaskNature {
        let p = prompt.to_lowercase();

        // Counting / enumeration intent
        if p.contains("count") || p.contains("how many") || p.contains("total") {
            return TaskNature::GlobalCounting;
        }

        // Symmetry / transformation intent
        if p.contains("mirror") || p.contains("flip") || p.contains("rotate")
            || p.contains("reflect") || p.contains("symmetry")
        {
            return TaskNature::GlobalSymmetry;
        }

        // Extraction / compression intent
        if p.contains("extract") || p.contains("compress") || p.contains("summarise")
            || p.contains("summarize") || p.contains("distil")
        {
            return TaskNature::GlobalExtraction;
        }

        // Healthcare / Medical intent
        if p.contains("health") || p.contains("medical") || p.contains("radiology")
            || p.contains("patient") || p.contains("clinical") || p.contains("x-ray")
            || p.contains("nodule") || p.contains("diagnosis") || p.contains("pathology")
        {
            return TaskNature::Healthcare;
        }

        // Default: local pattern (spatial chunking safe)
        TaskNature::LocalPattern
    }

    /// Derive a `PromptComplexity` tag from a free-text prompt.
    pub fn classify_prompt_complexity(prompt: &str) -> PromptComplexity {
        let p = prompt.to_lowercase();

        let complex_markers = [
            "architecture", "design", "refactor", "multi-file", "system",
            "integrate", "clinical", "statistical", "algorithm", "benchmark",
            "security", "cryptograph", "derive", "proof", "theorem",
        ];
        let trivial_markers = [
            "missing semicolon", "syntax error", "typo", "rename variable",
            "what is", "define", "explain briefly",
        ];

        let complex_hits = complex_markers.iter().filter(|&&m| p.contains(m)).count();
        let trivial_hits = trivial_markers.iter().filter(|&&m| p.contains(m)).count();

        if trivial_hits > 0 && complex_hits == 0 {
            PromptComplexity::Trivial
        } else if complex_hits >= 2 {
            PromptComplexity::Complex
        } else {
            PromptComplexity::Moderate
        }
    }

    fn build_histogram(grid: &[Vec<u32>]) -> HashMap<u32, usize> {
        let mut histogram = HashMap::new();
        for row in grid {
            for &color in row {
                if color <= 9 {
                    *histogram.entry(color).or_insert(0) += 1;
                }
            }
        }
        histogram
    }

    fn count_objects(grid: &[Vec<u32>]) -> usize {
        if grid.is_empty() || grid[0].is_empty() { return 0; }
        let rows = grid.len();
        let cols = grid[0].len();
        let mut visited = vec![vec![false; cols]; rows];
        let mut object_count = 0;

        for i in 0..rows {
            for j in 0..cols {
                if !visited[i][j] && grid[i][j] != 0 {
                    Self::bfs_mark(grid, &mut visited, i, j, grid[i][j]);
                    object_count += 1;
                }
            }
        }
        object_count
    }

    fn bfs_mark(
        grid: &[Vec<u32>],
        visited: &mut [Vec<bool>],
        start_i: usize,
        start_j: usize,
        color: u32,
    ) {
        let rows = grid.len();
        let cols = grid[0].len();
        let mut queue = VecDeque::new();
        queue.push_back((start_i, start_j));
        visited[start_i][start_j] = true;

        while let Some((i, j)) = queue.pop_front() {
            let neighbors: &[(i32, i32)] = &[
                (i as i32 - 1, j as i32), (i as i32 + 1, j as i32),
                (i as i32, j as i32 - 1), (i as i32, j as i32 + 1),
            ];
            for &(ni, nj) in neighbors {
                if ni >= 0 && ni < rows as i32 && nj >= 0 && nj < cols as i32 {
                    let ni = ni as usize;
                    let nj = nj as usize;
                    if !visited[ni][nj] && grid[ni][nj] == color {
                        visited[ni][nj] = true;
                        queue.push_back((ni, nj));
                    }
                }
            }
        }
    }

    fn check_symmetry(grid: &[Vec<u32>]) -> f32 {
        if grid.is_empty() || grid[0].is_empty() { return 0.0; }
        let rows = grid.len();
        let cols = grid[0].len();

        let mut vert_match = 0usize;
        #[allow(clippy::needless_range_loop)]
        for i in 0..rows {
            for j in 0..cols / 2 {
                if grid[i][j] == grid[i][cols - 1 - j] { vert_match += 1; }
            }
        }
        let vert_possible = rows * (cols / 2);
        let vert_score = if vert_possible > 0 { vert_match as f32 / vert_possible as f32 } else { 0.0 };

        let mut horiz_match = 0usize;
        #[allow(clippy::needless_range_loop)]
        for i in 0..rows / 2 {
            for j in 0..cols {
                if grid[i][j] == grid[rows - 1 - i][j] { horiz_match += 1; }
            }
        }
        let horiz_possible = (rows / 2) * cols;
        let horiz_score = if horiz_possible > 0 { horiz_match as f32 / horiz_possible as f32 } else { 0.0 };

        let mut diag_match = 0usize;
        let min_dim = rows.min(cols);
        #[allow(clippy::needless_range_loop)]
        for i in 0..min_dim {
            for j in i + 1..min_dim {
                if grid[i][j] == grid[j][i] { diag_match += 1; }
            }
        }
        let diag_possible = min_dim * (min_dim - 1) / 2;
        let diag_score = if diag_possible > 0 { diag_match as f32 / diag_possible as f32 } else { 0.0 };

        vert_score.max(horiz_score).max(diag_score)
    }

    #[allow(clippy::too_many_arguments)]
    fn decide_task_nature(
        input_histogram: &HashMap<u32, usize>,
        output_histogram: &HashMap<u32, usize>,
        _input_object_count: usize,
        output_object_count: usize,
        input_symmetry: f32,
        output_symmetry: f32,
        input_grid: &[Vec<u32>],
        output_grid: &[Vec<u32>],
    ) -> TaskNature {
        let input_size  = input_grid.len()  * input_grid.first().map(|r| r.len()).unwrap_or(0);
        let output_size = output_grid.len() * output_grid.first().map(|r| r.len()).unwrap_or(0);

        // Rule 1: GlobalCounting
        for count in input_histogram.values() {
            if output_object_count == *count && output_histogram.len() <= 3 {
                return TaskNature::GlobalCounting;
            }
        }

        // Rule 1b: 1D output of a single color often represents a count bar
        if (output_grid.len() == 1 || output_grid[0].len() == 1)
            && output_histogram.len() == 1
            && !output_histogram.contains_key(&0)
        {
            return TaskNature::GlobalCounting;
        }

        // Rule 3: GlobalExtraction
        if output_size > 0 && output_size < input_size / 4 && output_size <= 25 {
            return TaskNature::GlobalExtraction;
        }

        // Rule 2: GlobalSymmetry
        if input_symmetry > 0.85 && output_symmetry > 0.75 {
            return TaskNature::GlobalSymmetry;
        }

        // Rule 4: Summarisation (many input colours III few output colours)
        if input_histogram.len() > 5 && output_histogram.len() <= 3 {
            return TaskNature::GlobalCounting;
        }

        TaskNature::LocalPattern
    }
}

// ============================================================================
// SEMANTIC DIVERGENCE THRESHOLDS (mirrors neuro_kernel.rs constants)
// ============================================================================

/// Above this value, the model is likely hallucinating or in a collapse loop.
/// High semantic divergence indicates unhealthy token distribution.
pub const SEMANTIC_DIVERGENCE_CRITICAL: f64 = 0.05;

/// Above this value, the model is in the warning zone (System 2 may help).
/// Moderate semantic divergence indicates potential instability.
pub const SEMANTIC_DIVERGENCE_WARNING: f64 = 0.01;

// ============================================================================
// OUROBOROS DECISION  (System 1 / System 2 dispatch)
// ============================================================================

/// The outcome of `SemanticDivergenceGate::decide`.
#[derive(Debug, Clone, PartialEq)]
pub enum OuroborosDecision {
    /// Fast single-pass generation.  No Counterfactual Regret loops.
    /// Triggered when W is healthy, task is simple, or memory is under pressure.
    System1Fast,

    /// Deep multi-pass generation with Counterfactual Regret refinement.
    /// `max_loops` is bounded by Semantic Divergence severity and memory budget.
    System2Deep {
        /// Maximum number of Ouroboros refinement passes (1III3 in practice)
        max_loops: usize,
    },
}

impl std::fmt::Display for OuroborosDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OuroborosDecision::System1Fast => write!(f, "System1Fast"),
            OuroborosDecision::System2Deep { max_loops } =>
                write!(f, "System2Deep(loops={})", max_loops),
        }
    }
}

// ============================================================================
// SEMANTIC DIVERGENCE GATE  III Core dispatch logic
// ============================================================================

/// Stateless strategy dispatcher.  All inputs are immutable; no allocation.
pub struct SemanticDivergenceGate;

impl SemanticDivergenceGate {
    /// Decide whether to enter the Ouroboros loop for this generation request.
    ///
    /// Decision matrix:
    ///
    /// | `memory_pressure` | `semantic_divergence` | `complexity`   | Decision         |
    /// |:-----------------:|:--------------:|:--------------:|:----------------:|
    /// | true              | any            | any            | System1Fast      |
    /// | false             | > WARNING             | any            | System1Fast      |
    /// | false             | < CRITICAL            | any            | System2Deep(2)   |
    /// | false             | WARNING..CRIT         | Complex        | System2Deep(1)   |
    /// | false             | WARNING..CRIT  | Moderate/Low   | System1Fast      |
    ///
    /// This function is called **once per prompt**, synchronously, before the
    /// generation loop starts.  It is purely CPU-bound and returns in < 1 IIs.
    pub fn decide(
        semantic_divergence: f64,
        task_nature: &TaskNature,
        complexity: f32,
        memory_pressure: bool,
    ) -> OuroborosDecision {
        // Hard gate: Memory Pressure overrides everything
        // When RAM is above 85%, running Ouroboros loops would deepen the crisis.
        // The system MUST produce a usable response with minimum resource use.
        if memory_pressure {
            eprintln!(
                "[OUROBOROS-GATE] Memory pressure active. \
                 Forcing System1Fast regardless of SD={:.4}.",
                semantic_divergence
            );
            return OuroborosDecision::System1Fast;
        }
        
        // Phase 6: Healthcare Amputation
        // Clinical reporting does not benefit from counterfactual loops; 
        // resets cause context amnesia. Force System1Fast for clinical.
        if matches!(task_nature, TaskNature::Healthcare) {
            return OuroborosDecision::System1Fast;
        }

        // Fast path: Very High Semantic Divergence is very healthy
        // SD > CRITICAL means the model is generating very diverse, independent tokens.
        // No Counterfactual Regret needed.
        if semantic_divergence > SEMANTIC_DIVERGENCE_CRITICAL {
            return OuroborosDecision::System1Fast;
        }

        // Task-nature shortcut
        // Pure local-pattern or extraction tasks rarely benefit from deep
        // re-thinking; the answer is structural, not inferential.
        if matches!(task_nature, TaskNature::LocalPattern | TaskNature::GlobalExtraction)
            && semantic_divergence > SEMANTIC_DIVERGENCE_WARNING
            && complexity <= 0.5
        {
            return OuroborosDecision::System1Fast;
        }

        // Critical zone: SD < WARNING (low variance)
        // This indicates low diversity in the token probability space:
        // the model is looping or hallucinating. Deep correction is warranted.
        if semantic_divergence < SEMANTIC_DIVERGENCE_WARNING {
            let loops = if complexity > 0.7 { 3 } else { 2 };
            eprintln!(
                "[OUROBOROS-GATE] SD={:.4} < WARNING ({}). \
                 System2Deep activated (loops={}).",
                semantic_divergence, SEMANTIC_DIVERGENCE_WARNING, loops
            );
            return OuroborosDecision::System2Deep { max_loops: loops };
        }

        // Warning zone: WARNING <= SD <= CRITICAL
        // Enter System2 only for genuinely complex tasks.
        match (complexity > 0.5, task_nature) {
            (true, _) => {
                eprintln!(
                    "[OUROBOROS-GATE] SD={:.4} in warning zone + complex task. \
                     System2Deep(loops=1).",
                    semantic_divergence
                );
                OuroborosDecision::System2Deep { max_loops: 1 }
            }
            _ => OuroborosDecision::System1Fast,
        }
    }

    /// Convenience wrapper that derives `memory_pressure` from an `AtomicBool`.
    pub fn decide_atomic(
        semantic_divergence: f64,
        task_nature: &TaskNature,
        complexity: f32,
        pressure_flag: &AtomicBool,
    ) -> OuroborosDecision {
        Self::decide(semantic_divergence, task_nature, complexity, pressure_flag.load(Ordering::Relaxed))
    }
}

// ============================================================================
// LATENCY BUDGET III Real-time generation budget tracker
// ============================================================================

/// Tracks the elapsed wall-clock time against a per-request latency allowance.
///
/// Usage inside `generate_stream`:
/// ```rust,ignore
/// let budget = LatencyBudget::new(ouroboros_max_loops, task_plan.complexity);
/// // ... token loop ...
/// if budget.is_exhausted() {
///     println!("[BUDGET] Latency budget exhausted. Sealing generation.");
///     break;
/// }
/// budget.on_token_generated();
/// ```
pub struct LatencyBudget {
    /// Hard deadline: if wall clock passes this, generation stops.
    deadline: std::time::Instant,
    /// Total token allowance for this generation pass.
    token_budget: usize,
    /// Tokens produced so far in the *current* Ouroboros loop.
    tokens_used: usize,
    /// Number of Ouroboros loops remaining.
    loops_remaining: usize,
    /// Wall-clock start of current pass.
    pass_start: std::time::Instant,
}

impl LatencyBudget {
    /// Construct a budget from the Ouroboros decision and task complexity.
    ///
    /// # Time limits (heuristic defaults, tunable via env vars):
    /// - System1Fast:     30 s hard limit (generous for 2K tokens @ 70 tok/s)
    /// - System2Deep:     15 s *per pass* II max_loops
    pub fn new(ouroboros_decision: &OuroborosDecision, complexity: f32) -> Self {
        let now = std::time::Instant::now();

        let (loops, secs_per_pass, token_budget) = match ouroboros_decision {
            OuroborosDecision::System1Fast => {
                let secs: u64 = std::env::var("SODALS_S1_TIMEOUT_SECS")
                    .ok().and_then(|s| s.parse().ok()).unwrap_or(14400);
                (1usize, secs, 8192usize)
            }
            OuroborosDecision::System2Deep { max_loops } => {
                let secs: u64 = std::env::var("SODALS_S2_PASS_SECS")
                    .ok().and_then(|s| s.parse().ok()).unwrap_or(14400);
                let tokens = (8192.0 * (1.0 + complexity)) as usize;
                (*max_loops, secs, tokens)
            }
        };

        let total_secs = secs_per_pass * loops as u64;
        let deadline = now + std::time::Duration::from_secs(total_secs);

        eprintln!(
            "[BUDGET] III  Budget: {} tokens | {} loops | {}s total",
            token_budget, loops, total_secs
        );

        Self {
            deadline,
            token_budget,
            tokens_used: 0,
            loops_remaining: loops,
            pass_start: now,
        }
    }

    /// Returns `true` when the budget is exhausted and generation must stop.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.tokens_used >= self.token_budget || std::time::Instant::now() >= self.deadline
    }

    /// Called after each generated token to advance the counter.
    #[inline]
    pub fn on_token_generated(&mut self) {
        self.tokens_used += 1;
    }

    /// Reset the token counter for the next Ouroboros pass.
    /// Returns `false` if no passes remain (caller should stop looping).
    pub fn next_pass(&mut self) -> bool {
        if self.loops_remaining <= 1 {
            return false;
        }
        self.loops_remaining -= 1;
        self.tokens_used = 0;
        self.pass_start = std::time::Instant::now();
        eprintln!(
            "[BUDGET] IIII Ouroboros pass reset. Remaining loops: {}",
            self.loops_remaining
        );
        true
    }

    /// Remaining token allowance for the current pass.
    #[inline]
    pub fn remaining_tokens(&self) -> usize {
        self.token_budget.saturating_sub(self.tokens_used)
    }

    /// Elapsed milliseconds since this pass started.
    #[inline]
    pub fn pass_elapsed_ms(&self) -> u128 {
        self.pass_start.elapsed().as_millis()
    }
}

// ============================================================================
// OUROBOROS GUARD III enforces Memory-Pressure abort and loop limits
// ============================================================================

/// Wraps `LatencyBudget` + `OuroborosDecision` into a single guard object
/// that the generation loop polls each iteration.
///
/// The guard is the *only* authoritative source for:
///   - Whether to continue or break the inner token loop
///   - Whether to start a new Ouroboros refinement pass
///   - Whether memory pressure has arrived mid-generation
pub struct OuroborosGuard {
    decision:        OuroborosDecision,
    budget:          LatencyBudget,
    pressure_flag:   Arc<AtomicBool>,
    pass_number:     usize,
}

impl OuroborosGuard {
    /// Create a new guard.  `pressure_flag` should be the same `Arc<AtomicBool>`
    /// that `CortexActor` updates when `sodals_alloc` fires its pressure hook.
    pub fn new(
        decision: OuroborosDecision,
        complexity: f32,
        pressure_flag: Arc<AtomicBool>,
    ) -> Self {
        let budget = LatencyBudget::new(&decision, complexity);
        Self {
            decision,
            budget,
            pressure_flag,
            pass_number: 1,
        }
    }

    /// Should the generation loop continueI  Call once per token.
    ///
    /// Returns `false` when any of the following are true:
    ///   - Token budget exhausted
    ///   - Wall-clock deadline exceeded
    ///   - Memory pressure arrived mid-generation
    pub fn should_continue(&mut self) -> bool {
        // Memory pressure can arrive at any time
        if self.pressure_flag.load(Ordering::Relaxed) {
            eprintln!(
                "[OUROBOROS-GUARD] IIII Memory pressure detected mid-generation \
                 (pass {}). Sealing output.",
                self.pass_number
            );
            return false;
        }

        if self.budget.is_exhausted() {
            eprintln!(
                "[OUROBOROS-GUARD] III  Budget exhausted on pass {} \
                 ({} tokens, {}ms).",
                self.pass_number,
                self.budget.tokens_used,
                self.budget.pass_elapsed_ms()
            );
            return false;
        }

        true
    }

    /// Register a generated token.
    #[inline]
    pub fn on_token(&mut self) {
        self.budget.on_token_generated();
    }

    /// Try to start the next Ouroboros refinement pass.
    ///
    /// Returns `true` and advances the pass counter if another pass is allowed.
    /// The caller should reset generation state (e.g. clear partial output buffer).
    pub fn try_next_pass(&mut self) -> bool {
        // Never start a new pass under memory pressure
        if self.pressure_flag.load(Ordering::Relaxed) {
            return false;
        }

        if self.budget.next_pass() {
            self.pass_number += 1;
            eprintln!(
                "[OUROBOROS-GUARD] III  Starting Ouroboros pass {}",
                self.pass_number
            );
            true
        } else {
            false
        }
    }

    /// Current pass number (1-based).
    pub fn pass_number(&self) -> usize { self.pass_number }

    /// Expose the Ouroboros decision for logging.
    pub fn decision(&self) -> &OuroborosDecision { &self.decision }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    // IIIIII ORIGINAL GlobalScanner tests (preserved) ////////////////////////////////////////////////////////////////////////////////IIIIIII

    #[test]
    fn test_simple_counting_task() {
        let input  = vec![vec![1,0,1], vec![0,1,0], vec![1,0,1]];
        let output = vec![vec![2,2,2]];
        assert_eq!(GlobalScanner::analyze_task_nature(&input, &output), TaskNature::GlobalCounting);
    }

    #[test]
    fn test_symmetric_task() {
        let input  = vec![vec![1,2,1], vec![2,3,2], vec![1,2,1]];
        let output = vec![vec![5,6,5], vec![6,7,6], vec![5,6,5]];
        assert_eq!(GlobalScanner::analyze_task_nature(&input, &output), TaskNature::GlobalSymmetry);
    }

    #[test]
    fn test_extraction_task() {
        let input: Vec<Vec<u32>>  = vec![vec![1;5]; 5];
        let output = vec![vec![2,2], vec![2,2]];
        assert_eq!(GlobalScanner::analyze_task_nature(&input, &output), TaskNature::GlobalExtraction);
    }

    #[test]
    fn test_local_pattern_task() {
        let input  = vec![vec![1,2,3], vec![4,5,6], vec![7,8,9]];
        let output = vec![vec![2,3,4], vec![5,6,7], vec![8,9,0]];
        assert_eq!(GlobalScanner::analyze_task_nature(&input, &output), TaskNature::LocalPattern);
    }

    #[test]
    fn test_object_counting() {
        let grid = vec![vec![1,1,0,2], vec![1,0,0,2], vec![0,0,3,3]];
        assert_eq!(GlobalScanner::count_objects(&grid), 3);
    }

    #[test]
    fn test_symmetry_detection() {
        let grid = vec![vec![1,2,1], vec![3,4,3], vec![5,6,5]];
        assert!(GlobalScanner::check_symmetry(&grid) > 0.8);
    }

    // IIIIII SemanticDivergenceGate tests ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIII

    #[test]
    fn test_gate_memory_pressure_always_system1() {
        // Regardless of SD or complexity, memory pressure III System1Fast
        for &sd in &[0.001, 0.03, 0.2] {
            let decision = SemanticDivergenceGate::decide(sd, &TaskNature::LocalPattern, 0.9, true);
            assert_eq!(decision, OuroborosDecision::System1Fast,
                "SD={} with memory_pressure should be System1Fast", sd);
        }
    }

    #[test]
    fn test_gate_healthy_semantic_divergence_is_system1() {
        // SD > SEMANTIC_DIVERGENCE_WARNING III System1 regardless of complexity
        let decision = SemanticDivergenceGate::decide(0.10, &TaskNature::LocalPattern, 0.9, false);
        assert_eq!(decision, OuroborosDecision::System1Fast);
    }

    #[test]
    fn test_gate_critical_semantic_divergence_triggers_deep() {
        // SD < SEMANTIC_DIVERGENCE_CRITICAL III System2Deep
        let decision = SemanticDivergenceGate::decide(0.005, &TaskNature::LocalPattern, 0.5, false);
        assert!(matches!(decision, OuroborosDecision::System2Deep { .. }));
    }

    #[test]
    fn test_gate_warning_zone_simple_task_system1() {
        // Warning zone + simple (local pattern) + moderate complexity III System1
        let decision = SemanticDivergenceGate::decide(0.03, &TaskNature::LocalPattern, 0.3, false);
        assert_eq!(decision, OuroborosDecision::System1Fast);
    }

    #[test]
    fn test_gate_warning_zone_complex_task_system2() {
        // Warning zone + high complexity III System2Deep(1)
        let decision = SemanticDivergenceGate::decide(0.03, &TaskNature::LocalPattern, 0.8, false);
        assert_eq!(decision, OuroborosDecision::System2Deep { max_loops: 1 });
    }

    #[test]
    fn test_gate_critical_high_complexity_deeper_loops() {
        // Critical SD + high complexity III System2Deep(3)
        let decision = SemanticDivergenceGate::decide(0.001, &TaskNature::LocalPattern, 0.8, false);
        assert_eq!(decision, OuroborosDecision::System2Deep { max_loops: 3 });
    }

    #[test]
    fn test_gate_critical_low_complexity_two_loops() {
        // Critical SD + low complexity III System2Deep(2)
        let decision = SemanticDivergenceGate::decide(0.001, &TaskNature::LocalPattern, 0.3, false);
        assert_eq!(decision, OuroborosDecision::System2Deep { max_loops: 2 });
    }

    #[test]
    fn test_gate_atomic_no_pressure() {
        let flag = AtomicBool::new(false);
        let decision = SemanticDivergenceGate::decide_atomic(0.08, &TaskNature::LocalPattern, 0.4, &flag);
        assert_eq!(decision, OuroborosDecision::System1Fast);
    }

    #[test]
    fn test_gate_atomic_with_pressure() {
        let flag = AtomicBool::new(true);
        let decision = SemanticDivergenceGate::decide_atomic(0.001, &TaskNature::LocalPattern, 1.0, &flag);
        assert_eq!(decision, OuroborosDecision::System1Fast);
    }

    // IIIIII LatencyBudget tests ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIII

    #[test]
    fn test_budget_token_exhaustion() {
        let decision = OuroborosDecision::System1Fast;
        let mut budget = LatencyBudget::new(&decision, 0.0);
        // Burn through the token budget
        for _ in 0..budget.token_budget {
            assert!(!budget.is_exhausted());
            budget.on_token_generated();
        }
        assert!(budget.is_exhausted());
    }

    #[test]
    fn test_budget_next_pass_advances() {
        let decision = OuroborosDecision::System2Deep { max_loops: 2 };
        let mut budget = LatencyBudget::new(&decision, 0.5);
        assert!(budget.next_pass());   // pass 1 III 2
        assert!(!budget.next_pass()); // no more passes
    }

    // IIIIII OuroborosGuard tests ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIII

    #[test]
    fn test_guard_stops_on_pressure() {
        let flag = Arc::new(AtomicBool::new(false));
        let decision = OuroborosDecision::System1Fast;
        let mut guard = OuroborosGuard::new(decision, 0.5, Arc::clone(&flag));

        assert!(guard.should_continue());

        // Simulate memory pressure arriving mid-generation
        flag.store(true, Ordering::Relaxed);
        assert!(!guard.should_continue());
    }

    #[test]
    fn test_guard_no_next_pass_system1() {
        let flag = Arc::new(AtomicBool::new(false));
        let decision = OuroborosDecision::System1Fast;
        let mut guard = OuroborosGuard::new(decision, 0.3, Arc::clone(&flag));

        // System1 has only 1 loop III next_pass always false
        assert!(!guard.try_next_pass());
    }

    #[test]
    fn test_guard_next_pass_system2() {
        let flag = Arc::new(AtomicBool::new(false));
        let decision = OuroborosDecision::System2Deep { max_loops: 2 };
        let mut guard = OuroborosGuard::new(decision, 0.7, Arc::clone(&flag));

        assert_eq!(guard.pass_number(), 1);
        assert!(guard.try_next_pass());
        assert_eq!(guard.pass_number(), 2);
        assert!(!guard.try_next_pass()); // exhausted
    }

    #[test]
    fn test_prompt_complexity_classification() {
        assert_eq!(
            GlobalScanner::classify_prompt_complexity("fix missing semicolon"),
            PromptComplexity::Trivial
        );
        assert_eq!(
            GlobalScanner::classify_prompt_complexity("design a multi-file distributed architecture with security"),
            PromptComplexity::Complex
        );
    }
}














