// src/engine/chimera_router.rs

use std::collections::HashMap;
use crate::semantic_network::ResonanceNetwork;

// ============================================================================
// ANALOGICAL REASONING ENGINE PROTOCOL
// Topological Transfer for Novel Problems
// ============================================================================

pub const ANALOGICAL_REASONING_PROTOCOL: &str = r#"
<analogical_reasoning_protocol>
You are an Analogical Reasoning Engine powered by topological transfer learning.

Before writing code, you MUST perform a Topological Gap Analysis in <thinking> tags:

STEP 1: EXACT MATCH SCAN
Check if this problem exists in your training data.
- "Towers of Hanoi" III Yes, common I
- "River crossing with 7 entities" III Novel II

STEP 2: NOVELTY DETECTION
If novel: Acknowledge that direct recall is insufficient.
State: "This problem has novelty gap I = X%"

STEP 3: FIND SEMANTIC ANCHORS
Identify known algorithms with topological similarity:
- State Space Search: BFS, DFS, A*, Dijkstra
- Constraint Satisfaction: Backtracking, DP
- Flow problems: Max Flow, Network Flow

Example: "This problem shares:
 III 70% topology with Towers of Hanoi (State Space Search)
 III 15% topology with Ozturk Algorithm (Flow constraints)
 III 10% novel constraints"

STEP 4: TOPOLOGICAL TRANSFER
Apply the transfer:
"I will transfer:
 1. Recurrence relation from Hanoi III State representation
 2. Flow constraints from Ozturk III Capacity checking
 
Bridging the gap:
 III Derive new impedance rule for novel constraint
 III Add phase constraint: II > I/2 (orthogonal states)"

STEP 5: IMPLEMENTATION
Only AFTER Steps 1-4, write the code with:
III Comment header explaining topological transfer
III #[test] functions verifying transferred logic

FORBIDDEN:
I Skip <thinking> for complex problems
I Hallucinate algorithms not in training
I Claim 100% novelty when anchors exist
</analogical_reasoning_protocol>
"#;

#[derive(Clone, Debug)]
pub struct TaskPlan {
    pub complexity: f32,          // 0.0-1.0: task complexity score
    pub needs_design: bool,       // Does this need GEOMETER architectural planningI
    pub is_syntax_only: bool,     // Is this just syntax fixingI
    pub pure_inference: bool,     // Disables QMEM external cognitive steering
    /// Multiple-choice benchmark (MMLU Pro, FSIQ, etc.) — enables logit mask
    pub is_multiple_choice: bool,
    /// Option letters detected in the prompt (e.g. A–E)
    pub mc_option_letters: Vec<char>,
    pub is_health: bool,          // Phase 62: Clinical/Medical domain shield
    pub steps: Vec<String>,       // GEOMETER's to-do list
    pub mode_history: Vec<String>, // Track mode switches
    
    // RESOURCE MANAGER: Dynamic token allocation
    pub needs_new_test: bool,     // Does this task require test generationI
    pub estimated_tokens: usize,  // Estimated token budget for this task
    
    // SEMANTIC LINK: Compressed communication protocol
    
    // RUBIK'S TOPOLOGY: Semantic color mapping
    pub color_map: HashMap<String, String>,  // filename -> color (Red=Entity, Yellow=Logic, Blue=Main, Green=Test)
    
    // HOLOGRAPHIC PROPHECY: Geometric problem-solving data
    pub phase_angles: Vec<f64>,
    pub rhythm_score: f64,
    pub semantic_divergence_score: f64,
    pub keystone_nodes: Vec<String>,
    pub fractured_nodes: Vec<String>,
    pub geometric_moves: Vec<String>,
    pub semantic_log: Vec<String>,    // PGN-like semantic commands
    pub impact_radius: Vec<String>,
}

// GEOMETER SUPERVISOR: Monitors execution and detects loops
#[derive(Clone, Debug)]
pub struct PlanMonitor {
    pub expected_files: Vec<String>,      // Files from GEOMETER's plan
    pub completed_steps: Vec<String>,     // Files/steps actually written
    pub step_positions: Vec<usize>,       // Character positions where each step started
    pub loop_detected: bool,              // Flag for loop detection
    pub prune_position: Option<usize>,    // Where to cut the output
}

impl Default for PlanMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PlanMonitor {
    pub fn new() -> Self {
        Self {
            expected_files: Vec::new(),
            completed_steps: Vec::new(),
            step_positions: Vec::new(),
            loop_detected: false,
            prune_position: None,
        }
    }
    
    // SUPERVISOR: Detect if a file/step is being repeated
    pub fn check_for_loop(&mut self, filename: &str, position: usize) -> bool {
        // Check if this file was already completed
        if let Some(first_occurrence) = self.completed_steps.iter().position(|s| s == filename) {
            // This is a repeat!
            eprintln!("[SUPERVISOR] II Loop Detected: '{}' was already written at step {}", 
                     filename, first_occurrence + 1);
            
            self.loop_detected = true;
            self.prune_position = Some(position);
            return true;
        }
        
        // New step, record it
        self.completed_steps.push(filename.to_string());
        self.step_positions.push(position);
        false
    }
    
    // SUPERVISOR: Parse and track code blocks
    pub fn track_code_block(&mut self, text: &str, current_position: usize) -> bool {
        // Detect code block markers: ```filename or --- filename ---
        let lines: Vec<&str> = text.lines().collect();
        
        for line in lines {
            let trimmed = line.trim();
            
            // Pattern 1: ```rust:filename.rs or ```python:script.py
            if trimmed.starts_with("```") && trimmed.contains(':') {
                if let Some(filename) = trimmed.split(':').nth(1) {
                    let clean_name = filename.trim().trim_end_matches("```");
                    if !clean_name.is_empty() {
                        return self.check_for_loop(clean_name, current_position);
                    }
                }
            }
            
            // Pattern 2: --- src/auth.rs ---
            if trimmed.starts_with("---") && trimmed.ends_with("---") {
                let parts: Vec<&str> = trimmed.split("---").collect();
                if parts.len() >= 2 {
                    let filename = parts[1].trim();
                    if !filename.is_empty() && filename.contains('.') {
                        return self.check_for_loop(filename, current_position);
                    }
                }
            }
            
            // Pattern 3: File: filename.rs or // filename.rs
            if trimmed.starts_with("File:") || trimmed.starts_with("//") {
                for word in trimmed.split_whitespace() {
                    if word.contains('.') && (
                        word.ends_with(".rs") ||
                        word.ends_with(".py") ||
                        word.ends_with(".js") ||
                        word.ends_with(".ts") ||
                        word.ends_with(".cpp")
                    ) {
                        let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '_' && c != '/');
                        return self.check_for_loop(clean, current_position);
                    }
                }
            }
        }
        
        false // No loop detected
    }
}

impl Default for TaskPlan {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskPlan {
    pub fn new() -> Self {
        Self {
            complexity: 0.0,
            needs_design: false,
            is_syntax_only: false,
            pure_inference: false,
            is_multiple_choice: false,
            mc_option_letters: Vec::new(),
            is_health: false,
            steps: Vec::new(),
            mode_history: Vec::new(),
            needs_new_test: true,
            estimated_tokens: 8192,
            semantic_log: Vec::new(),
            color_map: HashMap::new(),
            phase_angles: Vec::new(),
            rhythm_score: 0.0,
            semantic_divergence_score: 0.0,
            keystone_nodes: Vec::new(),
            fractured_nodes: Vec::new(),
            geometric_moves: Vec::new(),
            impact_radius: Vec::new(),
        }
    }
    
    pub fn analyze_from_prompt(prompt: &str) -> Self {
        let p = prompt.to_lowercase();
        let mut plan = Self::new();
        
        // Complexity indicators
        let high_complexity_markers = ["architecture", "design", "refactor", "structure", "system", "complex", "multi", "integrate"];
        
        // Only match actual code syntax errors (narrowed)
        let syntax_markers = [
            "missing semicolon",
            "syntax error",
            "typo in function",
            "typo in variable",
            "unexpected token",
            "invalid syntax",
            "unterminated string",
            "indentation error",
            "missing parenthesis",
            "missing bracket",
            "missing brace",
        ];
        
        // Scientific/benchmark keywords that indicate complex reasoning is needed
        let scientific_markers = [
            "clinical", "statistical", "analysis", "calculate", "regression",
            "correlation", "variance", "probability", "hypothesis", "dataset",
            "benchmark", "experiment", "validation", "metrics", "performance",
            "algorithm", "optimization", "simulation", "modeling", "inference",
            "prediction", "classification", "clustering", "anomaly", "detection",
            "security", "privacy", "pii", "encryption", "compliance", "healthcare",
            "medical", "trial", "patient", "drug", "efficacy", "safety",
        ];
        
        let complexity_score = high_complexity_markers.iter()
            .filter(|&m| p.contains(m))
            .count() as f32 / high_complexity_markers.len() as f32;
        
        // Check for scientific/benchmark keywords - these override syntax-only detection
        let has_scientific_context = scientific_markers.iter().any(|&m| p.contains(m));
        
        // Only mark as syntax-only if it matches actual syntax patterns AND has no scientific context
        let matches_syntax_pattern = syntax_markers.iter().any(|&m| p.contains(m));
        
        plan.is_health = scientific_markers.iter().any(|&m| p.contains(m)) || p.contains("clinical") || p.contains("radiology") || p.contains("medical");
        plan.is_syntax_only = matches_syntax_pattern && !has_scientific_context;

        plan.is_multiple_choice = crate::semantic_parser::MmluLogitMask::detect_multiple_choice_prompt(prompt);
        if plan.is_multiple_choice {
            plan.mc_option_letters =
                crate::semantic_parser::MmluLogitMask::parse_option_letters_from_prompt(prompt);
            let lower_prompt = prompt.to_lowercase();
            if lower_prompt.contains("reasoning") || lower_prompt.contains("step-by-step") || lower_prompt.contains("thought") || lower_prompt.contains("think") {
                plan.estimated_tokens = 2048;
            } else {
                plan.estimated_tokens = 32;
            }
        }
        
        plan.complexity = complexity_score;
        plan.needs_design = complexity_score > 0.3 && !plan.is_syntax_only && !plan.is_multiple_choice;
        
        plan
    }
    
    // SEMANTIC LINK: Parse GEOMETER's plan into semantic codes
    pub fn generate_semantic_plan(&mut self, plan_text: &str) {
        self.semantic_log.clear();
        
        let lines: Vec<&str> = plan_text.lines().collect();
        
        for line in lines {
            let trimmed = line.trim();
            
            // Parse plan items into semantic codes
            if let Some(semantic_code) = Self::parse_plan_to_semantic(trimmed) {
                self.semantic_log.push(semantic_code);
            }
        }
        
        if !self.semantic_log.is_empty() {
            eprintln!("[GEO-LINK] II Semantic Plan Generated:");
            for (idx, code) in self.semantic_log.iter().enumerate() {
                eprintln!("  [{}.] {}", idx + 1, code);
            }
        }
    }
    
    // SEMANTIC LINK: Convert natural language plan to semantic code
    fn parse_plan_to_semantic(line: &str) -> Option<String> {
        let lower = line.to_lowercase();
        
        // Detect FIX operations
        if lower.contains("fix") || lower.contains("correct") || lower.contains("repair") {
            if let Some(target) = Self::extract_target(&lower) {
                return Some(format!("FIX({})", target));
            }
        }
        
        // Detect CREATE operations
        if lower.contains("create") || lower.contains("add") || lower.contains("implement") {
            if lower.contains("test") {
                if let Some(target) = Self::extract_target(&lower) {
                    return Some(format!("TEST({})", target));
                }
            } else if let Some(target) = Self::extract_target(&lower) {
                return Some(format!("CREATE({})", target));
            }
        }
        
        // Detect DELETE operations
        if lower.contains("delete") || lower.contains("remove") {
            if let Some(target) = Self::extract_target(&lower) {
                return Some(format!("DEL({})", target));
            }
        }
        
        None
    }
    
    // SEMANTIC LINK: Extract target (class.method or filename) from text
    pub fn extract_target(text: &str) -> Option<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for word in words {
            // Check for Class.method pattern
            if word.contains('.') && word.chars().next()?.is_uppercase() {
                return Some(word.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '_').to_string());
            }
            
            // Check for filename pattern
            if word.ends_with(".rs") || word.ends_with(".py") || word.ends_with(".js") {
                return Some(word.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '_').to_string());
            }
        }
        
        None
    }

    pub fn extract_plan_from_output(&mut self, output: &str) {
        let lines: Vec<&str> = output.lines().collect();
        
        for line in lines {
            let trimmed = line.trim();
            
            if trimmed.starts_with(|c: char| c.is_numeric() && trimmed.contains('.')) ||
               trimmed.starts_with("- ") ||
               trimmed.starts_with("* ") ||
               trimmed.contains("TODO:") ||
               trimmed.contains("Step ") {
                
                for word in trimmed.split_whitespace() {
                    if word.contains('.') && (
                        word.ends_with(".rs") ||
                        word.ends_with(".py") ||
                        word.ends_with(".js") ||
                        word.ends_with(".ts") ||
                        word.ends_with(".cpp") ||
                        word.ends_with(".go") ||
                        word.ends_with(".java")
                    ) {
                        let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '_' && c != '/');
                        if !self.steps.contains(&clean_word.to_string()) {
                            self.steps.push(clean_word.to_string());
                        }
                    }
                }
            }
        }
    }
    
    pub fn assign_colors(&mut self) {
        self.color_map.clear();
        
        for file in &self.steps {
            let color = Self::determine_file_color(file);
            self.color_map.insert(file.clone(), color.clone());
            eprintln!("[RUBIK] II {} III {} node", file, color);
        }
    }
    
    fn determine_file_color(filename: &str) -> String {
        let lower = filename.to_lowercase();
        
        if lower.contains("entity") || lower.contains("model") || 
           lower.contains("user") || lower.contains("data") ||
           lower.contains("wizard") || lower.contains("character") {
            return "II RED (Entity)".to_string();
        }
        
        if lower.contains("service") || lower.contains("logic") || 
           lower.contains("handler") || lower.contains("processor") ||
           lower.contains("manager") {
            return "II YELLOW (Logic)".to_string();
        }
        
        if lower.contains("main") || lower.contains("controller") || 
           lower.contains("api") || lower.contains("endpoint") {
            return "II BLUE (Main)".to_string();
        }
        
        if lower.contains("test") || lower.starts_with("test_") {
            return "II GREEN (Test)".to_string();
        }
        
        "I GRAY (Other)".to_string()
    }
    
    pub fn check_color_dependency(&self, edited_file: &str, _semantic_network: &ResonanceNetwork) -> Vec<String> {
        let mut warnings = Vec::new();
        
        if let Some(edited_color) = self.color_map.get(edited_file) {
            if edited_color.contains("YELLOW") {
                for (file, color) in &self.color_map {
                    if color.contains("RED") {
                        warnings.push(format!(
                            "[RUBIK] II  Phase: Editing {} (Yellow) affects {} (Red)",
                            edited_file, file
                        ));
                    }
                }
            }
            
            if edited_color.contains("BLUE") {
                for file in self.color_map.keys() {
                    if file != edited_file {
                        warnings.push(format!(
                            "[RUBIK] II Main {} (Blue prioritized)",
                            edited_file
                        ));
                    }
                }
            }
        }
        
        warnings
    }
    
    pub fn estimate_resources(&mut self, prompt: &str, context: &str) {
        let p = prompt.to_lowercase();
        let ctx = context.to_lowercase();
        
        let test_requested = p.contains("add test") || 
                           p.contains("write test") || 
                           p.contains("create test") ||
                           p.contains("test coverage");
        
        let has_existing_tests = ctx.contains("tests/") || 
                                ctx.contains("test_") ||
                                ctx.contains("__tests__");
        
        self.needs_new_test = test_requested || !has_existing_tests;
        
        let mut budget = 1024_usize;
        budget += prompt.len() / 5;
        
        if !self.is_syntax_only {
            budget *= 2;
        }
        
        if self.needs_new_test {
            budget += 512;
        }
        
        self.estimated_tokens = budget.min(8192);
    }
}

#[derive(Clone, Debug)]
pub struct NeuroConfig {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub min_p: f64,
    pub repeat_penalty: f32,
    pub euler_resonance: bool,
    pub physics_enabled: bool,
    pub strict_determinism: bool,
    /// Suppress verbose internal debug logging during generation (benchmark mode)
    pub quiet: bool,
    /// Override for the minimum-think gate (tokens). `None` = auto: simple QA gets a
    /// small cap (so EOS/termination is not suppressed for ~1000 tokens), while design/
    /// multi-step tasks keep the full MIN_THINK_TOKENS default.
    pub min_think_tokens: Option<usize>,
    /// Diagnostic "minimal mode": bypass the heavy logit orchestration (semantic
    /// steering, ARKHE bridge, oracle bias) so generation behaves like a plain sampler.
    /// Used to bisect which subsystem distorts the model output.
    pub minimal_mode: bool,
    /// True when the caller explicitly overrode temperature/top_p/repeat_penalty
    /// (e.g. via CLI flags). In that case prepare_generation_context must NOT overwrite
    /// them with NeuroConfig::detect()'s per-mode defaults.
    pub sampling_override: bool,
}

// GATEKEEPER: Rapid Coherence Filter
pub struct TextCoherenceFilter {
    history: Vec<String>,
    window_size: usize,
    sentence_history: Vec<Vec<num_complex::Complex64>>,
    current_sentence: String,
    // Phase 2: Hierarchical coherence levels
    paragraph_history: Vec<String>,
    paragraph_window_size: usize,
}

// Phase 2: Coherence decision types
pub enum CoherenceDecision {
    Continue,
    Backtrack(usize),  // Backtrack N tokens
    Break,             // Circuit break
}

impl Default for TextCoherenceFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl TextCoherenceFilter {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            window_size: 32,
            sentence_history: Vec::new(),
            current_sentence: String::new(),
            // Phase 2: Initialize paragraph-level tracking
            paragraph_history: Vec::new(),
            paragraph_window_size: 5,
        }
    }

    fn word_to_signature(word: &str) -> Vec<num_complex::Complex64> {
        let mut seed: u64 = 0;
        for b in word.bytes() {
            seed = seed.wrapping_mul(31).wrapping_add(b as u64);
        }
        (0..1024).map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let a = ((seed & 0xFF) as f64 / 255.0) * 2.0 - 1.0;
            let b = (((seed >> 8) & 0xFF) as f64 / 255.0) * 2.0 - 1.0;
            num_complex::Complex64::new(a, b)
        }).collect()
    }

    // Phase 2: Hierarchical coherence check with 3 levels
    pub fn check_coherence_hierarchical(&mut self, new_text: &str, full_response: &str) -> CoherenceDecision {
        if new_text.trim().is_empty() {
            return CoherenceDecision::Continue;
        }

        // Level 1: Token-level rapid repetition check
        let token_score = self.check_token_level(new_text);
        if token_score < 0.2 {
            return CoherenceDecision::Backtrack(10);
        }

        // Level 2: Sentence-level semantic coherence
        let sentence_score = self.check_sentence_level(new_text);
        if sentence_score < 0.3 {
            return CoherenceDecision::Backtrack(5);
        }

        // Level 3: Paragraph-level repetition
        let paragraph_score = self.check_paragraph_level(new_text, full_response);
        if paragraph_score < 0.2 {
            return CoherenceDecision::Break;
        }

        CoherenceDecision::Continue
    }

    fn check_token_level(&mut self, new_text: &str) -> f32 {
        if new_text.chars().count() >= 3 && self.history.contains(&new_text.to_string()) {
            return 0.1;
        }

        self.history.push(new_text.to_string());
        if self.history.len() > self.window_size {
            self.history.remove(0);
        }

        let char_count = new_text.chars().count();
        if char_count > 5 {
            let unique_chars = new_text.chars().collect::<std::collections::HashSet<_>>().len();
            let ratio = unique_chars as f32 / char_count as f32;
            if ratio < 0.2 {
                return 0.2;
            }
        }
        1.0
    }

    fn check_sentence_level(&mut self, new_text: &str) -> f32 {
        self.current_sentence.push_str(new_text);
        if new_text.contains('.') || new_text.contains('\n') || new_text.contains('?') || new_text.contains('!') {
            let sentence = self.current_sentence.trim().to_lowercase();
            self.current_sentence.clear();

            if sentence.len() >= 20 {
                let mut sig = vec![num_complex::Complex64::new(0.0, 0.0); 1024];
                let mut count = 0;
                for word in sentence.split_whitespace() {
                    let clean_w = word.trim_matches(|c: char| !c.is_alphabetic());
                    if clean_w.len() < 3 { continue; }
                    let w_sig = Self::word_to_signature(clean_w);
                    for i in 0..1024 { sig[i] += w_sig[i]; }
                    count += 1;
                }

                if count > 0 {
                    let mut max_sim = 0.0;
                    let norm_sig = sig.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
                    if norm_sig > 1e-6 {
                        for hist_sig in &self.sentence_history {
                            let norm_hist = hist_sig.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
                            let dot: num_complex::Complex64 = sig.iter().zip(hist_sig.iter()).map(|(a, b)| a * b.conj()).sum();
                            let sim = dot.norm() / (norm_sig * norm_hist);
                            if sim > max_sim { max_sim = sim; }
                        }
                    }

                    self.sentence_history.push(sig);
                    if self.sentence_history.len() > 10 {
                        self.sentence_history.remove(0);
                    }

                    // Phase 3 Fix: Earlier intervention - reduced threshold from 0.85 to 0.75
                    if max_sim > 0.75 {
                        eprintln!("[HIERARCHICAL FILTER] Sentence-level >75% similarity detected.");
                        return 0.1;
                    }
                }
            }
        }
        1.0
    }

    fn check_paragraph_level(&mut self, new_text: &str, full_response: &str) -> f32 {
        // Only evaluate on a real paragraph boundary (newline). A '.' mid-paragraph is
        // not a paragraph end, so it must not snapshot a still-growing paragraph: that
        // would compare a later, longer paragraph against its own shorter prefix and
        // produce a false >0.8 match. Sentence-level repetition is already caught by
        // Level 2 (which fires on '.'/'?'/'!'/newline).
        if !new_text.contains('\n') {
            return 1.0;
        }
        let paragraph = full_response.split('\n').last().unwrap_or(full_response);
        if paragraph.len() > 50 {
            for hist_para in &self.paragraph_history {
                let similarity = self.compute_string_similarity(paragraph, hist_para);
                if similarity > 0.8 {
                    eprintln!("[HIERARCHICAL FILTER] Paragraph-level >80% similarity detected.");
                    return 0.1;
                }
            }

            self.paragraph_history.push(paragraph.to_string());
            if self.paragraph_history.len() > self.paragraph_window_size {
                self.paragraph_history.remove(0);
            }
        }
        1.0
    }

    fn compute_string_similarity(&self, s1: &str, s2: &str) -> f32 {
        let words1: std::collections::HashSet<&str> = s1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = s2.split_whitespace().collect();

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        intersection as f32 / union as f32
    }

    // Phase 2: N-gram repetition detection
    pub fn detect_ngram_repetition(&self, tokens: &[u32], n: usize) -> Option<usize> {
        if tokens.len() < 2 * n {
            return None;
        }

        let window = &tokens[tokens.len() - n..];
        let prev_window = &tokens[tokens.len() - 2*n..tokens.len() - n];

        if window == prev_window {
            // Only treat as a loop when the repeated n-gram is degenerate (at most 2
            // distinct tokens, e.g. "the the" or "A B A B"). Legitimate repeated phrases
            // in structured answers (section headers, "Kural 1'i Uygula:", results lists)
            // contain many distinct tokens and must not trigger a destructive backtrack.
            let mut distinct_tokens: Vec<u32> = Vec::new();
            for &t in window {
                if !distinct_tokens.contains(&t) {
                    distinct_tokens.push(t);
                    if distinct_tokens.len() > 2 {
                        return None;
                    }
                }
            }
            Some(n)
        } else {
            None
        }
    }

    // Phase 4: Pattern-based detection for common loop patterns
    pub fn detect_common_loop_patterns(&self, text: &str) -> Option<String> {
        let common_patterns = [
            "Confidence:",
            "Answer:",
            "Explanation:",
            "**Confidence",
            "**Answer",
            "**Explanation",
        ];

        for pattern in &common_patterns {
            let count = text.matches(pattern).count();
            if count >= 3 {
                eprintln!("[PATTERN DETECTION] Common loop pattern '{}' repeated {} times.", pattern, count);
                return Some(pattern.to_string());
            }
        }

        // Detect markdown bold loops only when the markers are tightly clustered in the
        // recent output (degenerate streams like "** ** ** ** ** **"), not legitimate
        // bold formatting spread across a long response (e.g. "**CPU**: 50%" repeated
        // across sections of a normal answer).
        let recent = if text.len() > 120 {
            // Back up to a UTF-8 char boundary: slicing mid-multi-byte-char panics.
            let mut start = text.len() - 120;
            while start > 0 && !text.is_char_boundary(start) {
                start -= 1;
            }
            &text[start..]
        } else {
            text
        };
        let bold_count = recent.matches("**").count();
        if bold_count >= 6 {
            // Only treat markdown bold as a loop when the recent window is *dominated* by
            // the marker (degenerate streams like "** ** ** ** ** **" or "**\n**\n**"),
            // i.e. almost no real content between the markers. Legitimate bold formatting
            // spread across a structured answer (e.g. "**CPU**: 50%" / "**Kural 1:** ...")
            // is normal and must not trigger a destructive backtrack.
            let info_chars = recent.chars()
                .filter(|c| *c != '*' && *c != ' ' && *c != '\n' && *c != '\t' && *c != '\r')
                .count();
            if info_chars <= 8 {
                eprintln!("[PATTERN DETECTION] Markdown bold loop ({} '**' markers, only {} info chars in recent window).", bold_count, info_chars);
                return Some("**".to_string());
            }
            eprintln!("[PATTERN DETECTION] Markdown bold repeated {} times in recent window but with real content ({} info chars) — allowed.", bold_count, info_chars);
        }

        None
    }

    // Phase 4: Character-level pattern detection for complex loops
    pub fn detect_character_pattern_repetition(&self, text: &str) -> Option<String> {
        if text.len() < 30 {
            return None;
        }

        // Look for repeating character sequences with minor variations
        // e.g., "**Confidence\n:**\n\n**Confidence\n:**"
        let chars: Vec<char> = text.chars().collect();
        let window_size = 15;

        if chars.len() < 2 * window_size {
            return None;
        }

        let last_window: String = chars[chars.len() - window_size..].iter().collect();
        let prev_window: String = chars[chars.len() - 2 * window_size..chars.len() - window_size].iter().collect();

        // Check if windows are similar (allowing for minor variations)
        let similarity = self.compute_string_similarity(&last_window, &prev_window);
        if similarity > 0.85 {
            eprintln!("[CHARACTER PATTERN] Character-level repetition detected (similarity: {:.2}).", similarity);
            return Some(last_window);
        }

        None
    }

    // Keep original check_coherence for backward compatibility
    pub fn check_coherence(&mut self, new_text: &str, full_response: &str) -> f32 {
        match self.check_coherence_hierarchical(new_text, full_response) {
            CoherenceDecision::Continue => 1.0,
            CoherenceDecision::Backtrack(_) => 0.3,
            CoherenceDecision::Break => 0.1,
        }
    }
}

impl NeuroConfig {
    pub fn precision_mode() -> Self {
        Self { temperature: 0.1, top_p: 0.85, top_k: 40, min_p: 0.05, repeat_penalty: 1.15, euler_resonance: false, physics_enabled: false, strict_determinism: false, quiet: false, min_think_tokens: None, minimal_mode: false, sampling_override: false }
    }
    pub fn creative_mode() -> Self {
        Self { temperature: 0.9, top_p: 0.95, top_k: 100, min_p: 0.05, repeat_penalty: 1.15, euler_resonance: true, physics_enabled: true, strict_determinism: false, quiet: false, min_think_tokens: None, minimal_mode: false, sampling_override: false }
    }

    pub fn geometer_mode() -> Self {
        Self { 
            temperature: 0.4, 
            top_p: 0.92, 
            top_k: 70, 
            min_p: 0.03, 
            repeat_penalty: 1.15, 
            euler_resonance: true,
            physics_enabled: true, 
            strict_determinism: false,
            quiet: false,
            min_think_tokens: None,
            minimal_mode: false,
            sampling_override: false,
        }
    }

    pub fn engineer_mode() -> Self {
        Self { 
            temperature: 0.1, 
            top_p: 0.85, 
            top_k: 40, 
            min_p: 0.05, 
            repeat_penalty: 1.15, 
            euler_resonance: false,
            physics_enabled: false, 
            strict_determinism: true,
            quiet: false,
            min_think_tokens: None,
            minimal_mode: false,
            sampling_override: false,
        }
    }
}

impl Default for NeuroConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 60,
            min_p: 0.08,
            repeat_penalty: 1.15,
            euler_resonance: true,
            physics_enabled: true,
            strict_determinism: false,
            quiet: false,
            min_think_tokens: None,
            minimal_mode: false,
            sampling_override: false,
        }
    }
}

impl NeuroConfig {
    pub fn detect(prompt: &str) -> (Self, TaskPlan) {
        let analysis_text = if let Some(pos) = prompt.rfind("SORU:") { &prompt[pos..] } else { prompt };
        let p = analysis_text.to_lowercase();
        
        let task_plan = TaskPlan::analyze_from_prompt(prompt);
        

        
        if task_plan.is_syntax_only {
            return (Self::engineer_mode(), task_plan);
        }

        if task_plan.is_multiple_choice {
            return (Self::engineer_mode(), task_plan);
        }
        
        if task_plan.needs_design {
            return (Self::geometer_mode(), task_plan);
        }
        
        let tabular_triggers = ["csv", "json", "data_summary", "anomaly", "metrics", "cost_comparison_data", "table", "rows", "columns", "finops"];
        if tabular_triggers.iter().any(|&t| p.contains(t)) {
            return (Self::engineer_mode(), task_plan);
        }
        
        let deep_logic_triggers = ["kod", "script", "algoritma", "debug", "hata", "analiz", "optimize", "bash", "python", "rust"];
        
        if deep_logic_triggers.iter().any(|&t| p.contains(t)) {
            return (Self::geometer_mode(), task_plan);
        }
        
        let math_triggers = ["+", "-", "*", "/", "=", "x", "denklem", "iIIIlem", "sonuIII", "kaIII", "IIIIIIz", "hesapla", "kod", "fonksiyon", "integral", "tIIIrev"];
        
        let is_math = math_triggers.iter().any(|&t| p.contains(t));

        if is_math { 
            (Self::precision_mode(), task_plan)
        }
        else if p.contains("IIIiir") || p.contains("hayal") || p.contains("hikaye") { 
            (Self::creative_mode(), task_plan)
        }
        else { 
            (Self::default(), task_plan)
        }
    }
}
