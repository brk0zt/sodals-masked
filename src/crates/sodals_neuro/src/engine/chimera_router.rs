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
}

// GATEKEEPER: Rapid Coherence Filter
pub struct TextCoherenceFilter {
    history: Vec<String>,
    window_size: usize,
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
            window_size: 32 
        }
    }

    pub fn check_coherence(&mut self, new_text: &str) -> f32 {
        if new_text.trim().is_empty() {
            return 1.0;
        }

        let score = if new_text.chars().count() >= 3 && self.history.contains(&new_text.to_string()) {
            0.1
        } else {
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
        };

        score
    }
}

impl NeuroConfig {
    pub fn precision_mode() -> Self {
        Self { temperature: 0.1, top_p: 0.85, top_k: 40, min_p: 0.05, repeat_penalty: 1.15, euler_resonance: false, physics_enabled: false, strict_determinism: false, quiet: false }
    }
    pub fn creative_mode() -> Self {
        Self { temperature: 0.9, top_p: 0.95, top_k: 100, min_p: 0.05, repeat_penalty: 1.15, euler_resonance: true, physics_enabled: true, strict_determinism: false, quiet: false }
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
