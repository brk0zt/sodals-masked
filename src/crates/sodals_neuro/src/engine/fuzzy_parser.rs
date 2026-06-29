// src/engine/fuzzy_parser.rs
// FUZZY CONTEXT EXTRACTOR — Surgical AST Leaf Extraction for the Sensory Cortex
//
// Standard agents feed entire files into context windows. We do NOT have a
// context window. We have a continuous ODE flow that ingests BOUNDARY CONDITIONS.
//
// The AstHunter traverses the workspace using the `ignore` crate (ripgrep-speed
// file walking) and extracts ONLY the specific function/struct body — the AST
// leaf — that is relevant. This leaf is then routed directly to
// McpSensoryCortex::ingest_ast_node, converting it into a PhasorBoundaryCondition
// for the RK4 solver.
//
// NO file slurping. NO full-file context. NO text chunking.
// Only surgical extraction of the exact symbol definition.

use std::path::{Path, PathBuf};
use std::collections::HashMap;

use ignore::WalkBuilder;

use crate::semantic_parser::{DOMAIN_LOGIC, DOMAIN_SPATIAL, DOMAIN_TEMPORAL, DOMAIN_ABSTRACT};
use crate::engine::mcp_hdc_bridge::{McpSensoryCortex, PhasorBoundaryCondition, DefaultMcpSensoryCortex};

// ============================================================================
// AST NODE — The extracted symbol leaf
// ============================================================================

/// A surgically extracted AST leaf — a single function/struct/enum/trait body.
/// This is NOT a full file. It is the minimal unit of semantic information
/// that the physics engine needs to construct a boundary condition.
#[derive(Debug, Clone)]
pub struct AstNode {
    /// The symbol name (e.g., "PonderingCore", "rk4_step", "Arkhe")
    pub name: String,
    /// The kind of the symbol (function, struct, enum, trait, impl, const, type)
    pub kind: AstNodeKind,
    /// The source text of the symbol body (just the definition, not the whole file)
    pub source_text: String,
    /// The file where this symbol was found
    pub file_path: PathBuf,
    /// Line number where the symbol starts
    pub line_start: usize,
    /// Line number where the symbol ends
    pub line_end: usize,
    /// Neuro-symbolic domain classification
    pub domain_flag: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AstNodeKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Const,
    TypeAlias,
    Use,
    Module,
    Unknown,
}

impl std::fmt::Display for AstNodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AstNodeKind::Function => write!(f, "fn"),
            AstNodeKind::Struct => write!(f, "struct"),
            AstNodeKind::Enum => write!(f, "enum"),
            AstNodeKind::Trait => write!(f, "trait"),
            AstNodeKind::Impl => write!(f, "impl"),
            AstNodeKind::Const => write!(f, "const"),
            AstNodeKind::TypeAlias => write!(f, "type"),
            AstNodeKind::Use => write!(f, "use"),
            AstNodeKind::Module => write!(f, "mod"),
            AstNodeKind::Unknown => write!(f, "unknown"),
        }
    }
}

// ============================================================================
// AST HUNTER — The surgical workspace extractor
// ============================================================================

/// The surgical workspace extractor. Uses ripgrep-speed file walking via
/// the `ignore` crate to traverse the workspace, then performs lightweight
/// regex-based symbol extraction (foundation for tree-sitter integration).
///
/// The extracted AST leaves are routed directly to McpSensoryCortex::ingest_ast_node,
/// converting them into PhasorBoundaryConditions for the RK4 solver.
pub struct AstHunter {
    /// Workspace root directory
    workspace_root: PathBuf,
    /// The sensory cortex for converting AST nodes into boundary conditions
    sensory_cortex: DefaultMcpSensoryCortex,
    /// Cached symbol index: symbol_name → (file_path, line_number)
    /// Built on first traversal, used for O(1) lookup on subsequent hunts
    symbol_index: HashMap<String, Vec<(PathBuf, usize)>>,
    /// Whether the symbol index has been built
    index_built: bool,
    /// File extensions to search
    search_extensions: Vec<&'static str>,
}

impl AstHunter {
    /// Create a new AstHunter for the given workspace root.
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            workspace_root,
            sensory_cortex: DefaultMcpSensoryCortex::new(),
            symbol_index: HashMap::new(),
            index_built: false,
            search_extensions: vec!["rs", "py", "toml", "json"],
        }
    }

    /// Create with custom sensory cortex parameters.
    pub fn with_sensory_cortex(
        workspace_root: PathBuf,
        sensory_cortex: DefaultMcpSensoryCortex,
    ) -> Self {
        Self {
            workspace_root,
            sensory_cortex,
            symbol_index: HashMap::new(),
            index_built: false,
            search_extensions: vec!["rs", "py", "toml", "json"],
        }
    }

    /// Build the symbol index by walking the entire workspace once.
    /// This is the expensive operation — subsequent `hunt_symbol` calls
    /// use the cached index for O(1) lookup.
    pub fn build_index(&mut self) -> Result<usize, String> {
        self.symbol_index.clear();
        let mut total_symbols = 0;

        let walker = WalkBuilder::new(&self.workspace_root)
            .hidden(true)           // skip hidden files
            .git_ignore(true)       // respect .gitignore
            .git_global(true)       // respect global gitignore
            .git_exclude(true)      // respect .git/info/exclude
            .build();

        for result in walker {
            match result {
                Ok(entry) => {
                    let path = entry.path();
                    if !path.is_file() {
                        continue;
                    }

                    let ext = path.extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("");

                    if !self.search_extensions.contains(&ext) {
                        continue;
                    }

                    if let Ok(symbols) = self.extract_symbols_from_file(path) {
                        for (name, line) in symbols {
                            self.symbol_index
                                .entry(name)
                                .or_default()
                                .push((path.to_path_buf(), line));
                            total_symbols += 1;
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        self.index_built = true;
        Ok(total_symbols)
    }

    /// Hunt for a specific symbol by name across the workspace.
    /// Returns all AST nodes matching the symbol name, with their full
    /// source bodies extracted and ready for ingestion into the
    /// McpSensoryCortex.
    ///
    /// If the index has not been built, this will trigger a full workspace walk.
    pub fn hunt_symbol(&mut self, symbol_name: &str) -> Vec<AstNode> {
        if !self.index_built {
            let _ = self.build_index();
        }

        let locations = match self.symbol_index.get(symbol_name) {
            Some(locs) => locs.clone(),
            None => return Vec::new(),
        };

        let mut results = Vec::new();

        for (file_path, hint_line) in locations {
            if let Some(node) = self.extract_symbol_body(&file_path, symbol_name, hint_line) {
                results.push(node);
            }
        }

        results
    }

    /// Hunt for a symbol and immediately convert it to a PhasorBoundaryCondition
    /// via the McpSensoryCortex. This is the hot-path: symbol → phasor → ODE.
    pub fn hunt_and_ingest(&mut self, symbol_name: &str) -> Vec<PhasorBoundaryCondition> {
        let nodes = self.hunt_symbol(symbol_name);
        nodes
            .iter()
            .map(|node| {
                self.sensory_cortex.ingest_ast_node(
                    &node.kind.to_string(),
                    &node.source_text,
                    node.domain_flag,
                )
            })
            .collect()
    }

    /// Hunt for all symbols matching a regex pattern.
    pub fn hunt_pattern(&mut self, pattern: &str) -> Vec<AstNode> {
        if !self.index_built {
            let _ = self.build_index();
        }

        let regex = match regex::Regex::new(pattern) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        let matching_names: Vec<String> = self
            .symbol_index
            .keys()
            .filter(|name| regex.is_match(name))
            .cloned()
            .collect();

        let mut results = Vec::new();
        for name in matching_names {
            results.extend(self.hunt_symbol(&name));
        }
        results
    }

    /// Get the number of indexed symbols.
    pub fn index_size(&self) -> usize {
        self.symbol_index.len()
    }

    /// Check if the index has been built.
    pub fn is_indexed(&self) -> bool {
        self.index_built
    }

    // ========================================================================
    // INTERNAL: Lightweight symbol extraction (regex-based, tree-sitter foundation)
    // ========================================================================

    /// Extract symbol names and their line numbers from a single file.
    /// This is a lightweight regex-based extraction that serves as the
    /// foundation for tree-sitter integration. When tree-sitter is wired in
    /// via an MCP tool, this method will be replaced with proper AST parsing.
    fn extract_symbols_from_file(&self, path: &Path) -> Result<Vec<(String, usize)>, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        let mut symbols = Vec::new();

        // Rust symbol patterns
        let patterns = [
            (regex::Regex::new(r"^\s*(?:pub\s+)?fn\s+(\w+)").unwrap(), AstNodeKind::Function),
            (regex::Regex::new(r"^\s*(?:pub\s+)?struct\s+(\w+)").unwrap(), AstNodeKind::Struct),
            (regex::Regex::new(r"^\s*(?:pub\s+)?enum\s+(\w+)").unwrap(), AstNodeKind::Enum),
            (regex::Regex::new(r"^\s*(?:pub\s+)?trait\s+(\w+)").unwrap(), AstNodeKind::Trait),
            (regex::Regex::new(r"^\s*(?:pub\s+)?impl\s+(?:\w+\s+)?(?:for\s+)?(\w+)").unwrap(), AstNodeKind::Impl),
            (regex::Regex::new(r"^\s*(?:pub\s+)?const\s+(\w+)").unwrap(), AstNodeKind::Const),
            (regex::Regex::new(r"^\s*(?:pub\s+)?type\s+(\w+)").unwrap(), AstNodeKind::TypeAlias),
        ];

        for (line_num, line) in content.lines().enumerate() {
            for (pattern, _kind) in &patterns {
                if let Some(captures) = pattern.captures(line) {
                    if let Some(name) = captures.get(1) {
                        symbols.push((name.as_str().to_string(), line_num + 1));
                    }
                }
            }
        }

        Ok(symbols)
    }

    /// Extract the full body of a specific symbol from a file, given a hint line.
    /// Uses brace-depth tracking to find the complete definition body.
    fn extract_symbol_body(
        &self,
        file_path: &Path,
        symbol_name: &str,
        hint_line: usize,
    ) -> Option<AstNode> {
        let content = std::fs::read_to_string(file_path).ok()?;
        let lines: Vec<&str> = content.lines().collect();

        if hint_line == 0 || hint_line > lines.len() {
            return None;
        }

        // Find the start of the definition (the hint line or nearby)
        let start_idx = hint_line.saturating_sub(1);
        let start_line = lines.get(start_idx)?;

        // Determine the kind and domain from the start line
        let (kind, domain_flag) = classify_symbol(start_line);

        // Track brace depth to find the end of the definition
        let mut brace_depth = 0i32;
        let mut found_open_brace = false;
        let mut end_idx = start_idx;

        for i in start_idx..lines.len() {
            let line = lines[i];
            for ch in line.chars() {
                match ch {
                    '{' => {
                        brace_depth += 1;
                        found_open_brace = true;
                    }
                    '}' => {
                        brace_depth -= 1;
                    }
                    _ => {}
                }
            }

            if found_open_brace && brace_depth <= 0 {
                end_idx = i;
                break;
            }

            // Safety: don't read more than 200 lines for a single symbol
            if i - start_idx > 200 {
                end_idx = i;
                break;
            }
        }

        // If no braces found (e.g., type alias without braces), take just the line
        if !found_open_brace {
            end_idx = start_idx;
        }

        let source_text = lines[start_idx..=end_idx].join("\n");

        Some(AstNode {
            name: symbol_name.to_string(),
            kind,
            source_text,
            file_path: file_path.to_path_buf(),
            line_start: start_idx + 1,
            line_end: end_idx + 1,
            domain_flag,
        })
    }
}

/// Classify a symbol definition line into its AST kind and neuro-symbolic domain.
fn classify_symbol(line: &str) -> (AstNodeKind, u32) {
    let trimmed = line.trim();

    if trimmed.contains("fn ") {
        // Functions with loop/iter in name → TEMPORAL
        // Functions with compute/process → LOGIC
        // Other functions → LOGIC (default)
        let domain = if trimmed.contains("loop")
            || trimmed.contains("iter")
            || trimmed.contains("next")
            || trimmed.contains("step")
        {
            DOMAIN_TEMPORAL
        } else {
            DOMAIN_LOGIC
        };
        (AstNodeKind::Function, domain)
    } else if trimmed.contains("struct ") {
        (AstNodeKind::Struct, DOMAIN_SPATIAL)
    } else if trimmed.contains("enum ") {
        (AstNodeKind::Enum, DOMAIN_LOGIC)
    } else if trimmed.contains("trait ") {
        (AstNodeKind::Trait, DOMAIN_LOGIC)
    } else if trimmed.contains("impl ") {
        (AstNodeKind::Impl, DOMAIN_LOGIC)
    } else if trimmed.contains("const ") {
        (AstNodeKind::Const, DOMAIN_ABSTRACT)
    } else if trimmed.contains("type ") {
        (AstNodeKind::TypeAlias, DOMAIN_SPATIAL)
    } else {
        (AstNodeKind::Unknown, DOMAIN_ABSTRACT)
    }
}
