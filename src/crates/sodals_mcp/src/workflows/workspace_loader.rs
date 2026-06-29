// src/crates/sodals_mcp/src/workflows/workspace_loader.rs
// ═══════════════════════════════════════════════════════════════════════
// Workspace Loader — ICM Filesystem Hierarchy Traversal
// ═══════════════════════════════════════════════════════════════════════
//
// Implements ICM (Interpretable Context Methodology) workspace traversal:
// - Load Layer 0: AGENT.md (root identity)
// - Load Layer 1: CONTEXT.md (global context)
// - Traverse numbered stages (01 → 02 → 03 → 04 → 05)
// - Load stage-specific CONTEXT.md
// - Load reference skills from references/ folder
// - Execute stage with loaded context
// - Pass output to next stage
//
// ═══════════════════════════════════════════════════════════════════════

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::registry::ToolHandler;
use crate::types::*;

// ============================================================================
// WORKSPACE STRUCTURE
// ============================================================================

/// Represents the ICM workspace structure
#[derive(Debug, Clone)]
pub struct Workspace {
    pub path: PathBuf,
    pub agent: AgentIdentity,
    pub global_context: GlobalContext,
    pub stages: Vec<Stage>,
}

/// Layer 0: Root Identity (AGENT.md)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIdentity {
    pub persona: String,
    pub safety_rules: Vec<String>,
    pub tool_usage: Vec<String>,
    pub output_constraints: Vec<String>,
}

/// Layer 1: Global Context (CONTEXT.md)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalContext {
    pub mission: String,
    pub objectives: Vec<String>,
    pub success_metrics: HashMap<String, String>,
    pub constraints: Vec<String>,
}

/// Stage definition
#[derive(Debug, Clone)]
pub struct Stage {
    pub name: String,
    pub number: usize,
    pub path: PathBuf,
    pub context: StageContext,
    pub references: Vec<ReferenceSkill>,
}

/// Stage-specific context (CONTEXT.md)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageContext {
    pub purpose: String,
    pub input_contract: serde_json::Value,
    pub output_contract: serde_json::Value,
    pub processing_steps: Vec<String>,
}

/// Reference skill (Layer 3: references/)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceSkill {
    pub name: String,
    pub content: String,
    pub path: PathBuf,
}

/// Workspace execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceResult {
    pub stage_outputs: HashMap<String, serde_json::Value>,
    pub final_output: serde_json::Value,
    pub execution_time_ms: u64,
}

// ============================================================================
// WORKSPACE LOADER
// ============================================================================

/// Loads and traverses ICM workspace structure
pub struct WorkspaceLoader {
    workspace_path: PathBuf,
}

impl WorkspaceLoader {
    pub fn new(workspace_path: impl Into<PathBuf>) -> Self {
        Self {
            workspace_path: workspace_path.into(),
        }
    }

    /// Load complete workspace structure
    pub fn load(&self) -> Result<Workspace> {
        info!("Loading workspace from: {:?}", self.workspace_path);

        // Validate workspace exists
        if !self.workspace_path.exists() {
            anyhow::bail!("Workspace path does not exist: {:?}", self.workspace_path);
        }

        // Load Layer 0: AGENT.md
        let agent = self.load_agent_identity()?;

        // Load Layer 1: CONTEXT.md
        let global_context = self.load_global_context()?;

        // Load numbered stages
        let stages = self.load_stages()?;

        info!("Workspace loaded successfully");
        info!("  Agent: {}", agent.persona);
        info!("  Stages: {}", stages.len());

        Ok(Workspace {
            path: self.workspace_path.clone(),
            agent,
            global_context,
            stages,
        })
    }

    /// Load Layer 0: AGENT.md
    fn load_agent_identity(&self) -> Result<AgentIdentity> {
        let agent_path = self.workspace_path.join("AGENT.md");
        
        if !agent_path.exists() {
            anyhow::bail!("AGENT.md not found at: {:?}", agent_path);
        }

        let content = std::fs::read_to_string(&agent_path)
            .context("Failed to read AGENT.md")?;

        // Parse markdown content (simplified parsing)
        let persona = extract_section(&content, "Core Persona")
            .unwrap_or_else(|| "SODALS AI Agent".to_string());
        
        let safety_rules = extract_list_section(&content, "Safety Rules")
            .unwrap_or_default();
        
        let tool_usage = extract_list_section(&content, "Tool Usage")
            .unwrap_or_default();
        
        let output_constraints = extract_list_section(&content, "Output Constraints")
            .unwrap_or_default();

        debug!("Loaded agent identity: {}", persona);

        Ok(AgentIdentity {
            persona,
            safety_rules,
            tool_usage,
            output_constraints,
        })
    }

    /// Load Layer 1: CONTEXT.md
    fn load_global_context(&self) -> Result<GlobalContext> {
        let context_path = self.workspace_path.join("CONTEXT.md");
        
        if !context_path.exists() {
            anyhow::bail!("CONTEXT.md not found at: {:?}", context_path);
        }

        let content = std::fs::read_to_string(&context_path)
            .context("Failed to read CONTEXT.md")?;

        // Parse markdown content (simplified parsing)
        let mission = extract_section(&content, "Mission Statement")
            .unwrap_or_else(|| "Industrial AI reasoning".to_string());
        
        let objectives = extract_list_section(&content, "Core Objectives")
            .unwrap_or_default();
        
        // Parse success metrics (simplified)
        let success_metrics = HashMap::new(); // TODO: Parse metrics table
        
        let constraints = extract_list_section(&content, "High-Level Constraints")
            .unwrap_or_default();

        debug!("Loaded global context: {}", mission);

        Ok(GlobalContext {
            mission,
            objectives,
            success_metrics,
            constraints,
        })
    }

    /// Load numbered stages (01, 02, 03, 04, 05)
    fn load_stages(&self) -> Result<Vec<Stage>> {
        let mut stages = Vec::new();

        // Look for numbered directories (01_*, 02_*, etc.)
        for entry in self.workspace_path.read_dir()? {
            let entry = entry?;
            let path = entry.path();

            if !path.is_dir() {
                continue;
            }

            let name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            // Check if it's a numbered stage (starts with two digits)
            if let Some(num_str) = name.get(..2) {
                if let Ok(number) = num_str.parse::<usize>() {
                    if number >= 1 && number <= 99 {
                        let stage = self.load_stage(&path, number, name)?;
                        stages.push(stage);
                    }
                }
            }
        }

        // Sort by stage number
        stages.sort_by_key(|s| s.number);

        Ok(stages)
    }

    /// Load single stage
    fn load_stage(&self, stage_path: &Path, number: usize, name: &str) -> Result<Stage> {
        debug!("Loading stage: {} ({})", name, number);

        // Load stage-specific CONTEXT.md
        let context_path = stage_path.join("CONTEXT.md");
        let context = if context_path.exists() {
            let content = std::fs::read_to_string(&context_path)
                .context("Failed to read stage CONTEXT.md")?;

            let purpose = extract_section(&content, "Stage Purpose")
                .unwrap_or_else(|| format!("Stage {}", number));
            
            // Parse input/output contracts (simplified - assume JSON in markdown)
            let input_contract = serde_json::json!({}); // TODO: Parse actual contract
            let output_contract = serde_json::json!({}); // TODO: Parse actual contract
            
            let processing_steps = extract_list_section(&content, "Processing Steps")
                .unwrap_or_default();

            StageContext {
                purpose,
                input_contract,
                output_contract,
                processing_steps,
            }
        } else {
            warn!("CONTEXT.md not found for stage: {}", name);
            StageContext {
                purpose: format!("Stage {}", number),
                input_contract: serde_json::json!({}),
                output_contract: serde_json::json!({}),
                processing_steps: vec![],
            }
        };

        // Load reference skills from references/ folder
        let references_path = stage_path.join("references");
        let references = if references_path.exists() {
            self.load_reference_skills(&references_path)?
        } else {
            warn!("references/ folder not found for stage: {}", name);
            vec![]
        };

        debug!("Loaded stage: {} with {} reference skills", name, references.len());

        Ok(Stage {
            name: name.to_string(),
            number,
            path: stage_path.to_path_buf(),
            context,
            references,
        })
    }

    /// Load reference skills from references/ folder
    fn load_reference_skills(&self, references_path: &Path) -> Result<Vec<ReferenceSkill>> {
        let mut skills = Vec::new();

        for entry in references_path.read_dir()? {
            let entry = entry?;
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            let ext = path.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");

            if ext != "md" {
                continue;
            }

            let name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            let content = std::fs::read_to_string(&path)
                .context("Failed to read reference skill")?;

            skills.push(ReferenceSkill {
                name,
                content,
                path,
            });
        }

        Ok(skills)
    }
}

// ============================================================================
// WORKSPACE EXECUTOR
// ============================================================================

/// Executes workspace traversal with stage-to-stage context passing
pub struct WorkspaceExecutor {
    workspace: Workspace,
}

impl WorkspaceExecutor {
    pub fn new(workspace: Workspace) -> Self {
        Self { workspace }
    }

    /// Execute workspace traversal
    pub async fn execute(&self, initial_input: serde_json::Value) -> Result<WorkspaceResult> {
        info!("Starting workspace execution");
        
        let start_time = std::time::Instant::now();
        let mut stage_outputs = HashMap::new();
        let mut current_input = initial_input;

        // Execute each stage in order
        for stage in &self.workspace.stages {
            info!("Executing stage: {}", stage.name);

            // Load stage context
            let stage_context = &stage.context;

            // Load reference skills
            let references = &stage.references;

            // TODO: Execute stage with loaded context and references
            // For now, this is a stub that passes through the input
            let stage_output = self.execute_stage(&current_input, stage_context, references).await?;

            // Store stage output
            stage_outputs.insert(stage.name.clone(), stage_output.clone());

            // Pass output to next stage
            current_input = stage_output;

            info!("Stage {} completed", stage.name);
        }

        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        info!("Workspace execution completed in {}ms", execution_time_ms);

        Ok(WorkspaceResult {
            stage_outputs,
            final_output: current_input,
            execution_time_ms,
        })
    }

    /// Execute single stage (stub implementation)
    async fn execute_stage(
        &self,
        input: &serde_json::Value,
        context: &StageContext,
        references: &[ReferenceSkill],
    ) -> Result<serde_json::Value> {
        debug!("Executing stage with purpose: {}", context.purpose);
        debug!("  Reference skills: {}", references.len());

        // TODO: Integrate with actual ToolHandler execution
        // For now, return input as-is (stub)
        Ok(input.clone())
    }
}

// ============================================================================
// MARKDOWN PARSING HELPERS
// ============================================================================

/// Extract a section from markdown content
fn extract_section(content: &str, section_name: &str) -> Option<String> {
    let lines: Vec<&str> = content.lines().collect();
    
    let mut in_section = false;
    let mut section_content = Vec::new();
    
    for line in lines {
        if line.starts_with("## ") && line.contains(section_name) {
            in_section = true;
            continue;
        }
        
        if in_section {
            if line.starts_with("## ") {
                break;
            }
            section_content.push(line);
        }
    }
    
    let content = section_content.join("\n").trim().to_string();
    
    if content.is_empty() {
        None
    } else {
        Some(content)
    }
}

/// Extract a list section from markdown content
fn extract_list_section(content: &str, section_name: &str) -> Option<Vec<String>> {
    let section = extract_section(content, section_name)?;
    
    let items: Vec<String> = section
        .lines()
        .filter(|line| line.starts_with("- ") || line.starts_with("* "))
        .map(|line| line[2..].trim().to_string())
        .collect();
    
    if items.is_empty() {
        None
    } else {
        Some(items)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_section() {
        let content = r#"
## Test Section
This is test content
More content

## Other Section
Other content
"#;
        
        let result = extract_section(content, "Test Section");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "This is test content\nMore content");
    }

    #[test]
    fn test_extract_list_section() {
        let content = r#"
## Test List
- Item 1
- Item 2
- Item 3
"#;
        
        let result = extract_list_section(content, "Test List");
        assert!(result.is_some());
        let items = result.unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], "Item 1");
    }
}
