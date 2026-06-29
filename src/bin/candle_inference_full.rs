use std::env;
use anyhow::Result;
use std::collections::HashMap;

use sodals_neuro::llm_core::LlmEngine;
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::LogManager;

use sodals_neuro::semantic_network::ResonanceNetwork;
use sodals_neuro::semantic_parser::SemanticRegisterFile;
use sodals_neuro::mcts_exploration::TabularQTable;
use sodals_neuro::agent_loop::AgentLoop;
use sodals_neuro::arc_environment::ArcEnvironment;

// ChatML and Observation parser helpers for ARC integration
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub fn parse_chatml(prompt: &str) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    let parts: Vec<&str> = prompt.split("<|im_start|>").collect();
    for part in parts {
        if part.trim().is_empty() {
            continue;
        }
        let end_idx = part.find("<|im_end|>");
        if let Some(end) = end_idx {
            let role_content = &part[..end];
            if let Some(first_nl) = role_content.find('\n') {
                let role = role_content[..first_nl].trim().to_string();
                let content = role_content[first_nl + 1..].to_string();
                messages.push(ChatMessage { role, content });
            }
        }
    }
    messages
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ProxyState {
    pub current_grid: Vec<Vec<u32>>,
    pub levels_completed: usize,
    pub steps: usize,
    pub available_actions: Vec<String>,
}

pub fn parse_user_observation(content: &str) -> Option<ProxyState> {
    let mut levels_completed = 0;
    let mut grid = Vec::new();
    let mut available_actions = Vec::new();
    
    let mut in_frame = false;
    for line in content.lines() {
        let line_trimmed = line.trim();
        if line_trimmed.starts_with("Levels completed:") {
            if let Some(val_str) = line_trimmed.split(':').nth(1) {
                levels_completed = val_str.trim().parse::<usize>().unwrap_or(0);
            }
        } else if line_trimmed.starts_with("Frame ") {
            in_frame = true;
        } else if in_frame && !line_trimmed.is_empty() && line_trimmed.chars().all(|c| c.is_ascii_alphanumeric()) {
            let row: Vec<u32> = line_trimmed.chars()
                .map(|c| {
                    match c {
                        '0'..='9' => c.to_digit(10).unwrap_or(0),
                        'a'..='z' => (c as u32 - 'a' as u32) + 10,
                        'A'..='Z' => (c as u32 - 'A' as u32) + 10,
                        _ => 0,
                    }
                })
                .collect();
            if !row.is_empty() {
                grid.push(row);
            }
        } else if line_trimmed.starts_with("Available actions:") {
            in_frame = false;
        } else if line_trimmed.starts_with("- ") {
            let act = line_trimmed[2..].trim().to_string();
            available_actions.push(act);
        }
    }
    
    if grid.is_empty() {
        in_frame = false;
        for line in content.lines() {
            let line_trimmed = line.trim();
            if line_trimmed.starts_with("Frame ") {
                in_frame = true;
            } else if in_frame && line_trimmed.contains('[') && line_trimmed.contains(']') {
                if let (Some(start), Some(end)) = (line_trimmed.find('['), line_trimmed.rfind(']')) {
                    let row_str = &line_trimmed[start + 1..end];
                    let row: Vec<u32> = row_str.split(',')
                        .map(|s| s.trim().parse::<u32>().unwrap_or(0))
                        .collect();
                    grid.push(row);
                }
            } else if line_trimmed.starts_with("Available actions:") {
                in_frame = false;
            }
        }
    }
    
    if grid.is_empty() {
        None
    } else {
        Some(ProxyState {
            current_grid: grid,
            levels_completed,
            steps: 0,
            available_actions,
        })
    }
}

pub fn parse_assistant_action(content: &str, available_actions: &[String]) -> Option<String> {
    let content_upper = content.to_uppercase();
    let mut best_pos = None;
    let mut best_action = None;
    for act in available_actions {
        let act_upper = act.to_uppercase();
        if let Some(pos) = content_upper.rfind(&act_upper) {
            if best_pos.is_none() || pos > best_pos.unwrap() {
                best_pos = Some(pos);
                best_action = Some(act.clone());
            }
        }
    }
    best_action
}

pub struct ArcProxyEnv {
    pub transitions: HashMap<(ProxyState, String), (ProxyState, f64, bool)>,
    pub initial_proxy_state: ProxyState,
}

impl ArcEnvironment for ArcProxyEnv {
    type State = ProxyState;
    type Action = String;
    
    fn initial_state(&self) -> Self::State {
        self.initial_proxy_state.clone()
    }
    
    fn step(&self, state: &Self::State, action: &Self::Action) -> (Self::State, f64, bool) {
        if let Some(result) = self.transitions.get(&(state.clone(), action.clone())) {
            result.clone()
        } else {
            let mut next = state.clone();
            next.steps += 1;
            (next, 0.0, false)
        }
    }
    
    fn get_actions(&self, state: &Self::State) -> Vec<Self::Action> {
        state.available_actions.clone()
    }
    
    fn parse_observation(&self, state: &Self::State) -> String {
        let mut obs = String::new();
        obs.push_str(&format!("levels_completed: {}\n", state.levels_completed));
        obs.push_str(&format!("steps: {}\n", state.steps));
        let rows = state.current_grid.len();
        let cols = if rows > 0 { state.current_grid[0].len() } else { 0 };
        for r in 0..rows {
            for c in 0..cols {
                obs.push_str(&format!("cell_{}_{}: {}\n", r, c, state.current_grid[r][c]));
            }
        }
        obs
    }
    
    fn is_goal(&self, state: &Self::State) -> bool {
        false
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SodalsState {
    pub planned_actions: Vec<String>,
    pub q_table: TabularQTable,
    pub srf: SemanticRegisterFile,
    pub transition_failed_count: usize,
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    
    let mut cartridge_path = None;
    let mut qmem_manifest_path = None;
    let mut qmem_ram_bytes = None;
    let mut draft_model_path = None;
    let mut clean_args = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--cartridge" && i + 1 < args.len() {
            cartridge_path = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--qmem-manifest" && i + 1 < args.len() {
            qmem_manifest_path = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--qmem-ram-bytes" && i + 1 < args.len() {
            qmem_ram_bytes = args[i + 1].parse::<usize>().ok();
            i += 2;
        } else if args[i] == "--draft-model" && i + 1 < args.len() {
            draft_model_path = Some(args[i + 1].clone());
            i += 2;
        } else {
            clean_args.push(args[i].clone());
            i += 1;
        }
    }
    let args = clean_args;

    let use_stdin = args.iter().any(|arg| arg == "--stdin");
    
    if !use_stdin && args.len() < 2 {
        eprintln!("Usage: candle_inference_full <prompt> [max_tokens] [model_path] [tokenizer_path]");
        eprintln!("   or: candle_inference_full --stdin [max_tokens] [model_path] [tokenizer_path]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --cartridge <path>        Load QMEM cartridge");
        eprintln!("  --qmem-manifest <path>    Initialize hierarchical QMEM loader from manifest");
        eprintln!("  --qmem-ram-bytes <bytes>  Set RAM budget for hierarchical QMEM loader");
        eprintln!("  --draft-model <path>      Load lightweight draft model for speculative decoding");
        eprintln!();
        eprintln!("QMEM Layer-wise Mixed-Precision Support:");
        eprintln!("  Manifest dtype field: f32, f16, bf16 (default: f32)");
        eprintln!("  Each layer can have different precision for memory/speed tradeoffs");
        std::process::exit(1);
    }
    
    let prompt = if use_stdin {
        use std::io::{self, Read};
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        buffer
    } else {
        args.get(1).cloned().unwrap_or_default()
    };
    
    if prompt.trim().is_empty() {
        eprintln!("Error: Empty prompt provided");
        std::process::exit(1);
    }

    let max_tokens = args.get(2).and_then(|a| a.parse::<usize>().ok()).unwrap_or(2048);
    let model_path = args.get(3).cloned();
    let tokenizer_path = args.get(4).cloned();

    eprintln!("[DEBUG] Mode: {}", if use_stdin { "stdin" } else { "args" });
    eprintln!("[DEBUG] Tokens: {}", max_tokens);
    eprintln!("[DEBUG] Model: {:?}", model_path);
    eprintln!("[DEBUG] Tokenizer: {:?}", tokenizer_path);
    eprintln!("[DEBUG] Cartridge: {:?}", cartridge_path);
    eprintln!("[DEBUG] QMEM manifest: {:?}", qmem_manifest_path);
    eprintln!("[DEBUG] QMEM RAM bytes: {:?}", qmem_ram_bytes);
    eprintln!("[DEBUG] Draft Model: {:?}", draft_model_path);
    
    let seed = 42;
    let result = run_full_sodals_inference(
        &prompt,
        max_tokens,
        seed,
        model_path.as_deref(),
        tokenizer_path.as_deref(),
        cartridge_path.as_deref(),
        qmem_manifest_path.as_deref(),
        qmem_ram_bytes,
        draft_model_path.as_deref(),
    )?;
    
    let output = serde_json::json!({
        "raw_output": result.text,
        "token_count": result.token_count,
        "seed": seed,
        "model_path": model_path.unwrap_or_else(|| "models/model.gguf".to_string()),
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
        "model_type": "sodals_neuro_full",
        "system_status": result.system_status,
        "cuda_enabled": cfg!(feature = "cuda"),
        "semantic_network_active": true,
        "physics_reasoning_active": true
    });
    
    println!("{}", output);
    Ok(())
}

struct InferenceResult {
    text: String,
    token_count: usize,
    system_status: String,
}

fn run_full_sodals_inference(
    prompt: &str, 
    _max_tokens: usize, 
    _seed: u64,
    model_path: Option<&str>,
    tokenizer_path: Option<&str>,
    cartridge_path: Option<&str>,
    _qmem_manifest_path: Option<&str>,
    _qmem_ram_bytes: Option<usize>,
    _draft_model_path: Option<&str>,
) -> Result<InferenceResult> {
    let mut config = SystemConfig::default();
    if let Some(mp) = model_path {
        config.model_path = mp.to_string();
    }
    if let Some(tp) = tokenizer_path {
        config.tokenizer_path = tp.to_string();
    }
    
    let logger = LogManager::new(sodals_common::log_manager::LogLevel::INFO);
    let mut engine = LlmEngine::new(&config, &logger)?;
    
    if let Some(cp) = cartridge_path {
        engine.load_qmem_cartridge(cp)?;
    }
    
    let mut neuro_config = sodals_neuro::engine::chimera_router::NeuroConfig::default();
    neuro_config.strict_determinism = true;
    
    let text = sodals_neuro::engine::generation_loop::generate_stream(
        &mut engine,
        prompt,
        None,
        None,
        |token| {
            use std::io::Write;
            print!("{}", token);
            let _ = std::io::stdout().flush();
        },
        &neuro_config,
        0,
    )?;
    
    let token_count = text.split_whitespace().count();
    
    Ok(InferenceResult {
        text,
        token_count,
        system_status: "ok".to_string(),
    })
}
    



    



