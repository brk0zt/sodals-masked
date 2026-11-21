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

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    
    let mut cartridge_path = None;
    let mut qmem_manifest_path = None;
    let mut qmem_ram_bytes = None;
    let mut mab_dir_path = None;
    let mut draft_model_path = None;
    let mut prompt_override = None;
    let mut max_tokens_override = None;
    let mut temperature = None;
    let mut repeat_penalty = None;
    let mut image_path = None;
    let mut model_path = None;
    let mut tokenizer_path = None;
    let mut min_think_override = None;
    let mut minimal_mode = false;
    let mut clean_args = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--cartridge" && i + 1 < args.len() {
            cartridge_path = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--mab-dir" && i + 1 < args.len() {
            mab_dir_path = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--image" && i + 1 < args.len() {
            image_path = Some(args[i + 1].clone());
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
        } else if args[i] == "--prompt" && i + 1 < args.len() {
            prompt_override = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--max-tokens" && i + 1 < args.len() {
            max_tokens_override = args[i + 1].parse::<usize>().ok();
            i += 2;
        } else if args[i] == "--temperature" && i + 1 < args.len() {
            temperature = args[i + 1].parse::<f64>().ok();
            i += 2;
        } else if args[i] == "--repeat-penalty" && i + 1 < args.len() {
            repeat_penalty = args[i + 1].parse::<f32>().ok();
            i += 2;
        } else if args[i] == "--model" && i + 1 < args.len() {
            model_path = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--tokenizer" && i + 1 < args.len() {
            tokenizer_path = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--min-think" && i + 1 < args.len() {
            min_think_override = args[i + 1].parse::<usize>().ok();
            i += 2;
        } else if args[i] == "--minimal-mode" {
            minimal_mode = true;
            i += 1;
        } else {
            clean_args.push(args[i].clone());
            i += 1;
        }
    }
    let args = clean_args;

    let use_stdin = args.iter().any(|arg| arg == "--stdin");
    
    // A --prompt override satisfies the argument requirement even when no positional
    // prompt was given (the option parser consumed it out of clean_args).
    if !use_stdin && prompt_override.is_none() && args.len() < 2 {
        eprintln!("Usage: candle_inference_full <prompt> [max_tokens] [model_path] [tokenizer_path]");
        eprintln!("   or: candle_inference_full --stdin [max_tokens] [model_path] [tokenizer_path]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --image <path>            Load image for multimodal inference");
        eprintln!("  --cartridge <path>        Load QMEM cartridge");
        eprintln!("  --qmem-manifest <path>    Initialize hierarchical QMEM loader from manifest");
        eprintln!("  --qmem-ram-bytes <bytes>  Set RAM budget for hierarchical QMEM loader");
        eprintln!("  --draft-model <path>      Load lightweight draft model for speculative decoding");
        eprintln!("  --model <path>            Override GGUF model path (default: ./models/model.gguf)");
        eprintln!("  --tokenizer <path>        Override tokenizer path (default: ./models/tokenizer.json)");
        eprintln!("  --min-think <tokens>      Override min-think gate (0 = no forced thinking)");
        eprintln!();
        eprintln!("QMEM Layer-wise Mixed-Precision Support:");
        eprintln!("  Manifest dtype field: f32, f16, bf16 (default: f32)");
        eprintln!("  Each layer can have different precision for memory/speed tradeoffs");
        std::process::exit(1);
    }
    
    let prompt = prompt_override.unwrap_or_else(|| {
        if use_stdin {
            use std::io::{self, Read};
            let mut buffer = String::new();
            io::stdin().read_to_string(&mut buffer).unwrap_or_default();
            if buffer.starts_with('\u{feff}') {
                buffer.remove(0);
            }
            buffer.replace("\r\n", "\n")
        } else {
            args.get(1).cloned().unwrap_or_default()
        }
    });
    
    if prompt.trim().is_empty() {
        eprintln!("Error: Empty prompt provided");
        std::process::exit(1);
    }

    let max_tokens = max_tokens_override.unwrap_or_else(|| {
        args.get(2).and_then(|a: &String| a.parse::<usize>().ok()).unwrap_or(2048)
    });
    let model_path = model_path.or_else(|| args.get(3).cloned());
    let tokenizer_path = tokenizer_path.or_else(|| args.get(4).cloned());

    eprintln!("[DEBUG] Mode: {}", if use_stdin { "stdin" } else { "args" });
    eprintln!("[DEBUG] Tokens: {}", max_tokens);
    eprintln!("[DEBUG] Model: {:?}", model_path);
    eprintln!("[DEBUG] Tokenizer: {:?}", tokenizer_path);
    eprintln!("[DEBUG] Cartridge: {:?}", cartridge_path);
    eprintln!("[DEBUG] QMEM manifest: {:?}", qmem_manifest_path);
    eprintln!("[DEBUG] QMEM RAM bytes: {:?}", qmem_ram_bytes);
    eprintln!("[DEBUG] MAB Directory: {:?}", mab_dir_path);
    eprintln!("[DEBUG] Draft Model: {:?}", draft_model_path);
    eprintln!("[DEBUG] Image: {:?}", image_path);
    
    let seed = 42;
    let result = run_full_sodals_inference(
        &prompt,
        max_tokens,
        seed,
        model_path.as_deref(),
        tokenizer_path.as_deref(),
        cartridge_path.as_deref(),
        mab_dir_path.as_deref(),
        qmem_manifest_path.as_deref(),
        qmem_ram_bytes,
        draft_model_path.as_deref(),
        temperature,
        repeat_penalty,
        image_path.as_deref(),
        min_think_override,
        minimal_mode,
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

    // Fix: Add guaranteed newline before JSON output to prevent parsing issues
    // when progressive token printing doesn't end with newline
    println!(); // Ensure we're on a fresh line for stdout
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
    max_tokens: usize, 
    _seed: u64,
    model_path: Option<&str>,
    tokenizer_path: Option<&str>,
    cartridge_path: Option<&str>,
    mab_dir_path: Option<&str>,
    _qmem_manifest_path: Option<&str>,
    _qmem_ram_bytes: Option<usize>,
    _draft_model_path: Option<&str>,
    temperature: Option<f64>,
    repeat_penalty: Option<f32>,
    image_path: Option<&str>,
    min_think_override: Option<usize>,
    minimal_mode: bool,
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
    } else if let Some(mab_dir) = mab_dir_path {
        engine.init_cartridge_library(mab_dir, 1.414);
        engine.load_best_cartridge(None)?;
    }
    
    let mut neuro_config = sodals_neuro::engine::chimera_router::NeuroConfig::default();
    if let Some(t) = temperature {
        neuro_config.temperature = t;
        neuro_config.sampling_override = true;
        if t == 0.0 {
            neuro_config.strict_determinism = true;
        }
    }
    if let Some(rp) = repeat_penalty {
        neuro_config.repeat_penalty = rp;
        neuro_config.sampling_override = true;
    }
    if let Some(mt) = min_think_override {
        neuro_config.min_think_tokens = Some(mt);
    }
    if minimal_mode {
        neuro_config.minimal_mode = true;
    }
    
    eprintln!("[DEBUG] token_to_id('<think>') = {:?}", engine.tokenizer.token_to_id("<think>"));
    
    let text = if let Some(img_path) = image_path {
        let image_bytes = std::fs::read(img_path)?;
        engine.generate_multimodal(
            prompt,
            &[("image/jpeg", &image_bytes)],
            &neuro_config,
            max_tokens
        )?
    } else {
        sodals_neuro::engine::generation_loop::generate_stream(
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
            0, // visual_token_count
            max_tokens,
        )?
    };
    
    let token_count = engine.tokenizer.encode(text.as_str(), false)
        .map(|e| e.get_ids().len())
        .unwrap_or_else(|_| text.split_whitespace().count());
    
    if mab_dir_path.is_some() {
        engine.observe_cartridge_reward();
    }
    
    Ok(InferenceResult {
        text,
        token_count,
        system_status: "ok".to_string(),
    })
}
    



    





