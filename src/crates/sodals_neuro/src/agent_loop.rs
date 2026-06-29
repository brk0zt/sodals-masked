// src/crates/sodals_neuro/src/agent_loop.rs
// ============================================================================
// SODALS ARC-AGI-3 AGENT INTERACTION LOOP
// ============================================================================

use crate::arc_environment::{ArcEnvironment, GenericEnvironmentAdapter};
use crate::arkhe_pipeline::ArkheCognitionPipeline;
use crate::semantic_network::{BifurcationAction, ResonanceNetwork};
use crate::semantic_parser::SemanticRegisterFile;
use crate::mcts_exploration::{MctsSearch, BeamSearch, TabularQTable, get_qmem_state_hash};
use crate::llm_core::LlmEngine;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{HashMap, HashSet, VecDeque};

pub struct AgentLoop<E: ArcEnvironment> {
    pub env: E,
    pub pipeline: ArkheCognitionPipeline,
    pub adapter: GenericEnvironmentAdapter,
    pub max_steps: usize,
    pub mcts_iterations: usize,
    pub q_table: TabularQTable,
    pub current_state: Option<E::State>,
    pub planned_actions: Vec<String>,
    pub transition_failed_count: usize,
    pub clause_db: CdclClauseDb,
    pub step_counter: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictClause {
    pub antecedents: Vec<String>,
    pub conflicting_action: String,
    pub frequency: u32,
    pub first_seen: u64,
}

impl ConflictClause {
    pub fn matches(&self, current_state_keys: &[String]) -> bool {
        let state_set: HashSet<&String> = current_state_keys.iter().collect();
        self.antecedents.iter().all(|a| state_set.contains(a))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdclClauseDb {
    pub clauses: Vec<ConflictClause>,
    action_index: HashMap<String, Vec<usize>>,
    pub total_conflicts: u64,
    pub clauses_fired: u64,
}

impl CdclClauseDb {
    pub fn new() -> Self {
        Self {
            clauses: Vec::new(),
            action_index: HashMap::new(),
            total_conflicts: 0,
            clauses_fired: 0,
        }
    }

    pub fn learn_conflict(
        &mut self,
        antecedents: Vec<String>,
        conflicting_action: String,
        current_step: u64,
    ) -> &ConflictClause {
        self.total_conflicts += 1;

        let clause_index = self.clauses.iter().position(|c| {
            c.conflicting_action == conflicting_action
                && c.antecedents.len() == antecedents.len()
                && antecedents.iter().all(|a| c.antecedents.contains(a))
        });

        if let Some(idx) = clause_index {
            let clause = &mut self.clauses[idx];
            clause.frequency += 1;
            eprintln!(
                "[CDCL] Conflict reinforced (freq={}): {:?} -> {}",
                clause.frequency, clause.antecedents, clause.conflicting_action
            );
            clause
        } else {
            let idx = self.clauses.len();
            self.clauses.push(ConflictClause {
                antecedents: antecedents.clone(),
                conflicting_action: conflicting_action.clone(),
                frequency: 1,
                first_seen: current_step,
            });
            self.action_index
                .entry(conflicting_action.clone())
                .or_default()
                .push(idx);
            eprintln!(
                "[CDCL] New conflict learned: {:?} -> {}",
                antecedents, conflicting_action
            );
            &self.clauses[idx]
        }
    }

    pub fn is_action_blocked(
        &self,
        action: &str,
        current_state_keys: &[String],
        min_frequency: u32,
    ) -> bool {
        let Some(indices) = self.action_index.get(action) else { return false; };

        indices.iter().any(|&idx| {
            let clause = &self.clauses[idx];
            clause.frequency >= min_frequency && clause.matches(current_state_keys)
        })
    }

    pub fn blocked_actions(
        &self,
        current_state_keys: &[String],
        min_frequency: u32,
    ) -> Vec<String> {
        let mut blocked = Vec::new();
        for clause in &self.clauses {
            if clause.frequency >= min_frequency && clause.matches(current_state_keys) {
                if !blocked.contains(&clause.conflicting_action) {
                    blocked.push(clause.conflicting_action.clone());
                }
            }
        }
        blocked
    }

    pub fn export_to_exp(&self, min_frequency: u32) -> String {
        let mut lines = Vec::new();
        for clause in &self.clauses {
            if clause.frequency < min_frequency {
                continue;
            }
            let antecedent = clause.antecedents.join("|");
            let weight = (clause.frequency as f64 / 10.0).min(1.0);
            lines.push(format!(
                "\"{}\" TRANSITION_FAILED \"{}\" [{:.2}, 0.0];",
                antecedent, clause.conflicting_action, weight
            ));
        }
        lines.join("\n")
    }

    pub fn load_from_file(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(json) => serde_json::from_str(&json).unwrap_or_else(|_| Self::new()),
            Err(_) => Self::new(),
        }
    }

    pub fn save_to_file(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(&self.clauses)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

impl<E: ArcEnvironment> AgentLoop<E> {
    pub fn new(env: E, max_steps: usize, mcts_iterations: usize) -> Self {
        Self {
            env,
            pipeline: ArkheCognitionPipeline::new(),
            adapter: GenericEnvironmentAdapter::new(),
            max_steps,
            mcts_iterations,
            q_table: TabularQTable::new(),
            current_state: None,
            planned_actions: Vec::new(),
            transition_failed_count: 0,
            clause_db: CdclClauseDb::new(),
            step_counter: 0,
        }
    }
    pub fn initialize(
        &mut self,
        engine: Option<&mut LlmEngine>,
        net_g: &mut ResonanceNetwork,
        srf: &mut SemanticRegisterFile,
    ) -> Result<(), String> {
        let initial_state = self.env.initial_state();
        self.current_state = Some(initial_state.clone());
        let obs = self.env.parse_observation(&initial_state);
        
        if let Some(eng) = engine {
            eprintln!("[AGENT LOOP] Phase 1: Calling LLM to synthesize initial QMEM DSL world model...");
            let system_prompt = "You are QYPTOS, an analytical core. Generate QMEM DSL rules describing the objects, constraints, and causal relations.";
            let prompt = format!(
                "<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\nInitial Observation:\n{}\n\nGenerate QMEM DSL rules describing the objects, constraints, and causal relations.\nFormat:\n\"entity_a\" RELATION \"entity_b\" [energy, phase];\nUse only relation types: CAUSES, MEASURES, CORRELATES_PHASE, RESONATES_WITH, REWARD_OBSERVED, TRANSITION_FAILED, RULE_CONFIRMED.\n<|im_end|>\n<|im_start|>assistant\n",
                system_prompt, obs
            );
            
            let mut config = crate::llm_core::NeuroConfig::default();
            config.quiet = true;
            config.temperature = 0.1;
            
            if let Ok(dsl_output) = crate::engine::generation_loop::generate_stream(
                eng, &prompt, None, None, |_t| {}, &config, 0
            ) {
                eprintln!("[AGENT LOOP] Sentezlenen DSL:\n{}", dsl_output);
                if let Ok(cst) = crate::compiler::qmem_compiler::compile(&dsl_output) {
                    crate::compiler::qmem_linker::link_cst_into_network(&cst, net_g);
                    eprintln!("[AGENT LOOP] DSL linked into net_g successfully.");
                }
            }
        }
        
        self.adapter.embed_observation(&obs, net_g);
        self.replan(net_g);
        Ok(())
    }

    pub fn replan(&mut self, net_g: &mut ResonanceNetwork) {
        let state = self.current_state.clone().unwrap_or_else(|| self.env.initial_state());
        let obs = self.env.parse_observation(&state);
        
        let mut start_ids = Vec::new();
        for line in obs.lines() {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() == 2 {
                let key = parts[0].trim();
                let val_str = parts[1].trim();
                let label = format!("obs_{}_{}", key, val_str);
                if let Some(&id) = net_g.labels.get(&label) {
                    start_ids.push(id);
                }
            }
        }
        
        let target_ids = if let Some(&id) = net_g.labels.get("reward") {
            vec![id]
        } else {
            net_g.labels.iter()
                .filter(|(k, _)| k.contains("reward") || k.contains("target"))
                .map(|(_, &v)| v)
                .collect()
        };
        
        if start_ids.is_empty() || target_ids.is_empty() {
            eprintln!("[AGENT LOOP] Replan: start or target concepts not found in network. planned_actions left empty.");
            self.planned_actions = Vec::new();
            return;
        }
        
        let warp_map = net_g.compute_rope_warp_map("motor test");
        let traj = net_g.find_hamiltonian_trajectory_warped(&start_ids, &target_ids, 25, &warp_map);
        
        let id_to_label: HashMap<usize, &String> = net_g.labels.iter()
            .map(|(k, v)| (*v, k)).collect();
            
        let mut new_plan = Vec::new();
        for &id in &traj {
            if let Some(label) = id_to_label.get(&id) {
                if label.starts_with("obs_action_") {
                    let act_str = &label["obs_action_".len()..];
                    new_plan.push(act_str.to_string());
                }
            }
        }
        
        eprintln!("[AGENT LOOP] Replan: calculated Hamiltonian path: {:?}", new_plan);
        self.planned_actions = new_plan;
        self.transition_failed_count = 0;
    }

    pub fn step(
        &mut self,
        net_f: &mut ResonanceNetwork,
        net_g: &mut ResonanceNetwork,
        srf: &mut SemanticRegisterFile,
    ) -> Result<(E::Action, Vec<(String, String, f64)>), String> {
        let state = self.current_state.clone().unwrap_or_else(|| self.env.initial_state());
        let obs = self.env.parse_observation(&state);
        let current_state_keys: Vec<String> = parse_state_properties(&obs)
            .into_iter()
            .filter(|(k, _)| k != "steps")
            .map(|(k, v)| format!("obs_{}_{}", k, v))
            .collect();
        
        let actions = self.env.get_actions(&state);
        if actions.is_empty() {
            return Err("No actions available in current state".to_string());
        }
        
        // A. TabularQTable O(1) Bypass: if state is known and has a strong positive Q-value action, return it immediately
        let s_hash = get_qmem_state_hash(&self.env, &state);
        let mut best_q_action = None;
        let mut best_q = f64::NEG_INFINITY;
        let mut has_strong_signal = false;
        
        for action in &actions {
            let q = self.q_table.get(&s_hash, &format!("{:?}", action));
            if q > best_q {
                best_q = q;
                best_q_action = Some(action.clone());
            }
            if q > 10.0 { // Strong positive signal threshold
                has_strong_signal = true;
            }
        }
        
        if has_strong_signal {
            if let Some(action) = best_q_action {
                eprintln!("[AGENT LOOP] Step: TabularQTable O(1) bypass triggered with Q={:.2}: {:?}", best_q, action);
                return Ok((action, Vec::new()));
            }
        }
        
        // B. Wronskian pre-pruning & Twin Action Redundancy Elimination
        let mut pruned_actions = Vec::new();
        let mut seen_next_states = std::collections::HashSet::new();
        
        for action in &actions {
            let (next_state, _, _) = self.env.step(&state, action);
            if seen_next_states.contains(&next_state) {
                continue; // Prune duplicate state transitions
            }
            
            let child_obs = self.env.parse_observation(&next_state);
            let mut net_child = ResonanceNetwork::new(4096);
            self.adapter.embed_observation(&child_obs, &mut net_child);
            net_child.step(&[]);
            
            self.pipeline.run(net_f, &net_child, &format!("{:?}", action), srf);
            let wronskian = self.pipeline.state.wronskian;
            
            if wronskian >= 0.001 && self.pipeline.state.gate_open {
                pruned_actions.push(action.clone());
                seen_next_states.insert(next_state);
            }
        }
        
        let mut candidate_actions: Vec<E::Action> = if pruned_actions.is_empty() {
            actions.clone()
        } else {
            pruned_actions
        };
        
        let blocked_actions = self.clause_db.blocked_actions(&current_state_keys, 2);
        if !blocked_actions.is_empty() {
            eprintln!("[CDCL] Blocking {} actions: {:?}", blocked_actions.len(), blocked_actions);
        }
        let filtered_candidate_actions: Vec<E::Action> = candidate_actions
            .into_iter()
            .filter(|action| {
                let action_label = format!("{:?}", action);
                !self.clause_db.is_action_blocked(&action_label, &current_state_keys, 2)
            })
            .collect();
        if filtered_candidate_actions.is_empty() {
            eprintln!("[CDCL] All candidate actions blocked by CDCL; reverting to original actions.");
            candidate_actions = actions.clone();
        } else {
            candidate_actions = filtered_candidate_actions;
        }
        
        // C. CliffordBridge Direct Decision Layer: if candidate actions k <= 4, choose directly using h_phase
        if candidate_actions.len() <= 4 {
            let mut best_action = None;
            let mut best_hphase = f64::NEG_INFINITY;
            let mut all_scores = Vec::new();
            
            for action in &candidate_actions {
                let (next_state, _, _) = self.env.step(&state, action);
                let child_obs = self.env.parse_observation(&next_state);
                let mut net_child = ResonanceNetwork::new(4096);
                self.adapter.embed_observation(&child_obs, &mut net_child);
                
                let (h_phase, _) = self.pipeline.bridge.project(net_f, &net_child, 1.0);
                let score = (std::f64::consts::PI - h_phase.abs()).max(0.0);
                all_scores.push(score);
                
                if score > best_hphase {
                    best_hphase = score;
                    best_action = Some(action.clone());
                }
            }
            
            // Check if scores are discriminative (not all identical within a tiny epsilon)
            let mut discriminative = false;
            if !all_scores.is_empty() {
                let first = all_scores[0];
                for &s in &all_scores[1..] {
                    if (s - first).abs() > 1e-5 {
                        discriminative = true;
                        break;
                    }
                }
            }
            
            if discriminative {
                if let Some(action) = best_action {
                    eprintln!("[AGENT LOOP] Step: CliffordBridge direct decision made (k={}): {:?}", candidate_actions.len(), action);
                    return Ok((action, Vec::new()));
                }
            } else {
                eprintln!("[AGENT LOOP] Step: CliffordBridge scores are non-discriminative. Falling back to search.");
            }
        }
        
        // D. Check planned Hamiltonian actions
        if !self.planned_actions.is_empty() {
            let next_planned = self.planned_actions.remove(0);
            if let Some(action) = candidate_actions.iter().find(|a| format!("{:?}", a) == next_planned) {
                eprintln!("[AGENT LOOP] Step: Executing planned Hamiltonian action: {:?}", action);
                return Ok((action.clone(), Vec::new()));
            }
            eprintln!("[AGENT LOOP] Step: Planned action {} not available/valid in candidate list. Clearing plan.", next_planned);
            self.planned_actions.clear();
        }
        
        // E. Fallback to Trie lookup / Beam Search / MCTS
        let rule_count = count_rule_confirmed_in_exp("agent_experience.exp");
        let confidence = if rule_count >= 5 {
            if let Ok(content) = std::fs::read_to_string("agent_experience.exp") {
                let action_strs: Vec<String> = candidate_actions.iter().map(|a| format!("{:?}", a)).collect();
                calculate_predictive_confidence(&obs, &action_strs, &content)
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        let bifurcation_action = net_g.check_bifurcation_action();
        let mut fallback_iterations = self.mcts_iterations;
        if let BifurcationAction::DeepMcts { extra_depth } = bifurcation_action {
            fallback_iterations = self.mcts_iterations.saturating_add(extra_depth as usize);
            eprintln!("[BIFURCATION] Deep MCTS recommended, increasing iterations to {}", fallback_iterations);
        }

        let mut mcts_updates = Vec::new();
        let action_opt = if confidence >= 0.75 {
            eprintln!("[AGENT LOOP] Step: Local confidence {:.2} >= 0.75. Running Beam Search...", confidence);
            let beam = BeamSearch::new(&self.env, 5, &self.q_table);
            beam.search(state.clone(), 5, &mut self.pipeline, net_f, net_g, srf)
        } else {
            if let BifurcationAction::TryAlternative = bifurcation_action {
                eprintln!("[BIFURCATION] Critical regime: trying alternative action selection.");
            }
            eprintln!("[AGENT LOOP] Step: Local confidence {:.2} < 0.75. Running MCTS...", confidence);
            let mut search = MctsSearch::new(&self.env, state.clone(), &self.q_table);
            let opt = search.search(fallback_iterations, &mut self.pipeline, net_f, srf);
            
            mcts_updates = search.nodes.iter()
                .filter(|node| node.visits > 0 && node.action.is_some() && node.parent.is_some())
                .map(|node| {
                    let parent_idx = node.parent.unwrap();
                    let parent_state = &search.nodes[parent_idx].state;
                    let parent_hash = get_qmem_state_hash(&self.env, parent_state);
                    let mcts_q = node.value / node.visits as f64;
                    let action_str = format!("{:?}", node.action.as_ref().unwrap());
                    (parent_hash, action_str, mcts_q)
                })
                .collect();
            opt
        };
        
        let chosen_action = action_opt.ok_or_else(|| "Search failed to select an action".to_string())?;
        Ok((chosen_action, mcts_updates))
    }

    /// Circular agent loop: observes, searches, steps, and refines hypothesis on feedback
    pub fn run(
        &mut self,
        net_f: &mut ResonanceNetwork,
        net_g: &mut ResonanceNetwork,
        srf: &mut SemanticRegisterFile,
    ) -> Result<Vec<E::Action>, String> {
        if self.current_state.is_none() {
            self.initialize(None, net_g, srf)?;
        }

        self.clause_db = CdclClauseDb::load_from_file("agent_clauses.json");
        let mut current_state = self.current_state.clone().unwrap();
        let mut path = Vec::new();

        let exp_path = "agent_experience.exp";
        // Clean the experience file at the start of a new run
        let _ = std::fs::remove_file(exp_path);

        eprintln!("[AGENT LOOP] Starting circular interaction loop...");

        let result = (|| {
            for step in 0..self.max_steps {
                self.step_counter += 1;
                if self.env.is_goal(&current_state) {
                    eprintln!("[AGENT LOOP] Goal achieved at step {}!", step);
                    return Ok(path);
                }

                // 1. Convert state observation to text and embed as QMEM nodes
                let obs = self.env.parse_observation(&current_state);
                let current_state_keys: Vec<String> = parse_state_properties(&obs)
                    .into_iter()
                    .filter(|(k, _)| k != "steps")
                    .map(|(k, v)| format!("obs_{}_{}", k, v))
                    .collect();
                self.adapter.embed_observation(&obs, net_g);

                // Load and link compiled .exp experiences into net_g
                if std::path::Path::new(exp_path).exists() {
                    if let Ok(exp_content) = std::fs::read_to_string(exp_path) {
                        if let Ok(cst) = crate::compiler::qmem_compiler::compile(&exp_content) {
                            crate::compiler::qmem_linker::link_cst_into_network(&cst, net_g);
                        }
                    }
                }

                // 2. Perform step using physical planning
                let (action, mcts_updates) = self.step(net_f, net_g, srf)?;

                // 3. Step the interactive environment
                let (next_state, reward, is_terminal) = self.env.step(&current_state, &action);
                path.push(action.clone());
            
            // 4. Update Tabular Q-Table (Temporal Difference update)
            let s_hash = get_qmem_state_hash(&self.env, &current_state);
            let next_s_hash = get_qmem_state_hash(&self.env, &next_state);
            let next_actions: Vec<String> = self.env.get_actions(&next_state).iter().map(|a| format!("{:?}", a)).collect();
            self.q_table.update(s_hash, format!("{:?}", action), reward, next_s_hash, &next_actions);
            
            // 5. MCTS Supervised Q-Table Training
            for (parent_hash, act_str, mcts_q) in mcts_updates {
                self.q_table.update_from_mcts(parent_hash, act_str, mcts_q);
            }
            
            // 6. Feedback Verification Loop & Dynamic Program Synthesis (Hypothesis Refinement)
            let is_no_op = next_state == current_state;
            let is_failure = (is_terminal && !self.env.is_goal(&next_state)) || is_no_op;
            
            // Synthesize and write experience QMEM DSL statements
            let current_obs = self.env.parse_observation(&current_state);
            let next_obs = self.env.parse_observation(&next_state);
            let _ = append_experience_rules(exp_path, &action, &current_obs, &next_obs, reward, is_failure);
            
            if is_failure {
                self.transition_failed_count += 1;
                let action_label = format!("{:?}", action);
                let failure_label = "obs_steps";
                
                eprintln!("[HYPOTHESIS REFINEMENT] Failure feedback (collision/terminal): {:?}. Inhibiting concept.", action);

                let clause = self.clause_db.learn_conflict(
                    current_state_keys.clone(),
                    action_label.clone(),
                    self.step_counter,
                );
                if clause.frequency >= 2 {
                    if let Ok(mut file) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(exp_path)
                    {
                        let clause_line = format!(
                            "\"{}\" TRANSITION_FAILED \"{}\" [{:.2}, 0.0];\n",
                            clause.antecedents.join("|"),
                            clause.conflicting_action,
                            (clause.frequency as f64 / 10.0).min(1.0)
                        );
                        let _ = std::io::Write::write_all(&mut file, clause_line.as_bytes());
                    }
                }

                // Ensure nodes exist or are registered
                let act_node_label = format!("obs_action_{}", action_label);
                let act_id = net_g.labels.get(&act_node_label).copied().unwrap_or_else(|| {
                    let mut sig = vec![num_complex::Complex64::new(0.0, 0.0); 64];
                    for j in 0..64 {
                        sig[j] = num_complex::Complex64::from_polar(1.0, j as f64 * 0.1);
                    }
                    net_g.add_labeled_concept(&act_node_label, 0.5, sig)
                });
                
                let fail_id = net_g.labels.get(failure_label).copied().unwrap_or_else(|| {
                    let mut sig = vec![num_complex::Complex64::new(0.0, 0.0); 64];
                    for j in 0..64 {
                        sig[j] = num_complex::Complex64::from_polar(1.0, j as f64 * 0.2);
                    }
                    net_g.add_labeled_concept(failure_label, 0.8, sig)
                });

                // Dynamic negative coupling in the semantic grid
                net_g.connect_semantic(act_id, fail_id, -1.0);
                
                // Physics engine momentum damping on failure node to reduce selection probability
                if act_id < net_g.momenta.amplitudes.len() {
                    net_g.momenta.amplitudes[act_id] *= 0.05;
                }
                
                // Trigger replanning since previous path failed
                eprintln!("[AGENT LOOP] Path failed (failure count {}). Triggering replanning...", self.transition_failed_count);
                self.replan(net_g);
            }
            
            current_state = next_state;
            self.current_state = Some(current_state.clone());
            if is_terminal && !self.env.is_goal(&current_state) {
                eprintln!("[AGENT LOOP] Terminal state hit without reaching goal");
                break;
            }
        }

        if self.env.is_goal(&current_state) {
            Ok(path)
        } else {
            Err("Failed to reach target goal within maximum interactions budget".to_string())
        }
    })();

        let _ = self.clause_db.save_to_file("agent_clauses.json");

        // Episode isolation: preserve structural QMEM knowledge while clearing
        // transient physical resonance state and marking the agent ready for a
        // fresh next run.
        net_g.clear_state();
        self.current_state = None;

        result
    }
}

fn parse_state_properties(obs: &str) -> HashMap<String, String> {
    let mut props = HashMap::new();
    for line in obs.lines() {
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() == 2 {
            props.insert(parts[0].trim().to_string(), parts[1].trim().to_string());
        }
    }
    props
}

fn append_experience_rules<A: std::fmt::Debug>(
    exp_path: &str,
    action: &A,
    curr_obs: &str,
    next_obs: &str,
    reward: f64,
    is_failure: bool,
) -> std::io::Result<()> {
    use std::fs::OpenOptions;
    use std::io::Write;
    
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(exp_path)?;
        
    let action_str = format!("{:?}", action);
    let curr_props = parse_state_properties(curr_obs);
    let next_props = parse_state_properties(next_obs);
    
    // 1. Synthesize state transitions
    // Keep a list of changed properties
    let mut changed_props = Vec::new();
    for (k, curr_val) in &curr_props {
        if k == "steps" { continue; }
        if let Some(next_val) = next_props.get(k) {
            if curr_val != next_val {
                changed_props.push((k.clone(), curr_val.clone(), next_val.clone()));
            }
        }
    }
    
    // Unchanged properties excluding "steps"
    let mut unchanged_props = Vec::new();
    for (k, curr_val) in &curr_props {
        if k == "steps" { continue; }
        if !changed_props.iter().any(|(ck, _, _)| ck == k) {
            unchanged_props.push((k.clone(), curr_val.clone()));
        }
    }
    
    for (k, curr_val, next_val) in &changed_props {
        // A. Single-variable rule (baseline)
        writeln!(
            file,
            "\"obs_{}_{}\" CAUSES \"obs_action_{}\" [1.0, 0.0];",
            k, curr_val, action_str
        )?;
        writeln!(
            file,
            "\"obs_action_{}\" RULE_CONFIRMED \"obs_{}_{}\" [1.0, 0.0];",
            action_str, k, next_val
        )?;
        
        // B. Pairwise rules combining the changed key with each unchanged key (excluding steps)
        for (uk, uval) in &unchanged_props {
            let label_change_curr = format!("obs_{}_{}", k, curr_val);
            let label_change_next = format!("obs_{}_{}", k, next_val);
            let label_unchanged = format!("obs_{}_{}", uk, uval);
            
            let joint_curr = if label_change_curr < label_unchanged {
                format!("{}|{}", label_change_curr, label_unchanged)
            } else {
                format!("{}|{}", label_unchanged, label_change_curr)
            };
            
            let joint_next = if label_change_next < label_unchanged {
                format!("{}|{}", label_change_next, label_unchanged)
            } else {
                format!("{}|{}", label_unchanged, label_change_next)
            };
            
            writeln!(
                file,
                "\"{}\" CAUSES \"obs_action_{}\" [1.0, 0.0];",
                joint_curr, action_str
            )?;
            writeln!(
                file,
                "\"obs_action_{}\" RULE_CONFIRMED \"{}\" [1.0, 0.0];",
                action_str, joint_next
            )?;
        }
    }
    
    // 2. Failure experience
    if is_failure {
        writeln!(
            file,
            "\"obs_action_{}\" TRANSITION_FAILED \"obs_steps\" [1.0, 0.0];",
            action_str
        )?;
    }
    
    // 3. Reward experience
    if reward > 0.0 {
        writeln!(
            file,
            "\"obs_action_{}\" REWARD_OBSERVED \"reward\" [{:.1}, 0.0];",
            action_str, reward
        )?;
    }
    
    Ok(())
}

fn count_rule_confirmed_in_exp(exp_path: &str) -> usize {
    if let Ok(content) = std::fs::read_to_string(exp_path) {
        content.lines().filter(|line| line.contains("RULE_CONFIRMED")).count()
    } else {
        0
    }
}

fn calculate_predictive_confidence(
    state_obs: &str,
    actions: &[String],
    exp_content: &str,
) -> f64 {
    let props = parse_state_properties(state_obs);
    let mut active_concepts = std::collections::HashSet::new();
    for (k, v) in &props {
        if k != "steps" {
            active_concepts.insert(format!("obs_{}_{}", k, v));
        }
    }

    // Pairwise
    let keys: Vec<&String> = props.keys().filter(|&&ref k| k != "steps").collect();
    for i in 0..keys.len() {
        for j in (i+1)..keys.len() {
            let ka = keys[i];
            let kb = keys[j];
            let val_a = &props[ka];
            let val_b = &props[kb];
            let label_a = format!("obs_{}_{}", ka, val_a);
            let label_b = format!("obs_{}_{}", kb, val_b);
            let joint_label = if label_a < label_b {
                format!("{}|{}", label_a, label_b)
            } else {
                format!("{}|{}", label_b, label_a)
            };
            active_concepts.insert(joint_label);
        }
    }

    let mut causes_map = HashMap::new();
    let mut confirmed_set = std::collections::HashSet::new();

    for line in exp_content.lines() {
        if line.contains("CAUSES") {
            let parts: Vec<&str> = line.split("CAUSES").collect();
            if parts.len() == 2 {
                let pre = parts[0].trim().trim_matches('"');
                let act_part = parts[1].split('[').next().unwrap_or("").trim().trim_matches('"');
                if active_concepts.contains(pre) {
                    causes_map.entry(act_part.to_string()).or_insert_with(Vec::new).push(pre.to_string());
                }
            }
        } else if line.contains("RULE_CONFIRMED") {
            let parts: Vec<&str> = line.split("RULE_CONFIRMED").collect();
            if parts.len() == 2 {
                let act_part = parts[0].trim().trim_matches('"');
                confirmed_set.insert(act_part.to_string());
            }
        }
    }

    if actions.is_empty() {
        return 1.0;
    }

    let mut predicted_actions_count = 0;
    for act in actions {
        let act_label = format!("obs_action_{}", act);
        if causes_map.contains_key(&act_label) && confirmed_set.contains(&act_label) {
            predicted_actions_count += 1;
        }
    }

    predicted_actions_count as f64 / actions.len() as f64
}

// ============================================================================
// UNIT TESTS - CDCL Clause Database
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdcl_learns_and_blocks() {
        let mut db = CdclClauseDb::new();

        // Learn conflict
        db.learn_conflict(
            vec!["obs_x_3".to_string(), "obs_color_blue".to_string()],
            "action_right".to_string(),
            1,
        );
        db.learn_conflict( // Second observation → frequency 2 → reliable
            vec!["obs_x_3".to_string(), "obs_color_blue".to_string()],
            "action_right".to_string(),
            2,
        );

        // Same state → action should be blocked
        let state = vec!["obs_x_3".to_string(), "obs_color_blue".to_string()];
        assert!(db.is_action_blocked("action_right", &state, 2),
            "Learned conflict should block the action");

        // Different state → action should be free
        let other_state = vec!["obs_x_5".to_string(), "obs_color_red".to_string()];
        assert!(!db.is_action_blocked("action_right", &other_state, 2),
            "Different state should not be blocked");
    }

    #[test]
    fn test_cdcl_export_to_qmem_dsl() {
        let mut db = CdclClauseDb::new();
        db.learn_conflict(vec!["obs_wall".to_string()], "action_forward".to_string(), 1);
        db.learn_conflict(vec!["obs_wall".to_string()], "action_forward".to_string(), 2);

        let exp = db.export_to_exp(2);
        assert!(exp.contains("TRANSITION_FAILED"), "Should export as QMEM DSL");
        assert!(exp.contains("action_forward"), "Should contain the conflicting action");
    }

    #[test]
    fn test_cdcl_blocked_actions() {
        let mut db = CdclClauseDb::new();
        db.learn_conflict(vec!["obs_x_3".to_string()], "action_right".to_string(), 1);
        db.learn_conflict(vec!["obs_x_3".to_string()], "action_right".to_string(), 2);
        db.learn_conflict(vec!["obs_x_3".to_string()], "action_left".to_string(), 1);
        db.learn_conflict(vec!["obs_x_3".to_string()], "action_left".to_string(), 2);

        let state = vec!["obs_x_3".to_string()];
        let blocked = db.blocked_actions(&state, 2);
        assert_eq!(blocked.len(), 2, "Should block both actions");
        assert!(blocked.contains(&"action_right".to_string()));
        assert!(blocked.contains(&"action_left".to_string()));
    }
}
