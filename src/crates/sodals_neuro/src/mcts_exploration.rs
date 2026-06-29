// src/crates/sodals_neuro/src/mcts_exploration.rs
// ============================================================================
// SODALS ARC-AGI-3 MCTS EXPLORATION POLICY
// ============================================================================

use crate::arc_environment::ArcEnvironment;
use crate::arkhe_pipeline::ArkheCognitionPipeline;
use crate::semantic_network::ResonanceNetwork;
use crate::semantic_parser::SemanticRegisterFile;
use std::collections::HashMap;

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct TrieNode {
    pub children: HashMap<String, TrieNode>,
    pub q_values: HashMap<String, f64>,
}

impl TrieNode {
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            q_values: HashMap::new(),
        }
    }

    /// Recursively compresses Trie single-child nodes into compressed/Radix paths
    pub fn compress(&mut self) {
        let child_keys: Vec<String> = self.children.keys().cloned().collect();
        for key in child_keys {
            if let Some(mut child) = self.children.remove(&key) {
                child.compress();
                if child.children.len() == 1 && child.q_values.is_empty() {
                    let (grandchild_key, grandchild) = child.children.into_iter().next().unwrap();
                    let new_key = format!("{}|{}", key, grandchild_key);
                    self.children.insert(new_key, grandchild);
                } else {
                    self.children.insert(key, child);
                }
            }
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct TabularQTable {
    pub root: TrieNode,
    pub alpha: f64,
    pub gamma: f64,
    pub count: usize,
}

impl TabularQTable {
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
            alpha: 0.1,
            gamma: 0.9,
            count: 0,
        }
    }

    fn split_hash(s_hash: &str) -> Vec<String> {
        s_hash.split('|').map(|s| s.to_string()).collect()
    }

    /// Radix Tree traversal with path compression matching
    pub fn get(&self, s_hash: &str, a_str: &str) -> f64 {
        let segments = Self::split_hash(s_hash);
        let mut curr = &self.root;
        let mut i = 0;
        while i < segments.len() {
            let mut matched = false;
            for (child_key, next_node) in &curr.children {
                let child_segs = Self::split_hash(child_key);
                if i + child_segs.len() <= segments.len() && segments[i..i+child_segs.len()] == child_segs[..] {
                    curr = next_node;
                    i += child_segs.len();
                    matched = true;
                    break;
                }
            }
            if !matched {
                return 0.0;
            }
        }
        curr.q_values.get(a_str).copied().unwrap_or(0.0)
    }

    /// Update with path compression and feedback-driven adaptive learning rate (alpha)
    pub fn update(&mut self, s_hash: String, a_str: String, reward: f64, next_s_hash: String, next_actions: &[String]) {
        let current_q = self.get(&s_hash, &a_str);
        let mut max_next_q = 0.0;
        if !next_actions.is_empty() {
            let mut best = f64::NEG_INFINITY;
            for na in next_actions {
                let q = self.get(&next_s_hash, na);
                if q > best { best = q; }
            }
            if best.is_finite() {
                max_next_q = best;
            }
        }
        
        let target = reward + self.gamma * max_next_q;
        // Geri Beslemeli alpha Optimizasyonu: Adjust learning rate based on temporal difference error
        let td_error = (target - current_q).abs();
        let adaptive_alpha = (self.alpha * (1.0 + 0.1 * td_error)).clamp(0.01, 0.5);
        let new_q = current_q + adaptive_alpha * (target - current_q);
        
        let segments = Self::split_hash(&s_hash);
        let mut curr = &mut self.root;
        let mut i = 0;
        while i < segments.len() {
            let mut matched_key = None;
            for child_key in curr.children.keys() {
                let child_segs = Self::split_hash(child_key);
                if i + child_segs.len() <= segments.len() && segments[i..i+child_segs.len()] == child_segs[..] {
                    matched_key = Some((child_key.clone(), child_segs.len()));
                    break;
                }
            }
            if let Some((m_key, len)) = matched_key {
                curr = curr.children.get_mut(&m_key).unwrap();
                i += len;
            } else {
                let remaining = segments[i..].join("|");
                curr = curr.children.entry(remaining).or_insert_with(TrieNode::new);
                break;
            }
        }
        if curr.q_values.is_empty() {
            self.count += 1;
        }
        curr.q_values.insert(a_str, new_q);
        
        // Trigger Radix Tree compression
        self.root.compress();
    }

    /// Update with path compression and feedback-driven adaptive learning rate (alpha) from MCTS
    pub fn update_from_mcts(&mut self, s_hash: String, a_str: String, mcts_q: f64) {
        let current_q = self.get(&s_hash, &a_str);
        // Geri Beslemeli alpha Optimizasyonu: Adjust learning rate based on discrepancy
        let diff = (mcts_q - current_q).abs();
        let adaptive_alpha = (self.alpha * (1.0 + 0.1 * diff)).clamp(0.01, 0.5);
        let new_q = current_q + adaptive_alpha * (mcts_q - current_q);
        
        let segments = Self::split_hash(&s_hash);
        let mut curr = &mut self.root;
        let mut i = 0;
        while i < segments.len() {
            let mut matched_key = None;
            for child_key in curr.children.keys() {
                let child_segs = Self::split_hash(child_key);
                if i + child_segs.len() <= segments.len() && segments[i..i+child_segs.len()] == child_segs[..] {
                    matched_key = Some((child_key.clone(), child_segs.len()));
                    break;
                }
            }
            if let Some((m_key, len)) = matched_key {
                curr = curr.children.get_mut(&m_key).unwrap();
                i += len;
            } else {
                let remaining = segments[i..].join("|");
                curr = curr.children.entry(remaining).or_insert_with(TrieNode::new);
                break;
            }
        }
        if curr.q_values.is_empty() {
            self.count += 1;
        }
        curr.q_values.insert(a_str, new_q);
        
        // Trigger Radix Tree compression
        self.root.compress();
    }

    pub fn get_local_alpha(&self, s_hash: &str) -> f64 {
        let segments = Self::split_hash(s_hash);
        let mut curr = &self.root;
        let mut i = 0;
        while i < segments.len() {
            let mut matched = false;
            for (child_key, next_node) in &curr.children {
                let child_segs = Self::split_hash(child_key);
                if i + child_segs.len() <= segments.len() && segments[i..i+child_segs.len()] == child_segs[..] {
                    curr = next_node;
                    i += child_segs.len();
                    matched = true;
                    break;
                }
            }
            if !matched {
                return 0.0;
            }
        }
        let explored_actions = curr.q_values.len() as f64;
        (explored_actions / 5.0).clamp(0.0, 0.8)
    }
}

pub fn get_qmem_state_hash<E: ArcEnvironment>(env: &E, state: &E::State) -> String {
    let obs = env.parse_observation(state);
    let mut keys: Vec<String> = Vec::new();
    for line in obs.lines() {
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() == 2 {
            let key = parts[0].trim();
            let val_str = parts[1].trim();
            keys.push(format!("obs_{}_{}", key, val_str));
        }
    }
    keys.sort();
    keys.join("|")
}

pub struct MctsNode<S, A> {
    pub state: S,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub action: Option<A>,
    pub visits: u32,
    pub value: f64,
    pub reward: f64,
}

pub struct MctsSearch<'a, E: ArcEnvironment> {
    pub env: &'a E,
    pub nodes: Vec<MctsNode<E::State, E::Action>>,
    pub c_puct: f64,
    pub q_table: &'a TabularQTable,
}

impl<'a, E: ArcEnvironment> MctsSearch<'a, E> {
    pub fn new(env: &'a E, initial_state: E::State, q_table: &'a TabularQTable) -> Self {
        let root = MctsNode {
            state: initial_state,
            parent: None,
            children: Vec::new(),
            action: None,
            visits: 0,
            value: 0.0,
            reward: 0.0,
        };
        Self {
            env,
            nodes: vec![root],
            c_puct: 1.414,
            q_table,
        }
    }
    
    /// UCT Selection phase incorporating CliffordBridge phase priority, Tabular Q-Table bias, Progressive Widening, and Adaptive Blending
    fn select(&self, node_idx: usize, pipeline: &mut ArkheCognitionPipeline, net_f: &ResonanceNetwork) -> usize {
        let mut current = node_idx;
        while !self.nodes[current].children.is_empty() {
            let parent_visits = self.nodes[current].visits as f64;
            
            // Semantic Density Tabanlı Widening: Scale widening_alpha exponent based on parent concept density
            let parent_obs = self.env.parse_observation(&self.nodes[current].state);
            let mut net_parent = ResonanceNetwork::new(256);
            let adapter = crate::arc_environment::GenericEnvironmentAdapter::new();
            adapter.embed_observation(&parent_obs, &mut net_parent);
            
            let active_nodes_count = net_parent.labels.len() as f64;
            let density_factor = (active_nodes_count / 10.0).clamp(0.2, 0.8);
            let widening_alpha = 0.3 + 0.4 * density_factor;
            
            // Progressive Widening: limit candidate children by k * N^alpha
            let max_allowed = (1.5 * parent_visits.powf(widening_alpha)).ceil() as usize;
            let children_len = self.nodes[current].children.len();
            let allowed_children = &self.nodes[current].children[..std::cmp::min(children_len, max_allowed.max(1))];
            
            let mut best_child = allowed_children[0];
            let mut best_score = -f64::INFINITY;
            
            let parent_hash = get_qmem_state_hash(self.env, &self.nodes[current].state);
            
            for &child_idx in allowed_children {
                let child = &self.nodes[child_idx];
                let mcts_q = if child.visits > 0 { child.value / child.visits as f64 } else { 0.0 };
                
                let action_str = format!("{:?}", child.action.as_ref().unwrap());
                let tabular_q = self.q_table.get(&parent_hash, &action_str);
                
                // Adaptive Blending: Q_combined = (1-alpha)*Q_mcts + alpha*Q_tabular
                let alpha = self.q_table.get_local_alpha(&parent_hash);
                let q = if child.visits > 0 {
                    (1.0 - alpha) * mcts_q + alpha * tabular_q
                } else {
                    tabular_q
                };
                
                // Embed current child state and compute CliffordBridge h_phase as prior probability bias
                let obs_text = self.env.parse_observation(&child.state);
                let mut net_child = ResonanceNetwork::new(256);
                let adapter = crate::arc_environment::GenericEnvironmentAdapter::new();
                adapter.embed_observation(&obs_text, &mut net_child);
                
                let (h_phase, _) = pipeline.bridge.project(net_f, &net_child, 1.0);
                let bridge_p = (std::f64::consts::PI - h_phase.abs()).max(0.0) / std::f64::consts::PI;
                
                let u = self.c_puct * bridge_p * (parent_visits.sqrt() / (1.0 + child.visits as f64));
                let score = q + u;
                if score > best_score {
                    best_score = score;
                    best_child = child_idx;
                }
            }
            current = best_child;
        }
        current
    }
    
    /// Expand node and prune states that violate Wronskian constraints or closed God's Gate
    pub fn expand(&mut self, node_idx: usize, pipeline: &mut ArkheCognitionPipeline, net_f: &ResonanceNetwork, srf: &mut SemanticRegisterFile) {
        let state = self.nodes[node_idx].state.clone();
        let actions = self.env.get_actions(&state);
        
        let adapter = crate::arc_environment::GenericEnvironmentAdapter::new();
        let parent_obs = self.env.parse_observation(&state);
        let mut net_parent = ResonanceNetwork::new(256);
        adapter.embed_observation(&parent_obs, &mut net_parent);
        
        for action in actions {
            let (next_state, reward, _) = self.env.step(&state, &action);
            
            // Build temporary network state for the next transition
            let child_obs = self.env.parse_observation(&next_state);
            let mut net_child = ResonanceNetwork::new(256);
            adapter.embed_observation(&child_obs, &mut net_child);
            net_child.step(&[]);
            
            // Verify via Arkhe Pipeline
            pipeline.run(net_f, &net_child, &format!("{:?}", action), srf);
            let wronskian = pipeline.state.wronskian;
            
            // Budama (Pruning): Wronskian is too small (linear dependency) or God's Gate constraints violated
            if wronskian < 0.001 || !pipeline.state.gate_open {
                continue;
            }
            
            let child_idx = self.nodes.len();
            self.nodes.push(MctsNode {
                state: next_state,
                parent: Some(node_idx),
                children: Vec::new(),
                action: Some(action),
                visits: 0,
                value: 0.0,
                reward,
            });
            self.nodes[node_idx].children.push(child_idx);
        }
    }
    
    /// Simulation / Rollout phase biased by Tabular Q-Table and QMEM Yöntem Arkhesi (action sequence)
    /// Uses a fast in-memory snapshot of the semantic network during rollout to preserve
    /// the global network state while exploring hypothetical action sequences.
    fn rollout(
        &self,
        state: &E::State,
        pipeline: &ArkheCognitionPipeline,
        net_f: &mut ResonanceNetwork,
    ) -> f64 {
        let snapshot = net_f.snapshot();
        let mut current_state = state.clone();
        let mut total_reward = 0.0;
        let mut steps = 0;

        let action_seq = &pipeline.state.action_sequence;

        while steps < 5 {
            if self.env.is_goal(&current_state) {
                total_reward += 100.0;
                break;
            }

            let actions = self.env.get_actions(&current_state);
            if actions.is_empty() {
                break;
            }

            // Rollout guided by Tabular Q-values first, then fall back to action sequence
            let current_hash = get_qmem_state_hash(self.env, &current_state);
            let chosen_action = actions.iter()
                .max_by(|a, b| {
                    let qa = self.q_table.get(&current_hash, &format!("{:?}", a));
                    let qb = self.q_table.get(&current_hash, &format!("{:?}", b));
                    qa.partial_cmp(&qb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .filter(|a| {
                    let q = self.q_table.get(&current_hash, &format!("{:?}", a));
                    q != 0.0
                })
                .unwrap_or_else(|| {
                    actions.iter()
                        .find(|a| action_seq.contains(&format!("{:?}", a)))
                        .unwrap_or(&actions[0])
                });

            let (next, reward, is_terminal) = self.env.step(&current_state, chosen_action);
            total_reward += reward;
            current_state = next;
            if is_terminal {
                break;
            }
            steps += 1;
        }

        net_f.restore_snapshot(&snapshot);
        total_reward
    }

    /// Backpropagate rollout scores back up the tree
    fn backpropagate(&mut self, node_idx: usize, value: f64) {
        let mut current = Some(node_idx);
        while let Some(idx) = current {
            self.nodes[idx].visits += 1;
            self.nodes[idx].value += value;
            current = self.nodes[idx].parent;
        }
    }
    
    fn compute_depth(&self, node_idx: usize) -> usize {
        let mut depth = 0;
        let mut curr = node_idx;
        while let Some(parent_idx) = self.nodes[curr].parent {
            depth += 1;
            curr = parent_idx;
        }
        depth
    }

    /// Main MCTS search execution
    pub fn search(
        &mut self,
        iterations: usize,
        pipeline: &mut ArkheCognitionPipeline,
        net_f: &mut ResonanceNetwork,
        srf: &mut SemanticRegisterFile,
    ) -> Option<E::Action> {
        let effective_iterations = iterations.min(50);
        let max_depth = 3;
        
        for _ in 0..effective_iterations {
            let leaf_idx = self.select(0, pipeline, net_f);
            
            // Depth pruning: if depth >= max_depth, skip expand and run rollout directly on leaf
            let depth = self.compute_depth(leaf_idx);
            if depth >= max_depth {
                let reward = self.rollout(&self.nodes[leaf_idx].state, pipeline, net_f);
                let total_val = self.nodes[leaf_idx].reward + reward;
                self.backpropagate(leaf_idx, total_val);
                continue;
            }
            
            if self.nodes[leaf_idx].children.is_empty() {
                self.expand(leaf_idx, pipeline, net_f, srf);
            }
            
            let sim_idx = if !self.nodes[leaf_idx].children.is_empty() {
                self.nodes[leaf_idx].children[0]
            } else {
                leaf_idx
            };
            
            let reward = self.rollout(&self.nodes[sim_idx].state, pipeline, net_f);
            let total_val = self.nodes[sim_idx].reward + reward;
            self.backpropagate(sim_idx, total_val);
        }
        
        self.nodes[0].children.iter()
            .max_by(|&&a, &&b| {
                let va = self.nodes[a].visits;
                let vb = self.nodes[b].visits;
                va.cmp(&vb)
            })
            .and_then(|&idx| self.nodes[idx].action.clone())
    }
}

pub struct BeamSearch<'a, E: ArcEnvironment> {
    pub env: &'a E,
    pub beam_width: usize,
    pub q_table: &'a TabularQTable,
}

impl<'a, E: ArcEnvironment> BeamSearch<'a, E> {
    pub fn new(env: &'a E, beam_width: usize, q_table: &'a TabularQTable) -> Self {
        Self { env, beam_width, q_table }
    }
    
    /// Guided Beam Search incorporating h_phase, Tabular Q, and Wronskian / God's Gate constraints.
    pub fn search(
        &self,
        initial_state: E::State,
        max_depth: usize,
        pipeline: &mut ArkheCognitionPipeline,
        net_f: &ResonanceNetwork,
        net_g: &ResonanceNetwork,
        srf: &mut SemanticRegisterFile,
    ) -> Option<E::Action> {
        // Elements in beam: (state, first_action, cumulative_score)
        let mut beam: Vec<(E::State, Option<E::Action>, f64)> = vec![(initial_state.clone(), None, 0.0)];
        
        let adapter = crate::arc_environment::GenericEnvironmentAdapter::new();
        
        for depth in 0..max_depth {
            let mut candidates = Vec::new();
            
            for (state, first_act, score) in &beam {
                if self.env.is_goal(state) {
                    // Goal reached, keep it in the beam with a huge reward boost
                    candidates.push((state.clone(), first_act.clone(), *score + 1000.0));
                    continue;
                }
                
                let actions = self.env.get_actions(state);
                for action in actions {
                    let action_str = format!("{:?}", action);
                    
                    // Pruning check 1: Check if TransitionFailed exists in net_g for this action
                    let action_label = format!("obs_action_{:?}", action);
                    if let Some(&act_id) = net_g.labels.get(&action_label) {
                        let failure_label = "obs_steps";
                        if let Some(&fail_id) = net_g.labels.get(failure_label) {
                            if net_g.has_link(act_id, fail_id) {
                                // Failed transition, skip
                                continue;
                            }
                        }
                    }
                    
                    let (next_state, reward, is_terminal) = self.env.step(state, &action);
                    
                    // Build temporary network state for the next transition
                    let child_obs = self.env.parse_observation(&next_state);
                    let mut net_child = ResonanceNetwork::new(256);
                    adapter.embed_observation(&child_obs, &mut net_child);
                    net_child.step(&[]);
                    
                    // Pruning check 2: Verify via Arkhe Pipeline (Wronskian & God's Gate)
                    pipeline.run(net_f, &net_child, &action_str, srf);
                    let wronskian = pipeline.state.wronskian;
                    
                    if wronskian < 0.001 || !pipeline.state.gate_open {
                        // Violates constraints, prune
                        continue;
                    }
                    
                    // Heuristic score calculation: CliffordBridge h_phase
                    let (h_phase, _) = pipeline.bridge.project(net_f, &net_child, 1.0);
                    let heuristic = (std::f64::consts::PI - h_phase.abs()).max(0.0) / std::f64::consts::PI;
                    
                    // Tabular Q component
                    let parent_hash = get_qmem_state_hash(self.env, state);
                    let tabular_q = self.q_table.get(&parent_hash, &action_str);
                    
                    // Blend score dynamically
                    let alpha = self.q_table.get_local_alpha(&parent_hash);
                    let transition_score = (1.0 - alpha) * heuristic + alpha * tabular_q + reward;
                    
                    let new_first_act = if depth == 0 {
                        Some(action.clone())
                    } else {
                        first_act.clone()
                    };
                    
                    candidates.push((next_state, new_first_act, score + transition_score));
                }
            }
            
            if candidates.is_empty() {
                break;
            }
            
            // Sort candidates by cumulative score descending
            candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            
            // Keep top K (beam_width)
            candidates.truncate(self.beam_width);
            beam = candidates;
        }
        
        // Return first action of the best path in the final beam
        beam.first().and_then(|(_, first_act, _)| first_act.clone())
    }
}

