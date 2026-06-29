// src/crates/sodals_neuro/src/arc_environment.rs
// ============================================================================
// SODALS ARC-AGI-3 INTERACTIVE ENVIRONMENT ABSTRACTIONS
// ============================================================================

use std::collections::HashMap;
use num_complex::Complex64;
use crate::semantic_network::ResonanceNetwork;

/// Abstract interface representing interactive environments (grid, simulation, game)
pub trait ArcEnvironment {
    type State: Clone + std::fmt::Debug + Eq + std::hash::Hash;
    type Action: Clone + std::fmt::Debug + Eq;
    
    fn initial_state(&self) -> Self::State;
    fn step(&self, state: &Self::State, action: &Self::Action) -> (Self::State, f64, bool); // (next_state, reward, is_terminal)
    fn get_actions(&self, state: &Self::State) -> Vec<Self::Action>;
    fn parse_observation(&self, state: &Self::State) -> String; // text description
    fn is_goal(&self, state: &Self::State) -> bool;
}

/// Generic adapter converting raw structured text/log observations into VSA network nodes
pub struct GenericEnvironmentAdapter;

impl GenericEnvironmentAdapter {
    pub fn new() -> Self {
        Self
    }
    
    /// Parse key-value properties from observation and ingest into semantic network
    pub fn embed_observation(&self, observation: &str, net: &mut ResonanceNetwork) {
        let mut variables = Vec::new();
        for line in observation.lines() {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() == 2 {
                let key = parts[0].trim();
                let val_str = parts[1].trim();
                let val_f64 = val_str.parse::<f64>().unwrap_or(1.0);
                variables.push((key.to_string(), val_str.to_string(), val_f64));
            }
        }
        
        let mut individual_labels = Vec::new();
        for (key, val_str, val_f64) in &variables {
            if key == "steps" {
                continue;
            }
            let label = format!("obs_{}_{}", key, val_str);
            individual_labels.push((label, *val_f64));
        }
        
        // Also embed steps if present, but don't pair it
        for (key, val_str, val_f64) in &variables {
            if key == "steps" {
                let label = format!("obs_steps_{}", val_str);
                if !net.labels.contains_key(&label) {
                    let mut sig = vec![Complex64::new(0.0, 0.0); 64];
                    for j in 0..64 {
                        let phase = (val_f64 * j as f64 * 0.1) + (j as f64 * 0.05);
                        sig[j] = Complex64::from_polar(1.0, phase);
                    }
                    net.add_labeled_concept(&label, 0.1 * (val_f64.abs() + 1.0), sig);
                }
            }
        }

        // Embed individual labels
        for (label, val) in &individual_labels {
            if net.labels.contains_key(label) {
                continue;
            }
            
            let mut sig = vec![Complex64::new(0.0, 0.0); 64];
            for j in 0..64 {
                let phase = (val * j as f64 * 0.1) + (j as f64 * 0.05);
                sig[j] = Complex64::from_polar(1.0, phase);
            }
            net.add_labeled_concept(label, 0.1 * (val.abs() + 1.0), sig);
        }
        
        // Embed pairwise joint concepts using element-wise complex multiplication (VSA binding)
        for i in 0..individual_labels.len() {
            for j in (i + 1)..individual_labels.len() {
                let (label_a, val_a) = &individual_labels[i];
                let (label_b, val_b) = &individual_labels[j];
                
                let joint_label = if label_a < label_b {
                    format!("{}|{}", label_a, label_b)
                } else {
                    format!("{}|{}", label_b, label_a)
                };
                
                if net.labels.contains_key(&joint_label) {
                    continue;
                }
                
                let mut sig = vec![Complex64::new(0.0, 0.0); 64];
                for k in 0..64 {
                    let phase_a = (val_a * k as f64 * 0.1) + (k as f64 * 0.05);
                    let phase_b = (val_b * k as f64 * 0.1) + (k as f64 * 0.05);
                    let sig_a = Complex64::from_polar(1.0, phase_a);
                    let sig_b = Complex64::from_polar(1.0, phase_b);
                    sig[k] = sig_a * sig_b; // Holographic binding
                }
                let val_joint = val_a + val_b;
                net.add_labeled_concept(&joint_label, 0.1 * (val_joint.abs() + 1.0), sig);
            }
        }
    }
}

// ============================================================================
// SIMULATION GRID GAME ENVIRONMENT (TEST IMPLEMENTATION)
// ============================================================================

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct GridGameState {
    pub player_x: i32,
    pub player_y: i32,
    pub target_x: i32,
    pub target_y: i32,
    pub steps: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum GridGameAction {
    Up,
    Down,
    Left,
    Right,
}

pub struct GridGameEnv {
    pub width: i32,
    pub height: i32,
}

impl ArcEnvironment for GridGameEnv {
    type State = GridGameState;
    type Action = GridGameAction;
    
    fn initial_state(&self) -> Self::State {
        GridGameState {
            player_x: 0,
            player_y: 0,
            target_x: 3,
            target_y: 3,
            steps: 0,
        }
    }
    
    fn step(&self, state: &Self::State, action: &Self::Action) -> (Self::State, f64, bool) {
        let mut next = state.clone();
        next.steps += 1;
        
        match action {
            GridGameAction::Up => next.player_y = (next.player_y + 1).min(self.height - 1),
            GridGameAction::Down => next.player_y = (next.player_y - 1).max(0),
            GridGameAction::Left => next.player_x = (next.player_x - 1).max(0),
            GridGameAction::Right => next.player_x = (next.player_x + 1).min(self.width - 1),
        }
        
        let reached = next.player_x == next.target_x && next.player_y == next.target_y;
        let is_terminal = reached || next.steps >= 20;
        
        let dist_before = (state.target_x - state.player_x).abs() + (state.target_y - state.player_y).abs();
        let dist_after = (next.target_x - next.player_x).abs() + (next.target_y - next.player_y).abs();
        
        let reward = if reached {
            100.0
        } else if next.player_x == state.player_x && next.player_y == state.player_y {
            -2.0 // Hit boundary (no-op collision penalty)
        } else if dist_after < dist_before {
            1.0  // Moved closer
        } else {
            -1.0 // Moved away
        };
        
        (next, reward, is_terminal)
    }
    
    fn get_actions(&self, _state: &Self::State) -> Vec<Self::Action> {
        vec![GridGameAction::Up, GridGameAction::Down, GridGameAction::Left, GridGameAction::Right]
    }
    
    fn parse_observation(&self, state: &Self::State) -> String {
        format!(
            "player_x: {}\nplayer_y: {}\ntarget_x: {}\ntarget_y: {}\nsteps: {}",
            state.player_x, state.player_y, state.target_x, state.target_y, state.steps
        )
    }
    
    fn is_goal(&self, state: &Self::State) -> bool {
        state.player_x == state.target_x && state.player_y == state.target_y
    }
}
