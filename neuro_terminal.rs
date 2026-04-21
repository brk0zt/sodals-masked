// src/tui/neuro_terminal.rs
// COGNITIVE TELEMETRY TUI — Nuclear Reactor Control Panel
//
// This is NOT a chat interface. There are no message bubbles, no user/assistant turns,
// no conversation histories. This is a real-time physics instrument panel that renders
// the continuous-time state of the Töz Core: Arkhe activation, RK4 phase alignment,
// Semantic Divergence Z-Score, and Ouroboros refinement pass count.
//
// Layout:
//   ┌──────────────────────┬──────────────────────┐
//   │   Physical Realm     │  Cognitive Telemetry │
//   │   (Files/AST/MCP)    │  (Arkhe/RK4/Z/Pass) │
//   ├──────────────────────┴──────────────────────┤
//   │              Autonomy Log                    │
//   │         (Actuator streaming log)             │
//   └─────────────────────────────────────────────┘

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph},
};

use crate::qmem_compiler::Arkhe;
use crate::engine::autonomy_gate::AutonomyLevel;

// ============================================================================
// COGNITIVE TELEMETRY STATE — The physics snapshot rendered by the TUI
// ============================================================================

/// Snapshot of the Core's continuous-time physics state.
/// Updated every frame by the engine loop; the TUI only reads this.
#[derive(Debug, Clone)]
pub struct CognitiveTelemetry {
    /// Which Arkhe is currently dominant in the ODE flow
    pub active_arkhe: Arkhe,
    /// RK4 integrator phase alignment [0.0, 1.0]
    /// 1.0 = perfect alignment (all four RK stages converge)
    pub rk4_phase_alignment: f64,
    /// Semantic Divergence Z-Score from the Oracle
    /// Low = coherent, High = constructive interference failure
    pub divergence_z_score: f64,
    /// Current Ouroboros refinement pass number
    pub ouroboros_pass_count: u32,
    /// Current autonomy level (determined by AutonomyGate)
    pub autonomy_level: AutonomyLevel,
}

impl Default for CognitiveTelemetry {
    fn default() -> Self {
        Self {
            active_arkhe: Arkhe::Conditions,
            rk4_phase_alignment: 0.0,
            divergence_z_score: 0.0,
            ouroboros_pass_count: 0,
            autonomy_level: AutonomyLevel::Full,
        }
    }
}

// ============================================================================
// PHYSICAL REALM STATE — File system / AST / MCP tool inventory
// ============================================================================

/// Items displayed in the left "Physical Realm" panel.
#[derive(Debug, Clone)]
pub enum PhysicalRealmItem {
    File { path: String, loaded: bool },
    AstNode { kind: String, domain: String },
    McpTool { name: String, status: McpToolStatus },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpToolStatus {
    Idle,
    Active,
    Error,
}

// ============================================================================
// AUTONOMY LOG ENTRY — Streaming actuator action log
// ============================================================================

#[derive(Debug, Clone)]
pub struct AutonomyLogEntry {
    pub timestamp_ms: u64,
    pub actuator: String,
    pub action: String,
    pub autonomy_level: AutonomyLevel,
    pub divergence_at_execution: f64,
}

// ============================================================================
// NEURO TERMINAL — The main TUI struct
// ============================================================================

pub struct NeuroTerminal {
    /// Current physics telemetry snapshot
    pub telemetry: CognitiveTelemetry,
    /// Physical realm items (file system, AST, MCP tools)
    pub physical_realm: Vec<PhysicalRealmItem>,
    /// Streaming autonomy log (ring buffer, newest last)
    pub autonomy_log: Vec<AutonomyLogEntry>,
    /// Maximum log entries before eviction
    pub max_log_entries: usize,
}

impl NeuroTerminal {
    pub fn new() -> Self {
        Self {
            telemetry: CognitiveTelemetry::default(),
            physical_realm: Vec::new(),
            autonomy_log: Vec::new(),
            max_log_entries: 500,
        }
    }

    /// Push an actuator log entry, evicting oldest if over capacity.
    pub fn log_action(
        &mut self,
        timestamp_ms: u64,
        actuator: impl Into<String>,
        action: impl Into<String>,
        autonomy_level: AutonomyLevel,
        divergence: f64,
    ) {
        if self.autonomy_log.len() >= self.max_log_entries {
            self.autonomy_log.remove(0);
        }
        self.autonomy_log.push(AutonomyLogEntry {
            timestamp_ms,
            actuator: actuator.into(),
            action: action.into(),
            autonomy_level,
            divergence_at_execution: divergence,
        });
    }

    // ========================================================================
    // RENDER — The draw function
    // ========================================================================

    pub fn draw(&self, frame: &mut Frame) {
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(65), // Top: Physical Realm + Cognitive Telemetry
                Constraint::Percentage(35), // Bottom: Autonomy Log
            ])
            .split(frame.area());

        // Top half: split into left (Physical Realm) and right (Cognitive Telemetry)
        let top_split = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(40), // Physical Realm
                Constraint::Percentage(60), // Cognitive Telemetry
            ])
            .split(outer[0]);

        self.draw_physical_realm(frame, top_split[0]);
        self.draw_cognitive_telemetry(frame, top_split[1]);
        self.draw_autonomy_log(frame, outer[1]);
    }

    // ========================================================================
    // LEFT PANEL: Physical Realm
    // ========================================================================

    fn draw_physical_realm(&self, frame: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self
            .physical_realm
            .iter()
            .map(|item| match item {
                PhysicalRealmItem::File { path, loaded } => {
                    let icon = if *loaded { "●" } else { "○" };
                    let style = if *loaded {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    };
                    ListItem::new(Line::from(Span::styled(
                        format!(" {} {}", icon, path),
                        style,
                    )))
                }
                PhysicalRealmItem::AstNode { kind, domain } => {
                    let domain_color = match domain.as_str() {
                        "Logic" => Color::Cyan,
                        "Temporal" => Color::Yellow,
                        "Spatial" => Color::Magenta,
                        _ => Color::White,
                    };
                    ListItem::new(Line::from(vec![
                        Span::styled(" ◆ ".to_string(), Style::default().fg(domain_color)),
                        Span::styled(kind.clone(), Style::default().fg(Color::White)),
                        Span::styled(
                            format!(" [{}]", domain),
                            Style::default().fg(domain_color),
                        ),
                    ]))
                }
                PhysicalRealmItem::McpTool { name, status } => {
                    let (icon, color) = match status {
                        McpToolStatus::Idle => ("■", Color::DarkGray),
                        McpToolStatus::Active => ("▶", Color::Green),
                        McpToolStatus::Error => ("✕", Color::Red),
                    };
                    ListItem::new(Line::from(Span::styled(
                        format!(" {} {}", icon, name),
                        Style::default().fg(color),
                    )))
                }
            })
            .collect();

        let list = List::new(items).block(
            Block::default()
                .title(" PHYSICAL REALM ")
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::White))
                .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        );

        frame.render_widget(list, area);
    }

    // ========================================================================
    // RIGHT PANEL: Cognitive Telemetry — Physics gauges and metrics
    // ========================================================================

    fn draw_cognitive_telemetry(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(" COGNITIVE TELEMETRY ")
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::White))
            .title_style(
                Style::default()
                    .fg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            );

        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Split the telemetry panel into rows for each metric
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Active Arkhe
                Constraint::Length(3), // RK4 Phase Alignment
                Constraint::Length(3), // Semantic Divergence Z-Score
                Constraint::Length(3), // Ouroboros Pass Count
                Constraint::Length(3), // Autonomy Level
                Constraint::Min(0),    // Spacer
            ])
            .split(inner);

        // --- Active Arkhe ---
        let arkhe_label = format!(
            " Active Arkhe: {}",
            match self.telemetry.active_arkhe {
                Arkhe::Conditions => "CONDITIONS (Mass Conservation)",
                Arkhe::Harmony => "HARMONY (Wave Interference)",
                Arkhe::Isomorphism => "ISOMORPHISM (Group Theory)",
                Arkhe::BooleanLogic => "BOOLEAN LOGIC (Ternary K-Maps)",
            }
        );
        let arkhe_color = match self.telemetry.active_arkhe {
            Arkhe::Conditions => Color::Blue,
            Arkhe::Harmony => Color::Green,
            Arkhe::Isomorphism => Color::Magenta,
            Arkhe::BooleanLogic => Color::Cyan,
        };
        let arkhe_para = Paragraph::new(Line::from(Span::styled(
            arkhe_label,
            Style::default().fg(arkhe_color).add_modifier(Modifier::BOLD),
        )));
        frame.render_widget(arkhe_para, rows[0]);

        // --- RK4 Phase Alignment Gauge ---
        let rk4_ratio = (self.telemetry.rk4_phase_alignment.clamp(0.0, 1.0)) as f64;
        let rk4_color = if rk4_ratio > 0.8 {
            Color::Green
        } else if rk4_ratio > 0.5 {
            Color::Yellow
        } else {
            Color::Red
        };
        let rk4_gauge = Gauge::default()
            .block(Block::default().title(" RK4 Phase Alignment "))
            .gauge_style(Style::default().fg(rk4_color).bg(Color::Black))
            .ratio(rk4_ratio)
            .label(format!("{:.3}", self.telemetry.rk4_phase_alignment));
        frame.render_widget(rk4_gauge, rows[1]);

        // --- Semantic Divergence Z-Score Gauge ---
        let z = self.telemetry.divergence_z_score;
        // Invert for gauge: low divergence = good (high gauge), high = bad (low gauge)
        let z_ratio = (1.0 - z.clamp(0.0, 1.0)) as f64;
        let z_color = if z < 0.1 {
            Color::Green
        } else if z < 0.5 {
            Color::Yellow
        } else {
            Color::Red
        };
        let z_gauge = Gauge::default()
            .block(Block::default().title(" Semantic Divergence Z-Score "))
            .gauge_style(Style::default().fg(z_color).bg(Color::Black))
            .ratio(z_ratio)
            .label(format!("Z={:.4}", z));
        frame.render_widget(z_gauge, rows[2]);

        // --- Ouroboros Pass Count ---
        let pass_para = Paragraph::new(Line::from(Span::styled(
            format!(" Ouroboros Pass: {}", self.telemetry.ouroboros_pass_count),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )));
        frame.render_widget(pass_para, rows[3]);

        // --- Autonomy Level ---
        let (aut_label, aut_color) = match self.telemetry.autonomy_level {
            AutonomyLevel::Full => ("FULL AUTONOMY", Color::Green),
            AutonomyLevel::Deferred => ("DEFERRED (Picard Iteration)", Color::Yellow),
            AutonomyLevel::HaltAndPrompt => ("HALT — CONSTRUCTIVE INTERFERENCE FAILURE", Color::Red),
        };
        let aut_para = Paragraph::new(Line::from(Span::styled(
            format!(" Autonomy: {}", aut_label),
            Style::default()
                .fg(aut_color)
                .add_modifier(Modifier::BOLD | Modifier::SLOW_BLINK),
        )));
        frame.render_widget(aut_para, rows[4]);
    }

    // ========================================================================
    // BOTTOM PANEL: Autonomy Log — Streaming actuator action matrix
    // ========================================================================

    fn draw_autonomy_log(&self, frame: &mut Frame, area: Rect) {
        // Show the last N entries that fit in the area
        let visible_height = area.height.saturating_sub(2) as usize; // subtract borders
        let start = self.autonomy_log.len().saturating_sub(visible_height);
        let visible = &self.autonomy_log[start..];

        let items: Vec<ListItem> = visible
            .iter()
            .map(|entry| {
                let level_color = match entry.autonomy_level {
                    AutonomyLevel::Full => Color::Green,
                    AutonomyLevel::Deferred => Color::Yellow,
                    AutonomyLevel::HaltAndPrompt => Color::Red,
                };
                let level_tag = match entry.autonomy_level {
                    AutonomyLevel::Full => "FULL",
                    AutonomyLevel::Deferred => "DEFER",
                    AutonomyLevel::HaltAndPrompt => "HALT!",
                };
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("[{:08}] ", entry.timestamp_ms),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(
                        format!("{:<12}", entry.actuator),
                        Style::default().fg(Color::Cyan),
                    ),
                    Span::styled(
                        format!("{:<30}", entry.action),
                        Style::default().fg(Color::White),
                    ),
                    Span::styled(
                        format!("Z={:.3} ", entry.divergence_at_execution),
                        Style::default().fg(Color::Magenta),
                    ),
                    Span::styled(level_tag, Style::default().fg(level_color)),
                ]))
            })
            .collect();

        let list = List::new(items).block(
            Block::default()
                .title(" AUTONOMY LOG ")
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::White))
                .title_style(
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
        );

        frame.render_widget(list, area);
    }
}

impl Default for NeuroTerminal {
    fn default() -> Self {
        Self::new()
    }
}
