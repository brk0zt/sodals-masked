// src/engine/lsp_watcher.rs
// ASYNCHRONOUS LSP WATCHER — Continuous Compilation Diagnostics as Physics Impedance
//
// Standard agents run `cargo check` synchronously after writing. We run it
// CONTINUOUSLY in the background, parsing JSON diagnostics and converting
// compilation errors into PhysicsImpedance — topological weights that the
// PonderingCore treats as physical obstacles in the latent manifold.
//
// A syntax error is NOT a text message to display. It is a TOPOLOGICAL
// OBSTRUCTION — a high-impedance region in the ODE flow that the RK4
// integrator must route around or dissipate via Picard iteration.
//
// Architecture:
//   cargo check --message-format=json (background tokio task)
//       │
//       ▼
//   JSON line parsing → DiagnosticEvent
//       │
//       ▼
//   PhysicsImpedance conversion (error severity → impedance magnitude)
//       │
//       ▼
//   mpsc::channel → PonderingCore (ingests as boundary condition)

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::process::Command;
use tokio::io::AsyncBufReadExt;
use serde::Deserialize;

use crate::semantic_parser::{DOMAIN_LOGIC, DOMAIN_SPATIAL, DOMAIN_TEMPORAL, DOMAIN_ABSTRACT};
use crate::engine::mcp_hdc_bridge::PhasorBoundaryCondition;

// ============================================================================
// CARGO DIAGNOSTIC JSON TYPES — Subset of cargo check --message-format=json
// ============================================================================

/// Top-level cargo check JSON message (we only care about compiler diagnostics)
#[derive(Debug, Clone, Deserialize)]
pub struct CargoMessage {
    pub reason: Option<String>,
    /// Nested compiler message (present when reason == "compiler-message")
    pub message: Option<CompilerMessage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompilerMessage {
    pub message: String,
    pub level: DiagnosticLevel,
    pub spans: Vec<DiagnosticSpan>,
    pub code: Option<DiagnosticCode>,
    /// Children are sub-diagnostics (e.g., "help: consider removing this")
    pub children: Vec<CompilerMessage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DiagnosticSpan {
    pub file_name: String,
    pub line_start: usize,
    pub line_end: usize,
    pub column_start: usize,
    pub column_end: usize,
    pub text: Vec<DiagnosticText>,
    pub is_primary: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DiagnosticText {
    pub text: String,
    pub highlight_start: usize,
    pub highlight_end: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DiagnosticCode {
    pub code: String,
    pub explanation: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticLevel {
    Error,
    Warning,
    Note,
    Help,
    #[serde(other)]
    Unknown,
}

// ============================================================================
// PHYSICS IMPEDANCE — Compilation errors as topological weights
// ============================================================================

/// A compilation diagnostic converted into a physical impedance in the
/// latent manifold. The PonderingCore treats this as a topological
/// obstruction — a high-impedance region that the RK4 integrator must
/// either route around or dissipate via Picard iteration.
///
/// Impedance magnitude is derived from diagnostic severity:
///   - Error   → High impedance (0.8-1.0): Hard obstruction, flow must stop or reroute
///   - Warning → Medium impedance (0.3-0.6): Soft friction, flow is dampened
///   - Note    → Low impedance (0.05-0.15): Informational, negligible friction
///   - Help    → Negative impedance (-0.1): Assistive, actually reduces friction
#[derive(Debug, Clone)]
pub struct PhysicsImpedance {
    /// The source file where the impedance originates
    pub file_path: PathBuf,
    /// Line range in the source file
    pub line_range: (usize, usize),
    /// The diagnostic level converted to impedance magnitude
    pub impedance: f64,
    /// The neuro-symbolic domain this impedance affects
    /// (determined by the error code — type errors are SPATIAL,
    ///  borrow checker errors are TEMPORAL, logic errors are LOGIC)
    pub domain_flag: u32,
    /// The original diagnostic message (for TUI display only)
    pub message: String,
    /// The Rust error code (e.g., E0308, E0425)
    pub code: Option<String>,
    /// The phasor boundary condition derived from this impedance
    /// (ready for injection into the PonderingCore)
    pub boundary_condition: Option<PhasorBoundaryCondition>,
}

impl PhysicsImpedance {
    /// Convert a compiler diagnostic into a physics impedance.
    /// The impedance magnitude is severity-dependent; the domain flag
    /// is determined by the error code classification.
    pub fn from_compiler_message(msg: &CompilerMessage) -> Vec<Self> {
        let base_impedance = match msg.level {
            DiagnosticLevel::Error => 0.9,
            DiagnosticLevel::Warning => 0.4,
            DiagnosticLevel::Note => 0.1,
            DiagnosticLevel::Help => -0.1,
            DiagnosticLevel::Unknown => 0.2,
        };

        let domain_flag = classify_error_domain(msg.code.as_ref().map(|c| c.code.as_str()));

        msg.spans
            .iter()
            .filter(|s| s.is_primary)
            .map(|span| {
                // Modulate impedance by span size — larger errors = higher impedance
                let span_lines = (span.line_end - span.line_start).max(1) as f64;
                let modulated = base_impedance * (1.0 + span_lines * 0.05).min(2.0);

                Self {
                    file_path: PathBuf::from(&span.file_name),
                    line_range: (span.line_start, span.line_end),
                    impedance: modulated,
                    domain_flag,
                    message: msg.message.clone(),
                    code: msg.code.as_ref().map(|c| c.code.clone()),
                    boundary_condition: None, // Filled by the watcher after construction
                }
            })
            .collect()
    }
}

/// Classify a Rust error code into a neuro-symbolic domain.
///
/// - E0xxx type mismatches, trait bounds → SPATIAL (data structure topology)
/// - E0xxx borrow/lifetime errors → TEMPORAL (ownership over time)
/// - E0xxx unreachable/unresolved → LOGIC (control flow)
/// - Everything else → ABSTRACT
fn classify_error_domain(code: Option<&str>) -> u32 {
    match code {
        Some(c) if c.starts_with("E0")
            && (c.contains("28")  // E0282: type annotations needed
                || c.contains("30")  // E0308: mismatched types
                || c.contains("43")  // E0433: unresolved import
                || c.contains("41")  // E0412: unresolved name
                || c.contains("77")) // E0777: impossible to satisfy trait bound
        => DOMAIN_SPATIAL,

        Some(c) if c.contains("49")  // E0495: cannot borrow
            || c.contains("50")      // E0502: borrowed as mutable
            || c.contains("51")      // E0515: cannot return reference
            || c.contains("52")      // E0521: data race
            || c.contains("59")      // E0597: temporary value dropped
            || c.contains("58")      // E0583: file not found
        => DOMAIN_TEMPORAL,

        Some(c) if c.contains("27")  // E0277: trait bound not satisfied
            || c.contains("46")      // E0463: macro not found
        => DOMAIN_LOGIC,

        _ => DOMAIN_ABSTRACT,
    }
}

// ============================================================================
// DIAGNOSTIC EVENT — The message sent over the channel
// ============================================================================

/// Event sent from the background watcher to consumers (PonderingCore, TUI).
#[derive(Debug, Clone)]
pub enum DiagnosticEvent {
    /// New compilation diagnostics detected, converted to impedance
    ImpedanceDetected(Vec<PhysicsImpedance>),
    /// Cargo check completed (cycle done)
    CheckCycleComplete {
        error_count: usize,
        warning_count: usize,
    },
    /// The watcher encountered an internal error
    WatcherError(String),
}

// ============================================================================
// DIAGNOSTIC WATCHER — The async background watcher
// ============================================================================

/// Continuous background watcher that runs `cargo check --message-format=json`
/// on the workspace and converts compilation diagnostics into PhysicsImpedance
/// that is fed to the PonderingCore via a channel.
///
/// This is the SODALS equivalent of an LSP server — but instead of displaying
/// squiggly lines in an editor, we inject impedance into the ODE flow.
pub struct DiagnosticWatcher {
    /// Path to the workspace root (where Cargo.toml lives)
    workspace_root: PathBuf,
    /// Sender half of the diagnostic channel
    tx: mpsc::UnboundedSender<DiagnosticEvent>,
    /// Receiver half (consumed by the PonderingCore or TUI)
    rx: mpsc::UnboundedReceiver<DiagnosticEvent>,
    /// Interval between check cycles (milliseconds)
    check_interval_ms: u64,
    /// Whether the watcher is currently active
    active: Arc<std::sync::atomic::AtomicBool>,
}

impl DiagnosticWatcher {
    /// Create a new watcher for the given workspace root.
    pub fn new(workspace_root: PathBuf, check_interval_ms: u64) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self {
            workspace_root,
            tx,
            rx,
            check_interval_ms,
            active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Get a clone of the sender for wiring into other subsystems.
    pub fn sender(&self) -> mpsc::UnboundedSender<DiagnosticEvent> {
        self.tx.clone()
    }

    /// Take the receiver (can only be called once). The consumer (PonderingCore)
    /// calls this to receive diagnostic events.
    pub fn take_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<DiagnosticEvent>> {
        // mpsc::UnboundedReceiver is not Clone, so we can only hand it out once
        Some(std::mem::replace(&mut self.rx, {
            // Create a dummy channel to replace the taken receiver
            let (_, dummy_rx) = mpsc::unbounded_channel();
            dummy_rx
        }))
    }

    /// Check if the watcher is currently active.
    pub fn is_active(&self) -> bool {
        self.active.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Start the background watcher. Spawns a tokio task that runs
    /// `cargo check --message-format=json` continuously.
    pub fn start(&self) {
        self.active.store(true, std::sync::atomic::Ordering::SeqCst);

        let workspace_root = self.workspace_root.clone();
        let tx = self.tx.clone();
        let interval = self.check_interval_ms;
        let active = self.active.clone();

        tokio::spawn(async move {
            while active.load(std::sync::atomic::Ordering::SeqCst) {
                if let Err(e) = run_check_cycle(&workspace_root, &tx).await {
                    let _ = tx.send(DiagnosticEvent::WatcherError(format!(
                        "Check cycle failed: {}", e
                    )));
                }

                tokio::time::sleep(std::time::Duration::from_millis(interval)).await;
            }
        });
    }

    /// Stop the background watcher.
    pub fn stop(&self) {
        self.active.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Run a single `cargo check --message-format=json` cycle and parse the output.
async fn run_check_cycle(
    workspace_root: &PathBuf,
    tx: &mpsc::UnboundedSender<DiagnosticEvent>,
) -> Result<(), String> {
    let mut child = Command::new("cargo")
        .args(["check", "--message-format=json"])
        .current_dir(workspace_root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn cargo check: {}", e))?;

    let stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
    let reader = tokio::io::BufReader::new(stdout);
    let mut lines = reader.lines();

    let mut error_count = 0usize;
    let mut warning_count = 0usize;
    let mut all_impedances = Vec::new();

    while let Some(line) = lines.next_line().await.map_err(|e| e.to_string())? {
        let msg: CargoMessage = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(_) => continue, // Not all lines are JSON diagnostics
        };

        // We only process compiler messages
        if msg.reason.as_deref() != Some("compiler-message") {
            continue;
        }

        if let Some(compiler_msg) = &msg.message {
            match compiler_msg.level {
                DiagnosticLevel::Error => error_count += 1,
                DiagnosticLevel::Warning => warning_count += 1,
                _ => {}
            }

            let impedances = PhysicsImpedance::from_compiler_message(compiler_msg);
            all_impedances.extend(impedances);
        }
    }

    // Wait for the process to finish
    let _ = child.wait().await;

    // Send impedances to the channel
    if !all_impedances.is_empty() {
        let _ = tx.send(DiagnosticEvent::ImpedanceDetected(all_impedances));
    }

    let _ = tx.send(DiagnosticEvent::CheckCycleComplete {
        error_count,
        warning_count,
    });

    Ok(())
}
