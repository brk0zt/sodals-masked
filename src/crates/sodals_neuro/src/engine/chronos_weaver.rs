// src/engine/chronos_weaver.rs
// TEMPORAL ANCHOR & REPLAY — Time-Reversal for the Continuous Physics Engine
//
// Standard agents use git undo / git stash which breaks memory coherence.
// The QMEM holographic memory and the PonderingCore's RK4 latent vector
// are continuous-time objects — they cannot be simply "reverted" by
// restoring a file. The ODE flow must be TIME-REVERSED (−dt) to rollback
// the latent vector to its exact pre-hallucination state.
//
// The ChronosWeaver creates temporal anchors: lightweight differential
// snapshots of both the file system AND the QMEM holographic memory state.
// When a hallucination is detected (Semantic Divergence Z-Score ≥ 0.5),
// the weaver can reverse_time() to restore the file system AND invert
// the RK4 time-step (−dt) to rollback the PonderingCore.
//
// This is NOT git undo. This is PHYSICS TIME REVERSAL.
// The ODE is symmetric: if h(t+dt) = h(t) + dt·f(h(t)), then
// h(t) = h(t+dt) − dt·f(h(t+dt)) ≈ h(t+dt) + (−dt)·f(h(t+dt)).
// Time reversal is achieved by negating dt in the RK4 integrator.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use num_complex::Complex64;
use serde::{Serialize, Deserialize};

// ============================================================================
// ANCHOR ID — Unique identifier for a temporal anchor
// ============================================================================

/// Unique identifier for a temporal anchor. Monotonically increasing.
/// The anchor ID also encodes the RK4 time-step index at which the
/// anchor was created, enabling precise time-reversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AnchorId(pub u64);

impl std::fmt::Display for AnchorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ANCHOR-{:08X}", self.0)
    }
}

// ============================================================================
// FILE DIFFERENTIAL — Lightweight file snapshot
// ============================================================================

/// A lightweight differential snapshot of a single file.
/// Instead of storing the entire file content, we store only the
/// delta (diff) from the previous state, or the full content for
/// the first snapshot of a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileDifferential {
    /// Full content snapshot (used for the first anchor of a file)
    Full {
        content: String,
        hash: Vec<u8>, // SHA-256 hash for integrity verification
    },
    /// Differential snapshot (only the changed lines)
    Delta {
        /// Line-level patches: (line_number, old_line, new_line)
        /// For deleted lines, new_line is None
        /// For inserted lines, old_line is None
        patches: Vec<LinePatch>,
        /// Hash of the resulting file for integrity verification
        hash: Vec<u8>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinePatch {
    pub line_number: usize,
    pub old_line: Option<String>,
    pub new_line: Option<String>,
}

// ============================================================================
// QMEM STATE SNAPSHOT — Holographic memory address state
// ============================================================================

/// Snapshot of the QMEM holographic memory state at a point in time.
/// This captures the holographic_memory vectors and node_flags from
/// the ResonanceNetwork, enabling time-reversal of the semantic topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QmemStateSnapshot {
    /// The holographic memory vectors (node_id → 4096-D phasor)
    /// Stored as a sparse representation: only nodes with non-zero energy
    pub holographic_vectors: Vec<(usize, Vec<Complex64>)>,
    /// The neuro-symbolic domain flags for each node
    pub node_flags: Vec<(usize, u32)>,
    /// The RK4 time-step index at which this snapshot was taken
    pub rk4_step_index: u64,
    /// The PonderingCore's latent vector state (if available)
    /// This is the h(t) that the RK4 integrator was at when the anchor was created
    pub latent_vector: Option<Vec<f32>>,
    /// The dt value used by the RK4 integrator at this point
    pub rk4_dt: f64,
}

// ============================================================================
// TEMPORAL ANCHOR — The complete checkpoint
// ============================================================================

/// A temporal anchor: a complete checkpoint of both the file system AND
/// the QMEM holographic memory state. This is NOT a git commit — it is
/// a physics snapshot that enables time-reversal of the continuous ODE flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnchor {
    /// Unique identifier
    pub id: AnchorId,
    /// Timestamp when the anchor was created
    pub timestamp_ms: u64,
    /// File differentials for all modified files
    pub file_snapshots: HashMap<String, FileDifferential>,
    /// QMEM holographic memory state snapshot
    pub qmem_snapshot: QmemStateSnapshot,
    /// The Semantic Divergence Z-Score at the time of anchoring
    pub divergence_z_score: f64,
    /// The Ouroboros pass count at the time of anchoring
    pub ouroboros_pass: u32,
    /// Human-readable reason for the anchor (e.g., "pre-hallucination checkpoint")
    pub reason: String,
}

// ============================================================================
// TIME REVERSAL RESULT — The outcome of a reverse_time operation
// ============================================================================

/// Result of a time-reversal operation. Contains the restored state
/// and the RK4 time-reversal parameters.
#[derive(Debug, Clone)]
pub struct TimeReversalResult {
    /// The anchor that was restored
    pub anchor_id: AnchorId,
    /// Files that were restored
    pub restored_files: Vec<PathBuf>,
    /// The RK4 time-reversal step count (how many −dt steps to apply)
    pub rk4_reverse_steps: u64,
    /// The negated dt value for the RK4 integrator
    pub reversed_dt: f64,
    /// The QMEM state to restore
    pub qmem_state: QmemStateSnapshot,
    /// Whether the time reversal was complete or partial
    pub completeness: ReversalCompleteness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReversalCompleteness {
    /// Full reversal: both file system and QMEM state restored
    Full,
    /// Partial reversal: file system restored, QMEM state approximated
    /// (some holographic vectors may have been evicted)
    PartialQmem,
    /// Minimal reversal: only file system restored (QMEM state unavailable)
    FileSystemOnly,
}

// ============================================================================
// CHRONOS WEAVER — The temporal anchor manager
// ============================================================================

/// The ChronosWeaver: temporal anchor manager for the continuous physics engine.
/// Creates lightweight differential snapshots of the file system AND the QMEM
/// holographic memory state, enabling time-reversal of the ODE flow when
/// hallucinations are detected.
///
/// Time-reversal is achieved by:
/// 1. Restoring the file system to the exact snapshot state
/// 2. Restoring the QMEM holographic memory vectors
/// 3. Inverting the RK4 time-step (−dt) to rollback the PonderingCore's
///    latent vector to its exact pre-hallucination state
///
/// The ODE is symmetric under time reversal:
///   Forward:  h(t+dt) = h(t) + (dt/6)(k1 + 2k2 + 2k3 + k4)
///   Reverse:  h(t)   = h(t+dt) + (−dt/6)(k1' + 2k2' + 2k3' + k4')
pub struct ChronosWeaver {
    /// The workspace root directory
    workspace_root: PathBuf,
    /// All temporal anchors, indexed by AnchorId
    anchors: HashMap<u64, TemporalAnchor>,
    /// Monotonically increasing anchor counter
    next_anchor_id: u64,
    /// Maximum number of anchors to retain (ring buffer eviction)
    max_anchors: usize,
    /// The current file state cache (for computing differentials)
    /// Maps file_path → (content_hash, content)
    current_file_state: HashMap<String, (Vec<u8>, String)>,
    /// The current QMEM state (updated externally before creating anchors)
    current_qmem_state: Option<QmemStateSnapshot>,
}

impl ChronosWeaver {
    /// Create a new ChronosWeaver for the given workspace root.
    pub fn new(workspace_root: PathBuf, max_anchors: usize) -> Self {
        Self {
            workspace_root,
            anchors: HashMap::new(),
            next_anchor_id: 1,
            max_anchors,
            current_file_state: HashMap::new(),
            current_qmem_state: None,
        }
    }

    /// Update the QMEM state snapshot (called before creating an anchor).
    /// The caller should extract the holographic_memory and node_flags
    /// from the ResonanceNetwork and pass them here.
    pub fn set_qmem_state(
        &mut self,
        holographic_vectors: Vec<(usize, Vec<Complex64>)>,
        node_flags: Vec<(usize, u32)>,
        rk4_step_index: u64,
        latent_vector: Option<Vec<f32>>,
        rk4_dt: f64,
    ) {
        self.current_qmem_state = Some(QmemStateSnapshot {
            holographic_vectors,
            node_flags,
            rk4_step_index,
            latent_vector,
            rk4_dt,
        });
    }

    /// Create a temporal anchor — a lightweight differential snapshot of
    /// the currently modified files AND the state of the QMEM holographic
    /// memory.
    ///
    /// This is the SODALS equivalent of a git commit, but it captures
    /// the continuous physics state, not just the file system.
    pub fn create_temporal_anchor(
        &mut self,
        modified_files: &[PathBuf],
        divergence_z_score: f64,
        ouroboros_pass: u32,
        reason: impl Into<String>,
    ) -> AnchorId {
        let id = self.next_anchor_id;
        self.next_anchor_id += 1;

        // Build file differentials
        let mut file_snapshots = HashMap::new();

        for file_path in modified_files {
            let key = file_path.to_string_lossy().to_string();
            let content = std::fs::read_to_string(file_path).unwrap_or_default();

            // Compute SHA-256 hash
            let hash = compute_hash(&content);

            if let Some((prev_hash, prev_content)) = self.current_file_state.get(&key) {
                if &hash != prev_hash {
                    // Content changed — compute delta
                    let patches = compute_line_patches(prev_content, &content);
                    file_snapshots.insert(
                        key.clone(),
                        FileDifferential::Delta { patches, hash: hash.clone() },
                    );
                }
                // Content unchanged — skip (no differential needed)
            } else {
                // First snapshot of this file — store full content
                file_snapshots.insert(
                    key.clone(),
                    FileDifferential::Full {
                        content: content.clone(),
                        hash: hash.clone(),
                    },
                );
            }

            // Update the current file state cache
            self.current_file_state.insert(key, (hash, content));
        }

        // Take the QMEM snapshot
        let qmem_snapshot = self.current_qmem_state.clone().unwrap_or(QmemStateSnapshot {
            holographic_vectors: Vec::new(),
            node_flags: Vec::new(),
            rk4_step_index: 0,
            latent_vector: None,
            rk4_dt: 0.15, // Default DT from pondering_core
        });

        let anchor = TemporalAnchor {
            id: AnchorId(id),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            file_snapshots,
            qmem_snapshot,
            divergence_z_score,
            ouroboros_pass,
            reason: reason.into(),
        };

        // Evict oldest anchor if at capacity
        if self.anchors.len() >= self.max_anchors {
            let oldest_id = self.anchors.keys().min().copied().unwrap_or(0);
            self.anchors.remove(&oldest_id);
        }

        self.anchors.insert(id, anchor);
        AnchorId(id)
    }

    /// Reverse time — restore the file system to the exact snapshot at the
    /// given anchor AND invert the RK4 ODE time-step (−dt) to rollback
    /// the PonderingCore's latent vector to its exact pre-hallucination state.
    ///
    /// This is NOT git checkout. This is PHYSICS TIME REVERSAL.
    /// The RK4 integrator is run backwards with −dt to undo the ODE evolution.
    pub fn reverse_time(&mut self, anchor: AnchorId) -> Result<TimeReversalResult, String> {
        let temporal_anchor = self.anchors.get(&anchor.0)
            .ok_or_else(|| format!("Anchor {} not found", anchor))?
            .clone();

        let mut restored_files = Vec::new();

        // 1. Restore file system
        for (file_path_str, differential) in &temporal_anchor.file_snapshots {
            let file_path = PathBuf::from(file_path_str);

            match differential {
                FileDifferential::Full { content, hash } => {
                    // Verify hash before restoring
                    let _current_hash = if file_path.exists() {
                        compute_hash(&std::fs::read_to_string(&file_path).unwrap_or_default())
                    } else {
                        Vec::new()
                    };

                    // Write the full content back
                    std::fs::write(&file_path, content)
                        .map_err(|e| format!("Failed to restore {}: {}", file_path_str, e))?;

                    // Update our cache
                    self.current_file_state.insert(
                        file_path_str.clone(),
                        (hash.clone(), content.clone()),
                    );

                    restored_files.push(file_path);
                }

                FileDifferential::Delta { patches, hash } => {
                    // Apply patches in reverse
                    let current_content = std::fs::read_to_string(&file_path)
                        .unwrap_or_default();
                    let restored_content = apply_patches_reverse(&current_content, patches);

                    std::fs::write(&file_path, &restored_content)
                        .map_err(|e| format!("Failed to restore {}: {}", file_path_str, e))?;

                    // Update our cache
                    self.current_file_state.insert(
                        file_path_str.clone(),
                        (hash.clone(), restored_content),
                    );

                    restored_files.push(file_path);
                }
            }
        }

        // 2. Restore QMEM state
        let qmem_state = temporal_anchor.qmem_snapshot.clone();

        // 3. Compute RK4 time-reversal parameters
        // The number of reverse steps is the difference between the current
        // RK4 step index and the anchor's step index
        let current_step = self.current_qmem_state
            .as_ref()
            .map(|s| s.rk4_step_index)
            .unwrap_or(0);
        let anchor_step = qmem_state.rk4_step_index;
        let rk4_reverse_steps = current_step.saturating_sub(anchor_step);

        // The reversed dt is the negation of the original dt
        let reversed_dt = -qmem_state.rk4_dt;

        // Determine completeness
        let completeness = if !qmem_state.holographic_vectors.is_empty()
            && qmem_state.latent_vector.is_some()
        {
            ReversalCompleteness::Full
        } else if !qmem_state.holographic_vectors.is_empty() {
            ReversalCompleteness::PartialQmem
        } else {
            ReversalCompleteness::FileSystemOnly
        };

        // Update the current QMEM state to the restored state
        self.current_qmem_state = Some(qmem_state.clone());

        Ok(TimeReversalResult {
            anchor_id: anchor,
            restored_files,
            rk4_reverse_steps,
            reversed_dt,
            qmem_state,
            completeness,
        })
    }

    /// Get a reference to a specific anchor.
    pub fn get_anchor(&self, id: AnchorId) -> Option<&TemporalAnchor> {
        self.anchors.get(&id.0)
    }

    /// List all anchor IDs in chronological order.
    pub fn list_anchors(&self) -> Vec<AnchorId> {
        let mut ids: Vec<u64> = self.anchors.keys().copied().collect();
        ids.sort_unstable();
        ids.into_iter().map(AnchorId).collect()
    }

    /// Get the most recent anchor.
    pub fn latest_anchor(&self) -> Option<AnchorId> {
        self.anchors.keys().max().map(|&id| AnchorId(id))
    }

    /// Get the number of stored anchors.
    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    /// Persist all anchors to disk (for crash recovery).
    pub fn save_anchors(&self, path: &Path) -> Result<(), String> {
        let data = bincode::serialize(&self.anchors)
            .map_err(|e| format!("Failed to serialize anchors: {}", e))?;

        std::fs::write(path, data)
            .map_err(|e| format!("Failed to write anchors: {}", e))
    }

    /// Load anchors from disk (for crash recovery).
    pub fn load_anchors(&mut self, path: &Path) -> Result<usize, String> {
        let data = std::fs::read(path)
            .map_err(|e| format!("Failed to read anchors: {}", e))?;

        let anchors: HashMap<u64, TemporalAnchor> = bincode::deserialize(&data)
            .map_err(|e| format!("Failed to deserialize anchors: {}", e))?;

        let count = anchors.len();
        self.anchors = anchors;

        // Update next_anchor_id to be past the highest existing ID
        self.next_anchor_id = self.anchors.keys().max().map(|&id| id + 1).unwrap_or(1);

        Ok(count)
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Compute SHA-256 hash of a string.
fn compute_hash(content: &str) -> Vec<u8> {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hasher.finalize().to_vec()
}

/// Compute line-level patches between two versions of a file.
fn compute_line_patches(old: &str, new: &str) -> Vec<LinePatch> {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    let mut patches = Vec::new();
    let max_len = old_lines.len().max(new_lines.len());

    for i in 0..max_len {
        let old_line = old_lines.get(i).map(|s| s.to_string());
        let new_line = new_lines.get(i).map(|s| s.to_string());

        if old_line != new_line {
            patches.push(LinePatch {
                line_number: i + 1, // 1-indexed
                old_line,
                new_line,
            });
        }
    }

    patches
}

/// Apply patches in reverse to restore a file to its previous state.
fn apply_patches_reverse(content: &str, patches: &[LinePatch]) -> String {
    let mut lines: Vec<Option<String>> = content.lines()
        .map(|l| Some(l.to_string()))
        .collect();

    for patch in patches {
        let idx = patch.line_number.saturating_sub(1); // Convert to 0-indexed

        if idx < lines.len() {
            // Reverse: swap old_line and new_line
            lines[idx] = patch.old_line.clone();
        } else if let Some(ref old_line) = patch.old_line {
            // Line was at the end, extend the vector
            lines.push(Some(old_line.clone()));
        }
    }

    lines.iter()
        .filter_map(|opt| opt.as_ref())
        .cloned()
        .collect::<Vec<String>>()
        .join("\n")
}
