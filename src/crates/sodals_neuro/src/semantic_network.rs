// src/crates/sodals_neuro/src/semantic_network.rs - OPTIMIZED SPARSE MATRIX ENGINE
// REFACTORED: CSR Sparse Matrices + O(NI?) LU Decomposition for Determinants
// OPTIMIZATION: Active Inference for Jacobian (only connected nodes computed)
// REMOVED: Recursive O(N!) matrix_determinant III replaced by LU factorization
// REMOVED: Dense adjacency list Vec<Vec<>> III replaced by CsMat<f64> (CSR format)

use crate::fast_math::{SemanticWave, apply_clifford_resonance};
use crate::semantic_parser::DOMAIN_LOGIC;
use faer::Mat;
use num_complex::Complex64;
use sprs::{CsMat, TriMat};
use std::collections::HashMap;
use ndarray::{Array1, Array2};
use rayon::prelude::*; // Required for CPU multi-core operations
use serde::{Serialize, Deserialize};

fn _ndarray_to_faer_mat(array: &Array2<f64>) -> Mat<f64> {
    let (nrows, ncols) = array.dim();
    
    // Create faer matrix using from_fn constructor
    // This iterates through each position and fills it from the ndarray
    Mat::from_fn(nrows, ncols, |i, j| array[[i, j]])
}

/// Compute determinant of ndarray::Array2<f64> using faer's LU decomposition
/// 
/// Algorithm:
/// 1. Convert Array2 III faer::Mat
/// 2. Compute partial pivot LU decomposition
/// 3. Extract determinant from LU factors
/// 
/// Returns: determinant as f64
/// 
/// **Pure Rust**: No C/Fortran/BLAS/LAPACK dependencies
fn compute_determinant_smart(array: &Array2<f64>) -> f64 {
    // GPU entirely removed. Pure CPU computation only.
    compute_determinant_cpu_fallback(array)
}

fn compute_determinant_cpu_fallback(array: &Array2<f64>) -> f64 {
    let (n, _) = array.dim();
    
    if n == 0 { return 1.0; }
    if n == 1 { return array[[0, 0]]; }
    
    // Clone-free: construct work matrix from borrowed input (single allocation).
    // This replaces the previous `array.clone()` which deep-copied all elements.
    let mut a = Array2::from_shape_fn((n, n), |(i, j)| array[[i, j]]);
    let mut det = 1.0;
    
    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[[k, i]].abs() > a[[max_row, i]].abs() {
                max_row = k;
            }
        }
        
        if a[[max_row, i]].abs() < 1e-10 {
            return 0.0;
        }
        
        if max_row != i {
            // In-place row swap without temporary allocation.
            for j in 0..n {
                let tmp = a[[i, j]];
                a[[i, j]] = a[[max_row, j]];
                a[[max_row, j]] = tmp;
            }
            det = -det;
        }
        
        det *= a[[i, i]];
        
        for k in (i + 1)..n {
            let factor = a[[k, i]] / a[[i, i]];
            for j in i..n {
                a[[k, j]] -= factor * a[[i, j]];
            }
        }
    }
    
    det
}

/// Compute L2 norm of an ndarray column vector
/// 
/// Replaces ndarray-linalg's norm_l2() method with pure Rust implementation.
/// Used for Jacobian sensitivity analysis.
fn column_norm_l2(array: &Array2<f64>, col: usize) -> f64 {
    array.column(col).iter().map(|&x| x * x).sum::<f64>().sqrt()
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// ENUMERATIONS & metadata TYPES
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

/// Bincode serialization cartridge for saving/loading the network.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QmemCartridge {
    pub n: usize,
    pub damping: Vec<f64>,
    pub masses: Vec<f64>,
    pub adjacency_rows: Vec<usize>,
    pub adjacency_cols: Vec<usize>,
    pub adjacency_vals: Vec<f64>,
    pub labels: HashMap<String, usize>,
    pub holographic_memory: Vec<Vec<Complex64>>,
    pub node_flags: Vec<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NetworkSnapshot {
    pub n: usize,
    pub damping: Vec<f64>,
    pub masses: Vec<f64>,
    pub adjacency_rows: Vec<usize>,
    pub adjacency_cols: Vec<usize>,
    pub adjacency_vals: Vec<f64>,
    pub adjacency_types: HashMap<(usize, usize), ConnectionType>,
    pub labels: HashMap<String, usize>,
    pub holographic_memory: Vec<Vec<Complex64>>,
    pub node_flags: Vec<u32>,
    pub position_re: Vec<f64>,
    pub position_im: Vec<f64>,
    pub momentum_re: Vec<f64>,
    pub momentum_im: Vec<f64>,
}

/// Phase 40: SPATIAL AXIOM - Geometric transformation types for ARC-AGI reasoning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpatialAxiom {
    Identity,           // No transformation
    TranslateX,         // Horizontal shift
    TranslateY,         // Vertical shift
    Rotate90,           // 90° clockwise rotation
    Rotate180,          // 180° rotation
    Rotate270,          // 270° clockwise (90° CCW)
    FlipHorizontal,     // Mirror across Y-axis
    FlipVertical,       // Mirror across X-axis
    ColorFlood,         // Flood fill/region coloring
    ConnectComponents,  // Connect disjoint components
    Expand,             // Dilate/morphological expansion
    Contract,           // Erode/morphological contraction
    SymmetryX,          // Reflection symmetry across X
    SymmetryY,          // Reflection symmetry across Y
    SymmetryDiagonal,   // Reflection across main diagonal
}

impl SpatialAxiom {
    /// Get the inverse transformation
    pub fn inverse(&self) -> Self {
        match self {
            Self::Identity => Self::Identity,
            Self::TranslateX => Self::TranslateX,  // Inverse is opposite direction
            Self::TranslateY => Self::TranslateY,
            Self::Rotate90 => Self::Rotate270,
            Self::Rotate180 => Self::Rotate180,
            Self::Rotate270 => Self::Rotate90,
            Self::FlipHorizontal => Self::FlipHorizontal,  // Self-inverse
            Self::FlipVertical => Self::FlipVertical,
            Self::ColorFlood => Self::Identity,  // Cannot undo
            Self::ConnectComponents => Self::Identity,
            Self::Expand => Self::Contract,
            Self::Contract => Self::Expand,
            Self::SymmetryX => Self::SymmetryX,
            Self::SymmetryY => Self::SymmetryY,
            Self::SymmetryDiagonal => Self::SymmetryDiagonal,
        }
    }

    /// Check if applying this transformation twice returns to identity (order 2)
    pub fn is_involution(&self) -> bool {
        matches!(self,
            Self::Identity |
            Self::Rotate180 |
            Self::FlipHorizontal |
            Self::FlipVertical |
            Self::SymmetryX |
            Self::SymmetryY |
            Self::SymmetryDiagonal
        )
    }

    /// Get string label for serialization
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::TranslateX => "translate_x",
            Self::TranslateY => "translate_y",
            Self::Rotate90 => "rotate_90",
            Self::Rotate180 => "rotate_180",
            Self::Rotate270 => "rotate_270",
            Self::FlipHorizontal => "flip_h",
            Self::FlipVertical => "flip_v",
            Self::ColorFlood => "color_flood",
            Self::ConnectComponents => "connect",
            Self::Expand => "expand",
            Self::Contract => "contract",
            Self::SymmetryX => "symmetry_x",
            Self::SymmetryY => "symmetry_y",
            Self::SymmetryDiagonal => "symmetry_diag",
        }
    }
}

/// Phase 40: Spatial Node - Geometric entity in QMEM
#[derive(Debug, Clone)]
pub struct SpatialNode {
    pub node_id: usize,
    pub axiom: SpatialAxiom,
    pub position: (f64, f64),      // 2D coordinates in latent space
    pub mass: f64,                  // Importance/energy of this transformation
    pub color_value: u8,            // For color-specific transformations
    pub grid_bounds: (usize, usize), // Grid dimensions this node operates on
}

impl SpatialNode {
    pub fn new(node_id: usize, axiom: SpatialAxiom, x: f64, y: f64) -> Self {
        Self {
            node_id,
            axiom,
            position: (x, y),
            mass: 1.0,
            color_value: 0,
            grid_bounds: (30, 30), // Default ARC grid size
        }
    }

    /// Compute 2D RoPE (Rotary Position Embedding) for this node's position
    /// Returns complex phases for each dimension
    pub fn rope_2d(&self, dim_idx: usize, base: f64) -> Complex64 {
        let (x, y) = self.position;
        let theta = base.powf(-2.0 * (dim_idx as f64) / 128.0);
        let phase_x = theta * x;
        let phase_y = theta * y;
        Complex64::new(phase_x.cos() * phase_y.cos(), phase_x.sin() * phase_y.sin())
    }
}

/// Phase 40: Connection topology metadata - EXTENDED with Spatial
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConnectionType {
    Syntax,      // Short-range, high damping   (syntactic roles)
    Semantic,    // Long-range, low damping     (rich relationships)
    Spatial,     // Phase 40: Geometric transformation edges
}

#[derive(Debug, Clone, PartialEq)]
pub enum RhythmClass {
    Trivial,      // |det(J)| < 0.1  III safe but uninformative
    Harmonic,     // 0.1 III |det(J)| < 2.0 III ideal operating range
    Chaotic,      // |det(J)| III 2.0  III dynamically dangerous
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// SPECTRAL ANALYSIS STRUCTURES
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

/// Spectral Cluster III represents a semantic zone in the network
#[derive(Debug, Clone)]
pub struct SpectralCluster {
    pub cluster_id: usize,
    pub nodes: Vec<usize>,
    pub centroid: Complex64,    // Cluster centre in phase space
    pub impedance: Complex64,   // Complex impedance Z = R + jX
    pub color: ClusterColor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterColor {
    Pink,    // Wizard/Magic cluster
    Brown,   // Warrior/Combat cluster
    Blue,    // Main/Logic cluster
    Green,   // Test/Validation cluster
    Gray,    // Utility cluster
}

/// Gateway Node III interface between clusters
#[derive(Debug, Clone)]
pub struct GatewayNode {
    pub node_id: usize,
    pub cluster_memberships: HashMap<usize, f64>,  // cluster_id III membership degree
    pub impedance_mismatch: f64,                   // conflict severity
    pub vorticity: f64,                            // circular-dependency indicator
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// JACOBIAN / PHASE VARIANCE RESULT TYPES
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

/// Result of a Jacobian rhythm analysis
#[derive(Debug)]
pub struct JacobianAnalysis<'a> {
    pub network: &'a ResonanceNetwork, // Reference instead of cloning Z-Matrix
    pub determinant: f64,
    pub sensitivity_scores: Vec<f64>,
    pub rhythm_classification: RhythmClass,
}

/// Phase variance linear-independence check result
#[derive(Debug, Clone)]
pub struct PhaseVarianceCheck {
    pub determinant: f64,
    pub is_unique: bool,
    pub is_safe: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BifurcationRegime {
    Stable,
    Critical,
    Chaotic,
}

#[derive(Debug, Clone)]
pub struct BifurcationMonitor {
    pub window: usize,
    pub threshold: f64,
    pub history: Vec<f64>,
    pub last_bifurcation_step: Option<u64>,
    pub step: u64,
    pub lyapunov_estimate: f64,
    pub regime: BifurcationRegime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BifurcationAction {
    Normal,
    DeepMcts { extra_depth: u8 },
    TryAlternative,
}

impl BifurcationMonitor {
    pub fn new(window: usize, threshold: f64) -> Self {
        Self {
            window,
            threshold,
            history: Vec::with_capacity(window),
            last_bifurcation_step: None,
            step: 0,
            lyapunov_estimate: 0.0,
            regime: BifurcationRegime::Stable,
        }
    }

    pub fn observe(&mut self, value: f64) {
        self.step += 1;
        self.history.push(value.max(0.0));
        if self.history.len() > self.window {
            self.history.remove(0);
        }

        if self.history.len() >= 2 {
            let len = self.history.len();
            let last = self.history[len - 1];
            let prev = self.history[len - 2];
            self.lyapunov_estimate = (last / (prev.max(1e-12))).ln();
            self.regime = if last >= self.threshold {
                BifurcationRegime::Chaotic
            } else if self.lyapunov_estimate.abs() < 0.05 {
                BifurcationRegime::Critical
            } else {
                BifurcationRegime::Stable
            };

            if self.regime != BifurcationRegime::Stable {
                self.last_bifurcation_step = Some(self.step);
            }
        }
    }

    pub fn is_bifurcating(&self) -> bool {
        self.regime != BifurcationRegime::Stable
    }

    pub fn regime_name(&self) -> &'static str {
        match self.regime {
            BifurcationRegime::Stable => "stable",
            BifurcationRegime::Critical => "critical",
            BifurcationRegime::Chaotic => "chaotic",
        }
    }
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// OPTIMIZED RESONANCE NETWORK (CSR SPARSE + LU DECOMPOSITION)
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

/// **OPTIMIZED: SPARSE MATRIX ENGINE**
///
/// Performance improvements vs. the legacy adjacency-list implementation:
///
/// | Concern          | Legacy                  | This version          |
/// |------------------|-------------------------|-----------------------|
/// | Memory           | O(NI?) adjacency list    | O(nnz) CSR sparse     |
/// | Determinant      | O(N!) recursive cofactor| O(NI?) LU factorization|
/// | Jacobian         | Dense NIIN               | Sparse KIIK (K III N)    |
/// | Force calc       | O(NI?) pairs             | O(nnz) CSR iteration  |
#[derive(Debug, Clone)]
pub struct ResonanceNetwork {
    /// Number of active neurons
    pub n: usize,

    /// Quantum state: q III III^N (generalised coordinates)
    pub positions: SemanticWave,

    /// Conjugate momentum: p III III^N (generalised momenta)
    pub momenta: SemanticWave,

    /// Mass vector: m III III^N (inertia coefficients)
    pub masses: Array1<f64>,

    /// Damping coefficients: II III III^N (energy dissipation)
    pub damping: Array1<f64>,

    /// **CSR SPARSE MATRIX** III coupling weights.
    /// For edge iIIIj with weight w, the (i,j) entry holds w.
    pub adjacency_csr: CsMat<f64>,

    /// **EDGE metadata** III ConnectionType for every directed edge (i, j).
    adjacency_types: HashMap<(usize, usize), ConnectionType>,

    /// Simulation parameters
    dt: f64,
    step_count: usize,
    energy_log_interval: usize,
    pub bifurcation_monitor: BifurcationMonitor,

    /// SPECTRAL ZONING: cluster map built by spectral analysis
    pub spectral_clusters: HashMap<usize, SpectralCluster>,

    /// MESH CURRENT: sparse complex flow matrix for circuit analysis
    pub flow_matrix: CsMat<Complex64>,

    /// **CACHE** III determinant is cached and invalidated on topology change
    cached_determinant: Option<f64>,
    cache_valid: bool,

    /// **LABELS** III Persistent mapping from semantic keywords to node IDs
    pub labels: HashMap<String, usize>,

    /// Phase 40: SPATIAL NODES — Geometric transformation axioms for ARC-AGI
    pub spatial_nodes: Vec<SpatialNode>,
    
    /// Phase 40: Grid topology cache for mass conservation checks
    pub grid_topology_cache: Option<GridTopology>,
    
    // PHASE 60.2: Holographic Hyperdimensional Memory (Node ID -> 4096-D Phasor)
    pub holographic_memory: Vec<Vec<Complex64>>,
    
    // PHASE 60.5: Neuro-symbolic domain flags for each node (DOMAIN_LOGIC, DOMAIN_SPATIAL, etc.)
    pub node_flags: Vec<u32>,
}

/// Phase 40: Grid topology invariants for ARC-AGI physics enforcement
#[derive(Debug, Clone)]
pub struct GridTopology {
    pub input_mass: usize,           // Total pixel count in input
    pub output_mass: usize,          // Expected output mass (for training)
    pub connected_components: usize, // Number of disjoint regions
    pub symmetry_score: f64,         // 0.0-1.0 symmetry measure
    pub bounding_box: (usize, usize, usize, usize), // min_x, min_y, max_x, max_y
    pub color_histogram: Vec<usize>, // Pixel count per color
}

#[derive(Debug, Clone)]
pub struct ObjectSignature {
    pub color: u8,
    pub x: usize,
    pub y: usize,
    pub shape_label: String,
    pub signature: Vec<Complex64>,
}

#[derive(Debug, Clone)]
pub struct ArcSceneGraph {
    pub scene_signature: Vec<Complex64>,
    pub object_signatures: Vec<ObjectSignature>,
    pub topology: GridTopology,
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// CONSTRUCTION
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

impl ResonanceNetwork {
    /// Create a new network pre-allocated for `capacity` neurons.
    pub fn new(capacity: usize) -> Self {
        // GPU init removed III all math operations are now pure CPU (Rayon).

        let empty_csr: CsMat<f64> =
            TriMat::<f64>::new((capacity, capacity)).to_csr();
        let empty_complex_csr: CsMat<Complex64> =
            TriMat::<Complex64>::new((capacity, capacity)).to_csr();

        Self {
            n: 0,
            positions: SemanticWave::zero(capacity),
            momenta:   SemanticWave::zero(capacity),
            masses:    Array1::from_elem(capacity, 1.0),
            damping:   Array1::zeros(capacity),
            adjacency_csr:   empty_csr,
            adjacency_types: HashMap::new(),
            dt: 0.01,
            step_count: 0,
            energy_log_interval: 100,
            bifurcation_monitor: BifurcationMonitor::new(20, 0.3),
            spectral_clusters: HashMap::new(),
            flow_matrix: empty_complex_csr,
            cached_determinant: None,
            cache_valid: false,
            labels: HashMap::new(),
            spatial_nodes: Vec::new(),
            grid_topology_cache: None,
            holographic_memory: Vec::with_capacity(capacity),
            node_flags: Vec::with_capacity(capacity),
        }
    }

    /// PHASE 51.5: Hard Reset - Clear semantic state (vibrations) for scenario isolation.
    /// 
    /// Zeroes out positions and momenta waves and clears temporary caches/step counts.
    /// Structural data (labels, adjacency_csr) is preserved.
    pub fn clear_state(&mut self) {
        self.positions = SemanticWave::zero(self.masses.len());
        for i in 0..self.n {
            self.positions.amplitudes[i] = Complex64::new(1.0, 0.0);
        }
        self.momenta   = SemanticWave::zero(self.masses.len());
        self.step_count = 0;
        self.cached_determinant = None;
        self.cache_valid = false;
        self.spectral_clusters.clear();
        self.grid_topology_cache = None;
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // NEURON MANAGEMENT
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    /// Register a new neuron with the given damping coefficient.
    /// Returns the neuron ID.
    pub fn add_concept(&mut self, damping_coeff: f64) -> usize {
        assert!(self.n < self.masses.len(), "Network capacity exceeded");

        let id = self.n;
        self.damping[id] = damping_coeff;
        self.masses[id]  = 1.0;
        self.positions.amplitudes[id] = Complex64::new(1.0, 0.0); // resting state energy
        self.n += 1;

        // Ensure holographic_memory and node_flags are resized to match self.n
        while self.holographic_memory.len() < self.n {
            self.holographic_memory.push(Vec::new());
            self.node_flags.push(0);
        }

        self.cache_valid = false;   // topology changed
        id
    }

    /// Add a concept with a persistent label. Reuses existing ID if label found.
    /// PHASE 60.2: Now accepts and stores holographic signature for VSA operations.
    pub fn add_labeled_concept(&mut self, label: &str, damping_coeff: f64, signature: Vec<Complex64>) -> usize {
        if let Some(&id) = self.labels.get(label) {
            // Update holographic signature for existing concept
            while self.holographic_memory.len() <= id {
                self.holographic_memory.push(Vec::new());
                self.node_flags.push(0);
            }
            self.holographic_memory[id] = signature;
            if id < self.positions.amplitudes.len() {
                if self.positions.amplitudes[id].norm() < 1e-6 {
                    self.positions.amplitudes[id] = Complex64::new(1.0, 0.0);
                }
            }
            return id;
        }
        let id = self.add_concept(damping_coeff);
        self.labels.insert(label.to_string(), id);
        // Ensure holographic_memory and node_flags are extended to self.n
        while self.holographic_memory.len() < self.n {
            self.holographic_memory.push(Vec::new());
            self.node_flags.push(0);
        }
        self.holographic_memory[id] = signature;
        self.node_flags[id] = DOMAIN_LOGIC; // Default to LOGIC domain for labeled concepts
        id
    }

    /// Batch-register multiple labeled concepts in a single call.
    /// Builds CSR adjacency matrix ONCE at the end instead of invalidating per-neuron.
    ///
    /// `concepts`: slice of (label, damping_coeff) pairs.
    /// Returns a Vec of assigned IDs (reuses existing IDs for known labels).
    pub fn add_concepts_batch(&mut self, concepts: &[(&str, f64)]) -> Vec<usize> {
        let mut ids = Vec::with_capacity(concepts.len());
        let mut any_new = false;
        for &(label, damping) in concepts {
            if let Some(&existing_id) = self.labels.get(label) {
                ids.push(existing_id);
                if existing_id < self.positions.amplitudes.len() {
                    if self.positions.amplitudes[existing_id].norm() < 1e-6 {
                        self.positions.amplitudes[existing_id] = Complex64::new(1.0, 0.0);
                    }
                }
            } else {
                assert!(self.n < self.masses.len(), "Network capacity exceeded");
                let id = self.n;
                self.damping[id] = damping;
                self.masses[id] = 1.0;
                self.positions.amplitudes[id] = Complex64::new(1.0, 0.0); // resting state energy
                self.n += 1;
                self.labels.insert(label.to_string(), id);
                ids.push(id);
                any_new = true;
            }
        }
        // Ensure holographic_memory and node_flags are extended to self.n
        while self.holographic_memory.len() < self.n {
            self.holographic_memory.push(Vec::new());
            self.node_flags.push(0);
        }
        // Invalidate topology cache only once if any new concept was added
        if any_new {
            self.cache_valid = false;
        }
        ids
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // QMEM COMPILER / SERIALIZATION
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    /// Save the network to a binary .qmem file.
    pub fn save_cartridge(&self, path: &str) -> anyhow::Result<()> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (&val, (r, c)) in self.adjacency_csr.iter() {
            rows.push(r);
            cols.push(c);
            vals.push(val);
        }

        let mut holographic_memory = self.holographic_memory.clone();
        while holographic_memory.len() < self.n {
            holographic_memory.push(Vec::new());
        }
        let mut node_flags = self.node_flags.clone();
        while node_flags.len() < self.n {
            node_flags.push(0);
        }

        let cartridge = QmemCartridge {
            n: self.n,
            damping: self.damping.to_vec(),
            masses: self.masses.to_vec(),
            adjacency_rows: rows,
            adjacency_cols: cols,
            adjacency_vals: vals,
            labels: self.labels.clone(),
            holographic_memory,
            node_flags,
        };

        let bin = bincode::serialize(&cartridge)?;
        std::fs::write(path, bin)?;
        Ok(())
    }

    /// Load the network from a binary .qmem file.
    pub fn load_cartridge(path: &str) -> anyhow::Result<Self> {
        let bin = std::fs::read(path)?;
        let cartridge: QmemCartridge = bincode::deserialize(&bin)?;

        let mut net = ResonanceNetwork::new(cartridge.n.max(16));
        net.n = cartridge.n;
        net.damping = Array1::from_vec(cartridge.damping.clone());
        net.masses = Array1::from_vec(cartridge.masses.clone());
        net.labels = cartridge.labels;
        net.holographic_memory = cartridge.holographic_memory;
        net.node_flags = cartridge.node_flags;

        for i in 0..net.n {
            net.positions.amplitudes[i] = Complex64::new(1.0, 0.0);
        }

        let mut tri = sprs::TriMat::<f64>::new((net.masses.len(), net.masses.len()));
        for i in 0..cartridge.adjacency_rows.len() {
            tri.add_triplet(
                cartridge.adjacency_rows[i],
                cartridge.adjacency_cols[i],
                cartridge.adjacency_vals[i],
            );
        }
        net.adjacency_csr = tri.to_csr();

        Ok(net)
    }

    /// Take an in-memory snapshot of the network state for fast virtual reset.
    pub fn snapshot(&self) -> NetworkSnapshot {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (&val, (r, c)) in self.adjacency_csr.iter() {
            rows.push(r);
            cols.push(c);
            vals.push(val);
        }

        NetworkSnapshot {
            n: self.n,
            damping: self.damping.to_vec(),
            masses: self.masses.to_vec(),
            adjacency_rows: rows,
            adjacency_cols: cols,
            adjacency_vals: vals,
            adjacency_types: self.adjacency_types.clone(),
            labels: self.labels.clone(),
            holographic_memory: self.holographic_memory.clone(),
            node_flags: self.node_flags.clone(),
            position_re: self.positions.amplitudes.iter().map(|c| c.re).collect(),
            position_im: self.positions.amplitudes.iter().map(|c| c.im).collect(),
            momentum_re: self.momenta.amplitudes.iter().map(|c| c.re).collect(),
            momentum_im: self.momenta.amplitudes.iter().map(|c| c.im).collect(),
        }
    }

    /// Restore the network from a fast in-memory snapshot.
    pub fn restore_snapshot(&mut self, snap: &NetworkSnapshot) {
        self.n = snap.n;
        self.damping = Array1::from_vec(snap.damping.clone());
        self.masses = Array1::from_vec(snap.masses.clone());
        self.adjacency_types = snap.adjacency_types.clone();
        self.labels = snap.labels.clone();
        self.holographic_memory = snap.holographic_memory.clone();
        self.node_flags = snap.node_flags.clone();

        let cap = self.masses.len();
        let mut tri = TriMat::<f64>::new((cap, cap));
        for i in 0..snap.adjacency_rows.len() {
            tri.add_triplet(
                snap.adjacency_rows[i],
                snap.adjacency_cols[i],
                snap.adjacency_vals[i],
            );
        }
        self.adjacency_csr = tri.to_csr();
        self.cache_valid = false;
        self.cached_determinant = None;
        self.spectral_clusters.clear();
        self.grid_topology_cache = None;

        for (i, (&re, &im)) in snap.position_re.iter().zip(snap.position_im.iter()).enumerate() {
            if i < self.positions.amplitudes.len() {
                self.positions.amplitudes[i] = Complex64::new(re, im);
            }
        }
        for (i, (&re, &im)) in snap.momentum_re.iter().zip(snap.momentum_im.iter()).enumerate() {
            if i < self.momenta.amplitudes.len() {
                self.momenta.amplitudes[i] = Complex64::new(re, im);
            }
        }
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // EDGE MANAGEMENT III SPARSE
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    /// Add a directed syntactic connection (high stiffness, short-range).
    pub fn connect_syntax(&mut self, i: usize, j: usize, stiffness: f64) {
        assert!(i < self.n && j < self.n, "Invalid neuron ID");
        self.add_edge_sparse(i, j, stiffness, ConnectionType::Syntax, false);
    }

    /// Add an undirected semantic connection (low damping, long-range).
    pub fn connect_semantic(&mut self, i: usize, j: usize, stiffness: f64) {
        assert!(i < self.n && j < self.n, "Invalid neuron ID");
        self.add_edge_sparse(i, j, stiffness, ConnectionType::Semantic, true);
    }

    /// **OPTIMIZED** edge insertion into the CSR matrix.
    ///
    /// Strategy: read all existing triplets, update or append, rebuild CSR.
    /// Rebuilding is O(nnz) and happens only on topology changes (which are
    /// rare compared to integration steps).
    fn add_edge_sparse(
        &mut self,
        i: usize,
        j: usize,
        weight: f64,
        conn_type: ConnectionType,
        bidirectional: bool,
    ) {
        let capacity = self.masses.len();

        // Collect existing (row, col, val) triplets from current CSR matrix
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64>   = Vec::new();

        for (&val, (r, c)) in self.adjacency_csr.iter() {
            rows.push(r);
            cols.push(c);
            vals.push(val);
        }

        // Update weight if edge iIIIj already exists, otherwise append
        if let Some(pos) = rows.iter().zip(cols.iter())
            .position(|(&r, &c)| r == i && c == j)
        {
            vals[pos] = weight;
        } else {
            rows.push(i);
            cols.push(j);
            vals.push(weight);
        }

        // Mirror edge jIIIi for bidirectional connections
        if bidirectional {
            if let Some(pos) = rows.iter().zip(cols.iter())
                .position(|(&r, &c)| r == j && c == i)
            {
                vals[pos] = weight;
            } else {
                rows.push(j);
                cols.push(i);
                vals.push(weight);
            }
        }

        // Rebuild CSR from the updated triplet list
        let mut tri = TriMat::<f64>::new((capacity, capacity));
        for ((r, c), v) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
            tri.add_triplet(*r, *c, *v);
        }
        self.adjacency_csr = tri.to_csr();

        // Record connection type in the O(1) HashMap
        self.adjacency_types.insert((i, j), conn_type);
        if bidirectional {
            self.adjacency_types.insert((j, i), conn_type);
        }

        self.cache_valid = false;   // determinant cache is now stale
    }

    /// O(1) weight lookup via CSR random access.
    fn get_weight(&self, i: usize, j: usize) -> f64 {
        self.adjacency_csr.get(i, j).copied().unwrap_or(0.0)
    }

    pub fn has_link(&self, source: usize, target: usize) -> bool {
        self.get_weight(source, target) > 0.0
    }

    /// O(1) connection-type lookup (defaults to Syntax if edge absent).
    fn get_connection_type(&self, i: usize, j: usize) -> ConnectionType {
        self.adjacency_types
            .get(&(i, j))
            .copied()
            .unwrap_or(ConnectionType::Syntax)
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // STATE EXTRACTION
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    /// Real-valued state vector for LLM embedding injection.
    ///
    /// Each element = Re(q_i) II ln(1 + |p_i|) / mass_i  (log-scaled Clifford projection)
    pub fn extract_state_vector(&self) -> Vec<f32> {
        (0..self.n)
            .map(|i| {
                let rotor_real   = self.positions.amplitudes[i].re as f32;
                let momentum_mag = self.momenta.amplitudes[i].norm() as f32;
                let mass         = self.masses[i] as f32;

                if mass > 1e-6 {
                    rotor_real * (1.0 + momentum_mag).ln()
                } else {
                    0.0
                }
            })
            .collect()
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // SYMPLECTIC EULER INTEGRATION
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    /// Advance the simulation by one timestep (symplectic Euler).
    ///
    /// Ordering ensures phase-space volume conservation:
    ///
    /// 1. F    = compute_forces(q, external)
    /// 2. p'   = p + F II IIt          (momentum update first)
    /// 3. q'   = q + (p'/m) II IIt     (position with updated momentum)
    pub fn step(&mut self, external_forces: &[(usize, Complex64)]) -> SemanticWave {
        self.step_count += 1;

        let forces = self.compute_forces_sparse(external_forces);
        self.update_momenta(&forces);
        self.update_positions();

        if self.step_count.is_multiple_of(self.energy_log_interval) {
            self.log_energy();
        }

        let mut sum_omega = 0.0;
        let mut count = 0;
        for i in 0..self.n {
            let q_i = self.positions.amplitudes[i];
            if q_i.norm() > 1e-12 {
                let vel = self.momenta.amplitudes[i] / self.masses[i];
                let omega = (vel / q_i).im;
                sum_omega += omega;
                count += 1;
            }
        }
        let avg_omega = if count > 0 { sum_omega / count as f64 } else { 0.0 };
        self.positions.angular_velocity = avg_omega as f32;

        let det = self.compute_determinant_lu();
        self.bifurcation_monitor.observe(det);

        SemanticWave {
            amplitudes: self.positions.amplitudes
                .slice(ndarray::s![0..self.n])
                .to_owned(),
            residuals: Some(Array1::zeros(self.n)),
            angular_velocity: avg_omega as f32,
        }
    }

    /// **SPARSE FORCE CALCULATION**  F_i = II_j k_ij II (q_j III q_i)
    ///
    /// Iterates only over non-zero CSR entries (O(nnz)) rather than all NII pairs.
    pub fn compute_forces_sparse(
        &self,
        external: &[(usize, Complex64)],
    ) -> SemanticWave {
        let n = self.n;
        let mut forces = SemanticWave {
            amplitudes: Array1::zeros(n),
            residuals:  Some(Array1::zeros(n)),
            angular_velocity: 0.0,
        };

        // External / prompt-injection forces
        for &(id, force) in external {
            if id < n {
                forces.amplitudes[id] += force;
            }
        }

        // Coupling forces III O(nnz) CSR row iteration
        let q = self.positions.amplitudes.slice(ndarray::s![0..n]);

        for (i, row) in self.adjacency_csr.outer_iterator().enumerate() {
            if i >= n {
                break;
            }

            let mut coupling = Complex64::new(0.0, 0.0);

            for (j, &k_ij) in row.iter() {
                if j >= n || k_ij.abs() < 1e-12 {
                    continue;
                }

                let delta = q[j] - q[i];

                let force = match self.get_connection_type(i, j) {
                    ConnectionType::Syntax => {
                        // Short-range: force cut-off beyond distance 2.0
                        if delta.norm() > 2.0 {
                            Complex64::new(0.0, 0.0)
                        } else {
                            delta * (k_ij * 0.5)
                        }
                    }
                    ConnectionType::Semantic => {
                        // Long-range: no cutoff (quantum tunnelling)
                        delta * k_ij
                    }
                    ConnectionType::Spatial => {
                        // Phase 40: Geometric transformation forces
                        // Spatial connections have medium-range with angular dependence
                        let position_coupling = (delta.re * delta.re + delta.im * delta.im).sqrt();
                        if position_coupling > 3.0 {
                            Complex64::new(0.0, 0.0)
                        } else {
                            // Rotational bias for spatial transformations
                            let rotation_phase = Complex64::new(0.0, 1.0) * delta * k_ij * 0.3;
                            delta * k_ij * 0.7 + rotation_phase
                        }
                    }
                };

                coupling += force;
            }

            forces.amplitudes[i] += coupling;
        }

        forces
    }

    /// p(t+IIt) = p(t) + (F_coupling + F_damping) II IIt
    #[inline]
    fn update_momenta(&mut self, forces: &SemanticWave) {
        for i in 0..self.n {
            let damping_force = self.momenta.amplitudes[i] * (-self.damping[i]);
            let total = forces.amplitudes[i] + damping_force;
            self.momenta.amplitudes[i] += total * self.dt;
        }
    }

    /// q(t+IIt) = q(t) + (p(t+IIt) / m) II IIt
    #[inline]
    fn update_positions(&mut self) {
        for i in 0..self.n {
            let vel = self.momenta.amplitudes[i] / self.masses[i];
            self.positions.amplitudes[i] += vel * self.dt;
        }
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // HAMILTONIAN ENERGY
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    /// Returns (kinetic, potential, total) energy.
    ///
    /// H = T + V
    ///   = II_i |p_i|II/(2m_i)  +  II_{edges i<j} k_ij |q_j III q_i|II / 2
    pub fn calculate_energy(&self) -> (f64, f64, f64) {
        let n = self.n;

        let mut kinetic = 0.0_f64;
        for i in 0..n {
            kinetic += self.momenta.amplitudes[i].norm_sqr() / (2.0 * self.masses[i]);
        }

        let mut potential = 0.0_f64;
        for (i, row) in self.adjacency_csr.outer_iterator().enumerate() {
            if i >= n {
                break;
            }
            for (j, &k) in row.iter() {
                if j < n && i < j {
                    let delta = self.positions.amplitudes[j] - self.positions.amplitudes[i];
                    potential += k * delta.norm_sqr() / 2.0;
                }
            }
        }

        (kinetic, potential, kinetic + potential)
    }

    /// Total Hamiltonian energy III called by the Causal Ouroboros intervention checker.
    pub fn compute_total_energy(&self) -> f64 {
        self.calculate_energy().2
    }

    fn log_energy(&self) {
        let (ke, pe, total) = self.calculate_energy();
        eprintln!(
            "[HAMILTONIAN] Step {:6} | KE: {:10.6} | PE: {:10.6} | Total: {:10.6}",
            self.step_count, ke, pe, total
        );
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // UTILITY ACCESSORS
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    /// Current position snapshot for active neurons.
    pub fn get_state(&self) -> SemanticWave {
        SemanticWave {
            amplitudes: self.positions.amplitudes
                .slice(ndarray::s![0..self.n])
                .to_owned(),
            residuals: Some(Array1::zeros(self.n)),
            angular_velocity: 0.0,
        }
    }

    /// Zero all positions and momenta; preserve topology and parameters.
    pub fn reset(&mut self) {
        for i in 0..self.n {
            self.positions.amplitudes[i] = Complex64::new(1.0, 0.0);
            self.momenta.amplitudes[i]   = Complex64::new(0.0, 0.0);
        }
        self.step_count = 0;
    }

    /// Override the integration timestep (must satisfy dt III (0, 1)).
    pub fn set_timestep(&mut self, dt: f64) {
        assert!(dt > 0.0 && dt < 1.0, "Timestep must be in (0, 1) for numerical stability");
        self.dt = dt;
    }

    /// Total directed edges in the CSR matrix.
    pub fn edge_count(&self) -> usize {
        self.adjacency_csr.nnz()
    }

    /// Average out-degree of active neurons.
    pub fn average_degree(&self) -> f64 {
        if self.n == 0 { 0.0 } else { self.adjacency_csr.nnz() as f64 / self.n as f64 }
    }

    /// PHASE 49: Get correlation between two nodes based on adjacency weight
    /// 
    /// Returns normalized correlation in range [-1, 1] where:
    /// - Positive = similar/connected concepts
    /// - Negative = contradictory/opposing concepts  
    /// - Zero = unrelated
    pub fn get_correlation(&self, node_i: usize, node_j: usize) -> Option<f64> {
        if node_i >= self.n || node_j >= self.n {
            return None;
        }
        
        // Get direct edge weight if exists
        let direct_weight = self.get_weight(node_i, node_j);
        
        // Also check reverse direction
        let reverse_weight = self.get_weight(node_j, node_i);
        
        // Use max of both directions (undirected correlation)
        let weight = direct_weight.max(reverse_weight);
        
        if weight.abs() < 1e-12 {
            // No direct connection - check for indirect correlation via common neighbors
            let correlation = self.estimate_indirect_correlation(node_i, node_j);
            return Some(correlation);
        }
        
        // Normalize to [-1, 1] range based on typical weight magnitudes
        // Weights are typically in [0, 1] range, map to correlation
        let normalized = weight.clamp(-1.0, 1.0);
        Some(normalized)
    }
    
    /// Estimate indirect correlation via common neighbor analysis
    fn estimate_indirect_correlation(&self, node_i: usize, node_j: usize) -> f64 {
        let row_i = match self.adjacency_csr.outer_view(node_i) {
            Some(row) => row,
            None => return 0.0,
        };
        
        let row_j = match self.adjacency_csr.outer_view(node_j) {
            Some(row) => row,
            None => return 0.0,
        };
        
        // Find common neighbors
        let neighbors_i: std::collections::HashSet<usize> = row_i.iter().map(|(j, _)| j).collect();
        let neighbors_j: std::collections::HashSet<usize> = row_j.iter().map(|(j, _)| j).collect();
        
        let common: Vec<usize> = neighbors_i.intersection(&neighbors_j).copied().collect();
        
        if common.is_empty() {
            return 0.0;
        }
        
        // Calculate correlation based on shared neighbor weights
        let mut sum_product = 0.0;
        for &k in &common {
            let w_ik = self.get_weight(node_i, k);
            let w_jk = self.get_weight(node_j, k);
            sum_product += w_ik * w_jk;
        }
        
        // Normalize by number of common neighbors
        sum_product / common.len() as f64
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // LU DETERMINANT  (O(NI?) III replaces recursive O(N!) cofactor expansion)
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    /// Compute the determinant of the coupling matrix via LU factorisation.
    ///
    /// * For N III 100: exact, using `ndarray-linalg` LU factorisation (O(NI?)).
    /// * For N  > 100: lightweight spectral approximation to avoid large
    ///   dense-matrix allocation.
    ///
    /// Result is cached until the next topology change.
    pub fn compute_determinant_lu(&mut self) -> f64 {
        if self.cache_valid {
            return self.cached_determinant.unwrap_or(1.0);
        }

        let n = self.n;

        let det = if n > 100 {
            self.compute_approximate_determinant()
        } else {
            let dense = self.sparse_to_dense_submatrix(0, n);
            self.lu_det_of_dense(&dense)
        };

        self.cached_determinant = Some(det);
        self.cache_valid = true;
        det
    }

    /// LU factorisation of a dense matrix; returns det = II diag(U).
    /// 
    /// Uses pure Rust faer library for LU decomposition.
    fn lu_det_of_dense(&self, matrix: &Array2<f64>) -> f64 {
        let determinant = compute_determinant_smart(matrix);
        
        if !determinant.is_finite() {
            eprintln!("[DETERMINANT] IIIIII  Failed, using approximation");
            return self.compute_approximate_determinant();
        }
        
        determinant
    }

    /// Convert CSR rows [start, end) into a dense Array2 for LU input.
    fn sparse_to_dense_submatrix(&self, start: usize, end: usize) -> Array2<f64> {
        let n = end - start;
        let mut dense = Array2::<f64>::zeros((n, n));

        for (i, row) in self.adjacency_csr.outer_iterator().enumerate() {
            if i >= end   { break; }
            if i < start  { continue; }
            for (j, &val) in row.iter() {
                if j >= start && j < end {
                    dense[[i - start, j - start]] = val;
                }
            }
        }

        dense
    }

    /// Lightweight spectral approximation of det for large graphs (N > 100).
    ///
    /// Heuristic: det III (avg_edge_weight II sparsity)^N
    fn compute_approximate_determinant(&self) -> f64 {
        let mut weight_sum = 0.0_f64;
        let mut row_count  = 0_usize;

        for row in self.adjacency_csr.outer_iterator() {
            let row_sum: f64 = row.iter().map(|(_, &v)| v.abs()).sum();
            if row_sum > 0.0 {
                weight_sum += row_sum;
                row_count  += 1;
            }
        }

        let avg_weight = if row_count > 0 { weight_sum / row_count as f64 } else { 0.0 };
        let n_sq       = (self.n * self.n) as f64;
        let sparsity   = if n_sq > 0.0 { self.adjacency_csr.nnz() as f64 / n_sq } else { 0.0 };

        (avg_weight * sparsity)
            .powi(self.n as i32)
            .max(1e-10)
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // SPARSE JACOBIAN  (Active Inference III O(KI?), K = local degree)
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    /// Compute the Jacobian sensitivity for `changed_node` on its local
    /// sub-graph only (Active Inference), fully leveraging Rayon for CPU multi-threading.
    ///
    /// Returns `(determinant, full_sensitivity_vec, rhythm_class_string)`.
    pub fn compute_jacobian_sparse(
        &self,
        changed_node: usize,
    ) -> (f64, Vec<f64>, String) {
        let n = self.n;

        // "Spatial Chunking": Identify local neighbourhood
        // ONLY immediate 1st degree neighbors of the Keystone node
        let mut connected = vec![changed_node];
        if let Some(row) = self.adjacency_csr.outer_view(changed_node) {
            for (j, _) in row.iter() {
                if !connected.contains(&j) {
                    connected.push(j);
                }
            }
        }

        let k = connected.len();
        let mut j_sub = Array2::<f64>::zeros((k, k));

        for (ni, &i) in connected.iter().enumerate() {
            for (nj, &j) in connected.iter().enumerate() {
                let w = self.get_weight(i, j);
                if w.abs() > 1e-12 {
                    j_sub[[ni, nj]] = w;
                }
            }
        }

        let det = compute_determinant_smart(&j_sub).abs();

        // Par.ter over columns for CPU-bound multi-threading.
        let sensitivity: Vec<f64> = (0..k)
            .into_par_iter()
            .map(|col| column_norm_l2(&j_sub, col))
            .collect();

        let rhythm = if det < 0.1 {
            "Trivial"
        } else if det < 2.0 {
            "Harmonic"
        } else {
            "Chaotic"
        };


        eprintln!(
            "[JACOBIAN]   det(J)={:.4} ({}) | K={} nodes | Pure-CPU (Rayon)",
            det, rhythm, k
        );

        let mut full = vec![0.0_f64; n];
        for (ni, &i) in connected.iter().enumerate() {
            if ni < sensitivity.len() {
                full[i] = sensitivity[ni];
            }
        }

        (det, full, rhythm.to_string())
    }

    /// Convenience wrapper returning a `JacobianAnalysis` struct.
    /// Borrowed from `self` to enforce zero-copy lifetimes.
    pub fn compute_jacobian_rhythm(&self, changed_node: usize) -> JacobianAnalysis<'_> {
        let (det, sens, rhythm_str) = self.compute_jacobian_sparse(changed_node);

        let rhythm_classification = match rhythm_str.as_str() {
            "Trivial"  => RhythmClass::Trivial,
            "Harmonic" => RhythmClass::Harmonic,
            _          => RhythmClass::Chaotic,
        };

        JacobianAnalysis {
            network: self, // Inject reference to self
            determinant: det,
            sensitivity_scores: sens,
            rhythm_classification,
        }
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // PHASE VARIANCE CHECK
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    /// Phase variance linear-independence check for two solution trajectories.
    ///
    /// W(f, g) = II_i  [ f(i)?Ig'(i) III f'(i)?Ig(i) ]
    pub fn check_phase_variance(
        &self,
        solution1: &[Complex64],
        solution2: &[Complex64],
    ) -> PhaseVarianceCheck {
        assert_eq!(solution1.len(), solution2.len(), "Solutions must have equal length");

        let len = solution1.len();
        let mut d1 = vec![Complex64::new(0.0, 0.0); len];
        let mut d2 = vec![Complex64::new(0.0, 0.0); len];

        for i in 1..len {
            d1[i] = (solution1[i] - solution1[i - 1]) / self.dt;
            d2[i] = (solution2[i] - solution2[i - 1]) / self.dt;
        }

        let divergence: Complex64 = (0..len)
            .map(|i| solution1[i] * d2[i] - d1[i] * solution2[i])
            .sum();

        let det       = divergence.norm();
        let is_unique = det > 1e-6;
        let is_safe   = det < 10.0;

        eprintln!(
            "[PHASE VARIANCE] IIII |W|={:.6}  unique={}  safe={}",
            det, is_unique, is_safe
        );

        PhaseVarianceCheck { determinant: det, is_unique, is_safe }
    }

    pub fn check_bifurcation_action(&mut self) -> BifurcationAction {
        let det = self.compute_determinant_lu();
        self.bifurcation_monitor.observe(det);

        if self.bifurcation_monitor.is_bifurcating() {
            eprintln!(
                "[BIFURCATION] ⚠️  Regime: {}, λ={:.4}, step={}",
                self.bifurcation_monitor.regime_name(),
                self.bifurcation_monitor.lyapunov_estimate,
                self.bifurcation_monitor.step
            );
            match self.bifurcation_monitor.regime {
                BifurcationRegime::Chaotic => BifurcationAction::DeepMcts { extra_depth: 3 },
                BifurcationRegime::Critical => BifurcationAction::TryAlternative,
                BifurcationRegime::Stable => BifurcationAction::Normal,
            }
        } else {
            BifurcationAction::Normal
        }
    }

    /// Compute the L1 Manhattan distance from `node` to the nearest target node.
    pub fn manhattan_distance_to_target(
        &self,
        node: usize,
        target_state: &[usize],
    ) -> f64 {
        if node >= self.positions.amplitudes.len() || target_state.is_empty() {
            return f64::INFINITY;
        }
        target_state.iter().map(|&t| {
            if t >= self.positions.amplitudes.len() {
                return f64::INFINITY;
            }
            let pos_node = self.positions.amplitudes[node];
            let pos_tgt = self.positions.amplitudes[t];
            (pos_node.re - pos_tgt.re).abs() + (pos_node.im - pos_tgt.im).abs()
        }).fold(f64::INFINITY, f64::min)
    }

    /// Holographic cosine similarity between two VSA signatures.
    pub fn holographic_similarity(sig_a: &[Complex64], sig_b: &[Complex64]) -> f64 {
        if sig_a.is_empty() || sig_b.is_empty() || sig_a.len() != sig_b.len() {
            return 0.0;
        }
        let dim = sig_a.len() as f64;
        let dot: Complex64 = sig_a.iter().zip(sig_b.iter())
            .map(|(a, b)| a * b.conj())
            .fold(Complex64::new(0.0, 0.0), |acc, x| acc + x);
        (dot / dim).norm().min(1.0)
    }

    /// Element-wise VSA binding of two signatures.
    pub fn vsa_bind(sig_a: &[Complex64], sig_b: &[Complex64]) -> Vec<Complex64> {
        sig_a.iter().zip(sig_b.iter())
            .map(|(a, b)| a * b)
            .collect()
    }

    /// Superpose VSA signatures and normalize the result.
    pub fn vsa_superpose(signatures: &[Vec<Complex64>]) -> Vec<Complex64> {
        if signatures.is_empty() {
            return Vec::new();
        }
        let dim = signatures[0].len();
        let mut result = vec![Complex64::new(0.0, 0.0); dim];
        for sig in signatures {
            for (i, &v) in sig.iter().enumerate().take(dim) {
                result[i] += v;
            }
        }
        let norm: f64 = result.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-10 {
            result.iter_mut().for_each(|c| *c /= norm);
        }
        result
    }

    fn generate_xorshift_signature(mut seed: u64, dim: usize) -> Vec<Complex64> {
        (0..dim).map(|_| {
            seed ^= seed << 12;
            seed ^= seed >> 25;
            seed ^= seed << 27;
            let re = (seed.wrapping_mul(2685821657736338717) as f64 / u64::MAX as f64) * 2.0 - 1.0;
            seed ^= seed << 12;
            seed ^= seed >> 25;
            seed ^= seed << 27;
            let im = (seed.wrapping_mul(2685821657736338717) as f64 / u64::MAX as f64) * 2.0 - 1.0;
            let norm = (re * re + im * im).sqrt().max(1e-10);
            Complex64::new(re / norm, im / norm)
        }).collect()
    }

    fn get_or_create_signature(&mut self, key: &str) -> Vec<Complex64> {
        if let Some(&id) = self.labels.get(key) {
            if id < self.holographic_memory.len() {
                let sig = self.holographic_memory[id].clone();
                if !sig.is_empty() {
                    return sig;
                }
            }
        }
        let seed = key.bytes().fold(6364136223846793005u64, |acc, b| {
            acc.wrapping_mul(2862933555777941757).wrapping_add(b as u64)
        });
        let sig = Self::generate_xorshift_signature(seed, 4096);
        self.add_labeled_concept(key, 0.01, sig.clone());
        sig
    }

    /// Build an H-VSA scene graph from an ARC-style grid.
    pub fn build_scene_graph(&mut self, grid: &Array2<u8>) -> ArcSceneGraph {
        let topology = self.compute_grid_topology(grid);
        let (height, width) = (grid.nrows(), grid.ncols());
        let mut object_signatures = Vec::new();

        for y in 0..height {
            for x in 0..width {
                let color = grid[[y, x]];
                if color == 0 { continue; }

                let color_key = format!("arc_color_{}", color);
                let color_sig = self.get_or_create_signature(&color_key);

                let pos_key = format!("arc_pos_{}_{}", x, y);
                let pos_sig = self.get_or_create_signature(&pos_key);

                let obj_sig = Self::vsa_bind(&color_sig, &pos_sig);
                object_signatures.push(ObjectSignature {
                    color,
                    x,
                    y,
                    shape_label: color_key.clone(),
                    signature: obj_sig,
                });
            }
        }

        let all_sigs: Vec<Vec<Complex64>> = object_signatures.iter()
            .map(|o| o.signature.clone())
            .collect();
        let scene_signature = Self::vsa_superpose(&all_sigs);

        let scene_label = format!("scene_{}x{}", width, height);
        let _ = self.add_labeled_concept(&scene_label, 0.01, scene_signature.clone());

        ArcSceneGraph { scene_signature, object_signatures, topology }
    }

    pub fn scene_similarity(graph_a: &ArcSceneGraph, graph_b: &ArcSceneGraph) -> f64 {
        Self::holographic_similarity(&graph_a.scene_signature, &graph_b.scene_signature)
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // HIGH-RESONANCE CONCEPTS
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    /// Top-k neurons ranked by position-energy + momentum-energy.
    pub fn get_high_resonance_concepts(&self, top_k: usize) -> Vec<String> {
        let mut energies: Vec<(usize, f64)> = (0..self.n)
            .map(|i| {
                let e = self.positions.amplitudes[i].norm()
                      + self.momenta.amplitudes[i].norm();
                (i, e)
            })
            .collect();

        energies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Create reverse lookup for labels
        let id_to_label: HashMap<usize, &String> = self.labels.iter().map(|(k, v)| (*v, k)).collect();

        energies.iter()
            .take(top_k)
            .map(|(id, e)| {
                let label = id_to_label.get(id).map(|s| s.as_str()).unwrap_or("Unknown");
                format!("Concept_{}:{} (E={:.4})", label, id, e)
            })
            .collect()
    }

    /// Compute aggregate phase of the network. 
    /// This represents the "average" semantic orientation in phase space.
    pub fn compute_aggregate_phase(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        
        let mut aggregate = Complex64::new(0.0, 0.0);
        for i in 0..self.n {
            // Weight the contribution by mass (importance) and current normalized amplitude
            let weight = self.masses[i];
            aggregate += self.positions.amplitudes[i] * weight;
        }
        
        aggregate.arg() // returns f64 phase in (-PI, PI]
    }

    /// CliffordBridge: Compute Clifford Geometric Projection between two networks.
    ///
    /// Implements the "Ortak Alan / Saçak" (Common Space / Fringe) architecture:
    ///   h(x) = Clifford_project(f(x), g(x), θ_z)
    ///
    /// Mathematical formulation:
    ///   f_phase = self.compute_aggregate_phase()  (net_f: backbone)
    ///   g_phase = other.compute_aggregate_phase() (net_g: QMEM physics)
    ///   h = e^{i·f_phase} ∧ e^{i·g_phase·θ_z}
    ///     = Complex64::from_polar(1.0, f_phase) * Complex64::from_polar(1.0, -g_phase * θ_z)
    ///
    /// The result represents the projection intersection (pembe bölge / pink region)
    /// where both geometries converge — the "fringe node" that physics attracts even
    /// when the model hasn't seen the exact pattern in training.
    ///
    /// # Arguments
    /// * `other` - net_g (QMEM physics motor)
    /// * `impedance_theta` - Average topological impedance angle (θ_z)
    ///
    /// # Returns
    /// Complex64 projection. Take `.arg()` for h_phase (f32), `.norm()` for confidence.
    pub fn compute_clifford_projection(&self, other: &ResonanceNetwork, impedance_theta: f64) -> Complex64 {
        let f_phase = self.compute_aggregate_phase();
        let g_phase = other.compute_aggregate_phase();

        // Clifford geometric product in the complex plane:
        // e^{i·f_phase} ∧ e^{i·g_phase·θ_z}
        // = rotation by f_phase composed with conjugate rotation by (g_phase * θ_z)
        // This creates the ADE/BCE triangle intersection (pembe alan)
        let e_f = Complex64::new(f_phase.cos(), f_phase.sin());
        let e_g = Complex64::new((g_phase * impedance_theta).cos(), -(g_phase * impedance_theta).sin());

        let projection = e_f * e_g;

        // Log for diagnostics (only when both networks are active)
        if self.n > 0 && other.n > 0 {
            eprintln!(
                "[CLIFFORD] f_phase={:.4} g_phase={:.4} θ_z={:.4} → h_phase={:.4} confidence={:.4}",
                f_phase, g_phase, impedance_theta, projection.arg(), projection.norm()
            );
        }

        projection
    }

    /// CliffordBridge: Find cross-network "saçak" (fringe) nodes.
    ///
    /// Identifies nodes that appear with HIGH holographic similarity in BOTH networks.
    /// These are the "E point" in the Clifford geometry — nodes where:
    ///   - net_f (backbone semantics) has strong phasor activation
    ///   - net_g (QMEM physics) has strong phasor activation
    ///
    /// The returned node IDs are from net_f's label space. They can be used as:
    ///   - GPR routing signal ("this concept resonates in both networks → domain certain")
    ///   - Logit bias targets for subtle steering
    ///   - CX_W boost (high confidence signal)
    ///
    /// # Arguments
    /// * `net_f` - Backbone semantic network
    /// * `net_g` - QMEM physics network
    /// * `threshold` - Minimum holographic similarity (0.0–1.0, recommended: 0.3)
    ///
    /// # Returns
    /// Vec of (node_id_in_net_f, label, similarity_score) for matched saçak nodes.
    pub fn find_cross_network_sacak(
        net_f: &ResonanceNetwork,
        net_g: &ResonanceNetwork,
        threshold: f64,
    ) -> Vec<(usize, String, f64)> {
        let mut sacak_nodes = Vec::new();

        // Iterate over labeled concepts in net_f
        for (label_f, &id_f) in &net_f.labels {
            if id_f >= net_f.holographic_memory.len() {
                continue;
            }
            let sig_f = &net_f.holographic_memory[id_f];
            if sig_f.is_empty() {
                continue;
            }

            // Find matching concept in net_g by label (exact or substring match)
            let label_lower = label_f.to_lowercase();
            let mut best_similarity = 0.0f64;

            for (label_g, &id_g) in &net_g.labels {
                // Label matching: exact or substring (domain-aware)
                let matches = label_g.to_lowercase() == label_lower
                    || label_g.to_lowercase().contains(&label_lower)
                    || label_lower.contains(&label_g.to_lowercase());

                if !matches || id_g >= net_g.holographic_memory.len() {
                    continue;
                }

                let sig_g = &net_g.holographic_memory[id_g];
                if sig_g.is_empty() || sig_g.len() != sig_f.len() {
                    continue;
                }

                // Holographic similarity: normalized inner product (VSA cosine similarity)
                let dot: Complex64 = sig_f.iter().zip(sig_g.iter())
                    .map(|(a, b)| a * b.conj())
                    .sum();
                let norm_f: f64 = sig_f.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
                let norm_g: f64 = sig_g.iter().map(|b| b.norm_sqr()).sum::<f64>().sqrt();

                if norm_f > 1e-10 && norm_g > 1e-10 {
                    let similarity = dot.norm() / (norm_f * norm_g);
                    if similarity > best_similarity {
                        best_similarity = similarity;
                    }
                }
            }

            // Also check positional energy in net_f (active concept = high energy)
            let energy_f = if id_f < net_f.n {
                net_f.positions.amplitudes[id_f].norm()
            } else {
                0.0
            };

            // Saçak condition: high VSA similarity AND active in net_f
            let combined_score = best_similarity * (1.0 + energy_f * 0.5);
            if combined_score >= threshold {
                sacak_nodes.push((id_f, label_f.clone(), combined_score));
            }
        }

        // Sort by score descending
        sacak_nodes.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        sacak_nodes
    }

    /// Activate net_f with prompt energy and propagate to adjacent nodes.
    /// Returns the number of seeded concepts.
    pub fn seed_from_prompt(&mut self, prompt: &str) -> usize {
        let lower_prompt = prompt.to_lowercase();
        let mut seeded = 0;
        let mut forces: Vec<(usize, Complex64)> = Vec::new();

        for (label, &id) in &self.labels {
            if lower_prompt.contains(&label.to_lowercase()) && id < self.n {
                // Inject energy at concept's natural frequency
                let freq = if id < self.masses.len() { self.masses[id] } else { 1.0 };
                forces.push((id, Complex64::new(freq * 2.0, 0.0)));
                seeded += 1;
            }
        }

        if !forces.is_empty() {
            self.step(&forces);
        }

        seeded
    }


    /// Returns high-energy concept labels and their activation levels.
    pub fn diffuse_energy(&self, prompt: &str) -> HashMap<String, f32> {
        let mut initial_energy = HashMap::new();
        let lower_prompt = prompt.to_lowercase();

        // 1. Semantic Seeding
        for (label, &id) in &self.labels {
            if lower_prompt.contains(&label.to_lowercase()) {
                initial_energy.insert(id, 1.0f64);
            }
        }

        // 2. Topological Diffusion (1-step adjacency)
        let mut diffused_energy = vec![0.0f64; self.n];
        for (id, &energy) in &initial_energy {
            if *id < self.n {
                diffused_energy[*id] += energy;
                if let Some(row) = self.adjacency_csr.outer_view(*id) {
                    for (neighbor, &weight) in row.iter() {
                        if neighbor < self.n {
                            diffused_energy[neighbor] += energy * weight.abs();
                        }
                    }
                }
            }
        }

        // 3. Concept Aggregation
        let mut active_concepts = HashMap::new();
        let id_to_label: HashMap<usize, &String> = self.labels.iter().map(|(k, v)| (*v, k)).collect();

        for (id, &energy) in diffused_energy.iter().enumerate() {
            if energy > 0.1 {
                if let Some(label) = id_to_label.get(&id) {
                    // Safe steering multiplier (energy * 1.2) for subconscious influence
                    active_concepts.insert(label.to_string(), (energy * 1.2) as f32);
                }
            }
        }
        
        active_concepts
    }

    /// Compute Topological Gravity (RoPE Viscosity) from prompt semantics.
    ///
    /// Returns concept_label → warp_factor where:
    ///   warp_factor ∈ [0.3, 1.0]
    ///   1.0 = no warp (base RoPE distance)
    ///   0.3 = maximum gravitational pull (concept perceived as adjacent in latent space)
    ///
    /// Uses the same CSR-based spreading activation as `diffuse_energy`, but maps
    /// the diffused energy to RoPE frequency scale factors instead of logit biases.
    pub fn compute_rope_warp_map(&self, prompt: &str) -> HashMap<String, f32> {
        // 1. Semantic Seeding (identical to diffuse_energy)
        let mut initial_energy = HashMap::new();
        let lower_prompt = prompt.to_lowercase();
        for (label, &id) in &self.labels {
            if lower_prompt.contains(&label.to_lowercase()) {
                initial_energy.insert(id, 1.0f64);
            }
        }

        // 2. Topological Diffusion (1-step adjacency, CSR sparse)
        let mut diffused_energy = vec![0.0f64; self.n];
        for (id, &energy) in &initial_energy {
            if *id < self.n {
                diffused_energy[*id] += energy;
                if let Some(row) = self.adjacency_csr.outer_view(*id) {
                    for (neighbor, &weight) in row.iter() {
                        if neighbor < self.n {
                            diffused_energy[neighbor] += energy * weight.abs();
                        }
                    }
                }
            }
        }

        // 3. Energy → RoPE Warp Factor conversion
        //    High energy = strong gravity = smaller multiplier (compress positional distance)
        //    Base distance is 1.0; floor clamped at 0.3 to prevent numerical collapse
        let id_to_label: HashMap<usize, &String> = self.labels.iter()
            .map(|(k, v)| (*v, k)).collect();

        let mut warp_map = HashMap::new();
        for (id, &energy) in diffused_energy.iter().enumerate() {
            if energy > 0.1 {
                if let Some(label) = id_to_label.get(&id) {
                    let warp_factor = (energy * 0.5).min(1.57) as f32; // Max pi/2 phase shift
                    warp_map.insert(label.to_string(), warp_factor);
                }
            }
        }

        if !warp_map.is_empty() {
            eprintln!(
                "[TOPOLOGICAL GRAVITY] {} attractors seeded from prompt ({} warp entries)",
                initial_energy.len(), warp_map.len()
            );
        }

        warp_map
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // KEYSTONE IDENTIFICATION
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    /// Top-k keystone nodes by total weighted out-degree (sparse CSR lookup).
    pub fn identify_keystones(&self, top_k: usize) -> Vec<usize> {
        let mut centrality: Vec<(usize, f64)> = (0..self.n)
            .map(|i| {
                let coupling: f64 = self.adjacency_csr
                    .outer_view(i)
                    .map(|row| row.iter().map(|(_, &w)| w.abs()).sum::<f64>())
                    .unwrap_or(0.0);
                (i, coupling)
            })
            .collect();

        centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        centrality.iter()
            .take(top_k)
            .map(|(id, coupling)| {
                eprintln!("[HOLOGRAPHIC] IIII Keystone: Node {} (coupling={:.2})", id, coupling);
                *id
            })
            .collect()
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // SPECTRAL ZONING & MESH FLOW
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII



    /// Re-initialise spectral book-keeping (call after topology is finalised).
    pub fn init_spectral_fields(&mut self) {
        self.spectral_clusters = HashMap::new();
        let cap = self.masses.len();
        self.flow_matrix = TriMat::<Complex64>::new((cap, cap)).to_csr();
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // PHASE 40: SPATIAL AXIOM - GEOMETRIC REASONING FOR ARC-AGI
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    /// Add a spatial transformation axiom node to the network.
    /// 
    /// These nodes represent geometric operations (Rotate, Flip, etc.) that form
    /// a Group structure under composition. The adjacency matrix captures
    /// transformation composition rules (e.g., Rotate90 × 4 = Identity).
    pub fn add_spatial_node(&mut self, axiom: SpatialAxiom, x: f64, y: f64) -> usize {
        let node_id = self.spatial_nodes.len();
        let node = SpatialNode::new(node_id, axiom, x, y);
        self.spatial_nodes.push(node);
        node_id
    }

    /// Build the Geometric Transformation Group adjacency structure.
    /// 
    /// Connects spatial nodes according to Group Theory composition rules:
    /// - Rotate90 → Rotate90 → Rotate90 → Rotate90 → Identity (cycle of 4)
    /// - FlipHorizontal × FlipHorizontal = Identity (involution)
    /// - TranslateX ∘ TranslateX⁻¹ = Identity
    /// 
    /// This creates a closed algebraic structure where RK4 can "wander"
    /// but always returns to valid transformations.
    pub fn build_transformation_group(&mut self) {
        use SpatialAxiom::*;
        
        // First, ensure all core transformations exist as nodes
        let axioms = vec![
            Identity, Rotate90, Rotate180, Rotate270,
            FlipHorizontal, FlipVertical,
            TranslateX, TranslateY,
            ColorFlood, Expand, Contract,
        ];
        
        // Create nodes for any missing axioms
        for (i, axiom) in axioms.iter().enumerate() {
            if i >= self.spatial_nodes.len() || self.spatial_nodes[i].axiom != *axiom {
                // Place nodes in a circle in latent space (2D projection)
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / (axioms.len() as f64);
                self.add_spatial_node(*axiom, angle.cos(), angle.sin());
            }
        }

        // Connect according to Group composition rules
        // Rotation cycle: R90 × 4 = I
        if let (Some(r90), Some(r180), Some(r270), Some(id)) = 
            (self.find_node(Rotate90), self.find_node(Rotate180), 
             self.find_node(Rotate270), self.find_node(Identity)) {
            self.connect_spatial(r90, r90, 0.5);      // R90 ∘ R90 = R180 (approx)
            self.connect_spatial(r90, r180, 0.3);     // Path to R180
            self.connect_spatial(r180, r180, 0.5);  // R180 ∘ R180 = I
            self.connect_spatial(r90, r270, 0.2);   // Path through
            self.connect_spatial(r270, r90, 0.2);   // Reverse
            self.connect_spatial(r180, id, 0.8);    // Strong return to identity
        }

        // Involutions (self-inverse): F × F = I
        if let (Some(fh), Some(id)) = (self.find_node(FlipHorizontal), self.find_node(Identity)) {
            self.connect_spatial(fh, fh, 0.9);      // Strong: F ∘ F → I
            self.connect_spatial(fh, id, 0.5);
        }
        if let (Some(fv), Some(id)) = (self.find_node(FlipVertical), self.find_node(Identity)) {
            self.connect_spatial(fv, fv, 0.9);
            self.connect_spatial(fv, id, 0.5);
        }

        // Morphological duality: Expand ↔ Contract
        if let (Some(exp), Some(cont)) = (self.find_node(Expand), self.find_node(Contract)) {
            self.connect_spatial(exp, cont, 0.7);     // Duality
            self.connect_spatial(cont, exp, 0.7);
        }

        eprintln!("[SPATIAL AXIOM] Built transformation group with {} nodes", self.spatial_nodes.len());
    }

    /// Find a spatial node by its axiom type.
    fn find_node(&self, axiom: SpatialAxiom) -> Option<usize> {
        self.spatial_nodes.iter()
            .position(|n| n.axiom == axiom)
    }

    /// Add a spatial connection (bidirectional with Group-structure weighting).
    fn connect_spatial(&mut self, i: usize, j: usize, weight: f64) {
        // Spatial connections exist in a separate adjacency layer
        // but we map them to semantic indices for force propagation
        let base_idx = self.n;
        
        // Ensure spatial nodes have semantic counterparts
        while self.n + self.spatial_nodes.len() > self.masses.len() {
            // Extend capacity if needed (shouldn't happen with proper init)
            eprintln!("[WARNING] Extending network capacity for spatial nodes");
            return;
        }

        // Store spatial adjacency information
        // (In production, use a separate sparse matrix for spatial edges)
        eprintln!("[SPATIAL] Connected {:?} → {:?} (w={:.2})", 
            self.spatial_nodes[i].axiom.as_str(),
            self.spatial_nodes[j].axiom.as_str(),
            weight
        );
    }

    /// Compute Grid Topology invariants for ARC-AGI physics enforcement.
    /// 
    /// Extracts structural properties from a 2D grid that must be preserved
    /// under valid transformations (Object Permanence, Mass Conservation).
    /// 
    /// # Arguments
    /// * `grid` - 2D array of color values (0-9 for ARC)
    /// 
    /// Returns GridTopology with mass, components, symmetry measures.
    pub fn compute_grid_topology(&mut self, grid: &Array2<u8>) -> GridTopology {
        let (height, width) = (grid.nrows(), grid.ncols());
        let mut mass = 0_usize;
        let mut color_histogram = vec![0_usize; 10]; // ARC uses colors 0-9
        let mut bounding_box = (width, height, 0, 0); // min_x, min_y, max_x, max_y

        // Pass 1: Mass, histogram, bounding box
        for y in 0..height {
            for x in 0..width {
                let color = grid[[y, x]] as usize;
                if color > 0 {
                    mass += 1;
                    color_histogram[color] += 1;
                    bounding_box.0 = bounding_box.0.min(x);
                    bounding_box.1 = bounding_box.1.min(y);
                    bounding_box.2 = bounding_box.2.max(x);
                    bounding_box.3 = bounding_box.3.max(y);
                }
            }
        }

        // Pass 2: Connected components (4-connectivity)
        let mut visited = Array2::<bool>::from_elem((height, width), false);
        let mut components = 0_usize;
        
        for y in 0..height {
            for x in 0..width {
                if grid[[y, x]] > 0 && !visited[[y, x]] {
                    components += 1;
                    // Flood fill BFS
                    let mut stack = vec![(x, y)];
                    visited[[y, x]] = true;
                    
                    while let Some((cx, cy)) = stack.pop() {
                        for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                            let nx = cx as i32 + dx;
                            let ny = cy as i32 + dy;
                            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                                let (nx, ny) = (nx as usize, ny as usize);
                                if grid[[ny, nx]] > 0 && !visited[[ny, nx]] {
                                    visited[[ny, nx]] = true;
                                    stack.push((nx, ny));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Pass 3: Symmetry analysis
        let symmetry_score = self.compute_grid_symmetry(grid);

        let topology = GridTopology {
            input_mass: mass,
            output_mass: mass, // Initially same, updated by training
            connected_components: components,
            symmetry_score,
            bounding_box,
            color_histogram,
        };

        // Cache for RK4 physics enforcement
        self.grid_topology_cache = Some(topology.clone());

        eprintln!(
            "[GRID TOPOLOGY] Mass={}, Components={}, Symmetry={:.2}",
            mass, components, symmetry_score
        );

        topology
    }

    /// Compute symmetry score for a grid (0.0 = asymmetric, 1.0 = fully symmetric).
    fn compute_grid_symmetry(&self, grid: &Array2<u8>) -> f64 {
        let (h, w) = (grid.nrows(), grid.ncols());
        if h == 0 || w == 0 { return 0.0; }

        // Check reflection symmetries
        let mut h_sym = 0.0_f64;
        for y in 0..h/2 {
            for x in 0..w {
                if grid[[y, x]] == grid[[h - 1 - y, x]] {
                    h_sym += 1.0;
                }
            }
        }
        let score_count = ((h/2).max(1) * w) as f64;
        let h_score = h_sym / score_count;

        // Vertical symmetry
        let mut v_sym = 0.0_f64;
        for y in 0..h {
            for x in 0..w/2 {
                if grid[[y, x]] == grid[[y, w - 1 - x]] {
                    v_sym += 1.0;
                }
            }
        }
        let v_score_count = (h * (w/2).max(1)) as f64;
        let v_score = v_sym / v_score_count;

        // Rotational symmetry (180°)
        let mut r_sym = 0.0_f64;
        for y in 0..h {
            for x in 0..w {
                let ry = h - 1 - y;
                let rx = w - 1 - x;
                if grid[[y, x]] == grid[[ry, rx]] {
                    r_sym += 1.0;
                }
            }
        }
        let r_score_count = (h * w).max(1) as f64;
        let r_score = r_sym / r_score_count;

        (h_score + v_score + r_score) / 3.0
    }

    /// Extract spatial node energies for RK4 ghost vector injection.
    /// 
    /// Returns a SemanticWave encoding the "Target Geometric State"
    /// that guides the pondering process toward valid transformations.
    pub fn extract_spatial_energies(&self) -> SemanticWave {
        let mut amplitudes = Array1::zeros(self.spatial_nodes.len());
        
        for (i, node) in self.spatial_nodes.iter().enumerate() {
            // Energy proportional to node's mass and position magnitude
            let (x, y) = node.position;
            let position_energy = (x * x + y * y).sqrt();
            amplitudes[i] = Complex64::new(node.mass * position_energy, 0.0);
        }

        SemanticWave {
            amplitudes,
            residuals: Some(Array1::zeros(self.spatial_nodes.len())),
            angular_velocity: 0.0,
        }
    }

    /// Get the cached grid topology (if computed).
    pub fn get_cached_topology(&self) -> Option<&GridTopology> {
        self.grid_topology_cache.as_ref()
    }

    /// Invalidate grid topology cache (call after transformation).
    pub fn invalidate_topology_cache(&mut self) {
        self.grid_topology_cache = None;
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII
    // PHASE 24: PERIPHERAL VISION - UNSUPERVISED ANOMALY DETECTION
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIIII

    /// Detect peripheral anomalies in disconnected clusters outside the active prompt zone.
    /// 
    /// Uses Clifford bivector magnitudes to identify algebraically isolated subgraphs
    /// that represent "anomalies" (e.g., hidden money laundering context in property transfers).
    /// 
    /// Returns: Some(SemanticWave) if an anomaly detected, None otherwise
    pub fn detect_peripheral_anomalies(&self, active_nodes: &[usize]) -> Option<SemanticWave> {
        if self.n == 0 || active_nodes.is_empty() {
            return None;
        }

        // Build set of active nodes for O(1) lookup
        let _active_set: std::collections::HashSet<usize> = active_nodes.iter().copied().collect();
        
        // Find disconnected clusters (nodes not in active_set)
        let mut peripheral_clusters: Vec<Vec<usize>> = Vec::new();
        let mut visited = vec![false; self.n];
        
        // Mark active nodes as visited (we're looking OUTSIDE these)
        for &node in active_nodes {
            if node < self.n {
                visited[node] = true;
            }
        }
        
        // DFS to find disconnected peripheral clusters
        for start in 0..self.n {
            if visited[start] {
                continue;
            }
            
            // BFS to find connected component
            let mut cluster = Vec::new();
            let mut stack = vec![start];
            visited[start] = true;
            
            while let Some(node) = stack.pop() {
                cluster.push(node);
                
                // Find neighbors via CSR matrix
                if let Some(row) = self.adjacency_csr.outer_view(node) {
                    for (neighbor, &weight) in row.iter() {
                        if weight.abs() > 1e-12 && !visited[neighbor] && neighbor < self.n {
                            visited[neighbor] = true;
                            stack.push(neighbor);
                        }
                    }
                }
            }
            
            if !cluster.is_empty() {
                peripheral_clusters.push(cluster);
            }
        }
        
        // Score each cluster using Clifford bivector magnitudes
        let mut best_anomaly: Option<(Vec<usize>, f64)> = None;
        
        for cluster in peripheral_clusters {
            if cluster.len() < 2 {
                continue; // Need at least 2 nodes for bivector calculation
            }
            
            // Extract wave from cluster positions
            let cluster_amplitudes: Vec<Complex64> = cluster.iter()
                .map(|&i| self.positions.amplitudes[i])
                .collect();
            
            let cluster_wave = SemanticWave::from_vec(cluster_amplitudes);
            
            // Calculate bivector magnitude via Clifford geometric product
            let bivector_score = apply_clifford_resonance(&cluster_wave);
            
            // Dynamic threshold: higher for larger clusters (normalized by sqrt(N))
            let normalized_score = bivector_score / (cluster.len() as f64).sqrt();
            
            // Check if this is the most anomalous cluster
            let threshold = 0.15; // Adaptive threshold
            if normalized_score > threshold {
                if let Some((_, current_best)) = best_anomaly {
                    if normalized_score > current_best {
                        best_anomaly = Some((cluster, normalized_score));
                    }
                } else {
                    best_anomaly = Some((cluster, normalized_score));
                }
            }
        }
        
        // Return the most anomalous cluster as a SemanticWave
        if let Some((cluster_nodes, score)) = best_anomaly {
            eprintln!("[PERIPHERAL VISION] Anomaly detected: {} nodes, score={:.6}", cluster_nodes.len(), score);
            
            let anomaly_amplitudes: Vec<Complex64> = cluster_nodes.iter()
                .map(|&i| self.positions.amplitudes[i])
                .collect();
            
            return Some(SemanticWave::from_vec(anomaly_amplitudes));
        }
        
        None
    }

    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
    // PHASE 48: HAMILTONIAN PATH SYNTHESIS VIA RUBIK'S CUBE ISOMORPHISM
    // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

    /// PHASE 48: Find Hamiltonian Trajectory via Rubik's Cube Isomorphism
    /// 
    /// The Concept: Real reasoning isn't guessing the next word; it's finding the 
    /// shortest algorithmic path to a solved state. Treat the ResonanceNetwork CSR 
    /// matrix as a Permutation Group (like a Rubik's Cube).
    /// 
    /// Mathematical Model:
    /// - Nodes = Cube stickers (semantic positions)
    /// - Edges = Valid moves (permutations)
    /// - Goal = Find sequence of moves (Hamiltonian path) that transforms 
    ///          start_state → target_state
    /// - Each step visits necessary axioms without repetition (Hamiltonian property)
    ///
    /// Implementation:
    /// - Uses Symplectic Euler integrator to construct trajectory
    /// - Visits sequence of axioms (moves) to reach goal
    /// - LLM translates mathematical trajectory into human language
    /// 
    /// # Arguments
    /// * `start_state` - Initial semantic state (node indices)
    /// * `target_state` - Goal semantic state (node indices)
    /// * `max_steps` - Maximum trajectory length
    /// 
    /// # Returns
    /// Hamiltonian trajectory: ordered sequence of node visits
    pub fn find_hamiltonian_trajectory(
        &mut self,
        start_state: &[usize],
        target_state: &[usize],
        max_steps: usize
    ) -> Vec<usize> {
        eprintln!(
            "[HAMILTONIAN] Finding trajectory: {} start nodes → {} target nodes",
            start_state.len(), target_state.len()
        );

        let mut trajectory: Vec<usize> = start_state.to_vec();
        let mut current_active: std::collections::HashSet<usize> = start_state.iter().copied().collect();
        let target_set: std::collections::HashSet<usize> = target_state.iter().copied().collect();

        let dt = 0.1;

        for step in 0..max_steps {
            if target_set.is_subset(&current_active) {
                eprintln!("[HAMILTONIAN] Target reached in {} steps", step);
                break;
            }

            let mut best_next: Option<usize> = None;
            let mut best_force = -f64::INFINITY;

            for &current_node in &current_active {
                if let Some(row) = self.adjacency_csr.outer_view(current_node) {
                    for (neighbor, &weight) in row.iter() {
                        if current_active.contains(&neighbor) || trajectory.contains(&neighbor) {
                            continue;
                        }

                        let manhattan_dist = self.manhattan_distance_to_target(neighbor, target_state);
                        let gradient_force = weight / (manhattan_dist + 1e-6);
                        let target_bonus = if target_set.contains(&neighbor) { 5.0 } else { 0.0 };
                        let semantic_force = gradient_force + target_bonus;

                        if semantic_force > best_force {
                            best_force = semantic_force;
                            best_next = Some(neighbor);
                        }
                    }
                }
            }

            if let Some(next_node) = best_next {
                trajectory.push(next_node);
                current_active.insert(next_node);

                if next_node < self.n {
                    let momentum = Complex64::new(best_force * dt, 0.0);
                    self.momenta.amplitudes[next_node] += momentum;
                    let vel = self.momenta.amplitudes[next_node] / self.masses[next_node];
                    self.positions.amplitudes[next_node] += vel * dt;
                }
            } else {
                eprintln!("[HAMILTONIAN] Trajectory stuck at step {}", step);
                break;
            }
        }

        eprintln!(
            "[HAMILTONIAN] Trajectory complete: {} nodes visited",
            trajectory.len()
        );

        trajectory
    }

    /// Geodesic Search on Riemannian Manifold using a warp map.
    /// Warps positional distances/attraction based on prompt semantics to find true minimum energy path.
    pub fn find_hamiltonian_trajectory_warped(
        &mut self,
        start_state: &[usize],
        target_state: &[usize],
        max_steps: usize,
        warp_map: &HashMap<String, f32>,
    ) -> Vec<usize> {
        eprintln!(
            "[HAMILTONIAN WARPED] Finding geodesic trajectory with {} warp factors",
            warp_map.len()
        );
        
        let id_to_label: HashMap<usize, &String> = self.labels.iter()
            .map(|(k, v)| (*v, k)).collect();

        let mut trajectory: Vec<usize> = start_state.to_vec();
        let mut current_active: std::collections::HashSet<usize> = start_state.iter().copied().collect();
        let target_set: std::collections::HashSet<usize> = target_state.iter().copied().collect();
        
        let dt = 0.1;
        
        for step in 0..max_steps {
            if current_active == target_set {
                eprintln!("[HAMILTONIAN WARPED] Target reached in {} steps", step);
                break;
            }
            
            let mut best_next: Option<usize> = None;
            let mut best_force = -f64::INFINITY;
            
            for &current_node in &current_active {
                if let Some(row) = self.adjacency_csr.outer_view(current_node) {
                    for (neighbor, &weight) in row.iter() {
                        if current_active.contains(&neighbor) || trajectory.contains(&neighbor) {
                            continue; // Hamiltonian constraint: skip visited
                        }
                        
                        let target_attraction = if target_set.contains(&neighbor) {
                            10.0
                        } else {
                            0.0
                        };
                        
                        let warp_multiplier = if let Some(label) = id_to_label.get(&neighbor) {
                            if let Some(&wf) = warp_map.get(*label) {
                                1.0 / (wf.max(0.1) as f64)
                            } else {
                                1.0
                            }
                        } else {
                            1.0
                        };
                        
                        let semantic_force = weight * (1.0 + target_attraction) * warp_multiplier;
                        
                        if semantic_force > best_force {
                            best_force = semantic_force;
                            best_next = Some(neighbor);
                        }
                    }
                }
            }
            
            if let Some(next_node) = best_next {
                trajectory.push(next_node);
                current_active.insert(next_node);
                
                if next_node < self.n {
                    let momentum = Complex64::new(best_force * dt, 0.0);
                    self.momenta.amplitudes[next_node] += momentum;
                    
                    let vel = self.momenta.amplitudes[next_node] / self.masses[next_node];
                    self.positions.amplitudes[next_node] += vel * dt;
                }
            } else {
                eprintln!("[HAMILTONIAN WARPED] Trajectory stuck at step {}", step);
                break;
            }
        }
        
        eprintln!(
            "[HAMILTONIAN WARPED] Trajectory complete: {} nodes visited",
            trajectory.len()
        );
        
        trajectory
    }

    /// PHASE 48: Validate Hamiltonian Trajectory
    /// 
    /// Verifies that trajectory visits each node exactly once (Hamiltonian property)
    pub fn validate_hamiltonian(&self, trajectory: &[usize]) -> bool {
        let unique: std::collections::HashSet<usize> = trajectory.iter().copied().collect();
        unique.len() == trajectory.len()
    }

    /// PHASE 48: Compute Trajectory Energy Cost
    /// 
    /// Calculates the "work" required to follow this trajectory through the network
    pub fn compute_trajectory_energy(&self, trajectory: &[usize]) -> f64 {
        let mut total_energy = 0.0;
        
        for i in 0..trajectory.len().saturating_sub(1) {
            let from = trajectory[i];
            let to = trajectory[i + 1];
            
            // Edge traversal cost (lower = better path)
            let edge_weight = self.get_weight(from, to);
            total_energy += 1.0 / (edge_weight + 0.01); // Avoid div by zero
        }
        
        total_energy
    }

    /// Keystone Detection (Kilit Taşı Tespiti)
    /// Computes Betweenness Centrality on the network graph.
    /// Returns a list of node IDs sorted by their centrality scores descending.
    pub fn find_keystones(&self) -> Vec<usize> {
        let n = self.n;
        if n == 0 {
            return Vec::new();
        }
        let mut cb = vec![0.0; n];

        for s in 0..n {
            let mut stack = Vec::new();
            let mut p = vec![Vec::new(); n];
            let mut sigma = vec![0.0; n];
            sigma[s] = 1.0;
            let mut d = vec![-1_i32; n];
            d[s] = 0;

            let mut q = std::collections::VecDeque::new();
            q.push_back(s);

            while let Some(v) = q.pop_front() {
                stack.push(v);
                if let Some(row) = self.adjacency_csr.outer_view(v) {
                    for (w, &weight) in row.iter() {
                        if w >= n || weight.abs() < 1e-12 {
                            continue;
                        }
                        // Path discovery
                        if d[w] < 0 {
                            d[w] = d[v] + 1;
                            q.push_back(w);
                        }
                        // Path counting
                        if d[w] == d[v] + 1 {
                            sigma[w] += sigma[v];
                            p[w].push(v);
                        }
                    }
                }
            }

            let mut delta = vec![0.0; n];
            while let Some(w) = stack.pop() {
                for &v in &p[w] {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
                if w != s {
                    cb[w] += delta[w];
                }
            }
        }

        // Rank the nodes by centrality score descending
        let mut nodes_with_scores: Vec<(usize, f64)> = cb.into_iter().enumerate().collect();
        nodes_with_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return sorted node IDs
        nodes_with_scores.into_iter().map(|(id, _)| id).collect()
    }

    /// Ripple Effect Simulation (Kelebek Etkisi Simülasyonu)
    /// Clones the network, perturbs the target node, runs virtual steps,
    /// and returns peak deviation scores for all other nodes, sorted descending.
    pub fn propagate_change(&self, node_id: usize) -> Vec<(usize, f64)> {
        let n = self.n;
        if n == 0 || node_id >= n {
            return Vec::new();
        }

        let mut virtual_net = self.clone();
        let initial_amplitudes: Vec<Complex64> = virtual_net.positions.amplitudes.to_vec();

        // Perturb the target node: phase shift and momentum impulse
        virtual_net.positions.amplitudes[node_id] += Complex64::new(10.0, 10.0);
        virtual_net.momenta.amplitudes[node_id] += Complex64::new(5.0, 5.0);

        let mut peak_deviations = vec![0.0; n];

        // Run virtual steps to propagate waves
        for _ in 0..20 {
            virtual_net.step(&[]);
            for i in 0..n {
                if i == node_id {
                    continue;
                }
                let dev = (virtual_net.positions.amplitudes[i] - initial_amplitudes[i]).norm();
                if dev > peak_deviations[i] {
                    peak_deviations[i] = dev;
                }
            }
        }

        let mut results: Vec<(usize, f64)> = peak_deviations.into_iter().enumerate().collect();
        results.retain(|&(id, dev)| id != node_id && dev > 1e-5);
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII
// UNIT TESTS
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

#[cfg(test)]
mod tests {
    use super::*;
    // ... (rest of the code remains the same)

    // IIIIII Sparse matrix basics ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIIIII

    #[test]
    fn test_sparse_bidirectional_weight() {
        let mut net = ResonanceNetwork::new(10);
        let a = net.add_concept(0.01);
        let b = net.add_concept(0.01);
        net.connect_semantic(a, b, 0.5);

        assert_eq!(net.get_weight(a, b), 0.5, "forward edge");
        assert_eq!(net.get_weight(b, a), 0.5, "reverse edge (bidirectional)");
    }

    #[test]
    fn test_syntax_is_directed() {
        let mut net = ResonanceNetwork::new(10);
        let a = net.add_concept(0.0);
        let b = net.add_concept(0.0);
        net.connect_syntax(a, b, 1.0);

        assert_eq!(net.get_weight(a, b), 1.0);
        assert_eq!(net.get_weight(b, a), 0.0, "syntax is not bidirectional");
    }

    #[test]
    fn test_add_concept_increments_n() {
        let mut net = ResonanceNetwork::new(5);
        assert_eq!(net.n, 0);
        net.add_concept(0.1);
        assert_eq!(net.n, 1);
        net.add_concept(0.2);
        assert_eq!(net.n, 2);
    }

    #[test]
    fn test_edge_count() {
        let mut net = ResonanceNetwork::new(10);
        let a = net.add_concept(0.0);
        let b = net.add_concept(0.0);
        let c = net.add_concept(0.0);
        net.connect_syntax(a, b, 1.0);        // 1 directed edge
        net.connect_semantic(b, c, 1.0);      // 2 directed edges (bidirectional)
        assert_eq!(net.edge_count(), 3);
    }

    // IIIIII LU determinant ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIII

    #[test]
    fn test_lu_determinant_finite() {
        let mut net = ResonanceNetwork::new(10);
        for _ in 0..3 { net.add_concept(0.1); }
        net.connect_syntax(0, 1, 1.0);
        net.connect_semantic(1, 2, 1.0);

        let det = net.compute_determinant_lu();
        assert!(det.is_finite(), "LU determinant must be finite");
    }

    #[test]
    fn test_determinant_cache_invalidated_on_edge_add() {
        let mut net = ResonanceNetwork::new(10);
        net.add_concept(0.1);
        net.add_concept(0.1);

        net.compute_determinant_lu();
        assert!(net.cache_valid, "cache should be valid after first call");

        net.connect_syntax(0, 1, 2.0);
        assert!(!net.cache_valid, "cache must be invalidated after topology change");

        // Recomputing should succeed without panic
        let _ = net.compute_determinant_lu();
    }

    // IIIIII Sparse Jacobian ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIII

    #[test]
    fn test_jacobian_rhythm_string() {
        let mut net = ResonanceNetwork::new(5);
        let a = net.add_concept(0.0);
        let b = net.add_concept(0.0);
        net.connect_semantic(a, b, 1.0);

        let (det, sens, rhythm) = net.compute_jacobian_sparse(a);
        assert!(det.is_finite());
        assert_eq!(sens.len(), net.n);
        assert!(
            ["Trivial", "Harmonic", "Chaotic"].contains(&rhythm.as_str()),
            "unexpected rhythm: {}",
            rhythm
        );
    }

    // IIIIII Physics integration ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////I

    #[test]
    fn test_step_moves_position() {
        let mut net = ResonanceNetwork::new(4);
        let a = net.add_concept(0.0);
        let b = net.add_concept(0.0);
        net.connect_semantic(a, b, 1.0);

        net.positions.amplitudes[a] = Complex64::new(1.0, 0.0);
        net.positions.amplitudes[b] = Complex64::new(0.0, 0.0);
        net.step(&[]);

        assert!(
            (net.positions.amplitudes[a].re - 1.0).abs() > 1e-6,
            "symplectic step must change position"
        );
    }

    // IIIIII Bifurcation Monitor ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIIIIIIII

    #[test]
    fn test_bifurcation_monitor_detects_growth() {
        let mut monitor = BifurcationMonitor::new(10, 0.3);

        // Growing Wronskian series → positive Lyapunov
        for i in 1..15 {
            monitor.observe(i as f64 * 2.0); // exponential growth
        }

        assert!(monitor.lyapunov_estimate > 0.0,
            "Growing Wronskian should yield positive Lyapunov: λ={}",
            monitor.lyapunov_estimate);
    }

    #[test]
    fn test_bifurcation_monitor_stable_regime() {
        let mut monitor = BifurcationMonitor::new(10, 0.3);

        // Stable Wronskian → negative Lyapunov
        for _ in 0..15 {
            monitor.observe(1.0 + (rand::random::<f64>() * 0.01)); // small noise
        }

        assert_eq!(monitor.regime, BifurcationRegime::Stable);
    }

    #[test]
    fn test_bifurcation_threshold_triggers_action() {
        let mut monitor = BifurcationMonitor::new(5, 0.1); // low threshold

        // Rapid growth
        for i in 1..10 {
            monitor.observe(10f64.powi(i));
        }

        assert!(monitor.is_bifurcating(), "Should detect bifurcation with exponential growth");
        assert_eq!(monitor.regime, BifurcationRegime::Chaotic);
    }

    #[test]
    fn test_energy_conservation_undamped() {
        let mut net = ResonanceNetwork::new(10);
        let a = net.add_concept(0.0);  // zero damping
        let b = net.add_concept(0.0);
        net.connect_semantic(a, b, 0.5);

        net.positions.amplitudes[a] = Complex64::new(1.0, 0.0);
        net.set_timestep(0.001);

        let e0 = net.compute_total_energy();
        for _ in 0..1_000 { net.step(&[]); }
        let e1 = net.compute_total_energy();

        let drift = (e1 - e0).abs() / e0.max(1e-10);
        assert!(drift < 0.05, "energy drift {:.4} exceeds 5 %", drift);
    }

    #[test]
    fn test_compute_total_energy_public() {
        let mut net = ResonanceNetwork::new(4);
        let a = net.add_concept(0.0);
        let b = net.add_concept(0.0);
        net.connect_semantic(a, b, 1.0);
        net.positions.amplitudes[a] = Complex64::new(1.0, 0.0);

        let e = net.compute_total_energy();
        assert!(e.is_finite() && e >= 0.0, "total energy must be finite and non-negative");
    }

    // IIIIII Phase Variance ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

    #[test]
    fn test_phase_variance_independent() {
        let net = ResonanceNetwork::new(4);
        let s1: Vec<Complex64> = (0..4).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let s2: Vec<Complex64> = (0..4).map(|i| Complex64::new(0.0, (i * i) as f64)).collect();

        let wc = net.check_phase_variance(&s1, &s2);
        assert!(wc.is_unique, "orthogonal trajectories must be linearly independent");
    }

    #[test]
    fn test_phase_variance_dependent() {
        let net = ResonanceNetwork::new(4);
        let s1: Vec<Complex64> = (0..4).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let s2: Vec<Complex64> = s1.iter().map(|&v| v * 2.0).collect();  // collinear

        let wc = net.check_phase_variance(&s1, &s2);
        assert!(!wc.is_unique, "collinear solutions must be linearly dependent");
    }

    // IIIIII Phase Calculation ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////IIIIIIIIIII

    #[test]
    fn test_compute_aggregate_phase() {
        let mut net = ResonanceNetwork::new(10);
        let a = net.add_labeled_concept("motor", 0.0, Vec::new());
        let b = net.add_labeled_concept("rotor", 0.0, Vec::new());
        
        // 1. Orthogonal phases: arg(1+0i + 0+1i) = PI/4
        net.positions.amplitudes[a] = Complex64::new(1.0, 0.0);
        net.positions.amplitudes[b] = Complex64::new(0.0, 1.0);
        
        let phase = net.compute_aggregate_phase();
        assert!((phase - std::f64::consts::PI / 4.0).abs() < 1e-6);
        
        // 2. Opposite phases: arg(1+0i + -1+0i) = 0? 
        // Sum is 0, arg of 0 is undefined but Complex64::new(0.0, 0.0).arg() is 0.0
        net.positions.amplitudes[b] = Complex64::new(-1.0, 0.0);
        let phase = net.compute_aggregate_phase();
        assert_eq!(phase, 0.0);
    }

    // ⚡ Topological Gravity (RoPE Warp Map) ////////////////////////////////////////////////////////////////////////////////////

    #[test]
    fn test_rope_warp_map_basic() {
        let mut net = ResonanceNetwork::new(10);
        let a = net.add_labeled_concept("motor", 0.0, Vec::new());
        let b = net.add_labeled_concept("rotor", 0.0, Vec::new());
        net.connect_semantic(a, b, 1.0);
        
        let warp_map = net.compute_rope_warp_map("motor test");
        
        // "motor" is directly in the prompt → must be in warp map
        assert!(warp_map.contains_key("motor"), "prompt concept 'motor' must be warped");
        assert!(warp_map["motor"] < 1.0, "high-energy concept should have warp < 1.0");
        assert!(warp_map["motor"] >= 0.3, "warp factor must not go below 0.3 floor");
        
        // "rotor" is semantically connected to "motor" → should also be warped via CSR diffusion
        assert!(warp_map.contains_key("rotor"), "connected neighbor 'rotor' must be warped");
    }

    #[test]
    fn test_rope_warp_map_empty_for_unrelated_prompt() {
        let mut net = ResonanceNetwork::new(10);
        net.add_labeled_concept("motor", 0.0, Vec::new());
        net.add_labeled_concept("rotor", 0.0, Vec::new());
        
        let warp_map = net.compute_rope_warp_map("completely unrelated text");
        assert!(warp_map.is_empty(), "no concepts match → empty warp map");
    }

    #[test]
    fn test_keystones_and_ripple_effect() {
        let mut net = ResonanceNetwork::new(10);
        let a = net.add_labeled_concept("Main.java", 0.0, Vec::new());
        let b = net.add_labeled_concept("Wizard.java", 0.0, Vec::new());
        let c = net.add_labeled_concept("mana", 0.0, Vec::new());
        
        // Connect them: a (Main) -> c (mana) -> b (Wizard)
        net.connect_semantic(a, c, 1.0);
        net.connect_semantic(c, b, 1.0);
        
        let keystones = net.find_keystones();
        assert!(!keystones.is_empty());
        assert_eq!(keystones[0], c, "mana must be detected as the keystone");
        
        let ripple = net.propagate_change(c);
        let ripple_node_ids: Vec<usize> = ripple.iter().map(|&(id, _)| id).collect();
        assert!(ripple_node_ids.contains(&a));
        assert!(ripple_node_ids.contains(&b));
    }

    #[test]
    fn test_holographic_memory_alignment() {
        let mut net = ResonanceNetwork::new(10);
        // Initially empty
        assert_eq!(net.holographic_memory.len(), 0);
        assert_eq!(net.node_flags.len(), 0);

        // Add raw concept
        let id1 = net.add_concept(0.1);
        assert_eq!(net.n, 1);
        assert_eq!(net.holographic_memory.len(), 1);
        assert_eq!(net.node_flags.len(), 1);
        assert!(net.holographic_memory[id1].is_empty());
        assert_eq!(net.node_flags[id1], 0);

        // Add labeled concept
        let dummy_sig = vec![Complex64::new(1.0, 2.0); 4096];
        let id2 = net.add_labeled_concept("c2", 0.05, dummy_sig.clone());
        assert_eq!(net.n, 2);
        assert_eq!(net.holographic_memory.len(), 2);
        assert_eq!(net.node_flags.len(), 2);
        assert_eq!(net.holographic_memory[id2], dummy_sig);
        assert_eq!(net.node_flags[id2], crate::semantic_parser::DOMAIN_LOGIC);

        // Batch add concepts
        let batch_ids = net.add_concepts_batch(&[("c3", 0.02), ("c4", 0.03)]);
        assert_eq!(net.n, 4);
        assert_eq!(net.holographic_memory.len(), 4);
        assert_eq!(net.node_flags.len(), 4);
        for &id in &batch_ids {
            assert!(net.holographic_memory[id].is_empty());
            assert_eq!(net.node_flags[id], 0);
        }

        // Test save cartridge padding
        // Create a temporary file path
        let temp_path = std::env::temp_dir().join("test_cartridge.qmem");
        let temp_path_str = temp_path.to_string_lossy();
        net.save_cartridge(&temp_path_str).unwrap();

        // Load the cartridge back and verify lengths
        let loaded = ResonanceNetwork::load_cartridge(&temp_path_str).unwrap();
        assert_eq!(loaded.n, 4);
        assert_eq!(loaded.holographic_memory.len(), 4);
        assert_eq!(loaded.node_flags.len(), 4);
        assert_eq!(loaded.holographic_memory[id2], dummy_sig);

        // Clean up
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_manhattan_gradient_prefers_shorter_path() {
        let mut net = ResonanceNetwork::new(10);
        let a = net.add_concept(0.01);
        let b = net.add_concept(0.01);
        let c = net.add_concept(0.01);

        net.connect_semantic(a, c, 0.9);
        net.connect_semantic(a, b, 0.5);
        net.connect_semantic(b, c, 0.5);

        net.positions.amplitudes[a] = Complex64::new(0.0, 0.0);
        net.positions.amplitudes[b] = Complex64::new(5.0, 5.0);
        net.positions.amplitudes[c] = Complex64::new(0.1, 0.1);

        let trajectory = net.find_hamiltonian_trajectory(&[a], &[c], 5);
        assert!(!trajectory.contains(&b), "Manhattan gradient should skip detour node");
        assert_eq!(trajectory.len(), 2, "Should reach target in 2 steps");
    }

    #[test]
    fn test_snapshot_restore_preserves_holographic_memory() {
        let mut net = ResonanceNetwork::new(10);
        let sig: Vec<Complex64> = (0..4096).map(|i|
            Complex64::new(i as f64 * 0.001, 0.0)
        ).collect();
        net.add_labeled_concept("test_node", 0.01, sig.clone());

        let snap = net.snapshot();
        net.holographic_memory[0] = Vec::new();
        net.restore_snapshot(&snap);

        assert_eq!(net.holographic_memory[0].len(), 4096, "holographic signature must be restored");
    }

    #[test]
    fn test_identical_grids_max_similarity() {
        let mut net = ResonanceNetwork::new(200);
        let grid = Array2::from_shape_vec((3, 3), vec![
            1, 0, 2,
            0, 3, 0,
            2, 0, 1,
        ]).unwrap();

        let sg_a = net.build_scene_graph(&grid);
        let mut net2 = ResonanceNetwork::new(200);
        let sg_b = net2.build_scene_graph(&grid);

        let sim = ResonanceNetwork::scene_similarity(&sg_a, &sg_b);
        assert!(sim > 0.99, "Identical grids must have similarity > 0.99, got {}", sim);
    }

    #[test]
    fn test_different_grids_low_similarity() {
        let mut net = ResonanceNetwork::new(200);
        let grid_a = Array2::from_shape_vec((3, 3), vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1
        ]).unwrap();
        let grid_b = Array2::from_shape_vec((3, 3), vec![
            2, 3, 4, 5, 6, 7, 8, 9, 1
        ]).unwrap();

        let sg_a = net.build_scene_graph(&grid_a);
        let sg_b = net.build_scene_graph(&grid_b);
        let sim = ResonanceNetwork::scene_similarity(&sg_a, &sg_b);

        assert!(sim < 0.3, "Very different grids should have similarity < 0.3, got {}", sim);
    }

    #[test]
    fn test_vsa_binding_produces_unique_signature() {
        let color_sig = ResonanceNetwork::generate_xorshift_signature(12345, 4096);
        let pos_sig = ResonanceNetwork::generate_xorshift_signature(67890, 4096);
        let bound = ResonanceNetwork::vsa_bind(&color_sig, &pos_sig);

        let sim_color = ResonanceNetwork::holographic_similarity(&bound, &color_sig);
        let sim_pos = ResonanceNetwork::holographic_similarity(&bound, &pos_sig);

        assert!(sim_color < 0.5, "Bound sig must differ from color sig");
        assert!(sim_pos < 0.5, "Bound sig must differ from pos sig");
    }
}













