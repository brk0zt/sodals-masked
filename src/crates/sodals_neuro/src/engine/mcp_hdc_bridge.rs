// src/engine/mcp_hdc_bridge.rs
// MCP-TO-HDC BRIDGE — Sensory Cortex for the Continuous Physics Engine
//
// Standard agents read files as strings and append them to text prompts.
// We read files as BOUNDARY CONDITIONS — physical constraints that are
// hashed into continuous dense phasor arrays and pushed to the GPU's
// ODE solver as initial/boundary conditions, NOT appended to a text prompt.
//
// This bridge takes parsed AST nodes (from tree-sitter via MCP tools) and
// .experience axiom files, and transforms them into the Complex64 phasor
// representation that the HspOracle and RK4 integrator operate on.
//
// NO text chunking. NO vector databases. NO RAG.
// Memory is 1-million-dimensional sparse topological weights mapped to 4KB NVMe blocks.

use num_complex::Complex64;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::engine::oracle_worker::HspOracle;
use crate::semantic_parser::{DOMAIN_LOGIC, DOMAIN_SPATIAL, DOMAIN_TEMPORAL};

// ============================================================================
// PHASOR DIMENSIONALITY — Size of the dense phasor before HSP projection
// ============================================================================

/// Dimensionality of the dense phasor array produced by the sensory cortex.
/// This is the same dimensionality as the RK4 latent manifold (4096-D).
const PHASOR_DIMENSIONS: usize = 4096;

// ============================================================================
// MCP SENSORY CORTEX TRAIT — Interface for ingesting external reality
// ============================================================================

/// The sensory cortex trait that maps external reality (AST nodes, experience
/// axioms, MCP tool outputs) into the continuous phasor representation that
/// the physics engine operates on.
///
/// Implementations of this trait are the ONLY way external information enters
/// the Core's ODE flow. There is no text prompt. There is no RAG. There are
/// only boundary conditions.
pub trait McpSensoryCortex {
    /// Ingest a parsed AST node (from tree-sitter via an MCP tool) and prepare
    /// it for the HspOracle. The `domain_flag` determines which neuro-symbolic
    /// domain (Logic, Temporal, Spatial) the node belongs to, which affects
    /// the phasor's spectral characteristics.
    fn ingest_ast_node(&self, node_kind: &str, source_text: &str, domain_flag: u32)
        -> PhasorBoundaryCondition;

    /// Ingest an experience axiom from a .experience file and hash it into a
    /// continuous dense phasor array (simulating the Hilbert transform pipeline)
    /// so it can be pushed to the GPU's ODE solver as a boundary condition,
    /// rather than appending it to a text prompt.
    fn ingest_experience_axiom(&self, parsed_rule: &str) -> Vec<Complex64>;

    /// Project a dense phasor boundary condition into the 1,000,000-D sparse
    /// space via the HspOracle, producing an SDR suitable for QMEM storage.
    fn project_to_sparse_memory(&self, phasor: &[Complex64]) -> Vec<u32>;
}

// ============================================================================
// PHASOR BOUNDARY CONDITION — The output of sensory ingestion
// ============================================================================

/// A boundary condition for the RK4 ODE solver, derived from external reality.
/// This is NOT a text string — it is a set of complex-valued initial conditions
/// that constrain the continuous flow of the Core's latent manifold.
#[derive(Debug, Clone)]
pub struct PhasorBoundaryCondition {
    /// The dense 4096-D phasor array (amplitude + phase per dimension)
    pub phasor: Vec<Complex64>,
    /// Neuro-symbolic domain flag (DOMAIN_LOGIC, DOMAIN_TEMPORAL, DOMAIN_SPATIAL)
    pub domain_flag: u32,
    /// Source identifier (e.g., AST node kind or experience axiom name)
    pub source: String,
    /// Spectral energy of the phasor (L2 norm), used for priority weighting
    pub energy: f64,
}

impl PhasorBoundaryCondition {
    /// Compute the L2 energy of this boundary condition's phasor.
    pub fn compute_energy(&mut self) {
        self.energy = self
            .phasor
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>()
            .sqrt();
    }
}

// ============================================================================
// DEFAULT MCP SENSORY CORTEX IMPLEMENTATION
// ============================================================================

/// The default implementation of the sensory cortex that uses deterministic
/// spectral hashing to map textual inputs into continuous phasor arrays.
///
/// The hashing strategy:
/// 1. Compute a deterministic seed from the input string.
/// 2. For each of the 4096 dimensions, derive a phase angle from the seed.
/// 3. The domain flag modulates the base frequency and decay characteristics.
/// 4. The result is a dense Complex64 array that can be:
///    a. Injected directly as an RK4 boundary condition, OR
///    b. Projected to 1M-D sparse space via HspOracle for QMEM storage.
pub struct DefaultMcpSensoryCortex {
    /// Base frequency for Logic domain phasors
    pub logic_base_freq: f64,
    /// Base frequency for Temporal domain phasors
    pub temporal_base_freq: f64,
    /// Base frequency for Spatial domain phasors
    pub spatial_base_freq: f64,
}

impl DefaultMcpSensoryCortex {
    pub fn new() -> Self {
        Self {
            logic_base_freq: 0.5,
            temporal_base_freq: 0.2,
            spatial_base_freq: 0.8,
        }
    }

    /// Deterministic hash of a string to a u64 seed.
    fn hash_seed(input: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        let seed = hasher.finish();
        if seed == 0 { 0xDEADBEEFCAFEBABE } else { seed }
    }

    /// Xorshift64* PRNG step for deterministic sequence generation.
    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        state.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Get base frequency and decay for a given domain flag.
    fn domain_spectral_params(&self, domain_flag: u32) -> (f64, f64) {
        if (domain_flag & DOMAIN_LOGIC) != 0 {
            (self.logic_base_freq, 0.1)
        } else if (domain_flag & DOMAIN_TEMPORAL) != 0 {
            (self.temporal_base_freq, 0.05)
        } else if (domain_flag & DOMAIN_SPATIAL) != 0 {
            (self.spatial_base_freq, 0.2)
        } else {
            (0.3, 0.15) // Default/Abstract domain
        }
    }

    /// Generate a dense phasor array from a seed and domain parameters.
    /// Simulates the Hilbert transform pipeline: each dimension gets a
    /// phase angle derived from the seed, modulated by the domain's
    /// base frequency, with amplitude decay.
    fn generate_phasor(seed: u64, base_freq: f64, decay: f64) -> Vec<Complex64> {
        let mut state = seed;
        let mut phasor = Vec::with_capacity(PHASOR_DIMENSIONS);

        for i in 0..PHASOR_DIMENSIONS {
            // Derive phase angle from deterministic PRNG
            let rand_val = Self::xorshift(&mut state);
            let phase = (rand_val as f64 / u64::MAX as f64) * 2.0 * std::f64::consts::PI;

            // Amplitude decays with dimension index, modulated by base frequency
            let amplitude = (base_freq * (i as f64 + 1.0).ln() * decay).exp()
                * (1.0 - decay * (i as f64 / PHASOR_DIMENSIONS as f64));

            phasor.push(Complex64::from_polar(amplitude, phase));
        }

        phasor
    }
}

impl Default for DefaultMcpSensoryCortex {
    fn default() -> Self {
        Self::new()
    }
}

impl McpSensoryCortex for DefaultMcpSensoryCortex {
    fn ingest_ast_node(
        &self,
        node_kind: &str,
        source_text: &str,
        domain_flag: u32,
    ) -> PhasorBoundaryCondition {
        let composite = format!("{}:{}", node_kind, source_text);
        let seed = Self::hash_seed(&composite);
        let (base_freq, decay) = self.domain_spectral_params(domain_flag);
        let phasor = Self::generate_phasor(seed, base_freq, decay);

        let mut bc = PhasorBoundaryCondition {
            phasor,
            domain_flag,
            source: node_kind.to_string(),
            energy: 0.0,
        };
        bc.compute_energy();
        bc
    }

    fn ingest_experience_axiom(&self, parsed_rule: &str) -> Vec<Complex64> {
        // Hash the textual instruction into a continuous dense phasor array.
        // This simulates the Hilbert transform pipeline: the rule's semantic
        // content is encoded as spectral energy across 4096 complex dimensions,
        // ready to be pushed to the GPU ODE solver as a boundary condition.
        //
        // Experience axioms default to DOMAIN_LOGIC (they are imperative rules).
        let seed = Self::hash_seed(parsed_rule);
        let (base_freq, decay) = self.domain_spectral_params(DOMAIN_LOGIC);
        Self::generate_phasor(seed, base_freq, decay)
    }

    fn project_to_sparse_memory(&self, phasor: &[Complex64]) -> Vec<u32> {
        // Project the dense 4096-D phasor into 1,000,000-D sparse space
        // via the HspOracle's deterministic dimensional expander.
        HspOracle::project_to_sparse(phasor)
    }
}

// ============================================================================
// BOUNDARY CONDITION BATCH — For injecting multiple conditions at once
// ============================================================================

/// A batch of boundary conditions to be injected into the RK4 ODE solver
/// simultaneously. This represents the complete set of physical constraints
/// derived from a single MCP tool invocation or file read.
#[derive(Debug, Clone)]
pub struct BoundaryConditionBatch {
    /// The individual boundary conditions
    pub conditions: Vec<PhasorBoundaryCondition>,
    /// Combined phasor (element-wise superposition of all conditions)
    pub combined_phasor: Vec<Complex64>,
    /// Total spectral energy of the batch
    pub total_energy: f64,
}

impl BoundaryConditionBatch {
    /// Create a batch from multiple boundary conditions by superposing
    /// their phasors (element-wise addition with normalization).
    pub fn from_conditions(conditions: Vec<PhasorBoundaryCondition>) -> Self {
        let mut combined = vec![Complex64::new(0.0, 0.0); PHASOR_DIMENSIONS];
        let mut total_energy = 0.0;

        for bc in &conditions {
            total_energy += bc.energy;
            for (i, c) in bc.phasor.iter().enumerate() {
                if i < combined.len() {
                    combined[i] += c;
                }
            }
        }

        // Normalize by the number of conditions to prevent amplitude explosion
        if !conditions.is_empty() {
            let n = conditions.len() as f64;
            for c in combined.iter_mut() {
                *c /= n;
            }
        }

        Self {
            conditions,
            combined_phasor: combined,
            total_energy,
        }
    }

    /// Project the combined phasor into 1M-D sparse memory for QMEM storage.
    pub fn to_sparse_sdr(&self) -> Vec<u32> {
        HspOracle::project_to_sparse(&self.combined_phasor)
    }
}
