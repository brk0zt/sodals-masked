// src/qmem_compiler.rs
// Phase 43: The Axiomatic Arkhe Distiller
//
// This module procedurally generates ternary weight matrices that encode
// Universal Physical and Geometric Axioms (The "Arkhes") without training data.
//
// "We are NOT building a lookup table. We are NOT memorizing ARC-AGI training JSONs.
//  Overfitting is a sin against Artificial General Intelligence."
//
// The Arkhes encode Core Knowledge Priors:
// - Mass Conservation (Object Permanence)
// - Constructive Wave Interference (Harmonic Alignment)
// - Group Theory Invariants (Topological Similarity)

use crate::nvme_direct::{HilbertSpatialIndex, QMEM_MAX_SIZE};
use std::collections::HashMap;

/// Ternary weight encoding: [-1, 0, 1] only
/// This forces the network to learn symbolic rules, not floating-point interpolation.
pub type TernaryValue = i8;

/// The Three Fundamental Arkhes (Core Knowledge Priors)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arkhe {
    /// Mass Conservation / Object Permanence
    /// Penalizes pixel creation/destruction. Translation invariance yields 0 residual.
    Conditions,
    
    /// Constructive Wave Interference / Harmonic Alignment
    /// Topological low-pass filter. Compatible phases amplify (+1), incompatible cancel (-1).
    Harmony,
    
    /// Topological Similarity / Group Theory Invariants
    /// Rotation invariance: R_90^4 = I. Internal topology preserved under transformation.
    Isomorphism,
    
    /// PHASE 48: Boolean Logic Gates as Continuous Karnaugh Maps
    /// Ternary weights [-1, 0, 1] encode AND, OR, XOR as hardware logic gates
    /// Superposition yields destructive interference for invalid logic states
    BooleanLogic,
}

impl Arkhe {
    pub fn as_str(&self) -> &'static str {
        match self {
            Arkhe::Conditions => "Arkhe of Conditions (Mass Conservation)",
            Arkhe::Harmony => "Arkhe of Harmony (Wave Interference)",
            Arkhe::Isomorphism => "Arkhe of Isomorphism (Group Theory)",
            Arkhe::BooleanLogic => "Arkhe of BooleanLogic (Ternary K-Maps)",
        }
    }
}

/// PHASE 48: Boolean Logic Gate Types for Ternary K-Maps
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogicGate {
    AND, OR, XOR, NAND, NOR, XNOR,
}

impl LogicGate {
    fn as_u8(&self) -> u8 {
        match self {
            LogicGate::AND => 0,
            LogicGate::OR => 1,
            LogicGate::XOR => 2,
            LogicGate::NAND => 3,
            LogicGate::NOR => 4,
            LogicGate::XNOR => 5,
        }
    }
}

/// Spatial parameterization for axiom mapping
/// Each axiom maps to a 2D grid coordinate via Hilbert curve
#[derive(Debug, Clone)]
pub struct AxiomSpatialParams {
    /// Primary parameter (angle, phase, rotation count)
    pub param1: f32,
    /// Secondary parameter (scale, frequency, symmetry order)
    pub param2: f32,
    /// Tertiary parameter (translation dx, wavelength, group element)
    pub param3: f32,
}

/// The Axiomatic Arkhe Distiller
/// 
/// Procedurally generates ternary weight blocks that encode universal physics
/// without training data exposure. This is Fluid Intelligence, not memorization.
pub struct ArkheDistiller {
    /// Hilbert spatial indexer for SSD organization
    hilbert: HilbertSpatialIndex,
    
    /// Block counter for sequential allocation
    #[allow(dead_code)]
    next_block_index: u64,
    
    /// Axiom-to-SSD mapping registry (uses Arkhe only as key, params stored separately)
    axiom_registry: HashMap<Arkhe, Vec<(AxiomSpatialParams, u64)>>,
}

impl ArkheDistiller {
    /// Initialize the Arkhe Distiller with 16-bit Hilbert order
    /// This provides 65536 x 65536 spatial resolution for axiom organization
    pub fn new() -> Self {
        let hilbert = HilbertSpatialIndex::new(16); // 2^16 = 65536 grid
        
        Self {
            hilbert,
            next_block_index: 0,
            axiom_registry: HashMap::new(),
        }
    }

    // =========================================================================
    // ARKHE 1: CONDITIONS (Mass Conservation / Object Permanence)
    // =========================================================================
    
    /// Generate ternary weights for Mass Conservation axiom
    /// 
    /// Mathematical Model:
    /// - Input: Grid state vector (flattened 2D grid)
    /// - Operation: Compute mass (pixel count) before and after transformation
    /// - Penalty: |mass_after - mass_before| (should be 0 for valid transformations)
    /// 
    /// Ternary Encoding:
    /// - W[i,j] = 1 if pixel i contributes to mass count
    /// - W[i,j] = -1 if pixel j is created/destroyed (violation)
    /// - W[i,j] = 0 otherwise (neutral)
    /// 
    /// # Arguments
    /// * `grid_size` - Width/height of square grid (e.g., 30 for ARC-AGI)
    /// * `translation` - (dx, dy) translation vector
    /// 
    /// # Returns
    /// 4KB block of ternary weights encoding mass conservation constraints
    pub fn distill_mass_conservation(&mut self, grid_size: u32, translation: (i32, i32)) -> (u64, [u8; 4096]) {
        let axiom = Arkhe::Conditions;
        let params = AxiomSpatialParams {
            param1: translation.0 as f32,
            param2: translation.1 as f32,
            param3: grid_size as f32,
        };
        
        // Map to Hilbert spatial coordinate
        // x = normalized translation magnitude, y = grid scale
        let (dx, dy) = translation;
        let trans_magnitude = ((dx * dx + dy * dy) as f32).sqrt();
        let max_trans = (grid_size as f32) * 2.0_f32.sqrt();
        
        let x = ((trans_magnitude / max_trans) * 65535.0) as u32;
        let y = ((grid_size as f32 / 30.0) * 65535.0) as u32; // ARC max is 30x30
        
        // Generate ternary weight matrix
        let mut weights = vec![0i8; grid_size as usize * grid_size as usize];
        
        // Mass conservation: Each pixel at (i,j) contributes +1 to mass count
        // Violation detection: If pixel disappears without appearing elsewhere = -1
        for row in 0..grid_size {
            for col in 0..grid_size {
                let idx = (row * grid_size + col) as usize;
                
                // Source pixel contributes +1 to mass
                weights[idx] = 1;
                
                // Check if translation keeps pixel in bounds
                let new_row = row as i32 + dy;
                let new_col = col as i32 + dx;
                
                if new_row < 0 || new_row >= grid_size as i32 ||
                   new_col < 0 || new_col >= grid_size as i32 {
                    // Translation moves pixel out of bounds = destruction
                    // This triggers a -1 penalty for mass violation
                    weights[idx] = -1;
                }
            }
        }
        
        // Pack into 4KB block
        let block = self.pack_ternary_block(&weights, axiom, &params);
        
        // Compute SSD address via Hilbert curve
        let hilbert_index = self.hilbert.encode(x, y);
        let ssd_address = self.hilbert.to_ssd_address(hilbert_index, QMEM_MAX_SIZE.try_into().unwrap());
        
        self.axiom_registry.entry(axiom).or_insert_with(Vec::new).push((params, ssd_address));
        
        eprintln!(
            "[ARKHE CONDITIONS] Mass Conservation: grid={}x{}, trans=({},{}) -> SSD=0x{:016X}",
            grid_size, grid_size, dx, dy, ssd_address
        );
        
        (ssd_address, block)
    }

    // =========================================================================
    // ARKHE 2: HARMONY (Constructive Wave Interference)
    // =========================================================================
    
    /// Generate ternary weights for Harmonic Alignment axiom
    /// 
    /// Mathematical Model:
    /// - Phase alignment: compatible features amplify, incompatible cancel
    /// - Ternary rule: +1 for constructive interference, -1 for destructive
    /// 
    /// This acts as a topological low-pass filter on the feature space.
    /// 
    /// # Arguments
    /// * `phase1` - First phase angle [0, 2π)
    /// * `phase2` - Second phase angle [0, 2π)
    /// * `tolerance` - Alignment tolerance in radians
    /// 
    /// # Returns
    /// 4KB block encoding wave interference rules
    pub fn distill_harmonic_alignment(&mut self, phase1: f32, phase2: f32, tolerance: f32) -> (u64, [u8; 4096]) {
        let axiom = Arkhe::Harmony;
        let params = AxiomSpatialParams {
            param1: phase1,
            param2: phase2,
            param3: tolerance,
        };
        
        // Map phases to spatial coordinates
        // x = phase1 normalized, y = phase difference
        let phase_diff = (phase2 - phase1).abs();
        let normalized_diff = (phase_diff / std::f32::consts::TAU).min(1.0);
        let normalized_p1 = (phase1 / std::f32::consts::TAU).rem_euclid(1.0);
        
        let x = (normalized_p1 * 65535.0) as u32;
        let y = (normalized_diff * 65535.0) as u32;
        
        // Generate interference matrix (64x64 for 4KB)
        let size = 64usize;
        let mut weights = vec![0i8; size * size];
        
        // Ternary interference rules
        for i in 0..size {
            for j in 0..size {
                let idx = i * size + j;
                
                // Compute local phase at position (i,j)
                let local_phase = 2.0 * std::f32::consts::PI * (i as f32 / size as f32);
                
                // Phase alignment with tolerance
                let align1 = (local_phase - phase1).abs() < tolerance;
                let align2 = (local_phase - phase2).abs() < tolerance;
                
                weights[idx] = match (align1, align2) {
                    (true, true) => 1,   // Constructive: both phases align
                    (true, false) => -1, // Destructive: only phase1
                    (false, true) => -1, // Destructive: only phase2
                    (false, false) => 0, // Neutral: neither aligns
                };
            }
        }
        
        let block = self.pack_ternary_block(&weights, axiom, &params);
        let hilbert_index = self.hilbert.encode(x, y);
        let ssd_address = self.hilbert.to_ssd_address(hilbert_index, QMEM_MAX_SIZE.try_into().unwrap());
        
        self.axiom_registry.entry(axiom).or_insert_with(Vec::new).push((params, ssd_address));
        
        eprintln!(
            "[ARKHE HARMONY] Wave Interference: phase1={:.3}, phase2={:.3}, tol={:.3} -> SSD=0x{:016X}",
            phase1, phase2, tolerance, ssd_address
        );
        
        (ssd_address, block)
    }

    // =========================================================================
    // ARKHE 3: ISOMORPHISM (Group Theory / Rotation Invariants)
    // =========================================================================
    
    /// Generate ternary weights for Rotation Invariance (Group Theory)
    /// 
    /// Mathematical Model:
    /// - Cyclic group C4: {I, R_90, R_180, R_270}
    /// - Property: R_90^4 = I (4-fold symmetry)
    /// - Internal topology must be invariant under group action
    /// 
    /// Ternary Encoding:
    /// - +1: Topology preserved under rotation
    /// - -1: Topology violated (e.g., asymmetric feature at rotation center)
    /// - 0: Neutral position
    /// 
    /// # Arguments
    /// * `rotation_order` - Order of rotation (4 for 90°, 2 for 180°, etc.)
    /// * `grid_size` - Size of square grid
    /// 
    /// # Returns
    /// 4KB block encoding group invariants
    pub fn distill_rotation_invariant(&mut self, rotation_order: u32, grid_size: u32) -> (u64, [u8; 4096]) {
        let axiom = Arkhe::Isomorphism;
        let params = AxiomSpatialParams {
            param1: rotation_order as f32,
            param2: grid_size as f32,
            param3: 0.0,
        };
        
        // Map to spatial: x = rotation order, y = grid size
        let x = ((rotation_order as f32 / 8.0) * 65535.0) as u32; // Max 8-fold
        let y = ((grid_size as f32 / 30.0) * 65535.0) as u32;
        
        // Generate invariance matrix
        let size = grid_size as usize;
        let mut weights = vec![0i8; size * size];
        
        // Center of rotation
        let cx = (size - 1) as f32 / 2.0;
        let cy = (size - 1) as f32 / 2.0;
        
        // For each position, check if it's invariant under rotation
        for row in 0..size {
            for col in 0..size {
                let idx = row * size + col;
                
                // Convert to centered coordinates
                let dx = col as f32 - cx;
                let dy = row as f32 - cy;
                
                // Compute distance from center (invariant under rotation)
                let r = (dx * dx + dy * dy).sqrt();
                
                // Check all rotation positions
                let angle_step = 2.0 * std::f32::consts::PI / rotation_order as f32;
                let mut all_same = true;
                
                for k in 1..rotation_order {
                    let theta = angle_step * k as f32;
                    let rot_dx = dx * theta.cos() - dy * theta.sin();
                    let rot_dy = dx * theta.sin() + dy * theta.cos();
                    
                    // Check if rotated position maps to valid grid
                    let rot_col = (rot_dx + cx).round() as i32;
                    let rot_row = (rot_dy + cy).round() as i32;
                    
                    if rot_col < 0 || rot_col >= size as i32 ||
                       rot_row < 0 || rot_row >= size as i32 {
                        all_same = false;
                        break;
                    }
                }
                
                // Mark invariant positions with +1, boundary conditions with -1
                weights[idx] = if r < 0.8 {
                    0 // Center is rotationally symmetric by definition
                } else if all_same {
                    1 // Position preserves topology under rotation
                } else {
                    -1 // Violation: asymmetric feature
                };
            }
        }
        
        let block = self.pack_ternary_block(&weights, axiom, &params);
        let hilbert_index = self.hilbert.encode(x, y);
        let ssd_address = self.hilbert.to_ssd_address(hilbert_index, QMEM_MAX_SIZE.try_into().unwrap());
        
        self.axiom_registry.entry(axiom).or_insert_with(Vec::new).push((params, ssd_address));
        
        eprintln!(
            "[ARKHE ISOMORPHISM] Rotation Invariant: order={}, grid={}x{} -> SSD=0x{:016X}",
            rotation_order, grid_size, grid_size, ssd_address
        );
        
        (ssd_address, block)
    }

    // =========================================================================
    // PHASE 48: TERNARY KARNAUGH MAPS (Hardware Logic Gates)
    // =========================================================================
    
    /// PHASE 48: Distill Boolean Logic Gates as Continuous Karnaugh Maps
    /// 
    /// The Concept: The ternary weights [-1, 0, 1] on the SSD are not just for dot products.
    /// They are physical logic gates encoded as continuous Karnaugh Maps (K-Maps).
    /// 
    /// Mathematical Model:
    /// - K-Map is a 2D truth table: rows = input A, columns = input B
    /// - Ternary encoding: -1 = FALSE (destructive interference), 
    ///                     0 = DON'T CARE (neutral),
    ///                     1 = TRUE (constructive interference)
    /// - When Split-Complex projection fetches these weights, it enforces combinatorial logic
    /// - Example: If Concept A and Concept B are XOR, their superposition yields
    ///           total destructive interference when A == B, amplification when A != B
    /// 
    /// # Arguments
    /// * `gate_type` - The logic gate: AND, OR, XOR, NAND, NOR, XNOR
    /// * `input_dim` - Dimension of input space (determines K-Map size)
    /// 
    /// # Returns
    /// 4KB block encoding the continuous K-Map for the logic gate
    pub fn distill_boolean_logic(&mut self, gate_type: LogicGate, input_dim: u32) -> (u64, [u8; 4096]) {
        let axiom = Arkhe::BooleanLogic;
        let params = AxiomSpatialParams {
            param1: gate_type.as_u8() as f32,
            param2: input_dim as f32,
            param3: 0.0,
        };
        
        // Map to spatial: x = gate type, y = input dimension
        let x = ((gate_type.as_u8() as f32 / 8.0) * 65535.0) as u32;
        let y = ((input_dim as f32 / 64.0) * 65535.0) as u32;
        
        // Generate K-Map matrix (64x64 for 4KB block)
        let size = 64usize;
        let mut weights = vec![0i8; size * size];
        
        // PHASE 48: Continuous K-Map Encoding
        // Map 2D index (i, j) to continuous inputs (a, b) ∈ [-1, 1]
        for i in 0..size {
            for j in 0..size {
                let idx = i * size + j;
                
                // Normalize to [-1, 1] range (continuous inputs)
                let a = 2.0 * (i as f32 / (size as f32 - 1.0)) - 1.0;
                let b = 2.0 * (j as f32 / (size as f32 - 1.0)) - 1.0;
                
                // Compute gate output in ternary space
                weights[idx] = match gate_type {
                    LogicGate::AND => {
                        // AND: both must be positive for TRUE
                        if a > 0.3 && b > 0.3 { 1 }
                        else if a < -0.3 || b < -0.3 { -1 }
                        else { 0 }
                    }
                    LogicGate::OR => {
                        // OR: at least one positive for TRUE
                        if a > 0.3 || b > 0.3 { 1 }
                        else if a < -0.3 && b < -0.3 { -1 }
                        else { 0 }
                    }
                    LogicGate::XOR => {
                        // XOR: exactly one positive for TRUE
                        let a_pos = a > 0.1;
                        let b_pos = b > 0.1;
                        if a_pos != b_pos { 1 }
                        else if a_pos == b_pos && (a.abs() > 0.3 || b.abs() > 0.3) { -1 }
                        else { 0 }
                    }
                    LogicGate::NAND => {
                        // NAND: NOT (A AND B)
                        if a > 0.3 && b > 0.3 { -1 }
                        else if a < -0.3 || b < -0.3 { 1 }
                        else { 0 }
                    }
                    LogicGate::NOR => {
                        // NOR: NOT (A OR B)
                        if a > 0.3 || b > 0.3 { -1 }
                        else if a < -0.3 && b < -0.3 { 1 }
                        else { 0 }
                    }
                    LogicGate::XNOR => {
                        // XNOR: NOT (A XOR B) = equivalence
                        let a_pos = a > 0.1;
                        let b_pos = b > 0.1;
                        if a_pos == b_pos { 1 }
                        else { -1 }
                    }
                };
            }
        }
        
        let block = self.pack_ternary_block(&weights, axiom, &params);
        let hilbert_index = self.hilbert.encode(x, y);
        let ssd_address = self.hilbert.to_ssd_address(hilbert_index, QMEM_MAX_SIZE.try_into().unwrap());
        
        self.axiom_registry.entry(axiom).or_insert_with(Vec::new).push((params, ssd_address));
        
        eprintln!(
            "[ARKHE BOOLEAN] Logic Gate: {:?}, dim={} -> SSD=0x{:016X}",
            gate_type, input_dim, ssd_address
        );
        
        (ssd_address, block)
    }

    /// PHASE 48: Apply Ternary K-Map Logic via Dot Product
    /// 
    /// Performs the continuous logic operation by projecting inputs onto the K-Map.
    /// Returns ternary result: -1 (FALSE), 0 (UNDEFINED), 1 (TRUE)
    /// 
    /// # Arguments
    /// * `gate_weights` - The K-Map weights from distill_boolean_logic
    /// * `input_a` - First continuous input ∈ [-1, 1]
    /// * `input_b` - Second continuous input ∈ [-1, 1]
    /// 
    /// # Returns
    /// Ternary logic result: -1, 0, or 1
    pub fn apply_ternary_logic(gate_weights: &[i8], input_a: f32, input_b: f32) -> i8 {
        let size = (gate_weights.len() as f32).sqrt() as usize;
        
        // Map continuous inputs to K-Map indices
        let i = ((input_a + 1.0) / 2.0 * (size as f32 - 1.0)).clamp(0.0, size as f32 - 1.0) as usize;
        let j = ((input_b + 1.0) / 2.0 * (size as f32 - 1.0)).clamp(0.0, size as f32 - 1.0) as usize;
        
        // Look up in K-Map
        let idx = i * size + j;
        gate_weights.get(idx).copied().unwrap_or(0)
    }

    // =========================================================================
    // PACKING UTILITIES
    // =========================================================================
    
    /// Pack ternary weights into 4KB block with metadata
    /// 
    /// Format:
    /// - Bytes 0-3: Magic "ARKH"
    /// - Byte 4: Arkhe type (0=Conditions, 1=Harmony, 2=Isomorphism, 3=BooleanLogic)
    /// - Bytes 5-7: Reserved
    /// - Bytes 8-15: Hilbert index
    /// - Bytes 16-31: Spatial parameters (3 x f32)
    /// - Bytes 32-4095: Ternary weights (2 bits each: 00=-1, 01=0, 10=1)
    fn pack_ternary_block(&self, weights: &[i8], arkhe: Arkhe, params: &AxiomSpatialParams) -> [u8; 4096] {
        let mut block = [0u8; 4096];
        
        // Magic header
        block[0..4].copy_from_slice(b"ARKH");
        
        // Arkhe type
        block[4] = match arkhe {
            Arkhe::Conditions => 0,
            Arkhe::Harmony => 1,
            Arkhe::Isomorphism => 2,
            Arkhe::BooleanLogic => 3,
        };
        
        // Hilbert index (placeholder - filled during distillation)
        // block[8..16] reserved
        
        // Spatial parameters
        block[16..20].copy_from_slice(&params.param1.to_le_bytes());
        block[20..24].copy_from_slice(&params.param2.to_le_bytes());
        block[24..28].copy_from_slice(&params.param3.to_le_bytes());
        
        // Pack ternary weights (2 bits per weight = 4 weights per byte)
        // Encoding: 00=-1, 01=0, 10=1, 11=reserved
        let max_weights = (4096 - 32) * 4; // 16,256 weights max
        let weight_count = weights.len().min(max_weights);
        
        for i in 0..weight_count {
            let byte_idx = 32 + i / 4;
            let bit_offset = (i % 4) * 2;
            
            let encoded = match weights[i] {
                -1 => 0b00,
                0 => 0b01,
                1 => 0b10,
                _ => 0b11, // Invalid = reserved
            };
            
            block[byte_idx] |= encoded << bit_offset;
        }
        
        block
    }
    
    /// Unpack ternary weights from 4KB block
    pub fn unpack_ternary_block(block: &[u8; 4096], count: usize) -> Vec<i8> {
        let mut weights = Vec::with_capacity(count);
        
        for i in 0..count {
            let byte_idx = 32 + i / 4;
            let bit_offset = (i % 4) * 2;
            
            let encoded = (block[byte_idx] >> bit_offset) & 0b11;
            
            let weight = match encoded {
                0b00 => -1,
                0b01 => 0,
                0b10 => 1,
                _ => 0, // Reserved = neutral
            };
            
            weights.push(weight);
        }
        
        weights
    }

    // =========================================================================
    // BATCH DISTILLATION
    // =========================================================================
    
    /// Distill all Core Priors for ARC-AGI domain
    /// 
    /// Generates comprehensive axiom coverage:
    /// - Mass conservation for all translations in [-5, 5] range
    /// - Harmonic alignment for phase pairs
    /// - Rotation invariants for 90°, 180°, 270°
    pub fn distill_arc_agi_priors(&mut self) -> Vec<(u64, [u8; 4096])> {
        let mut blocks = Vec::new();
        
        eprintln!("[ARKHE DISTILLER] Beginning ARC-AGI Core Priors distillation...");
        
        // Arkhe 1: Mass Conservation for common translations
        for dx in -5..=5i32 {
            for dy in -5..=5i32 {
                if dx == 0 && dy == 0 { continue; }
                let (addr, block) = self.distill_mass_conservation(10, (dx, dy));
                blocks.push((addr, block));
            }
        }
        
        // Arkhe 2: Harmonic alignment for phase pairs
        let phase_steps = 8; // 8 phase positions around circle
        for i in 0..phase_steps {
            for j in (i + 1)..phase_steps {
                let phase1 = 2.0 * std::f32::consts::PI * (i as f32 / phase_steps as f32);
                let phase2 = 2.0 * std::f32::consts::PI * (j as f32 / phase_steps as f32);
                let (addr, block) = self.distill_harmonic_alignment(phase1, phase2, 0.5);
                blocks.push((addr, block));
            }
        }
        
        // Arkhe 3: Rotation invariants
        for &order in &[2u32, 4, 6, 8] {
            for &size in &[10u32, 15, 20, 30] {
                let (addr, block) = self.distill_rotation_invariant(order, size);
                blocks.push((addr, block));
            }
        }
        
        eprintln!(
            "[ARKHE DISTILLER] Complete: {} axiom blocks distilled",
            blocks.len()
        );
        
        blocks
    }
    
    /// Get registry statistics
    pub fn get_statistics(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        
        for (arkhe, entries) in &self.axiom_registry {
            let key = arkhe.as_str().to_string();
            *stats.entry(key).or_insert(0) += entries.len();
        }
        
        stats.insert("Total Blocks".to_string(), self.axiom_registry.len());
        
        stats
    }
}

impl Default for ArkheDistiller {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// UNIT TESTS
// =========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mass_conservation() {
        let mut distiller = ArkheDistiller::new();
        let (addr, block) = distiller.distill_mass_conservation(5, (1, 0));
        
        assert!(addr > 0);
        assert!(addr & 0xFFF == 0); // 4KB aligned
        assert_eq!(&block[0..4], b"ARKH");
        assert_eq!(block[4], 0); // Conditions = 0
        
        // Unpack and verify structure
        let weights = ArkheDistiller::unpack_ternary_block(&block, 25);
        assert_eq!(weights.len(), 25);
        
        // Check that weights are ternary (-1, 0, 1)
        for w in &weights {
            assert!(*w >= -1 && *w <= 1);
        }
    }
    
    #[test]
    fn test_harmonic_alignment() {
        let mut distiller = ArkheDistiller::new();
        let (addr, block) = distiller.distill_harmonic_alignment(0.0, 1.57, 0.5);
        
        assert!(addr > 0);
        assert_eq!(&block[0..4], b"ARKH");
        assert_eq!(block[4], 1); // Harmony = 1
    }
    
    #[test]
    fn test_rotation_invariant() {
        let mut distiller = ArkheDistiller::new();
        let (addr, block) = distiller.distill_rotation_invariant(4, 10);
        
        assert!(addr > 0);
        assert_eq!(&block[0..4], b"ARKH");
        assert_eq!(block[4], 2); // Isomorphism = 2
        
        let weights = ArkheDistiller::unpack_ternary_block(&block, 100);
        // Center should be 0 (symmetric)
        assert_eq!(weights[44], 0); // Approximate center of 10x10
    }
    
    #[test]
    fn test_arc_agi_priors() {
        let mut distiller = ArkheDistiller::new();
        let blocks = distiller.distill_arc_agi_priors();
        
        assert!(!blocks.is_empty());
        
        let stats = distiller.get_statistics();
        eprintln!("Arkhe Statistics: {:?}", stats);
        
        assert!(stats.contains_key("Arkhe of Conditions (Mass Conservation)"));
        assert!(stats.contains_key("Arkhe of Harmony (Wave Interference)"));
        assert!(stats.contains_key("Arkhe of Isomorphism (Group Theory)"));
    }
    
    // PHASE 48: Ternary K-Map Tests
    
    #[test]
    fn test_boolean_logic_and() {
        let mut distiller = ArkheDistiller::new();
        let (addr, block) = distiller.distill_boolean_logic(LogicGate::AND, 32);
        
        assert!(addr > 0);
        assert!(addr & 0xFFF == 0); // 4KB aligned
        assert_eq!(&block[0..4], b"ARKH");
        assert_eq!(block[4], 3); // BooleanLogic = 3
        
        // Unpack and verify
        let weights = ArkheDistiller::unpack_ternary_block(&block, 64*64);
        assert_eq!(weights.len(), 4096);
        
        // Test AND truth table via K-Map
        // Both positive inputs should yield TRUE (1)
        let result = ArkheDistiller::apply_ternary_logic(&weights, 0.8, 0.8);
        assert_eq!(result, 1, "AND(0.8, 0.8) should be TRUE");
        
        // One negative input should yield FALSE (-1)
        let result = ArkheDistiller::apply_ternary_logic(&weights, 0.8, -0.8);
        assert!(result == -1 || result == 0, "AND(0.8, -0.8) should not be TRUE");
    }
    
    #[test]
    fn test_boolean_logic_xor() {
        let mut distiller = ArkheDistiller::new();
        let (addr, block) = distiller.distill_boolean_logic(LogicGate::XOR, 32);
        
        let weights = ArkheDistiller::unpack_ternary_block(&block, 64*64);
        
        // XOR: different inputs = TRUE, same inputs = FALSE
        let result = ArkheDistiller::apply_ternary_logic(&weights, 0.8, -0.8);
        assert_eq!(result, 1, "XOR(0.8, -0.8) should be TRUE");
        
        let result = ArkheDistiller::apply_ternary_logic(&weights, 0.8, 0.8);
        assert_eq!(result, -1, "XOR(0.8, 0.8) should be FALSE");
    }
    
    #[test]
    fn test_all_logic_gates() {
        let mut distiller = ArkheDistiller::new();
        
        for gate in [LogicGate::AND, LogicGate::OR, LogicGate::XOR, 
                     LogicGate::NAND, LogicGate::NOR, LogicGate::XNOR] {
            let (addr, block) = distiller.distill_boolean_logic(gate, 16);
            assert!(addr > 0, "{:?} should have valid address", gate);
            assert_eq!(&block[0..4], b"ARKH", "{:?} should have magic header", gate);
        }
    }
}
