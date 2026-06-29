# QMEM Architecture: Detailed Code Reference Guide

This document provides exact file paths and line numbers to verify every claim about QMEM's architecture.

---

## 1. QMEM LOADER: Where Data is Read

### File: `src/crates/sodals_neuro/src/qmem_loader/mod.rs`

#### Struct Definition (Lines 1-80)
```rust
// Line 8-20: QmemLoaderConfig
pub struct QmemLoaderConfig {
    pub ram_first: bool,           // Prefer loading layers into RAM first
    pub max_ram_bytes: usize,      // Maximum RAM bytes to use for loaded layers
    pub device: Device,            // Device where loaded tensors should be allocated
}

// Line 24-35: QmemDType
pub enum QmemDType {
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "bf16")]
    BF16,
}

// Line 64-73: QmemLayer structure
pub struct QmemLayer {
    pub layer_index: usize,
    pub tensor: Option<Tensor>,
    pub mmap: Option<Arc<Mmap>>,
    pub on_disk_path: PathBuf,
    pub offset: u64,
    pub length: usize,
    pub resident_bytes: usize,
    pub dtype: QmemDType,
}

// Line 76-82: QmemLoader structure
pub struct QmemLoader {
    pub config: QmemLoaderConfig,
    pub layers: Arc<Mutex<HashMap<usize, QmemLayer>>>,
    pub lru_queue: Arc<Mutex<VecDeque<usize>>>,
    pub resident_bytes: Arc<Mutex<usize>>,
}
```

#### Critical Function: read_layer_bytes (Lines 204-226)
```rust
pub fn read_layer_bytes(&self, layer_idx: usize) -> Result<Vec<u8>> {
    let mut layers = self.layers.lock().unwrap();
    let layer = layers.get_mut(&layer_idx)
        .with_context(|| format!("QMEM layer {} not found", layer_idx))?;

    // Load on-demand: RAM first, fallback to mmap
    if layer.tensor.is_none() && layer.mmap.is_none() {
        if self.config.ram_first {
            self.load_layer_into_ram(layer).or_else(|_| self.load_layer_via_mmap(layer))?;
        } else {
            self.load_layer_via_mmap(layer)?;
        }
    }

    // Return as raw u8 vector
    if let Some(ref tensor) = layer.tensor {
        return tensor.to_vec1::<u8>().with_context(|| format!("Failed to copy QMEM tensor bytes from layer {}", layer_idx));
    }

    if let Some(ref mmap) = layer.mmap {
        let start = layer.offset as usize;
        let end = start.checked_add(layer.length)?;
        return Ok(mmap[start..end].to_vec());  // ← Returns raw bytes, no semantic processing
    }

    Err(anyhow::anyhow!("QMEM layer {} is unavailable", layer_idx))
}
```

**Key Points:**
- **Line 214-218**: Data loaded on-demand (RAM or mmap)
- **Line 220-221**: Returns `Vec<u8>` — **NO semantic conversion**
- **Line 225**: Returns raw byte slice as vector

---

## 2. QMEM INJECTION: Attention Bias Application

### File: `src/crates/sodals_neuro/src/models/qwen_custom.rs`

#### Location: QAttention::forward (Lines 892-909)

```rust
// Line 892-909: QMEM Runtime Attention Bias
if let Some(loader) = qmem_loader {
    match loader.read_layer_bytes(layer_idx) {
        Ok(layer_bytes) if !layer_bytes.is_empty() => {
            // Line 895: Transform bytes → scalar via deterministic fold
            let signal = layer_bytes.iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64));
            
            // Line 896: Compute bias
            // Max value: (u64_max % 1024) × 1e-4 = 1023 × 1e-4 ≈ 0.1023
            // Min value: 0 × 1e-4 = 0
            // Range: [0, +0.1023] logits (always positive due to modulo)
            let bias = ((signal % 1024) as f64) * 1e-4;
            
            // Line 899: DIRECT logit modification (pre-softmax)
            attention_scores = (attention_scores + bias)?;
        }
        Ok(_) => {
            // Empty layer bytes are ignored
        }
        Err(e) => {
            eprintln!("[QMEM] failed to read layer {}: {}", layer_idx, e);
        }
    }
}
```

**Key Points:**
- **Line 895**: Byte fold operation (O(N) reduction of 4KB → u64)
- **Line 897**: No semantic processing — pure hash
- **Line 899**: Adds to attention scores **BEFORE softmax** (direct logit modification)
- **Magnitude**: ±0.1 logits (approximately 10% of typical scale ~±1.0)

#### How This Gets Called (Line 1150)

```rust
// Line 1150 in QDecoderLayer::forward():
// QMEM loader passed as parameter to attention forward
qmem_loader,
```

#### Location in Forward Pass (Line 1137-1150)

```rust
fn forward(
    &self,
    xs: &Tensor,
    attention_mask: Option<&Tensor>,
    seqlen_offset: usize,
    adapter: Option<&NeuroAdapter>,
    poles_ptr: Option<u64>,
    impedance_ptr: Option<u64>,
    flags_ptr: Option<u64>,
    token_concept_map_ptr: Option<u64>,
    layer_idx: usize,
    num_layers: usize,
    n_nodes: Option<i32>,
    phase: Option<&PhaseSignal>,
    qmem_loader: Option<&Arc<QmemLoader>>,  // ← Line 1137: QMEM loader parameter
    gpr_qmem_router: Option<&Arc<Phase45GprQmemRouter>>,
) -> Result<Tensor> {
    let norm_out = self.input_layernorm.forward(xs)?;

    let attn_out = self.self_attn.forward(
        &norm_out,
        attention_mask,
        seqlen_offset,
        adapter,
        poles_ptr,
        impedance_ptr,
        flags_ptr,
        token_concept_map_ptr,
        layer_idx,
        num_layers,
        n_nodes,
        phase,
        qmem_loader,  // ← Line 1150: Passed to attention forward
        gpr_qmem_router,
    )?;
```

---

## 3. QMEM NOT INTEGRATED WITH RESONANCE NETWORK

### File: `src/crates/sodals_neuro/src/qmem_builder.rs`

#### Unused Import Warning (Line 21)

```rust
// Line 21: UNUSED IMPORT
use crate::semantic_network::ResonanceNetwork;

// This import appears in .cargo_check_output.txt:
// warning: unused import: `crate::semantic_network::ResonanceNetwork`
//   --> src\crates\sodals_neuro\src\qmem_builder.rs:21:5
// 21 | use crate::semantic_network::ResonanceNetwork;
```

**Implication:** ResonanceNetwork is imported but never used in QMEM building logic.

### File: `src/crates/sodals_neuro/src/semantic_network.rs`

#### ResonanceNetwork Structure (Lines 100-200)

```rust
// Lines 100-120 (from read results):
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QmemCartridge {
    pub n: usize,
    pub damping: Vec<f64>,
    pub masses: Vec<f64>,
    pub adjacency_rows: Vec<usize>,
    pub adjacency_cols: Vec<usize>,
    pub adjacency_vals: Vec<f64>,
    pub labels: HashMap<String, usize>,
    pub holographic_memory: Vec<Vec<Complex64>>,  // ← Complex64 phasors (NOT u8)
    pub node_flags: Vec<u32>,
}
```

**Key Finding:**
- ResonanceNetwork stores `Vec<Vec<Complex64>>` (holographic memory)
- QMEM loader returns `Vec<u8>` (raw bytes)
- **NO conversion path exists** between the two

---

## 4. CLIFFORD ALGEBRA: Orthogonal to QMEM

### File: `src/crates/sodals_neuro/src/arkhe_pipeline.rs`

#### CliffordBridge Structure (Lines 26-45)

```rust
/// CliffordBridge: Two ResonanceNetworks with Clifford Geometric Product
/// f(x) + g(x) → h(x) = Clifford_project(net_f, net_g, θ_z)
pub struct CliffordBridge {
    /// Last computed h_phase (radian)
    pub last_h_phase: f64,

    /// Last computed projection strength [0, 1]
    pub last_confidence: f64,

    /// Last found cross-network nodes (label, similarity_score)
    pub last_sacak: Vec<(String, f64)>,

    /// 4096-D VSA bound signature
    pub last_bound_signature: Vec<Complex64>,
}
```

#### CliffordBridge::project (Lines 56-150)

```rust
/// Clifford projection: f(x) + g(x) → h(x)
///
/// Two-layer computation:
/// 1. MACROSCOPIC: Weight aggregate phase with impedance
/// 2. VSA: 4096-D phasor binding with viscosity
pub fn project(
    &mut self,
    net_f: &ResonanceNetwork,    // ← Backbone Qwen output network
    net_g: &ResonanceNetwork,    // ← QMEM physics network (separate init)
    impedance_theta: f64,
) -> (f64, f64) {
    // Line 67-70: Get aggregate phases from each network
    let f_phase = net_f.compute_aggregate_phase();
    let g_phase = net_g.compute_aggregate_phase();

    // Line 72-80: Clifford geometric product in complex plane
    let e_f = Complex64::new(f_phase.cos(), f_phase.sin());
    let e_g = Complex64::new(
        (g_phase * impedance_theta).cos(),
        -(g_phase * impedance_theta).sin(),  // conjugate
    );
    let macro_projection = e_f * e_g;  // ← Clifford product
    let macro_h_phase = macro_projection.arg();
    let macro_confidence = macro_projection.norm().min(1.0);

    // ... VSA binding code ...
    // Line 100-120: Use holographic_memory from BOTH networks
    let top_f = Self::find_top_energy_node(net_f);
    let top_g = Self::find_top_energy_node(net_g);
    
    let vsa_result = top_f.and_then(|id_f| {
        top_g.and_then(|id_g| {
            let sig_f = net_f.holographic_memory.get(id_f)?;
            let sig_g = net_g.holographic_memory.get(id_g)?;
            // ... bind_phasors operation ...
        })
    });
}
```

**Key Finding:**
- **Line 67-70**: CliffordBridge works with abstract semantic PHASES, not raw bytes
- **Line 100-120**: Uses ResonanceNetwork.holographic_memory (Complex64)
- **QMEM loader** returns `Vec<u8>` — **INCOMPATIBLE**
- `net_g` is initialized separately; no observed coupling to QMEM loader

---

## 5. GRADIENT FLOW: ABSENT

### File: `src/crates/sodals_neuro/src/models/qwen_custom.rs`

#### QMEM Bias Computation (Line 895)

```rust
// Line 895: NO Tensor intermediate, pure Rust arithmetic
let signal = layer_bytes.iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64));

// This is Rust u64 arithmetic, not candle_core::Tensor
// Therefore: NO autograd tracking, NO gradient flow
```

**Why No Gradients:**
1. `layer_bytes` is `Vec<u8>` — not a Tensor
2. Fold operation is raw Rust arithmetic — not a Tensor operation
3. Result added to Tensor (line 899), but the computation path is non-differentiable

---

## 6. EPHEMERAL SESSION-BASED STATE: Routing Statistics Only

### File: `src/crates/sodals_neuro/src/phase45_gpr_qmem_router.rs`

#### LayerRoutingStats (Lines 20-50)

```rust
#[derive(Debug, Clone)]
pub struct LayerRoutingStats {
    /// Layer index (0 to num_layers-1)
    pub layer_idx: usize,
    
    /// EMA of bypass decisions (0.0 = always runs, 1.0 = always bypassed)
    pub bypass_rate_ema: f32,
    
    /// Number of forward passes through this layer
    pub forward_count: u32,
    
    /// EMA of z_i magnitudes (phase interference)
    pub z_i_ema: f32,
    
    /// EMA of semantic divergence
    pub semantic_divergence_ema: f32,
    
    /// EMA of Jacobian norm
    pub jacobian_norm_ema: f32,
    
    /// Loading priority for QMEM (0.0 = prefer mmap, 1.0 = prefer RAM)
    pub ram_loading_priority: f32,
    
    /// Estimated memory savings from bypassing (in bytes)
    pub estimated_memory_saved: u64,
}

impl LayerRoutingStats {
    pub fn new(layer_idx: usize) -> Self {
        Self {
            layer_idx,
            bypass_rate_ema: 0.0,          // ← Initialized to 0
            forward_count: 0,              // ← No prior state
            z_i_ema: 0.0,
            semantic_divergence_ema: 0.0,
            jacobian_norm_ema: 0.0,
            ram_loading_priority: 1.0,    // ← All RAM at start
            estimated_memory_saved: 0,
        }
    }

    /// Update routing statistics from phase signal information
    pub fn update(&mut self, bypass_decision: bool, z_i: f32, 
                  semantic_divergence: f32, jacobian_norm: f32, alpha: f32) {
        self.forward_count = self.forward_count.saturating_add(1);

        // EMA update
        let target = if bypass_decision { 1.0 } else { 0.0 };
        self.bypass_rate_ema = (1.0 - alpha) * self.bypass_rate_ema + alpha * target;

        // Recompute RAM loading priority based on bypass rate
        self.ram_loading_priority = 1.0 - self.bypass_rate_ema;
    }
}
```

**Ephemeral Properties:**
- **Line 40**: `new()` initializes all stats to 0 (NO persistence)
- **Line 51-60**: `update()` uses EMA (exponential moving average)
- **Line 52-54**: Reset on each session start

**Why Ephemeral:**
- Statistics stored in `Arc<Mutex<HashMap<usize, LayerRoutingStats>>>`
- Map is never serialized to disk
- Session restart → all `new()` calls → stats reset to 0.0

---

## 7. FORWARD PASS: Model Entry Point

### File: `src/crates/sodals_neuro/src/models/qwen_custom.rs`

#### Qwen2Model::forward (Lines 1320-1410)

```rust
// Line 1320-1330: Function signature
fn forward(
    &self,
    input_ids: &Tensor,
    seqlen_offset: usize,
    poles_ptr: Option<u64>,
    impedance_ptr: Option<u64>,
    flags_ptr: Option<u64>,
    n_nodes: Option<i32>,
    phase: Option<&PhaseSignal>,
    ghost_prefix: Option<&Tensor>,
    token_concept_map_ptr: Option<u64>,
    qmem_loader: Option<&Arc<QmemLoader>>,  // ← Line 1330: QMEM loader parameter
) -> Result<Tensor> {
    let mut chunk_xs = input_ids.clone();
    
    // Line 1393-1410: Pass QMEM loader to each layer
    for (idx, layer) in self.layers.iter().enumerate() {
        chunk_xs = layer.forward(
            &chunk_xs,
            attention_mask.as_ref(),
            seqlen_offset,
            adapter.as_ref(),
            poles_ptr,
            impedance_ptr,
            flags_ptr,
            token_concept_map_ptr,
            idx,
            self.num_layers,
            n_nodes,
            phase,
            qmem_loader,  // ← Line 1406: Passed to layer forward
            gpr_qmem_router.as_ref(),
        )?;
    }
}
```

#### Qwen2LanguageModel::generate (Line 1886)

```rust
// Line 1886: QMEM loader passed from LLM engine
let hidden_states = self.model.forward(
    input_ids, seqlen_offset, poles_ptr, impedance_ptr, flags_ptr, 
    n_nodes, phase, ghost_prefix, token_concept_map_ptr, 
    self.qmem_loader.as_ref()  // ← Line 1886: From self.qmem_loader
)?;
```

---

## 8. QMEM LOADER INITIALIZATION

### File: `src/crates/sodals_neuro/src/engine/llm_engine.rs`

#### Initialization (Lines 433-441)

```rust
// Line 433-441: init_qmem_loader function
pub fn init_qmem_loader(&mut self, manifest_path: &std::path::Path, max_ram_bytes: usize) -> anyhow::Result<()> {
    let qmem_loader = QmemLoader::new(QmemLoaderConfig {
        ram_first: true,
        max_ram_bytes,
        device: self.device.clone(),
    });
    
    qmem_loader.init_from_manifest(manifest_path)?;
    
    self.qmem_loader = Some(Arc::new(qmem_loader));
    Ok(())
}
```

**Key Points:**
- **Line 434-438**: QmemLoaderConfig with RAM preference
- **Line 439**: Loads manifest (JSON) with layer metadata
- **Line 441**: Stores in model (Arc-wrapped for sharing across layers)

---

## 9. SEMANTIC DIVERGENCE USAGE (Indirect QMEM Effect)

### File: `src/crates/sodals_neuro/src/models/qwen_custom.rs`

#### PhaseSignal Structure (Lines 440-460)

```rust
pub struct PhaseSignal {
    pub theta: f32,
    pub semantic_divergence: f32,    // ← Quality metric [0, 1]
    pub jacobian_norm: f32,
}

impl PhaseSignal {
    pub fn new(theta: f32, semantic_divergence: f32, jacobian_norm: f32) -> Self {
        Self {
            theta,
            semantic_divergence: semantic_divergence.clamp(0.0, 1.0),
            jacobian_norm,
        }
    }
}
```

#### Semantic Divergence Used in Attention (Lines 919-926)

```rust
// Line 919-926: Semantic Divergence Gate
if let Some(p) = phase {
    if p.semantic_divergence < 1.0 {
        let penalty = (1.0 - p.semantic_divergence) * -10.0;
        attention_scores = (attention_scores + penalty as f64)?;
    }
}
```

**Indirect QMEM Effect:**
1. QMEM bias → slightly different attention output
2. Different attention → marginally different hidden states
3. Different hidden states → RK4 pondering produces different semantic_divergence
4. semantic_divergence influences attention penalty (line 921)

**But:** This is **NOT direct QMEM → reasoning** coupling; it's probabilistic downstream effect.

---

## 10. NO DIRECT QMEM → RESONANCE NETWORK COUPLING

### Evidence Summary

| File | Line | What's Missing |
|------|------|----------------|
| qmem_loader/mod.rs | 204 | No conversion `Vec<u8>` → `Complex64` |
| qmem_builder.rs | 21 | Unused ResonanceNetwork import |
| semantic_network.rs | 100 | holographic_memory never populated from QMEM |
| arkhe_pipeline.rs | 67 | CliffordBridge gets phases from ResonanceNetwork, not QMEM |
| qwen_custom.rs | 892 | QMEM bytes hashed directly, no semantic processing |

---

## 11. COMPLETE FUNCTION CALL CHAIN

```
Token Input
    ↓
Qwen2LanguageModel::generate() [qwen_custom.rs:1850+]
    ↓
Qwen2Model::forward() [qwen_custom.rs:1320]
    └─ qmem_loader passed: line 1406
    ↓
    for layer in self.layers [line 1393]
        ↓
        QDecoderLayer::forward() [qwen_custom.rs:1137]
            └─ qmem_loader passed: line 1150
            ↓
            QAttention::forward() [qwen_custom.rs:707]
                └─ qmem_loader.read_layer_bytes() [qmem_loader.rs:204]
                    ↓ Returns Vec<u8>
                    ↓
                    signal = byte_fold(u64) [qwen_custom.rs:895]
                    ↓
                    bias = (signal % 1024) × 1e-4 [qwen_custom.rs:897]
                    ↓
                    attention_scores += bias [qwen_custom.rs:899]
    ↓
Output Logits
```

---

## Summary: Code Location Quick Reference

| Task | File | Line(s) |
|------|------|---------|
| **QMEM reads bytes** | qmem_loader/mod.rs | 204-226 |
| **QMEM applies bias** | qwen_custom.rs | 892-909 |
| **Unused ResonanceNetwork** | qmem_builder.rs | 21 |
| **CliffordBridge (orthogonal)** | arkhe_pipeline.rs | 22-150 |
| **GPR Routing Stats (ephemeral)** | phase45_gpr_qmem_router.rs | 20-60 |
| **Forward pass entry** | qwen_custom.rs | 1320-1410 |
| **Init QMEM loader** | llm_engine.rs | 433-441 |
| **Semantic divergence effect** | qwen_custom.rs | 919-926 |

