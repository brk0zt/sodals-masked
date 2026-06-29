// PHASE 1: QMEM Byte Tensor Conversion Plan
// ════════════════════════════════════════════════════════════════════════════════
// Goal: Convert QMEM from static byte cache → Learnable Tensor with Gradient Support
// Timeline: ~5-7 days
// ════════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// FILE 1: src/crates/sodals_neuro/src/qmem_loader/mod.rs
// ═══════════════════════════════════════════════════════════════════════════════

// CHANGE 1.1: Add learnable parameter wrapper struct
/*
BEFORE (current - read-only bytes):
    pub fn read_layer_bytes(&self, layer_idx: usize) -> Result<Vec<u8>> {
        // Returns raw bytes
        // Hash folding happens later in qwen_custom.rs
    }

AFTER (new - learnable tensor):
    /// Learnable QMEM parameter for a layer
    pub struct QmemParameter {
        /// Original read-only cartridge tensor (immutable reference)
        cartridge_base: Tensor,
        
        /// Learned delta/adjustment tensor (trainable, requires_grad=true)
        /// Shape matches cartridge_base
        /// Initially zeros, accumulated during training
        learned_delta: Tensor,
        
        /// Enable gradient computation
        requires_grad: bool,
        
        /// Optimizer state (Adam moments)
        m_t: Option<Tensor>,  // First moment (mean)
        v_t: Option<Tensor>,  // Second moment (variance)
        t: u64,               // Timestep for bias correction
    }

    impl QmemParameter {
        pub fn new(cartridge_base: Tensor, device: &Device) -> Result<Self> {
            let shape = cartridge_base.shape();
            let learned_delta = Tensor::zeros(shape, cartridge_base.dtype(), device)?;
            
            Ok(Self {
                cartridge_base,
                learned_delta,
                requires_grad: true,
                m_t: None,
                v_t: None,
                t: 0,
            })
        }
        
        /// Get effective parameter: cartridge_base + learned_delta
        pub fn get_effective_tensor(&self) -> Result<Tensor> {
            (&self.cartridge_base + &self.learned_delta)
        }
        
        /// Gradient descent step (Adam optimizer)
        pub fn adam_step(
            &mut self,
            gradient: &Tensor,
            learning_rate: f32,
            beta1: f32,  // 0.9
            beta2: f32,  // 0.999
            eps: f32,    // 1e-8
        ) -> Result<()> {
            self.t += 1;
            let t = self.t as f32;
            
            // Initialize moments if first time
            if self.m_t.is_none() {
                self.m_t = Some(Tensor::zeros_like(&self.learned_delta)?);
                self.v_t = Some(Tensor::zeros_like(&self.learned_delta)?);
            }
            
            // m_t = beta1 * m_{t-1} + (1-beta1) * grad
            let m_t = self.m_t.as_ref().unwrap();
            let new_m = (
                &((m_t * beta1 as f64)? + &((gradient * (1.0 - beta1) as f64)?)?)?
            )?;
            
            // v_t = beta2 * v_{t-1} + (1-beta2) * grad²
            let v_t = self.v_t.as_ref().unwrap();
            let grad_sq = gradient.sqr()?;
            let new_v = (
                &((v_t * beta2 as f64)? + &((grad_sq * (1.0 - beta2) as f64)?)?)?
            )?;
            
            // Bias correction
            let m_hat = (&new_m * (1.0 / (1.0 - beta1.powf(t)))?)? ;
            let v_hat = (&new_v * (1.0 / (1.0 - beta2.powf(t)))?)? ;
            
            // Update: delta = delta - lr * m_hat / (sqrt(v_hat) + eps)
            let denom = (&(v_hat.sqrt()?)? + eps)?;
            let update = (&m_hat / &denom)?;
            self.learned_delta = (&self.learned_delta - &((update * learning_rate as f64)?)?)?;
            
            self.m_t = Some(new_m);
            self.v_t = Some(new_v);
            
            Ok(())
        }
    }
```

    pub fn get_layer_tensor(&self, layer_idx: usize) -> Result<QmemParameter> {
        // NEW METHOD: Returns QmemParameter (trainable) instead of raw bytes
        let mut layers = self.layers.lock().unwrap();
        let layer = layers.get_mut(&layer_idx)
            .with_context(|| format!("QMEM layer {} not found", layer_idx))?;

        if layer.tensor.is_none() && layer.mmap.is_none() {
            if self.config.ram_first {
                self.load_layer_into_ram(layer).or_else(|_| self.load_layer_via_mmap(layer))?;
            } else {
                self.load_layer_via_mmap(layer)?;
            }
        }

        let tensor = if let Some(ref t) = layer.tensor {
            t.clone()
        } else if let Some(ref mmap) = layer.mmap {
            let start = layer.offset as usize;
            let end = start.checked_add(layer.length)
                .with_context(|| format!("QMEM layer {} size invalid", layer_idx))?;
            Tensor::from_slice(&mmap[start..end], (layer.length,), &self.config.device)?
        } else {
            return Err(anyhow::anyhow!("QMEM layer {} unavailable", layer_idx));
        };

        // Convert to f32 for gradient support
        let tensor_f32 = tensor.to_dtype(DType::F32)?;
        QmemParameter::new(tensor_f32, &self.config.device)
    }
*/

// ═══════════════════════════════════════════════════════════════════════════════
// FILE 2: src/crates/sodals_neuro/src/models/qwen_custom.rs
// ═══════════════════════════════════════════════════════════════════════════════

// CHANGE 2.1: Add QmemParameter storage to Qwen2Model
/*
BEFORE:
    pub struct Qwen2Model {
        // ... existing fields ...
        pub qmem_loader: Option<Arc<QmemLoader>>,
        pub gpr_qmem_router: Arc<Phase45GprQmemRouter>,
    }

AFTER:
    pub struct Qwen2Model {
        // ... existing fields ...
        pub qmem_loader: Option<Arc<QmemLoader>>,
        pub gpr_qmem_router: Arc<Phase45GprQmemRouter>,
        
        // Phase 50: Trainable QMEM parameters per layer
        /// HashMap: layer_idx → QmemParameter
        /// Initialized in new(), updated during backprop
        pub qmem_parameters: Arc<Mutex<HashMap<usize, qmem_loader::QmemParameter>>>,
    }
*/

// CHANGE 2.2: Initialize QMEM parameters in Qwen2Model::new()
/*
BEFORE:
    impl Qwen2Model {
        pub fn new(/* ... */) -> Result<Self> {
            // ... existing initialization ...
            Ok(Self {
                // ... existing fields ...
                qmem_loader: qmem_loader,
                gpr_qmem_router: Arc::new(...),
            })
        }
    }

AFTER:
    impl Qwen2Model {
        pub fn new(/* ... */) -> Result<Self> {
            // ... existing initialization ...
            
            // Phase 50: Initialize QMEM parameters if loader present
            let qmem_parameters = if let Some(ref loader) = qmem_loader {
                let mut params = HashMap::new();
                let num_layers = loader.num_layers();
                
                for layer_idx in 0..num_layers {
                    match loader.get_layer_tensor(layer_idx) {
                        Ok(param) => {
                            params.insert(layer_idx, param);
                            eprintln!("[QMEM] Layer {} parameter initialized (trainable)", layer_idx);
                        }
                        Err(e) => {
                            eprintln!("[QMEM] Warning: Layer {} initialization failed: {}", layer_idx, e);
                        }
                    }
                }
                Arc::new(Mutex::new(params))
            } else {
                Arc::new(Mutex::new(HashMap::new()))
            };
            
            Ok(Self {
                // ... existing fields ...
                qmem_loader,
                gpr_qmem_router: Arc::new(...),
                qmem_parameters,  // NEW
            })
        }
    }
*/

// CHANGE 2.3: Use QMEM parameters in attention forward pass
/*
BEFORE (in QAttention::forward):
    // Line ~895: Hash-based bias computation
    let color_seed = color.wrapping_mul(0x9e3779b9).wrapping_add(0x85ebca6b);
    let bias = ((hash % 1024) as f32 * 1e-4) - 0.0512;
    attention_scores = (attention_scores.to_f32()? + bias)?;

AFTER:
    // Get learnable QMEM parameter for this layer
    if let Some(ref qmem_params) = self.qmem_parameters {
        let params = qmem_params.lock().unwrap();
        if let Some(param) = params.get(&layer_idx) {
            // Get effective parameter tensor (cartridge + learned_delta)
            let effective_qmem = param.get_effective_tensor()?;
            
            // Project QMEM tensor to attention score space [batch, heads, seq_q, seq_k]
            // Simple projection: hash current q,k tensors + project QMEM
            let qmem_attention_bias = self.project_qmem_to_attention(
                &effective_qmem,
                &qs_r,
                &k_cache,
                b_size,
                seq_len,
                seq_k,
            )?;
            
            // Add learnable QMEM bias to attention scores
            attention_scores = (&attention_scores + &qmem_attention_bias)?;
        }
    }
*/

// CHANGE 2.4: Add projection helper method to QAttention
/*
    fn project_qmem_to_attention(
        &self,
        qmem_tensor: &Tensor,
        q: &Tensor,  // [batch, num_heads, seq_q, head_dim]
        k: &Tensor,  // [batch, num_kv_heads, seq_k, head_dim]
        batch: usize,
        seq_len: usize,
        seq_k: usize,
    ) -> Result<Tensor> {
        // Map QMEM dimensions to attention space
        // QMEM shape: typically [4096] or layer_size
        // Attention shape: [batch, num_heads, seq_q, seq_k]
        
        // Simple projection: hash QMEM to attention grid
        let qmem_size = qmem_tensor.dim(0)?;
        let mut bias_grid = vec![0.0f32; batch * self.num_heads * seq_len * seq_k];
        
        for i in 0..qmem_size {
            let qmem_val = qmem_tensor.get(i)?.to_f32() as f32;
            
            // Distribute QMEM values across attention positions
            let batch_idx = i % batch;
            let head_idx = (i / batch) % self.num_heads;
            let q_idx = (i / (batch * self.num_heads)) % seq_len;
            let k_idx = (i / (batch * self.num_heads * seq_len)) % seq_k;
            
            let pos = batch_idx * (self.num_heads * seq_len * seq_k) +
                      head_idx * (seq_len * seq_k) +
                      q_idx * seq_k +
                      k_idx;
            
            bias_grid[pos] = qmem_val * 0.01;  // Scale down for stability
        }
        
        Tensor::from_vec(
            bias_grid,
            (batch, self.num_heads, seq_len, seq_k),
            q.device()
        )
    }
*/

// ═══════════════════════════════════════════════════════════════════════════════
// FILE 3: New file - src/crates/sodals_neuro/src/qmem_optimizer.rs
// ═══════════════════════════════════════════════════════════════════════════════

/*
pub struct QmemOptimizer {
    /// Learning rate for QMEM parameter updates
    learning_rate: f32,
    /// Beta1 for Adam optimizer (default 0.9)
    beta1: f32,
    /// Beta2 for Adam optimizer (default 0.999)
    beta2: f32,
    /// Epsilon for numerical stability (default 1e-8)
    eps: f32,
    /// L2 regularization weight (prevent divergence)
    weight_decay: f32,
}

impl QmemOptimizer {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-4,
        }
    }
    
    /// Apply gradient updates to all QMEM parameters
    pub fn step(
        &self,
        qmem_parameters: &Arc<Mutex<HashMap<usize, qmem_loader::QmemParameter>>>,
        gradients: &HashMap<usize, Tensor>,  // layer_idx → gradient tensor
    ) -> Result<()> {
        let mut params = qmem_parameters.lock().unwrap();
        
        for (layer_idx, param) in params.iter_mut() {
            if let Some(gradient) = gradients.get(layer_idx) {
                // L2 regularization: add weight decay to gradient
                let reg_gradient = (
                    gradient + &(param.learned_delta.clone() * self.weight_decay as f64)?
                )?;
                
                // Apply Adam step
                param.adam_step(&reg_gradient, self.learning_rate, self.beta1, self.beta2, self.eps)?;
            }
        }
        
        Ok(())
    }
    
    /// Save optimized QMEM parameters to disk
    pub fn save_checkpoint(
        &self,
        qmem_parameters: &Arc<Mutex<HashMap<usize, qmem_loader::QmemParameter>>>,
        path: &Path,
    ) -> Result<()> {
        let params = qmem_parameters.lock().unwrap();
        
        let mut checkpoint = serde_json::json!({
            "qmem_parameters": {},
        });
        
        for (layer_idx, param) in params.iter() {
            let learned_delta_vec: Vec<f32> = param.learned_delta.to_vec1()?;
            checkpoint["qmem_parameters"][layer_idx.to_string()] = serde_json::json!({
                "learned_delta": learned_delta_vec,
            });
        }
        
        std::fs::write(path, serde_json::to_string_pretty(&checkpoint)?)?;
        eprintln!("[QMEM] Saved optimized parameters to {}", path.display());
        
        Ok(())
    }
}
*/

// ═══════════════════════════════════════════════════════════════════════════════
// Implementation Order
// ═══════════════════════════════════════════════════════════════════════════════
/*
Step 1: Create QmemParameter struct and add to qmem_loader/mod.rs (Day 1-2)
Step 2: Add get_layer_tensor() method to QmemLoader (Day 2)
Step 3: Add qmem_parameters field to Qwen2Model (Day 2-3)
Step 4: Initialize QMEM parameters in Qwen2Model::new() (Day 3)
Step 5: Modify QAttention::forward to use QmemParameter (Day 3-4)
Step 6: Add project_qmem_to_attention() helper (Day 4)
Step 7: Create qmem_optimizer.rs module (Day 4-5)
Step 8: Integration tests (Day 5-6)
Step 9: Benchmarking WITH/WITHOUT (Day 6-7)

Testing Checklist:
  [ ] Unit test: QmemParameter creation
  [ ] Unit test: adam_step() convergence
  [ ] Integration test: Full forward pass with QMEM
  [ ] Integration test: Gradient computation
  [ ] Benchmark: Performance comparison (with/without)
  [ ] Benchmark: Memory usage (learnable params)
  [ ] Accuracy test: FailureSensorIQ before/after
*/
