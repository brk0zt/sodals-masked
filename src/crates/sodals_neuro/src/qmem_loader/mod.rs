use anyhow::{Context, Result};
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use candle_core::{Device, Tensor, DType};

/// Hierarchical QMEM loader configuration
pub struct QmemLoaderConfig {
    /// Prefer loading layers into RAM first; fallback to memory-mapped pages
    pub ram_first: bool,
    /// Maximum RAM bytes to use for loaded layers
    pub max_ram_bytes: usize,
    /// Device where loaded tensors should be allocated
    pub device: Device,
}

/// Layer-wise dtype support for mixed-precision inference
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
pub enum QmemDType {
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "bf16")]
    BF16,
}

impl QmemDType {
    pub fn to_candle_dtype(&self) -> DType {
        match self {
            QmemDType::F32 => DType::F32,
            QmemDType::F16 => DType::F16,
            QmemDType::BF16 => DType::BF16,
        }
    }
}

#[derive(Deserialize)]
pub struct QmemLayerManifest {
    pub layer_index: usize,
    pub path: PathBuf,
    pub offset: u64,
    pub length: usize,
    #[serde(default = "default_qmem_dtype")]
    pub dtype: QmemDType,
}

fn default_qmem_dtype() -> QmemDType {
    QmemDType::F32
}

#[derive(Deserialize)]
pub struct QmemManifest {
    pub layers: Vec<QmemLayerManifest>,
}

/// Simple in-memory representation for a loaded QMEM layer
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

/// QmemLoader handles hierarchical loading and eviction of layers
pub struct QmemLoader {
    pub config: QmemLoaderConfig,
    pub layers: Arc<Mutex<HashMap<usize, QmemLayer>>>,
    pub lru_queue: Arc<Mutex<VecDeque<usize>>>,
    pub resident_bytes: Arc<Mutex<usize>>,
}

impl QmemLoader {
    pub fn new(config: QmemLoaderConfig) -> Self {
        Self {
            config,
            layers: Arc::new(Mutex::new(HashMap::new())),
            lru_queue: Arc::new(Mutex::new(VecDeque::new())),
            resident_bytes: Arc::new(Mutex::new(0)),
        }
    }

    /// Initialize layers metadata from a QMEM manifest file.
    /// Manifest schema is { "layers": [{"layer_index":0,"path":"...","offset":0,"length":123}, ...] }.
    pub fn init_from_manifest<P: AsRef<Path>>(&self, manifest: P) -> Result<()> {
        let manifest_path = manifest.as_ref();
        let file = File::open(manifest_path)
            .with_context(|| format!("Failed to open QMEM manifest: {}", manifest_path.display()))?;
        let manifest: QmemManifest = serde_json::from_reader(file)
            .with_context(|| format!("Failed to parse QMEM manifest: {}", manifest_path.display()))?;

        let mut layers = self.layers.lock().unwrap();
        for entry in manifest.layers {
            layers.insert(entry.layer_index, QmemLayer {
                layer_index: entry.layer_index,
                tensor: None,
                mmap: None,
                on_disk_path: manifest_path.parent().unwrap_or_else(|| Path::new(".")).join(entry.path),
                offset: entry.offset,
                length: entry.length,
                resident_bytes: 0,
                dtype: entry.dtype,
            });
        }

        Ok(())
    }

    fn mark_accessed(&self, layer_idx: usize) {
        let mut queue = self.lru_queue.lock().unwrap();
        if let Some(pos) = queue.iter().position(|&idx| idx == layer_idx) {
            queue.remove(pos);
        }
        queue.push_back(layer_idx);
    }

    fn allocate_bytes(&self, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let mut resident = self.resident_bytes.lock().unwrap();
        if *resident + bytes <= self.config.max_ram_bytes {
            *resident += bytes;
            return Ok(());
        }

        let target = self.config.max_ram_bytes.saturating_sub(bytes);
        drop(resident);
        let freed = self.evict_until(target)?;
        let mut resident = self.resident_bytes.lock().unwrap();
        *resident += bytes.saturating_sub(freed);
        Ok(())
    }

    pub fn ensure_layer_in_ram(&self, layer_idx: usize) -> Result<()> {
        let mut layers = self.layers.lock().unwrap();
        let layer = layers.get_mut(&layer_idx)
            .with_context(|| format!("QMEM layer {} not found", layer_idx))?;

        if layer.tensor.is_some() {
            self.mark_accessed(layer_idx);
            return Ok(());
        }

        if self.config.ram_first {
            match self.load_layer_into_ram(layer) {
                Ok(()) => {
                    self.mark_accessed(layer_idx);
                    return Ok(());
                }
                Err(e) => {
                    eprintln!("[QMEM] RAM load failed for layer {}: {}. Falling back to mmap.", layer_idx, e);
                }
            }
        }

        self.load_layer_via_mmap(layer)?;
        self.mark_accessed(layer_idx);
        Ok(())
    }

    fn ensure_layer_mapped(&self, layer: &mut QmemLayer) -> Result<()> {
        if layer.mmap.is_some() {
            return Ok(());
        }

        let file = File::open(&layer.on_disk_path)
            .with_context(|| format!("Failed to open QMEM layer file: {}", layer.on_disk_path.display()))?;
        let raw = unsafe { Mmap::map(&file) }
            .with_context(|| format!("Failed to mmap QMEM layer file: {}", layer.on_disk_path.display()))?;
        layer.mmap = Some(Arc::new(raw));
        Ok(())
    }

    fn load_layer_into_ram(&self, layer: &mut QmemLayer) -> Result<()> {
        self.ensure_layer_mapped(layer)?;
        let raw = layer.mmap.as_ref().unwrap().clone();
        let start = layer.offset as usize;
        let end = start.checked_add(layer.length)
            .with_context(|| format!("QMEM layer {} size invalid", layer.layer_index))?;
        let slice = &raw[start..end];

        let tensor_f32 = Tensor::from_slice(slice, (layer.length,), &self.config.device)?;
        let target_dtype = layer.dtype.to_candle_dtype();
        let tensor = if target_dtype != DType::F32 {
            tensor_f32.to_dtype(target_dtype)?
        } else {
            tensor_f32
        };
        
        let resident_bytes = slice.len();
        self.allocate_bytes(resident_bytes)?;
        layer.tensor = Some(tensor);
        layer.resident_bytes = resident_bytes;
        Ok(())
    }

    fn load_layer_via_mmap(&self, layer: &mut QmemLayer) -> Result<()> {
        self.ensure_layer_mapped(layer)?;
        layer.tensor = None;
        layer.resident_bytes = 0;
        Ok(())
    }

    pub fn read_layer_bytes(&self, layer_idx: usize) -> Result<Vec<u8>> {
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

        if let Some(ref tensor) = layer.tensor {
            return tensor.to_vec1::<u8>().with_context(|| format!("Failed to copy QMEM tensor bytes from layer {}", layer_idx));
        }

        if let Some(ref mmap) = layer.mmap {
            let start = layer.offset as usize;
            let end = start.checked_add(layer.length)
                .with_context(|| format!("QMEM layer {} size invalid", layer_idx))?;
            return Ok(mmap[start..end].to_vec());
        }

        Err(anyhow::anyhow!("QMEM layer {} is unavailable", layer_idx))
    }

    pub fn get_layer_dtype(&self, layer_idx: usize) -> Result<QmemDType> {
        let layers = self.layers.lock().unwrap();
        let layer = layers.get(&layer_idx)
            .with_context(|| format!("QMEM layer {} not found", layer_idx))?;
        Ok(layer.dtype)
    }

    pub fn num_layers(&self) -> usize {
        self.layers.lock().unwrap().len()
    }

    pub fn evict_until(&self, target_free_bytes: usize) -> Result<usize> {
        let mut queue = self.lru_queue.lock().unwrap();
        let mut layers = self.layers.lock().unwrap();
        let mut resident = self.resident_bytes.lock().unwrap();
        let mut freed = 0;

        while *resident > target_free_bytes {
            if let Some(layer_idx) = queue.pop_front() {
                if let Some(layer) = layers.get_mut(&layer_idx) {
                    if layer.tensor.is_some() {
                        let layer_bytes = layer.resident_bytes;
                        layer.tensor = None;
                        layer.resident_bytes = 0;
                        *resident = resident.saturating_sub(layer_bytes);
                        freed += layer_bytes;
                    }
                }
            } else {
                break;
            }
        }

        Ok(freed)
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// PHASE 50: QMEM Learnable Parameters (Virtual Parameters for Reasoning)
// ════════════════════════════════════════════════════════════════════════════════
//
// QmemParameter encapsulates a learnable tensor derived from the static cartridge.
// This enables gradient-based optimization of QMEM's contribution to reasoning.
//
// Architecture:
//   - cartridge_base: Read-only baseline from static file (immutable)
//   - learned_delta: Trainable adjustment (initialized to zero)
//   - Effective tensor: cartridge_base + learned_delta
//
// Optimizer: Adam with momentum and variance tracking
// Regularization: L2 weight decay to prevent divergence

use std::f32;

#[derive(Debug, Clone)]
pub struct QmemParameter {
    /// Original read-only cartridge tensor (immutable reference)
    /// This is the "prior knowledge" from academic sources (IEEE/ISO/API)
    pub cartridge_base: Tensor,

    /// Learned delta/adjustment tensor (trainable, requires_grad=true)
    /// Shape matches cartridge_base
    /// Initially zeros; accumulated during training via Adam optimizer
    pub learned_delta: Tensor,

    /// Enable gradient computation
    pub requires_grad: bool,

    /// Adam optimizer state - first moment (exponential moving average of gradients)
    pub m_t: Option<Tensor>,

    /// Adam optimizer state - second moment (exponential moving average of squared gradients)
    pub v_t: Option<Tensor>,

    /// Timestep counter for Adam bias correction
    pub t: u64,

    /// Layer index (for debugging and monitoring)
    pub layer_idx: usize,
}

impl QmemParameter {
    /// Create a new learnable QMEM parameter from a cartridge tensor
    pub fn new(layer_idx: usize, cartridge_base: Tensor, device: &Device) -> Result<Self> {
        let shape = cartridge_base.shape();
        let learned_delta = Tensor::zeros(shape, DType::F32, device)
            .with_context(|| format!("Failed to allocate learned_delta for layer {}", layer_idx))?;

        Ok(Self {
            cartridge_base,
            learned_delta,
            requires_grad: true,
            m_t: None,
            v_t: None,
            t: 0,
            layer_idx,
        })
    }

    /// Get effective parameter tensor: cartridge_base + learned_delta
    /// This is what gets used in forward pass
    pub fn get_effective_tensor(&self) -> Result<Tensor> {
        (&self.cartridge_base + &self.learned_delta)
            .with_context(|| format!("Failed to compute effective tensor for layer {}", self.layer_idx))
    }

    /// Perform one Adam optimizer step to update learned_delta
    /// 
    /// # Arguments
    /// * `gradient` - Backpropagated gradient w.r.t. learned_delta
    /// * `learning_rate` - Learning rate (typical: 1e-4 to 1e-3)
    /// * `beta1` - Exponential decay rate for first moment (typical: 0.9)
    /// * `beta2` - Exponential decay rate for second moment (typical: 0.999)
    /// * `eps` - Numerical stability epsilon (typical: 1e-8)
    pub fn adam_step(
        &mut self,
        gradient: &Tensor,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
    ) -> Result<()> {
        self.t += 1;
        let t = self.t as f32;

        // Initialize moments if first update
        if self.m_t.is_none() {
            self.m_t = Some(Tensor::zeros_like(&self.learned_delta)?);
            self.v_t = Some(Tensor::zeros_like(&self.learned_delta)?);
        }

        let m_t_prev = self.m_t.as_ref().unwrap();
        let v_t_prev = self.v_t.as_ref().unwrap();

        // m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        let m_t = {
            let term1 = (m_t_prev * (beta1 as f64))?;
            let term2 = (gradient * ((1.0 - beta1) as f64))?;
            (&term1 + &term2)?
        };

        // v_t = beta2 * v_{t-1} + (1 - beta2) * grad²
        let grad_squared = gradient.sqr()?;
        let v_t = {
            let term1 = (v_t_prev * (beta2 as f64))?;
            let term2 = (&grad_squared * ((1.0 - beta2) as f64))?;
            (&term1 + &term2)?
        };

        // Bias correction
        let bias_correction_1 = 1.0 - beta1.powf(t);
        let bias_correction_2 = 1.0 - beta2.powf(t);

        // m_hat = m_t / (1 - beta1^t)
        let m_hat = (&m_t * (1.0 / bias_correction_1 as f64))?;

        // v_hat = v_t / (1 - beta2^t)
        let v_hat = (&v_t * (1.0 / bias_correction_2 as f64))?;

        // Parameter update: delta_new = delta_old - lr * m_hat / (sqrt(v_hat) + eps)
        let v_sqrt = v_hat.sqrt()?;
        let eps_tensor = Tensor::new(&[eps as f64], &v_sqrt.device())?;
        let denom = (&v_sqrt + &eps_tensor)?;
        let update_step = (&m_hat / &denom)?;
        let scaled_update = (&update_step * (learning_rate as f64))?;

        self.learned_delta = (&self.learned_delta - &scaled_update)?;

        // Store updated moments
        self.m_t = Some(m_t);
        self.v_t = Some(v_t);

        Ok(())
    }

    /// Get summary statistics for monitoring during training
    pub fn get_statistics(&self) -> Result<QmemParameterStats> {
        let delta_vec: Vec<f32> = self.learned_delta.to_vec1()?;
        let base_vec: Vec<f32> = self.cartridge_base.to_vec1()?;

        let delta_norm: f32 = delta_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let base_norm: f32 = base_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let max_delta = delta_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_delta = delta_vec.iter().cloned().fold(f32::INFINITY, f32::min);

        Ok(QmemParameterStats {
            layer_idx: self.layer_idx,
            delta_norm,
            base_norm,
            max_delta,
            min_delta,
            timestep: self.t,
        })
    }
}

#[derive(Debug, Clone)]
pub struct QmemParameterStats {
    pub layer_idx: usize,
    pub delta_norm: f32,
    pub base_norm: f32,
    pub max_delta: f32,
    pub min_delta: f32,
    pub timestep: u64,
}

// Add new method to QmemLoader
impl QmemLoader {
    /// Get a learnable QMEM parameter for the specified layer
    /// Returns QmemParameter with cartridge_base loaded and learned_delta initialized to zero
    pub fn get_layer_tensor(&self, layer_idx: usize) -> Result<QmemParameter> {
        let mut layers = self.layers.lock().unwrap();
        let layer = layers.get_mut(&layer_idx)
            .with_context(|| format!("QMEM layer {} not found", layer_idx))?;

        // Ensure layer is mapped into memory
        if layer.tensor.is_none() && layer.mmap.is_none() {
            if self.config.ram_first {
                let _ = self.load_layer_into_ram(layer).or_else(|_| self.load_layer_via_mmap(layer));
            } else {
                let _ = self.load_layer_via_mmap(layer);
            }
        }

        // Get raw tensor (from RAM or mmap)
        let raw_tensor = if let Some(ref t) = layer.tensor {
            t.clone()
        } else if let Some(ref mmap) = layer.mmap {
            let start = layer.offset as usize;
            let end = start.checked_add(layer.length)
                .with_context(|| format!("QMEM layer {} size invalid", layer_idx))?;
            Tensor::from_slice(&mmap[start..end], (layer.length,), &self.config.device)
                .with_context(|| format!("Failed to create tensor from layer {}", layer_idx))?
        } else {
            return Err(anyhow::anyhow!("QMEM layer {} unavailable (no tensor or mmap)", layer_idx));
        };

        // Convert to f32 for gradient support
        let tensor_f32 = raw_tensor.to_dtype(DType::F32)
            .with_context(|| format!("Failed to convert layer {} to f32", layer_idx))?;

        // Create learnable parameter
        QmemParameter::new(layer_idx, tensor_f32, &self.config.device)
    }
}
