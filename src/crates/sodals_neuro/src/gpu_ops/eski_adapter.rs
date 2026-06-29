// src/adapter.rs - Async Prefetching Sidecar CUDA Adapter
//
// SODALS Whitepaper II5: True Double Buffering III VRAM-backed DMA, no PCIe reads
//
// ARCHITECTURE:
//   CPU prepares layer N+1 impedance in pinned host RAM (staging),
//   then cudaMemcpyAsync DMA-copies it to a pre-allocated VRAM buffer.
//   The CUDA kernel reads impedance from VRAM (1000+ GB/s) not PCIe (32 GB/s).
//
//   NeuroAdapter holds TWO PinnedBuffers (double-buffer rotation):
//   - buffer[front]: currently being read by the GPU kernel
//   - buffer[back]:  being filled by CPU in parallel via DMA
//   After each layer the front/back indices are swapped.

use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_nn::{Embedding, Module};
use std::ffi::c_void;

use crate::gpu_ops::SodalsCudaAccess;

// ============================================================================
// FFI declarations for host_memory.cpp
// ============================================================================

#[cfg(feature = "cuda")]
extern "C" {
    fn alloc_pinned_memory(bytes: usize) -> *mut c_void;
    fn free_pinned_memory(ptr: *mut c_void);
    fn alloc_device_vram(bytes: usize) -> *mut c_void;
    fn free_device_vram(ptr: *mut c_void);
    /// Returns 0 on success (cudaSuccess).
    fn async_memcpy_to_device(
        host_src: *const c_void,
        dev_dst:  *mut c_void,
        bytes:    usize,
        stream:   *mut c_void,  // cudaStream_t; null III default stream
    ) -> i32;
    fn synchronize_stream(stream: *mut c_void);
    fn stream_pool_create(num_streams: i32) -> *mut c_void;
    fn stream_pool_destroy(pool: *mut c_void);
    fn stream_pool_get_next(pool: *mut c_void) -> *mut c_void;
    fn stream_pool_sync_all(pool: *mut c_void);
}

// ============================================================================
// ASYNC NEURO ADAPTER III double-buffer VRAM prefetch
// ============================================================================

pub struct NeuroAdapter {
    device_id:        u32,
    pub impedance_factor: f64,
    /// Two VRAM-backed staging buffers; CPU fills [back], GPU reads [front].
    buffers: [std::sync::Mutex<Option<PinnedBuffer>>; 2],
    /// Index into `buffers` that is currently safe for the GPU to read.
    front:   std::sync::atomic::AtomicUsize,
    /// Dedicated non-blocking CUDA stream for DMA transfers.
    #[cfg(feature = "cuda")]
    dma_stream: *mut c_void,
    #[cfg(not(feature = "cuda"))]
    _dma_stream: (),
}

// SAFETY: The raw stream pointer is owned by NeuroAdapter and
// never aliased across threads without synchronisation.
unsafe impl Send for NeuroAdapter {}
unsafe impl Sync for NeuroAdapter {}

impl NeuroAdapter {
    pub fn new(device_id: u32, impedance_factor: f64, is_cuda: bool) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        let dma_stream = if is_cuda {
            unsafe { stream_pool_create(1) }
        } else {
            std::ptr::null_mut()
        };

        Ok(Self {
            device_id,
            impedance_factor,
            buffers: [std::sync::Mutex::new(None), std::sync::Mutex::new(None)],
            front:   std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "cuda")]
            dma_stream,
            #[cfg(not(feature = "cuda"))]
            _dma_stream: (),
        })
    }

    /// Queue impedance for layer N+1 into the *back* VRAM buffer via DMA.
    /// Call this while the GPU is still computing layer N.
    pub fn prefetch_impedance(&self, data_f16: &[u16]) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let back = 1 - self.front.load(std::sync::atomic::Ordering::SeqCst);
        let mut guard = self.buffers[back].lock().map_err(|e| e.to_string())?;

        let bytes = data_f16.len() * 2; // each f16 = 2 bytes

        // Allocate / resize buffer as needed
        let buf = match guard.as_mut() {
            Some(b) if b.size >= bytes => b,
            _ => {
                *guard = Some(PinnedBuffer::new(bytes)?);
                guard.as_mut().unwrap()
            }
        };

        #[cfg(feature = "cuda")]
        {
            // Copy host data into pinned staging
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data_f16.as_ptr() as *const u8,
                    buf.host_ptr as *mut u8,
                    bytes,
                );
            }
            // Async DMA from pinned host III VRAM (non-blocking on host)
            let rc = unsafe {
                async_memcpy_to_device(
                    buf.host_ptr as *const c_void,
                    buf.vram_ptr as *mut c_void,
                    bytes,
                    self.dma_stream,
                )
            };
            if rc != 0 {
                return Err(format!("cudaMemcpyAsync failed with code {rc}").into());
            }
        }

        Ok(())
    }

    /// Wait for the DMA to the back buffer to complete, then swap front/back.
    /// Returns the VRAM device pointer that the GPU kernel should read.
    pub fn swap_and_get_vram_ptr(&self) -> std::result::Result<u64, Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        unsafe { synchronize_stream(self.dma_stream); }

        // Atomically promote back III front
        let prev = self.front.fetch_xor(1, std::sync::atomic::Ordering::SeqCst);
        let new_front = 1 - prev;

        let guard = self.buffers[new_front].lock().map_err(|e| e.to_string())?;
        match guard.as_ref() {
            Some(b) => Ok(b.vram_ptr),
            None    => Err("No VRAM buffer in front slot after swap".into()),
        }
    }

    /// Apply physics bias using the address already resident in VRAM.
    /// `iptr` must be a VRAM address (returned by `swap_and_get_vram_ptr`).
    pub fn apply_bias_raw(
        &self,
        wptr: u64,
        iptr: u64,
        b: i32, h: i32, q: i32, k: i32,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        unsafe {
            physics_mask::apply_mask(
                std::ptr::null_mut() as *mut c_void,
                wptr as *mut c_void,
                iptr as *const c_void,
                b, h, q, k,
            );
        }
        Ok(())
    }

    /// Block host until all async GPU ops on the default stream complete.
    pub fn synchronize(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        unsafe { synchronize_stream(std::ptr::null_mut()); }
        Ok(())
    }

    // Kept for callers that still pass a pre-computed Candle tensor impedance
    // directly as an F16 device pointer (existing path in qwen_custom.rs).
    pub fn apply_modulation_raw(
        &self, _act_ptr: u64, _imp_ptr: u64,
        _b: i32, _h: i32, _q: i32, _d: i32, _scaling: f32,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> { Ok(()) }
}

impl Drop for NeuroAdapter {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe { stream_pool_destroy(self.dma_stream); }
    }
}

// ============================================================================
// PINNED BUFFER III Pinned host staging + VRAM destination (Double-Buffer node)
// ============================================================================

struct PinnedBuffer {
    /// Write-combining pinned host RAM (staging area for DMA)
    host_ptr: *mut c_void,
    /// Pre-allocated VRAM buffer (final destination read by the CUDA kernel)
    vram_ptr: u64,
    /// Capacity in bytes
    size: usize,
}

// SAFETY: raw pointers are managed exclusively by the owning Mutex guard.
unsafe impl Send for PinnedBuffer {}

impl PinnedBuffer {
    fn new(bytes: usize) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        {
            let host_ptr = unsafe { alloc_pinned_memory(bytes) };
            if host_ptr.is_null() {
                return Err("cudaHostAlloc failed (out of pinned memory)".into());
            }
            let vram_raw = unsafe { alloc_device_vram(bytes) };
            if vram_raw.is_null() {
                unsafe { free_pinned_memory(host_ptr); }
                return Err("cudaMalloc for VRAM buffer failed".into());
            }
            Ok(Self { host_ptr, vram_ptr: vram_raw as u64, size: bytes })
        }
        #[cfg(not(feature = "cuda"))]
        Ok(Self { host_ptr: std::ptr::null_mut(), vram_ptr: 0, size: bytes })
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            if !self.host_ptr.is_null() { free_pinned_memory(self.host_ptr); }
            if self.vram_ptr != 0     { free_device_vram(self.vram_ptr as *mut c_void); }
        }
    }
}

// ============================================================================
// CONFIG
// ============================================================================

#[derive(Debug, Clone)]
pub struct Qwen2Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

// ============================================================================
// ROTARY EMBEDDING
// ============================================================================

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    dim: usize,
}

impl RotaryEmbedding {
    fn new(cfg: &Qwen2Config, device: &Device) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let theta    = cfg.rope_theta as f32;
        let inv_freq: Vec<_> = (0..head_dim).step_by(2)
            .map(|i| 1.0 / theta.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq.clone(), inv_freq.len(), device)?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?;
        // freqs: [max_pos, head_dim/2] III candle rope() expects cos/sin of this exact shape
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?.contiguous()?;
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;
        Ok(Self { sin, cos, dim: head_dim })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _n_head, seq_len, _head_dim) = q.dims4()?;
        // candle rope() expects: xs 4D [b, h, seq, head_dim], cos/sin 2D [seq, head_dim/2]
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?.contiguous()?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?.contiguous()?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ============================================================================
// RMS NORM
// ============================================================================

struct QRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl QRmsNorm {
    fn new(w: QTensor, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: w.dequantize(&w.device())?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n   = x.dim(D::Minus1)?;
        let x2  = x.sqr()?.sum_keepdim(D::Minus1)?;
        let x2s = (x2 / n as f64)?;
        let den = (x2s + self.eps)?.sqrt()?;
        let xn  = x.broadcast_div(&den)?;
        xn.broadcast_mul(&self.weight)
    }
}

// ============================================================================
// MLP
// ============================================================================

struct QMlp {
    gate: QMatMul,
    up:   QMatMul,
    down: QMatMul,
}

impl QMlp {
    fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        r: &mut R,
        i: usize,
        d: &Device,
    ) -> Result<Self> {
        Ok(Self {
            gate: QMatMul::from_qtensor(ct.tensor(r, &format!("blk.{i}.ffn_gate.weight"), d)?)?,
            up:   QMatMul::from_qtensor(ct.tensor(r, &format!("blk.{i}.ffn_up.weight"),   d)?)?,
            down: QMatMul::from_qtensor(ct.tensor(r, &format!("blk.{i}.ffn_down.weight"), d)?)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // FIX: QMatMul expects 2D input, but we have 3D [batch, seq_len, hidden_dim]
        // Reshape to 2D [batch*seq_len, hidden_dim], apply matmul, then reshape back
        let ORIGINAL_shape = x.shape();
        let (batch_size, seq_len, hidden_dim) = x.dims3()?;
        let x_2d = x.reshape((batch_size * seq_len, hidden_dim))?;
        
        let g   = self.gate.forward(&x_2d)?;
        let u   = self.up.forward(&x_2d)?;
        let ga  = candle_nn::ops::silu(&g)?;
        let hid = (ga * u)?;
        let out = self.down.forward(&hid)?;
        
        // Reshape back to 3D [batch_size, seq_len, hidden_dim]
        out.reshape((batch_size, seq_len, hidden_dim))
    }
}

// ============================================================================
// ATTENTION (with impedance-aligned physics kernel + fixed RoPE)
// ============================================================================

struct QAttention {
    pub q: QMatMul,
    pub q_bias: Option<candle_core::Tensor>,
    pub k: QMatMul,
    pub k_bias: Option<candle_core::Tensor>,
    pub v: QMatMul,
    pub v_bias: Option<candle_core::Tensor>,
    pub o: QMatMul,
    pub o_bias: Option<candle_core::Tensor>,
    pub heads: usize,
    pub kv_heads: usize,
    pub groups: usize,
    pub head_dim: usize,
    pub rope: Arc<RotaryEmbedding>,
    pub kv_cache: std::sync::Mutex<Option<(Tensor, Tensor)>>,
}

impl QAttention {
    pub fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        r: &mut R,
        cfg: &Qwen2Config,
        i: usize,
        rope: Arc<RotaryEmbedding>,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        
        let q = QMatMul::from_qtensor(ct.tensor(r, &format!("blk.{i}.attn_q.weight"),      device)?)?;
        let q_bias = ct.tensor(r, &format!("blk.{i}.attn_q.bias"), device).ok().and_then(|t| t.dequantize(device).ok());
        
        let k = QMatMul::from_qtensor(ct.tensor(r, &format!("blk.{i}.attn_k.weight"),      device)?)?;
        let k_bias = ct.tensor(r, &format!("blk.{i}.attn_k.bias"), device).ok().and_then(|t| t.dequantize(device).ok());
        
        let v = QMatMul::from_qtensor(ct.tensor(r, &format!("blk.{i}.attn_v.weight"),      device)?)?;
        let v_bias = ct.tensor(r, &format!("blk.{i}.attn_v.bias"), device).ok().and_then(|t| t.dequantize(device).ok());
        
        let o = QMatMul::from_qtensor(ct.tensor(r, &format!("blk.{i}.attn_output.weight"), device)?)?;
        let o_bias = ct.tensor(r, &format!("blk.{i}.attn_output.bias"), device).ok().and_then(|t| t.dequantize(device).ok());
        
        Ok(Self {
            q, q_bias,
            k, k_bias,
            v, v_bias,
            o, o_bias,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim,
            rope,
            kv_cache: std::sync::Mutex::new(None),
        })
    }

    #[cfg(feature = "cuda")]
    fn forward(
        &self,
        xs: &candle_core::Tensor,
        attention_mask: Option<&candle_core::Tensor>,
        seqlen_offset: usize,
        adapter: Option<&NeuroAdapter>,
        impedance: Option<&candle_core::Tensor>,
    ) -> candle_core::Result<candle_core::Tensor> {
        println!("[DEBUG] ADAPTER QAttention::forward - ENTRY POINT - IMMEDIATE!");
        
        let (b_size, seq_len, _) = xs.dims3()?;

        // FIX: QMatMul expects 2D input, but we have 3D [batch, seq_len, hidden_dim]
        // Reshape to 2D [batch*seq_len, hidden_dim], apply matmul, then reshape back
        let xs_2d = xs.reshape((b_size * seq_len, xs.dim(2)?))?;

        println!("[DEBUG] ADAPTER QAttention::forward - xs_2d.shape = {:?}", xs_2d.shape());

        let mut query_states = self.q.forward(&xs_2d)?;
        if let Some(b) = &self.q_bias { query_states = query_states.broadcast_add(b)?; }
        
        let mut key_states = self.k.forward(&xs_2d)?;
        if let Some(b) = &self.k_bias { key_states = key_states.broadcast_add(b)?; }
        
        let mut value_states = self.v.forward(&xs_2d)?;
        if let Some(b) = &self.v_bias { value_states = value_states.broadcast_add(b)?; }

        println!("[DEBUG] ADAPTER QAttention::forward - After QKV projections");
        println!("[DEBUG] ADAPTER QAttention::forward - query_states.shape = {:?}", query_states.shape());
        println!("[DEBUG] ADAPTER QAttention::forward - key_states.shape = {:?}", key_states.shape());
        let query_states = query_states.reshape((b_size, seq_len, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let key_states = key_states.reshape((b_size, seq_len, self.kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let value_states = value_states.reshape((b_size, seq_len, self.kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        println!("[DEBUG] ADAPTER QAttention::forward - After reshape+transpose: query_states.shape = {:?}", query_states.shape());

        // RoPE with corrected 4D cos/sin shape
        let (query_states, key_states) = self.rope.apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        // KV cache grow-and-concat
        let (k_cache, v_cache) = {
            let mut cache = self.kv_cache.lock().unwrap();
            if seqlen_offset == 0 {
                *cache = Some((key_states.to_dtype(DType::F16)?, value_states.to_dtype(DType::F16)?));
                (key_states, value_states)
            } else if let Some((prev_k_f16, prev_v_f16)) = cache.take() {
                let prev_k_f32 = prev_k_f16.to_dtype(DType::F32)?;
                drop(prev_k_f16);
                let new_k = candle_core::Tensor::cat(&[&prev_k_f32, &key_states], 2)?;
                drop(prev_k_f32);
                
                let prev_v_f32 = prev_v_f16.to_dtype(DType::F32)?;
                drop(prev_v_f16);
                let new_v = candle_core::Tensor::cat(&[&prev_v_f32, &value_states], 2)?;
                drop(prev_v_f32);
                
                *cache = Some((new_k.to_dtype(DType::F16)?, new_v.to_dtype(DType::F16)?));
                (new_k, new_v)
            } else {
                *cache = Some((key_states.to_dtype(DType::F16)?, value_states.to_dtype(DType::F16)?));
                (key_states, value_states)
            }
        };

        // GQA repeat
        let (k_rep, v_rep) = if self.groups == 1 {
            (k_cache.clone(), v_cache.clone())
        } else {
            let (b, n_kv, s, d) = k_cache.dims4()?;
            let k_exp = k_cache.unsqueeze(2)?.expand((b, n_kv, self.groups, s, d))?.contiguous()?.reshape((b, n_kv * self.groups, s, d))?.contiguous()?;
            let v_exp = v_cache.unsqueeze(2)?.expand((b, n_kv, self.groups, s, d))?.contiguous()?.reshape((b, n_kv * self.groups, s, d))?.contiguous()?;
            (k_exp, v_exp)
        };

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (query_states.matmul(&k_rep.transpose(2, 3)?)? * scale)?;

        if let Some(m) = attention_mask {
            attn_weights = attn_weights.broadcast_add(m)?;
        }

        // IIIIII Physics kernel injection (decode-step only) ////////////////////////////////////////////////////////////IIIIII
        if let (Some(a), Some(i)) = (adapter, impedance) {
            let (_, _, q_len, k_len) = attn_weights.dims4()?;
            if q_len == 1 {
                // Ensure F32 + contiguous for the C++ kernel
                let attn_f32_safe = attn_weights.to_dtype(candle_core::DType::F32)?.contiguous()?;

                // Align impedance to current k_len (grows as KV cache grows)
                let imp_f32  = i.to_dtype(candle_core::DType::F32)?;
                let imp_len  = imp_f32.dim(0)?;
                let imp_aligned_f32 = if imp_len < k_len {
                    let pad = Tensor::zeros((k_len - imp_len,), DType::F32, imp_f32.device())?;
                    Tensor::cat(&[&imp_f32, &pad], 0)?
                } else if imp_len > k_len {
                    imp_f32.narrow(0, 0, k_len)?
                } else {
                    imp_f32
                };
                let imp_f16 = imp_aligned_f32.to_dtype(candle_core::DType::F16)?.contiguous()?;

                let wptr = attn_f32_safe.get_physical_address()?;
                let iptr = imp_f16.get_physical_address()?;

                a.apply_bias_raw(wptr, iptr, 1, self.heads as i32, q_len as i32, k_len as i32)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

                // CRITICAL FIX: GPU RACE CONDITION. Force synchronization so the C++ kernel 
                // finishes writing before Candle's softmax tries to read the memory!
                a.synchronize().map_err(|e| candle_core::Error::Msg(e.to_string()))?;

                attn_weights = attn_f32_safe;
            }
        }

        let w = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let o = w.matmul(&v_rep)?;
        let out_unproj = o.transpose(1, 2)?.reshape((b_size, seq_len, self.heads * self.head_dim))?;
        
        // FIX: QMatMul expects 2D input, but we have 3D [batch, seq_len, hidden_dim]
        // Reshape to 2D [batch*seq_len, hidden_dim], apply matmul, then reshape back
        let out_unproj_2d = out_unproj.reshape((b_size * seq_len, self.heads * self.head_dim))?;
        let mut out = self.o.forward(&out_unproj_2d)?;
        if let Some(b) = &self.o_bias { out = out.broadcast_add(b)?; }
        // Reshape back to 3D [batch, seq_len, intermediate_size]
        let intermediate_size = out.dim(1)?;
        out.reshape((b_size, seq_len, intermediate_size))
    }

    // (the forward_async path in Qwen2Model::forward_async also benefits from
    //  the same impedance-alignment logic; kept here for sync path)
    #[cfg(feature = "cuda")]
    pub async fn forward_async(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        adapter: Option<&NeuroAdapter>,
        impedance: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = xs.dims3()?;
        
        // FIX: QMatMul expects 2D input, but we have 3D [batch, seq_len, hidden_dim]
        // Reshape to 2D [batch*seq_len, hidden_dim], apply matmul, then reshape back
        let xs_2d = xs.reshape((b_size * seq_len, xs.dim(2)?))?;
        
        let mut query_states = self.q.forward(&xs_2d)?;
        if let Some(b) = &self.q_bias { query_states = query_states.broadcast_add(b)?; }
        
        let mut key_states = self.k.forward(&xs_2d)?;
        if let Some(b) = &self.k_bias { key_states = key_states.broadcast_add(b)?; }
        
        let mut value_states = self.v.forward(&xs_2d)?;
        if let Some(b) = &self.v_bias { value_states = value_states.broadcast_add(b)?; }
        
        // Reshape back to 3D and then to 4D for attention
        let query_states = query_states.reshape((b_size, seq_len, self.heads * self.head_dim))?
            .reshape((b_size, seq_len, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let key_states = key_states.reshape((b_size, seq_len, self.kv_heads * self.head_dim))?
            .reshape((b_size, seq_len, self.kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let value_states = value_states.reshape((b_size, seq_len, self.kv_heads * self.head_dim))?
            .reshape((b_size, seq_len, self.kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        
        let (query_states, key_states) = self.rope.apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;
        
        // CRITICAL FIX: KV CACHE INJECTION FOR ASYNC PIPELINE
        let (k_cache, v_cache) = {
            let mut cache = self.kv_cache.lock().unwrap();
            if seqlen_offset == 0 {
                *cache = Some((key_states.to_dtype(DType::F16)?, value_states.to_dtype(DType::F16)?));
                (key_states, value_states)
            } else if let Some((prev_k_f16, prev_v_f16)) = cache.take() {
                let prev_k_f32 = prev_k_f16.to_dtype(DType::F32)?;
                drop(prev_k_f16);
                let new_k = Tensor::cat(&[&prev_k_f32, &key_states], 2)?;
                drop(prev_k_f32);
                
                let prev_v_f32 = prev_v_f16.to_dtype(DType::F32)?;
                drop(prev_v_f16);
                let new_v = Tensor::cat(&[&prev_v_f32, &value_states], 2)?;
                drop(prev_v_f32);
                
                *cache = Some((new_k.to_dtype(DType::F16)?, new_v.to_dtype(DType::F16)?));
                (new_k, new_v)
            } else {
                *cache = Some((key_states.to_dtype(DType::F16)?, value_states.to_dtype(DType::F16)?));
                (key_states, value_states)
            }
        };
        
        // CRITICAL FIX: GQA REPEAT FOR QWEN2 IN ASYNC PIPELINE
        let num_kv_groups = self.heads / self.kv_heads;
        let (k_rep, v_rep) = if num_kv_groups == 1 {
            (k_cache.clone(), v_cache.clone())
        } else {
            let (b, n_kv, s, d) = k_cache.dims4()?;
            let k_expanded = k_cache.unsqueeze(2)?.expand((b, n_kv, num_kv_groups, s, d))?.contiguous()?.reshape((b, n_kv * num_kv_groups, s, d))?.contiguous()?;
            let v_expanded = v_cache.unsqueeze(2)?.expand((b, n_kv, num_kv_groups, s, d))?.contiguous()?.reshape((b, n_kv * num_kv_groups, s, d))?.contiguous()?;
            (k_expanded, v_expanded)
        };
        
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (query_states.matmul(&k_rep.transpose(2, 3)?)? * scale)?;
        
        if let Some(mask) = attention_mask { 
            attn_weights = attn_weights.broadcast_add(&mask)?; 
        }
        
        if let (Some(a), Some(i)) = (adapter, impedance) {
            let (_, _, q_len, k_len) = attn_weights.dims4()?;

            // ////////////////////I [DEBUG-PROBE-8] ADAPTER CALL ////////////////////I
let imp_shape = i.shape().dims().to_vec();
println!(
    "[PROBE-8] Adapter called: q_len={} k_len={} imp_shape={:?} dtype={:?}",
    q_len, k_len, imp_shape, i.dtype()
);

            if q_len == 1 {
                let mut attn_f32_safe = attn_weights.to_dtype(candle_core::DType::F32)?.contiguous()?;
                
                // CRITICAL FIX: Missing impedance alignment logic in async method!
                let imp_f32  = i.to_dtype(candle_core::DType::F32)?;
                let imp_len  = imp_f32.dim(0)?;
                let imp_aligned_f32 = if imp_len < k_len {
                    let pad = candle_core::Tensor::zeros((k_len - imp_len,), candle_core::DType::F32, imp_f32.device())?;
                    candle_core::Tensor::cat(&[&imp_f32, &pad], 0)?
                } else if imp_len > k_len {
                    imp_f32.narrow(0, 0, k_len)?
                } else {
                    imp_f32
                };
                let imp_f16 = imp_aligned_f32.to_dtype(candle_core::DType::F16)?.contiguous()?;
                let wptr = attn_f32_safe.get_physical_address()?;
                let iptr = imp_f16.get_physical_address()?;
                
                a.apply_bias_raw(wptr, iptr, 1, self.heads as i32, q_len as i32, k_len as i32)
                    .map_err(|e: Box<dyn std::error::Error>| candle_core::Error::Msg(e.to_string()))?;
                
                // Prevent GPU Race Condition
                a.synchronize().map_err(|e: Box<dyn std::error::Error>| candle_core::Error::Msg(e.to_string()))?;
                // ////////////////////I [DEBUG-PROBE-9] ADAPTER DONE ////////////////////I
let nan_check = attn_weights
    .to_dtype(candle_core::DType::F32)
    .and_then(|t| t.sum_all())
    .and_then(|t| t.to_scalar::<f32>());
println!(
    "[PROBE-9] Post-adapter attn_sum={:?} (NaN here = kernel corruption)",
    nan_check
);
                attn_weights = attn_f32_safe;
            }
        }
        
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v_rep)?;
        
        let out_unproj = attn_output.transpose(1, 2)?.reshape((b_size, seq_len, self.heads * self.head_dim))?;
        
        // FIX: QMatMul expects 2D input, but we have 3D [batch, seq_len, hidden_dim]
        // Reshape to 2D [batch*seq_len, hidden_dim], apply matmul, then reshape back
        let out_unproj_2d = out_unproj.reshape((b_size * seq_len, self.heads * self.head_dim))?;
        let mut out = self.o.forward(&out_unproj_2d)?;
        if let Some(b) = &self.o_bias { out = out.broadcast_add(b)?; }
        // Reshape back to 3D [batch, seq_len, intermediate_size]
        let intermediate_size = out.dim(1)?;
        out.reshape((b_size, seq_len, intermediate_size))
    }
}

// ============================================================================
// DECODER LAYER
// ============================================================================

struct QDecoderLayer {
    attn: QAttention,
    mlp:  QMlp,
    ln1:  QRmsNorm,
    ln2:  QRmsNorm,
}

impl QDecoderLayer {
    #[cfg(feature = "cuda")]
    async fn forward_async(
        &self,
        x: &Tensor,
        m: Option<&Tensor>,
        off: usize,
        adapter: Option<&NeuroAdapter>,
        imp: Option<&Tensor>,
    ) -> Result<Tensor> {
        let ln1_out  = self.ln1.forward(x)?;
        let attn_out = self.attn.forward_async(&ln1_out, m, off, adapter, imp).await?;
        let x1       = (attn_out + x)?;
        let ln2_out  = self.ln2.forward(&x1)?;
        let mlp_out  = self.mlp.forward(&ln2_out)?;
        (mlp_out + &x1)
    }
}

// ============================================================================
// MODEL (async pipeline with prefetch)
// ============================================================================

pub struct Qwen2Model {
    embed:  Embedding,
    layers: Vec<QDecoderLayer>,
    norm:   QRmsNorm,
    device: Device,

    #[cfg(feature = "cuda")]
    pub physics_adapter: Option<Arc<NeuroAdapter>>,

    #[cfg(feature = "cuda")]
    current_impedance: Option<Tensor>,

    #[cfg(feature = "cuda")]
    next_impedance: Option<Vec<f32>>,
}

impl Qwen2Model {
    pub fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        r: &mut R,
        cfg: &Qwen2Config,
        device: Device,
    ) -> Result<Self> {
        let emb_w = ct.tensor(r, "token_embd.weight", &device)?.dequantize(&device)?;
        let embed = Embedding::new(emb_w, cfg.hidden_size);
        let rope  = Arc::new(RotaryEmbedding::new(cfg, &device)?);

        let mut layers = vec![];
        for i in 0..cfg.num_hidden_layers {
            layers.push(QDecoderLayer {
                attn: QAttention::new(ct, r, cfg, i, rope.clone(), &device)?,
                mlp:  QMlp::new(ct, r, i, &device)?,
                ln1:  QRmsNorm::new(ct.tensor(r, &format!("blk.{i}.attn_norm.weight"), &device)?, cfg.rms_norm_eps)?,
                ln2:  QRmsNorm::new(ct.tensor(r, &format!("blk.{i}.ffn_norm.weight"), &device)?,  cfg.rms_norm_eps)?,
            });
        }
        let norm = QRmsNorm::new(ct.tensor(r, "output_norm.weight", &device)?, cfg.rms_norm_eps)?;

        #[cfg(feature = "cuda")]
        let physics_adapter = if device.is_cuda() {
            Some(Arc::new(NeuroAdapter::new(0, 0.8, true)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?))
        } else {
            None
        };

        Ok(Self {
            embed, layers, norm, device,
            #[cfg(feature = "cuda")] physics_adapter,
            #[cfg(feature = "cuda")] current_impedance: None,
            #[cfg(feature = "cuda")] next_impedance: None,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn prefetch_next_impedance(&mut self, impedance: Vec<f32>) {
        self.next_impedance = Some(impedance);
    }

    /// Forward pass with async prefetching pipeline.
    pub async fn forward_async(&mut self, ids: &Tensor, off: usize) -> Result<Tensor> {
        let mut x = self.embed.forward(ids)?;

        #[cfg(feature = "cuda")]
        {
            let adapter = self.physics_adapter.as_deref();

            for (i, layer) in self.layers.iter().enumerate() {
                // Prefetch next layer's impedance in background
                 if let (Some(ref next_imp), Some(adp)) = (&self.next_impedance, adapter) {
                    if i < self.layers.len() - 1 {
                        // Convert f32 impedance to f16 bits for the VRAM double-buffer
                        let imp_u16: Vec<u16> = next_imp.iter()
                            .map(|&v| half::f16::from_f32(v).to_bits())
                            .collect();
                        adp.prefetch_impedance(&imp_u16)
                            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    }
                }

                let imp = self.current_impedance.take();
                x = layer.forward_async(&x, None, off, adapter, imp.as_ref()).await?;

                // Double-buffer swap
                if let Some(next) = self.next_impedance.take() {
                    self.current_impedance = Some(
                        Tensor::from_vec(next.clone(), (next.len(),), &self.device)?
                    );
                }
            }

            if let Some(ref adp) = adapter {
                adp.synchronize()
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            }
        }

        self.norm.forward(&x)
    }
}

// ============================================================================
// SYNC WRAPPER
// ============================================================================

impl Qwen2Model {
    pub fn forward(&mut self, ids: &Tensor, off: usize) -> Result<Tensor> {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.forward_async(ids, off))
    }
}

// ============================================================================
// FFI III physics_mask.cu
// ============================================================================

pub mod physics_mask {
    use std::ffi::c_void;
    
    #[link(name = "physics_mask", kind = "static")]
    extern "C" {
        /// C++ apply_mask wrapper III launches apply_physics_bias_kernel with
        /// the corrected 2D grid: dim3(b*h, ceil(q/BLOCK_SIZE)).
        pub fn apply_mask(
            stream: *mut c_void,   // cudaStream_t; nullptr = default stream
            wptr:   *mut c_void,   // F32 attention scores
            iptr:   *const c_void, // F16 impedance values (length == k)
            b: i32,
            h: i32,
            q: i32,
            k: i32,
        );
    }
}















