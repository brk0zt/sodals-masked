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
use crate::gpu_ops::memory_utils::SodalsCudaAccess;
use crate::gpu_ops::physics_mask;

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
    fn synchronize_stream(stream: *mut c_void) -> i32;
    fn stream_pool_create(num_streams: i32) -> *mut c_void;
    fn stream_pool_destroy(pool: *mut c_void);
    fn cudaGetDeviceFlags(flags: *mut i32) -> i32;
    // BARE-METAL VRAM SAFETY: Direct FFI for synchronization - do NOT rely on framework wrappers
    pub fn cudaDeviceSynchronize() -> i32;
    pub fn cudaSetDevice(device_id: i32) -> i32;
}

// ============================================================================
// ASYNC NEURO ADAPTER - double-buffer VRAM prefetch (DMA memory manager only)
// ASYNC NEURO ADAPTER - triple-buffer VRAM prefetch (DMA memory manager only)
// ============================================================================

#[allow(dead_code)]
pub struct NeuroAdapter {
    device_id:    u32,
    pub impedance_factor: f64,
    /// Three VRAM-backed staging buffers; provides full temporal isolation for async DMA.
    /// Rotation: N (Read) | N+1 (Prefilling) | N-1 (Just finished reading)
    buffers: [std::sync::Mutex<Option<PinnedBuffer>>; 3],
    /// Index of the 'front' buffer currently being read by the GPU.
    front:   std::sync::atomic::AtomicUsize,
    /// Dedicated non-blocking CUDA stream for DMA transfers.
    /// PHASE 3: Wrapped in Mutex to prevent cross-thread driver races.
    #[cfg(feature = "cuda")]
    dma_stream: std::sync::Mutex<*mut c_void>,
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
            buffers: [
                std::sync::Mutex::new(None),
                std::sync::Mutex::new(None),
                std::sync::Mutex::new(None),
            ],
            front: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "cuda")]
            dma_stream: std::sync::Mutex::new(dma_stream),
            #[cfg(not(feature = "cuda"))]
            _dma_stream: (),
        })
    }

    /// Queue impedance for layer N+1 into the *back* VRAM buffer via DMA.
    /// Call this while the GPU is still computing layer N.
    pub fn prefetch_impedance(&self, sparse_data: &[(usize, u16)], vocab_size: usize) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let bytes = vocab_size * 2; // each f16 = 2 bytes
        if bytes == 0 {
            return Ok(());
        }

        // ORACLE THREAD SAFETY: Skip GPU operations if no context.
        #[cfg(feature = "cuda")]
        unsafe {
            // ORACLE THREAD SAFETY: Bind context before ANY driver call.
            // Oracle thread is a tokio::spawn_blocking - it has no implicit CUDA context.
            let rc = cudaSetDevice(self.device_id as i32);
            if rc != 0 {
                // No context available on this thread; silently skip prefetch.
                // Generation continues without Oracle bias - correct degraded mode.
                return Ok(());
            }
            
            // Secondary check: verify context is actually initialized (not just device selected)
            let mut flags: i32 = 0;
            if cudaGetDeviceFlags(&mut flags) != 0 {
                return Ok(());
            }
        }

        // TRIPLE BUFFERING: Prefetch into the NEXT buffer after the current front one.
        // Rotation: (front + 1) % 3
        let current_front = self.front.load(std::sync::atomic::Ordering::SeqCst);
        let back = (current_front + 1) % 3;

        #[cfg(feature = "cuda")]
        let _stream_guard = self.dma_stream.lock().map_err(|e| e.to_string())?;
        
        let mut guard = self.buffers[back].lock().map_err(|e| e.to_string())?;

        // Allocate / resize buffer as needed
        let _buf = match guard.as_mut() {
            Some(b) if b.size >= bytes => b,
            _ => {
                // VRAM SAFETY: Over-allocate to 1MB minimum to prevent constant reallocation SegFaults
                // during autoregressive generation when Oracle thread continuously adds tokens.
                let safe_size = bytes.max(1024 * 1024);
                *guard = Some(PinnedBuffer::new(safe_size)?);
                guard.as_mut().unwrap()
            }
        };

        #[cfg(feature = "cuda")]
        {
            // Build dense array efficiently
            let mut dense = vec![0u16; vocab_size];
            for &(idx, val) in sparse_data {
                if idx < vocab_size { dense[idx] = val; }
            }

            // Copy host data into pinned staging
            unsafe {
                std::ptr::copy_nonoverlapping(
                    dense.as_ptr() as *const u8,
                    _buf._host_ptr as *mut u8,
                    bytes,
                );
            }
            // Async DMA from pinned host III VRAM (non-blocking on host)
            let rc = unsafe {
                let stream_guard = self.dma_stream.lock().map_err(|e| e.to_string())?;
                async_memcpy_to_device(
                    _buf._host_ptr as *const c_void,
                    _buf.vram_ptr as *mut c_void,
                    bytes,
                    *stream_guard,
                )
            };
            if rc != 0 {
                return Err(format!("cudaMemcpyAsync failed with code {rc}").into());
            }
        }

        Ok(())
    }

    /// Wait for the DMA to the prefill buffer to complete, then rotate.
    pub fn swap_and_get_vram_ptr(&self) -> std::result::Result<u64, Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        unsafe {
            // Triple-Buffering ensures the PREVIOUSLY read buffer (current_front)
            // is isolated from the JUST PREFILLED buffer (back).
            let stream_guard = self.dma_stream.lock().map_err(|e| e.to_string())?;
            synchronize_stream(*stream_guard);
        }

        // Rotate: Move to the buffer we just prefetched.
        let prev_front = self.front.load(std::sync::atomic::Ordering::SeqCst);
        let next_front = (prev_front + 1) % 3;
        self.front.store(next_front, std::sync::atomic::Ordering::SeqCst);

        let guard = self.buffers[next_front].lock().map_err(|e| e.to_string())?;
        match guard.as_ref() {
            Some(b) => Ok(b.vram_ptr),
            None    => Err("No VRAM buffer in front slot after swap".into()),
        }
    }

    /// Apply physics bias using the address already resident in VRAM.
    /// `iptr` must be a VRAM address (returned by `swap_and_get_vram_ptr`).
    pub fn apply_bias_raw(
        &self,
        _wptr: u64,
        _iptr: u64,
        _b: i32, _h: i32, _q: i32, _k: i32,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        unsafe {
            physics_mask::apply_mask(
                std::ptr::null_mut() as *mut c_void,
                _wptr as *mut c_void,
                _iptr as *const c_void,
                _b, _h, _q, _k,
            );
        }
        Ok(())
    }

    /// Block host until all async GPU ops on the DEFAULT COMPUTE stream complete.
    /// PHASE 7: We use this instead of Device::synchronize() to avoid deadlocking with DMA.
    pub fn synchronize(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        unsafe { synchronize_stream(std::ptr::null_mut()); }
        Ok(())
    }

    // Kept for callers that still pass a pre-computed Candle tensor impedance
    // directly as an F16 device pointer (existing path in qwen_custom.rs).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_modulation_raw(
        &self, _act_ptr: u64, _imp_ptr: u64,
        _b: i32, _h: i32, _q: i32, _d: i32, _scaling: f32,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> { Ok(()) }

    /// Diagnostic Phase 3.5: Hard-Zero all VRAM buffers to break persistent math attractors.
    pub fn clear_buffers(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        eprintln!("[NeuroAdapter] 🌊 Neutralizing Triple-Buffer VRAM sidecar...");
        
        #[cfg(feature = "cuda")]
        {
            // CRITICAL: Bind CUDA context to THIS thread before ANY driver call.
            // clear_buffers() may be called from Ouroboros loop threads that have
            // no implicit CUDA context. Without this, async_memcpy_to_device() 
            // dereferences an invalid context pointer -> STATUS_ACCESS_VIOLATION.
            unsafe {
                let rc = cudaSetDevice(self.device_id as i32);
                if rc != 0 {
                    eprintln!("[NeuroAdapter] cudaSetDevice failed ({}), skipping VRAM clear.", rc);
                    self.front.store(0, std::sync::atomic::Ordering::SeqCst);
                    return Ok(());
                }
            }

            // Use the DEFAULT stream (null) for synchronous zeroing, NOT the DMA stream.
            // The DMA stream was created on the engine's init thread context.
            // Using null stream here is safe: it's a synchronization point on THIS context.
            for i in 0..3 {
                if let Ok(mut guard) = self.buffers[i].lock() {
                    if let Some(buf) = guard.as_mut() {
                        // Null pointer guard: skip uninitialized buffers
                        if buf._host_ptr.is_null() || buf.vram_ptr == 0 || buf.size == 0 {
                            continue;
                        }
                        unsafe {
                            // Zero host staging first (pure CPU, always safe)
                            std::ptr::write_bytes(buf._host_ptr as *mut u8, 0, buf.size);
                            
                            // DMA to VRAM on DEFAULT stream (null) - context-agnostic
                            let rc = async_memcpy_to_device(
                                buf._host_ptr as *const std::ffi::c_void,
                                buf.vram_ptr as *mut std::ffi::c_void,
                                buf.size,
                                std::ptr::null_mut(), // DEFAULT stream - safe cross-thread
                            );
                            if rc != 0 {
                                eprintln!("[NeuroAdapter] ⚠  DMA zeroing failed for buffer {} (code {})", i, rc);
                            }
                        }
                    }
                }
            }
            
            // Synchronize DEFAULT stream to ensure VRAM is clean before returning
            unsafe { synchronize_stream(std::ptr::null_mut()); }
        }
        
        self.front.store(0, std::sync::atomic::Ordering::SeqCst);
        eprintln!("[NeuroAdapter] ✅ Sidecar neutralized.");
        Ok(())
    }
}

impl Drop for NeuroAdapter {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if let Ok(guard) = self.dma_stream.lock() {
            unsafe { stream_pool_destroy(*guard); }
        }
    }
}

// ============================================================================
// PINNED BUFFER III Pinned host staging + VRAM destination (Double-Buffer node)
// ============================================================================

struct PinnedBuffer {
    /// Write-combining pinned host RAM (staging area for DMA)
    _host_ptr: *mut c_void,
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
            // ORACLE THREAD SAFETY: Check CUDA context before allocation.
            // Background threads without context must skip GPU allocation.
            unsafe {
                let mut flags: i32 = 0;
                let ctx_check = cudaGetDeviceFlags(&mut flags);
                if ctx_check != 0 { // cudaSuccess = 0
                    // No CUDA context - return dummy buffer (CPU-only mode for this thread)
                    return Ok(Self {
                        _host_ptr: std::ptr::null_mut(),
                        vram_ptr: 0,
                        size: bytes
                    });
                }
            }
            
            let host_ptr = unsafe { alloc_pinned_memory(bytes) };
            if host_ptr.is_null() {
                return Err("cudaHostAlloc failed (out of pinned memory)".into());
            }
            let vram_raw = unsafe { alloc_device_vram(bytes) };
            if vram_raw.is_null() {
                unsafe { free_pinned_memory(host_ptr); }
                return Err("cudaMalloc for VRAM buffer failed".into());
            }
            Ok(Self { _host_ptr: host_ptr, vram_ptr: vram_raw as u64, size: bytes })
        }
        #[cfg(not(feature = "cuda"))]
        Ok(Self {
            _host_ptr: std::ptr::null_mut(),
            vram_ptr: 0,
            size: bytes
        })
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            // ORACLE THREAD SAFETY: Check CUDA context before freeing.
            let mut flags: i32 = 0;
            let ctx_check = cudaGetDeviceFlags(&mut flags);
            if ctx_check != 0 { // cudaSuccess = 0
                // No CUDA context - skip all GPU frees (leak memory, OS will clean on exit)
                return;
            }
            
            // Sadece Host (RAM) belleğini temizle, VRAM ve Sync işlemlerini bypass et.
            if !self._host_ptr.is_null() { 
                free_pinned_memory(self._host_ptr); 
            }
            // VRAM pointer'ına dokunmuyoruz. GPU hala okuyor olabilir.
        }
    }
}

// ============================================================================
// FFI Exports for unified Qwen2Model in qwen_custom.rs
// ============================================================================

/// Get VRAM device pointer from front buffer for physics kernel injection.
pub fn get_vram_ptr(adapter: &NeuroAdapter) -> std::result::Result<u64, Box<dyn std::error::Error>> {
    let front = adapter.front.load(std::sync::atomic::Ordering::SeqCst);
    let guard = adapter.buffers[front].lock().map_err(|e| e.to_string())?;
    match guard.as_ref() {
        Some(b) => Ok(b.vram_ptr),
        None    => Err("No VRAM buffer in front slot".into()),
    }
}
