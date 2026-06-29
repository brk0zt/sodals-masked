// src/gpu_ops/memory_utils.rs
// SODALS GPU Memory Access Abstraction Layer (candle-core 0.8.2 compatible)
// 
// This module provides a clean, isolated interface for extracting physical GPU memory
// addresses from candle-core tensors, supporting the sidecar physics adapter pattern.
//
// Design Principle: "Matematiksel KIIrlIIk" (Mathematical Blindness)
// - System treats GPU memory as opaque physical addresses (u64)
// - No tensor semantics; only raw device pointers
// - Decoupled from candle-core's internal stream management
// 
// API Compatibility Notes:
// - candle-core 0.8.2: device_ptr() returns CUdeviceptr directly
// - Stream management handled internally by candle-core
// - CudaStorage.device is a public field for stream access

use candle_core::{Tensor, Result as CandleResult, Error as CandleError};
use std::ops::Deref;
use std::mem;

use std::sync::Arc;

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;

// BARE-METAL CUDA FFI: Direct runtime API calls for synchronization
#[cfg(feature = "cuda")]
extern "C" {
    pub fn cudaDeviceSynchronize() -> i32;
    // ORACLE THREAD SAFETY: Check CUDA context before GPU operations
    pub fn cudaGetDeviceFlags(flags: *mut i32) -> i32;
}

/// SodalsCudaAccess: Extension trait for extracting physical GPU addresses from tensors
/// 
/// This trait provides a single, clean interface for getting the raw GPU memory address
/// of a tensor, hiding all the complexity of CUDA stream management and pointer extraction.
pub trait SodalsCudaAccess {
    /// Get the physical GPU memory address of this tensor's data as a u64 pointer
    /// 
    /// # Returns
    /// - `Ok(u64)`: The CUDA device pointer cast to u64
    /// - `Err(CandleError)`: If tensor is not in CUDA storage or stream access fails
    /// 
    /// # Example
    /// ```ignore
    /// let attention_ptr = attn_weights.get_physical_address()?;
    /// let impedance_ptr = impedance_tensor.get_physical_address()?;
    /// adapter.apply_bias_raw(attention_ptr, impedance_ptr, ...)?;
    /// ```
    fn get_physical_address(&self) -> CandleResult<u64>;
}

#[cfg(feature = "cuda")]
impl SodalsCudaAccess for Tensor {
    fn get_physical_address(&self) -> CandleResult<u64> {
        let (storage, _layout) = self.storage_and_layout();

        match &*storage.deref() {
            candle_core::Storage::Cuda(s) => {
                match s.as_cuda_slice::<f32>() {
                    Ok(slice) => {
                        // candle-core 0.8.2: Direct pointer access without tuple
                        let device = &s.device;

                        // Get CUDA stream (cudarc 0.13: cu_stream() returns &*mut CUstream_st, not Result)
                        let _stream = device.cu_stream(); 
                        
                        // candle-core / cudarc 0.13: device_ptr() takes no arguments
                        let ptr = slice.device_ptr();
                        let alignment = mem::align_of::<f32>();  // = 4 bytes
                        let is_aligned = (*ptr as usize) % alignment == 0;
                        let address = *ptr as u64;
                        
                        // III VALIDATION 1: Null Pointer Check
                        // Catches CUDA driver bugs that return null
                        debug_assert!(
                            address != 0,
                            "CUDA driver returned null pointer for tensor storage: \
                             This indicates a driver bug or memory corruption. \
                             Device memory may be exhausted or driver state corrupted."
                        );
                        
                        // III VALIDATION 2: Alignment Check
                        // f32 requires 4-byte alignment (built-in to CUDA)
                        debug_assert!(
                            is_aligned,
                            "Misaligned CUDA pointer: 0x{:X} is not aligned to {} bytes. \
                             This causes undefined behavior on load/store.",
                            address, alignment
                        );
                        
                        // III VALIDATION 3: Type Consistency Check
                        debug_assert_eq!(
                            mem::size_of::<f32>(), 4,
                            "f32 size is not 4 bytes; data layout will be corrupted"
                        );
                        
                        Ok(address)
                    }
                    Err(e) => Err(CandleError::Msg(format!(
                        "Failed to get CUDA slice: {}",
                        e
                    ))),
                }
            }
            _ => Err(CandleError::Msg(
                "Tensor is not stored in CUDA memory".to_string(),
            )),
        }
    }
}

#[cfg(not(feature = "cuda"))]
impl SodalsCudaAccess for Tensor {
    fn get_physical_address(&self) -> CandleResult<u64> {
        Err(CandleError::Msg(
            "CUDA not compiled: gpu memory access unavailable".to_string(),
        ))
    }
}

// ============================================================================
// Lifetime-Safe CUDA Pointer III PinnedCudaPtr
// ============================================================================
//
// PROBLEM: `get_physical_address()` returns a raw u64. If the source `Tensor`
// is dropped before the CUDA kernel finishes reading, the pointer becomes
// dangling III CUDA Error 700 (Illegal Memory Access) III device hang.
//
// SOLUTION: `PinnedCudaPtr` bundles the address with an `Arc<Tensor>` clone.
// The tensor's VRAM backing is freed only when the LAST Arc<Tensor> drops.
// By holding `PinnedCudaPtr` alive until after `adapter.synchronize()`, the
// caller guarantees that the kernel finishes before the VRAM is reclaimed.
//
// Usage:
//   let _pin_attn = attn_f32_safe.pinned_address()?;
//   let _pin_imp  = imp_f16_aligned.pinned_address()?;
//   adapter.apply_bias_raw(*_pin_attn, *_pin_imp, ...)?;
//   adapter.synchronize()?;
//   // _pin_attn, _pin_imp drop here III VRAM freed safely AFTER sync.

/// A CUDA device pointer whose source tensor is kept alive via an Arc clone.
pub struct PinnedCudaPtr {
    address: u64,
    /// Keeps the tensor's VRAM allocation alive until this is dropped.
    pub(crate) _tensor: Arc<Tensor>,
}

impl PinnedCudaPtr {
    /// Address of the tensor's CUDA storage (same value as `get_physical_address`).
    pub fn address(&self) -> u64 { self.address }
}

/// Dereferences to the raw u64 address for ergonomic use in FFI calls.
impl Deref for PinnedCudaPtr {
    type Target = u64;
    fn deref(&self) -> &u64 { &self.address }
}

pub trait SodalsCudaPinnedAccess {
    /// Wrap this tensor's CUDA pointer in a lifetime-safe `PinnedCudaPtr`.
    /// The tensor MUST NOT be dropped before the CUDA kernel finishes.
    /// Keep the returned value alive until after `adapter.synchronize()`.
    fn pinned_address(self: Arc<Self>) -> CandleResult<PinnedCudaPtr>;
}

impl SodalsCudaPinnedAccess for Tensor {
    fn pinned_address(self: Arc<Self>) -> CandleResult<PinnedCudaPtr> {
        let address = self.get_physical_address()?;
        Ok(PinnedCudaPtr { address, _tensor: self })
    }
}


// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(feature = "cuda")]
extern "C" {
    // Phase 21: Zero-copy mapped memory for QMEM cartridges
    // CPU writes to host_ptr, GPU reads from dev_ptr directly over PCIe
    fn allocate_zero_copy_qmem(bytes: usize, host_ptr: *mut *mut std::ffi::c_void, dev_ptr: *mut *mut std::ffi::c_void) -> i32;
    fn free_zero_copy_qmem(host_ptr: *mut std::ffi::c_void);
    #[allow(dead_code)]
    fn get_device_pointer(host_ptr: *mut std::ffi::c_void, dev_ptr: *mut *mut std::ffi::c_void) -> i32;
}

#[cfg(feature = "cuda")]
/// Zero-copy memory cartridge for QMEM semantic tensors.
/// Uses cudaHostAllocMapped - CPU writes to host_ptr, GPU reads from dev_ptr directly.
/// No explicit cudaMemcpy needed - memory is mapped and immediately available to GPU.
#[cfg(feature = "cuda")]
pub struct ZeroCopyCartridge {
    host_ptr: *mut std::ffi::c_void,
    dev_ptr: u64,
    capacity: usize,
}

#[cfg(feature = "cuda")]
impl ZeroCopyCartridge {
    /// Allocate zero-copy mapped memory for QMEM data.
    /// Returns error if CUDA allocation fails.
    pub fn new(capacity: usize) -> Result<Self, String> {
        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        
        let bytes = capacity * std::mem::size_of::<f32>();
        
        unsafe {
            let result = allocate_zero_copy_qmem(bytes, &mut host_ptr, &mut dev_ptr);
            if result != 0 || host_ptr.is_null() {
                return Err(format!("Failed to allocate zero-copy memory: CUDA error {}", result));
            }
        }
        
        Ok(Self {
            host_ptr,
            dev_ptr: dev_ptr as u64,
            capacity,
        })
    }
    
    /// Get the raw device pointer for passing to CUDA kernels.
    /// This pointer can be used directly in GPU kernels - no copy needed.
    pub fn device_ptr(&self) -> u64 {
        self.dev_ptr
    }
    
    /// Get the host pointer for CPU writing.
    /// Write to this pointer, and the data is immediately available to GPU.
    pub fn host_ptr(&self) -> *mut std::ffi::c_void {
        self.host_ptr
    }
    
    /// Get capacity (number of f32 elements)
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Write f32 data directly to the mapped memory.
    /// Because it's zero-copy mapped, the GPU sees this immediately.
    pub fn write_f32(&mut self, data: &[f32]) -> Result<(), String> {
        if data.len() > self.capacity {
            return Err(format!("Data size {} exceeds capacity {}", data.len(), self.capacity));
        }
        
        unsafe {
            let ptr = self.host_ptr as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Ok(())
    }
    
    /// Write (f32, f32) pairs for complex poles (frequency, decay)
    pub fn write_poles(&mut self, frequencies: &[f32], decays: &[f32]) -> Result<(), String> {
        if frequencies.len() != decays.len() {
            return Err("Frequency and decay arrays must have same length".to_string());
        }
        if frequencies.len() * 2 > self.capacity {
            return Err("Pole data exceeds capacity".to_string());
        }
        
        unsafe {
            let ptr = self.host_ptr as *mut f32;
            for (i, (f, d)) in frequencies.iter().zip(decays.iter()).enumerate() {
                *ptr.add(i * 2) = *f;
                *ptr.add(i * 2 + 1) = *d;
            }
        }
        Ok(())
    }
    
    /// Write impedance values (single f32 per node)
    pub fn write_impedance(&mut self, values: &[f32]) -> Result<(), String> {
        if values.len() > self.capacity {
            return Err("Impedance data exceeds capacity".to_string());
        }
        
        unsafe {
            let ptr = self.host_ptr as *mut f32;
            std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
        }
        Ok(())
    }
    
    /// Write neuro flags (u32 bitmasks)
    pub fn write_flags(&mut self, flags: &[u32]) -> Result<(), String> {
        if flags.len() > self.capacity {
            return Err("Flags data exceeds capacity".to_string());
        }
        
        unsafe {
            let ptr = self.host_ptr as *mut u32;
            std::ptr::copy_nonoverlapping(flags.as_ptr(), ptr, flags.len());
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for ZeroCopyCartridge {
    fn drop(&mut self) {
        // ORACLE THREAD SAFETY: Check CUDA context before freeing.
        // Background threads without context must skip GPU deallocation.
        unsafe {
            let mut flags: i32 = 0;
            let ctx_check = cudaGetDeviceFlags(&mut flags);
            if ctx_check != 0 { // cudaSuccess = 0
                // No CUDA context - skip GPU frees (leak memory, OS will clean on exit)
                return;
            }
        }
        
        if !self.host_ptr.is_null() {
            unsafe {
                // BARE-METAL VRAM SAFETY: Force GPU to finish all async kernels before freeing memory.
                // Prevents STATUS_ACCESS_VIOLATION use-after-free race condition.
                cudaDeviceSynchronize();
                free_zero_copy_qmem(self.host_ptr);
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
/// CPU fallback stub for ZeroCopyCartridge when CUDA is not available
pub struct ZeroCopyCartridge;

#[cfg(not(feature = "cuda"))]
impl ZeroCopyCartridge {
    pub fn new(_capacity: usize) -> Result<Self, String> {
        Err("ZeroCopyCartridge requires CUDA feature".to_string())
    }
    pub fn device_ptr(&self) -> u64 { 0 }
    pub fn capacity(&self) -> usize { 0 }
    pub fn write_poles(&mut self, _freqs: &[f32], _decays: &[f32]) -> Result<(), String> {
        Err("ZeroCopyCartridge requires CUDA feature".to_string())
    }
    pub fn write_impedance(&mut self, _imp: &[f32]) -> Result<(), String> {
        Err("ZeroCopyCartridge requires CUDA feature".to_string())
    }
    pub fn write_flags(&mut self, _flags: &[u32]) -> Result<(), String> {
        Err("ZeroCopyCartridge requires CUDA feature".to_string())
    }
}

#[cfg(not(feature = "cuda"))]
/// CPU fallback stub for ensure_zero_copy_capacity
pub fn ensure_zero_copy_capacity(_cartridge: &mut Option<ZeroCopyCartridge>, _required_size: usize) -> Result<(), String> {
    // No-op on CPU-only builds
    Ok(())
}

#[cfg(feature = "cuda")]
/// Reallocate if needed - grows the cartridge if context size exceeds current capacity.
pub fn ensure_zero_copy_capacity(cartridge: &mut Option<ZeroCopyCartridge>, required_size: usize) -> Result<(), String> {
    if let Some(ref mut existing) = cartridge {
        if existing.capacity() >= required_size {
            return Ok(());
        }
    }
    
    // BARE-METAL VRAM SAFETY: Synchronize before dropping old cartridge to ensure GPU
    // is not actively using the memory we're about to free. Prevents use-after-free.
    unsafe {
        cudaDeviceSynchronize();
    }
    
    // VRAM SAFETY: Over-allocate massively to completely eradicate mid-flight dynamic 
    // resizing (which drops pointers while the GPU is actively reading them).
    let new_capacity = required_size.max(32768);
    
    // Create new larger cartridge (old one drops after sync)
    let new_cartridge = ZeroCopyCartridge::new(new_capacity)?;
    *cartridge = Some(new_cartridge);
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_get_physical_address() -> CandleResult<()> {
        let device = Device::new_cuda(0)?;
        let tensor = Tensor::zeros((10, 10), candle_core::DType::F32, &device)?;
        
        let addr = tensor.get_physical_address()?;
        assert!(addr > 0, "Physical address should be non-zero");
        Ok(())
    }

    #[test]
    fn test_multiple_tensors_different_addresses() -> CandleResult<()> {
        let device = Device::new_cuda(0)?;
        let t1 = Tensor::zeros((5, 5), candle_core::DType::F32, &device)?;
        let t2 = Tensor::ones((5, 5), candle_core::DType::F32, &device)?;
        
        let addr1 = t1.get_physical_address()?;
        let addr2 = t2.get_physical_address()?;
        
        assert_ne!(addr1, addr2, "Different tensors should have different addresses");
        Ok(())
    }
}














