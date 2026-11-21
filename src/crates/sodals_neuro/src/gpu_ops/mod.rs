// src/gpu_ops/mod.rs
// GPU Operations Module for SODALS Physics Kernel Sidecar

pub mod adapter;
pub use adapter::{NeuroAdapter, get_vram_ptr};

// GPU Memory Access Abstraction (Available on all platforms)
pub mod memory_utils;
pub use memory_utils::SodalsCudaAccess;

pub mod diagnostics;
pub use diagnostics::TensorTracer;

pub mod spectral_ffi;
pub use spectral_ffi::apply_fused_resonance_gemm;

pub mod atan2_ops;
pub use atan2_ops::apply_atan2_cuda;

pub mod complex_gemm_ffi;

pub mod physics_mask;
pub mod gnn_mask;











