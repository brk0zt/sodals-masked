// src/crates/sodals_neuro/src/gpu_ops/atan2_ops.rs
// FFI Bridge for the atan2 CUDA kernel

use std::ffi::c_void;

extern "C" {
    /// Applies atan2f(y, x) element-wise on device.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream for asynchronous execution
    /// * `out_ptr` - F32 pointer to output tensor
    /// * `y_ptr` - F32 pointer to Y (imaginary) tensor
    /// * `x_ptr` - F32 pointer to X (real) tensor
    /// * `num_elements` - Number of elements to process
    pub fn apply_atan2_cuda(
        stream: *mut c_void,
        out_ptr: *mut c_void,
        y_ptr: *const c_void,
        x_ptr: *const c_void,
        num_elements: i32,
    );
}
