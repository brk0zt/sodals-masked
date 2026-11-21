// src/gpu_ops/gnn_mask.rs
// FFI binding for GNN message pass CUDA kernel

#[cfg(feature = "cuda")]
use std::os::raw::{c_int, c_void, c_float};

#[cfg(feature = "cuda")]
extern "C" {
    pub fn apply_gnn_mask(
        stream: *mut c_void,
        wptr: *mut c_void,       // float* scores [b, h, q, k]
        q_nodes: *const c_int,
        k_nodes: *const c_int,
        indptr: *const usize,
        indices: *const c_int,
        dptr: *const c_void,     // float* data
        b: c_int,
        h: c_int,
        q: c_int,
        k: c_int,
        gnn_weight: c_float,
    );
}

// Dummy fallback for non-CUDA builds
#[cfg(not(feature = "cuda"))]
pub fn apply_gnn_mask(
    _stream: *mut std::os::raw::c_void,
    _wptr: *mut std::os::raw::c_void,
    _q_nodes: *const std::os::raw::c_int,
    _k_nodes: *const std::os::raw::c_int,
    _indptr: *const usize,
    _indices: *const std::os::raw::c_int,
    _dptr: *const std::os::raw::c_void,
    _b: std::os::raw::c_int,
    _h: std::os::raw::c_int,
    _q: std::os::raw::c_int,
    _k: std::os::raw::c_int,
    _gnn_weight: std::os::raw::c_float,
) {}
