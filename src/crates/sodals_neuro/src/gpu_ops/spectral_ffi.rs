// src/crates/sodals_neuro/src/gpu_ops/spectral_ffi.rs
use candle_core::{Tensor, Result, Device};
use std::ffi::c_void;
use crate::gpu_ops::memory_utils::SodalsCudaAccess;

extern "C" {
    fn launch_fused_resonance(
        stream: *mut c_void,
        attention_scores: u64,
        q_states: u64,
        k_states: u64,
        laplace_poles: u64,
        impedance: u64,
        neuro_flags: u64,
        token_concept_map: u64,
        layer_idx: i32,
        num_layers: i32,
        batch_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        seq_q: i32,
        seq_k: i32,
        head_dim_cplx: i32,
        n_nodes: i32,
    );
}

/// Phase 21.5: Zero-Copy Fused GEMM Attention
/// 
/// Now accepts raw u64 device pointers for poles, impedance, and flags.
/// These are zero-copy mapped memory from ZeroCopyCartridge - GPU reads directly over PCIe.
/// Pass 0 for None (null pointer).
pub fn apply_fused_resonance_gemm(
    attn_scores: &Tensor,
    q: &Tensor,
    k: &Tensor,
    poles_ptr: Option<u64>,
    impedance_ptr: Option<u64>,
    flags_ptr: Option<u64>,
    token_concept_map_ptr: Option<u64>,
    layer_idx: i32,
    num_layers: i32,
    n_nodes: i32,
) -> Result<()> {
    // Fused Resonance GEMM relies on Double-Buffer Native VRAM

    match q.device() {
        Device::Cuda(_) => {},
        _ => return Err(candle_core::Error::Msg("Fused Resonance requires a Bare-Metal CUDA device.".to_string())),
    }

    let (batch_size, num_heads, seq_q, head_dim) = q.dims4()?;
    let (_, num_kv_heads, seq_k, k_head_dim) = k.dims4()?;
    
    if head_dim != k_head_dim {
        return Err(candle_core::Error::Msg(format!(
            "Head dimension mismatch: Q={}! K={}", head_dim, k_head_dim
        )));
    }

    // Guard: If num_kv_heads is 0, the CUDA kernel will divide by zero and hang the GPU
    if num_kv_heads == 0 {
        return Err(candle_core::Error::Msg(
            "num_kv_heads is 0 — cannot launch CUDA kernel (division by zero).".to_string()
        ));
    }

    if std::env::var("SODALS_VERBOSE").is_ok() {
        eprintln!("[CUDA] apply_fused_resonance_gemm: batch={}, heads={}, kv_heads={}, seq_q={}, seq_k={}, head_dim={}, n_nodes={}, layer={}/{}",
            batch_size, num_heads, num_kv_heads, seq_q, seq_k, head_dim, n_nodes, layer_idx, num_layers);
    }

    let head_dim_cplx = (head_dim / 2) as i32;
    assert!(head_dim_cplx <= 128, "Head dimension exceeds CUDA kernel bounds (MAX_HEAD_DIM_CPLX=128)");

    let q_ptr = q.get_physical_address()?;
    let k_ptr = k.get_physical_address()?;
    let attn_ptr = attn_scores.get_physical_address()?;
    
    // Phase 21.5: Raw u64 pointers - already device addresses from ZeroCopyCartridge
    let p_ptr = poles_ptr.unwrap_or(0);
    let imp_ptr = impedance_ptr.unwrap_or(0);
    let f_ptr = flags_ptr.unwrap_or(0);
    let t_map_ptr = token_concept_map_ptr.unwrap_or(0);

    unsafe {
        launch_fused_resonance(
            std::ptr::null_mut(),
            attn_ptr,
            q_ptr,
            k_ptr,
            p_ptr,
            imp_ptr,
            f_ptr,
            t_map_ptr,
            layer_idx,
            num_layers,
            batch_size as i32,
            num_heads as i32,
            num_kv_heads as i32,
            seq_q as i32,
            seq_k as i32,
            head_dim_cplx,
            n_nodes,
        );
    }
    
    let _ = attn_scores.device(); 
    Ok(())
}
