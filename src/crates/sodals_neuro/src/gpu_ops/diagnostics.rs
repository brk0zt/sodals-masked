// src/gpu_ops/diagnostics.rs
// Diagnostics module for tracing tensor operations and capturing asynchronous CUDA faults.

use candle_core::{Device, Result, Tensor};

pub struct TensorTracer;

impl TensorTracer {
    /// Traces a tensor operation by enforcing strict device synchronization
    /// before and after the operation. If a silent CUDA fault (e.g. out-of-bounds,
    /// segfault) occurs, the trailing synchronization will catch it.
    pub fn trace_op<F>(name: &str, device: &Device, op: F) -> Result<Tensor>
    where
        F: FnOnce() -> Result<Tensor>,
    {
        // For CPU devices, just run the op normally without sync overhead.
        if !device.is_cuda() {
            return op();
        }

        // println!("[TRACE-START] Starting operation: {}", name);

        // 1. Sync before to clear any previous asynchronous faults.
        // We force synchronization by doing a tiny GPU->CPU copy.
        #[cfg(feature = "cuda")]
        {
            let _sync = Tensor::zeros((1,), candle_core::DType::F32, device)?.to_device(&Device::Cpu)?;
        }

        // 2. Perform the actual operation
        let result = op();

        // 3. Sync after to catch faults during THIS operation
        #[cfg(feature = "cuda")]
        {
            if device.is_cuda() {
                // Force synchronization block
                // Note: if `t` is massive this might be slow, but for debugging this is exactly what we need.
                // If we just want to sync the stream, we can copy a dummy 1-element tensor. 
                // Let's do a dummy copy instead to avoid OOM by copying the huge weight matrix `t` entirely.
                let _dummy_sync = Tensor::zeros((1,), candle_core::DType::F32, device)?.to_device(&Device::Cpu)?;
            }
        }

        match &result {
            Ok(_t) => {
                // println!("[TRACE-OK] Operation {} completed successfully. Return Shape: {:?}", name, t.shape());
            }
            Err(e) => {
                eprintln!("[TRACE-FAIL] Operation {} failed with error: {}", name, e);
            }
        }

        result
    }
}

pub fn log_vram_usage(step: &str) {
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits"
        ])
        .output()
    {
        if let Ok(stdout) = String::from_utf8(output.stdout) {
            let parts: Vec<&str> = stdout.trim().split(',').collect();
            if parts.len() == 3 {
                eprintln!("[VRAM-DIAG] Step: {:<20} | Used: {} MB | Free: {} MB | Total: {} MB", 
                         step, parts[0].trim(), parts[1].trim(), parts[2].trim());
            }
        }
    }
}














