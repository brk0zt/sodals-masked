# bci_telepathy.py
import torch
import sodals_bci # Compiled PyO3 Rust module
from tensor import GraphTensor # Phase 62.1 Triton Primitive

def transmit_to_rust(graph_tensor: GraphTensor):
    """
    Extracts the raw CUDA memory address from the GraphTensor
    and fires it directly into the Rust RK4 PonderingCore.
    """
    # Ensure the tensor is contiguous in VRAM and is float32 (Real amplitudes for RK4)
    # The Hilbert Transform output is complex64; we extract absolute amplitudes for the Arkhe field.
    if graph_tensor.hd_vector is not None:
        amplitudes = torch.abs(graph_tensor.hd_vector).to(torch.float32).contiguous()
        
        # O(1) Zero-Copy FFI Transmission
        ptr = amplitudes.data_ptr()
        dim = amplitudes.shape[-1]
        
        # Fire into Rust!
        sodals_bci.inject_ghost_tensor(ptr, dim)
        print(f"[TELEPATHY] Ghost Vector injected at VRAM Address: {hex(ptr)}")
