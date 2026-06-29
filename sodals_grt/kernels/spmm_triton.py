import torch
import triton
import triton.language as tl

@triton.jit
def spmm_sdr_kernel(
    row_ptr, col_idx, values,
    x_ptr, sdr_mask_ptr,
    out_ptr,
    num_rows, num_cols, in_features,
    stride_x_row, stride_x_col,
    stride_out_row, stride_out_col,
    stride_mask_row, stride_mask_col,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    pid_feat = tl.program_id(1)
    
    if row_id >= num_rows:
        return

    feat_offset = pid_feat * BLOCK_SIZE
    feat_idx = feat_offset + tl.arange(0, BLOCK_SIZE)
    
    feat_mask = feat_idx < in_features
    
    # Load SDR mask for the output row and current feature block
    mask_ptrs = sdr_mask_ptr + row_id * stride_mask_row + feat_idx * stride_mask_col
    sdr_active = tl.load(mask_ptrs, mask=feat_mask, other=0.0)
    
    # Prune FLOPs at the warp scheduling level
    compute_mask = (sdr_active > 0.0) & feat_mask
    if not tl.math.sum(compute_mask, axis=0) > 0:
        return

    row_start = tl.load(row_ptr + row_id)
    row_end = tl.load(row_ptr + row_id + 1)
    
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for edge_idx in range(row_start, row_end):
        col = tl.load(col_idx + edge_idx)
        val = tl.load(values + edge_idx)
        
        # Strictly masked coalesced load to prevent out-of-bounds segfaults
        x_ptrs = x_ptr + col * stride_x_row + feat_idx * stride_x_col
        x_vals = tl.load(x_ptrs, mask=compute_mask, other=0.0)
        
        accumulator += val * x_vals

    out_ptrs = out_ptr + row_id * stride_out_row + feat_idx * stride_out_col
    tl.store(out_ptrs, accumulator, mask=compute_mask)


def spmm_triton_forward(row_ptr: torch.Tensor, col_idx: torch.Tensor, values: torch.Tensor, x: torch.Tensor, sdr_mask: torch.Tensor) -> torch.Tensor:
    """
    Launch bounds-checked native SDR SpMM computation.
    All inputs must be contiguous on CUDA device.
    """
    assert x.is_cuda and row_ptr.is_cuda and col_idx.is_cuda and values.is_cuda and sdr_mask.is_cuda
    assert x.is_contiguous()
    assert sdr_mask.is_contiguous()

    num_rows = row_ptr.shape[0] - 1
    num_cols, in_features = x.shape
    
    out = torch.zeros((num_rows, in_features), device=x.device, dtype=x.dtype)
    
    # Optimize warp occupancy based on feature vector quantization
    BLOCK_SIZE = triton.next_power_of_2(min(in_features, 128))
    if BLOCK_SIZE < 16:
        BLOCK_SIZE = 16
        
    grid = lambda meta: (num_rows, triton.cdiv(in_features, meta['BLOCK_SIZE']))
    
    spmm_sdr_kernel[grid](
        row_ptr, col_idx, values,
        x, sdr_mask,
        out,
        num_rows, num_cols, in_features,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        sdr_mask.stride(0), sdr_mask.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
