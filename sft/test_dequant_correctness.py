import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import sft.smart_app_control_compat as compat

def test():
    print("Testing 4-bit dequantization correctness...")
    # Create a small 4-bit quantized tensor
    K = 4096
    N = 256 # 1 block
    blocksize = 64
    shapeB = (N, K)
    
    # 2 values per byte -> N * K // 2 bytes
    B_u8 = torch.randint(0, 256, (N * (K // 2),), dtype=torch.uint8, device='cuda')
    absmax = torch.rand(N * K // blocksize, dtype=torch.float32, device='cuda') + 0.1
    from bitsandbytes.backends.utils import _get_4bit_code
    code = _get_4bit_code('nf4', 'cuda')
    
    # Dequantize with custom op
    out = torch.ops.bitsandbytes.dequantize_4bit(B_u8, absmax, blocksize, 'nf4', shapeB, torch.float16)
    print("Dequant shape:", out.shape)
    print("Any NaN/Inf?", torch.isnan(out).any().item(), torch.isinf(out).any().item())
    
    # Check GEMV kernel vs Dequant + Linear
    x = torch.randn(1, K, dtype=torch.float16, device='cuda')
    ref_y = torch.nn.functional.linear(x, out)
    
    gemv_y = torch.ops.bitsandbytes.gemv_4bit(x, B_u8, shapeB, absmax, code, blocksize)
    # Test LM_HEAD dimensions: N=248320, K=4096
    print("\n--- Testing Large LM_HEAD (N=248320, K=4096) ---")
    N_large = 248320
    shape_large = (N_large, K)
    B_large = torch.randint(0, 256, (N_large * (K // 2),), dtype=torch.uint8, device='cuda')
    absmax_large = torch.rand(N_large * K // blocksize, dtype=torch.float32, device='cuda') + 0.1
    
    # M=1 (Token Generation) with GEMV
    x_gen = torch.randn(1, 1, K, dtype=torch.float16, device='cuda')
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    y_gen = torch.ops.bitsandbytes.gemv_4bit(x_gen, B_large, shape_large, absmax_large, code, blocksize)
    t1.record()
    torch.cuda.synchronize()
    print(f"GEMV M=1 latency: {t0.elapsed_time(t1):.2f} ms | Shape: {y_gen.shape} | Finite: {torch.isfinite(y_gen).all().item()}")

    # Test chunked dequant byte slicing for M > 4
    start = 0
    end = 32768
    b_start = (start * K) // 2
    b_end = (end * K) // 2
    abs_start = (start * K) // blocksize
    abs_end = (end * K) // blocksize
    B_chunk = B_large[b_start:b_end]
    abs_chunk = absmax_large[abs_start:abs_end]
    dq_chunk = torch.ops.bitsandbytes.dequantize_4bit(B_chunk, abs_chunk, blocksize, 'nf4', (end - start, K), torch.float16)
    print(f"Chunked dequant shape: {dq_chunk.shape} | Finite: {torch.isfinite(dq_chunk).all().item()}")


if __name__ == "__main__":
    test()
