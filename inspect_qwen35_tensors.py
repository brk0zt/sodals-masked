#!/usr/bin/env python3
"""Inspect tensor names and metadata in Qwen3.5-9B GGUF model"""

try:
    from gguf import GGUFReader
    reader = GGUFReader('models/Qwen3.5-9B-Q4_K_M.gguf')
    
    print("=== GGUF Metadata ===")
    # Access fields directly
    for field_name in reader.fields:
        if any(k in field_name.lower() for k in ['attention', 'head', 'hidden', 'embedding', 'block', 'layer', 'feed', 'rope']):
            parts = reader.fields[field_name].parts
            print(f"{field_name}: {parts}")
    
    # Get all tensor names
    tensors = [t.name for t in reader.tensors]
    
    # Filter for blk.0 tensors
    blk0_tensors = sorted([t for t in tensors if 'blk.0.' in t])
    
    print("\n=== Block 0 Tensors in Qwen3.5-9B ===")
    for t in blk0_tensors:
        tensor = reader.tensors[tensors.index(t)]
        print(f"{t}: shape={tensor.shape}")
    
    # Check for specific attention tensors
    print("\n=== Checking for specific tensor patterns ===")
    patterns = [
        'blk.0.attn_output.weight',
        'blk.0.attn_output.bias',
        'blk.0.ssm_out.weight',
        'blk.0.ssm_out.bias',
        'blk.0.attn_q.weight',
        'blk.0.attn_k.weight',
        'blk.0.attn_v.weight',
        'blk.0.attn_qkv.weight',
    ]
    
    for pattern in patterns:
        exists = pattern in tensors
        print(f"{pattern}: {'FOUND' if exists else 'NOT FOUND'}")
        
except ImportError:
    print("gguf module not found. Trying candle...")
    try:
        import candle
        print("candle module found but tensor inspection not directly available")
    except ImportError:
        print("Neither gguf nor candle available for tensor inspection")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
