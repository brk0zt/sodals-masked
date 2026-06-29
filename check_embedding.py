#!/usr/bin/env python3
import gguf
import numpy as np

# GGUF dosyasını aç
reader = gguf.GGUFReader("models/Qwen3.5-9B-Q4_K_M.gguf")

# Embedding tensor'ını bul
print("GGUF tensor keys containing 'emb' or 'token':")
for tensor in reader.tensors:
    if 'emb' in tensor.name.lower() or 'token' in tensor.name.lower():
        print(f"  {tensor.name}: shape={tensor.shape}, dtype={tensor.type}")

# token_embd.weight var mı?
emb_tensor = None
for tensor in reader.tensors:
    if tensor.name == 'token_embd.weight':
        emb_tensor = tensor
        break

if emb_tensor:
    data = emb_tensor.data
    print(f"\ntoken_embd.weight found:")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  First 5 values: {data.flatten()[:5]}")
    print(f"  Mean: {data.mean():.6f}")
    print(f"  Std: {data.std():.6f}")
else:
    print("\ntoken_embd.weight NOT found in GGUF!")
    print("Available tensor keys:")
    for tensor in reader.tensors[:50]:
        print(f"  {tensor.name}")
