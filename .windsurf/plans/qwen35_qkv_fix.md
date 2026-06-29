# Qwen3.5 QKV Dimension Mismatch Fix

## Problem
The Qwen3.5-9B model has QKV weight shape [4096, 8192], producing 8192 dimensions, but the code expects:
- 6144 (Q+K+V with GQA: 4096 + 1024 + 1024)
- 5120 (Q+KV shared: 4096 + 1024)

## Current Configuration (from GGUF metadata)
- hidden_size: 4096
- num_attention_heads: 16
- num_key_value_heads: 4
- head_dim: 256
- QKV weight shape: [4096, 8192]

## Analysis
8192 = 4096 + 4096, suggesting the layout is:
- Q: 4096 (16 heads * 256 head_dim)
- K: 4096 (full hidden size, not GQA-reduced)
- V: might be derived from K or separate tensor

OR

8192 = 4096 + 2048 + 2048, suggesting:
- Q: 4096 (16 heads * 256)
- K: 2048 (8 heads * 256)
- V: 2048 (8 heads * 256)

## Required Fix
Update `src\crates\sodals_neuro\src\models\qwen_custom.rs` line 728-760 to handle the 8192-dimension QKV layout:

1. Add case for `qkv_dim == 8192`:
   - Extract Q: first 4096 dims
   - Extract K: next 2048 dims (if layout is Q+K+V with 8 KV heads)
   - Extract V: last 2048 dims
   - OR extract K as full 4096 and use GQA logic to reduce

2. May need to update head configuration if K/V use different head counts than metadata suggests

## Implementation Location
File: `src\crates\sodals_neuro\src\models\qwen_custom.rs`
Function: `QAttention::forward`
Lines: ~728-760 (QKV projection logic)
