"""
SODALS v2.1 — Expand SDPO-A Adapter from Rank 4 to Rank 8
==========================================================
Expands lora_A and lora_B tensors in `qwen_lora_adapter_sdpo_anchored` from r=4 to r=8:
  - lora_A: [4, in_dim] -> [8, in_dim] (rows 0..3: old weights, rows 4..7: kaiming uniform / 0.01 randn)
  - lora_B: [out_dim, 4] -> [out_dim, 8] (cols 0..3: old weights, cols 4..7: exact zeros)
  - alpha: 8 -> 16 (preserves exact alpha/r = 2 scaling factor)

This guarantees ΔW_new = ΔW_old at step 0 (exact identity preservation),
while unlocking 4 additional orthogonal degrees of freedom for training!
"""

import os
import json
import shutil
import torch
from safetensors.torch import load_file, save_file

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
SRC_DIR = os.path.join(SFT_DIR, "qwen_lora_adapter_sdpo_anchored")
DST_DIR = os.path.join(SFT_DIR, "qwen_lora_adapter_sdpo_anchored_r8")

def expand_adapter():
    os.makedirs(DST_DIR, exist_ok=True)
    
    # 1. Update adapter_config.json
    with open(os.path.join(SRC_DIR, "adapter_config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    
    config["r"] = 8
    config["lora_alpha"] = 16
    
    with open(os.path.join(DST_DIR, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[EXPAND] Saved updated adapter_config.json with r=8, alpha=16 to {DST_DIR}")
    
    # 2. Expand safetensors
    src_weights_path = os.path.join(SRC_DIR, "adapter_model.safetensors")
    tensors = load_file(src_weights_path)
    new_tensors = {}
    
    torch.manual_seed(1337)
    for k, v in tensors.items():
        if "lora_A.weight" in k:
            # v shape: [4, in_features] -> [8, in_features]
            old_r, in_dim = v.shape
            assert old_r == 4, f"Expected r=4, got {old_r}"
            extra_A = 0.01 * torch.randn(4, in_dim, dtype=v.dtype)
            new_A = torch.cat([v, extra_A], dim=0)
            new_tensors[k] = new_A
        elif "lora_B.weight" in k:
            # v shape: [out_features, 4] -> [out_features, 8]
            out_dim, old_r = v.shape
            assert old_r == 4, f"Expected r=4, got {old_r}"
            extra_B = torch.zeros(out_dim, 4, dtype=v.dtype)
            new_B = torch.cat([v, extra_B], dim=1)
            new_tensors[k] = new_B
        else:
            new_tensors[k] = v
            
    dst_weights_path = os.path.join(DST_DIR, "adapter_model.safetensors")
    save_file(new_tensors, dst_weights_path)
    print(f"[EXPAND] Saved {len(new_tensors)} expanded r=8 tensors to {dst_weights_path}")
    
    # 3. Copy other auxiliary files (tokenizer, etc.)
    for fname in os.listdir(SRC_DIR):
        if fname not in ["adapter_config.json", "adapter_model.safetensors"]:
            src_f = os.path.join(SRC_DIR, fname)
            dst_f = os.path.join(DST_DIR, fname)
            if os.path.isfile(src_f):
                shutil.copy2(src_f, dst_f)
    print(f"[EXPAND] Copied tokenizer and config files successfully.")
    print("[SUCCESS] SDPO-A r=8 Warm-Start Adapter Ready!")

if __name__ == "__main__":
    expand_adapter()
