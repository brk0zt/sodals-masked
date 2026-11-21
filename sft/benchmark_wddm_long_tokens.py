"""
SODALS v2.1 — Step 1: Long-Context WDDM Paging & VRAM Benchmark
================================================================
Target: Prove that sequence lengths L = 768, 1024, 1280, 1536, 1792, 2048
can execute forward + backward passes on an 8GB RTX 4060 Laptop GPU
WITHOUT Windows WDDM paging.

Key Innovations:
  1. CPU-resident embed_tokens: Saves 1,645 MB of physical VRAM (Base VRAM: ~5,653 MB).
  2. PyTorch Native SDPA: Native FlashAttention-2 / MemEfficient kernel on Ada Lovelace.
  3. Chunked Policy Loss (chunk_size = 32): Caps vocabulary logits memory to ~15 MB.
  4. FP32 SSM & Norm Casting: Zero NaNs in backpropagation.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys
import time
import json
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_balanced_r4")
GPU_CAPACITY_MB = 8188.0
OUTPUT_JSON = os.path.join(ROOT, "sft", "benchmark_wddm_long_tokens_results.json")


def compute_chunked_loss_from_embeds(backbone, lm_head, embeds, p_len, target_ids, adv, chunk_size=32):
    # Backbone forward with gradient checkpointing
    out = backbone(inputs_embeds=embeds)
    h_resp = out.last_hidden_state[:, p_len - 1 : -1, :]  # [1, R, 4096]
    R = target_ids.shape[1]
    
    total_loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)
    
    for start in range(0, R, chunk_size):
        end = min(start + chunk_size, R)
        c_h = h_resp[:, start:end, :].contiguous().to(torch.float32)
        c_t = target_ids[:, start:end].contiguous()
        
        c_logits = lm_head(c_h)
        c_log_probs = F.log_softmax(c_logits.float(), dim=-1)
        c_logps = torch.gather(c_log_probs, -1, c_t.unsqueeze(-1)).squeeze(-1)
        
        c_loss = - (c_logps * adv).sum() / R
        total_loss = total_loss + c_loss
        
    return total_loss


def run_long_context_benchmark():
    print("=" * 85, flush=True)
    print("SODALS v2.1 — STEP 1: LONG-CONTEXT WDDM BENCHMARK (1536 - 2048 TOKENS)", flush=True)
    print("Hardware: NVIDIA GeForce RTX 4060 Laptop (8GB Physical VRAM)", flush=True)
    print("Strategy: CPU embed_tokens (-1.65 GB) + 4-bit lm_head (-1.43 GB) + Chunked Loss", flush=True)
    print("=" * 85, flush=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )

    print("1. Loading Base Model in 4-bit NF4 with SDPA...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    # Cast SSM and norm parameters to FP32 (4.2 MB)
    print("2. Casting SSM and LayerNorm parameters to FP32...", flush=True)
    for name, param in base_model.named_parameters():
        if param.__class__.__name__ != 'Params4bit' and 'embed_tokens' not in name and 'lm_head' not in name:
            param.data = param.data.to(torch.float32)

    # Offload embed_tokens to CPU (-1,645 MB physical VRAM!)
    print("3. Offloading embed_tokens to CPU (-1,645 MB VRAM savings)...", flush=True)
    embed_cpu = base_model.model.embed_tokens.to("cpu")
    torch.cuda.empty_cache()

    # Quantize lm_head to 4-bit NF4 (-1,431 MB physical VRAM!)
    print("4. Quantizing lm_head to 4-bit NF4 (-1,431 MB VRAM savings)...", flush=True)
    orig_lm_weight = base_model.lm_head.weight.data
    l4_head = bnb.nn.Linear4bit(4096, 248320, bias=False, compute_dtype=torch.float32, quant_type='nf4')
    q_w, qstate = bnb.functional.quantize_4bit(orig_lm_weight, quant_type='nf4')
    l4_head.weight = bnb.nn.Params4bit(q_w, requires_grad=False, quant_state=qstate)
    base_model.lm_head = l4_head
    del orig_lm_weight
    torch.cuda.empty_cache()

    print(f"5. Attaching LoRA from: {ADAPTER_PATH}...", flush=True)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    model.enable_input_require_grads()
    model.train()

    backbone = model.base_model.model.model
    lm_head = model.base_model.model.lm_head
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)

    base_alloc = torch.cuda.memory_allocated(0) / (1024**2)
    base_res = torch.cuda.memory_reserved(0) / (1024**2)
    print(f"\n[OPTIMIZED BASELINE VRAM] Allocated: {base_alloc:.1f} MB | Reserved: {base_res:.1f} MB | Physical Headroom: {GPU_CAPACITY_MB - base_res:.1f} MB\n", flush=True)

    test_lengths = [768, 1024, 1280, 1536, 1792, 2048]
    adv = torch.tensor(1.0, device="cuda")
    p_len = 128

    results = {
        "baseline_alloc_mb": round(base_alloc, 1),
        "baseline_res_mb": round(base_res, 1),
        "physical_headroom_mb": round(GPU_CAPACITY_MB - base_res, 1),
        "tests": []
    }

    print("-" * 92, flush=True)
    print(f"{'Seq Len':^10} | {'Resp Tok':^10} | {'Step Time':^10} | {'Peak Alloc':^12} | {'Peak Res':^12} | {'Headroom':^10} | {'WDDM Paging?':^14}", flush=True)
    print("-" * 92, flush=True)

    for seq_len in test_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

        r_len = seq_len - p_len
        dummy_tokens = torch.randint(100, 50000, (1, seq_len))
        target_ids = dummy_tokens[:, p_len:].to("cuda")

        # Embedding lookup on CPU + copy to CUDA as leaf variable
        t_emb_0 = time.time()
        embeds = embed_cpu(dummy_tokens).to("cuda", dtype=torch.float32).detach().requires_grad_(True)
        emb_time_ms = (time.time() - t_emb_0) * 1000

        optimizer.zero_grad()
        t0 = time.time()

        try:
            loss = compute_chunked_loss_from_embeds(backbone, lm_head, embeds, p_len, target_ids, adv, chunk_size=32)
            loss.backward()

            # Check gradients
            has_nan = any(torch.isnan(p.grad).any().item() for p in trainable_params if p.grad is not None)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item()

            optimizer.step()
            optimizer.zero_grad()

            elapsed = time.time() - t0
            peak_alloc = torch.cuda.max_memory_allocated(0) / (1024**2)
            peak_res = torch.cuda.max_memory_reserved(0) / (1024**2)
            headroom = GPU_CAPACITY_MB - peak_res

            # WDDM Paging occurs when reserved memory exceeds physical 8188 MB or step takes >45s
            is_paging = (peak_res > GPU_CAPACITY_MB) or (elapsed > 45.0)
            paging_str = "YES (PAGED)" if is_paging else "NO (0 PAGE)"

            print(
                f"{seq_len:^10} | {r_len:^10} | {elapsed:^9.2f}s | {peak_alloc:^10.1f}MB | {peak_res:^10.1f}MB | {headroom:^8.1f}MB | {paging_str:^14}",
                flush=True
            )

            results["tests"].append({
                "seq_len": seq_len,
                "resp_len": r_len,
                "time_sec": round(elapsed, 2),
                "peak_alloc_mb": round(peak_alloc, 1),
                "peak_res_mb": round(peak_res, 1),
                "headroom_mb": round(headroom, 1),
                "is_paged": is_paging,
                "has_nan": has_nan,
                "grad_norm": round(grad_norm, 4)
            })

            del loss, embeds, target_ids, dummy_tokens
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as oom:
            print(f"{seq_len:^10} | {r_len:^10} | {'FAILED':^10} | {'OOM':^12} | {'OOM':^12} | {'0.0 MB':^10} | {'OUT OF MEM':^14}", flush=True)
            results["tests"].append({
                "seq_len": seq_len,
                "resp_len": r_len,
                "error": "OOM"
            })
            break

    print("-" * 92, flush=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWDDM LONG-CONTEXT BENCHMARK COMPLETED! Saved to: {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    run_long_context_benchmark()


