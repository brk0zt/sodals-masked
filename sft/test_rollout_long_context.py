"""
SODALS v2.1 — Step 1 Verification: Long-Context Rollout Generation
==================================================================
Test autoregressive generation up to 1024-1536 tokens using:
  1. CPUEmbeddingWrapper (embed_tokens in CPU RAM, zero VRAM footprint)
  2. 4-bit NF4 lm_head (508 MB instead of 1,940 MB)
  3. SDPA FlashAttention-2 kernels
Verifies tokens/second generation speed, KV-cache VRAM usage, and zero WDDM paging.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys
import time
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_balanced_r4")
GPU_CAPACITY_MB = 8188.0


class CPUEmbeddingWrapper(nn.Module):
    def __init__(self, cpu_embed):
        super().__init__()
        self.cpu_embed = cpu_embed
    def forward(self, input_ids):
        ids_cpu = input_ids.to("cpu")
        embeds = self.cpu_embed(ids_cpu)
        return embeds.to("cuda")


def run_rollout_test():
    print("=" * 80, flush=True)
    print("SODALS v2.1 — LONG-CONTEXT ROLLOUT GENERATION VERIFICATION", flush=True)
    print("=" * 80, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )

    print("1. Loading Model with SDPA...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa"
    )

    # Cast SSM & Norm
    for name, param in base_model.named_parameters():
        if param.__class__.__name__ != 'Params4bit' and 'embed_tokens' not in name and 'lm_head' not in name:
            param.data = param.data.to(torch.float32)

    # Wrap embed_tokens to CPU
    print("2. Wrapping embed_tokens to CPU RAM...", flush=True)
    embed_cpu = base_model.model.embed_tokens.to("cpu")
    base_model.model.embed_tokens = CPUEmbeddingWrapper(embed_cpu)
    torch.cuda.empty_cache()

    # Quantize lm_head to 4-bit NF4
    print("3. Quantizing lm_head to 4-bit NF4...", flush=True)
    orig_lm_weight = base_model.lm_head.weight.data
    l4_head = bnb.nn.Linear4bit(4096, 248320, bias=False, compute_dtype=torch.float32, quant_type='nf4')
    q_w, qstate = bnb.functional.quantize_4bit(orig_lm_weight, quant_type='nf4')
    l4_head.weight = bnb.nn.Params4bit(q_w, requires_grad=False, quant_state=qstate)
    base_model.lm_head = l4_head
    del orig_lm_weight
    torch.cuda.empty_cache()

    print("4. Attaching LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    base_alloc = torch.cuda.memory_allocated(0) / (1024**2)
    base_res = torch.cuda.memory_reserved(0) / (1024**2)
    print(f"\n[READY VRAM] Allocated: {base_alloc:.1f} MB | Reserved: {base_res:.1f} MB | Headroom: {GPU_CAPACITY_MB - base_res:.1f} MB\n", flush=True)

    # Test prompt from MATH-500
    prompt_text = "Please solve the following math problem step by step: Let P(x) = x^4 + 2x^3 - 7x^2 - 8x + 12. Find the sum of the squares of the real roots of P(x) = 0."
    prompt = f"<|im_start|>system\nYou are an expert mathematician. Think through the problem thoroughly step-by-step, showing all intermediate deductions.<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n<thought>\n"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    p_len = input_ids.shape[1]
    print(f"Prompt tokens: {p_len}", flush=True)
    print("Generating up to 1024 new tokens...", flush=True)

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    t_gen = time.time() - t0
    new_tokens = out.shape[1] - p_len
    tok_per_sec = new_tokens / t_gen if t_gen > 0 else 0

    peak_alloc = torch.cuda.max_memory_allocated(0) / (1024**2)
    peak_res = torch.cuda.max_memory_reserved(0) / (1024**2)
    headroom = GPU_CAPACITY_MB - peak_res

    gen_text = tokenizer.decode(out[0, p_len:], skip_special_tokens=False)

    print("\n" + "=" * 80)
    print("ROLLOUT GENERATION RESULTS:")
    print(f"Generated Tokens:  {new_tokens} tokens")
    print(f"Generation Time:   {t_gen:.2f} s")
    print(f"Generation Speed:  {tok_per_sec:.2f} tokens/s")
    print(f"Peak VRAM Alloc:   {peak_alloc:.1f} MB")
    print(f"Peak VRAM Reserved:{peak_res:.1f} MB")
    print(f"Physical Headroom: {headroom:.1f} MB")
    print(f"WDDM Paging:       {'YES (PAGED)' if peak_res > GPU_CAPACITY_MB else 'NO (0 PAGE)'}")
    print("=" * 80)
    print("\nFirst 300 characters of generation:\n", gen_text[:300])
    print("\nLast 300 characters of generation:\n", gen_text[-300:] if len(gen_text) > 300 else gen_text)


if __name__ == "__main__":
    run_rollout_test()
