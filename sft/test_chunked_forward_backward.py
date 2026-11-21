"""
Test chunked lm_head forward + backward on Qwen 3.5 9B NF4
Goal: Determine the exact maximum sequence length (384, 512, 768)
that fits cleanly in 8GB VRAM without triggering WDDM paging.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_balanced_r4")


def compute_chunked_loss(backbone, lm_head, input_ids, p_len, adv, chunk_size=64):
    out = backbone(input_ids)
    hidden = out.last_hidden_state  # [1, S, 4096]
    
    resp_hidden = hidden[:, p_len - 1 : -1, :]  # [1, R, 4096]
    target_ids = input_ids[:, p_len:]            # [1, R]
    R = target_ids.shape[1]
    
    total_loss = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)
    
    for start in range(0, R, chunk_size):
        end = min(start + chunk_size, R)
        c_h = resp_hidden[:, start:end, :].contiguous()
        c_t = target_ids[:, start:end].contiguous()
        
        c_logits = lm_head(c_h)  # [1, chunk, 248320]
        c_log_probs = F.log_softmax(c_logits, dim=-1)
        c_logps = torch.gather(c_log_probs, -1, c_t.unsqueeze(-1)).squeeze(-1)
        
        c_loss = - (c_logps * adv).sum() / R
        total_loss = total_loss + c_loss
        
    return total_loss


def run_benchmark():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("Loading model onto GPU (device_map={'':0})...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    
    print(f"Attaching LoRA adapter from: {ADAPTER_PATH}...", flush=True)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    model.enable_input_require_grads()
    model.train()

    base_m = model.base_model.model if hasattr(model, "base_model") else model
    lm_head = getattr(base_m, "lm_head", None) or model.get_output_embeddings()
    if lm_head.weight.requires_grad:
        lm_head.weight.requires_grad = False

    backbone = model.base_model.model.model
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)

    base_alloc = torch.cuda.memory_allocated(0) / (1024**2)
    print(f"Base Allocated VRAM: {base_alloc:.1f} MB (Capacity: 8188 MB)\n", flush=True)

    test_lengths = [384, 512, 640, 768]
    adv = torch.tensor(1.0, device="cuda")

    for seq_len in test_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        
        p_len = 128
        r_len = seq_len - p_len
        dummy_input = torch.randint(100, 50000, (1, seq_len), device="cuda")
        
        optimizer.zero_grad()
        t0 = time.time()
        try:
            loss = compute_chunked_loss(backbone, lm_head, dummy_input, p_len, adv, chunk_size=32)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step_time = time.time() - t0
            
            peak_alloc = torch.cuda.max_memory_allocated(0) / (1024**2)
            peak_res = torch.cuda.max_memory_reserved(0) / (1024**2)
            margin = 8188 - peak_res
            
            status = "OK (SERIN & HIZLI)" if peak_res < 7800 else "UYARI (WDDM Paging Riski)"
            print(f"[L={seq_len:4d} (Resp={r_len:4d})] Süre: {step_time:5.2f}s | Tepe Alloc: {peak_alloc:6.1f}MB | Tepe Res: {peak_res:6.1f}MB | Kalan: {margin:5.1f}MB | {status}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[L={seq_len:4d} (Resp={r_len:4d})] OOM: VRAM Yetersiz!", flush=True)
            break
        except Exception as e:
            print(f"[L={seq_len:4d} (Resp={r_len:4d})] Hata: {e}", flush=True)
            break

    print("\nBenchmark tamamlandı.", flush=True)


if __name__ == "__main__":
    run_benchmark()
