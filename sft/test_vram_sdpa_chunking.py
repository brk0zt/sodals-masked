"""
SODALS v2.1 — Aşama 1: Saf Altyapı Doğrulaması (SDPA + Chunked Log-Probs)
=======================================================================
Hedef: 8GB VRAM (RTX 4060 Laptop) üzerinde 1536 ve 2048 token uzunluğunda
geri yayılım (backward pass) çalıştırıp WDDM paging olmadan tepe VRAM ve
hız ölçümü yapmak.

Kullanılan Saf Altyapı Bileşenleri:
1. attn_implementation="sdpa" (PyTorch Native FlashAttention / Memory-Efficient Attention)
2. Chunked lm_head + log_softmax (chunk_size=128): Vokabüler tensörü [1, L, 152064] yerine [1, 128, 152064]
3. LoRA rank r=4, gradient checkpointing aktif, lm_head dondurulmuş.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_balanced_r4")


def compute_chunked_logps(backbone, lm_head, input_ids, p_len, chunk_size=128):
    """
    Computes per-token log-probabilities without instantiating the massive
    [1, seq_len, vocab_size] logits tensor all at once in VRAM.
    """
    # 1. Forward through backbone (gradient checkpointed)
    out = backbone(input_ids)
    hidden = out.last_hidden_state  # [1, S, 4096]
    
    # Target response tokens: from p_len to S-1
    # Hidden states predicting them: from p_len - 1 to S - 2
    resp_hidden = hidden[:, p_len - 1 : -1, :]  # [1, R, 4096]
    target_ids = input_ids[:, p_len:]            # [1, R]
    R = target_ids.shape[1]
    
    logps_list = []
    
    for start in range(0, R, chunk_size):
        end = min(start + chunk_size, R)
        c_h = resp_hidden[:, start:end, :].contiguous()
        c_t = target_ids[:, start:end].contiguous()
        
        # Project only this chunk through lm_head
        c_logits = lm_head(c_h)  # [1, chunk_size, vocab]
        c_log_probs = F.log_softmax(c_logits, dim=-1)
        c_logps = torch.gather(c_log_probs, -1, c_t.unsqueeze(-1)).squeeze(-1)
        logps_list.append(c_logps)
        
    return torch.cat(logps_list, dim=1)  # [1, R]


def test_infrastructure(seq_len=1536):
    print(f"\n{'='*75}")
    print(f"TEST: Sequence Length = {seq_len} Tokens (Prompt=256, Response={seq_len-256})")
    print(f"{'='*75}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("1. Loading Qwen 3.5 9B with attn_implementation='sdpa'...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    
    print(f"2. Attaching LoRA adapter from: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    model.enable_input_require_grads()

    base_m = model.base_model.model if hasattr(model, "base_model") else model
    lm_head = getattr(base_m, "lm_head", None) or model.get_output_embeddings()
    if lm_head.weight.requires_grad:
        lm_head.weight.requires_grad = False

    backbone = model.base_model.model.model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    print(f"Model loaded in {time.time()-t0:.2f}s.")
    torch.cuda.reset_peak_memory_stats(0)
    vram_before = torch.cuda.memory_allocated(0) / (1024**2)
    print(f"Base VRAM allocated: {vram_before:.1f} MB")

    # Create synthetic test sequence
    p_len = 256
    dummy_input_ids = torch.randint(100, 50000, (1, seq_len), device="cuda")

    print(f"\n3. Running Forward + Chunked Log-Probs (chunk_size=128)...")
    t_fwd = time.time()
    logps = compute_chunked_logps(backbone, lm_head, dummy_input_ids, p_len, chunk_size=128)
    fwd_time = time.time() - t_fwd
    print(f"   Forward finished in {fwd_time:.2f}s. Output shape: {logps.shape}")

    # Synthetic advantage and dummy loss
    dummy_adv = torch.tensor(1.2, device="cuda")
    loss = - (logps * dummy_adv).mean()

    print(f"4. Running Backward Pass (Gradient Checkpointed across {seq_len} tokens)...")
    t_bwd = time.time()
    loss.backward()
    bwd_time = time.time() - t_bwd
    print(f"   Backward finished in {bwd_time:.2f}s.")

    optimizer.step()
    optimizer.zero_grad()

    peak_allocated = torch.cuda.max_memory_allocated(0) / (1024**2)
    peak_reserved = torch.cuda.max_memory_reserved(0) / (1024**2)

    print(f"\n{'='*75}")
    print(f"SONUÇLAR (Sequence Length: {seq_len} Tokens):")
    print(f"  Forward Süresi:          {fwd_time:.2f} saniye")
    print(f"  Backward Süresi:         {bwd_time:.2f} saniye")
    print(f"  Toplam Adım Süresi:      {fwd_time + bwd_time:.2f} saniye")
    print(f"  Tepe Ayrılan VRAM:       {peak_allocated:.1f} MB ({peak_allocated/1024:.2f} GB)")
    print(f"  Tepe Rezerve VRAM:       {peak_reserved:.1f} MB ({peak_reserved/1024:.2f} GB)")
    print(f"  8GB Donanım Emniyet Payı:{8188 - peak_reserved:.1f} MB BOŞTA")
    if peak_reserved < 7400:
        print(f"  [BAŞARI] WDDM Paging SINIRININ ÇOK ALTINDA! (Sıfır Disk/RAM Takası)")
    else:
        print(f"  [UYARI] VRAM sınıra yakın.")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    test_infrastructure(seq_len=1536)
