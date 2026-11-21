import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_balanced_r4")

class CPUEmbeddingWrapper(nn.Module):
    def __init__(self, cpu_embed):
        super().__init__()
        self.cpu_embed = cpu_embed
    def forward(self, input_ids):
        return self.cpu_embed(input_ids.to("cpu")).to("cuda")

print("Loading model with CPU embed_tokens and native FP16 lm_head...", flush=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    attn_implementation="sdpa"
)
base_model.config.use_cache = False
base_model.gradient_checkpointing_enable()

# SSM / Norm FP32
for name, param in base_model.named_parameters():
    if param.__class__.__name__ != 'Params4bit' and 'embed_tokens' not in name and 'lm_head' not in name:
        param.data = param.data.to(torch.float32)

# CPU embed_tokens
embed_cpu = base_model.model.embed_tokens.to("cpu")
base_model.model.embed_tokens = CPUEmbeddingWrapper(embed_cpu)
type(base_model).device = property(lambda self: torch.device("cuda:0"))

# Native FP16 lm_head (frozen)
base_model.lm_head.weight.requires_grad = False
base_model.lm_head.to(torch.float16)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
type(model).device = property(lambda self: torch.device("cuda:0"))
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
model.enable_input_require_grads()

backbone = model.base_model.model.model
lm_head = model.base_model.model.lm_head
trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(0)

base_alloc = torch.cuda.memory_allocated(0) / (1024**2)
base_res = torch.cuda.memory_reserved(0) / (1024**2)
print(f"Base Allocated: {base_alloc:.1f} MB | Reserved: {base_res:.1f} MB | Headroom: {8188.0 - base_res:.1f} MB", flush=True)

# Test sequence length 1536 (prompt=128, resp=1408)
seq_len = 1536
p_len = 128
r_len = seq_len - p_len
dummy_tokens = torch.randint(100, 50000, (1, seq_len))
target_ids = dummy_tokens[:, p_len:].to("cuda")
embeds = embed_cpu(dummy_tokens).to("cuda", dtype=torch.float32).detach().requires_grad_(True)

print(f"\nTesting Forward + Backward for SeqLen = {seq_len} (r_len = {r_len})...", flush=True)
t0 = time.time()

# 1. Forward
out = backbone(inputs_embeds=embeds)
h_resp = out.last_hidden_state[:, p_len - 1 : -1, :]

# 2. Chunked Loss with FP16 lm_head
loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)
for start in range(0, r_len, 32):
    end = min(start + 32, r_len)
    c_h = h_resp[:, start:end, :].contiguous().to(torch.float16)
    c_t = target_ids[:, start:end].contiguous()
    c_logits = lm_head(c_h)
    c_log_probs = F.log_softmax(c_logits.float(), dim=-1)
    c_logps = torch.gather(c_log_probs, -1, c_t.unsqueeze(-1)).squeeze(-1)
    c_loss = - c_logps.sum() / r_len
    loss = loss + c_loss

# 3. Backward
loss.backward()
optimizer.step()
optimizer.zero_grad()

elapsed = time.time() - t0
peak_alloc = torch.cuda.max_memory_allocated(0) / (1024**2)
peak_res = torch.cuda.max_memory_reserved(0) / (1024**2)
is_paged = (peak_res > 8188.0) or (elapsed > 45.0)

print("=" * 70, flush=True)
print(f"Elapsed Time:       {elapsed:.2f} s")
print(f"Peak Allocated:     {peak_alloc:.1f} MB")
print(f"Peak Reserved:      {peak_res:.1f} MB")
print(f"Physical Headroom:  {8188.0 - peak_res:.1f} MB")
print(f"WDDM Paged?         {'YES (PAGED)' if is_paged else 'NO (0 PAGE!)'}")
print("=" * 70, flush=True)
