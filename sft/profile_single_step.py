import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
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

def print_mem(tag):
    alloc = torch.cuda.memory_allocated(0) / (1024**2)
    res = torch.cuda.memory_reserved(0) / (1024**2)
    max_res = torch.cuda.max_memory_reserved(0) / (1024**2)
    print(f"[{tag:25s}] Alloc: {alloc:7.1f} MB | Res: {res:7.1f} MB | MaxRes: {max_res:7.1f} MB", flush=True)

print("1. Loading Tokenizer & Model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
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

# CPU embed
embed_cpu = base_model.model.embed_tokens.to("cpu")
base_model.model.embed_tokens = CPUEmbeddingWrapper(embed_cpu)
type(base_model).device = property(lambda self: torch.device("cuda:0"))

# 4-bit lm_head
orig_lm_weight = base_model.lm_head.weight.data
l4_head = bnb.nn.Linear4bit(4096, 248320, bias=False, compute_dtype=torch.float32, quant_type='nf4')
q_w, qstate = bnb.functional.quantize_4bit(orig_lm_weight, quant_type='nf4')
l4_head.weight = bnb.nn.Params4bit(q_w, requires_grad=False, quant_state=qstate)
base_model.lm_head = l4_head
del orig_lm_weight
torch.cuda.empty_cache()

# LoRA
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

torch.cuda.reset_peak_memory_stats(0)
print_mem("Initial Setup")

prompt_str = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n"
p_enc = tokenizer(prompt_str, return_tensors="pt").to("cuda")
p_len = p_enc.input_ids.shape[1]

# Rollout
model.eval()
model.config.use_cache = True
t0 = time.time()
print("\nStarting Rollout generation...", flush=True)
with torch.no_grad():
    out = model.generate(**p_enc, max_new_tokens=300, do_sample=True, temperature=0.7)
gen_ids = out[0][p_len:]
print(f"Rollout done in {time.time()-t0:.2f}s ({len(gen_ids)} tokens)")
print_mem("After Rollout")

# Backward
model.train()
model.config.use_cache = False
torch.cuda.empty_cache()
print_mem("After train mode switch")

seq_ids = torch.cat([p_enc.input_ids, gen_ids.unsqueeze(0)], dim=1)
target_ids = gen_ids.unsqueeze(0).to("cuda")
r_len = gen_ids.size(0)

# Reference Forward
t_ref = time.time()
seq_embeds = embed_cpu(seq_ids.cpu()).to("cuda", dtype=torch.float32)
with torch.no_grad():
    with model.disable_adapter():
        ref_out = backbone(inputs_embeds=seq_embeds)
        ref_h = ref_out.last_hidden_state[:, p_len - 1 : -1, :]
        ref_logps_list = []
        for start in range(0, r_len, 32):
            end = min(start + 32, r_len)
            c_h = ref_h[:, start:end, :].contiguous().to(torch.float32)
            c_t = target_ids[:, start:end].contiguous()
            c_logits = lm_head(c_h)
            c_log_probs = F.log_softmax(c_logits.float(), dim=-1)
            c_logps = torch.gather(c_log_probs, -1, c_t.unsqueeze(-1)).squeeze(-1)
            ref_logps_list.append(c_logps)
        ref_logps = torch.cat(ref_logps_list, dim=-1)
        del ref_out, ref_h, ref_logps_list
        torch.cuda.empty_cache()
print(f"Reference Forward done in {time.time()-t_ref:.2f}s")
print_mem("After Ref Forward")

# Active Forward
t_act = time.time()
act_embeds = seq_embeds.detach().requires_grad_(True)
backbone_out = backbone(inputs_embeds=act_embeds)
h_resp = backbone_out.last_hidden_state[:, p_len - 1 : -1, :]
print(f"Active Forward done in {time.time()-t_act:.2f}s")
print_mem("After Active Forward")

# Chunked Loss + Backward
t_loss = time.time()
loss = torch.tensor(0.0, device="cuda", dtype=torch.float32)
for start in range(0, r_len, 32):
    end = min(start + 32, r_len)
    c_h = h_resp[:, start:end, :].contiguous().to(torch.float32)
    c_t = target_ids[:, start:end].contiguous()
    c_ref_logps = ref_logps[:, start:end]
    c_logits = lm_head(c_h)
    c_log_probs = F.log_softmax(c_logits.float(), dim=-1)
    c_cur_logps = torch.gather(c_log_probs, -1, c_t.unsqueeze(-1)).squeeze(-1)
    c_ratio = torch.clamp(c_ref_logps - c_cur_logps, -10.0, 10.0)
    c_kl = torch.exp(c_ratio) - c_ratio - 1.0
    c_loss = - (c_cur_logps * 1.0 - 0.04 * c_kl).sum() / r_len
    loss = loss + c_loss

loss.backward()
print(f"Chunked Loss + Backward done in {time.time()-t_loss:.2f}s")
print_mem("After Backward")

optimizer.step()
optimizer.zero_grad()
print_mem("After Optimizer Step")
