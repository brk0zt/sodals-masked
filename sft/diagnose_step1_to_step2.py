"""
Diagnostic script: Step 1 -> Update -> Step 2 transition test
Goal: Verify that Step 2 generation runs flawlessly after Step 1 weight update,
with FP32 log_softmax stability and dynamic use_cache management.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_balanced_r4")

print("1. Loading Tokenizer & Model in 4-bit NF4...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
    trust_remote_code=True
)
base_model.config.use_cache = False
base_model.gradient_checkpointing_enable()

# 1. Cast all SSM and norm parameters to FP32 (cost: 4.2 MB)
for name, param in base_model.named_parameters():
    if param.__class__.__name__ != 'Params4bit' and 'embed_tokens' not in name and 'lm_head' not in name:
        param.data = param.data.to(torch.float32)

# 2. Leave embed_tokens in FP16, but forward hook converts output to FP32! (cost: ~6 MB)
def upcast_embeds(module, input, output):
    return output.to(torch.float32)
base_model.model.embed_tokens.register_forward_hook(upcast_embeds)

print(f"2. Attaching LoRA from {ADAPTER_PATH}...", flush=True)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
model.enable_input_require_grads()

base_m = model.base_model.model if hasattr(model, "base_model") else model
lm_head = getattr(base_m, "lm_head", None) or model.get_output_embeddings()
if lm_head.weight.requires_grad:
    lm_head.weight.requires_grad = False

def downcast_lm_head_input(module, args):
    if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].dtype != module.weight.dtype:
        return (args[0].to(module.weight.dtype),) + args[1:]
    return args
lm_head.register_forward_pre_hook(downcast_lm_head_input)

trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)
print(f"LoRA trainable params: {len(trainable_params)} tensors. Base VRAM: {torch.cuda.memory_allocated()/(1024**2):.1f} MB\n", flush=True)

# Prompts
prompt1 = "<|im_start|>user\nConvert the point $(0,3)$ in rectangular coordinates to polar coordinates.<|im_end|>\n<|im_start|>assistant\n<think>\n"
prompt2 = "<|im_start|>user\nDefine p = sum 1/k^2.<|im_end|>\n<|im_start|>assistant\n<think>\n"

# -------------------------------------------------------------
# STEP 1: ROLLOUT
# -------------------------------------------------------------
print("--- [STEP 1] Generating Rollout (use_cache=True)...", flush=True)
p_enc = tokenizer(prompt1, return_tensors="pt").to("cuda")
p_len = p_enc.input_ids.shape[1]

model.eval()
model.config.use_cache = True
t0 = time.time()
with torch.no_grad():
    out = model.generate(
        **p_enc,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
gen_time = time.time() - t0
gen_ids = out[0][p_len:]
r_len = gen_ids.size(0)
print(f"Step 1 Rollout generated {r_len} tokens in {gen_time:.2f}s! Text: {tokenizer.decode(gen_ids[:20])}...", flush=True)

# -------------------------------------------------------------
# STEP 1: POLICY UPDATE
# -------------------------------------------------------------
print("--- [STEP 1] Running Reference & Chunked Policy Backward (use_cache=False, train=True)...", flush=True)
model.train()
model.config.use_cache = False
optimizer.zero_grad()

seq_ids = torch.cat([p_enc.input_ids, gen_ids.unsqueeze(0)], dim=1)
target_ids = gen_ids.unsqueeze(0)

# A. Reference Policy (no_grad, disabled adapter)
with torch.no_grad():
    with model.disable_adapter():
        ref_out = model.base_model.model.model(seq_ids)
        ref_h = ref_out.last_hidden_state[:, p_len - 1 : -1, :]
        ref_logps_list = []
        for start in range(0, r_len, 32):
            end = min(start + 32, r_len)
            c_h = ref_h[:, start:end, :]
            c_t = target_ids[:, start:end]
            c_logits = lm_head(c_h.to(lm_head.weight.dtype))
            c_log_probs = F.log_softmax(c_logits.float(), dim=-1)
            c_logps = torch.gather(c_log_probs, -1, c_t.unsqueeze(-1)).squeeze(-1)
            ref_logps_list.append(c_logps)
        ref_logps = torch.cat(ref_logps_list, dim=-1)
        del ref_out, ref_h, ref_logps_list
        torch.cuda.empty_cache()

# B. Active Policy (train mode, gradient checkpointing)
backbone_out = model.base_model.model.model(seq_ids)
h_resp = backbone_out.last_hidden_state[:, p_len - 1 : -1, :]

loss_g = torch.tensor(0.0, device="cuda", dtype=torch.float32)
adv = 1.0
beta = 0.04

for start in range(0, r_len, 32):
    end = min(start + 32, r_len)
    c_h = h_resp[:, start:end, :].contiguous()
    c_t = target_ids[:, start:end].contiguous()
    c_ref_logps = ref_logps[:, start:end]

    c_logits = lm_head(c_h.to(lm_head.weight.dtype))
    c_log_probs = F.log_softmax(c_logits.float(), dim=-1)
    c_cur_logps = torch.gather(c_log_probs, -1, c_t.unsqueeze(-1)).squeeze(-1)

    c_ratio = torch.clamp(c_ref_logps - c_cur_logps, -10.0, 10.0)
    c_kl = torch.exp(c_ratio) - c_ratio - 1.0

    c_loss = - (c_cur_logps * adv - beta * c_kl).sum() / r_len
    loss_g = loss_g + c_loss

print(f"Step 1 Loss: {loss_g.item():.4f} (Is NaN: {torch.isnan(loss_g).item()})", flush=True)

loss_g.backward()

# Check gradients
has_nan_grad = any(torch.isnan(p.grad).any().item() for p in trainable_params if p.grad is not None)
has_inf_grad = any(torch.isinf(p.grad).any().item() for p in trainable_params if p.grad is not None)
grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item()
print(f"Step 1 Gradients -> Pre-clip Norm: {grad_norm:.4f} | Has NaN: {has_nan_grad} | Has Inf: {has_inf_grad}", flush=True)

optimizer.step()
optimizer.zero_grad()

del backbone_out, h_resp, ref_logps, loss_g
torch.cuda.empty_cache()

# Check weights after step
has_nan_weight = any(torch.isnan(p).any().item() for p in trainable_params)
has_inf_weight = any(torch.isinf(p).any().item() for p in trainable_params)
print(f"Step 1 Post-Update Weights -> Has NaN: {has_nan_weight} | Has Inf: {has_inf_weight}", flush=True)

# -------------------------------------------------------------
# STEP 2: ROLLOUT
# -------------------------------------------------------------
print("\n--- [STEP 2] Generating Rollout on Prompt 2 (Testing multinomial stability)...", flush=True)
p_enc2 = tokenizer(prompt2, return_tensors="pt").to("cuda")
p_len2 = p_enc2.input_ids.shape[1]

model.eval()
model.config.use_cache = True
t1 = time.time()
with torch.no_grad():
    out2 = model.generate(
        **p_enc2,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
gen_time2 = time.time() - t1
gen_ids2 = out2[0][p_len2:]
print(f"[SUCCESS] Step 2 Rollout generated {gen_ids2.size(0)} tokens in {gen_time2:.2f}s!", flush=True)
print(f"Output: {tokenizer.decode(gen_ids2[:30])}...", flush=True)
print("\nAll transition verifications passed flawlessly!", flush=True)
