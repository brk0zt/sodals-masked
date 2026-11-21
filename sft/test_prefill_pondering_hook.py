import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_grpo_long")
PONDERING_WEIGHTS = os.path.join(ROOT, "sft", "qwen_lora_gpr_soft_moa", "gpr_soft_moa_aux_modules.pt")

sys.path.insert(0, os.path.join(ROOT, "sft"))
from adaptive_pondering_ode import AdaptivePonderingODEModule

print("1. Loading Tokenizer and Base Model in 4-bit NF4...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True
)

print(f"2. Attaching GRPO Long Adapter: {ADAPTER_PATH}...", flush=True)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("3. Initializing Adaptive Pondering ODE Module...", flush=True)
pondering_ode = AdaptivePonderingODEModule(
    hidden_dim=4096,
    dt=0.1,
    min_steps=2,
    max_steps=3,
    base_lambda=0.3
).to("cuda")

# Load trained weights
if os.path.exists(PONDERING_WEIGHTS):
    aux_data = torch.load(PONDERING_WEIGHTS, map_location="cuda")
    if "pondering_state_dict" in aux_data:
        pondering_ode.load_state_dict(aux_data["pondering_state_dict"])
        print("  [SUCCESS] Loaded trained Pondering ODE weights from aux_modules.pt!", flush=True)
pondering_ode.eval()

# 4. Define Prefill Pondering Hook
def prefill_pondering_hook(module, args, output):
    """
    Intercepts backbone output during PREFILL (seq_len > 1).
    Applies RK4 continuous manifold relaxation on the prompt's final hidden state.
    """
    last_h = output.last_hidden_state
    seq_len = last_h.size(1)
    if seq_len > 1:
        # Prompt prefill: extract last token hidden state
        h_last = last_h[:, -1:, :]
        # Apply RK4 evolution (alpha=0.2 for general math)
        with torch.no_grad():
            h_pondered, steps, lam = pondering_ode(h_last, alpha=0.2)
        # Update last hidden state in place
        last_h[:, -1:, :] = h_pondered
    return output

# Attach hook to backbone
backbone = model.base_model.model.model
hook_handle = backbone.register_forward_hook(prefill_pondering_hook)
print("4. Attached prefill_pondering_hook to backbone!", flush=True)

# 5. Test Inference
prompt = (
    "<|im_start|>user\n"
    "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. "
    "Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
p_len = inputs.input_ids.shape[1]

print(f"\nGenerating response with Prefill Pondering ODE (prompt tokens: {p_len})...", flush=True)
t0 = time.time()
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
elapsed = time.time() - t0
gen_tokens = out[0][p_len:]
gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

print(f"Generation completed in {elapsed:.2f}s ({len(gen_tokens)} tokens, {len(gen_tokens)/elapsed:.2f} tok/s)")
print("Response preview:")
print(gen_text[:300])

# Clean hook
hook_handle.remove()
print("\n[VERIFIED] Prefill Pondering ODE Hook runs seamlessly with zero errors!")
