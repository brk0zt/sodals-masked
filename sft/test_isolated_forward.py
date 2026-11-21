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

class CPUEmbeddingWrapper(nn.Module):
    def __init__(self, cpu_embed):
        super().__init__()
        self.cpu_embed = cpu_embed
    def forward(self, input_ids):
        return self.cpu_embed(input_ids.to("cpu")).to("cuda")

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

for name, param in base_model.named_parameters():
    if param.__class__.__name__ != 'Params4bit' and 'embed_tokens' not in name and 'lm_head' not in name:
        param.data = param.data.to(torch.float32)

embed_cpu = base_model.model.embed_tokens.to("cpu")
base_model.model.embed_tokens = CPUEmbeddingWrapper(embed_cpu)
type(base_model).device = property(lambda self: torch.device("cuda:0"))

orig_lm_weight = base_model.lm_head.weight.data
l4_head = bnb.nn.Linear4bit(4096, 248320, bias=False, compute_dtype=torch.float32, quant_type='nf4')
q_w, qstate = bnb.functional.quantize_4bit(orig_lm_weight, quant_type='nf4')
l4_head.weight = bnb.nn.Params4bit(q_w, requires_grad=False, quant_state=qstate)
base_model.lm_head = l4_head
del orig_lm_weight
torch.cuda.empty_cache()

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
type(model).device = property(lambda self: torch.device("cuda:0"))
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
model.enable_input_require_grads()

backbone = model.base_model.model.model
dummy_tokens = torch.randint(100, 50000, (1, 128))
embeds = embed_cpu(dummy_tokens).to("cuda", dtype=torch.float32)

# TEST 1: Forward before anything
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(0)
print("TEST 1: Forward without generate or disable_adapter...")
out1 = backbone(inputs_embeds=embeds)
print(f"Test 1 Peak Res: {torch.cuda.max_memory_reserved(0)/(1024**2):.1f} MB")
del out1
torch.cuda.empty_cache()

# TEST 2: Forward with disable_adapter
torch.cuda.reset_peak_memory_stats(0)
print("TEST 2: Forward with disable_adapter...")
with torch.no_grad():
    with model.disable_adapter():
        out2 = backbone(inputs_embeds=embeds)
print(f"Test 2 Peak Res: {torch.cuda.max_memory_reserved(0)/(1024**2):.1f} MB")
del out2
torch.cuda.empty_cache()

# TEST 3: Generate first, then forward
torch.cuda.reset_peak_memory_stats(0)
print("TEST 3: Running generate (100 tokens)...")
model.eval()
model.config.use_cache = True
p_enc = tokenizer("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n", return_tensors="pt").to("cuda")
with torch.no_grad():
    gen_out = model.generate(**p_enc, max_new_tokens=50)
print(f"Generate Peak Res: {torch.cuda.max_memory_reserved(0)/(1024**2):.1f} MB")

# Now forward after generate
model.train()
model.config.use_cache = False
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(0)
print("TEST 4: Forward after generate...")
out4 = backbone(inputs_embeds=embeds)
print(f"Test 4 Peak Res: {torch.cuda.max_memory_reserved(0)/(1024**2):.1f} MB")
del out4
torch.cuda.empty_cache()

# TEST 5: Backward
torch.cuda.reset_peak_memory_stats(0)
print("TEST 5: Backward after generate...")
embeds_grad = embeds.detach().requires_grad_(True)
out5 = backbone(inputs_embeds=embeds_grad)
loss = out5.last_hidden_state.sum()
loss.backward()
print(f"Test 5 Peak Res (Backward): {torch.cuda.max_memory_reserved(0)/(1024**2):.1f} MB")
