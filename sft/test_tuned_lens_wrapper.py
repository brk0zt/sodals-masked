import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tuned_lens_wrapper import TunedLensWrapper

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_D1_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_grpo_long")
JLENS_PATH = os.path.join(ROOT, "jlens_weights.safetensors")

print("Initializing Unit Test for TunedLensWrapper...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb,
    device_map={"": 0},
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, ADAPTER_D1_PATH, adapter_name="expert_general_math")
model.eval()

wrapper = TunedLensWrapper(JLENS_PATH, layers=(28, 30), device="cuda", dtype=torch.float16)

prompt = "Question: Find the value of x if 2x + 10 = 30.\nAnswer: 10"
inp = tok(prompt, return_tensors="pt").to("cuda")

t0 = time.time()
report = wrapper.evaluate_sequence(model, tok, inp.input_ids)
dur = time.time() - t0

print("\n--- Tuned Lens Stability Report ---")
print(f"Mean Stability:     {report.mean_stability}")
print(f"Answer Stability:   {report.answer_stability}")
print(f"Min Stability:      {report.min_stability}")
print(f"Cos Layer 28:       {report.cos_layer28}")
print(f"Cos Layer 30:       {report.cos_layer30}")
print(f"Top-1 Agreement:    {report.top1_agreement} ('{report.top1_token}')")
print(f"Impedance:          {report.impedance}")
print(f"Geometric Score:    {report.geometric_score}")
print(f"Evaluation Time:    {dur*1000:.1f} ms")
print(f"Peak VRAM:          {torch.cuda.max_memory_allocated() / (1024**2):.1f} MB")
print("Unit Test Passed successfully!")
