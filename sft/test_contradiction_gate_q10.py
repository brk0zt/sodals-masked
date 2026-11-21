import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_grpo_long")

sys.path.insert(0, os.path.join(ROOT, "scratch"))
from test_subscript_verifier import verify_math_chunk

print("1. Loading Tokenizer & Model in 4-bit...", flush=True)
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
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

problem = (
    "The expression $2\\cdot 3 \\cdot 4\\cdot 5+1$ is equal to 121, since multiplication is carried out before addition. "
    "However, we can obtain values other than 121 for this expression if we are allowed to change it by inserting parentheses. "
    "For example, we can obtain 144 by writing [\n"
    "(2\\cdot (3\\cdot 4)) \\cdot (5+1) = 144.\n"
    "]In total, how many values can be obtained from the expression $2\\cdot 3\\cdot 4 \\cdot 5 + 1$ by inserting parentheses? "
    "(Note that rearranging terms is not allowed, only inserting parentheses)."
)

prompt = (
    f"<|im_start|>user\n{problem}\n"
    f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
    f"<|im_start|>assistant\n<think>\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
p_len = inputs.input_ids.shape[1]

print("\n2. Generating initial greedy response...", flush=True)
t0 = time.time()
with torch.no_grad():
    out1 = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
gen_text1 = tokenizer.decode(out1[0][p_len:], skip_special_tokens=True).strip()
print(f"Initial generation completed in {time.time()-t0:.1f}s.")

# Check CST PRM Verifier
is_valid, count, contras = verify_math_chunk(gen_text1)
print(f"CST Verifier: Valid={is_valid}, Verified Eq Count={count}, Contradictions={contras}")

if not is_valid:
    print("\n3. CONTRADICTION DETECTED! Triggering Contradiction Gate...", flush=True)
    contra_summary = "; ".join(contras[:3])
    retry_prompt = (
        f"<|im_start|>user\n{problem}\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
        f"Initial calculation note: Careful verification reveals an arithmetic discrepancy in {contra_summary}. "
        f"Let me list all distinct groupings of the expression (2 * 3 * 4 * 5 + 1) explicitly and evaluate each value correctly from first principles:\n"
    )
    r_inputs = tokenizer(retry_prompt, return_tensors="pt").to("cuda")
    rp_len = r_inputs.input_ids.shape[1]
    
    t1 = time.time()
    with torch.no_grad():
        out2 = model.generate(
            **r_inputs,
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    gen_text2 = tokenizer.decode(out2[0][rp_len:], skip_special_tokens=True).strip()
    print(f"Retry generation completed in {time.time()-t1:.1f}s.")
    print("\n=== RETRY RESPONSE PREVIEW ===")
    print(gen_text2[:800])
    
    is_valid2, count2, contras2 = verify_math_chunk(gen_text2)
    print(f"\nCST Verifier after Retry: Valid={is_valid2}, Verified Eq Count={count2}, Contradictions={contras2}")
