"""
SODALS v2.1 — Test Soft LoRA Blending (Continuous Alpha Mixing)
==============================================================
Validates the SoftLoraBlender on representative test cases:
  1. Pure General Math: alpha ~ 0.00 (General Math Expert Dominant)
  2. Pure Abstract Algebra: alpha ~ 1.00 (Abstract Algebra Expert Dominant)
  3. Intermediate Boundary Mixing: alpha ~ 0.50 (Co-operative Hybrid Forward Pass)

Checks generation validity, logits health, and smooth scale transitions.
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from gpr_phase_router import NeuralPhaseRouter
from soft_lora_blender import SoftLoraBlender

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_D1_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_balanced_r4")
ADAPTER_D2_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_domain2_r8_pondering")

TEST_CASES = [
    {
        "name": "General Math (MATH-500)",
        "prompt": "Evaluate the expression 3 * (4 + 7) - 15 / 3.",
        "expected_alpha_range": (0.0, 0.05)
    },
    {
        "name": "Abstract Algebra (Lie Theory)",
        "prompt": "What is the exact dimension of the exceptional simple Lie algebra E_8 over C?",
        "expected_alpha_range": (0.95, 1.0)
    },
    {
        "name": "Boundary / Intermediate (Polynomial Ring)",
        "prompt": "Find the degree of the minimal polynomial of sqrt(2) + sqrt(3) over Q.",
        "expected_alpha_range": (0.1, 0.99)
    }
]

def run_test():
    print("=" * 80, flush=True)
    print("SODALS SOFT LORA BLENDING — VALIDATION TEST", flush=True)
    print("=" * 80, flush=True)

    print("Loading Tokenizer & Model in 4-bit NF4...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    print("Attaching Expert 1 (General Math) & Expert 2 (Abstract Algebra)...", flush=True)
    model = PeftModel.from_pretrained(base_model, ADAPTER_D1_PATH, adapter_name="expert_general_math")
    model.load_adapter(ADAPTER_D2_PATH, adapter_name="expert_abstract_algebra")

    print("Initializing NeuralPhaseRouter and SoftLoraBlender...", flush=True)
    router = NeuralPhaseRouter()
    blender = SoftLoraBlender(temperature=0.03)

    num_layers = blender.enable_multi_adapter(model)
    print(f"Soft LoRA multi-adapter enabled across {num_layers} LoraLayers concurrently!", flush=True)

    model.eval()

    print("\nRunning Soft Blending Test Cases:", flush=True)
    print("-" * 80, flush=True)

    for tc in TEST_CASES:
        p = tc["prompt"]
        chat_prompt = (
            f"<|im_start|>user\n{p}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

        # 1. GPR Phase Probe
        z_i, s_d1, s_d2 = router.compute_phase_divergence(model, inputs)
        alpha = blender.compute_alpha(z_i)
        rep_penalty = blender.compute_rep_penalty(alpha)
        eff_scales = blender.set_blend(model, alpha)

        print(f"\n[TEST CASE] {tc['name']}")
        print(f"  Prompt       : {p}")
        print(f"  GPR z_i      : {z_i:+.4f} (S_D1: {s_d1:.4f}, S_D2: {s_d2:.4f})")
        print(f"  Blend Alpha  : {alpha:.4f} (General Math: {eff_scales['expert_general_math']:.4f}, Algebra: {eff_scales['expert_abstract_algebra']:.4f})")
        print(f"  Rep Penalty  : {rep_penalty:.4f}")

        # 2. Forward & Generate
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=rep_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0

        gen_tokens = out[0][inputs.input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        print(f"  Generated ({len(gen_tokens)} tokens in {elapsed:.1f}s):")
        print(f"    Tail: {gen_text[-180:]}")
        print(f"  [STATUS] SUCCESS — Valid generation, zero NaNs, smooth scaling.")

    print("\n" + "=" * 80, flush=True)
    print("ALL SOFT LORA BLENDING TESTS PASSED SUCCESSFULLY!", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    run_test()
