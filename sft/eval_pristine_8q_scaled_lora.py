"""
Evaluate Pristine 8Q Domain Benchmark with Scaled LoRA (Alpha = 0.50 & 0.25)
=============================================================================
Tests whether the optimal Pareto LoRA scale (alpha = 0.50) preserves domain-specific
mastery on the 8 Pristine STEM questions while maintaining 76% on general MATH-500.

Inference: Pure Greedy (T=0, max_new_tokens=1024)
"""

import sys
import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_balanced_r4")
OUTPUT_JSON = os.path.join(ROOT, "pristine_8q_scaled_lora_results.json")

PRISTINE_QUESTIONS = [
    {
        "id": "pristine_01_lie_f4_dim",
        "domain": "Lie Algebras / Exceptional Groups",
        "prompt": "What is the dimension of the exceptional simple Lie algebra f_4 over the complex numbers?",
        "expected_gt": "52"
    },
    {
        "id": "pristine_02_top_grassmannian_gr24",
        "domain": "Topology / Schubert Calculus",
        "prompt": "Compute the topological Euler characteristic chi(Gr(2, 4)) of the complex Grassmannian of 2-planes in C^4.",
        "expected_gt": "6"
    },
    {
        "id": "pristine_03_geom_canonical_g7",
        "domain": "Algebraic Geometry / Riemann Surfaces",
        "prompt": "Let C be a smooth compact Riemann surface of genus g = 7. Compute the exact degree of the canonical divisor K on C.",
        "expected_gt": "12"
    },
    {
        "id": "pristine_04_num_elliptic_f97",
        "domain": "Number Theory / Elliptic Curves",
        "prompt": "Let E be an elliptic curve defined over the finite field F_97 having trace of Frobenius a_97 = 6. How many F_97-rational points #E(F_97) does the curve have?",
        "expected_gt": "92"
    },
    {
        "id": "pristine_05_rep_su5_adjoint_dim",
        "domain": "Representation Theory",
        "prompt": "What is the dimension of the adjoint representation of the special unitary Lie algebra su(5)?",
        "expected_gt": "24"
    },
    {
        "id": "pristine_06_symp_sp6_dim",
        "domain": "Symplectic Geometry / Matrix Groups",
        "prompt": "Compute the dimension of the real symplectic Lie algebra sp(6, R).",
        "expected_gt": "21"
    },
    {
        "id": "pristine_07_top_cp5_euler",
        "domain": "Complex Geometry / Projective Spaces",
        "prompt": "Compute the topological Euler characteristic chi(CP^5) of the 5-dimensional complex projective space.",
        "expected_gt": "6"
    },
    {
        "id": "pristine_08_num_ramanujan_tau2",
        "domain": "Modular Forms / Arithmetic",
        "prompt": "What is the exact value of the Ramanujan tau function at n = 2, tau(2)?",
        "expected_gt": "-24"
    }
]


def extract_answer(text: str) -> str:
    # Match Answer: ...
    m = re.search(r"Answer:\s*([^\n\.]+)", text)
    if m:
        return m.group(1).strip()
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        start = idx + len(r"\boxed{")
        end = text.find("}", start)
        if end != -1:
            return text[start:end].strip()
    return ""

import re

def set_lora_scale(model, scale: float):
    for module in model.modules():
        if hasattr(module, "scaling") and hasattr(module, "lora_alpha") and hasattr(module, "r"):
            for adapter_name in module.scaling:
                alpha = module.lora_alpha[adapter_name]
                r = module.r[adapter_name]
                module.scaling[adapter_name] = (alpha / r) * scale


def run_pristine_eval():
    print("=" * 80)
    print("EVALUATING PRISTINE 8Q DOMAIN BENCHMARK ACROSS SCALING FACTORS")
    print("=" * 80)

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
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    all_results = {}

    for scale in [0.50, 0.25, 0.00, 1.00]:
        print(f"\n--- TESTING SCALE {scale:.2f} ---")
        if scale == 0.0:
            # Disable adapter
            set_lora_scale(model, 0.0)
        else:
            set_lora_scale(model, scale)

        pass_count = 0
        scale_data = []

        for q in PRISTINE_QUESTIONS:
            qid = q["id"]
            gt = q["expected_gt"]
            prompt = q["prompt"]

            chat_prompt = (
                f"<|im_start|>user\n{prompt}\nPlease reason step by step, and put your final answer after 'Answer:'.<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n"
            )
            inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            elapsed = time.time() - t0

            gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            tokens_gen = len(out[0]) - inputs.input_ids.shape[1]

            ans = extract_answer(gen_text)
            is_pass = (ans == gt or gt in ans)

            if is_pass:
                pass_count += 1
                v = "PASS"
            else:
                v = "FAIL"

            print(f"[{qid}] {v} ({elapsed:.1f}s, {tokens_gen} toks) | GT: {gt} | Pred: {ans}")
            scale_data.append({"id": qid, "gt": gt, "pred": ans, "verdict": v, "tokens": tokens_gen, "time": elapsed})

        all_results[str(scale)] = {"pass": pass_count, "total": len(PRISTINE_QUESTIONS), "details": scale_data}
        print(f"Scale {scale:.2f} Score: {pass_count} / 8 ({100*pass_count/8:.1f}%)")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("FINAL DUAL BENCHMARK MATRIX (MATH-500 & PRISTINE 8Q):")
    print("=" * 80)
    print(f"Scale 0.00 (Base)  : MATH-500 = 74.0% (37/50) | Pristine 8Q = {all_results['0.0']['pass']}/8")
    print(f"Scale 0.25 (LoRA)  : MATH-500 = 76.0% (38/50) | Pristine 8Q = {all_results['0.25']['pass']}/8")
    print(f"Scale 0.50 (LoRA)  : MATH-500 = 76.0% (38/50) | Pristine 8Q = {all_results['0.5']['pass']}/8")
    print(f"Scale 1.00 (LoRA)  : MATH-500 = 56.0% (28/50) | Pristine 8Q = {all_results['1.0']['pass']}/8")


if __name__ == "__main__":
    run_pristine_eval()
