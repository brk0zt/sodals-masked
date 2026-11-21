"""
SODALS v3.0 — End-to-End Evaluation with MCTS + SymPy Symbolic Verification
===========================================================================
Evaluates the model using Token-Level MCTS with runtime SymPy verification
on the 4 OOD perturbation questions (especially Question 3 where the base LoRA made a sign error).
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
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_with_pondering")
ODE_WEIGHTS_PATH = os.path.join(ADAPTER_PATH, "arkhe_ode_proj.pt")

sys.path.insert(0, os.path.join(ROOT, "sft"))
from token_mcts_engine import TokenMctsEngine

PERTURBED_QUESTIONS = [
    {
        "id": "unseen_geom_canonical_degree_g4",
        "topic": "Canonical Divisor Genus 4 (Failed without MCTS)",
        "prompt": "Let C be a smooth compact Riemann surface of genus g = 4. Compute the exact degree of the canonical divisor K on C.",
        "expected_gt": "6"
    },
    {
        "id": "unseen_top_euler_cp4",
        "topic": "Euler Characteristic CP^4",
        "prompt": "Compute the topological Euler characteristic \\chi(\\mathbb{C}P^4) of the 4-dimensional complex projective space.",
        "expected_gt": "5"
    },
    {
        "id": "unseen_lie_dim_e6",
        "topic": "Dimension of Simple Lie Algebra e_6",
        "prompt": "What is the dimension of the exceptional simple Lie algebra \\mathfrak{e}_6 over the complex numbers?",
        "expected_gt": "78"
    },
    {
        "id": "unseen_num_point_count_f53",
        "topic": "Elliptic Curve F_53 a_p = -4",
        "prompt": "Let E be an elliptic curve defined over the finite field \\mathbb{F}_{53} having trace of Frobenius a_{53} = -4. How many \\mathbb{F}_{53}-rational points #E(\\mathbb{F}_{53}) does the curve have?",
        "expected_gt": "58"
    }
]


def run_evaluation():
    print("="*80)
    print("SODALS v3.0: EVALUATION WITH TOKEN MCTS + SYMPY SYMBOLIC VERIFICATION")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Base Model in 4-bit NF4 onto GPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    print(f"Loading LoRA Adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("Initializing TokenMctsEngine with SymPy Symbolic Verifier...")
    engine = TokenMctsEngine(
        model=model,
        tokenizer=tokenizer,
        num_branches=3,
        mcts_steps=8,
        chunk_token_limit=96,
        min_reasoning_tokens=80,
        max_total_tokens=2048,
        c_puct=1.414,
        temperature=0.6,
        ode_weights_path=ODE_WEIGHTS_PATH,
        verbose=True
    )

    results = []

    for i, q in enumerate(PERTURBED_QUESTIONS):
        print(f"\n{'='*80}\nEVALUATING QUESTION {i+1}/{len(PERTURBED_QUESTIONS)}: {q['topic']}\n{'='*80}")
        print(f"Prompt: {q['prompt']}")
        print(f"Expected Ground Truth: {q['expected_gt']}\n")

        res = engine.generate(q["prompt"])

        clean_text = res["response"]
        gt = q["expected_gt"]
        is_exact = f"Answer: {gt}" in clean_text or f"answer is: {gt}" in clean_text or f"Answer: {gt}." in clean_text
        has_gt = gt in clean_text

        verdict = "PASS (Exact)" if is_exact else ("PARTIAL (GT in text)" if has_gt else "FAIL")

        print(f"\n[FINAL RESPONSE FOR {q['id']}]:\n{clean_text}\n")
        print(f"Tokens: {res['tokens_generated']} | Time: {res['elapsed_seconds']}s | Verdict: {verdict}")

        results.append({
            "id": q["id"],
            "topic": q["topic"],
            "expected_gt": gt,
            "response": clean_text,
            "tokens": res["tokens_generated"],
            "elapsed": res["elapsed_seconds"],
            "verdict": verdict
        })

    out_file = os.path.join(ROOT, "sft_mcts_sympy_ood_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[EVALUATION COMPLETE] Results saved to {out_file}")


if __name__ == "__main__":
    run_evaluation()
