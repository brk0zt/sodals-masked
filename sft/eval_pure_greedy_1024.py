"""
Pristine 8Q Test: Pure LoRA Greedy with Full Token Budget (max_new_tokens=1024)
=============================================================================
Tests whether the 5/8 -> 7/8 performance gap was purely a token budget artifact (512 vs 1024)
rather than a symbolic verification effect.
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

PRISTINE_8Q_BENCHMARK = [
    {
        "id": "pristine_01_lie_f4_dim",
        "domain": "Lie Theory",
        "prompt": "Compute the exact dimension of the exceptional simple Lie algebra \\mathfrak{f}_4 over the complex numbers.",
        "expected_gt": "52"
    },
    {
        "id": "pristine_02_top_grassmannian_gr24",
        "domain": "Algebraic Topology",
        "prompt": "Compute the topological Euler characteristic \\chi(\\text{Gr}(2, 4)) of the complex Grassmannian manifold of 2-planes in \\mathbb{C}^4.",
        "expected_gt": "6"
    },
    {
        "id": "pristine_03_geom_canonical_g7",
        "domain": "Algebraic Geometry",
        "prompt": "Let C be a smooth compact Riemann surface of genus g = 7. Compute the exact degree of the canonical divisor K on C.",
        "expected_gt": "12"
    },
    {
        "id": "pristine_04_num_elliptic_f97",
        "domain": "Arithmetic Geometry",
        "prompt": "Let E be an elliptic curve defined over the finite field \\mathbb{F}_{97} with trace of Frobenius a_{97} = 6. Compute the exact number of rational points #E(\\mathbb{F}_{97}).",
        "expected_gt": "92"
    },
    {
        "id": "pristine_05_rep_su5_adjoint_dim",
        "domain": "Representation Theory",
        "prompt": "What is the dimension of the adjoint representation of the special unitary Lie algebra \\mathfrak{su}(5)?",
        "expected_gt": "24"
    },
    {
        "id": "pristine_06_symp_sp6_dim",
        "domain": "Symplectic Geometry",
        "prompt": "Compute the exact dimension of the real symplectic Lie algebra \\mathfrak{sp}(6, \\mathbb{R}) of 6x6 Hamiltonian matrices.",
        "expected_gt": "21"
    },
    {
        "id": "pristine_07_top_cp5_euler",
        "domain": "Complex Geometry",
        "prompt": "Compute the topological Euler characteristic \\chi(\\mathbb{C}P^5) of the 5-dimensional complex projective space.",
        "expected_gt": "6"
    },
    {
        "id": "pristine_08_num_ramanujan_tau2",
        "domain": "Modular Forms / Number Theory",
        "prompt": "What is the exact value of the Ramanujan tau function \\tau(2), defined as the Fourier coefficient of the modular discriminant \\Delta(z)?",
        "expected_gt": "-24"
    }
]


def test_pure_greedy_1024():
    print("="*80)
    print("PURE LoRA GREEDY CONTROL TEST (max_new_tokens=1024)")
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

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    results = []
    pass_count = 0

    for i, q in enumerate(PRISTINE_8Q_BENCHMARK):
        gt = q["expected_gt"]
        chat_prompt = f"<|im_start|>user\n{q['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        tokens_gen = len(out[0]) - inputs.input_ids.shape[1]

        is_exact = f"Answer: {gt}" in text or f"answer is: {gt}" in text or f"Answer: {gt}." in text or f"Answer: {gt}\n" in text
        verdict = "PASS (Exact)" if is_exact else ("PARTIAL" if gt in text else "FAIL")
        if is_exact:
            pass_count += 1

        print(f"Q{i+1} [{q['id']}] (GT: {gt}) -> Verdict: {verdict} | Tokens: {tokens_gen} | Time: {elapsed:.2f}s")
        print(f"Tail snippet:\n{text[-150:]}\n{'-'*60}")

        results.append({
            "id": q["id"],
            "gt": gt,
            "verdict": verdict,
            "tokens": tokens_gen,
            "time": elapsed,
            "tail": text[-200:]
        })

    print(f"\nFINAL PURE GREEDY (1024 TOKENS) SCORE: {pass_count} / {len(PRISTINE_8Q_BENCHMARK)} ({100*pass_count/len(PRISTINE_8Q_BENCHMARK):.1f}%)")
    
    with open(os.path.join(ROOT, "pure_greedy_1024_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    test_pure_greedy_1024()
