"""
SODALS v2.1 — Domain 2 (Soyut Cebir & Lie Teorisi) 20Q Evaluation Suite
======================================================================
Evaluates models / LoRA adapters on a canonical 20-question Abstract Algebra benchmark:
  - 10 Lie Algebra Dimension Problems (sp(6)=21 diagnostic, F4=52, G2=14, E6=78, E7=133, E8=248, sl(4)=15, so(8)=28, sp(4)=10, so(5)=10)
  - 4 Group Theory Problems (A5 order=60, Sylow-7 count=1, Aut(Q8) order=24, Z(D8) order=2)
  - 3 Ring & Field Theory Problems (units of Z[i]=4, F27 mult group order=26, R[x]/(x^2+1) dim=2)
  - 3 Galois Theory Problems (Q(sqrt2, sqrt3) deg=4, Q(zeta_7) deg=6, Gal(x^3-2) order=6)

Inference Settings:
  - Greedy decoding (Temperature = 0.0, Top-p = 1.0)
  - 4-bit NF4 quantization on RTX 4060
  - Exact match / regex numerical parsing for ground-truth verification
"""

import sys
import os
import json
import re
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
SFT_DIR = os.path.join(ROOT, "sft")

ADAPTER_PRESETS = {
    "frozen_base": "",
    "baseline": os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4"),
    "sdpo_a": os.path.join(SFT_DIR, "qwen_lora_adapter_sdpo_anchored"),
    "domain2": os.path.join(SFT_DIR, "qwen_lora_adapter_domain2"),
    "sdpo_d2": os.path.join(SFT_DIR, "qwen_lora_adapter_domain2"),
    "placebo": os.path.join(SFT_DIR, "qwen_lora_adapter_placebo_control"),
    "sdpo_pure": os.path.join(SFT_DIR, "qwen_lora_adapter_sdpo"),
    "domain2_pondering": os.path.join(SFT_DIR, "qwen_lora_adapter_domain2_pondering"),
    "domain2_r8_pondering": os.path.join(SFT_DIR, "qwen_lora_adapter_domain2_r8_pondering"),
    "domain2_r8_surgical_patch": os.path.join(SFT_DIR, "qwen_lora_adapter_domain2_r8_surgical_patch"),
    "interleaved_r8": os.path.join(SFT_DIR, "qwen_lora_adapter_interleaved_d1_d2_r8"),
}

DOMAIN2_20Q_BENCHMARK = [
    # --- Lie Algebras & Root Systems (10 Questions) ---
    {
        "id": "d2_01_lie_sp6_dim",
        "topic": "Dimension of sp(6, C)",
        "prompt": "What is the exact dimension of the complex symplectic Lie algebra \\mathfrak{sp}(6, \\mathbb{C}) (also denoted \\mathfrak{sp}(6))?",
        "expected_gt": "21",
        "formula": "C_3 family: n(2n+1) for n=3 -> 3 * 7 = 21"
    },
    {
        "id": "d2_02_lie_sp4_dim",
        "topic": "Dimension of sp(4, C)",
        "prompt": "What is the exact dimension of the complex symplectic Lie algebra \\mathfrak{sp}(4, \\mathbb{C})?",
        "expected_gt": "10",
        "formula": "C_2 family: n(2n+1) for n=2 -> 2 * 5 = 10"
    },
    {
        "id": "d2_03_lie_f4_dim",
        "topic": "Dimension of Exceptional Lie algebra F4",
        "prompt": "Compute the exact dimension of the exceptional simple Lie algebra F_4 over the complex numbers.",
        "expected_gt": "52",
        "formula": "Cartan: rank + |Phi| = 4 + 48 = 52"
    },
    {
        "id": "d2_04_lie_g2_dim",
        "topic": "Dimension of Exceptional Lie algebra G2",
        "prompt": "Compute the exact dimension of the exceptional Lie algebra G_2.",
        "expected_gt": "14",
        "formula": "Cartan: rank + |Phi| = 2 + 12 = 14"
    },
    {
        "id": "d2_05_lie_e6_dim",
        "topic": "Dimension of Exceptional Lie algebra E6",
        "prompt": "Compute the exact dimension of the exceptional simple Lie algebra E_6.",
        "expected_gt": "78",
        "formula": "Cartan: rank + |Phi| = 6 + 72 = 78"
    },
    {
        "id": "d2_06_lie_e7_dim",
        "topic": "Dimension of Exceptional Lie algebra E7",
        "prompt": "Compute the exact dimension of the exceptional simple Lie algebra E_7.",
        "expected_gt": "133",
        "formula": "Cartan: rank + |Phi| = 7 + 126 = 133"
    },
    {
        "id": "d2_07_lie_e8_dim",
        "topic": "Dimension of Exceptional Lie algebra E8",
        "prompt": "Compute the exact dimension of the exceptional simple Lie algebra E_8.",
        "expected_gt": "248",
        "formula": "Cartan: rank + |Phi| = 8 + 240 = 248"
    },
    {
        "id": "d2_08_lie_sl4_dim",
        "topic": "Dimension of sl(4, C)",
        "prompt": "What is the dimension of the special linear Lie algebra \\mathfrak{sl}(4, \\mathbb{C}) of traceless 4x4 matrices?",
        "expected_gt": "15",
        "formula": "A_3 family: 4^2 - 1 = 15"
    },
    {
        "id": "d2_09_lie_so8_dim",
        "topic": "Dimension of so(8, C)",
        "prompt": "What is the dimension of the orthogonal Lie algebra \\mathfrak{so}(8, \\mathbb{C}) of skew-symmetric 8x8 matrices?",
        "expected_gt": "28",
        "formula": "D_4 family: n(2n-1) for n=4 -> 4 * 7 = 28"
    },
    {
        "id": "d2_10_lie_so5_dim",
        "topic": "Dimension of so(5, C)",
        "prompt": "What is the dimension of the orthogonal Lie algebra \\mathfrak{so}(5, \\mathbb{C})?",
        "expected_gt": "10",
        "formula": "B_2 family: n(2n+1) for n=2 -> 2 * 5 = 10"
    },

    # --- Group Theory (4 Questions) ---
    {
        "id": "d2_11_grp_a5_order",
        "topic": "Order of Alternating Group A5",
        "prompt": "What is the order of the alternating group A_5 (the smallest non-abelian simple group)?",
        "expected_gt": "60",
        "formula": "5! / 2 = 60"
    },
    {
        "id": "d2_12_grp_sylow7_count",
        "topic": "Sylow 7-subgroups in Order 21 Group",
        "prompt": "How many Sylow 7-subgroups are there in any group of order 21?",
        "expected_gt": "1",
        "formula": "n_7 = 1 mod 7 and n_7 | 3 -> n_7 = 1"
    },
    {
        "id": "d2_13_grp_aut_q8_order",
        "topic": "Order of Aut(Q8)",
        "prompt": "What is the order of the automorphism group \\mathrm{Aut}(Q_8) of the quaternion group Q_8?",
        "expected_gt": "24",
        "formula": "Aut(Q_8) is isomorphic to S_4, order 24"
    },
    {
        "id": "d2_14_grp_center_d8_order",
        "topic": "Order of Center of D8",
        "prompt": "What is the order of the center Z(D_8) of the dihedral group of order 8 (symmetry group of the square)?",
        "expected_gt": "2",
        "formula": "Z(D_8) = {1, r^2} -> order 2"
    },

    # --- Ring & Field Theory (3 Questions) ---
    {
        "id": "d2_15_ring_units_zi_count",
        "topic": "Number of Units in Z[i]",
        "prompt": "How many units (invertible elements) are there in the ring of Gaussian integers \\mathbb{Z}[i]?",
        "expected_gt": "4",
        "formula": "Units are {1, -1, i, -i} -> 4"
    },
    {
        "id": "d2_16_ring_f27_mult_order",
        "topic": "Order of Multiplicative Group of F27",
        "prompt": "What is the order of the multiplicative group \\mathbb{F}_{27}^\\times of the finite field with 27 elements?",
        "expected_gt": "26",
        "formula": "q - 1 = 27 - 1 = 26"
    },
    {
        "id": "d2_17_ring_c_poly_dim",
        "topic": "Dimension of R[x]/(x^2+1) over R",
        "prompt": "What is the dimension of \\mathbb{R}[x] / (x^2 + 1) as a vector space over \\mathbb{R}?",
        "expected_gt": "2",
        "formula": "Degree of minimal polynomial x^2+1 is 2 -> dim = 2 (isomorphic to C)"
    },

    # --- Galois Theory (3 Questions) ---
    {
        "id": "d2_18_gal_biquadratic_deg",
        "topic": "Degree of Q(sqrt(2), sqrt(3)) over Q",
        "prompt": "What is the extension degree [\\mathbb{Q}(\\sqrt{2}, \\sqrt{3}) : \\mathbb{Q}]?",
        "expected_gt": "4",
        "formula": "Tower law: 2 * 2 = 4"
    },
    {
        "id": "d2_19_gal_cyclotomic_7_deg",
        "topic": "Degree of 7th Cyclotomic Field over Q",
        "prompt": "What is the degree [\\mathbb{Q}(\\zeta_7) : \\mathbb{Q}], where \\zeta_7 is a primitive 7th root of unity?",
        "expected_gt": "6",
        "formula": "Euler totient phi(7) = 6"
    },
    {
        "id": "d2_20_gal_x3_minus_2_order",
        "topic": "Order of Galois Group of x^3 - 2 over Q",
        "prompt": "What is the order of the Galois group of the polynomial f(x) = x^3 - 2 over \\mathbb{Q}?",
        "expected_gt": "6",
        "formula": "Splitting field Q(2^(1/3), omega) has degree 3 * 2 = 6 (isomorphic to S_3)"
    }
]


def extract_numerical_answer(text: str) -> str:
    """Extract final answer from CoT or direct response."""
    # Look for \boxed{...}
    boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
    if boxed:
        clean = boxed[-1].strip()
        num_m = re.search(r'-?\d+', clean)
        if num_m:
            return num_m.group(0)
        return clean

    # Look for Answer: ...
    ans_m = re.search(r'Answer:\s*([^\n]+)', text, re.IGNORECASE)
    if ans_m:
        clean = ans_m.group(1).strip()
        num_m = re.search(r'-?\d+', clean)
        if num_m:
            return num_m.group(0)
        return clean

    # Fallback to last number in text
    all_nums = re.findall(r'\b(-?\d+)\b', text)
    if all_nums:
        return all_nums[-1]
    return ""


def evaluate_domain2(adapter_key_or_path: str, max_tokens: int = 1024):
    adapter_path = ADAPTER_PRESETS.get(adapter_key_or_path, adapter_key_or_path)
    print("=" * 80)
    print("SODALS v2.1: DOMAIN 2 (SOYUT CEBİR) 20Q EVALUATION")
    print(f"Adapter: {adapter_key_or_path} -> {adapter_path}")
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

    print("Loading base model in 4-bit NF4...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    if os.path.exists(adapter_path):
        print(f"Attaching LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print(f"Warning: Adapter path {adapter_path} not found. Running frozen base!")
        model = base_model

    model.eval()

    results = []
    correct_count = 0
    t0 = time.time()

    print("\n" + "-" * 80)
    for idx, q in enumerate(DOMAIN2_20Q_BENCHMARK):
        sys_msg = (
            "You are an expert mathematician specializing in abstract algebra and Lie theory. "
            "Solve the problem step by step with exact mathematical rigor. "
            "Conclude your final answer clearly as 'Answer: <value>' or '\\boxed{<value>'."
        )
        prompt = (
            f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n{q['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        extracted = extract_numerical_answer(gen_text)
        expected = q["expected_gt"]

        is_correct = (extracted == expected)
        if is_correct:
            correct_count += 1
            verdict = "PASS ✓"
        else:
            verdict = "FAIL ✗"

        print(f"[{idx+1:02d}/20] {q['id']:<24} | Pred: {extracted:<8} | Expected: {expected:<5} | {verdict}", flush=True)

        results.append({
            "id": q["id"],
            "topic": q["topic"],
            "prompt": q["prompt"],
            "expected_gt": expected,
            "extracted_answer": extracted,
            "verdict": is_correct,
            "full_response": gen_text[:500]  # Store preview
        })

    elapsed = time.time() - t0
    accuracy = (correct_count / len(DOMAIN2_20Q_BENCHMARK)) * 100.0

    print("-" * 80)
    print(f"FINAL SCORE: {correct_count}/{len(DOMAIN2_20Q_BENCHMARK)} ({accuracy:.2f}%) in {elapsed:.1f}s")
    print("=" * 80)

    out_file = os.path.join(ROOT, f"domain2_20q_{adapter_key_or_path}_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "adapter": adapter_key_or_path,
            "adapter_path": adapter_path,
            "score": f"{correct_count}/{len(DOMAIN2_20Q_BENCHMARK)}",
            "accuracy": accuracy,
            "elapsed_seconds": elapsed,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_file}")
    return correct_count, len(DOMAIN2_20Q_BENCHMARK)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default="sdpo_a", choices=list(ADAPTER_PRESETS.keys()) + ["frozen_base"])
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    evaluate_domain2(args.adapter, args.max_tokens)
