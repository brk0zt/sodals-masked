"""
SDPO Phase 5: Rigorous Multi-Benchmark Verification Suite
=========================================================
Tests the newly trained SDPO adapter (sft/qwen_lora_adapter_sdpo) against:
  1. SDPO Held-Out Test Pool (35 questions, ZERO training leakage) -> Generalization test
  2. Representative 50-Question MATH-500 -> Catastrophic Forgetting Tripwire
  3. Pristine 8Q STEM Benchmark -> Domain integrity test

Outputs: sdpo_verification_master_report.json
"""

import sys
import os
import json
import re
import time
import math
import sympy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
SDPO_ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_sdpo")
HELDOUT_FAILURES_PATH = os.path.join(ROOT, "sft", "sdpo_heldout_failures.json")
MATH500_50_PATH = os.path.join(ROOT, "math500_stratified50_results.json")
OUTPUT_JSON = os.path.join(ROOT, "sdpo_verification_master_report.json")


def extract_boxed_answer(text: str) -> str:
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        depth = 0
        start = idx + len(r"\boxed{")
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                if depth == 0:
                    return text[start:i].strip()
                depth -= 1
        return text[start:].strip()

    m = re.search(r"(?:Answer|answer|the answer is)[:\s]+([^\n\.]+)", text)
    if m:
        return m.group(1).strip()

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return lines[-1].strip()

    return ""


def clean_latex(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = s.replace(r"\$", "").replace("$", "").strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"^\circ", "").replace("^{\\circ}", "").replace("°", "")
    s = s.replace(" ", "")
    return s


def to_sympy(s: str):
    s = clean_latex(s)
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = s.replace("^", "**")
    try:
        return sympy.sympify(s)
    except Exception:
        return None


def check_math_match(pred: str, gt: str) -> bool:
    cp = clean_latex(pred)
    cg = clean_latex(gt)
    if cp == cg:
        return True

    cp_bare = re.sub(r"^[a-zA-Z]\s*=\s*", "", cp)
    cg_bare = re.sub(r"^[a-zA-Z]\s*=\s*", "", cg)
    if cp_bare == cg_bare:
        return True

    if r"\pm" in gt:
        base_parts = gt.split(r"\pm")
        if len(base_parts) == 2:
            a, b = base_parts[0].strip(), base_parts[1].strip()
            p1 = f"{a}+{b},{a}-{b}"
            p2 = f"{a}-{b},{a}+{b}"
            if cp in [clean_latex(p1), clean_latex(p2)]:
                return True

    sym_gt = to_sympy(gt)
    sym_pred = to_sympy(pred)
    if sym_gt is not None and sym_pred is not None:
        try:
            if sympy.simplify(sym_gt - sym_pred) == 0:
                return True
        except Exception:
            pass
        try:
            if abs(float(sym_gt.evalf()) - float(sym_pred.evalf())) < 1e-6:
                return True
        except Exception:
            pass

    return False


def set_lora_scale(model, scale: float):
    for module in model.modules():
        if hasattr(module, "scaling") and hasattr(module, "lora_alpha") and hasattr(module, "r"):
            for adapter_name in module.scaling:
                alpha = module.lora_alpha[adapter_name]
                r = module.r[adapter_name]
                module.scaling[adapter_name] = (alpha / r) * scale


def evaluate_set(model, tokenizer, items, prompt_fn, max_tokens=2048, label="Benchmark"):
    print("\n" + "=" * 80)
    print(f"RUNNING EVALUATION: {label} (N={len(items)})")
    print("=" * 80)

    pass_count = 0
    results = []

    for i, item in enumerate(items):
        prob = item["problem"]
        gt = item["ground_truth"]
        subj = item.get("subject", "General")
        lvl = item.get("level", "?")

        prompt = prompt_fn(prob)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0

        gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred = extract_boxed_answer(gen_text)
        is_pass = check_math_match(pred, gt)

        if is_pass:
            pass_count += 1
            v = "PASS"
        else:
            v = "FAIL"

        print(f"[{i+1:02d}/{len(items)}] [{subj:22s} | Lvl {lvl}] {v} ({elapsed:.1f}s) | GT: {gt} | Pred: {pred}")

        results.append({
            "index": i + 1,
            "problem": prob,
            "ground_truth": gt,
            "predicted": pred,
            "verdict": v,
            "elapsed_seconds": round(elapsed, 2)
        })

    acc = 100.0 * pass_count / len(items)
    print(f"\n[{label} RESULT] {pass_count} / {len(items)} ({acc:.1f}%)")
    return pass_count, len(items), acc, results


def run_full_sdpo_verification():
    print("=" * 80)
    print("SDPO MASTER VERIFICATION PROTOCOL")
    print("=" * 80)

    # 1. Load Datasets
    with open(HELDOUT_FAILURES_PATH, "r", encoding="utf-8") as f:
        heldout_cases = json.load(f)["cases"]
    print(f"Loaded {len(heldout_cases)} strictly held-out failure questions.")

    with open(MATH500_50_PATH, "r", encoding="utf-8") as f:
        math500_50_cases = json.load(f)
    print(f"Loaded {len(math500_50_cases)} representative MATH-500 control questions.")

    # Pristine 8Q STEM
    pristine_8q = [
        {"subject": "Lie Theory", "problem": "What is the dimension of the exceptional Lie algebra F4?", "ground_truth": "52"},
        {"subject": "Modular Forms", "problem": "Compute the Ramanujan tau function tau(2).", "ground_truth": "-24"},
        {"subject": "Symplectic Geom", "problem": "What is the dimension of the symplectic Lie algebra sp(6)?", "ground_truth": "21"},
        {"subject": "Arithmetic Geom", "problem": "What is the rank of the elliptic curve y^2 = x^3 - x over Q?", "ground_truth": "0"},
        {"subject": "Algebraic Topology", "problem": "Compute the homotopy group pi_3(S^2).", "ground_truth": "\\mathbb{Z}"},
        {"subject": "Representation", "problem": "What is the dimension of the standard irreducible representation of SU(3)?", "ground_truth": "3"},
        {"subject": "Complex Analysis", "problem": "What is the residue of f(z) = 1/(z^2 + 1) at z = i?", "ground_truth": "-\\frac{i}{2}"},
        {"subject": "Characteristic Class", "problem": "What is the Euler characteristic of the 2-sphere S^2?", "ground_truth": "2"}
    ]

    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Base Model in 4-bit NF4...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    print(f"Loading SDPO Adapter from {SDPO_ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, SDPO_ADAPTER_PATH)
    set_lora_scale(model, 0.25)
    model.eval()

    def default_math_prompt(p):
        return f"<|im_start|>user\n{p}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n<think>\n"

    # Test 1: Held-out failures (Zero-Shot Generalization)
    h_p, h_t, h_acc, h_dets = evaluate_set(
        model, tokenizer, heldout_cases, default_math_prompt, max_tokens=2048, label="SDPO 35 Held-Out Failures (Generalization Test)"
    )

    # Test 2: Representative 50-Question MATH-500 (Forgetting Tripwire)
    m_p, m_t, m_acc, m_dets = evaluate_set(
        model, tokenizer, math500_50_cases, default_math_prompt, max_tokens=2048, label="MATH-500 50-Question Control (Forgetting Tripwire)"
    )

    # Test 3: Pristine 8Q STEM
    p_p, p_t, p_acc, p_dets = evaluate_set(
        model, tokenizer, pristine_8q, default_math_prompt, max_tokens=1024, label="Pristine 8Q STEM Benchmark"
    )

    summary_report = {
        "heldout_test_pool": {
            "label": "35 Strictly Held-Out Failures (Pure Generalization)",
            "pass_count": h_p,
            "total": h_t,
            "accuracy": h_acc,
            "details": h_dets
        },
        "math500_50_control": {
            "label": "50-Question MATH-500 Control (Catastrophic Forgetting Tripwire)",
            "pass_count": m_p,
            "total": m_t,
            "accuracy": m_acc,
            "details": m_dets
        },
        "pristine_8q": {
            "label": "Pristine 8Q STEM Benchmark",
            "pass_count": p_p,
            "total": p_t,
            "accuracy": p_acc,
            "details": p_dets
        }
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("SDPO MASTER VERIFICATION REPORT COMPLETED")
    print("=" * 80)
    print(f"1. 35 Held-Out Failures (Generalization) : {h_p} / {h_t} ({h_acc:.1f}%) [Was 0% before!]")
    print(f"2. MATH-500 50Q Control (Forgetting)     : {m_p} / {m_t} ({m_acc:.1f}%) [Baseline: 76.0-78.0%]")
    print(f"3. Pristine 8Q STEM Benchmark            : {p_p} / {p_t} ({p_acc:.1f}%) [Baseline: 7/8]")
    print(f"Full report saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    run_full_sdpo_verification()
