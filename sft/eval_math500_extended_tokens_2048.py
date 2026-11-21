"""
MATH-500 High-Capacity Benchmark (max_new_tokens = 2048): Deep Reasoning Comparison
===================================================================================
Evaluates:
  1. Saf Base Model (alpha = 0.00)
  2. Optimal LoRA (alpha = 0.25)
under the official deep reasoning token budget (max_new_tokens=2048) on MATH-500 (N=50).

Inference: Pure Greedy (T=0, max_new_tokens=2048, repetition_penalty=1.05)
Auto-saves after EVERY question to guarantee zero progress loss.
"""

import sys
import os
import json
import re
import time
import math
import torch
import sympy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_balanced_r4")
INPUT_JSON = os.path.join(ROOT, "math500_stratified50_results.json")
OUTPUT_JSON = os.path.join(ROOT, "math500_extended_2048_results.json")


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
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = s.replace("^", "**")
    try:
        return sympy.sympify(s)
    except Exception:
        return None


def check_math_match(pred: str, gt: str) -> bool:
    if clean_latex(pred) == clean_latex(gt):
        return True

    if r"\pm" in gt:
        base_parts = gt.split(r"\pm")
        if len(base_parts) == 2:
            a, b = base_parts[0].strip(), base_parts[1].strip()
            p1 = f"{a}+{b},{a}-{b}"
            p2 = f"{a}-{b},{a}+{b}"
            if clean_latex(pred) in [clean_latex(p1), clean_latex(p2)]:
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


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def set_lora_scale(model, scale: float):
    for module in model.modules():
        if hasattr(module, "scaling") and hasattr(module, "lora_alpha") and hasattr(module, "r"):
            for adapter_name in module.scaling:
                alpha = module.lora_alpha[adapter_name]
                r = module.r[adapter_name]
                module.scaling[adapter_name] = (alpha / r) * scale


def run_extended_eval():
    print("=" * 80)
    print("MATH-500 EXTENDED REASONING EVALUATION (max_new_tokens = 2048)")
    print("=" * 80)

    # 1. Load benchmark questions
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        benchmark_questions = json.load(f)
    print(f"Loaded {len(benchmark_questions)} stratified benchmark questions from {INPUT_JSON}.")

    subjects = [
        "Algebra", "Intermediate Algebra", "Prealgebra",
        "Number Theory", "Precalculus", "Geometry", "Counting & Probability"
    ]

    all_results = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"Loaded existing progress: {list(all_results.keys())}")
        except Exception:
            all_results = {}

    # 2. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Model in 4-bit NF4 onto GPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    arms = [
        ("0.00", 0.00, "Saf Base Model (No LoRA)"),
        ("0.25", 0.25, "Optimal LoRA (alpha = 0.25)")
    ]

    for arm_key, scale, label in arms:
        if arm_key in all_results and len(all_results[arm_key].get("details", [])) == len(benchmark_questions):
            print(f"\n[SKIP] {label} already completed ({all_results[arm_key]['pass_count']}/{len(benchmark_questions)}).")
            continue

        print("\n" + "=" * 80)
        print(f"STARTING {label.upper()} WITH 2048 TOKEN BUDGET")
        print("=" * 80)

        set_lora_scale(model, scale)
        pass_count = 0
        scale_details = []
        subj_stats = {s: {"pass": 0, "total": 0} for s in subjects}
        t_start = time.time()

        for i, item in enumerate(benchmark_questions):
            prob = item["problem"]
            gt = item["ground_truth"]
            subj = item["subject"]
            lvl = item.get("level", "?")
            uid = item.get("unique_id", f"math500_{i+1}")

            subj_stats[subj]["total"] += 1

            chat_prompt = (
                f"<|im_start|>user\n{prob}\n"
                f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n"
            )
            inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            elapsed = time.time() - t0

            gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            tokens_gen = len(out[0]) - inputs.input_ids.shape[1]

            pred = extract_boxed_answer(gen_text)
            is_pass = check_math_match(pred, gt)

            if is_pass:
                pass_count += 1
                subj_stats[subj]["pass"] += 1
                verdict = "PASS"
            else:
                verdict = "FAIL"

            print(f"[{i+1}/50] [{arm_key}] [{subj:22s} | Lvl {lvl}] {verdict} ({elapsed:.1f}s, {tokens_gen} toks) | GT: {gt} | Pred: {pred}")

            scale_details.append({
                "index": i + 1,
                "unique_id": uid,
                "subject": subj,
                "level": lvl,
                "ground_truth": gt,
                "predicted": pred,
                "verdict": verdict,
                "tokens": tokens_gen,
                "time": round(elapsed, 2)
            })

            total_time = time.time() - t_start
            acc = pass_count / len(benchmark_questions)
            low, high = wilson_ci(pass_count, len(benchmark_questions))

            all_results[arm_key] = {
                "label": label,
                "pass_count": pass_count,
                "total": len(benchmark_questions),
                "accuracy": acc,
                "ci_low": low,
                "ci_high": high,
                "subj_stats": subj_stats,
                "time_minutes": total_time / 60,
                "details": scale_details
            }

            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n[{label} COMPLETE] Score: {pass_count} / {len(benchmark_questions)} ({acc*100:.1f}%) in {total_time/60:.1f} min.")

    print("\n" + "=" * 80)
    print("FINAL 2048-TOKEN EXTENDED BENCHMARK SUMMARY (N=50):")
    print("=" * 80)
    for k, data in all_results.items():
        print(f"{data['label']:35s}: {data['pass_count']:2d} / {data['total']} ({data['accuracy']*100:5.1f}%) | 95% Wilson CI: [{data['ci_low']*100:5.1f}%, {data['ci_high']*100:5.1f}%]")

    print("\n--- SUBJECT-BY-SUBJECT BREAKDOWN (2048 TOKENS) ---")
    header = f"{'Subject':22s} | {'Base (0.00)':15s} | {'LoRA (0.25)':15s} | {'Delta (Δ)':10s}"
    print(header)
    print("-" * len(header))
    for s in subjects:
        b_p = all_results.get('0.00', {}).get('subj_stats', {}).get(s, {}).get('pass', 0)
        b_t = all_results.get('0.00', {}).get('subj_stats', {}).get(s, {}).get('total', 1)
        l_p = all_results.get('0.25', {}).get('subj_stats', {}).get(s, {}).get('pass', 0)
        l_t = all_results.get('0.25', {}).get('subj_stats', {}).get(s, {}).get('total', 1)
        b_pct = 100.0 * b_p / b_t
        l_pct = 100.0 * l_p / l_t
        diff = l_pct - b_pct
        print(f"{s:22s} | {b_p:2d}/{b_t:2d} ({b_pct:5.1f}%)   | {l_p:2d}/{l_t:2d} ({l_pct:5.1f}%)   | {diff:+5.1f}%")


if __name__ == "__main__":
    run_extended_eval()
