"""
Fresh Held-Out 50-Question MATH-500 Benchmark (Disjoint Test Set)
================================================================
Evaluates the Frozen Base Model (alpha=0.00) and Scaled LoRA (alpha=0.25, 0.50)
on a completely fresh, held-out, disjoint 50-question stratified sample of MATH-500.

Auto-saves after EVERY question to ensure zero progress loss.

Inference: Pure Greedy (T=0, max_new_tokens=1024, repetition_penalty=1.05)
Grading: SymPy algebraic normalization.
"""

import sys
import os
import json
import re
import time
import math
import torch
import sympy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_balanced_r4")
PREV_JSON = os.path.join(ROOT, "math500_stratified50_results.json")
OUTPUT_JSON = os.path.join(ROOT, "math500_fresh50_heldout_results.json")


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


def run_fresh_heldout_eval():
    print("=" * 80)
    print("FRESH HELD-OUT 50-QUESTION MATH-500 BENCHMARK (DISJOINT SET)")
    print("=" * 80)

    # 1. Load full MATH-500 dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # 2. Exclude previously used 50 questions
    with open(PREV_JSON, "r", encoding="utf-8") as f:
        prev_data = json.load(f)
    used_ids = set(x["unique_id"] for x in prev_data)
    print(f"Loaded {len(used_ids)} previously evaluated question IDs to exclude.")

    fresh_pool = [x for x in ds if x["unique_id"] not in used_ids]
    print(f"Remaining fresh question pool: {len(fresh_pool)} questions.")

    # 3. Stratify 50 fresh questions across the 7 subjects
    subjects = [
        "Algebra", "Intermediate Algebra", "Prealgebra",
        "Number Theory", "Precalculus", "Geometry", "Counting & Probability"
    ]
    target_per_subj = {
        "Algebra": 8,
        "Intermediate Algebra": 8,
        "Prealgebra": 7,
        "Number Theory": 7,
        "Precalculus": 7,
        "Geometry": 7,
        "Counting & Probability": 6
    }

    by_subj = {s: [] for s in subjects}
    for item in fresh_pool:
        s = item["subject"]
        if s in by_subj:
            by_subj[s].append(item)

    selected_fresh = []
    for s, count in target_per_subj.items():
        items = sorted(by_subj[s], key=lambda x: str(x.get("level", "")))
        step = max(1, len(items) // count)
        picked = items[::step][:count]
        selected_fresh.extend(picked)

    print(f"Selected {len(selected_fresh)} fresh, disjoint benchmark questions.")

    # Load existing state if any
    all_scales_results = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                all_scales_results = json.load(f)
            print(f"Found existing progress in {OUTPUT_JSON}: {list(all_scales_results.keys())}")
        except Exception:
            all_scales_results = {}

    # 4. Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Model onto GPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    for scale in [0.00, 0.25, 0.50]:
        scale_key = str(scale)
        if scale_key in all_scales_results and len(all_scales_results[scale_key].get("details", [])) == len(selected_fresh):
            print(f"\n[SKIP] Scale {scale:.2f} already completed ({all_scales_results[scale_key]['pass_count']}/{len(selected_fresh)}).")
            continue

        print("\n" + "=" * 80)
        print(f"EVALUATING FRESH HELD-OUT SET AT SCALE = {scale:.2f}")
        print("=" * 80)

        set_lora_scale(model, scale)
        pass_count = 0
        scale_details = []
        subj_stats = {s: {"pass": 0, "total": 0} for s in subjects}
        t_start = time.time()

        for i, item in enumerate(selected_fresh):
            prob = item["problem"]
            gt = item["answer"]
            subj = item["subject"]
            lvl = item.get("level", "?")
            uid = item["unique_id"]

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
                    max_new_tokens=1024,
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

            print(f"[{i+1}/50] [Scale {scale:.2f}] [{subj:22s} | Lvl {lvl}] {verdict} ({elapsed:.1f}s, {tokens_gen} toks) | GT: {gt} | Pred: {pred}")

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
            acc = pass_count / len(selected_fresh)
            low, high = wilson_ci(pass_count, len(selected_fresh))

            all_scales_results[scale_key] = {
                "pass_count": pass_count,
                "total": len(selected_fresh),
                "accuracy": acc,
                "ci_low": low,
                "ci_high": high,
                "subj_stats": subj_stats,
                "time_minutes": total_time / 60,
                "details": scale_details
            }

            # Auto-save after EVERY question
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(all_scales_results, f, indent=2, ensure_ascii=False)

        print(f"\n[SCALE {scale:.2f} FRESH SET COMPLETE] Score: {pass_count} / 50 ({acc*100:.1f}%) | 95% Wilson CI: [{low*100:.1f}%, {high*100:.1f}%]")

    print("\n" + "=" * 80)
    print("FINAL UNBIASED HELD-OUT BENCHMARK MATRIX (50 FRESH DISJOINT QUESTIONS):")
    print("=" * 80)
    for s, data in all_scales_results.items():
        print(f"Scale {s:4s} : {data['pass_count']:2d} / {data['total']} ({data['accuracy']*100:5.1f}%) | 95% Wilson CI: [{data['ci_low']*100:5.1f}%, {data['ci_high']*100:5.1f}%]")

    print("\n--- SUBJECT-BY-SUBJECT COMPARISON ON FRESH SET ---")
    header = f"{'Subject':22s} | {'Base (0.00)':12s} | {'Scale 0.25':12s} | {'Scale 0.50':12s}"
    print(header)
    print("-" * len(header))
    for s in subjects:
        b00 = f"{all_scales_results['0.0']['subj_stats'][s]['pass']}/{all_scales_results['0.0']['subj_stats'][s]['total']}"
        s25 = f"{all_scales_results['0.25']['subj_stats'][s]['pass']}/{all_scales_results['0.25']['subj_stats'][s]['total']}"
        s50 = f"{all_scales_results['0.5']['subj_stats'][s]['pass']}/{all_scales_results['0.5']['subj_stats'][s]['total']}"
        print(f"{s:22s} | {b00:12s} | {s25:12s} | {s50:12s}")


if __name__ == "__main__":
    run_fresh_heldout_eval()
