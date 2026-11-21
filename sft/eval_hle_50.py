"""
SODALS v2.1: Canonical 50-Question Stratified HLE Evaluation Suite
==================================================================
Evaluates either SDPO-A or Balanced Baseline on the locked 50-question
stratified text-only HLE dataset (hle_50_stratified_test_cases.json).

Usage:
  python sft/eval_hle_50.py --model sdpo_a
  python sft/eval_hle_50.py --model baseline

Features:
  - Resumes automatically from disk if interrupted
  - Saves progress after EVERY question
  - Logs category-level breakdown
  - Uses deterministic greedy decoding (T=0, max_new_tokens=2048)
"""

import sys
import os
import json
import re
import time
import argparse
import math
import sympy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
TEST_CASES_PATH = os.path.join(SFT_DIR, "hle_50_stratified_test_cases.json")

ADAPTERS = {
    "sdpo_a": {
        "path": os.path.join(SFT_DIR, "qwen_lora_adapter_sdpo_anchored"),
        "label": "Anchored SDPO (SDPO-A)",
        "output_file": os.path.join(ROOT, "hle_50_sdpo_a_results.json")
    },
    "baseline": {
        "path": os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4"),
        "label": "Balanced Baseline (r=4)",
        "output_file": os.path.join(ROOT, "hle_50_baseline_results.json")
    }
}


def extract_boxed_answer(text: str) -> str:
    """Extracts boxed answer, or falls back to last line / Answer: prefix."""
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
    s = re.sub(r"\\mathbb\{([^}]+)\}", r"\1", s)
    s = s.replace(r"\$", "").replace("$", "").strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"^\circ", "").replace("^{\\circ}", "").replace("°", "")
    s = s.replace(" ", "")
    s = re.sub(r"\\mathbb\{Z\}_2|Z_2|Z/2Z|\\mathbb\{Z\}/2\\mathbb\{Z\}", "Z_2", s)
    s = re.sub(r"\\mathbb\{Z\}|Z", "Z", s)
    return s


def to_sympy(s: str):
    s = clean_latex(s)
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = s.replace(r"\pi", "pi")
    s = s.replace("^", "**")
    try:
        return sympy.sympify(s)
    except Exception:
        return None


def check_hle_match(pred: str, gt: str, answer_type: str = "") -> bool:
    pred_clean = clean_latex(pred)
    gt_clean = clean_latex(gt)

    if pred_clean.lower() == gt_clean.lower():
        return True

    # Strip prefixes like "x = "
    pred_bare = re.sub(r"^[a-zA-Z]\s*=\s*", "", pred_clean)
    gt_bare = re.sub(r"^[a-zA-Z]\s*=\s*", "", gt_clean)
    if pred_bare.lower() == gt_bare.lower():
        return True

    # Check for single-letter multiple choice
    if len(gt_clean) == 1 and gt_clean.isalpha():
        # Check if pred has format "(A)" or "A"
        letter_match = re.search(r"\b([A-Ja-j])\b", pred)
        if letter_match and letter_match.group(1).upper() == gt_clean.upper():
            return True

    # SymPy numerical / algebraic check
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


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def set_lora_scale(model, scale: float):
    for module in model.modules():
        if hasattr(module, "scaling") and hasattr(module, "lora_alpha") and hasattr(module, "r"):
            for adapter_name in module.scaling:
                alpha = module.lora_alpha[adapter_name]
                r = module.r[adapter_name]
                module.scaling[adapter_name] = (alpha / r) * scale


def run_evaluation(model_key: str):
    config = ADAPTERS[model_key]
    adapter_path = config["path"]
    model_label = config["label"]
    output_path = config["output_file"]

    print("=" * 80, flush=True)
    print(f"HLE 50-QUESTION EVALUATION: {model_label}", flush=True)
    print(f"Adapter Path: {adapter_path}", flush=True)
    print(f"Output Path:  {output_path}", flush=True)
    print("=" * 80, flush=True)

    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} locked HLE test cases.", flush=True)

    # Check for resume
    evaluated_ids = set()
    results_list = []
    pass_count = 0
    cat_stats = {}

    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            results_list = saved_data.get("details", [])
            for r in results_list:
                evaluated_ids.add(r["id"])
                c = r.get("category", "Other")
                if c not in cat_stats:
                    cat_stats[c] = {"pass": 0, "total": 0}
                cat_stats[c]["total"] += 1
                if r["verdict"] == "PASS":
                    cat_stats[c]["pass"] += 1
                    pass_count += 1
            print(f"Resuming: {len(evaluated_ids)}/{len(cases)} already completed ({pass_count} passed).", flush=True)
        except Exception as e:
            print(f"Error loading existing progress: {e}", flush=True)
            evaluated_ids = set()
            results_list = []

    if len(evaluated_ids) >= len(cases):
        print("All 50 questions already evaluated! Exiting.", flush=True)
        return

    # Load Model
    print("Loading Tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Base Model in 4-bit NF4 onto GPU...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    print(f"Loading LoRA Adapter from {adapter_path}...", flush=True)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    set_lora_scale(model, 0.25)
    model.eval()

    print("\nStarting HLE Inference Loop (max_new_tokens=2048, greedy T=0)...", flush=True)
    t_start = time.time()

    for idx, item in enumerate(cases, 1):
        qid = item["id"]
        if qid in evaluated_ids:
            continue

        q_text = item["question"]
        gt = item["ground_truth"]
        atype = item.get("answer_type", "")
        cat = item.get("category", "Other")
        raw_subj = item.get("raw_subject", "")

        if cat not in cat_stats:
            cat_stats[cat] = {"pass": 0, "total": 0}
        cat_stats[cat]["total"] += 1

        chat_prompt = (
            f"<|im_start|>user\n{q_text}\n"
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

        gen_tokens = out[0][inputs.input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        tokens_gen = len(gen_tokens)

        pred = extract_boxed_answer(gen_text)
        is_pass = check_hle_match(pred, gt, atype)

        if is_pass:
            pass_count += 1
            cat_stats[cat]["pass"] += 1
            verdict = "PASS"
        else:
            verdict = "FAIL"

        current_total = len(results_list) + 1
        current_acc = 100.0 * pass_count / current_total

        print(f"[{current_total:02d}/50] [{cat:22s}] {verdict} ({elapsed:.1f}s, {tokens_gen} toks) | Live: {pass_count}/{current_total} ({current_acc:.1f}%) | GT: {gt[:30]} | Pred: {pred[:30]}", flush=True)

        results_list.append({
            "index": current_total,
            "id": qid,
            "category": cat,
            "raw_subject": raw_subj,
            "question": q_text,
            "answer_type": atype,
            "ground_truth": gt,
            "predicted": pred,
            "verdict": verdict,
            "tokens": tokens_gen,
            "elapsed_seconds": round(elapsed, 2),
            "response_preview": gen_text[-300:] if len(gen_text) > 300 else gen_text
        })
        evaluated_ids.add(qid)

        # Save to JSON after EVERY question
        low, high = wilson_ci(pass_count, current_total)
        payload = {
            "model_key": model_key,
            "model_label": model_label,
            "adapter_path": adapter_path,
            "total_evaluated": current_total,
            "total_target": len(cases),
            "pass_count": pass_count,
            "accuracy": round(current_acc, 2),
            "ci_95_low": round(low * 100, 2),
            "ci_95_high": round(high * 100, 2),
            "category_breakdown": cat_stats,
            "elapsed_minutes": round((time.time() - t_start) / 60, 2),
            "details": results_list
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80, flush=True)
    print(f"EVALUATION COMPLETED: {model_label}", flush=True)
    print(f"Score: {pass_count} / {len(cases)} ({100.0 * pass_count / len(cases):.2f}%)", flush=True)
    low, high = wilson_ci(pass_count, len(cases))
    print(f"95% Wilson CI: [{low*100:.2f}%, {high*100:.2f}%]", flush=True)
    print("Category Breakdown:", flush=True)
    for c, stats in sorted(cat_stats.items()):
        c_acc = (100.0 * stats["pass"] / stats["total"]) if stats["total"] > 0 else 0
        print(f"  {c:25s}: {stats['pass']}/{stats['total']} ({c_acc:.1f}%)", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sdpo_a", "baseline"], default="sdpo_a",
                        help="Which model adapter to evaluate (sdpo_a or baseline)")
    args = parser.parse_args()
    run_evaluation(args.model)
