"""
SODALS v2.1: Canonical 100-Question STEM MMLU-Pro Evaluation Suite
==================================================================
Evaluates either SDPO-A or Balanced Baseline on the locked 100-question
STEM MMLU-Pro dataset (mmlu_pro_100_stem_cases.json).

Usage:
  python sft/eval_mmlu_pro_100.py --model sdpo_a
  python sft/eval_mmlu_pro_100.py --model baseline

Features:
  - Resumes automatically from disk if interrupted
  - Saves progress after EVERY question
  - Logs category-level breakdown (math, physics, chemistry, CS, engineering)
  - Uses deterministic greedy decoding (T=0, max_new_tokens=1024)
"""

import sys
import os
import json
import re
import time
import argparse
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
TEST_CASES_PATH = os.path.join(SFT_DIR, "mmlu_pro_100_stem_cases.json")

ADAPTERS = {
    "sdpo_a": {
        "path": os.path.join(SFT_DIR, "qwen_lora_adapter_sdpo_anchored"),
        "label": "Anchored SDPO (SDPO-A)",
        "output_file": os.path.join(ROOT, "mmlu_pro_100_sdpo_a_results.json")
    },
    "baseline": {
        "path": os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4"),
        "label": "Balanced Baseline (r=4)",
        "output_file": os.path.join(ROOT, "mmlu_pro_100_baseline_results.json")
    }
}

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def extract_mc_letter(text: str) -> str:
    """Extracts multiple-choice letter (A-J)."""
    # 1. Look for boxed letter first
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        depth = 0
        start = idx + len(r"\boxed{")
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                if depth == 0:
                    cand = text[start:i].strip()
                    m = re.search(r"([A-Ja-j])", cand)
                    if m:
                        return m.group(1).upper()
                    break
                depth -= 1

    # 2. Look for explicit "Answer: X" or "The correct answer is (X)"
    m = re.search(r"(?:Answer|answer|the answer is)[:\s]+(?:\()?([A-Ja-j])(?:\))?", text)
    if m:
        return m.group(1).upper()

    # 3. Look at the last lines for a standalone letter
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in reversed(lines[-5:]):
        m = re.search(r"\b([A-Ja-j])\b", l)
        if m:
            return m.group(1).upper()

    return ""


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


def format_mmlu_pro_prompt(item: dict) -> str:
    q = item["question"]
    opts = item["options"]
    opt_lines = []
    for i, opt in enumerate(opts):
        letter = LETTERS[i] if i < len(LETTERS) else f"({i})"
        opt_lines.append(f"({letter}) {opt}")
    options_text = "\n".join(opt_lines)

    return (
        f"<|im_start|>user\n"
        f"Question: {q}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Please reason step by step, and put your final choice letter within \\boxed{{}}.<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )


def run_evaluation(model_key: str, max_questions: int = 20):
    config = ADAPTERS[model_key]
    adapter_path = config["path"]
    model_label = config["label"]
    output_path = config["output_file"]

    print("=" * 80, flush=True)
    print(f"MMLU-PRO {max_questions}-QUESTION STEM EVALUATION: {model_label}", flush=True)
    print(f"Adapter Path: {adapter_path}", flush=True)
    print(f"Output Path:  {output_path}", flush=True)
    print("=" * 80, flush=True)

    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)[:max_questions]
    print(f"Loaded {len(cases)} locked MMLU-Pro STEM test cases (capped at {max_questions}).", flush=True)

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
                evaluated_ids.add(r["question_id"])
                c = r.get("category", "other")
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
        print("All 100 questions already evaluated! Exiting.", flush=True)
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

    print("\nStarting MMLU-Pro Inference Loop (max_new_tokens=1024, greedy T=0)...", flush=True)
    t_start = time.time()

    for idx, item in enumerate(cases, 1):
        qid = item["question_id"]
        if qid in evaluated_ids:
            continue

        gt = item["ground_truth"].strip().upper()
        cat = item.get("category", "other")

        if cat not in cat_stats:
            cat_stats[cat] = {"pass": 0, "total": 0}
        cat_stats[cat]["total"] += 1

        prompt = format_mmlu_pro_prompt(item)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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

        gen_tokens = out[0][inputs.input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        tokens_gen = len(gen_tokens)

        pred = extract_mc_letter(gen_text)
        is_pass = (pred == gt)

        if is_pass:
            pass_count += 1
            cat_stats[cat]["pass"] += 1
            verdict = "PASS"
        else:
            verdict = "FAIL"

        current_total = len(results_list) + 1
        current_acc = 100.0 * pass_count / current_total

        print(f"[{current_total:03d}/100] [{cat:18s}] {verdict} ({elapsed:.1f}s, {tokens_gen} toks) | Live: {pass_count}/{current_total} ({current_acc:.1f}%) | GT: {gt} | Pred: {pred}", flush=True)

        results_list.append({
            "index": current_total,
            "question_id": qid,
            "category": cat,
            "ground_truth": gt,
            "predicted": pred,
            "verdict": verdict,
            "tokens": tokens_gen,
            "elapsed_seconds": round(elapsed, 2),
            "response_preview": gen_text[-250:] if len(gen_text) > 250 else gen_text
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
    print(f"MMLU-PRO EVALUATION COMPLETED: {model_label}", flush=True)
    print(f"Score: {pass_count} / {len(cases)} ({100.0 * pass_count / len(cases):.2f}%)", flush=True)
    low, high = wilson_ci(pass_count, len(cases))
    print(f"95% Wilson CI: [{low*100:.2f}%, {high*100:.2f}%]", flush=True)
    print("Category Breakdown:", flush=True)
    for c, stats in sorted(cat_stats.items()):
        c_acc = (100.0 * stats["pass"] / stats["total"]) if stats["total"] > 0 else 0
        print(f"  {c:20s}: {stats['pass']}/{stats['total']} ({c_acc:.1f}%)", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sdpo_a", "baseline"], default="sdpo_a",
                        help="Which model adapter to evaluate (sdpo_a or baseline)")
    parser.add_argument("--max_questions", type=int, default=20,
                        help="Maximum questions to evaluate (default: 20)")
    args = parser.parse_args()
    run_evaluation(args.model, args.max_questions)
