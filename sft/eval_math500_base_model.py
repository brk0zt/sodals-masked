"""
MATH-500 Stratified 50-Question Control Benchmark: FROZEN BASE MODEL (No LoRA)
=============================================================================
Evaluates the un-finetuned Frozen Qwen 3.5 9B Base Model on the exact same 50 questions
to isolate the exact delta of LoRA fine-tuning and test for catastrophic forgetting / retention.

Inference: Pure Greedy (T=0, max_new_tokens=1024, repetition_penalty=1.05)
Grading: SymPy algebraic normalization and extraction.
"""

import sys
import os
import json
import re
import time
import torch
import sympy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
INPUT_JSON = os.path.join(ROOT, "math500_stratified50_results.json")
OUTPUT_JSON = os.path.join(ROOT, "math500_base_model_control50_results.json")


def extract_boxed_answer(text: str) -> str:
    """Extracts answer inside \\boxed{...} or fallback to Answer: ..."""
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


def run_base_model_control():
    print("=" * 80)
    print("CONTROL EXPERIMENT: FROZEN BASE MODEL (No LoRA) ON MATH-500 (50 Questions)")
    print("=" * 80)

    # 1. Load exact 50 questions from previous run
    print("Loading exact 50 questions from:", INPUT_JSON)
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        previous_data = json.load(f)
    print(f"Loaded {len(previous_data)} benchmark questions.")

    # 2. Load Base Model and Tokenizer (NO LoRA PEFT)
    print("\nLoading Tokenizer from:", MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Pure Frozen Base Model in 4-bit NF4 onto GPU (NO LoRA ADAPTER)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    model.eval()

    results = []
    pass_count = 0
    start_all = time.time()
    by_subject = {}

    print("\n" + "=" * 80)
    print("STARTING BASE MODEL CONTROL INFERENCE (50 Questions)")
    print("=" * 80)

    for i, item in enumerate(previous_data):
        prob = item["problem"]
        gt_ans = item["ground_truth"]
        subj = item["subject"]
        level = item.get("level", "?")
        uid = item.get("unique_id", f"math500_{i+1}")

        if subj not in by_subject:
            by_subject[subj] = {"pass": 0, "total": 0}
        by_subject[subj]["total"] += 1

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

        pred_ans = extract_boxed_answer(gen_text)
        is_correct = check_math_match(pred_ans, gt_ans)

        if is_correct:
            pass_count += 1
            by_subject[subj]["pass"] += 1
            verdict = "PASS"
        else:
            verdict = "FAIL"

        print(f"[{i+1}/50] [{subj:22s} | Level {level}] {verdict} ({elapsed:.1f}s, {tokens_gen} toks) | GT: {gt_ans} | Pred: {pred_ans}")

        results.append({
            "index": i + 1,
            "unique_id": uid,
            "subject": subj,
            "level": level,
            "problem": prob,
            "ground_truth": gt_ans,
            "predicted_answer": pred_ans,
            "verdict": verdict,
            "tokens": tokens_gen,
            "elapsed_seconds": round(elapsed, 2),
            "response_tail": gen_text[-300:]
        })

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    total_time = time.time() - start_all
    accuracy = 100.0 * pass_count / len(previous_data)

    print("\n" + "=" * 80)
    print(f"BASE MODEL (NO LoRA) MATH-500 SCORE: {pass_count} / {len(previous_data)} ({accuracy:.1f}%)")
    print(f"Total Time: {total_time/60:.1f} minutes | Average Latency: {total_time/len(previous_data):.1f}s/question")
    print("=" * 80)

    print("\n--- BASE MODEL PERFORMANCE BY SUBJECT ---")
    for subj, stats in by_subject.items():
        p = stats["pass"]
        t = stats["total"]
        print(f"  {subj:25s}: {p} / {t} ({100.0 * p / t:.1f}%)")


if __name__ == "__main__":
    run_base_model_control()
