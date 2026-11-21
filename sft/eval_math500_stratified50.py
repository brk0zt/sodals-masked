"""
MATH-500 Stratified 50-Question Benchmark
=========================================
Evaluates the SODALS Balanced r=4 LoRA adapter on a 50-question stratified sample
of the official HuggingFaceH4/MATH-500 benchmark across all 7 subjects and difficulty levels.

Inference: Pure Greedy (T=0, max_new_tokens=1024)
Grading: Standard MATH boxed-expression extraction and normalization.
"""

import sys
import os
import json
import re
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_balanced_r4")
OUTPUT_JSON = os.path.join(ROOT, "math500_stratified50_results.json")


def extract_boxed_answer(text: str) -> str:
    """Extracts answer inside \\boxed{...} or fallback to Answer: ..."""
    # Look for \boxed{...}
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        # Match matching braces
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

    # Fallback to Answer: ...
    m = re.search(r"(?:Answer|answer|the answer is)[:\s]+([^\n\.]+)", text)
    if m:
        return m.group(1).strip()

    # Fallback to last line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return lines[-1].strip()

    return ""


def clean_latex(s: str) -> str:
    """Normalizes basic LaTeX formatting for comparison."""
    s = s.strip()
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = s.replace(r"\$", "").replace("$", "").strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(" ", "")
    return s


def check_math_match(pred: str, gt: str) -> bool:
    """Checks if predicted answer matches ground truth."""
    pred_clean = clean_latex(pred)
    gt_clean = clean_latex(gt)

    if pred_clean == gt_clean:
        return True

    # Numeric equivalence fallback
    try:
        if float(pred_clean) == float(gt_clean):
            return True
    except Exception:
        pass

    return False


def run_math500_eval():
    print("=" * 80)
    print("SODALS v3.2: MATH-500 STRATIFIED 50-QUESTION BENCHMARK")
    print("=" * 80)

    # 1. Load MATH-500 dataset
    print("Loading HuggingFaceH4/MATH-500 dataset...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"Total dataset size: {len(ds)} questions.")

    # 2. Select 50 stratified questions across subjects
    # Subject breakdown: ~7 questions per subject (7 x 7 = 49 + 1 = 50)
    subjects = [
        "Algebra", "Intermediate Algebra", "Prealgebra",
        "Number Theory", "Precalculus", "Geometry", "Counting & Probability"
    ]
    
    selected_samples = []
    by_subject = {}
    for item in ds:
        subj = item["subject"]
        if subj not in by_subject:
            by_subject[subj] = []
        by_subject[subj].append(item)

    target_per_subj = {
        "Algebra": 8,
        "Intermediate Algebra": 8,
        "Prealgebra": 7,
        "Number Theory": 7,
        "Precalculus": 7,
        "Geometry": 7,
        "Counting & Probability": 6
    }

    for subj, count in target_per_subj.items():
        # Stratify by level (1 to 5) if possible
        items = sorted(by_subject.get(subj, []), key=lambda x: str(x.get("level", "")))
        step = max(1, len(items) // count)
        picked = items[::step][:count]
        selected_samples.extend(picked)

    print(f"Selected {len(selected_samples)} stratified benchmark questions.")
    for subj in subjects:
        c = sum(1 for x in selected_samples if x["subject"] == subj)
        print(f"  - {subj:25s}: {c} questions")

    # 3. Load Model and Tokenizer
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

    print("Loading Base Model in 4-bit NF4 onto GPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    print("Loading Balanced r=4 LoRA Adapter from:", ADAPTER_PATH)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # 4. Evaluation Loop
    results = []
    pass_count = 0
    start_all = time.time()

    print("\n" + "=" * 80)
    print("STARTING INFERENCE ON 50 STRATIFIED MATH-500 QUESTIONS")
    print("=" * 80)

    for i, item in enumerate(selected_samples):
        prob = item["problem"]
        gt_ans = item["answer"]
        subj = item["subject"]
        level = item.get("level", "?")
        uid = item.get("unique_id", f"math500_{i+1}")

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
            verdict = "PASS"
        else:
            verdict = "FAIL"

        print(f"[{i+1}/50] [{subj} | {level}] {verdict} ({elapsed:.1f}s, {tokens_gen} toks) | GT: {gt_ans} | Pred: {pred_ans}")

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

        # Save intermediate results safely
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    total_time = time.time() - start_all
    accuracy = 100.0 * pass_count / len(selected_samples)

    print("\n" + "=" * 80)
    print(f"MATH-500 STRATIFIED 50 FINAL SCORE: {pass_count} / {len(selected_samples)} ({accuracy:.1f}%)")
    print(f"Total Time: {total_time/60:.1f} minutes | Average Latency: {total_time/len(selected_samples):.1f}s/question")
    print("=" * 80)

    # Breakdown by Subject
    print("\n--- PERFORMANCE BY SUBJECT ---")
    for subj in subjects:
        subj_results = [r for r in results if r["subject"] == subj]
        if subj_results:
            c = sum(1 for r in subj_results if r["verdict"] == "PASS")
            print(f"  {subj:25s}: {c} / {len(subj_results)} ({100.0 * c / len(subj_results):.1f}%)")


if __name__ == "__main__":
    run_math500_eval()
