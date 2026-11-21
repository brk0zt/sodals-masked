"""
Full 500-Question Official MATH-500 Benchmark: Anchored SDPO (SDPO-A)
=====================================================================
Runs the complete 500 questions of HuggingFaceH4/MATH-500 on the SDPO-A adapter
with a 4096-token reasoning ceiling (identical to the Balanced Baseline benchmark).

Adapter: sft/qwen_lora_adapter_sdpo_anchored (alpha=0.25)
Output:  math500_full500_sdpo_a_results.json

Inference: Pure Greedy (T=0, max_new_tokens=4096, repetition_penalty=1.05)
Auto-saves after EVERY question to guarantee zero progress loss and seamless resume capability.
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
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_sdpo_anchored")
OUTPUT_JSON = os.path.join(ROOT, "math500_full500_sdpo_a_results.json")


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
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def set_lora_scale(model, scale: float):
    for module in model.modules():
        if hasattr(module, "scaling") and hasattr(module, "lora_alpha") and hasattr(module, "r"):
            for adapter_name in module.scaling:
                alpha = module.lora_alpha[adapter_name]
                r = module.r[adapter_name]
                module.scaling[adapter_name] = (alpha / r) * scale


def run_full500_eval():
    print("=" * 80, flush=True)
    print("FULL 500-QUESTION MATH-500 BENCHMARK: ANCHORED SDPO (SDPO-A)", flush=True)
    print(f"Adapter Path: {ADAPTER_PATH}", flush=True)
    print(f"Output File:  {OUTPUT_JSON}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load entire MATH-500 dataset
    print("Loading official HuggingFaceH4/MATH-500 dataset (all 500 questions)...", flush=True)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"Loaded {len(ds)} questions.", flush=True)

    subjects = [
        "Algebra", "Intermediate Algebra", "Prealgebra",
        "Number Theory", "Precalculus", "Geometry", "Counting & Probability"
    ]

    # Resume capability
    evaluated_ids = set()
    results_list = []
    pass_count = 0
    subj_stats = {s: {"pass": 0, "total": 0} for s in subjects}

    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            results_list = saved_data.get("details", [])
            for r in results_list:
                evaluated_ids.add(r["unique_id"])
                s = r["subject"]
                if s in subj_stats:
                    subj_stats[s]["total"] += 1
                    if r["verdict"] == "PASS":
                        subj_stats[s]["pass"] += 1
                if r["verdict"] == "PASS":
                    pass_count += 1
            print(f"Resuming from existing run: {len(evaluated_ids)} / {len(ds)} questions already evaluated ({pass_count} passed).", flush=True)
        except Exception as e:
            print(f"Could not load previous state: {e}", flush=True)
            evaluated_ids = set()
            results_list = []

    if len(evaluated_ids) >= len(ds):
        print("All 500 questions already completed! Exiting.", flush=True)
        return

    # 2. Load Model & Tokenizer
    print("\nLoading Tokenizer...", flush=True)
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

    print("Loading Anchored SDPO (SDPO-A) Adapter and setting scale = 0.25...", flush=True)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    set_lora_scale(model, 0.25)
    model.eval()

    print("\n" + "=" * 80, flush=True)
    print("STARTING FULL 500-QUESTION SDPO-A EVALUATION LOOP", flush=True)
    print("=" * 80, flush=True)

    t_start = time.time()

    for i, item in enumerate(ds):
        uid = item.get("unique_id", f"math500_{i+1}")
        if uid in evaluated_ids:
            continue

        prob = item["problem"]
        gt = item["answer"]
        subj = item["subject"]
        lvl = item.get("level", "?")

        if subj not in subj_stats:
            subj_stats[subj] = {"pass": 0, "total": 0}
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
                max_new_tokens=4096,
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
        is_pass = check_math_match(pred, gt)

        if is_pass:
            pass_count += 1
            subj_stats[subj]["pass"] += 1
            verdict = "PASS"
        else:
            verdict = "FAIL"

        current_total = len(results_list) + 1
        current_acc = 100.0 * pass_count / current_total
        print(f"[{current_total:03d}/500] [{subj:22s} | Lvl {lvl}] {verdict} ({elapsed:.1f}s, {tokens_gen} toks) | Live: {pass_count}/{current_total} ({current_acc:.1f}%) | GT: {gt} | Pred: {pred}", flush=True)

        results_list.append({
            "index": current_total,
            "unique_id": uid,
            "subject": subj,
            "level": lvl,
            "problem": prob,
            "ground_truth": gt,
            "predicted": pred,
            "verdict": verdict,
            "tokens": tokens_gen,
            "elapsed_seconds": round(elapsed, 2)
        })
        evaluated_ids.add(uid)

        # Save to JSON after EVERY question
        low, high = wilson_ci(pass_count, current_total)
        payload = {
            "model": "Qwen 3.5 9B + SDPO-A (alpha=0.25)",
            "adapter_path": ADAPTER_PATH,
            "quantization": "4-bit NF4",
            "max_new_tokens": 4096,
            "total_evaluated": current_total,
            "total_dataset": len(ds),
            "pass_count": pass_count,
            "accuracy": round(current_acc, 2),
            "ci_95_low": round(low * 100, 2),
            "ci_95_high": round(high * 100, 2),
            "subj_stats": subj_stats,
            "elapsed_minutes": round((time.time() - t_start) / 60, 2),
            "details": results_list
        }

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80, flush=True)
    print(f"FULL 500-QUESTION SDPO-A BENCHMARK COMPLETED: {pass_count} / {len(ds)} ({100.0*pass_count/len(ds):.2f}%)", flush=True)
    low, high = wilson_ci(pass_count, len(ds))
    print(f"95% Wilson Confidence Interval: [{low*100:.2f}%, {high*100:.2f}%]", flush=True)
    print("Baseline reference was: 389 / 500 (77.80%)", flush=True)
    delta = (100.0 * pass_count / len(ds)) - 77.80
    print(f"Net Delta vs Balanced Baseline: {delta:+.2f}% ({pass_count - 389:+d} questions)", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    run_full500_eval()
