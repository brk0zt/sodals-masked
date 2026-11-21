"""
SDPO Multi-Seed & Multi-Model Empirical Ablation Harness
========================================================
Executes rigorous evaluation across multiple independent random seeds (e.g. 42, 123, 999)
and calculates mean, standard deviation, and Wilson score 95% confidence intervals.

Supports:
  - Any model backbone (Qwen 3.5 9B, Llama-3.1 8B, etc.)
  - Any PEFT adapter (Base, Placebo, Pure SDPO, SDPO-A)
  - Any benchmark (sdpo_heldout_100_pool.json, pristine_24_frontier_stem.json, etc.)
"""

import sys
import os
import json
import re
import math
import argparse
import time
import torch
import sympy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

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
    s = re.sub(r"\\left\(", "(", s)
    s = re.sub(r"\\right\)", ")", s)
    s = re.sub(r"\\left\[", "[", s)
    s = re.sub(r"\\right\]", "]", s)
    s = s.replace("$", "").replace("\\$", "").replace(",", "").strip()
    return s

def check_equiv(pred: str, gt: str) -> bool:
    cp = clean_latex(pred)
    cg = clean_latex(gt)
    if cp == cg:
        return True

    # Floating point comparison
    try:
        if abs(float(cp) - float(cg)) < 1e-4:
            return True
    except Exception:
        pass

    # SymPy symbolic equivalence
    try:
        sp_p = sympy.sympify(cp, rational=True)
        sp_g = sympy.sympify(cg, rational=True)
        if sympy.simplify(sp_p - sp_g) == 0:
            return True
    except Exception:
        pass

    return False

def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def run_evaluation(args):
    print("=" * 80)
    print("SDPO MULTI-SEED & MULTI-MODEL ABLATION HARNESS")
    print(f"Model Path:      {args.model_path}")
    print(f"Adapter Path:    {args.adapter_path}")
    print(f"Benchmark Suite: {args.benchmark_json}")
    print(f"Seeds:           {args.seeds}")
    print("=" * 80)

    with open(args.benchmark_json, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    cases = bench_data.get("cases", [])
    if args.limit and args.limit > 0:
        cases = cases[:args.limit]
    print(f"Loaded {len(cases)} benchmark questions for evaluation.")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Load Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load 4-bit Base Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("Loading Base Model in 4-bit NF4...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    base_model.eval()

    # Attach Adapter if provided
    if args.adapter_path and args.adapter_path.lower() != "none" and os.path.exists(args.adapter_path):
        print(f"Attaching LoRA Adapter from: {args.adapter_path}...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        model.eval()
    else:
        print("Running in PURE BASE MODEL mode (No Adapter).")
        model = base_model

    seed_results = {}
    seed_accuracies = []

    for s in seeds:
        print(f"\n---> Running Evaluation for Seed {s}...")
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

        pass_count = 0
        details = []

        for idx, item in enumerate(cases):
            problem = item["problem"]
            gt = item["ground_truth"]
            uid = item.get("unique_id", f"case_{idx}")

            prompt = f"<|im_start|>user\n{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                gen_tokens = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=False, # Deterministic greedy
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            resp_ids = gen_tokens[0][inputs.input_ids.shape[1]:]
            resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
            pred = extract_boxed_answer(resp_text)
            verdict = "PASS" if check_equiv(pred, gt) else "FAIL"

            if verdict == "PASS":
                pass_count += 1

            details.append({
                "index": idx,
                "unique_id": uid,
                "subject": item.get("subject", "Unknown"),
                "level": item.get("level", 0),
                "ground_truth": gt,
                "predicted": pred,
                "verdict": verdict
            })

            if (idx + 1) % 10 == 0 or (idx + 1) == len(cases):
                print(f"  [Seed {s}] Progress: {idx+1}/{len(cases)} | Passed: {pass_count}/{idx+1} ({pass_count/(idx+1)*100:.1f}%)")

        acc = (pass_count / len(cases)) * 100.0 if cases else 0.0
        ci_low, ci_high = wilson_interval(pass_count, len(cases))
        seed_accuracies.append(acc)

        seed_results[str(s)] = {
            "seed": s,
            "pass_count": pass_count,
            "total": len(cases),
            "accuracy": acc,
            "ci_95": [round(ci_low * 100, 2), round(ci_high * 100, 2)],
            "details": details
        }

    # Statistical Aggregation
    mean_acc = sum(seed_accuracies) / len(seed_accuracies) if seed_accuracies else 0.0
    if len(seed_accuracies) > 1:
        variance = sum((x - mean_acc) ** 2 for x in seed_accuracies) / (len(seed_accuracies) - 1)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0.0

    master_summary = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "benchmark_file": args.benchmark_json,
        "evaluated_cases": len(cases),
        "seeds": seeds,
        "mean_accuracy": round(mean_acc, 2),
        "std_deviation": round(std_dev, 2),
        "seed_breakdown": seed_results
    }

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(master_summary, f, indent=2, ensure_ascii=False)
        print(f"\n[REPORT SAVED] Multi-seed summary saved to: {args.output_json}")

    print("\n" + "=" * 80)
    print("MULTI-SEED ABLATION SUMMARY")
    print(f"Mean Accuracy:    {mean_acc:.2f}% ± {std_dev:.2f}%")
    for s_str, res in seed_results.items():
        print(f"  Seed {s_str}: {res['pass_count']}/{res['total']} ({res['accuracy']:.2f}%) [95% CI: {res['ci_95'][0]}% - {res['ci_95'][1]}%]")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Seed & Multi-Model Evaluation Harness")
    parser.add_argument("--model_path", type=str, default=r"c:\Users\Burak\Desktop\sodals_core\models\Qwen 3.5 9B")
    parser.add_argument("--adapter_path", type=str, default="none")
    parser.add_argument("--benchmark_json", type=str, default=r"c:\Users\Burak\Desktop\sodals_core\sft\sdpo_heldout_100_pool.json")
    parser.add_argument("--seeds", type=str, default="42,123,999")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for dry-run testing")
    parser.add_argument("--output_json", type=str, default=r"c:\Users\Burak\Desktop\sodals_core\multiseed_ablation_report.json")
    args = parser.parse_args()

    run_evaluation(args)
