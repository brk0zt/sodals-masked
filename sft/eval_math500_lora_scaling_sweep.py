"""
MATH-500 LoRA Dynamic Scaling Sweep (Alpha = 0.50 & Alpha = 0.25)
================================================================
Evaluates the SODALS LoRA adapter at reduced scaling factors (alpha = 0.50 and alpha = 0.25)
on the exact same 50 stratified MATH-500 questions to find the optimal Pareto frontier
between general mathematical reasoning (Base model = 74%) and domain adaptation.

Inference: Pure Greedy (T=0, max_new_tokens=1024, repetition_penalty=1.05)
Grading: SymPy algebraic normalization.
"""

import sys
import os
import json
import re
import time
import torch
import sympy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_balanced_r4")
INPUT_JSON = os.path.join(ROOT, "math500_stratified50_results.json")


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
    count = 0
    for module in model.modules():
        if hasattr(module, "scaling") and hasattr(module, "lora_alpha") and hasattr(module, "r"):
            for adapter_name in module.scaling:
                alpha = module.lora_alpha[adapter_name]
                r = module.r[adapter_name]
                module.scaling[adapter_name] = (alpha / r) * scale
                count += 1
    print(f"[LoRA Scaling] Set scaling factor = {scale:.2f} across {count} LoRA adapter layers.")


def run_scaling_sweep(scale_values=[0.50, 0.25]):
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        benchmark_questions = json.load(f)
    print(f"Loaded {len(benchmark_questions)} benchmark questions from {INPUT_JSON}.")

    print("Loading Tokenizer...")
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

    print(f"Loading LoRA Adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    sweep_summary = {}

    for scale in scale_values:
        print("\n" + "=" * 80)
        print(f"STARTING MATH-500 EVALUATION AT LoRA SCALE = {scale:.2f}")
        print("=" * 80)

        set_lora_scale(model, scale)
        output_file = os.path.join(ROOT, f"math500_lora_scale_{int(scale*100):03d}_results.json")

        results = []
        pass_count = 0
        start_scale = time.time()
        by_subject = {}

        for i, item in enumerate(benchmark_questions):
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

            print(f"[{i+1}/50] [Scale {scale:.2f}] [{subj:22s} | Lvl {level}] {verdict} ({elapsed:.1f}s, {tokens_gen} toks) | GT: {gt_ans} | Pred: {pred_ans}")

            results.append({
                "index": i + 1,
                "scale": scale,
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

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        total_time = time.time() - start_scale
        accuracy = 100.0 * pass_count / len(benchmark_questions)
        sweep_summary[scale] = {
            "pass_count": pass_count,
            "total": len(benchmark_questions),
            "accuracy": accuracy,
            "by_subject": by_subject,
            "time_minutes": total_time / 60
        }

        print(f"\n[SCALE {scale:.2f} COMPLETE] Score: {pass_count} / {len(benchmark_questions)} ({accuracy:.1f}%) in {total_time/60:.1f} min.")

    print("\n" + "=" * 80)
    print("ALL SCALING SWEEPS COMPLETED! SUMMARY TABLE:")
    print("=" * 80)
    print(f"Base Model (Scale 0.00) : 37 / 50 (74.0%) [Pre-computed]")
    for s, data in sweep_summary.items():
        print(f"LoRA Scale {s:.2f}        : {data['pass_count']} / {data['total']} ({data['accuracy']:.1f}%)")
    print(f"Full LoRA  (Scale 1.00) : 28 / 50 (56.0%) [Pre-computed]")


if __name__ == "__main__":
    run_scaling_sweep([0.50, 0.25])
