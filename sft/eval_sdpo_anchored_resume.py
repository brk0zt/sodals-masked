"""
Resume Script for SDPO-A MATH-500 Control Evaluation (Questions 36 to 50)
==========================================================================
Picks up exactly from question 36, evaluates questions 36-50, merges with
the 35 already saved questions, and finalizes sdpo_anchored_verification_results.json.
"""

import sys
import os
import json
import re
import time
import sympy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_sdpo_anchored")
MATH500_50_PATH = os.path.join(ROOT, "math500_stratified50_results.json")
RESULTS_JSON = os.path.join(ROOT, "sdpo_anchored_verification_results.json")
CHECKPOINT_JSON = os.path.join(ROOT, "sft", "sdpo_anchored_resume_checkpoint.json")


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
    s = s.replace(r"\pi", "pi")
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


def main():
    with open(CHECKPOINT_JSON, "r", encoding="utf-8") as f:
        cp_info = json.load(f)
    start_idx = cp_info.get("last_completed_index", 35)

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    with open(MATH500_50_PATH, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    remaining_cases = all_cases[start_idx:]
    print(f"Resuming from question {start_idx + 1} to 50 ({len(remaining_cases)} questions remaining)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    set_lora_scale(model, 0.25)
    model.eval()

    existing_details = full_data["math500_50_control"]["details"]

    for i, item in enumerate(remaining_cases):
        global_idx = start_idx + i + 1
        prob = item["problem"]
        gt = item["ground_truth"]
        subj = item.get("subject", "General")
        lvl = item.get("level", "?")

        prompt = f"<|im_start|>user\n{prob}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n<think>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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
        pred = extract_boxed_answer(gen_text)
        is_pass = check_math_match(pred, gt)
        v = "PASS" if is_pass else "FAIL"

        print(f"[{global_idx:02d}/50] [{subj:22s} | Lvl {lvl}] {v} ({elapsed:.1f}s) | GT: {gt} | Pred: {pred}", flush=True)

        existing_details.append({
            "index": global_idx,
            "subject": subj,
            "problem": prob,
            "ground_truth": gt,
            "predicted": pred,
            "verdict": v,
            "elapsed_seconds": round(elapsed, 2)
        })

        # Update JSON incrementally after each question
        total_passed = sum(1 for e in existing_details if e["verdict"] == "PASS")
        full_data["math500_50_control"] = {
            "label": "50-Question MATH-500 Control Check",
            "status": "COMPLETED" if len(existing_details) == 50 else "PARTIAL",
            "completed_count": len(existing_details),
            "total": 50,
            "pass_count": total_passed,
            "accuracy": round(100.0 * total_passed / len(existing_details), 2),
            "details": existing_details
        }
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)

    print("\n[SUCCESS] All 50 questions of MATH-500 completed and merged!")


if __name__ == "__main__":
    main()
