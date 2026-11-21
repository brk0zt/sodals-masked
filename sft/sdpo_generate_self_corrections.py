"""
SDPO Phase 2: Model Self-Correction Generation on Train Failures Only
====================================================================
Prompts Qwen 3.5 9B with diagnostic feedback ONLY on the 76 Train Failures.
The 35 Held-Out failures remain strictly untouched.

The model generates self-correction reasoning trajectories.
Only trajectories whose final answer is strictly verified by SymPy/GT
are admitted into the SDPO training dataset (sdpo_verified_trajectories.jsonl).
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
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_balanced_r4")
TRAIN_FAILURES_JSON = os.path.join(ROOT, "sft", "sdpo_train_failures.json")
OUTPUT_TRAJECTORIES_JSONL = os.path.join(ROOT, "sft", "sdpo_verified_trajectories.jsonl")


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


def generate_critique(prob: str, gt: str, pred: str, subj: str) -> str:
    """
    Diagnostic critique specifying the mathematical checkpoint for self-correction.
    """
    if "Intermediate Algebra" in subj:
        return (
            "Pay close attention to defining an auxiliary polynomial or symmetry. "
            "If rational functions or roots are involved, clear denominators and check "
            "whether roots of the auxiliary equation can be directly evaluated at the target point."
        )
    elif "Algebra" in subj:
        return (
            "Double-check algebraic expansions, sign flips, and extraneous roots. "
            "Express the final answer cleanly as the exact value or set of values requested."
        )
    elif "Prealgebra" in subj:
        return (
            "Carefully re-evaluate the order of operations, parenthesis placement, or boundary cases. "
            "Calculate all intermediate arithmetic steps explicitly from first principles."
        )
    elif "Number Theory" in subj:
        return (
            "Examine prime factorizations and modular arithmetic constraints. "
            "Check for coprime conditions and verify that no factors were double counted."
        )
    elif "Precalculus" in subj:
        return (
            "Apply exact trigonometric identities or coordinate transform formulas. "
            "Ensure that vectors and complex conjugates maintain proper signs in each component."
        )
    elif "Geometry" in subj:
        return (
            "Use standard geometric theorems (Power of a Point, Stewart's, cyclic quadrilaterals) "
            "and verify all intermediate lengths using the Pythagorean theorem."
        )
    else:
        return (
            "Carefully analyze combinatorial cases or complementary counting. "
            "Verify all distinct configurations without overcounting identical arrangements."
        )


def main():
    print("=" * 80)
    print("SDPO: MODEL SELF-CORRECTION GENERATION (TRAIN FAILURES ONLY, N=76)")
    print("=" * 80)

    with open(TRAIN_FAILURES_JSON, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    candidates = train_data["cases"]
    print(f"Loaded {len(candidates)} train failure candidates from {TRAIN_FAILURES_JSON}.")

    # Load Model & Tokenizer
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
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    set_lora_scale(model, 0.25)
    model.eval()

    # Clear previous output
    if os.path.exists(OUTPUT_TRAJECTORIES_JSONL):
        os.remove(OUTPUT_TRAJECTORIES_JSONL)

    verified_corrections = []
    total = len(candidates)
    passed = 0

    print("\n" + "=" * 80)
    print("RUNNING CONDITIONAL SELF-CORRECTION GENERATION")
    print("=" * 80)

    for i, item in enumerate(candidates):
        prob = item["problem"]
        gt = item["ground_truth"]
        err_pred = item["predicted"]
        subj = item["subject"]
        lvl = item.get("level", "?")
        uid = item.get("unique_id", f"case_{i+1}")

        critique = generate_critique(prob, gt, err_pred, subj)

        prompt = (
            f"<|im_start|>user\n{prob}\n\n"
            f"[Previous Attempt Feedback]: A previous attempt yielded '{err_pred}', which was incorrect. "
            f"Critique: {critique}\n\n"
            f"Please reason step by step, rigorously correct the calculation, and place your final answer in \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )

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

        if is_pass:
            passed += 1
            verdict = "SELF-CORRECTED (PASS)"
            correction_entry = {
                "unique_id": uid,
                "subject": subj,
                "level": lvl,
                "problem": prob,
                "critique": critique,
                "ground_truth": gt,
                "erroneous_pred": err_pred,
                "corrected_pred": pred,
                "solution_cot": gen_text,
                "messages": [
                    {
                        "role": "user",
                        "content": f"{prob}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
                    },
                    {
                        "role": "assistant",
                        "content": f"<think>\n{gen_text}"
                    }
                ]
            }
            verified_corrections.append(correction_entry)

            with open(OUTPUT_TRAJECTORIES_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(correction_entry, ensure_ascii=False) + "\n")
        else:
            verdict = "STILL FAILED"

        print(f"[{i+1:02d}/{total}] [{subj:22s} | Lvl {lvl}] {verdict} ({elapsed:.1f}s) | GT: {gt} | Pred: {pred}")

    print("\n" + "=" * 80)
    print(f"SDPO SELF-CORRECTION RESULTS: {passed} / {total} ({100*passed/total:.1f}%) Successfully Corrected!")
    print(f"Saved {len(verified_corrections)} verified self-correction trajectories to {OUTPUT_TRAJECTORIES_JSONL}")
    print("=" * 80)


if __name__ == "__main__":
    main()
