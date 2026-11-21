"""
Pristine 24Q Frontier STEM Benchmark Evaluator
==============================================
Evaluates adapters on the 24 canonical frontier STEM questions:
  - Lie Theory (F4, G2, E8)
  - Modular Forms & Analytic Number Theory (tau(2), tau(3), zeta(4))
  - Symplectic Geometry (sp(6), sp(4), T*Q)
  - Arithmetic Geometry (rank, torsion, rational points over F5)
  - Algebraic Topology (pi_3(S^2), pi_4(S^3), pi_1(RP^2))
  - Representation Theory (SU(3) fund, SU(3) adj, Spin(7))
  - Complex Analysis (residues, winding number)
  - Characteristic Classes (S^2, CP^2, T^2)

Outputs: pristine_24_comparative_report.json
"""

import sys
import os
import json
import re
import time
import argparse
import sympy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
BENCHMARK_PATH = os.path.join(ROOT, "sft", "pristine_24_frontier_stem.json")
OUTPUT_JSON = os.path.join(ROOT, "pristine_24_comparative_report.json")

ADAPTERS = {
    "Balanced_Baseline": os.path.join(ROOT, "sft", "qwen_lora_adapter_balanced_r4"),
    "SDPO": os.path.join(ROOT, "sft", "qwen_lora_adapter_sdpo"),
    "Placebo_Control": os.path.join(ROOT, "sft", "qwen_lora_adapter_control"),
    "Interleaved_R8": os.path.join(ROOT, "sft", "qwen_lora_adapter_interleaved_d1_d2_r8")
}


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
    s = re.sub(r"\\mathbb\{([^}]+)\}", r"\1", s)
    s = s.replace(r"\$", "").replace("$", "").strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"^\circ", "").replace("^{\\circ}", "").replace("°", "")
    s = s.replace(" ", "")
    # Canonicalize common representations
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


def check_math_match(pred: str, gt: str) -> bool:
    cp = clean_latex(pred)
    cg = clean_latex(gt)
    if cp == cg:
        return True

    cp_bare = re.sub(r"^[a-zA-Z]\s*=\s*", "", cp)
    cg_bare = re.sub(r"^[a-zA-Z]\s*=\s*", "", cg)
    if cp_bare == cg_bare:
        return True

    # Check topology equivalence Z_2
    if cg in ["Z_2", "Z/2Z", "\\mathbb{Z}_2", "\\mathbb{Z}/2\\mathbb{Z}"]:
        if cp in ["Z_2", "Z/2Z", "\\mathbb{Z}_2", "\\mathbb{Z}/2\\mathbb{Z}"]:
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


def evaluate_adapter(adapter_name: str, adapter_path: str, base_model, tokenizer, questions, scale: float = 1.0):
    print("\n" + "=" * 80)
    print(f"EVALUATING ON PRISTINE 24Q: {adapter_name} (LoRA Scale Multiplier: {scale})")
    print("=" * 80)

    model = PeftModel.from_pretrained(base_model, adapter_path)
    if scale != 1.0:
        set_lora_scale(model, scale)
    model.eval()

    passed = 0
    results = []

    for i, q in enumerate(questions):
        prob = q["problem"]
        gt = q["ground_truth"]
        subj = q["subject"]
        qid = q["id"]

        prompt = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
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

        gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred = extract_boxed_answer(gen_text)
        is_pass = check_math_match(pred, gt)

        if is_pass:
            passed += 1
            verdict = "PASS"
        else:
            verdict = "FAIL"

        print(f"[{i+1:02d}/24] [{subj:22s}] {verdict} ({elapsed:.1f}s) | GT: {gt} | Pred: {pred}", flush=True)

        results.append({
            "id": qid,
            "subject": subj,
            "problem": prob,
            "ground_truth": gt,
            "predicted": pred,
            "verdict": verdict,
            "elapsed_seconds": round(elapsed, 2)
        })

    acc = 100.0 * passed / len(questions)
    print(f"[{adapter_name} SCORE] {passed} / {len(questions)} ({acc:.1f}%)")

    # Clean unload of adapter
    del model
    torch.cuda.empty_cache()

    return {
        "pass_count": passed,
        "total": len(questions),
        "accuracy": acc,
        "details": results
    }


def main():
    parser = argparse.ArgumentParser(description="Pristine 24Q STEM Benchmark Evaluator")
    parser.add_argument("--adapter", type=str, default="Interleaved_R8",
                        help=f"Adapter to evaluate: {list(ADAPTERS.keys())} or 'all' (default: Interleaved_R8)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="LoRA scale multiplier (1.0 = native PEFT scale, 0.25 = legacy damping)")
    args = parser.parse_args()

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} canonical frontier STEM questions from {BENCHMARK_PATH}.")

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

    # Load existing report to avoid re-evaluating baseline adapters
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                full_report = json.load(f)
        except Exception:
            full_report = {}
    else:
        full_report = {}

    if args.adapter == "all":
        targets = ADAPTERS
    else:
        if args.adapter not in ADAPTERS:
            print(f"Unknown adapter '{args.adapter}'. Available: {list(ADAPTERS.keys())}")
            sys.exit(1)
        targets = {args.adapter: ADAPTERS[args.adapter]}

    for name, path in targets.items():
        if os.path.exists(path):
            key_name = f"{name}_Scale{args.scale}" if args.scale != 0.25 else name
            res = evaluate_adapter(name, path, base_model, tokenizer, questions, scale=args.scale)
            res["lora_scale"] = args.scale
            full_report[key_name] = res
        else:
            print(f"Warning: Adapter path {path} not found.")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("PRISTINE 24Q COMPARATIVE REPORT COMPLETED")
    print("=" * 80)
    for name, res in full_report.items():
        scale_info = f"(scale={res.get('lora_scale', '0.25')})"
        print(f"  - {name:24s} {scale_info:14s}: {res['pass_count']} / {res['total']} ({res['accuracy']:.1f}%)")
    print(f"Full comparative results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
