"""
SODALS v2.1 — 1536-Token GRPO Long vs Baseline Frozen Architecture Evaluation (150 MATH-500 Questions)
========================================================================================================
Rigorous Head-to-Head A/B Evaluation:
  - Baseline: 122 / 150 (81.33%) from math500_domain2_soft_blended_170_results.json
      * Train 25 subset (Q1-Q25): 23 / 25 (92.00%)
      * Held-out 125 subset (Q26-Q150): 99 / 125 (79.20%)
  - Challenger: Expert 1 replaced by newly trained `qwen_lora_adapter_grpo_long` (1536-token trained)
  - Everything else held 100% constant:
      * Base model: Qwen 3.5 9B (4-bit NF4)
      * GPR Phase Router: `gpr_domain_anchors.pt`
      * Soft LoRA Blender: same threshold, same temperature, same base scales
      * Expert 2: `qwen_lora_adapter_domain2_r8_pondering`
      * Decoding: Pure Greedy (T=0, max_new_tokens=2048, repetition_penalty=dynamic)

Saves live to: sft/eval_grpo_long_math500_150_results.json
"""

import sys
import os
import json
import re
import time
import math
import argparse
import torch
import sympy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
DEFAULT_ADAPTER_D1 = os.path.join(ROOT, "sft", "qwen_lora_adapter_grpo_long")
ADAPTER_D2_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_domain2_r8_pondering")
BASELINE_JSON = os.path.join(ROOT, "math500_domain2_soft_blended_170_results.json")
DEFAULT_OUTPUT_JSON = os.path.join(ROOT, "sft", "eval_grpo_long_math500_150_results.json")

# Import project router & blender
sys.path.insert(0, os.path.join(ROOT, "sft"))
from gpr_phase_router import NeuralPhaseRouter
from soft_lora_blender import SoftLoraBlender


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

    all_nums = re.findall(r'\b(-?\d+)\b', text)
    if all_nums:
        return all_nums[-1]

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
    s = re.sub(r'^[a-zA-Z]\s*=\s*', '', s)
    s = re.sub(r'\s*(?:cm|units|degrees|m|inches)\b', '', s, flags=re.I)
    return s.strip()


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
    cp = clean_latex(pred)
    cg = clean_latex(gt)
    if cp == cg:
        return True

    if cg.lstrip('-').isdigit():
        num_m = re.findall(r'-?\d+', cp)
        if num_m and num_m[-1] == cg:
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


def run_grpo_eval(adapter_d1_path=DEFAULT_ADAPTER_D1, output_json=DEFAULT_OUTPUT_JSON):
    print("=" * 80, flush=True)
    print("SODALS v2.1 — 1536-TOKEN GRPO vs FROZEN BASELINE (150 MATH-500 QUESTIONS)", flush=True)
    print(f"Challenger Adapter: {adapter_d1_path}", flush=True)
    print("Baseline reference: 122/150 (81.33%) | Train: 23/25 (92.0%) | Held-Out: 99/125 (79.2%)", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Baseline Reference
    print(f"Loading baseline results from: {BASELINE_JSON}...", flush=True)
    with open(BASELINE_JSON, "r", encoding="utf-8") as f:
        base_raw = json.load(f)["details"][:150]
    base_map = {item["unique_id"]: item for item in base_raw}
    print(f"Loaded {len(base_map)} baseline question records.", flush=True)

    # 2. Load 150 MATH-500 Questions
    print("Loading 150 MATH-500 questions from HuggingFace...", flush=True)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    questions = []
    for i in range(150):
        it = ds[i]
        questions.append({
            "index": i + 1,
            "unique_id": it.get("unique_id", f"math500_{i+1}"),
            "subject": it["subject"],
            "level": it.get("level", "?"),
            "problem": it["problem"],
            "ground_truth": it["answer"],
            "is_train_subset": (i < 25)
        })
    print(f"Loaded {len(questions)} questions (25 Train Subset + 125 Held-out Validation).", flush=True)

    # 3. Resume Check
    evaluated_ids = set()
    results_list = []
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                saved = json.load(f)
            results_list = saved.get("details", [])
            for r in results_list:
                evaluated_ids.add(r["unique_id"])
            print(f"Resuming: {len(evaluated_ids)} / 150 questions already evaluated.", flush=True)
        except Exception as e:
            print(f"Resume failed ({e}), starting fresh.", flush=True)
            results_list = []
            evaluated_ids = set()

    if len(evaluated_ids) >= 150:
        print("All 150 questions already evaluated! Exiting.", flush=True)
        return

    # 4. Load Models & Adapters
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
    print("Loading Base Model Qwen 3.5 9B in 4-bit NF4...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    print(f"Attaching Expert 1 (1536-Token GRPO): {adapter_d1_path}...", flush=True)
    model = PeftModel.from_pretrained(base_model, adapter_d1_path, adapter_name="expert_general_math")
    print(f"Attaching Expert 2 (Abstract Algebra): {ADAPTER_D2_PATH}...", flush=True)
    model.load_adapter(ADAPTER_D2_PATH, adapter_name="expert_abstract_algebra")

    # 5. Initialize GPR Router & Soft Blender
    router = NeuralPhaseRouter(anchors_path=os.path.join(ROOT, "sft", "gpr_domain_anchors.pt"))
    blender = SoftLoraBlender(threshold=router.threshold, temperature=0.03)
    num_blended_layers = blender.enable_multi_adapter(model)
    print(f"Soft LoRA Blender enabled across {num_blended_layers} layers.", flush=True)
    model.eval()

    print("\n" + "=" * 80, flush=True)
    print("STARTING 150-QUESTION INFERENCE (Pure Greedy, T=0, Repetition Penalty=Dynamic)", flush=True)
    print("=" * 80, flush=True)

    t_start = time.time()
    gains = 0
    drops = 0

    for idx, item in enumerate(questions):
        uid = item["unique_id"]
        if uid in evaluated_ids:
            continue

        q_idx = item["index"]
        prob = item["problem"]
        gt = item["ground_truth"]
        subj = item["subject"]
        level = item["level"]
        is_train = item["is_train_subset"]
        sub_type = "TRAIN-25" if is_train else "HELDOUT"

        base_item = base_map.get(uid, {})
        base_verdict = base_item.get("verdict", "UNKNOWN")
        base_pred = base_item.get("predicted", "")

        # Probe inputs for GPR routing
        probe_prompt = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        inputs = tokenizer(probe_prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        # GPR routing
        z_i, _, route_time_ms = router.compute_phase_divergence(model, inputs)
        alpha = blender.compute_alpha(z_i)
        rep_penalty = blender.compute_rep_penalty(alpha)
        eff_scales = blender.set_blend(model, alpha)

        # Generate Greedy Response
        t0 = time.time()
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                repetition_penalty=rep_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0

        gen_tokens = gen_out[0][prompt_len:]
        num_new_tokens = len(gen_tokens)
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Extract answer & verify
        pred = extract_boxed_answer(gen_text)
        is_correct = check_math_match(pred, gt)
        verdict = "PASS" if is_correct else "FAIL"

        # Compare with Baseline
        delta_symbol = " "
        if verdict == "PASS" and base_verdict == "FAIL":
            delta_symbol = "★ GAIN"
            gains += 1
        elif verdict == "FAIL" and base_verdict == "PASS":
            delta_symbol = "✗ DROP"
            drops += 1
        elif verdict == "PASS" and base_verdict == "PASS":
            delta_symbol = "= PASS"
        else:
            delta_symbol = "= FAIL"

        results_list.append({
            "index": q_idx,
            "unique_id": uid,
            "subject": subj,
            "level": level,
            "is_train_subset": is_train,
            "subset": sub_type,
            "problem": prob,
            "ground_truth": gt,
            "predicted": pred,
            "verdict": verdict,
            "baseline_verdict": base_verdict,
            "baseline_pred": base_pred,
            "delta": delta_symbol.strip(),
            "alpha": round(alpha, 4),
            "generated_tokens": num_new_tokens,
            "time_seconds": round(elapsed, 2),
            "full_response": gen_text
        })
        evaluated_ids.add(uid)

        total_pass = sum(1 for r in results_list if r["verdict"] == "PASS")
        curr_total = len(results_list)
        base_pass_so_far = sum(1 for r in results_list if r.get("baseline_verdict") == "PASS")

        # Running stats for subsets
        train_items = [r for r in results_list if r["is_train_subset"]]
        held_items = [r for r in results_list if not r["is_train_subset"]]
        train_pass = sum(1 for r in train_items if r["verdict"] == "PASS")
        train_base = sum(1 for r in train_items if r.get("baseline_verdict") == "PASS")
        held_pass = sum(1 for r in held_items if r["verdict"] == "PASS")
        held_base = sum(1 for r in held_items if r.get("baseline_verdict") == "PASS")

        print(
            f"[{q_idx:03d}/150] [{sub_type:7s}] [{subj[:14]:14s}] "
            f"GRPO: {verdict:4s} | Base: {base_verdict:4s} | {delta_symbol:6s} | "
            f"Score: {total_pass}/{curr_total} ({100*total_pass/curr_total:.1f}%) vs Base: {base_pass_so_far}/{curr_total} ({100*base_pass_so_far/curr_total:.1f}%) | "
            f"Net: {gains - drops:+d} (G:{gains} D:{drops}) | Tok: {num_new_tokens:4d} | {elapsed:.1f}s",
            flush=True
        )

        # Live save after every question
        payload = {
            "experiment": "SODALS v2.1 — 1536-Token GRPO vs Baseline Frozen Architecture (150 MATH-500 Questions)",
            "challenger_adapter": adapter_d1_path,
            "baseline_reference": "math500_domain2_soft_blended_170_results.json (expert_general_math balanced_r4)",
            "total_evaluated": curr_total,
            "total_benchmark": 150,
            "total_pass_grpo": total_pass,
            "total_pass_baseline": base_pass_so_far,
            "net_delta": gains - drops,
            "gains_count": gains,
            "drops_count": drops,
            "accuracy_grpo": round(100.0 * total_pass / curr_total, 2),
            "accuracy_baseline": round(100.0 * base_pass_so_far / curr_total, 2),
            "train_25_subset": {
                "evaluated": len(train_items),
                "pass_grpo": train_pass,
                "pass_baseline": train_base,
                "acc_grpo": round(100.0 * train_pass / max(1, len(train_items)), 2),
                "acc_baseline": round(100.0 * train_base / max(1, len(train_items)), 2)
            },
            "heldout_125_subset": {
                "evaluated": len(held_items),
                "pass_grpo": held_pass,
                "pass_baseline": held_base,
                "acc_grpo": round(100.0 * held_pass / max(1, len(held_items)), 2),
                "acc_baseline": round(100.0 * held_base / max(1, len(held_items)), 2)
            },
            "elapsed_minutes": round((time.time() - t_start) / 60, 2),
            "details": results_list
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80, flush=True)
    print("EVALUATION COMPLETED!", flush=True)
    print(f"Total GRPO Passes: {total_pass} / 150 ({100*total_pass/150:.2f}%) vs Baseline: {base_pass_so_far} / 150 ({100*base_pass_so_far/150:.2f}%)", flush=True)
    print(f"Held-out 125Q (Clean OOD): {held_pass} / 125 ({100*held_pass/125:.2f}%) vs Baseline: 99 / 125 (79.20%)", flush=True)
    print(f"Train 25Q Subset: {train_pass} / 25 ({100*train_pass/25:.2f}%) vs Baseline: 23 / 25 (92.00%)", flush=True)
    print(f"Net Gain/Loss: {gains - drops:+d} questions (Gains: {gains}, Drops: {drops})", flush=True)
    print(f"Results saved to: {output_json}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 1536-Token GRPO on MATH-500 150Q")
    parser.add_argument("--adapter_d1", type=str, default=DEFAULT_ADAPTER_D1, help="Challenger adapter path")
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_JSON, help="Output evaluation JSON")
    args = parser.parse_args()

    run_grpo_eval(args.adapter_d1, args.output_json)
