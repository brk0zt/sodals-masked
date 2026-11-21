"""
SODALS v2.1 — Step 3: Module 1 Isolated Ablation (Prefill Pondering ODE Integration)
==================================================================================
Strict Isolated Ablation on 150 MATH-500 Questions:
  - Baseline 0 (SFT Frozen Baseline): 122 / 150 (81.33%)
  - Baseline 1 (1536-Token GRPO Long): 123 / 150 (82.00%)
  - Challenger: 1536-Token GRPO Long + Module 1 (Adaptive Prefill Pondering ODE)
      * Explicit RK4 integration on prompt's final semantic latent state
      * Contemplation depth dynamically modulated by GPR alpha (N = 2 to 5 steps)
      * Zero KV-cache or autoregressive decoding overhead (prefill-only hook)
      * Master weights loaded from gpr_soft_moa_aux_modules.pt

Protocol: "Önce doğrula, sonra inşa et" (Verify first, build second)
Evaluates 150 MATH-500 questions in pure greedy mode (T=0, repetition_penalty=dynamic).
Saves live to: sft/eval_grpo_long_pondering_150_results.json
"""

import sys
import os
import json
import re
import time
import math
import argparse
import torch
import torch.nn as nn
import sympy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_D1_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_grpo_long")
ADAPTER_D2_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_domain2_r8_pondering")
PONDERING_WEIGHTS = os.path.join(ROOT, "sft", "qwen_lora_gpr_soft_moa", "gpr_soft_moa_aux_modules.pt")
BASELINE_SFT_JSON = os.path.join(ROOT, "math500_domain2_soft_blended_170_results.json")
BASELINE_GRPO_JSON = os.path.join(ROOT, "sft", "eval_grpo_long_math500_150_results.json")
OUTPUT_JSON = os.path.join(ROOT, "sft", "eval_grpo_long_pondering_150_results.json")

# Import project router, blender & pondering ODE
sys.path.insert(0, os.path.join(ROOT, "sft"))
from gpr_phase_router import NeuralPhaseRouter
from soft_lora_blender import SoftLoraBlender
from adaptive_pondering_ode import AdaptivePonderingODEModule


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


def run_pondering_eval(output_json=OUTPUT_JSON):
    print("=" * 80, flush=True)
    print("SODALS v2.1 — STEP 3: PREFILL PONDERING ODE ABLATION (150 MATH-500 QUESTIONS)", flush=True)
    print(f"Base Adapter: {ADAPTER_D1_PATH}", flush=True)
    print("Baseline 0 (SFT):  122/150 (81.33%) | Train: 23/25 (92.0%) | Held-Out: 99/125 (79.2%)", flush=True)
    print("Baseline 1 (GRPO): 123/150 (82.00%) | Train: 23/25 (92.0%) | Held-Out: 100/125 (80.0%)", flush=True)
    print("=" * 80, flush=True)

    # 1. Load SFT and GRPO Baseline References
    sft_map = {}
    if os.path.exists(BASELINE_SFT_JSON):
        with open(BASELINE_SFT_JSON, "r", encoding="utf-8") as f:
            sft_raw = json.load(f)["details"][:150]
        sft_map = {item["unique_id"]: item for item in sft_raw}
        print(f"Loaded {len(sft_map)} SFT baseline records.", flush=True)

    grpo_map = {}
    if os.path.exists(BASELINE_GRPO_JSON):
        with open(BASELINE_GRPO_JSON, "r", encoding="utf-8") as f:
            grpo_raw = json.load(f)["details"][:150]
        grpo_map = {item["unique_id"]: item for item in grpo_raw}
        print(f"Loaded {len(grpo_map)} GRPO baseline records.", flush=True)

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
            print(f"Resume check failed ({e}), starting fresh.", flush=True)
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

    print(f"Attaching Expert 1 (1536-Token GRPO): {ADAPTER_D1_PATH}...", flush=True)
    model = PeftModel.from_pretrained(base_model, ADAPTER_D1_PATH, adapter_name="expert_general_math")
    print(f"Attaching Expert 2 (Abstract Algebra): {ADAPTER_D2_PATH}...", flush=True)
    model.load_adapter(ADAPTER_D2_PATH, adapter_name="expert_abstract_algebra")

    # 5. Initialize GPR Router & Soft Blender
    router = NeuralPhaseRouter(anchors_path=os.path.join(ROOT, "sft", "gpr_domain_anchors.pt"))
    blender = SoftLoraBlender(threshold=router.threshold, temperature=0.03)
    num_blended_layers = blender.enable_multi_adapter(model)
    print(f"Soft LoRA Blender enabled across {num_blended_layers} layers.", flush=True)

    # 6. Initialize Adaptive Pondering ODE Module & Prefill Hook
    print("Initializing Adaptive Pondering ODE Module...", flush=True)
    pondering_ode = AdaptivePonderingODEModule(
        hidden_dim=4096,
        dt=0.1,
        min_steps=2,
        max_steps=5,
        base_lambda=0.3
    ).to("cuda")

    if os.path.exists(PONDERING_WEIGHTS):
        aux_data = torch.load(PONDERING_WEIGHTS, map_location="cuda", weights_only=False)
        if "pondering_state_dict" in aux_data:
            pondering_ode.load_state_dict(aux_data["pondering_state_dict"])
            print("  [SUCCESS] Loaded trained Pondering ODE weights from aux_modules.pt!", flush=True)
        else:
            print("  [WARNING] pondering_state_dict not in aux_data, using initialized weights.", flush=True)
    else:
        print(f"  [WARNING] Pondering weights file not found: {PONDERING_WEIGHTS}", flush=True)
    pondering_ode.eval()

    # Context holder for prefill pondering hook
    pondering_ctx = {
        "enabled": False,
        "alpha": 0.2,
        "steps_run": 0,
        "lambda_eff": 0.3,
        "hook_call_count": 0
    }

    def prefill_pondering_hook(module, args, output):
        """
        Intercepts backbone output during prompt PREFILL (seq_len > 1).
        Evolves the prompt's final semantic representation over the continuous latent manifold.
        Auto-bypasses autoregressive generation (seq_len == 1) for zero inference latency overhead.
        """
        if not pondering_ctx["enabled"]:
            return output
        last_h = output.last_hidden_state
        if last_h.size(1) > 1:
            h_last = last_h[:, -1:, :]
            alpha_val = pondering_ctx["alpha"]
            with torch.no_grad():
                h_pondered, steps, lam = pondering_ode(h_last, alpha=alpha_val)
            last_h[:, -1:, :] = h_pondered
            pondering_ctx["steps_run"] = steps
            pondering_ctx["lambda_eff"] = round(lam, 4)
            pondering_ctx["hook_call_count"] += 1
        return output

    # Register hook on transformer backbone
    backbone = model.base_model.model.model
    hook_handle = backbone.register_forward_hook(prefill_pondering_hook)
    print("Attached prefill_pondering_hook to transformer backbone successfully!", flush=True)

    model.eval()

    print("\n" + "=" * 80, flush=True)
    print("STARTING 150-QUESTION INFERENCE: GRPO LONG + PREFILL PONDERING ODE", flush=True)
    print("=" * 80, flush=True)

    t_start = time.time()
    gains_vs_grpo = 0
    drops_vs_grpo = 0
    gains_vs_sft = 0
    drops_vs_sft = 0

    # Calculate existing gains/drops if resuming
    for r in results_list:
        v = r["verdict"]
        gv = r.get("grpo_verdict", "UNKNOWN")
        sv = r.get("sft_verdict", "UNKNOWN")
        if v == "PASS" and gv == "FAIL":
            gains_vs_grpo += 1
        elif v == "FAIL" and gv == "PASS":
            drops_vs_grpo += 1
        if v == "PASS" and sv == "FAIL":
            gains_vs_sft += 1
        elif v == "FAIL" and sv == "PASS":
            drops_vs_sft += 1

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

        sft_item = sft_map.get(uid, {})
        sft_verdict = sft_item.get("verdict", "UNKNOWN")
        sft_pred = sft_item.get("predicted", "")

        grpo_item = grpo_map.get(uid, {})
        grpo_verdict = grpo_item.get("verdict", "UNKNOWN")
        grpo_pred = grpo_item.get("predicted", "")

        # Probe inputs for GPR routing
        probe_prompt = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        inputs = tokenizer(probe_prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        # 1. GPR routing (pondering hook disabled during probe)
        pondering_ctx["enabled"] = False
        z_i, _, _ = router.compute_phase_divergence(model, inputs)
        alpha = blender.compute_alpha(z_i)
        rep_penalty = blender.compute_rep_penalty(alpha)
        eff_scales = blender.set_blend(model, alpha)

        # 2. Configure Pondering Context for Prefill Hook
        pondering_ctx["enabled"] = True
        pondering_ctx["alpha"] = alpha
        pondering_ctx["steps_run"] = 0

        # 3. Generate Greedy Response with Prefill Pondering ODE Active
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

        # Turn off pondering hook after generation
        pondering_ctx["enabled"] = False
        rk4_steps_executed = pondering_ctx["steps_run"]
        lam_eff_applied = pondering_ctx["lambda_eff"]

        gen_tokens = gen_out[0][prompt_len:]
        num_new_tokens = len(gen_tokens)
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # 4. Extract boxed answer & verify
        pred = extract_boxed_answer(gen_text)
        is_correct = check_math_match(pred, gt)
        verdict = "PASS" if is_correct else "FAIL"

        # Compare with GRPO Baseline
        delta_vs_grpo_sym = "= PASS" if (verdict == "PASS" and grpo_verdict == "PASS") else (
            "★ GAIN" if (verdict == "PASS" and grpo_verdict == "FAIL") else (
                "✗ DROP" if (verdict == "FAIL" and grpo_verdict == "PASS") else "= FAIL"
            )
        )
        if verdict == "PASS" and grpo_verdict == "FAIL":
            gains_vs_grpo += 1
        elif verdict == "FAIL" and grpo_verdict == "PASS":
            drops_vs_grpo += 1

        # Compare with SFT Baseline
        if verdict == "PASS" and sft_verdict == "FAIL":
            gains_vs_sft += 1
        elif verdict == "FAIL" and sft_verdict == "PASS":
            drops_vs_sft += 1

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
            "grpo_verdict": grpo_verdict,
            "grpo_pred": grpo_pred,
            "sft_verdict": sft_verdict,
            "sft_pred": sft_pred,
            "delta_vs_grpo": delta_vs_grpo_sym.strip(),
            "alpha": round(alpha, 4),
            "rk4_steps": rk4_steps_executed,
            "lambda_eff": lam_eff_applied,
            "generated_tokens": num_new_tokens,
            "time_seconds": round(elapsed, 2),
            "full_response": gen_text
        })
        evaluated_ids.add(uid)

        total_pass = sum(1 for r in results_list if r["verdict"] == "PASS")
        curr_total = len(results_list)
        grpo_pass_so_far = sum(1 for r in results_list if r.get("grpo_verdict") == "PASS")
        sft_pass_so_far = sum(1 for r in results_list if r.get("sft_verdict") == "PASS")

        # Running stats for subsets
        train_items = [r for r in results_list if r["is_train_subset"]]
        held_items = [r for r in results_list if not r["is_train_subset"]]
        train_pass = sum(1 for r in train_items if r["verdict"] == "PASS")
        train_grpo = sum(1 for r in train_items if r.get("grpo_verdict") == "PASS")
        train_sft = sum(1 for r in train_items if r.get("sft_verdict") == "PASS")

        held_pass = sum(1 for r in held_items if r["verdict"] == "PASS")
        held_grpo = sum(1 for r in held_items if r.get("grpo_verdict") == "PASS")
        held_sft = sum(1 for r in held_items if r.get("sft_verdict") == "PASS")

        print(
            f"[{q_idx:03d}/150] [{sub_type:7s}] [{subj[:14]:14s}] "
            f"Ponder: {verdict:4s} | GRPO: {grpo_verdict:4s} | SFT: {sft_verdict:4s} | {delta_vs_grpo_sym:6s} | "
            f"Score: {total_pass}/{curr_total} ({100*total_pass/curr_total:.1f}%) | "
            f"Net vs GRPO: {gains_vs_grpo - drops_vs_grpo:+d} (G:{gains_vs_grpo} D:{drops_vs_grpo}) | "
            f"RK4: {rk4_steps_executed} st | Tok: {num_new_tokens:4d} | {elapsed:.1f}s",
            flush=True
        )

        # Live save after every single question
        payload = {
            "experiment": "SODALS v2.1 — Step 3: Module 1 Isolated Ablation (Prefill Pondering ODE Integration)",
            "module_tested": "Module 1: Adaptive Prefill Pondering ODE (RK4 Latent Relaxation)",
            "adapter_path": ADAPTER_D1_PATH,
            "weights_path": PONDERING_WEIGHTS,
            "total_evaluated": curr_total,
            "total_benchmark": 150,
            "total_pass_pondering": total_pass,
            "total_pass_grpo": grpo_pass_so_far,
            "total_pass_sft": sft_pass_so_far,
            "net_delta_vs_grpo": gains_vs_grpo - drops_vs_grpo,
            "gains_vs_grpo_count": gains_vs_grpo,
            "drops_vs_grpo_count": drops_vs_grpo,
            "net_delta_vs_sft": gains_vs_sft - drops_vs_sft,
            "gains_vs_sft_count": gains_vs_sft,
            "drops_vs_sft_count": drops_vs_sft,
            "accuracy_pondering": round(100.0 * total_pass / curr_total, 2),
            "accuracy_grpo": round(100.0 * grpo_pass_so_far / curr_total, 2),
            "accuracy_sft": round(100.0 * sft_pass_so_far / curr_total, 2),
            "train_25_subset": {
                "evaluated": len(train_items),
                "pass_pondering": train_pass,
                "pass_grpo": train_grpo,
                "pass_sft": train_sft,
                "acc_pondering": round(100.0 * train_pass / max(1, len(train_items)), 2),
                "acc_grpo": round(100.0 * train_grpo / max(1, len(train_items)), 2),
                "acc_sft": round(100.0 * train_sft / max(1, len(train_items)), 2)
            },
            "heldout_125_subset": {
                "evaluated": len(held_items),
                "pass_pondering": held_pass,
                "pass_grpo": held_grpo,
                "pass_sft": held_sft,
                "acc_pondering": round(100.0 * held_pass / max(1, len(held_items)), 2),
                "acc_grpo": round(100.0 * held_grpo / max(1, len(held_items)), 2),
                "acc_sft": round(100.0 * held_sft / max(1, len(held_items)), 2)
            },
            "elapsed_minutes": round((time.time() - t_start) / 60, 2),
            "details": results_list
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80, flush=True)
    print("MODULE 1 (PREFILL PONDERING ODE) EVALUATION COMPLETED!", flush=True)
    print(f"Total Passes: {total_pass} / 150 ({100*total_pass/150:.2f}%)", flush=True)
    print(f"  vs GRPO Long: {grpo_pass_so_far} / 150 ({100*grpo_pass_so_far/150:.2f}%) [Net: {gains_vs_grpo - drops_vs_grpo:+d}]", flush=True)
    print(f"  vs SFT Base:  {sft_pass_so_far} / 150 ({100*sft_pass_so_far/150:.2f}%) [Net: {gains_vs_sft - drops_vs_sft:+d}]", flush=True)
    print(f"Held-out 125Q (Clean OOD): {held_pass} / 125 ({100*held_pass/125:.2f}%) vs GRPO: {held_grpo}/125 vs SFT: {held_sft}/125", flush=True)
    print(f"Train 25Q Subset:          {train_pass} / 25 ({100*train_pass/25:.2f}%) vs GRPO: {train_grpo}/25 vs SFT: {train_sft}/25", flush=True)
    print(f"Results saved to: {output_json}", flush=True)


if __name__ == "__main__":
    run_pondering_eval()
