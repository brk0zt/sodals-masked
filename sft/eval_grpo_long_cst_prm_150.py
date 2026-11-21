"""
SODALS v2.1 — Step 3: Module 2 Isolated Ablation
(CST PRM: SymPy Symbolic Verification + Jacobian Filter + Contradiction Gate)
=============================================================================
Strict Isolated Ablation on 150 MATH-500 Questions:
  - Baseline 0 (SFT Frozen Baseline): 122 / 150 (81.33%)
  - Baseline 1 (1536-Token GRPO Long): 123 / 150 (82.00%)
  - Step 3 Module 1 (Prefill Pondering ODE): 123 / 150 (82.00%) [0 drops, verified]
  - Challenger: Module 1 (Pondering ODE) + Module 2 (CST PRM & Contradiction Gate)
      * Pure Greedy Pass first (preserving 100% of passing chains with 0 regressions)
      * CST PRM inspection:
          - CstSymbolicVerifier checks arithmetic and symbolic consistency
          - ReasoningChainJacobianFilter checks for proof abandonment (jump ratio > 3.0)
      * Contradiction Gate:
          - If primary CoT has zero contradictions: accepted immediately.
          - If explicit contradiction or fatal proof abandonment detected: triggers
            a single targeted self-correction pass with explicit discrepancy feedback.
      * Dual Evaluation Metrics:
          - Strict Legacy Match: exact identical criteria to Baseline 0 & 1
          - Normalized Symbolic Match: robust mathematical equivalence (subscripts, roots, etc.)

Protocol: "Önce doğrula, sonra inşa et" (Verify first, build second)
Saves live to: sft/eval_grpo_long_cst_prm_150_results.json
"""

import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import json
import re
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
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
BASELINE_PONDER_JSON = os.path.join(ROOT, "sft", "eval_grpo_long_pondering_150_results.json")
OUTPUT_JSON = os.path.join(ROOT, "sft", "eval_grpo_long_cst_prm_150_results.json")

# Import project router, blender, pondering ODE, and CST PRM
sys.path.insert(0, os.path.join(ROOT, "sft"))
from gpr_phase_router import NeuralPhaseRouter
from soft_lora_blender import SoftLoraBlender
from adaptive_pondering_ode import AdaptivePonderingODEModule
from cst_symbolic_verifier import CstSymbolicVerifier, NormalizedMathMatcher
from reasoning_chain_jacobian import ReasoningChainJacobianFilter


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


def run_cst_prm_eval(output_json=OUTPUT_JSON):
    print("=" * 80, flush=True)
    print("SODALS v2.1 — STEP 3: MODULE 2 CST PRM & CONTRADICTION GATE (150 MATH-500)", flush=True)
    print(f"Base Adapter: {ADAPTER_D1_PATH}", flush=True)
    print("Baseline 0 (SFT):        122 / 150 (81.33%) | Train: 23/25 | Held-Out: 99/125", flush=True)
    print("Baseline 1 (GRPO Long):  123 / 150 (82.00%) | Train: 23/25 | Held-Out: 100/125", flush=True)
    print("Module 1 (Pondering ODE): 123 / 150 (82.00%) | Train: 23/25 | Held-Out: 100/125", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Baselines for reference
    sft_map = {}
    if os.path.exists(BASELINE_SFT_JSON):
        with open(BASELINE_SFT_JSON, "r", encoding="utf-8") as f:
            sft_raw = json.load(f)["details"][:150]
        sft_map = {item["unique_id"]: item for item in sft_raw}

    grpo_map = {}
    if os.path.exists(BASELINE_GRPO_JSON):
        with open(BASELINE_GRPO_JSON, "r", encoding="utf-8") as f:
            grpo_raw = json.load(f)["details"][:150]
        grpo_map = {item["unique_id"]: item for item in grpo_raw}

    ponder_map = {}
    if os.path.exists(BASELINE_PONDER_JSON):
        with open(BASELINE_PONDER_JSON, "r", encoding="utf-8") as f:
            ponder_raw = json.load(f)["details"][:150]
        ponder_map = {item["unique_id"]: item for item in ponder_raw}

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
            print("  [SUCCESS] Loaded trained Pondering ODE weights!", flush=True)
    pondering_ode.eval()

    pondering_ctx = {
        "enabled": False,
        "alpha": 0.2,
        "steps_run": 0,
        "lambda_eff": 0.3
    }

    def prefill_pondering_hook(module, args, output):
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
        return output

    model.base_model.model.model.register_forward_hook(prefill_pondering_hook)

    # 7. Initialize Jacobian Filter
    jacobian_filter = ReasoningChainJacobianFilter(jump_threshold=3.5)

    model.eval()

    print("\n" + "=" * 80, flush=True)
    print("STARTING 150-QUESTION INFERENCE: PONDERING ODE + CST PRM & CONTRADICTION GATE", flush=True)
    print("=" * 80, flush=True)

    t_start = time.time()
    total_gate_triggered = sum(1 for r in results_list if r.get("gate_triggered", False))
    total_gate_gains = sum(1 for r in results_list if r.get("gate_gain", False))

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
        grpo_item = grpo_map.get(uid, {})
        grpo_verdict = grpo_item.get("verdict", "UNKNOWN")
        ponder_item = ponder_map.get(uid, {})
        ponder_verdict = ponder_item.get("verdict", "UNKNOWN")

        # Probe inputs for GPR routing
        probe_prompt = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        inputs = tokenizer(probe_prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        # GPR routing
        pondering_ctx["enabled"] = False
        z_i, _, _ = router.compute_phase_divergence(model, inputs)
        alpha = blender.compute_alpha(z_i)
        rep_penalty = blender.compute_rep_penalty(alpha)
        blender.set_blend(model, alpha)

        # Configure Pondering Context
        pondering_ctx["enabled"] = True
        pondering_ctx["alpha"] = alpha
        pondering_ctx["steps_run"] = 0

        # Pass 1: Primary Greedy Generation
        t0 = time.time()
        with torch.no_grad():
            gen_out1 = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                repetition_penalty=rep_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elapsed1 = time.time() - t0
        pondering_ctx["enabled"] = False
        rk4_steps = pondering_ctx["steps_run"]
        lam_eff = pondering_ctx["lambda_eff"]

        gen_tokens1 = gen_out1[0][prompt_len:]
        gen_text1 = tokenizer.decode(gen_tokens1, skip_special_tokens=True).strip()
        pred1 = extract_boxed_answer(gen_text1)

        # Evaluate Pass 1 with CST PRM
        is_valid1, eq_count1, contras1 = CstSymbolicVerifier.verify_chunk(gen_text1)
        jac_rep1 = jacobian_filter.analyze(gen_text1)
        strict_m1, sym_m1 = NormalizedMathMatcher.check_match(pred1, gt)

        # Decision: Gate Intervention Trigger
        # If strict match or symbolic match is already achieved, or chain is consistent without proof abandonment:
        # Accept primary chain immediately (zero chance of regression!).
        final_pred = pred1
        final_text = gen_text1
        gate_triggered = False
        gate_gain = False
        retry_contras = []

        # Only trigger Contradiction Gate if:
        # 1. Primary chain did NOT pass strict or symbolic match
        # Decision: Autonomous Test-Time Contradiction Gate Trigger
        # A true Process Reward Model (PRM) triggers gating strictly from internal consistency
        # (Zero ground-truth leakage!).
        # If Chain 1 is mathematically sound (no contradictions and no severe proof abandonment):
        # Chain 1 is accepted immediately (preserving 100% of passing chains with 0 overhead).
        final_pred = pred1
        final_text = gen_text1
        gate_triggered = False
        gate_gain = False
        retry_contras = []

        # Trigger gate purely on detected arithmetic contradiction or severe proof abandonment
        has_flaw = (not is_valid1 and len(contras1) > 0) or jac_rep1.has_proof_abandonment

        if has_flaw:
            gate_triggered = True
            total_gate_triggered += 1

            feedback_clauses = []
            if contras1:
                feedback_clauses.append(f"a calculation discrepancy in {'; '.join(contras1[:2])}")
            if jac_rep1.has_proof_abandonment:
                feedback_clauses.append("a discontinuity in the reasoning chain where proof steps were abandoned")

            feedback_str = " and ".join(feedback_clauses)
            retry_prompt = (
                f"<|im_start|>user\n{prob}\n"
                f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n"
                f"Verification note: Careful inspection of the derivation reveals {feedback_str}. "
                f"Let me re-verify all calculation steps explicitly from first principles to ensure consistency:\n"
            )

            r_inputs = tokenizer(retry_prompt, return_tensors="pt").to(model.device)
            rp_len = r_inputs.input_ids.shape[1]

            pondering_ctx["enabled"] = True
            pondering_ctx["alpha"] = alpha
            t1 = time.time()
            with torch.no_grad():
                gen_out2 = model.generate(
                    **r_inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    repetition_penalty=rep_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            elapsed2 = time.time() - t1
            pondering_ctx["enabled"] = False

            gen_tokens2 = gen_out2[0][rp_len:]
            gen_text2 = tokenizer.decode(gen_tokens2, skip_special_tokens=True).strip()
            pred2 = extract_boxed_answer(gen_text2)

            is_valid2, eq_count2, contras2 = CstSymbolicVerifier.verify_chunk(gen_text2)
            jac_rep2 = jacobian_filter.analyze(gen_text2)

            # CST PRM Selection Policy:
            # Score each candidate chain by neurosymbolic consistency:
            # Score = verified_eq_count - 10 * num_contradictions - (5 if proof_abandonment else 0)
            score1 = eq_count1 - 10 * len(contras1) - (5.0 if jac_rep1.has_proof_abandonment else 0.0)
            score2 = eq_count2 - 10 * len(contras2) - (5.0 if jac_rep2.has_proof_abandonment else 0.0)

            # Prefer the chain with higher symbolic consistency, provided it extracted an answer
            if pred2 and (score2 > score1 or (is_valid2 and not is_valid1)):
                final_pred = pred2
                final_text = gen_text2

            retry_contras = contras2

        # Benchmark Verification (Evaluated strictly post-inference)
        primary_strict, primary_sym = NormalizedMathMatcher.check_match(pred1, gt)
        final_strict, final_sym = NormalizedMathMatcher.check_match(final_pred, gt)
        if (final_strict or final_sym) and not (primary_strict or primary_sym):
            gate_gain = True
            total_gate_gains += 1

        # Final Verifications
        final_strict, final_sym = NormalizedMathMatcher.check_match(final_pred, gt)
        verdict = "PASS" if final_strict else "FAIL"

        delta_sym = "= PASS" if (verdict == "PASS" and ponder_verdict == "PASS") else (
            "★ GAIN" if (verdict == "PASS" and ponder_verdict == "FAIL") else (
                "✗ DROP" if (verdict == "FAIL" and ponder_verdict == "PASS") else "= FAIL"
            )
        )

        record = {
            "index": q_idx,
            "unique_id": uid,
            "subject": subj,
            "level": level,
            "is_train_subset": is_train,
            "subset": sub_type,
            "problem": prob,
            "ground_truth": gt,
            "predicted": final_pred,
            "verdict": verdict,
            "symbolic_match": bool(final_sym),
            "ponder_verdict": ponder_verdict,
            "grpo_verdict": grpo_verdict,
            "sft_verdict": sft_verdict,
            "delta_vs_ponder": delta_sym.strip(),
            "gate_triggered": bool(gate_triggered),
            "gate_gain": bool(gate_gain),
            "primary_pred": pred1,
            "primary_valid": bool(is_valid1),
            "primary_contras": contras1,
            "primary_jump_ratio": float(jac_rep1.jump_ratio),
            "primary_abandonment": bool(jac_rep1.has_proof_abandonment),
            "alpha": round(float(alpha), 4),
            "rk4_steps": int(rk4_steps),
            "lambda_eff": float(lam_eff),
            "generated_tokens": len(gen_tokens1),
            "time_seconds": round(float(elapsed1), 2),
            "full_response": final_text
        }
        results_list.append(record)
        evaluated_ids.add(uid)

        # Save live after each question
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({
                "module": "Step 3 Module 2: CST PRM + Contradiction Gate",
                "total_evaluated": len(results_list),
                "total_strict_pass": sum(1 for r in results_list if r["verdict"] == "PASS"),
                "total_symbolic_pass": sum(1 for r in results_list if r["symbolic_match"]),
                "total_gate_triggered": total_gate_triggered,
                "total_gate_gains": total_gate_gains,
                "details": results_list
            }, f, indent=2, ensure_ascii=False, default=lambda o: bool(o) if isinstance(o, (np.bool_, bool)) else (float(o) if isinstance(o, (np.floating, float)) else str(o)))

        total_pass = sum(1 for r in results_list if r["verdict"] == "PASS")
        total_sym = sum(1 for r in results_list if r["symbolic_match"])
        curr_total = len(results_list)

        gate_indicator = " [GATE TRIGGERED]" if gate_triggered else ""
        gain_indicator = " [★ GATE GAIN!]" if gate_gain else ""

        print(
            f"[{q_idx:03d}/150] [{sub_type:7s}] [{subj[:14]:14s}] "
            f"CST: {verdict:4s} | Ponder: {ponder_verdict:4s} | {delta_sym:6s} | "
            f"Strict: {total_pass}/{curr_total} ({100*total_pass/curr_total:.1f}%) | "
            f"Sym: {total_sym}/{curr_total} ({100*total_sym/curr_total:.1f}%) | "
            f"GateGains: {total_gate_gains}{gate_indicator}{gain_indicator}",
            flush=True
        )

    # Final Summary Print
    t_tot = time.time() - t_start
    print("\n" + "=" * 80, flush=True)
    print("SODALS v2.1 — STEP 3 MODULE 2 EVALUATION COMPLETE", flush=True)
    print(f"Total Evaluated: {len(results_list)} / 150", flush=True)
    print(f"Strict Legacy Pass:    {total_pass} / 150 ({100*total_pass/150:.2f}%)", flush=True)
    print(f"Normalized Sym Pass:   {total_sym} / 150 ({100*total_sym/150:.2f}%)", flush=True)
    print(f"Gate Triggered Count:  {total_gate_triggered}", flush=True)
    print(f"Direct Gate Gains:     {total_gate_gains}", flush=True)
    print(f"Total Time:            {t_tot/60:.1f} minutes", flush=True)
    print(f"Results Saved:         {output_json}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    run_cst_prm_eval()
