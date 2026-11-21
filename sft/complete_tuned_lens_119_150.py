"""
Fast & Exact GPU Completion of Step 3 Module 3 (Questions 119..150)
===================================================================
Since Pass 1 (greedy T=0 with Adaptive Pondering ODE) is 100% deterministic and
verified bit-for-bit identical (118/118) to `sft/eval_grpo_long_pondering_150_results.json`,
and Pass 2 (greedy T=0 retry on arithmetic contradiction / proof abandonment when
mean_stability >= 0.65) is verified bit-for-bit identical (5/5) to
`sft/eval_grpo_long_cst_prm_150_results.json`, this script:
  1. Verifies on Q112, Q117, Q118 that reconstructing `gen_out1` from
     `probe_prompt + ponder_response + eos_token` yields the exact same
     GPU TunedLensWrapper metrics (`mean_stability`, `answer_stability`, `top1_agreement`).
  2. Evaluates Q119..Q150 on the GPU using `TunedLensWrapper`, `CstSymbolicVerifier`,
     and `ReasoningChainJacobianFilter`, computing exact Tri-Modal PRM scores
     (`score1` vs `score2`) and falling back to full `model.generate` if any new
     stability trigger (`mean_stability < 0.65`) occurs.
"""

import os
import sys
import json
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_D1_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_grpo_long")
ADAPTER_D2_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_domain2_r8_pondering")
PONDERING_WEIGHTS = os.path.join(ROOT, "sft", "qwen_lora_gpr_soft_moa", "gpr_soft_moa_aux_modules.pt")
JLENS_PATH = os.path.join(ROOT, "jlens_weights.safetensors")

BASELINE_SFT_JSON = os.path.join(ROOT, "math500_domain2_soft_blended_170_results.json")
BASELINE_GRPO_JSON = os.path.join(ROOT, "sft", "eval_grpo_long_math500_150_results.json")
BASELINE_PONDER_JSON = os.path.join(ROOT, "sft", "eval_grpo_long_pondering_150_results.json")
BASELINE_CST_JSON = os.path.join(ROOT, "sft", "eval_grpo_long_cst_prm_150_results.json")
OUTPUT_JSON = os.path.join(ROOT, "sft", "eval_grpo_long_tuned_lens_150_results.json")

sys.path.insert(0, os.path.join(ROOT, "sft"))
from gpr_phase_router import NeuralPhaseRouter
from soft_lora_blender import SoftLoraBlender
from adaptive_pondering_ode import AdaptivePonderingODEModule
from cst_symbolic_verifier import CstSymbolicVerifier, NormalizedMathMatcher
from reasoning_chain_jacobian import ReasoningChainJacobianFilter
from tuned_lens_wrapper import TunedLensWrapper
from eval_grpo_long_tuned_lens_150 import extract_boxed_answer


def build_seq_tensor(tokenizer, prompt_text: str, response_text: str, saved_gen_len: int, device) -> torch.Tensor:
    p_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    r_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if saved_gen_len < 2048:
        full_ids = p_ids + r_ids + [tokenizer.eos_token_id]
    else:
        full_ids = p_ids + r_ids[:2048]
    return torch.tensor([full_ids], dtype=torch.long, device=device)


def main():
    with open(BASELINE_SFT_JSON, "r", encoding="utf-8") as f:
        sft_map = {item["unique_id"]: item for item in json.load(f)["details"][:150]}
    with open(BASELINE_GRPO_JSON, "r", encoding="utf-8") as f:
        grpo_map = {item["unique_id"]: item for item in json.load(f)["details"][:150]}
    with open(BASELINE_PONDER_JSON, "r", encoding="utf-8") as f:
        ponder_list = json.load(f)["details"][:150]
        ponder_map = {item["unique_id"]: item for item in ponder_list}
    with open(BASELINE_CST_JSON, "r", encoding="utf-8") as f:
        cst_map = {item["unique_id"]: item for item in json.load(f)["details"][:150]}
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        saved_lens = json.load(f)
        results_list = saved_lens.get("details", [])
        lens_by_idx = {item["index"]: item for item in results_list}
        evaluated_ids = {item["unique_id"] for item in results_list}

    print(f"Loaded {len(results_list)}/150 existing Module 3 records.")

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
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_D1_PATH, adapter_name="expert_general_math")
    model.load_adapter(ADAPTER_D2_PATH, adapter_name="expert_abstract_algebra")

    router = NeuralPhaseRouter(anchors_path=os.path.join(ROOT, "sft", "gpr_domain_anchors.pt"))
    blender = SoftLoraBlender(threshold=router.threshold, temperature=0.03)
    blender.enable_multi_adapter(model)

    jacobian_filter = ReasoningChainJacobianFilter(jump_threshold=3.5)
    tuned_lens = TunedLensWrapper(JLENS_PATH, layers=(28, 30), device="cuda", dtype=torch.float16)
    model.eval()

    # Step 1: Verify exact parity on Q112, Q117, Q118
    print("\n--- Verifying GPU TunedLens parity on Q112, Q117, Q118 ---", flush=True)
    for test_idx in [112, 117, 118]:
        p_item = ponder_list[test_idx - 1]
        l_item = lens_by_idx[test_idx]
        prob = p_item["problem"]
        probe_prompt = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        blender.set_blend(model, p_item["alpha"])
        seq_t = build_seq_tensor(tokenizer, probe_prompt, p_item["full_response"], p_item["generated_tokens"], model.device)
        rep = tuned_lens.evaluate_sequence(model, tokenizer, seq_t)
        print(
            f"Q{test_idx}: saved_stab={l_item['primary_mean_stability']:.4f} vs recomputed={rep.mean_stability:.4f} | "
            f"saved_ans_stab={l_item['primary_answer_stability']:.4f} vs recomputed={rep.answer_stability:.4f} | "
            f"saved_top1={l_item['primary_top1_agreement']} vs recomputed={rep.top1_agreement}",
            flush=True,
        )

    # Step 2: Evaluate Q119..Q150
    print("\n--- Evaluating Q119..Q150 with GPU TunedLens & Tri-Modal PRM ---", flush=True)
    total_gate_triggered = sum(1 for r in results_list if r.get("gate_triggered", False))
    total_gate_gains = sum(1 for r in results_list if r.get("gate_gain", False))

    for p_item in ponder_list:
        uid = p_item["unique_id"]
        if uid in evaluated_ids:
            continue

        q_idx = p_item["index"]
        prob = p_item["problem"]
        gt = p_item["ground_truth"]
        subj = p_item["subject"]
        level = p_item["level"]
        is_train = p_item["is_train_subset"]
        sub_type = "TRAIN-25" if is_train else "HELDOUT"

        sft_verdict = sft_map.get(uid, {}).get("verdict", "UNKNOWN")
        grpo_verdict = grpo_map.get(uid, {}).get("verdict", "UNKNOWN")
        ponder_verdict = p_item.get("verdict", "UNKNOWN")
        cst_item = cst_map.get(uid, {})
        cst_verdict = cst_item.get("verdict", "UNKNOWN")

        alpha = p_item["alpha"]
        rk4_steps = p_item["rk4_steps"]
        lam_eff = p_item["lambda_eff"]
        elapsed1 = p_item["time_seconds"]
        gen_len1 = p_item["generated_tokens"]
        gen_text1 = p_item["full_response"]
        pred1 = extract_boxed_answer(gen_text1)

        probe_prompt = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        blender.set_blend(model, alpha)
        gen_out1 = build_seq_tensor(tokenizer, probe_prompt, gen_text1, gen_len1, model.device)

        is_valid1, eq_count1, contras1 = CstSymbolicVerifier.verify_chunk(gen_text1)
        jac_rep1 = jacobian_filter.analyze(gen_text1)
        lens_rep1 = tuned_lens.evaluate_sequence(model, tokenizer, gen_out1)

        final_pred = pred1
        final_text = gen_text1
        gate_triggered = False
        gate_gain = False

        has_flaw = (not is_valid1 and len(contras1) > 0) or jac_rep1.has_proof_abandonment or (lens_rep1.mean_stability < 0.65)

        if has_flaw:
            gate_triggered = True
            total_gate_triggered += 1

            feedback_clauses = []
            if contras1:
                feedback_clauses.append(f"a calculation discrepancy in {'; '.join(contras1[:2])}")
            if jac_rep1.has_proof_abandonment:
                feedback_clauses.append("a discontinuity in the reasoning chain where proof steps were abandoned")
            if lens_rep1.mean_stability < 0.65:
                feedback_clauses.append("a semantic drift in intermediate reasoning steps")

            feedback_str = " and ".join(feedback_clauses)
            retry_prompt = (
                f"<|im_start|>user\n{prob}\n"
                f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n"
                f"Verification note: Careful inspection of the derivation reveals {feedback_str}. "
                f"Let me re-verify all calculation steps explicitly from first principles to ensure consistency:\n"
            )

            # If mean_stability >= 0.65 and CST already executed the exact same retry prompt and stored Pass 2:
            if lens_rep1.mean_stability >= 0.65 and cst_item.get("gate_triggered", False) and cst_item.get("full_response") != gen_text1:
                gen_text2 = cst_item["full_response"]
                r_ids = tokenizer.encode(gen_text2, add_special_tokens=False)
                saved_len2 = len(r_ids) + (1 if len(r_ids) < 2048 else 0)
                gen_out2 = build_seq_tensor(tokenizer, retry_prompt, gen_text2, saved_len2, model.device)
            else:
                r_inputs = tokenizer(retry_prompt, return_tensors="pt").to(model.device)
                rp_len = r_inputs.input_ids.shape[1]
                rep_penalty = blender.compute_rep_penalty(alpha)
                with torch.no_grad():
                    gen_out2 = model.generate(
                        **r_inputs,
                        max_new_tokens=2048,
                        do_sample=False,
                        repetition_penalty=rep_penalty,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                gen_tokens2 = gen_out2[0][rp_len:]
                gen_text2 = tokenizer.decode(gen_tokens2, skip_special_tokens=True).strip()

            pred2 = extract_boxed_answer(gen_text2)
            is_valid2, eq_count2, contras2 = CstSymbolicVerifier.verify_chunk(gen_text2)
            jac_rep2 = jacobian_filter.analyze(gen_text2)
            lens_rep2 = tuned_lens.evaluate_sequence(model, tokenizer, gen_out2)

            score1 = (
                eq_count1
                - 10 * len(contras1)
                + (5.0 if is_valid1 else 0.0)
                - (5.0 if jac_rep1.has_proof_abandonment else 0.0)
                - 2.0 * max(0.0, jac_rep1.jump_ratio - 2.0)
                + lens_rep1.geometric_score
            )
            score2 = (
                eq_count2
                - 10 * len(contras2)
                + (5.0 if is_valid2 else 0.0)
                - (5.0 if jac_rep2.has_proof_abandonment else 0.0)
                - 2.0 * max(0.0, jac_rep2.jump_ratio - 2.0)
                + lens_rep2.geometric_score
            )
            print(
                f"  -> [GATE Q{q_idx}] pred1={pred1} (score1={score1:.2f}, stab1={lens_rep1.mean_stability:.4f}) | "
                f"pred2={pred2} (score2={score2:.2f}, stab2={lens_rep2.mean_stability:.4f})",
                flush=True,
            )
            if pred2 and (score2 > score1):
                final_pred = pred2
                final_text = gen_text2

        primary_strict, primary_sym = NormalizedMathMatcher.check_match(pred1, gt)
        final_strict, final_sym = NormalizedMathMatcher.check_match(final_pred, gt)
        if (final_strict or final_sym) and not (primary_strict or primary_sym):
            gate_gain = True
            total_gate_gains += 1

        verdict = "PASS" if final_strict else "FAIL"
        delta_sym = "= PASS" if (verdict == "PASS" and cst_verdict == "PASS") else (
            "★ GAIN" if (verdict == "PASS" and cst_verdict == "FAIL") else (
                "✗ DROP" if (verdict == "FAIL" and cst_verdict == "PASS") else "= FAIL"
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
            "cst_verdict": cst_verdict,
            "ponder_verdict": ponder_verdict,
            "grpo_verdict": grpo_verdict,
            "sft_verdict": sft_verdict,
            "delta_vs_cst": delta_sym.strip(),
            "gate_triggered": bool(gate_triggered),
            "gate_gain": bool(gate_gain),
            "primary_pred": pred1,
            "primary_valid": bool(is_valid1),
            "primary_contras": contras1,
            "primary_jump_ratio": float(jac_rep1.jump_ratio),
            "primary_abandonment": bool(jac_rep1.has_proof_abandonment),
            "primary_mean_stability": float(lens_rep1.mean_stability),
            "primary_answer_stability": float(lens_rep1.answer_stability),
            "primary_top1_agreement": bool(lens_rep1.top1_agreement),
            "alpha": round(float(alpha), 4),
            "rk4_steps": int(rk4_steps),
            "lambda_eff": float(lam_eff),
            "generated_tokens": int(gen_len1),
            "time_seconds": round(float(elapsed1), 2),
            "full_response": final_text,
        }
        results_list.append(record)
        evaluated_ids.add(uid)

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "module": "Step 3 Module 3: Tuned Lens (J-Lens) & Geometric Stability Gate",
                    "total_evaluated": len(results_list),
                    "total_strict_pass": sum(1 for r in results_list if r["verdict"] == "PASS"),
                    "total_symbolic_pass": sum(1 for r in results_list if r["symbolic_match"]),
                    "total_gate_triggered": total_gate_triggered,
                    "total_gate_gains": total_gate_gains,
                    "details": results_list,
                },
                f,
                indent=2,
                ensure_ascii=False,
                default=lambda o: bool(o)
                if isinstance(o, (np.bool_, bool))
                else (float(o) if isinstance(o, (np.floating, float)) else str(o)),
            )

        total_pass = sum(1 for r in results_list if r["verdict"] == "PASS")
        total_sym = sum(1 for r in results_list if r["symbolic_match"])
        curr_total = len(results_list)
        gate_indicator = " [GATE TRIGGERED]" if gate_triggered else ""
        gain_indicator = " [★ GATE GAIN!]" if gate_gain else ""
        print(
            f"[{q_idx:03d}/150] [{sub_type:7s}] [{subj[:14]:14s}] "
            f"Lens: {verdict:4s} | CST: {cst_verdict:4s} | {delta_sym:6s} | "
            f"Strict: {total_pass}/{curr_total} ({100*total_pass/curr_total:.1f}%) | "
            f"Sym: {total_sym}/{curr_total} ({100*total_sym/curr_total:.1f}%) | "
            f"Stab: {lens_rep1.mean_stability:.3f}{gate_indicator}{gain_indicator}",
            flush=True,
        )

        del gen_out1
        torch.cuda.empty_cache()

    print("\n" + "=" * 80, flush=True)
    print("STEP 3 MODULE 3 EVALUATION COMPLETE (150/150)", flush=True)
    print(f"Strict Legacy Pass:  {total_pass}/150 ({100*total_pass/150:.2f}%)", flush=True)
    print(f"Normalized Sym Pass: {total_sym}/150 ({100*total_sym/150:.2f}%)", flush=True)
    print(f"Gate Triggered:      {total_gate_triggered}", flush=True)
    print(f"Gate Gains:          {total_gate_gains}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
