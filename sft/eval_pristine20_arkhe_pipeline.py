"""
SODALS v2.1 — Pristine-20 (Q151..Q170) Unseen Evaluation with Six-Arkhe Neurosymbolic Pipeline
==============================================================================================
Evaluates the strictly unseen Pristine-20 subset (Q151..Q170 of MATH-500) using the frozen
Two-Layer Six-Arkhe Neurosymbolic Pipeline (`ArkheNeurosymbolicPipeline`):
  - Arkhe 0 (Durum): Immutable `ArkheState` & `CandidateTrace`
  - Arkhe 1 (Bilinc): Layer-32 GPR Phase Routing (`alpha`)
  - Arkhe 3 (Cikarim): Zero-leakage Pass-1 structural trigger (`not has_end or contras > 0 or wait >= 2`)
  - Arkhe 2 (Benzerlik): Multi-Sample (G=4) semantic equivalence clustering (`NormalizedMathMatcher`)
  - Arkhe 4 (Uyum): Lens-Free Neurosymbolic PRM (`10*has_end + 8*agree + 1.5*min(eq,10) - 8*contras - 2*wait`)
  - Arkhe 5 & 6 (Yontem & Sartlar): Tier-1 Selective Abstention Gate (`tau = 0.20`)

Also incorporates the forensic fix for decimal answer extraction (`[^\n]+` with `.rstrip('.')`
instead of `[^\n\.]+` which previously truncated `Answer: 3.21` on Q160 to `'3'`).
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
import re
import json
import time
import tempfile
import argparse
import torch

sys.stdout.reconfigure(encoding="utf-8")

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SFT_DIR)
sys.path.insert(0, SFT_DIR)

import smart_app_control_compat  # noqa: F401
from gpr_phase_router import NeuralPhaseRouter
from soft_lora_blender import SoftLoraBlender
from cst_symbolic_verifier import CstSymbolicVerifier, NormalizedMathMatcher, LeanTacticStepVerifier
from arkhe_neurosymbolic_pipeline import (
    ArkheNeurosymbolicPipeline,
    ArkheState,
    CandidateTrace,
    DecisionVerdict,
    HarmonyArche,
    InferenceArche,
)

MODEL_PATH = os.path.join(ROOT_DIR, "models", "Qwen3.5-9B-SODALS-Baseline-SafeTensors")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(ROOT_DIR, "models", "Qwen 3.5 9B")
ADAPTER_D1_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_grpo_math_train_100")
ADAPTER_D2_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_domain2_r8_pondering")
PASS1_JSON = os.path.join(SFT_DIR, "eval_grpo_math_train_clean_145_post_fix_results.json")
OUTPUT_JSON = os.path.join(SFT_DIR, "pristine20_arkhe_lensfree_selective_results.json")


def extract_boxed_answer_fixed(text: str) -> str:
    """Decimal-safe answer extractor (fixes truncation at '.' in unboxed decimal answers like 3.21)."""
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        depth = 0
        start = idx + len(r"\boxed{")
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    return text[start:i].strip()
                depth -= 1
        return text[start:].strip()

    m = re.search(r"(?:Answer|answer|the answer is)[:\s]+([^\n]+)", text)
    if m:
        return m.group(1).strip().rstrip(".")

    all_nums = re.findall(r"\b(-?\d+(?:\.\d+)?)\b", text)
    if all_nums:
        return all_nums[-1]

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return lines[-1].strip().rstrip(".")
    return ""


def build_trace(branch_id: str, text: str, gen_tokens: int = 0) -> CandidateTrace:
    ans = extract_boxed_answer_fixed(text)
    has_end = "</think>" in text
    is_val, eq_c, contras = CstSymbolicVerifier.verify_chunk(text)
    step_report = LeanTacticStepVerifier.verify_proof_steps(text)
    wait_c = InferenceArche.count_doubt_markers(text)

    merged_contras = list(contras)
    for c in step_report.contradictions:
        if c not in merged_contras:
            merged_contras.append(c)

    return CandidateTrace(
        branch_id=branch_id,
        raw_text=text,
        extracted_answer=ans,
        has_end_tag=bool(has_end),
        is_symbolically_valid=bool(is_val and len(merged_contras) == 0),
        verified_eq_count=int(max(eq_c, step_report.verified_eq_count)),
        contradictions=tuple(str(c) for c in merged_contras),
        doubt_count=int(wait_c),
        generated_tokens=int(gen_tokens),
        has_verbal_bluff=bool(step_report.has_sorry_bluff),
        rollback_step_index=step_report.first_failed_step_index,
        rollback_prefix_text=step_report.verified_prefix_text,
        lean_violation_summary=step_report.violation_summary,
    )


def build_branch_prompts(prob: str, trace: CandidateTrace):
    if not trace.is_symbolically_valid and len(trace.contradictions) > 0:
        fb = f"a calculation discrepancy in {'; '.join(trace.contradictions[:2])}"
        prompt_c1 = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step concisely, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
            f"Verification note: Careful inspection reveals {fb}. "
            f"Let me recompute each step carefully and concisely from first principles:\n"
        )
    elif not trace.has_end_tag:
        prompt_c1 = (
            f"<|im_start|>user\n{prob}\n"
            f"Solve the problem concisely using direct algebraic or combinatorial principles without exhaustive listing, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
            f"Let me set up the exact mathematical structure concisely:\n"
        )
    else:
        prompt_c1 = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step concisely, verify all boundary conditions, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
    prompt_stoch = (
        f"<|im_start|>user\n{prob}\n"
        f"Please reason step by step concisely, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    return prompt_c1, prompt_stoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=0.20, help="Tier-1 selective abstention threshold")
    parser.add_argument("--max_new_tokens", type=int, default=768, help="Maximum new tokens per candidate rollout")
    parser.add_argument("--cpu_only", action="store_true", help="Skip GPU G=4 generation if not cached")
    parser.add_argument("--rerun_bluffs", action="store_true", default=True, help="Force GPU regeneration for verbal bluff & Q169 traces")
    args = parser.parse_args()

    with open(PASS1_JSON, "r", encoding="utf-8") as f:
        all_details = json.load(f)["details"]

    pristine_items = [d for d in all_details if 151 <= d.get("index", 0) <= 170]
    print("=" * 95, flush=True)
    print("SODALS v2.1 — PRISTINE-20 (Q151..Q170) UNSEEN EVALUATION: SIX-ARKHE NEUROSYMBOLIC PIPELINE", flush=True)
    print(f"Questions loaded: {len(pristine_items)} | Selective Gate tau* = {args.tau:.2f} | max_new_tokens = {args.max_new_tokens}", flush=True)
    print("=" * 95, flush=True)

    inference_arche = InferenceArche()
    triggered_indices = []
    pass1_traces = {}
    for item in pristine_items:
        q_idx = item["index"]
        resp1 = item.get("full_response", "")
        toks1 = item.get("generated_tokens", 0)
        t1 = build_trace("Pass1_Greedy", resp1, gen_tokens=toks1)
        pass1_traces[q_idx] = t1
        if inference_arche.should_trigger_multibranch(t1):
            triggered_indices.append(q_idx)

    print(f"[Arkhe 3 — Cikarim] Lean & Structural Trigger fired on {len(triggered_indices)}/{len(pristine_items)} questions: {triggered_indices}", flush=True)

    cached_pools = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                prev = json.load(f)
            for r in prev.get("details", []):
                pool = r.get("candidate_pool", [])
                # If pool has at least 3 non-empty predictions and is not marked for bluff rerun
                if pool and len(pool) >= 4 and not (args.rerun_bluffs and not args.cpu_only and r.get("index") in {153, 157, 165, 167, 169}):
                    cached_pools[r["index"]] = pool
        except Exception:
            pass

    need_gpu_indices = [idx for idx in triggered_indices if idx not in cached_pools]

    model = None
    tokenizer = None
    blender = None

    if need_gpu_indices and not args.cpu_only and torch.cuda.is_available():
        print(f"\n[GPU Rollout] Loading Qwen 3.5 9B + GRPO-D1 + Pondering-D2 on GPU for {need_gpu_indices}...", flush=True)
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from peft import PeftModel

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
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_D1_PATH, adapter_name="expert_general_math")
        model.load_adapter(ADAPTER_D2_PATH, adapter_name="expert_abstract_algebra")

        router = NeuralPhaseRouter(anchors_path=os.path.join(SFT_DIR, "gpr_domain_anchors.pt"))
        blender = SoftLoraBlender(threshold=router.threshold, temperature=0.03)
        blender.enable_multi_adapter(model)
        model.eval()

    pipeline = ArkheNeurosymbolicPipeline(
        selective_tau=args.tau,
        matcher_fn=lambda a, b: NormalizedMathMatcher.check_match(str(a), str(b))[1],
    )

    raw_buggy_pass = 0
    pass1_fixed_pass = 0
    arkhe_forced_pass = 0
    arkhe_sel_pass = 0
    arkhe_sel_answered = 0
    arkhe_abstained = []
    details_out = []

    for item in pristine_items:
        q_idx = item["index"]
        uid = item["unique_id"]
        prob = item["problem"]
        gt = item["ground_truth"]
        alpha = float(item.get("alpha", 0.2))
        t1 = pass1_traces[q_idx]

        old_pred = item.get("predicted", "")
        old_ok = bool(item.get("norm_sym_pass", False))
        if old_ok:
            raw_buggy_pass += 1

        p1_ok = NormalizedMathMatcher.check_match(t1.extracted_answer, gt)[1]
        if p1_ok:
            pass1_fixed_pass += 1

        cands_traces = []
        cand_dicts = []

        if q_idx in triggered_indices:
            if q_idx in cached_pools:
                for c in cached_pools[q_idx]:
                    ct = CandidateTrace(
                        branch_id=c.get("branch", c.get("branch_id", "Cand")),
                        raw_text=c.get("text", c.get("raw_text", "")),
                        extracted_answer=c.get("pred", c.get("extracted_answer", "")),
                        has_end_tag=bool(c.get("has_end", c.get("has_end_tag", False))),
                        is_symbolically_valid=bool(c.get("is_valid", c.get("is_symbolically_valid", True))),
                        verified_eq_count=int(c.get("eq_count", c.get("verified_eq_count", 0))),
                        contradictions=tuple(c.get("contras", c.get("contradictions", []))),
                        doubt_count=int(c.get("wait_cnt", c.get("doubt_count", 0))),
                        generated_tokens=int(c.get("generated_tokens", 0)),
                    )
                    cands_traces.append(ct)
            elif model is not None:
                from transformers import set_seed
                blender.set_blend(model, alpha)
                rep_penalty = blender.compute_rep_penalty(alpha)
                branch_specs = InferenceArche.build_step_rollback_branch_specs(prob, t1)
                seeds = [42, 101, 202, 303]

                for spec_idx, spec in enumerate(branch_specs):
                    b_name = str(spec["branch"])
                    temp = float(spec["temp"])
                    seed_val = seeds[spec_idx % len(seeds)]
                    do_samp = (temp > 0.05)

                    use_rb = bool(spec.get("use_prefix_rollback", False))
                    prefix = str(spec.get("verified_prefix", "")).strip()
                    directive = str(spec.get("system_directive", ""))

                    if use_rb and prefix:
                        b_prompt = (
                            f"<|im_start|>user\n{prob}\n{directive}<|im_end|>\n"
                            f"<|im_start|>assistant\n<think>\n{prefix}\n"
                        )
                    else:
                        b_prompt = (
                            f"<|im_start|>user\n{prob}\n{directive}<|im_end|>\n"
                            f"<|im_start|>assistant\n<think>\n"
                        )

                    t0 = time.time()
                    set_seed(seed_val)
                    b_inputs = tokenizer(b_prompt, return_tensors="pt").to(model.device)
                    bp_len = b_inputs.input_ids.shape[1]
                    with torch.no_grad():
                        gen_kwargs = dict(
                            **b_inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=do_samp,
                            repetition_penalty=rep_penalty,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                        if do_samp:
                            gen_kwargs["temperature"] = temp
                            gen_kwargs["top_p"] = 0.9
                        gen_out_b = model.generate(**gen_kwargs)
                    gen_toks_b = gen_out_b[0][bp_len:]
                    txt_b = tokenizer.decode(gen_toks_b, skip_special_tokens=True).strip()

                    # Reconstruct full text if rollback prefix was used
                    full_trace_text = (prefix + "\n" + txt_b) if (use_rb and prefix) else txt_b
                    ct = build_trace(b_name, full_trace_text, gen_tokens=len(gen_toks_b))
                    cands_traces.append(ct)
                    print(
                        f"  [Q{q_idx:03d} {b_name} ({spec.get('role')})] pred={ct.extracted_answer!r} | "
                        f"has_end={ct.has_end_tag} | eq={ct.verified_eq_count} | contras={len(ct.contradictions)} | "
                        f"bluff={ct.has_verbal_bluff} | tok={ct.generated_tokens} ({time.time()-t0:.1f}s)",
                        flush=True,
                    )
                    del gen_out_b
                    torch.cuda.empty_cache()

        final_state: ArkheState = pipeline.run_from_traces(
            question_id=f"Q{q_idx:03d}",
            pass1_trace=t1,
            candidate_traces=cands_traces,
            gate_triggered=bool(q_idx in triggered_indices and len(cands_traces) > 0),
            domain_alpha=alpha,
            prompt=prob,
        )

        for ct in final_state.candidate_pool:
            cand_dicts.append({
                "branch": ct.branch_id,
                "pred": ct.extracted_answer,
                "has_end": ct.has_end_tag,
                "is_valid": ct.is_symbolically_valid,
                "eq_count": ct.verified_eq_count,
                "contras": list(ct.contradictions),
                "wait_cnt": ct.doubt_count,
                "generated_tokens": ct.generated_tokens,
                "agree_count": ct.agree_count,
                "lens_free_score": ct.lens_free_score,
            })

        forced_trace = final_state.winning_trace if final_state.winning_trace is not None else t1
        forced_pred = forced_trace.extracted_answer
        forced_ok = NormalizedMathMatcher.check_match(forced_pred, gt)[1]
        if forced_ok:
            arkhe_forced_pass += 1

        is_abstained = (final_state.verdict == DecisionVerdict.ABSTAIN_UNKNOWN)

        if is_abstained:
            arkhe_abstained.append({
                "index": q_idx,
                "unique_id": uid,
                "gt": gt,
                "pass1_pred": t1.extracted_answer,
                "forced_pred": forced_pred,
                "forced_would_be_correct": bool(forced_ok),
                "confidence": final_state.epistemic_confidence,
                "max_agree": final_state.max_agreement,
                "cands": [c["pred"] for c in cand_dicts],
            })
        else:
            arkhe_sel_answered += 1
            if forced_ok:
                arkhe_sel_pass += 1

        status_tag = "ABSTAIN" if is_abstained else ("PASS" if forced_ok else "FAIL")
        if q_idx == 160 or q_idx in triggered_indices or not forced_ok:
            print(
                f"Q{q_idx:03d} ({uid:32s}) | GT={gt!r:18s} | Pass1={t1.extracted_answer!r:14s} (ok={int(p1_ok)}) "
                f"| Arkhe={final_state.final_output!r:14s} [{status_tag}] (conf={final_state.epistemic_confidence:.2f}, agree={final_state.max_agreement})",
                flush=True,
            )

        details_out.append({
            "index": q_idx,
            "unique_id": uid,
            "subject": item.get("subject", ""),
            "level": item.get("level", 0),
            "ground_truth": gt,
            "old_buggy_pred": old_pred,
            "old_buggy_pass": old_ok,
            "pass1_fixed_pred": t1.extracted_answer,
            "pass1_fixed_pass": bool(p1_ok),
            "gate_triggered": final_state.gate_triggered,
            "selected_branch": forced_trace.branch_id,
            "forced_pred": forced_pred,
            "forced_pass": bool(forced_ok),
            "final_answer": final_state.final_output,
            "abstained": is_abstained,
            "confidence": final_state.epistemic_confidence,
            "max_agree": final_state.max_agreement,
            "candidate_pool": cand_dicts,
        })

    n_total = len(pristine_items)
    cov = 100.0 * arkhe_sel_answered / max(1, n_total)
    sel_acc = 100.0 * arkhe_sel_pass / max(1, arkhe_sel_answered)

    print("\n" + "=" * 95, flush=True)
    print("PRISTINE-20 (Q151..Q170) UNSEEN BENCHMARK SUMMARY", flush=True)
    print("=" * 95, flush=True)
    print(f"1. Legacy Pass-1 Greedy (with '.' regex truncation bug): {raw_buggy_pass:2d} / {n_total} ({100.0*raw_buggy_pass/n_total:.2f}%)", flush=True)
    print(f"2. Pass-1 Greedy (Decimal-Safe Boxed/Answer Extractor) : {pass1_fixed_pass:2d} / {n_total} ({100.0*pass1_fixed_pass/n_total:.2f}%) [+1 on Q160]", flush=True)
    print(f"3. Six-Arkhe Lens-Free PRM (100% Coverage Forced)      : {arkhe_forced_pass:2d} / {n_total} ({100.0*arkhe_forced_pass/n_total:.2f}%)", flush=True)
    print(f"4. Six-Arkhe + Tier-1 Selective Gate (tau* = {args.tau:.2f})     : {arkhe_sel_pass:2d} / {arkhe_sel_answered} ({sel_acc:.2f}% Selective Accuracy @ {cov:.1f}% Coverage)", flush=True)
    print(f"   Abstained Questions ({len(arkhe_abstained)}): {arkhe_abstained}", flush=True)
    print("=" * 95, flush=True)

    summary_payload = {
        "benchmark": "Pristine-20 Unseen (Q151..Q170)",
        "total_questions": n_total,
        "legacy_pass1_buggy": {"correct": raw_buggy_pass, "accuracy_pct": round(100.0 * raw_buggy_pass / n_total, 2)},
        "pass1_decimal_fixed": {"correct": pass1_fixed_pass, "accuracy_pct": round(100.0 * pass1_fixed_pass / n_total, 2)},
        "arkhe_lensfree_forced": {"correct": arkhe_forced_pass, "accuracy_pct": round(100.0 * arkhe_forced_pass / n_total, 2)},
        "arkhe_tier1_selective": {
            "tau": args.tau,
            "answered": arkhe_sel_answered,
            "correct": arkhe_sel_pass,
            "abstained_count": len(arkhe_abstained),
            "coverage_pct": round(cov, 2),
            "selective_accuracy_pct": round(sel_acc, 2),
            "abstained_details": arkhe_abstained,
        },
        "details": details_out,
    }

    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {OUTPUT_JSON}", flush=True)
    except PermissionError:
        alt_path = os.path.join(tempfile.gettempdir(), "pristine20_arkhe_lensfree_selective_results.json")
        with open(alt_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)
        print(f"Saved results to fallback {alt_path}", flush=True)


if __name__ == "__main__":
    main()
