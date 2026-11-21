"""
SODALS v2.1 — Pristine-50 (Q151..Q200) Always-On G=4 Compute-Matched Control & Logprob Benchmark
==================================================================================================
Executes Claude's requested unbiased control & logprob experiment on the exploratory dev slice
(Q151..Q200, keeping confirmatory Q201..Q500 completely untouched):
  1. Pass-1 Greedy (+ full teacher-forced token/step logprobs across ALL N=50 questions -> unbiased AUROC).
  2. Always-On Unconditional G=4 Standard Rollouts (S1..S4, temp=0.6) across ALL N=50 questions:
       - `Avg@4`: mean single-sample accuracy over 200 rollouts
       - `Majority@4`: semantic equivalence plurality baseline (zero Arkhe re-ranking)
       - `Pass@4`: oracle upper bound (at least 1 correct in S1..S4)
       - `Arkhe_ZeroSteer`: Similarity + Harmony + Conditions over exact same S1..S4 pool (100% compute-matched!)
  3. Arkhe Gate-Triggered Repair (`Arkhe_Full`): runs targeted Step-Rollback / MethodArche branches
     only when `InferenceArche` triggers, reporting Forced & Selective (`tau=0.20`) accuracies + exact McNemar p.
"""

import argparse
from dataclasses import replace as dc_replace
import gc
import hashlib
import json
import math
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SFT_DIR)
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

import smart_app_control_compat  # noqa: F401
import numpy as np
import torch

from arkhe_neurosymbolic_pipeline import (
    ArkheNeurosymbolicPipeline,
    ArkheState,
    CandidateTrace,
    DecisionVerdict,
    InferenceArche,
    SimilarityArche,
)
from cst_symbolic_verifier import (
    CstSymbolicVerifier,
    NormalizedMathMatcher,
    SymbolicStepChecker,
)
from env_reproducibility import collect_runtime_metadata
from eval_pristine30_q171_q200_arkhe_pipeline import (
    ANCHORS_PATH,
    ADAPTER_D1_PATH,
    ADAPTER_D2_PATH,
    BASELINE_JSON,
    CPUEmbeddingWrapper,
    MATH500_JSONL,
    MODEL_PATH,
    build_trace,
    prevent_windows_sleep,
    safe_json_dump,
)
from gpr_phase_router import NeuralPhaseRouter
from soft_lora_blender import SoftLoraBlender

DEFAULT_OUT_JSON = os.path.join(SFT_DIR, "pristine50_always_on_control_results.json")
FALLBACK_OUT_JSON = os.path.join(tempfile.gettempdir(), "pristine50_always_on_control_results.json")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _exact_mcnemar_p(b_wins: int, c_losses: int) -> float:
    n = b_wins + c_losses
    if n == 0:
        return 1.0
    k = min(b_wins, c_losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** (n - 1))
    return min(1.0, float(p))


def _paired_summary(flags_a: List[bool], flags_b: List[bool]) -> Dict[str, Any]:
    gains = sum(1 for a, b in zip(flags_a, flags_b) if (not a) and b)
    losses = sum(1 for a, b in zip(flags_a, flags_b) if a and (not b))
    return {
        "a_correct": sum(flags_a),
        "b_correct": sum(flags_b),
        "gains_a0_b1": gains,
        "losses_a1_b0": losses,
        "net": gains - losses,
        "exact_mcnemar_p": round(_exact_mcnemar_p(gains, losses), 4),
    }


def _compute_auroc(scores: List[float], labels: List[int]) -> Optional[float]:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else (0.5 if p == n else 0.0)) for p in pos for n in neg)
    return round(float(wins / (len(pos) * len(neg))), 4)


def _majority_vote_answer(answers: List[str], sim_arche: SimilarityArche) -> str:
    """Pure semantic plurality vote (ties broken deterministically by first cluster appearance)."""
    non_empty = [a for a in answers if (a or "").strip()]
    if not non_empty:
        return answers[0] if answers else ""
    clusters: List[List[str]] = []
    for ans in non_empty:
        placed = False
        for cl in clusters:
            if sim_arche.are_equivalent(ans, cl[0]):
                cl.append(ans)
                placed = True
                break
        if not placed:
            clusters.append([ans])
    best_cluster = max(clusters, key=len)
    return best_cluster[0]


def _compute_metrics(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(details)
    if n == 0:
        return {}
    base_f = [bool(r["baseline_correct"]) for r in details]
    p1_f = [bool(r["pass1_correct"]) for r in details]
    maj4_f = [bool(r["majority4_correct"]) for r in details]
    pass4_f = [bool(r["pass4_correct"]) for r in details]
    zero_steer_f = [bool(r["arkhe_zerosteer_forced_correct"]) for r in details]
    full_f = [bool(r["arkhe_full_forced_correct"]) for r in details]

    zs_sel = [r for r in details if r["arkhe_zerosteer_verdict"] != DecisionVerdict.ABSTAIN_UNKNOWN.name]
    full_sel = [r for r in details if r["arkhe_full_verdict"] != DecisionVerdict.ABSTAIN_UNKNOWN.name]
    zs_sel_ok = sum(1 for r in zs_sel if r["arkhe_zerosteer_forced_correct"])
    full_sel_ok = sum(1 for r in full_sel if r["arkhe_full_forced_correct"])

    avg4_hits = sum(sum(1 for c in r["always_on_std_pool"] if c["is_correct"]) for r in details)
    avg4_total = sum(len(r["always_on_std_pool"]) for r in details)

    lp_rows = [r for r in details if r.get("pass1_min_step_logprob") is not None]
    y_p1 = [1 if r["pass1_correct"] else 0 for r in lp_rows]
    min_lps = [float(r["pass1_min_step_logprob"]) for r in lp_rows]
    mean_lps = [float(r["pass1_mean_logprob"]) for r in lp_rows]

    return {
        "n": n,
        "baseline_sft_frozen": {"correct": sum(base_f), "acc_pct": round(100.0 * sum(base_f) / n, 2)},
        "pass1_greedy": {"correct": sum(p1_f), "acc_pct": round(100.0 * sum(p1_f) / n, 2)},
        "avg_at_4": {"correct": avg4_hits, "total_rollouts": avg4_total, "acc_pct": round(100.0 * avg4_hits / max(1, avg4_total), 2)},
        "majority_at_4": {"correct": sum(maj4_f), "acc_pct": round(100.0 * sum(maj4_f) / n, 2)},
        "pass_at_4": {"correct": sum(pass4_f), "acc_pct": round(100.0 * sum(pass4_f) / n, 2)},
        "arkhe_zerosteer_forced": {"correct": sum(zero_steer_f), "acc_pct": round(100.0 * sum(zero_steer_f) / n, 2)},
        "arkhe_zerosteer_selective": {
            "correct": zs_sel_ok,
            "answered": len(zs_sel),
            "acc_pct": round(100.0 * zs_sel_ok / max(1, len(zs_sel)), 2),
            "coverage_pct": round(100.0 * len(zs_sel) / n, 2),
        },
        "arkhe_full_forced": {"correct": sum(full_f), "acc_pct": round(100.0 * sum(full_f) / n, 2)},
        "arkhe_full_selective": {
            "correct": full_sel_ok,
            "answered": len(full_sel),
            "acc_pct": round(100.0 * full_sel_ok / max(1, len(full_sel)), 2),
            "coverage_pct": round(100.0 * len(full_sel) / n, 2),
        },
        "gate_triggered_count": sum(1 for r in details if r.get("multibranch_triggered")),
        "unbiased_pass1_logprob_auroc": {
            "n": len(lp_rows),
            "min_step_logprob": _compute_auroc(min_lps, y_p1),
            "mean_token_logprob": _compute_auroc(mean_lps, y_p1),
        },
        "paired_comparisons": {
            "majority4_vs_pass1": _paired_summary(p1_f, maj4_f),
            "arkhe_zerosteer_vs_majority4": _paired_summary(maj4_f, zero_steer_f),
            "arkhe_full_vs_pass1": _paired_summary(p1_f, full_f),
            "arkhe_full_vs_majority4": _paired_summary(maj4_f, full_f),
            "arkhe_full_vs_baseline": _paired_summary(base_f, full_f),
        },
    }


def main():
    prevent_windows_sleep()
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Pristine-50 Always-On G=4 Control & Logprob Benchmark")
    parser.add_argument("--start_idx", type=int, default=151)
    parser.add_argument("--end_idx", type=int, default=200)
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--std_temp", type=float, default=0.6)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_std_samples", action="store_true", help="Batch G=4 standard samples with num_return_sequences=4")
    parser.add_argument("--logprob_threshold", type=float, default=None)
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    print("=" * 80)
    print(f"SODALS v2.1 — PRISTINE-50 ALWAYS-ON G=4 CONTROL (Q{args.start_idx}..Q{args.end_idx})")
    print(f"Tau*: {args.tau:.2f} | StdTemp: {args.std_temp:.2f} | MaxTok: {args.max_new_tokens} | LogprobGate: {args.logprob_threshold}")
    print("=" * 80)

    with open(MATH500_JSONL, "r", encoding="utf-8") as f:
        all_lines = [json.loads(line) for line in f]

    target_items = []
    for one_idx in range(args.start_idx, args.end_idx + 1):
        if 1 <= one_idx <= len(all_lines):
            q = all_lines[one_idx - 1]
            target_items.append(
                {
                    "index": one_idx,
                    "problem": q.get("problem", ""),
                    "ground_truth": q.get("answer", ""),
                    "subject": q.get("subject", "General Math"),
                    "level": q.get("level", 3),
                    "unique_id": q.get("unique_id", f"MATH_500_{one_idx:04d}"),
                }
            )

    baseline_map: Dict[int, Dict[str, Any]] = {}
    baseline_by_uid: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(BASELINE_JSON):
        b_data = json.load(open(BASELINE_JSON, "r", encoding="utf-8"))
        for item in b_data.get("details", []):
            idx = item.get("index")
            uid = str(item.get("unique_id", ""))
            pred = str(item.get("predicted", ""))
            gt_b = str(item.get("ground_truth", ""))
            entry = {
                "predicted": pred,
                "is_correct": bool(NormalizedMathMatcher.check_match(pred, gt_b)[1]),
            }
            if idx is not None:
                baseline_map[int(idx)] = entry
            if uid:
                baseline_by_uid[uid] = entry

    # Checkpoint resume
    run_fallback = (
        FALLBACK_OUT_JSON
        if os.path.abspath(args.output_json) == os.path.abspath(DEFAULT_OUT_JSON)
        else os.path.join(tempfile.gettempdir(), os.path.basename(args.output_json))
    )
    merged_by_idx: Dict[int, Dict[str, Any]] = {}
    for cand in dict.fromkeys([args.output_json, run_fallback]):
        if os.path.exists(cand):
            try:
                prev = json.load(open(cand, "r", encoding="utf-8"))
                for row in prev.get("details", []):
                    if row.get("index") is not None:
                        merged_by_idx[int(row["index"])] = row
            except Exception:
                pass

    results: List[Dict[str, Any]] = [merged_by_idx[it["index"]] for it in target_items if it["index"] in merged_by_idx]
    completed_indices = {r["index"] for r in results}
    if completed_indices:
        print(f"[RESUME] Loaded {len(completed_indices)}/{len(target_items)} completed questions from checkpoint.", flush=True)

    if len(completed_indices) == len(target_items):
        print("[RESUME] All questions already completed in checkpoint!", flush=True)
        return

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    print("[GPU INIT] Loading Qwen 3.5 9B with Dual LoRA & CPUEmbeddingWrapper...", flush=True)
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    for sp in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(sp)
        if isinstance(tid, int) and tid >= 0 and tid not in eos_token_ids:
            eos_token_ids.append(tid)

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
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    target_dev = torch.device("cuda:0")
    if hasattr(base_model, "model") and hasattr(base_model.model, "embed_tokens"):
        embed_cpu = base_model.model.embed_tokens.to(device="cpu", dtype=torch.float16)
        base_model.model.embed_tokens = CPUEmbeddingWrapper(embed_cpu, target_device="cuda:0")
        type(base_model).device = property(lambda _m, d=target_dev: d)
    if hasattr(base_model, "lm_head"):
        base_model.lm_head = base_model.lm_head.to(torch.float16)
    torch.cuda.empty_cache()

    model = PeftModel.from_pretrained(base_model, ADAPTER_D1_PATH, adapter_name="expert_general_math")
    if os.path.exists(ADAPTER_D2_PATH):
        model.load_adapter(ADAPTER_D2_PATH, adapter_name="expert_abstract_algebra")

    router = NeuralPhaseRouter(anchors_path=ANCHORS_PATH) if os.path.exists(ANCHORS_PATH) else None
    blender = SoftLoraBlender(threshold=(router.threshold if router else 0.004913), temperature=0.03)
    blender.enable_multi_adapter(model)
    model.eval()
    print(f"[GPU INIT] Ready in {time.time() - t_load:.1f}s | VRAM: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MiB", flush=True)

    backbone = model.base_model.model.model if hasattr(model.base_model.model, "model") else model.model
    lm_head = model.get_output_embeddings()
    embed_layer = base_model.model.embed_tokens if hasattr(base_model, "model") else model.get_input_embeddings()

    def compute_pass1_logprob_stats(prompt_ids_tensor, gen_ids_tensor):
        if gen_ids_tensor.numel() < 2:
            return None, None
        full_ids = torch.cat([prompt_ids_tensor.cpu(), gen_ids_tensor.unsqueeze(0).cpu()], dim=1)
        p_len_loc = prompt_ids_tensor.shape[1]
        r_len_loc = gen_ids_tensor.shape[0]
        if p_len_loc + r_len_loc > 2560:
            r_len_loc = 2560 - p_len_loc
            full_ids = full_ids[:, :2560]
        with torch.no_grad():
            embeds = embed_layer(full_ids.to("cuda:0"))
            out = backbone(inputs_embeds=embeds, use_cache=False)
            resp_h = out.last_hidden_state[:, p_len_loc - 1 : -1, :]
            t_ids = full_ids[:, p_len_loc:].to("cuda:0")
            lps = []
            for s_i in range(0, r_len_loc, 64):
                e_i = min(s_i + 64, r_len_loc)
                c_logits = lm_head(resp_h[:, s_i:e_i, :].contiguous()).float()
                c_lp = torch.nn.functional.log_softmax(c_logits, dim=-1)
                tok_lp = torch.gather(c_lp, -1, t_ids[:, s_i:e_i].unsqueeze(-1)).squeeze(-1)
                lps.append(tok_lp[0].cpu())
                del c_logits, c_lp
            del out, resp_h, embeds
        arr = torch.cat(lps, dim=0).numpy()
        step_means = [float(arr[k : k + 32].mean()) for k in range(0, len(arr), 32) if len(arr[k : k + 32]) >= 8]
        min_step = min(step_means) if step_means else float(arr.mean())
        return float(arr.mean()), float(min_step)

    sim_arche = SimilarityArche(matcher_fn=lambda a, b: bool(NormalizedMathMatcher.check_match(str(a), str(b))[1]))
    pipeline = ArkheNeurosymbolicPipeline(
        selective_tau=args.tau,
        matcher_fn=lambda a, b: bool(NormalizedMathMatcher.check_match(str(a), str(b))[1]),
        logprob_trigger_threshold=args.logprob_threshold,
    )
    inference_arche = InferenceArche(logprob_trigger_threshold=args.logprob_threshold)

    code_hashes = {
        fn: _sha256_file(os.path.join(SFT_DIR, fn))
        for fn in ("cst_symbolic_verifier.py", "arkhe_neurosymbolic_pipeline.py", "eval_pristine50_always_on_control.py")
        if os.path.exists(os.path.join(SFT_DIR, fn))
    }
    runtime_meta = collect_runtime_metadata()
    seeds = [42, 101, 202, 303]
    active_out_path = args.output_json

    for item in target_items:
        q_idx = item["index"]
        if q_idx in completed_indices:
            continue

        prob = item["problem"]
        gt = item["ground_truth"]
        low = prob.lower()
        algebra_cues = ("group", "ring", "field", "abelian", "isomorphism", "homomorphism", "eigenvalue", "matrix", "determinant")
        alpha = 0.65 if any(k in low for k in algebra_cues) else 0.20
        blender.set_blend(model, alpha)
        rep_penalty = blender.compute_rep_penalty(alpha)

        std_prompt = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step concisely, verify all algebraic transitions, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )

        # 1. Pass-1 Greedy + Logprobs (always measured on all N=50)
        t0 = time.time()
        p1_inputs = tokenizer(std_prompt, return_tensors="pt").to("cuda:0")
        p1_len = p1_inputs.input_ids.shape[1]
        with torch.no_grad():
            p1_out = model.generate(
                **p1_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=rep_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_ids,
            )
        p1_toks = p1_out[0][p1_len:]
        p1_text = tokenizer.decode(p1_toks, skip_special_tokens=True).strip()
        p1_mean_lp, p1_min_lp = compute_pass1_logprob_stats(p1_inputs.input_ids, p1_toks)
        t1 = build_trace("Pass1_Greedy", p1_text, gen_tokens=len(p1_toks), prompt=prob)
        if p1_mean_lp is not None:
            t1 = dc_replace(t1, mean_token_logprob=p1_mean_lp, min_step_logprob=p1_min_lp)
        p1_ok = bool(NormalizedMathMatcher.check_match(t1.extracted_answer, gt)[1])
        del p1_out
        torch.cuda.empty_cache()

        # 2. Always-On Unconditional G=4 Standard Samples (Compute-Matched Majority@4 & Pass@4)
        std_traces: List[CandidateTrace] = []
        std_pool_records: List[Dict[str, Any]] = []
        if args.batch_std_samples:
            set_seed(seeds[0])
            with torch.no_grad():
                s_batch_out = model.generate(
                    **p1_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.std_temp,
                    top_p=0.9,
                    num_return_sequences=len(seeds),
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_ids,
                )
            for s_idx, seed_val in enumerate(seeds, start=1):
                s_toks = s_batch_out[s_idx - 1][p1_len:]
                # Trim trailing pad/eos tokens in batched decode
                s_list = s_toks.tolist()
                while s_list and s_list[-1] == tokenizer.pad_token_id:
                    s_list.pop()
                s_toks_trimmed = torch.tensor(s_list, dtype=s_toks.dtype)
                s_text = tokenizer.decode(s_toks_trimmed, skip_special_tokens=True).strip()
                tr = build_trace(f"Std_{s_idx}", s_text, gen_tokens=len(s_list), prompt=prob)
                is_c = bool(NormalizedMathMatcher.check_match(tr.extracted_answer, gt)[1])
                std_traces.append(tr)
                std_pool_records.append(
                    {
                        "branch": tr.branch_id,
                        "seed": seed_val,
                        "pred": tr.extracted_answer,
                        "is_correct": is_c,
                        "has_end": tr.has_end_tag,
                        "is_valid": tr.is_symbolically_valid,
                        "eq_count": tr.verified_eq_count,
                        "contras": list(tr.contradictions),
                        "tokens": tr.generated_tokens,
                    }
                )
            del s_batch_out
            torch.cuda.empty_cache()
        else:
            for s_idx, seed_val in enumerate(seeds, start=1):
                set_seed(seed_val)
                with torch.no_grad():
                    s_out = model.generate(
                        **p1_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.std_temp,
                        top_p=0.9,
                        repetition_penalty=1.0,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos_token_ids,
                    )
                s_toks = s_out[0][p1_len:]
                s_text = tokenizer.decode(s_toks, skip_special_tokens=True).strip()
                tr = build_trace(f"Std_{s_idx}", s_text, gen_tokens=len(s_toks), prompt=prob)
                is_c = bool(NormalizedMathMatcher.check_match(tr.extracted_answer, gt)[1])
                std_traces.append(tr)
                std_pool_records.append(
                    {
                        "branch": tr.branch_id,
                        "seed": seed_val,
                        "pred": tr.extracted_answer,
                        "is_correct": is_c,
                        "has_end": tr.has_end_tag,
                        "is_valid": tr.is_symbolically_valid,
                        "eq_count": tr.verified_eq_count,
                        "contras": list(tr.contradictions),
                        "tokens": tr.generated_tokens,
                    }
                )
                del s_out
                torch.cuda.empty_cache()

        maj4_pred = _majority_vote_answer([t.extracted_answer for t in std_traces], sim_arche)
        maj4_ok = bool(NormalizedMathMatcher.check_match(maj4_pred, gt)[1])
        pass4_ok = any(r["is_correct"] for r in std_pool_records)

        # Compute-Matched Arkhe Zero-Steer over exact S1..S4 pool
        zs_state: ArkheState = pipeline.run_from_traces(
            question_id=f"Q{q_idx:03d}",
            pass1_trace=t1,
            candidate_traces=std_traces,
            gate_triggered=True,
            domain_alpha=alpha,
            prompt=prob,
        )
        zs_pred = zs_state.winning_trace.extracted_answer if zs_state.winning_trace else maj4_pred
        zs_ok = bool(NormalizedMathMatcher.check_match(zs_pred, gt)[1])

        # 3. Arkhe Gate-Triggered Repair Branches (only if InferenceArche fires)
        triggered = inference_arche.should_trigger_multibranch(t1)
        repair_traces: List[CandidateTrace] = []
        repair_pool_records: List[Dict[str, Any]] = []
        if triggered:
            specs = InferenceArche.build_step_rollback_branch_specs(prob, t1)
            for b_i, spec in enumerate(specs):
                temp = float(spec["temp"])
                seed_val = seeds[b_i % len(seeds)]
                do_samp = temp > 0.05
                use_rb = bool(spec.get("use_prefix_rollback", False))
                prefix = str(spec.get("verified_prefix", "")).strip()
                directive = str(spec.get("system_directive", ""))
                b_prompt = (
                    f"<|im_start|>user\n{prob}\n{directive}<|im_end|>\n<|im_start|>assistant\n<think>\n{prefix}\n"
                    if (use_rb and prefix)
                    else f"<|im_start|>user\n{prob}\n{directive}<|im_end|>\n<|im_start|>assistant\n<think>\n"
                )
                set_seed(seed_val)
                b_in = tokenizer(b_prompt, return_tensors="pt").to("cuda:0")
                bp_len = b_in.input_ids.shape[1]
                g_kw = dict(
                    **b_in,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_samp,
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_ids,
                )
                if do_samp:
                    g_kw["temperature"] = temp
                    g_kw["top_p"] = 0.9
                with torch.no_grad():
                    b_out = model.generate(**g_kw)
                b_toks = b_out[0][bp_len:]
                b_txt = tokenizer.decode(b_toks, skip_special_tokens=True).strip()
                full_b_txt = (prefix + "\n" + b_txt) if (use_rb and prefix) else b_txt
                ct = build_trace(str(spec["branch"]), full_b_txt, gen_tokens=len(b_toks), prompt=prob)
                repair_traces.append(ct)
                repair_pool_records.append(
                    {
                        "branch": ct.branch_id,
                        "role": str(spec.get("role", "")),
                        "pred": ct.extracted_answer,
                        "is_correct": bool(NormalizedMathMatcher.check_match(ct.extracted_answer, gt)[1]),
                        "has_end": ct.has_end_tag,
                        "is_valid": ct.is_symbolically_valid,
                        "eq_count": ct.verified_eq_count,
                        "contras": list(ct.contradictions),
                        "tokens": ct.generated_tokens,
                    }
                )
                del b_out
                torch.cuda.empty_cache()

            full_state = pipeline.run_from_traces(
                question_id=f"Q{q_idx:03d}",
                pass1_trace=t1,
                candidate_traces=repair_traces,
                gate_triggered=True,
                domain_alpha=alpha,
                prompt=prob,
            )
            full_pred = full_state.winning_trace.extracted_answer if full_state.winning_trace else t1.extracted_answer
            full_ok = bool(NormalizedMathMatcher.check_match(full_pred, gt)[1])
            full_verdict = full_state.verdict.name
            full_conf = full_state.epistemic_confidence
        else:
            full_pred = t1.extracted_answer
            full_ok = p1_ok
            full_verdict = DecisionVerdict.COMMIT_ANSWER.name if t1.has_end_tag else DecisionVerdict.ABSTAIN_UNKNOWN.name
            full_conf = 0.90 if (t1.has_end_tag and t1.is_symbolically_valid) else 0.40

        b_entry = baseline_by_uid.get(item["unique_id"]) or baseline_map.get(q_idx, {"predicted": "", "is_correct": False})
        row_rec = {
            "index": q_idx,
            "unique_id": item["unique_id"],
            "subject": item["subject"],
            "level": item["level"],
            "ground_truth": gt,
            "baseline_predicted": b_entry["predicted"],
            "baseline_correct": bool(b_entry["is_correct"]),
            "pass1_predicted": t1.extracted_answer,
            "pass1_correct": p1_ok,
            "pass1_tokens": t1.generated_tokens,
            "pass1_has_end": t1.has_end_tag,
            "pass1_eq_count": t1.verified_eq_count,
            "pass1_contras": list(t1.contradictions),
            "pass1_mean_logprob": round(p1_mean_lp, 4) if p1_mean_lp is not None else None,
            "pass1_min_step_logprob": round(p1_min_lp, 4) if p1_min_lp is not None else None,
            "always_on_std_pool": std_pool_records,
            "majority4_predicted": maj4_pred,
            "majority4_correct": maj4_ok,
            "pass4_correct": pass4_ok,
            "arkhe_zerosteer_predicted": zs_pred,
            "arkhe_zerosteer_forced_correct": zs_ok,
            "arkhe_zerosteer_verdict": zs_state.verdict.name,
            "arkhe_zerosteer_conf": zs_state.epistemic_confidence,
            "multibranch_triggered": bool(triggered),
            "repair_pool": repair_pool_records,
            "arkhe_full_predicted": full_pred,
            "arkhe_full_forced_correct": full_ok,
            "arkhe_full_verdict": full_verdict,
            "arkhe_full_conf": full_conf,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        results.append(row_rec)
        completed_indices.add(q_idx)

        metrics = _compute_metrics(results)
        out_payload = {
            "benchmark": f"Pristine-50 Always-On G=4 Control (Q{args.start_idx}..Q{args.end_idx})",
            "runtime_metadata": runtime_meta,
            "frozen_code_sha256": code_hashes,
            "total_questions": len(target_items),
            "processed_questions": len(results),
            "metrics": metrics,
            "details": results,
        }
        active_out_path = safe_json_dump(out_payload, active_out_path, fallback_path=run_fallback)
        gc.collect()
        torch.cuda.empty_cache()

        lp_str = f"{p1_min_lp:.3f}" if p1_min_lp is not None else "NA"
        print(
            f"[{len(results)}/{len(target_items)}] Q{q_idx:03d} ({row_rec['elapsed_sec']}s) | "
            f"P1={'PASS' if p1_ok else 'FAIL'} (lp={lp_str}) | Maj@4={'PASS' if maj4_ok else 'FAIL'} | "
            f"Pass@4={'PASS' if pass4_ok else 'FAIL'} | ArkheZS={'PASS' if zs_ok else 'FAIL'} | "
            f"ArkheFull={'PASS' if full_ok else 'FAIL'} (trig={triggered}) || "
            f"Cum P1={metrics['pass1_greedy']['correct']}/{len(results)} "
            f"Maj4={metrics['majority_at_4']['correct']}/{len(results)} "
            f"ArkZS={metrics['arkhe_zerosteer_forced']['correct']}/{len(results)} "
            f"ArkFull={metrics['arkhe_full_forced']['correct']}/{len(results)}",
            flush=True,
        )

    print("=" * 80)
    print(f"PRISTINE-50 ALWAYS-ON CONTROL COMPLETE! Saved: {active_out_path}")
    print(json.dumps(_compute_metrics(results), indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
