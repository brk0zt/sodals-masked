"""
SODALS v2.1 — Phase 2B: Independent 2x2 Validation of SymbolicStepChecker on 400 MATH-Train Rollouts
====================================================================================================
Claude Audit #3 Requirement:
  - Precision jump (33% -> 73%) on Clean-125 was tuned on that dev pool.
  - Evaluate the FROZEN `SymbolicStepChecker` (NO rule modifications allowed!) on an independent
    pool of 400 labeled Hendrycks MATH-train rollouts (100 prompts x 4 sampled rollouts).
  - Outputs the out-of-sample 2x2 table: `{contras > 0} x {correct / wrong}`, Precision, Recall, FPR.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from typing import Any, Dict, List

if "--online" not in sys.argv:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
else:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SFT_DIR)
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

if sys.platform == "win32" and sys.version_info >= (3, 13):
    import smart_app_control_compat  # noqa: F401

import torch

from cst_symbolic_verifier import NormalizedMathMatcher, SymbolicStepChecker
from env_reproducibility import collect_runtime_metadata
from eval_pristine30_q171_q200_arkhe_pipeline import (
    CPUEmbeddingWrapper,
    MODEL_PATH,
    prevent_windows_sleep,
    safe_json_dump,
)

PROMPTS_JSON = os.path.join(SFT_DIR, "math_train_100_grpo_prompts.json")
ADAPTER_SFT = os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4")
DEFAULT_OUT = os.path.join(SFT_DIR, "symbolic_checker_independent_400_results.json")
FALLBACK_OUT = os.path.join(tempfile.gettempdir(), "symbolic_checker_independent_400_results.json")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_boxed(text: str) -> str:
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
    m = re.search(r"(?:Answer|answer|the answer is)[:\s]+([^\n\.]+)", text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"\b(-?\d+(?:\.\d+)?)\b", text)
    return nums[-1] if nums else ""


def set_lora_scale(model, scale: float):
    for module in model.modules():
        if hasattr(module, "scaling") and hasattr(module, "lora_alpha") and hasattr(module, "r"):
            for name in module.scaling:
                module.scaling[name] = (module.lora_alpha[name] / module.r[name]) * scale


def compute_2x2(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    fp_examples = []
    for item in rows:
        for r in item["rollouts"]:
            ok = bool(r["is_correct"])
            has_contra = bool(r["has_contra"])
            if ok:
                if has_contra:
                    fp += 1
                    if len(fp_examples) < 10:
                        fp_examples.append({
                            "prompt_index": item["prompt_index"],
                            "rollout_idx": r["rollout_idx"],
                            "pred": r["predicted"],
                            "gt": item["ground_truth"],
                            "contras": r["contradictions"],
                        })
                else:
                    tn += 1
            else:
                if has_contra:
                    tp += 1
                else:
                    fn += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    return {
        "total_rollouts": tp + fp + fn + tn,
        "TP_contra_wrong": tp,
        "FP_contra_correct": fp,
        "FN_clean_wrong": fn,
        "TN_clean_correct": tn,
        "precision_pct": round(100.0 * prec, 2),
        "recall_pct": round(100.0 * rec, 2),
        "false_positive_rate_pct": round(100.0 * fpr, 2),
        "sample_fp_examples": fp_examples,
    }


def main():
    prevent_windows_sleep()
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Independent 400 MATH-train Rollout 2x2 Audit")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--adapter_path", type=str, default=ADAPTER_SFT)
    parser.add_argument("--fp16", action="store_true", help="Load full FP16 multi-GPU model (e.g. Kaggle T4 x2)")
    parser.add_argument("--online", action="store_true", help="Allow HuggingFace Hub downloads")
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_g4", action=argparse.BooleanOptionalAction, default=True, help="Use num_return_sequences=G batched decode (default: True)")
    parser.add_argument("--rescore_only", action="store_true")
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.online:
        from cloud_runtime_utils import configure_hf_env
        configure_hf_env(online=True)

    prompts_data = json.load(open(PROMPTS_JSON, "r", encoding="utf-8"))["prompts"][: args.num_prompts]
    merged: Dict[int, Dict[str, Any]] = {}
    for cand in (args.output_json, FALLBACK_OUT):
        if os.path.exists(cand):
            try:
                prev = json.load(open(cand, "r", encoding="utf-8"))
                for row in prev.get("details", []):
                    merged[int(row["prompt_index"])] = row
            except Exception:
                pass

    if args.rescore_only:
        rows = [merged[p["prompt_index"]] for p in prompts_data if p["prompt_index"] in merged]
        for item in rows:
            for r in item["rollouts"]:
                rep = SymbolicStepChecker.verify_proof_steps(
                    r["full_text"], prompt=item["problem"], include_sorry_in_contradictions=False
                )
                r["contradictions"] = list(rep.contradictions)
                r["has_contra"] = len(rep.contradictions) > 0
                r["eq_count"] = rep.verified_eq_count
                r["has_verbal_bluff"] = rep.has_sorry_bluff
                r["is_correct"] = bool(NormalizedMathMatcher.check_match(r["predicted"], item["ground_truth"])[1])
        summary_2x2 = compute_2x2(rows)
        payload = {
            "experiment": "Independent 400 MATH-Train Rollout Audit (Frozen SymbolicStepChecker)",
            "runtime_metadata": collect_runtime_metadata(),
            "checker_sha256": _sha256_file(os.path.join(SFT_DIR, "cst_symbolic_verifier.py")),
            "prompts_count": len(rows),
            "summary_2x2": summary_2x2,
            "details": rows,
        }
        safe_json_dump(payload, args.output_json, FALLBACK_OUT)
        print("RESCORE 2x2:", json.dumps(summary_2x2, indent=2), flush=True)
        return

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    print("=" * 80)
    print(f"INDEPENDENT 400 ROLLOUT BENCHMARK | N={len(prompts_data)} x G={args.group_size} | batch={args.batch_g4} | fp16={args.fp16}")
    print("=" * 80, flush=True)

    if args.fp16 or args.online or args.model_path != MODEL_PATH:
        from cloud_runtime_utils import get_primary_device, load_qwen_model_and_tokenizer
        base_model, tokenizer, _rt = load_qwen_model_and_tokenizer(
            model_path=args.model_path,
            quantize_4bit=not args.fp16,
            online=True if args.online else None,
        )
        entry_dev = get_primary_device(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
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
        entry_dev = target_dev

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    for sp in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(sp)
        if isinstance(tid, int) and tid >= 0 and tid not in eos_ids:
            eos_ids.append(tid)

    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    set_lora_scale(model, 0.25)
    model.eval()
    print(f"[GPU INIT] Ready | VRAM: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MiB", flush=True)

    rows: List[Dict[str, Any]] = [merged[p["prompt_index"]] for p in prompts_data if p["prompt_index"] in merged]
    done_idx = {r["prompt_index"] for r in rows}
    checker_sha = _sha256_file(os.path.join(SFT_DIR, "cst_symbolic_verifier.py"))

    for i, p in enumerate(prompts_data):
        p_idx = int(p["prompt_index"])
        if p_idx in done_idx:
            continue
        prob = p["problem"]
        gt = p["ground_truth"]
        prompt = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(entry_dev)
        t0 = time.time()
        rollouts_out = []

        if args.batch_g4:
            set_seed(1000 + p_idx)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temp,
                    top_p=0.95,
                    num_return_sequences=args.group_size,
                    repetition_penalty=1.03,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )
            seqs = [out[g][inputs.input_ids.shape[1]:] for g in range(args.group_size)]
        else:
            seqs = []
            for g in range(args.group_size):
                set_seed(1000 + p_idx * 10 + g)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temp,
                        top_p=0.95,
                        repetition_penalty=1.03,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos_ids,
                    )
                seqs.append(out[0][inputs.input_ids.shape[1]:])

        for g, seq_ids in enumerate(seqs):
            txt = tokenizer.decode(seq_ids, skip_special_tokens=True).strip()
            pred = extract_boxed(txt)
            ok = bool(NormalizedMathMatcher.check_match(pred, gt)[1])
            rep = SymbolicStepChecker.verify_proof_steps(txt, prompt=prob, include_sorry_in_contradictions=False)
            rollouts_out.append({
                "rollout_idx": g,
                "tokens": int(seq_ids.numel()),
                "predicted": pred,
                "is_correct": ok,
                "has_contra": len(rep.contradictions) > 0,
                "contradictions": list(rep.contradictions),
                "eq_count": rep.verified_eq_count,
                "has_verbal_bluff": rep.has_sorry_bluff,
                "full_text": txt,
            })

        dt = time.time() - t0
        rows.append({
            "prompt_index": p_idx,
            "train_row_id": p.get("train_row_id"),
            "subject": p.get("subject"),
            "level": p.get("level"),
            "problem": prob,
            "ground_truth": gt,
            "elapsed_sec": round(dt, 2),
            "rollouts": rollouts_out,
        })
        rows.sort(key=lambda r: r["prompt_index"])
        summary_2x2 = compute_2x2(rows)
        payload = {
            "experiment": "Independent 400 MATH-Train Rollout Audit (Frozen SymbolicStepChecker)",
            "runtime_metadata": collect_runtime_metadata(),
            "checker_sha256": checker_sha,
            "prompts_count": len(rows),
            "summary_2x2": summary_2x2,
            "details": rows,
        }
        safe_json_dump(payload, args.output_json, FALLBACK_OUT)
        print(
            f"[{len(rows):03d}/{len(prompts_data)}] P{p_idx:03d} ({dt:5.1f}s) | "
            f"TP={summary_2x2['TP_contra_wrong']} FP={summary_2x2['FP_contra_correct']} "
            f"FN={summary_2x2['FN_clean_wrong']} TN={summary_2x2['TN_clean_correct']} "
            f"(Prec={summary_2x2['precision_pct']}%, Rec={summary_2x2['recall_pct']}%)",
            flush=True,
        )


if __name__ == "__main__":
    main()
