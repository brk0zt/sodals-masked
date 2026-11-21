"""
SODALS v2.1 — Phase 3 Evaluation Harness for Disjoint MATH-Train Dev Pool (N=500)
=================================================================================
Evaluates an adapter (`balanced_r4` baseline, `GRPO-1 ZPD`, or `GRPO-2 DynLambda`)
on `sft/math_train_disjoint_dev500.json` (level-matched to MATH-500, 0% overlap).
Supports checkpoint resume, exact greedy decoding, frozen NormalizedMathMatcher,
and two-sided exact binomial McNemar comparison against a baseline JSON.
"""

import argparse
import hashlib
import json
import math
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

from cst_symbolic_verifier import NormalizedMathMatcher
from env_reproducibility import collect_runtime_metadata
from eval_pristine30_q171_q200_arkhe_pipeline import (
    CPUEmbeddingWrapper,
    MODEL_PATH,
    prevent_windows_sleep,
    safe_json_dump,
)

DEFAULT_DEV500 = os.path.join(SFT_DIR, "math_train_disjoint_dev500.json")
DEFAULT_ADAPTER = os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4")


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
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""


def set_lora_scale(model, scale: float):
    for module in model.modules():
        if hasattr(module, "scaling") and hasattr(module, "lora_alpha") and hasattr(module, "r"):
            for name in module.scaling:
                module.scaling[name] = (module.lora_alpha[name] / module.r[name]) * scale


def exact_mcnemar(b_wins: int, c_losses: int) -> float:
    n = b_wins + c_losses
    if n == 0:
        return 1.0
    k = min(b_wins, c_losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** (n - 1))
    return min(1.0, float(p))


def main():
    prevent_windows_sleep()
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate Adapter on Disjoint Dev-500 Pool")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--fp16", action="store_true", help="Load full FP16 multi-GPU model (e.g. Kaggle T4 x2)")
    parser.add_argument("--online", action="store_true", help="Allow HuggingFace Hub downloads")
    parser.add_argument("--dev_json", type=str, default=DEFAULT_DEV500)
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--baseline_json", type=str, default=None)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--d2_adapter_path", type=str, default=None, help="Optional D2 Algebra LoRA adapter")
    parser.add_argument("--svon_identity_lora", type=str, default=None, help="Optional Svon identity LoRA adapter")
    parser.add_argument("--enable_svon_cgw", action="store_true", help="Upgrade model with Svon-CGW hybrid layers")
    parser.add_argument("--svon_cgw_weights", type=str, default=None, help="Path to svon_cgw_hybrid.safetensors")
    args = parser.parse_args()

    if args.online:
        from cloud_runtime_utils import configure_hf_env
        configure_hf_env(online=True)

    prompts = json.load(open(args.dev_json, "r", encoding="utf-8"))["prompts"][: args.limit]
    fallback_out = os.path.join(tempfile.gettempdir(), os.path.basename(args.output_json))
    merged: Dict[int, Dict[str, Any]] = {}
    for cand in (args.output_json, fallback_out):
        if os.path.exists(cand):
            try:
                prev = json.load(open(cand, "r", encoding="utf-8"))
                for row in prev.get("details", []):
                    merged[int(row["prompt_index"])] = row
            except Exception:
                pass

    base_by_idx: Dict[int, bool] = {}
    if args.baseline_json and os.path.exists(args.baseline_json):
        bd = json.load(open(args.baseline_json, "r", encoding="utf-8"))
        for row in bd.get("details", []):
            base_by_idx[int(row["prompt_index"])] = bool(row["is_correct"])

    rows: List[Dict[str, Any]] = [merged[p["prompt_index"]] for p in prompts if p["prompt_index"] in merged]
    done_idx = {r["prompt_index"] for r in rows}

    if len(done_idx) < len(prompts):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

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

        if args.d2_adapter_path or args.svon_identity_lora:
            from cloud_runtime_utils import mount_production_adapters
            model, _blender, _mounted = mount_production_adapters(
                base_model,
                d1_adapter_path=args.adapter_path,
                d2_adapter_path=args.d2_adapter_path,
                svon_adapter_path=args.svon_identity_lora,
                d1_scale=args.scale,
                alpha=0.20,
            )
        else:
            model = PeftModel.from_pretrained(base_model, args.adapter_path)
            set_lora_scale(model, args.scale)

        if args.enable_svon_cgw:
            from svon_kuramoto_equilibrium import load_svon_cgw_state_dict, upgrade_model_to_svon_cgw
            upgrade_model_to_svon_cgw(model, phase_rank=16, freeze_base=True)
            if args.svon_cgw_weights and os.path.exists(args.svon_cgw_weights):
                load_svon_cgw_state_dict(model, args.svon_cgw_weights)

        model.eval()

        for p in prompts:
            idx = int(p["prompt_index"])
            if idx in done_idx:
                continue
            prompt = (
                f"<|im_start|>user\n{p['problem']}\n"
                f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(entry_dev)
            set_seed(42)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            dt = time.time() - t0
            gen_ids = out[0][inputs.input_ids.shape[1]:]
            txt = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            pred = extract_boxed(txt)
            ok = bool(NormalizedMathMatcher.check_match(pred, p["ground_truth"])[1])

            rows.append({
                "prompt_index": idx,
                "train_row_id": p["train_row_id"],
                "level": p["level"],
                "subject": p["subject"],
                "ground_truth": p["ground_truth"],
                "predicted": pred,
                "is_correct": ok,
                "tokens": int(gen_ids.numel()),
                "text_sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
                "elapsed_sec": round(dt, 2),
            })
            rows.sort(key=lambda r: r["prompt_index"])
            done_idx.add(idx)

            correct_so_far = sum(1 for r in rows if r["is_correct"])
            payload = {
                "adapter_path": args.adapter_path,
                "runtime_metadata": collect_runtime_metadata(),
                "evaluated": len(rows),
                "correct": correct_so_far,
                "acc_pct": round(100.0 * correct_so_far / len(rows), 2),
                "details": rows,
            }
            safe_json_dump(payload, args.output_json, fallback_out)
            print(f"[{len(rows):03d}/{len(prompts)}] Q{idx:03d} L{p['level']} | ok={ok} | acc={payload['acc_pct']}% | {dt:.1f}s", flush=True)

    correct_cnt = sum(1 for r in rows if r["is_correct"])
    paired = None
    if base_by_idx:
        common = [r for r in rows if r["prompt_index"] in base_by_idx]
        wins = sum(1 for r in common if (not base_by_idx[r["prompt_index"]]) and r["is_correct"])
        losses = sum(1 for r in common if base_by_idx[r["prompt_index"]] and (not r["is_correct"]))
        paired = {
            "n_common": len(common),
            "baseline_correct": sum(1 for r in common if base_by_idx[r["prompt_index"]]),
            "model_correct": sum(1 for r in common if r["is_correct"]),
            "gains": wins,
            "losses": losses,
            "net": wins - losses,
            "exact_mcnemar_p": round(exact_mcnemar(wins, losses), 4),
        }

    final_payload = {
        "adapter_path": args.adapter_path,
        "runtime_metadata": collect_runtime_metadata(),
        "evaluated": len(rows),
        "correct": correct_cnt,
        "acc_pct": round(100.0 * correct_cnt / max(1, len(rows)), 2),
        "paired_vs_baseline": paired,
        "details": rows,
    }
    saved = safe_json_dump(final_payload, args.output_json, fallback_out)
    print(f"DEV500 EVAL COMPLETE -> {saved} | {correct_cnt}/{len(rows)} ({final_payload['acc_pct']}%) | paired={paired}", flush=True)


if __name__ == "__main__":
    main()
