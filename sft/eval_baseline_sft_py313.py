"""
SODALS v2.1 — Same-Environment (Python 3.13) SFT Baseline Capture (Exploratory Q026..Q200)
===========================================================================================
Phase 0 found 24/30 (80.0%) greedy token identity between legacy Py 3.10 and new Py 3.13 stacks.
To prevent cross-stack 4-bit kernel drift from confounding comparisons, this script captures
the canonical `SFT Baseline` (`qwen_lora_adapter_balanced_r4`, scale=0.25, do_sample=False)
strictly inside the frozen Python 3.13 stack on exploratory slices (`Q026..Q200`).
Confirmatory Q201..Q500 remains untouched until Phase 4 protocol lock.
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

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SFT_DIR)
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

if sys.version_info >= (3, 13):
    import smart_app_control_compat  # noqa: F401

import torch

from cst_symbolic_verifier import NormalizedMathMatcher
from env_reproducibility import collect_runtime_metadata
from eval_pristine30_q171_q200_arkhe_pipeline import (
    CPUEmbeddingWrapper,
    MATH500_JSONL,
    MODEL_PATH,
    prevent_windows_sleep,
    safe_json_dump,
)

ADAPTER_SFT = os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4")
DEFAULT_OUT = os.path.join(SFT_DIR, "baseline_sft_py313_q026_q200_results.json")
FALLBACK_OUT = os.path.join(tempfile.gettempdir(), "baseline_sft_py313_q026_q200_results.json")


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


def main():
    prevent_windows_sleep()
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Capture Py 3.13 Same-Stack SFT Baseline")
    parser.add_argument("--start_idx", type=int, default=26)
    parser.add_argument("--end_idx", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUT)
    parser.add_argument("--allow_confirmatory_range", action="store_true", help="Required flag to run beyond Q200.")
    args = parser.parse_args()

    if args.end_idx > 200 and not args.allow_confirmatory_range:
        raise ValueError("Reserve protection: Q201..Q500 requires explicit --allow_confirmatory_range after protocol lock.")

    with open(MATH500_JSONL, "r", encoding="utf-8") as f:
        all_lines = [json.loads(line) for line in f]

    items = []
    for one_idx in range(args.start_idx, args.end_idx + 1):
        q = all_lines[one_idx - 1]
        items.append({
            "index": one_idx,
            "unique_id": q.get("unique_id", f"MATH_500_{one_idx:04d}"),
            "subject": q.get("subject", ""),
            "level": q.get("level", ""),
            "problem": q["problem"],
            "ground_truth": q["answer"],
        })

    merged: Dict[int, Dict[str, Any]] = {}
    for cand in (args.output_json, FALLBACK_OUT):
        if os.path.exists(cand):
            try:
                prev = json.load(open(cand, "r", encoding="utf-8"))
                for r in prev.get("details", []):
                    merged[int(r["index"])] = r
            except Exception:
                pass

    results: List[Dict[str, Any]] = [merged[it["index"]] for it in items if it["index"] in merged]
    done_idx = {r["index"] for r in results}
    if len(done_idx) == len(items):
        print(f"[RESUME] All {len(items)} items already completed in {args.output_json}.", flush=True)
        return

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

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

    model = PeftModel.from_pretrained(base_model, ADAPTER_SFT)
    set_lora_scale(model, 0.25)
    model.eval()
    runtime_meta = collect_runtime_metadata()

    for it in items:
        idx = it["index"]
        if idx in done_idx:
            continue
        prompt = (
            f"<|im_start|>user\n{it['problem']}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
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
        ok = bool(NormalizedMathMatcher.check_match(pred, it["ground_truth"])[1])

        results.append({
            "index": idx,
            "unique_id": it["unique_id"],
            "subject": it["subject"],
            "level": it["level"],
            "ground_truth": it["ground_truth"],
            "predicted": pred,
            "frozen_correct": ok,
            "tokens": int(gen_ids.numel()),
            "text_sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
            "elapsed_sec": round(dt, 2),
        })
        results.sort(key=lambda r: r["index"])
        done_idx.add(idx)
        ok_cnt = sum(1 for r in results if r["frozen_correct"])
        payload = {
            "benchmark": f"SFT Baseline Py313 Same-Stack (Q{args.start_idx}..Q{args.end_idx})",
            "runtime_metadata": runtime_meta,
            "total": len(items),
            "completed": len(results),
            "frozen_correct": ok_cnt,
            "acc_pct": round(100.0 * ok_cnt / len(results), 2),
            "details": results,
        }
        saved = safe_json_dump(payload, args.output_json, FALLBACK_OUT)
        print(f"[{len(results):03d}/{len(items)}] Q{idx:03d} | ok={ok} | Cum={ok_cnt}/{len(results)} ({payload['acc_pct']}%) | {dt:.1f}s -> {os.path.basename(saved)}", flush=True)


if __name__ == "__main__":
    main()
