"""
SODALS v2.1 — Phase 0: Environment Equivalence Test (Python 3.10 Legacy vs Python 3.13 New)
==========================================================================================
Measures greedy generation equivalence between:
  - Legacy stack: Python 3.10.11 (Windows.old) + torch 2.5.1+cu121 + transformers 5.16.0.dev0 + peft 0.14.0 + bnb 0.50.1
  - New stack:    Python 3.13.7 + torch 2.8.0+cu126 + transformers 5.23.0 + peft 0.21.3 + bnb 0.50.2

Runs identical model weights (`Qwen3.5-9B-SODALS-Baseline-SafeTensors` + `qwen_lora_adapter_balanced_r4`,
scale=0.25), identical prompt, identical greedy decoding (`do_sample=False`, `repetition_penalty=1.05`,
`max_new_tokens=1024`), and compares exact generated token ID lists, final extracted answers, and
frozen NormalizedMathMatcher verdicts (also cross-checking against stored Baseline Full-500).
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
import torch.nn as nn

from cst_symbolic_verifier import NormalizedMathMatcher
from env_reproducibility import collect_runtime_metadata

MODEL_PATH = os.path.join(ROOT_DIR, "models", "Qwen3.5-9B-SODALS-Baseline-SafeTensors")
ADAPTER_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4")
MATH500_JSONL = r"C:\Users\Burak\.cache\huggingface\hub\datasets--HuggingFaceH4--MATH-500\snapshots\6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be\test.jsonl"
BASELINE_JSON = os.path.join(ROOT_DIR, "math500_full500_lora_4096_results.json")


class _CPUEmbeddingWrapper(nn.Module):
    """Exact discrete CPU embedding lookup to preserve 8 GB VRAM headroom."""

    def __init__(self, cpu_embed: nn.Module, target_device: str = "cuda:0"):
        super().__init__()
        self.cpu_embed = cpu_embed
        self.target_device = target_device

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.cpu_embed(input_ids.to("cpu"))
        return embeds.to(self.target_device)


def extract_boxed_answer(text: str) -> str:
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
            for adapter_name in module.scaling:
                alpha = module.lora_alpha[adapter_name]
                r = module.r[adapter_name]
                module.scaling[adapter_name] = (alpha / r) * scale


def safe_write_json(payload: Any, path: str):
    targets = [path, os.path.join(tempfile.gettempdir(), os.path.basename(path))]
    for t in targets:
        try:
            with open(t, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return t
        except PermissionError:
            continue
    raise PermissionError(f"Failed writing {path}")


def run_env_capture(args):
    sys.stdout.reconfigure(encoding="utf-8")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    out_json = args.output_json or os.path.join(SFT_DIR, f"env_equiv_{args.tag}.json")
    print("=" * 80)
    print(f"ENV EQUIVALENCE CAPTURE [{args.tag}] | Python {sys.version.split()[0]} | Q{args.start_idx}..Q{args.end_idx}")
    print("=" * 80, flush=True)

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

    # Checkpoint resume support
    done_by_idx: Dict[int, Dict[str, Any]] = {}
    for cand in (out_json, os.path.join(tempfile.gettempdir(), os.path.basename(out_json))):
        if os.path.exists(cand):
            try:
                prev = json.load(open(cand, "r", encoding="utf-8"))
                for row in prev.get("details", []):
                    done_by_idx[int(row["index"])] = row
            except Exception:
                pass

    results: List[Dict[str, Any]] = [done_by_idx[it["index"]] for it in items if it["index"] in done_by_idx]
    if len(results) == len(items):
        print(f"[RESUME] All {len(items)} questions already captured in {out_json}.", flush=True)
        return

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
        base_model.model.embed_tokens = _CPUEmbeddingWrapper(embed_cpu, target_device="cuda:0")
        type(base_model).device = property(lambda _m, d=target_dev: d)
    if hasattr(base_model, "lm_head"):
        base_model.lm_head = base_model.lm_head.to(torch.float16)
    torch.cuda.empty_cache()

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    set_lora_scale(model, 0.25)
    model.eval()
    print(f"[GPU INIT] Ready | VRAM: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MiB", flush=True)

    for it in items:
        idx = it["index"]
        if idx in done_by_idx:
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
        gen_ids = [int(x) for x in out[0][inputs.input_ids.shape[1]:].tolist()]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred = extract_boxed_answer(gen_text)
        is_ok = bool(NormalizedMathMatcher.check_match(pred, it["ground_truth"])[1])

        row = {
            "index": idx,
            "unique_id": it["unique_id"],
            "ground_truth": it["ground_truth"],
            "predicted": pred,
            "frozen_correct": is_ok,
            "tokens_gen": len(gen_ids),
            "gen_text_sha256": hashlib.sha256(gen_text.encode("utf-8")).hexdigest(),
            "gen_token_ids": gen_ids,
            "elapsed_sec": round(dt, 2),
        }
        results.append(row)
        results.sort(key=lambda r: r["index"])
        done_by_idx[idx] = row

        payload = {
            "tag": args.tag,
            "runtime_metadata": collect_runtime_metadata(),
            "start_idx": args.start_idx,
            "end_idx": args.end_idx,
            "max_new_tokens": args.max_new_tokens,
            "completed": len(results),
            "frozen_correct_count": sum(1 for r in results if r["frozen_correct"]),
            "details": results,
        }
        saved_at = safe_write_json(payload, out_json)
        print(
            f"[{len(results):02d}/{len(items)}] Q{idx:03d} | ok={is_ok} | tok={len(gen_ids):4d} | "
            f"{dt:5.1f}s | pred={pred!r} (saved -> {os.path.basename(saved_at)})",
            flush=True,
        )


def _resolve_existing(path: str) -> str:
    if os.path.exists(path):
        return path
    fb = os.path.join(tempfile.gettempdir(), os.path.basename(path))
    return fb if os.path.exists(fb) else path


def compare_envs(py310_json: str, py313_json: str, report_json: str):
    sys.stdout.reconfigure(encoding="utf-8")
    d10 = json.load(open(_resolve_existing(py310_json), "r", encoding="utf-8"))
    d13 = json.load(open(_resolve_existing(py313_json), "r", encoding="utf-8"))
    b_map = {}
    if os.path.exists(BASELINE_JSON):
        b_data = json.load(open(BASELINE_JSON, "r", encoding="utf-8"))
        for r in b_data.get("details", []):
            b_map[str(r.get("unique_id", ""))] = r

    map10 = {r["index"]: r for r in d10.get("details", [])}
    map13 = {r["index"]: r for r in d13.get("details", [])}
    common = sorted(set(map10.keys()) & set(map13.keys()))

    token_identical = 0
    pred_identical = 0
    verdict_identical = 0
    first_diffs = []
    vs_stored_10_ok = 0
    vs_stored_13_ok = 0
    per_item = []

    for idx in common:
        r10 = map10[idx]
        r13 = map13[idx]
        t10 = r10["gen_token_ids"]
        t13 = r13["gen_token_ids"]
        tok_eq = (t10 == t13)
        if tok_eq:
            token_identical += 1
        else:
            first_diff = next((i for i, (a, b) in enumerate(zip(t10, t13)) if a != b), min(len(t10), len(t13)))
            first_diffs.append(first_diff)
        p_eq = (r10["predicted"] == r13["predicted"]) or bool(
            NormalizedMathMatcher.check_match(r10["predicted"], r13["predicted"])[1]
        )
        v_eq = (bool(r10["frozen_correct"]) == bool(r13["frozen_correct"]))
        if p_eq:
            pred_identical += 1
        if v_eq:
            verdict_identical += 1

        sb = b_map.get(r10["unique_id"], {})
        sb_ok = bool(NormalizedMathMatcher.check_match(str(sb.get("predicted", "")), r10["ground_truth"])[1]) if sb else None
        if sb_ok is not None:
            if sb_ok == bool(r10["frozen_correct"]):
                vs_stored_10_ok += 1
            if sb_ok == bool(r13["frozen_correct"]):
                vs_stored_13_ok += 1

        per_item.append({
            "index": idx,
            "unique_id": r10["unique_id"],
            "token_identical": tok_eq,
            "first_diff_pos": None if tok_eq else first_diff,
            "len_py310": len(t10),
            "len_py313": len(t13),
            "pred_py310": r10["predicted"],
            "pred_py313": r13["predicted"],
            "ok_py310": bool(r10["frozen_correct"]),
            "ok_py313": bool(r13["frozen_correct"]),
            "stored_baseline_pred": sb.get("predicted"),
            "stored_baseline_ok": sb_ok,
        })

    summary = {
        "n_common": len(common),
        "runtime_py310": d10.get("runtime_metadata"),
        "runtime_py313": d13.get("runtime_metadata"),
        "token_identical_count": token_identical,
        "token_identical_rate": round(token_identical / max(1, len(common)), 4),
        "pred_equiv_count": pred_identical,
        "pred_equiv_rate": round(pred_identical / max(1, len(common)), 4),
        "verdict_identical_count": verdict_identical,
        "py310_correct": sum(1 for idx in common if map10[idx]["frozen_correct"]),
        "py313_correct": sum(1 for idx in common if map13[idx]["frozen_correct"]),
        "verdict_match_vs_stored_baseline": {"py310": vs_stored_10_ok, "py313": vs_stored_13_ok},
        "mean_first_diff_token_pos": round(sum(first_diffs) / len(first_diffs), 2) if first_diffs else None,
        "recommendation": (
            "ENV_EQUIVALENT"
            if token_identical / max(1, len(common)) >= 0.90 and verdict_identical >= len(common) - 1
            else "REGENERATE_BASELINE_IN_PY313"
        ),
        "details": per_item,
    }
    saved = safe_write_json(summary, report_json)
    print("=" * 80)
    print(f"ENV EQUIVALENCE SUMMARY (N={len(common)}) -> {saved}")
    print(f"Token-identical: {token_identical}/{len(common)} ({summary['token_identical_rate']*100:.1f}%)")
    print(f"Pred-equiv:      {pred_identical}/{len(common)} ({summary['pred_equiv_rate']*100:.1f}%)")
    print(f"Verdict-match:   {verdict_identical}/{len(common)} | Py310={summary['py310_correct']} vs Py313={summary['py313_correct']}")
    print(f"Recommendation:  {summary['recommendation']}")
    print("=" * 80, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 0 Environment Equivalence Test")
    parser.add_argument("--tag", type=str, choices=["py310", "py313", "compare"], required=True)
    parser.add_argument("--start_idx", type=int, default=26)
    parser.add_argument("--end_idx", type=int, default=55)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--py310_json", type=str, default=os.path.join(SFT_DIR, "env_equiv_py310.json"))
    parser.add_argument("--py313_json", type=str, default=os.path.join(SFT_DIR, "env_equiv_py313.json"))
    parser.add_argument("--report_json", type=str, default=os.path.join(SFT_DIR, "ENV_EQUIVALENCE_REPORT_PY310_VS_PY313.json"))
    args = parser.parse_args()

    if args.tag == "compare":
        compare_envs(args.py310_json, args.py313_json, args.report_json)
    else:
        run_env_capture(args)


if __name__ == "__main__":
    main()
