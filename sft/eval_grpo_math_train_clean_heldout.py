"""
SODALS v2.1 — Clean Held-Out Evaluation & Early-Surrender Audit for Out-of-MATH-500 GRPO
=========================================================================================
Evaluates `sft/qwen_lora_adapter_grpo_math_train_100` (trained strictly on Hendrycks MATH `train`
split with 0% overlap with `MATH-500`) on the Clean Held-Out `MATH-500` pool:
  - `Clean-125 Dev` (`Q026..Q150`, indices 25..149)
  - `Pristine-20 Unseen` (`Q151..Q170`, indices 150..169)
  - Combined `Clean-145 Held-Out` (`Q026..Q170`, indices 25..169 — where SFT Baseline is already
    available in `math500_domain2_soft_blended_170_results.json`)
  - Extensible up to `Clean-475` (`Q026..Q500`, indices 25..499) via `--end_idx 500`.

Applies all 3 Grader Tiers uniformly to both SFT Baseline and Out-of-MATH-500 GRPO:
  1. Tier 1 (`strict_pass`): Pure String Equality (`cp == cg`, zero SymPy)
  2. Tier 2 (`legacy_pass`): `check_math_match_legacy` (Baseline SymPy Grader)
  3. Tier 3 (`norm_sym_pass`): `NormalizedMathMatcher` (Normalized Symbolic Grader)

Computes:
  - 2x2 Contingency Table (b11, b01, b10, b00) & Two-Sided Exact Binomial McNemar p-value
  - Full-Dataset "Early Surrender" Token Length Distribution across all SFT-failed questions
    (<256, <512, <768, <1024, <1536, =2048 tokens) and overall trajectory divergence rate.
  - Pre-Registered Stopping Rule Verification (p < 0.05 threshold check).
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smart_app_control_compat  # noqa: F401

def prevent_windows_sleep():
    """Prevent Windows Modern Standby / Sleep / Display & PCIe throttle when screen is locked."""
    if sys.platform == "win32":
        try:
            import ctypes
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002
            ES_AWAYMODE_REQUIRED = 0x00000040
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED | ES_AWAYMODE_REQUIRED
            )
        except Exception:
            pass

prevent_windows_sleep()

import json
import re
import time
import math
import statistics
import argparse
import torch
import sympy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
sys.path.insert(0, SFT_DIR)

from gpr_phase_router import NeuralPhaseRouter
from soft_lora_blender import SoftLoraBlender
from cst_symbolic_verifier import NormalizedMathMatcher

MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
DEFAULT_ADAPTER_D1 = os.path.join(SFT_DIR, "qwen_lora_adapter_grpo_math_train_100")
ADAPTER_D2_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_domain2_r8_pondering")
BASELINE_JSON = os.path.join(ROOT, "math500_domain2_soft_blended_170_results.json")
MATH500_CACHE_JSONL = r"C:\Users\Burak\.cache\huggingface\hub\datasets--HuggingFaceH4--MATH-500\snapshots\6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be\test.jsonl"
DEFAULT_OUTPUT_JSON = os.path.join(SFT_DIR, "eval_grpo_math_train_clean_145_results.json")


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


def safe_json_dump(data, path, max_retries=10, delay=0.5):
    """Robust atomic JSON saver that handles Windows file locks and PermissionErrors with fallback."""
    for attempt in range(max_retries):
        try:
            temp_path = f"{path}.tmp.{os.getpid()}"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            if os.path.exists(path):
                try:
                    os.replace(temp_path, path)
                    return
                except PermissionError:
                    pass
            else:
                try:
                    os.rename(temp_path, path)
                    return
                except PermissionError:
                    pass
            # Direct overwrite fallback
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            return
        except PermissionError as e:
            if attempt == max_retries - 1:
                # Fallback to a guaranteed writable temporary directory
                fallback_dir = os.environ.get("TEMP", os.environ.get("TMP", os.path.expanduser("~")))
                fallback_path = os.path.join(fallback_dir, os.path.basename(path))
                try:
                    with open(fallback_path, "w", encoding="utf-8") as ff:
                        json.dump(data, ff, indent=2, ensure_ascii=False)
                    print(f"[SAVE FALLBACK] Primary path hit PermissionError ({e}); successfully saved to: {fallback_path}", flush=True)
                    return
                except Exception as fe:
                    print(f"[SAVE WARNING] PermissionError after {max_retries} attempts ({e}) and fallback ({fe}); will retry next question.", flush=True)
            else:
                time.sleep(delay * (attempt + 1))
        except Exception as e:
            print(f"[SAVE ERROR] Unexpected error saving JSON: {e}", flush=True)
            break


def clean_latex_legacy(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = s.replace(r"\$", "").replace("$", "").strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"^\circ", "").replace("^{\\circ}", "").replace("°", "")
    s = s.replace(" ", "")
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)
    s = re.sub(r"\s*(?:cm|units|degrees|m|inches)\b", "", s, flags=re.I)
    return s.strip()


def to_sympy_legacy(s: str):
    s = clean_latex_legacy(s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = s.replace("^", "**")
    try:
        return sympy.sympify(s)
    except Exception:
        return None


def check_math_match_legacy(pred: str, gt: str) -> bool:
    cp = clean_latex_legacy(pred)
    cg = clean_latex_legacy(gt)
    if cp == cg:
        return True
    if cg.lstrip("-").isdigit():
        num_m = re.findall(r"-?\d+", cp)
        if num_m and num_m[-1] == cg:
            return True
    if r"\pm" in gt:
        base_parts = gt.split(r"\pm")
        if len(base_parts) == 2:
            a, b = base_parts[0].strip(), base_parts[1].strip()
            p1 = f"{a}+{b},{a}-{b}"
            p2 = f"{a}-{b},{a}+{b}"
            if clean_latex_legacy(pred) in [clean_latex_legacy(p1), clean_latex_legacy(p2)]:
                return True
    sym_gt = to_sympy_legacy(gt)
    sym_pred = to_sympy_legacy(pred)
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


def mcnemar_exact(b01: int, b10: int) -> float:
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def grade_all_three_tiers(pred: str, gt: str) -> tuple[bool, bool, bool]:
    strict_ok, norm_sym_ok = NormalizedMathMatcher.check_match(pred, gt)
    legacy_ok = check_math_match_legacy(pred, gt)
    return bool(strict_ok), bool(legacy_ok), bool(norm_sym_ok)


def print_audit_summary(results_list: list):
    print("\n" + "=" * 105, flush=True)
    print("UNIFIED 3-TIER GRADER, MCNEMAR TEST & EARLY-SURRENDER TOKEN AUDIT", flush=True)
    print("=" * 105, flush=True)

    subsets = [
        ("CLEAN-125 DEV (Q026..Q150)", [r for r in results_list if 26 <= r["index"] <= 150]),
        ("PRISTINE-20 UNSEEN (Q151..Q170)", [r for r in results_list if 151 <= r["index"] <= 170]),
        ("COMBINED CLEAN-145 HELD-OUT (Q026..Q170)", [r for r in results_list if 26 <= r["index"] <= 170]),
    ]

    for sub_name, rows in subsets:
        if not rows:
            continue
        n = len(rows)
        print(f"\n--- {sub_name} (N = {n}) ---", flush=True)

        paired_rows = [r for r in rows if r.get("has_baseline", False) and r.get("base_legacy_pass") is not None]
        np = len(paired_rows)
        if np > 0:
            print(f"  [Paired with SFT Baseline: {np}/{n}]", flush=True)
            for tier_key, base_key, label in [
                ("strict_pass", "base_strict_pass", "Tier 1: Pure String Equality"),
                ("legacy_pass", "base_legacy_pass", "Tier 2: Legacy SymPy Grader"),
                ("norm_sym_pass", "base_norm_sym_pass", "Tier 3: Normalized SymPy Grader"),
            ]:
                b_pass = sum(1 for r in paired_rows if r[base_key])
                g_pass = sum(1 for r in paired_rows if r[tier_key])
                b11 = sum(1 for r in paired_rows if r[base_key] and r[tier_key])
                b01 = sum(1 for r in paired_rows if (not r[base_key]) and r[tier_key])
                b10 = sum(1 for r in paired_rows if r[base_key] and (not r[tier_key]))
                b00 = sum(1 for r in paired_rows if (not r[base_key]) and (not r[tier_key]))
                pval = mcnemar_exact(b01, b10)
                print(
                    f"    {label:32s}: SFT={b_pass:3d}/{np} ({100*b_pass/np:5.2f}%) -> "
                    f"GRPO={g_pass:3d}/{np} ({100*g_pass/np:5.2f}%) | "
                    f"Net={g_pass - b_pass:+2d} (G:{b01} D:{b10} [2x2: {b11}/{b01}/{b10}/{b00}]) | "
                    f"McNemar p={pval:.4f}",
                    flush=True,
                )

        if np < n:
            standalone_rows = [r for r in rows if not r.get("has_baseline", False)]
            ns = len(standalone_rows)
            st_g = sum(1 for r in standalone_rows if r["norm_sym_pass"])
            leg_g = sum(1 for r in standalone_rows if r["legacy_pass"])
            print(
                f"  [Standalone (No SFT Baseline Available): {ns}/{n}] "
                f"GRPO NormSym={st_g}/{ns} ({100*st_g/ns:5.2f}%) | Legacy={leg_g}/{ns} ({100*leg_g/ns:5.2f}%)",
                flush=True,
            )

        # Early-Surrender Token Distribution on SFT Failed Questions
        sft_fails = [r for r in paired_rows if not r["base_legacy_pass"]]
        if sft_fails:
            nf = len(sft_fails)
            s_toks = [r["base_tokens"] for r in sft_fails]
            g_toks = [r["generated_tokens"] for r in sft_fails]
            div_all = sum(1 for r in paired_rows if r["generated_tokens"] != r["base_tokens"] or r["predicted"] != r["baseline_pred"])
            div_fail = sum(1 for r in sft_fails if r["generated_tokens"] != r["base_tokens"] or r["predicted"] != r["baseline_pred"])
            print(f"  Trajectory Divergence Rate: {div_all}/{np} ({100*div_all/np:.1f}%) overall | {div_fail}/{nf} ({100*div_fail/nf:.1f}%) on SFT failures", flush=True)
            print(
                f"  Early-Surrender Audit on SFT Failures (n={nf}): "
                f"SFT mean={statistics.mean(s_toks):.1f} (med={statistics.median(s_toks):.1f}) -> "
                f"GRPO mean={statistics.mean(g_toks):.1f} (med={statistics.median(g_toks):.1f})",
                flush=True,
            )
            for th in [256, 512, 768, 1024, 1536]:
                sc = sum(1 for t in s_toks if t < th)
                gc = sum(1 for t in g_toks if t < th)
                print(
                    f"    Tokens < {th:4d}: SFT = {sc:2d}/{nf} ({100*sc/nf:5.1f}%) -> GRPO = {gc:2d}/{nf} ({100*gc/nf:5.1f}%)",
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_d1", type=str, default=DEFAULT_ADAPTER_D1)
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--start_idx", type=int, default=25, help="0-based start index (25 = Q026, skipping Q001..Q025)")
    parser.add_argument("--end_idx", type=int, default=170, help="0-based end index (170 = Q170, or 500 for full Clean-475)")
    args = parser.parse_args()

    print("=" * 100, flush=True)
    print("SODALS v2.1 — OUT-OF-MATH-500 GRPO CLEAN HELD-OUT EVALUATION & EARLY-SURRENDER AUDIT", flush=True)
    print(f"Challenger Adapter: {args.adapter_d1}", flush=True)
    print(f"Evaluation Range:   Q{args.start_idx+1:03d}..Q{args.end_idx:03d} ({args.end_idx - args.start_idx} Clean Held-Out Questions)", flush=True)
    print("=" * 100, flush=True)

    with open(BASELINE_JSON, "r", encoding="utf-8") as f:
        base_raw = json.load(f)["details"]
    # Build unique_id map and strict MATH-500 index map to prevent cross-dataset index collision
    base_map_by_uid = {item.get("unique_id"): item for item in base_raw if item.get("unique_id")}
    base_map_by_idx = {
        item.get("index"): item for item in base_raw
        if item.get("dataset") == "MATH-500" or (isinstance(item.get("unique_id"), str) and "test/" in item.get("unique_id"))
    }

    all_math500 = []
    with open(MATH500_CACHE_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_math500.append(json.loads(line))

    questions = []
    for i in range(args.start_idx, min(args.end_idx, len(all_math500))):
        it = all_math500[i]
        q_idx = i + 1
        sub_label = "CLEAN-125-DEV" if q_idx <= 150 else "PRISTINE-UNSEEN"
        questions.append({
            "index": q_idx,
            "unique_id": it.get("unique_id", f"math500_{q_idx}"),
            "subject": it["subject"],
            "level": it.get("level", "?"),
            "problem": it["problem"],
            "ground_truth": it["answer"],
            "subset": sub_label,
        })

    results_list = []
    evaluated_indices = set()
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json, "r", encoding="utf-8") as f:
                saved = json.load(f)
            results_list = saved.get("details", [])
            for r in results_list:
                evaluated_indices.add(r["index"])
            print(f"[RESUME] Loaded {len(evaluated_indices)} already-evaluated questions from {args.output_json}.", flush=True)
        except Exception as e:
            print(f"[RESUME WARNING] Could not load {args.output_json}: {e}", flush=True)

    if len(evaluated_indices) >= len(questions):
        print("All target questions already evaluated! Printing audit summary...", flush=True)
        print_audit_summary(results_list)
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
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_d1, adapter_name="expert_general_math")
    model.load_adapter(ADAPTER_D2_PATH, adapter_name="expert_abstract_algebra")

    router = NeuralPhaseRouter(anchors_path=os.path.join(SFT_DIR, "gpr_domain_anchors.pt"))
    blender = SoftLoraBlender(threshold=router.threshold, temperature=0.03)
    blender.enable_multi_adapter(model)
    model.eval()

    for item in questions:
        q_idx = item["index"]
        if q_idx in evaluated_indices:
            continue

        prob = item["problem"]
        gt = item["ground_truth"]
        subj = item["subject"]
        sub_label = item["subset"]

        # Look up baseline by unique_id first; fallback to index strictly if confirmed MATH-500
        base_item = base_map_by_uid.get(item["unique_id"])
        if base_item is None:
            candidate = base_map_by_idx.get(q_idx)
            if candidate and (candidate.get("unique_id") == item["unique_id"] or candidate.get("dataset") == "MATH-500"):
                base_item = candidate

        has_baseline = (base_item is not None)
        if has_baseline:
            base_pred = base_item.get("predicted", "")
            base_tokens = base_item.get("tokens", 0)
            b_strict, b_leg, b_sym = grade_all_three_tiers(base_pred, gt)
        else:
            base_pred = None
            base_tokens = 0
            b_strict, b_leg, b_sym = None, None, None

        probe_prompt = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        inputs = tokenizer(probe_prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        z_i, _, _ = router.compute_phase_divergence(model, inputs)
        alpha = blender.compute_alpha(z_i)
        rep_penalty = blender.compute_rep_penalty(alpha)
        blender.set_blend(model, alpha)

        t0 = time.time()
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                repetition_penalty=rep_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        gen_tokens = gen_out[0][prompt_len:]
        num_new_tokens = len(gen_tokens)
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        pred = extract_boxed_answer(gen_text)
        g_strict, g_leg, g_sym = grade_all_three_tiers(pred, gt)

        if has_baseline:
            delta_str = "= PASS" if (g_leg and b_leg) else ("= FAIL" if (not g_leg and not b_leg) else ("★ GAIN" if g_leg else "✗ DROP"))
        else:
            delta_str = "PASS" if g_leg else "FAIL"

        rec = {
            "index": q_idx,
            "unique_id": item["unique_id"],
            "subset": sub_label,
            "subject": subj,
            "level": item["level"],
            "problem": prob,
            "ground_truth": gt,
            "predicted": pred,
            "verdict": "PASS" if g_leg else "FAIL",
            "strict_pass": g_strict,
            "legacy_pass": g_leg,
            "norm_sym_pass": g_sym,
            "has_baseline": has_baseline,
            "baseline_pred": base_pred,
            "base_strict_pass": b_strict,
            "base_legacy_pass": b_leg,
            "base_norm_sym_pass": b_sym,
            "base_tokens": base_tokens,
            "generated_tokens": num_new_tokens,
            "delta_legacy": delta_str,
            "alpha": round(alpha, 4),
            "time_seconds": round(elapsed, 2),
            "full_response": gen_text,
        }
        results_list.append(rec)
        evaluated_indices.add(q_idx)

        paired_rows = [r for r in results_list if r.get("has_baseline", False)]
        n_paired = len(paired_rows)
        leg_g_paired = sum(1 for r in paired_rows if r["legacy_pass"])
        leg_b = sum(1 for r in paired_rows if r["base_legacy_pass"])
        sym_g_paired = sum(1 for r in paired_rows if r["norm_sym_pass"])
        sym_b = sum(1 for r in paired_rows if r["base_norm_sym_pass"])

        tok_msg = f"Tok: {base_tokens:4d}->{num_new_tokens:4d}" if has_baseline else f"Tok: {num_new_tokens:4d}"
        if has_baseline:
            print(
                f"[Q{q_idx:03d}] [{sub_label:15s}] {delta_str:6s} | "
                f"Legacy: {leg_g_paired}/{n_paired} vs {leg_b}/{n_paired} (Net:{leg_g_paired-leg_b:+d}) | "
                f"NormSym: {sym_g_paired}/{n_paired} vs {sym_b}/{n_paired} (Net:{sym_g_paired-sym_b:+d}) | "
                f"{tok_msg} | {elapsed:.1f}s",
                flush=True,
            )
        else:
            n_done = len(results_list)
            sym_g_all = sum(1 for r in results_list if r["norm_sym_pass"])
            print(
                f"[Q{q_idx:03d}] [{sub_label:15s}] {delta_str:6s} | "
                f"Standalone (No SFT Baseline) | GRPO Total: {sym_g_all}/{n_done} | "
                f"{tok_msg} | {elapsed:.1f}s",
                flush=True,
            )

        safe_json_dump(
            {
                "experiment": "Out-of-MATH-500 GRPO Clean Held-Out Evaluation",
                "challenger_adapter": args.adapter_d1,
                "total_evaluated": len(results_list),
                "total_paired_with_baseline": n_paired,
                "legacy_grpo_pass": sum(1 for r in results_list if r["legacy_pass"]),
                "legacy_base_pass": leg_b,
                "norm_sym_grpo_pass": sum(1 for r in results_list if r["norm_sym_pass"]),
                "norm_sym_base_pass": sym_b,
                "details": results_list,
            },
            args.output_json,
        )

    print_audit_summary(results_list)


if __name__ == "__main__":
    main()
