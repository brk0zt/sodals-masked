"""
SODALS v2.1 — Pristine-30 (Q171..Q200) Unseen MATH-500 Benchmark Evaluation
============================================================================
Evaluates 30 strictly unseen MATH-500 test questions (indices 170..199) using the
full Six-Arkhe Neurosymbolic Architecture:
  - Base Model: Qwen 3.5 9B (4-bit NF4 quantized + CPUEmbeddingWrapper for 8.5 tok/s)
  - Adapters: Dual LoRA (D1: Math Train 100 + D2: Domain2 Pondering)
  - Routing & Manifold: NeuralPhaseRouter + SoftLoraBlender + AdaptivePonderingODEModule
  - Verifiers: LeanTacticStepVerifier (ring, linarith_subst, norm_num, no_sorry) + CstSymbolicVerifier
  - Pipeline:
      * Arkhe 0 (Durum): Immutable ArkheState & CandidateTrace monads
      * Arkhe 1 (Bilinç): Layer-32 GPR Phase Routing alpha in [0, 1]
      * Arkhe 3 (Çıkarım): Zero-leakage Pass-1 structural & Lean tactic trigger
      * Arkhe 2 (Benzerlik): Multi-sample semantic equivalence clustering (NormalizedMathMatcher)
      * Arkhe 4 (Uyum): Lens-Free Neurosymbolic PRM scoring
      * Arkhe 5 & 6 (Yöntem & Şartlar): Tier-1 Selective Abstention Gate (tau* = 0.20)
  - Baseline Comparison: Direct pairwise comparison against SFT Baseline (`math500_full500_lora_4096_results.json`).
"""

import os
import sys

if "--online" not in sys.argv:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import re
import json
import time
import tempfile
import argparse
import torch

sys.stdout.reconfigure(encoding="utf-8")

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SFT_DIR)
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

import smart_app_control_compat  # noqa: F401
from gpr_phase_router import NeuralPhaseRouter
from soft_lora_blender import SoftLoraBlender
from adaptive_pondering_ode import AdaptivePonderingODEModule
from cst_symbolic_verifier import CstSymbolicVerifier, NormalizedMathMatcher, LeanTacticStepVerifier
from env_reproducibility import collect_runtime_metadata
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
ANCHORS_PATH = os.path.join(SFT_DIR, "gpr_domain_anchors.pt")
MATH500_JSONL = r"C:\Users\Burak\.cache\huggingface\hub\datasets--HuggingFaceH4--MATH-500\snapshots\6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be\test.jsonl"
BASELINE_JSON = os.path.join(ROOT_DIR, "math500_full500_lora_4096_results.json")
OUTPUT_JSON = os.path.join(SFT_DIR, "pristine30_q171_q200_arkhe_results.json")
FALLBACK_OUTPUT_JSON = os.path.join(tempfile.gettempdir(), "pristine30_q171_q200_arkhe_results.json")


def prevent_windows_sleep():
    """Prevent Windows Modern Standby / Sleep during long benchmark runs."""
    if sys.platform == "win32":
        try:
            import ctypes
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
        except Exception:
            pass


class CPUEmbeddingWrapper(torch.nn.Module):
    """Offloads the 248,320 x 4,096 token embedding table to CPU RAM, cutting VRAM by 1.65 GB."""
    def __init__(self, cpu_embed: torch.nn.Module, target_device: str = "cuda:0"):
        super().__init__()
        self.cpu_embed = cpu_embed
        self.target_device = target_device

    @property
    def weight(self):
        return self.cpu_embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.cpu_embed(input_ids.to("cpu"))
        return embeds.to(device=self.target_device, dtype=torch.float16)


def extract_boxed_answer_fixed(text: str) -> str:
    """Decimal-safe answer extractor."""
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


def build_trace(branch_id: str, text: str, gen_tokens: int = 0, prompt: str = "") -> CandidateTrace:
    ans = extract_boxed_answer_fixed(text)
    has_end = "</think>" in text
    is_val, eq_c, contras = CstSymbolicVerifier.verify_chunk(text)
    step_report = LeanTacticStepVerifier.verify_proof_steps(text, prompt=prompt)
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


def _atomic_write_json(data, target_path, max_retries=8, delay=0.3):
    tmp_path = target_path + f".tmp_{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    for attempt in range(max_retries):
        try:
            os.replace(tmp_path, target_path)
            return target_path
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                try:
                    with open(target_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    return target_path
                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass


def safe_json_dump(data, path, fallback_path=FALLBACK_OUTPUT_JSON, max_retries=8, delay=0.3):
    """Writes checkpoint JSON atomically, falling back to tempfile.gettempdir() on PermissionError."""
    try:
        return _atomic_write_json(data, path, max_retries=max_retries, delay=delay)
    except PermissionError:
        if fallback_path and os.path.abspath(fallback_path) != os.path.abspath(path):
            print(
                f"  [WARN] PermissionError writing to {path}; switching checkpoint path to {fallback_path}",
                flush=True,
            )
            return _atomic_write_json(data, fallback_path, max_retries=max_retries, delay=delay)
        raise


def rescore_checkpoint(output_json_arg, fallback_path=FALLBACK_OUTPUT_JSON):
    """Re-grades an existing saved Pristine-30 checkpoint offline with updated NormalizedMathMatcher."""
    candidates = [p for p in [output_json_arg, fallback_path] if p and os.path.exists(p)]
    if not candidates:
        raise FileNotFoundError(
            f"No Pristine-30 checkpoint found at {output_json_arg!r} or {fallback_path!r}"
        )

    best_path = None
    best_data = None
    for cand in candidates:
        with open(cand, "r", encoding="utf-8") as f:
            d = json.load(f)
        if best_data is None or len(d.get("details", [])) > len(best_data.get("details", [])):
            best_path = cand
            best_data = d

    old_metrics = dict(best_data.get("metrics", {}))
    details = best_data.get("details", [])

    pass1_ok_cnt = 0
    forced_ok_cnt = 0
    sel_answered_cnt = 0
    sel_ok_cnt = 0
    base_ok_cnt = 0
    changed_rows = []

    for rec in details:
        q_idx = rec.get("index")
        gt = rec.get("ground_truth", "")
        p1_pred = rec.get("pass1_predicted", "")
        win_pred = rec.get("winning_predicted", "")
        base_pred = rec.get("baseline_predicted", "")

        old_p1 = bool(rec.get("pass1_correct", False))
        old_forced = bool(rec.get("arkhe_forced_correct", False))
        old_base = bool(rec.get("baseline_correct", False))

        new_p1 = bool(NormalizedMathMatcher.check_match(p1_pred, gt)[1])
        new_forced = bool(NormalizedMathMatcher.check_match(win_pred, gt)[1])
        new_base = old_base or bool(NormalizedMathMatcher.check_match(base_pred, gt)[1])

        if (new_p1, new_forced, new_base) != (old_p1, old_forced, old_base):
            changed_rows.append((q_idx, gt, p1_pred, old_p1, new_p1, win_pred, old_forced, new_forced, base_pred, old_base, new_base))

        rec["pass1_correct"] = new_p1
        rec["arkhe_forced_correct"] = new_forced
        rec["baseline_correct"] = new_base

        pass1_ok_cnt += int(new_p1)
        forced_ok_cnt += int(new_forced)
        if rec.get("arkhe_verdict") != DecisionVerdict.ABSTAIN_UNKNOWN.name:
            sel_answered_cnt += 1
            sel_ok_cnt += int(new_forced)
        base_ok_cnt += int(new_base)

    n = len(details)
    cur_cov = (sel_answered_cnt / max(1, n)) * 100.0
    cur_sel_acc = (sel_ok_cnt / max(1, sel_answered_cnt)) * 100.0
    cur_forced_acc = (forced_ok_cnt / max(1, n)) * 100.0
    cur_p1_acc = (pass1_ok_cnt / max(1, n)) * 100.0
    cur_base_acc = (base_ok_cnt / max(1, n)) * 100.0

    best_data["processed_questions"] = n
    best_data["metrics"] = {
        "baseline_sft_correct": base_ok_cnt,
        "baseline_sft_acc": cur_base_acc,
        "pass1_greedy_correct": pass1_ok_cnt,
        "pass1_greedy_acc": cur_p1_acc,
        "arkhe_forced_correct": forced_ok_cnt,
        "arkhe_forced_acc": cur_forced_acc,
        "arkhe_selective_correct": sel_ok_cnt,
        "arkhe_selective_answered": sel_answered_cnt,
        "arkhe_selective_acc": cur_sel_acc,
        "coverage_pct": cur_cov,
    }

    saved_path = safe_json_dump(best_data, best_path, fallback_path=fallback_path)

    print("=" * 80)
    print(f"PRISTINE-30 OFFLINE RE-SCORING COMPLETE ({n} questions)")
    print(f"Checkpoint loaded : {best_path}")
    print(f"Checkpoint updated: {saved_path}")
    print("-" * 80)
    print(
        f"BEFORE -> Pass-1: {old_metrics.get('pass1_greedy_correct')}/{n} ({old_metrics.get('pass1_greedy_acc', 0.0):.2f}%) | "
        f"Arkhe Forced: {old_metrics.get('arkhe_forced_correct')}/{n} ({old_metrics.get('arkhe_forced_acc', 0.0):.2f}%) | "
        f"Selective: {old_metrics.get('arkhe_selective_correct')}/{old_metrics.get('arkhe_selective_answered')} ({old_metrics.get('arkhe_selective_acc', 0.0):.2f}% @ {old_metrics.get('coverage_pct', 0.0):.1f}% cov) | "
        f"Baseline: {old_metrics.get('baseline_sft_correct')}/{n} ({old_metrics.get('baseline_sft_acc', 0.0):.2f}%)"
    )
    print(
        f"AFTER  -> Pass-1: {pass1_ok_cnt}/{n} ({cur_p1_acc:.2f}%) | "
        f"Arkhe Forced: {forced_ok_cnt}/{n} ({cur_forced_acc:.2f}%) | "
        f"Selective: {sel_ok_cnt}/{sel_answered_cnt} ({cur_sel_acc:.2f}% @ {cur_cov:.1f}% cov) | "
        f"Baseline: {base_ok_cnt}/{n} ({cur_base_acc:.2f}%)"
    )
    if changed_rows:
        print("Changed verdicts:")
        for row in changed_rows:
            q_idx, gt, p1_p, op1, np1, wp, ofo, nfo, bp, ob, nb = row
            print(
                f"  Q{q_idx:03d} GT={gt!r}: Pass1={p1_p!r} ({op1}->{np1}), "
                f"Arkhe={wp!r} ({ofo}->{nfo}), Baseline={bp!r} ({ob}->{nb})"
            )
    else:
        print("Changed verdicts: none")
    print("=" * 80)
    return saved_path, best_data


def main():
    prevent_windows_sleep()

    parser = argparse.ArgumentParser(description="Pristine-30 (Q171..Q200) MATH-500 Unseen Benchmark")
    parser.add_argument("--tau", type=float, default=0.20, help="Tier-1 selective abstention threshold")
    parser.add_argument("--start_idx", type=int, default=171, help="Start question index (1-based, default 171)")
    parser.add_argument("--end_idx", type=int, default=200, help="End question index (1-based, inclusive, default 200)")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum new tokens per candidate rollout")
    parser.add_argument("--output_json", type=str, default=OUTPUT_JSON, help="Primary output JSON path (falls back to tempdir on PermissionError)")
    parser.add_argument("--rescore_only", action="store_true", help="Re-grade saved checkpoint offline without running GPU inference")
    parser.add_argument("--logprob_threshold", type=float, default=None, help="Trigger G=4 multi-branch when Pass-1 min_step_logprob < threshold (e.g. -0.75)")
    parser.add_argument("--indices", type=str, default="", help="Optional comma-separated 1-based MATH-500 indices (overrides start_idx..end_idx range filter)")
    args = parser.parse_args()

    if args.rescore_only:
        rescore_fb = (
            FALLBACK_OUTPUT_JSON
            if os.path.abspath(args.output_json) == os.path.abspath(OUTPUT_JSON)
            else os.path.join(tempfile.gettempdir(), os.path.basename(args.output_json))
        )
        rescore_checkpoint(args.output_json, fallback_path=rescore_fb)
        return

    print("=" * 80)
    print(f"SODALS v2.1 — PRISTINE UNSEEN BENCHMARK EVALUATION (Q{args.start_idx}..Q{args.end_idx})")
    print(f"Six-Arkhe Pipeline + Lean Tactic Verifier + SoftLoRA + RK4 Manifold Pondering")
    print(f"Tau* Threshold: {args.tau:.2f} | Max Tokens: {args.max_new_tokens} | Logprob Gate: {args.logprob_threshold}")
    print("=" * 80)

    # 1. Load MATH-500 test questions
    with open(MATH500_JSONL, "r", encoding="utf-8") as f:
        all_lines = [json.loads(line) for line in f]

    explicit_indices = None
    if args.indices.strip():
        explicit_indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]

    target_items = []
    idx_iter = explicit_indices if explicit_indices is not None else list(range(args.start_idx, args.end_idx + 1))
    for one_based_idx in idx_iter:
        i = one_based_idx - 1
        if 0 <= i < len(all_lines):
            q = all_lines[i]
            target_items.append({
                "index": one_based_idx,
                "problem": q.get("problem", ""),
                "ground_truth": q.get("answer", ""),
                "subject": q.get("subject", "General Math"),
                "level": q.get("level", 3),
                "unique_id": q.get("unique_id", f"MATH_500_{one_based_idx:04d}"),
            })

    print(f"Loaded {len(target_items)} unseen test questions from MATH-500.")

    # 2. Load baseline comparison data
    baseline_map = {}
    if os.path.exists(BASELINE_JSON):
        try:
            with open(BASELINE_JSON, "r", encoding="utf-8") as f:
                b_data = json.load(f)
            for item in b_data.get("details", []):
                idx = item.get("index")
                if idx is not None:
                    baseline_map[idx] = {
                        "predicted": item.get("predicted", ""),
                        "is_correct": item.get("verdict") == "PASS",
                    }
        except Exception as e:
            print(f"Warning loading baseline JSON: {e}")

    # 3. Load Model and Adapters on GPU with CPUEmbeddingWrapper
    print("\n[GPU INIT] Loading Qwen 3.5 9B with Dual LoRA & CPUEmbeddingWrapper on cuda:0...")
    t_load_start = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_token_ids = []
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)
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

    # CPU Embedding Offload
    target_dev = torch.device("cuda:0")
    if hasattr(base_model, "model") and hasattr(base_model.model, "embed_tokens"):
        embed_cpu = base_model.model.embed_tokens.to(device="cpu", dtype=torch.float16)
        base_model.model.embed_tokens = CPUEmbeddingWrapper(embed_cpu, target_device="cuda:0")
        type(base_model).device = property(lambda _m, d=target_dev: d)
    if hasattr(base_model, "lm_head"):
        base_model.lm_head = base_model.lm_head.to(torch.float16)
    torch.cuda.empty_cache()

    # LoRA Adapters
    model = PeftModel.from_pretrained(base_model, ADAPTER_D1_PATH, adapter_name="expert_general_math")
    if os.path.exists(ADAPTER_D2_PATH):
        model.load_adapter(ADAPTER_D2_PATH, adapter_name="expert_abstract_algebra")

    router = NeuralPhaseRouter(anchors_path=ANCHORS_PATH) if os.path.exists(ANCHORS_PATH) else None
    blender = SoftLoraBlender(threshold=(router.threshold if router else 0.004913), temperature=0.03)
    blender.enable_multi_adapter(model)

    ode_module = AdaptivePonderingODEModule(hidden_dim=4096, dt=0.1, min_steps=2, max_steps=5).to("cuda:0")
    model.eval()

    vram_used = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"[GPU INIT] Ready in {time.time()-t_load_start:.1f}s | GPU VRAM Allocated: {vram_used:.1f} MiB\n")

    pipeline = ArkheNeurosymbolicPipeline(
        selective_tau=args.tau,
        matcher_fn=lambda a, b: NormalizedMathMatcher.check_match(str(a), str(b))[1],
        logprob_trigger_threshold=args.logprob_threshold,
    )
    inference_arche = InferenceArche(logprob_trigger_threshold=args.logprob_threshold)
    runtime_meta = collect_runtime_metadata()
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
        import numpy as _np
        arr = torch.cat(lps, dim=0).numpy()
        step_means = [float(arr[k : k + 32].mean()) for k in range(0, len(arr), 32) if len(arr[k : k + 32]) >= 8]
        min_step = min(step_means) if step_means else float(arr.mean())
        return float(arr.mean()), float(min_step)

    # 4. Check for existing checkpoint across primary & tempdir fallback paths to resume seamlessly
    results = []
    completed_indices = set()
    active_output_path = args.output_json
    run_fallback_json = os.path.join(tempfile.gettempdir(), os.path.basename(args.output_json))
    resume_candidates = (
        [args.output_json, OUTPUT_JSON, FALLBACK_OUTPUT_JSON]
        if os.path.abspath(args.output_json) == os.path.abspath(OUTPUT_JSON)
        else [args.output_json, run_fallback_json]
    )
    merged_by_idx = {}
    for cand_path in dict.fromkeys(resume_candidates):
        if os.path.exists(cand_path):
            try:
                with open(cand_path, "r", encoding="utf-8") as f:
                    prev_data = json.load(f)
                for r in prev_data.get("details", []):
                    r_idx = r.get("index")
                    if r_idx is not None:
                        merged_by_idx[r_idx] = r
            except Exception:
                pass
    if merged_by_idx:
        results = [merged_by_idx[k] for k in sorted(merged_by_idx.keys())]
        completed_indices = set(merged_by_idx.keys())
        print(f"[RESUME] Loaded {len(completed_indices)} completed questions from checkpoint.")

    # Benchmark Counters
    pass1_correct_count = 0
    arkhe_forced_correct_count = 0
    arkhe_sel_correct_count = 0
    arkhe_sel_answered_count = 0
    baseline_correct_count = 0
    total_processed = 0

    # Running Loop over target items
    for item in target_items:
        q_idx = item["index"]
        prob = item["problem"]
        gt = item["ground_truth"]
        subj = item["subject"]
        lvl = item["level"]
        uid = item["unique_id"]

        # If already done in checkpoint, load from results
        if q_idx in completed_indices:
            prev_item = next(r for r in results if r["index"] == q_idx)
            total_processed += 1
            if prev_item.get("pass1_correct"):
                pass1_correct_count += 1
            if prev_item.get("arkhe_forced_correct"):
                arkhe_forced_correct_count += 1
            if prev_item.get("arkhe_verdict") == "COMMIT_ANSWER":
                arkhe_sel_answered_count += 1
                if prev_item.get("arkhe_forced_correct"):
                    arkhe_sel_correct_count += 1
            if prev_item.get("baseline_correct"):
                baseline_correct_count += 1
            continue

        print(f"\n{'='*75}")
        print(f"[{total_processed+1}/{len(target_items)}] Question Q{q_idx:03d} | {subj} Level {lvl}")
        print(f"Problem: {prob[:100]}...")
        print(f"Ground Truth: {gt}")
        print(f"{'='*75}")

        # Compute domain alpha
        low = prob.lower()
        algebra_cues = ("group", "ring", "field", "abelian", "isomorphism", "homomorphism", "eigenvalue", "matrix", "determinant", "grup", "halka", "cisim", "özdeğer", "matris", "polinom")
        alpha = 0.65 if any(k in low for k in algebra_cues) else 0.20
        blender.set_blend(model, alpha)
        rep_penalty = blender.compute_rep_penalty(alpha)

        # Standard MATH prompt with native thinking template
        p1_prompt = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step concisely, verify all algebraic transitions, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )

        # --- PASS-1 GREEDY GENERATION ---
        t0 = time.time()
        inputs = tokenizer(p1_prompt, return_tensors="pt").to("cuda:0")
        p_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=rep_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_ids,
            )
        p1_toks = gen_out[0][p_len:]
        p1_text = tokenizer.decode(p1_toks, skip_special_tokens=True).strip()
        elapsed_p1 = time.time() - t0
        tok_speed = len(p1_toks) / max(0.01, elapsed_p1)

        p1_mean_lp, p1_min_step_lp = (None, None)
        if args.logprob_threshold is not None:
            p1_mean_lp, p1_min_step_lp = compute_pass1_logprob_stats(inputs.input_ids, p1_toks)

        t1 = build_trace("Pass1_Greedy", p1_text, gen_tokens=len(p1_toks), prompt=prob)
        if p1_mean_lp is not None:
            from dataclasses import replace as _dc_replace
            t1 = _dc_replace(t1, mean_token_logprob=p1_mean_lp, min_step_logprob=p1_min_step_lp)
        p1_ok = NormalizedMathMatcher.check_match(t1.extracted_answer, gt)[1]

        lp_tag = f" | min_step_lp={p1_min_step_lp:.3f}" if p1_min_step_lp is not None else ""
        print(
            f"  [Pass-1] pred={t1.extracted_answer!r} | GT={gt!r} | "
            f"Match={'PASS' if p1_ok else 'FAIL'} | has_end={t1.has_end_tag} | "
            f"eq={t1.verified_eq_count} | contras={len(t1.contradictions)} | "
            f"bluff={t1.has_verbal_bluff}{lp_tag} | {len(p1_toks)} toks @ {tok_speed:.1f} tok/s ({elapsed_p1:.1f}s)",
            flush=True,
        )

        del gen_out
        torch.cuda.empty_cache()

        # Check Arkhe-3 Structural, Lean & Logprob Trigger
        struct_only_triggered = (
            (not t1.has_end_tag)
            or (not t1.is_symbolically_valid)
            or (len(t1.contradictions) > 0)
            or (t1.doubt_count > inference_arche.max_doubt_allowed)
            or bool(t1.has_verbal_bluff)
            or (t1.rollback_step_index is not None)
        )
        multibranch_triggered = inference_arche.should_trigger_multibranch(t1)
        triggered_by_lp_only = bool(multibranch_triggered and not struct_only_triggered)
        cands_traces = []

        if multibranch_triggered:
            print(
                f"  [Arkhe-3 Çıkarım Gate] Triggered multi-branch rollouts! "
                f"(contras={len(t1.contradictions)}, bluff={t1.has_verbal_bluff}, has_end={t1.has_end_tag}, lp_only={triggered_by_lp_only})"
            )
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

                t_b_start = time.time()
                set_seed(seed_val)
                b_inputs = tokenizer(b_prompt, return_tensors="pt").to("cuda:0")
                bp_len = b_inputs.input_ids.shape[1]

                with torch.no_grad():
                    gen_kwargs = dict(
                        **b_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=do_samp,
                        repetition_penalty=1.00,  # Strict 1.0 to prevent history poisoning
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos_token_ids,
                    )
                    if do_samp:
                        gen_kwargs["temperature"] = temp
                        gen_kwargs["top_p"] = 0.9
                    gen_out_b = model.generate(**gen_kwargs)

                gen_toks_b = gen_out_b[0][bp_len:]
                txt_b = tokenizer.decode(gen_toks_b, skip_special_tokens=True).strip()
                full_trace_text = (prefix + "\n" + txt_b) if (use_rb and prefix) else txt_b

                ct = build_trace(b_name, full_trace_text, gen_tokens=len(gen_toks_b), prompt=prob)
                cands_traces.append(ct)
                b_match = NormalizedMathMatcher.check_match(ct.extracted_answer, gt)[1]
                print(
                    f"    [{b_name} ({spec.get('role')})] pred={ct.extracted_answer!r} | "
                    f"Match={'PASS' if b_match else 'FAIL'} | has_end={ct.has_end_tag} | "
                    f"eq={ct.verified_eq_count} | contras={len(ct.contradictions)} | "
                    f"bluff={ct.has_verbal_bluff} | {len(gen_toks_b)} toks ({time.time()-t_b_start:.1f}s)",
                    flush=True,
                )
                del gen_out_b
                torch.cuda.empty_cache()
        else:
            print("  [Arkhe-3 Çıkarım Gate] Pass-1 structurally SOUND & verified. Multi-branch bypassed (Zero GPU overhead).")

        # Run Six-Arkhe Pipeline Decision Monad
        final_state: ArkheState = pipeline.run_from_traces(
            question_id=f"Q{q_idx:03d}",
            pass1_trace=t1,
            candidate_traces=cands_traces,
            gate_triggered=bool(multibranch_triggered and len(cands_traces) > 0),
            domain_alpha=alpha,
            prompt=prob,
        )

        winning_ans = final_state.winning_trace.extracted_answer if final_state.winning_trace else t1.extracted_answer
        forced_ok = NormalizedMathMatcher.check_match(winning_ans, gt)[1]

        is_abstain = (final_state.verdict == DecisionVerdict.ABSTAIN_UNKNOWN)
        conf = final_state.epistemic_confidence

        # Baseline info (also apply current NormalizedMathMatcher so Q199 etc. match fairly)
        b_info = baseline_map.get(q_idx, {})
        base_pred = b_info.get("predicted", "")
        base_ok = bool(b_info.get("is_correct", False)) or bool(NormalizedMathMatcher.check_match(base_pred, gt)[1])

        print(
            f"  ─> Arkhe Outcome: Forced Pred={winning_ans!r} ({'PASS' if forced_ok else 'FAIL'}) | "
            f"Conf={conf:.2f} | Verdict={final_state.verdict.name} | Baseline={base_pred!r} ({'PASS' if base_ok else 'FAIL'})"
        )

        total_processed += 1
        if p1_ok:
            pass1_correct_count += 1
        if forced_ok:
            arkhe_forced_correct_count += 1
        if not is_abstain:
            arkhe_sel_answered_count += 1
            if forced_ok:
                arkhe_sel_correct_count += 1
        if base_ok:
            baseline_correct_count += 1

        cand_pool_dicts = []
        for ct in final_state.candidate_pool:
            cand_pool_dicts.append({
                "branch": ct.branch_id,
                "pred": ct.extracted_answer,
                "has_end": ct.has_end_tag,
                "is_valid": ct.is_symbolically_valid,
                "eq_count": ct.verified_eq_count,
                "contras": list(ct.contradictions),
                "wait_cnt": ct.doubt_count,
                "has_verbal_bluff": ct.has_verbal_bluff,
                "tokens": ct.generated_tokens,
            })

        q_record = {
            "index": q_idx,
            "unique_id": uid,
            "subject": subj,
            "level": lvl,
            "problem": prob,
            "ground_truth": gt,
            "alpha": alpha,
            "pass1_predicted": t1.extracted_answer,
            "pass1_correct": p1_ok,
            "pass1_tokens": t1.generated_tokens,
            "pass1_has_end": t1.has_end_tag,
            "pass1_eq_count": t1.verified_eq_count,
            "pass1_contras": list(t1.contradictions),
            "pass1_has_bluff": t1.has_verbal_bluff,
            "pass1_mean_logprob": round(p1_mean_lp, 4) if p1_mean_lp is not None else None,
            "pass1_min_step_logprob": round(p1_min_step_lp, 4) if p1_min_step_lp is not None else None,
            "multibranch_triggered": multibranch_triggered,
            "triggered_by_logprob_only": triggered_by_lp_only,
            "winning_predicted": winning_ans,
            "arkhe_forced_correct": forced_ok,
            "arkhe_verdict": final_state.verdict.name,
            "epistemic_confidence": conf,
            "baseline_predicted": base_pred,
            "baseline_correct": base_ok,
            "candidate_pool": cand_pool_dicts,
        }
        results.append(q_record)

        # Cumulative running stats
        cur_cov = (arkhe_sel_answered_count / total_processed) * 100
        cur_sel_acc = (arkhe_sel_correct_count / max(1, arkhe_sel_answered_count)) * 100
        cur_forced_acc = (arkhe_forced_correct_count / total_processed) * 100
        cur_p1_acc = (pass1_correct_count / total_processed) * 100
        cur_base_acc = (baseline_correct_count / total_processed) * 100

        print(
            f"  [Progress {total_processed}/{len(target_items)}] "
            f"Pass-1: {pass1_correct_count}/{total_processed} ({cur_p1_acc:.1f}%) | "
            f"Arkhe Forced: {arkhe_forced_correct_count}/{total_processed} ({cur_forced_acc:.1f}%) | "
            f"Selective: {arkhe_sel_correct_count}/{arkhe_sel_answered_count} ({cur_sel_acc:.1f}% @ {cur_cov:.1f}% cov) | "
            f"Baseline: {baseline_correct_count}/{total_processed} ({cur_base_acc:.1f}%)"
        )

        out_summary = {
            "benchmark": f"MATH-500 Unseen (Q{args.start_idx}..Q{args.end_idx})",
            "runtime_metadata": runtime_meta,
            "logprob_threshold": args.logprob_threshold,
            "start_index": args.start_idx,
            "end_index": args.end_idx,
            "total_questions": len(target_items),
            "processed_questions": total_processed,
            "tau_threshold": args.tau,
            "metrics": {
                "baseline_sft_correct": baseline_correct_count,
                "baseline_sft_acc": cur_base_acc,
                "pass1_greedy_correct": pass1_correct_count,
                "pass1_greedy_acc": cur_p1_acc,
                "arkhe_forced_correct": arkhe_forced_correct_count,
                "arkhe_forced_acc": cur_forced_acc,
                "arkhe_selective_correct": arkhe_sel_correct_count,
                "arkhe_selective_answered": arkhe_sel_answered_count,
                "arkhe_selective_acc": cur_sel_acc,
                "coverage_pct": cur_cov,
            },
            "details": results,
        }
        active_output_path = safe_json_dump(out_summary, active_output_path, fallback_path=run_fallback_json)

    print("\n" + "=" * 80)
    print("SODALS v2.1 — PRISTINE-30 UNSEEN BENCHMARK COMPLETE!")
    print("=" * 80)
    print(f"Results successfully saved to: {active_output_path}")


if __name__ == "__main__":
    main()
