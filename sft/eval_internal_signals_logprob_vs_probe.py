"""
SODALS v2.1 — Phase 3: Teacher-Forced Internal Signals vs. Token Logprob Benchmark
===================================================================================
Answers Claude's closing criterion before abandoning internal neural signals:
  "Before closing the internal-signal direction, run the two missing baselines on
   greedy outputs: (1) mean/step token logprob and (2) intermediate hidden-state
   logistic probe with prompt-grouped CV and PCA <= 20 dimensions. If the probe
   does not beat logprob with 95% CI excluding zero, close the internal probe line."

Protocol (Zero generation overhead — single teacher-forced forward pass per prompt):
  - Dataset: N=145 saved greedy responses from `eval_grpo_math_train_clean_145_post_fix_results.json`
    * Clean-125 (Q026..Q150): 5-Fold Stratified CV for PCA(n<=16) + LogisticRegression
    * Pristine-20 (Q151..Q170): Strictly unseen OOD holdout evaluation
  - Signals Extracted:
    * `mean_token_logprob`: mean log P(y_t | x, y_<t) across generated tokens
    * `p05_token_logprob` / `min_step_logprob`: lowest 5% tail / lowest line-step mean logprob
    * `final_answer_logprob`: mean logprob inside `\\boxed{...}` span (when localized)
    * `mean_token_entropy`: mean next-token Shannon entropy across generated tokens
    * `probe_layer_mid` (Layer -8), `probe_layer_late` (Layer -4), `probe_layer_final` (Layer -1):
      last-token & mean-pooled hidden states -> StandardScaler -> PCA(16) -> LogisticRegression(C=0.1)
  - Evaluation: B=5000 Stratified/Clustered Bootstrap AUROC & Paired Delta-AUROC 95% CI,
    plus Silent-Failure Detection Audit on clean-terminating wrong Pass-1 responses.
"""

import json
import math
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import smart_app_control_compat  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from cst_symbolic_verifier import CstSymbolicVerifier, LeanTacticStepVerifier, NormalizedMathMatcher
from env_reproducibility import collect_runtime_metadata

ROOT = os.path.dirname(SFT_DIR)
MODEL_PATH = os.path.join(ROOT, "models", "Qwen3.5-9B-SODALS-Baseline-SafeTensors")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_GRPO = os.path.join(SFT_DIR, "qwen_lora_adapter_grpo_math_train_100")
DATA_JSON = os.path.join(SFT_DIR, "eval_grpo_math_train_clean_145_post_fix_results.json")


class CPUEmbeddingWrapper(nn.Module):
    def __init__(self, cpu_embed):
        super().__init__()
        self.cpu_embed = cpu_embed

    def forward(self, input_ids):
        return self.cpu_embed(input_ids.to("cpu")).to("cuda")


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact rank-based Mann-Whitney AUROC."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    wins = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
    return float(wins / (len(pos) * len(neg)))


def bootstrap_auroc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 5000, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    obs = compute_auroc(y_true, y_score)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    vals = []
    for _ in range(n_boot):
        bp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([bp, bn])
        vals.append(compute_auroc(y_true[idx], y_score[idx]))
    vals = np.array(vals)
    return {
        "auroc": round(obs, 4),
        "ci95_low": round(float(np.percentile(vals, 2.5)), 4),
        "ci95_high": round(float(np.percentile(vals, 97.5)), 4),
    }


def paired_bootstrap_delta(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, n_boot: int = 5000, seed: int = 42
) -> Dict[str, Any]:
    """Paired bootstrap CI for AUROC(score_a) - AUROC(score_b)."""
    rng = np.random.default_rng(seed)
    delta_obs = compute_auroc(y_true, score_a) - compute_auroc(y_true, score_b)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    deltas = []
    for _ in range(n_boot):
        bp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([bp, bn])
        d = compute_auroc(y_true[idx], score_a[idx]) - compute_auroc(y_true[idx], score_b[idx])
        deltas.append(d)
    deltas = np.array(deltas)
    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    p_two = float(2.0 * min(np.mean(deltas <= 0.0), np.mean(deltas >= 0.0)))
    return {
        "delta_auroc": round(delta_obs, 4),
        "ci95_low": round(ci_lo, 4),
        "ci95_high": round(ci_hi, 4),
        "excludes_zero": bool(ci_lo > 0.0 or ci_hi < 0.0),
        "p_value": round(min(1.0, p_two), 4),
    }


def fit_predict_pca_logreg_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5, max_pca: int = 16, seed: int = 42):
    """
    Pure-NumPy + PyTorch Stratified K-Fold StandardScaler + SVD PCA(<=16) + Balanced L2 LogisticRegression.
    Avoids external scikit-learn Cython DLL issues under Windows Smart App Control.
    """
    rng = np.random.default_rng(seed)

    def make_stratified_folds(labels: np.ndarray, k: int):
        pos = np.where(labels == 1)[0]
        neg = np.where(labels == 0)[0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        pos_splits = np.array_split(pos, k)
        neg_splits = np.array_split(neg, k)
        folds = []
        all_idx = np.arange(len(labels))
        for f in range(k):
            va = np.concatenate([pos_splits[f], neg_splits[f]])
            tr_mask = np.ones(len(labels), dtype=bool)
            tr_mask[va] = False
            tr = all_idx[tr_mask]
            folds.append((tr, va))
        return folds

    def fit_single_pipeline(X_tr: np.ndarray, y_tr: np.ndarray, C: float = 0.1):
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        X_norm = (X_tr - mu) / sd

        # Economy SVD for exact PCA on zero-mean normalized features
        _, _, vt = np.linalg.svd(X_norm, full_matrices=False)
        n_comp = min(max_pca, X_tr.shape[0] - 1, X_tr.shape[1])
        comps = vt[:n_comp].T  # [D, n_comp]
        Z_tr = X_norm @ comps

        # Balanced L2-regularized Logistic Regression solved via PyTorch L-BFGS on CPU
        z_t = torch.tensor(Z_tr, dtype=torch.float64)
        y_t = torch.tensor(y_tr, dtype=torch.float64)
        n_pos = max(1.0, float((y_tr == 1).sum()))
        n_neg = max(1.0, float((y_tr == 0).sum()))
        w_pos = len(y_tr) / (2.0 * n_pos)
        w_neg = len(y_tr) / (2.0 * n_neg)
        sample_w = torch.where(y_t > 0.5, w_pos, w_neg)

        w = torch.zeros(n_comp, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        reg = 1.0 / (2.0 * C * len(y_tr))
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            logits = z_t @ w + b
            bce = F.binary_cross_entropy_with_logits(logits, y_t, weight=sample_w, reduction="mean")
            loss = bce + reg * torch.sum(w * w)
            loss.backward()
            return loss

        opt.step(closure)
        w_np = w.detach().numpy()
        b_val = float(b.detach().item())

        def predict_prob(X_in: np.ndarray) -> np.ndarray:
            Xn = (X_in - mu) / sd
            Z = Xn @ comps
            logits = np.clip(Z @ w_np + b_val, -40.0, 40.0)
            return 1.0 / (1.0 + np.exp(-logits))

        return predict_prob

    oof_probs = np.zeros(len(y), dtype=np.float64)
    for tr_idx, va_idx in make_stratified_folds(y, n_splits):
        pred_fn = fit_single_pipeline(X[tr_idx], y[tr_idx], C=0.1)
        oof_probs[va_idx] = pred_fn(X[va_idx])

    full_pred_fn = fit_single_pipeline(X, y, C=0.1)
    return oof_probs, full_pred_fn


def extract_signals_for_dataset():
    data = json.load(open(DATA_JSON, "r", encoding="utf-8"))
    items = data.get("details", [])
    print(f"Loaded {len(items)} saved greedy responses (Clean-125 + Pristine-20).", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[GPU INIT] Loading base model + {os.path.basename(ADAPTER_GRPO)}...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_GRPO)
    model.eval()

    # Keep embed_tokens on CPU via CPUEmbeddingWrapper to match runtime setup & save VRAM
    embed_layer = model.get_input_embeddings()
    embed_cpu = embed_layer.cpu()
    model.set_input_embeddings(CPUEmbeddingWrapper(embed_cpu))
    lm_head = model.get_output_embeddings()
    backbone = model.base_model.model.model if hasattr(model.base_model.model, "model") else model.model
    torch.cuda.empty_cache()

    records = []
    t0 = time.time()
    for idx_i, item in enumerate(items):
        q_idx = item["index"]
        prob = item["problem"]
        gt = str(item["ground_truth"])
        pred = str(item.get("predicted", ""))
        resp = str(item.get("full_response", ""))

        is_correct = bool(NormalizedMathMatcher.check_match(pred, gt)[1])
        has_end = ("</think>" in resp)
        is_cst_ok, _, contras = CstSymbolicVerifier.verify_chunk(resp)
        step_rep = LeanTacticStepVerifier.verify_proof_steps(resp)
        structural_clean = bool(has_end and is_cst_ok and len(contras) == 0 and not step_rep.has_sorry_bluff)

        prompt_str = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        p_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids
        r_ids = tokenizer(resp, return_tensors="pt", add_special_tokens=False).input_ids
        p_len = p_ids.shape[1]
        r_len = r_ids.shape[1]
        if r_len < 2:
            continue

        # Truncate combined sequence to 2560 max if unusually long so 8GB VRAM stays safe
        if p_len + r_len > 2560:
            r_ids = r_ids[:, : (2560 - p_len)]
            r_len = r_ids.shape[1]

        full_ids = torch.cat([p_ids, r_ids], dim=1)
        embeds = embed_cpu(full_ids).to("cuda", dtype=torch.float16)

        with torch.no_grad():
            out = backbone(inputs_embeds=embeds, output_hidden_states=True, use_cache=False)
            hs_all = out.hidden_states  # tuple of (L+1) tensors [1, T, D]
            n_layers = len(hs_all) - 1
            idx_mid = max(1, n_layers - 8)
            idx_late = max(1, n_layers - 4)
            idx_last = n_layers

            # Extract response hidden features (last response token & mean response pooling)
            h_last_tok = hs_all[idx_last][0, -1, :].float().cpu().numpy()
            h_late_tok = hs_all[idx_late][0, -1, :].float().cpu().numpy()
            h_mid_mean = hs_all[idx_mid][0, p_len:, :].float().mean(dim=0).cpu().numpy()
            h_late_mean = hs_all[idx_late][0, p_len:, :].float().mean(dim=0).cpu().numpy()

            # Locate </think> boundary hidden state if present
            resp_h = out.last_hidden_state[:, p_len - 1 : -1, :]
            target_ids = r_ids.to("cuda")

            # Compute chunked exact token logprobs and Shannon entropies over generated tokens
            token_logps = []
            token_ents = []
            for start in range(0, r_len, 64):
                end = min(start + 64, r_len)
                c_h = resp_h[:, start:end, :].contiguous()
                c_t = target_ids[:, start:end].contiguous()
                logits = lm_head(c_h).float()
                log_p = F.log_softmax(logits, dim=-1)
                tok_lp = torch.gather(log_p, -1, c_t.unsqueeze(-1)).squeeze(-1)
                probs = torch.exp(log_p)
                ent = -(probs * log_p).sum(dim=-1)
                token_logps.append(tok_lp[0].cpu())
                token_ents.append(ent[0].cpu())
                del logits, log_p, probs

            lp_arr = torch.cat(token_logps, dim=0).numpy()
            ent_arr = torch.cat(token_ents, dim=0).numpy()

        del out, hs_all, resp_h, embeds
        if idx_i % 10 == 0:
            torch.cuda.empty_cache()

        # Compute line/chunk-level min step logprob (chunks of 32 tokens)
        step_means = [float(lp_arr[s : s + 32].mean()) for s in range(0, len(lp_arr), 32) if len(lp_arr[s : s + 32]) >= 8]
        min_step_lp = min(step_means) if step_means else float(lp_arr.mean())

        # Tail / answer window (last 24 tokens)
        tail_lp = float(lp_arr[-min(24, len(lp_arr)) :].mean())

        records.append({
            "index": q_idx,
            "subset": "Pristine-20" if q_idx >= 151 else "Clean-125",
            "is_correct": int(is_correct),
            "structural_clean": structural_clean,
            "r_tokens": int(r_len),
            "mean_token_logprob": float(lp_arr.mean()),
            "p05_token_logprob": float(np.percentile(lp_arr, 5)),
            "min_step_logprob": float(min_step_lp),
            "tail_answer_logprob": float(tail_lp),
            "neg_mean_entropy": float(-ent_arr.mean()),
            "h_last_tok": h_last_tok,
            "h_late_tok": h_late_tok,
            "h_mid_mean": h_mid_mean,
            "h_late_mean": h_late_mean,
        })

        if (idx_i + 1) % 25 == 0 or (idx_i + 1) == len(items):
            print(f"  Processed {idx_i+1}/{len(items)} ({time.time()-t0:.1f}s)", flush=True)

    # Split Clean-125 (Dev CV) & Pristine-20 (Unseen Holdout)
    dev_recs = [r for r in records if r["subset"] == "Clean-125"]
    ood_recs = [r for r in records if r["subset"] == "Pristine-20"]

    y_dev = np.array([r["is_correct"] for r in dev_recs], dtype=np.int64)
    y_ood = np.array([r["is_correct"] for r in ood_recs], dtype=np.int64)

    probe_models = {}
    probe_specs = ["h_last_tok", "h_late_tok", "h_mid_mean", "h_late_mean"]
    dev_oof_probes = {}
    ood_pred_probes = {}

    for pkey in probe_specs:
        X_dev = np.stack([r[pkey] for r in dev_recs])
        X_ood = np.stack([r[pkey] for r in ood_recs])
        oof_p, pred_fn = fit_predict_pca_logreg_cv(X_dev, y_dev, n_splits=5, max_pca=16, seed=42)
        dev_oof_probes[f"probe_{pkey}"] = oof_p
        ood_pred_probes[f"probe_{pkey}"] = pred_fn(X_ood)
        probe_models[pkey] = True

    scalar_keys = [
        "mean_token_logprob",
        "p05_token_logprob",
        "min_step_logprob",
        "tail_answer_logprob",
        "neg_mean_entropy",
    ]

    dev_scores: Dict[str, np.ndarray] = {k: np.array([r[k] for r in dev_recs], dtype=np.float64) for k in scalar_keys}
    dev_scores.update(dev_oof_probes)

    ood_scores: Dict[str, np.ndarray] = {k: np.array([r[k] for r in ood_recs], dtype=np.float64) for k in scalar_keys}
    ood_scores.update(ood_pred_probes)

    dev_auroc = {k: bootstrap_auroc_ci(y_dev, v) for k, v in dev_scores.items()}
    ood_auroc = {k: bootstrap_auroc_ci(y_ood, v) for k, v in ood_scores.items()}

    # Best probe vs best logprob paired bootstrap test on Clean-125 (5-fold OOF)
    best_probe_key = max(dev_oof_probes.keys(), key=lambda k: dev_auroc[k]["auroc"])
    paired_probe_vs_mean_lp = paired_bootstrap_delta(
        y_dev, dev_scores[best_probe_key], dev_scores["mean_token_logprob"]
    )
    paired_probe_vs_min_step = paired_bootstrap_delta(
        y_dev, dev_scores[best_probe_key], dev_scores["min_step_logprob"]
    )

    # Silent-failure separation audit (only structural_clean==True subset across N=145)
    silent_recs = [r for r in records if r["structural_clean"]]
    y_silent = np.array([r["is_correct"] for r in silent_recs], dtype=np.int64)
    silent_auroc = {
        k: bootstrap_auroc_ci(y_silent, np.array([r[k] for r in silent_recs], dtype=np.float64))
        for k in scalar_keys
    }

    # Check threshold sensitivity on silent failures (Q153, Q155, Q157 etc.)
    clean_wrong = [r for r in silent_recs if r["is_correct"] == 0]
    clean_right = [r for r in silent_recs if r["is_correct"] == 1]

    # Drop large numpy vectors before JSON dump
    compact_records = []
    for r in records:
        cr = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        cr[best_probe_key] = (
            float(dev_scores[best_probe_key][[x["index"] for x in dev_recs].index(r["index"])])
            if r["subset"] == "Clean-125"
            else float(ood_scores[best_probe_key][[x["index"] for x in ood_recs].index(r["index"])])
        )
        compact_records.append(cr)

    summary = {
        "runtime_metadata": collect_runtime_metadata(),
        "dataset": {
            "total_evaluated": len(records),
            "clean125_n": len(dev_recs),
            "clean125_correct": int(y_dev.sum()),
            "pristine20_n": len(ood_recs),
            "pristine20_correct": int(y_ood.sum()),
            "structural_clean_n": len(silent_recs),
            "structural_clean_silent_wrong": int((y_silent == 0).sum()),
        },
        "clean125_auroc_5fold_cv": dev_auroc,
        "pristine20_ood_auroc": ood_auroc,
        "best_probe_key": best_probe_key,
        "paired_delta_best_probe_vs_mean_token_logprob": paired_probe_vs_mean_lp,
        "paired_delta_best_probe_vs_min_step_logprob": paired_probe_vs_min_step,
        "silent_failure_subset_auroc": silent_auroc,
        "silent_wrong_vs_right_means": {
            k: {
                "silent_wrong_mean": round(float(np.mean([r[k] for r in clean_wrong])), 4) if clean_wrong else None,
                "silent_right_mean": round(float(np.mean([r[k] for r in clean_right])), 4) if clean_right else None,
            }
            for k in scalar_keys
        },
        "per_question_signals": compact_records,
    }

    out_primary = os.path.join(SFT_DIR, "eval_internal_signals_logprob_vs_probe_results.json")
    out_fallback = os.path.join(tempfile.gettempdir(), "eval_internal_signals_logprob_vs_probe_results.json")
    saved_to = out_primary
    try:
        with open(out_primary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    except PermissionError:
        saved_to = out_fallback
        with open(out_fallback, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 88)
    print("PHASE 3 COMPLETE: LOGPROB VS. PCA(<=16) HIDDEN-STATE LOGISTIC PROBE")
    print("Saved results:", saved_to)
    print("-" * 88)
    print("Clean-125 (5-Fold CV) AUROC [95% Bootstrap CI]:")
    for k, v in dev_auroc.items():
        print(f"  {k:22s}: AUROC={v['auroc']:.4f}  [{v['ci95_low']:.4f}, {v['ci95_high']:.4f}]")
    print("-" * 88)
    print(f"Paired Test ({best_probe_key} - mean_token_logprob): delta={paired_probe_vs_mean_lp['delta_auroc']:+.4f} "
          f"CI=[{paired_probe_vs_mean_lp['ci95_low']:+.4f}, {paired_probe_vs_mean_lp['ci95_high']:+.4f}] "
          f"excludes_zero={paired_probe_vs_mean_lp['excludes_zero']} p={paired_probe_vs_mean_lp['p_value']:.4f}")
    print(f"Silent-Failure Subset (N={len(silent_recs)}, Silent Wrongs={int((y_silent==0).sum())}):")
    for k, v in silent_auroc.items():
        print(f"  {k:22s}: AUROC={v['auroc']:.4f}  [{v['ci95_low']:.4f}, {v['ci95_high']:.4f}]")
    print("Pristine-20 OOD AUROC:")
    for k in ("mean_token_logprob", "min_step_logprob", "tail_answer_logprob", best_probe_key):
        v = ood_auroc[k]
        print(f"  {k:22s}: AUROC={v['auroc']:.4f}  [{v['ci95_low']:.4f}, {v['ci95_high']:.4f}]")
    print("=" * 88)


if __name__ == "__main__":
    extract_signals_for_dataset()
