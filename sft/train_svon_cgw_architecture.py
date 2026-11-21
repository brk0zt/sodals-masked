"""
Svon-CGW (Epoch III) — Two-Stage Phase-Contrastive & GRPO Training Engine
=========================================================================
Trains the `Svon-CGW` (Holomorphic CVNN-DeltaNet + 2-Hop Relational Sheaf-GNN)
parameters (~2.1M trainable parameters, < 9 MB VRAM overhead) on top of the
frozen 4-bit NF4 base model:
  - Stage 1 (Phase-Contrastive Self-Distillation):
      * Constructive phase resonance (1 - cos(Delta phi)) across valid algebraic steps
      * Destructive wave cancellation (1 + cos(Delta phi_err)) on injected contradictory steps
      * Submanifold KL anchor: KL(P_CGW || P_Base) <= 0.01
  - Stage 2 (CGW-GRPO):
      * Joint reinforcement evolution of complex phase velocities `omega_t` and
        2-hop graph diffusion gains `alpha_hop`, exporting `.safetensors` + `.qmem`.

NOTE: Scheduled strictly for Epoch III (after Epoch I & Epoch II complete on GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SFT_DIR)
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

from cloud_runtime_utils import (
    get_primary_device,
    load_qwen_model_and_tokenizer,
    maybe_enable_sac_compat,
    mount_production_adapters,
    prevent_sleep_if_windows,
)

maybe_enable_sac_compat()

import torch
import torch.nn.functional as F

from svon_kuramoto_equilibrium import (
    _locate_decoder_layers,
    collect_svon_cgw_state_dict,
    compute_kuramoto_order_parameter,
    export_svon_cgw_to_rust_qmem,
    upgrade_model_to_svon_cgw,
)
from svon_cvnn_deltanet import HolomorphicCVNNDeltaNetWrapper


def compute_phase_contrastive_loss(
    coherent_phases: List[torch.Tensor],
    contradictory_phases: Optional[List[torch.Tensor]] = None,
    prompt_len: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes the Complex Wave Interference Loss across CVNN layers:
      1. L_sync:   encourages consecutive valid reasoning steps to align in phase
                   (1 - cos(phi_{t+1} - phi_t) -> 0).
      2. L_cancel: forces contradictory/corrupted post-prompt steps toward phase opposition
                   Delta phi = pi (1 + cos(phi_err - phi_clean) -> 0, i.e., cos -> -1).
    Uses LEFT-aligned post-prompt token slicing (`[prompt_len : prompt_len + min_t]`) so
    shared prompt prefixes are never pushed apart and different-length completions stay aligned.
    """
    if not coherent_phases:
        zero = torch.tensor(0.0)
        return zero, {"loss_sync": 0.0, "loss_cancel": 0.0, "mean_r_order": 1.0}

    p_start = max(0, int(prompt_len))
    sync_terms = []
    r_orders = []
    for p in coherent_phases:
        # p: [B, T, H_k, C]
        p_resp = p[:, p_start:] if p_start < p.shape[1] else p
        if p_resp.shape[1] > 1:
            d_phi = p_resp[:, 1:] - p_resp[:, :-1]
            sync_terms.append((1.0 - torch.cos(d_phi)).mean())
        r_orders.append(compute_kuramoto_order_parameter(p_resp).mean())

    loss_sync = (
        torch.stack([t.to(coherent_phases[0].device) for t in sync_terms]).mean()
        if sync_terms
        else torch.tensor(0.0, device=coherent_phases[0].device)
    )

    cancel_terms = []
    if contradictory_phases and len(contradictory_phases) == len(coherent_phases):
        for p_clean, p_err in zip(coherent_phases, contradictory_phases):
            c_slice = p_clean[:, p_start:] if p_start < p_clean.shape[1] else p_clean
            e_slice = p_err[:, p_start:] if p_start < p_err.shape[1] else p_err
            min_t = min(c_slice.shape[1], e_slice.shape[1])
            if min_t <= 0:
                continue
            p_err_al = e_slice[:, :min_t].to(c_slice.device)
            diff = p_err_al - c_slice[:, :min_t].detach()
            # Target cos(diff) -> -1 (destructive interference angle pi)
            cancel_terms.append((1.0 + torch.cos(diff)).mean().to(coherent_phases[0].device))

    loss_cancel = (
        torch.stack(cancel_terms).mean()
        if cancel_terms
        else torch.tensor(0.0, device=coherent_phases[0].device)
    )
    total = loss_sync + loss_cancel
    meta = {
        "loss_sync": round(float(loss_sync.item()), 6),
        "loss_cancel": round(float(loss_cancel.item()), 6),
        "mean_r_order": round(float(torch.stack([r.to(coherent_phases[0].device) for r in r_orders]).mean().item()), 6) if r_orders else 1.0,
    }
    return total, meta


def train_svon_cgw(args):
    prevent_sleep_if_windows()
    sys.stdout.reconfigure(encoding="utf-8")

    q4 = False if getattr(args, "fp16", False) else getattr(args, "quantize_4bit", None)
    model, tokenizer, rt_meta = load_qwen_model_and_tokenizer(
        model_path=args.model_path,
        quantize_4bit=q4,
        online=True if getattr(args, "online", False) else None,
    )
    model, _blender, mounted_prod = mount_production_adapters(
        model,
        d1_adapter_path=getattr(args, "d1_adapter_path", None),
        d2_adapter_path=getattr(args, "d2_adapter_path", None),
        svon_adapter_path=getattr(args, "svon_adapter_path", None),
    )
    rt_meta["mounted_prod_adapters"] = mounted_prod
    entry_dev = get_primary_device(model)
    print(f"[SVON-CGW RUNTIME] {json.dumps(rt_meta)}", flush=True)

    up_info = upgrade_model_to_svon_cgw(model, phase_rank=args.phase_rank, freeze_base=True)
    print(f"[SVON-CGW UPGRADE] {json.dumps(up_info)}", flush=True)

    # Base weights remain frozen in eval mode, while CGW wrappers enter train mode so
    # complex rotations & 2-hop sheaf message passing stay differentiable from step 1.
    model.eval()
    for mod in model.modules():
        if mod.__class__.__name__ in ("HolomorphicCVNNDeltaNetWrapper", "RelationalSheafGNNAttention"):
            mod.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    if os.path.exists(args.prompts_json):
        prompts = json.load(open(args.prompts_json, "r", encoding="utf-8"))["prompts"][: args.steps]
    else:
        prompts = [
            {"problem": f"Solve 2*x + {i} = {2*i + 6} for x.", "ground_truth": "3"}
            for i in range(1, args.steps + 1)
        ]

    layers = _locate_decoder_layers(model)
    cvnn_layers = [
        (i, l.linear_attn)
        for i, l in enumerate(layers)
        if hasattr(l, "linear_attn") and isinstance(l.linear_attn, HolomorphicCVNNDeltaNetWrapper)
    ]

    # Precompute frozen base logits (at exact zero-init t=0) for Submanifold KL anchor regularization
    base_kl_anchors: List[torch.Tensor] = []
    cgw_mods = [
        mod
        for mod in model.modules()
        if mod.__class__.__name__ in ("HolomorphicCVNNDeltaNetWrapper", "RelationalSheafGNNAttention")
    ]
    for m in cgw_mods:
        m.eval()
    with torch.no_grad():
        for item in prompts:
            prob = item["problem"]
            gt = item["ground_truth"]
            clean_text = f"Problem: {prob}\nStep 1: Let us set up the algebraic equations carefully.\nStep 2: Verified result is \\boxed{{{gt}}}."
            enc_c = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=256).to(entry_dev)
            base_kl_anchors.append(model(**enc_c, use_cache=False).logits[:, -1, :].detach().float().cpu())
    for m in cgw_mods:
        m.train()

    kl_anchor_weight = float(getattr(args, "kl_anchor_weight", 0.20) or 0.20)

    for step_i, item in enumerate(prompts, start=1):
        prob = item["problem"]
        gt = item["ground_truth"]
        prompt_prefix = f"Problem: {prob}\n"
        clean_text = f"{prompt_prefix}Step 1: Let us set up the algebraic equations carefully.\nStep 2: Verified result is \\boxed{{{gt}}}."
        corrupt_text = f"{prompt_prefix}Step 1: Assume 1 = 2 by contradiction.\nStep 2: Unverified jump yields \\boxed{{999999}}."

        prompt_tok_len = int(tokenizer(prompt_prefix, return_tensors="pt", truncation=True, max_length=220).input_ids.shape[1])
        enc_clean = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=256).to(entry_dev)
        enc_corr = tokenizer(corrupt_text, return_tensors="pt", truncation=True, max_length=256).to(entry_dev)
        eff_prompt_len = min(prompt_tok_len, max(0, int(enc_clean.input_ids.shape[1]) - 2))

        opt.zero_grad()
        out_clean = model(**enc_clean, use_cache=False)
        clean_phases = [cw.last_cumulative_phase for _, cw in cvnn_layers if cw.last_cumulative_phase is not None]

        # Contradictory trajectory forward for destructive phase-contrastive signal (post-prompt left-aligned!)
        _ = model(**enc_corr, use_cache=False)
        corr_phases = [cw.last_cumulative_phase for _, cw in cvnn_layers if cw.last_cumulative_phase is not None]

        loss_phase, phase_meta = compute_phase_contrastive_loss(clean_phases, corr_phases, prompt_len=eff_prompt_len)

        # Mask prompt tokens with -100 so CE only trains on solution tokens
        logits_clean = out_clean.logits[:, :-1, :].float()
        targets_clean = enc_clean.input_ids[:, 1:].to(logits_clean.device).clone()
        if eff_prompt_len > 1:
            targets_clean[:, : eff_prompt_len - 1] = -100
        loss_ce = F.cross_entropy(
            logits_clean.reshape(-1, logits_clean.shape[-1]),
            targets_clean.reshape(-1),
            ignore_index=-100,
        )

        # Submanifold KL anchor KL(P_CGW || P_Base)
        cur_last_logits = out_clean.logits[:, -1, :].float()
        base_last_logits = base_kl_anchors[step_i - 1].to(device=cur_last_logits.device, dtype=torch.float32)
        p_cgw = F.softmax(cur_last_logits, dim=-1)
        loss_kl = (p_cgw * (F.log_softmax(cur_last_logits, dim=-1) - F.log_softmax(base_last_logits, dim=-1))).sum(dim=-1).mean()

        loss = (
            loss_ce
            + float(args.phase_loss_weight) * loss_phase.to(loss_ce.device)
            + kl_anchor_weight * loss_kl.to(loss_ce.device)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        print(
            f"[SVON-CGW STEP {step_i:02d}/{len(prompts)}] total={loss.item():.4f} ce={loss_ce.item():.4f} "
            f"kl={loss_kl.item():.5f} sync={phase_meta['loss_sync']:.4f} cancel={phase_meta['loss_cancel']:.4f} R={phase_meta['mean_r_order']:.4f}",
            flush=True,
        )

    grpo_steps = int(getattr(args, "cgw_grpo_steps", 0) or 0)
    if grpo_steps > 0:
        stage2_mode = str(getattr(args, "stage2_mode", "grpo") or "grpo").lower()
        group_size = max(2, int(getattr(args, "group_size", 4) or 4))
        grpo_max_new_tokens = max(32, int(getattr(args, "grpo_max_new_tokens", 512) or 512))
        grpo_temp = float(getattr(args, "grpo_temp", 0.6) or 0.6)
        grpo_prompts = prompts[:grpo_steps]

        if stage2_mode == "grpo":
            from cst_symbolic_verifier import NormalizedMathMatcher, SymbolicStepChecker

            matcher = NormalizedMathMatcher()
            checker = SymbolicStepChecker()

            def _score_rollout(prob_txt: str, completion_txt: str, gt_txt: str) -> Tuple[float, Dict[str, Any]]:
                pred_ans = matcher.extract_boxed(completion_txt)
                _, is_correct = matcher.check_match(pred_ans, gt_txt)
                outcome_r = 2.0 if is_correct else -1.0
                rep = checker.verify_proof_steps(completion_txt, prompt=prob_txt)
                eq_bonus = min(0.40, 0.04 * float(getattr(rep, "verified_eq_count", 0)))
                contra_pen = 0.35 * float(len(getattr(rep, "contradictions", []) or []))
                sorry_pen = 0.30 if getattr(rep, "has_sorry_bluff", False) else 0.0
                total_r = outcome_r + eq_bonus - contra_pen - sorry_pen
                return total_r, {
                    "correct": bool(is_correct),
                    "eq": int(getattr(rep, "verified_eq_count", 0)),
                    "contras": int(len(getattr(rep, "contradictions", []) or [])),
                    "sorry": bool(getattr(rep, "has_sorry_bluff", False)),
                }

            active_grpo_steps = 0
            for g_idx, item in enumerate(grpo_prompts, start=1):
                prob = item["problem"]
                gt = item["ground_truth"]
                prompt_prefix = f"Problem: {prob}\nSolve step by step and place the final answer in \\boxed{{...}}.\n"
                enc_p = tokenizer(prompt_prefix, return_tensors="pt", truncation=True, max_length=256).to(entry_dev)
                p_len = int(enc_p.input_ids.shape[1])

                # 1. Sample G=4 on-policy rollouts from frozen D1+D2+Svon + trainable CGW layers
                for m in cgw_mods:
                    m.eval()
                with torch.no_grad():
                    gen_ids = model.generate(
                        **enc_p,
                        max_new_tokens=grpo_max_new_tokens,
                        do_sample=True,
                        temperature=grpo_temp,
                        top_p=0.95,
                        num_return_sequences=group_size,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                for m in cgw_mods:
                    m.train()

                rewards: List[float] = []
                metas: List[Dict[str, Any]] = []
                seq_tensors: List[torch.Tensor] = []
                for gi in range(gen_ids.shape[0]):
                    comp_ids = gen_ids[gi, p_len:]
                    comp_txt = tokenizer.decode(comp_ids, skip_special_tokens=True)
                    r_val, r_meta = _score_rollout(prob, comp_txt, gt)
                    rewards.append(r_val)
                    metas.append(r_meta)
                    seq_tensors.append(gen_ids[gi : gi + 1])

                r_t = torch.tensor(rewards, dtype=torch.float32)
                r_std = float(r_t.std(unbiased=False).item())
                if r_std < 1e-6:
                    print(
                        f"[SVON-CGW GRPO {g_idx:02d}/{len(grpo_prompts)}] zero-variance skip "
                        f"(mean_r={r_t.mean().item():.3f}, solved={sum(m['correct'] for m in metas)}/{len(metas)})",
                        flush=True,
                    )
                    continue

                adv = torch.clamp((r_t - r_t.mean()) / (r_std + 1e-6), -2.0, 2.0)
                best_i = int(torch.argmax(r_t).item())
                worst_i = int(torch.argmin(r_t).item())

                opt.zero_grad()
                pg_loss_val = 0.0
                best_phases = None
                worst_phases = None
                for gi, seq_i in enumerate(seq_tensors):
                    if seq_i.shape[1] <= p_len:
                        continue
                    out_i = model(input_ids=seq_i, use_cache=False)
                    lp_i = F.log_softmax(out_i.logits[:, :-1, :].float(), dim=-1)
                    tok_lp = lp_i.gather(-1, seq_i[:, 1:].to(lp_i.device).unsqueeze(-1)).squeeze(-1)
                    resp_lp = tok_lp[:, max(0, p_len - 1) :].mean()
                    loss_i = -(float(adv[gi].item()) / float(len(seq_tensors))) * resp_lp
                    if gi == best_i:
                        best_phases = [cw.last_cumulative_phase for _, cw in cvnn_layers if cw.last_cumulative_phase is not None]
                    elif gi == worst_i:
                        worst_phases = [cw.last_cumulative_phase for _, cw in cvnn_layers if cw.last_cumulative_phase is not None]
                    loss_i.backward(retain_graph=(gi in (best_i, worst_i)))
                    pg_loss_val += float(loss_i.item())

                phase_meta_g = {"loss_cancel": 0.0, "mean_r_order": 0.0}
                if best_phases and worst_phases and rewards[best_i] > rewards[worst_i]:
                    loss_ph_g, phase_meta_g = compute_phase_contrastive_loss(best_phases, worst_phases, prompt_len=p_len)
                    (float(args.phase_loss_weight) * loss_ph_g).backward()

                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                active_grpo_steps += 1
                print(
                    f"[SVON-CGW GRPO {g_idx:02d}/{len(grpo_prompts)}] pg={pg_loss_val:.4f} "
                    f"mean_r={r_t.mean().item():.3f} solved={sum(m['correct'] for m in metas)}/{len(metas)} "
                    f"cancel={phase_meta_g['loss_cancel']:.4f} R={phase_meta_g['mean_r_order']:.4f}",
                    flush=True,
                )
        else:
            # Legacy fast synthetic DPO preference margin fallback
            for g_idx, item in enumerate(grpo_prompts, start=1):
                prob = item["problem"]
                gt = item["ground_truth"]
                prompt_prefix = f"Problem: {prob}\n"
                traj_pos = f"{prompt_prefix}Step 1: Set up verified balance equations.\nStep 2: Verified answer is \\boxed{{{gt}}}."
                traj_neg = f"{prompt_prefix}Step 1: Skip derivation via unverified shortcut.\nStep 2: Flawed answer is \\boxed{{0}}."
                prompt_tok_len = int(tokenizer(prompt_prefix, return_tensors="pt", truncation=True, max_length=220).input_ids.shape[1])
                enc_pos = tokenizer(traj_pos, return_tensors="pt", truncation=True, max_length=256).to(entry_dev)
                enc_neg = tokenizer(traj_neg, return_tensors="pt", truncation=True, max_length=256).to(entry_dev)
                eff_p = min(prompt_tok_len, max(0, min(enc_pos.input_ids.shape[1], enc_neg.input_ids.shape[1]) - 2))

                opt.zero_grad()
                out_pos = model(**enc_pos, use_cache=False)
                p_pos = [cw.last_cumulative_phase for _, cw in cvnn_layers if cw.last_cumulative_phase is not None]
                out_neg = model(**enc_neg, use_cache=False)
                p_neg = [cw.last_cumulative_phase for _, cw in cvnn_layers if cw.last_cumulative_phase is not None]

                lp_pos = F.log_softmax(out_pos.logits[:, :-1, :].float(), dim=-1)
                tok_lp_pos_all = lp_pos.gather(-1, enc_pos.input_ids[:, 1:].to(lp_pos.device).unsqueeze(-1)).squeeze(-1)
                tok_lp_pos = tok_lp_pos_all[:, max(0, eff_p - 1):].mean()

                lp_neg = F.log_softmax(out_neg.logits[:, :-1, :].float(), dim=-1)
                tok_lp_neg_all = lp_neg.gather(-1, enc_neg.input_ids[:, 1:].to(lp_neg.device).unsqueeze(-1)).squeeze(-1)
                tok_lp_neg = tok_lp_neg_all[:, max(0, eff_p - 1):].mean()

                pref_loss = -F.logsigmoid(2.0 * (tok_lp_pos - tok_lp_neg.to(tok_lp_pos.device)))
                phase_loss_g, g_meta = compute_phase_contrastive_loss(p_pos, p_neg, prompt_len=eff_p)
                grpo_obj = pref_loss - 0.1 * tok_lp_pos + float(args.phase_loss_weight) * phase_loss_g.to(tok_lp_pos.device)
                grpo_obj.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                print(
                    f"[SVON-CGW DPO {g_idx:02d}/{len(grpo_prompts)}] obj={grpo_obj.item():.4f} "
                    f"pref={pref_loss.item():.4f} lp_pos={tok_lp_pos.item():.4f} cancel={g_meta['loss_cancel']:.4f} R={g_meta['mean_r_order']:.4f}",
                    flush=True,
                )

    state = collect_svon_cgw_state_dict(model)
    exp = export_svon_cgw_to_rust_qmem(state, out_dir=args.out_dir)
    print(f"[SVON-CGW EXPORT] {json.dumps(exp, indent=2)}", flush=True)
    return exp


def main():
    parser = argparse.ArgumentParser(description="Epoch III Svon-CGW Hybrid CVNN+GNN Trainer")
    parser.add_argument("--model_path", type=str, default=os.path.join(ROOT_DIR, "models", "Qwen3.5-9B-SODALS-Baseline-SafeTensors"))
    parser.add_argument("--prompts_json", type=str, default=os.path.join(SFT_DIR, "math_train_hard_mined_zpd_prompts.json"))
    parser.add_argument("--out_dir", type=str, default=os.path.join(SFT_DIR, "svon_cgw_checkpoints"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--phase_rank", type=int, default=16)
    parser.add_argument("--phase_loss_weight", type=float, default=0.25)
    parser.add_argument("--quantize_4bit", action="store_true", default=None)
    parser.add_argument("--fp16", action="store_true", help="Force FP16 dual-GPU execution (Kaggle T4 x2)")
    parser.add_argument("--online", action="store_true", help="Allow HuggingFace Hub downloads")
    parser.add_argument("--cgw_grpo_steps", type=int, default=0, help="Stage 2 CGW-GRPO group-relative steps")
    parser.add_argument("--stage2_mode", type=str, choices=["grpo", "dpo"], default="grpo", help="Stage 2 mode: real ZPD G=4 SymbolicStepChecker GRPO (default) or legacy synthetic DPO")
    parser.add_argument("--group_size", type=int, default=4, help="Number of sampled rollouts per prompt in Stage 2 CGW-GRPO")
    parser.add_argument("--grpo_max_new_tokens", type=int, default=512, help="Max new tokens per rollout in Stage 2 CGW-GRPO")
    parser.add_argument("--grpo_temp", type=float, default=0.6, help="Sampling temperature in Stage 2 CGW-GRPO")
    parser.add_argument("--d1_adapter_path", type=str, default=None, help="Optional D1 GRPO adapter path")
    parser.add_argument("--d2_adapter_path", type=str, default=None, help="Optional D2 Algebra adapter path")
    parser.add_argument("--svon_adapter_path", type=str, default=None, help="Optional Svon identity adapter path")
    args = parser.parse_args()
    train_svon_cgw(args)


if __name__ == "__main__":
    main()
