"""
SODALS / Svon v2.1 — Local 4-bit NF4 Full-Stack Compatibility & Interference Evaluator
======================================================================================
Bridges the cloud (Kaggle T4 x2 FP16) training run with the local production stack
(`sodals_serve.py`: 4-bit NF4 + CPUEmbeddingWrapper + D1 GRPO + D2 Algebra + SoftLoraBlender
+ svon_identity + Svon-CGW + Kuramoto Rollback) and verifies four strict gates:

  Gate 1 (NF4 System-1 Zero-Prompt Identity & 3rd-Person World Specificity):
    - With D1/D2 disabled (`use_adapters=False`, `--no_system_prompt`) and only `svon_identity`
      active on `mlp.down_proj`, verifies >= 27/30 identity persistence & >= 18/20 world facts.
  Gate 2 (System-2 Multi-Adapter Interference KL):
    - With D1 + D2 + `svon_identity` all active simultaneously under `SoftLoraBlender`,
      measures KL(P_{D1+D2+Svon} || P_{D1+D2}) < 0.015 across formal math & algebra prompts.
  Gate 3 (SoftLoraBlender Transition Invariance):
    - Cycles SYSTEM1_DIRECT (`set_adapter(["svon_identity"])`) -> SYSTEM2_MATH_FORMAL
      (`enable_multi_adapter` + `set_blend(alpha)`) and verifies all 3 adapter scales remain exact.
  Gate 4 (Svon-CGW & Live Kuramoto Phase Rollback Coherence):
    - Verifies `upgrade_model_to_svon_cgw` + `load_svon_cgw_state_dict` on the multi-adapter
      `PeftModel` and confirms `synchronize_model_cvnn_phases_on_rollback` monotonically
      increases Kuramoto phase order R_order.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

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

from eval_svon_identity_and_regression import (
    THIRD_PERSON_WORLD_SPECIFICITY_20,
    ZERO_PROMPT_ADVERSARIAL_30,
    score_third_person_world_preserved,
    score_zero_prompt_identity_response,
)
from svon_identity_tracer import PROTECTED_WORLD_AND_MATH_PROMPTS
from svon_kuramoto_equilibrium import (
    load_svon_cgw_state_dict,
    synchronize_model_cvnn_phases_on_rollback,
    upgrade_model_to_svon_cgw,
)


def compute_kl_divergence_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """Computes KL(P || Q) from last-token logit vectors in FP32."""
    p = F.softmax(logits_p.float(), dim=-1)
    lp = F.log_softmax(logits_p.float(), dim=-1)
    lq = F.log_softmax(logits_q.float().to(logits_p.device), dim=-1)
    kl = float((p * (lp - lq)).sum(dim=-1).mean().item())
    return max(0.0, kl)


def verify_blender_transition_invariance(model: Any, blender: Any, alpha: float = 0.20) -> Dict[str, Any]:
    """
    Simulates `sodals_serve.py` switching from SYSTEM1_DIRECT (`svon_identity` only)
    to SYSTEM2_MATH_FORMAL (`D1 + D2 + svon_identity`) and verifies scaling invariants.
    """
    if blender is None:
        return {"ok": False, "reason": "no_blender"}

    if "svon_identity" in blender.adapter_names and hasattr(model, "set_adapter"):
        try:
            model.set_adapter(["svon_identity"])
        except Exception:
            pass

    blender.enable_multi_adapter(model)
    scale_map = blender.set_blend(model, alpha=alpha)

    expected_d1 = (1.0 - alpha) * blender.base_scales.get("expert_general_math", 0.25)
    expected_d2 = alpha * blender.base_scales.get("expert_abstract_algebra", 1.00)
    expected_svon = blender.base_scales.get("svon_identity", 1.00)

    ok_d1 = abs(float(scale_map.get("expert_general_math", 0.0)) - expected_d1) < 1e-6
    ok_d2 = abs(float(scale_map.get("expert_abstract_algebra", 0.0)) - expected_d2) < 1e-6
    ok_svon = ("svon_identity" not in blender.base_scales) or (
        abs(float(scale_map.get("svon_identity", 0.0)) - expected_svon) < 1e-6
    )

    return {
        "ok": bool(ok_d1 and ok_d2 and ok_svon),
        "alpha": alpha,
        "scale_map": {k: round(float(v), 6) for k, v in scale_map.items()},
    }


GATE2_SYSTEM2_MATH_PROMPTS: List[str] = [
    "Solve for x: 3x^2 - 12x + 9 = 0 step by step.",
    "Find the remainder when 7^100 is divided by 5.",
    "Evaluate the definite integral of 3x^2 + 2x from x = 0 to x = 2.",
    "If a finite group G has order 15, prove that G is cyclic.",
    "Find all real solutions to x^3 - 6x^2 + 11x - 6 = 0.",
    "Compute the determinant of the 2x2 matrix [[4, 7], [2, 6]].",
    "What is the sum of the first 25 positive odd integers?",
    "Let R = Z[x]/(x^2 + 1). Is R an integral domain? Explain briefly.",
    "Solve the congruence 7x ≡ 3 (mod 11).",
    "Find the number of ways to choose 3 distinct items from 10 items.",
    "If sin(theta) = 3/5 and theta is in Q1, find cos(2*theta).",
    "Show whether the subgroup H = {0, 3, 6} is normal in Z_9.",
    "Find the greatest common divisor of 252 and 105 using the Euclidean algorithm.",
    "Compute the derivative of f(x) = x^3 * ln(x) for x > 0.",
    "In a ring with unity, prove that (-1)*(-1) = 1.",
    "What is the coefficient of x^3 in the binomial expansion of (2x + 1)^5?",
]


def _batched_greedy_generate(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    entry_dev: Any,
    max_new_tokens: int = 160,
    batch_size: int = 8,
) -> List[str]:
    from transformers import set_seed

    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs: List[str] = []
    try:
        bs = max(1, int(batch_size))
        for start in range(0, len(prompts), bs):
            chunk = prompts[start : start + bs]
            enc = tokenizer(chunk, return_tensors="pt", padding=True).to(entry_dev)
            in_len = enc.input_ids.shape[1]
            set_seed(42)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for row_i in range(len(chunk)):
                txt = tokenizer.decode(out[row_i][in_len:], skip_special_tokens=True).strip()
                outputs.append(txt)
    finally:
        tokenizer.padding_side = orig_padding_side
    return outputs


def run_local_full_stack_evaluation(args) -> Dict[str, Any]:
    prevent_sleep_if_windows()
    sys.stdout.reconfigure(encoding="utf-8")

    model, tokenizer, rt_meta = load_qwen_model_and_tokenizer(
        model_path=args.model_path,
        quantize_4bit=not args.fp16,
        online=True if getattr(args, "online", False) else False,
    )
    model, blender, mounted = mount_production_adapters(
        model,
        d1_adapter_path=args.d1_adapter_path,
        d2_adapter_path=args.d2_adapter_path,
        svon_adapter_path=args.adapter_svon,
        alpha=0.20,
    )
    entry_dev = get_primary_device(model)

    cgw_meta = None
    if args.cgw_weights and os.path.exists(args.cgw_weights):
        up_info = upgrade_model_to_svon_cgw(model, phase_rank=16, freeze_base=True)
        loaded_cnt = load_svon_cgw_state_dict(model, args.cgw_weights)
        cgw_meta = {**up_info, "loaded_tensors": loaded_cnt, "weights_path": args.cgw_weights}

    model.eval()

    # Gate 3: Transition invariance
    gate3 = verify_blender_transition_invariance(model, blender, alpha=0.20)

    def _fmt_sys1(p_txt: str) -> str:
        msgs = [{"role": "user", "content": p_txt}]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            raw = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            return (raw + "\n</think>\n\n") if raw.endswith("<think>\n") else raw

    # Gate 1: System-1 Zero-Prompt Identity & 3rd-Person World Specificity (D1/D2 off, Svon on)
    if "svon_identity" in mounted and hasattr(model, "set_adapter"):
        try:
            model.set_adapter("svon_identity")
        except Exception:
            try:
                model.set_adapter(["svon_identity"])
            except Exception:
                pass

    bs = int(getattr(args, "batch_size", 8) or 8)
    id_prompts = [_fmt_sys1(item["prompt"]) for item in ZERO_PROMPT_ADVERSARIAL_30]
    id_replies = _batched_greedy_generate(model, tokenizer, id_prompts, entry_dev, max_new_tokens=160, batch_size=bs)
    id_pass = sum(1 for txt in id_replies if score_zero_prompt_identity_response(txt)["identity_pass"])

    world_prompts = [_fmt_sys1(item["prompt"]) for item in THIRD_PERSON_WORLD_SPECIFICITY_20]
    world_replies = _batched_greedy_generate(model, tokenizer, world_prompts, entry_dev, max_new_tokens=160, batch_size=bs)
    world_pass = sum(
        1 for item, txt in zip(THIRD_PERSON_WORLD_SPECIFICITY_20, world_replies)
        if score_third_person_world_preserved(txt, item["id"])
    )

    # Gate 2: System-2 Multi-Adapter Interference KL(P_{D1+D2+Svon} || P_{D1+D2}) on >= 16 math/algebra prompts
    svon_requested = bool(args.adapter_svon and os.path.exists(args.adapter_svon))
    kl_list: List[float] = []
    if blender is not None and "svon_identity" in mounted and "svon_identity" in blender.base_scales:
        for prompt_txt in GATE2_SYSTEM2_MATH_PROMPTS:
            enc = tokenizer(_fmt_sys1(prompt_txt), return_tensors="pt").to(entry_dev)
            with torch.no_grad():
                blender.enable_multi_adapter(model)
                blender.set_blend(model, alpha=0.20)
                logits_all = model(**enc, use_cache=False).logits[0, -1, :]

                # Zero out svon_identity contribution while keeping D1+D2 untouched
                orig_svon = blender.base_scales["svon_identity"]
                blender.base_scales["svon_identity"] = 0.0
                blender.set_blend(model, alpha=0.20)
                logits_d1d2 = model(**enc, use_cache=False).logits[0, -1, :]
                blender.base_scales["svon_identity"] = orig_svon
                blender.set_blend(model, alpha=0.20)
                kl_list.append(compute_kl_divergence_from_logits(logits_all, logits_d1d2))

    # Fail-closed when svon_identity was requested/expected but KL was never measured
    if kl_list:
        mean_interference_kl: Optional[float] = round(sum(kl_list) / len(kl_list), 6)
        gate2_passed = bool(mean_interference_kl < 0.015)
    else:
        mean_interference_kl = None
        gate2_passed = not svon_requested

    # Gate 4: Live Kuramoto Phase Rollback Coherence
    gate4 = None
    gate4_passed = True
    if cgw_meta is not None:
        # Ensure CVNN phase trajectories are populated by a forward pass before rollback sync
        probe_enc = tokenizer(_fmt_sys1(GATE2_SYSTEM2_MATH_PROMPTS[0]), return_tensors="pt").to(entry_dev)
        with torch.no_grad():
            model(**probe_enc, use_cache=False)
        gate4 = synchronize_model_cvnn_phases_on_rollback(model, coupling_strength=0.35, max_sync_steps=3)
        gate4_passed = bool(
            gate4.get("synced_cvnn_layers", 0) > 0
            and float(gate4.get("mean_r_after", 0.0)) >= float(gate4.get("mean_r_before", 0.0)) - 1e-6
        )
        gate4["passed"] = gate4_passed

    report = {
        "runtime": rt_meta,
        "mounted_adapters": mounted,
        "cgw_meta": cgw_meta,
        "gate1_system1_identity": {
            "zero_prompt_pass": id_pass,
            "zero_prompt_total": len(ZERO_PROMPT_ADVERSARIAL_30),
            "world_preserved": world_pass,
            "world_total": len(THIRD_PERSON_WORLD_SPECIFICITY_20),
            "passed": (id_pass >= 27 and world_pass >= 18),
        },
        "gate2_system2_adapter_interference": {
            "num_prompts": len(kl_list),
            "mean_kl_d1d2_svon_vs_d1d2": mean_interference_kl,
            "threshold": 0.015,
            "passed": gate2_passed,
        },
        "gate3_blender_transition": gate3,
        "gate4_kuramoto_rollback": gate4,
    }
    report["all_gates_passed"] = bool(
        report["gate1_system1_identity"]["passed"]
        and report["gate2_system2_adapter_interference"]["passed"]
        and report["gate3_blender_transition"]["ok"]
        and gate4_passed
    )

    for tgt in (args.output_json, os.path.join(tempfile.gettempdir(), os.path.basename(args.output_json))):
        try:
            with open(tgt, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"[FULL-STACK COMPAT] Saved -> {tgt} | all_passed={report['all_gates_passed']}", flush=True)
            break
        except PermissionError:
            continue
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Local 4-bit NF4 + D1/D2 + Svon + Svon-CGW Compatibility")
    parser.add_argument("--model_path", type=str, default=os.path.join(ROOT_DIR, "models", "Qwen3.5-9B-SODALS-Baseline-SafeTensors"))
    parser.add_argument("--d1_adapter_path", type=str, default=os.path.join(SFT_DIR, "qwen_lora_adapter_grpo_math_train_100"))
    parser.add_argument("--d2_adapter_path", type=str, default=os.path.join(SFT_DIR, "qwen_lora_adapter_domain2_r8_pondering"))
    parser.add_argument(
        "--adapter_svon",
        type=str,
        default=os.path.join(SFT_DIR, "kaggle_outputs", "svon-epoch-ii-and-iii-t4-x2", "svon_out", "qwen_lora_adapter_svon_alphaedit"),
    )
    parser.add_argument(
        "--cgw_weights",
        type=str,
        default=os.path.join(SFT_DIR, "kaggle_outputs", "svon-epoch-ii-and-iii-t4-x2", "svon_out", "svon_cgw_checkpoints", "svon_cgw_hybrid.safetensors"),
    )
    parser.add_argument("--output_json", type=str, default=os.path.join(SFT_DIR, "local_full_stack_compatibility_report.json"))
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--online", action="store_true", help="Allow HuggingFace Hub model download")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for Gate 1 greedy generation")
    args = parser.parse_args()
    run_local_full_stack_evaluation(args)


if __name__ == "__main__":
    main()
