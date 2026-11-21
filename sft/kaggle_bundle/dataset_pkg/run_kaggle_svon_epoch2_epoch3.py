"""
Kaggle T4 x2 / Linux Cloud Orchestrator for Epoch II (Svon) & Epoch III (Svon-CGW)
===================================================================================
Offloads heavy Epoch II (Identity Tweezer Surgery) and Epoch III (Holomorphic
CVNN-DeltaNet + 2-Hop Sheaf-GNN training) onto dual-GPU Kaggle T4 x2 (32 GB VRAM)
without touching local Epoch I evaluation.

Usage on Kaggle / Linux:
  python run_kaggle_svon_epoch2_epoch3.py --stage all --fp16 --online \
      --model_path Qwen/Qwen3.5-9B --work_dir /kaggle/working/svon_out
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import zipfile
from typing import Any, Dict, Optional

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
if SFT_DIR not in sys.path:
    sys.path.insert(0, SFT_DIR)

from cloud_runtime_utils import configure_hf_env, detect_gpu_profile, maybe_enable_sac_compat

maybe_enable_sac_compat()


def _release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def resolve_default_model_path(online: bool = False) -> str:
    local_cand = os.path.join(os.path.dirname(SFT_DIR), "models", "Qwen3.5-9B-SODALS-Baseline-SafeTensors")
    if os.path.exists(local_cand):
        return local_cand
    for root in ("/kaggle/input", os.getcwd()):
        if os.path.exists(root):
            for dirpath, _dirnames, filenames in os.walk(root):
                if "config.json" in filenames and any("qwen" in dirpath.lower() or "sodals" in dirpath.lower() for _ in [0]):
                    if any(fn.endswith(".safetensors") for fn in filenames):
                        return dirpath
    return "Qwen/Qwen3.5-9B" if online else local_cand


def bundle_artifacts(work_dir: str, zip_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _dirnames, filenames in os.walk(work_dir):
            # Exclude huge 18 GB full FP16 model directory from automatic small artifact zip unless requested
            if "fp16_edited_model" in dirpath:
                continue
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                arc = os.path.relpath(fp, work_dir)
                zf.write(fp, arcname=arc)
    return zip_path


def run_cloud_pipeline(args) -> Dict[str, Any]:
    configure_hf_env(online=args.online)
    gpu_prof = detect_gpu_profile()
    print(f"[CLOUD RUNNER] GPU Profile: {json.dumps(gpu_prof)}", flush=True)

    model_path = args.model_path or resolve_default_model_path(online=args.online)
    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    q4 = False if args.fp16 else (True if args.quantize_4bit else None)

    auto_d1 = os.path.join(SFT_DIR, "prod_adapters", "d1")
    auto_d2 = os.path.join(SFT_DIR, "prod_adapters", "d2")
    d1_path = args.d1_adapter_path or (auto_d1 if os.path.exists(auto_d1) else None)
    d2_path = args.d2_adapter_path or (auto_d2 if os.path.exists(auto_d2) else None)

    summary: Dict[str, Any] = {
        "gpu_profile": gpu_prof,
        "model_path": model_path,
        "work_dir": work_dir,
        "d1_adapter_path": d1_path,
        "d2_adapter_path": d2_path,
    }

    if args.stage in ("epoch2", "all"):
        from svon_identity_tracer import run_gpu_identity_trace
        from svon_alphaedit_surgeon import run_alphaedit_surgery
        from eval_svon_identity_and_regression import run_gpu_svon_validation

        trace_json = os.path.join(work_dir, "svon_tweezer_neuron_map.json")
        adapter_dir = os.path.join(work_dir, "qwen_lora_adapter_svon_alphaedit")
        val_json = os.path.join(work_dir, "svon_identity_regression_report.json")

        print("\n=== [EPOCH II.1] Tweezer SwiGLU Neuron Trace ===", flush=True)
        run_gpu_identity_trace(
            model_path=model_path,
            output_json=trace_json,
            top_layers=args.top_layers,
            neurons_per_layer=args.neurons_per_layer,
            min_ratio=args.min_ratio,
            quantize_4bit=q4,
            online=args.online,
        )
        _release_gpu_memory()

        print("\n=== [EPOCH II.2] Null-Space AlphaEdit Surgery (Path A SVD LoRA) ===", flush=True)
        surg = run_alphaedit_surgery(
            model_path=model_path,
            trace_json=trace_json,
            adapter_out_dir=adapter_dir,
            ridge_lambda=args.ridge_lambda,
            max_rel_frob=args.max_rel_frob,
            max_rank=args.max_rank,
            export_fp16_dir=None,
            quantize_4bit=q4,
            online=args.online,
            d1_adapter_path=d1_path,
            d2_adapter_path=d2_path,
        )
        _release_gpu_memory()
        summary["epoch2_surgery"] = surg

        print("\n=== [EPOCH II.3] Zero-Prompt & 3rd-Person Specificity Validation ===", flush=True)
        eval_ns = argparse.Namespace(
            model_path=model_path,
            adapter_path=adapter_dir,
            output_json=val_json,
            fp16=args.fp16,
            quantize_4bit=args.quantize_4bit,
            online=args.online,
        )
        run_gpu_svon_validation(eval_ns)
        _release_gpu_memory()
        val_report = json.load(open(val_json, "r", encoding="utf-8"))
        summary["epoch2_validation"] = {
            "zero_prompt_pass": val_report["zero_prompt_identity_pass"],
            "world_preserved": val_report["third_person_world_preserved"],
            "mean_kl": val_report["mean_next_token_kl"],
            "passes_gates": val_report["passes_gates"],
        }

        if val_report.get("passes_gates") and args.export_path_b_on_pass:
            fp16_dir = os.path.join(work_dir, "fp16_edited_model")
            print(f"\n=== [EPOCH II.4] Gates PASSED -> Exporting Path B FP16 Model -> {fp16_dir} ===", flush=True)
            run_alphaedit_surgery(
                model_path=model_path,
                trace_json=trace_json,
                adapter_out_dir=adapter_dir,
                ridge_lambda=args.ridge_lambda,
                max_rel_frob=args.max_rel_frob,
                max_rank=args.max_rank,
                export_fp16_dir=fp16_dir,
                quantize_4bit=False,
                online=args.online,
            )

    if args.stage in ("epoch3", "all"):
        from train_svon_cgw_architecture import train_svon_cgw

        cgw_dir = os.path.join(work_dir, "svon_cgw_checkpoints")
        svon_cand = os.path.join(work_dir, "qwen_lora_adapter_svon_alphaedit")
        prompts_json = args.prompts_json or os.path.join(SFT_DIR, "math_train_hard_mined_zpd_prompts.json")
        print("\n=== [EPOCH III] Training Svon-CGW (CVNN-DeltaNet + 2-Hop Sheaf-GNN) ===", flush=True)
        cgw_ns = argparse.Namespace(
            model_path=model_path,
            prompts_json=prompts_json,
            out_dir=cgw_dir,
            steps=args.cgw_steps,
            cgw_grpo_steps=args.cgw_grpo_steps,
            lr=args.cgw_lr,
            phase_rank=args.cgw_phase_rank,
            phase_loss_weight=args.cgw_phase_loss_weight,
            fp16=args.fp16,
            quantize_4bit=args.quantize_4bit,
            online=args.online,
            d1_adapter_path=d1_path,
            d2_adapter_path=d2_path,
            svon_adapter_path=svon_cand if os.path.exists(svon_cand) else None,
        )
        cgw_exp = train_svon_cgw(cgw_ns)
        summary["epoch3_cgw_export"] = cgw_exp

    summary_path = os.path.join(work_dir, "cloud_pipeline_summary.json")
    if os.path.exists(summary_path):
        try:
            prev_summary = json.load(open(summary_path, "r", encoding="utf-8"))
            if isinstance(prev_summary, dict):
                prev_summary.update(summary)
                summary = prev_summary
        except Exception:
            pass
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.out_zip:
        zpath = bundle_artifacts(work_dir, args.out_zip)
        summary["bundle_zip"] = zpath
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[CLOUD RUNNER] Bundled lightweight download artifacts -> {zpath}", flush=True)

    print(f"[CLOUD RUNNER] Complete -> {summary_path}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Kaggle T4 x2 Orchestrator for Svon Epoch II & III")
    parser.add_argument("--stage", choices=["epoch2", "epoch3", "all"], default="all")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--prompts_json", type=str, default=None)
    parser.add_argument("--work_dir", type=str, default="/kaggle/working/svon_out" if os.path.exists("/kaggle") else os.path.join(SFT_DIR, "cloud_svon_out"))
    parser.add_argument("--out_zip", type=str, default="/kaggle/working/svon_kaggle_artifacts.zip" if os.path.exists("/kaggle") else None)
    parser.add_argument("--fp16", action="store_true", help="Run full FP16 across Dual-T4 (32 GB VRAM)")
    parser.add_argument("--quantize_4bit", action="store_true", help="Force 4-bit NF4")
    parser.add_argument("--online", action="store_true", help="Enable HuggingFace Hub downloads")
    parser.add_argument("--export_path_b_on_pass", action="store_true")

    # Epoch II hyperparams
    parser.add_argument("--top_layers", type=int, default=8)
    parser.add_argument("--neurons_per_layer", type=int, default=256)
    parser.add_argument("--min_ratio", type=float, default=1.25)
    parser.add_argument("--ridge_lambda", type=float, default=2.0)
    parser.add_argument("--max_rel_frob", type=float, default=0.014)
    parser.add_argument("--max_rank", type=int, default=24)

    # Epoch III hyperparams
    parser.add_argument("--cgw_steps", type=int, default=50)
    parser.add_argument("--cgw_grpo_steps", type=int, default=15)
    parser.add_argument("--cgw_lr", type=float, default=1e-3)
    parser.add_argument("--cgw_phase_rank", type=int, default=16)
    parser.add_argument("--cgw_phase_loss_weight", type=float, default=0.25)
    parser.add_argument("--d1_adapter_path", type=str, default=None)
    parser.add_argument("--d2_adapter_path", type=str, default=None)

    args = parser.parse_args()
    run_cloud_pipeline(args)


if __name__ == "__main__":
    main()
