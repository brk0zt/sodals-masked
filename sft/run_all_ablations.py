"""
SODALS v2.1 — Master Ablation Suite Runner
===========================================
Sequentially executes:
  1. Ablation 1: Only Hamiltonian Curriculum Ordering (Stages 1-5)
  2. Ablation 2: Only Arkhe Focal Loss (Dynamic w_i)
  3. Ablation 4: Hybrid + Pondering Core (CliffordBridge Logit Steering Bridge)

For each ablation:
  - Trains the 9B model on RTX 4060 GPU (72 steps, batch 1, grad_accum 4)
  - Merges weights into standalone SafeTensors
  - Evaluates 8 Official CAIS HLE questions (closed-book greedy)
  - Evaluates 4 OOD Mathematical Perturbation tests
"""

import os
import sys
import subprocess
import time

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")

ABLATIONS = [
    {
        "name": "Ablation 1: Only Hamiltonian Ordering",
        "train_script": os.path.join(SFT_DIR, "train_qwen_lora_only_hamiltonian.py"),
        "adapter_dir": os.path.join(SFT_DIR, "qwen_lora_adapter_only_hamiltonian"),
        "merged_dir": os.path.join(ROOT, "models", "Qwen3.5-9B-SODALS-Only-Hamiltonian-SafeTensors"),
        "hle_out": os.path.join(ROOT, "hle_results_only_hamiltonian_20260818.json"),
        "pert_out": os.path.join(ROOT, "hle_perturbations_only_hamiltonian_20260818.json")
    },
    {
        "name": "Ablation 2: Only Arkhe Focal Loss",
        "train_script": os.path.join(SFT_DIR, "train_qwen_lora_only_focal_loss.py"),
        "adapter_dir": os.path.join(SFT_DIR, "qwen_lora_adapter_only_focal_loss"),
        "merged_dir": os.path.join(ROOT, "models", "Qwen3.5-9B-SODALS-Only-FocalLoss-SafeTensors"),
        "hle_out": os.path.join(ROOT, "hle_results_only_focal_loss_20260818.json"),
        "pert_out": os.path.join(ROOT, "hle_perturbations_only_focal_loss_20260818.json")
    },
    {
        "name": "Ablation 4: Hybrid + Pondering Core Bridge",
        "train_script": os.path.join(SFT_DIR, "train_qwen_lora_with_pondering.py"),
        "adapter_dir": os.path.join(SFT_DIR, "qwen_lora_adapter_with_pondering"),
        "merged_dir": os.path.join(ROOT, "models", "Qwen3.5-9B-SODALS-With-Pondering-SafeTensors"),
        "hle_out": os.path.join(ROOT, "hle_results_with_pondering_20260818.json"),
        "pert_out": os.path.join(ROOT, "hle_perturbations_with_pondering_20260818.json")
    }
]


def run_pipeline():
    print("=" * 80)
    print("  🚀 SODALS v2.1 MASTER ABLATION PIPELINE LAUNCHED")
    print("=" * 80)

    for idx, ab in enumerate(ABLATIONS, 1):
        name = ab["name"]
        print(f"\n{'='*80}")
        print(f"  [{idx}/{len(ABLATIONS)}] STARTING: {name}")
        print(f"{'='*80}")

        # 1. Train
        print(f"\n[STEP 1/3] Executing Training Script: {ab['train_script']}...")
        t0 = time.time()
        res_train = subprocess.run([sys.executable, ab["train_script"]], cwd=ROOT)
        if res_train.returncode != 0:
            print(f"❌ Training FAILED for {name} with return code {res_train.returncode}")
            sys.exit(1)
        print(f"✅ Training completed in {(time.time() - t0)/60:.1f} minutes.")

        # 2. Merge & Evaluate
        print(f"\n[STEP 2/3] Executing Weight Merge and HLE + OOD Evaluations...")
        eval_script = os.path.join(SFT_DIR, "merge_and_eval_ablation.py")
        cmd = [
            sys.executable, eval_script,
            "--adapter", ab["adapter_dir"],
            "--merged_dir", ab["merged_dir"],
            "--hle_out", ab["hle_out"],
            "--pert_out", ab["pert_out"],
            "--tag", name
        ]
        res_eval = subprocess.run(cmd, cwd=ROOT)
        if res_eval.returncode != 0:
            print(f"❌ Evaluation FAILED for {name} with return code {res_eval.returncode}")
            sys.exit(1)
        print(f"✅ Merge and Evaluation completed successfully for {name}!")

    # 3. Final Master Report Compilation
    print(f"\n{'='*80}")
    print("  [STEP 3/3] Compiling Master Factorial Comparison Reports...")
    print(f"{'='*80}")
    rep_script = os.path.join(SFT_DIR, "build_ultimate_master_report.py")
    subprocess.run([sys.executable, rep_script], cwd=ROOT)
    print("🎉 ALL ABLATION RUNS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_pipeline()
