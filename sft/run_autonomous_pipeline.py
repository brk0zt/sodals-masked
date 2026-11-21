"""
SODALS v2.1 — Master Autonomous Pipeline: 1536-Token GRPO Training + 150Q Greedy Evaluation
=============================================================================================
Fully automated end-to-end execution:
  Phase 1: Long-Context GRPO Training (25 prompts, G=4, max_new_tokens=1536)
           Target: expert_general_math (balanced_r4) -> qwen_lora_adapter_grpo_long
           Saved to: sft/grpo_long_results.json
  Phase 2: Immediate Head-to-Head Evaluation (150 MATH-500 Questions, Greedy T=0)
           Target: Full GPR-MoA architecture with qwen_lora_adapter_grpo_long
           Saved to: sft/eval_grpo_long_math500_150_results.json
"""

import os
import sys
import time
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
PYTHON_EXE = sys.executable

def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t_str}] {msg}", flush=True)

def main():
    log("=" * 85)
    log("SODALS v2.1 — FULLY AUTONOMOUS TRAINING & EVALUATION PIPELINE")
    log("=" * 85)

    # -------------------------------------------------------------
    # PHASE 1: 1536-TOKEN GRPO TRAINING
    # -------------------------------------------------------------
    log(">>> STARTING PHASE 1: 1536-Token GRPO Training (25 Prompts, G=4, max_new_tokens=1536)...")
    train_cmd = [
        PYTHON_EXE, "-u", os.path.join(SFT_DIR, "train_grpo_long_context.py"),
        "--num_prompts", "25",
        "--start_idx", "0",
        "--max_new_tokens", "1536",
        "--group_size", "4",
        "--beta", "0.04",
        "--lr", "1e-5",
        "--adapter_in", os.path.join(SFT_DIR, "qwen_lora_adapter_balanced_r4"),
        "--adapter_out", os.path.join(SFT_DIR, "qwen_lora_adapter_grpo_long"),
        "--output_json", os.path.join(SFT_DIR, "grpo_long_results.json")
    ]

    t0_train = time.time()
    train_proc = subprocess.run(train_cmd, cwd=ROOT)
    train_duration = (time.time() - t0_train) / 60

    if train_proc.returncode != 0:
        log(f"[ERROR] Phase 1 training exited with error code {train_proc.returncode}! Aborting pipeline.")
        sys.exit(train_proc.returncode)

    log(f"[SUCCESS] Phase 1 Training completed in {train_duration:.1f} minutes!")
    log(f"Trained adapter saved to: {os.path.join(SFT_DIR, 'qwen_lora_adapter_grpo_long')}")

    # Give GPU a 10s cooldown & memory release
    time.sleep(10)

    # -------------------------------------------------------------
    # PHASE 2: 150Q GREEDY EVALUATION
    # -------------------------------------------------------------
    log("\n" + "=" * 85)
    log(">>> STARTING PHASE 2: MATH-500 150Q Head-to-Head Greedy Evaluation (T=0)...")
    log("=" * 85)

    eval_cmd = [
        PYTHON_EXE, "-u", os.path.join(SFT_DIR, "eval_grpo_long_math500_150.py"),
        "--adapter_d1", os.path.join(SFT_DIR, "qwen_lora_adapter_grpo_long"),
        "--output_json", os.path.join(SFT_DIR, "eval_grpo_long_math500_150_results.json")
    ]

    t0_eval = time.time()
    eval_proc = subprocess.run(eval_cmd, cwd=ROOT)
    eval_duration = (time.time() - t0_eval) / 60

    if eval_proc.returncode != 0:
        log(f"[ERROR] Phase 2 evaluation exited with error code {eval_proc.returncode}!")
        sys.exit(eval_proc.returncode)

    log(f"[SUCCESS] Phase 2 Evaluation completed in {eval_duration:.1f} minutes!")
    total_pipeline_time = train_duration + eval_duration
    log("\n" + "=" * 85)
    log(f"AUTONOMOUS PIPELINE FULLY COMPLETED IN {total_pipeline_time:.1f} MINUTES ({total_pipeline_time/60:.2f} HOURS)!")
    log(f"Training Results:   {os.path.join(SFT_DIR, 'grpo_long_results.json')}")
    log(f"Evaluation Results: {os.path.join(SFT_DIR, 'eval_grpo_long_math500_150_results.json')}")
    log("=" * 85)

if __name__ == "__main__":
    main()
