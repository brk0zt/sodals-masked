"""
SODALS v2.1 — Chained Post-Fix Evaluation Runner
=================================================
1. Monitors the active Clean-145 evaluation process (PID 21976).
2. As soon as PID 21976 terminates and writes its final checkpoint,
   summarizes the pre-fix baseline results.
3. Automatically launches the clean held-out evaluation on the POST-FIX codebase:
   - Output: sft/eval_grpo_math_train_clean_145_post_fix_results.json
   - Full 145 held-out test prompts (Indices 25..170)
   - Evaluates with the active dual-LoRA Soft Blender, balanced-brace SymPy grader,
     and false-positive-free CST verifier.
4. Generates a head-to-head comparison diff between pre-fix and post-fix runs.
"""

import os
import sys
import time
import json
import subprocess
import psutil

TARGET_PID = 21976
PYTHON_EXE = r"C:\Windows.old\Users\Burak\AppData\Local\Programs\Python\Python310\python.exe"
SFT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SFT_DIR)
PRE_FIX_JSON = os.path.join(SFT_DIR, "eval_grpo_math_train_clean_145_results.json")
POST_FIX_JSON = os.path.join(SFT_DIR, "eval_grpo_math_train_clean_145_post_fix_results.json")
EVAL_SCRIPT = os.path.join(SFT_DIR, "eval_grpo_math_train_clean_heldout.py")

print(f"[CHAIN-RUNNER] Monitoring active baseline evaluation PID {TARGET_PID}...", flush=True)

# 1. Wait for PID 21976 to finish
while True:
    try:
        proc = psutil.Process(TARGET_PID)
        if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
            time.sleep(15)
        else:
            break
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        break

print(f"\n[CHAIN-RUNNER] Baseline evaluation process PID {TARGET_PID} has completed!", flush=True)
time.sleep(5)  # allow disk flush and lock release

# 2. Check pre-fix results
if os.path.exists(PRE_FIX_JSON):
    try:
        with open(PRE_FIX_JSON, "r", encoding="utf-8") as f:
            d = json.load(f)
        total = d.get("total_evaluated", len(d.get("details", [])))
        pass_count = d.get("norm_sym_grpo_pass", 0)
        base_pass = d.get("norm_sym_base_pass", 0)
        print(f"[CHAIN-RUNNER] Pre-Fix Baseline Run Finished: GRPO {pass_count}/{total} PASS ({100.0 * pass_count / max(1, total):.1f}%) vs Base {base_pass}/{total} ({100.0 * base_pass / max(1, total):.1f}%).", flush=True)
    except Exception as e:
        print(f"[CHAIN-RUNNER] Notice on pre-fix JSON inspection: {e}", flush=True)

# 3. Launch Post-Fix evaluation
cmd = [
    PYTHON_EXE,
    "-u",
    EVAL_SCRIPT,
    "--start_idx", "25",
    "--end_idx", "170",
    "--output_json", POST_FIX_JSON
]

print(f"\n[CHAIN-RUNNER] ===========================================================================", flush=True)
print(f"[CHAIN-RUNNER] LAUNCHING POST-FIX SODALS EVALUATION (Indices 25..170, Clean Heldout)", flush=True)
print(f"[CHAIN-RUNNER] Output target: {POST_FIX_JSON}", flush=True)
print(f"[CHAIN-RUNNER] ===========================================================================\n", flush=True)

env = os.environ.copy()
env["HF_HUB_OFFLINE"] = "1"
env["TRANSFORMERS_OFFLINE"] = "1"
env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

proc = subprocess.Popen(cmd, cwd=REPO_DIR, env=env)
proc.wait()

print(f"\n[CHAIN-RUNNER] POST-FIX EVALUATION COMPLETE with exit code {proc.returncode}!", flush=True)

# 4. Generate Head-to-Head Comparative Summary
if os.path.exists(PRE_FIX_JSON) and os.path.exists(POST_FIX_JSON):
    try:
        with open(PRE_FIX_JSON, "r", encoding="utf-8") as f:
            pre_data = json.load(f)
        with open(POST_FIX_JSON, "r", encoding="utf-8") as f:
            post_data = json.load(f)

        pre_pass = pre_data.get("norm_sym_grpo_pass", 0)
        post_pass = post_data.get("norm_sym_grpo_pass", 0)
        total = post_data.get("total_evaluated", len(post_data.get("details", [])))

        print("\n" + "=" * 75, flush=True)
        print("HEAD-TO-HEAD COMPARATIVE SUMMARY (PRE-FIX vs POST-FIX)", flush=True)
        print("=" * 75, flush=True)
        print(f"Pre-Fix Evaluation  : {pre_pass}/{total} PASS ({100.0 * pre_pass / max(1, total):.2f}%)", flush=True)
        print(f"Post-Fix Evaluation : {post_pass}/{total} PASS ({100.0 * post_pass / max(1, total):.2f}%)", flush=True)
        delta = post_pass - pre_pass
        sign = "+" if delta >= 0 else ""
        print(f"Net Pass Delta      : {sign}{delta} solved problems ({sign}{100.0 * delta / max(1, total):.2f}%)", flush=True)
        print("=" * 75, flush=True)
    except Exception as e:
        print(f"[CHAIN-RUNNER] Could not generate comparative summary: {e}", flush=True)
