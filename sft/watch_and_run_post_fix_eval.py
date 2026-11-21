"""
SODALS v2.1 — Robust Chained Post-Fix Evaluation Watcher
=========================================================
Monitors eval_grpo_math_train_clean_145_results.json until all 145 questions
(Q026 through Q170) are fully evaluated and saved.
Once completed, it immediately launches the Clean-145 evaluation on the
post-fix codebase (targeting eval_grpo_math_train_clean_145_post_fix_results.json).
"""

import os
import sys
import time
import json
import subprocess

SFT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SFT_DIR)
PRE_FIX_JSON = os.path.join(SFT_DIR, "eval_grpo_math_train_clean_145_results.json")
POST_FIX_JSON = os.path.join(SFT_DIR, "eval_grpo_math_train_clean_145_post_fix_results.json")
EVAL_SCRIPT = os.path.join(SFT_DIR, "eval_grpo_math_train_clean_heldout.py")
PYTHON_EXE = r"C:\Windows.old\Users\Burak\AppData\Local\Programs\Python\Python310\python.exe"
TARGET_TOTAL = 145

print(f"[POST-FIX WATCHER] Initialized. Monitoring {PRE_FIX_JSON} for {TARGET_TOTAL} questions...", flush=True)

last_count = -1
stagnant_checks = 0

while True:
    current_count = 0
    if os.path.exists(PRE_FIX_JSON):
        try:
            with open(PRE_FIX_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            details = data.get("details", [])
            current_count = len(details)
            last_q = details[-1]["index"] if details else None
            
            if current_count != last_count:
                print(f"[POST-FIX WATCHER] Progress: {current_count}/{TARGET_TOTAL} questions evaluated (Last: Q{last_q:03d}).", flush=True)
                last_count = current_count
                stagnant_checks = 0
            else:
                stagnant_checks += 1
                
            if current_count >= TARGET_TOTAL:
                print(f"\n[POST-FIX WATCHER] BASELINE RUN COMPLETE! All {TARGET_TOTAL} questions evaluated.", flush=True)
                break
        except Exception as e:
            # File might be mid-atomic-write
            pass
            
    time.sleep(20)

# Allow 15 seconds for parent process to completely terminate and release GPU VRAM
print("[POST-FIX WATCHER] Waiting 15s for GPU memory release...", flush=True)
time.sleep(15)

# Launch post-fix evaluation directly
cmd = [
    PYTHON_EXE,
    "-u",
    EVAL_SCRIPT,
    "--start_idx", "25",
    "--end_idx", "170",
    "--output_json", POST_FIX_JSON
]

print(f"\n[POST-FIX WATCHER] ===========================================================================", flush=True)
print(f"[POST-FIX WATCHER] LAUNCHING POST-FIX EVALUATION (Q026..Q170)", flush=True)
print(f"[POST-FIX WATCHER] Output: {POST_FIX_JSON}", flush=True)
print(f"[POST-FIX WATCHER] ===========================================================================\n", flush=True)

env = os.environ.copy()
env["HF_HUB_OFFLINE"] = "1"
env["TRANSFORMERS_OFFLINE"] = "1"
env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

proc = subprocess.run(cmd, cwd=REPO_DIR, env=env)
print(f"\n[POST-FIX WATCHER] Post-fix evaluation exited with code {proc.returncode}.", flush=True)

# Generate final comparative report
if os.path.exists(PRE_FIX_JSON) and os.path.exists(POST_FIX_JSON):
    try:
        with open(PRE_FIX_JSON, "r", encoding="utf-8") as f:
            pre = json.load(f)
        with open(POST_FIX_JSON, "r", encoding="utf-8") as f:
            post = json.load(f)
            
        pre_pass = pre.get("norm_sym_grpo_pass", 0)
        post_pass = post.get("norm_sym_grpo_pass", 0)
        tot = len(post.get("details", []))
        
        print("\n" + "="*75, flush=True)
        print("FINAL AUDIT COMPARISON: PRE-FIX vs POST-FIX", flush=True)
        print("="*75, flush=True)
        print(f"Pre-Fix Baseline   : {pre_pass}/{tot} PASS ({100.0*pre_pass/max(1,tot):.2f}%)", flush=True)
        print(f"Post-Fix Evaluation: {post_pass}/{tot} PASS ({100.0*post_pass/max(1,tot):.2f}%)", flush=True)
        print(f"Net Gain           : {post_pass - pre_pass:+d} questions", flush=True)
        print("="*75, flush=True)
    except Exception as e:
        print(f"[POST-FIX WATCHER] Error generating summary: {e}", flush=True)
