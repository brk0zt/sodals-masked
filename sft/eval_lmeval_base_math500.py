"""
Official EleutherAI lm-eval Benchmark: Qwen 3.5 9B Base Model on MATH-500
==========================================================================
Instantiates the model with BitsAndBytesConfig in 4-bit NF4 and passes it directly
to lm_eval's HFLM wrapper to evaluate on minerva_math500 / hendrycks_math500.
"""

import sys
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import lm_eval
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
OUTPUT_JSON = os.path.join(ROOT, "lmeval_base_math500_results.json")

print("=" * 80)
print("RUNNING OFFICIAL ELEUTHERAI LM-EVAL HARNESS ON BASE MODEL (QWEN 3.5 9B)")
print("=" * 80)

print("Loading Tokenizer from:", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading Base Model in 4-bit NF4 onto GPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True
)

print("Wrapping model with lm_eval HFLM...")
lm = HFLM(
    pretrained=base_model,
    tokenizer=tokenizer,
    batch_size=1,
    max_length=2048
)

print("Running official lm-eval simple_evaluate on hendrycks_math500 (50 questions limit)...")
results = evaluator.simple_evaluate(
    model=lm,
    tasks=["hendrycks_math500"],
    limit=50,
    log_samples=True
)

print("\n" + "=" * 80)
print("LM-EVAL OFFICIAL EVALUATION COMPLETED!")
print("=" * 80)

if "results" in results:
    for task_name, metrics in results["results"].items():
        print(f"\nTask: {task_name}")
        for metric_name, val in metrics.items():
            if isinstance(val, (int, float)):
                print(f"  {metric_name:30s}: {val:.4f} ({val*100:.2f}%)")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

print(f"\nDetailed evaluation log saved to {OUTPUT_JSON}")
