"""
SODALS v2.1 — 30-Question Extended HLE Benchmark for Plain LoRA Baseline
=========================================================================
Runs greedy closed-book evaluation on 30 official CAIS HLE Mathematics & Physics
questions on the standalone Plain LoRA Baseline SafeTensors model.

Reuses existing 8 questions from `hle_results_baseline_control_20260818.json`
and evaluates the 22 new unseen questions to achieve a tight Wilson 95% CI (< 15%).
"""

import os
import sys
import json
import time
import math
import logging
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MERGED_MODEL_PATH = os.path.join(ROOT, "models", "Qwen3.5-9B-SODALS-Baseline-SafeTensors")
QUESTION_IDS_30_PATH = os.path.join(ROOT, "hle_30_test_question_ids.json")
EXISTING_8_RESULTS_PATH = os.path.join(ROOT, "hle_results_baseline_control_20260818.json")
OUTPUT_30_RESULTS_PATH = os.path.join(ROOT, "hle_results_baseline_control_30q_20260818.json")


def compute_wilson_ci(correct: int, total: int, z: float = 1.96):
    """Computes exact Wilson score 95% confidence interval."""
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denominator = 1.0 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denominator
    margin = (z * math.sqrt((p * (1 - p) / total) + (z**2) / (4 * total**2))) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def main():
    logging.info("=== SODALS BASELINE 30-QUESTION EXTENDED HLE EVALUATION ===")
    logging.info(f"Target Model: {MERGED_MODEL_PATH}")
    logging.info(f"Question List: {QUESTION_IDS_30_PATH}")
    logging.info(f"Output File: {OUTPUT_30_RESULTS_PATH}")

    # Load 30 target question IDs
    with open(QUESTION_IDS_30_PATH, "r", encoding="utf-8") as f:
        target_30_ids = json.load(f)

    # Load existing 8 results if available
    results_30 = {}
    if os.path.exists(EXISTING_8_RESULTS_PATH):
        with open(EXISTING_8_RESULTS_PATH, "r", encoding="utf-8") as f:
            existing_8 = json.load(f)
        for qid in target_30_ids[:8]:
            if qid in existing_8:
                results_30[qid] = existing_8[qid]
        logging.info(f"Loaded {len(results_30)} existing questions from 8Q run. 22 new questions to evaluate.")

    # Load HLE dataset
    logging.info("Loading cais/hle test split...")
    dataset = load_dataset("cais/hle", split="test").to_dict()
    all_q = [dict(zip(dataset.keys(), v)) for v in zip(*dataset.values())]
    id_to_q = {q["id"]: q for q in all_q}

    # Identify pending questions
    pending_ids = [qid for qid in target_30_ids if qid not in results_30]
    logging.info(f"Total target: {len(target_30_ids)} | Already done: {len(results_30)} | Remaining to run: {len(pending_ids)}")

    if not pending_ids:
        logging.info("All 30 questions already evaluated!")
        with open(OUTPUT_30_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results_30, f, indent=2, ensure_ascii=False)
        return

    # Load model onto GPU in 4-bit NF4
    logging.info("Loading Baseline SafeTensors Model in 4-bit NF4 onto GPU...")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    model.eval()

    # Evaluate pending questions
    for idx, qid in enumerate(pending_ids, 1):
        q_item = id_to_q.get(qid)
        if not q_item:
            logging.warning(f"Question ID {qid} not found in HLE dataset!")
            continue

        prompt = q_item["question"]
        subject = q_item.get("category", "STEM")
        gt_answer = q_item.get("answer", "")

        logging.info(f"\n[{idx}/{len(pending_ids)}] (Overall: {len(results_30)+1}/30) Evaluating ID: {qid} [{subject}]")
        logging.info(f"Question preview: {prompt[:120]}...")

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        elapsed = time.time() - start_time
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        resp_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        results_30[qid] = {
            "response": resp_text,
            "ground_truth": gt_answer,
            "category": subject,
            "error": "",
            "returncode": 0,
            "status": "success",
            "usage": {
                "completion_tokens": len(gen_tokens),
                "elapsed_seconds": round(elapsed, 2),
                "tokens_per_second": round(len(gen_tokens) / elapsed, 2) if elapsed > 0 else 0
            }
        }

        # Save progress after every question
        with open(OUTPUT_30_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results_30, f, indent=2, ensure_ascii=False)

        logging.info(f"Completed in {elapsed:.1f}s ({len(gen_tokens)} tokens). Ground Truth: '{gt_answer}'")

    logging.info(f"\n🎉 ALL 30 HLE QUESTIONS COMPLETED FOR BASELINE CONTROL!")
    logging.info(f"Results saved to: {OUTPUT_30_RESULTS_PATH}")


if __name__ == "__main__":
    main()
