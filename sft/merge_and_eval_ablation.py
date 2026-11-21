"""
SODALS v2.1 — Universal Ablation Merge & Evaluator
===================================================
Merges any given LoRA adapter into SafeTensors, then runs:
  1. 8 Official CAIS HLE Benchmark questions (Greedy, Temp 0.0)
  2. 4 Out-of-Distribution (OOD) Mathematical Perturbation tests
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
BASE_MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
TEST_IDS_PATH = os.path.join(ROOT, "hle-main", "hle_eval", "test_question_ids.json")

PERTURBATIONS = [
    {
        "test_id": "unseen_top_euler_cp4",
        "topic": "Euler Characteristic of Complex Projective Space CP^4",
        "prompt": "Compute the topological Euler characteristic \\chi(\\mathbb{C}P^4) of the 4-dimensional complex projective space.",
        "expected_ground_truth": "5",
    },
    {
        "test_id": "unseen_lie_dim_e6",
        "topic": "Dimension of Exceptional Lie Algebra E6",
        "prompt": "What is the dimension of the exceptional simple Lie algebra \\mathfrak{e}_6 over the complex numbers?",
        "expected_ground_truth": "78",
    },
    {
        "test_id": "unseen_geom_canonical_degree_g4",
        "topic": "Degree of Canonical Divisor for Genus 4 Riemann Surface",
        "prompt": "Let C be a smooth compact Riemann surface of genus g = 4. Compute the exact degree of the canonical divisor K on C.",
        "expected_ground_truth": "6",
    },
    {
        "test_id": "unseen_num_point_count_f53",
        "topic": "Rational Point Count for Elliptic Curve over F_53 with a_p = -4",
        "prompt": "Let E be an elliptic curve defined over the finite field \\mathbb{F}_{53} having trace of Frobenius a_{53} = -4. How many \\mathbb{F}_{53}-rational points #E(\\mathbb{F}_{53}) does the curve have?",
        "expected_ground_truth": "58",
    }
]


def merge_lora(adapter_path: str, merged_output_dir: str):
    logging.info(f"=== MERGING LORA: {adapter_path} -> {merged_output_dir} ===")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    logging.info("Loading Base Model on CPU for Clean SafeTensors Export...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    logging.info("Attaching LoRA adapter and merging weights...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()

    logging.info(f"Saving merged SafeTensors to {merged_output_dir}...")
    os.makedirs(merged_output_dir, exist_ok=True)
    merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_output_dir)
    logging.info(f"[SUCCESS] Standalone Merged Model saved to {merged_output_dir}")

    # Free memory
    del peft_model
    del merged_model
    del base_model
    torch.cuda.empty_cache()


def run_evaluations(merged_model_path: str, hle_out_json: str, pert_out_json: str, tag: str):
    logging.info(f"=== EVALUATING ABLATION: {tag} ===")
    logging.info(f"Loading Model in 4-bit NF4 onto GPU...")

    tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    model.eval()

    # ─── PART 1: 8 HLE BENCHMARK QUESTIONS ────────────────────────────────────
    logging.info(f"--- Running 8 HLE Benchmark Questions for {tag} ---")
    dataset = load_dataset("cais/hle", split="test").to_dict()
    all_q = [dict(zip(dataset.keys(), v)) for v in zip(*dataset.values())]
    id_to_question = {q["id"]: q for q in all_q}

    with open(TEST_IDS_PATH, "r", encoding="utf-8") as f:
        target_ids = json.load(f)

    hle_results = {}
    for idx, qid in enumerate(target_ids, 1):
        q_item = id_to_question.get(qid)
        if not q_item:
            continue
        prompt = q_item["question"]
        logging.info(f"[{idx}/8] Evaluating HLE Question {qid}...")

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

        hle_results[qid] = {
            "response": resp_text,
            "error": "",
            "returncode": 0,
            "status": "success",
            "usage": {
                "completion_tokens": len(gen_tokens),
                "elapsed_seconds": round(elapsed, 2),
                "tokens_per_second": round(len(gen_tokens) / elapsed, 2) if elapsed > 0 else 0
            }
        }
        logging.info(f"[{idx}/8] Done in {elapsed:.1f}s ({len(gen_tokens)} tokens)")

    with open(hle_out_json, "w", encoding="utf-8") as f:
        json.dump(hle_results, f, indent=2, ensure_ascii=False)
    logging.info(f"[HLE SAVED] -> {hle_out_json}")

    # ─── PART 2: 4 OOD PERTURBATIONS ─────────────────────────────────────────
    logging.info(f"--- Running 4 OOD Perturbation Tests for {tag} ---")
    pert_results = []
    for test in PERTURBATIONS:
        prompt = test["prompt"]
        logging.info(f"Evaluating Perturbation: {test['test_id']}...")

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        elapsed = time.time() - start_time
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        resp_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        pert_results.append({
            "test_id": test["test_id"],
            "topic": test["topic"],
            "prompt": test["prompt"],
            "expected_ground_truth": test["expected_ground_truth"],
            "memorized_original_answer": "N/A (Unseen OOD Test)",
            "model_response": resp_text,
            "tokens": len(gen_tokens),
            "elapsed_seconds": round(elapsed, 2)
        })
        logging.info(f"Done in {elapsed:.1f}s")

    with open(pert_out_json, "w", encoding="utf-8") as f:
        json.dump(pert_results, f, indent=2, ensure_ascii=False)
    logging.info(f"[PERTURBATIONS SAVED] -> {pert_out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--merged_dir", required=True)
    parser.add_argument("--hle_out", required=True)
    parser.add_argument("--pert_out", required=True)
    parser.add_argument("--tag", default="Ablation")
    args = parser.parse_args()

    merge_lora(args.adapter, args.merged_dir)
    run_evaluations(args.merged_dir, args.hle_out, args.pert_out, args.tag)
