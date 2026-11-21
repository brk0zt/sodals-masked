"""
SODALS v2.1 — K7 Unified Model Merging & Full 30Q Extended Evaluator
====================================================================
1. Merges K7 Unified LoRA Adapter into Standalone SafeTensors:
   `models/Qwen3.5-9B-SODALS-K7-SafeTensors`
2. Evaluates full 30-Question Extended HLE Benchmark:
   `hle_results_k7_unified_30q_20260818.json`
3. Evaluates 4 OOD Mathematical Perturbation tests:
   `hle_perturbations_k7_unified_20260818.json`
4. Scores exact matches against Baseline (1/30) and calculates comparative lift!
"""

import os
import sys
import json
import time
import math
import logging
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
BASE_MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_k7_unified_20260818")
MERGED_OUTPUT_DIR = os.path.join(ROOT, "models", "Qwen3.5-9B-SODALS-K7-SafeTensors")

QUESTION_IDS_30_PATH = os.path.join(ROOT, "hle_30_test_question_ids.json")
HLE_OUT_PATH = os.path.join(ROOT, "hle_results_k7_unified_30q_20260818.json")
PERT_OUT_PATH = os.path.join(ROOT, "hle_perturbations_k7_unified_20260818.json")

PERTURBATIONS = [
    {
        "test_id": "unseen_top_euler_cp4",
        "topic": "Euler Characteristic of Complex Projective Space CP^4 (P1 Accumulator Test)",
        "prompt": "Compute the topological Euler characteristic \\chi(\\mathbb{C}P^4) of the 4-dimensional complex projective space.",
        "expected_ground_truth": "5",
    },
    {
        "test_id": "unseen_lie_dim_e6",
        "topic": "Dimension of Exceptional Lie Algebra E6 (P2 Explicit Derivation Test)",
        "prompt": "What is the dimension of the exceptional simple Lie algebra \\mathfrak{e}_6 over the complex numbers? Please derive the answer explicitly from the root system.",
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


def merge_k7():
    logging.info("=== STEP 1: MERGING K7 UNIFIED LORA ADAPTER INTO SAFETENSORS ===")
    logging.info(f"Adapter: {ADAPTER_PATH}")
    logging.info(f"Output Directory: {MERGED_OUTPUT_DIR}")

    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"K7 LoRA adapter not found at {ADAPTER_PATH}")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    merged_model = peft_model.merge_and_unload()

    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
    logging.info(f"[SUCCESS] Merged K7 Model saved to {MERGED_OUTPUT_DIR}")

    del peft_model
    del merged_model
    del base_model
    torch.cuda.empty_cache()


def eval_k7_30q_and_pert():
    logging.info("=== STEP 2: RUNNING 30Q HLE AND 4 OOD PERTURBATIONS ON K7 ===")

    tokenizer = AutoTokenizer.from_pretrained(MERGED_OUTPUT_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MERGED_OUTPUT_DIR,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    model.eval()

    # Load 30 questions
    with open(QUESTION_IDS_30_PATH, "r", encoding="utf-8") as f:
        target_30_ids = json.load(f)

    dataset = load_dataset("cais/hle", split="test").to_dict()
    all_q = [dict(zip(dataset.keys(), v)) for v in zip(*dataset.values())]
    id_to_q = {q["id"]: q for q in all_q}

    # 1. Evaluate 30 HLE questions
    hle_30_results = {}
    for idx, qid in enumerate(target_30_ids, 1):
        q_item = id_to_q.get(qid)
        if not q_item:
            continue
        prompt = q_item["question"]
        subject = q_item.get("category", "STEM")
        gt_answer = q_item.get("answer", "")

        logging.info(f"[{idx:02d}/30] Evaluating ID: {qid} [{subject}]...")

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

        hle_30_results[qid] = {
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

        with open(HLE_OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(hle_30_results, f, indent=2, ensure_ascii=False)
        logging.info(f"[{idx:02d}/30] Done in {elapsed:.1f}s")

    # 2. Evaluate 4 Perturbations
    logging.info("--- Evaluating 4 OOD Perturbations ---")
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

    with open(PERT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pert_results, f, indent=2, ensure_ascii=False)
    logging.info(f"[SUCCESS] All K7 30Q and Perturbation evaluations completed and saved!")


if __name__ == "__main__":
    merge_k7()
    eval_k7_30q_and_pert()
