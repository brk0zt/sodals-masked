"""
SODALS v3.0 — Token-Level MCTS Benchmark Evaluator
===================================================
Runs HLE questions and OOD perturbations using Token-Level
Monte Carlo Tree Search (MCTS) reasoning with hypothesis branching,
Clifford process rewards, CDCL pruning, and minimum reasoning depth.
"""

import os
import sys
import json
import time
import torch
import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
SFT_DIR = os.path.join(ROOT, "sft")
BASE_MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(SFT_DIR, "qwen_lora_adapter_k7_unified_20260818")
OJA_PATH = os.path.join(ADAPTER_PATH, "sodals_oja_b_matrices.pt")
QUESTION_IDS_30_PATH = os.path.join(ROOT, "hle_30_test_question_ids.json")
MCTS_OUT_PATH = os.path.join(ROOT, "hle_results_k8_token_mcts_20260820.json")

sys.path.insert(0, SFT_DIR)
from token_mcts_engine import TokenMctsEngine

import transformers.trainer
import transformers.utils.import_utils
import transformers.utils

def _safe_bypass():
    pass

transformers.utils.import_utils.check_torch_load_is_safe = _safe_bypass
transformers.utils.check_torch_load_is_safe = _safe_bypass
transformers.trainer.check_torch_load_is_safe = _safe_bypass

_orig_torch_load = torch.load
def _custom_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _custom_torch_load


def evaluate_with_mcts(num_questions: int = 10, min_reasoning_tokens: int = 1500):
    logging.info("=" * 80)
    logging.info("  SODALS v3.0 — TOKEN-LEVEL MCTS DEEP REASONING BENCHMARK")
    logging.info(f"  Target Questions: {num_questions} | Min Reasoning Tokens: {min_reasoning_tokens}")
    logging.info("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    logging.info(f"Loading Base Model on GPU (4-bit NF4)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    logging.info(f"Attaching K7 LoRA Adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    oja_path = OJA_PATH if os.path.exists(OJA_PATH) else None
    engine = TokenMctsEngine(
        model=model,
        tokenizer=tokenizer,
        num_branches=3,
        mcts_steps=18,
        chunk_token_limit=64,
        min_reasoning_tokens=min_reasoning_tokens,
        max_total_tokens=4096,
        c_puct=1.414,
        temperature=0.7,
        oja_path=oja_path,
        verbose=True
    )

    with open(QUESTION_IDS_30_PATH, "r", encoding="utf-8") as f:
        target_ids = json.load(f)[:num_questions]

    dataset = load_dataset("cais/hle", split="test").to_dict()
    all_q = [dict(zip(dataset.keys(), v)) for v in zip(*dataset.values())]
    id_to_q = {q["id"]: q for q in all_q}

    results = {}
    for idx, qid in enumerate(target_ids, 1):
        q_item = id_to_q.get(qid)
        if not q_item:
            continue
        prompt = q_item["question"]
        gt_answer = q_item.get("answer", "")
        category = q_item.get("category", "STEM")

        logging.info(f"\n{'='*75}\n[{idx:02d}/{len(target_ids)}] Evaluating ID: {qid} [{category}]")
        logging.info(f"Ground Truth: {gt_answer[:80]}...\n{'='*75}")

        mcts_res = engine.generate(prompt)

        results[qid] = {
            "response": mcts_res["response"],
            "ground_truth": gt_answer,
            "category": category,
            "tokens_generated": mcts_res["tokens_generated"],
            "mcts_depth": mcts_res["mcts_depth"],
            "mcts_value_final": mcts_res["mcts_value_final"],
            "srf_final": mcts_res["srf_final"],
            "elapsed_seconds": mcts_res["elapsed_seconds"],
            "tokens_per_second": mcts_res["tokens_per_second"],
            "status": "success"
        }

        # Periodic save
        with open(MCTS_OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        torch.cuda.empty_cache()

    logging.info("\n" + "=" * 80)
    logging.info("  MCTS EVALUATION COMPLETE")
    logging.info(f"  Saved to: {MCTS_OUT_PATH}")
    logging.info("=" * 80)


if __name__ == "__main__":
    evaluate_with_mcts(num_questions=5, min_reasoning_tokens=1200)
