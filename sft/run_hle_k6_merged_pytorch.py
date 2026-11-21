import argparse
import json
import logging
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MERGED_MODEL_PATH = os.path.join(ROOT, "models", "Qwen3.5-9B-SODALS-Merged-SafeTensors")
OUT_JSON = os.path.join(ROOT, "hle_k6_merged_weights_Qwen3.5-9B.json")
STATUS_JSON = os.path.join(ROOT, "hle_k6_merged_weights_status.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--test_ids", default=os.path.join(ROOT, "hle-main", "hle_eval", "test_question_ids.json"))
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun all test questions")
    args = parser.parse_args()

    logging.info(f"=== K6 STANDALONE MERGED MODEL HLE BENCHMARK ===")
    logging.info(f"Loading Model from: {MERGED_MODEL_PATH}")

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

    logging.info(f"Loading HLE test dataset...")
    dataset = load_dataset("cais/hle", split="test").to_dict()
    all_q = [dict(zip(dataset.keys(), v)) for v in zip(*dataset.values())]
    id_to_question = {q["id"]: q for q in all_q}

    with open(args.test_ids, "r", encoding="utf-8") as f:
        target_ids = json.load(f)

    results = {}
    if os.path.exists(OUT_JSON) and not args.force_rerun:
        try:
            with open(OUT_JSON, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

    total = len(target_ids)
    logging.info(f"Target questions for K6_MERGED_WEIGHTS: {total}")

    for idx, qid in enumerate(target_ids, 1):
        if qid in results and not args.force_rerun:
            logging.info(f"[{idx}/{total}] Already completed QID: {qid}")
            continue

        q_item = id_to_question.get(qid)
        if not q_item:
            logging.warning(f"QID {qid} not found in HLE dataset.")
            continue

        prompt_text = q_item["question"]
        logging.info(f"[{idx}/{total}] Running K6_MERGED_WEIGHTS on QID: {qid}...")

        messages = [
            {"role": "user", "content": prompt_text}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - start_time

        gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
        completion_tokens = len(gen_tokens)
        resp_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        results[qid] = {
            "response": resp_text,
            "error": "",
            "returncode": 0,
            "status": "success",
            "usage": {
                "completion_tokens": completion_tokens,
                "elapsed_seconds": round(elapsed, 2),
                "tokens_per_second": round(completion_tokens / max(elapsed, 0.001), 2)
            }
        }

        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        with open(STATUS_JSON, "w", encoding="utf-8") as f:
            json.dump({
                "config": "K6_MERGED_WEIGHTS_ONLY",
                "current_index": idx,
                "total_questions": total,
                "current_qid": qid,
                "completed": len(results),
                "elapsed_seconds": round(elapsed, 2),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, f, ensure_ascii=False, indent=2)

        logging.info(f"[{idx}/{total}] Done QID: {qid} in {elapsed:.1f}s ({completion_tokens} tokens, {completion_tokens/max(elapsed, 0.001):.1f} t/s)")

    logging.info(f"=== K6_MERGED_WEIGHTS Benchmark Completed! Saved to {OUT_JSON} ===")

if __name__ == "__main__":
    main()
