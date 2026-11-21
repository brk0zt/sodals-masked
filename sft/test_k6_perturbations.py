import json
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Ensure utf-8 stdout encoding on Windows
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MERGED_MODEL_PATH = os.path.join(ROOT, "models", "Qwen3.5-9B-SODALS-Merged-SafeTensors")
OUT_JSON = os.path.join(ROOT, "hle_k6_perturbation_test_results.json")

PERTURBED_QUESTIONS = [
    {
        "id": "unseen_top_euler_cp4",
        "original_topic": "Euler Characteristic of Complex Projective Space CP^4",
        "prompt": (
            "Compute the topological Euler characteristic \\chi(\\mathbb{C}P^4) of the 4-dimensional complex projective space."
        ),
        "expected_ground_truth": "5",
        "memorized_original_answer": "N/A (Unseen OOD Test)"
    },
    {
        "id": "unseen_lie_dim_e6",
        "original_topic": "Dimension of Exceptional Lie Algebra E6",
        "prompt": (
            "What is the dimension of the exceptional simple Lie algebra \\mathfrak{e}_6 over the complex numbers?"
        ),
        "expected_ground_truth": "78",
        "memorized_original_answer": "N/A (Unseen OOD Test)"
    },
    {
        "id": "unseen_geom_canonical_degree_g4",
        "original_topic": "Degree of Canonical Divisor for Genus 4 Riemann Surface",
        "prompt": (
            "Let C be a smooth compact Riemann surface of genus g = 4. Compute the exact degree of the canonical divisor K on C."
        ),
        "expected_ground_truth": "6",
        "memorized_original_answer": "N/A (Unseen OOD Test)"
    },
    {
        "id": "unseen_num_point_count_f53",
        "original_topic": "Rational Point Count for Elliptic Curve over F_53 with a_p = -4",
        "prompt": (
            "Let E be an elliptic curve defined over the finite field \\mathbb{F}_{53} having trace of Frobenius a_{53} = -4. "
            "How many \\mathbb{F}_{53}-rational points #E(\\mathbb{F}_{53}) does the curve have?"
        ),
        "expected_ground_truth": "58",
        "memorized_original_answer": "N/A (Unseen OOD Test)"
    }
]

def run_perturbation_test():
    print("=== K6 PERTURBATION & OVERFITTING ROBUSTNESS TEST ===")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print("Loading K6 Standalone Merged Model on RTX 4060 GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    model.eval()

    results = []
    for idx, item in enumerate(PERTURBED_QUESTIONS, 1):
        print(f"\n[{idx}/4] Testing: {item['original_topic']}...")
        messages = [{"role": "user", "content": item["prompt"]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0

        gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
        resp_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        print(f"Generated ({len(gen_tokens)} tok, {elapsed:.1f}s):")
        print(resp_text)

        results.append({
            "test_id": item["id"],
            "topic": item["original_topic"],
            "prompt": item["prompt"],
            "expected_ground_truth": item["expected_ground_truth"],
            "memorized_original_answer": item["memorized_original_answer"],
            "k6_response": resp_text,
            "tokens": len(gen_tokens),
            "elapsed_seconds": round(elapsed, 2)
        })

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Perturbation test completed! Saved to {OUT_JSON}")

if __name__ == "__main__":
    run_perturbation_test()
