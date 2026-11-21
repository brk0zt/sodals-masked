"""
SODALS v2.1 — Comprehensive GPR Phase Router Accuracy Benchmark (170Q)
======================================================================
Tests the Neural Gaussian Phase Router (GPR) across the complete 170-question
benchmark (150 MATH-500 + 20 Domain 2 Abstract Algebra) with ZERO regex or keywords.
Verifies routing accuracy, separation margin, and probe latency.
"""

import os
import sys
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from gpr_phase_router import NeuralPhaseRouter

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
DOMAIN2_REF_PATH = os.path.join(ROOT, "domain2_20q_interleaved_r8_results.json")
OUTPUT_REPORT_PATH = os.path.join(ROOT, "gpr_router_170q_accuracy_report.json")

def load_170_questions():
    items = []
    # 150 MATH-500
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for i in range(150):
        items.append({
            "index": i + 1,
            "id": ds[i].get("unique_id", f"math500_{i+1}"),
            "prompt": ds[i]["problem"],
            "expected_adapter": "expert_general_math",
            "domain": "MATH-500 (General STEM)"
        })

    # 20 Domain 2
    with open(DOMAIN2_REF_PATH, "r", encoding="utf-8") as f:
        d2_raw = json.load(f)["results"]
    for i, it in enumerate(d2_raw):
        items.append({
            "index": 150 + i + 1,
            "id": it["id"],
            "prompt": it["prompt"],
            "expected_adapter": "expert_abstract_algebra",
            "domain": "Domain 2 (Abstract Algebra)"
        })

    return items

def run_test():
    print("=" * 80, flush=True)
    print("SODALS GPR NEURAL PHASE ROUTER — 170 QUESTION ACCURACY AUDIT", flush=True)
    print("=" * 80, flush=True)

    items = load_170_questions()
    print(f"Loaded {len(items)} benchmark questions (150 MATH-500 + 20 Domain 2).", flush=True)

    print("Loading Tokenizer & Model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    model.eval()

    print("Initializing NeuralPhaseRouter...", flush=True)
    router = NeuralPhaseRouter()
    print(f"Router loaded: Calibrated Layer {router.layer_idx}, Threshold {router.threshold:.6f}, Fisher Separation {router.fisher_separation:.2f}", flush=True)

    print("\nEvaluating all 170 questions...", flush=True)
    correct_count = 0
    math500_correct = 0
    domain2_correct = 0
    total_latency = 0.0
    details = []

    for idx, it in enumerate(items):
        chat_prompt = (
            f"<|im_start|>user\n{it['prompt']}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

        t0 = time.time()
        routed_adapter, z_i, meta = router.route(model, inputs)
        elapsed_ms = (time.time() - t0) * 1000.0
        total_latency += elapsed_ms

        expected = it["expected_adapter"]
        is_correct = (routed_adapter == expected)

        if is_correct:
            correct_count += 1
            if it["domain"] == "Domain 2 (Abstract Algebra)":
                domain2_correct += 1
            else:
                math500_correct += 1

        verdict = "CORRECT" if is_correct else "MISROUTED"
        if (idx + 1) % 25 == 0 or not is_correct or (idx + 1) > 150:
            print(f"[{idx+1:03d}/170] [{it['id']:25s}] Expected: {expected:24s} | Routed: {routed_adapter:24s} | z_i: {z_i:+.4f} | {verdict} ({elapsed_ms:.1f}ms)", flush=True)

        details.append({
            "index": it["index"],
            "id": it["id"],
            "domain": it["domain"],
            "expected_adapter": expected,
            "routed_adapter": routed_adapter,
            "z_i": z_i,
            "confidence": meta["confidence"],
            "is_correct": is_correct,
            "latency_ms": round(elapsed_ms, 2)
        })

    avg_latency = total_latency / len(items)
    accuracy = 100.0 * correct_count / len(items)
    math500_acc = 100.0 * math500_correct / 150
    domain2_acc = 100.0 * domain2_correct / 20

    print("\n" + "=" * 80, flush=True)
    print(f"GPR NEURAL PHASE ROUTER ACCURACY REPORT:")
    print(f"  Overall Routing Accuracy  : {correct_count} / {len(items)} ({accuracy:.2f}%)")
    print(f"  MATH-500 (General STEM)   : {math500_correct} / 150 ({math500_acc:.2f}%)")
    print(f"  Domain 2 (Abstract Algebra): {domain2_correct} / 20 ({domain2_acc:.2f}%)")
    print(f"  Average Probe Latency      : {avg_latency:.2f} ms")
    print("=" * 80, flush=True)

    report = {
        "benchmark": "SODALS GPR Neural Phase Router 170Q Audit",
        "total_questions": len(items),
        "correct_routing": correct_count,
        "accuracy_pct": round(accuracy, 2),
        "math500_accuracy_pct": round(math500_acc, 2),
        "domain2_accuracy_pct": round(domain2_acc, 2),
        "average_latency_ms": round(avg_latency, 2),
        "calibrated_layer": router.layer_idx,
        "calibrated_threshold": router.threshold,
        "details": details
    }

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Detailed audit saved to: {OUTPUT_REPORT_PATH}", flush=True)

if __name__ == "__main__":
    run_test()
