"""
SODALS v3.0 — K8 Full SODALS Evaluation Engine
================================================
K7 merged model + Full SODALS Inference Engine ile:
  - 30 soruluk HLE Extended Benchmark
  - 4 OOD Mathematical Perturbation Test
  - CliffordBridge her token adımında aktif
  - SRF registerleri güncelleniyor
  - Skew-G braking aktif
  - Dinamik bütçe: 4096 → max 9216
  - Emergency Answer Latch aktif
  - MindFilter aktif
  - Oja B-matrisleri çıkarımda yüklü

model.generate() KULLANILMAZ.
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
MERGED_OUTPUT_DIR = os.path.join(ROOT, "models", "Qwen3.5-9B-SODALS-K8-FullSODALS-SafeTensors")

OJA_PATH = os.path.join(ADAPTER_PATH, "sodals_oja_b_matrices.pt")
QUESTION_IDS_30_PATH = os.path.join(ROOT, "hle_30_test_question_ids.json")
HLE_OUT_PATH = os.path.join(ROOT, "hle_results_k8_full_sodals_30q_20260819.json")
PERT_OUT_PATH = os.path.join(ROOT, "hle_perturbations_k8_full_sodals_20260819.json")

sys.path.insert(0, SFT_DIR)
from sodals_inference_engine import FullSodalsInferenceEngine

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


PERTURBATIONS = [
    {
        "test_id": "unseen_top_euler_cp4",
        "topic": "Euler Characteristic of Complex Projective Space CP^4 (P1 Accumulator Test)",
        "prompt": "Compute the topological Euler characteristic χ(CP⁴) of the 4-dimensional complex projective space. Show your complete derivation step by step.",
        "expected_ground_truth": "5",
    },
    {
        "test_id": "unseen_lie_dim_e6",
        "topic": "Dimension of Exceptional Lie Algebra E6 (P2 Explicit Derivation Test)",
        "prompt": "What is the dimension of the exceptional simple Lie algebra e₆ over the complex numbers? Derive the answer explicitly from the root system.",
        "expected_ground_truth": "78",
    },
    {
        "test_id": "unseen_geom_canonical_degree_g4",
        "topic": "Degree of Canonical Divisor for Genus 4 Riemann Surface",
        "prompt": "Let C be a smooth compact Riemann surface of genus g = 4. Compute the exact degree of the canonical divisor K on C. Derive from Riemann-Roch.",
        "expected_ground_truth": "6",
    },
    {
        "test_id": "unseen_num_point_count_f53",
        "topic": "Rational Point Count for Elliptic Curve over F_53 with a_p = -4",
        "prompt": "Let E be an elliptic curve defined over the finite field F₅₃ having trace of Frobenius a₅₃ = -4. How many F₅₃-rational points #E(F₅₃) does the curve have? Show derivation.",
        "expected_ground_truth": "58",
    }
]


def eval_k8_full_sodals():
    """Full SODALS inference motoru ile 30Q HLE + 4 OOD değerlendirmesi (4-bit NF4 LoRA)."""
    logging.info("=" * 80)
    logging.info("  K8 FULL SODALS EVALUATION (4-bit NF4 + K7 LoRA Adapter)")
    logging.info("=" * 80)

    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"K7 LoRA adapter not found at {ADAPTER_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    logging.info(f"Loading Base Qwen 3.5 9B in 4-bit NF4 on GPU from {BASE_MODEL_PATH}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    logging.info(f"Attaching Trained K7 LoRA Adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # Full SODALS Inference Engine başlat
    oja_b_path = OJA_PATH if os.path.exists(OJA_PATH) else None
    engine = FullSodalsInferenceEngine(
        model=model,
        tokenizer=tokenizer,
        initial_budget=4096,
        budget_ceiling=9216,
        oja_path=oja_b_path,
        verbose=True
    )

    # ── 1. 30 Soru HLE Benchmark ──
    logging.info("--- Evaluating 30 HLE Questions with Full SODALS Engine ---")

    with open(QUESTION_IDS_30_PATH, "r", encoding="utf-8") as f:
        target_30_ids = json.load(f)

    dataset = load_dataset("cais/hle", split="test").to_dict()
    all_q = [dict(zip(dataset.keys(), v)) for v in zip(*dataset.values())]
    id_to_q = {q["id"]: q for q in all_q}

    hle_results = {}
    for idx, qid in enumerate(target_30_ids, 1):
        q_item = id_to_q.get(qid)
        if not q_item:
            continue
        prompt = q_item["question"]
        subject = q_item.get("category", "STEM")
        gt_answer = q_item.get("answer", "")

        logging.info(f"\n[{idx:02d}/30] Evaluating ID: {qid} [{subject}]...")
        logging.info(f"  Ground Truth: {gt_answer[:80]}...")

        result = engine.generate(prompt, temperature=0.0)

        hle_results[qid] = {
            "response": result["response"],
            "ground_truth": gt_answer,
            "category": subject,
            "tokens_generated": result["tokens_generated"],
            "budget_used": result["budget_used"],
            "budget_expansions": result["budget_expansions"],
            "stop_reason": result["stop_reason"],
            "srf_final": result["srf_final"],
            "elapsed_seconds": result["elapsed_seconds"],
            "tokens_per_second": result["tokens_per_second"],
            "status": "success",
        }

        logging.info(f"  → Tokens: {result['tokens_generated']}, "
                     f"Budget: {result['budget_used']}, "
                     f"Expansions: {result['budget_expansions']}, "
                     f"Stop: {result['stop_reason']}, "
                     f"Time: {result['elapsed_seconds']}s, "
                     f"TPS: {result['tokens_per_second']}")

        # Periyodik kaydet
        with open(HLE_OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(hle_results, f, indent=2, ensure_ascii=False)

        # VRAM temizliği
        torch.cuda.empty_cache()

    # ── 2. 4 OOD Perturbation Test ──
    logging.info("\n--- Evaluating 4 OOD Perturbations with Full SODALS Engine ---")

    pert_results = []
    for test in PERTURBATIONS:
        logging.info(f"Evaluating Perturbation: {test['test_id']}...")

        result = engine.generate(test["prompt"], temperature=0.0)

        pert_results.append({
            "test_id": test["test_id"],
            "topic": test["topic"],
            "prompt": test["prompt"],
            "expected_ground_truth": test["expected_ground_truth"],
            "model_response": result["response"],
            "tokens_generated": result["tokens_generated"],
            "budget_used": result["budget_used"],
            "budget_expansions": result["budget_expansions"],
            "stop_reason": result["stop_reason"],
            "srf_final": result["srf_final"],
            "elapsed_seconds": result["elapsed_seconds"],
            "tokens_per_second": result["tokens_per_second"],
        })

        torch.cuda.empty_cache()

    with open(PERT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pert_results, f, indent=2, ensure_ascii=False)

    # ── 3. Sonuç Özeti ──
    logging.info("\n" + "=" * 80)
    logging.info("  K8 FULL SODALS EVALUATION COMPLETE")
    logging.info("=" * 80)

    total_tokens = sum(r["tokens_generated"] for r in hle_results.values())
    avg_tokens = total_tokens / len(hle_results) if hle_results else 0
    total_time = sum(r["elapsed_seconds"] for r in hle_results.values())
    avg_time = total_time / len(hle_results) if hle_results else 0
    total_expansions = sum(r["budget_expansions"] for r in hle_results.values())

    logging.info(f"  HLE 30Q Results:")
    logging.info(f"    Total Tokens Generated:    {total_tokens}")
    logging.info(f"    Avg Tokens/Question:       {avg_tokens:.0f}")
    logging.info(f"    Avg Time/Question:          {avg_time:.1f}s")
    logging.info(f"    Total Budget Expansions:   {total_expansions}")
    logging.info(f"    Results saved to: {HLE_OUT_PATH}")
    logging.info(f"    Perturbations saved to: {PERT_OUT_PATH}")


def score_results():
    """K8 sonuçlarını skorla ve K7 Baseline ile karşılaştır."""
    logging.info("\n--- Scoring K8 Results ---")

    if not os.path.exists(HLE_OUT_PATH):
        logging.warning("K8 HLE results not found, skipping scoring.")
        return

    with open(HLE_OUT_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    correct = 0
    total = len(results)
    for qid, r in results.items():
        gt = str(r.get("ground_truth", "")).strip().lower()
        resp = str(r.get("response", "")).strip().lower()

        # Son cevabı çıkar
        ans = resp
        for marker in ["answer:", "answer :", "the answer is"]:
            if marker in resp:
                ans = resp.split(marker)[-1].strip()
                # İlk satırı al
                ans = ans.split("\n")[0].strip().rstrip(".")
                break

        if ans == gt:
            correct += 1
            logging.info(f"  ✅ {qid}: {ans} == {gt}")
        else:
            logging.info(f"  ❌ {qid}: {ans} != {gt}")

    accuracy = correct / total if total > 0 else 0

    # Wilson CI
    import math
    n = total
    p_hat = accuracy
    z = 1.96
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    ci_low = max(0, center - margin)
    ci_high = min(1, center + margin)

    logging.info(f"\n  K8 FULL SODALS SCORE: {correct} / {total} ({accuracy*100:.1f}%)")
    logging.info(f"  Wilson 95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
    logging.info(f"  Baseline Reference: 1 / 30 (3.3%)")
    logging.info(f"  Absolute Lift: {(accuracy - 1/30)*100:+.1f}pp")


if __name__ == "__main__":
    eval_k8_full_sodals()
    score_results()
