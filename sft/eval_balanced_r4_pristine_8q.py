"""
SODALS v3.2 — Evaluation of Balanced r=4 LoRA Adapter on Pristine 8Q Benchmark
==============================================================================
Compares 3 arms using the newly trained clean adapter (sft/qwen_lora_adapter_balanced_r4/):
  Arm 1: Pure Greedy LoRA (Clean r=4, 805 balanced samples)
  Arm 2: Greedy + SymPy Runtime Verifier (Zero-Search Baseline)
  Arm 3: Token-Level MCTS with Verified-Reward
"""

import sys
import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
ADAPTER_PATH = os.path.join(ROOT, "sft", "qwen_lora_adapter_balanced_r4")
ODE_WEIGHTS_PATH = os.path.join(ADAPTER_PATH, "arkhe_ode_proj.pt")

sys.path.insert(0, os.path.join(ROOT, "sft"))
from token_mcts_engine import TokenMctsEngine, GreedySympyVerifierEngine

PRISTINE_8Q_BENCHMARK = [
    {
        "id": "pristine_01_lie_f4_dim",
        "domain": "Lie Theory",
        "prompt": "Compute the exact dimension of the exceptional simple Lie algebra \\mathfrak{f}_4 over the complex numbers.",
        "expected_gt": "52",
        "first_principles_basis": "Rank 4 Cartan subalgebra + 48 roots in root system F_4 -> 4 + 48 = 52."
    },
    {
        "id": "pristine_02_top_grassmannian_gr24",
        "domain": "Algebraic Topology",
        "prompt": "Compute the topological Euler characteristic \\chi(\\text{Gr}(2, 4)) of the complex Grassmannian manifold of 2-planes in \\mathbb{C}^4.",
        "expected_gt": "6",
        "first_principles_basis": "Schubert cell decomposition or binomial coefficient \\binom{4}{2} = 6."
    },
    {
        "id": "pristine_03_geom_canonical_g7",
        "domain": "Algebraic Geometry",
        "prompt": "Let C be a smooth compact Riemann surface of genus g = 7. Compute the exact degree of the canonical divisor K on C.",
        "expected_gt": "12",
        "first_principles_basis": "Riemann-Roch / Serre duality / Gauss-Bonnet: deg(K) = 2g - 2 = 2(7) - 2 = 12."
    },
    {
        "id": "pristine_04_num_elliptic_f97",
        "domain": "Arithmetic Geometry",
        "prompt": "Let E be an elliptic curve defined over the finite field \\mathbb{F}_{97} with trace of Frobenius a_{97} = 6. Compute the exact number of rational points #E(\\mathbb{F}_{97}).",
        "expected_gt": "92",
        "first_principles_basis": "Hasse-Weil theorem: #E(F_q) = q + 1 - a_q -> 97 + 1 - 6 = 92."
    },
    {
        "id": "pristine_05_rep_su5_adjoint_dim",
        "domain": "Representation Theory",
        "prompt": "What is the dimension of the adjoint representation of the special unitary Lie algebra \\mathfrak{su}(5)?",
        "expected_gt": "24",
        "first_principles_basis": "dim(su(N)) = N^2 - 1 -> 5^2 - 1 = 25 - 1 = 24."
    },
    {
        "id": "pristine_06_symp_sp6_dim",
        "domain": "Symplectic Geometry",
        "prompt": "Compute the exact dimension of the real symplectic Lie algebra \\mathfrak{sp}(6, \\mathbb{R}) of 6x6 Hamiltonian matrices.",
        "expected_gt": "21",
        "first_principles_basis": "dim(sp(2n, R)) = n(2n + 1). For 2n = 6 (n = 3): 3(2*3 + 1) = 3(7) = 21."
    },
    {
        "id": "pristine_07_top_cp5_euler",
        "domain": "Complex Geometry",
        "prompt": "Compute the topological Euler characteristic \\chi(\\mathbb{C}P^5) of the 5-dimensional complex projective space.",
        "expected_gt": "6",
        "first_principles_basis": "\\chi(CP^n) = n + 1 -> 5 + 1 = 6."
    },
    {
        "id": "pristine_08_num_ramanujan_tau2",
        "domain": "Modular Forms / Number Theory",
        "prompt": "What is the exact value of the Ramanujan tau function \\tau(2), defined as the Fourier coefficient of the modular discriminant \\Delta(z)?",
        "expected_gt": "-24",
        "first_principles_basis": "q-expansion of \\Delta(q) = q \\prod_{n=1}^\\infty (1 - q^n)^{24} = q - 24q^2 + 252q^3 - ... -> \\tau(2) = -24."
    }
]


def run_pristine_benchmark():
    print("="*80)
    print("SODALS v3.2: EVALUATION OF BALANCED r=4 LoRA ADAPTER ON PRISTINE 8Q")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Base Model in 4-bit NF4...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    print(f"Loading Balanced r=4 LoRA Adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("Initializing GreedySympyVerifierEngine & TokenMctsEngine with Verified-Reward...")
    greedy_sympy_engine = GreedySympyVerifierEngine(model=model, tokenizer=tokenizer, max_tokens=512, verbose=True)
    mcts_engine = TokenMctsEngine(
        model=model,
        tokenizer=tokenizer,
        num_branches=3,
        mcts_steps=8,
        chunk_token_limit=96,
        min_reasoning_tokens=60,
        max_total_tokens=2048,
        c_puct=1.414,
        temperature=0.4,
        ode_weights_path=ODE_WEIGHTS_PATH,
        verbose=True
    )

    results = []

    for i, q in enumerate(PRISTINE_8Q_BENCHMARK):
        print(f"\n{'='*80}\nPRISTINE QUESTION {i+1}/8: [{q['domain']}] {q['id']}\n{'='*80}")
        print(f"Prompt: {q['prompt']}")
        print(f"Expected Ground Truth: {q['expected_gt']}")
        print(f"First Principles Basis: {q['first_principles_basis']}\n")

        gt = q["expected_gt"]

        # ARM 1: Pure LoRA Greedy Generation
        print("--- [ARM 1: Pure LoRA Greedy Generation] ---")
        chat_prompt = f"<|im_start|>user\n{q['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            greedy_out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elapsed_greedy = time.time() - t0
        greedy_text = tokenizer.decode(greedy_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        greedy_exact = f"Answer: {gt}" in greedy_text or f"answer is: {gt}" in greedy_text or f"Answer: {gt}." in greedy_text or f"Answer: {gt}\n" in greedy_text
        greedy_verdict = "PASS (Exact)" if greedy_exact else ("PARTIAL" if gt in greedy_text else "FAIL")
        print(f"Greedy Time: {elapsed_greedy:.2f}s | Verdict: {greedy_verdict}")
        print(f"Greedy Snippet:\n{greedy_text[-200:]}\n")

        # ARM 2: Greedy + SymPy Step-by-Step Verifier (Fast Zero-Search Baseline)
        print("--- [ARM 2: Greedy + SymPy Verifier (Zero-Search)] ---")
        gs_res = greedy_sympy_engine.generate(q["prompt"])
        gs_text = gs_res["response"]
        gs_exact = f"Answer: {gt}" in gs_text or f"answer is: {gt}" in gs_text or f"Answer: {gt}." in gs_text or f"Answer: {gt}\n" in gs_text
        gs_verdict = "PASS (Exact)" if gs_exact else ("PARTIAL" if gt in gs_text else "FAIL")
        print(f"Greedy+SymPy Time: {gs_res['elapsed_seconds']:.2f}s | Verified Eqs: {gs_res['verified_equations']} | Corrections: {gs_res['corrections_applied']} | Verdict: {gs_verdict}")
        print(f"Greedy+SymPy Snippet:\n{gs_text[-200:]}\n")

        # ARM 3: Token-Level MCTS with SymPy-Conditional Reward
        print("--- [ARM 3: Token-Level MCTS (Verified-Reward)] ---")
        mcts_res = mcts_engine.generate(q["prompt"])
        mcts_text = mcts_res["response"]
        mcts_exact = f"Answer: {gt}" in mcts_text or f"answer is: {gt}" in mcts_text or f"Answer: {gt}." in mcts_text or f"Answer: {gt}\n" in mcts_text
        mcts_verdict = "PASS (Exact)" if mcts_exact else ("PARTIAL" if gt in mcts_text else "FAIL")
        print(f"MCTS Time: {mcts_res['elapsed_seconds']:.2f}s | Tokens: {mcts_res['tokens_generated']} | Verdict: {mcts_verdict}")
        print(f"MCTS Snippet:\n{mcts_text[-200:]}\n")

        results.append({
            "id": q["id"],
            "domain": q["domain"],
            "prompt": q["prompt"],
            "expected_gt": gt,
            "greedy_verdict": greedy_verdict,
            "greedy_response": greedy_text,
            "greedy_time": elapsed_greedy,
            "greedy_sympy_verdict": gs_verdict,
            "greedy_sympy_response": gs_text,
            "greedy_sympy_time": gs_res["elapsed_seconds"],
            "greedy_sympy_verified_eqs": gs_res["verified_equations"],
            "greedy_sympy_corrections": gs_res["corrections_applied"],
            "mcts_verdict": mcts_verdict,
            "mcts_response": mcts_text,
            "mcts_time": mcts_res["elapsed_seconds"],
            "mcts_tokens": mcts_res["tokens_generated"]
        })

    out_file = os.path.join(ROOT, "pristine_8q_balanced_r4_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*80}\n[BALANCED r=4 BENCHMARK COMPLETE] Saved to {out_file}\n{'='*80}")


if __name__ == "__main__":
    run_pristine_benchmark()
