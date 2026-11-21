"""
SODALS v2.1 — Faz 3: Jointly-Trained GPR Soft-MoA 170-Question Benchmark
========================================================================
Evaluates the end-to-end trained model:
  - Base Model: Qwen 3.5 9B (4-bit NF4)
  - Jointly Trained Dual Adapters:
      1. expert_general_math (sft/qwen_lora_gpr_soft_moa/expert_general_math)
      2. expert_abstract_algebra (sft/qwen_lora_gpr_soft_moa/expert_abstract_algebra)
  - Jointly Trained Differentiable GPR Gate (Layer 32 anchors + learned tau=0.00567)
  - Wronskian Orthogonal Low-Rank Subspaces
  - Continuous Soft LoRA Blending: Delta W = (1-alpha)*s1*W1 + alpha*s2*W2
  - Evaluated on 170 Questions (150 MATH-500 + 20 Domain 2 Canonical)

Output: math500_domain2_faz3_trained_170_results.json
"""

import sys
import os
import json
import re
import time
import math
import torch
import sympy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, r"c:\Users\Burak\Desktop\sodals_core\sft")
from gpr_differentiable_gate import DifferentiableGPRGate
from soft_lora_blender import SoftLoraBlender

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen 3.5 9B")
SFT_DIR = os.path.join(ROOT, "sft")
ADAPTER_DIR = os.path.join(SFT_DIR, "qwen_lora_gpr_soft_moa")
ADAPTER_D1_PATH = os.path.join(ADAPTER_DIR, "expert_general_math")
ADAPTER_D2_PATH = os.path.join(ADAPTER_DIR, "expert_abstract_algebra")
AUX_PATH = os.path.join(ADAPTER_DIR, "gpr_soft_moa_aux_modules.pt")
OUTPUT_JSON = os.path.join(ROOT, "math500_domain2_faz3_trained_170_results.json")
DOMAIN2_REF_PATH = os.path.join(ROOT, "domain2_20q_interleaved_r8_results.json")


def extract_boxed_answer(text: str) -> str:
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        depth = 0
        start = idx + len(r"\boxed{")
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                if depth == 0:
                    return text[start:i].strip()
                depth -= 1
        return text[start:].strip()

    m = re.search(r"(?:Answer|answer|the answer is)[:\s]+([^\n\.]+)", text)
    if m:
        return m.group(1).strip()

    all_nums = re.findall(r'\b(-?\d+)\b', text)
    if all_nums:
        return all_nums[-1]

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return lines[-1].strip()

    return ""


def clean_latex(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = s.replace(r"\$", "").replace("$", "").strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"^\circ", "").replace("^{\\circ}", "").replace("°", "")
    s = s.replace(" ", "")
    s = re.sub(r'^[a-zA-Z]\s*=\s*', '', s)
    s = re.sub(r'\s*(?:cm|units|degrees|m|inches)\b', '', s, flags=re.I)
    return s.strip()


def to_sympy(s: str):
    s = clean_latex(s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = s.replace("^", "**")
    try:
        return sympy.sympify(s)
    except Exception:
        return None


def check_math_match(pred: str, gt: str) -> bool:
    cp = clean_latex(pred)
    cg = clean_latex(gt)
    if cp == cg:
        return True

    if cg.lstrip('-').isdigit():
        num_m = re.findall(r'-?\d+', cp)
        if num_m and num_m[-1] == cg:
            return True

    if r"\pm" in gt:
        base_parts = gt.split(r"\pm")
        if len(base_parts) == 2:
            a, b = base_parts[0].strip(), base_parts[1].strip()
            p1 = f"{a}+{b},{a}-{b}"
            p2 = f"{a}-{b},{a}+{b}"
            if clean_latex(pred) in [clean_latex(p1), clean_latex(p2)]:
                return True

    sym_gt = to_sympy(gt)
    sym_pred = to_sympy(pred)
    if sym_gt is not None and sym_pred is not None:
        try:
            if sympy.simplify(sym_gt - sym_pred) == 0:
                return True
        except Exception:
            pass
        try:
            if abs(float(sym_gt.evalf()) - float(sym_pred.evalf())) < 1e-6:
                return True
        except Exception:
            pass

    return False


def load_170_benchmark():
    items = []
    # 1. 150 MATH-500 questions
    print("Loading 150 MATH-500 questions...", flush=True)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for i in range(150):
        it = ds[i]
        items.append({
            "index": i + 1,
            "dataset": "MATH-500",
            "unique_id": it.get("unique_id", f"math500_{i+1}"),
            "subject": it["subject"],
            "level": it.get("level", "?"),
            "problem": it["problem"],
            "ground_truth": it["answer"],
            "domain": "Domain 1 (General STEM)"
        })

    # 2. 20 Domain 2 canonical questions
    print("Loading 20 Domain 2 Abstract Algebra questions...", flush=True)
    with open(DOMAIN2_REF_PATH, "r", encoding="utf-8") as f:
        d2_raw = json.load(f)["results"]
    for i, it in enumerate(d2_raw):
        items.append({
            "index": 150 + i + 1,
            "dataset": "Domain 2 20Q",
            "unique_id": it["id"],
            "subject": it.get("topic", "Abstract Algebra"),
            "level": "Graduate",
            "problem": it["prompt"],
            "ground_truth": it["expected_gt"],
            "domain": "Domain 2 (Abstract Algebra)"
        })

    print(f"Total combined benchmark: {len(items)} questions (150 MATH-500 + 20 Domain 2)", flush=True)
    return items


def run_eval():
    print("=" * 80, flush=True)
    print("SODALS v2.1 — FAZ 3 JOINTLY-TRAINED GPR SOFT-MOA BENCHMARK (170 QUESTIONS)", flush=True)
    print("Dual Trained Adapters + Learned GPR Gate (tau=0.00567) + Continuous Blending", flush=True)
    print("=" * 80, flush=True)

    items = load_170_benchmark()

    evaluated_ids = set()
    results_list = []
    pass_count = 0

    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                saved = json.load(f)
            results_list = saved.get("details", [])
            for r in results_list:
                evaluated_ids.add(r["unique_id"])
                if r["verdict"] == "PASS":
                    pass_count += 1
            print(f"Resuming: {len(evaluated_ids)} / {len(items)} already evaluated ({pass_count} passed).", flush=True)
        except Exception as e:
            print(f"Error loading progress: {e}", flush=True)
            evaluated_ids = set()
            results_list = []

    if len(evaluated_ids) >= len(items):
        print("All 170 questions already evaluated! Exiting.", flush=True)
        return

    # Load Tokenizer & Base Model
    print("Loading Tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Base Model in 4-bit NF4 onto GPU...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )

    # Attach both trained adapters
    print(f"Attaching Trained Expert 1 (General Math): {ADAPTER_D1_PATH}...", flush=True)
    model = PeftModel.from_pretrained(base_model, ADAPTER_D1_PATH, adapter_name="expert_general_math")
    print(f"Attaching Trained Expert 2 (Abstract Algebra): {ADAPTER_D2_PATH}...", flush=True)
    model.load_adapter(ADAPTER_D2_PATH, adapter_name="expert_abstract_algebra")

    # Load trained Gate
    print(f"Loading Trained GPR Gate from {AUX_PATH}...", flush=True)
    aux_data = torch.load(AUX_PATH, map_location="cpu", weights_only=False)
    gate = DifferentiableGPRGate(
        hidden_dim=4096,
        init_anchor_d1=aux_data["gate_state_dict"]["anchor_d1"].float(),
        init_anchor_d2=aux_data["gate_state_dict"]["anchor_d2"].float(),
        init_tau=float(aux_data["tau"]),
        temperature=float(aux_data["temperature"])
    ).to("cuda")
    gate.eval()
    print(f"Trained Gate loaded with learned tau={gate.tau.item():.6f}, temperature={gate.temperature}", flush=True)

    # Blender setup
    blender = SoftLoraBlender(threshold=float(aux_data["tau"]), temperature=float(aux_data["temperature"]))
    num_layers = blender.enable_multi_adapter(model)
    print(f"Soft LoRA multi-adapter active across {num_layers} LoraLayers concurrently!", flush=True)

    model.eval()

    print("\n" + "=" * 80, flush=True)
    print("STARTING BENCHMARK EVALUATION LOOP", flush=True)
    print("=" * 80, flush=True)

    t_start = time.time()

    for item in items:
        uid = item["unique_id"]
        if uid in evaluated_ids:
            continue

        prob = item["problem"]
        gt = item["ground_truth"]
        subj = item["subject"]
        ds_name = item["dataset"]
        idx = item["index"]

        # 1. Prepare raw prompt for GPR probe
        probe_prompt = (
            f"<|im_start|>user\n{prob}\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        probe_inputs = tokenizer(probe_prompt, return_tensors="pt").to(model.device)

        # 2. GPR Phase Probe & Continuous Alpha Calculation
        t_route0 = time.time()
        with torch.no_grad():
            with model.disable_adapter():
                probe_out = model(**probe_inputs, output_hidden_states=True)
                h_probe = probe_out.hidden_states[32]
                del probe_out

            alpha_tensor, z_i_tensor, _ = gate(h_probe, attention_mask=probe_inputs["attention_mask"])
            alpha = float(alpha_tensor.detach().cpu().item())
            z_i = float(z_i_tensor.detach().cpu().item())

        rep_penalty = blender.compute_rep_penalty(alpha)
        eff_scales = blender.set_blend(model, alpha)
        route_time_ms = (time.time() - t_route0) * 1000.0

        # Format prompt according to dominant expert
        if alpha >= 0.5:
            sys_msg = (
                "You are an expert mathematician specializing in abstract algebra and Lie theory. "
                "Solve the problem step by step with exact mathematical rigor. "
                "Conclude your final answer clearly within \\boxed{}."
            )
            chat_prompt = (
                f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
                f"<|im_start|>user\n{prob}\n"
                f"Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n"
            )
        else:
            chat_prompt = probe_prompt

        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                repetition_penalty=rep_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0

        gen_tokens = out[0][inputs.input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        tokens_gen = len(gen_tokens)

        pred = extract_boxed_answer(gen_text)
        is_pass = check_math_match(pred, gt)

        if is_pass:
            pass_count += 1
            verdict = "PASS"
        else:
            verdict = "FAIL"

        current_total = len(results_list) + 1
        current_acc = 100.0 * pass_count / current_total
        print(f"[{current_total:03d}/170] [{ds_name:10s} | {subj:20s}] -> [alpha={alpha:.4f} | z_i={z_i:+.4f} | rep={rep_penalty:.3f}] {verdict} ({elapsed:.1f}s) | Live: {pass_count}/{current_total} ({current_acc:.1f}%) | GT: {gt} | Pred: {pred}", flush=True)

        results_list.append({
            "index": idx,
            "dataset": ds_name,
            "unique_id": uid,
            "subject": subj,
            "level": item["level"],
            "problem": prob,
            "ground_truth": gt,
            "predicted": pred,
            "verdict": verdict,
            "alpha": round(alpha, 4),
            "z_i": round(z_i, 4),
            "rep_penalty": round(rep_penalty, 4),
            "scale_general": round(eff_scales["expert_general_math"], 4),
            "scale_algebra": round(eff_scales["expert_abstract_algebra"], 4),
            "route_time_ms": round(route_time_ms, 2),
            "tokens": tokens_gen,
            "elapsed_seconds": round(elapsed, 2)
        })
        evaluated_ids.add(uid)

        # Save live after every question
        payload = {
            "experiment": "SODALS v2.1 — Faz 3 Jointly-Trained GPR Soft-MoA (170 Questions)",
            "mechanism": "Jointly-Trained Dual LoRA (Wronskian Regularized) + Learned GPR Gate (tau=0.00567)",
            "total_evaluated": current_total,
            "total_benchmark": len(items),
            "pass_count": pass_count,
            "accuracy": round(current_acc, 2),
            "elapsed_minutes": round((time.time() - t_start) / 60, 2),
            "details": results_list
        }
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    math500_pass = sum(1 for r in results_list if r["dataset"] == "MATH-500" and r["verdict"] == "PASS")
    domain2_pass = sum(1 for r in results_list if r["dataset"] == "Domain 2 20Q" and r["verdict"] == "PASS")

    print("\n" + "=" * 80, flush=True)
    print("FAZ 3 JOINTLY-TRAINED GPR SOFT-MOA 170-QUESTION BENCHMARK COMPLETED!", flush=True)
    print(f"MATH-500 (150 Questions):       {math500_pass} / 150 ({100.0*math500_pass/150:.2f}%)", flush=True)
    print(f"Domain 2 (20 Questions):        {domain2_pass} / 20 ({100.0*domain2_pass/20:.2f}%)", flush=True)
    print(f"TOTAL BENCHMARK (170 Questions): {pass_count} / {len(items)} ({100.0*pass_count/len(items):.2f}%)", flush=True)
    print(f"Results saved to: {OUTPUT_JSON}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    run_eval()
