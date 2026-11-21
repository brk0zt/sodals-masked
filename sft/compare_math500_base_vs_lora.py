import sys
import json
import re
import sympy
import math

sys.stdout.reconfigure(encoding='utf-8')

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
    return s

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
    if clean_latex(pred) == clean_latex(gt):
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

def wilson_ci(k, n, z=1.96):
    """Calculates Wilson score interval for binomial proportion."""
    if n == 0:
        return 0, 0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)

with open(r"c:\Users\Burak\Desktop\sodals_core\math500_base_model_control50_results.json", "r", encoding="utf-8") as f:
    base_data = json.load(f)

with open(r"c:\Users\Burak\Desktop\sodals_core\math500_stratified50_results.json", "r", encoding="utf-8") as f:
    lora_data = json.load(f)

base_correct = 0
lora_correct = 0
subjects = ["Algebra", "Intermediate Algebra", "Prealgebra", "Number Theory", "Precalculus", "Geometry", "Counting & Probability"]

subj_stats = {s: {"base_pass": 0, "lora_pass": 0, "total": 0} for s in subjects}
flip_lora_won = []
flip_base_won = []

for i in range(len(base_data)):
    b_item = base_data[i]
    l_item = lora_data[i]

    gt = b_item["ground_truth"]
    b_pred = b_item["predicted_answer"]
    l_pred = l_item["predicted_answer"]
    subj = b_item["subject"]

    subj_stats[subj]["total"] += 1

    b_pass = check_math_match(b_pred, gt)
    l_pass = check_math_match(l_pred, gt)

    if b_pass:
        base_correct += 1
        subj_stats[subj]["base_pass"] += 1

    if l_pass:
        lora_correct += 1
        subj_stats[subj]["lora_pass"] += 1

    if l_pass and not b_pass:
        flip_lora_won.append((i+1, subj, b_item.get("level", "?"), gt, b_pred, l_pred))
    elif b_pass and not l_pass:
        flip_base_won.append((i+1, subj, b_item.get("level", "?"), gt, b_pred, l_pred))

N = len(base_data)
base_acc = base_correct / N
lora_acc = lora_correct / N

b_low, b_high = wilson_ci(base_correct, N)
l_low, l_high = wilson_ci(lora_correct, N)

print("=" * 80)
print("MASTER MATH-500 ISOLATED DELTA ANALYSIS: BASE MODEL vs SODALS LoRA")
print("=" * 80)
print(f"Sample Size (N)              : {N} stratified questions (MATH-500)")
print(f"Base Model (Frozen, No LoRA) : {base_correct} / {N} ({base_acc*100:.1f}%) | 95% Wilson CI: [{b_low*100:.1f}%, {b_high*100:.1f}%]")
print(f"SODALS Balanced r=4 LoRA     : {lora_correct} / {N} ({lora_acc*100:.1f}%) | 95% Wilson CI: [{l_low*100:.1f}%, {l_high*100:.1f}%]")
print(f"NET LoRA DELTA (Δ)           : {lora_acc*100 - base_acc*100:+.1f} percentage points ({lora_correct - base_correct:+d} questions)")
print("=" * 80)

print("\n--- SUBJECT-BY-SUBJECT BREAKDOWN ---")
print(f"{'Subject':25s} | {'Base Model':15s} | {'LoRA Model':15s} | {'Delta (Δ)':10s}")
print("-" * 72)
for s in subjects:
    bp = subj_stats[s]["base_pass"]
    lp = subj_stats[s]["lora_pass"]
    tot = subj_stats[s]["total"]
    b_pct = 100.0 * bp / tot
    l_pct = 100.0 * lp / tot
    diff = l_pct - b_pct
    print(f"{s:25s} | {bp:2d}/{tot:2d} ({b_pct:5.1f}%)   | {lp:2d}/{tot:2d} ({l_pct:5.1f}%)   | {diff:+5.1f}%")

print("\n" + "=" * 80)
print(f"QUESTIONS WHERE BASE MODEL WON (Base Correct, LoRA Wrong) [Total: {len(flip_base_won)}]:")
print("=" * 80)
for idx, subj, lvl, gt, bp, lp in flip_base_won:
    print(f"Q{idx:02d} [{subj:22s} | Lvl {lvl}] GT: {gt} | Base: {bp} (PASS) | LoRA: {lp} (FAIL)")

print("\n" + "=" * 80)
print(f"QUESTIONS WHERE LoRA WON (LoRA Correct, Base Wrong) [Total: {len(flip_lora_won)}]:")
print("=" * 80)
for idx, subj, lvl, gt, bp, lp in flip_lora_won:
    print(f"Q{idx:02d} [{subj:22s} | Lvl {lvl}] GT: {gt} | Base: {bp} (FAIL) | LoRA: {lp} (PASS)")
