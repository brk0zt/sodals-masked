import sys
import os
import json
import re
import sympy
import math

sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"c:\Users\Burak\Desktop\sodals_core"

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

    # Check set equivalence (e.g., "3 - 2\sqrt{2}, 3 + 2\sqrt{2}" vs "3 \pm 2\sqrt{2}")
    if r"\pm" in gt:
        base_parts = gt.split(r"\pm")
        if len(base_parts) == 2:
            a, b = base_parts[0].strip(), base_parts[1].strip()
            # Possibilities
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

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)

files = {
    "0.00 (Frozen Base)": os.path.join(ROOT, "math500_base_model_control50_results.json"),
    "0.25 (LoRA Scale)": os.path.join(ROOT, "math500_lora_scale_025_results.json"),
    "0.50 (LoRA Scale)": os.path.join(ROOT, "math500_lora_scale_050_results.json"),
    "1.00 (Full LoRA)": os.path.join(ROOT, "math500_stratified50_results.json"),
}

summary = {}
subjects = ["Algebra", "Intermediate Algebra", "Prealgebra", "Number Theory", "Precalculus", "Geometry", "Counting & Probability"]

for label, fpath in files.items():
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    correct = 0
    by_subj = {s: {"pass": 0, "total": 0} for s in subjects}

    for item in data:
        subj = item["subject"]
        by_subj[subj]["total"] += 1
        gt = item["ground_truth"]
        pred = item["predicted_answer"]
        if check_math_match(pred, gt):
            correct += 1
            by_subj[subj]["pass"] += 1

    low, high = wilson_ci(correct, len(data))
    summary[label] = {
        "correct": correct,
        "total": len(data),
        "acc": correct / len(data),
        "ci_low": low,
        "ci_high": high,
        "by_subj": by_subj
    }

print("=" * 80)
print("MASTER LoRA PARETO SCALING ANALYSIS ON MATH-500 (N=50)")
print("=" * 80)
for label, res in summary.items():
    print(f"{label:22s}: {res['correct']:2d} / {res['total']:2d} ({res['acc']*100:5.1f}%) | 95% Wilson CI: [{res['ci_low']*100:5.1f}%, {res['ci_high']*100:5.1f}%]")

print("\n" + "=" * 80)
print("SUBJECT BREAKDOWN ACROSS ALL 4 SCALES:")
print("=" * 80)
header = f"{'Subject':22s} | {'Base (0.00)':12s} | {'Scale 0.25':12s} | {'Scale 0.50':12s} | {'Full (1.00)':12s}"
print(header)
print("-" * len(header))
for s in subjects:
    b00 = f"{summary['0.00 (Frozen Base)']['by_subj'][s]['pass']}/{summary['0.00 (Frozen Base)']['by_subj'][s]['total']}"
    s25 = f"{summary['0.25 (LoRA Scale)']['by_subj'][s]['pass']}/{summary['0.25 (LoRA Scale)']['by_subj'][s]['total']}"
    s50 = f"{summary['0.50 (LoRA Scale)']['by_subj'][s]['pass']}/{summary['0.50 (LoRA Scale)']['by_subj'][s]['total']}"
    s10 = f"{summary['1.00 (Full LoRA)']['by_subj'][s]['pass']}/{summary['1.00 (Full LoRA)']['by_subj'][s]['total']}"
    print(f"{s:22s} | {b00:12s} | {s25:12s} | {s50:12s} | {s10:12s}")
