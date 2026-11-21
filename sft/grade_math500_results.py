import json
import re
import sympy

with open(r"c:\Users\Burak\Desktop\sodals_core\math500_stratified50_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def clean_latex(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = s.replace(r"\$", "").replace("$", "").strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(" ", "")
    return s

def to_sympy(s: str):
    s = clean_latex(s)
    # convert \frac{a}{b} or \frac ab to ((a)/(b))
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = s.replace("^", "**")
    try:
        return sympy.sympify(s)
    except Exception:
        return None

strict_pass = sum(1 for q in data if q["verdict"] == "PASS")
math_pass = 0
rescued = []
final_by_subject = {}

for q in data:
    subj = q["subject"]
    if subj not in final_by_subject:
        final_by_subject[subj] = {"pass": 0, "total": 0}
    final_by_subject[subj]["total"] += 1

    gt = q["ground_truth"]
    pred = q["predicted_answer"]
    is_correct = False
    
    # 1. Exact / Clean string match
    if clean_latex(pred) == clean_latex(gt):
        is_correct = True
    else:
        # 2. SymPy algebraic equivalence
        sym_gt = to_sympy(gt)
        sym_pred = to_sympy(pred)
        if sym_gt is not None and sym_pred is not None:
            try:
                if sympy.simplify(sym_gt - sym_pred) == 0:
                    is_correct = True
            except Exception:
                pass
            if not is_correct:
                try:
                    if abs(float(sym_gt.evalf()) - float(sym_pred.evalf())) < 1e-6:
                        is_correct = True
                except Exception:
                    pass

    if is_correct:
        math_pass += 1
        final_by_subject[subj]["pass"] += 1
        if q["verdict"] == "FAIL":
            rescued.append((q["index"], subj, q.get("level", "?"), gt, pred))

print("=" * 80)
print(f"MATH-500 STRATIFIED 50: FINAL ACCURATE RESULTS")
print("=" * 80)
print(f"Strict String Match Score     : {strict_pass} / 50 ({100*strict_pass/50:.1f}%)")
print(f"True Mathematical Score (SymPy) : {math_pass} / 50 ({100*math_pass/50:.1f}%)")
print("-" * 80)

print("\nFormatting False-Negatives Rescued by Algebraic Normalization:")
for idx, subj, lvl, gt, pred in rescued:
    print(f"  Q{idx:02d} [{subj:22s} | Level {lvl}]: GT=\"{gt}\" <==> Pred=\"{pred}\" (Mathematically Equivalent)")

print("\n" + "=" * 80)
print("FINAL ACCURACY BY SUBJECT (50 Stratified Questions):")
print("=" * 80)
for subj, stats in final_by_subject.items():
    p = stats["pass"]
    t = stats["total"]
    print(f"  {subj:25s}: {p} / {t} ({100.0 * p / t:.1f}%)")
