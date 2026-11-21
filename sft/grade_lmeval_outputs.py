import json
import re
import sympy
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r"c:\Users\Burak\Desktop\sodals_core\lmeval_base_math500_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data.get("samples", {}).get("hendrycks_math500", [])

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
    if clean_latex(gt) in clean_latex(pred) and len(clean_latex(gt)) > 2:
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

def extract_answer(text: str) -> str:
    # 1. Check \boxed{...}
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
    
    # 2. Check Answer: ...
    m = re.findall(r"(?:Answer|answer|the answer is)[:\s]+([^\n\.]+)", text)
    if m:
        return m[-1].strip()
    
    # 3. Fallback to last line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return lines[-1].strip()
    return ""

correct = 0
total = len(samples)

print("=" * 80)
print(f"GRADING OFFICIAL LM-EVAL GENERATED OUTPUTS (N={total})")
print("=" * 80)

for i, s in enumerate(samples):
    gt = s.get("target", "")
    resps = s.get("filtered_resps", [""])[0]
    pred = extract_answer(resps)
    is_match = check_math_match(pred, gt)
    if is_match:
        correct += 1
        v = "PASS"
    else:
        v = "FAIL"
    print(f"[{i+1:02d}/50] {v} | GT: {gt[:30]:30s} | Pred: {pred[:30]:30s}")

print("=" * 80)
print(f"LM-EVAL GENERATION ACCURACY (Qwen 3.5 9B Base in 4-bit NF4): {correct} / {total} ({100*correct/total:.1f}%)")
print("=" * 80)
