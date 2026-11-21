import json
import sys
import re
import sympy

sys.stdout.reconfigure(encoding='utf-8')

with open('hle_50_sdpo_a_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(f"Total Questions: {len(d['details'])}")
print(f"Total Time: {d['elapsed_minutes']} minutes")

def clean_latex(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\mathbb\{([^}]+)\}", r"\1", s)
    s = s.replace(r"\$", "").replace("$", "").strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"^\circ", "").replace("^{\\circ}", "").replace("°", "")
    s = s.replace(" ", "")
    s = re.sub(r"\\mathbb\{Z\}_2|Z_2|Z/2Z|\\mathbb\{Z\}/2\\mathbb\{Z\}", "Z_2", s)
    s = re.sub(r"\\mathbb\{Z\}|Z", "Z", s)
    return s

def to_sympy(s: str):
    s = clean_latex(s)
    s = re.sub(r"^[a-zA-Z\\]+\s*=\s*", "", s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = s.replace(r"\pi", "pi")
    s = s.replace("^", "**")
    try:
        return sympy.sympify(s)
    except Exception:
        return None

verified_passes = []

for r in d['details']:
    idx = r['index']
    cat = r['category']
    gt = str(r['ground_truth']).strip()
    pred = str(r['predicted']).strip()
    
    # Check 1: exact string
    if clean_latex(pred).lower() == clean_latex(gt).lower():
        verified_passes.append((idx, cat, gt, pred, "Exact String Match"))
        continue
        
    # Check 2: bare prefix strip
    p_bare = re.sub(r"^[a-zA-Z\\]+\s*=\s*", "", clean_latex(pred))
    g_bare = re.sub(r"^[a-zA-Z\\]+\s*=\s*", "", clean_latex(gt))
    if p_bare.lower() == g_bare.lower() and len(p_bare) > 0:
        verified_passes.append((idx, cat, gt, pred, "Prefix-stripped Match"))
        continue
        
    # Check 3: SymPy equivalence
    sg = to_sympy(gt)
    sp = to_sympy(pred)
    if sg is not None and sp is not None:
        try:
            if sympy.simplify(sg - sp) == 0:
                verified_passes.append((idx, cat, gt, pred, "SymPy Equivalence"))
                continue
        except Exception:
            pass
        try:
            if abs(float(sg.evalf()) - float(sp.evalf())) < 1e-5:
                verified_passes.append((idx, cat, gt, pred, "Float Precision Match"))
                continue
        except Exception:
            pass

print(f"\nTotal Verified Passes after Symbolic Normalization: {len(verified_passes)}")
for p in verified_passes:
    print(f"  [Q{p[0]:02d} | {p[1]}] Method: {p[4]}")
    print(f"     GT:   {p[2]}")
    print(f"     Pred: {p[3]}")
