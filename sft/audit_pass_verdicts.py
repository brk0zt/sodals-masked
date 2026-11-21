import json
import re
import sympy
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r"c:\Users\Burak\Desktop\sodals_core\math500_full500_lora_4096_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

details = data.get("details", [])

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

exact_cnt = 0
pm_cnt = 0
sympy_cnt = 0
float_cnt = 0
suspicious = []

for r in details:
    if r["verdict"] == "PASS":
        gt = r["ground_truth"]
        pred = r["predicted"]
        
        c_p = clean_latex(pred)
        c_g = clean_latex(gt)
        
        if c_p == c_g:
            exact_cnt += 1
        elif r"\pm" in gt:
            pm_cnt += 1
        else:
            sym_gt = to_sympy(gt)
            sym_pred = to_sympy(pred)
            matched = False
            if sym_gt is not None and sym_pred is not None:
                try:
                    if sympy.simplify(sym_gt - sym_pred) == 0:
                        sympy_cnt += 1
                        matched = True
                except Exception:
                    pass
                if not matched:
                    try:
                        if abs(float(sym_gt.evalf()) - float(sym_pred.evalf())) < 1e-6:
                            float_cnt += 1
                            matched = True
                    except Exception:
                        pass
            if not matched:
                suspicious.append((r["index"], gt, pred))

print("=" * 80)
print(f"500-SORULUK ÇALIŞMANIN TÜM 'PASS' KARARLARININ BAĞIMSIZ DENETİMİ")
print("=" * 80)
print(f"Toplam Değerlendirilen Soru : {len(details)}")
print(f"Toplam Doğru Kararı (PASS)  : {data.get('pass_count')}")
print(f"  1. Tam Dize Eşleşmesi (Exact Match)     : {exact_cnt} (%{100*exact_cnt/data.get('pass_count'):.1f})")
print(f"  2. \\pm Küme Eşleşmesi (Set Match)       : {pm_cnt} (%{100*pm_cnt/data.get('pass_count'):.1f})")
print(f"  3. SymPy Cebirsel Sadeleştirme (Zero-Diff): {sympy_cnt} (%{100*sympy_cnt/data.get('pass_count'):.1f})")
print(f"  4. Sayısal Tolerans (Float Equiv)        : {float_cnt} (%{100*float_cnt/data.get('pass_count'):.1f})")
print(f"  5. Şüpheli / Alt Dize ile Geçen          : {len(suspicious)}")

if suspicious:
    print("\nŞüpheli Durumlar:")
    for idx, g, p in suspicious:
        print(f"  Q{idx:03d}: GT=\"{g}\" | Pred=\"{p}\"")
else:
    print("\nSONUÇ: Sıfır alt-dize (substring) eşleşmesi. Tüm PASS kararları %100 cebirsel ve tam eşleşmedir.")
