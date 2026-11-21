import json
import sys
import sympy
import re

sys.stdout.reconfigure(encoding='utf-8')

with open('hle_50_sdpo_a_results.json', 'r', encoding='utf-8') as f:
    sdpo = json.load(f)

with open('hle_50_baseline_results.json', 'r', encoding='utf-8') as f:
    base = json.load(f)

print("=" * 80)
print("HLE 50-QUESTION DIRECT A/B COMPARISON REPORT")
print("=" * 80)

def is_sympy_match(gt, pred):
    s_gt = re.sub(r"^[a-zA-Z\\]+\s*=\s*", "", gt).replace(r"\$", "").strip()
    s_pred = re.sub(r"^[a-zA-Z\\]+\s*=\s*", "", pred).replace(r"\$", "").strip()
    try:
        sg = sympy.sympify(s_gt)
        sp = sympy.sympify(s_pred)
        if sg is not None and sp is not None and sympy.simplify(sg - sp) == 0:
            return True
    except Exception:
        pass
    return False

agreements = 0
disagreements = 0

print(f"{'Idx':3s} | {'Category':20s} | {'Ground Truth':20s} | {'Baseline':20s} | {'SDPO-A':20s} | Match?")
print("-" * 95)

base_correct = 0
sdpo_correct = 0

for b, s in zip(base['details'], sdpo['details']):
    idx = b['index']
    cat = b['category']
    gt = str(b['ground_truth'])
    b_pred = str(b['predicted'])
    s_pred = str(s['predicted'])
    
    # Grading check
    b_pass = (b['verdict'] == 'PASS') or is_sympy_match(gt, b_pred)
    s_pass = (s['verdict'] == 'PASS') or is_sympy_match(gt, s_pred)
    
    if b_pass: base_correct += 1
    if s_pass: sdpo_correct += 1
    
    is_agree = (b_pred.strip().lower() == s_pred.strip().lower())
    if is_agree:
        agreements += 1
    else:
        disagreements += 1
        
    diff_mark = "EQUAL" if is_agree else "DIFF"
    if b_pass or s_pass:
        diff_mark += f" [B:{'PASS' if b_pass else 'FAIL'}, S:{'PASS' if s_pass else 'FAIL'}]"
        
    gt_s = gt[:18] + ".." if len(gt) > 18 else gt
    bp_s = b_pred[:18] + ".." if len(b_pred) > 18 else b_pred
    sp_s = s_pred[:18] + ".." if len(s_pred) > 18 else s_pred
    
    if not is_agree or b_pass or s_pass:
        print(f"{idx:02d}  | {cat:20s} | {gt_s:20s} | {bp_s:20s} | {sp_s:20s} | {diff_mark}")

print("=" * 95)
print(f"Summary:")
print(f"  Total Questions:      50")
print(f"  Identical Predictions:{agreements} / 50 ({100.0*agreements/50:.1f}%)")
print(f"  Different Predictions:{disagreements} / 50 ({100.0*disagreements/50:.1f}%)")
print(f"  Baseline Verified:    {base_correct} / 50 ({100.0*base_correct/50:.1f}%) [Q6: Set Theory, Q24: Physics]")
print(f"  SDPO-A Verified:      {sdpo_correct} / 50 ({100.0*sdpo_correct/50:.1f}%) [Q24: Physics]")
print("=" * 95)
