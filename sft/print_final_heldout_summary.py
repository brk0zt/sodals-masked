import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r"c:\Users\Burak\Desktop\sodals_core\math500_fresh50_heldout_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 80)
print("BAĞIMSIZ TAZE HELD-OUT (50 AYRIK SORU) NİHAİ BENCHMARK MATRİSİ")
print("=" * 80)
for k in ["0.0", "0.25", "0.5"]:
    if k in data:
        sdata = data[k]
        dets = sdata.get("details", [])
        cnt = sdata.get("pass_count", 0)
        tot = len(dets)
        ci_low = sdata.get("ci_low", 0.0) * 100
        ci_high = sdata.get("ci_high", 0.0) * 100
        label = "Saf Base Model (alpha=0.00)" if k == "0.0" else f"Dinamik LoRA (alpha={k})"
        print(f"{label:30s}: {cnt:2d} / {tot:2d} ({100*cnt/max(1,tot):5.1f}%) | 95% Wilson CI: [{ci_low:5.1f}%, {ci_high:5.1f}%]")

subjects = ["Algebra", "Intermediate Algebra", "Prealgebra", "Number Theory", "Precalculus", "Geometry", "Counting & Probability"]

print("\n" + "=" * 80)
print("KONU BAZLI KARŞILAŞTIRMA (50 Taze Soru):")
print("=" * 80)
header = f"{'Subject':22s} | {'Base (0.00)':12s} | {'LoRA (0.25)':12s} | {'LoRA (0.50)':12s}"
print(header)
print("-" * len(header))
for s in subjects:
    b00 = f"{data['0.0']['subj_stats'][s]['pass']}/{data['0.0']['subj_stats'][s]['total']}"
    s25 = f"{data['0.25']['subj_stats'][s]['pass']}/{data['0.25']['subj_stats'][s]['total']}"
    s50_p = data.get('0.5', {}).get('subj_stats', {}).get(s, {}).get('pass', 0)
    s50_t = data.get('0.5', {}).get('subj_stats', {}).get(s, {}).get('total', 0)
    s50 = f"{s50_p}/{s50_t}"
    print(f"{s:22s} | {b00:12s} | {s25:12s} | {s50:12s}")
