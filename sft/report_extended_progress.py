import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r"c:\Users\Burak\Desktop\sodals_core\math500_extended_2048_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

base = data.get("0.00", {})
lora = data.get("0.25", {})

subjects = ["Algebra", "Intermediate Algebra", "Prealgebra", "Number Theory", "Precalculus", "Geometry", "Counting & Probability"]

print("=" * 80)
print("YÜKSEK KAPASİTELİ (2048 TOKEN) BENCHMARK: ANLIK İLERLEME VE KONU BAZLI DÖKÜM")
print("=" * 80)
print(f"1. Saf Base Model (alpha=0.00) : {base.get('pass_count', 0)} / {len(base.get('details', []))} ({base.get('accuracy', 0)*100:.1f}%) [TAMAMLANDI]")
lora_dets = lora.get("details", [])
lora_cnt = lora.get("pass_count", 0)
print(f"2. Optimal LoRA   (alpha=0.25) : {lora_cnt} / {len(lora_dets)} ({100*lora_cnt/max(1, len(lora_dets)):.1f}%) [{len(lora_dets)}/50 Devam Ediyor]")
print("-" * 80)

print(f"{'Subject':25s} | {'Base (2048 tok)':18s} | {'LoRA (2048 tok Anlık)':22s}")
print("-" * 72)
for s in subjects:
    b_p = base.get("subj_stats", {}).get(s, {}).get("pass", 0)
    b_t = base.get("subj_stats", {}).get(s, {}).get("total", 0)
    l_p = lora.get("subj_stats", {}).get(s, {}).get("pass", 0)
    l_t = lora.get("subj_stats", {}).get(s, {}).get("total", 0)
    b_str = f"{b_p}/{b_t} ({100*b_p/max(1,b_t):.1f}%)" if b_t > 0 else "—"
    l_str = f"{l_p}/{l_t} ({100*l_p/max(1,l_t):.1f}%)" if l_t > 0 else "—"
    print(f"{s:25s} | {b_str:18s} | {l_str:22s}")
