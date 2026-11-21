import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r"c:\Users\Burak\Desktop\sodals_core\math500_fresh50_heldout_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

base = data["0.0"]
s25 = data["0.25"]
s50_eval = data.get("0.5", {})
s50_done = len(s50_eval.get("details", []))
s50_pass = s50_eval.get("pass_count", 0)

subjects = ["Algebra", "Intermediate Algebra", "Prealgebra", "Number Theory", "Precalculus", "Geometry", "Counting & Probability"]

print("=" * 80)
print("TAZE HELD-OUT 50-SORULUK BENCHMARK: ANLIK KESİNLEŞMİŞ KARŞILAŞTIRMA MATRİSİ")
print("=" * 80)
print(f"1. Saf Base Model (Scale 0.00) : {base['pass_count']} / {base['total']} ({base['accuracy']*100:.1f}%) [TAMAMLANDI]")
print(f"2. Optimal LoRA   (Scale 0.25) : {s25['pass_count']} / {s25['total']} ({s25['accuracy']*100:.1f}%) [TAMAMLANDI]")
print(f"3. LoRA Scale 0.50 (Scale 0.50) : {s50_pass} / {s50_done} ({100*s50_pass/max(1, s50_done):.1f}%) [{s50_done}/50 İlerliyor]")
print("-" * 80)

print(f"{'Subject':25s} | {'Base (0.00)':15s} | {'Scale 0.25':15s}")
print("-" * 60)
for s in subjects:
    b_pass = base["subj_stats"][s]["pass"]
    b_tot = base["subj_stats"][s]["total"]
    l_pass = s25["subj_stats"][s]["pass"]
    l_tot = s25["subj_stats"][s]["total"]
    print(f"{s:25s} | {b_pass:2d}/{b_tot:2d} ({100*b_pass/b_tot:5.1f}%)   | {l_pass:2d}/{l_tot:2d} ({100*l_pass/l_tot:5.1f}%)")

flips_base_won = []
flips_lora_won = []
for i in range(50):
    b_det = base["details"][i]
    l_det = s25["details"][i]
    if b_det["verdict"] == "PASS" and l_det["verdict"] == "FAIL":
        flips_base_won.append((i+1, b_det["subject"], b_det["level"], b_det["ground_truth"], b_det["predicted"], l_det["predicted"]))
    elif b_det["verdict"] == "FAIL" and l_det["verdict"] == "PASS":
        flips_lora_won.append((i+1, b_det["subject"], b_det["level"], b_det["ground_truth"], b_det["predicted"], l_det["predicted"]))

print(f"\nBase Modelin Kazandığı Sorular (Base Doğru, LoRA Yanlış) [Toplam: {len(flips_base_won)}]:")
for idx, subj, lvl, gt, bp, lp in flips_base_won:
    print(f"  Q{idx:02d} [{subj:22s} | Lvl {lvl}] GT: {gt} | Base: {bp} (PASS) | LoRA: {lp} (FAIL)")

print(f"\nLoRA'nın Kazandığı Sorular (LoRA Doğru, Base Yanlış) [Toplam: {len(flips_lora_won)}]:")
for idx, subj, lvl, gt, bp, lp in flips_lora_won:
    print(f"  Q{idx:02d} [{subj:22s} | Lvl {lvl}] GT: {gt} | Base: {bp} (FAIL) | LoRA: {lp} (PASS)")
