import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r"c:\Users\Burak\Desktop\sodals_core\math500_full500_lora_4096_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

tot = data.get("total_evaluated", 0)
cnt = data.get("pass_count", 0)
acc = data.get("accuracy", 0.0) * 100
low = data.get("ci_low", 0.0) * 100
high = data.get("ci_high", 0.0) * 100
elapsed = data.get("elapsed_minutes", 0.0)

print("=" * 80)
print("TAM 500-SORULUK MATH-500 BENCHMARK: ANLIK CANLI SKOR TABLOSU")
print("=" * 80)
print(f"Model              : {data.get('model')}")
print(f"Tamamlanan Soru    : {tot} / 500 Soru (%{tot*100/500:.1f})")
print(f"Anlık Doğru Sayısı : {cnt} / {tot}")
print(f"Anlık Başarı Oranı : %{acc:.2f}")
print(f"95% Wilson Güven   : [%{low:.2f} – %{high:.2f}]")
print(f"Geçen Süre         : {elapsed:.1f} dakika (Ortalama {elapsed*60/max(1,tot):.1f} saniye/soru)")
print("-" * 80)

print("KONU BAZLI ANLIK DÖKÜM:")
print(f"{'Subject':25s} | {'Doğru / Toplam':18s} | {'Başarı Oranı':15s}")
print("-" * 65)
for s, stats in data.get("subj_stats", {}).items():
    p = stats["pass"]
    t = stats["total"]
    if t > 0:
        print(f"{s:25s} | {p:3d} / {t:3d}            | %{100*p/t:5.1f}")
