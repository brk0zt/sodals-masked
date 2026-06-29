import pandas as pd
import json
import os

# Parquet dosyasını bul (HuggingFace cache yapısına göre değişebilir, indirilen klasöre bak)
parquet_path = "data/SWE-bench_Lite/data/test-00000-of-00001.parquet"

if not os.path.exists(parquet_path):
    print(f"❌ Hata: {parquet_path} bulunamadı. Lütfen indirme scriptini çalıştırdığından emin ol.")
    exit(1)

print(f"🔄 Dönüştürülüyor: {parquet_path} -> swe_lite.json")

# Okuma ve JSON'a çevirme
df = pd.read_parquet(parquet_path)
records = df.to_dict(orient="records")

# Sadece ihtiyacımız olan alanları alalım (Instance ID ve Problem Description)
cleaned_records = []
for item in records:
    cleaned_records.append({
        "instance_id": item["instance_id"],
        "problem_statement": item["problem_statement"],
        "repo": item["repo"],
        "base_commit": item["base_commit"]
    })

output_path = "data/swe_lite.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_records, f, indent=2)

print(f"✅ Hazır! {len(cleaned_records)} adet problem Rust için hazırlandı.")