import json
import subprocess
import os
from datasets import load_dataset
from tqdm import tqdm

# Bizim Rust Motorunu Çağıran Fonksiyon
def call_sodals_engine(problem_statement):
    # Rust projesinin olduğu dizin
    rust_project_path = "/c/Users/Burak/Desktop/sodals_core" 
    
    # Komutu hazırla (Release modunda çalıştır ki hızlı olsun)
    # --q argümanı ile soruyu yolluyoruz
    command = ["cargo", "run", "--release", "--", "--mode", "engineer", "--prompt", problem_statement]
    
    try:
        # Rust motorunu çalıştır
        result = subprocess.run(
            command, 
            cwd=rust_project_path,
            capture_output=True, 
            text=True, 
            encoding='utf-8'
        )
        return result.stdout
    except Exception as e:
        return f"Error: {e}"

def main():
    # 1. SWE-Bench Lite setini indir (Tam set çok büyük, Lite ile başla)
    print("📥 Veri seti indiriliyor...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    predictions = []

    # 2. Her soruyu çözmeye çalış
    for instance in tqdm(dataset):
        instance_id = instance["instance_id"]
        problem_text = instance["problem_statement"]
        
        print(f"\n🧠 SODALS Çözüyor: {instance_id}")
        
        # Rust Motoruna Sor
        raw_output = call_sodals_engine(problem_text)
        
        # --- ÖNEMLİ: Çıktıdan Sadece Kod Bloğunu Ayıkla ---
        # Bizim motor markdown (```) veriyor, bunu patch formatına uydurmalıyız.
        # Basit bir parsing mantığı (Geliştirilebilir):
        model_patch = raw_output 
        if "```diff" in raw_output:
            model_patch = raw_output.split("```diff")[1].split("```")[0]
        elif "```python" in raw_output:
             # Eğer python kodu verdiyse bunu diff'e çevirmek için ek işlem gerekir
             # Şimdilik ham çıktıyı alıyoruz.
             model_patch = raw_output

        # 3. Sonucu kaydet
        predictions.append({
            "instance_id": instance_id,
            "model_patch": model_patch,
            "model_name_or_path": "SODALS_v3_Physics"
        })

    # 4. Tahmin dosyasını yaz
    with open("sodals_predictions.jsonl", "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
            
    print("✅ Tahminler 'sodals_predictions.jsonl' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()