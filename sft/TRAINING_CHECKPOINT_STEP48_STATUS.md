# SODALS v2.1 — GRPO Training Checkpoint Status (Step 48 / 100)

**Tarih / Saat:** 23 Ekim 2026, 13:17 (Yerel Saat)  
**Eğitim Betiği:** `sft/train_grpo_math_train_100.py`  
**Model & Adapter:** `models/Qwen 3.5 9B` + `sft/qwen_lora_adapter_grpo_math_train_100`  
**Veri Kümesi:** `sft/math_train_100_grpo_prompts.json` (Hendrycks MATH Train Split, 0% MATH-500 örtüşmesi)

---

## 1. Genel İlerleme ve Telemetri Özeti

- **Tamamlanan Adım:** **48 / 100** (%48 tamamlandı)
- **Aktif Gradyan Adımı:** **45 / 48 (%93.8)**
- **Çözülen Soru Sayısı:** **33 / 48 (%68.8)** Hendrycks MATH Seviye-5 problemi başarıyla çözüldü
- **Son Model Ağırlıkları:** `sft/qwen_lora_adapter_grpo_math_train_100/adapter_model.safetensors` (Diskte güvenle saklandı, 3.94 MB)
- **Son JSON Günlüğü:** `sft/grpo_math_train_100_results.json` (48 adımın tüm rollout, ödül ve tahmin kayıtları diske yazıldı)
- **Canlı Süreç Durumu:** Görev arka planda aktif çalışıyor (`task-1901`, PID: 11308), Step 49 rollout üretimi devam ediyor.
- **GPU Donanım Sağlığı:** VRAM: **4.861 MiB / 8.188 MiB** (3.3 GB temiz fiziksel boşluk), Sıcaklık: 59°C, Güç: 28W.

---

## 2. Son Tamamlanan Adımların Doğruluk Tablosu

| Adım | Alan | Çözülen Rollout | Ort. Token | Adım Süresi | Kayıp | Üretilen Tahminler | Gerçek Cevap (GT) |
|---|---|---|---|---|---|---|---|
| **Step 43** | Counting & Probability | **3 / 4** | 292.8 | 7.8 dk | -0.0088 | `'\frac{160}{729}'`, `'160/2187'`, `'160/2187'`, `'\frac{160}{2187}'` | **`\frac{160}{2187}`** (Doğru) |
| **Step 44** | Intermediate Algebra | **4 / 4** | 513.2 | 13.1 dk | +0.0101 | `'13'`, `'13'`, `'13'`, `'13'` | **`13`** (Tam İsabet) |
| **Step 45** | Number Theory | **1 / 4** | 568.2 | 16.9 dk | +0.0084 | `'99'`, `'24'`, `'27'`, `'101'` | **`24`** (Tam İsabet) |
| **Step 46** | Counting & Probability | **1 / 4** | 945.8 | 26.0 dk | -0.0023 | `'14'`, `'10'`, `'16'`, `'41'` | **`16`** (Tam İsabet) |
| **Step 47** | Counting & Probability | 0 / 4 | 488.8 | 10.6 dk | +0.0000 | `'314'`, `'331'`, `'785'`, `'827'` | `341` |
| **Step 48** | Number Theory | **2 / 4** | 626.8 | 17.3 dk | -0.0118 | `'3'`, `'30'`, `'30'`, `'32'` | **`30`** (Tam İsabet) |
| **Step 49** | Algebra *(Canlı)* | *Sürüyor* | *Üretiliyor* | *Hesaplanıyor* | *Hesaplanıyor* | *Rollout 3/4* | `19` |

---

## 3. Resume (Yeniden Başlatma) Garantisi

Eğer bilgisayarı kapatırsanız veya yeniden başlatırsanız:
```powershell
cd C:\Users\Burak\Desktop\sodals_core
C:\Windows.old\Users\Burak\AppData\Local\Programs\Python\Python310\python.exe -u sft\train_grpo_math_train_100.py --num_prompts 100
```
Eğitim betiği diskteki `sft/grpo_math_train_100_results.json` dosyasını otomatik okuyarak tamamlanan adımları atlayıp kaldığı yerden güvenle devam edecektir.
