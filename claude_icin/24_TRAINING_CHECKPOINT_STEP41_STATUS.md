# SODALS v2.1 — GRPO Training Checkpoint Status (Step 41 / 100)

**Tarih / Saat:** 22 Ekim 2026, 16:47 (Yerel Saat)  
**Eğitim Betiği:** `sft/train_grpo_math_train_100.py`  
**Model & Adapter:** `models/Qwen 3.5 9B` + `sft/qwen_lora_adapter_grpo_math_train_100`  
**Veri Kümesi:** `sft/math_train_100_grpo_prompts.json` (Hendrycks MATH Train Split, 0% MATH-500 örtüşmesi)

---

## 1. Genel İlerleme ve Telemetri Özeti

- **Tamamlanan Adım:** **41 / 100** (%41 tamamlandı)
- **Aktif Gradyan Adımı:** **39 / 41 (%95.1)** (Önceki 25-promptluk pilotta %60 idi)
- **Çözülen Soru Sayısı:** **28 / 41 (%68.3)** Hendrycks MATH Seviye-5 problemi başarıyla çözüldü
- **Son Model Ağırlıkları:** `sft/qwen_lora_adapter_grpo_math_train_100/adapter_model.safetensors` (Diskte güvenle saklandı, 3.94 MB)
- **Son JSON Günlüğü:** `sft/grpo_math_train_100_results.json` (41 adımın tüm rollout, ödül ve tahmin kayıtları diske yazıldı)
- **GPU Donanım Durumu:** Görev kontrollü olarak durduruldu, VRAM: **0 MiB / 8188 MiB**, GPU sıcaklığı: 57°C.

---

## 2. Son Tamamlanan Adımların Doğruluk Tablosu

| Adım | Alan | Çözülen Rollout | Ort. Token | Adım Süresi | Kayıp | Üretilen Tahminler | Gerçek Cevap (GT) |
|---|---|---|---|---|---|---|---|
| **Step 37** | Counting & Probability | **4 / 4** | 202.8 | 6.9 dk | +0.0153 | `'8788'`, `'8788'`, `'8788'`, `'8788'` | **`8788`** (Tam İsabet) |
| **Step 38** | Intermediate Algebra | **2 / 4** | 306.0 | 9.1 dk | +0.0058 | `'25/6'`, `'25/6'`, `'10/3'`, `'8/3'` | **`\frac{25}{6}`** (Doğru) |
| **Step 39** | Algebra | **4 / 4** | 247.8 | 6.7 dk | +0.0044 | `'10'`, `'10'`, `'10'`, `'10'` | **`10`** (Tam İsabet) |
| **Step 40** | Algebra | **4 / 4** | 260.2 | 6.4 dk | +0.0036 | `'19'`, `'19'`, `'19'`, `'19'` | **`19`** (Tam İsabet) |
| **Step 41** | Prealgebra | **4 / 4** | 205.0 | 3.7 dk | +0.0000 | `'170'`, `'170'`, `'170'`, `'170'` | **`170`** (Tam İsabet) |

---

## 3. Bilgisayar Açıldığında Tek Komutla Devam Etme Protokolü

Bilgisayarı açtığınızda eğitim kaldığı yerden (Adım 42) otomatik olarak devam edebilir:

```powershell
cd C:\Users\Burak\Desktop\sodals_core
C:\Windows.old\Users\Burak\AppData\Local\Programs\Python\Python310\python.exe -u sft\train_grpo_math_train_100.py --num_prompts 100
```

Eğitim betiği diskteki `sft/grpo_math_train_100_results.json` dosyasını otomatik okuyacak (`[RESUME] Found 41 completed steps`), Step 1–41'i atlayıp doğrudan **Step 42 / 100** üzerinden çalışmaya başlayacaktır.
