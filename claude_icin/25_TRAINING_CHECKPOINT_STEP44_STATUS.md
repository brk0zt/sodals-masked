# SODALS v2.1 — GRPO Training Checkpoint Status (Step 44 / 100)

**Tarih / Saat:** 22 Ekim 2026, 17:45 (Yerel Saat)  
**Eğitim Betiği:** `sft/train_grpo_math_train_100.py`  
**Model & Adapter:** `models/Qwen 3.5 9B` + `sft/qwen_lora_adapter_grpo_math_train_100`  
**Veri Kümesi:** `sft/math_train_100_grpo_prompts.json` (Hendrycks MATH Train Split, 0% MATH-500 örtüşmesi)

---

## 1. Genel İlerleme ve Telemetri Özeti

- **Tamamlanan Adım:** **44 / 100** (%44 tamamlandı)
- **Aktif Gradyan Adımı:** **42 / 44 (%95.5)**
- **Çözülen Soru Sayısı:** **30 / 44 (%68.2)** Hendrycks MATH Seviye-5 problemi başarıyla çözüldü
- **Son Model Ağırlıkları:** `sft/qwen_lora_adapter_grpo_math_train_100/adapter_model.safetensors` (Diskte güvenle saklandı, 3.94 MB)
- **Son JSON Günlüğü:** `sft/grpo_math_train_100_results.json` (44 adımın tüm rollout, ödül ve tahmin kayıtları diske yazıldı)
- **GPU Donanım Durumu:** Görev kontrollü olarak durduruldu, VRAM: **0 MiB / 8188 MiB**, GPU sıcaklığı: 57°C.

---

## 2. Son Tamamlanan Adımların Doğruluk Tablosu

| Adım | Alan | Çözülen Rollout | Ort. Token | Adım Süresi | Kayıp | Üretilen Tahminler | Gerçek Cevap (GT) |
|---|---|---|---|---|---|---|---|
| **Step 39** | Algebra | **4 / 4** | 247.8 | 6.7 dk | +0.0044 | `'10'`, `'10'`, `'10'`, `'10'` | **`10`** (Tam İsabet) |
| **Step 40** | Algebra | **4 / 4** | 260.2 | 6.4 dk | +0.0036 | `'19'`, `'19'`, `'19'`, `'19'` | **`19`** (Tam İsabet) |
| **Step 41** | Prealgebra | **4 / 4** | 205.0 | 3.7 dk | +0.0000 | `'170'`, `'170'`, `'170'`, `'170'` | **`170`** (Tam İsabet) |
| **Step 42** | Intermediate Algebra | 0 / 4 | 692.8 | 19.5 dk | +0.0136 | `'-17'`, `'-1'`, `'-183/44'`, `'1'` | `-\frac{11}{2}` |
| **Step 43** | Counting & Probability | **3 / 4** | 292.8 | 7.8 dk | -0.0088 | `'\frac{160}{729}'`, `'160/2187'`, `'160/2187'`, `'\frac{160}{2187}'` | **`\frac{160}{2187}`** (Doğru) |
| **Step 44** | Intermediate Algebra | **4 / 4** | 513.2 | 13.1 dk | +0.0101 | `'13'`, `'13'`, `'13'`, `'13'` | **`13`** (Tam İsabet) |

---

## 3. Bilgisayar Açıldığında Tek Komutla Devam Etme Protokolü

Bilgisayarı açtığınızda eğitim kaldığı yerden (Adım 45) otomatik olarak devam edebilir:

```powershell
cd C:\Users\Burak\Desktop\sodals_core
C:\Windows.old\Users\Burak\AppData\Local\Programs\Python\Python310\python.exe -u sft\train_grpo_math_train_100.py --num_prompts 100
```

Eğitim betiği diskteki `sft/grpo_math_train_100_results.json` dosyasını otomatik okuyacak (`[RESUME] Found 44 completed steps`), Step 1–44'ü atlayıp doğrudan **Step 45 / 100** üzerinden çalışmaya başlayacaktır.
