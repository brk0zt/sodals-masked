# SODALS v2.1 — GRPO Training Checkpoint Status (Step 60 / 100)

**Tarih / Saat:** 2 Kasım 2026, 17:49 (Yerel Saat)  
**Eğitim Betiği:** `sft/train_grpo_math_train_100.py`  
**Model & Adapter:** `models/Qwen 3.5 9B` + `sft/qwen_lora_adapter_grpo_math_train_100`  
**Veri Kümesi:** `sft/math_train_100_grpo_prompts.json` (Hendrycks MATH Train Split, 0% MATH-500 örtüşmesi)

---

## 1. Genel İlerleme ve Telemetri Özeti

- **Tamamlanan Adım:** **60 / 100** (%60 tamamlandı)
- **Aktif Gradyan Adımı:** **55 / 60 (%91.7)** (Politika gradyanı adımların %91.7'sinde aktif olarak güncellendi)
- **Çözülen Soru Sayısı:** **39 / 60 (%65.0)** Hendrycks MATH Seviye-5 problemi tam doğru çözüldü
- **Son Model Ağırlıkları:** `sft/qwen_lora_adapter_grpo_math_train_100/adapter_model.safetensors` (Diskte güvenle saklandı, 3.94 MB)
- **Son JSON Günlüğü:** `sft/grpo_math_train_100_results.json` (60 adımın tüm rollout, ödül ve tahmin kayıtları diske yazıldı)
- **GPU Donanım Durumu:** Görev kontrollü olarak durduruldu, VRAM: **0 MiB / 8188 MiB**, GPU sıcaklığı: 56°C.

---

## 2. Son Tamamlanan Adımların Doğruluk Tablosu

| Adım | Alan | Çözülen Rollout | Ort. Token | Adım Süresi | Kayıp | Üretilen Tahminler | Gerçek Cevap (GT) |
|---|---|---|---|---|---|---|---|
| **Step 54** | Algebra | **4 / 4** | 204.5 | 8.3 dk | -0.0086 | `'109'`, `'109'`, `'109'`, `'109'` | **`109`** (Tam İsabet) |
| **Step 55** | Prealgebra | **4 / 4** | 288.8 | 9.5 dk | +0.0078 | `'17'`, `'17'`, `'17'`, `'17'` | **`17`** (Tam İsabet) |
| **Step 56** | Intermediate Algebra | **4 / 4** | 711.2 | 20.0 dk | -0.0053 | `'1247'`, `'1247'`, `'1247'`, `'1247'` | **`1247`** (Tam İsabet) |
| **Step 57** | Prealgebra | **4 / 4** | 180.0 | 3.8 dk | +0.0000 | `'\frac{99}{100}'`, `'\frac{99}{100}'`, `'\frac{99}{100}'`, `'99/100'` | **`\frac{99}{100}`** (Tam İsabet) |
| **Step 58** | Intermediate Algebra | 0 / 4 | 728.5 | 22.4 dk | -0.0482 | `'156'`, `'0'`, `'-8'`, `'0'` | `84` |
| **Step 59** | Intermediate Algebra | 0 / 4 | 537.8 | 15.5 dk | -0.0150 | `'1225'`, `'725'`, `'1225'`, `'1225'` | `1850` |
| **Step 60** | Geometry | 0 / 4 | 946.0 | 26.8 dk | +0.0282 | `'783'`, `'2'`, `'25'`, `'14'` | `31` |

---

## 3. Bilgisayar Açıldığında Tek Komutla Devam Etme Protokolü

Bilgisayarı açtığınızda eğitim kaldığı yerden (Adım 61) otomatik olarak devam edebilir:

```powershell
cd C:\Users\Burak\Desktop\sodals_core
C:\Windows.old\Users\Burak\AppData\Local\Programs\Python\Python310\python.exe -u sft\train_grpo_math_train_100.py --num_prompts 100
```

Eğitim betiği diskteki `sft/grpo_math_train_100_results.json` dosyasını otomatik okuyacak (`[RESUME] Found 60 completed steps`), Step 1–60'ı atlayıp doğrudan **Step 61 / 100** üzerinden çalışmaya başlayacaktır.
