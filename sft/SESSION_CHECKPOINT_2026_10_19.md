# SODALS v2.1 — Oturum Kayıt Noktası (Session Checkpoint — 20 Ekim 2026, 10:30)

## 1. Kaydedilen Güncel Eğitim Durumu (`22 / 30` Tamamlandı, `23. Adım` Rollout Logu Kaydedildi)

- **Arka Plan Görevi (`task-1029`):** Temiz şekilde durduruldu, aktif GPU işlemi kalmadı.
- **Diske Kaydedilen Adaptör Checkpoint'i:** `sft/qwen_lora_adapter_grpo_math_train_100/` (`22. adım` sonu, **21 / 22 [%95.5] aktif gradyan adımı**, **16 / 22 [%72.73] çözülen Level-5 soru**).
- **23. Adımda Gözlemlenen Canlı Kanıt (`task-1029.log`):**
  - `Rollout 1 (348 token, kısa/erken bitirme)` $\to$ `Yanlış (\frac{123}{169}, Ödül: -0.62)`
  - `Rollout 2 (1349 token, derin türetim)` $\to$ **`DOĞRU (\frac{507}{112}, Ödül: +2.85)`**
  - `Rollout 3 (1177 token)` $\to$ `Yanlış (\frac{1352}{297}, Ödül: -0.35)`
  - `Rollout 4 (1536 token)` $\to$ `Yanlış (16, Ödül: -1.42)`

---

## 2. Döndüğünde ("devam et" Dediğinde) Yapılacak İşlem

1. Kalan son 8 adım (`23..30`) tamamlanacak (veya dilersen şimdiden 21 aktif gradyan adımına ulaşmış olan `22. adım` checkpoint'i üzerinde doğrudan `sft/eval_grpo_math_train_clean_heldout.py` ile `Clean-145` değerlendirmesi başlatılacak).
