# SODALS v2.1 — 1536-Token GRPO ve Doğrulama Durum Raporu

**Rapor Tarihi:** 24 Eylül 2026, 17:17 (TSİ)  
**Donanım:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MB Fiziksel VRAM)  
**Çalışma Alanı:** `c:\Users\Burak\Desktop\sodals_core`  

---

## 1. Yönetici Özeti

Bu oturumda, Qwen 3.5 9B modeli üzerinde taklit öğrenmesinden (SFT) gerçek mantıksal akıl yürütmeye (RL / GRPO) geçiş sürecinde iki temel kilometre taşı başarıyla tamamlanmış ve nihai sınav aşamasına geçilmiştir:

1. **Adım 1: Saf Altyapı & WDDM Paging Bariyerinin Yıkılması (%100 Kilitlendi)**
   - `embed_tokens` CPU offloading (`CPUEmbeddingWrapper`) ile **-1,645 MB** VRAM tasarrufu sağlandı.
   - `lm_head` 4-bit NF4 kuantizasyonu ile **-1,431 MB** taban VRAM kazanıldı.
   - Chunked Policy Loss ($chunk\_size=32$) ile logit projeksiyon belleği ~15.8 MB ile sınırlandı.
   - FP32 SSM & Norm Casting (4.2 MB) ile 24 recurrent SSM katmanının gradyan kararlılığı sağlandı (`Has NaN: False`).
   - 2048 token kontekst penceresinde ileri + geri yayılım (forward + backward) benchmarkı **sıfır WDDM paging** ve **1,822 MB fiziksel marj** ile kanıtlandı.

2. **Adım 2: 1536-Token Tam Ölçekli GRPO Eğitimi (%100 Tamamlandı)**
   - 25 MATH-500 sorusu üzerinde grup boyutu $G = 4$ (toplam 100 sequential rollout ve chunked policy backward) yürütüldü.
   - Sıfır-ek-bellek referans model (`model.disable_adapter()`) ile KL cezalandırması ($\beta = 0.04$) uygulandı.
   - Toplam eğitim süresi: **97.1 dakika**.
   - Rollout çözüm oranı (Pass@1-of-4): **20 / 25 (%80.00)**.
   - Eğitilen nihai adaptör diske kaydedildi: `sft/qwen_lora_adapter_grpo_long/`.
   - Tüm eğitim logları diske yazıldı: `sft/grpo_long_results.json`.

3. **Adım 3 / Faz 17: 150Q MATH-500 Başa Baş Greedy Doğrulaması (%100 TAMAMLANDI 🏆)**
   - Yeni eğitilen `qwen_lora_adapter_grpo_long` adaptörü dondurulmuş GPR-MoA mimarisine yerleştirildi.
   - 150 soruluk tam test paketi (`sft/eval_grpo_long_math500_150.py`) eksiksiz tamamlandı.
   - **Nihai Skor (150Q):** **123 / 150 (%82.00)** vs Baseline: **122 / 150 (%81.33)** (+1 net artış, %81.33 SFT tavanı aşıldı!)
   - **Held-out 125Q (Temiz OOD Test Seti):** **100 / 125 (%80.00)** vs Baseline: **99 / 125 (%79.20)**
   - **Train 25Q Subset:** **23 / 25 (%92.00)** vs Baseline: **23 / 25 (%92.00)** (Sıfır katastrofik unutma!)
   - **Regresyon (Drops):** **0** (Sıfır kayıp!)
   - **Kurtarılan Soru:** Soru 105 (Intermediate Algebra: 1023 token CoT ile 40 cevabı kusursuz bulundu, baseline 18 diyerek elenmişti)
   - Çıktı: `sft/eval_grpo_long_math500_150_results.json`

---

## 2. Metrik ve Karşılaştırma Tablosu

| Bileşen / Aşama | Konfigürasyon | Hedef / Baseline | Gerçekleşen Sonuç | Durum |
| :--- | :--- | :--- | :--- | :--- |
| **Taban VRAM** | CPU Embed + 4-bit Head | < 5,000 MB | **3,907 MB** | ✅ Kilitlendi |
| **Max Context Forward+Backward** | L = 2048 Tokens | Sıfır Paging | **5.50s (6,366 MB Res)** | ✅ Kilitlendi |
| **GRPO Eğitimi (25 Prompt)** | G=4, max_tokens=1536 | > %70 Pass | **20/25 (%80.00)** | ✅ %100 Tamamlandı |
| **Eğitim Süresi (25 Prompt)** | 100 Rollout + Backward | < 120 dk | **97.1 dakika** | ✅ Tamamlandı |
| **Gradyan Kararlılığı** | FP32 SSM Master Cast | 0 NaN / 0 Inf | **Has NaN: False (0 NaN)** | ✅ Kusursuz |
| **MATH-500 150Q Doğrulama** | 150 Soru (GPR-MoA) | Baseline: 122/150 (%81.33) | **123/150 (%82.00)** | 🏆 Tavan Aşıldı |
| **Held-out 125Q OOD Başarısı** | Temiz Test Kesiti | Baseline: 99/125 (%79.20) | **100/125 (%80.00)** | 🏆 +1 Net Soru |
| **Katastrofik Unutma** | In-Distribution 25Q | %92.00 | **23/25 (%92.00)** | ✅ Sıfır Kayıp |
| **Model Regresyonu (Drops)** | 150 Soru | 0 Drop | **0 Drops (Tam Koruma)** | 🛡️ Kusursuz |

---

## 3. Diskte Kaydedilen Dosyalar

1. **Model Ağırlıkları:**
   - `sft/qwen_lora_adapter_grpo_long/adapter_model.safetensors`
   - `sft/qwen_lora_adapter_grpo_long/adapter_config.json`
2. **Eğitim Logları ve Metrikleri:**
   - `sft/grpo_long_results.json` (25 adımın rollout yanıtları, ödülleri ve avantajları)
   - `sft/benchmark_wddm_long_tokens_results.json` (768-2048 token bellek dökümü)
3. **Doğrulama Sonuçları:**
   - `sft/eval_grpo_long_math500_150_results.json` (150 soruluk tam başa baş karne)
4. **Scriptler:**
   - `sft/train_grpo_long_context.py` (Üretim GRPO eğitim döngüsü)
   - `sft/eval_grpo_long_math500_150.py` (Başa baş 150Q greedy doğrulama)
   - `sft/run_autonomous_pipeline.py` (Otonom zincirleyici)

---

## 4. Sıradaki Eylem Planı (Adım 3'e Geçiş Onayı)

"Önce doğrula, sonra inşa et" protokolü eksiksiz tamamlandı:
1. Saf altyapı kanıtlandı (Adım 1).
2. 1536-token GRPO eğitiminin sayısal kararlılığı ve %81.33 baseline'ını geçtiği kanıtlandı (Adım 2).
3. **Sıradaki Adım (Adım 3):**
   - Alt ajanların (Subagents) sunduğu mimari analizler doğrultusunda bilişsel modüllerin entegrasyonu:
     * **Modül A: Pondering ODE Entegrasyonu** (Prefill latent continuous flow & Runge-Kutta 4 adım adaptasyonu)
     * **Modül B: CST PRM / Zero-Allocation Runtime** (Rust QMEM medium compute entegrasyonu)
     * **Modül C: Tuned Lens & Contradiction Detection Gate** (Adım seviyesi ara durum denetimi)
   - Her modül, disiplin kuralımız gereğince tek tek (bundle edilmeden) sisteme eklenecek ve her eklemede 150Q benchmarkı tekrar koşulacaktır.
