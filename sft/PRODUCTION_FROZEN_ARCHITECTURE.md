# SODALS v2.1 — Üretim Mimarisi Dondurma Spesifikasyonu (Production Frozen Architecture)

**Tarih:** 19 Eylül 2026  
**Durum:** RESMİ OLARAK DONDURULDU (OFFICIALLY FROZEN & LOCKED) 🔒  
**Mimar:** SODALS AI Research Team  
**Doğrulama Benchmark'ı:** 170 Soruluk Birleşik Test Paketi (150 MATH-500 + 20 Domain 2 Soyut Cebir & Lie Teorisi)

---

## 1. Yönetici Özeti ve Temel Mimari Aksiyom

Dokuz aylık kapsamlı araştırma ve beş aşamalı faktöriyel doğrulama sonucunda, çoklu uzman modellerinde en yüksek akıl yürütme performansı, en düşük çıkarım gecikmesi ve sıfır yıkıcı unutma (catastrophic forgetting) sağlayan mimari kesinleşmiştir:

> ### 🏛️ Temel Mimari Aksiyom (The Subspace Isolation Law)
> **"Uzman adaptörler kendi müfredatlarında kesinlikle İZOLE eğitilmeli; birleştirme ve ağırlık harmanlaması ortak gradyan eğitimiyle DEĞİL, yalnızca çıkarım anında nöral GPR faz ayrımı ile SÜREKLİ HARMANLAMA (Soft Blending) şeklinde yapılmalıdır."**

İki veya daha fazla adaptörün ağırlıklarını ortak gradyan döngüsünde (joint training) güncellemek, Wronskian Stiefel izometri regülarizasyonu ($\|AA^T - I\|_F^2$) ve çok düşük öğrenme oranlarında bile kaçınılmaz bir **altuzay sürtünmesine (cross-subspace perturbation, $A_1 A_2^T \neq 0$)** ve temsil erozyonuna yol açmaktadır. Bu nedenle çoklu adaptör ortak eğitimi resmi olarak amorti edilmiş (deprecated), **Eğitimsiz Soft Blender (GPR Phase Router)** nihai üretim standardı olarak kilitlenmiştir.

---

## 2. Dondurulan Ağırlıklar ve Modül Konfigürasyonu

| Bileşen | Tanım / Rol | Dondurulan Dosya Yolu | Hiperparametreler |
| :--- | :--- | :--- | :--- |
| **Temel Model (Base LLM)** | 4-bit NF4 Quantized Backbone | `models/Qwen 3.5 9B` | 4-bit NF4, double quant, FP16 compute |
| **Uzman 1 (General Math)** | Genel Matematik, Lise/Olimpiyat, STEM | `sft/qwen_lora_adapter_balanced_r4/` | Rank $r=4$, $\alpha=8$, Native Scale $= 0.25$ |
| **Uzman 2 (Abstract Algebra)** | Soyut Cebir, Lie Teorisi, Galois, Kök Uzayı | `sft/qwen_lora_adapter_domain2_r8_pondering/` | Rank $r=8$, $\alpha=16$, Native Scale $= 1.00$ |
| **Nöral Yönlendirici (GPR Router)** | Katman 32 Fisher-Kalibre Çapa Yönlendiricisi | `sft/gpr_phase_router.py` + `sft/gpr_domain_anchors.pt` | Fisher Ayrımı $= 15.30$, Eşik $\tau = 0.004913$ |
| **Sürekli Harmanlayıcı (Soft Blender)** | Eşzamanlı Çoklu-Adaptör Ağırlık Modülatörü | `sft/soft_lora_blender.py` | Sıcaklık $T = 0.03$, 128 LoraLayer aktif |

---

## 3. Matematiksel Çıkarım Mekanizması

1. **Katman 32 Nöral Çapa Problaması:**
   Model bir kullanıcı promptu aldığında, saf temel modelin Katman 32 gizli durumları ($\mathbf{h}_{32}$) çıkarılır ve dondurulmuş çapa vektörlerine olan kosinüs mesafesi üzerinden skalar faz sapması $z_i$ hesaplanır:
   $$z_i = \text{dist}(\mathbf{h}_{32}, \mathbf{c}_{D1}) - \text{dist}(\mathbf{h}_{32}, \mathbf{c}_{D2}) - \tau$$

2. **Sürekli Harmanlama Oranı ($\alpha$):**
   $$z_i \to \alpha = \sigma\left(\frac{z_i}{T}\right) \in [0, 1], \quad T = 0.03$$
   - Genel Matematik / Lise STEM: $z_i \approx -0.15 \implies \alpha \in [0.001, 0.030]$
   - Soyut Cebir / Lie Teorisi: $z_i \approx +0.15 \implies \alpha \in [0.950, 0.998]$

3. **Eşzamanlı LoRA İleri Geçişi:**
   Tüm 32 transformatör katmanında (128 LoRA katmanı) ağırlık farkı anlık ölçeklenir:
   $$\Delta W = (1 - \alpha) \cdot s_1 W_1 + \alpha \cdot s_2 W_2$$
   - $s_1 = 0.25$ (Uzman 1), $s_2 = 1.00$ (Uzman 2)

4. **Dinamik Tekrar Cezası (Repetition Penalty):**
   $$\text{rep\_penalty} = (1 - \alpha) \cdot 1.05 + \alpha \cdot 1.00$$

---

## 4. 170 Soruluk Faktöriyel Benchmark Kanıtı

Aşağıdaki tablo, beş farklı mimari aşamanın aynı 170 soru üzerindeki kesin, katı eşleşme (strict exact-match) sonuçlarını sunmaktadır:

| # | Mimari / Aşama | Yönlendirme | Ağırlık Tipi | MATH-500 (150Q) | Domain 2 (20Q) | TOPLAM (170Q) |
|:---:|:---|:---|:---|:---:|:---:|:---:|
| 1 | **Regex Router MoA** | Katı Regex/Kelime | İzole Adaptörler | 122 / 150 (81.33%) | 16 / 20 (80.00%) | 138 / 170 (81.18%) |
| **2** | **Eğitimsiz Soft Blender (DONDURULAN)** | **Sürekli GPR ($\alpha$)** | **İzole Adaptörler** | **122 / 150 (81.33%)** | **16 / 20 (80.00%)** | **138 / 170 (81.18%) 🏆** |
| 3 | Faz 3 Ham Joint Model ($B=0$) | Diferansiyellenebilir Kapı | Sıfırdan Ortak Eğitim | 111 / 150 (74.00%) | **17 / 20 (85.00%)** | 128 / 170 (75.29%) |
| 4 | Faz 3 Optimize (Kalibre Çapa) | Sürekli GPR | Sıfırdan Ortak Eğitim | 118 / 150 (78.67%) | **17 / 20 (85.00%)** | 135 / 170 (79.41%) |
| 5 | Faz 13 Warm-Start Joint Model | Sürekli GPR | İnce Ayarlı Ortak Eğitim | 120 / 150 (80.00%) | **17 / 20 (85.00%)** | 137 / 170 (80.59%) |

### Analiz ve Doğrulamalar:
- **En Yüksek Skor:** Eğitimsiz Soft Blender (138/170, %81.18), regex baseli'ın tam başarımını yakalamış ve kırılgan anahtar kelime eşleştirmesini tamamen ortadan kaldırmıştır.
- **Sıfır Çapraz Bozulma:** Uzman 1'in olgun ağırlıkları (122/150) sıfır bozulmayla korunurken, Uzman 2'nin soyut cebir başarıları (16/20) tam verimle sisteme entegre edilmiştir.
- **Düşük Gecikme:** GPR yönlendirmesi her soru için ortalama yalnızca **1.3 ms** ek süre gerektirir.

---

## 5. Çıkarım Çalıştırma Standardı (Production Inference Execution)

Üretim ortamında bu dondurulmuş sistemi çalıştırmak için kullanılan standart script:
```bash
python sft/eval_soft_moa_170.py
```
Veya tekil sorgu çıkarımı için:
```python
from sft.gpr_phase_router import NeuralPhaseRouter
from sft.soft_lora_blender import SoftLoraBlender

# 1. Router ve Blender'ı başlat
router = NeuralPhaseRouter(anchors_path="sft/gpr_domain_anchors.pt")
blender = SoftLoraBlender(threshold=router.threshold, temperature=0.03)
num_layers = blender.enable_multi_adapter(model)

# 2. Giriş promptunu GPR ile analiz et ve alpha'yı belirle
z_i, _, _ = router.compute_phase_divergence(model, probe_inputs)
alpha = blender.compute_alpha(z_i)
rep_penalty = blender.compute_rep_penalty(alpha)

# 3. LoRA ölçeklerini anlık harmanla
blender.set_blend(model, alpha)

# 4. Çıkarımı gerçekleştir
output = model.generate(**inputs, repetition_penalty=rep_penalty)
```

---

## 6. Sonuç ve Gelecek Yol Haritası

- **Dondurulan Alan:** Çoklu adaptör birleştirme ve yönlendirme mimarisi (MoA / Soft Blender). Bu alanda yeni bir ortak eğitim veya birleştirme deneyi yapılmayacaktır.
- **Açık Bırakılan Alan:** Tekil uzmanların kendi iç tavanlarının yükseltilmesi (örneğin yalnızca `expert_general_math` üzerinde yapılacak tekil RL / GRPO pilot deneyleri). Bu deneyler mevcut routing mimarisine dokunmadan, sadece uzman kartuşunun kendi iç doğruluğunu artırma hedefi güdecektir.
