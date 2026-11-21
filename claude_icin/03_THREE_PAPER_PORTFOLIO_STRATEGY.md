# SODALS Üçlü Makale Portföy Stratejisi (Three-Paper Portfolio Strategy)
## Akademik Ayrıştırma, Hedef Dergi/Konferanslar ve Yayın Yol Haritası

**Tarih:** 19 Eylül 2026  
**Yazar:** Burak Öztürk & SODALS AI Research Group  
**Amaç:** Tek bir mega-makale ("kitchen-sink paper") yerine, her biri tek bir spesifik bilimsel problemi derinlemesine çözen, yöntemleri ve deneyleri su geçirmez **üç bağımsız akademik yayının** planlanması.

---

## Genel Bakış: 3 Makalenin Konumlandırma Matrisi

```mermaid
graph TD
    subgraph P1["MAKALE 1 (AMİRAL GEMİSİ)"]
        T1["SDPO, Long-Context GRPO & Pondering ODE"]
        V1["Hedef: NeurIPS / ICML / ICLR / TMLR"]
        C1["Katkı: SymPy doğrulamalı SDPO, Plasebo Kontrolü (0/35 vs 4/35), Sıfır-Paging VRAM Mimarisi (3.42GB tasarruf), 1536-Token GRPO ile SFT tavanının aşılması (123/150 %82.0, 0 drop), RK4 Manifold Pondering ODE"]
    end

    subgraph P2["MAKALE 2 (SİSTEM & ALGORİTMA)"]
        T2["Continuous Soft-MoA & Subspace Isolation"]
        V2["Hedef: ICLR Tiny Papers / COLING / CoNLL / MLSys"]
        C2["Katkı: 170Q Multi-Domain MoA, Alt-uzay Girişim Yasası (Joint Training erozyonu vs Untrained Soft Blender %81.18), Layer-32 GPR Fisher ayrımı"]
    end

    subgraph P3["MAKALE 3 (UYGULAMA & CPS)"]
        T3["EdgeSSM: Zero-Allocation Inference Engine"]
        V3["Hedef: IEEE IoT Journal / IEEE TII (Q1) / MLSys"]
        C3["Katkı: Rust sıfır-tahsisli çalışma zamanı (100MB RAM, 28.5ms), GGUF tile dizilimi, CDCL SSM re-prefill senkronizasyonu, AssetOpsBench (%80.0)"]
    end
```

---

## 1. MAKALE 1: SDPO, Long-Context GRPO ve Sürekli Manifold Regülasyonu (Amiral Gemisi)

- **Tam Başlık:** *Self-Distillation Policy Optimization (SDPO), Long-Context GRPO, and Continuous Manifold Stabilization in Frontier Reasoning*
- **Ana Dosya:** [arxiv/arXiv_SODALS_v2.1_Master_Paper.md](file:///c:/Users/Burak/Desktop/sodals_core/arxiv/arXiv_SODALS_v2.1_Master_Paper.md)
- **Durum:** **Makale taslağı tamamlandı; SDPO plasebo kontrolü, sıfır-paging VRAM mimarisi, 1536-token GRPO eğitimi ve Pondering ODE doğrulamaları eklendi.**

### Bilimsel Problem ve Hipotez:
- *Problem:* (1) LLM'lerde kendi kendini düzeltme (self-correction) eğitiminin gerçekten nedensel bir öğrenme refleksi mi yarattığı yoksa rastgele ağırlık oynaması mı olduğu; (2) Tüketici sınıfı donanımda (8GB VRAM) uzun akıl yürütme pencerelerinin ($L=1536\text{--}2048$) WDDM sürücü paging cezası olmadan açılıp açılamayacağı; (3) Takviyeli öğrenmenin (RL) SFT tavanını katastrofik unutma olmadan aşıp aşamayacağı.
- *Hipotez:* SymPy doğrulamalı yörüngelerle SDPO eğitimi nedensel hata onarım refleksi sağlar; CPU embedding offload + NF4 head kuantizasyonu + chunked policy loss ile VRAM tabanı 7.3GB'dan 3.9GB'a indirilerek 8GB GPU'da sıfır paging ile 1536-token GRPO çalıştırılabilir; bu sayede MATH-500 üzerinde SFT tavanı aşılarak 123/150 (%82.00) rekoruna sıfır regresyonla ulaşılır.

### Ana Deneysel Bulgular:
1. **Negatif Plasebo Kontrolü:** 17 adımlık plasebo modeli görülemeyen 35 sınır sorusunda **0 / 35 (%0.0)** düzeltme yaparken, Saf SDPO **4 / 35 (%11.4)** net düzeltme sağlamıştır (iki yönlü Fisher $p = 0.114$, tek yönlü $p = 0.057$, McNemar $p \approx 0.125$; $N=35$ sınırlı güçtedir, $N \ge 100$ hedeflenmektedir).
2. **$F_4$ Anomalisi ve Factual Anchor Takası:** $F_4=52$ bilgisini korumak için 3 adet alan anchor'ı eklendiğinde (SDPO-A), $F_4=52$ başarıyla korunmuş (21/24, %87.5) ancak held-out hata kurtarma oranı **4/35'ten 2/35'e (%5.71)** düşmüştür (Pareto kapasite takası).
3. **Sıfır-Paging VRAM Mimarisi (3.42 GB Tasarruf):** Host CPU embedding aktarımı (-1,645 MB), 4-bit NF4 LM Head (-1,431 MB) ve Chunked Policy Loss ($chunk\_size=32$, logit belleği 15.8 MB) ile model tabanı 3,907 MB'a çekilmiş; $L=2048$ token ileri+geri yayılımında 0 WDDM paging ile 1,822 MB serbest fiziksel alan korunmuştur (Rollout hızı: 58.57 tok/s).
4. **1536-Token GRPO ile SFT Tavanının Aşılması:** 25 promptluk ($G=4$, 100 rollout) takviyeli öğrenme döngüsü sonrası MATH-500 150-soruluk başa baş testinde **123 / 150 (%82.00)** skoru elde edilerek dondurulmuş SFT tabanı (122/150, %81.33) aşılmıştır. 150 soruda **0 regresyon (sıfır kayıp)** yaşanmış, temiz test setinde 100/125 (%80.00) OOD genellemesi sağlanmış ve Q105 sorusu 1,023 tokenlık derin CoT ile kusursuz çözülmüştür.
5. **Prefill Pondering ODE (RK4 Manifold Regülasyonu):** 4. derece Runge-Kutta sürekli manifold gevşetmesi, prompt prefill aşamasında top-1 greedy kararlarını bozmadan 123/150 (%82.00) rekorunu 0 regresyonla stabilize etmiştir.

### Hedef Mecralar:
- **Konferanslar:** NeurIPS / ICML / ICLR / ACL.
- **Dergi:** Transactions on Machine Learning Research (TMLR).

---

## 2. MAKALE 2: Tüketici Donanımında Continuous Soft-MoA & Alt-Uzay İzolasyonu (Sistem/Algoritma)

- **Tam Başlık:** *Learned Continuous Soft Mixture of Low-Rank Adapters for Multi-Domain Fine-Tuning on Constrained Hardware*
- **Ana Dosya:** [arxiv/arXiv_SODALS_v2.1_Soft_MoA_Paper.md](file:///c:/Users/Burak/Desktop/sodals_core/arxiv/arXiv_SODALS_v2.1_Soft_MoA_Paper.md)
- **Çekirdek Kodlar:** [sft/gpr_differentiable_gate.py](file:///c:/Users/Burak/Desktop/sodals_core/sft/gpr_differentiable_gate.py), [sft/soft_lora_blender.py](file:///c:/Users/Burak/Desktop/sodals_core/sft/soft_lora_blender.py), [sft/eval_soft_moa_170.py](file:///c:/Users/Burak/Desktop/sodals_core/sft/eval_soft_moa_170.py)
- **Durum:** **Makale taslağı tamamlandı; 170-soruluk çok alanlı ablasyon tamamlandı; alt-uzay girişim etkisi (Subspace Interference Effect) ampirik olarak kanıtlandı ve donduruldu.**

### Bilimsel Problem ve Hipotez:
- *Problem:* 8GB VRAM'e sahip tüketici donanımında birden fazla alana (Genel STEM ve Soyut Cebir) ait LoRA adaptörlerinin ortak eğitilmesi çapraz kovaryans ($A_1 A_2^T \neq 0$) nedeniyle katastrofik girişime yol açar.
- *Hipotez:* Katman 32 latent aktivasyonlarından beslenen GPR yönlendiricisi ve sürekli harmanlayıcı ($\Delta W = (1-\alpha)s_1 W_1 + \alpha s_2 W_2$), adaptörleri eğitimde kesin olarak izole tutup yalnızca çıkarım anında harmanlayarak en yüksek doğruluğu sağlar.

### Ana Mühendislik ve Deneysel Katkılar:
1. **170-Soruluk Çoklu Alan Doğrulaması:** 150 MATH-500 + 20 Soyut Cebir sorusunda Eğitimsiz Soft Blender **138 / 170 (%81.18)** ile zirve skoru vermiştir.
2. **Ortak Eğitimin Çöküşü (Subspace Interference Law):** Uçtan uca ortak eğitim (Faz 3 Ham: 128/170, Faz 3 Optimize: 135/170, Faz 13 Warm-Start: 137/170) yapıldığında modelin ilk kez çözdüğü $E_7=133$ Lie boyutu silinmiş (`70`), $\mathfrak{sp}(6)=21$ bozulmuş (`1`) ve net gerileme yaşanmıştır.
3. **Üretim Standardı:** Adaptörlerin izole eğitilip çıkarım anında GPR ile yumuşak harmanlanması (Untrained Soft Blender) sıfır unutma ve 1.3ms gecikmeyle resmî üretim standardı olarak kilitlenmiştir.

### Hedef Mecralar:
- **Konferans:** ICLR 2027 Tiny Papers / COLING 2027 / CoNLL 2027.
- **Workshop:** MLSys Workshop on Efficient Systems for Foundation Models.

---

## 3. MAKALE 3: EdgeSSM: Sıfır-Bellek Tahsisli Durum-Uzay Modeli Çıkarım Motoru (Sistem/Edge CPS)

- **Tam Başlık:** *EdgeSSM: A Deterministic, Zero-Allocation State Space Model Inference Engine for Industrial Cyber-Physical Systems*
- **Ana Dosya:** [`SODALS_arXiv_Paper.md`](file:///c:/Users/Burak/Desktop/sodals_core/SODALS_arXiv_Paper.md)
- **Durum:** **Makale taslağı tamamlandı; sahte fizik terimlerinden arındırılarak saf uç-sistemleri (Edge Systems) ve Deterministik SSM çıkarım mimarisi olarak yeniden yazıldı.**

### Bilimsel Problem ve Hipotez:
- *Problem:* Endüstriyel nesnelerin interneti ve siber-fiziksel sistemlerde (CPS) derin sinir ağları çalıştırmak; deterministik olmayan GC duraklamaları, yüksek bellek gereksinimi ve tekrarlayan SSM durumlarının geri alınamazlığı (irreversibility) nedeniyle güvensizdir.
- *Hipotez:* Rust ile yazılmış sıfır-tahsisli (zero-allocation) bir çalışma zamanı (`RtRingBuffer`, 100MB RAM), GGUF `tile` tensör dizilimi, CUDA CSR çizge dikkat yönlendirmesi, deterministik hash-tabanlı kural öncülleri ve CDCL re-prefill önbellek senkronizasyonu; 9B parametreli bir modeli RTX 4060 seviyesinde 28.5ms gecikmeyle çalıştırarak GPT-4o ile tanısal pariteye ulaştırır.

### Ana Deneysel Bulgular:
1. **Endüstriyel Benchmark Doğrulaması:** AssetOpsBench ($N=120$) üzerinde %80.0 tanısal tam eşleşme (+21.7 puan açık-ağırlık tabanı üstünde, $p=0.0004$), SRE görevinde %89.5, FinOps görevinde %93.3 başarı; GPT-4o ile Wilson güven aralıklarında tam istatistiksel parite ($p=0.87$).
2. **GGUF `tile` Hata Düzeltmesi:** `repeat_interleave` yerine `tile` tensör dizilimine geçilmesi, Layer-0 kosinüs benzerliğini 0.0300'dan 0.9996'ya çıkarmış, KL sapmasını $7.22$'den $3.83 \times 10^{-8}$'e düşürmüştür.
3. **CDCL ve SSM Durum Senkronizasyonu:** Geriye dönüş sırasında KV önbelleklerinin sıfırlanıp 0.22ms içinde paralel re-prefill yapılması ile tekrarlayan durum tutarlılığı garanti altına alınmıştır.

### Hedef Mecralar:
- **Q1 Dergiler:** IEEE Transactions on Industrial Informatics (TII), IEEE Internet of Things Journal, ACM Transactions on Embedded Computing Systems (TECS).
- **Konferanslar:** MLSys, ACM EdgeSys.

---

## 4. Uygulama ve Teslim Takvimi

| Hafta | Makale 1 (SDPO) | Makale 2 (Soft-MoA) | Makale 3 (Endüstriyel AI) |
|:---:|:---|:---|:---|
| **Hafta 1** | ✅ Metin ve formal bölümler tamamlandı | ✅ Autograd kopukluğu onarıldı | Kod temizliği ve framing revizyonu |
| **Hafta 2** | Held-out soru sayısını 100'e çıkarma | 304 örnekli tam eğitim döngüsü | QMEM kartridj ablasyonu (kartridjli vs kartridjsiz) |
| **Hafta 3** | Llama-3.1-8B üzerinde test koşusu | Discrete vs Soft-MoA ablasyon tablosu | OEDB 100 senaryo raporu hazırlanması |
| **Hafta 4** | arXiv v2.1 submission & PDF derleme | Makale taslağının yazımı | IEEE formatına dönüştürme |

---

## Sonuç

> **Bu 3 makalelik strateji sayesinde:**
> 1. İddialar ve kod arasında hiçbir tutarsızlık kalmaz; hakemlerin veya Prof. Dr. İrem Ersöz Kaya'nın eleştireceği "dead code" veya "abartılı fizik teorisi" riski sıfırlanır.
> 2. Projenin tek ve riskli bir zar atışı olmak yerine, **farklı akademik mecralara hitap eden 3 güçlü kurşunu** olur.
> 3. En hazır ve en orijinal çalışma olan **SDPO (Makale 1)**, doğrudan arXiv ve top-tier workshop'lara sunulabilecek kalitede somut bir amiral gemisine dönüşmüştür.
