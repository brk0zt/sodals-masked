# SODALS v2.1: Temiz 125 Held-Out MATH-500 Doğrulama ve Ayrıştırma Raporu

**Tarih:** 14 Ekim 2026  
**Denetim Kapsamı:** GRPO Eğitim Promptlarının (25Q) Test Kümesinden Ayrıştırılması ve 125 Temiz Held-Out Soruda Bağımsız İstatistiksel Değerlendirme  
**Ayrıştırılmış Veri Dosyaları:**
- **Temiz Test Kümesi (Held-Out 125Q):** [`sft/math500_heldout_clean_125_dataset.json`](file:///c:/Users/Burak/Desktop/sodals_core/sft/math500_heldout_clean_125_dataset.json)
- **Eğitim Seti Rekonstrüksiyonu (Train 25Q):** [`sft/math500_training_pilot_25_dataset.json`](file:///c:/Users/Burak/Desktop/sodals_core/sft/math500_training_pilot_25_dataset.json)
- **Temiz Test Sonuç JSON:** [`sft/math500_heldout_clean_125_results.json`](file:///c:/Users/Burak/Desktop/sodals_core/sft/math500_heldout_clean_125_results.json)
- **Eğitim Seti Sonuç JSON:** [`sft/math500_training_pilot_25_results.json`](file:///c:/Users/Burak/Desktop/sodals_core/sft/math500_training_pilot_25_results.json)

---

## 1. YÖNETİCİ ÖZETİ VE HAKEM PERSPEKTİFİ

| Model / Mimari | Temiz Held-Out (125Q) | %95 Wilson Güven Aralığı | Net Kazanç | McNemar Testi | Karar |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Frozen SFT Baseline (`balanced_r4`)** | **99 / 125 (%79.20)** | [%71.27, %85.39] | — | — | Taban Doğruluğu |
| **Long-Context GRPO ($L=1536$)** | **100 / 125 (%80.00)** | [%72.14, %86.07] | **+1 Soru** | **$p = 1.0000$** | **İstatistiki Olarak Anlamsız (Non-Significant)** |
| **Prefill Pondering ODE (RK4)** | **100 / 125 (%80.00)** | [%72.14, %86.07] | **0 Soru** | **$p = 1.0000$** | **Tam Null Sonuç (0 Kazanç, 0 Kayıp)** |

> [!CAUTION]
> **Akademik Dürüstlük Kuralı:**  
> Temiz 125 soruluk test kümesinde elde edilen **99 -> 100 net değişimi**, eşli McNemar testinde **p = 1.0000** değerine sahiptir ve güven aralıkları tamamen örtüşmektedir. Bu nedenle *"GRPO ile SFT tavanı aşıldı"* veya *"Süper-insan akıl yürütme elde edildi"* iddiası ampirik olarak **savunulamaz**.  
> **Savunulabilir Gerçek Katkı:**  
> (1) 8GB VRAM sınırında sıfır WDDM paging ile 1536-token çıkarım altyapısının kurulabilmesi;  
> (2) 125 sorunun tamamında **0 regresyon (sıfır unutma)** ile politika stabilitesinin korunması;  
> (3) Q105 sorusunda 1,023 tokenlık otonom CoT ile çözüme ulaşılmasıdır.

---

## 2. EŞLİ KONTENJANS MATRİSLERİ (PAIRED CONTINGENCY MATRICES)

### A. Frozen SFT Baseline vs. Long-Context GRPO (N = 125)

```
                       GRPO PASS        GRPO FAIL        Toplam
    Baseline PASS      99               0 (Drop)         99
    Baseline FAIL      1 (Gain/Q105)    25               26
    Toplam             100              25               125
```

- **Uyuşumsuz Çiftler (Discordant Pairs):** b = 0 (Kayıp), c = 1 (Kazanç).
- **Exact Binomial İki Yönlü p-Değeri:** p = 1.0000.
- **10,000 İterasyonlu Bootstrap Güven Aralığı (Delta = Acc_GRPO - Acc_SFT):** [%0.00, %2.40] (Sıfır noktasını içeriyor).

### B. Long-Context GRPO vs. Prefill Pondering ODE (N = 125)

```
                       Pondering PASS   Pondering FAIL   Toplam
    GRPO PASS          100              0                100
    GRPO FAIL          0                25               25
    Toplam             100              25               125
```

- **Uyuşumsuz Çiftler:** 0 kazanç, 0 kayıp.  
- **Sonuç:** Pondering ODE, greedy modda 125 sorunun tamamında GRPO ile birebir aynı kararları üretmiştir (**Null Result / İnvaryans Operatörü**).

---

## 3. KONU VE ZORLUK SEVİYESİ DAĞILIMI (TEMİZ 125 HELD-OUT)

### Konu Dağılımı:
| Konu (Subject) | Soru Sayısı | Baseline SFT | Long GRPO | Pondering ODE | Net Fark |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Algebra** | 29 | 28 / 29 (%96.6) | 28 / 29 (%96.6) | 28 / 29 (%96.6) | 0 |
| **Counting & Probability** | 10 | 9 / 10 (%90.0) | 9 / 10 (%90.0) | 9 / 10 (%90.0) | 0 |
| **Geometry** | 8 | 6 / 8 (%75.0) | 6 / 8 (%75.0) | 6 / 8 (%75.0) | 0 |
| **Intermediate Algebra** | 30 | 20 / 30 (%66.7) | **21 / 30 (%70.0)** | **21 / 30 (%70.0)** | **+1 (Q105)** |
| **Number Theory** | 14 | 9 / 14 (%64.3) | 9 / 14 (%64.3) | 9 / 14 (%64.3) | 0 |
| **Prealgebra** | 22 | 17 / 22 (%77.3) | 17 / 22 (%77.3) | 17 / 22 (%77.3) | 0 |
| **Precalculus** | 12 | 10 / 12 (%83.3) | 10 / 12 (%83.3) | 10 / 12 (%83.3) | 0 |

### Zorluk Seviyesi Dağılımı:
| Seviye (Level) | Soru Sayısı | Baseline SFT | Long GRPO | Pondering ODE |
|:---|:---:|:---:|:---:|:---:|
| **Level 1** | 12 | 12 / 12 (%100.0) | 12 / 12 (%100.0) | 12 / 12 (%100.0) |
| **Level 2** | 24 | 22 / 24 (%91.7) | 22 / 24 (%91.7) | 22 / 24 (%91.7) |
| **Level 3** | 29 | 24 / 29 (%82.8) | 24 / 29 (%82.8) | 24 / 29 (%82.8) |
| **Level 4** | 31 | 24 / 31 (%77.4) | 24 / 31 (%77.4) | 24 / 31 (%77.4) |
| **Level 5** | 29 | 17 / 29 (%58.6) | **18 / 29 (%62.1)** | **18 / 29 (%62.1)** |

---

## 4. EĞİTİM SETİ REKONSTRÜKSİYONU (TRAIN 25Q SANITY CHECK)

Eğitim sırasında kullanılan 25 prompt, eğitim sonrasında modelin kendi eğitim verisi üzerindeki davranışını ve ezberleme/unutma dengesini denetlemek için ayrı bir kontrol seti olarak değerlendirilmiştir:

- **Örneklem:** 25 soru
- **Baseline SFT Skoru:** 23 / 25 (%92.00)
- **Long-Context GRPO Skoru:** 23 / 25 (%92.00)
- **Pondering ODE Skoru:** 23 / 25 (%92.00)
- **Net Delta:** 0 (Katastrofik unutma yaşanmamıştır, eğitim promptlarında %100 istikrar korunmuştur).

---

## 5. SONUÇ VE BİLİMSEL PERSPEKTİF

Bu ayrıştırma ile:
1. **Veri Sızıntısı (Contamination) Sıfırlandı:** 25 eğitim sorusu testten tamamen çıkarıldı; 125 held-out soru bağımsız bir dosya olarak izole edildi.
2. **Hakem İtirazları Önceden Yanıtlandı:** $p = 1.0000$ değeri gizlenmek yerine doğrudan ana tabloda şeffaf bir dille sunuldu.
3. **Mühendislik Başarısı Netleşti:** Tüketici sınıfı 8GB GPU üzerinde WDDM paging olmaksızın çalışan uzun bağlamlı RL altyapısı, 0 regresyon başarısıyla birlikte sağlam bir zemine oturtuldu.
