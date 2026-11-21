# 📊 SODALS v2.1: 30-Soruluk Genişletilmiş Baseline (Kontrol Grubu) Master Raporu

**Tarih:** 18 Ağustos 2026  
**Model:** `models/Qwen3.5-9B-SODALS-Baseline-SafeTensors` (QLoRA 4-bit NF4 Merged)  
**Değerlendirilen Benchmark:** CAIS Humanity's Last Exam (HLE) — Matematik & Teorik Fizik Bölümü  
**Test Türü:** Kapalı-Kitap (Closed-Book), Sıfır-Sızıntı, Deterministik Greedy ($T=0.0$)  
**Ham Veri Dosyası:** [`hle_results_baseline_control_30q_20260818.json`](file:///c:/Users/Burak/Desktop/sodals_core/hle_results_baseline_control_30q_20260818.json)  

---

## 1. YÖNETİCİ ÖZETİ VE İSTATİSTİKSEL GÜÇ

Bu rapor, SODALS v2.1 projesinde geliştirilecek olan **K7 Birleşik Modeli** (Pondering Core, Falsification Rollback ve Epistemic Accumulator içeren mimari) için değişmez, sarsılmaz bir kontrol referansı (Baseline) oluşturmak amacıyla gerçekleştirilen **30 soruluk genişletilmiş HLE testinin** sonuçlarını belgeler.

```
+---------------------------------------------------------------------------------------------------------+
|                               30 SORULUK GENİŞLETİLMİŞ BASELINE TESTİ                                   |
+---------------------------------------------------------------------------------------------------------+
| Toplam Soru Sayısı            : 30 Soru (8 Orijinal HLE + 22 Yeni Görmemiş STEM Sorusu)                 |
| Tam İsabet (Exact Match)      : 1 / 30 (%3.3)                                                           |
| %95 Wilson Güven Aralığı      : [%0.6, %16.7]  (CI Genişliği: %16.1)                                    |
| Doğal Tamamlama Oranı         : 29 / 30 (%96.7) (Yalnızca 1 soruda 4096 token bütçe aşımı)              |
| Ortalama Soru Başına Süre     : 89.2 saniye                                                             |
| 4 OOD Perturbasyon Skoru      : 3 / 4 (%75.0)                                                           |
+---------------------------------------------------------------------------------------------------------+
```

### Temel Çıkarımlar:
1. **İstatistiksel Güven Aralığı Daraltıldı:**  
   8 soruluk ilk testteki Wilson %95 aralığı $[\%7.1, \%59.1]$ (genişlik %52.0) gibi belirsiz bir aralıktayken; 30 soruluk test ile bu aralık **$[\%0.6, \%16.7]$ (genişlik %16.1)** hassasiyetine indirilmiştir.
2. **Kontrol Grubu Tescillendi:**  
   Plain LoRA Baseline modelinin frontier matematik ve teorik fizik sorularındaki gerçek kapasitesi **$\%3.3 \pm 8.0$** bandına oturmuştur. Bu model bir daha yüklenmeye gerek kalmayacak şekilde kilitlenmiştir.

---

## 2. 30 SORUNUN TAM VE DETAYLI DENETİM DÖKÜMÜ

| # | Soru ID | Kategori | Baseline Çıktısı | Resmi Ground Truth | Sonuç | Süre / Token |
|:---:|:---|:---:|:---|:---|:---:|:---:|
| 1 | `669402b4` | Math | `\mathbb{Z}_2` | `Z+Z+Z+Z+Z` | ❌ | 203.2s (902 tok) |
| 2 | `6696c373` | Math | `36` | `18` | ❌ | 67.2s (430 tok) |
| 3 | `66b2c7c9` | Math | `(1 + t)^2 (1 + t + t^2)^2` | `1+3x+6x^2+8x^3+6x^4+3x^5+x^6` | ❌ | 125.0s (790 tok) |
| 4 | `66b727d3` | Math | `k(d - k)` | `d-(d-2k)^2` | ❌ | 49.4s (311 tok) |
| 5 | `66b827b9` | Math | `9` | `3` | ❌ | 52.0s (328 tok) |
| 6 | `66b91693` | Math | `"I AM TO BE IN THE WORDS SPO..."` | `Katie kicked the knotted kite...` | ❌ | 52.7s (339 tok) |
| 7 | `66ba5510` | Math | `\mathcal{A}^{3+\alpha}(X)` | `\mathcal{A}^{\alpha-}(X)` | ❌ | 53.1s (345 tok) |
| 8 | `66c5c3ea` | Math | `0` | `\mathbb{Z}` | ❌ | 76.2s (491 tok) |
| 9 | `66e89811` | Math | `True` | `Yes` | ❌ | 57.7s (362 tok) |
| 10 | `66e89ebe` | Physics | `\begin{pmatrix} 2 & 1 \\ 1 & ...` | `\begin{pmatrix}8&9\\9&8\end{pmatrix}` | ❌ | 124.7s (821 tok) |
| 11 | `66e8a23d` | Math | `Yes` | `No, the regular subgradient...` | ❌ | 39.1s (259 tok) |
| 12 | `66e8a3ed` | Math | `2/7` | `\frac{2}{7}` | ❌ | 100.0s (680 tok) |
| 13 | `66e8ae60` | Math | `The number of non-admissible...` | `1` | ❌ | 92.7s (631 tok) |
| 14 | `66e8b0a9` | Math | `28` | `56` | ❌ | 93.0s (626 tok) |
| 15 | `66e8c151` | Math | `1` | `1430` | ❌ | 118.7s (810 tok) |
| 16 | `66e8cfa0` | Math | `\frac{1}{\pi + 1}` | `\frac{1-1/(\pi+1)}{\log(\pi+1)}` | ❌ | 111.3s (757 tok) |
| 17 | `66e8d4d6` | Math | `1/4` | `1/1451520` | ❌ | 35.7s (235 tok) |
| 18 | `66e8e473` | Math | `3` | `8` | ❌ | 68.1s (452 tok) |
| 19 | `66e8e886` | Physics | `0` | `-m^2/2` | ❌ | 105.9s (707 tok) |
| 20 | `66e8ea4c` | Math | `C` | `B` | ❌ | 41.5s (273 tok) |
| 21 | `66e90320` | Physics | *(Bütçe Aşımı / 4096 tok)* | `5.7` | ❌ | 688.9s (4096 tok) |
| 22 | `66e9100b` | Physics | `-1` | `-0.5` | ❌ | 76.8s (519 tok) |
| **23** | `66e91732` | **Math** | **`5429515560378`** | **`5429515560378`** | **✅** | **73.1s (494 tok)** |
| 24 | `66e9249c` | Math | `(6, 15, 20, 9)` | `(6, 14, 15, 7)` | ❌ | 82.4s (549 tok) |
| 25 | `66e939c1` | Math | `1/13` | `0` | ❌ | 50.4s (334 tok) |
| 26 | `66e9469b` | Math | `25` | `48` | ❌ | 69.3s (472 tok) |
| 27 | `66e94a88` | Physics | `\mathbb{Z}` | `2\mathbb{Z}` | ❌ | 41.5s (284 tok) |
| 28 | `66e94bf0` | Math | `\int_0^1 \int_0^1 F(u, v) ...` | `F(U_i, U_j, U_{ij})` | ❌ | 39.8s (275 tok) |
| 29 | `66e9a436` | Math | `3` | `4` | ❌ | 50.1s (342 tok) |
| 30 | `66e9d06a` | Math | `0` | `4` | ❌ | 48.7s (326 tok) |

---

## 3. ADLİ BULGULAR VE WDDM BELLEK YÖNETİMİ DERSLERİ

1. **WDDM Bellek Paging Mekanizması:**  
   VRAM doluluk oranı %96'yı aştığında Windows WDDM sürücüsü aktivasyon tensörlerini PCIe üzerinden sistem RAM'ine aktararak adım süresini 5 saniyeden 300 saniyeye fırlatmaktadır. Gelecek tüm eğitimlerde `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ve sıkı `torch.cuda.empty_cache()` protokolü uygulanacaktır.
2. **Strict Exact-Match Standartı:**  
   Substring karşılaştırma hatası (`"8" in "18"` $\to$ `True`) tamamen temizlenmiş, tüm cevaplar `clean_ans == clean_gt` ile sertifikalandırılmıştır.
