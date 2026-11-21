# SODALS v2.1 — Adım 3 (Modül 1, 2, 3 ve 4) Tamamlanmış Deney ve Temiz 125 Held-Out Adli Denetim Raporu

**Tarih:** 19 Ekim 2026  
**Değerlendirme Protokolü:** Sıfır Veri Sızıntısı (25 GRPO Eğitim Sorusu tamamen ayrıştırılmış, **Temiz 125 Held-Out** test kümesi izole edilmiş) ve Sıfır Değerlendirici Kayması (*Zero Grader Drift* — tüm 6 aşama hem **Legacy Baseline Grader** hem de **Normalized Symbolic Grader** ile birebir aynı kriterlerle puanlanmıştır).

---

## 1. Yönetici Özeti: Adım 3 (Modül 1, 2, 3 ve 4) Gerçekte Ne Üretti?

Adım 3 kapsamında test edilen dört çıkarım-zamanı (*test-time*) modülünün tamamı (**150/150 soru = 25 Eğitim Pilot + 125 Temiz Held-Out**) RTX 4060 Laptop GPU (8 GB VRAM, 4-bit NF4) üzerinde tamamlanmış ve tüm modeller **aynı değerlendirici (grader)** altında çapraz denetime tabi tutulmuştur.

Denetim sonucunda **dört kritik bilimsel bulgu** tespit edilmiştir:

1. **Değerlendirici Kayması (*Grader Drift*) Yanılsamasının Tespiti ve Düzeltilmesi:**
   - Modül 2 (`eval_grpo_long_cst_prm_150.py`), Modül 3 (`eval_grpo_long_tuned_lens_150.py`) ve Modül 4 (`eval_grpo_long_best_of_g_prm_150.py`) betiklerinde devreye alınan `NormalizedMathMatcher`, LaTeX formatlama farklarını (`11\sqrt{2}` vs `11\sqrt2`, `-\pi/6` vs `-\frac{\pi}{6}`, `East` vs `\text{east}`, `4210_5` vs `4210_{5}`) doğru kabul etmektedir.
   - Ancak bu **4 soruda (Q032, Q054, Q098, Q128 — tamamı Temiz 125 Held-Out kümesinde)** SFT Baseline, GRPO Long ve Modül 1 (Pondering ODE) modelleri de **zaten birebir aynı doğru cevabı üretmiştir!**
   - Tüm 6 aşama **aynı** değerlendirici ile puanlandığında, Modül 2 ve Modül 3'te görünen `+3 / +4` puanlık artışın **modelin yeni soru çözmesinden değil, değerlendiricinin LaTeX format normalizasyonundan** kaynaklandığı kanıtlanmıştır.

2. **Modül 2 (Saf CST Sembolik PRM Kapısı) ile Modül 3 (Tuned Lens + Rafine Tri-Modal PRM Kapısı) Farkı:**
   - **Modül 2 (`eval_grpo_long_cst_prm_150_results.json`):** Saf regex tabanlı sembolik doğrulayıcı, **Q055 (Temiz 125 Held-Out)** sorusunda modelin doğru ürettiği `\frac{3}{20} = \frac{15}{100}` adımını hatalı ayrıştırarak sahte bir çelişki (`0.15 != 3.0`) üretmiş ve doğru olan `0.15` (`PASS`) cevabını ikinci geçişte `0` (`FAIL`) cevabıyla ezerek **1 soruluk gerilemeye (regression)** yol açmıştır (`100/125 -> 99/125` Legacy; `104/125 -> 103/125` Normalized Symbolic).
   - **Modül 3 (`eval_grpo_long_tuned_lens_150_results.json`):** Rafine edilmiş CST ayrıştırıcısı ve **Tuned Lens (Katman 28 & 30 $\to$ Katman 31 afin projeksiyonu) + Tri-Modal PRM** birleşimi, **Q055** üzerindeki sahte pozitif tetiklemeyi ortadan kaldırmış (`0.15` `PASS` korunmuş) ve Temiz 125 Held-Out kümesinde tetiklenen **9 sorunun tamamında** doğru cevapları %100 koruyarak **sıfır gerileme (`0 drop`)** sağlamıştır (`100/125` Legacy; `104/125` Normalized Symbolic).

3. **Greedy ($T=0$) Çıkarımda "Özgüvenli Yanılgı" (*Confident Hallucination*) Bulgusu:**
   - Temiz 125 Held-Out kümesinde Katman 28 ve Katman 30 Tuned Lens projeksiyonlarının Katman 31 ile ortalama kosinüs kararlılığı (`primary_mean_stability`) **doğru çözülen sorularda ($n=104$) $0.7756$**, **yanlış çözülen sorularda ($n=21$) ise $0.7867$** olarak ölçülmüştür (minimum kararlılık: $0.7266$, Top-1 kelime dağarcığı uzlaşısı: $124/125 = \%99.20$).
   - Bu bulgu, 9B parametreli modelin deterministik ($T=0$) çıkarımda hata yaparken içsel katmanlar arasında kararsızlık yaşamadığını, aksine **yüksek katmanlar-arası geometrik uzlaşı ile özgüvenli biçimde yanlış türetim yaptığını** deneysel olarak kanıtlamaktadır. Dolayısıyla tek bir greedy yeniden deneme (*single greedy retry*) yeni soru kazandıramamakta (`0 gate gain`); gerçek doğruluk artışı için çeşitlendirilmiş örnekleme ($G > 1$ aday üretimi + PRM seçimi) gerekmektedir.

4. **Modül 4 (Hedefli Best-of-$G$ [$G=4$] Nöro-Sembolik Rollout & Tri-Modal PRM Seçimi) ile Gerçek Sızıntısız Held-Out Kazanımı (`+2 Net Soru`):**
   - Yalnızca yapısal kusur barındıran (**15 / 150 soru**: `</think>` kesilmesi, SymPy CST cebirsel/taban aritmetiği çelişkisi veya $\ge 2$ öz-şüphe belirteci) yörüngelerde devreye giren **Kör Tetikleyici (Blind Trigger)**, $G=4$ paralel dal (`1` hedefli deterministik doğrulama + `3` stokastik $T=0.6$ dal) üretmiş ve adayları **Tri-Modal PRM (`SymPy CST + Tuned Lens + Öz-Uzlaşı`)** ile puanlamıştır.
   - **Temiz 125 Held-Out** kümesinde **sıfır gerileme (`0 regression`)** ile **2 gerçek yeni soru (`Q120` ve `Q138`)** çözülmüştür:
     - **`Q120` (`HELDOUT`, `GT = -9`):** Tek geçişli Greedy Pass-1 (`pred = '2'`) 2048 tokenda tekrarlama döngüsüne girip kesilmişken (`end=False, valid=False`), `Best-of-4` dallarından `Cand4_Sample_S3 = '-9'` (`11` doğrulanmış denklem, `0` çelişki, PRM skoru `+35.63`) seçilerek **`FAIL -> PASS`** dönüşümü sağlanmıştır.
     - **`Q138` (`HELDOUT`, `GT = 204_5`):** Tek geçişli Greedy Pass-1 (`pred = '224_5'`) ve stokastik dalların ikisi (`Cand3 = '224_5'`, `Cand4 = '224_5'`) yanlış cevabı çoğunluk oyuyla (3/5 oy) tekrar etmesine rağmen, **SymPy taban-$n$ (`_eval_base_n`) sembolik doğrulayıcımız** `224_5` ve `244_5` dallarındaki taban-5 aritmetik çelişkilerini yakalayıp (`-16.7`, `-16.9`, `-1.5` ceza puanı), **tek cebirsel tutarlı dal olan `Cand2_Sample_S1 = '204_5'` (`+27.4` PRM skoru)** adayını seçerek **çoğunluk oylamasını (Majority Voting) mağlup etmiş ve `FAIL -> PASS`** dönüşümü sağlamıştır!
   - Böylece **Temiz 125 Held-Out** başarısı:
     - **Legacy Grader:** `99/125 (%79.20 SFT) -> 100/125 (%80.00 GRPO) -> 102/125 (%81.60 Modül 4 Best-of-G PRM)` (**SFT'ye göre +3 soru / +%2.40**)
     - **Normalized Symbolic Grader:** `103/125 (%82.40 SFT) -> 104/125 (%83.20 GRPO) -> 106/125 (%84.80 Modül 4 Best-of-G PRM)` (**SFT'ye göre +3 soru / +%2.40**) seviyesine ulaşmıştır!

---

## 2. Birleştirilmiş Değerlendirici (Unified Grader) Tablosu — Tüm 6 Aşama

Aşağıdaki tabloda tüm 6 aşama, **birebir aynı iki değerlendirici** ile hem **Temiz 125 Held-Out (`index 26..150`)** hem **25 Eğitim Pilot (`index 1..25`)** hem de **Birleşik 150 (`index 1..150`)** üzerinde raporlanmıştır:

| Aşama / Model | Temiz 125 Held-Out (Legacy Grader) | Temiz 125 Held-Out (Normalized Symbolic) | 25 Eğitim Pilot (Legacy / NormSym) | Birleşik 150 (Legacy / NormSym) | SFT Baseline'a Göre Net Fark (Temiz 125) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0. SFT Soft-Blended Baseline** | **99 / 125 (%79.20)** | **103 / 125 (%82.40)** | 23 / 25 (%92.00) | 122/150 (%81.33) / 126/150 (%84.00) | Referans |
| **1. 1536-Token GRPO Long** | **100 / 125 (%80.00)** | **104 / 125 (%83.20)** | 23 / 25 (%92.00) | 123/150 (%82.00) / 127/150 (%84.67) | **+1 Soru (+%0.80)** (`Q105` çözüldü) |
| **2. Adım 3 — Modül 1 (Pondering ODE)** | **100 / 125 (%80.00)** | **104 / 125 (%83.20)** | 23 / 25 (%92.00) | 123/150 (%82.00) / 127/150 (%84.67) | **+1 Soru (+%0.80)** (GRPO ile %100 aynı) |
| **3. Adım 3 — Modül 2 (Saf CST PRM Kapısı, $G=1$)** | **99 / 125 (%79.20)** | **103 / 125 (%82.40)** | 23 / 25 (%92.00) | 122/150 (%81.33) / 126/150 (%84.00) | **0 Soru (%0.00)** (`Q055` gerilemesi: -1) |
| **4. Adım 3 — Modül 3 (Tuned Lens + PRM, $G=1$)** | **100 / 125 (%80.00)** | **104 / 125 (%83.20)** | 23 / 25 (%92.00) | 123/150 (%82.00) / 127/150 (%84.67) | **+1 Soru (+%0.80)** (`Q055` kurtarıldı, 0 kayıp) |
| **5. Adım 3 — Modül 4 (Hedefli Best-of-$G$ [$G=4$] PRM)** | **102 / 125 (%81.60)** | **106 / 125 (%84.80)** | 23 / 25 (%92.00) | 125/150 (%83.33) / 129/150 (%86.00) | **+3 Soru (+%2.40)** (`Q105, Q120, Q138` çözüldü, 0 kayıp) |

---

## 3. Değerlendirici Normalizasyonunun Etkilediği 4 Soru (Temiz 125 Held-Out)

Aşağıdaki 4 soruda **tüm 6 model de (SFT, GRPO, Modül 1, Modül 2, Modül 3, Modül 4) ilk geçişte (Pass 1) matematiksel olarak doğru cevabı üretmiştir**. Eski `check_math_match` (Legacy Grader) bu 4 cevabı salt yazım biçimi yüzünden `FAIL` sayarken, `NormalizedMathMatcher` (`NormSym`) doğru olarak `PASS` kabul etmektedir:

| Soru No | Küme | Konu | Model Tahmini (Tüm 6 Modelde Aynı) | Ground Truth (`MATH-500`) | Legacy Grader | Normalized Symbolic Grader | Normalizasyon Nedeni |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Q032** | `CLEAN-125` | Prealgebra | `11\sqrt{2}` | `11\sqrt2` | `FAIL` | `PASS` | Köklü ifadede süslü parantez normalizasyonu (`\sqrt2` $\leftrightarrow$ `\sqrt{2}`) |
| **Q054** | `CLEAN-125` | Precalculus | `-\pi/6` | `-\frac{\pi}{6}` | `FAIL` | `PASS` | SymPy dönüşümünde `\pi` $\to$ `pi` sembolik eşdeğerliği |
| **Q098** | `CLEAN-125` | Prealgebra | `East` | `\text{east}` | `FAIL` | `PASS` | Büyük/küçük harf duyarsız (`case-insensitive`) metin eşleşmesi |
| **Q128** | `CLEAN-125` | Number Theory | `4210_5` | `4210_{5}` | `FAIL` | `PASS` | Taban gösteriminde alt-indis parantez normalizasyonu (`_{5}` $\leftrightarrow$ `_5`) |

> **Metodolojik Not:** Makalede değerlendirici değişikliğinden kaynaklanan bu +4 soruluk fark **asla** algoritmik kazanım olarak sunulmamalıdır. Tüm tablolar aynı değerlendirici altında hizalanmıştır.

---

## 4. Tüm 150 Soru Boyunca Aşamalar Arasında Doğruluğu Değişen Soruların Özeti

| Soru No | Küme | Ground Truth | 0. SFT Baseline | 1. GRPO Long | 2. Modül 1 (Pondering) | 3. Modül 2 (CST, $G=1$) | 4. Modül 3 (Lens, $G=1$) | 5. Modül 4 (Best-of-$G=4$ PRM) | Net Durum Açıklaması |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Q055** | `CLEAN-125` | `0.15` | `0.15` (**PASS**) | `0.15` (**PASS**) | `0.15` (**PASS**) | `0` (**FAIL** ✗) | `0.15` (**PASS** ★) | `0.15` (**PASS** ★) | Modül 2 sahte çelişkiyle bozdu; **Modül 3 & 4 kurtardı** |
| **Q105** | `CLEAN-125` | `40` | `18` (**FAIL**) | `40` (**PASS** ★) | `40` (**PASS** ★) | `40` (**PASS** ★) | `40` (**PASS** ★) | `40` (**PASS** ★) | **GRPO Long'un Held-Out kazanımı (+1)** |
| **Q120** | `CLEAN-125` | `-9` | `4` (**FAIL**) | `2` (**FAIL**) | `2` (**FAIL**) | `13` (**FAIL**) | `13` (**FAIL**) | **`-9` (PASS ★★)** | **Modül 4 Best-of-$G$ PRM kazanımı (+1)** (`Cand4_Sample_S3` `+35.6`) |
| **Q138** | `CLEAN-125` | `204_5` | `224_5` (**FAIL**) | `224_5` (**FAIL**) | `224_5` (**FAIL**) | `244_5` (**FAIL**) | `244_5` (**FAIL**) | **`204_5` (PASS ★★)** | **Modül 4 Best-of-$G$ PRM kazanımı (+1)** (`Cand2_Sample_S1` `+27.4`, çoğunluk oyunu yendi!) |

---

## 5. Modül 4 (`Best-of-G [G=4] Neurosymbolic PRM`) Kör Tetiklenen 15 Sorunun Tam Tablosu

Modül 4'te kör tetikleme kuralı (`not has_end1` veya `not v1 and len(contras1)>0` veya `wait1 >= 2`) 150 soruda toplam **15 soruda** (2'si `TRAIN-25`, 13'ü `CLEAN-125`) ateşlenmiştir:

| Soru No | Küme | Ground Truth | Pass 1 Tahmin (`end, valid, wait`) | Üretilen $G=4$ Adayları (`Tahmin (PRM Skoru)`) | PRM Tarafından Seçilen Dal | Sonuç Değişimi |
| :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| **Q012** | `TRAIN-25` | `\frac{3}{56}` | `2/15` (`T, T, 2`) | `['1/9(19.8)', '1/9(25.6)', '1/9(23.6)', '109/882(17.7)']` | `Cand2_Sample_S1 = '1/9'` | `FAIL -> FAIL` |
| **Q020** | `TRAIN-25` | `3` | `3` (`T, T, 2`) | `['3(15.5)', '3(15.6)', '3(17.5)', '3(17.4)']` | `Cand3_Sample_S2 = '3'` | **`PASS -> PASS` (Korundu)** |
| **Q026** | `CLEAN-125` | `1,-2` | `0, 5` (`T, T, 2`) | `['-1, 2(19.6)', '0, 1(21.6)', '0, 5(21.6)', '0, 1(23.5)']` | `Cand4_Sample_S3 = '0, 1'` | `FAIL -> FAIL` |
| **Q027** | `CLEAN-125` | `144` | `2` (`F, T, 5`) | `['1008(25.2)', '432(31.2)', '216(23.2)', '504(35.4)']` | `Cand4_Sample_S3 = '504'` | `FAIL -> FAIL` |
| **Q044** | `CLEAN-125` | `70\sqrt{2}` | `80` (`F, T, 3`) | `['100(29.4)', '30(-100.0)', '0(-100.0)', '80(-42.3)']` | `Cand1_Targeted_T0 = '100'` | `FAIL -> FAIL` |
| **Q063** | `CLEAN-125` | `4` | `4` (`T, F, 0`) | `['4(-5.6)', '4(-22.5)', '4(-7.4)', '4(19.4)']` | `Pass1_Greedy = '4'` | **`PASS -> PASS` (Korundu)** |
| **Q087** | `CLEAN-125` | `240` | `240` (`T, F, 0`) | `['240(19.3)', '240(19.1)', '240(-24.9)', '240(-24.9)']` | `Pass1_Greedy = '240'` | **`PASS -> PASS` (Korundu)** |
| **Q097** | `CLEAN-125` | `1\pm\sqrt{19}` | `x\in\mathbb{R}\setminus...` (`T, T, 2`) | `['14(11.6)', '1, 5(15.6)', '5, -1(15.6)', '41(-100.0)']` | `Pass1_Greedy` | `FAIL -> FAIL` |
| **Q101** | `CLEAN-125` | `145^\circ` | `115` (`T, T, 2`) | `['75(-48.9)', '95(15.3)', '125(11.4)', '140(15.1)']` | `Cand2_Sample_S1 = '95'` | `FAIL -> FAIL` |
| **Q109** | `CLEAN-125` | `2` | `2` (`T, T, 2`) | `['2(17.3)', '2(17.2)', '2(17.4)', '4/3(15.3)']` | `Cand3_Sample_S2 = '2'` | **`PASS -> PASS` (Korundu)** |
| **Q120** | `CLEAN-125` | `-9` | `2` (`F, F, 0`) | `['1(-100.0)', '-9(10.7)', '-9(-4.1)', '-9(35.6)']` | **`Cand4_Sample_S3 = '-9'`** | **`FAIL -> PASS` (★ KAZANIM)** |
| **Q121** | `CLEAN-125` | `\text{even}` | `odd` (`T, T, 2`) | `['odd(15.2)', 'odd(13.4)', 'odd(11.5)', 'odd(15.2)']` | `Cand1_Targeted_T0 = 'odd'` | `FAIL -> FAIL` |
| **Q124** | `CLEAN-125` | `8` | `10` (`T, F, 2`) | `['12(25.3)', '8(-1.5)', '8(-1.8)', '7(4.2)']` | `Cand1_Targeted_T0 = '12'` | `FAIL -> FAIL` |
| **Q127** | `CLEAN-125` | `1+274i` | `2` (`F, T, 4`) | `['(1+265i)/(1+263i)(17.7)', '(1+270i)/(1-268i)(17.6)', '1(21.5)', '274137(-100.0)']` | `Cand3_Sample_S2 = '1'` | `FAIL -> FAIL` |
| **Q138** | `CLEAN-125` | `204_5` | `224_5` (`T, F, 0`) | `['244_5(-1.5)', '204_5(27.4)', '224_5(-16.7)', '224_5(-16.9)']` | **`Cand2_Sample_S1 = '204_5'`** | **`FAIL -> PASS` (★ KAZANIM)** |

### Temiz 125 Held-Out Tuned Lens İstatistiksel Özeti
- **Ortalama Katmanlar-Arası Geometrik Kararlılık (`primary_mean_stability`, Katman 28 & 30 $\to$ 31):** $0.7775 \pm 0.0191$ ($\min = 0.7266, \max = 0.8452$)
- **Son Token Cevap Kararlılığı (`primary_answer_stability`):** $0.6705 \pm 0.0282$
- **Top-1 Kelime Dağarcığı Uzlaşısı (`primary_top1_agreement`):** $124 / 125$ (**%99.20**)
- **PASS vs. FAIL Ortalama Kararlılık (Greedy Pass-1):** PASS ($n=104$): $0.7756$ vs. FAIL ($n=21$): $0.7867$
- **Ortalama RK4 Adım Sayısı (`rk4_steps`):** $2.04$ adım

