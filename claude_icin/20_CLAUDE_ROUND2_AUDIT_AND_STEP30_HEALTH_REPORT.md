# SODALS v2.1 — Claude 2. Tur Metodolojik Denetim, 30. Adım Sağlık/Güvenlik Kapısı ve Modül 4 Çoğunluk Oylaması (Majority Vote) Karşılaştırma Raporu

**Tarih:** 21 Ekim 2026  
**Doğrulama Betiği:** `sft/audit_claude_round2_checks.py`  
**Ham Kanıt Dosyası (150 Soru Bayt-Bayt Karşılaştırma):** `sft/mod2_vs_grpo_150_byte_comparison_proof.json`  
**Eğitim Telemetrisi (30 Adım, 120 Rollout):** `sft/grpo_math_train_100_results.json`

---

## 1. Modül 2 vs. GRPO (147 / 150 Bayt-Bayt Aynı Çıktı) Ham Kanıt Dosyası

Claude'un talebi doğrultusunda, SFT Baseline (`sft_pred`), GRPO Long (`grpo_pred`) ve Modül 2 (`mod2_pred`) modellerinin **150 sorunun tamamındaki ham tahmin metinlerini ve 3 katmanlı notlayıcı (`strict`, `legacy`, `norm_sym`) çıktılarını** yan yana içeren bağımsız kanıt dosyası `sft/mod2_vs_grpo_150_byte_comparison_proof.json` olarak üretilmiştir:
- **150 sorunun 147'sinde (`%98.0`)** `grpo_pred == mod2_pred` bayt-bayt (`True`) aynıdır.
- Modül 2'nin doğru çözdüğü **126 sorunun 126'sında (`%100.0`)** `grpo_pred == mod2_pred` bayt-bayt (`True`) aynıdır.
- Farklılaşan yalnızca **3 soru vardır (`Q055`, `Q120`, `Q138`)**:
  - **`Q055` (`CLEAN-125`, `GT = '0.15'`):** `grpo_pred = '0.15'` (`PASS`) $\to$ `mod2_pred = '0'` (`FAIL` — `0 kazanç, 1 kayıp`).
  - **`Q120` (`CLEAN-125`, `GT = '-9'`):** `grpo_pred = '2'` (`FAIL`) $\to$ `mod2_pred = '13'` (`FAIL`).
  - **`Q138` (`CLEAN-125`, `GT = '204_5'`):** `grpo_pred = '224_5'` (`FAIL`) $\to$ `mod2_pred = '244_5'` (`FAIL`).

---

## 2. İlk 25 Soruluk Pilottaki "Türetim Israrı" İfadesinin Makaleden Çıkarılması

Claude'un mekanizma testi uyarısı doğrultusunda:
- İlk 25 soruluk GRPO pilotunda başarısız sorulardaki erken bitirme oranları (`<512 token`: `15/26 = %57.7`) değişmediği ve `Q105` (`844 -> 1023` token) tekil bir örnek olduğu için, `arxiv/arXiv_SODALS_v2.1_Master_Paper.md` ve `arxiv/paper1_master_reasoning.tex` metinlerinden ilk pilota atfedilen **"türetim ısrarı / deliberation persistence"** ifadeleri tamamen çıkarılmış; `Q105` yalnızca tekil bir vaka (`n01=1, n10=0, McNemar p=1.0000`) olarak çerçevelenmiştir.

---

## 3. Yeni GRPO Koşusunda Ödül Üst Sınırı (Reward Cap) ve 30. Adım Sağlık/Güvenlik Denetimi

Claude'un **2. ve 3. maddelerde** işaret ettiği *"Ödül üst sınırını ve ödülün doğruluk terimini aşıp aşamadığını şimdi kontrol edin, yüksek ödüllü yanlış rollout'lardan örnek bakın; 30. adım kapısını yalnızca zarar ve sağlık için kullanın"* kontrolü, tamamlanan **30 adımın (`120` rollout)** tamamı üzerinde çalıştırılmıştır:

### 3.1 Matematiksel Ödül Üst Sınırı ve Ayrışma Boşluğu (*Zero Reward Hacking Proof*)
`sft/train_grpo_math_train_100.py` içindeki `compute_reward` fonksiyonunda **SymPy denklem bonusu (`r_sympy`) zaten `min(0.40, 0.04 * eq_count)` ile katı bir üst sınıra (hard cap) sahiptir**:
- **Yanlış Bir Rollout'un Alabileceği Teorik Maksimum Ödül:**
  $$R_{\text{wrong, max}} = -1.00 \text{ (sonuç)} + 0.30 \text{ (format)} + 0.40 \text{ (SymPy üst sınırı, }\ge 10\text{ denklem)} + 0.20 \text{ (uzunluk)} = \mathbf{-0.1000}$$
- **Doğru Bir Rollout'un Alabileceği Teorik Minimum Ödül (`\boxed{}` ile):**
  $$R_{\text{correct, min}} = +2.00 \text{ (sonuç)} + 0.10 \text{ (kesilmiş box)} - 0.25 \text{ (çelişki)} + 0.00 = \mathbf{+1.8500} \quad (\text{veya box'sız } +1.2500)$$
- **Gerçekleşen 120 Rollout Dağılımı (`Steps 1..30`):**
  - **Doğru Rollout'lar (`67 / 120`):** Minimum = **`+1.4800`**, Maksimum = **`+2.8500`**
  - **Yanlış Rollout'lar (`53 / 120`):** Maksimum = **`-0.1000`**, Minimum = **`-1.7188`**
  - **Ampirik Ayrışma Boşluğu (*Separation Gap*):** En yüksek ödüllü yanlış rollout (`-0.1000`) ile en düşük ödüllü doğru rollout (`+1.4800`) arasında **`+1.5800` puanlık kesin bir boşluk** vardır. Çok sayıda doğru ara denklem yazan hiçbir yanlış rollout, doğru cevabın ödülüne yaklaşamamaktadır (`%0` örtüşme).

### 3.2 En Yüksek Ödüllü 5 Yanlış Rollout'un İncelemesi (`Steps 1..30`)
| Adım | Konu | Rollout Ödülü | Token Sayısı | Model Tahmini (`pred`) | Ground Truth (`gt`) | Neden `-0.10` / `-0.34` Aldı? |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Step 10** | Prealgebra | **`-0.1000`** | `593` | `'87653'` | `'98752'` | $7!$ çarpanlarını doğru açtı ($\ge 10$ geçerli denklem, `+0.40` tavan), ancak rakam kombinasyonunda hata yaptı (diğer 3 rollout `'98752'` ile `+1.97..+2.77` aldı). |
| **Step 01** | Prealgebra | **`-0.3400`** | `399` | `'90'` | `'100'` | 1-500 arası 9 rakamı sayımında yüzler basamağını eksik topladı (`+0.16` denklem bonusu; doğru 2 rollout `+2.53` ve `+2.57` aldı). |
| **Step 18** | Algebra | **`-0.3800`** | `432` | `'1.2'` | `'438'` | Ara cebirsel adımlar tutarlıydı (`+0.12` denklem bonusu), son yerine koymada hata yaptı. |
| **Step 26** | Interm. Algebra | **`-0.4200`** | `677` | `'393.5625'` | `'717'` | 2 denklem doğrulandı (`+0.08`), `</think>` ve `\boxed{}` tamamdı (`+0.30`). |
| **Step 05** | Prealgebra | **`-0.5000`** | `390` | `'39'` | `'52'` | Asal tarihleri aylara göre sayarken 0 denklem bonusu (`+0.00`), tam format (`+0.30`) ve uzunluk (`+0.20`) aldı. |

### 3.3 30. Adım Sağlık ve Güvenlik Kapısı (*Health & Safety Gate*) Özeti
- **Aktif Gradyan Oranı:** **`29 / 30 (%96.7)`** adımda aktif politika gradyanı (`std_r > 0`) üretilmiştir.
- **Çözülen Level-5 Eğitim Sorusu Oranı:** **`22 / 30 (%73.33)`** soruda en az 1 doğru rollout elde edilmiştir.
- **Uzunluk Kararlılığı (Patlama / Çökme Kontrolü):**
  - `Adım 1..15` ortalama rollout uzunluğu: **`421.6 token`** (medyan: `394.5`)
  - `Adım 16..30` ortalama rollout uzunluğu: **`502.1 token`** (medyan: `349.5`)
  - Modelde uzunluk patlaması (*length explosion*) veya format çökmesi (*mode collapse*) yaşanmamıştır.
- **Karar:** Claude'un 3. maddesi uyarınca 30. adım kapısı **yalnızca sağlık ve güvenlik doğrulaması** olarak başarıyla geçilmiş olup, etkinlik kararı **100. adım sonunda `Clean-475` üzerinde eşli McNemar testiyle** verilmek üzere eğitim 31. adımdan 100. adıma devam ettirilmektedir.

---

## 4. Modül 4 (`Best-of-4 + PRM`) vs. Saf Çoğunluk Oylaması (`Majority Voting`) Kontrolü (GPU'suz)

Claude'un **4. maddede** istediği kritik kontrol:
> *"Adil kıyas, aynı 4 örnek üzerinde çoğunluk oyu. PRM seçimi çoğunluk oyunu geçmiyorsa PRM bir şey eklemiyor. 4 örnek depolandıysa bu kontrol GPU'suzdur."*

`sft/eval_grpo_long_best_of_g_prm_150_results.json` içinde depolanan **birebir aynı aday havuzu (`candidate_pool`)** üzerinde **Saf Çoğunluk Oylaması (`MV@4`: yeni üretilen 4 aday üzerinde çoğunluk oyu; `MV@5`: Pass-1 dahil 5 aday üzerinde çoğunluk oyu)** ile **Tri-Modal PRM (`Modül 4`)** karşılaştırılmıştır:

### 4.1 Temiz 125 Held-Out Karşılaştırma Tablosu (Aynı Depolanmış Adaylar Üzerinde)

| Seçim Yöntemi (Aynı Aday Havuzu) | Temiz 125 (Legacy Grader) | Temiz 125 (Normalized SymPy) | $G=1$'e Göre Net Fark | Çoğunluk Oyuna (`MV@4`) Göre PRM Farkı |
| :--- | :---: | :---: | :---: | :---: |
| **Pass-1 Greedy ($G=1$)** | 100 / 125 (%80.00) | 104 / 125 (%83.20) | Referans | — |
| **Saf Çoğunluk Oyu (`MV@4` — 4 Yeni Aday)** | **102 / 125 (%81.60)** | **106 / 125 (%84.80)** | **+2 Soru (`Q120, Q124`)** | Referans (`MV@4`) |
| **Saf Çoğunluk Oyu (`MV@5` — Tüm 5 Aday)** | **102 / 125 (%81.60)** | **106 / 125 (%84.80)** | **+2 Soru (`Q120, Q124`)** | `0` Fark |
| **Tri-Modal PRM Seçimi (`Modül 4`)** | **102 / 125 (%81.60)** | **106 / 125 (%84.80)** | **+2 Soru (`Q120, Q138`)** | **`0` Net Fark (`+1 [Q138] / -1 [Q124]`, $p=1.0000$)** |

### 4.2 Soru Bazında `MV@4` ile `PRM` Farkının Adli Analizi
Tetiklenen 13 Held-Out sorunun 11'inde `MV@4` ve `PRM` aynı sonucu (`PASS`/`FAIL`) verirken, **2 soruda birbirini nötrlemiştir**:
1. **`Q138` (`GT = '204_5'` — PRM Kazandı, Çoğunluk Oyu Kaybetti):**
   - Adaylar: `['244_5', '204_5', '224_5', '224_5']` (Pass-1: `'224_5'`).
   - Saf Çoğunluk Oyu (`MV@4` ve `MV@5`) 3 oy alan yanlış `'224_5'` cevabını seçip **başarısız** olurken; SymPy taban-5 doğrulayıcısı `'224_5'` dallarındaki aritmetik çelişkiyi yakalayıp tek doğru dal olan `'204_5'` adayını seçmiştir (**PRM +1**).
2. **`Q124` (`GT = '8'` — Çoğunluk Oyu Kazandı, PRM Kaybetti):**
   - Adaylar: `['12(PRM: +25.3)', '8(PRM: -1.5)', '8(PRM: -1.8)', '7(PRM: +4.2)']` (Pass-1: `'10'`).
   - Stokastik dalların ikisi doğru cevap olan `'8'` değerini üretmiş ve **Saf Çoğunluk Oyu (`MV@4` ve `MV@5`) `'8'` cevabını seçerek soruyu çözmüştür!**
   - Ancak PRM, `'8'` diyen dallardaki ara metin ifadelerini sahte çelişki olarak cezalandırıp (`-1.5`, `-1.8`), daha çok denklem yazan yanlış `Cand1 = '12'` (`+25.3`) dalını seçerek çoğunluk oyunun doğru cevabını ezmiştir (**PRM -1**).

**Bilimsel Sonuç:** Claude'un 4. maddedeki öngörüsü de **%100 kanıtlanmıştır**. Modül 4'ün `+2` soruluk kazanımı (`104 -> 106`), $4\times$ test-zamanı örneklemesinden (*Best-of-4 sampling*) gelmektedir; aynı 4 örnek üzerinde **Tri-Modal PRM (`106/125`) ile Saf Çoğunluk Oylaması (`106/125`) arasında net doğruluk farkı sıfırdır** (`+1 kazanç [Q138], -1 kayıp [Q124], McNemar p=1.0000`). Makalede bu bulgu tam şeffaflıkla ("$G=4$ kazanımının örnekleme çeşitliliğinden geldiği, PRM'in Q138'de çoğunluk oyunu yenerken Q124'te sahte çelişkiyle çoğunluk oyunun gerisine düştüğü ve net skorda `106/125 = 106/125` berabere kaldığı") olarak raporlanacaktır.
