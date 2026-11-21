# SODALS v2.1 — Claude'un 1. ve 2. Maddelerinin Tam Uygulaması: Adli Notlayıcı Mutabakatı, Tüm Sorularda "Erken Pes Etme" (Token Dağılımı) Analizi ve Ön-Kayıtlı (Pre-Registered) Ölçekleme Protokolü

**Tarih:** 19 Ekim 2026  
**Betik ve Doğrulama Kaynağı:** `sft/audit_claude_items_1_and_2.py`, `sft/audit_step3_all_modules_clean_125.py`

---

## 1. Madde 1a & 1b (GPU'suz): 3 Katmanlı Notlayıcı Mutabakatı ve "Modül 2'de 116 vs 123 Biçim Kayması Var mı?" Sorusunun Kanıtı

Claude, `sft/eval_grpo_long_cst_prm_150_results.json` dosyasındaki üst-seviye özet alanlarına (`total_strict_pass: 116`, `total_symbolic_pass: 126`) bakarak şu çıkarımı yapmıştı:
> *"Aynı katı notlayıcıyla Modül 2 116, baseline 123. Karşılaştırma 3 kazanç, 10 kayıp çıkıyor. Modül 2'nin yalnızca sembolik geçtiği 10 sorunun 9'unu baseline katı notlayıcıyla zaten geçiyordu. Yani Modül 2 cevap biçimini değiştiriyor..."*

### Gerçekte Ne Oldu? (Birebir Metin Karşılaştırması)
1. **Modül 2, 150 sorunun 147'sinde (ve doğru çözdüğü 126 sorunun TAMAMINDA: 126/126) GRPO Long ile bayt-bayt (%100) BİREBİR AYNI cevabı (`predicted`) üretmiştir!**
   - Modül 2'nin GRPO Long'dan farklı bir cevap (`predicted`) ürettiği **yalnızca 3 soru vardır** (üçünde de kapı tetiklenmiştir):
     - **`Q055` (`CLEAN-125`, `GT = '0.15'`):** GRPO `'0.15'` (`PASS`) $\to$ Modül 2 `'0'` (`FAIL` — sahte çelişki yüzünden **1 kayıp**).
     - **`Q120` (`CLEAN-125`, `GT = '-9'`):** GRPO `'2'` (`FAIL`) $\to$ Modül 2 `'13'` (`FAIL`).
     - **`Q138` (`CLEAN-125`, `GT = '204_5'`):** GRPO `'224_5'` (`FAIL`) $\to$ Modül 2 `'244_5'` (`FAIL`).
   - Geri kalan **147 sorunun tamamında** Modül 2'nin ürettiği cevap metni (`predicted`), GRPO Long'un ürettiği cevap metniyle **karakteri karakterine aynıdır** (hiçbir biçim kayması / format değişikliği yoktur).

2. **Peki `116`, `122/123` ve `126/127` Sayıları Neden Farklı Çıktı?**
   - Çünkü kod tabanında üç farklı katılık seviyesinde eşleştirme tanımı bulunmaktaydı:
     - **Seviye 1 — Salt Dize Eşitliği (`strict_ok` = 116 / 150):** `NormalizedMathMatcher.check_match(pred, gt)` fonksiyonunun döndürdüğü ilk boolean değerdir (`cp == cg`). **Hiçbir SymPy sembolik eşdeğerliği içermez.** Örneğin model `(3, \frac{\pi}{2})` yerine eşdeğer bir kesir veya `\frac{27}{2}` yerine `13.5` yazdığında salt dize eşitliği `False` döner. SFT, GRPO ve Modül 1 de **bu salt dize eşitliğiyle** puanlandığında `123` değil, **116 / 150** (veya Q055 dahil **117 / 150**) almaktadır!
     - **Seviye 2 — Legacy Baseline Notlayıcısı (`check_math_match_legacy` = GRPO'da 123/150 [100/125], Modül 2'de 122/150 [99/125]):** SFT, GRPO ve Modül 1 betiklerinin içindeki orijinal notlayıcıdır. Salt dize eşitliğinin üzerine **temel SymPy (`to_sympy_legacy`)** kontrolü ekler ve 6 soruyu (Q055 ile birlikte 7 soruyu) zaten `PASS` kabul eder (`116 + 7 = 123`). Aynı Legacy notlayıcı Modül 2'ye uygulandığında sonuç `116` değil, **122 / 150 (Temiz 125'te 99 / 125)** çıkmaktadır (`0 kazanç, 1 kayıp [Q055]`).
     - **Seviye 3 — Normalized Symbolic Notlayıcı (`NormalizedMathMatcher` = GRPO'da 127/150 [104/125], Modül 2'de 126/150 [103/125]):** Legacy notlayıcının LaTeX ayrıştırma hatası yüzünden kaçırdığı **4 soruyu (`Q032: 11\sqrt{2}`, `Q054: -\pi/6`, `Q098: East`, `Q128: 4210_5` — tamamı Temiz 125'te)** doğru kabul eder. Bu 4 soruda **SFT, GRPO, Modül 1, Modül 2, Modül 3 ve Modül 4 modellerinin altısı da ilk geçişte birebir aynı cevabı üretmiştir.**

### Birleştirilmiş Notlayıcı Tablosu (Tüm Modeller Aynı 3 Notlayıcı Altında)

| Aşama / Model | Temiz 125 (Seviye 1: Salt Dize) | Temiz 125 (Seviye 2: Legacy SymPy) | Temiz 125 (Seviye 3: Norm. SymPy) | Birleşik 150 (Legacy / Norm. SymPy) | Gerçek Algoritmik Fark (SFT'ye Göre, Temiz 125) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **0. SFT Baseline** | 94 / 125 (%75.20) | **99 / 125 (%79.20)** | **103 / 125 (%82.40)** | 122/150 / 126/150 | Referans |
| **1. 1536-Token GRPO Long** | 95 / 125 (%76.00) | **100 / 125 (%80.00)** | **104 / 125 (%83.20)** | 123/150 / 127/150 | **+1 Soru (`Q105` çözüldü, 0 kayıp)** |
| **2. Modül 1 (Pondering ODE)** | 95 / 125 (%76.00) | **100 / 125 (%80.00)** | **104 / 125 (%83.20)** | 123/150 / 127/150 | **Null (GRPO ile %100 aynı)** |
| **3. Modül 2 (Saf CST PRM, $G=1$)** | 94 / 125 (%75.20) | **99 / 125 (%79.20)** | **103 / 125 (%82.40)** | 122/150 / 126/150 | **Negatif (GRPO'ya göre -1 kayıp: `Q055`)** |
| **4. Modül 3 (Tuned Lens + PRM, $G=1$)** | 95 / 125 (%76.00) | **100 / 125 (%80.00)** | **104 / 125 (%83.20)** | 123/150 / 127/150 | **Null (`Q055` hatasını düzeltti, +0 yeni soru)** |
| **5. Modül 4 (Best-of-$G=4$ PRM)** | 97 / 125 (%77.60) | **102 / 125 (%81.60)** | **106 / 125 (%84.80)** | 125/150 / 129/150 | **+3 Soru (`Q105, Q120, Q138` çözüldü, 0 kayıp)** |

---

## 2. Madde 1c (GPU'suz): Tüm Başarısız Sorularda "Erken Pes Etme" (*Early Surrender*) Mekanizmasının Sayımı

Claude'un sorduğu kritik soru şuydu:
> *"Aynı depolanmış çıktılarla 'erken pes etme' mekanizmasını tek soru yerine tüm sorularda sayın: baseline'ın başarısızları arasında bütçenin çok altında kısa tahminle biten üretimlerin oranı GRPO'da gerçekten düşüyor mu?"*

`sft/audit_claude_items_1_and_2.py` ile **Temiz 125 Held-Out (`Q026..Q150`)** kümesindeki **26 SFT başarısız sorunun tamamında** (ve tüm 125 soruda) token uzunluk dağılımını saydık:

### 2.1 Temiz 125 Held-Out Kümesinde Token Dağılımı Bulguları
1. **125 Sorunun 111'inde (%88.8) SFT ve GRPO Birebir Aynı Token Sayısını ve Birebir Aynı Cevabı Üretmiştir:**
   - Yalnızca **14 / 125 soruda** (ve SFT'nin başarısız olduğu 26 sorunun **yalnızca 7'sinde**: `Q044, Q095, Q097, Q105, Q111, Q120, Q121`) GRPO yörüngesi SFT'den ayrışmıştır.
2. **SFT'nin Başarısız Olduğu 26 Soruda (`n = 26`) Erken Bitirme Oranları:**
   - **Ortalama Token Uzunluğu (`n = 26`):** SFT = **693.0 token** (medyan: `387.0`) $\to$ GRPO = **714.0 token** (medyan: `401.0`) (**+21.0 token artış**).
   - **Eşik Altı Kısa Üretim Oranları (`n = 26`):**
     - `< 256 token`: SFT **10 / 26 (%38.5)** vs. GRPO **10 / 26 (%38.5)** (**Değişim: %0.0**)
     - `< 512 token`: SFT **15 / 26 (%57.7)** vs. GRPO **15 / 26 (%57.7)** (**Değişim: %0.0**)
     - `< 768 token`: SFT **17 / 26 (%65.4)** vs. GRPO **17 / 26 (%65.4)** (**Değişim: %0.0**)
     - `< 1024 token`: SFT **20 / 26 (%76.9)** vs. GRPO **20 / 26 (%76.9)** (**Değişim: %0.0**)
     - `< 1536 token`: SFT **22 / 26 (%84.6)** vs. GRPO **22 / 26 (%84.6)** (**Değişim: %0.0**)
     - `= 2048 token (tavan)`: SFT **4 / 26 (%15.4)** vs. GRPO **4 / 26 (%15.4)** (**Değişim: %0.0**)
3. **SFT ile GRPO'nun Farklılaştığı 7 Başarısız Held-Out Sorunun Tek Tek Dökümü:**
   - **`Q095` (`FAIL -> FAIL`):** `312 token -> 349 token` (`+37 token`, tahmin `'110' -> '110'`)
   - **`Q097` (`FAIL -> FAIL`):** `1066 token -> 1311 token` (`+245 token`, tahmin `'1, -1' -> 'x \in \mathbb{R} \setminus ...'`)
   - **`Q105` (`FAIL -> PASS ★`):** `844 token -> 1023 token` (`+179 token`, tahmin `'18' -> '40'`)
   - **`Q111` (`FAIL -> FAIL`):** `1282 token -> 1367 token` (`+85 token`, tahmin `'7' -> '1'`)
   - **`Q121` (`FAIL -> FAIL`):** `455 token -> 453 token` (`-2 token`, tahmin `'odd' -> 'odd'`)
   - **`Q044` (`FAIL -> FAIL`):** `2048 token -> 2048 token` (`0 fark`, tahmin `'65' -> '80'`)
   - **`Q120` (`FAIL -> FAIL`):** `2048 token -> 2048 token` (`0 fark`, tahmin `'4' -> '2'`; *Not: Bu soru Modül 4 Best-of-4 PRM aşamasında `-9` ile kurtarılmıştır*).

### 2.2 Neden 25 Adımlık GRPO Pilotu Yörüngelerin %88.8'ini Değiştirmedi? (Kök Neden Tespiti)
Bu adli tarama, 25 adımlık GRPO pilotunun neden yalnızca 14 soruda yörüngeyi değiştirdiğini ve genel "erken pes etme" oranını (`<512 token`: %57.7) neden düşürmediğini **iki somut mühendislik bulgusuyla** açıklamaktadır:
1. **Eğitimde Sıfır-Avantaj (*Zero-Advantage*) Adımları (`sft/grpo_long_results.json`):**
   - 25 eğitim sorusunun (`Q001..Q025`) 23'ünü SFT modeli zaten çözebildiği için, 25 GRPO adımının **10'unda** 4 rollout'un tamamı aynı ödülü almış (`solved_rollouts = 4` veya `0`, `std_r = 0`, `advantage = 0.0`), dolayısıyla **yalnızca 15 adımda aktif politika gradyanı (policy gradient) oluşmuştur** (`lr = 1e-5`, `beta_kl = 0.04`).
2. **Çıkarım Zamanında `SoftLoraBlender` Sönümlemesi (`0.24x` Katsayısı):**
   - `train_grpo_long_context.py` sırasında LoRA adaptörü tam ölçekte (`scale = 1.0`) eğitilmiş, ancak `eval_grpo_long_math500_150.py` sırasında `SoftLoraBlender` genel matematik adaptörünü `scale_general = 0.25 * (1 - alpha) ≈ 0.24` katsayısıyla çarparak **GRPO ağırlık güncellemesini çıkarım esnasında ~4 kat sönümlemiştir!**

**Sonuç:** Claude'un öngörüsü %100 doğrudur. Mevcut 25 adımlık pilotta "erken pes etmeyi önleme" mekanizması veri seti genelinde (`15/26 -> 15/26`) istatistiksel olarak gerçekleşmemiştir; `Q105` (`844 -> 1023` token) tekil bir kurtarmadır.

---

## 3. Madde 2: Çalıştırmadan Önce Yazılan Ön-Kayıtlı (Pre-Registered) Ölçekleme ve Durdurma Protokolü

Claude'un 2., 3. ve 4. maddeleri doğrultusunda, yeni bir GPU ölçekleme deneyi başlatılmadan önce **test havuzu, istatistiksel test, anlamlılık eşiği ve durdurma kuralı (stopping rule)** aşağıdaki şekilde ön-kayıt (*pre-registration*) altına alınmıştır:

### 3.1 Eğitim Havuzu ve Veri İzolasyonu (Sıfır Sızıntı Kuralı)
- **Eğitim Veri Kümesi (`N_train = 100` Prompt):**
  - `MATH-500` test setinden **HİÇBİR** soru kullanılmayacaktır.
  - Eğitim soruları yalnızca orijinal **Hendrycks MATH Eğitim Bölümü (`train` split)** içerisinden (`MATH-500` test kümesiyle %0 örtüşme kontrolü yapılarak) seçilecektir.
  - **Aktif Gradyan Filtresi (Zorluk Seçimi):** İlk pilottaki "4/4 doğru = sıfır gradyan" kaybını önlemek için, eğitim soruları **Level 3, Level 4 ve Level 5** zorluk seviyesindeki (cebirsel, sayı teorisi ve kombinatorik) problemlerden seçilecektir.
  - **Ölçek Tutarlılığı:** Eğitim ve çıkarım arasındaki `0.24x` adaptör sönümleme uyumsuzluğu kaldırılacak; değerlendirme sırasında eğitilen adaptör eğitimdeki efektif ölçeğiyle birebir aynı katsayıda çalıştırılacaktır.

### 3.2 Temiz Değerlendirme Havuzu (`Clean Held-Out Pool`)
- **Birincil Temiz Test Havuzu (`N_test = 475`):**
  - `HuggingFaceH4/MATH-500` içerisinden daha önce 25 soruluk pilotta kullanılan `Q001..Q025` tamamen dışarıda bırakılarak geriye kalan **475 Temiz Soru (`Q026..Q500`)** birincil held-out test havuzu olarak tanımlanmıştır.
  - **Alt-Havuz A (`Clean-125`, `Q026..Q150`):** Mevcut modüllerin (`Modül 1, 2, 3, 4`) eşik ve hiperparametre analizlerinde zaten gözlemlendiği için **"Geliştirme / Doğrulama (Dev) Alt-Kümesi"** olarak ayrılmıştır.
  - **Alt-Havuz B (`Pristine-350`, `Q151..Q500`):** Bugüne kadar hiçbir modelin, hiçbir kapının veya hiçbir eşik ayarının **bir kez bile görmediği 350 Tamamen Bakir Soru** birincil doğrulayıcı (*unseen confirmatory test set*) olarak kilitlenmiştir.

### 3.3 Birincil İstatistiksel Test ve Anlamlılık Eşiği ($\alpha = 0.05$)
- **Test:** İki-yönlü Tam Binom McNemar Testi (*Two-Sided Exact Binomial McNemar Test*):
  $$p = \min\left(1.0,\; 2 \sum_{i=0}^{\min(b_{01}, b_{10})} \binom{b_{01} + b_{10}}{i} 0.5^{b_{01} + b_{10}}\right)$$
  burada $b_{01}$ SFT Baseline'ın yanlış yapıp yeni modelin doğru yaptığı soru sayısı (`GAIN`), $b_{10}$ ise SFT Baseline'ın doğru yapıp yeni modelin yanlış yaptığı soru sayısıdır (`DROP`).
- **Kabul / Başarı Eşiği ($p < 0.05$ için gereken minimum net farklar):**
  - Eğer $b_{10} = 0$ (sıfır kayıp) ise: En az **$b_{01} \ge 6$ net kurtarma** ($p = 0.03125 < 0.05$).
  - Eğer $b_{10} = 1$ (1 kayıp) ise: En az **$b_{01} \ge 8$ kurtarma** (`+7` net soru, $p = 0.03906 < 0.05$).
  - Eğer $b_{10} = 2$ (2 kayıp) ise: En az **$b_{01} \ge 10$ kurtarma** (`+8` net soru, $p = 0.03857 < 0.05$).
  - Eğer $b_{10} = 5$ (5 kayıp) ise: En az **$b_{01} \ge 15$ kurtarma** (`+10` net soru / $N=475$'te yaklaşık **+%2.11** artış, $p = 0.04139 < 0.05$).

### 3.4 Katı Durdurma Kuralı (*Hard Stopping Rule*)
1. **Bilişsel Modüllerin ($G=1$ Pondering ODE, $G=1$ CST PRM, $G=1$ Tuned Lens) Durumu:**
   - `Clean-125` üzerinde $G=1$ deterministik çıkarımda **Modül 1 (`0 net`)**, **Modül 2 (`-1 net`)** ve **Modül 3 (`0 net`)** sıfır veya negatif katkı verdiği için **$G=1$ kapı mimarileri resmen durdurulmuş (park edilmiş)** ve makalede dürüstçe **"Null / Negatif Bulgu (Confident Hallucination under Greedy Decoding)"** olarak raporlanmıştır.
2. **GRPO Ölçekleme Durdurma Kuralı:**
   - `MATH train` bölümünden seçilen 100 soruluk GRPO eğitimi tamamlandıktan sonra, temiz test havuzunda McNemar $p < 0.05$ eşiği (yukarıdaki $b_{01}/b_{10}$ tablosu) **sağlanamazsa**, 8 GB VRAM sınırında tek-adaptörlü GRPO ölçekleme iddiası sonlandırılacak; makalenin ana iddiası **"4-bit NF4 + CPU Embedding Offload ile %46.5 VRAM tasarrufu (7,298 MB -> 3,907 MB) ve doğruluk kaybı olmadan ($p=1.0$) bellek-sınırlı RL fizibilitesi"** olarak sabitlenecektir.
3. **Modül 4 (`Best-of-G [G=4] Neurosymbolic Rollout + PRM`) Değerlendirme Kuralı:**
   - Modül 4'ün eşikleri `Clean-125` (`Dev` bölümü, `+2` kurtarma: `Q120, Q138`, `0` kayıp) üzerinde sabitlenmiştir. Bakir test kümesinde yalnızca **bir kez** (hiperparametre değiştirmeden) değerlendirilecek ve aynı McNemar $p < 0.05$ durdurma kuralına tabi olacaktır.
