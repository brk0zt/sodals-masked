# CLAUDE İÇİN AKADEMİK DENETİM VE DOĞRULAMA PAKETİ

Bu klasör, Claude'un bir önceki denetim turunda ortaya koyduğu 7 kritik araştırma bütünlüğü ve teknik tutarsızlık kategorisinin tamamının çözüldüğünü teyit ettirmek için derlenmiş resmi inceleme paketidir.

---

## 📂 Dosya Fihristi ve İçerikleri

| Sıra | Dosya Adı | Açıklama |
|:---:|:---|:---|
| **01** | `01_CLAUDE_VE_AKADEMISYEN_SUNUM_METNI.md` | **(ÖNERİLEN İLK OKUMA)** Claude'a doğrudan kopyalayıp yapıştırabileceğiniz, 7 maddenin çözümlerini tek tek anlatan Türkçe sunum metni. |
| **02** | `02_MASTER_ACADEMIC_AUDIT_DOSSIER.md` | Bağımsız akademisyenler için hazırlanmış, matematiksel ispatlar ve bootstrap/McNemar güven aralıklarını içeren kapsamlı İngilizce denetim dosyası. |
| **03** | `03_THREE_PAPER_PORTFOLIO_STRATEGY.md` | Tek bir mega-makale yerine 3 bağımsız makaleye ayrışan portföy stratejisi yol haritası. |
| **04** | `04_PAPER1_REASONING_MASTER.md` | **Makale 1 (Markdown):** SDPO, 3.42GB VRAM mimarisi, 1536-token GRPO, Temiz 125 Held-Out ($p=1.0000$), Q105 CoT, Adım 3 (Modül 1, 2, 3 ve 4) Birleştirilmiş Değerlendirici Altı Aşamalı Ablasyonu, *Confident Hallucination* ve Modül 4 Targeted Best-of-$G$ ($G=4$) PRM kazanımları (Q120, Q138 ile %84.80 NormSym / %81.60 Legacy). |
| **05** | `05_PAPER1_REASONING_LATEX.tex` | **Makale 1 (LaTeX):** NeurIPS / ICML / TMLR uyumlu, Tablo 5 (Altı Aşamalı Birleştirilmiş Ablasyon) ve 15/15 kaynakça eşleşmeli tam `.tex` dosyası. |
| **06** | `06_PAPER2_SOFT_MOA.md` | **Makale 2 (Markdown):** Alt-uzay Girişim Yasası ($E_7=133 \to 70$), Katman 32 GPR ve 170Q benchmarkta Eğitimsiz Soft Blender zirvesi (138/170, %81.18). |
| **07** | `07_PAPER2_SOFT_MOA_LATEX.tex` | **Makale 2 (LaTeX):** ICLR Tiny / MLSys uyumlu derlenmeye hazır tam `.tex` dosyası. |
| **08** | `08_PAPER3_EDGESSM_CPS.md` | **Makale 3 (Markdown):** Saf Rust deterministik motor, GGUF `tile` dizilimi, AssetOpsBench ($N=120$, %80.0), SRE/FinOps/EdgeTelemetry. (Sıfır matematik benchmarkı içerir). |
| **09** | `09_PAPER3_EDGESSM_IEEE_TII_LATEX.tex` | **Makale 3 (LaTeX):** *IEEE Transactions on Industrial Informatics* çift sütun dergi formatında derlenmeye hazır tam `.tex` dosyası. |
| **10** | `10_CLEAN_125_HELDOUT_AUDIT_REPORT.md` | 25 eğitim sorusu çıkarıldıktan sonra temiz 125 held-out kümesinde yapılan McNemar ($p=1.0000$) ve paired bootstrap istatistiksel raporu. |
| **11** | `11_clean_125_heldout_verification_results.json` | Temiz 125 kümesindeki soru bazlı ham model çıktıları ve doğrulanmış json sonuçları. |
| **12** | `12_math500_heldout_clean_125_dataset.json` | Sadece 125 temiz test sorusunu içeren izole veri seti. |
| **13** | `13_math500_training_pilot_25_dataset.json` | GRPO eğitiminde kullanılan 25 promptluk izole eğitim veri seti. |
| **14** | `14_STEP3_MODULES_1_2_3_CLEAN_125_AUDIT_REPORT.md` | **Adım 3 (Modül 1, 2, 3 ve 4) Tamamlanmış 150/150 GPU Deneyi ve Temiz 125 Adli Denetim Raporu:** Değerlendirici kayması (*Grader Drift*), Q055 gerilemesi/kurtarılması, Greedy $T=0$ *Confident Hallucination* ($0.7867$ vs $0.7756$) analizi ve Modül 4 (Best-of-$G$ PRM) ile Q120/Q138 Held-Out kazanımları tablosu. |
| **15** | `15_math500_heldout_clean_125_module3_tuned_lens.json` | Modül 3 (Tuned Lens & Tri-Modal PRM) için izole edilmiş Temiz 125 Held-Out soru bazlı tam GPU sonuçları. |
| **16** | `16_math500_heldout_clean_125_module4_best_of_g_prm.json` | **Modül 4 (Targeted Best-of-$G$ PRM)** için izole edilmiş Temiz 125 Held-Out soru bazlı tam GPU sonuçları (106/125 = %84.80 NormSym, 102/125 = %81.60 Legacy). |
| **17** | `17_math500_training_pilot_25_module4_best_of_g_prm.json` | Modül 4 için 25 Eğitim Pilot sorusu soru bazlı tam GPU sonuçları (23/25 = %92.00). |
| **18** | `18_Burak_Ozturk_CV.md` | Burak Öztürk güncellenmiş ve doğrulanmış teknik CV'si (Tüm şablon boşlukları temizlenmiş, EdgeSSM/SODALS mimari kazanımları ve MATH-500 Held-Out %84.80 puanı işlenmiş). |

---

## 💬 Claude'a Gönderilecek Örnek Mesaj Şablonu

Claude'a doğrudan şu mesajı iletebilirsiniz:

```text
Claude merhaba, 

Önceki turda ortaya koyduğun eleştirileri (sayı tekrarı, GRPO veri sızıntısı, McNemar p=1.0000 ve SFT tavanı gerçeği, Q105 token sayısı, KL vs. Kosinüs katman ayrımı, 0.22ms reset vs. GPU re-prefill süresi, O(N log deg) ve Nogood Cache düzeltmeleri, FailureSensorIQ vs. AssetOpsBench atıf hizalaması) ciddiyetle ele aldık.

Ayrıca Adım 3 (Modül 1: Pondering ODE, Modül 2: CST PRM, Modül 3: Tuned Lens & Tri-Modal PRM, Modül 4: Targeted Best-of-G PRM) değerlendirmelerinin tamamını (150/150) GPU'da bitirip tüm 6 aşamayı birebir aynı değerlendiriciler altında Temiz 125 Held-Out kümesinde çapraz denetledik:
- Değerlendirici normalizasyonundan gelen +4 soruluk farkı (Q032, Q054, Q098, Q128) model kazanımı gibi sunmak yerine dürüstçe ayrıştırdık (Temiz 125: SFT 99/125 Legacy & 103/125 NormSym -> GRPO/Modül 1/Modül 3 100/125 Legacy & 104/125 NormSym).
- Greedy T=0 çıkarımda yanlış çözülen soruların katmanlar-arası kararlılığının (0.7867) doğru çözülenlerden (0.7756) biraz daha yüksek olduğunu ("Confident Hallucination") ve tekil deterministik yeniden denemenin bu yüzden yeni soru kazandıramadığını kanıtladık.
- Modül 4'te (Targeted Best-of-G [G=4] Neurosymbolic PRM) yalnızca yapısal kusur barındıran 15/150 yörüngede çoklu-dal açarak ve Tri-Modal PRM ile puanlayarak Temiz 125 Held-Out kümesinde sıfır gerileme (0 drop) ile +2 gerçek soru kazandık:
  * Q120 (GT: -9): 2048 tokenda tekrara düşen Pass 1 (pred: 2) yerine 11 doğrulanmış denklem içeren Cand4 (-9, +35.6 PRM) seçildi.
  * Q138 (GT: 204_5): Pass 1 ve iki stokastik dalın hatalı 224_5 cevabını (3'e 1 çoğunluk oyu) tekrarlamasına rağmen, SymPy taban-n PRM'i hatalı dalları eledi ve tek tutarlı dal olan Cand2'yi (204_5, +27.4 PRM) seçerek çoğunluk oylamasını mağlup etti!
- Böylece Temiz 125 Held-Out başarısı 104/125'ten 106/125'e (%84.80 NormSym / 102/125 %81.60 Legacy) yükseldi (SFT'ye göre +3 soru / +%2.40 net artış).

Yapılan düzeltmelerin tam dökümünü içeren "01_CLAUDE_VE_AKADEMISYEN_SUNUM_METNI.md", "04_PAPER1_REASONING_MASTER.md" ve "14_STEP3_MODULES_1_2_3_CLEAN_125_AUDIT_REPORT.md" dosyalarını ekte iletiyorum. Araştırma bütünlüğü ve bilimsel savunulabilirlik açısından inceleyip teyit eder misin?
```
