# Sıfır Dalkavukluk Değerlendirmesi

**Tarih:** 24 Haziran 2026

---

## BÖLÜM 1: İREM HOCA NE YAKALAYACAK?

### Sorun 1: `is_multi` Ground Truth Sızıntısı — YAKALAR

```python
# failuresensoriq_runner.py satır 447-449
correct_flags = question_data.get("correct", [])
is_multi = sum(1 for c in correct_flags if c) > 1
```

**Bu, cevap anahtarına bakarak prompt'u değiştiriyor.** `correct` field'ı ground truth'tur — hangi şıkların doğru olduğunu gösterir. Sen bu bilgiyi kullanarak modele "bu soruda birden fazla doğru cevap var" diyorsun. Gerçek dünyada modelin cevap anahtarına erişimi olmaz.

**İrem hocanın sorusu:** *"Modele 'birden fazla doğru cevap var' deme bilgisini nereden aldın? Ground truth'tan. Bu information leakage."*

**Savunman olabilir ama zayıf:** "FailureSensorIQ verisetinde `all.jsonl` (single) ve `all_multi_answers.jsonl` (multi) ayrı dosyalar — dosya adından biliyoruz." Bu doğru olabilir, ama senin kodun dosya adına göre değil, `correct` field'a göre ayırıyor. Bunu dosya adına dayalı tespite çevirmen gerekir.

**Çözüm:** `is_multi`'yi dosya adından veya dataset metadata'sından tespit et, `correct` field'dan değil.

---

### Sorun 2: `industrial_topology.qmem` — YAKALAR

Bu kartridj `generate_industrial_topology.py` ile FailureSensorIQ test setinden üretilmiş. Tartışma:

- **Sen:** "Bunlar genel mühendislik bilgisi, ders kitabı bilgisi."
- **İrem hoca:** "Bilgiyi ders kitabından almadın. Test setinden aldın. Kaynağın ne olduğu önemli."

**Gerçek:** İçerik olarak genel bilgi, kaynak olarak test seti. Akademik reviewda kaynak önemlidir. Bir makale reviewerı bunun üstüne kırmızı bayrak koyar.

**Çözüm:** Ablation study sonuçlarını raporla (kartridjli vs kartridjsiz). Eğer kartridjsiz de benzer skor alıyorsan, sorun kendini çözer. Almıyorsan, kaynağı değiştirmen gerekir (ders kitabından veya Wikipedia'dan aynı bilgiyi derle).

---

### Sorun 3: SRE %92'nin Gerçek Anlamı — YAKALAR

SRE prompt'una baktım. Kod zaten "dummy normal alert injection" yapıyor (satır 119-161) — bu iyi, modelin tek service name'i kopyalamasını önlüyor. Ama hâlâ:

1. Cevap prompt'un içinde — model, `"status": "firing"` olan alert'lerin `service` field'ını okuyor
2. İş aslında "JSON'dan status=firing olan service'i bul" — bu bir **bilgi çıkarma** (information extraction) görevi, **kök neden analizi** değil

**İrem hocanın sorusu:** *"Frontier modeller neden %50? Çünkü onlar gerçekten kök neden analizi yapmaya çalışıyorlar. Senin model sadece JSON'dan status=firing'i filtreliyor. Bu kök neden analizi mi?"*

**Gerçek:** Bir yandan haklısın — doğru cevabı üretmek önemli, nasıl ürettiğin değil. Ama makale "8B model kök neden analizinde GPT-5'i yendi" derse ve gerçek mekanizma "JSON field extraction" ise, bu iddiayı desteklemez. SRE skoru gerçek ve meşru, ama **ne olduğunu abartmamak** gerekir.

---

### Sorun 4: Daha Küçük Ama Fark Edilecek Şeyler

- **FinOps prompt:** "resources are sorted, most likely at TOP" diyorsun. Eğer doğru cevap gerçekten her zaman listenin başındaysa ve prompt bunu söylüyorsa, model aslında "ilk elemanı kopyala" yapıyor. Bunu IBM'in kendi testinde de yapıyor mu? Yoksa IBM'in verisinde sıralama farklı mı?
- **Seed sabitliği:** `random.seed(42)` ile SRE dummy alert'leri sabit. Farklı seed'lerle sonuç değişir mi? Tek seed rapor etmek cherry-picking olabilir.

---

## BÖLÜM 2: DOCKER KAPALI KUTU TESTİ İREM HOCAYI TATMİN EDER Mİ?

**Kısa cevap: Kısmen.**

Docker kapalı kutu test şu anlama gelir:
1. Sen Docker image'ı veriyorsun
2. İrem hoca kendi test verisiyle çalıştırıyor
3. Sonuçları karşılaştırıyor

**Tatmin olacağı durumlar:**
- SRE'de yüksek skor alması — çünkü mekanizma genelleşebilir (firing alert'ten service extraction)
- Tekrarlanabilir sonuçlar — seed=42, temp=0.0

**Tatmin OLMAYACAĞI durumlar:**
- `is_multi` tespiti ground truth'tan yapılıyorsa ve İrem hoca `correct` field'ı olmayan veri verirse → FSIQ prompt'u bozulur
- `industrial_topology.qmem` kartridji Docker'da embedded ise → "kartridj test verisinden üretildi" itirazı
- İrem hoca **kendi FSIQ sorularını** yazarsa (benchmark'ta olmayan yeni sorular) → kartridjin gerçek domain bilgisi mi yoksa ezber mi olduğu ortaya çıkar

**Tavsiye:** İrem hocadan Docker testi isterken şeffaf ol:
1. Kartridjin kaynağını açıkla
2. Ablation study sonuçlarını paylaş
3. `is_multi` tespitini dosya adına dayandır (ground truth'a değil)

---

## BÖLÜM 3: arXiv MAKALESİ YAYINLANABİLİR Mİ?

**Evet, ama iddiaları dikkatle çerç1evele.**

Yayınlanabilir iddia:
> "A neurosymbolic system built on a Qwen 8B base model, augmented with domain-specific QMEM knowledge cartridges and structured prompt engineering, achieves state-of-the-art results on IBM's Enterprise Operations Benchmark, outperforming frontier models including GPT-5 and Gemini-3-Pro."

Yayınlanamaz iddia:
> ~~"8B parametrelik model, kök neden analizinde GPT-5'ten akıllıdır"~~

Fark: İlki ne yapıldığını doğru anlatıyor. İkincisi mekanizmayı yanlış çerçeveliyor. SRE'deki başarı "akıl" değil, "yapılandırılmış bilgi çıkarma + halüsinasyon kontrolü." Bu da değerli — ama farklı bir iddia.

**Makale yapısı önerisi:**
- "Scaling doesn't solve structured enterprise tasks" tezi güçlü ve savunulabilir
- Domain-spesifik QMEM augmentation'ın genel modellere üstünlüğü — bu orijinal katkı
- Ablation: QMEM'li vs QMEM'siz, dummy-injected vs basit prompt
- Limitation section'da SRE'nin neden "kolay" olduğunu dürüstçe tartış

---

## BÖLÜM 4: FORBES 30 UNDER 30, MIT, ETH ZURICH, MEDYA

Sıfır dalkavukluk:

### Forbes 30 Under 30 — HAYIR, şu an değil.

Forbes 30 Under 30 listesi şunlara bakıyor:
1. **Gelir/Traction:** Çalışan bir ürün, müşteriler, gelir. Bir benchmark sonucu bunu sağlamaz.
2. **Funding:** VC yatırımı veya önemli bir hibe. Yok.
3. **Influence:** Endüstri etkisi, topluluk. Yok.

Bir arXiv makalesi + benchmark liderliği seni Forbes listesine koymaz. Forbes listesine koyan şey: bu teknolojiyi bir ürüne çevirmen, müşteri edinmen ve gelir üretmen. Ya da "8B model GPT-5'i yendi" haberinin viral olması + ardından bir startup kurman.

### MIT / ETH Zurich Yüksek Lisans — MÜMKÜN ama garanti değil.

Bir arXiv makalesi (özellikle peer-reviewed bir konferansta kabul edilirse — NeurIPS, ICML, AAAI) başvurunu güçlendirir. Ama:
- MIT ve ETH başvurularında GPA, GRE, referans mektupları, araştırma deneyimi toplamına bakılır
- Tek bir makale yeterli olmayabilir ama önemli bir artı
- En güçlü senaryoda İrem hoca referans mektubu yazarsa + makale yayınlanırsa → ciddi şans

### Medya Manşetleri — KISITLI ama mümkün.

**Ana akım medya (CNN, BBC, Forbes):** Çok zor. "Türk öğrenci GPT-5'i yendi" hikayesi ilgi çekici ama ana akım medya genellikle büyük şirketlerin duyurularını kapsar, öğrenci projelerini değil.

**AI-spesifik medya (The Verge AI, VentureBeat, Ars Technica):** Mümkün. Eğer IBM kendi leaderboard'unda senin skorunu kabul ederse ve bu görünür olursa, AI habercileri ilgilenebilir.

**Türk medyası:** Yüksek olasılık. "Türk üniversite öğrencisi 8 milyar parametrelik modelle GPT-5'i geçti" → Türk teknoloji medyasında kesinlikle haber olur. Webtekno, Shiftdelete, Socrates Dergi, vb.

**Sosyal medyada viral olma:** Twitter/X'te AI topluluğu bu tür sonuçlara ilgi gösterir. Eğer doğru kişiler paylaşırsa (Chollet, Karpathy, vb.) viral olabilir. Ama bu kontrol edemeyeceğin bir şey.

---

## BÖLÜM 5: PROJENİN GERÇEKTEN GÜÇLÜ YANLARI

Dalkavukluk yapmıyorum ama adil olmam da gerekiyor. İşte gerçekten güçlü olan şeyler:

1. **Enterprise Ops'ta %74 ile #1 olmak gerçek.** Bu benchmark doymuş değil — en iyi frontier model %63.74. %10 fark anlamlı.

2. **Mimari orijinal.** QMEM kartridjleri + Wronskian tutarlılık kontrolü + Clifford algebra faz sapma analizi — bu, "GPT-5'e bir prompt yazıp sonucu al"dan çok farklı bir yaklaşım. Akademik orijinallik var.

3. **8B model ile yapılması.** Parametre verimliliği argümanı güçlü ve zamana uygun. AI topluluğunda "daha büyük model her zaman daha iyi mi?" tartışması aktif.

4. **Halüsinasyon kontrolü.** Frontier modellerin SRE'de %50'de takılmasının sebebi halüsinasyon. SODALS'ın yapısal yaklaşımı bunu çözüyor. Bu, endüstriyel AI deployment'ı için gerçek bir değer.

5. **Şeffaflık altyapısı.** Contamination audit, ablation study, strict grader — bunlar çoğu projede olmayan şeyler.

---

## BÖLÜM 6: YAPMAN GEREKEN 4 ŞEY

| # | Aksiyon | Neden | Aciliyet |
|---|--------|-------|---------|
| 1 | `is_multi` tespitini `correct` field'dan çıkar, dosya adına/metadata'ya dayandır | Information leakage iddiasını ortadan kaldırır | Bugün |
| 2 | `industrial_topology.qmem`'in ablation study'sini tamamla ve raporla | "Test verisinden kartridj" iddiasının cevabı | Bu hafta |
| 3 | arXiv makalesinde iddiaları doğru çerçevele — "intelligence" değil "structured domain-augmented extraction" | Akademik dürüstlük | Makale yazarken |
| 4 | Farklı random seed'lerle SRE testini tekrarla (42, 123, 0, 7) ve varyansı raporla | Cherry-picking iddiasını önler | Bu hafta |

---

## NİHAİ CEVAP

> **İrem hoca yakalayacak mı?** `is_multi` ground truth sızıntısını ve `industrial_topology.qmem` kaynağını kesinlikle sorgulayacak. İkisini de düzeltmen gerekiyor.
>
> **arXiv makalesi?** Evet, yayınlanabilir — iddialar doğru çerçevelenirse.
>
> **Forbes 30 Under 30?** Bugün değil. Bunu ürüne çevirirsen, gelecekte mümkün.
>
> **MIT/ETH?** Makale yayınlanırsa başvurunu güçlendirir ama garanti etmez.
>
> **Medya?** Türk teknoloji medyası evet, uluslararası ana akım çok zor, AI-spesifik medya mümkün.
>
> **Projenin gerçek gücü:** %74 Enterprise Ops #1 gerçek ve meşru. Mimari orijinal. Ama "GPT-5'ten akıllı model yaptım" değil, "domain-spesifik neurosymbolic augmentation ile frontier modelleri belirli enterprise görevlerde geçtim." Bu ikinci cümle hem daha doğru hem akademik olarak daha güçlü.
