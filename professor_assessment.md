# SODALS Değerlendirme — Profesör İrem Ersöz Kaya'ya Sunum

**Tarih:** 24 Haziran 2026  
**Hazırlayan:** Kod Analizi (Bağımsız Denetim Perspektifi)  
**Amaç:** İrem Hoca "bunda bir hile olmalı" dediğinde ne bulur? Bu rapor onu önceden simüle eder.

---

## 1. SONUÇLARIN ÖZETİ

| Benchmark | Skor | Sıra | Fark |
|-----------|------|------|------|
| ITBench (SRE + FinOps avg) | %83.33 | **#1** | Gemini-3 Pro'dan +25 puan |
| AssetOps (FSIQ) | %64.67 | #9 | Claude Sonnet 4.5'ten +0.47 |
| **Enterprise Ops Average** | **%74.00** | **#1** | Gemini-3 Pro'dan **+10.26 puan** |

---

## 2. İREM HOCA NE SORAR? — DÜRÜST CEVAPLAR

### Soru 1: "SRE'de %92 mi? Bu 8B model GPT-5'i 48 puan farkla mı yenmiş? Saçmalık."

**İrem Hoca haklı olarak şüphelenecek. Ama şu anda kodda olan mekanizmayı açıklayalım:**

Kodun son halinde ([sre_runner.py satır 119-161](file:///c:/Users/Burak/Desktop/sodals_core/scripts/benchmark/sre_runner.py#L119-L161)) **dummy normal alert injection** yapılıyor:

```python
# 15 adet bilinen servis ismi listesi
all_services = ["frontend-service", "ad-service", "shipping-service", ...]

# Gerçek firing alert'ler + dummy "normal" alert'ler karıştırılıyor
for s in all_services:
    if s not in firing_services:
        normal_alerts.append({"alertname": "ServiceStatusNormal", "service": s, "status": "normal"})

# Karıştırılıp birlikte sunuluyor
random.shuffle(injected_alerts)
```

**Bu iyi bir adım.** Model artık tek bir service name'i kopyalayamıyor — 12 alert arasından "firing" olanı seçmesi gerekiyor. Bu, entity extraction'dan classification'a geçiş.

**Ama İrem Hoca şunu soracak:** "Firing alert'ler `status: firing` yazıyor, dummy'ler `status: normal` yazıyor. Model sadece `status == firing` olan alert'in service adını mı kopyalıyor?"

**Dürüst cevap: Evet, büyük ihtimalle.** Bu hâlâ **filtreleme görevi** — reasoning değil. Prompt'ta `"status": "firing"` vs `"status": "normal"` yazdığı sürece, model bir if/else mantığıyla doğru cevabı buluyor. GPT-5'in aynı görevi yapamaması, GPT-5'in daha kötü olduğu anlamına gelmez — GPT-5 muhtemelen fazla reasoning yapıp halüsinasyon üretiyordur.

> [!WARNING]
> **Akademik Dürüstlük Notu:** SRE %92, modelin "SRE uzmanı" olduğunu kanıtlamaz. Modelin **yapılandırılmış JSON'dan status=firing filtresi yapıp entity name extract etmekte** iyi olduğunu kanıtlar. arXiv makalesinde bunu böyle çerçevelemek gerekir. "SRE reasoning'de lider" yerine "structured alert triage'da lider" demek daha doğru.

---

### Soru 2: "FailureSensorIQ'da `is_multi` bilgisini nereden alıyorsunuz?"

**Bu kritik bir soru.** Kodun son halini inceledim ([failuresensoriq_runner.py satır 449](file:///c:/Users/Burak/Desktop/sodals_core/scripts/benchmark/failuresensoriq_runner.py#L449)):

```python
is_multi = "multi" in Path(dataset_path).name.lower()
```

**Verdict: TEMİZ.** `is_multi` bilgisi dosya adından (`all_multi_answers.jsonl` vs `all.jsonl`) tespit ediliyor — ground truth `correct` field'ından değil. Bu, IBM'in kendi dataset yapısını kullanmak demek. Herhangi bir yarışmacı da aynı bilgiye sahip.

**Ama İrem Hoca şunu soracak:** "Diğer yarışmacılar da multi-answer sorularda farklı prompt mu kullanıyor?"

**Dürüst cevap:** Bilmiyoruz. Leaderboard'daki modellerin çoğu **tek prompt** ile hem single hem multi soruları çözüyor. SODALS'ın soru tipine göre prompt değiştirmesi **haksız avantaj değil** (bilgi herkese açık) ama **standart uygulama da değil**. arXiv'de bu ayrımın şeffafça belirtilmesi gerekir.

---

### Soru 3: "industrial_topology.qmem nereden geliyor?"

[generate_industrial_topology.py](file:///c:/Users/Burak/Desktop/sodals_core/scripts/benchmark/generate_industrial_topology.py) FailureSensorIQ veri setindeki `correct` field'larını okuyup `"sensör" MEASURES "arıza"` formatında QMEM DSL'e çeviriyor.

**İrem Hoca'nın göreceği:**
1. Script, FailureSensorIQ `all.jsonl` dosyasından doğru cevapları okuyor (satır 54-65)
2. `industrial_topology.qmem` (8.8 MB) aktif olarak kök dizinde duruyor
3. Bu kartridj modelin logit'lerini etkiliyor

**En kötü senaryo yorumu:** "Test setinden doğru cevapları çıkarıp modele öğretiyorsunuz."

**En iyi senaryo yorumu:** "Standard mühendislik bilgisini (sensör-arıza korelasyonları) QMEM formatında kodluyorsunuz. Bu bilgiler herhangi bir condition monitoring textbook'unda var."

**Gerçek:** İkisinin arasında. Bilgi kaynağı benchmark'ın kendisi olduğu için "textbook knowledge" savunması zayıf. Ama çıkarılan bilgi soru-cevap mapping'i değil, genel fizik ilişkileri olduğu için "answer key leakage" de değil.

> [!CAUTION]
> **İrem Hoca'nın en sert eleştirisi buraya gelecek.** Çözüm:
> 1. Ablation study'yi çalıştırın: kartridjli vs kartridjsiz (`run_ablation_study.py` zaten var)
> 2. Kartridjsiz skoru da raporlayın
> 3. Eğer kartridjsiz skor da %50+ ise, kartridj "boost" sağlıyor ama "hile" değil
> 4. Eğer kartridjsiz skor %20 ise, kartridj *esasen cevapları sağlıyor* demek — bu savunulması çok zor

---

### Soru 4: "FinOps'ta prompt'a 'ilk kaynak en muhtemel' yazmışsınız. Bu cevabı vermek değil mi?"

Kodun son halini inceledim ([finops_runner.py satır 120-132](file:///c:/Users/Burak/Desktop/sodals_core/scripts/benchmark/finops_runner.py#L120-L132)):

```python
user_content = f"""...
Task:
Identify the resource name or ID that is causing the cost anomaly from the 
cost_comparison_data. You must analyze the metrics (mean_hist, anomaly_cost, 
and diff) for each resource to determine which one exhibits the most 
significant unexpected cost increase.

CRITICAL: Output ONLY that resource name wrapped in <answer> tags.
"""
```

**Verdict: Son halde TEMİZ.** "Analyze the metrics" diyor, "ilk eleman cevaptır" demiyor. Model telemetri verisini analiz etmeli. Bu fair.

---

### Soru 5: "Örneklem büyüklüğü yeterli mi?"

| Benchmark | Soru Sayısı | İstatistiksel Güç |
|-----------|-------------|-------------------|
| SRE | **16 senaryo** (ITBench v8'de) | ⚠️ ÇOK DÜŞÜK |
| FinOps | **15 senaryo** | ⚠️ ÇOK DÜŞÜK |
| FSIQ | ~8,296 soru | ✅ Yeterli |

> [!WARNING]
> **SRE'de 16 senaryo var.** Her bir senaryo %6.25 ağırlık taşıyor. 1 soru fazla/eksik doğru = %6.25 puan farkı. ITBench'teki %83.33 skoru, **birkaç şanslı/şanssız tahmin** ile %70-95 arasında salınabilir.
>
> İrem Hoca bunu anında görecek: "16 soruluk testte istatistiksel anlamlılık yok."
>
> **Savunma:** Bu IBM'in kendi benchmark yapısı. Tüm rakipler de aynı 16 senaryo üzerinde değerlendiriliyor. Aynı koşullar altında SODALS lider — ama evet, confidence interval geniş.

---

## 3. GERÇEKTEN HİLE VAR MI?

### Kod Taraması Sonuçları

| Kontrol | Sonuç | Detay |
|---------|-------|-------|
| Rust kodunda benchmark if/else | ✅ TEMİZ | 0 eşleşme |
| Hardcoded cevap | ✅ TEMİZ | 0 eşleşme |
| Ground truth sızıntısı (is_multi) | ✅ TEMİZ | Dosya adından, ground truth'tan değil |
| FinOps ground_truth leak | ✅ TEMİZ | `del cleaned_data["ground_truth"]` (satır 113) |
| FinOps anomaly.resource_id leak | ✅ TEMİZ | `del cleaned_data["anomaly"]["resource_id"]` (satır 117-118) |
| SRE dummy alert injection | ✅ TEMİZ | Görevi zorlaştırıyor |
| industrial_topology.qmem kaynağı | ⚠️ GRİ | Test verisinden türetilmiş domain bilgisi |
| Contamination audit aracı | ✅ Mevcut | `contamination_audit.py` |
| Ablation study aracı | ✅ Mevcut | `run_ablation_study.py` |

**Hile yok mu?** Hayır, klasik anlamda hile yok — hardcoded cevap, answer key lookup, conditional cheating yok. Ama **gri alanlar var** ve bunlar şeffafça raporlanmazsa, İrem Hoca bunları "kasıtlı olmasa bile metodolojik hata" olarak değerlendirebilir.

---

## 4. DOCKER KAPALI KUTU TESTİ İREM HOCA'YI TATMİN EDER Mİ?

### Docker Testi Neyi Kanıtlar?

- ✅ Modelin gerçekten çalıştığını
- ✅ Inference'ın reproducible olduğunu
- ✅ Hardcoded cevap olmadığını
- ✅ Sonuçların tutarlı olduğunu

### Docker Testi Neyi Kanıtlamaz?

- ❌ `industrial_topology.qmem`'in kaynağının etik olduğunu
- ❌ SRE'deki başarının "reasoning" olduğunu
- ❌ 16 soruluk testte istatistiksel anlamlılığı
- ❌ Prompt engineering'in "haksız avantaj" olup olmadığını

> **İrem Hoca Docker testi ile "hile yok" konusunda tatmin olabilir ama "bu gerçekten zekâ mı?" sorusunu soracak. Docker bunu yanıtlamaz.**

---

## 5. arXiv MAKALESİ YAYINLANABİLİR Mİ?

### "Ölçeklendirme Yasası Yetersiz" Tezi

Bu tez şunu iddia ediyor: "8B parametrelik neurosymbolic model, domain bilgisi ile donatıldığında, trilyon parametrelik genel amaçlı modelleri enterprise IT operasyonlarında geçer."

**Bu iddia doğru mu?**

| Boyut | Durum | Açıklama |
|-------|-------|----------|
| Enterprise Ops #1 | ✅ Veri destekliyor | %74 vs Gemini-3 %63.74 |
| SRE #1 | ✅ Veri destekliyor ama... | ...görev yapısı basit (alert filtreleme) |
| AssetOps #9 | ⚠️ Ortalama | Frontier modellerden geride |
| "Scaling yetersiz" tezi | ⚠️ Zayıf | Çünkü başarı scaling'i yenmekten değil, **task-specific optimization**'dan geliyor |

### Peer Review'da Gelecek Eleştiriler

1. **"SRE task trivial"** — Reviewer, SRE görevinin yapılandırılmış JSON'dan entity extraction olduğunu, reasoning gerektirmediğini söyleyecek. **Haklı.**

2. **"16 soruda istatistiksel anlamlılık yok"** — Reviewer, confidence interval hesaplaması isteyecek. 16 soruda %83 skor → 95% CI yaklaşık [%58, %96]. Bu çok geniş. **Haklı.**

3. **"QMEM kartridji data contamination"** — Reviewer, industrial_topology.qmem'in kaynağını soracak. Ablation study olmadan bu sorunu savunmak zor. **Haklı.**

4. **"AssetOps'ta #9 — bu scaling'i yenmiyor"** — Enterprise Ops ortalaması SRE'nin aşırı yüksek skoruyla çarpıtılmış. AssetOps'ta tek başına bakıldığında SODALS ortalamanın altında. **Haklı.**

5. **"Prompt engineering fair mi?"** — Multi-answer sorulara farklı prompt, SRE'ye dummy alert injection — bunlar engineering kararları ama rakipler bunları yapmamış olabilir. **Kısmen haklı.**

### arXiv'de Yayınlanabilir mi?

**Evet — ama tezi daraltmanız gerekiyor.**

| Tez | Yayınlanabilir mi? |
|-----|-------------------|
| ❌ "8B model scaling yasasını yendi" | Hayır — AssetOps #9 bunu çürütüyor |
| ❌ "8B model GPT-5'ten akıllı" | Hayır — SRE task yapısı bunu desteklemiyor |
| ✅ "Domain-specific neurosymbolic augmentation, IT operasyonlarında küçük modellerin performansını frontier seviyeye çıkarır" | **Evet** — bu kanıtlanabilir ve savunulabilir |
| ✅ "QMEM + structured alert processing, SRE incident triage'da SOTA" | **Evet** — dar ve doğru bir iddia |
| ✅ "Neurosymbolic hybrid architecture enables efficient enterprise AI deployment" | **Evet** — verimlilik ve deployment perspektifi |

---

## 6. KAGGLE IBM YETKİLİLERİ TASFİK EDER Mİ?

IBM'in Enterprise Operations benchmark'ı Kaggle üzerinde yayınlanıyor. IBM yetkilileri:

1. **Docker konteynerini çalıştıracak** → Sonuçlar tutarlıysa ✅
2. **Kodu inceleyecek** → Hardcoded cevap yok ✅
3. **QMEM kartridjini soracak** → Şeffaf belgelenmediyse ⚠️
4. **SRE task yapısını biliyorlar** — IBM bu benchmark'ı kendisi tasarladı. Alert JSON'dan entity extraction'ın kolay olduğunu biliyorlar. Yüksek SRE skorlarına şaşırmayabilirler ama **neden kendi modellerinin bunu yapamadığını** soracaklar.

**Tasdik olasılığı:** Docker testi geçerse ve QMEM kaynağı şeffafça belgelenirse, teknik tasdik **muhtemel**. Ama IBM yetkilileri muhtemelen "interesting engineering approach" diyecek, "revolutionary AI breakthrough" değil.

---

## 7. NİHAİ VERDİCT — SIFIR DALKAVUKLUK

### Güçlü Yanlar (gerçek)
1. **Enterprise Ops #1 gerçek.** Aynı benchmark, aynı koşullar, Gemini-3'ten 10 puan önde.
2. **Kod temiz.** Hardcoded cevap yok, ground truth sızıntısı yok.
3. **8B modelin frontier seviyede performansı gerçek.** AssetOps #9 bile, Qwen 235B'den (#14) ileride.
4. **Mühendislik kalitesi yüksek.** Dummy alert injection, thinking block stripping, strict grading — bunlar profesyonel iş.

### Zayıf Yanlar (saklamamanız gereken)
1. **SRE başarısı "reasoning" değil, "structured data filtering."** bunu arXiv'de doğru çerçevelemelisiniz.
2. **16 soruluk SRE testinde istatistiksel anlamlılık zayıf.** Confidence interval raporlayın.
3. **industrial_topology.qmem kaynağı gri alan.** Ablation study sonuçları olmadan savunulması zor.
4. **"Scaling yasası yetersiz" tezi çok güçlü bir iddia.** Veriler bunu tam desteklemiyor — AssetOps #9 buna ters düşüyor.
5. **Enterprise Ops Average, SRE'nin çarpıtıcı etkisiyle yükseliyor.** İrem Hoca bileşenlere ayrıştırınca görecek.

### Sonuç

> **Bu proje, 8B parametrelik bir modelin domain-spesifik augmentation ile enterprise IT operasyonlarında frontier modellere yakın ve bazı görevlerde üstün performans gösterebileceğini kanıtlıyor. Bu, yayınlanabilir ve değerli bir akademik katkıdır.**
>
> **Bu proje, "scaling yasasını yendiğini" veya "GPT-5'ten akıllı olduğunu" kanıtlamıyor. Bu iddiaları yaparsanız, peer review'da reddedilir ve İrem Hoca'yı tatmin etmez.**
>
> **Aradaki fark, başarı ile başarıyı nasıl çerçevelediğiniz arasındaki farktır.**

---

## 8. İREM HOCA'YI TATMİN ETMEK İÇİN YAPILMASI GEREKENLER

| # | Aksiyon | Süre | Neden |
|---|--------|------|-------|
| 1 | **Ablation study çalıştır** (kartridjli vs kartridjsiz) | 2-3 saat | industrial_topology savunması için zorunlu |
| 2 | **SRE confidence interval hesapla** | 30 dk | 16 soruda n küçük — CI raporlanmalı |
| 3 | **arXiv tezini daralt** | — | "Scaling yetersiz" → "Domain augmentation effective" |
| 4 | **SRE'yi "alert triage" olarak çerçevele** | — | "SRE reasoning" demeyin |
| 5 | **AssetOps #9'u açıkça raporla** | — | Cherry-picking yapmayın |
| 6 | **QMEM kaynağını belgele** | 30 dk | Şeffaflık en güçlü savunma |
