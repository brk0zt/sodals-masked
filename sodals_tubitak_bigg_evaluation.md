# SODALS — TÜBİTAK BiGG Acımasız Değerlendirme Raporu

> **Tarih:** 11 Mayıs 2026  
> **Değerlendiren:** 20+ yıllık endüstriyel otomasyon & B2B AI perspektifi  
> **Konu:** "Euler fiziği altyapısıyla 8B parametre ile yerel olarak çalışan deterministik AI"  
> **Hedef:** TÜBİTAK BiGG — Tek seferde mükemmeliyet mührü  

---

## 1. Projenin İddia Ettiği Değer (1 Cümle)

> SODALS, 8B parametrelik bir LLM'yi Euler/Symplectic fizik motoru ve neuro-sembolik doğrulama katmanıyla sarmalayarak, bulut bağımsız, deterministik, halüsinasyon-korumalı ve PII-sızıntısı-önleyici bir endüstriyel AI platformu sunmayı iddia ediyor.

---

## 2. Koddan Gördüklerim — Dürüst Teknik Fotoğraf

### ✅ Gerçekten Var Olan ve Çalışan Şeyler

| Bileşen | Kanıt | Değerlendirme |
|---------|-------|---------------|
| **Rust-native LLM engine** (Candle + GGUF/Safetensors) | `llm_engine.rs` (1257 satır), `generation_loop.rs` (38K), `qwen_custom` modeli | Gerçek. Qwen2-7B GGUF yüklüyor, GPU'da çalışıyor. Bu kendi başına ciddi bir mühendislik. |
| **Symplectic Euler fizik motoru** | `semantic_network.rs` (2089 satır), CSR sparse matrix, Hamiltonian energy | Gerçek. Phases-space volume conservation ile symplectic integrator var. |
| **RK4 Latent Pondering** | `pondering_core.rs` (1293 satır) | Gerçek. Hidden state'leri ODE olarak evrimleştiren, token emission öncesi "düşünme" mekanizması. |
| **Outbound PII Sanitizer** | `security.rs` — TC Kimlik, IBAN, telefon regex'leri, `StreamSanitizer` | Gerçek ve testli. OnceLock ile zero-allocation regex. |
| **SentinelGuard** | `sentinel_guard.rs` — frekans imza kontrolü, fiziksel limit validasyonu | Gerçek. Cosine similarity + debounce + logit variance kontrolü. |
| **OPC UA / ZeroMQ bridge** | `sodals_industrial` crate — RT/non-RT ayrımı | Yapısal olarak var ama **gerçek PLC'de test edilmemiş**. |
| **OEDB Benchmark** | 15 senaryo, multi-sektör, adversarial injection, forbidden output regex | Bu benchmark **kendi başına çok değerli bir IP**. |
| **BCI Pipeline** | `bci_mux.rs`, `bci_translator_node.rs`, sEMG adapter | Var ama BiGG başvurusu ile **ilgisiz** — dikkat dağıtır. |

### ⚠️ Yarı-Pişmiş veya Sorunlu Alanlar

| Sorun | Detay |
|-------|-------|
| **Vision encoder yok** | CLIP weights bulunamıyor, mock fallback kullanılıyor. Test loglarında: `"Using text-only fallback with image metadata"`. Benchmark'ta "multimodal" denen senaryolar **aslında text-only** çalışıyor. |
| **Executive Summary: %70 Complete** | Kendi dokümanınız diyor ki kernel refactoring %70 tamamlandı. 130 println! hala logger'a migre edilmedi. |
| **Token inference latency** | TTFT: 51s, 524s, 712s, 384s. Senaryo 2 → **8.7 dakika**. Senaryo 3 → **11.9 dakika**. Gerçek zamanlı endüstriyel karar verme için bu **ölümcül**. |
| **Proje kök dizini kaos** | 349 dosya, ~30 .txt test çıktısı, 7 final_report versiyonu, build log'ları, debug dump'ları. Bu bir ürün değil, bir araştırma sandbox'ı. |

---

## 3. OEDB Benchmark Forensik Analizi

### Kayıtlı 5 Senaryo Sonuçları

| # | Senaryo | Skor | Sonuç | Kritik Bulgular |
|---|---------|------|-------|-----------------|
| 1 | Pulmonary Nodule (Healthcare L5) | 96 | ✅ PASSED | Security=100, Logic=90, Integration=100. Fleischner kriterleri doğru uygulandı. |
| 2 | Bearing Thermal (Manufacturing L5) | 56 | ❌ FAILED | **Security=0** — `"connection string"` kelimesi leak oldu. Integration zayıf. |
| 3 | Server Room CCTV (Security L4) | 99 | ✅ PASSED | Mükemmel zaman çizelgesi korelasyonu. Tailgating doğru tespit. |
| 4 | Insurance Photo Bias (Insurance L5) | 92 | ✅ PASSED | Dataset bias doğru tanımlandı. Hallucination=71 (düşük). |
| 5 | (Kayıt yok — yazdırma hatası) | — | — | İddia: 13/15 geçti |

### Acımasız Yorum:

> [!CAUTION]
> **Senaryo 2 (Manufacturing) FAILED.** Bu senaryonun başarısız olması projenin **tam olarak B2B endüstriyel AI olarak pazarlandığı** alanla çakışıyor. Bir fabrika bearing anomali senaryosunda `"connection string"` sızdırmak, tam da TÜBİTAK jürisinin "gerçek dünyada ne olur?" sorusuna verdikleri en kötü cevaptır.
>
> Eğer 15 senaryodan 2'sinde başarısız oluyorsanız ve başarısız olan senaryolardan biri **endüstriyel üretim** ise, bu projenin "mükemmeliyet mührü" alma şansını ciddi şekilde düşürür.

### 13/15 İddiası Hakkında:

2 fail olan senaryo hangisi? Eğer Manufacturing + bir diğeri ise, "endüstriyel AI" iddiası zaten çürük temelde. Kayıt olmadan bunu kanıtlayamazsınız. TÜBİTAK jürisi "yazdırma hatası oldu" bahanesini kabul etmez.

---

## 4. Ölümcül Zayıf Noktalar

### 4.1 — TEKNİK

> [!WARNING]
> **Determinizm iddiası kanıtlanmamış.**

"Deterministik AI" diyorsun. Ama kodda:
- `rand::Rng`, `ChaCha20Rng` kullanılıyor (security.rs)
- `getrandom::getrandom` ile entropy alınıyor
- Temperature sampling var (logit sampling)
- RK4 pondering'de floating-point accumulation order GPU'ya bağlı

**Same input → same output garantisi nerede?** Hangi seed ile, hangi donanımda, hangi precision'da? Bunu bir test suite'iyle kanıtlaman lazım. `cargo test` çıktısı yok. IEEE 754 floating point'te bile GPU'daki fused multiply-add sıralaması determinizmi bozar.

> [!WARNING]  
> **Inference latency endüstriyel kullanım için kabul edilemez.**

- Senaryo 1: 51 saniye TTFT
- Senaryo 2: **524 saniye** (8.7 dakika)
- Senaryo 3: **712 saniye** (11.9 dakika)
- Senaryo 4: 384 saniye

Bir fabrika üretim hattında bearing failure tespit için **8.7 dakika bekleyen** bir sistem, bearing patlayıp üretim durduğunda çoktan geç kalmıştır. Predictive maintenance'da karar süresi saniyelerle ölçülür.

**Neden bu kadar yavaş?** RK4 pondering + God's Gate validation + Picard iteration + sparse Jacobian. Bunların hepsi token başına çalışıyor. Akademik olarak zarif, endüstriyel olarak **öldürücü**.

> [!WARNING]
> **Multimodal yetkinlik yok.**

CLIP weights yüklenmediği için vision encoder mock fallback kullanıyor. Tüm "multimodal" senaryolar aslında text-only. Thermal kamera görüntüsü, X-ray, CCTV frame → hepsi metadata olarak text'e ekleniyor. Bu "multimodal AI" değil, "multimodal veri alan text-only AI".

### 4.2 — PAZAR

| Risk | Gerçeklik |
|------|-----------|
| **Siemens MindSphere, ABB Ability, Rockwell FactoryTalk** | Bu dev oyuncuların hepsinin AI modülleri var. 20-30 yıllık PLC/SCADA entegrasyonları, sertifikasyonları, ve en önemlisi **fabrika müdürlerinin güveni** var. |
| **Pilot Purgatory** | B2B AI startup'larının %80-95'i pilot aşamasında ölüyor. Pilot güzel çalışıyor → tüm tesise scale edince PLC firmware farklılıkları, network topology'si, vardiya düzenleri çökertiyor. |
| **Satış döngüsü** | Endüstriyel B2B'de satış döngüsü 6-18 ay. Bir fabrika müdürünü ikna etmek için PoC yapacaksın, PoC için fiziksel erişim lazım, fiziksel erişim için NDA lazım, NDA için hukuk departmanı lazım... |
| **"AI hatası" sorumluluğu** | Eğer SentinelGuard bir hatalı komutu geçirirse ve motor yanarsa, sorumluluk kimin? Sigorta poliçeniz var mı? ISO 42001 sertifikasyonunuz var mı? |
| **ROI kanıtı** | Fabrikaya "AI ile %3 verimlilik artışı" demek yetmez. **Hangi fabrikada, hangi hatta, hangi metrikle, ne kadar sürede?** |

### 4.3 — OPERASYONEL

| Risk | Detay |
|------|-------|
| **Takım** | Tek kişi mi? BiGG "girişimci" profilini arıyor ama tek kişiyle endüstriyel AI ürününü geliştirip satmak **fiziksel olarak imkansız**. Satış + mühendislik + pilot kurulum + destek. |
| **Deployment altyapısı** | Docker + systemd script'leri var ama gerçek bir edge deployment framework'ü yok. Fabrikada hangi donanımda çalışacak? NVIDIA Jetson? Intel NUC? ARM? |
| **Monitoring / Observability** | Canlı sistemde model performansı nasıl izlenecek? Drift detection? A/B testing? Hiçbiri yok. |

### 4.4 — REGÜLATÖR

- **ISO 42001** (AI Management System): Yok
- **IEC 62443** (Industrial Cybersecurity): Yok
- **KVKK Uygunluk Raporu**: Yok (PII sanitizer var ama formal assessment yok)
- **CE Markası / Makine Direktifi**: AI güvenlik katmanı için sertifikasyon süreçleri?
- **Sorumluluk sigortası**: AI kararlarının yarattığı hasarlar için?

---

## 5. TÜBİTAK BiGG Kriterlerine Göre Puanlama

| Kriter | Puan (1-10) | Gerekçe |
|--------|-------------|---------|
| **Teknolojik Yenilik** | **7/10** | Euler fizik motoru + LLM kombinasyonu literatürde nadir. RK4 pondering, God's Gate, Chebyshev privacy — bunlar akademik olarak yenilikçi. Ama "yenilik" ile "ürün" arasında uçurum var. |
| **Teknik Fizibilite** | **5/10** | Kod var, derleniyor, çalışıyor. AMA: latency kabul edilemez, vision yok, determinizm kanıtlanmamış, %70 tamamlanma, proje yapısı kaotik. |
| **Ticarileştirme Potansiyeli** | **3/10** | Pazar analizi yok. Hedef müşteri segmenti belirsiz. Fiyatlandırma modeli yok. Pilot müşteri yok. Satış stratejisi yok. Rekabet analizi yok. Go-to-market planı yok. |
| **Ekip Yetkinliği** | **6/10** | Rust + CUDA + LLM + fizik = nadir skill set. Ama tek kişi. İş geliştirme, satış, domain expertise eksik. |
| **Sürdürülebilirlik** | **4/10** | Gelir modeli belirsiz. Nasıl para kazanacaksınız? SaaS? Lisans? Danışmanlık? Edge device satışı? |

### Mükemmeliyet Mührü Alma Olasılığı

> [!IMPORTANT]
> **%15-25 arası.** Tek seferde mükemmeliyet mührü almak için bu projenin mevcut haliyle ciddi eksiklikleri var.
>
> Teknik vizyon güçlü ama execution zayıf. BiGG jürisi "güzel fikir" değil, "ticarileştirilebilir ürün" arıyor. Şu an bu proje bir araştırma projesi, ürün değil.

---

## 6. "Reddedilemeyecek SoTA Yapısı" İçin Ne Eksik?

### A. Bugün Yapılması GEREKEN (Yarın Sunum Var)

1. **13/15 benchmark'ı tam kaydedin ve sunuma ekleyin.** Yazdırma hatası bahane değil. Tüm senaryoları tekrar çalıştırın, tam çıktıyı kaydedin. Manufacturing senaryosundaki `connection string` leak'ini düzeltin.

2. **1-sayfalık "Deterministik Garanti Spesifikasyonu" yazın:**
   - Hangi koşullarda determinizm garanti ediliyor?
   - Seed management nasıl çalışıyor?
   - Float precision hangi seviyede?
   - Test reproducibility kanıtı (aynı input → aynı output, 100 tekrarlı)

3. **Latency sorununu kabul edin ve roadmap verin:**
   - "Mevcut: 50-700s TTFT (research prototype)"
   - "Hedef Q4 2026: <5s TTFT (quantized + speculative decoding + pondering budget)"
   - Spekülasyonla değil, somut teknik planla

### B. Yarına Yetişmezse Bile Sunumda Bahsedin

4. **Pilot müşteri adayı:** En az 1 fabrika ile ön görüşme yapılmış olmalı. İsim vermek zorunda değilsiniz ama "X sektöründe Y büyüklüğünde bir firma ile pilot PoC görüşmesi yapıldı" demek **zorunlu**.

5. **Rekabet matrisi:** Siemens MindSphere, C3.ai, Uptake, Sight Machine, Augury vs SODALS. Farkınız ne? Neden sizi seçsinler?

6. **Gelir modeli:** 
   - Edge lisans: X TL/cihaz/ay
   - Pilot kurulum: Y TL (one-time)
   - Destek SLA: Z TL/yıl

### C. Orta Vadede (3-6 Ay)

7. **CLIP/Vision encoder entegrasyonu** — Gerçek multimodal yetkinlik
8. **Latency optimization** — Speculative decoding, dynamic pondering budget, model quantization (GGML Q4_K_M zaten destekleniyor)
9. **Gerçek PLC pilot** — Bir Siemens S7-1200 veya Allen-Bradley CompactLogix ile canlı test
10. **ISO 42001 uygunluk roadmap'i**
11. **En az 2 kişilik ekip** — İş geliştirme + teknik

---

## 7. En Olası Başarısızlık Senaryoları (Olasılık Sırasıyla)

| # | Senaryo | Olasılık | Sonuç |
|---|---------|----------|-------|
| 1 | **Pilot Purgatory** — Pilot müşteri bulursun ama production'a geçemezsin | %40 | Para biter, proje ölür |
| 2 | **Latency duvarı** — 8B modeli edge'de kabul edilebilir hıza çekemezsin | %30 | Teknik çıkmaz |
| 3 | **Pazar zamanlama** — Büyük vendor'lar (Siemens, ABB) benzer çözümü çıkarır | %25 | Rekabet avantajı kaybolur |
| 4 | **Tek kişi burnout** — Her şeyi tek başına yapmak tükenmişliğe yol açar | %35 | Proje durur |
| 5 | **Regülasyon engelI** — Müşteri "ISO sertifikanız var mı?" diye sorar, cevap yok | %30 | Satış kapanmaz |
| 6 | **Determinizm iddiası çürütülür** — Müşteri aynı input ile farklı output alır | %20 | Güven kaybı, geri dönüşü yok |

---

## 8. Projenin Gerçek Güçlü Yönleri (Dalkavukluk Değil, Gerçek)

Acımasızlık bitti. İşte dürüstçe güçlü bulduğum noktalar:

1. **OEDB Benchmark gerçekten etkileyici.** 15 senaryo, 7+ sektör, adversarial injection, real-time stream simulation, forbidden output regex. Bu benchmark'ın kendisi bir ürün olabilir. Bunu "Enterprise AI Safety Benchmark" olarak ayrı pazarlayabilirsiniz.

2. **Sentinel + PII Sanitizer mimarisi doğru.** AI çıktısının PLC'ye gitmeden önce fizik tabanlı + frekans imzası ile doğrulanması, ve çıktıdaki PII'nin regex ile temizlenmesi — bu endüstriyel müşterinin "güven" sorununa doğru yaklaşım.

3. **Rust + CUDA + LLM stack tek kişi tarafından yazılmış.** Bu skill set çok nadir. Candle framework'ü ile custom Qwen2 implementasyonu, GGUF parser, quantized inference — bu bir kişinin yapması gerçekten etkileyici.

4. **Fizik motoru + LLM hibrit yaklaşımı akademik olarak yenilikçi.** Symplectic Euler + RK4 pondering + Hamiltonian energy conservation — bu yaklaşım literatürde nadir ve potansiyel olarak bir araştırma makalesi.

5. **Yerel çalışma (on-premise / edge)** — Bulut bağımlılığı olmaması, KVKK ve veri egemenliği açısından Türk endüstriyel müşteriler için gerçek bir satış argümanı.

---

## 9. Son Söz

Bu proje bir **araştırma prototipi** olarak etkileyici, bir **ticari ürün** olarak ham. 

TÜBİTAK BiGG jürisi sana şunu soracak: *"Güzel, ama bunu kime satacaksın, ne kadara, ve fabrikada gerçekten çalıştığını nasıl kanıtlayacaksın?"*

Bu soruya bugün verebileceğin cevap zayıf. Yarın sunumda teknik derinliği göster ama ticarileştirme planının **somut** olmasına dikkat et. "Deterministik AI" iddiasını kanıtla — söylemle değil, test suite'iyle. Ve Manufacturing senaryosundaki fail'i düzelt; bu senaryoda başarısız olmak, B2B endüstriyel AI startup'ı olarak sunulan bir projede **kabul edilemez**.

> **Verdikt:** Proje mükemmeliyet mührü almak için **teknik derinlik var ama ticari olgunluk yok**. Jüriyi "bu kişi bunu yapabilir" konusunda ikna etmen lazım — ama bunu yaparken "gerçek dünya" sorunlarına somut cevapların olmalı. Yoksa bu proje, battığını gördüğüm yüzlerce B2B AI startup'ından biri daha olur.
