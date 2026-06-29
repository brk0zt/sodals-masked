# SODALS Rekabet Matrisi

> Versiyon: 1.0 | Tarih: 11 Mayıs 2026

## Endüstriyel AI Pazar Karşılaştırması

| Özellik | **SODALS** | Siemens MindSphere | C3.ai | Sight Machine | Augury | Uptake |
|---------|-----------|-------------------|-------|---------------|--------|--------|
| **Dağıtım Modeli** | ✅ Tam on-premise/edge | ❌ Bulut zorunlu | ❌ Bulut | ❌ Bulut | ⚠️ Hibrit | ❌ Bulut |
| **Veri Egemenliği (KVKK)** | ✅ Veri tesisten çıkmaz | ❌ Veri buluta gider | ❌ | ❌ | ⚠️ | ❌ |
| **LLM Tabanlı Analiz** | ✅ 8B yerel LLM | ❌ Rule-based | ⚠️ GPT API (bulut) | ❌ Klasik ML | ❌ | ❌ |
| **Fizik Tabanlı Doğrulama** | ✅ Euler + Sentinel | ❌ | ❌ | ❌ | ⚠️ Vibrasyon ML | ❌ |
| **Deterministik Çıktı** | ✅ Garanti (temp=0) | N/A | ❌ GPT non-det | N/A | N/A | N/A |
| **PII Otomatik Sanitizasyon** | ✅ Regex + Stream | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Halüsinasyon Koruması** | ✅ God's Gate + Sentinel | N/A | ❌ | N/A | N/A | N/A |
| **OPC UA Entegrasyonu** | ✅ Deterministik polling | ✅ Native | ⚠️ Connector | ⚠️ | ❌ | ⚠️ |
| **Kurulum Süresi** | Saatler | Aylar | Aylar | Haftalar | Haftalar | Haftalar |
| **Aylık Lisans (tesis)** | ~$500 | ~$5,000+ | ~$10,000+ | ~$3,000+ | ~$2,000+ | ~$3,000+ |
| **İnternet Bağımlılığı** | ❌ Yok | ✅ Zorunlu | ✅ Zorunlu | ✅ Zorunlu | ⚠️ Kısmi | ✅ Zorunlu |

## SODALS'ın Rekabet Avantajları

### 1. Veri Egemenliği (Regulatory Moat)
Türk fabrikalarında KVKK uyumu kritik. SODALS, verinin tesis dışına çıkmamasını garanti eden **tek çözüm**. Rakiplerin tümü bulut bağımlı — bu Türk regülasyonlarında risk oluşturur.

### 2. Maliyet Avantajı (10x Fark)
Büyük vendorların yıllık lisans maliyeti $36,000-$120,000/tesis. SODALS edge lisansı $6,000/yıl → **6-20x maliyet avantajı**.

### 3. Hız Avantajı (Kurulum)
Siemens MindSphere kurulumu aylar sürer (IT altyapısı, bulut entegrasyonu, güvenlik auditi). SODALS edge cihaz + OPC UA → **saatler içinde çalışır**.

### 4. Teknolojik Fark (Physics-Guided AI)
Rakiplerin hiçbiri LLM çıktısını fizik motoru ile doğrulamıyor. SODALS'ın Euler/Symplectic + SentinelGuard katmanı, endüstriyel AI'da **benzersiz bir güvenlik mekanizması**.

## Zayıf Noktalar (Dürüst Değerlendirme)

| Rakip Avantajı | SODALS Durumu |
|----------------|---------------|
| 20+ yıllık müşteri güveni (Siemens) | Yeni, henüz pilot müşteri yok |
| Geniş ekosistem (yüzlerce connector) | Yalnızca OPC UA + ZeroMQ |
| Kurumsal destek (7/24) | Tek kişilik ekip |
| Sertifikasyonlar (ISO, IEC) | Henüz sertifikasyon süreci başlamamış |
