# SODALS Determinizm Garanti Spesifikasyonu

> Versiyon: 1.0 | Tarih: 11 Mayıs 2026

## 1. Determinizm Tanımı

SODALS, aşağıdaki koşullar sağlandığında **aynı girdi → aynı çıktı** garantisi verir:

| Koşul | Değer |
|-------|-------|
| `temperature` | `0.0` (greedy argmax) |
| `strict_determinism` | `true` |
| Donanım sınıfı | Aynı GPU mimarisi (örn: NVIDIA Ampere) |
| Model dosyası | Aynı GGUF/Safetensors checkpoint |
| Tokenizer | Aynı `tokenizer.json` |

## 2. Determinizm Katmanları

### Katman 1: Token Sampling (✅ Garanti)
- `temperature=0.0` → softmax → argmax → **tek bir deterministik seçim**
- Sampling randomness **tamamen elimine** edilir
- `ChaCha20Rng` yalnızca security katmanında (encryption) kullanılır, inference path'te **asla**

### Katman 2: RK4 Pondering (✅ Kontrollü)
- `PonderingCore` sabit katsayılar kullanır (`α=0.15, β=0.08, γ=0.05`)
- RK4 integrasyonu sabit adım sayısı (`MAX_PONDER_STEPS=3`)
- Convergence threshold sabit (`EPSILON=1e-4`)
- Floating point aritmetiği aynı donanımda **bitwise deterministik**

### Katman 3: SentinelGuard (✅ Garanti)
- Cosine similarity threshold sabit (`0.85`)
- Debounce counter atomik ve deterministik
- Frekans imzası karşılaştırması `O(128) = O(1)`, dallanma-tahmin-edilebilir

### Katman 4: PII Sanitizer (✅ Garanti)
- Regex pattern'ları `OnceLock` ile tek seferlik derlenir
- Aynı input → aynı redaction output

## 3. Bilinen Sınırlamalar

### IEEE 754 Floating Point Davranışı
GPU'da fused multiply-add (FMA) operasyonlarının sıralama farklılıkları, **farklı GPU mimarileri arasında** (Ampere vs Hopper) son birkaç bitte farklılık yaratabilir. Bu durum:
- Aynı GPU modeli → **Deterministik** ✅
- Aynı GPU mimarisi (farklı model) → **Büyük olasılıkla deterministik** ⚠️
- Farklı GPU mimarisi → **Garanti edilmez** ❌

### Pondering Hassasiyeti
RK4 adımları sırasında birikimli float hataları, çok uzun bağlamlarda (>4K token) nadir durumlarda farklı convergence noktalarına yol açabilir. Bu durum `strict_determinism=true` ile minimize edilir.

## 4. Doğrulama Yöntemi

Determinizm, aşağıdaki test protokolü ile doğrulanır:

```
Protokol: 10x Tekrarlı Çalıştırma
1. Aynı prompt, aynı config, aynı model
2. Her çalıştırmada KV cache temizlenir
3. Tüm çıktılar byte-level karşılaştırılır
4. 10/10 eşleşme = PASS
```

## 5. Endüstriyel Determinizm vs Akademik Determinizm

| Özellik | Akademik | SODALS Endüstriyel |
|---------|----------|--------------------|
| Aynı output | Her cihazda | Aynı cihaz sınıfında |
| Audit trail | Yok | Her token loglanır (`YieldedToken`) |
| Fail-safe | Yok | `SentinelGuard` → `EMERGENCY_STOP` |
| Sorumluluk | Araştırmacı | `FailSafeCommand` + PLC izolasyonu |

> **Sonuç:** SODALS'ın determinizm garantisi, endüstriyel üretim ortamının gerçek ihtiyaçlarına (aynı tesis = aynı donanım) tam olarak uygundur.
