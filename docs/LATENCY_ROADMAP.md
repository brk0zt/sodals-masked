# SODALS Latency Optimization Roadmap

> Versiyon: 1.0 | Tarih: 11 Mayıs 2026

## Mevcut Durum (Baseline)

| Metrik | Değer | Koşullar |
|--------|-------|----------|
| TTFT (Time to First Token) | 50-700s | 8B model, CUDA, RK4 pondering aktif |
| Token/saniye | ~2-5 tok/s | Autoregressive decoding |
| Pondering overhead | ~30-40% | 3-step RK4 + God's Gate |

## Neden Bu Kadar Yavaş?

Mevcut pipeline'da her token üretimi sırasında:
1. **RK4 Pondering** (3 step × 4 k-vektör = 12 forward-benzeri hesaplama)
2. **God's Gate Validation** (cosine similarity + Picard iteration)
3. **Oracle Sidecar** (sparse Jacobian + semantic divergence)
4. **Ghost Vector Injection** (embedding lookup + L2 normalization)

Bu katmanlar akademik doğruluk için kritik, ancak endüstriyel deployment'ta **dinamik bütçeleme** gerektirir.

## Optimization Roadmap

### v1.1 — Dynamic Pondering Budget (Q3 2026)
**Teknik:** Chimera Router'ın `TaskPlan` analizi zaten sorgu kompleksitesini tespit ediyor. Basit sorgularda (factual Q&A, lookup) pondering'i bypass et.

```
Basit sorgu: pondering=0, God's Gate=skip → -60% latency
Orta sorgu:  pondering=1 step → -30% latency  
Karmaşık:   pondering=3 step (mevcut) → baseline
```

**Beklenen İyileşme:** Basit sorgularda TTFT 50s → 20s

### v1.2 — Speculative Decoding (Q3 2026)
**Teknik:** 1B draft model ile 4-8 token spekülatif üretim, 8B ana model ile doğrulama.

**Beklenen İyileşme:** -40% token generation süresi

### v1.3 — KV-Cache Quantization (Q4 2026)
**Teknik:** KV cache'i INT8'e quantize ederek VRAM kullanımını düşür → daha büyük batch / daha uzun context.

**Beklenen İyileşme:** -30% VRAM → paralel sorgu desteği

### v2.0 — Task-Specific Model Distillation (Q1 2027)
**Teknik:** 8B genel modelden 3B task-specific modeller distill et (prediktif bakım, kalite kontrol, anomali tespiti).

**Beklenen İyileşme:** TTFT 50s → 5-10s

### v2.1 — TensorRT/ONNX Runtime Backend (Q2 2027)
**Teknik:** Candle backend'i yerine NVIDIA TensorRT veya ONNX Runtime ile inference.

**Beklenen İyileşme:** -50% inference süresi

## Hedef Zaman Çizelgesi

```
2026 Q3: v1.1 + v1.2 → TTFT < 15s (basit), < 60s (karmaşık)
2026 Q4: v1.3         → Paralel sorgu desteği
2027 Q1: v2.0         → TTFT < 5s (task-specific)
2027 Q2: v2.1         → Production-grade performance
```

## Endüstriyel Bağlam

Predictive maintenance'da karar süreleri:

| Kullanım | Kabul Edilebilir Latency | SODALS Mevcut | SODALS v2.0 Hedef |
|----------|------------------------|---------------|-------------------|
| Alarm değerlendirme | < 30s | ⚠️ 50-700s | ✅ < 5s |
| Vardiya raporu | < 5 dk | ✅ Uygun | ✅ < 30s |
| Root cause analysis | < 30 dk | ✅ Uygun | ✅ < 2 dk |
| Stratejik planlama | < 1 saat | ✅ Uygun | ✅ < 5 dk |
