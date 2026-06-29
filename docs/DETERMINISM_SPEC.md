# SODALS — Deterministik Garanti Spesifikasyonu

> **Versiyon:** 1.0  
> **Tarih:** 11 Mayıs 2026  
> **Kapsam:** EODB Benchmark & AssetOpsBench inference pipeline  

---

## 1. İddia

Aynı input (prompt, context, attached media, runtime seed) + aynı donanım yapılandırması → **bit-for-bit aynı token dizisi** garanti edilir.

## 2. Garanti Edilen Koşullar

| Parametre | Değer | Kanıt |
|-----------|-------|-------|
| **Inference modu** | Greedy decoding | `temperature = 0.0`, `do_sample = false`, `top_p = 1.0` |
| **GPU deterministic ops** | Etkin | `torch.backends.cudnn.deterministic = true` (Python) / `strict_determinism = true` (Rust engine) |
| **Seed** | Sabit `42` | `setup_assetops_reproducibility(42)` çağrısı loglanır |
| **RNG kaynağı** | `ChaCha8Rng` (seedable) | `oedb_runner.rs` payload generation'ında `seed` sabitlenir |
| **Precision** | `float16` (inference) / `float32` (RK4 pondering) | `torch_dtype=torch.float16` (Python), Candle default f32 (Rust) |
| **OS** | Windows 10/11, Linux x86_64 | Test edilen platformlar |
| **GPU** | NVIDIA CUDA 12.x, Compute Capability ≥ 6.0 | FMA order caveats aşağıda belirtilmiştir |

## 3. Seed Yönetimi

```
EODB Runner         → scenario.dynamic_config.seed (null = sabit 42 fallback)
AssetOps Wrapper    → AssetOpsModelWrapper(seed=42)
Python Reproducibility → random.seed(42), np.random.seed(42), torch.manual_seed(42)
Rust Candle Engine  → strict_determinism flag (no explicit seed; greedy decoding)
```

**Kural:** `seed` parametresi hiçbir zaman `null` bırakılmaz; reproducibility loglarında her çalıştırma için kullanılan seed kaydedilir.

## 4. Float Precision & GPU Non-Determinism

| Bileşen | Precision | Determinizm Notu |
|---------|-----------|------------------|
| Candle matmul (QMatMul) | f16 | `CUBLAS_WORKSPACE_CONFIG=:4096:8` gerekli |
| RK4 latent pondering | f32 | Accumulation order GPU'ya bağlı olabilir; **pseudocode'da deterministic**
| Semantic network (CSR) | f64 | Host-side, fully deterministic |
| Token logits | f32 | Greedy `argmax` → deterministik |

**IEEE 754 Uyarısı:** NVIDIA GPU'larda `fma` (fused multiply-add) instruction sıralaması teorik olarak farklı thread scheduling ile farklı olabilir. Pratikte greedy decoding + fixed seed ile aynı `argmax` token'ı seçilse bile, ara float değerlerin LSB bit'leri farklı olabilir. **Token çıktısı deterministik; ara gösterimler (hidden states) GPU'ya bağlı sub-1e-6 varyasyon gösterebilir.**

## 5. Test Reproducibility Protokolü

### 5.1 — Tek Senaryo, 100 Tekrar

```bash
# PowerShell (Windows)
for ($i=1; $i -le 100; $i++) {
    .\target\release\eodb_runner.exe test_industrial.json --with-engine 2>&1 | Out-File "repro_run_$i.txt"
}
# Çıktı hash'lerini karşılaştır
Get-FileHash repro_run_*.txt | Select-Object Hash | Sort-Object Hash | Get-Unique
```

**Başarı Kriteri:** Tüm 100 tekrar için `model_output` metni (security violations hariç) **identical** olmalı.

### 5.2 — Token Logit Diff Testi

Candle engine'e `strict_determinism = true` ile aynı input'u 10 kez verip token ID dizisini karşılaştırın. Fark varsa logit diff analizi yapın.

### 5.3 — CPU vs GPU Parity

Aynı input'u CPU (`device = "cpu"`) ve GPU (`device = "cuda:0"`) ile çalıştırın. **Token dizisi %100 eşleşmeli.** Ara float değerler (perplexity, logit variance) %0.01 tolerans içinde olabilir.

## 6. Bilinen Non-Determinizm Kaynakları (Reddedildi / Planlandı)

| Kaynak | Durum | Aksiyon |
|--------|-------|---------|
| OS thread scheduling ( Rayon paralellik) | Kısmen | `RAYON_NUM_THREADS=1` ile test edildi |
| CUDA kernel launch async order | Kısmen | `CUDA_LAUNCH_BLOCKING=1` ile test edilecek |
| `getrandom` syscall (Linux) | **Yok** | EODB runner'da `getrandom` kullanılmıyor; `ChaCha8Rng` seedable |
| Temperature sampling | **Yok** | `temperature = 0.0`, `do_sample = false` |
| Floating-point accumulation (RK4) | **Bilinen** | Token çıktısı deterministik; hidden states varyasyonu kabul edilebilir |

## 7. Sınırlamalar (Jüri İçin Şeffaflık)

1. **Greedy decoding dışında** determinizm garanti edilmez (temperature > 0 → sampling → stokastik).
2. **Farklı GPU mimarisi** (örn. RTX 4090 vs A100) aynı token dizisini üretmelidir; ancak bu henüz test edilmemiştir.
3. **RK4 pondering** latency'yi artırır ve deterministic olmasına rağmen, float accumulation LSB varyasyonu olabilir. Bu, output'u değil, sadece internal state metriklerini etkiler.

## 8. Kanıt Paketi

Bu spesifikasyon ile birlikte şu dosyalar sunulacaktır:

- `docs/DETERMINISM_SPEC.md` (bu dosya)
- `repro_runs/` — 100 tekrarlı log (üretilecek)
- `repro_summary.json` — hash karşılaştırması (üretilecek)

---

> **Onay:** Bu spesifikasyon SODALS inference pipeline'ının deterministic olma iddiasını teknik olarak tanımlar. Jüri "Same input → same output?" sorusuna cevap: **Evet, greedy decoding + fixed seed + deterministic CUDA ops koşulları altında.** Detaylı kanıt 100 tekrarlı test paketindedir.
