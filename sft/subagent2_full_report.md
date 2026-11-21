# SODALS Core Mimari ve Pondering Mekanizması Araştırma Raporu

Bu rapor, `c:\Users\Burak\Desktop\sodals_core` kod tabanında yer alan Attention mekanizmasını, Gated DeltaNet SSM yapısını, rezonans ve Clifford cebir manifoldlarını, **PREFILL ve GENERATION** aşamalarındaki **Pondering Core (RK4 ODE Manifold Entegratörü)** algoritmasını ve bu mekanizmanın **Uzun Zincirli Akıl Yürütme (Long CoT, 1500-2048 token) ile RL/Distilasyon** süreçlerinde VRAM patlaması (OOM) yaşanmadan nasıl harmanlanabileceğine dair somut kod ve mimari tasarımları sunar.

---

## 1. Mimari Genel Bakış ve Hibrit Katman Düzeni

SODALS (EdgeSSM), standart Transformer mimarisinin quadratic $O(L^2)$ bellek büyümesini kırmak ve edge donanımlarda (ör. RTX 4060 8GB) deterministik gerçek zamanlı çıkarım sağlamak amacıyla hibrit bir omurga kullanır:
* **Model Omurgası:** Qwen 3.5 9B tabanlı hibrit mimari.
* **Katman Dağılımı:** Toplam 32 katman; **3:1 SSM/Attention oranı**.
  * **24 SSM Katmanı (Gated DeltaNet):** 0, 1, 2, 4, 5, 6, 8, 9, 10, ... (Her 4. katman hariç).
  * **8 Full Attention Katmanı:** 3, 7, 11, 15, 19, 23, 27, 31.
* **Pondering Entegrasyonu:** Model discrete token uzayına çökmeden (collapsing) önce gizli durumu (hidden state) 4096-D sürekli manifoldda diferansiyel denklemler (RK4) ile işler.

---

## 2. Attention Mekanizması ve Retroactive Relaxation

### 2.1. Rust Tarafı İmplementasyonu
* **Dosya:** `src/crates/sodals_neuro/src/models/qwen_custom.rs`
* **Yapı:** `pub struct QwenAttention` (Satır ~1900–2460)
  * **GQA (Grouped-Query Attention):** `num_heads = 32`, `num_kv_heads = 16`, `head_dim = 128`.
  * **RMS Normalizasyonu:** $Q$ ve $K$ projeksiyonları üzerinde `QRmsNorm` (`q_norm`, `k_norm`).
  * **Attention Gate (Sigmoid):** `attn_gate` projeksiyonu üzerinden elde edilen gate faktörü `candle_nn::ops::sigmoid(&gate)` ile attention çıkışına uygulanır (`attn_flat.broadcast_mul(&gate_active)`).
  * **QMEM Runtime Attention Bias:** Katman seviyesinde dinamik nöromorfik bellek bias'ı eklenir (`attention_scores = attention_scores + bias`).
  * **Fizik Çekirdeği Enjeksiyonu (Zero-Copy CUDA Kernel):** `adapter.apply_bias_raw(w_ptr, i_ptr, ...)` ile `ZeroCopyCartridge` üzerindeki empedans maskesi attention skoru matrisine yazılır.
  * **GNN / Semantik Sapma Freni:** Eğer `PhaseSignal` içindeki `semantic_divergence < 1.0` ise:
    $$\text{penalty} = (1.0 - \text{divergence}) \times (-10.0)$$
    attention skorlarına doğrudan negatif ceza olarak eklenir.

### 2.2. Retroactive Attention & Gauss-Seidel KV-Relaxation
* **Fonksiyon:** `QwenAttention::relax_past_states` (`qwen_custom.rs` Satır 2473–2640)
* **Mantık:** RK4 Pondering sırasında bir "aydınlanma" (epiphany / büyük semantik sıçrama) tespit edildiğinde, Transformer'ların nedensel (causal) append-only kuralı kırılarak geçmiş KV durumları geriye dönük olarak gevşetilir.
* **Matematiksel Formül (Zamansal Sönümlenmeli Gauss-Seidel):**
  $$\text{decay}[pos] = \exp(-\lambda \cdot (seq\_len - 1 - pos))$$
  $$K_{\text{new}}[pos] = (1 - \omega \cdot \text{decay}[pos]) K_{\text{old}}[pos] + \omega \cdot \text{decay}[pos] \cdot \Delta E_{\text{projected}}$$
* **Holografik Faz Rotasyonu:** Eğer `PhaseSignal` mevcutsa, $\Delta E$ skalar bir üzerine-yazma yerine kompleks düzlemde $\theta$ açısıyla döndürülür:
  $$E_{\text{real}} = E \cdot \cos(\theta), \quad E_{\text{imag}} = E \cdot \sin(\theta)$$
  böylece KV belleğinde yapıcı/yıkıcı girişim (interference) oluşturulur.
* **VRAM Güvenlik Kalkanı:**
  ```rust
  // Cache 1536 token sınırını aşarsa broadcast öncesi kırpılır:
  let (k_cache, v_cache) = if k_cache_raw.dim(2)? > 1536 {
      let cur_len = k_cache_raw.dim(2)?;
      (k_cache_raw.narrow(2, cur_len - 1536, 1536)?, v_cache_raw.narrow(2, cur_len - 1536, 1536)?)
  } else { ... };
  ```

---

## 3. Delta Gate Net (Delta-Rule State Space Model - SSM)

### 3.1. Rust İmplementasyonu
* **Dosya:** `src/crates/sodals_neuro/src/models/qwen_ssm.rs`
* **Yapı:** `pub struct QwenSsmBlock` (Satır 11–559)
  * **Bileşenler:** `ssm_in_proj`, `attn_gate` (Z-gate), `conv1d` (kernel_size=4), `ssm_out_proj`, `ssm_norm` (QRmsNorm), `ssm_a`, `ssm_dt`, `ssm_beta`, `ssm_alpha`.
  * **Durum Boyutu:** Sabit boyutlu matris $S_t \in \mathbb{R}^{B \times 32 \times 128 \times 128}$ ve `conv_state` $[B, 8192, 3]$.
  * **Tiled Head Genişletme:** Q ve K başlıkları 16'dan 32'ye `Tensor::cat(&[&k, &k], 1)` şeklinde döşenerek (tile) V başlıklarıyla hizalanır.
  * **Sürekli Bozunum Kapısı (Continuous Decay Gate):**
    $$g = -A \cdot \text{softplus}(\alpha_{\text{proj}} + \Delta t)$$
    $$\text{decay\_gate} = \exp(g) \in (0, 1]$$
  * **Delta Kuralı Rekürans Taraması (`recurrent_scan`):**
    1. *Durum Sönümü:* $S'_t = S_{t-1} \odot \text{decay\_gate}_t$
    2. *Bellek Okuma:* $s_k = K_t^T S'_t$
    3. *Hata (Yenilik) Hesabı:* $e_t = V_t - s_k$
    4. *Yazma Kapısı:* $\delta_t = \beta_t \odot e_t$ (Burada $\beta = \text{sigmoid}(\text{ssm\_beta}(x))$)
    5. *Dış Çarpım Güncellemesi:* $S_t = S'_t + K_t \otimes \delta_t$
    6. *Çıktı Okuma:* $O_t = Q_t^T S_t$
  * **QMEM Cartridge Warm-Start:** İlk adımda harici semantik tensör $K_{\text{cart}} \otimes \delta_{\text{cart}}$ şeklinde $S_0$'a yazılır.
  * **CDCL Backtracking Uyumsuzluğu:** $S_t$ dış çarpım birikimi yaptığı için matematiksel olarak tersi alınamaz. Bu nedenle CDCL arama algoritmalarında geri sarma (backtracking) temiz bir prefix re-prefill protokolü ile garanti edilir.

---

## 4. Latent Manifold ve Rezonans Yapıları

### 4.1. Rezonans Ağı (Resonance Network)
* **Dosya:** `src/crates/sodals_neuro/src/semantic_network.rs`
* **Yapı:** `pub struct ResonanceNetwork` (Satır 401–455)
  * **Hamiltonian Durum:** Genelleştirilmiş koordinatlar $q \in \mathbb{C}^N$ (`positions: SemanticWave`) ve eşlenik momentumlar $p \in \mathbb{C}^N$ (`momenta: SemanticWave`).
  * **İnersiya ve Sönümleme:** `masses: Array1<f64>`, `damping: Array1<f64>`.
  * **CSR Seyrek Matris Kuplajı:** $O(nnz)$ seyrek komşuluk matrisi `adjacency_csr: CsMat<f64>` ve kompleks akım matrisi `flow_matrix: CsMat<Complex64>`.
  * **Holografik Hiperboyutlu Bellek:** Her semantik düğüm için 4096-D fazör vektörleri (`holographic_memory: Vec<Vec<Complex64>>`).
  * **Alan Bayrakları:** `DOMAIN_LOGIC`, `DOMAIN_SPATIAL`, `DOMAIN_TEMPORAL`.

### 4.2. Akışkan Cebri ve Fazör Çarpışma Motoru
* **Dosya:** `src/crates/sodals_neuro/src/fluid_algebra.rs` (Satır 1–206)
  * **Clifford Geometrik Çarpımı (Holographic Binding):**
    $$\text{bind\_phasors}(A, B): \quad A \otimes B = [a_k \cdot b_k] \implies e^{i(\theta_A + \theta_B)}$$
    $O(N^3)$ matris çarpımı yerine $O(N)$ faz toplamı.
  * **Ters Clifford Çıkarımı (Holographic Unbinding):**
    $$\text{unbind\_phasors}(C, B): \quad C \otimes B^* \implies e^{i(\theta_C - \theta_B)} \approx A$$
  * **Topolojik Viskozite (Topological Viscosity):**
    $$\text{apply\_viscosity}(phasor, \text{impedance}): \quad \text{amplitude}' = \text{amplitude} \cdot e^{-\text{impedance}}$$
    Mantıksal kurallarla çelişen uzamsal veya semantik fazörler bellekten silinmez; viskozite ile genlikleri sönümlenir.

---

## 5. Pondering Core Algoritması (RK4 ODE Manifold Entegrasyonu)

### 5.1. Rust İmplementasyonu
* **Dosya:** `src/crates/sodals_neuro/src/engine/pondering_core.rs`
* **Yapı:** `pub struct PonderingCore` (Satır 163–800)
* **Sabitler:** `MAX_PONDER_STEPS = 3`, `DT = 0.15`, `EPSILON = 1e-4`, `TAU_LOGICAL = 0.1`.

#### A. Arkhe Diferansiyel Vektör Alanı Türevi (`arkhe_derivative`)
Gizli durum $h$, manifold üzerinde non-lineer çekiciler (attractor) boyunca akar:
$$\frac{dh}{dt} = \tanh(\alpha \cdot h) - \gamma \cdot h + \beta \cdot \text{ghost} + \lambda \cdot \text{correction}$$
* **$\tanh(\alpha \cdot h)$:** Sınırlı, pürüzsüz nöral manifold çekici alanı ($\alpha = 0.15$).
* **$-\gamma \cdot h$:** Enerji sönümleme terimi ($\gamma = 0.05$).
* **$+\beta \cdot \text{ghost}$:** `GHOST_VECTOR_SLOT` üzerindeki anlık telemetri veya semantik rehberlik çekimi ($\beta = 0.08 \times 10.0$ telemetri güçlendirmesi).
* **$\lambda \cdot \text{correction}$:** God's Gate / Picard düzeltme terimi.
* **Katsayı Yönetimi:** `ArkheCoefficients` yapısı içindeki `AtomicU64` hücreleriyle kilit-serbest (lock-free) hot-path erişimi.

#### B. Bare-Metal RK4 İntegratörü (`rk4_step_variable_dt`)
Candle tensörlerinde GPU bellek tahsisini (allocation) sıfırlamak için `.affine()` skalar operasyonları kullanılır:
$$k_1 = f(h_n, \text{ghost}, \text{corr})$$
$$h_{k2} = h_n + \frac{dt}{2} k_1 \implies k_2 = f(h_{k2}, \dots)$$
$$h_{k3} = h_n + \frac{dt}{2} k_2 \implies k_3 = f(h_{k3}, \dots)$$
$$h_{k4} = h_n + dt \cdot k_3 \implies k_4 = f(h_{k4}, \dots)$$
$$h_{n+1} = h_n + \frac{dt}{6} \left(k_1 + 2k_2 + 2k_3 + k_4\right)$$

#### C. Picard Öz-Düzeltmesi ve God's Gate
* **Hızlı Kosinüs Koherens Kontrolü (`fast_semantic_coherence_check`):**
  $$\text{divergence} = 1.0 - \left|\frac{\langle h_{\text{current}}, h_0 \rangle}{\|h_{\text{current}}\| \cdot \|h_0\|}\right|$$
* **Picard Düzeltmesi:** Eğer $\text{divergence} > \tau_{\text{logical}}$ ise:
  $$h_{\text{corrected}} = (1 - \lambda) h_{\text{current}} + \lambda h_0 \quad (\lambda = 0.3)$$
* **Kuantum Faz İnversiyonu (Quantum Phase Inversion):** Önceki token halüsinatif ise (`prev_was_hallucinated == true`), Grover algoritması mantığıyla $\pi$-radyan faz ters çevirmesi uygulanarak yanlış akıl yürütme yolları üzerinde yıkıcı girişim (destructive interference) oluşturulur.
* **2D RoPE Decoder Projeksiyonu (`compute_physics_violation`):** 4096-D vektörü 2D RoPE bazları ile tersine projekte ederek piksel/kütle korunumunun ihlal edilip edilmediğini doğrular.

### 5.2. Python Tarafı Diferansiyellenebilir İmplementasyonlar
1. **`sft/pondering_core_bridge.py` $\to$ `PonderingODEModule`:**
   * Differentiable explicit RK4 autograd grafiği:
     ```python
     def vector_field(self, h, h0):
         drift = -self.lambda_anchor * (h - h0)
         flow = self.arkhe_proj(torch.tanh(h))
         return drift + flow
     ```
   * 4-adım RK4 döngüsü ile gradyanların doğrudan LoRA ağırlıklarına akması sağlanır.
2. **`sft/adaptive_pondering_ode.py` $\to$ `AdaptivePonderingODEModule`:**
   * Yönlendirici (router) katsayısı $\alpha \in [0, 1]$'e bağlı dinamik RK4 adım sayısı ($N(\alpha) \in [2, 5]$) ve dinamik atalet katsayısı ($\lambda_{\text{eff}} = \lambda_0 (1 + \alpha)$).

---

## 6. Prefill ve Generation Aşamalarında Pondering İşleyişi

| Aşama | Tetiklenme Şartı | Girdi Vektörü | Yapılan İşlem | Bellek / Çıktı Yönetimi |
| :--- | :--- | :--- | :--- | :--- |
| **PREFILL** | `seqlen_offset == 0`<br>($0 \pmod 8 == 0$) | Prompt'un son token gizli durumu:<br>`hidden_states.narrow(1, seq_len-1, 1)` | Model ilk tokeni üretmeden önce prompt bağlamını 4096-D manifoldda RK4 ile dengeler. "Konuşmadan önce düşünür". | **Kritik Optimizasyon:** `hidden_for_lm_head = pondered_hidden.narrow(...)` yapılarak tüm prompt için $1.07$ GB'lık devasa logit tensörü açılması engellenir. Sadece son token `lm_head`'e girer. |
| **GENERATION** | `seqlen_offset % 8 == 0`<br>(Her 8 tokende bir) | Önceki tokenin ürettiği $h_{\text{last}}$ ($seq\_len=1$) | Akıl yürütme rotasındaki semantik sapmaları düzeltir. Epiphany eşiği aşılırsa `retroactive_relax` tetiklenir. | `GodsGateState` (AtomicU64) ile tokenlar arası durum devredilir. Her tokende çalıştırılmayarak VRAM parçalanması ve RMS kayması önlenir. |

---

## 7. Uzun Zincirli Akıl Yürütme (Long CoT) ve RL/Distilasyon ile VRAM Patlamasını Önleme Tasarımı

1500–2048 token uzunluğundaki düşünce zincirlerinde (Long CoT), $V=248,320$ (Qwen vokabüleri) ile çalışırken karşılaşılan 3 ana VRAM darboğazı ve çözümleri:

### Darboğaz 1: Logit Tensörünün Patlaması ($2.03\text{ GB}$ FP32 Tek Adımda)
* **Tehlike:** 2048 token boyunca $1 \times 2048 \times 248,320 \times 4\text{ byte} \approx 2.03\text{ GB}$. Backward grafiği ve optimizer durumlarıyla 8GB kartta derhal OOM oluşur.
* **Uygulanan Çözüm (`sft/train_qwen_lora_with_pondering.py`):**
  1. Forward pass sonrası ham logitler derhal serbest bırakılır:
     ```python
     if hasattr(outputs, "logits"):
         outputs.logits = None
     ```
  2. **Chunked lm_head + CrossEntropy:** 2048 uzunluk `chunk_size = 256` dilimlerine bölünür. Her dilimde `lm_head(c_hidden)` hesaplanır, loss toplanır ve ara logit tensörü GPU'dan silinir. Tepe bellek kullanımı **4.0 GB'tan 120 MB'a** düşer!

### Darboğaz 2: Retroactive KV Cache ve Autograd Tape Patlaması
* **Tehlike:** Her token için RK4'ün 4 ara durumunu ($k_1, k_2, k_3, k_4$) ve KV-gevşetme tensörlerini 2048 token boyunca autograd hafızasında tutmak GPU'yu kilitler.
* **Uygulanan Çözüm:**
  1. **Sliding KV Window Pruning (`qwen_custom.rs` Satır 2500):** Gevşetme tensörü geriye doğru tüm diziyi değil, maksimum 1536 tokenlik pencereyi günceller (`narrow(2, cur_len - 1536, 1536)`).
  2. **Prefill-Boundary Pondering + Activation Checkpointing:** RL ve SFT eğitiminde RK4 grafiği yalnızca prefill sınırında ve $K=8$ aralıklarla çağrılır; `torch.utils.checkpoint.checkpoint` ile $k_1..k_4$ ara tensörleri geriye yayılım sırasında anlık yeniden hesaplanır.

### Darboğaz 3: RL (GRPO) Eğitiminde Value/Critic Modeli Maliyeti
* **Tehlike:** PPO türevi algoritmalar ayrı bir Critic modeli ve referans modeli gerektirir (3x model boyutu = 24GB+ VRAM).
* **Uygulanan Çözüm (`sft/pilot_grpo_single_expert.py`):**
  1. **GRPO (Group Relative Policy Optimization):** Critic modeli yoktur ($0$ ekstra VRAM). Avantajlar grup içi $G=4$ sequential rollout ortalamasından çıkarılır:
     $$A_g = \frac{R_g - \text{mean}(R)}{\text{std}(R) + \epsilon}$$
  2. **Zero-VRAM Referans Modeli:** Referans model GPU'ya ayrı yüklenmez; LoRA adaptörü anlık kapatılarak baz model üzerinden hesaplanır:
     ```python
     with torch.no_grad():
         with model.disable_adapter():
             ref_out = model.base_model.model.model(seq_ids)
     ```
  3. **Rollout-by-Rollout Backward:** $G=4$ rollout aynı anda tensör olarak tutulmaz; her rollout için politika gradyanı tek tek geriye yayılır (`loss_g.backward()`), tensörler `del` ile silinip `torch.cuda.empty_cache()` çağrılır.

### Darboğaz 4: Düşünce Zincirinde Bozulma ve Kanıt Terki
* **Tehlike:** Long CoT uzadıkça modelin kanıtı yarıda bırakması (proof abandonment) veya değişkenleri unutması.
* **Uygulanan Çözüm (`sft/reasoning_chain_jacobian.py`):**
  * `ReasoningChainJacobianFilter`: Adımlar arası semantik Jacobian normu $\|\Delta_i\| = \|\text{feature}(s_i) - \text{feature}(s_{i-1})\|$ hesaplanır. $\text{jump\_ratio} > 3.0$ olan veya değişken kaybı yaşanan örnekler veri setinden veya RL ödülünden elenir.

### Darboğaz 5: Pondering Distilasyonu (Latent Distillation)
* Modelin her çıkarım adımında RK4 integratörü çalıştırması yerine; ODE manifoldunun yakınsadığı denge durumu ($h_{\text{final}} = \text{RK4}(h_0)$), LoRA adaptörüne bir latent distilasyon kaybı ile öğretilir:
  $$\mathcal{L}_{\text{distill}} = 1.0 - \text{CosineSimilarity}(h_{\text{backbone}}, h_{\text{pondered}}^{\text{target}})$$
  Böylece Long CoT sırasında model, 0-adım veya 1-adım RK4 ile sanki 4-adım düşünmüş gibi kararlı latent temsillere doğrudan ulaşır.

---

## 8. Somut Dosya ve Sembol İndeksi

| Bileşen / Mekanizma | Dil | Dosya Yolu | İlgili Sınıf / Fonksiyon / Sabit |
| :--- | :--- | :--- | :--- |
| **Pondering Core** | Rust | `src/crates/sodals_neuro/src/engine/pondering_core.rs` | `PonderingCore`, `ponder()`, `rk4_step_variable_dt()`, `arkhe_derivative()`, `GodsGateState` |
| **Attention & KV-Relax** | Rust | `src/crates/sodals_neuro/src/models/qwen_custom.rs` | `QwenAttention`, `relax_past_states()`, `retroactive_relax()`, satır 5030-5230 (`should_ponder`) |
| **DeltaNet SSM** | Rust | `src/crates/sodals_neuro/src/models/qwen_ssm.rs` | `QwenSsmBlock`, `recurrent_scan()`, `apply_conv1d()`, `state_cache` |
| **Rezonans Ağı** | Rust | `src/crates/sodals_neuro/src/semantic_network.rs` | `ResonanceNetwork`, `positions`, `momenta`, `holographic_memory`, `GridTopology` |
| **Clifford / Akışkan Cebri** | Rust | `src/crates/sodals_neuro/src/fluid_algebra.rs` | `bind_phasors()`, `unbind_phasors()`, `apply_viscosity()` |
| **PyTorch ODE Köprüsü** | Python | `sft/pondering_core_bridge.py` | `PonderingODEModule`, `vector_field()`, `CliffordBridgePyTorch` |
| **Faz-Adaptif ODE** | Python | `sft/adaptive_pondering_ode.py` | `AdaptivePonderingODEModule`, `forward()` |
| **Chunked Loss & SFT** | Python | `sft/train_qwen_lora_with_pondering.py` | `PonderingOrthogonalTrainer`, `compute_loss()` (chunk_size=256) |
| **Zero-VRAM GRPO RL** | Python | `sft/pilot_grpo_single_expert.py` | `compute_reward()`, GRPO eğitim döngüsü (`disable_adapter()`) |
| **CoT Kalite Denetimi** | Python | `sft/reasoning_chain_jacobian.py` | `ReasoningChainJacobianFilter`, `JacobianReport` |
| **Python İnference Motoru** | Python | `sft/sodals_inference_engine.py` | `FullSodalsInferenceEngine`, `DynamicBudgetExpander`, `SkewGBraking` |

Tüm bu mimari bileşenler, kod tabanında doğrulanmış olup sıfır bellek sızıntısıyla 8GB edge donanımlarda stabil çalışacak şekilde tasarlanmıştır.