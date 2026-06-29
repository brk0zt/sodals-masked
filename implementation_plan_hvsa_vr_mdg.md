# Implementation Plan
## H-VSA Scene Graph + Virtual Reset + Manhattan Distance Gradient
### SODALS Core — Koddan Türetilmiş Spesifikasyon

**Tarih:** 18 Haziran 2026  
**Hedef:** ARC-AGI-3 RHAE skoru iyileştirmesi + minimum hamle garantisi  
**Etkilenen dosyalar:** `semantic_network.rs`, `agent_loop.rs`, `mcts_exploration.rs`

---

## Önce: Mevcut Kodda Ne Var, Ne Eksik

### `find_hamiltonian_trajectory()` — Satır 1745

Şu an şöyle çalışıyor:
```
semantic_force = weight * (1.0 + target_attraction)
best_next = argmax(semantic_force)
```
Bu **L∞ norm** değil, **ağırlık çarpımı**. Manhattan mesafesi yok.
`target_attraction` sabit `10.0` — dinamik değil.
Hamiltonian constraint var ama sezgisel yok — körü körüne en ağır kenara gidiyor.

### `save_cartridge()` / `load_cartridge()` — Satır 497–548

`holographic_memory` zaten serialize ediliyor (önceki bug fix sonrası).
Virtual reset için gereken altyapı **hazır** — sadece RAM'e serialize edilmiyor, diske gidiyor.
Bunu RAM snapshot'a çevirmek gerekiyor.

### `holographic_memory: Vec<Vec<Complex64>>` — Satır 354

4096-D imzalar zaten var.
`holographic_similarity()` fonksiyonu **yok** — sadece `check_phase_variance()` var.
Scene graph için bu fonksiyon yazılmalı.

---

## Bölüm 1: Manhattan Distance Gradient

### Problem

`find_hamiltonian_trajectory()` şu an `semantic_force = weight * scalar` ile çalışıyor.
Bu Öklid mesafesine benzer, seyrek çözüm üretmiyor.
ARC-AGI-3'te "minimum hamle" = RHAE skoru. Manhattan gradyan burada işe yarıyor çünkü
L1 norm sparse minimizer — gereksiz ara düğümleri doğal olarak atlatıyor.

### Değişiklik: `semantic_network.rs`

**Yeni yardımcı fonksiyon — `manhattan_distance_to_target()`**

```rust
/// L1 (Manhattan) mesafe — sparse path için
/// |x_i - x_j| toplamı, Öklid'den farklı olarak köşe çözümler üretiyor
fn manhattan_distance_to_target(
    &self,
    node: usize,
    target_state: &[usize],
) -> f64 {
    // Her target node'a L1 mesafe hesapla, minimumunu al
    target_state.iter().map(|&t| {
        let pos_node = self.positions.amplitudes[node];
        let pos_tgt  = self.positions.amplitudes[t];
        // Re + Im bileşenleri üzerinden L1
        (pos_node.re - pos_tgt.re).abs() + (pos_node.im - pos_tgt.im).abs()
    }).fold(f64::INFINITY, f64::min)
}
```

**`find_hamiltonian_trajectory()` içindeki force hesabı değişiyor**

```rust
// ESKI (satır 1791):
let semantic_force = weight * (1.0 + target_attraction);

// YENİ — Manhattan gradyan tabanlı:
let manhattan_dist = self.manhattan_distance_to_target(neighbor, target_state);
// Mesafe azaldıkça force artar (ters orantı), weight katsayı olarak kalır
let gradient_force = weight / (manhattan_dist + 1e-6);
// target node'a doğrudan komşuysa güçlü çekim
let target_bonus = if target_set.contains(&neighbor) { 5.0 } else { 0.0 };
let semantic_force = gradient_force + target_bonus;
```

**Neden 10.0 sabit yerine 5.0?**
Eski `target_attraction = 10.0` çok baskın — Manhattan gradyanı eziyor.
5.0 ile gradient_force'a alan açılıyor, ikisi dengeli çalışıyor.

### Test

```rust
#[test]
fn test_manhattan_gradient_prefers_shorter_path() {
    let mut net = ResonanceNetwork::new(10);
    let a = net.add_concept(0.01);
    let b = net.add_concept(0.01);
    let c = net.add_concept(0.01);
    
    // a → c düz yol, a → b → c dolaşık yol
    net.connect_semantic(a, c, 0.9);
    net.connect_semantic(a, b, 0.5);
    net.connect_semantic(b, c, 0.5);
    
    // Konumları ayarla: c hedefe yakın, b uzak
    net.positions.amplitudes[a] = Complex64::new(0.0, 0.0);
    net.positions.amplitudes[b] = Complex64::new(5.0, 5.0); // uzak
    net.positions.amplitudes[c] = Complex64::new(0.1, 0.1); // hedefe yakın
    
    let trajectory = net.find_hamiltonian_trajectory(&[a], &[c], 5);
    
    // Manhattan gradyan ile a → c direkt gitmeli, b'ye uğramamalı
    assert!(!trajectory.contains(&b), "Manhattan gradient should skip detour node");
    assert_eq!(trajectory.len(), 2, "Should reach target in 2 steps");
}
```

---

## Bölüm 2: Virtual Reset

### Problem

`MctsSearch::rollout()` şu an ortamı gerçekten mutate ediyor — yanlış dal deneyince
gerçek `net_g` kirleniyor. Virtual reset olmadan her dallanma kalıcı iz bırakıyor.

Mevcut `save_cartridge()` disk I/O yapıyor — rollout döngüsünde çağrılamaz (~5ms latency × 1000 rollout = 5 saniye kayıp).

### Değişiklik 1: `semantic_network.rs` — RAM snapshot

```rust
/// In-memory snapshot — disk I/O yok, rollout için optimize
/// save_cartridge() ile aynı veriyi tutar ama Vec<u8> olarak bellekte
#[derive(Clone)]
pub struct NetworkSnapshot {
    pub n: usize,
    pub damping: Vec<f64>,
    pub masses: Vec<f64>,
    pub adjacency_rows: Vec<usize>,
    pub adjacency_cols: Vec<usize>,
    pub adjacency_vals: Vec<f64>,
    pub labels: HashMap<String, usize>,
    pub holographic_memory: Vec<Vec<Complex64>>,
    pub node_flags: Vec<u32>,
    // positions ve momenta da dahil — clear_state() sonrası restore için
    pub position_re: Vec<f64>,
    pub position_im: Vec<f64>,
    pub momentum_re: Vec<f64>,
    pub momentum_im: Vec<f64>,
}

impl ResonanceNetwork {
    /// RAM snapshot al — O(n + nnz), disk I/O yok
    pub fn snapshot(&self) -> NetworkSnapshot {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (&val, (r, c)) in self.adjacency_csr.iter() {
            rows.push(r); cols.push(c); vals.push(val);
        }
        NetworkSnapshot {
            n: self.n,
            damping: self.damping.to_vec(),
            masses: self.masses.to_vec(),
            adjacency_rows: rows,
            adjacency_cols: cols,
            adjacency_vals: vals,
            labels: self.labels.clone(),
            holographic_memory: self.holographic_memory.clone(),
            node_flags: self.node_flags.clone(),
            position_re: self.positions.amplitudes.iter().map(|c| c.re).collect(),
            position_im: self.positions.amplitudes.iter().map(|c| c.im).collect(),
            momentum_re: self.momenta.amplitudes.iter().map(|c| c.re).collect(),
            momentum_im: self.momenta.amplitudes.iter().map(|c| c.im).collect(),
        }
    }

    /// RAM snapshot'tan restore — yapısal yeniden inşa + durum geri yükleme
    pub fn restore_snapshot(&mut self, snap: &NetworkSnapshot) {
        self.n = snap.n;
        self.damping = Array1::from_vec(snap.damping.clone());
        self.masses  = Array1::from_vec(snap.masses.clone());
        self.labels  = snap.labels.clone();
        self.holographic_memory = snap.holographic_memory.clone();
        self.node_flags = snap.node_flags.clone();

        // CSR yeniden inşa
        let cap = self.masses.len();
        let mut tri = sprs::TriMat::<f64>::new((cap, cap));
        for i in 0..snap.adjacency_rows.len() {
            tri.add_triplet(
                snap.adjacency_rows[i],
                snap.adjacency_cols[i],
                snap.adjacency_vals[i],
            );
        }
        self.adjacency_csr = tri.to_csr();
        self.cache_valid = false;

        // Positions + momenta restore
        for (i, (re, im)) in snap.position_re.iter().zip(snap.position_im.iter()).enumerate() {
            if i < self.positions.amplitudes.len() {
                self.positions.amplitudes[i] = Complex64::new(*re, *im);
            }
        }
        for (i, (re, im)) in snap.momentum_re.iter().zip(snap.momentum_im.iter()).enumerate() {
            if i < self.momenta.amplitudes.len() {
                self.momenta.amplitudes[i] = Complex64::new(*re, *im);
            }
        }
    }
}
```

### Değişiklik 2: `mcts_exploration.rs` — Rollout'ta Virtual Reset

```rust
// MctsSearch::rollout() içinde, her dal denemesi önce snapshot alır:

fn rollout(&mut self, node_id: usize, env: &mut dyn ArcEnvironment, net_g: &mut ResonanceNetwork) -> f64 {
    // Virtual reset checkpoint
    let checkpoint = net_g.snapshot();
    
    let mut total_reward = 0.0;
    let mut current_state = self.nodes[node_id].state.clone();
    
    for _ in 0..self.config.rollout_depth {
        // Q-table'dan en iyi eylem veya rastgele
        let action = self.select_rollout_action(&current_state);
        let (next_state, reward, done) = env.step(&action);
        total_reward += reward;
        
        if done { break; }
        current_state = next_state;
    }
    
    // Rollout bitti → net_g'yi checkpoint'e geri al
    // Ortam kirlenmedi, gerçek adım atılmadı
    net_g.restore_snapshot(&checkpoint);
    
    total_reward
}
```

### Değişiklik 3: `agent_loop.rs` — Episode İzolasyonu

```rust
// AgentLoop::run() başında:
let episode_checkpoint = self.semantic_memory.snapshot();

// Her yeni bölüm/episode başında:
self.semantic_memory.restore_snapshot(&episode_checkpoint);
// NOT: .exp dosyası silinmiyor — öğrenme birikimi korunuyor
// Sadece net_g'nin fizik durumu sıfırlanıyor
```

### Test

```rust
#[test]
fn test_virtual_reset_no_contamination() {
    let mut net = ResonanceNetwork::new(20);
    let a = net.add_labeled_concept("start", 0.01, Vec::new());
    net.positions.amplitudes[a] = Complex64::new(1.0, 0.5);
    
    let snap = net.snapshot();
    
    // Net'i kirlet
    net.positions.amplitudes[a] = Complex64::new(99.0, 99.0);
    let b = net.add_concept(0.01);
    net.connect_syntax(a, b, 5.0);
    
    // Restore
    net.restore_snapshot(&snap);
    
    // Orijinal duruma dönmüş olmalı
    assert_eq!(net.n, 1, "node count must be restored");
    assert!((net.positions.amplitudes[a].re - 1.0).abs() < 1e-9, "position must be restored");
    assert_eq!(net.get_weight(a, 1), 0.0, "contamination edge must be gone");
}

#[test]
fn test_snapshot_restore_preserves_holographic_memory() {
    let mut net = ResonanceNetwork::new(10);
    let sig: Vec<Complex64> = (0..4096).map(|i| 
        Complex64::new(i as f64 * 0.001, 0.0)
    ).collect();
    net.add_labeled_concept("test_node", 0.01, sig.clone());
    
    let snap = net.snapshot();
    
    // Bellekte VSA imzasını boz
    net.holographic_memory[0] = Vec::new();
    
    // Restore
    net.restore_snapshot(&snap);
    
    assert_eq!(
        net.holographic_memory[0].len(), 
        4096, 
        "holographic signature must be restored"
    );
}
```

---

## Bölüm 3: H-VSA Scene Graph

### Problem

`primordial_cognition.qmem` ARC primitiflerini **düz konsept** olarak tutuyor:
`nesne`, `renk`, `konum`, `boyut` ayrı ayrı holografik imzalar.
Ama ARC görevlerinde asıl soru şu: "Bu gride benzer bir grid daha önce gördüm mü?"
Bunun için tek tek konsept benzerliği değil, **tüm sahne yapısının** VSA uzayında temsili lazım.

### H-VSA Bağlama İşlemi (Binding)

VSA'da hiyerarşik temsil için iki işlem:
- **Bağlama (binding):** `A ⊗ B = element-wise complex multiplication` — farklı rol/değer çiftleri
- **Süperpozisyon (superposition):** `A + B` — birden fazla özellik aynı vektörde

Sahne grafı bu ikisini birleştirir:
```
nesne_kırmızı_kare = bind(renk:kırmızı, şekil:kare, konum:(2,3))
sahne_vektörü = superpose(nesne_kırmızı_kare, nesne_mavi_daire, ...)
```
`sahne_vektörü` tüm sahneyi tek bir 4096-D imzada tutuyor.
`holographic_similarity(sahne_A, sahne_B)` → iki grid ne kadar benzer?

### Değişiklik 1: `semantic_network.rs` — `holographic_similarity()`

```rust
/// İki 4096-D phasor imzasının kosinüs benzerliği
/// H-VSA scene graph karşılaştırması için temel operasyon
/// 
/// Dönüş: 0.0 (tamamen farklı) → 1.0 (özdeş)
pub fn holographic_similarity(sig_a: &[Complex64], sig_b: &[Complex64]) -> f64 {
    if sig_a.is_empty() || sig_b.is_empty() || sig_a.len() != sig_b.len() {
        return 0.0;
    }
    let dim = sig_a.len() as f64;
    
    // Normalize edilmiş iç çarpım: (1/dim) * Σ a_k · conj(b_k)
    let dot: Complex64 = sig_a.iter().zip(sig_b.iter())
        .map(|(a, b)| a * b.conj())
        .fold(Complex64::new(0.0, 0.0), |acc, x| acc + x);
    
    // Büyüklük: |dot| / dim → [0, 1]
    (dot / dim).norm().min(1.0)
}

/// İki imzayı element-wise çarpar (VSA binding operasyonu)
/// "Kırmızı" + "Kare" → "Kırmızı Kare" bileşik konsept
pub fn vsa_bind(sig_a: &[Complex64], sig_b: &[Complex64]) -> Vec<Complex64> {
    sig_a.iter().zip(sig_b.iter())
        .map(|(a, b)| a * b)
        .collect()
}

/// Birden fazla imzayı süperpozisyona sokar, normalize eder
/// Bir sahnenin tüm nesnelerini tek imzada toplar
pub fn vsa_superpose(signatures: &[Vec<Complex64>]) -> Vec<Complex64> {
    if signatures.is_empty() { return Vec::new(); }
    let dim = signatures[0].len();
    
    let mut result = vec![Complex64::new(0.0, 0.0); dim];
    for sig in signatures {
        for (i, &v) in sig.iter().enumerate() {
            if i < dim { result[i] += v; }
        }
    }
    
    // L2 normalize — benzerlik hesabında boyuttan bağımsız
    let norm: f64 = result.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    if norm > 1e-10 {
        result.iter_mut().for_each(|c| *c /= norm);
    }
    result
}
```

### Değişiklik 2: `semantic_network.rs` — `ArcSceneGraph` struct

```rust
/// H-VSA Scene Graph — ARC grid'ini holografik uzayda temsil eder
/// 
/// Her grid için bir SceneGraph oluşturulur.
/// Tüm nesnelerin VSA süperpozisyonu "sahne imzası" verir.
/// İki sahne imzasının benzerliği = iki gridin yapısal benzerliği.
#[derive(Debug, Clone)]
pub struct ArcSceneGraph {
    /// Sahnenin toplam VSA imzası — tüm nesneler süperpozisyonda
    pub scene_signature: Vec<Complex64>,
    
    /// Nesne başına VSA imzaları — sorgu için
    pub object_signatures: Vec<ObjectSignature>,
    
    /// Grid topolojisi — mass conservation kontrolü için
    pub topology: GridTopology,
}

#[derive(Debug, Clone)]
pub struct ObjectSignature {
    pub color: u8,
    pub x: usize,
    pub y: usize,
    pub shape_label: String,
    pub signature: Vec<Complex64>,  // bind(color_sig, pos_sig, shape_sig)
}
```

### Değişiklik 3: `semantic_network.rs` — `build_scene_graph()`

```rust
impl ResonanceNetwork {
    /// ARC grid'inden H-VSA Scene Graph oluşturur
    /// 
    /// Her renk/konum çifti için VSA binding yapar,
    /// tüm nesneleri süperpozisyona sokar.
    /// Sonuç ResonanceNetwork'ün holographic_memory'sine de yazılır.
    pub fn build_scene_graph(&mut self, grid: &Array2<u8>) -> ArcSceneGraph {
        let topology = self.compute_grid_topology(grid);
        let (height, width) = (grid.nrows(), grid.ncols());
        let mut object_signatures: Vec<ObjectSignature> = Vec::new();
        
        // Renk ve pozisyon için temel VSA vektörleri
        // generate_holographic_signature() zaten var (qmem_compiler.rs'den)
        // Buraya import gerekecek veya inline Xorshift64*
        
        for y in 0..height {
            for x in 0..width {
                let color = grid[[y, x]];
                if color == 0 { continue; } // Boş hücre atla
                
                // Her renk için deterministik VSA imzası
                let color_key = format!("arc_color_{}", color);
                let color_sig = self.get_or_create_signature(&color_key);
                
                // Her konum için deterministik VSA imzası
                let pos_key = format!("arc_pos_{}_{}", x, y);
                let pos_sig = self.get_or_create_signature(&pos_key);
                
                // Renk + Konum binding → bu nesnenin imzası
                let obj_sig = Self::vsa_bind(&color_sig, &pos_sig);
                
                object_signatures.push(ObjectSignature {
                    color,
                    x, y,
                    shape_label: color_key.clone(),
                    signature: obj_sig,
                });
            }
        }
        
        // Tüm nesneleri süperpozisyona sok → sahne imzası
        let all_sigs: Vec<Vec<Complex64>> = object_signatures.iter()
            .map(|o| o.signature.clone())
            .collect();
        let scene_signature = Self::vsa_superpose(&all_sigs);
        
        // Sahneyi ResonanceNetwork'e kaydet
        let scene_label = format!("scene_{}x{}", width, height);
        let scene_id = self.add_labeled_concept(
            &scene_label, 0.01, scene_signature.clone()
        );
        
        eprintln!(
            "[H-VSA] Scene graph built: {} objects, scene_id={}",
            object_signatures.len(), scene_id
        );
        
        ArcSceneGraph { scene_signature, object_signatures, topology }
    }
    
    /// Mevcut imzayı getir veya Xorshift64* ile oluştur
    fn get_or_create_signature(&mut self, key: &str) -> Vec<Complex64> {
        if let Some(&id) = self.labels.get(key) {
            if id < self.holographic_memory.len() {
                let sig = self.holographic_memory[id].clone();
                if !sig.is_empty() { return sig; }
            }
        }
        // Xorshift64* deterministik üretim — qmem_compiler'daki ile aynı seed mantığı
        let seed = key.bytes().fold(6364136223846793005u64, |acc, b| {
            acc.wrapping_mul(2862933555777941757).wrapping_add(b as u64)
        });
        let sig = Self::generate_xorshift_signature(seed, 4096);
        self.add_labeled_concept(key, 0.01, sig.clone());
        sig
    }
    
    /// Xorshift64* ile deterministik 4096-D Complex64 vektör
    fn generate_xorshift_signature(mut seed: u64, dim: usize) -> Vec<Complex64> {
        (0..dim).map(|_| {
            seed ^= seed << 12;
            seed ^= seed >> 25;
            seed ^= seed << 27;
            let re = (seed.wrapping_mul(2685821657736338717) as f64 / u64::MAX as f64) * 2.0 - 1.0;
            seed ^= seed << 12;
            seed ^= seed >> 25;
            seed ^= seed << 27;
            let im = (seed.wrapping_mul(2685821657736338717) as f64 / u64::MAX as f64) * 2.0 - 1.0;
            let norm = (re * re + im * im).sqrt().max(1e-10);
            Complex64::new(re / norm, im / norm)
        }).collect()
    }
    
    /// İki sahne grafiğinin benzerliğini hesapla
    /// ARC-AGI-3'te "bu grid buna benziyor mu?" sorusunun cevabı
    pub fn scene_similarity(graph_a: &ArcSceneGraph, graph_b: &ArcSceneGraph) -> f64 {
        Self::holographic_similarity(&graph_a.scene_signature, &graph_b.scene_signature)
    }
}
```

### Değişiklik 4: `agent_loop.rs` — Scene Graph'ı Benzerlik Arkhesi'ne bağla

```rust
// AgentLoop::run() içinde, her ortam gözlemi geldiğinde:

if let Some(grid) = env.get_current_grid() {
    let current_scene = self.semantic_memory.build_scene_graph(&grid);
    
    // Geçmiş sahnelerle karşılaştır
    // episode_scenes: Vec<ArcSceneGraph> — episode boyunca birikir
    for past_scene in &self.episode_scenes {
        let sim = ResonanceNetwork::scene_similarity(&current_scene, past_scene);
        if sim > 0.75 {
            // Yüksek benzerlik → Benzerlik Arkhesi tetiklenir
            // Bu sahnenin kuralı daha önce öğrenilmiş olabilir
            eprintln!("[H-VSA] Scene similarity {:.3} → applying past rule", sim);
            // .exp dosyasındaki RULE_CONFIRMED kenarlarını önceliklendir
        }
    }
    self.episode_scenes.push(current_scene);
}
```

### Test

```rust
#[test]
fn test_identical_grids_max_similarity() {
    let mut net = ResonanceNetwork::new(200);
    let grid = Array2::from_shape_vec((3, 3), vec![
        1, 0, 2,
        0, 3, 0,
        2, 0, 1,
    ]).unwrap();
    
    let sg_a = net.build_scene_graph(&grid);
    // Aynı grid'i yeniden oluştur
    let mut net2 = ResonanceNetwork::new(200);
    let sg_b = net2.build_scene_graph(&grid);
    
    let sim = ResonanceNetwork::scene_similarity(&sg_a, &sg_b);
    assert!(sim > 0.99, "Identical grids must have similarity > 0.99, got {}", sim);
}

#[test]
fn test_different_grids_low_similarity() {
    let mut net = ResonanceNetwork::new(200);
    let grid_a = Array2::from_shape_vec((3, 3), vec![
        1, 1, 1, 1, 1, 1, 1, 1, 1
    ]).unwrap();
    let grid_b = Array2::from_shape_vec((3, 3), vec![
        2, 3, 4, 5, 6, 7, 8, 9, 1
    ]).unwrap();
    
    let sg_a = net.build_scene_graph(&grid_a);
    let sg_b = net.build_scene_graph(&grid_b);
    let sim = ResonanceNetwork::scene_similarity(&sg_a, &sg_b);
    
    assert!(sim < 0.3, "Very different grids should have similarity < 0.3, got {}", sim);
}

#[test]
fn test_vsa_binding_produces_unique_signature() {
    let color_sig = ResonanceNetwork::generate_xorshift_signature(12345, 4096);
    let pos_sig   = ResonanceNetwork::generate_xorshift_signature(67890, 4096);
    let bound     = ResonanceNetwork::vsa_bind(&color_sig, &pos_sig);
    
    // Bound imzası ne color ne pos ile aynı olmamalı
    let sim_color = ResonanceNetwork::holographic_similarity(&bound, &color_sig);
    let sim_pos   = ResonanceNetwork::holographic_similarity(&bound, &pos_sig);
    
    assert!(sim_color < 0.5, "Bound sig must differ from color sig");
    assert!(sim_pos   < 0.5, "Bound sig must differ from pos sig");
}
```

---

## Üç Tekniğin Birleşimi — ARC-AGI-3 Tam Akışı

```
Yeni grid gelir
  │
  ├─ build_scene_graph(grid) → ArcSceneGraph
  │   └─ H-VSA: her nesne bind(renk, konum) → süperpozisyon → scene_signature
  │
  ├─ scene_similarity(current, past_scenes) > 0.75?
  │   ├─ EVET → RULE_CONFIRMED kenarları öncelikli, kısa Beam Search
  │   └─ HAYIR → MCTS başlar
  │                │
  │                ├─ Her rollout başında: checkpoint = net_g.snapshot()  [Virtual Reset]
  │                ├─ Rollout koşar, dal denenir
  │                └─ Rollout biter: net_g.restore_snapshot(checkpoint)
  │
  ├─ find_hamiltonian_trajectory()                          [Manhattan Gradient]
  │   └─ semantic_force = weight / (manhattan_dist + 1e-6) + target_bonus
  │       → sparse çözüm → daha az ara düğüm → daha az hamle
  │
  └─ Eylem seçilir, ortama gönderilir
      → .exp'e RULE_CONFIRMED veya TRANSITION_FAILED yazılır
      → Bir sonraki adımda bu bilgi net_g'ye linker ile enjekte edilir
```

---

## Verification Plan

```powershell
# Birim testler
cargo test -p sodals_neuro --lib semantic_network::tests::test_manhattan_gradient_prefers_shorter_path
cargo test -p sodals_neuro --lib semantic_network::tests::test_virtual_reset_no_contamination
cargo test -p sodals_neuro --lib semantic_network::tests::test_snapshot_restore_preserves_holographic_memory
cargo test -p sodals_neuro --lib semantic_network::tests::test_identical_grids_max_similarity
cargo test -p sodals_neuro --lib semantic_network::tests::test_different_grids_low_similarity
cargo test -p sodals_neuro --lib semantic_network::tests::test_vsa_binding_produces_unique_signature

# Tam regresyon
cargo test -p sodals_neuro
```

**Regresyon kontrol noktaları:**
- `test_energy_conservation_undamped` hâlâ geçmeli — Manhattan değişikliği Symplectic Euler'i bozmamalı
- `test_lu_determinant_finite` hâlâ geçmeli — cache invalidation etkilenmemeli
- FailureSensorIQ %90 skoru düşmemeli — semantic_network değişiklikleri FSIQ pipeline'ını etkilememeli

---

## Öncelik Sırası

| Adım | İş | Süre | Risk |
|------|-----|------|------|
| 1 | `holographic_similarity()` + `vsa_bind()` + `vsa_superpose()` yaz | 1 gün | Düşük |
| 2 | `snapshot()` + `restore_snapshot()` yaz | 1 gün | Düşük |
| 3 | Manhattan gradient — `find_hamiltonian_trajectory()` güncelle | 1 gün | Düşük |
| 4 | `build_scene_graph()` yaz | 2 gün | Orta |
| 5 | `MctsSearch::rollout()` virtual reset entegrasyonu | 1 gün | Orta |
| 6 | `agent_loop.rs` scene similarity kontrolü | 1 gün | Orta |
| 7 | Tüm testler + regresyon | 1 gün | — |

**Toplam: ~8 iş günü**

`get_or_create_signature()`'daki Xorshift64* implementasyonu
`qmem_compiler.rs`'deki `generate_holographic_signature()` ile aynı mantığı kullanıyor.
Ortak bir modüle (`sodals_neuro::vsa_ops`) taşımak uzun vadede tekrar kodu önler.
