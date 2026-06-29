# Implementation Plan
## MAB + CDCL + Bifurcation — Kartridj Kütüphanesi
### SODALS Core — Koddan Türetilmiş Spesifikasyon

**Tarih:** 18 Haziran 2026  
**Hedef:** Büyük kartridj kütüphanesinde doğru kartridji seçmek, hataları tekrar etmemek, kritik karar noktasını tespit etmek  
**Etkilenen dosyalar:** `llm_engine.rs`, `semantic_network.rs`, `agent_loop.rs`, `qmem_compiler.rs`

---

## Mevcut Kod Analizi — Ne Var, Ne Eksik

### `load_qmem_cartridge()` — `llm_engine.rs:461`

```rust
// ŞU AN:
pub fn load_qmem_cartridge(&mut self, path: &str) -> anyhow::Result<()> {
    let new_network = ResonanceNetwork::load_cartridge(path)?;
    if let Ok(mut mem) = self.semantic_memory.write() {
        *mem = new_network;  // FULL REPLACE — önceki ağ tamamen siliniyor
    }
    self.active_steering = None;
    if let Ok(mut srf) = self.srf.lock() {
        srf.bx = 0.0;  // BX sıfırlanıyor — geçmiş bilgi kayboluyor
        srf.dx = 0.0;
    }
}
```

**Sorun:** Hangi kartridj ne zaman yüklendi, ne kadar başarılı oldu — hiçbir kayıt yok.  
Kartridj kütüphanesi büyüdükçe hangi kartridji seçeceğini bilmiyor.

### `SemanticRegisterFile` — `semantic_parser.rs:1017`

```
AX — aktif odak (token fazı)
BX — base context (sistem promptu)
DX — data register (QMEM enjeksiyonu)
CX_W — phase variance
CX_J — chaos jacobian proxy
```

**Sorun:** Kartridj seçim skoru, başarı geçmişi, hata clause'ları burada yok.

### `check_phase_variance()` — `semantic_network.rs`

Wronskian `W(f,g)` tek anlık ölçüm yapıyor. Zaman serisi yok — bifurcation tespiti imkansız.

### `.exp` dosyası — `agent_loop.rs`

`TRANSITION_FAILED` kenarları var ama **clause öğrenimi** yok:  
"A durumunda B eylemini yapma ÇÜNKÜ C çelişkisi çıkar" yapısı yok.

---

## Bölüm 1: Multi-Armed Bandit (MAB) — Kartridj Seçici

### Fikir

Her QMEM kartridji bir "kol" (arm). Her sorgu sonrası ödül gözlemleniyor:  
- Wronskian > threshold → başarılı enjeksiyon → ödül +1  
- Wronskian ≈ 0 → kartridj sorguyla ilgisiz → ödül 0  
- God's Gate kapandı → hallüsinasyon → ödül -1  

UCB1 formülü hangi kartridji deneyelim sorusuna cevap veriyor:  
`UCB1(i) = Q(i) + C * sqrt(ln(N) / n_i)`  
`Q(i)` = kartridj i'nin ortalama ödülü  
`N` = toplam deneme sayısı  
`n_i` = kartridj i'nin deneme sayısı  
`C` = keşif katsayısı (önerilen: 1.414 = sqrt(2))

### Yeni Struct: `CartridgeLibrary` — `llm_engine.rs`

```rust
/// MAB tabanlı kartridj kütüphanesi
/// Her .qmem dosyası bir "kol" — UCB1 ile en iyi seçilir
#[derive(Debug, Clone)]
pub struct CartridgeArm {
    /// Kartridj dosya yolu
    pub path: String,
    /// Toplam deneme sayısı
    pub pulls: u32,
    /// Kümülatif ödül toplamı
    pub reward_sum: f64,
    /// Son başarı oranı (rolling average, son 10 deneme)
    pub recent_rewards: std::collections::VecDeque<f64>,
    /// Bu kartridj hangi domain etiketlerine sahip (DSL'den)
    pub domain_tags: Vec<String>,
    /// Kartridj VSA imzası özeti — sorgu benzerliği için
    /// full 4096-D değil, ilk 64 boyutu (hız için)
    pub signature_summary: Vec<num_complex::Complex64>,
}

impl CartridgeArm {
    pub fn new(path: &str, domain_tags: Vec<String>) -> Self {
        Self {
            path: path.to_string(),
            pulls: 0,
            reward_sum: 0.0,
            recent_rewards: std::collections::VecDeque::with_capacity(10),
            domain_tags,
            signature_summary: Vec::new(),
        }
    }

    /// UCB1 skoru — yüksek skor = daha önce denenmeyi hak ediyor
    pub fn ucb1_score(&self, total_pulls: u32, exploration_c: f64) -> f64 {
        if self.pulls == 0 {
            return f64::INFINITY; // Hiç denenmemiş → önce dene
        }
        let q = self.reward_sum / self.pulls as f64;
        let exploration = exploration_c * ((total_pulls as f64).ln() / self.pulls as f64).sqrt();
        q + exploration
    }

    /// Ödül kaydet (rolling window)
    pub fn record_reward(&mut self, reward: f64) {
        self.pulls += 1;
        self.reward_sum += reward;
        self.recent_rewards.push_back(reward);
        if self.recent_rewards.len() > 10 {
            self.recent_rewards.pop_front();
        }
    }

    /// Son 10 denemenin ortalaması — trend tespiti için
    pub fn recent_average(&self) -> f64 {
        if self.recent_rewards.is_empty() { return 0.0; }
        self.recent_rewards.iter().sum::<f64>() / self.recent_rewards.len() as f64
    }
}

/// Tüm kütüphane yöneticisi
pub struct CartridgeLibrary {
    pub arms: Vec<CartridgeArm>,
    pub total_pulls: u32,
    pub exploration_c: f64,
    /// Şu an yüklü olan kartridj index'i
    pub active_arm: Option<usize>,
}

impl CartridgeLibrary {
    pub fn new(exploration_c: f64) -> Self {
        Self {
            arms: Vec::new(),
            total_pulls: 0,
            exploration_c,
            active_arm: None,
        }
    }

    /// Kütüphaneye yeni kartridj ekle
    pub fn register(&mut self, path: &str, domain_tags: Vec<String>) {
        self.arms.push(CartridgeArm::new(path, domain_tags));
    }

    /// UCB1 ile en iyi kartridji seç
    /// Ek filtre: sorgu VSA benzerliği — alakasız kartridj seçilmez
    pub fn select_best(
        &self,
        query_domain_hint: Option<&str>,
    ) -> Option<usize> {
        if self.arms.is_empty() { return None; }

        // Domain hint filtresi — varsa alakasız kolları hariç tut
        let candidates: Vec<usize> = if let Some(hint) = query_domain_hint {
            let filtered: Vec<usize> = self.arms.iter().enumerate()
                .filter(|(_, arm)| {
                    arm.domain_tags.iter().any(|tag| tag.contains(hint))
                })
                .map(|(i, _)| i)
                .collect();
            // Eşleşen yoksa tümünü dene
            if filtered.is_empty() {
                (0..self.arms.len()).collect()
            } else {
                filtered
            }
        } else {
            (0..self.arms.len()).collect()
        };

        // UCB1 ile en yüksek skor
        candidates.into_iter().max_by(|&a, &b| {
            let score_a = self.arms[a].ucb1_score(self.total_pulls, self.exploration_c);
            let score_b = self.arms[b].ucb1_score(self.total_pulls, self.exploration_c);
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Gözlemlenen ödülü ilgili kola kaydet
    pub fn record_reward(&mut self, arm_idx: usize, reward: f64) {
        if arm_idx < self.arms.len() {
            self.arms[arm_idx].record_reward(reward);
            self.total_pulls += 1;
        }
    }
}
```

### `LlmEngine`'e Entegrasyon — `llm_engine.rs`

```rust
pub struct LlmEngine {
    // ... mevcut alanlar ...
    
    /// MAB kartridj kütüphanesi — yeni eklendi
    pub cartridge_library: Option<CartridgeLibrary>,
}

impl LlmEngine {
    /// Kütüphaneyi başlat, tüm .qmem dosyalarını tara ve kaydet
    pub fn init_cartridge_library(&mut self, cartridge_dir: &str, exploration_c: f64) {
        let mut library = CartridgeLibrary::new(exploration_c);
        
        if let Ok(entries) = std::fs::read_dir(cartridge_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "qmem").unwrap_or(false) {
                    let path_str = path.to_string_lossy().to_string();
                    // Domain tag'ı dosya adından çıkar (örn: "industrial_topology" → "industrial")
                    let domain_tag = path.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.split('_').next().unwrap_or(s).to_string())
                        .unwrap_or_default();
                    library.register(&path_str, vec![domain_tag]);
                    eprintln!("[MAB] Registered cartridge: {}", path_str);
                }
            }
        }
        
        self.cartridge_library = Some(library);
        eprintln!("[MAB] Cartridge library initialized with {} arms", 
            self.cartridge_library.as_ref().map(|l| l.arms.len()).unwrap_or(0));
    }

    /// MAB ile kartridj seç ve yükle
    pub fn load_best_cartridge(&mut self, domain_hint: Option<&str>) -> anyhow::Result<()> {
        let (arm_idx, path) = {
            let library = self.cartridge_library.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Cartridge library not initialized"))?;
            let idx = library.select_best(domain_hint)
                .ok_or_else(|| anyhow::anyhow!("No cartridge available"))?;
            (idx, library.arms[idx].path.clone())
        };

        self.load_qmem_cartridge(&path)?;
        
        if let Some(lib) = &mut self.cartridge_library {
            lib.active_arm = Some(arm_idx);
        }
        
        eprintln!("[MAB] Selected arm #{}: {}", arm_idx, path);
        Ok(())
    }

    /// Ödül gözlemle — her inference sonrası çağrılır
    /// Wronskian determinantı ödül sinyali olarak kullanılır
    pub fn observe_cartridge_reward(&mut self) {
        let wronskian = if let Ok(mem) = self.semantic_memory.read() {
            mem.compute_determinant_lu()
        } else { 0.0 };

        // Ödül fonksiyonu:
        // det > 1.0  → güçlü lineer bağımsızlık → kartridj alakalı → +1.0
        // det ∈ [0.1, 1.0] → kısmi ilgi → +0.3
        // det < 0.1  → neredeyse bağımlı → kartridj alakasız → 0.0
        // God's Gate kapandıysa (cx_w < 0.2) → -0.5
        let cx_w = if let Ok(srf) = self.srf.lock() { srf.cx_w } else { 0.5 };

        let reward = if cx_w < 0.2 {
            -0.5 // Hallüsinasyon sinyali
        } else if wronskian > 1.0 {
            1.0
        } else if wronskian > 0.1 {
            0.3
        } else {
            0.0
        };

        if let Some(lib) = &mut self.cartridge_library {
            if let Some(arm_idx) = lib.active_arm {
                lib.record_reward(arm_idx, reward);
                eprintln!("[MAB] Arm #{} reward: {:.2} (det={:.4}, cx_w={:.3})", 
                    arm_idx, reward, wronskian, cx_w);
            }
        }
    }
}
```

### Test

```rust
#[test]
fn test_mab_unexplored_arm_selected_first() {
    let mut lib = CartridgeLibrary::new(1.414);
    lib.register("a.qmem", vec!["physics".to_string()]);
    lib.register("b.qmem", vec!["protocol".to_string()]);
    lib.register("c.qmem", vec!["chemistry".to_string()]);

    // Hiç denenmemiş → hepsi infinity → ilk seçilen 0 olmalı
    let selected = lib.select_best(None).unwrap();
    assert_eq!(selected, 0, "First unexplored arm should be selected");

    // 0'ı dene, yüksek ödül ver
    lib.record_reward(0, 1.0);
    // Şimdi 1 seçilmeli (unexplored → infinity)
    let selected = lib.select_best(None).unwrap();
    assert_eq!(selected, 1);
}

#[test]
fn test_mab_domain_hint_filters_arms() {
    let mut lib = CartridgeLibrary::new(1.414);
    lib.register("industrial.qmem", vec!["industrial".to_string()]);
    lib.register("math.qmem", vec!["math".to_string()]);
    lib.register("industrial2.qmem", vec!["industrial".to_string()]);

    // "industrial" hint ile sadece industrial kartridj seçilmeli
    let selected = lib.select_best(Some("industrial")).unwrap();
    assert!(selected == 0 || selected == 2, 
        "Domain hint should filter to industrial cartridges");
}

#[test]
fn test_mab_ucb1_prefers_high_reward_with_low_pull() {
    let mut lib = CartridgeLibrary::new(1.414);
    lib.register("a.qmem", vec![]);
    lib.register("b.qmem", vec![]);

    // a: 10 deneme, düşük ödül
    for _ in 0..10 { lib.record_reward(0, 0.1); }
    // b: 1 deneme, yüksek ödül
    lib.record_reward(1, 1.0);

    // UCB1 b'yi tercih etmeli (yüksek ödül + az deneme = yüksek UCB1)
    let score_a = lib.arms[0].ucb1_score(lib.total_pulls, lib.exploration_c);
    let score_b = lib.arms[1].ucb1_score(lib.total_pulls, lib.exploration_c);
    assert!(score_b > score_a, "High reward + low pulls = higher UCB1");
}
```

---

## Bölüm 2: CDCL (Conflict-Driven Clause Learning) — Hata Öğrenimi

### Fikir

SAT solver'lardan gelen teknik. Şu an `.exp` dosyasındaki `TRANSITION_FAILED` kenarları  
"bu hamle kötüydü" diyor ama **neden** kötüydü söylemiyor.

CDCL şunu ekliyor:  
`"agent_at_x_3 ∧ color_blue ∧ action_right → CONFLICT"`  
= "x=3, mavi renkte, sağa git → çelişki"  

Bu clause bir daha aynı durumda aynı eylemi engeller.  
Ve **daha da güçlü**: bu clause'dan türetilen başka çelişkileri de bloke eder.

### Yeni Struct: `CdclClauseDb` — `agent_loop.rs`'e eklenecek

```rust
/// Öğrenilmiş çelişki clause'u
/// SAT formülü: literals → CONFLICT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictClause {
    /// Çelişkiyi tetikleyen koşullar (QMEM state hash'leri)
    pub antecedents: Vec<String>,
    /// Hangi eylem bu çelişkiye yol açtı
    pub conflicting_action: String,
    /// Kaç kez gözlemlendi (frekans = güven skoru)
    pub frequency: u32,
    /// İlk gözlem zamanı (step sayısı)
    pub first_seen: u64,
}

impl ConflictClause {
    /// Bu clause mevcut durumla eşleşiyor mu?
    /// Tüm antecedent'lar mevcut state'te varsa → eşleşme
    pub fn matches(&self, current_state_keys: &[String]) -> bool {
        let state_set: std::collections::HashSet<&String> = current_state_keys.iter().collect();
        self.antecedents.iter().all(|a| state_set.contains(a))
    }
}

/// CDCL clause veritabanı
pub struct CdclClauseDb {
    pub clauses: Vec<ConflictClause>,
    /// Hızlı lookup: action → bu eylemi içeren clause index'leri
    action_index: std::collections::HashMap<String, Vec<usize>>,
    /// İstatistik
    pub total_conflicts: u64,
    pub clauses_fired: u64,
}

impl CdclClauseDb {
    pub fn new() -> Self {
        Self {
            clauses: Vec::new(),
            action_index: std::collections::HashMap::new(),
            total_conflicts: 0,
            clauses_fired: 0,
        }
    }

    /// Yeni çelişki öğren veya mevcut clause'u güçlendir
    pub fn learn_conflict(
        &mut self,
        antecedents: Vec<String>,
        conflicting_action: String,
        current_step: u64,
    ) {
        self.total_conflicts += 1;

        // Aynı clause daha önce öğrenildi mi?
        let existing = self.clauses.iter_mut().find(|c| {
            c.conflicting_action == conflicting_action &&
            c.antecedents.len() == antecedents.len() &&
            antecedents.iter().all(|a| c.antecedents.contains(a))
        });

        if let Some(clause) = existing {
            clause.frequency += 1;
            eprintln!("[CDCL] Conflict reinforced (freq={}): {:?} → {}",
                clause.frequency, clause.antecedents, clause.conflicting_action);
        } else {
            let idx = self.clauses.len();
            self.clauses.push(ConflictClause {
                antecedents: antecedents.clone(),
                conflicting_action: conflicting_action.clone(),
                frequency: 1,
                first_seen: current_step,
            });
            self.action_index
                .entry(conflicting_action.clone())
                .or_default()
                .push(idx);
            eprintln!("[CDCL] New conflict learned: {:?} → {}", antecedents, conflicting_action);
        }
    }

    /// Bu eylem mevcut durumda bloke edilmeli mi?
    /// Frekans eşiğini geçen clause'lar güvenilir kabul edilir
    pub fn is_action_blocked(
        &self,
        action: &str,
        current_state_keys: &[String],
        min_frequency: u32,
    ) -> bool {
        let Some(indices) = self.action_index.get(action) else { return false; };

        indices.iter().any(|&idx| {
            let clause = &self.clauses[idx];
            clause.frequency >= min_frequency && clause.matches(current_state_keys)
        })
    }

    /// Tüm bloke edilmiş eylemleri döndür (MCTS budama için)
    pub fn blocked_actions(
        &self,
        current_state_keys: &[String],
        min_frequency: u32,
    ) -> Vec<String> {
        let mut blocked = Vec::new();
        for clause in &self.clauses {
            if clause.frequency >= min_frequency && clause.matches(current_state_keys) {
                if !blocked.contains(&clause.conflicting_action) {
                    blocked.push(clause.conflicting_action.clone());
                }
            }
        }
        blocked
    }

    /// QMEM DSL'e dışa aktar — .exp dosyasıyla entegrasyon
    /// Her clause bir QMEM satırına dönüşür
    pub fn export_to_exp(&self, min_frequency: u32) -> String {
        let mut lines = Vec::new();
        for clause in &self.clauses {
            if clause.frequency < min_frequency { continue; }
            // "obs_x_3|obs_color_blue" TRANSITION_FAILED "action_right" [freq/10, 0.0];
            let antecedent = clause.antecedents.join("|");
            let weight = (clause.frequency as f64 / 10.0).min(1.0);
            lines.push(format!(
                "\"{}\" TRANSITION_FAILED \"{}\" [{:.2}, 0.0];",
                antecedent, clause.conflicting_action, weight
            ));
        }
        lines.join("\n")
    }

    /// Disk'ten yükle (JSON formatı)
    pub fn load_from_file(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(json) => serde_json::from_str(&json).unwrap_or_else(|_| Self::new()),
            Err(_) => Self::new(),
        }
    }

    /// Disk'e kaydet
    pub fn save_to_file(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(&self.clauses)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
```

### `AgentLoop`'a Entegrasyon — `agent_loop.rs`

```rust
pub struct AgentLoop {
    // ... mevcut alanlar ...

    /// CDCL clause veritabanı — yeni eklendi
    pub clause_db: CdclClauseDb,
    /// Mevcut adım sayacı (clause zaman damgası için)
    pub step_counter: u64,
}

impl AgentLoop {
    pub fn run(&mut self, env: &mut dyn ArcEnvironment) {
        // Episode başında CDCL dosyasını yükle
        self.clause_db = CdclClauseDb::load_from_file("agent_clauses.json");

        loop {
            let obs = env.observe();
            let state_keys = self.get_qmem_state_hash(&obs);

            // CDCL: bu adımda bloke edilmiş eylemleri hesapla
            let blocked = self.clause_db.blocked_actions(&state_keys, 2); // min 2 gözlem
            if !blocked.is_empty() {
                eprintln!("[CDCL] Blocking {} actions: {:?}", blocked.len(), blocked);
            }

            // MCTS arama — bloke edilmiş eylemleri hariç tut
            let action = self.mcts_search_with_exclusions(&obs, &blocked);

            let (next_obs, reward, done) = env.step(&action);
            self.step_counter += 1;

            // Çelişki tespiti
            if reward < -0.5 {
                // Başarısız geçiş → clause öğren
                self.clause_db.learn_conflict(
                    state_keys.clone(),
                    action.clone(),
                    self.step_counter,
                );

                // .exp dosyasını güncelle (her 10 adımda bir)
                if self.step_counter % 10 == 0 {
                    let exp_content = self.clause_db.export_to_exp(2);
                    if let Ok(mut file) = std::fs::OpenOptions::new()
                        .append(true).open("agent_experience.exp") {
                        use std::io::Write;
                        let _ = writeln!(file, "{}", exp_content);
                    }
                }
            }

            // .exp dosyasını her adımda .qmem'e link et
            // (mevcut agent_loop mantığı — değişmeden kalıyor)

            if done { break; }
        }

        // Episode sonunda clause DB'yi kaydet
        let _ = self.clause_db.save_to_file("agent_clauses.json");
    }
}
```

### `MctsSearch`'e CDCL Budama

```rust
// mcts_exploration.rs içinde:

impl MctsSearch {
    /// Yasal eylemler listesinden CDCL bloke edilmiş eylemleri çıkar
    fn filter_actions_with_cdcl(
        &self,
        actions: Vec<String>,
        state_keys: &[String],
        clause_db: &CdclClauseDb,
    ) -> Vec<String> {
        actions.into_iter()
            .filter(|action| !clause_db.is_action_blocked(action, state_keys, 2))
            .collect()
    }
}
```

### Test

```rust
#[test]
fn test_cdcl_learns_and_blocks() {
    let mut db = CdclClauseDb::new();

    // Çelişkiyi öğret
    db.learn_conflict(
        vec!["obs_x_3".to_string(), "obs_color_blue".to_string()],
        "action_right".to_string(),
        1,
    );
    db.learn_conflict( // İkinci gözlem → frequency 2 → güvenilir
        vec!["obs_x_3".to_string(), "obs_color_blue".to_string()],
        "action_right".to_string(),
        2,
    );

    // Aynı durum → eylem bloke edilmeli
    let state = vec!["obs_x_3".to_string(), "obs_color_blue".to_string()];
    assert!(db.is_action_blocked("action_right", &state, 2),
        "Learned conflict should block the action");

    // Farklı durum → eylem serbest
    let other_state = vec!["obs_x_5".to_string(), "obs_color_red".to_string()];
    assert!(!db.is_action_blocked("action_right", &other_state, 2),
        "Different state should not be blocked");
}

#[test]
fn test_cdcl_export_to_qmem_dsl() {
    let mut db = CdclClauseDb::new();
    db.learn_conflict(vec!["obs_wall".to_string()], "action_forward".to_string(), 1);
    db.learn_conflict(vec!["obs_wall".to_string()], "action_forward".to_string(), 1);

    let exp = db.export_to_exp(2);
    assert!(exp.contains("TRANSITION_FAILED"), "Should export as QMEM DSL");
    assert!(exp.contains("action_forward"), "Should contain the conflicting action");
}
```

---

## Bölüm 3: Bifurcation (Topolojik Çatallanma) Tespiti

### Fikir

`compute_determinant_lu()` anlık Wronskian veriyor.  
Bifurcation = Wronskian'ın zaman serisinde niteliksel davranış değişikliği.

Matematiksel olarak: Lyapunov üsteli `λ = lim(t→∞) (1/t) * ln|W(t) / W(0)|`  
- `λ > 0` → kaotik (zaman içinde Wronskian büyüyor) → bifurcation noktası yakın  
- `λ < 0` → kararlı (Wronskian küçülüyor) → güvenli bölge  
- `λ ≈ 0` → kritik nokta — tam bifurcation eşiği  

Pratik implementasyon: son N Wronskian değerinin logaritmik türevi.  
Türev işaret değiştirdiğinde → bifurcation.

### Yeni Struct: `BifurcationMonitor` — `semantic_network.rs`'e eklenecek

```rust
/// Wronskian zaman serisi izleyici
/// Bifurcation noktası = sistemin kararlı → kaotik geçiş yaptığı an
#[derive(Debug, Clone)]
pub struct BifurcationMonitor {
    /// Son N Wronskian değerleri (sliding window)
    wronskian_history: std::collections::VecDeque<f64>,
    /// Pencere boyutu
    window_size: usize,
    /// Bifurcation eşiği — bu değerin üzerinde kaotik
    pub chaos_threshold: f64,
    /// Lyapunov üsteli tahmini
    pub lyapunov_estimate: f64,
    /// Son bifurcation'ın tespit edildiği adım
    pub last_bifurcation_step: Option<u64>,
    /// Toplam adım
    pub step: u64,
}

impl BifurcationMonitor {
    pub fn new(window_size: usize, chaos_threshold: f64) -> Self {
        Self {
            wronskian_history: std::collections::VecDeque::with_capacity(window_size),
            window_size,
            chaos_threshold,
            lyapunov_estimate: 0.0,
            last_bifurcation_step: None,
            step: 0,
        }
    }

    /// Her Symplectic Euler adımında çağrılır
    pub fn observe(&mut self, wronskian: f64) {
        self.step += 1;
        self.wronskian_history.push_back(wronskian.abs().max(1e-10));
        if self.wronskian_history.len() > self.window_size {
            self.wronskian_history.pop_front();
        }

        // Yeterli geçmiş varsa Lyapunov üsteli hesapla
        if self.wronskian_history.len() >= 4 {
            self.update_lyapunov();
        }
    }

    /// Lyapunov üstelini güncelle
    /// λ ≈ (1/N) * Σ ln|W(t+1) / W(t)|
    fn update_lyapunov(&mut self) {
        let history: Vec<f64> = self.wronskian_history.iter().copied().collect();
        let n = history.len();
        if n < 2 { return; }

        let log_ratios: Vec<f64> = history.windows(2)
            .map(|w| (w[1] / w[0]).abs().ln())
            .collect();

        self.lyapunov_estimate = log_ratios.iter().sum::<f64>() / log_ratios.len() as f64;
    }

    /// Bifurcation var mı?
    /// λ pozitif ve eşik üzerinde → kaotik geçiş
    pub fn is_bifurcating(&mut self) -> bool {
        if self.lyapunov_estimate > self.chaos_threshold {
            self.last_bifurcation_step = Some(self.step);
            true
        } else {
            false
        }
    }

    /// GPR'e dinamik eşik ver
    /// Bifurcation varsa → daha derin MCTS
    /// Kararlıysa → hızlı Beam Search
    pub fn gpr_threshold(&self) -> f64 {
        if self.lyapunov_estimate > 0.0 {
            // Kaotik → daha hassas GPR (düşük eşik = daha çok katman aktive)
            1.0 - self.lyapunov_estimate.min(0.9)
        } else {
            // Kararlı → hızlı geç
            1.96 // standart Z-score eşiği
        }
    }

    /// Mevcut rejim
    pub fn regime(&self) -> &'static str {
        if self.lyapunov_estimate > self.chaos_threshold {
            "CHAOTIC"
        } else if self.lyapunov_estimate > 0.0 {
            "TRANSITIONAL"
        } else {
            "STABLE"
        }
    }
}
```

### `ResonanceNetwork`'e Entegrasyon — `semantic_network.rs`

```rust
pub struct ResonanceNetwork {
    // ... mevcut alanlar ...
    
    /// Bifurcation monitörü — yeni eklendi
    pub bifurcation_monitor: BifurcationMonitor,
}

impl ResonanceNetwork {
    pub fn new(capacity: usize) -> Self {
        Self {
            // ... mevcut init ...
            bifurcation_monitor: BifurcationMonitor::new(20, 0.3),
        }
    }
}
```

`symplectic_step()` içine Wronskian gözlemi ekle:

```rust
pub fn step(&mut self, external_forces: &[Complex64]) {
    // ... mevcut symplectic_step kodu ...

    // Bifurcation izleme — her adımda
    // NOT: compute_determinant_lu() pahalı, her 5 adımda bir
    if self.step_count % 5 == 0 {
        let det = self.compute_determinant_lu();
        self.bifurcation_monitor.observe(det);
        
        if self.bifurcation_monitor.is_bifurcating() {
            eprintln!(
                "[BIFURCATION] ⚠️  Regime: {}, λ={:.4}, step={}",
                self.bifurcation_monitor.regime(),
                self.bifurcation_monitor.lyapunov_estimate,
                self.bifurcation_monitor.step
            );
        }
    }
}
```

### `LlmEngine`'e Bifurcation → MAB Geri Bildirimi

```rust
impl LlmEngine {
    /// Bifurcation tespiti → MAB ödülünü negatif yap + MCTS derinleştir
    pub fn check_bifurcation_and_adapt(&mut self) -> BifurcationAction {
        let (regime, lyapunov) = if let Ok(mem) = self.semantic_memory.read() {
            (
                mem.bifurcation_monitor.regime(),
                mem.bifurcation_monitor.lyapunov_estimate,
            )
        } else {
            ("STABLE", 0.0)
        };

        match regime {
            "CHAOTIC" => {
                // Mevcut kartridj sistemi kararsızlaştırıyor → MAB ceza
                self.observe_cartridge_reward(); // reward zaten -0.5 alacak
                // MCTS derinliğini artır
                BifurcationAction::DeepMcts { extra_depth: 3 }
            }
            "TRANSITIONAL" => {
                // Dikkatli ol — alternatif kartridj dene
                BifurcationAction::TryAlternative
            }
            _ => {
                // Kararlı — normal akış
                BifurcationAction::Normal
            }
        }
    }
}

pub enum BifurcationAction {
    Normal,
    TryAlternative,
    DeepMcts { extra_depth: u32 },
}
```

### `AgentLoop`'a Bifurcation Entegrasyonu

```rust
impl AgentLoop {
    pub fn run(&mut self, env: &mut dyn ArcEnvironment) {
        loop {
            // ... gözlem, CDCL budama, eylem seçimi ...

            // Her 5 adımda bifurcation kontrol
            if self.step_counter % 5 == 0 {
                match self.engine.check_bifurcation_and_adapt() {
                    BifurcationAction::DeepMcts { extra_depth } => {
                        // MCTS'in rollout derinliğini geçici artır
                        self.mcts_config.rollout_depth += extra_depth;
                        eprintln!("[BIFURCATION] MCTS depth increased to {}", 
                            self.mcts_config.rollout_depth);
                    }
                    BifurcationAction::TryAlternative => {
                        // MAB'dan farklı bir kartridj yükle (mevcut kolu geç)
                        if let Some(lib) = &mut self.engine.cartridge_library {
                            // Mevcut kolu geçici olarak düşük skor ver
                            if let Some(arm) = lib.active_arm {
                                lib.record_reward(arm, -0.3);
                            }
                        }
                        let _ = self.engine.load_best_cartridge(None);
                    }
                    BifurcationAction::Normal => {}
                }
            }

            // MCTS derinliği bifurcation sonrası normala dön
            if self.step_counter % 20 == 0 {
                self.mcts_config.rollout_depth = self.mcts_config.default_rollout_depth;
            }
        }
    }
}
```

### Test

```rust
#[test]
fn test_bifurcation_monitor_detects_growth() {
    let mut monitor = BifurcationMonitor::new(10, 0.3);

    // Büyüyen Wronskian serisi → Lyapunov pozitif
    for i in 1..15 {
        monitor.observe(i as f64 * 2.0); // üstel büyüme
    }

    assert!(monitor.lyapunov_estimate > 0.0,
        "Growing Wronskian should yield positive Lyapunov: λ={}", 
        monitor.lyapunov_estimate);
}

#[test]
fn test_bifurcation_monitor_stable_regime() {
    let mut monitor = BifurcationMonitor::new(10, 0.3);

    // Kararlı Wronskian → Lyapunov negatif
    for _ in 0..15 {
        monitor.observe(1.0 + (rand::random::<f64>() * 0.01)); // küçük gürültü
    }

    assert_eq!(monitor.regime(), "STABLE");
}

#[test]
fn test_bifurcation_threshold_triggers_action() {
    let mut monitor = BifurcationMonitor::new(5, 0.1); // düşük eşik

    // Hızlı büyüme
    for i in 1..10 {
        monitor.observe(10f64.powi(i));
    }

    assert!(monitor.is_bifurcating(), "Should detect bifurcation with exponential growth");
    assert_eq!(monitor.regime(), "CHAOTIC");
}
```

---

## Üç Tekniğin Birleşimi — Kartridj Kütüphanesi Tam Akışı

```
Sistem başlar
  │
  ├─ init_cartridge_library("qmem_crystals/") → 50+ kartridj taranır, MAB başlar
  ├─ CdclClauseDb::load_from_file() → önceki session'ların hatalar yüklenir
  │
  Sorgu gelir
  │
  ├─ GPR: domain_hint tespit eder (node_flags + query VSA)
  ├─ MAB: load_best_cartridge(domain_hint) → UCB1 ile en iyi seçilir
  ├─ CDCL: blocked_actions() → bu durumda yasaklı eylemler hesaplanır
  ├─ MCTS: bloke edilmiş eylemleri hariç tutarak arar
  │
  Her 5 adımda:
  │
  ├─ BifurcationMonitor: Wronskian gözlemi → λ hesabı
  ├─ λ > eşik → KAOTIK → MCTS derinleştir + MAB'a ceza
  ├─ λ ∈ (0, eşik) → GEÇİŞ → alternatif kartridj dene
  ├─ λ < 0 → KARARLI → hızlı Beam Search
  │
  Eylem sonrası:
  │
  ├─ Başarısız → CDCL: learn_conflict() → clause veritabanına ekle
  ├─ MAB: observe_cartridge_reward() → Wronskian → ödül → UCB1 güncelleme
  ├─ Her 10 adımda: clause_db.export_to_exp() → .exp dosyasına TRANSITION_FAILED kenarları
  │
  Episode sonu:
  └─ clause_db.save_to_file() → kalıcı öğrenme
```

---

## Verification Plan

```powershell
# Birim testler
cargo test -p sodals_neuro --lib engine::llm_engine::tests::test_mab_unexplored_arm_selected_first
cargo test -p sodals_neuro --lib engine::llm_engine::tests::test_mab_domain_hint_filters_arms
cargo test -p sodals_neuro --lib agent_loop::tests::test_cdcl_learns_and_blocks
cargo test -p sodals_neuro --lib agent_loop::tests::test_cdcl_export_to_qmem_dsl
cargo test -p sodals_neuro --lib semantic_network::tests::test_bifurcation_monitor_detects_growth
cargo test -p sodals_neuro --lib semantic_network::tests::test_bifurcation_monitor_stable_regime

# Tam regresyon
cargo test -p sodals_neuro
cargo test -p sodals_kernel
```

**Kritik regresyon noktaları:**
- `load_qmem_cartridge()` hâlâ single-cartridge modda çalışabilmeli (library opsiyonel)
- `compute_determinant_lu()` her 5 adımda bir çağrılıyor — energy_conservation testi etkilenebilir, izle
- FailureSensorIQ skoru %90 altına düşmemeli

---

## Öncelik Sırası ve Süre

| Adım | İş | Süre | Risk |
|------|-----|------|------|
| 1 | `CartridgeArm` + `CartridgeLibrary` struct'ları | 1 gün | Düşük |
| 2 | `init_cartridge_library()` + `load_best_cartridge()` | 1 gün | Düşük |
| 3 | `observe_cartridge_reward()` — Wronskian → ödül | 0.5 gün | Düşük |
| 4 | `CdclClauseDb` + `learn_conflict()` + `is_action_blocked()` | 2 gün | Orta |
| 5 | `export_to_exp()` — CDCL → .exp entegrasyonu | 0.5 gün | Düşük |
| 6 | `BifurcationMonitor` + Lyapunov hesabı | 1 gün | Orta |
| 7 | `symplectic_step()` içine BifurcationMonitor.observe() | 0.5 gün | Düşük |
| 8 | `check_bifurcation_and_adapt()` + `BifurcationAction` | 1 gün | Orta |
| 9 | `AgentLoop::run()` üç tekniğin orkestrasyonu | 1 gün | Orta |
| 10 | Tüm testler + regresyon | 1 gün | — |

**Toplam: ~9.5 iş günü**

---

## Notlar

**`serde` bağımlılığı:** `CdclClauseDb` disk serializasyonu için `serde_json` gerektiriyor.  
`Cargo.toml`'a `serde = { features = ["derive"] }` ve `serde_json` eklenmeli.  
`sodals_neuro` crate'inde zaten `bincode` + `serde` var — `serde_json` küçük eklenti.

**`BifurcationMonitor` ve `clear_state()`:**  
`clear_state()` çağrıldığında `bifurcation_monitor`'ü sıfırlamak YANLIŞ olur —  
çünkü öğrenilmiş Lyapunov geçmişi değerli. Sadece `wronskian_history` temizlenebilir,  
`lyapunov_estimate` ve `last_bifurcation_step` korunmalı.

**Deterministik test:**  
`CartridgeLibrary::select_best()` aynı ödül geçmişiyle her zaman aynı kolu seçmeli.  
Tiebreak için `arm_idx` (küçük olan önce) kullan — rastgele tiebreak determinizmi bozar.
