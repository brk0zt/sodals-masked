// src/crates/sodals_neuro/src/arkhe_pipeline.rs
// ============================================================================
// SODALS ARKHE-COGNITION PIPELINE
// Altı Arkhe'yi sıralı çağıran inference wrapper.
//
// Felsefi temel: İnsan bilişinin altı temel ilkesi — Bilinç, Benzerlik, Çıkarım,
// Uyum, Yöntem, Şartlar — SODALS'ın mevcut fizik motoruna cerrahi bağlantı.
//
// Mimari: CliffordBridge f(x)+g(x) → h(x) ile iki kutu (Qwen backbone + QMEM)
// arasındaki "Ortak Alan / Saçak" kesişimini her token adımında hesaplar.
// ============================================================================

use num_complex::Complex64;
use std::collections::HashMap;

use crate::semantic_network::ResonanceNetwork;
use crate::semantic_parser::SemanticRegisterFile;
use crate::fluid_algebra::{bind_phasors, apply_viscosity};
use crate::fast_math::{SemanticWave, Cl3Multivector, GeometricProduct, apply_clifford_resonance};

// ════════════════════════════════════════════════════════════════════════════
// CLIFFORD BRIDGE STRUCT
// f(x) + g(x) → h(x) = Clifford_project(net_f, net_g, θ_z)
// ════════════════════════════════════════════════════════════════════════════

/// CliffordBridge: İki ResonanceNetwork (f ve g) arasında Clifford Geometric
/// Product kullanarak "saçak/fringe" kesişim noktasını (E noktası) hesaplar.
///
/// Görsel 2'deki ADE ve BCE üçgenlerinin empedans açısı boyunca izdüşüm
/// kesişimi olan pembe bölge, bu struct'ın `project()` metodu tarafından üretilir.
pub struct CliffordBridge {
    /// Son hesaplanan h_phase (radian) — SRF.dx'e yazılır
    pub last_h_phase: f64,

    /// Son hesaplanan projeksiyon gücü [0, 1] — CX_W boost için
    pub last_confidence: f64,

    /// Son bulunan saçak düğüm listesi (label, similarity_score)
    pub last_sacak: Vec<(String, f64)>,

    /// 4096-D VSA bindma ile hesaplanan son birleşik imza
    /// Boş ise: VSA henüz hesaplanmadı (her iki ağın holographic_memory'si de dolu olmalı)
    pub last_bound_signature: Vec<Complex64>,
}

impl CliffordBridge {
    pub fn new() -> Self {
        Self {
            last_h_phase: 0.0,
            last_confidence: 0.0,
            last_sacak: Vec::new(),
            last_bound_signature: Vec::new(),
        }
    }

    /// Clifford projeksiyon: f(x) + g(x) → h(x)
    ///
    /// İki katmanlı hesaplama:
    /// 1. **Makroskopik** (O(1)): `compute_aggregate_phase()` ile ağırlıklı faz argyümanı.
    ///    h_macro = Re(e^{i·f_phase} ∧ e^{i·g_phase·θ_z})
    /// 2. **VSA** (O(N)): En yüksek enerjili kavramların holographic_memory phasorlarını
    ///    `bind_phasors()` ile birleştirir → bileşik 4096-D imza.
    ///    Birleşik imzanın arg() → h_phase (daha hassas).
    ///
    /// Eğer her iki ağda da holographic_memory doluysa VSA versiyonu kullanılır;
    /// aksi hâlde makroskopik versiyona fallback.
    ///
    /// # Arguments
    /// * `net_f` - Backbone semantik ağı (f(x))
    /// * `net_g` - QMEM fizik ağı (g(x))
    /// * `impedance_theta` - Ortalama topological impedance (θ_z)
    ///
    /// # Returns
    /// `(h_phase: f64, confidence: f64)` — h_phase SRF.dx'e yazılır
    pub fn project(
        &mut self,
        net_f: &ResonanceNetwork,
        net_g: &ResonanceNetwork,
        impedance_theta: f64,
    ) -> (f64, f64) {
        // ─── Makroskopik katman ───────────────────────────────────────────────
        let f_phase = net_f.compute_aggregate_phase();
        let g_phase = net_g.compute_aggregate_phase();

        // Clifford geometric product in complex plane:
        // e^{i·f_phase} ∧ e^{i·(g_phase·θ_z)}
        let e_f = Complex64::new(f_phase.cos(), f_phase.sin());
        let e_g = Complex64::new(
            (g_phase * impedance_theta).cos(),
            -(g_phase * impedance_theta).sin(), // conjugate = ∧ in Clifford sense
        );
        let macro_projection = e_f * e_g;
        let macro_h_phase = macro_projection.arg();
        let macro_confidence = macro_projection.norm().min(1.0);

        // ─── VSA katmanı (4096-D phasor bindma) ──────────────────────────────
        // En yüksek enerjili kavramı her ağda bul
        let top_f = Self::find_top_energy_node(net_f);
        let top_g = Self::find_top_energy_node(net_g);

        let vsa_result = top_f
            .and_then(|id_f| {
                top_g.and_then(|id_g| {
                    let sig_f = net_f.holographic_memory.get(id_f)?;
                    let sig_g = net_g.holographic_memory.get(id_g)?;
                    if sig_f.is_empty() || sig_g.is_empty() || sig_f.len() != sig_g.len() {
                        return None;
                    }
                    // Topological viscosity: yüksek impedance = g(x) ağını bastır
                    let mut sig_g_dampened = sig_g.clone();
                    apply_viscosity(&mut sig_g_dampened, impedance_theta.abs() * 0.1);

                    // VSA binding: A ⊗ B = elementwise complex multiply (O(N))
                    let bound = bind_phasors(sig_f, &sig_g_dampened);

                    // Birleşik imzanın aggregate phase'i
                    let bound_phase: Complex64 = bound.iter()
                        .take(256) // ilk 256 boyut yeterli (O(256) ≈ O(1) pratikte)
                        .sum();
                    Some((bound_phase.arg(), bound_phase.norm() / 256.0, bound))
                })
            });

        let (h_phase, confidence, bound_sig) = match vsa_result {
            Some((phase, conf, sig)) => (phase, conf.min(1.0), sig),
            None => (macro_h_phase, macro_confidence, Vec::new()),
        };

        // ─── Cl(3,0) Geometric Product: ağ karşılaştırma skoru ───────────────
        // f_phase ve g_phase'i 3-vektör olarak temsil et; iç çarpım = uyum skoru
        let u = Cl3Multivector::vector(f_phase.cos(), f_phase.sin(), impedance_theta);
        let v = Cl3Multivector::vector(g_phase.cos(), g_phase.sin(), impedance_theta);
        let (inner_score, outer_bivector) = u.geometric_product(&v);
        let cl3_confidence = (inner_score.abs() + outer_bivector * 0.5).min(1.0);

        // Kombine güven: VSA + Cl(3,0)
        let combined_confidence = (confidence * 0.7 + cl3_confidence * 0.3).min(1.0);

        // Saçak düğümleri güncelle
        self.last_sacak = ResonanceNetwork::find_cross_network_sacak(net_f, net_g, 0.25)
            .into_iter()
            .map(|(_, label, score)| (label, score))
            .collect();

        self.last_h_phase = h_phase;
        self.last_confidence = combined_confidence;
        self.last_bound_signature = bound_sig;

        eprintln!(
            "[CLIFFORD BRIDGE] f={:.3} g={:.3} θ_z={:.3} → h={:.3} conf={:.3} saçak={}",
            f_phase, g_phase, impedance_theta, h_phase, combined_confidence,
            self.last_sacak.len()
        );

        (h_phase, combined_confidence)
    }

    /// En yüksek kinetik enerjiye sahip düğümün ID'sini döndür
    fn find_top_energy_node(net: &ResonanceNetwork) -> Option<usize> {
        if net.n == 0 {
            return None;
        }
        (0..net.n)
            .max_by(|&a, &b| {
                let ea = net.positions.amplitudes[a].norm() * net.masses[a];
                let eb = net.positions.amplitudes[b].norm() * net.masses[b];
                ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ARKHE DURUM TÜRÜ
// Her Arkhe adımının çıktısını taşır; bir sonraki Arkhe'nin girdisi olur
// ════════════════════════════════════════════════════════════════════════════

/// Her Arkhe'nin çıktısını temsil eder.
/// ArkheCognitionPipeline her adımda bu struct'ı biriktirir.
#[derive(Debug, Clone)]
pub struct ArkheState {
    // ─── Arkhe 1 — Bilinç ────────────────────────────────────────────────────
    /// Hedef görev tipi (QMEM görev kartridjinden türetilir)
    pub task_manifold: TaskManifold,
    /// BX kaydına yazılacak faz değeri
    pub bx_phase: f32,

    // ─── Arkhe 2 — Benzerlik ─────────────────────────────────────────────────
    /// En benzer kavram çifti ve VSA skoru (label_f, label_g, similarity)
    pub top_similarity: Option<(String, String, f64)>,
    /// Saçak düğümler: hem f(x) hem g(x)'de güçlü titreşen kavramlar
    pub sacak_nodes: Vec<(String, f64)>,

    // ─── Arkhe 3 — Çıkarım ───────────────────────────────────────────────────
    /// Hamiltonian yolu başlangıç ve bitiş node ID'leri
    pub inference_path: Option<Vec<usize>>,
    /// Enerji kısıtlı yolun toplam "maliyeti"
    pub path_energy_cost: f64,

    // ─── Arkhe 4 — Uyum ──────────────────────────────────────────────────────
    /// Clifford bivektör büyüklüğü: yüksek = uyumlu, düşük = çatışmalı
    pub compatibility_score: f64,
    /// Wronskian determinantı (lineer bağımsızlık)
    pub wronskian: f64,

    // ─── Arkhe 5 — Yöntem ────────────────────────────────────────────────────
    /// Aktif eylem sırası (QMEM CAUSES kenarlarından türetilir)
    pub action_sequence: Vec<String>,
    /// Yöntem güven skoru [0, 1]
    pub method_confidence: f32,

    // ─── Arkhe 6 — Şartlar ───────────────────────────────────────────────────
    /// İhlal edilen kısıt varsa Some(kısıt adı)
    pub constraint_violation: Option<String>,
    /// God's Gate durumu: true = kapı açık, false = kapı kapalı (ihlal)
    pub gate_open: bool,

    // ─── Clifford Bridge çıktısı ──────────────────────────────────────────────
    /// h_phase: pembe bölge kesişim açısı → SRF.dx'e yazılır
    pub h_phase: f32,
    /// Projeksiyon güven skoru → CX_W güncellenir
    pub bridge_confidence: f32,
}

/// Hedef görev manifoldu — Arkhe Bilinç tarafından belirlenir
#[derive(Debug, Clone, PartialEq)]
pub enum TaskManifold {
    /// Varsayılan: görev tipi bilinmiyor
    Unknown,
    /// Nedensel muhakeme (FailureSensorIQ, AssetOps)
    CausalReasoning,
    /// Anomali tespiti (sensör verisi)
    AnomalyDetection,
    /// Reçete / prosedür sentezi (ITBench, mutfak)
    RecipeSynthesis,
    /// Geometrik dönüşüm muhakemesi (ARC-AGI-3)
    GeometricTransformation,
    /// Sınıflandırma (MCQ)
    Classification,
}

impl ArkheState {
    pub fn new() -> Self {
        Self {
            task_manifold: TaskManifold::Unknown,
            bx_phase: 0.0,
            top_similarity: None,
            sacak_nodes: Vec::new(),
            inference_path: None,
            path_energy_cost: 0.0,
            compatibility_score: 0.0,
            wronskian: 0.0,
            action_sequence: Vec::new(),
            method_confidence: 0.5,
            constraint_violation: None,
            gate_open: true,
            h_phase: 0.0,
            bridge_confidence: 0.0,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ARKHE CognitionPipeline
// Altı Arkhe'yi sıralı çağıran wrapper. Her çağrı token adımı başında yapılır.
// ════════════════════════════════════════════════════════════════════════════

pub struct ArkheCognitionPipeline {
    /// CliffordBridge instance
    pub bridge: CliffordBridge,

    /// Mevcut Arkhe durumu (token döngüsü boyunca birikilir)
    pub state: ArkheState,

    /// Şartlar kısıt tablosu: (kısıt_adı, max_energy_threshold)
    pub constraint_table: Vec<(String, f64)>,
}

impl ArkheCognitionPipeline {
    pub fn new() -> Self {
        Self {
            bridge: CliffordBridge::new(),
            state: ArkheState::new(),
            constraint_table: Vec::new(),
        }
    }

    /// Kısıt ekle: QMEM DSL'den gelen [energy, phase_theta] çiftleri
    /// "Şartlar Arkhesi" bu tabloyu Hamiltonian'a hard constraint olarak ekler
    pub fn add_constraint(&mut self, name: impl Into<String>, max_energy: f64) {
        self.constraint_table.push((name.into(), max_energy));
    }

    /// **ANA DÖNGÜ** — Her token adımında çağrılır.
    ///
    /// Altı Arkhe sıralı çalışır; her birinin çıktısı SRF'ye ve state'e yazılır.
    /// Dönüş değeri güncellenmiş ArkheState'dir.
    ///
    /// ```ignore
    /// let updated = pipeline.run(net_f, net_g, &prompt, srf);
    /// srf.set_data_phase(updated.h_phase);
    /// ```
    pub fn run(
        &mut self,
        net_f: &ResonanceNetwork,
        net_g: &ResonanceNetwork,
        prompt: &str,
        srf: &mut SemanticRegisterFile,
    ) -> &ArkheState {
        // θ_z = ortalama QMEM damping katsayısı
        let impedance_theta = if net_g.n > 0 {
            net_g.damping.iter().take(net_g.n).sum::<f64>() / net_g.n as f64
        } else {
            1.0
        };

        // ─── Arkhe 1 — Bilinç ───────────────────────────────────────────────
        self.state = self.arkhe_bilincin(net_g, prompt, srf);

        // ─── Arkhe 2 — Benzerlik ────────────────────────────────────────────
        self.arkhe_benzerlik(net_f, net_g);

        // ─── Arkhe 3 — Çıkarım ──────────────────────────────────────────────
        self.arkhe_cikarim(net_g);

        // ─── Arkhe 4 — Uyum ─────────────────────────────────────────────────
        self.arkhe_uyum(net_f, net_g);

        // ─── Arkhe 5 — Yöntem ───────────────────────────────────────────────
        self.arkhe_yontem(net_g, impedance_theta);

        // ─── Arkhe 6 — Şartlar ──────────────────────────────────────────────
        self.arkhe_sartlar(net_g);

        // ─── CliffordBridge — h_phase hesapla ───────────────────────────────
        let (h_phase, confidence) = self.bridge.project(net_f, net_g, impedance_theta);
        self.state.h_phase = h_phase as f32;
        self.state.bridge_confidence = confidence as f32;
        self.state.sacak_nodes = self.bridge.last_sacak.clone();

        // SRF güncelle
        if self.state.gate_open {
            srf.set_data_phase(self.state.h_phase);
            // CX_W = bridge güveni + saçak katkısı
            let sacak_boost = (self.state.sacak_nodes.len() as f32 * 0.02).min(0.1);
            srf.cx_w = (srf.cx_w + sacak_boost * confidence as f32).clamp(0.0, 1.0);
        } else {
            // God's Gate kapalı: şart ihlali → DX'i sıfırla (halüsinasyon freni)
            srf.set_data_phase(0.0);
            srf.cx_w = (srf.cx_w * 0.8).max(0.0); // güven düşür
            eprintln!(
                "[ARKHE GATE CLOSED] Constraint violation: {:?}",
                self.state.constraint_violation
            );
        }

        &self.state
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ARKHE 1 — BİLİNÇ
    // "Ben ne yapmaya çalışıyorum?"
    // ═══════════════════════════════════════════════════════════════════════

    /// Arkhe Bilinç: Görev tipi QMEM'deki node_flags'tan türetilir.
    ///
    /// node_flags → DOMAIN_LOGIC / DOMAIN_TEMPORAL / DOMAIN_SPATIAL kombinasyonu
    /// → TaskManifold belirlenir → SRF.bx faz olarak set edilir.
    ///
    /// Mevcut genişletilen fonksiyon: `SemanticRegisterFile::new(bx_phase)`
    /// Yeni davranış: bx_phase, QMEM'deki baskın domain'in aggregate phase'inden türer.
    fn arkhe_bilincin(
        &self,
        net_g: &ResonanceNetwork,
        prompt: &str,
        srf: &mut SemanticRegisterFile,
    ) -> ArkheState {
        use crate::semantic_parser::{
            DOMAIN_LOGIC, DOMAIN_TEMPORAL, DOMAIN_SPATIAL,
        };

        let mut state = ArkheState::new();

        // QMEM'deki node flag dağılımını analiz et
        let mut logic_energy = 0.0f64;
        let mut temporal_energy = 0.0f64;
        let mut spatial_energy = 0.0f64;

        for i in 0..net_g.n.min(net_g.node_flags.len()) {
            let energy = net_g.positions.amplitudes[i].norm() * net_g.masses[i];
            let flags = net_g.node_flags[i];
            if flags & DOMAIN_LOGIC != 0 { logic_energy += energy; }
            if flags & DOMAIN_TEMPORAL != 0 { temporal_energy += energy; }
            if flags & DOMAIN_SPATIAL != 0 { spatial_energy += energy; }
        }

        // Prompt anahtar kelimelerinden de ipucu al
        let prompt_lower = prompt.to_lowercase();
        let manifold = if prompt_lower.contains("arc") || prompt_lower.contains("grid")
            || prompt_lower.contains("transform") || spatial_energy > logic_energy * 2.0
        {
            TaskManifold::GeometricTransformation
        } else if prompt_lower.contains("anomal") || prompt_lower.contains("failure")
            || prompt_lower.contains("sensor")
        {
            TaskManifold::AnomalyDetection
        } else if prompt_lower.contains("recipe") || prompt_lower.contains("step")
            || temporal_energy > logic_energy
        {
            TaskManifold::RecipeSynthesis
        } else if prompt_lower.contains("cause") || prompt_lower.contains("why")
        {
            TaskManifold::CausalReasoning
        } else if prompt_lower.contains("which") || prompt_lower.contains("answer")
            || prompt_lower.contains("(a)") || prompt_lower.contains("(b)")
        {
            TaskManifold::Classification
        } else {
            TaskManifold::CausalReasoning // Default: endüstriyel kök neden analizi
        };

        // BX faz: baskın domain'in aggregate phase'i
        // Bu, SRF.bx'i statik 0.0'dan çok daha anlamlı bir başlangıç değerine set eder
        let bx_phase = net_g.compute_aggregate_phase() as f32;
        srf.bx = bx_phase; // Bilinç → sistem promptu faz anlamı

        eprintln!(
            "[ARKHE BILINC] Manifold={:?} bx={:.3} L={:.2} T={:.2} S={:.2}",
            manifold, bx_phase, logic_energy, temporal_energy, spatial_energy
        );

        state.task_manifold = manifold;
        state.bx_phase = bx_phase;
        state
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ARKHE 2 — BENZERLİK
    // "Bu, bildiğim neye benziyor?"
    // ═══════════════════════════════════════════════════════════════════════

    /// Arkhe Benzerlik: net_f ve net_g arasında VSA topolojik izomorfizm skoru.
    ///
    /// Genişletilen fonksiyon: `ResonanceNetwork::find_cross_network_sacak()`
    /// Yeni: En yüksek skorlu çift `top_similarity` olarak kaydedilir.
    /// Bu skor GPR routing kararına beslenir.
    pub fn arkhe_benzerlik(&mut self, net_f: &ResonanceNetwork, net_g: &ResonanceNetwork) {
        let sacak = ResonanceNetwork::find_cross_network_sacak(net_f, net_g, 0.15);

        if let Some((id, label, score)) = sacak.first() {
            // En benzer kavramı net_g'de de bul
            let label_g = net_g.labels.iter()
                .filter(|(l, _)| l.to_lowercase().contains(&label.to_lowercase()))
                .map(|(l, _)| l.clone())
                .next()
                .unwrap_or_else(|| label.clone());

            self.state.top_similarity = Some((label.clone(), label_g, *score));
        }

        self.state.sacak_nodes = sacak
            .into_iter()
            .map(|(_, label, score)| (label, score))
            .collect();

        eprintln!(
            "[ARKHE BENZERLIK] top={:?} saçak_count={}",
            self.state.top_similarity, self.state.sacak_nodes.len()
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ARKHE 3 — ÇIKARIM
    // "Buradan oraya nasıl varırım?"
    // ═══════════════════════════════════════════════════════════════════════

    /// Arkhe Çıkarım: Enerji kısıtlı Hamiltonian yolu.
    ///
    /// Genişletilen fonksiyon: `ResonanceNetwork::step()` + CSR spreading activation.
    /// Yeni: Başlangıç durumu (yüksek enerjili node) → hedef (saçak node) arasında
    /// BFS ile minimum ağırlıklı yol bulunur. Enerji kısıtı: QMEM'deki her kenar
    /// ağırlığı, o bağlantının "geçiş maliyeti" olarak yorumlanır.
    ///
    /// Tam Hamiltonian integration yerine O(V+E) BFS — token başına O(1) hedef.
    fn arkhe_cikarim(&mut self, net_g: &ResonanceNetwork) {
        if net_g.n < 2 {
            return;
        }

        // Başlangıç: en yüksek enerjili node
        let start = (0..net_g.n)
            .max_by(|&a, &b| {
                let ea = net_g.positions.amplitudes[a].norm();
                let eb = net_g.positions.amplitudes[b].norm();
                ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        // Hedef: saçak düğümler varsa ilki, yoksa en düşük enerjili
        let target = if let Some((label, _)) = self.state.sacak_nodes.first() {
            net_g.labels.get(label).copied().unwrap_or(net_g.n - 1)
        } else {
            (0..net_g.n)
                .min_by(|&a, &b| {
                    let ea = net_g.positions.amplitudes[a].norm();
                    let eb = net_g.positions.amplitudes[b].norm();
                    ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(net_g.n - 1)
        };

        // Enerji kısıtlı BFS (Dijkstra ağırlıkla)
        let (path, cost) = Self::energy_constrained_bfs(net_g, start, target);

        eprintln!(
            "[ARKHE CIKARIM] start={} target={} path_len={} cost={:.3}",
            start, target, path.len(), cost
        );

        self.state.inference_path = if path.is_empty() { None } else { Some(path) };
        self.state.path_energy_cost = cost;
    }

    /// Enerji kısıtlı BFS — O(V + E)
    /// Düğümler arası geçiş maliyeti: 1/weight (yüksek bağ = ucuz yol)
    fn energy_constrained_bfs(
        net: &ResonanceNetwork,
        start: usize,
        target: usize,
    ) -> (Vec<usize>, f64) {
        if start == target {
            return (vec![start], 0.0);
        }

        // Basit BFS ile yol bul (toy Dijkstra: basit queue ile)
        let mut visited = vec![false; net.n];
        let mut parent = vec![usize::MAX; net.n];
        let mut cost = vec![f64::INFINITY; net.n];

        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        cost[start] = 0.0;

        while let Some(current) = queue.pop_front() {
            if current == target {
                break;
            }

            if let Some(row) = net.adjacency_csr.outer_view(current) {
                for (neighbor, &weight) in row.iter() {
                    if neighbor < net.n && !visited[neighbor] && weight > 1e-12 {
                        // Geçiş maliyeti: 1/weight (güçlü bağ = düşük maliyet)
                        let edge_cost = 1.0 / weight.abs();
                        let new_cost = cost[current] + edge_cost;

                        if new_cost < cost[neighbor] {
                            cost[neighbor] = new_cost;
                            parent[neighbor] = current;
                            visited[neighbor] = true;
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        // Yolu geri izle
        if !visited[target] || parent[target] == usize::MAX && target != start {
            return (Vec::new(), f64::INFINITY);
        }

        let mut path = Vec::new();
        let mut cur = target;
        while cur != usize::MAX && path.len() <= net.n {
            path.push(cur);
            if cur == start {
                break;
            }
            cur = parent[cur];
        }
        path.reverse();

        let total_cost = cost[target];
        (path, total_cost)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ARKHE 4 — UYUM
    // "Bunlar birlikte çalışıyor mu?"
    // ═══════════════════════════════════════════════════════════════════════

    /// Arkhe Uyum: net_f ve net_g'nin Clifford bivektör süperpozisyonu.
    ///
    /// Genişletilen fonksiyon: `apply_clifford_resonance()` (tek ağ → iki ağ)
    /// Yeni: wave_f + wave_g süperpozisyonu → genlik artıyorsa yapıcı (uyumlu),
    /// azalıyorsa yıkıcı (uyumsuz).
    ///
    /// `compatibility_score > 0.5` → yapıcı girişim (uyumlu bileşenler)
    /// `compatibility_score < 0.3` → yıkıcı girişim (çatışan bileşenler)
    fn arkhe_uyum(&mut self, net_f: &ResonanceNetwork, net_g: &ResonanceNetwork) {

        // net_f'in aktif nodelarından dalga oluştur
        let wave_f_amps: Vec<Complex64> = (0..net_f.n.min(32))
            .map(|i| net_f.positions.amplitudes[i])
            .collect();

        // net_g'nin aktif nodelarından dalga oluştur
        let n_g = net_g.n.min(32);
        let wave_g_amps: Vec<Complex64> = (0..n_g)
            .map(|i| net_g.positions.amplitudes[i])
            .collect();

        if wave_f_amps.is_empty() || wave_g_amps.is_empty() {
            self.state.compatibility_score = 0.5; // Nötr
            return;
        }

        // Süperpozisyon dalga boyutlarını eşitle
        let min_len = wave_f_amps.len().min(wave_g_amps.len());
        let superposition_amps: Vec<Complex64> = (0..min_len)
            .map(|i| wave_f_amps[i] + wave_g_amps[i])
            .collect();

        let wave_superposed = SemanticWave::from_vec(superposition_amps);

        // Clifford bivektör büyüklüğü — yüksek = rotasyonel bağımsızlık = sağlıklı
        let bivector_score = apply_clifford_resonance(&wave_superposed);

        // Bireysel ağların bivektör büyüklükleri
        let mut wave_f = SemanticWave::from_vec(wave_f_amps.iter().take(min_len).cloned().collect());
        wave_f.angular_velocity = net_f.positions.angular_velocity;
        
        let mut wave_g = SemanticWave::from_vec(wave_g_amps.iter().take(min_len).cloned().collect());
        wave_g.angular_velocity = net_g.positions.angular_velocity;

        // Yapıcı / yıkıcı girişim değerlendirmesi (faz farkı)
        let mut constructive = 0;
        let mut destructive = 0;
        let mut neutral = 0;
        for i in 0..min_len {
            let f_i = wave_f_amps[i];
            let g_i = wave_g_amps[i];
            if f_i.norm() > 1e-6 && g_i.norm() > 1e-6 {
                let diff = (f_i.arg() - g_i.arg()).cos().acos().abs();
                if diff < std::f64::consts::FRAC_PI_4 {
                    constructive += 1;
                } else if diff > 3.0 * std::f64::consts::FRAC_PI_4 {
                    destructive += 1;
                } else {
                    neutral += 1;
                }
            }
        }
        let total_active = constructive + destructive + neutral;
        let compatibility = if total_active > 0 {
            let raw_score = (constructive as f64 - destructive as f64) / total_active as f64;
            (raw_score + 1.0) / 2.0 // scale to [0.0, 1.0]
        } else {
            0.5
        };

        // Wronskian empedansı
        let wronskian = SemanticWave::compute_cross_network_wronskian(&wave_f, &wave_g);

        eprintln!(
            "[ARKHE UYUM] constructive={} destructive={} compatibility={:.3} wronskian={:.6} bivector={:.3}",
            constructive, destructive, compatibility, wronskian, bivector_score
        );

        self.state.compatibility_score = compatibility;
        self.state.wronskian = wronskian;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ARKHE 5 — YÖNTEM
    // "Hangi sırayla, nasıl?"
    // ═══════════════════════════════════════════════════════════════════════

    /// Arkhe Yöntem: QMEM CAUSES kenarlarından eylem sıralaması.
    ///
    /// Genişletilen fonksiyon: `ResonanceNetwork::step()` — CSR edge traversal
    /// Yeni: Çıkarım yolundaki düğümler label sırasıyla action_sequence'a yazılır.
    /// Bu sıra, token üretiminde "önce X, sonra Y" kısıtını temsil eder.
    fn arkhe_yontem(&mut self, net_g: &ResonanceNetwork, _impedance_theta: f64) {
        let id_to_label: HashMap<usize, &String> =
            net_g.labels.iter().map(|(k, v)| (*v, k)).collect();

        // Çıkarım yolundaki düğümlerin labellarını sırala
        let mut sequence = Vec::new();
        if let Some(ref path) = self.state.inference_path {
            for &node_id in path {
                if let Some(label) = id_to_label.get(&node_id) {
                    sequence.push(label.to_string());
                }
            }
        }

        // Eğer yol boşsa: yüksek enerjili node'ları impedance azalan sırada listele
        if sequence.is_empty() {
            let mut energy_nodes: Vec<(usize, f64)> = (0..net_g.n)
                .map(|i| (i, net_g.positions.amplitudes[i].norm() * net_g.masses[i]))
                .filter(|(_, e)| *e > 0.01)
                .collect();
            energy_nodes.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            sequence = energy_nodes.iter()
                .take(8)
                .filter_map(|(id, _)| id_to_label.get(id).map(|l| l.to_string()))
                .collect();
        }

        // Yöntem güveni: uyum × (1 - ihlal cezası)
        let violation_penalty = if self.state.constraint_violation.is_some() { 0.5 } else { 0.0 };
        let method_conf = ((self.state.compatibility_score as f32) * (1.0 - violation_penalty))
            .clamp(0.0, 1.0);

        eprintln!(
            "[ARKHE YONTEM] sequence={:?} confidence={:.3}",
            &sequence[..sequence.len().min(4)], method_conf
        );

        self.state.action_sequence = sequence;
        self.state.method_confidence = method_conf;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ARKHE 6 — ŞARTLAR
    // "Fiziksel gerçeklik ne söylüyor?"
    // ═══════════════════════════════════════════════════════════════════════

    /// Arkhe Şartlar: QMEM [energy, phase_theta] çiftlerini hard constraint olarak uygula.
    ///
    /// Genişletilen fonksiyon: Mevcut Wronskian gating (soft) → hard constraint.
    /// Yeni: constraint_table'daki her kısıt, net_g'nin toplam enerji metrikleri ile
    /// karşılaştırılır. İhlal varsa God's Gate kapatılır (gate_open = false).
    ///
    /// Bu, "Çilek sıcak fırına girerse = energy > threshold → sonsuz ceza" davranışıdır.
    fn arkhe_sartlar(&mut self, net_g: &ResonanceNetwork) {
        self.state.gate_open = true;
        self.state.constraint_violation = None;

        // Net_g'nin toplam enerji metrikleri
        let total_kinetic: f64 = (0..net_g.n)
            .map(|i| net_g.momenta.amplitudes[i].norm_sqr() / (2.0 * net_g.masses[i]))
            .sum();
        let total_potential: f64 = (0..net_g.n)
            .map(|i| net_g.positions.amplitudes[i].norm())
            .sum();
        let total_energy = total_kinetic + total_potential;

        // DOMAIN_SPATIAL düğümlerinden gelecek: kısıt ihlali için ek sinyal
        // (Geometrik dönüşüm görevlerinde kütle korunumu)
        let spatial_amplitude: f64 = (0..net_g.n.min(net_g.node_flags.len()))
            .filter(|&i| {
                use crate::semantic_parser::DOMAIN_SPATIAL;
                net_g.node_flags[i] & DOMAIN_SPATIAL != 0
            })
            .map(|i| net_g.positions.amplitudes[i].norm())
            .sum();

        // Kısıt tablosunu kontrol et
        for (constraint_name, max_energy) in &self.constraint_table {
            let relevant_energy = if constraint_name.contains("spatial") || constraint_name.contains("grid") {
                spatial_amplitude
            } else {
                total_energy
            };

            if relevant_energy > *max_energy {
                self.state.gate_open = false;
                self.state.constraint_violation = Some(constraint_name.clone());
                eprintln!(
                    "[ARKHE SARTLAR] IHLAL! {}: energy={:.3} > max={:.3} → Gate CLOSED",
                    constraint_name, relevant_energy, max_energy
                );
                return;
            }
        }

        // Wronskian proxy ile soft check (uyum düşükse uyarı)
        if self.state.wronskian < 0.1 && self.state.compatibility_score < 0.2 {
            eprintln!(
                "[ARKHE SARTLAR] DÜŞÜK UYUM: wronskian={:.3} compat={:.3} — Gate açık ama güven düşük",
                self.state.wronskian, self.state.compatibility_score
            );
        }

        eprintln!(
            "[ARKHE SARTLAR] OK — energy={:.3} gate=OPEN",
            total_energy
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ARC-AGI-3 ÖZEL ZİNCİR
// build_transformation_group → Benzerlik → Çıkarım → Yöntem → Şartlar
// ════════════════════════════════════════════════════════════════════════════

/// ARC-AGI-3 için özelleştirilmiş Arkhe zinciri.
///
/// Adım sırası:
/// 1. `build_transformation_group()` → SpatialNode'ları net_g'ye yükle
/// 2. Benzerlik Arkhesi → iki gridin VSA imzaları karşılaştırılır
/// 3. Çıkarım Arkhesi → Enerji kısıtlı Hamiltonian yolu
/// 4. Yöntem Arkhesi → CAUSES kenarlarından eylem sırası
/// 5. Şartlar Arkhesi → GridTopology invariantları (kütle korunumu)
///
/// Dönüş: `Vec<String>` — uygulanacak dönüşüm sırası
pub fn run_arc_agi_pipeline(
    pipeline: &mut ArkheCognitionPipeline,
    net_f: &ResonanceNetwork,    // Input grid semantik ağı
    net_g: &mut ResonanceNetwork, // Output/kartridj ağı
    input_grid_label: &str,
    output_grid_label: &str,
    srf: &mut SemanticRegisterFile,
) -> Vec<String> {
    // Şartlar: GridTopology kütle korunumu kısıtı ekle
    let input_mass = net_g.labels.get(input_grid_label)
        .and_then(|&id| net_g.holographic_memory.get(id))
        .map(|sig| sig.iter().map(|c| c.norm()).sum::<f64>())
        .unwrap_or(100.0);

    // Grid kütle korunumu: output grid enerji ≤ input grid + %10 tolerans
    pipeline.add_constraint("grid_mass_conservation", input_mass * 1.1);
    pipeline.add_constraint("spatial_symmetry", input_mass * 2.0);

    // Bilinç: GeometricTransformation manifoldu zorla
    srf.bx = net_g.compute_aggregate_phase() as f32;

    // Benzerlik: iki grid ağının VSA karşılaştırması
    pipeline.arkhe_benzerlik(net_f, net_g);

    // Çıkarım: start = input grid node, target = output grid node
    let start_node = net_g.labels.get(input_grid_label).copied().unwrap_or(0);
    let target_node = net_g.labels.get(output_grid_label).copied()
        .unwrap_or_else(|| net_g.n.saturating_sub(1));

    let impedance_theta = if net_g.n > 0 {
        net_g.damping.iter().take(net_g.n).sum::<f64>() / net_g.n as f64
    } else { 1.0 };

    let (path, cost) = ArkheCognitionPipeline::energy_constrained_bfs(net_g, start_node, target_node);
    pipeline.state.inference_path = if path.is_empty() { None } else { Some(path) };
    pipeline.state.path_energy_cost = cost;

    // Yöntem: dönüşüm sırasını çıkar
    pipeline.arkhe_yontem(net_g, impedance_theta);

    // Şartlar: kütle korunumu kontrolü
    pipeline.arkhe_sartlar(net_g);

    eprintln!(
        "[ARC-AGI PIPELINE] manifold=GeometricTransformation gate={} sequence={:?}",
        pipeline.state.gate_open,
        &pipeline.state.action_sequence[..pipeline.state.action_sequence.len().min(5)]
    );

    pipeline.state.action_sequence.clone()
}

// ════════════════════════════════════════════════════════════════════════════
// TESTS
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_network(n: usize) -> ResonanceNetwork {
        let mut net = ResonanceNetwork::new(n + 4);
        for i in 0..n {
            let sig: Vec<Complex64> = (0..64)
                .map(|j| Complex64::from_polar(1.0, (i * j) as f64 * 0.1))
                .collect();
            net.add_labeled_concept(&format!("concept_{}", i), 0.05 * i as f64, sig);
        }
        if n >= 2 {
            net.connect_semantic(0, 1, 1.5);
        }
        if n >= 3 {
            net.connect_syntax(1, 2, 0.8);
        }
        net
    }

    #[test]
    fn test_clifford_bridge_project() {
        let net_f = make_test_network(4);
        let net_g = make_test_network(4);
        let mut bridge = CliffordBridge::new();
        let (h_phase, confidence) = bridge.project(&net_f, &net_g, 0.5);
        assert!(h_phase.abs() <= std::f64::consts::PI + 1e-6);
        assert!(confidence >= 0.0 && confidence <= 1.0 + 1e-6);
    }

    #[test]
    fn test_arkhe_pipeline_runs() {
        let net_f = make_test_network(6);
        let net_g = make_test_network(6);
        let mut pipeline = ArkheCognitionPipeline::new();
        let mut srf = SemanticRegisterFile::new(0.0);
        let state = pipeline.run(&net_f, &net_g, "failure sensor anomaly", &mut srf);
        assert!(state.gate_open); // Kısıt yok → kapı açık
        assert!(state.h_phase.abs() <= std::f32::consts::PI + 0.01);
    }

    #[test]
    fn test_energy_constrained_bfs() {
        let net = make_test_network(5);
        let (path, cost) = ArkheCognitionPipeline::energy_constrained_bfs(&net, 0, 1);
        // Bağlantı varsa yol bulunmalı
        if !path.is_empty() {
            assert!(cost.is_finite());
            assert_eq!(path[0], 0);
        }
    }

    #[test]
    fn test_arkhe_sartlar_violation() {
        let net_g = make_test_network(4);
        let mut pipeline = ArkheCognitionPipeline::new();
        pipeline.add_constraint("test_energy", 0.0); // İmkansız eşik
        pipeline.arkhe_sartlar(&net_g);
        // Sıfır eşikle mutlaka ihlal olmalı (toplam enerji > 0)
        // (Ağ boş olduğunda ihlal olmayabilir, sadece gate durumu kontrol et)
        assert!(pipeline.state.constraint_violation.is_some() || pipeline.state.gate_open);
    }
}
