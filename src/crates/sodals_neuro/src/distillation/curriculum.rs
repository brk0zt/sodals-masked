// src/distillation/curriculum.rs
//
// QMEM Synthetic Curriculum Generator for Selective Reasoning Distillation.
//
// Prevents pre-training contamination and "distilled from GPT-4" accusations
// by generating training samples directly from QMEM axioms and goals.

use crate::compiler::qmem_compiler::{ContinuousSemanticTopology, GoalNode, CompositionRule, AxiomNode};

/// Category of reasoning pattern to distill into model weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningCategory {
    /// Analogical and composition reasoning (e.g., Arkhe stage transitions, X:Y :: A:B)
    Analogical,
    /// Goal-precondition physical prerequisite satisfaction (e.g., STRIPS/PDDL goal compliance)
    GoalPrecondition,
    /// Epsilon-delta style reasoning: limits, convergence, bounding arguments
    /// Phase signature: θ ≈ 0.0 (analysis-type linear reasoning)
    EpsilonDeltaReasoning,
    /// Topological invariance reasoning: properties preserved under homeomorphism
    /// Phase signature: θ ≈ π/2 (topology-type transformational reasoning)
    TopologicalInvariance,
    /// Abstract algebraic reasoning: groups, rings, fields, ideals, Galois correspondence
    /// Phase signature: θ ≈ 0.0 to π/2 (algebraic structure & isomorphism reasoning)
    AbstractAlgebraicReasoning,
    /// Lie algebraic reasoning: root systems, Dynkin diagrams, classical & exceptional dimension formulas
    /// Phase signature: θ ≈ 2π/3 (differential symmetry & Cartan/Dynkin geometry)
    LieAlgebraicReasoning,
}

/// A synthetic reasoning sample generated from QMEM topology.
#[derive(Debug, Clone)]
pub struct SyntheticReasoningSample {
    pub id: String,
    pub category: ReasoningCategory,
    pub prompt: String,
    pub target_reasoning_chain: String,
    pub correct_action: String,
    pub distractor_action: String,
    pub precondition_satisfied_by_correct: bool,
    pub precondition_satisfied_by_distractor: bool,
    /// Bayes confidence score from axiom composition depth (Gelistirme_Plani.md: "türetim-güveni")
    pub bayes_confidence: f64,
}

pub struct CurriculumGenerator;

impl CurriculumGenerator {
    /// Generate a dataset of synthetic reasoning samples from a compiled QMEM topology.
    pub fn generate_from_topology(cst: &ContinuousSemanticTopology) -> Vec<SyntheticReasoningSample> {
        let mut samples = Vec::new();

        // 1. Generate Goal-Precondition reasoning samples (Car-wash category)
        for (idx, goal) in cst.goals.iter().enumerate() {
            let sample = Self::build_goal_precondition_sample(goal, idx);
            samples.push(sample);
        }

        // 2. Generate Analogical & Composition reasoning samples
        for (idx, rule) in cst.rules.iter().enumerate() {
            let sample = Self::build_composition_sample(rule, idx);
            samples.push(sample);
        }

        // 3. Generate Mathematics domain samples if topology contains math axioms
        let has_math_axioms = cst.axioms.iter().any(|a| 
            a.name.contains("cauchy") || a.name.contains("tikiz") || a.name.contains("baglanti")
        );
        if has_math_axioms {
            samples.extend(Self::generate_mathematics_curriculum());
        }

        // 3.5 Generate Abstract Algebra domain samples if topology contains algebra axioms
        let has_algebra_axioms = cst.axioms.iter().any(|a|
            a.name.contains("lagrange") || a.name.contains("sylow") || a.name.contains("cartan")
            || a.name.contains("simplektik") || a.name.contains("ideal") || a.name.contains("galois")
        );
        if has_algebra_axioms {
            samples.extend(Self::generate_abstract_algebra_curriculum());
        }

        // 4. Fallback: if topology has no rules/goals, generate synthetic benchmark templates
        if samples.is_empty() {
            samples.extend(Self::generate_default_curriculum());
        }

        samples
    }

    /// Build a goal-precondition sample:
    /// Teaches the model that surface attraction/convenience (distractor) MUST NOT override
    /// physical prerequisite satisfaction (correct action).
    fn build_goal_precondition_sample(goal: &GoalNode, idx: usize) -> SyntheticReasoningSample {
        let prompt = format!(
            "Hedef: \"{}\". Bu hedefe ulaşmak için önkoşul: \"{}\". \
             Hangi eylem seçilmelidir?",
            goal.goal_state, goal.precondition
        );

        let correct_action = format!("arabayla_git(hedef=\"{}\")", goal.precondition);
        let distractor_action = "yuruyerek_git(kolaylik=yuksek)".to_string();

        let chain = format!(
            "1. Hedef analiz edildi: {}\n\
             2. Fiziksel önkoşul doğrulandı: {}\n\
             3. Eylem A ({}) önkoşulu SAĞLIYOR.\n\
             4. Eylem B ({}) yüzeysel sezgi sunuyor ama önkoşulu SAĞLAMIYOR.\n\
             Karar: {}",
            goal.goal_state, goal.precondition, correct_action, distractor_action, correct_action
        );

        SyntheticReasoningSample {
            id: format!("goal_sample_{}", idx),
            category: ReasoningCategory::GoalPrecondition,
            prompt,
            target_reasoning_chain: chain,
            correct_action,
            distractor_action,
            precondition_satisfied_by_correct: true,
            precondition_satisfied_by_distractor: false,
            bayes_confidence: 0.85,
        }
    }

    /// Build an analogical / composition sample:
    /// Teaches the model to compose two concepts into a conclusion using Arkhe principles.
    fn build_composition_sample(rule: &CompositionRule, idx: usize) -> SyntheticReasoningSample {
        let prompt = format!(
            "Kavram A: \"{}\", Kavram B: \"{}\". Bu iki kavramın bileşimi \
             Arkhe ilkelerine göre neyi türetir?",
            rule.concept_a, rule.concept_b
        );

        let correct_action = rule.target_concept.clone();
        let distractor_action = format!("{}_tersi", rule.target_concept);

        let chain = format!(
            "1. Arkhe Benzerlik Aşaması: \"{}\" ve \"{}\" eşleştirildi.\n\
             2. Arkhe Çıkarım Aşaması: Enerji={:.2}, Faz θ={:.2}.\n\
             3. Arkhe Uyum Aşaması: \"{}\" türetildi.",
            rule.concept_a, rule.concept_b, rule.energy, rule.phase_theta, correct_action
        );

        SyntheticReasoningSample {
            id: format!("composition_sample_{}", idx),
            category: ReasoningCategory::Analogical,
            prompt,
            target_reasoning_chain: chain,
            correct_action,
            distractor_action,
            precondition_satisfied_by_correct: true,
            precondition_satisfied_by_distractor: false,
            bayes_confidence: 0.90,
        }
    }

    /// Generate default curriculum containing canonical examples of both categories
    pub fn generate_default_curriculum() -> Vec<SyntheticReasoningSample> {
        vec![
            // Car wash canonical example
            SyntheticReasoningSample {
                id: "canonical_car_wash".into(),
                category: ReasoningCategory::GoalPrecondition,
                prompt: "Arabayı yıkatmak istiyorsunuz. Yıkama yeri 5 km uzakta. Yürüyerek gitmek daha spor ve kolay. Ne yapmalısınız?".into(),
                target_reasoning_chain: "1. Hedef: araba_yikanmis.\n2. Önkoşul: araba_konumu == yikama_yeri.\n3. Yürüyerek gitmek arabayı götürmez (önkoşul ihlali).\n4. Arabayla gitmek önkoşulu sağlar.\nKarar: arabayla_git.".into(),
                correct_action: "arabayla_git".into(),
                distractor_action: "yuruyerek_git".into(),
                precondition_satisfied_by_correct: true,
                precondition_satisfied_by_distractor: false,
                bayes_confidence: 0.85,
            },
            // Pasteurization thermal prerequisite canonical example
            SyntheticReasoningSample {
                id: "canonical_pasteurization".into(),
                category: ReasoningCategory::GoalPrecondition,
                prompt: "Süt sterilizasyonu tamamlanmalıdır (HACCP). Sıcaklık 72C'nin altında ama akış hızı yüksek. Ne yapılmalı?".into(),
                target_reasoning_chain: "1. Hedef: haccp_sterilizasyon.\n2. Önkoşul: sicaklik >= 72C.\n3. Akış hızını artırmak önkoşulu sağlamaz.\n4. Sıcaklığı artırmak önkoşulu sağlar.\nKarar: sicaklik_artir.".into(),
                correct_action: "sicaklik_artir".into(),
                distractor_action: "akis_hizi_artir".into(),
                precondition_satisfied_by_correct: true,
                precondition_satisfied_by_distractor: false,
                bayes_confidence: 0.85,
            },
            // Physics analogy composition canonical example
            SyntheticReasoningSample {
                id: "canonical_physics_analogy".into(),
                category: ReasoningCategory::Analogical,
                prompt: "Elektrik devresinde Akım ~ Gerilim ilişkisi veriliyor. Isı iletiminde Akı ~ Sıcaklık Farkı analojisi nasıl kurulur?".into(),
                target_reasoning_chain: "1. Gerilim <-> Sıcaklık Farkı (Potansiyel)\n2. Akım <-> Isı Akısı (Debi)\n3. Direnç <-> Termal Direnç\nAnaloji: Ohm Yasası Q = deltaT / R_th olarak haritalanır.".into(),
                correct_action: "thermal_ohm_law".into(),
                distractor_action: "electric_current_direct".into(),
                precondition_satisfied_by_correct: true,
                precondition_satisfied_by_distractor: false,
                bayes_confidence: 0.85,
            },
        ]
    }

    /// Generate mathematics domain curriculum:
    /// Epsilon-delta (analysis) + topological invariance reasoning patterns.
    pub fn generate_mathematics_curriculum() -> Vec<SyntheticReasoningSample> {
        vec![
            // Tıkızlık korunumu
            SyntheticReasoningSample {
                id: "math_tikizlik_korunumu".into(),
                category: ReasoningCategory::TopologicalInvariance,
                prompt: "f: X → Y sürekli bir fonksiyon ve X tıkız bir uzay ise, f(X)'in tıkız olduğunu neden biliyoruz?".into(),
                target_reasoning_chain: concat!(
                    "Arkhe İzi:\n",
                    "  Benzerlik: \"tıkızlık\" kavramı, \"her açık örtünün sonlu alt-örtüsü var\" tanımıyla rezonansa giriyor\n",
                    "  Çıkarım: surekli_fonksiyon AXIOM'u + tikiz_uzay AXIOM'u → tikizlik_korunumu RULE'u tetikleniyor\n",
                    "    (Hamiltonian yol: açık örtü → geri-görüntü açık → X'in tıkızlığından sonlu alt-örtü → görüntüde sonlu alt-örtü)\n",
                    "  Şartlar: Wronskian(surekli_fonksiyon, tikiz_uzay) = bağımsız → RULE geçerli\n",
                    "Cevap: f(X) tıkızdır, çünkü f'nin sürekliliği açık kümelerin geri-görüntüsünü açık tutar, ",
                    "X'in tıkızlığı her açık örtünün sonlu alt-örtüsünü garanti eder, ",
                    "bu ikisi birleşince f(X)'in her açık örtüsü de sonlu alt-örtüye indirgenebilir."
                ).into(),
                correct_action: "tikizlik_korunumu_RULE_uygula".into(),
                distractor_action: "sadece_sinirlilik_kontrol_et".into(),
                precondition_satisfied_by_correct: true,
                precondition_satisfied_by_distractor: false,
                bayes_confidence: 0.95,
            },
            // Cauchy yakınsaklık
            SyntheticReasoningSample {
                id: "math_cauchy_yakinsaklik".into(),
                category: ReasoningCategory::EpsilonDeltaReasoning,
                prompt: "Tam metrik uzayda her Cauchy dizisi neden yakınsar?".into(),
                target_reasoning_chain: concat!(
                    "Arkhe İzi:\n",
                    "  Benzerlik: \"Cauchy dizisi\" → \"elemanlar birbirine yaklaşıyor\" rezonansı\n",
                    "  Çıkarım: cauchy_dizisi AXIOM + tam_metrik_uzay AXIOM → yakinsaklik_garantisi RULE\n",
                    "    (Adımlar: Cauchy koşulu → sınırlılık → alt dizide yığılma noktası → tüm dizi o noktaya yakınsar)\n",
                    "  Şartlar: Wronskian(cauchy_dizisi, tam_metrik_uzay) = bağımsız → RULE geçerli\n",
                    "Cevap: Tam metrik uzay tanımı gereği her Cauchy dizisi yakınsar. Cauchy koşulu dizinin ",
                    "terimlerinin birbirine keyfi olarak yaklaşmasını garanti eder, tamlık ise bu yığılma ",
                    "noktasının uzayın kendi içinde var olduğunu garanti eder."
                ).into(),
                correct_action: "yakinsaklik_garantisi_RULE_uygula".into(),
                distractor_action: "sadece_monotonluk_kontrol_et".into(),
                precondition_satisfied_by_correct: true,
                precondition_satisfied_by_distractor: false,
                bayes_confidence: 0.95,
            },
            // Heine-Borel
            SyntheticReasoningSample {
                id: "math_heine_borel".into(),
                category: ReasoningCategory::TopologicalInvariance,
                prompt: "R^n uzayında bir alt kümenin tıkız olması için kapalı ve sınırlı olması neden hem gerek hem yeter şarttır?".into(),
                target_reasoning_chain: concat!(
                    "Arkhe İzi:\n",
                    "  Benzerlik: \"kapalı ve sınırlı\" → \"Heine-Borel teoremi\" rezonansı\n",
                    "  Çıkarım: kapali_sinirli_kume + reel_sayi_dogasi → heine_borel_kurali RULE\n",
                    "    (R'nin tamlığı → kapalı aralık tıkız → Tychonoff ile küp tıkız → kapalı alt küme tıkız)\n",
                    "  Şartlar: Metrik öklidyen ve sonlu boyutlu (R^n) olmalıdır.\n",
                    "Cevap: Heine-Borel teoremi gereğince, R^n sonlu boyutlu tam bir normlu uzay olduğu için ",
                    "kapalı ve sınırlı kümeler her açık örtüden sonlu alt-örtü çıkarılmasını garanti eder."
                ).into(),
                correct_action: "heine_borel_kurali_uygula".into(),
                distractor_action: "sadece_acik_kure_kontrol_et".into(),
                precondition_satisfied_by_correct: true,
                precondition_satisfied_by_distractor: false,
                bayes_confidence: 0.92,
            },
            // Bağlantılılık korunumu
            SyntheticReasoningSample {
                id: "math_baglanti_korunumu".into(),
                category: ReasoningCategory::TopologicalInvariance,
                prompt: "Sürekli fonksiyonun bağlantılı uzaydan görüntüsü neden bağlantılıdır?".into(),
                target_reasoning_chain: concat!(
                    "Arkhe İzi:\n",
                    "  Benzerlik: \"bağlantılılık\" → \"uzay iki ayrık açık kümeye bölünemez\"\n",
                    "  Çıkarım: surekli_fonksiyon + baglantili_uzay → baglanti_korunumu RULE\n",
                    "    (Varsayım: f(X) = A∪B ayrık açık → f⁻¹(A)∪f⁻¹(B) = X ayrık açık → X bağlantısız → çelişki)\n",
                    "  Şartlar: Wronskian(surekli_fonksiyon, baglantili_uzay) = bağımsız → RULE geçerli\n",
                    "Cevap: Çelişki kanıtı ile: f(X) bağlantısız varsayılırsa, geri-görüntüler X'i böler, ",
                    "bu X'in bağlantılılığıyla çelişir."
                ).into(),
                correct_action: "baglanti_korunumu_RULE_uygula".into(),
                distractor_action: "tekil_deger_kontrol_et".into(),
                precondition_satisfied_by_correct: true,
                precondition_satisfied_by_distractor: false,
                bayes_confidence: 0.91,
            },
            // Bolzano-Weierstrass
            SyntheticReasoningSample {
                id: "math_bolzano_weierstrass".into(),
                category: ReasoningCategory::EpsilonDeltaReasoning,
                prompt: "R'de her sınırlı dizinin yakınsak bir alt dizisi var. Bunu nasıl kanıtlarız?".into(),
                target_reasoning_chain: concat!(
                    "Arkhe İzi:\n",
                    "  Benzerlik: \"sınırlı dizi\" → \"sonlu aralıkta sıkışmış\" rezonansı\n",
                    "  Çıkarım: sinirli_dizi → altdizi_secimi → bolzano_weierstrass CAUSES zinciri\n",
                    "    (İç içe aralıklar: [a,b] → yarıla → sonsuz terim içeren yarıyı seç → monoton sınırlı alt dizi)\n",
                    "  Şartlar: R'nin tamlığı + sınırlılık → monoton yakınsaklık teoremi uygulanabilir\n",
                    "Cevap: İç içe aralıklar yöntemi ile: her adımda aralığı yarılayıp sonsuz terim ",
                    "içeren yarıyı seçerek monoton sınırlı alt dizi elde edilir, bu dizi yakınsar."
                ).into(),
                correct_action: "ic_ice_aralik_yontemi".into(),
                distractor_action: "dogrudan_limit_hesapla".into(),
                precondition_satisfied_by_correct: true,
                precondition_satisfied_by_distractor: false,
                bayes_confidence: 0.90,
            },
        ]
    }

    /// Generate Domain 2: Abstract Algebra curriculum
    /// Covers Group Theory, Rings/Ideals, Galois Theory, and Lie Algebra root/dimension formulas.
    pub fn generate_abstract_algebra_curriculum() -> Vec<SyntheticReasoningSample> {
        vec![
            // 1. Critical diagnostic: sp(6) Lie algebra dimension = 21
            SyntheticReasoningSample {
                id: "algebra_sp6_dimension".into(),
                category: ReasoningCategory::LieAlgebraicReasoning,
                prompt: "sp(6) Lie cebirinin (veya sp(2n, C) için n=3) boyutu nedir ve hangi formülle hesaplanır?".into(),
                target_reasoning_chain: concat!(
                    "Arkhe İzi:\n",
                    "  Benzerlik: \"sp(6)\" → simplektik Lie cebiri C_n ailesi rezonansı\n",
                    "  Çıkarım: sp(2n) cebirinde 2n = 6 olduğundan rank n = 3'tür.\n",
                    "    Klasik C_n boyutu: dim(sp(2n)) = n(2n+1).\n",
                    "    n=3 için: dim(sp(6)) = 3 * (2*3 + 1) = 3 * 7 = 21.\n",
                    "    (Not: Cartan rank+kök formülü dim = 3 + 18 = 21 ile de tam örtüşür; 12 hatası 2n(n-1) veya 4n yanlış eşlemesinden doğar).\n",
                    "  Şartlar: Wronskian(C_n_simplektik_sp, sp_6_lie_cebiri) = bağımsız → RULE geçerli\n",
                    "Cevap: sp(6) Lie cebirinin boyutu 21'dir (C_3 ailesi, formül: n(2n+1) = 3 × 7 = 21)."
                ).into(),
                correct_action: "simplektik_boyut_kurali_21".into(),
                distractor_action: "yanlis_boyut_12".into(),
                precondition_satisfied_by_correct: true,
                precondition_satisfied_by_distractor: false,
                bayes_confidence: 0.98,
            },
            // 2. Exceptional Lie algebra F4 dimension = 52
            SyntheticReasoningSample {
                id: "algebra_f4_dimension".into(),
                category: ReasoningCategory::LieAlgebraicReasoning,
                prompt: "İstisnai Lie cebiri F4'ün boyutu Cartan kök sistemi kullanılarak nasıl türetilir?".into(),
                target_reasoning_chain: concat!(
                    "Arkhe İzi:\n",
                    "  Benzerlik: \"F4\" → istisnai Lie cebiri ve Cartan kök ayrışımı\n",
                    "  Çıkarım: Cartan kök sistemi formülü: dim(g) = rank + |Phi| = rank + 2|Phi^+|.\n",
                    "    F4 cebirinin rankı 4'tür. Pozitif kök sayısı |Phi^+| = 24, toplam kök sayısı |Phi| = 48'dir.\n",
                    "    Boyut = 4 + 48 = 52.\n",
                    "  Şartlar: Wronskian(cartan_altcebiri, kok_sistemi_Phi) = bağımsız → RULE geçerli\n",
                    "Cevap: F4'ün boyutu 52'dir (rank 4 + 48 kök = 52)."
                ).into(),
                correct_action: "cartan_rank_arti_kokler_52".into(),
                distractor_action: "so8_d4_boyut_28".into(),
                precondition_satisfied_by_correct: true,
                precondition_satisfied_by_distractor: false,
                bayes_confidence: 0.98,
            },
            // 3. Sylow subgroups of order 21
            SyntheticReasoningSample {
                id: "algebra_sylow_count".into(),
                category: ReasoningCategory::AbstractAlgebraicReasoning,
                prompt: "Mertebesi 21 = 3 × 7 olan bir grupta Sylow-3 ve Sylow-7 altgruplarının olası sayıları nelerdir?".into(),
                target_reasoning_chain: concat!(
                    "Arkhe İzi:\n",
                    "  Benzerlik: \"Sylow altgrubu sayısı\" → Sylow 3. Teoremi rezonansı\n",
                    "  Çıkarım: |G| = 21 = 3 * 7.\n",
                    "    n_7 için: n_7 ≡ 1 (mod 7) ve n_7 | 3 → tek çözüm n_7 = 1 (Sylow-7 altgrubu normaldir).\n",
                    "    n_3 için: n_3 ≡ 1 (mod 3) ve n_3 | 7 → olası değerler n_3 = 1 veya n_3 = 7.\n",
                    "  Şartlar: Lagrange ve Sylow bölünebilirlik koşulları doğrulanmıştır.\n",
                    "Cevap: Sylow-7 altgrubu tek olup n_7 = 1'dir; Sylow-3 altgrup sayısı ise n_3 ∈ {1, 7} olabilir."
                ).into(),
                correct_action: "sylow_kisitlari_n7_1_n3_1_veya_7".into(),
                distractor_action: "tum_sylowlar_tekil".into(),
                precondition_satisfied_by_correct: true,
                precondition_satisfied_by_distractor: false,
                bayes_confidence: 0.95,
            },
            // 4. Maximal ideal and quotient field
            SyntheticReasoningSample {
                id: "algebra_maximal_ideal".into(),
                category: ReasoningCategory::AbstractAlgebraicReasoning,
                prompt: "R[x] polinom halkasında I = (x^2 + 1) idealinin maksimal olduğunu ve R[x]/I bölüm halkasının yapısını gösteriniz.".into(),
                target_reasoning_chain: concat!(
                    "Arkhe İzi:\n",
                    "  Benzerlik: \"maksimal ideal\" → \"bölüm halkası cisimdir\" teoremi\n",
                    "  Çıkarım: R[x] bir esas ideal bölgesidir (PID).\n",
                    "    x^2 + 1 polinomunun R üzerinde gerçel kökü yoktur, derece 2 olduğundan R üzerinde indirgenemezdir.\n",
                    "    PID üzerinde indirgenemez elemanın ürettiği ideal maksimaldir.\n",
                    "    Dolayısıyla R[x]/(x^2 + 1) bir cisimdir ve x mod I elemanı i = sqrt(-1) rolü oynayarak C'ye izomorftur.\n",
                    "  Şartlar: Maksimal ideal aksiyomu ve PID indirgenemezlik kuralı sağlanır.\n",
                    "Cevap: x^2 + 1 gerçel sayılar üzerinde indirgenemez olduğundan (x^2 + 1) maksimal idealdir ve R[x]/(x^2 + 1) ≅ C (karmaşık sayılar cismi) elde edilir."
                ).into(),
                correct_action: "maksimal_ideal_cisim_izomorf_C".into(),
                distractor_action: "sadece_asal_ideal".into(),
                precondition_satisfied_by_correct: true,
                precondition_satisfied_by_distractor: false,
                bayes_confidence: 0.96,
            },
            // 5. Galois group of biquadratic extension
            SyntheticReasoningSample {
                id: "algebra_galois_biquadratic".into(),
                category: ReasoningCategory::AbstractAlgebraicReasoning,
                prompt: "Q üzerinde x^4 - 5x^2 + 6 = (x^2 - 2)(x^2 - 3) polinomunun ayrışma cismi ve Galois grubu nedir?".into(),
                target_reasoning_chain: concat!(
                    "Arkhe İzi:\n",
                    "  Benzerlik: \"ayrışma cismi\" → Galois grubu otomorfizmaları\n",
                    "  Çıkarım: Polinomun kökleri ±sqrt(2) ve ±sqrt(3)'tür.\n",
                    "    Ayrışma cismi L = Q(sqrt(2), sqrt(3)).\n",
                    "    Kule kanunu: [L:Q] = [Q(sqrt(2), sqrt(3)):Q(sqrt(2))] * [Q(sqrt(2)):Q] = 2 * 2 = 4.\n",
                    "    Galois grubu Gal(L/Q) 4 elemanlıdır. Otomorfizmalar sqrt(2) → ±sqrt(2) ve sqrt(3) → ±sqrt(3) bağımsız işaret değişimleridir.\n",
                    "    Her elemanın mertebesi 2 olduğundan Gal(L/Q) ≅ Z_2 × Z_2 (Klein-4 grubu).\n",
                    "  Şartlar: Ayrılabilir normal genişleme ve Galois eşlemesi doğrulanmıştır.\n",
                    "Cevap: Ayrışma cismi Q(sqrt(2), sqrt(3)) olup Galois grubu Klein-4 grubu Z_2 × Z_2'ye izomorftur."
                ).into(),
                correct_action: "galois_grubu_klein_4_Z2_x_Z2".into(),
                distractor_action: "devirli_grup_Z4".into(),
                precondition_satisfied_by_correct: true,
                precondition_satisfied_by_distractor: false,
                bayes_confidence: 0.96,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curriculum_default_generation() {
        let samples = CurriculumGenerator::generate_default_curriculum();
        assert!(!samples.is_empty());
        assert!(samples.iter().any(|s| s.category == ReasoningCategory::GoalPrecondition));
        assert!(samples.iter().any(|s| s.category == ReasoningCategory::Analogical));
    }

    #[test]
    fn test_curriculum_from_cst() {
        let mut cst = ContinuousSemanticTopology::new();
        cst.goals.push(GoalNode {
            goal_state: "araba_yikanmis".into(),
            precondition: "araba_konumu_yikama".into(),
            energy: 1.0,
            phase_theta: 0.0,
        });

        let samples = CurriculumGenerator::generate_from_topology(&cst);
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].category, ReasoningCategory::GoalPrecondition);
        assert!(samples[0].prompt.contains("araba_yikanmis"));
    }

    #[test]
    fn test_mathematics_curriculum_generation() {
        let samples = CurriculumGenerator::generate_mathematics_curriculum();
        assert_eq!(samples.len(), 5);
        
        // Check both reasoning categories present
        assert!(samples.iter().any(|s| s.category == ReasoningCategory::EpsilonDeltaReasoning));
        assert!(samples.iter().any(|s| s.category == ReasoningCategory::TopologicalInvariance));
        
        // Check Bayes confidence scores are valid
        for sample in &samples {
            assert!(sample.bayes_confidence > 0.0 && sample.bayes_confidence <= 1.0,
                "Bayes confidence must be in (0,1]: {}", sample.bayes_confidence);
        }
        
        // Check Arkhe İzi format present
        assert!(samples[0].target_reasoning_chain.contains("Arkhe İzi"));
        assert!(samples[0].target_reasoning_chain.contains("Benzerlik"));
        assert!(samples[0].target_reasoning_chain.contains("Çıkarım"));
    }

    #[test]
    fn test_abstract_algebra_curriculum_generation() {
        let samples = CurriculumGenerator::generate_abstract_algebra_curriculum();
        assert_eq!(samples.len(), 5);

        // Check categories
        assert!(samples.iter().any(|s| s.category == ReasoningCategory::LieAlgebraicReasoning));
        assert!(samples.iter().any(|s| s.category == ReasoningCategory::AbstractAlgebraicReasoning));

        // Verify critical epistemic fix: sp(6) = 21 sample exists and is correct
        let sp6 = samples.iter().find(|s| s.id == "algebra_sp6_dimension").expect("sp(6) sample missing");
        assert!(sp6.target_reasoning_chain.contains("21"));
        assert!(sp6.correct_action.contains("21"));
        assert_eq!(sp6.category, ReasoningCategory::LieAlgebraicReasoning);

        // Verify F4 = 52 sample exists and is correct
        let f4 = samples.iter().find(|s| s.id == "algebra_f4_dimension").expect("F4 sample missing");
        assert!(f4.target_reasoning_chain.contains("52"));
        assert!(f4.correct_action.contains("52"));

        // Check Bayes confidence bounds
        for sample in &samples {
            assert!(sample.bayes_confidence >= 0.90 && sample.bayes_confidence <= 1.0,
                "Bayes confidence should be high for canonical algebra derivations: {}", sample.bayes_confidence);
            assert!(sample.target_reasoning_chain.contains("Arkhe İzi"));
            assert!(sample.precondition_satisfied_by_correct);
            assert!(!sample.precondition_satisfied_by_distractor);
        }
    }
}
