// src/crates/sodals_kernel/src/sentinel.rs - SELF-HEALING SENTINEL (ANALYTIC ENRICHED)

use tokio::sync::mpsc;
use sodals_net::protocol::SystemMsg;
use ndarray::Array1;
use rand::{Rng, thread_rng};
use num_complex::Complex64;

// Fast Math modülüne erişim (Skew Code ve SemanticWave için)
use sodals_neuro::fast_math::SemanticWave;
use crate::crisis_manager::{CrisisManager, CrisisTrigger};

// Wronskian Eşiği: Bu değerin altındaki determinantlar "Lineer Bağımlı/Çökme" kabul edilir.
pub const WRONSKIAN_THRESHOLD: f64 = 0.05;

// ============================================================================
// SEMANTIC KALMAN FILTER (Uncoupled Dimension-Wise Tracker)
// ============================================================================
// Recurvisively estimates the hidden true semantic trajectory of high-D vectors.
// Diagonal covariance assumption for O(D) execution time.
#[derive(Clone)]
pub struct SemanticKalmanFilter {
    estimates: Option<Array1<f32>>,
    uncertainties: Option<Array1<f32>>, // Diagonal Covariance Matrix P
    process_noise: f32,                 // Q (Expected semantic drift per step)
    measurement_noise: f32,             // R (Expected noise per vector snapshot)
}

impl SemanticKalmanFilter {
    pub fn new(process_noise: f32, measurement_noise: f32) -> Self {
        Self {
            estimates: None,
            uncertainties: None,
            process_noise,
            measurement_noise,
        }
    }

    pub fn step(&mut self, observed: &Array1<f32>) -> Array1<f32> {
        let d = observed.len();

        // Initialize filter on first vector
        let (x_hat, mut p) = match (&self.estimates, &self.uncertainties) {
            (Some(est), Some(unc)) => (est.clone(), unc.clone()),
            _ => {
                let initial_x = observed.clone();
                let initial_p = Array1::from_elem(d, 1.0f32);
                self.estimates = Some(initial_x.clone());
                self.uncertainties = Some(initial_p);
                return initial_x;
            }
        };

        // 1. Predict step (A = I, no motion model yet)
        // x_hat remains same, P = P + Q
        for i in 0..d {
            p[i] += self.process_noise;
        }

        // 2. Update Step (Fusion)
        let mut corrected = Array1::zeros(d);
        for i in 0..d {
            // Kalman Gain K = P / (P + R)
            let k = p[i] / (p[i] + self.measurement_noise);
            
            // Update estimate: x = x + K * (y - x)
            let innovation = observed[i] - x_hat[i];
            let new_est = x_hat[i] + k * innovation;
            corrected[i] = new_est;
            
            // Update Uncertainty: P = (1 - K) * P
            p[i] = (1.0 - k) * p[i];
        }

        // Store internal state for next recursive iteration
        self.estimates = Some(corrected.clone());
        self.uncertainties = Some(p);

        corrected
    }
}

// ============================================================================
// ADAPTIVE SENTINEL LAYER
// ============================================================================

pub struct AdaptiveSentinel {
    epsilon: f64, // Privacy Budget
    penalty_lambda: f64,
    sensitivity_alpha: f64,
    /// TELEMETRY ENRICHMENT: Kalman tracker for temporal stabilization
    kalman: SemanticKalmanFilter,
    /// CRISIS LOG AUTOMATION: Logs critical defense logic to disk
    crisis_mgr: CrisisManager,
}

impl AdaptiveSentinel {
    pub fn new(epsilon: f64, lambda: f64, alpha: f64) -> Self {
        Self { 
            epsilon, 
            penalty_lambda: lambda, 
            sensitivity_alpha: alpha,
            kalman: SemanticKalmanFilter::new(0.05, 0.2),
            crisis_mgr: CrisisManager::default(),
        }
    }

    /// Wronskian Determinantına dayalı sağlık kontrolü ve gürültü faktörü hesabı
    fn compute_wronskian_health(&self, vector: &Array1<f32>) -> (f64, bool) {
        // 1. Enerji Analizi (L1 Norm)
        let l1_energy: f32 = vector.fold(0.0, |acc, x| acc + x.abs());
        let dim = vector.len() as f32;
        let normalized_energy = (l1_energy / dim) as f64;

        // 2. Wronskian Benzerliği (Basitleştirilmiş Entropi Kontrolü)
        // Eğer enerji çok düşükse veya çok yüksekse (Gradient Explosion/Vanishing), sistem kararsızdır.
        let is_healthy = normalized_energy > WRONSKIAN_THRESHOLD && normalized_energy < 5.0;

        // 3. Dinamik Gürültü Çarpanı (Differential Privacy için)
        let dynamic_multiplier = 1.0 + self.penalty_lambda * (-self.sensitivity_alpha * normalized_energy).exp();
        
        (dynamic_multiplier, is_healthy)
    }

    /// PDE Partial Differential Equation based Heat Diffusion smoothing
    /// Treats vector as a continuous semantic topology and dissipates sudden energy spikes.
    fn apply_heat_diffusion(&self, vector: &Array1<f32>, iterations: usize, diffusion_coefficient: f32) -> Array1<f32> {
        let mut u = vector.clone();
        let n = u.len();
        if n < 3 { return u; }

        for _ in 0..iterations {
            let mut next_u = u.clone();
            // Apply finite difference Laplacian: dU/dt = alpha * d^2U/dx^2
            for i in 1..n-1 {
                let laplacian = u[i+1] - 2.0 * u[i] + u[i-1];
                next_u[i] = u[i] + diffusion_coefficient * laplacian;
            }
            // Boundary conditions (periodic fallback)
            next_u[0] = u[0] + diffusion_coefficient * (u[1] - 2.0 * u[0] + u[n-1]);
            next_u[n-1] = u[n-1] + diffusion_coefficient * (u[0] - 2.0 * u[n-1] + u[n-2]);
            
            u = next_u;
        }
        u
    }

    /// Vektörü güvence altına alır ve gerekirse ONARIR (Self-Healing)
    pub fn secure_and_heal(&mut self, raw_vector: &[f32]) -> anyhow::Result<(Vec<f32>, f64)> {
        // Ndarray formatına çevir
        let vector = Array1::from_vec(raw_vector.to_owned());
        
        // A. Teşhis (Diagnosis) - Wronskian Kontrolü
        let (noise_factor, is_healthy) = self.compute_wronskian_health(&vector);
        
        let final_vector = if is_healthy {
            // Durum A: Vektör Sağlıklı -> PDE Heat Diffusion + Gizlilik Gürültüsü
            // Smooth noise dynamically using PDE smoothing before adding Laplace noise
            let smoothed = self.apply_heat_diffusion(&vector, 2, 0.15);
            self.apply_laplace_noise(&smoothed, noise_factor)
        } else {
            // Durum B: Vektör HASTA -> Attractor Multi-Stage Correction
            eprintln!("[SENTINEL] ⚠️ System Instability Detected. Initiating Attractor Stabilization protocol...");
            
            // AUTOMATION: Trigger Crisis Manager and write automatic ledger documentation
            let _defensive_action = self.crisis_mgr.handle_emergency(
                CrisisTrigger::SevereWronskianDeviation, 
                noise_factor
            );
            eprintln!("[SENTINEL] Automated Document Ledger Updated: {:?}", _defensive_action);
            
            // 1. Skew Code Recovery (ABS Braking)
            let skew_corrected_raw = self.apply_skew_correction(&vector);
            let mut vector_node = Array1::from_vec(skew_corrected_raw);

            // 2. Kalman Trajectory Fusion
            // Bias the corrected vector back to the estimated underlying semantic state
            vector_node = self.kalman.step(&vector_node);

            // 3. Strong PDE Heat Dissipation (Aggressive smoothing to prevent exploding NaNs)
            let smoothed_vector = self.apply_heat_diffusion(&vector_node, 5, 0.25);
            
            smoothed_vector.to_vec()
        };

        Ok((final_vector, noise_factor))
    }

    /// Laplace Gürültüsü (Standart Gizlilik Prosedürü)
    fn apply_laplace_noise(&self, vector: &Array1<f32>, factor: f64) -> Vec<f32> {
        let mut rng = thread_rng();
        let scale = factor / self.epsilon;
        
        vector.iter().map(|&val| {
            // Laplace dağılımı simülasyonu: u ~ Uniform(-0.5, 0.5) -> -sgn(u) * ln(1-2|u|) * b
            let u: f64 = rng.gen::<f64>() - 0.5;
            let noise = -u.signum() * (1.0 - 2.0 * u.abs()).ln() * scale;
            (val as f64 + noise) as f32
        }).collect()
    }

    /// SKEW G-CODE CORRECTION (Makaledeki Onarım Algoritması)
    fn apply_skew_correction(&self, vector: &Array1<f32>) -> Vec<f32> {
        // 1. Reel Vektörü -> Karmaşık Dalga Fonksiyonuna (SemanticWave) Çevir
        // (Reel kısmı vektörden al, Sanal kısmı 0 kabul et)
        let complex_data: Vec<Complex64> = vector.iter()
            .map(|&x| Complex64::new(x as f64, 0.0))
            .collect();
        
        let mut wave = SemanticWave::from_vec(complex_data);

        // 2. Kuantizasyon (Gray Map): Complex64 -> F4 (Skew Code)
        let skew_code = wave.to_skew_code();

        // 3. Parity Check (Sağlama): Hata var mı? (Zaten biliyoruz ama prosedür gereği)
        if !wave.check_parity() {
            // 4. ONARIM (Heal): Skew Code'dan geri yükle
            // Bu işlem gürültüyü (bozuk fazları) "en yakın geçerli F4 sembolüne" yuvarlar.
            wave.heal_from_skew_code(&skew_code);
        }

        // 5. Geri Dönüş: SemanticWave -> Vec<f32>
        wave.amplitudes.iter()
            .map(|c| c.re as f32) // Sadece reel kısmı al (Sanal kısım gürültüydü, atıldı)
            .collect()
    }
}

pub struct SentinelActor {
    rx: mpsc::Receiver<SystemMsg>,
    kernel_tx: mpsc::Sender<SystemMsg>,
    engine: AdaptiveSentinel,
}

impl SentinelActor {
    pub fn new(rx: mpsc::Receiver<SystemMsg>, kernel_tx: mpsc::Sender<SystemMsg>) -> Self {
        // Epsilon=1.0 (Yüksek Gizlilik), Lambda=0.5 (Ceza), Alpha=2.0 (Hassasiyet)
        let engine = AdaptiveSentinel::new(1.0, 0.5, 2.0);
        Self { rx, kernel_tx, engine }
    }

    pub async fn run(mut self) {
        println!("[SENTINEL] Güvenlik Kalkanı & Skew Onarım Sistemi Aktif (PDE & Kalman Enhanced).");

        while let Some(msg) = self.rx.recv().await {
            if let SystemMsg::SanitizeVectorRequest { req_id, raw_vector } = msg {
                // NOTE: Now calling with &mut self to allow Kalman updates.
                match self.engine.secure_and_heal(&raw_vector) {
                    Ok((safe_vec, noise)) => { 
                        let _ = self.kernel_tx.send(SystemMsg::SanitizeVectorResponse {
                            req_id,
                            safe_vector: safe_vec,
                            noise_level: noise, 
                        }).await;
                    },
                    Err(e) => {
                        eprintln!("[SENTINEL] Kritik Hata: {}", e);
                        // Hata durumunda SIFIRLANMIŞ veri dönülür (FAIL-CLOSED)
                        let zeroed_vector = vec![0.0; raw_vector.len()];
                        let _ = self.kernel_tx.send(SystemMsg::SanitizeVectorResponse {
                            req_id,
                            safe_vector: zeroed_vector,
                            noise_level: 0.0,
                        }).await;
                    }
                }
            }
        }
    }
}