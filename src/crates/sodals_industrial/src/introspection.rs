//! Introspection (İçe Bakış) Modülü
//!
//! SODALS-IoT teknik spesifikasyonunda belirtilen matematiksel temellerin
//! zero-allocation (stack-only) Rust uygulamaları.
//!
//! Kurallar:
//! - Hot-path'te heap allocasyonu kesinlikle yasaktır (no allocations).
//! - Tüm matris ve vektör işlemleri sabit boyutlu stack dizileriyle gerçekleştirilir.

pub const TEMPLATE_SIZE: usize = 32;

/// Katman 0: Sensör Kalibrasyonu ve Bazal Çizgi (Z-Skor Sınır Kontrolü)
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct SensorCalibration {
    pub sensor_id: u32,
    pub baseline_mean: f64,
    pub baseline_std: f64,
    pub dtw_template: [f64; TEMPLATE_SIZE],
    pub token_id: u32,
}

impl Default for SensorCalibration {
    fn default() -> Self {
        Self {
            sensor_id: 0,
            baseline_mean: 0.0,
            baseline_std: 0.0,
            dtw_template: [0.0; TEMPLATE_SIZE],
            token_id: 0,
        }
    }
}

impl SensorCalibration {
    pub const fn new(
        sensor_id: u32,
        baseline_mean: f64,
        baseline_std: f64,
        dtw_template: [f64; TEMPLATE_SIZE],
        token_id: u32,
    ) -> Self {
        Self {
            sensor_id,
            baseline_mean,
            baseline_std,
            dtw_template,
            token_id,
        }
    }

    /// Sinyal okumasının normal baselinedan sapmasını 3-sigma kuralına göre test eder.
    pub fn is_anomalous(&self, reading: f64, deviation_multiplier: f64) -> bool {
        let std = if self.baseline_std < 1e-9 { 1e-9 } else { self.baseline_std };
        let z_score = (reading - self.baseline_mean).abs() / std;
        z_score > deviation_multiplier
    }
}

/// Katman 2.1: Wronskian Determinantı ile Lineer Bağımsızlık ve Anomali Tespiti
///
/// W(f1, f2) = f1 * f2' - f2 * f1'
/// dt: örnekleme aralığı (saniye)
pub fn compute_wronskian(f1: &[f64; 2], f2: &[f64; 2], dt: f64) -> f64 {
    let f1_t = f1[1];
    let f2_t = f2[1];
    
    // Merkezi fark ile türev yaklaşımı
    let f1_prime = (f1[1] - f1[0]) / dt;
    let f2_prime = (f2[1] - f2[0]) / dt;
    
    f1_t * f2_prime - f2_t * f1_prime
}

/// Katman 2.2: Jakobiyen Durum Uzayı Degradasyon Modeli
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StateSpace {
    pub temperature: f64,
    pub pressure: f64,
    pub rpm: f64,
    pub vibration: f64,
}

impl StateSpace {
    pub const fn new(temperature: f64, pressure: f64, rpm: f64, vibration: f64) -> Self {
        Self {
            temperature,
            pressure,
            rpm,
            vibration,
        }
    }

    pub fn to_array(&self) -> [f64; 4] {
        [self.temperature, self.pressure, self.rpm, self.vibration]
    }
}

/// İki ardışık durum uzayı noktası arasındaki değişim gradyanını yaklaştıran Jakobiyen matrisi
pub fn compute_jacobian(current: &StateSpace, previous: &StateSpace, dt: f64) -> [[f64; 4]; 4] {
    let mut jac = [[0.0; 4]; 4];
    let states_curr = current.to_array();
    let states_prev = previous.to_array();
    
    for i in 0..4 {
        let diff_y = states_curr[i] - states_prev[i];
        for j in 0..4 {
            let diff_x = states_curr[j] - states_prev[j];
            // Kısmi türev yaklaştırması: dy/dx * dt ile zaman ölçeği entegrasyonu
            jac[i][j] = if diff_x.abs() > 1e-9 {
                (diff_y / diff_x) / dt
            } else {
                0.0
            };
        }
    }
    jac
}

/// Power Iteration (Güç İterasyonu) yöntemiyle 4x4 matrisin dominant özdeğerini (dominant eigenvalue) hesaplar.
/// Zero-allocation ve deterministik çalışma garantisi sunar.
pub fn power_iteration_4x4(matrix: &[[f64; 4]; 4], max_iters: usize, tol: f64) -> f64 {
    let mut v = [1.0, 1.0, 1.0, 1.0];
    let mut v_new = [0.0; 4];
    let mut lambda = 0.0;
    
    for _ in 0..max_iters {
        // v_new = A * v
        for i in 0..4 {
            v_new[i] = 0.0;
            for j in 0..4 {
                v_new[i] += matrix[i][j] * v[j];
            }
        }
        
        // v_new norm hesabı
        let norm = (v_new[0].powi(2) + v_new[1].powi(2) + v_new[2].powi(2) + v_new[3].powi(2)).sqrt();
        if norm < 1e-9 {
            break;
        }
        
        let new_lambda = norm;
        
        // Normalize v
        for i in 0..4 {
            v[i] = v_new[i] / norm;
        }
        
        if (new_lambda - lambda).abs() < tol {
            lambda = new_lambda;
            break;
        }
        lambda = new_lambda;
    }
    lambda
}

/// Kalan Faydalı Ömür (RUL) Tahmini
pub fn estimate_rul(jacobian: &[[f64; 4]; 4], critical_threshold: f64, epsilon: f64) -> f64 {
    let lambda_max = power_iteration_4x4(jacobian, 50, 1e-6);
    critical_threshold / (lambda_max + epsilon)
}

/// Katman 3: LSQ (En Küçük Kareler) ve Causal Transpoz Ağırlık Güncellemesi
///
/// w_new = w_old + learning_rate * Aᵀ(b - A * w_old)
///
/// N: Geçmiş zaman pencerelerinin sayısı
/// SENSORS: İzlenen sensör sayısı
pub fn update_weights_lsq<const N: usize, const SENSORS: usize>(
    a: &[[f64; SENSORS]; N],
    b: &[f64; N],
    w_old: &[f64; SENSORS],
    learning_rate: f64,
    w_new: &mut [f64; SENSORS],
) {
    // 1. Residual hesabı: e = b - A * w_old
    let mut residuals = [0.0; N];
    for i in 0..N {
        let mut prediction = 0.0;
        for j in 0..SENSORS {
            prediction += a[i][j] * w_old[j];
        }
        residuals[i] = b[i] - prediction;
    }
    
    // 2. Gradyan yansıtma (Aᵀ * e)
    let mut gradient = [0.0; SENSORS];
    for j in 0..SENSORS {
        for i in 0..N {
            gradient[j] += a[i][j] * residuals[i];
        }
    }
    
    // 3. Ağırlık güncelleme
    for j in 0..SENSORS {
        w_new[j] = w_old[j] + learning_rate * gradient[j];
    }
}

/// Golden Teacher RL Boyut İndirgeme: En yüksek değere sahip K sensörü seçer
pub fn select_top_features<const SENSORS: usize, const K: usize>(
    combined_scores: &[f64; SENSORS],
) -> [usize; K] {
    let mut scored_indices = [(0.0, 0); SENSORS];
    for i in 0..SENSORS {
        scored_indices[i] = (combined_scores[i], i);
    }
    
    // Stack-safe selection sort
    let k_limit = K.min(SENSORS);
    for i in 0..k_limit {
        let mut max_idx = i;
        for j in (i + 1)..SENSORS {
            if scored_indices[j].0 > scored_indices[max_idx].0 {
                max_idx = j;
            }
        }
        if max_idx != i {
            scored_indices.swap(i, max_idx);
        }
    }
    
    let mut selected = [0; K];
    for i in 0..k_limit {
        selected[i] = scored_indices[i].1;
    }
    selected
}

/// Katman 4: Newton İleri Fark İnterpolasyon Düğümü (Eksik Veri Çözümü)
///
/// x: Zaman noktaları
/// y: Sensör değerleri
/// target_x: Değeri doldurulmak istenen zaman noktası
pub fn newton_interpolate(x: &[f64], y: &[f64], target_x: f64) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return y[0];
    }
    
    // Maksimum 16 nokta ile stack sınırlandırması
    let limit = n.min(16);
    let mut diff_table = [0.0; 16];
    for i in 0..limit {
        diff_table[i] = y[i];
    }
    
    // Bölünmüş farklar (divided differences) tablosunu çıkar
    for j in 1..limit {
        for i in (j..limit).rev() {
            let denominator = x[i] - x[i - j];
            diff_table[i] = if denominator.abs() > 1e-9 {
                (diff_table[i] - diff_table[i - 1]) / denominator
            } else {
                0.0
            };
        }
    }
    
    // Newton Polinomunu değerlendir
    let mut result = diff_table[0];
    let mut term = 1.0;
    for i in 1..limit {
        term *= target_x - x[i - 1];
        result += diff_table[i] * term;
    }
    result
}

/// Katman 1: Termodinamik Korunum Yasası / Fiziksel Gramer Kontrolü
pub fn validate_thermodynamic_grammar(
    winding_temp: f64,
    ambient_temp: f64,
    cooling_flow: f64,
) -> bool {
    // 1. Akış hızı negatif olamaz
    if cooling_flow < 0.0 {
        return false;
    }
    // 2. Sargı sıcaklığı ortam sıcaklığından belirgin derecede düşük olamaz (Termodinamik 2. Yasa)
    if winding_temp < ambient_temp - 5.0 {
        return false;
    }
    true
}
