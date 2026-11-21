//! ADIM 4: Sentinel Entegrasyon Şablonu
//!
//! Sentinel, Qwen 8B modelinin çıktısı PLC'ye yazılmadan ÖNCE
//! fizik tabanlı Euler/frekans imzası kontrolü yapar.
//!
//! Bu modül iki parçadan oluşur:
//! 1. `SentinelGuard` (Rust): ZeroMQ'dan veri alır, .qmem imzaları ile
//!    karşılaştırır, "izin ver / reddet" kararı üretir.
//! 2. `PyO3` adaptör köprüsü: Python tarafında çalışan Qwen 8B modelinin
//!    çıktısı (logit vektörü) Rust Sentinel'e gönderilir.
//!
//! Karmaşıklık:
//! - İmza karşılaştırması: O(128) = O(1) (sabit boyutlu 128-bant FFT).
//! - Euler açı hesaplama: O(1), LUT (Lookup Table) tabanlı.
//! - Bellek: Stack-only, alloc yok.

use crate::{
    qmem_format::FrequencySignatureBlock, IndustrialError,
    SensorPayload,
};
use core::sync::atomic::{AtomicU64, Ordering};

/// Sentinel karar sınırı.
/// Kosinüs benzerliği < 0.85 ise anomali olarak işaretlenir.
const COSINE_THRESHOLD: f32 = 0.85;

/// Referans (normal çalışma) frekans imzası.
/// Bu, kalibrasyon sırasında öğrenilir ve sabitlenir.
/// Init-time tek yüklenme; hot-path'te değişmez.
pub struct SentinelReference {
    pub signature: FrequencySignatureBlock,
    pub threshold_db: f32,
}

/// Sentinel Güvenlik Katmanı.
pub struct SentinelGuard {
    reference: SentinelReference,
    /// Son anomali zaman damgası
    last_anomaly_ns: AtomicU64,
    /// Arka arkaya kaç anomali sayıldı (debounce için)
    anomaly_counter: AtomicU64,
    /// Debounce eşiği: 3 ardışık anomali -> Fail-Safe
    debounce_threshold: u64,
}

impl SentinelGuard {
    pub fn new(reference: SentinelReference, debounce_threshold: u64) -> Self {
        Self {
            reference,
            last_anomaly_ns: AtomicU64::new(0),
            anomaly_counter: AtomicU64::new(0),
            debounce_threshold,
        }
    }

    /// Gelen sensör verisinin frekans imzasını referans ile karşılaştırır.
    /// O(128) = O(1), stack-only, branch-predictable.
    pub fn evaluate(&self, candidate: &FrequencySignatureBlock) -> SentinelDecision {
        let now = candidate.header.timestamp_ns;

        // 1. Anomali tespiti (enerji tabanlı)
        if candidate.is_anomalous(self.reference.threshold_db) {
            self.last_anomaly_ns.store(now, Ordering::Relaxed);
            let count = self.anomaly_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if count >= self.debounce_threshold {
                return SentinelDecision::Reject(SentinelRejectionReason::EnergyAnomaly);
            }
            return SentinelDecision::Hold; // Bekle, henüz reddetme
        }

        // 2. Spektral benzerlik (cosine similarity) — fail-closed on NaN/Inf!
        let sim = candidate.cosine_similarity(&self.reference.signature);
        if !sim.is_finite() || sim < COSINE_THRESHOLD {
            self.last_anomaly_ns.store(now, Ordering::Relaxed);
            let count = self.anomaly_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if count >= self.debounce_threshold {
                return SentinelDecision::Reject(SentinelRejectionReason::SpectralMismatch);
            }
            return SentinelDecision::Hold;
        }

        // 3. Normal: sayaç sıfırla, izin ver
        self.anomaly_counter.store(0, Ordering::Relaxed);
        SentinelDecision::Allow
    }

    /// Qwen 8B modelinin çıktısı (logit vektörü) Sentinel'e girmeden önce
    /// fiziksel uyumluluk kontrolü. Bu fonksiyon AI çıktısını
    /// doğrudan PLC komutuna dönüştürmez — sadece "güvenli mi?" diye sorar.
    ///
    /// `logits`: Qwen çıkışının son katman logit değerleri (örn: 8 eleman).
    /// `sensor`: Anlık sensör verisi (fiziksel sınırlar için).
    pub fn validate_ai_output(
        &self,
        logits: &[f32; 8],
        sensor: &SensorPayload,
    ) -> Result<(), IndustrialError> {
        // 1. Fiziksel sınır kontrolü — NaN/Inf fail-closed!
        if !sensor.temperature.is_finite()
            || !sensor.pressure.is_finite()
            || !sensor.motor_rpm.is_finite()
            || sensor.temperature > 200.0
            || sensor.pressure > 50.0
        {
            return Err(IndustrialError::SentinelRejected);
        }

        // 2. Logit kararlılığı (NaN/Inf fail-closed + variyans kontrolü)
        if logits.iter().any(|x| !x.is_finite()) {
            return Err(IndustrialError::SentinelRejected);
        }
        let mean = logits.iter().sum::<f32>() / 8.0;
        let variance = logits.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 8.0;
        if !variance.is_finite() || variance > 100.0 {
            // Model kararsız -> reddet
            return Err(IndustrialError::SentinelRejected);
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SentinelDecision {
    /// PLC'ye yazılabilir.
    Allow,
    /// Değerlendirme sürüyor, komut beklemede.
    Hold,
    /// Reddedildi. Fail-Safe komutu üretilmeli.
    Reject(SentinelRejectionReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SentinelRejectionReason {
    EnergyAnomaly,
    SpectralMismatch,
    PhysicalLimitExceeded,
    ModelInstability,
}

// ============================================================================
// PyO3 Adaptör Köprüsü (Python <-> Rust)
// ============================================================================
// Bu bölüm `pyo3` crate'i aktif olduğunda derlenir.
// Python tarafından çağrılacak fonksiyonları tanımlar.
//
// Kullanım (Python):
//   import sodals_industrial
//   guard = sodals_industrial.SentinelGuardPy(ref_sig_bytes)
//   decision = guard.evaluate(candidate_bytes)
// ============================================================================

#[cfg(feature = "python")]
pub mod pyo3_bridge {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    /// Python'dan erişilebilir Sentinel Guard wrapper'ı.
    #[pyclass]
    pub struct SentinelGuardPy {
        inner: SentinelGuard,
    }

    #[pymethods]
    impl SentinelGuardPy {
        #[new]
        fn new(reference_bytes: &[u8], threshold_db: f32, debounce: u64) -> PyResult<Self> {
            if reference_bytes.len() != 544 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Referans imza 544 byte olmalı",
                ));
            }
            let signature = unsafe {
                std::ptr::read_unaligned(reference_bytes.as_ptr() as *const FrequencySignatureBlock)
            };
            let reference = SentinelReference {
                signature,
                threshold_db,
            };
            Ok(Self {
                inner: SentinelGuard::new(reference, debounce),
            })
        }

        /// Python'dan `bytes` olarak gelen aday imzayı değerlendirir.
        /// Dönüş: 0=Allow, 1=Hold, 2=Reject
        fn evaluate(&self, candidate_bytes: &[u8]) -> PyResult<u8> {
            if candidate_bytes.len() != 544 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Aday imza 544 byte olmalı",
                ));
            }
            let candidate = unsafe {
                std::ptr::read_unaligned(
                    candidate_bytes.as_ptr() as *const FrequencySignatureBlock,
                )
            };
            match self.inner.evaluate(&candidate) {
                SentinelDecision::Allow => Ok(0),
                SentinelDecision::Hold => Ok(1),
                SentinelDecision::Reject(_) => Ok(2),
            }
        }

        /// Qwen 8B çıktı logitlerini ve sensör verisini doğrular.
        /// Dönüş: true=güvenli, false=reddedildi
        fn validate_ai_output(&self, logits: [f32; 8], sensor_bytes: &[u8]) -> PyResult<bool> {
            if sensor_bytes.len() != 24 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "SensorPayload 24 byte olmalı",
                ));
            }
            let sensor = unsafe {
                std::ptr::read_unaligned(sensor_bytes.as_ptr() as *const SensorPayload)
            };
            match self.inner.validate_ai_output(&logits, &sensor) {
                Ok(()) => Ok(true),
                Err(_) => Ok(false),
            }
        }
    }

    #[pymodule]
    fn sodals_industrial(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<SentinelGuardPy>()?;
        Ok(())
    }
}
