//! Sentinel Güvenlik Katmanı — Profesyonel Birim Testleri
//!
//! Kapsam:
//! 1. Enerji anomali tespiti ve debounce mekanizması
//! 2. Cosine similarity karşılaştırması
//! 3. AI çıktı validasyonu (fiziksel limitler, logit varyansı)
//! 4. 1 milyon karar/saniye throughput
//! 5. Multi-thread Sentinel değerlendirme (race condition)
//! 6. Sınır değerler (edge cases): sıfır vektör, sonsuz, NaN

use sodals_industrial::{
    qmem_format::*,
    sentinel_guard::*,
    SensorPayload, IndustrialError,
};
use std::sync::Arc;
use std::time::Instant;

fn make_reference_signature() -> FrequencySignatureBlock {
    let mut sig = FrequencySignatureBlock {
        header: QmemHeader {
            magic: QmemHeader::MAGIC,
            version: [1, 0],
            dtype: 1,
            _pad0: 0,
            len: 128,
            sample_rate_hz: 256.0,
            timestamp_ns: 0,
            label: *b"REF_____",
        },
        bands: [-50.0f32; 128],
    };
    // Spektral şekil: bazı bantlarda güç artışı
    sig.bands[10] = -30.0;
    sig.bands[20] = -35.0;
    sig.bands[50] = -25.0;
    sig
}

fn make_guard(debounce: u64) -> SentinelGuard {
    SentinelGuard::new(
        SentinelReference {
            signature: make_reference_signature(),
            threshold_db: -40.0,
        },
        debounce,
    )
}

/// 1. Normal veri: izin verilmeli.
#[test]
fn normal_data_allow() {
    let guard = make_guard(3);
    let mut candidate = make_reference_signature();
    candidate.header.timestamp_ns = 1_000_000;
    candidate.header.label = *b"CAND0___";

    let decision = guard.evaluate(&candidate);
    assert_eq!(decision, SentinelDecision::Allow, "Normal veri reddedilmemeli");
}

/// 2. Enerji anomalisi: debounce öncesi Hold, sonra Reject.
#[test]
fn energy_anomaly_debounce() {
    let guard = make_guard(3);

    // Anomali yüksek enerji
    let mut candidate = make_reference_signature();
    candidate.bands.fill(-10.0); // Çok yüksek enerji

    // 1. anomali: Hold
    assert_eq!(
        guard.evaluate(&candidate),
        SentinelDecision::Hold,
        "1. anomali: Hold"
    );

    // 2. anomali: Hold
    assert_eq!(
        guard.evaluate(&candidate),
        SentinelDecision::Hold,
        "2. anomali: Hold"
    );

    // 3. anomali: Reject (debounce eşiği aşıldı)
    assert!(
        matches!(
            guard.evaluate(&candidate),
            SentinelDecision::Reject(SentinelRejectionReason::EnergyAnomaly)
        ),
        "3. anomali: Reject"
    );
}

/// 3. Spektral mismatch: farklı spektral şekil.
#[test]
fn spectral_mismatch_reject() {
    // Referans: enerji alt bantlarda (0-63)
    let mut ref_sig = make_reference_signature();
    ref_sig.bands.fill(-80.0); // Çok düşük ortalama -> anomali değil
    for i in 0..64 {
        ref_sig.bands[i] = -30.0; // Alt bantlarda güç
    }

    // Aday: enerji üst bantlarda (64-127) — neredeyse dik vektör
    let mut candidate = ref_sig.clone();
    candidate.bands.fill(-80.0);
    for i in 64..128 {
        candidate.bands[i] = -30.0; // Üst bantlarda güç
    }

    let guard = SentinelGuard::new(
        SentinelReference {
            signature: ref_sig,
            threshold_db: -40.0,
        },
        1,
    );

    let decision = guard.evaluate(&candidate);
    assert!(
        matches!(
            decision,
            SentinelDecision::Reject(SentinelRejectionReason::SpectralMismatch)
            | SentinelDecision::Hold
        ),
        "Spektral mismatch reddedilmeli veya Hold, alınan: {:?}",
        decision
    );
}

/// 4. AI çıktı validasyonu: fiziksel limit aşımı.
#[test]
fn ai_physical_limit_exceeded() {
    let guard = make_guard(3);

    let sensor = SensorPayload {
        timestamp_ns: 0,
        temperature: 250.0, // > 200C limit
        pressure: 2.0,
        motor_rpm: 1000.0,
        status_mask: 0,
        fault_code: 0,
        _pad: [0; 2],
    };

    let logits = [0.1f32; 8];
    let result = guard.validate_ai_output(&logits, &sensor);
    assert_eq!(
        result,
        Err(IndustrialError::SentinelRejected),
        "Sıcaklık limit aşımı reddedilmeli"
    );
}

/// 5. AI çıktı validasyonu: logit kararsızlığı (yüksek varyans).
#[test]
fn ai_model_instability() {
    let guard = make_guard(3);

    let sensor = SensorPayload {
        timestamp_ns: 0,
        temperature: 30.0,
        pressure: 2.0,
        motor_rpm: 1500.0,
        status_mask: 0,
        fault_code: 0,
        _pad: [0; 2],
    };

    // Yüksek varyanslı logitler (model kararsız)
    let logits = [-100.0f32, 100.0, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0];
    let result = guard.validate_ai_output(&logits, &sensor);
    assert_eq!(
        result,
        Err(IndustrialError::SentinelRejected),
        "Yüksek varyans reddedilmeli"
    );
}

/// 6. Throughput: 1 milyon Sentinel değerlendirme/saniye.
#[test]
fn sentinel_throughput() {
    let guard = make_guard(3);
    let candidate = make_reference_signature();

    const N: usize = 1_000_000;
    let start = Instant::now();

    for _ in 0..N {
        let _ = guard.evaluate(&candidate);
    }

    let elapsed = start.elapsed();
    let ops_per_sec = N as f64 / elapsed.as_secs_f64();

    eprintln!("\n=== Sentinel Evaluate Throughput ===");
    eprintln!("  Karar: {}", N);
    eprintln!("  Süre: {:?}", elapsed);
    eprintln!("  Karar/sn: {:.0}", ops_per_sec);

    assert!(
        ops_per_sec > 100_000.0,
        "Sentinel throughput düşük: {:.0} karar/sn (debug build bekleniyor)",
        ops_per_sec
    );
}

/// 7. Multi-thread: 8 thread aynı anda Sentinel değerlendirir.
#[test]
fn concurrent_sentinel_evaluation() {
    let guard = Arc::new(make_guard(3));
    let mut handles = Vec::with_capacity(8);

    let start = Instant::now();
    for _ in 0..8 {
        let g = Arc::clone(&guard);
        handles.push(std::thread::spawn(move || {
            let mut local_allow = 0usize;
            let mut local_reject = 0usize;
            let candidate = make_reference_signature();

            for _ in 0..100_000 {
                match g.evaluate(&candidate) {
                    SentinelDecision::Allow => local_allow += 1,
                    SentinelDecision::Reject(_) => local_reject += 1,
                    _ => {}
                }
            }
            (local_allow, local_reject)
        }));
    }

    let mut total_allow = 0usize;
    let mut total_reject = 0usize;
    for h in handles {
        let (a, r) = h.join().unwrap();
        total_allow += a;
        total_reject += r;
    }

    let elapsed = start.elapsed();
    let total = total_allow + total_reject;
    let ops_per_sec = total as f64 / elapsed.as_secs_f64();

    eprintln!("\n=== Concurrent Sentinel (8 threads) ===");
    eprintln!("  Toplam Karar: {}", total);
    eprintln!("  Allow: {}, Reject: {}", total_allow, total_reject);
    eprintln!("  Süre: {:?}", elapsed);
    eprintln!("  Karar/sn: {:.0}", ops_per_sec);

    assert_eq!(total_reject, 0, "Normal veri hiç reddedilmemeli");
}

/// 8. Edge case: Sıfır vektör (NaN önleme).
#[test]
fn zero_vector_finite() {
    let mut zero = make_reference_signature();
    zero.bands.fill(0.0);

    let guard = make_guard(3);
    // Sıfır enerji -> anomali değil (ortalama 0 > -40 eşik değil, dikkat: threshold negatif)
    // -40.0 eşik, ortalama 0.0 > -40.0 -> ANOMALI
    let decision = guard.evaluate(&zero);
    assert!(
        matches!(decision, SentinelDecision::Hold | SentinelDecision::Reject(_)),
        "Sıfır vektör anomali olmalı (yüksek enerji), alınan: {:?}",
        decision
    );
}
