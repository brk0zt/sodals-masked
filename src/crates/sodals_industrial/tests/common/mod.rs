#![allow(dead_code)]
//! Profesyonel Test Yardımcıları ve Fabrika Fixture'ları
//!
//! Bu modül, tüm test paketleri için ortak:
//! - Mock veri üreticileri (deterministik seed ile)
//! - Performans ölçüm makroları
//! - Fabrika sensör topolojisi tanımları

use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use sodals_industrial::SensorPayload;
use std::time::{Duration, Instant};

/// Sabit seed ile deterministik mock üretici.
/// Tüm testler aynı seed ile çalışırsa sonuçlar tekrarlanabilir.
pub const MOCK_SEED: u64 = 0xDEAD_BEEF_CAFE_BABE;

/// Fabrika üretim bandı tanımı.
#[derive(Clone, Debug)]
pub struct ProductionLine {
    pub id: u8,
    pub name: &'static str,
    pub sensor_count: usize,
    pub line_type: LineType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineType {
    Bottling,
    Filling,
    Labeling,
    Packaging,
    Palletizing,
}

/// Coca-Cola benzeri 5 bantlı fabrika topolojisi.
pub fn coca_cola_factory_topology() -> Vec<ProductionLine> {
    vec![
        ProductionLine {
            id: 1,
            name: "Band-A: Şişeleme (Bottling)",
            sensor_count: 120,
            line_type: LineType::Bottling,
        },
        ProductionLine {
            id: 2,
            name: "Band-B: Dolum (Filling)",
            sensor_count: 110,
            line_type: LineType::Filling,
        },
        ProductionLine {
            id: 3,
            name: "Band-C: Etiketleme (Labeling)",
            sensor_count: 90,
            line_type: LineType::Labeling,
        },
        ProductionLine {
            id: 4,
            name: "Band-D: Paketleme (Packaging)",
            sensor_count: 100,
            line_type: LineType::Packaging,
        },
        ProductionLine {
            id: 5,
            name: "Band-E: Paletleme (Palletizing)",
            sensor_count: 80,
            line_type: LineType::Palletizing,
        },
    ]
}

/// Toplam sensör sayısını doğrular (500 adet hedefi).
#[test]
fn topology_has_500_sensors() {
    let lines = coca_cola_factory_topology();
    let total: usize = lines.iter().map(|l| l.sensor_count).sum();
    assert_eq!(total, 500, "Fabrika topolojisi 500 sensör bekleniyor");
}

/// Deterministik sensör verisi üretici.
pub fn generate_sensor_batch(
    count: usize,
    line_id: u8,
    anomaly_probability: f64,
) -> Vec<SensorPayload> {
    let mut rng = StdRng::seed_from_u64(MOCK_SEED + line_id as u64);
    (0..count)
        .map(|i| {
            let is_anomaly = rng.gen::<f64>() < anomaly_probability;
            SensorPayload {
                timestamp_ns: 1_700_000_000_000_000_000u64 + (i as u64 * 10_000_000),
                temperature: if is_anomaly {
                    rng.gen_range(150.0f32..300.0)
                } else {
                    rng.gen_range(15.0f32..45.0)
                },
                pressure: if is_anomaly {
                    rng.gen_range(30.0f32..80.0)
                } else {
                    rng.gen_range(0.5f32..8.0)
                },
                motor_rpm: rng.gen_range(800.0f32..3600.0),
                status_mask: if is_anomaly { 0x02 } else { 0x00 },
                fault_code: 0,
                _pad: [0; 2],
            }
        })
        .collect()
}

/// Latency istatistikleri (percentile hesaplama).
pub struct LatencyStats {
    pub samples: Vec<Duration>,
    pub min: Duration,
    pub max: Duration,
    pub mean: Duration,
    pub p50: Duration,
    pub p99: Duration,
    pub p999: Duration,
    pub std_dev_ns: f64,
}

impl LatencyStats {
    pub fn from_samples(mut samples: Vec<Duration>) -> Self {
        if samples.is_empty() {
            return Self {
                samples: vec![],
                min: Duration::ZERO,
                max: Duration::ZERO,
                mean: Duration::ZERO,
                p50: Duration::ZERO,
                p99: Duration::ZERO,
                p999: Duration::ZERO,
                std_dev_ns: 0.0,
            };
        }
        samples.sort();
        let n = samples.len();
        let total_ns: u128 = samples.iter().map(|d| d.as_nanos()).sum();
        let mean_ns = total_ns / n as u128;

        let mean_f = mean_ns as f64;
        let variance = samples
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_f;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;

        Self {
            min: samples[0],
            max: samples[n - 1],
            mean: Duration::from_nanos(mean_ns as u64),
            p50: samples[n * 50 / 100],
            p99: samples[n * 99 / 100],
            p999: samples[(n * 999).min(n * 1000 - 1) / 1000],
            std_dev_ns: variance.sqrt(),
            samples,
        }
    }

    pub fn report(&self, name: &str) {
        eprintln!("\n=== {} ===", name);
        eprintln!("  Samples : {}", self.samples.len());
        eprintln!("  Min     : {:?}", self.min);
        eprintln!("  Mean    : {:?}", self.mean);
        eprintln!("  P50     : {:?}", self.p50);
        eprintln!("  P99     : {:?}", self.p99);
        eprintln!("  P99.9   : {:?}", self.p999);
        eprintln!("  Max     : {:?}", self.max);
        eprintln!("  StdDev  : {:.2} ns", self.std_dev_ns);
    }
}

/// Bir closure'u `iterations` kere çalıştırır ve her iterasyonun latency'sini kaydeder.
pub fn bench_latency<F>(name: &str, iterations: usize, f: F)
where
    F: Fn(),
{
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        samples.push(start.elapsed());
    }
    let stats = LatencyStats::from_samples(samples);
    stats.report(name);
}

/// Bellek basıncı (memory pressure) simülasyonu: heap'te büyük bir blok ayrır,
/// test süresince cache thrashing yaratır.
pub fn simulate_memory_pressure(mb: usize) -> Vec<u8> {
    let mut v = vec![0u8; mb * 1024 * 1024];
    let mut rng = StdRng::seed_from_u64(MOCK_SEED);
    for i in (0..v.len()).step_by(4096) {
        v[i] = rng.gen();
    }
    v
}
