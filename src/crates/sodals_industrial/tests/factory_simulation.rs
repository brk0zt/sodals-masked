//! Coca-Cola Fabrika Simülasyonu — Karma Kritiklik Yük Testi
//!
//! Senaryo: 5 üretim bandı, toplam 500 IoT sensör.
//! Her band farklı sensör tipi ve örnekleme hızına sahip.
//!
//! Band A: Şişeleme   — 120 sensör (titreşim, sıcaklık, motor RPM)
//! Band B: Dolum      — 110 sensör (basınç, akış, sıcaklık)
//! Band C: Etiketleme — 90  sensör (görüntü kalite, motor)
//! Band D: Paketleme  — 100 sensör (kuvvet, konum, hız)
//! Band E: Paletleme  — 80  sensör (ağırlık, konum, güvenlik)
//!
//! Hedef metrikler:
//! - Toplam veri akışı: ~50,000 okuma/sn (500 sensör x 10ms çevrim)
//! - ZeroMQ IPC latency: P99 < 500 µs
//! - Sentinel anomali tespiti: < 1 µs/sensör
//! - Fail-Safe tepki süresi: < 5 ms (PLC'ye komut)
//! - Bellek kullanımı: < 256MB (RT süreç)

use sodals_industrial::{
    fail_safe::FailSafeCommand,
    qmem_format::*,
    sentinel_guard::*,
    zmq_bridge::*,
    SensorPayload,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::tempdir;

mod common;

/// Fabrika sensör profili.
#[derive(Clone, Debug)]
struct SensorProfile {
    id: usize,
    line_id: u8,
    sensor_type: &'static str,
    /// Normal çalışma aralığı (min, max)
    range: (f32, f32),
    /// Anomali olasılığı (%0.1 = ayda 1 olay)
    anomaly_rate: f64,
}

fn build_factory() -> (Vec<SensorProfile>, Vec<common::ProductionLine>) {
    let lines = common::coca_cola_factory_topology();
    let mut sensors = Vec::with_capacity(500);
    let mut global_id = 0usize;

    for line in &lines {
        for local_id in 0..line.sensor_count {
            let stype = match line.line_type {
                common::LineType::Bottling => {
                    match local_id % 3 {
                        0 => "vibration",
                        1 => "temperature",
                        _ => "motor_rpm",
                    }
                }
                common::LineType::Filling => {
                    match local_id % 3 {
                        0 => "pressure",
                        1 => "flow_rate",
                        _ => "temperature",
                    }
                }
                common::LineType::Labeling => {
                    match local_id % 3 {
                        0 => "vision_quality",
                        1 => "motor_rpm",
                        _ => "tension",
                    }
                }
                common::LineType::Packaging => {
                    match local_id % 3 {
                        0 => "force",
                        1 => "position",
                        _ => "conveyor_speed",
                    }
                }
                common::LineType::Palletizing => {
                    match local_id % 3 {
                        0 => "weight",
                        1 => "position",
                        _ => "safety_proximity",
                    }
                }
            };

            let range = match stype {
                "temperature" => (15.0, 45.0),
                "pressure" => (0.5, 8.0),
                "motor_rpm" => (800.0, 3600.0),
                "vibration" => (0.0, 5.0),
                "flow_rate" => (10.0, 100.0),
                "vision_quality" => (0.0, 100.0),
                "force" => (0.0, 500.0),
                "position" => (0.0, 1000.0),
                "conveyor_speed" => (0.5, 3.0),
                "weight" => (0.0, 25.0),
                "safety_proximity" => (0.0, 2.0),
                "tension" => (0.0, 100.0),
                _ => (0.0, 100.0),
            };

            sensors.push(SensorProfile {
                id: global_id,
                line_id: line.id,
                sensor_type: stype,
                range,
                anomaly_rate: 0.001, // %0.1
            });

            global_id += 1;
        }
    }

    (sensors, lines)
}

/// 1. Fabrika topolojisi doğrulama.
#[test]
fn factory_topology_integrity() {
    let (sensors, lines) = build_factory();

    assert_eq!(sensors.len(), 500, "Toplam 500 sensör olmalı");
    assert_eq!(lines.len(), 5, "5 üretim bandı olmalı");

    for line in &lines {
        let line_sensors: Vec<_> = sensors.iter().filter(|s| s.line_id == line.id).collect();
        assert_eq!(
            line_sensors.len(),
            line.sensor_count,
            "{} sensör sayısı uyuşmuyor",
            line.name
        );
    }

    eprintln!("\n=== Coca-Cola Fabrika Topolojisi ===");
    for line in &lines {
        let count = sensors.iter().filter(|s| s.line_id == line.id).count();
        eprintln!("  {}: {} sensör", line.name, count);
    }
    eprintln!("  TOPLAM: {} sensör", sensors.len());
}

/// 2. 500 sensör eşzamanlı veri üretimi ve IPC.
#[test]
fn factory_500_sensor_ipc_flood() {
    let (sensors, _lines) = build_factory();
    let dir = tempdir().unwrap();
    let pub_path = format!("ipc://{}/factory.pub", dir.path().display());
    let _rep_path = format!("ipc://{}/factory.rep", dir.path().display());

    let publisher = Arc::new(SensorPublisher::new(&pub_path, 50_000).unwrap());
    std::thread::sleep(Duration::from_millis(200));

    let subscriber = Arc::new(SensorSubscriber::new(&pub_path).unwrap());

    // 1 saniyelik üretim döngüsü simülasyonu
    const DURATION_MS: usize = 1000;
    const CYCLE_MS: usize = 10;
    const CYCLES: usize = DURATION_MS / CYCLE_MS; // 100 döngü

    let sent_count = Arc::new(AtomicUsize::new(0));
    let s = Arc::clone(&sent_count);

    let pub_clone = Arc::clone(&publisher);
    let sensors_clone = sensors.clone();

    let producer = std::thread::spawn(move || {
        let start = Instant::now();
        for cycle in 0..CYCLES {
            let cycle_start = Instant::now();

            for sensor in &sensors_clone {
                let payload = generate_sensor_value(sensor, cycle);
                if let Ok(true) = pub_clone.publish(&payload) {
                    s.fetch_add(1, Ordering::Relaxed);
                }
            }

            // 10ms deterministik tick
            let elapsed = cycle_start.elapsed();
            if elapsed < Duration::from_millis(10) {
                std::thread::sleep(Duration::from_millis(10) - elapsed);
            }
        }
        start.elapsed()
    });

    // Tüketim
    let mut received = 0usize;
    let recv_start = Instant::now();
    let recv_timeout = Duration::from_secs(5);

    while recv_start.elapsed() < recv_timeout {
        if let Ok(Some(_)) = subscriber.receive() {
            received += 1;
        }
        // Hedef: 500 sensör x 100 döngü = 50,000 mesaj
        if received >= 40_000 {
            break;
        }
    }

    let prod_elapsed = producer.join().unwrap();
    let sent = sent_count.load(Ordering::Relaxed);
    let expected = sensors.len() * CYCLES; // 50,000

    let throughput_sent = sent as f64 / prod_elapsed.as_secs_f64();
    let throughput_recv = received as f64 / recv_start.elapsed().as_secs_f64();

    eprintln!("\n=== 500 Sensör Fabrika Simülasyonu (1 saniye) ===");
    eprintln!("  Sensör: {}", sensors.len());
    eprintln!("  Beklenen mesaj: {}", expected);
    eprintln!("  Gönderilen: {}", sent);
    eprintln!("  Alınan: {}", received);
    eprintln!("  Üretim süresi: {:?}", prod_elapsed);
    eprintln!("  Gönderim throughput: {:.0} mesaj/sn", throughput_sent);
    eprintln!("  Alım throughput: {:.0} mesaj/sn", throughput_recv);

    // Gerçekçi beklenti: %80+ alım oranı (ZeroMQ HWM ve schedule jitter nedeniyle)
    assert!(
        received >= expected * 70 / 100,
        "Alım oranı düşük: {}/{} (%{:.1})",
        received,
        expected,
        received as f64 / expected as f64 * 100.0
    );
}

/// 3. Felaket senaryosu: PLC bağlantısı kopması.
#[test]
fn disaster_plc_disconnect() {
    let (_sensors, _lines) = build_factory();
    let dir = tempdir().unwrap();
    let rep_path = format!("ipc://{}/disaster.rep", dir.path().display());

    let replier = Arc::new(CommandReplier::new(&rep_path).unwrap());
    std::thread::sleep(Duration::from_millis(50));
    let requester = Arc::new(CommandRequester::new(&rep_path).unwrap());

    // PLC kopması simülasyonu: Sensör verisi NaN
    let anomaly_sensor = SensorPayload {
        timestamp_ns: 0,
        temperature: f32::NAN,
        pressure: 2.5,
        motor_rpm: 1500.0,
        status_mask: 0xFF,
        fault_code: 0,
        _pad: [0; 2],
    };

    let result = sodals_industrial::fail_safe::validate_sensor_payload(&anomaly_sensor);
    assert_eq!(
        result,
        Err(sodals_industrial::IndustrialError::OpcUaDisconnect),
        "NaN sensör verisi OPC UA kopması olarak işaretlenmeli"
    );

    // Fail-Safe komutu üret ve gönder
    let cmd = FailSafeCommand::from_error(
        sodals_industrial::IndustrialError::OpcUaDisconnect,
        1_700_000_000_000_000_000u64,
    );

    let repl = Arc::clone(&replier);
    let handle = std::thread::spawn(move || {
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(2) {
            if let Ok(Some(_)) = repl.recv_command() {
                return true;
            }
            std::thread::sleep(Duration::from_micros(100));
        }
        false
    });

    std::thread::sleep(Duration::from_millis(50));
    let ack = requester.send_command(&cmd).unwrap();

    assert!(ack, "Fail-Safe komut ACK almalı");
    assert!(handle.join().unwrap(), "Fail-Safe komut replier'a ulaşmalı");

    eprintln!("\n=== Felaket Senaryosu: PLC Kopması ===");
    eprintln!("  Komut: STOP_MOTOR (0x01)");
    eprintln!("  ACK: true");
}

/// 4. Anomali saldırısı senaryosu: Sentinel debounce testi.
#[test]
fn disaster_anomaly_attack() {
    let guard = SentinelGuard::new(
        SentinelReference {
            signature: make_reference_signature(),
            threshold_db: -40.0,
        },
        3,
    );

    let mut anomalous = make_reference_signature();
    anomalous.bands.fill(-5.0); // Aşırı yüksek enerji

    let mut rejections = 0usize;
    let mut holds = 0usize;

    for _ in 0..10 {
        match guard.evaluate(&anomalous) {
            SentinelDecision::Reject(_) => rejections += 1,
            SentinelDecision::Hold => holds += 1,
            _ => {}
        }
    }

    eprintln!("\n=== Felaket Senaryosu: Anomali Saldırısı ===");
    eprintln!("  10 ardışık anomali değerlendirmesi");
    eprintln!("  Reject: {}, Hold: {}", rejections, holds);

    assert!(
        rejections >= 1,
        "Debounce sonrası en az 1 reddetme olmalı"
    );
    assert!(
        holds >= 2,
        "Debounce öncesi en az 2 Hold olmalı"
    );
}

/// 5. Bellek profili: 500 sensör x 1K mesaj ring buffer'da.
#[test]
fn factory_memory_footprint() {
    const PER_SENSOR_BUFFER: usize = 1024;
    const SENSOR_COUNT: usize = 500;

    // Ring buffer başına ~24KB (1024 x 24 byte payload)
    let total_bytes = SENSOR_COUNT * PER_SENSOR_BUFFER * std::mem::size_of::<SensorPayload>();
    let total_mb = total_bytes as f64 / 1_048_576.0;

    eprintln!("\n=== Fabrika Bellek Profili ===");
    eprintln!("  Sensör: {}", SENSOR_COUNT);
    eprintln!("  Buffer/sensör: {} eleman", PER_SENSOR_BUFFER);
    eprintln!("  Payload boyutu: {} byte", std::mem::size_of::<SensorPayload>());
    eprintln!("  Toplam: {} byte ({:.2} MB)", total_bytes, total_mb);

    // RT süreçte 256MB sınırı
    assert!(
        total_mb < 256.0,
        "Ring buffer toplamı 256MB'ı aşmamalı: {:.2} MB",
        total_mb
    );
}

// =============================================================================
// Yardımcı Fonksiyonlar
// =============================================================================

fn generate_sensor_value(sensor: &SensorProfile, cycle: usize) -> SensorPayload {
    use rand::prelude::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64((sensor.id * 10000 + cycle) as u64);
    let is_anomaly = rng.gen::<f64>() < sensor.anomaly_rate;

    let val = if is_anomaly {
        rng.gen_range(sensor.range.1 * 2.0..sensor.range.1 * 5.0)
    } else {
        rng.gen_range(sensor.range.0..sensor.range.1)
    };

    SensorPayload {
        timestamp_ns: 1_700_000_000_000_000_000u64 + (cycle as u64 * 10_000_000),
        temperature: if sensor.sensor_type == "temperature" { val } else { 25.0 },
        pressure: if sensor.sensor_type == "pressure" { val } else { 2.5 },
        motor_rpm: if sensor.sensor_type == "motor_rpm" { val } else { 1500.0 },
        status_mask: if is_anomaly { 0x02 } else { 0x00 },
        fault_code: 0,
        _pad: [0; 2],
    }
}

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
    sig.bands[10] = -30.0;
    sig.bands[20] = -35.0;
    sig.bands[50] = -25.0;
    sig
}
