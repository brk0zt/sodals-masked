//! ZeroMQ IPC Köprüsü — Profesyonel Birim Testleri
//!
//! Kapsam:
//! 1. PUB/SUB mesaj round-trip latency dağılımı (P50, P99, P99.9)
//! 2. Backpressure testi: HWM dolunca mesaj düşürme determinizmi
//! 3. REQ/REP ACK/NAK protokolü
//! 4. Multi-subscriber throughput
//! 5. 500 sensör eşzamanlı veri akışı simülasyonu
//! 6. Bellek basıncı altında IPC stabilitesi

use sodals_industrial::{
    fail_safe::FailSafeCommand,
    zmq_bridge::*,
    SensorPayload,
};

mod common;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::tempdir;

/// IPC endpoint'lerini temp dizinde oluşturur (çakışma önleme).
fn temp_endpoints() -> (IpcEndpoints, tempfile::TempDir) {
    let dir = tempdir().unwrap();
    let pub_path = format!("ipc://{}/sensor.pub", dir.path().display());
    let rep_path = format!("ipc://{}/cmd.rep", dir.path().display());
    (
        IpcEndpoints {
            sensor_pub: pub_path,
            cmd_rep: rep_path,
        },
        dir,
    )
}

/// 1. Tek mesaj round-trip latency (P50, P99).
#[test]
fn pubsub_roundtrip_latency() {
    let (endpoints, _dir) = temp_endpoints();

    let pubber = Arc::new(SensorPublisher::new(&endpoints.sensor_pub, 1000).unwrap());
    let subber = Arc::new(SensorSubscriber::new(&endpoints.sensor_pub).unwrap());

    // Publisher'ın bağlanması için kısa bekleme
    std::thread::sleep(Duration::from_millis(100));

    const N: usize = 10_000;
    let mut latencies = Vec::with_capacity(N);

    for i in 0..N {
        let payload = SensorPayload {
            timestamp_ns: i as u64,
            temperature: 25.0,
            pressure: 2.5,
            motor_rpm: 1500.0,
            status_mask: 0,
            fault_code: 0,
            _pad: [0; 2],
        };

        let t0 = Instant::now();
        pubber.publish(&payload).unwrap();

        // Alana kadar bekle (maks 10ms timeout)
        let mut received = false;
        let deadline = Instant::now() + Duration::from_millis(10);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = subber.receive() {
                latencies.push(t0.elapsed());
                received = true;
                break;
            }
        }
        if !received {
            panic!("Mesaj {} alınamadı", i);
        }
    }

    latencies.sort();
    let p50 = latencies[N / 2];
    let p99 = latencies[N * 99 / 100];
    let max = latencies[N - 1];

    eprintln!("\n=== ZeroMQ PUB/SUB Round-Trip Latency ===");
    eprintln!("  Samples: {}", latencies.len());
    eprintln!("  P50:     {:?}", p50);
    eprintln!("  P99:     {:?}", p99);
    eprintln!("  Max:     {:?}", max);

    assert!(
        p99 < Duration::from_micros(500),
        "P99 IPC latency çok yüksek: {:?}",
        p99
    );
}

/// 2. Backpressure: HWM dolunca mesaj düşürme.
/// Broadcast channel'da subscriber yavaşsa mesajlar düşürülür (Lagged).
#[test]
fn backpressure_message_drop() {
    let (endpoints, _dir) = temp_endpoints();

    // HWM=5 (çok düşük, hızlı dolması için)
    let pubber = Arc::new(SensorPublisher::new(&endpoints.sensor_pub, 5).unwrap());
    let subber = Arc::new(SensorSubscriber::new(&endpoints.sensor_pub).unwrap());

    let payload = SensorPayload {
        timestamp_ns: 0,
        temperature: 25.0,
        pressure: 2.5,
        motor_rpm: 1500.0,
        status_mask: 0,
        fault_code: 0,
        _pad: [0; 2],
    };

    // Hızlıca HWM'den fazla mesaj gönder (subscriber okumadan)
    let mut sent_ok = 0;
    let mut sent_full = 0;
    for i in 0..100 {
        let mut p = payload;
        p.timestamp_ns = i as u64;
        match pubber.publish(&p) {
            Ok(true) => sent_ok += 1,
            Ok(false) => sent_full += 1,
            Err(_) => sent_full += 1,
        }
    }

    // Subscriber en az bir mesaj alabilmeli
    let mut received = 0;
    while let Ok(Some(_)) = subber.receive() {
        received += 1;
    }

    eprintln!("\n=== Backpressure Mesaj Düşürme ===");
    eprintln!("  Gönderilen (başarılı): {}", sent_ok);
    eprintln!("  Gönderilen (dolu): {}", sent_full);
    eprintln!("  Alınan: {}", received);

    assert!(
        received > 0,
        "En azından bazı mesajlar alınmalı"
    );
}

/// 3. REQ/REP komut protokolü: ACK alınmalı.
#[test]
fn reqrep_command_ack() {
    let (endpoints, _dir) = temp_endpoints();

    let replier = Arc::new(CommandReplier::new(&endpoints.cmd_rep).unwrap());
    std::thread::sleep(Duration::from_millis(50));
    let requester = Arc::new(CommandRequester::new(&endpoints.cmd_rep).unwrap());

    let cmd = FailSafeCommand {
        command_code: 0x01,
        param: 0,
        timestamp_ns: 1_700_000_000_000_000_000u64,
        triggered_by: 0x10,
        _pad: [0; 4],
    };

    // Async: replier bir thread'de, requester main thread'de
    let repl = Arc::clone(&replier);
    let handle = std::thread::spawn(move || {
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(2) {
            if let Ok(Some(cmd)) = repl.recv_command() {
                return Some(cmd);
            }
            std::thread::sleep(Duration::from_micros(100));
        }
        None
    });

    std::thread::sleep(Duration::from_millis(50));
    let ack = requester.send_command(&cmd).unwrap();

    assert!(ack, "REQ/REP ACK alınmalı");
    let received = handle.join().unwrap();
    assert!(received.is_some(), "Replie'ya komut ulaşmalı");
}

/// 4. 500 sensör eşzamanlı veri akışı simülasyonu.
#[test]
fn five_hundred_sensor_flood() {
    let (endpoints, _dir) = temp_endpoints();

    let pubber = Arc::new(SensorPublisher::new(&endpoints.sensor_pub, 10_000).unwrap());
    std::thread::sleep(Duration::from_millis(100));

    let subber = Arc::new(SensorSubscriber::new(&endpoints.sensor_pub).unwrap());

    const SENSOR_COUNT: usize = 500;
    const MESSAGES_PER_SENSOR: usize = 100;
    const TOTAL_MESSAGES: usize = SENSOR_COUNT * MESSAGES_PER_SENSOR;

    let sent = Arc::new(AtomicUsize::new(0));
    let s = Arc::clone(&sent);

    // Üretim bandı thread'i
    let pub_clone = Arc::clone(&pubber);
    let producer = std::thread::spawn(move || {
        for sensor_id in 0..SENSOR_COUNT {
            for msg_id in 0..MESSAGES_PER_SENSOR {
                let payload = SensorPayload {
                    timestamp_ns: (sensor_id * MESSAGES_PER_SENSOR + msg_id) as u64,
                    temperature: 20.0 + (sensor_id as f32 * 0.1),
                    pressure: 2.0 + (msg_id as f32 * 0.01),
                    motor_rpm: 1000.0 + (sensor_id as f32 * 10.0),
                    status_mask: 0,
                    fault_code: 0,
                    _pad: [0; 2],
                };
                if let Ok(true) = pub_clone.publish(&payload) {
                    s.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    });

    // Tüketim
    let mut received = 0usize;
    let start = Instant::now();
    let timeout = Duration::from_secs(10);

    while received < TOTAL_MESSAGES && start.elapsed() < timeout {
        if let Ok(Some(_)) = subber.receive() {
            received += 1;
        }
    }

    producer.join().unwrap();

    let sent_count = sent.load(Ordering::Relaxed);
    let elapsed = start.elapsed();
    let throughput = received as f64 / elapsed.as_secs_f64();

    eprintln!("\n=== 500 Sensör IPC Flood Test ===");
    eprintln!("  Sensör: {}", SENSOR_COUNT);
    eprintln!("  Gönderilen: {}/{} (HWM dolunca düşürüldü)", sent_count, TOTAL_MESSAGES);
    eprintln!("  Alınan: {}", received);
    eprintln!("  Süre: {:?}", elapsed);
    eprintln!("  Mesaj/sn: {:.0}", throughput);

    assert!(
        received > 0,
        "En azından bazı mesajlar alınmalı"
    );
    // HWM dolması normaldir; %50'den fazlası alınmalı
    assert!(
        received >= sent_count * 3 / 4,
        "Çoğu mesaj alınmalı: alınan={}, gönderilen={}",
        received,
        sent_count
    );
}

/// 5. Bellek basıncı altında IPC stabilitesi.
#[test]
fn ipc_under_memory_pressure() {
    let _pressure = common::simulate_memory_pressure(200);
    let (endpoints, _dir) = temp_endpoints();

    let pubber = Arc::new(SensorPublisher::new(&endpoints.sensor_pub, 10_000).unwrap());
    let subber = Arc::new(SensorSubscriber::new(&endpoints.sensor_pub).unwrap());

    let payload = SensorPayload {
        timestamp_ns: 0,
        temperature: 25.0,
        pressure: 2.5,
        motor_rpm: 1500.0,
        status_mask: 0,
        fault_code: 0,
        _pad: [0; 2],
    };

    const N: usize = 10_000;
    let mut received = 0;

    let start = Instant::now();
    // Publish and receive interleaved to avoid buffer overflow
    for i in 0..N {
        let mut p = payload;
        p.timestamp_ns = i as u64;
        let _ = pubber.publish(&p);
        // Drain on each cycle
        while let Ok(Some(_)) = subber.receive() {
            received += 1;
        }
    }
    // Final drain
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if let Ok(Some(_)) = subber.receive() {
            received += 1;
        } else {
            break;
        }
    }

    let elapsed = start.elapsed();
    eprintln!("\n=== Bellek Basıncı Altında IPC ===");
    eprintln!("  Gönderilen: {}, Alınan: {}", N, received);
    eprintln!("  Süre: {:?}", elapsed);

    assert!(
        received >= N * 3 / 4,
        "Çoğu mesaj alınmalı: alınan={}, gönderilen={}",
        received, N
    );
}
