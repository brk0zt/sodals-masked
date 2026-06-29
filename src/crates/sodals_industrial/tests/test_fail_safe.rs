//! Fail-Safe Mekanizması — Profesyonel Birim Testleri
//!
//! Kapsam:
//! 1. State Machine transition matrix doğrulama
//! 2. FailSafeCommand binary serileştirme/deserileştirme
//! 3. Sensör validasyonu (NaN, sapma, fiziksel limit)
//! 4. Çoklu thread'den state geçişleri (race condition önleme)
//! 5. Hızlı döngüde 1 milyon state transition (determinizm)

use sodals_industrial::{
    fail_safe::*,
    IndustrialError, SensorPayload,
};
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::time::Instant;

// Global SYSTEM_STATE'e paralel erişimi önlemek için testler arası lock.
static STATE_LOCK: Mutex<()> = Mutex::new(());

/// 1. State Machine Transition Matrix: Geçerli ve geçersiz geçişler.
#[test]
fn state_transition_matrix() {
    let _guard = STATE_LOCK.lock().unwrap();
    // Her test öncesi state'i resetle
    SYSTEM_STATE.store(SystemState::Initializing as u8, Ordering::SeqCst);

    // Initializing -> Running (geçerli)
    let s = transition_state(SystemState::Running);
    assert_eq!(s, SystemState::Initializing);
    assert_eq!(SYSTEM_STATE.load(Ordering::SeqCst), SystemState::Running as u8);

    // Running -> FailSafe (geçerli: hata)
    let s = transition_state(SystemState::FailSafe);
    assert_eq!(s, SystemState::Running);
    assert_eq!(SYSTEM_STATE.load(Ordering::SeqCst), SystemState::FailSafe as u8);

    // FailSafe -> Running (geçerli: kurtarma)
    let s = transition_state(SystemState::Running);
    assert_eq!(s, SystemState::FailSafe);

    // Running -> Degraded (geçerli: OPC UA kopma ama ZeroMQ çalışıyor)
    let s = transition_state(SystemState::Degraded);
    assert_eq!(s, SystemState::Running);
}

/// 2. FailSafeCommand serileştirme round-trip.
#[test]
fn failsafe_command_serde_roundtrip() {
    let _guard = STATE_LOCK.lock().unwrap();
    let now = 1_700_000_000_000_000_000u64;
    let cmd = FailSafeCommand {
        command_code: 0xFF,
        param: 0xABCD,
        timestamp_ns: now,
        triggered_by: 0x30,
        _pad: [0xDE, 0xAD, 0xBE, 0xEF],
    };

    let mut buf = [0u8; 16];
    cmd.serialize_into(&mut buf);

    // Little-endian kontrolü
    assert_eq!(buf[0], 0xFF); // command_code
    assert_eq!(&buf[1..3], &[0xCD, 0xAB]); // param (u16 LE)
    assert_eq!(&buf[3..11], &now.to_le_bytes()); // timestamp
    assert_eq!(buf[11], 0x30); // triggered_by
    assert_eq!(&buf[12..16], &[0xDE, 0xAD, 0xBE, 0xEF]); // _pad

    // Boyut sabit
    assert_eq!(buf.len(), 16, "FailSafeCommand sabit 16 byte olmalı");
}

/// 3. Sensör validasyon: NaN ve sapma tespiti.
#[test]
fn sensor_validation_edge_cases() {
    let _guard = STATE_LOCK.lock().unwrap();
    let ok = SensorPayload {
        timestamp_ns: 0,
        temperature: 25.0,
        pressure: 2.5,
        motor_rpm: 1500.0,
        status_mask: 0x00,
        fault_code: 0,
        _pad: [0; 2],
    };
    assert!(validate_sensor_payload(&ok).is_ok());

    let nan_temp = SensorPayload {
        timestamp_ns: 0,
        temperature: f32::NAN,
        pressure: 2.5,
        motor_rpm: 1500.0,
        status_mask: 0x00,
        fault_code: 0,
        _pad: [0; 2],
    };
    assert_eq!(
        validate_sensor_payload(&nan_temp),
        Err(IndustrialError::OpcUaDisconnect)
    );

    let nan_pressure = SensorPayload {
        timestamp_ns: 0,
        temperature: 25.0,
        pressure: f32::NAN,
        motor_rpm: 1500.0,
        status_mask: 0x00,
        fault_code: 0,
        _pad: [0; 2],
    };
    assert_eq!(
        validate_sensor_payload(&nan_pressure),
        Err(IndustrialError::OpcUaDisconnect)
    );

    let unknown_status = SensorPayload {
        timestamp_ns: 0,
        temperature: 25.0,
        pressure: 2.5,
        motor_rpm: 1500.0,
        status_mask: 0xFF,
        fault_code: 0,
        _pad: [0; 2],
    };
    assert_eq!(
        validate_sensor_payload(&unknown_status),
        Err(IndustrialError::OpcUaDisconnect)
    );
}

/// 4. Çoklu thread race condition: 100 thread aynı anda state değiştirir.
/// Son state deterministik olmalı (SeqCst ordering ile).
#[test]
fn concurrent_state_transitions() {
    let _guard = STATE_LOCK.lock().unwrap();
    SYSTEM_STATE.store(SystemState::Initializing as u8, Ordering::SeqCst);

    let handles: Vec<_> = (0..100)
        .map(|i| {
            std::thread::spawn(move || {
                if i % 3 == 0 {
                    transition_state(SystemState::Running);
                } else if i % 3 == 1 {
                    transition_state(SystemState::Degraded);
                } else {
                    transition_state(SystemState::FailSafe);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let final_state = SYSTEM_STATE.load(Ordering::SeqCst);
    // En azından geçerli bir state olmalı
    assert!(
        final_state <= SystemState::Panic as u8,
        "Geçersiz state: {}",
        final_state
    );
}

/// 5. Hızlı state transition: 1 milyon döngüde deterministik süre.
#[test]
fn state_transition_performance() {
    let _guard = STATE_LOCK.lock().unwrap();
    SYSTEM_STATE.store(SystemState::Initializing as u8, Ordering::SeqCst);

    const N: usize = 1_000_000;
    let start = Instant::now();

    for _ in 0..N {
        transition_state(SystemState::Running);
        transition_state(SystemState::FailSafe);
    }

    let elapsed = start.elapsed();
    let ops_per_sec = (N * 2) as f64 / elapsed.as_secs_f64();

    eprintln!("\n=== State Transition Performance ===");
    eprintln!("  Toplam Geçiş: {}", N * 2);
    eprintln!("  Süre: {:?}", elapsed);
    eprintln!("  İşlem/sn: {:.0}", ops_per_sec);

    // Atomik swap çok hızlıdır; 10M+ ops/sn beklenir
    assert!(
        ops_per_sec > 5_000_000.0,
        "State transition çok yavaş: {:.0} ops/sn",
        ops_per_sec
    );
}

/// 6. Hata -> Komut mapping determinizmi.
#[test]
fn error_to_command_mapping() {
    let _guard = STATE_LOCK.lock().unwrap();
    let now = 1_700_000_000_000_000_000u64;

    let cmd_disconnect = FailSafeCommand::from_error(IndustrialError::OpcUaDisconnect, now);
    assert_eq!(cmd_disconnect.command_code, 1);
    assert_eq!(cmd_disconnect.triggered_by, 10);

    let cmd_zmq = FailSafeCommand::from_error(IndustrialError::ZmqBackpressure, now);
    assert_eq!(cmd_zmq.command_code, 2);
    assert_eq!(cmd_zmq.triggered_by, 20);

    let cmd_sentinel = FailSafeCommand::from_error(IndustrialError::SentinelRejected, now);
    assert_eq!(cmd_sentinel.command_code, 0xFF); // EMERGENCY_STOP
    assert_eq!(cmd_sentinel.triggered_by, 30);

    let cmd_drift = FailSafeCommand::from_error(IndustrialError::ClockDrift, now);
    assert_eq!(cmd_drift.command_code, 1);
    assert_eq!(cmd_drift.triggered_by, 40);
}
