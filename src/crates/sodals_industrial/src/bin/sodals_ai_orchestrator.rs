//! SODALS Non-RT AI Orchestrator — Binary Giriş Noktası
//!
//! Görevleri:
//! 1. ZeroMQ SUB üzerinden sensör verisi al.
//! 2. Qwen 8B modelini çalıştır (Python/PyO3 çağrısı).
//! 3. SentinelGuard ile çıktıyı doğrula.
//! 4. Geçerse REQ/REP üzerinden PLC'ye komut gönder.

use sodals_industrial::{
    fail_safe::{FailSafeCommand, transition_state, SystemState},
    sentinel_guard::{SentinelGuard, SentinelReference},
    zmq_bridge::{CommandRequester, IpcEndpoints, SensorSubscriber},
};
use std::time::Duration;
use tracing::{error, info, warn};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    tracing_subscriber::fmt::init();
    info!("SODALS AI Orchestrator başlatılıyor...");

    let endpoints = IpcEndpoints::default();

    // --- 1. ZeroMQ SUB bağlan ---
    let subscriber = SensorSubscriber::new(&endpoints.sensor_pub)
        .expect("ZeroMQ SUB soketi oluşturulamadı");

    // --- 2. ZeroMQ REQ bağlan ---
    let requester = CommandRequester::new(&endpoints.cmd_rep)
        .expect("ZeroMQ REQ soketi oluşturulamadı");

    // --- 3. Sentinel Guard Yükle ---
    // Referans imza: /opt/sodals/qmem/reference.qmem dosyasından okunur.
    let ref_path = std::env::var("SENTINEL_QMEM_PATH")
        .unwrap_or_else(|_| "/opt/sodals/qmem/reference.qmem".to_string());

    let ref_bytes = std::fs::read(&ref_path).expect("Referans .qmem okunamadı");
    if ref_bytes.len() != 544 {
        panic!("Referans .qmem 544 byte olmalı, okunan: {}", ref_bytes.len());
    }

    let reference = unsafe {
        std::ptr::read_unaligned(ref_bytes.as_ptr() as *const sodals_industrial::qmem_format::FrequencySignatureBlock)
    };
    let sentinel = SentinelGuard::new(
        SentinelReference {
            signature: reference,
            threshold_db: -40.0, // dB eşik
        },
        3, // debounce: 3 ardışık anomali
    );

    info!("Sentinel Guard aktif. AI döngüsü başlıyor (50ms tick).");
    transition_state(SystemState::Running);

    let mut tick = tokio::time::interval(Duration::from_millis(50));

    loop {
        tick.tick().await;

        // Sensör verisi al
        let sensor = match subscriber.receive() {
            Ok(Some(s)) => s,
            Ok(None) => continue, // Yeni veri yok
            Err(e) => {
                error!("ZeroMQ alım hatası: {:?}", e);
                continue;
            }
        };

        // --- 4. Qwen 8B Çıktısı (Stub) ---
        // Gerçek entegrasyonda PyO3 ile Python modeli çağrılır.
        // Burada: rastgele logit vektörü (deterministik olmayan kısım izole).
        let logits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        // --- 5. Sentinel Doğrulama ---
        if let Err(e) = sentinel.validate_ai_output(&logits, &sensor) {
            warn!("Sentinel AI çıktısını reddetti: {:?}", e);
            let cmd = FailSafeCommand::from_error(e, sensor.timestamp_ns);
            if let Err(e) = requester.send_command(&cmd) {
                error!("Fail-Safe komutu gönderilemedi: {:?}", e);
            }
            continue;
        }

        // --- 6. PLC'ye Güvenli Komut Gönder (Örnek: motor hızını ayarla) ---
        // Gerçek uygulamada AI çıktısı bir komuta dönüştürülür.
        // Şimdilik: Sadece RUN komutu (0x00)
        let cmd = FailSafeCommand {
            command_code: 0x00,
            param: sensor.motor_rpm as u16,
            timestamp_ns: sensor.timestamp_ns,
            triggered_by: 0,
            _pad: [0; 4],
        };

        match requester.send_command(&cmd) {
            Ok(true) => info!("Komut PLC'ye iletildi: motor_rpm={}", sensor.motor_rpm),
            Ok(false) => warn!("PLC ACK alınamadı"),
            Err(e) => error!("Komut gönderim hatası: {:?}", e),
        }
    }
}
