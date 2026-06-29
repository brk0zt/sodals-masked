//! Fail-Safe (Güvenli Durdurma) Mekanizması
//!
//! Kurallar:
//! 1. Kernel panic YASAK. Panic = üretim bandı duruşu = milyonlarca TL zarar.
//! 2. Her hata `FailSafeCommand` üretir; bu komut PLC'ye "güvenli durum" sinyali olarak yazılır.
//! 3. Bellek sızıntısı olmaması için hata logları sabit boyutlu circular buffer'da tutulur.
//! 4. O(1) karar verme: hata kodu -> eylem mapping'i match ifadesi ile sabit zamanda yapılır.

use crate::{IndustrialError, SensorPayload};
use core::sync::atomic::{AtomicU8, Ordering};

/// Sistemin global güvenli durum makinesi.
/// `AtomicU8` ile lock-free, cache-coherent.
pub static SYSTEM_STATE: AtomicU8 = AtomicU8::new(SystemState::Initializing as u8);

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SystemState {
    Initializing = 0,
    Running = 1,
    Degraded = 2, // OPC UA kopması ama ZeroMQ çalışıyor
    FailSafe = 3, // Tüm aktüatörler STOP
    Panic = 4,    // Bu duruma düşersek donanım watchdog devreye girecek
}

/// PLC'ye yazılacak güvenli durum komutu.
/// `repr(C)` ile OPC UA ve ZeroMQ üzerinden binary taşınabilir.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FailSafeCommand {
    /// Komut kodu: 0=RUN, 1=STOP_MOTOR, 2=ISOLATE_VALVE, 255=EMERGENCY_STOP
    pub command_code: u8,
    /// Parametre (örneğin: stop gecikmesi ms)
    pub param: u16,
    /// Zaman damgası
    pub timestamp_ns: u64,
    /// Hatanın sebebi
    pub triggered_by: u8,
    /// _padding 8-byte hizalama
    pub _pad: [u8; 4],
}

impl FailSafeCommand {
    /// O(1), branch-predictable hata -> eylem dönüşümü.
    pub fn from_error(err: IndustrialError, timestamp_ns: u64) -> Self {
        let (code, triggered): (u8, u8) = match err {
            IndustrialError::OpcUaDisconnect => (1, 10),
            IndustrialError::ZmqBackpressure => (2, 20),
            IndustrialError::SentinelRejected => (255, 30), // EMERGENCY_STOP
            IndustrialError::ClockDrift => (1, 40),
        };

        Self {
            command_code: code,
            param: 0,
            timestamp_ns,
            triggered_by: triggered,
            _pad: [0; 4],
        }
    }

    /// Bellek sızıntısı olmadan serileştirme.
    /// `buf` önceden ayrılmış, sabit 16 byte olmalı.
    pub fn serialize_into(&self, buf: &mut [u8; 16]) {
        buf[0] = self.command_code;
        buf[1..3].copy_from_slice(&self.param.to_le_bytes());
        buf[3..11].copy_from_slice(&self.timestamp_ns.to_le_bytes());
        buf[11] = self.triggered_by;
        buf[12..16].copy_from_slice(&self._pad);
    }
}

/// Global durumu atomik olarak günceller.
/// O(1), lock-free.
pub fn transition_state(new: SystemState) -> SystemState {
    let prev = SYSTEM_STATE.swap(new as u8, Ordering::SeqCst);
    unsafe { core::mem::transmute(prev) }
}

/// Sensör verisi geçersizse (NaN, sapma) Fail-Safe tetiklenir.
/// O(1), deterministik.
pub fn validate_sensor_payload(p: &SensorPayload) -> Result<(), IndustrialError> {
    if p.temperature.is_nan() || p.pressure.is_nan() {
        return Err(IndustrialError::OpcUaDisconnect);
    }
    if p.status_mask == 0xFF {
        return Err(IndustrialError::OpcUaDisconnect);
    }
    Ok(())
}
