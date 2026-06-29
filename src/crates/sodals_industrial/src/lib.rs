//! SODALS Endüstriyel Zırh ve Haberleşme Köprüsü
//!
//! Karma Kritiklik (Mixed-Criticality) Mimarisi:
//! - RT Çekirdek (Core 0-1): OPC UA polling, sensör okuma
//! - Non-RT Çekirdek (Core 2+): Qwen 8B, Sentinel matematiksel hesaplama
//!
//! Bellek yönetimi: Hot-path'te heap allocasyonu YASAK.
//! Tüm sensör verileri önceden ayrılmış (pre-allocated) ring buffer'larda taşınır.

// Dışarıya açılan modüller
pub mod opcua_rt_client;
pub mod zmq_bridge;
pub mod sentinel_guard;
pub mod fail_safe;
pub mod qmem_format;
pub mod introspection;

// RT Ring Buffer ve Sabit Boyutlu Veri Yapıları
use core::fmt;

/// PLC'den okunan ham sensör verisi.
/// Bellek düzeni sabit, `repr(C)` ile FFI-güvenli.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SensorPayload {
    /// Unix timestamp nanosaniye (u64 -> 8 byte)
    pub timestamp_ns: u64,
    /// Sıcaklık [°C], f32
    pub temperature: f32,
    /// Basınç [bar], f32
    pub pressure: f32,
    /// Motor hızı [RPM], f32
    pub motor_rpm: f32,
    /// Durum bitmask (0: OK, 1: Uyarı, 2: Kritik)
    pub status_mask: u8,
    /// Hata kodu (0: Yok)
    pub fault_code: u8,
    /// _padding: 16-byte hizalama için (SIMD/IPC uyumluluğu)
    pub _pad: [u8; 2],
}

impl SensorPayload {
    /// SABIT_BOYUT: ZeroMQ mesaj boyutunu deterministik yapar.
    /// O(1) - heap allocasyonu yok.
    pub const SERIALIZED_SIZE: usize = core::mem::size_of::<Self>();

    /// Tüm alanları "güvenli durum" (fail-safe) değerlerine çeker.
    /// O(1), branch-predictable.
    pub fn to_failsafe() -> Self {
        Self {
            timestamp_ns: 0,
            temperature: f32::NAN,
            pressure: f32::NAN,
            motor_rpm: 0.0,
            status_mask: 0xFF, // Bilinmeyen durum
            fault_code: 0xFF,  // Bilinmeyen hata
            _pad: [0; 2],
        }
    }
}

/// Sıfır-tahsis, lock-free SPSC ring buffer.
/// `capacity` 2'nin kuvveti olmalı (mask ile modulo yapmak için).
/// O(1) write/read; cache-line ayrımı için `cache_padding` kullanılır.
pub struct RtRingBuffer<T, const N: usize> {
    // Bu yapı hot-path'te kullanılacaksa `crossbeam` crate'i yerine
    // elle yazılmış versiyon tercih edilebilir. Şimdilik sabit boyutlu
    // bir vec-backed yapı kullanıyoruz (init-time tek alloc).
    buffer: [T; N],
    write_idx: core::sync::atomic::AtomicUsize,
    read_idx: core::sync::atomic::AtomicUsize,
}

impl<T: Copy + Default, const N: usize> RtRingBuffer<T, N> {
    pub fn new() -> Self {
        assert!(N.is_power_of_two(), "Kapasite 2'nin kuvveti olmalı");
        Self {
            buffer: [T::default(); N],
            write_idx: core::sync::atomic::AtomicUsize::new(0),
            read_idx: core::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// O(1), lock-free, wait-free (SPSC için).
    pub fn push(&self, item: T) -> bool {
        let w = self.write_idx.load(core::sync::atomic::Ordering::Relaxed);
        let r = self.read_idx.load(core::sync::atomic::Ordering::Acquire);
        if w.wrapping_sub(r) >= N {
            return false; // Tam dolu, overwrite yok (deterministik)
        }
        let idx = w & (N - 1);
        unsafe {
            let ptr = self.buffer.as_ptr() as *mut T;
            core::ptr::write(ptr.add(idx), item);
        }
        self.write_idx
            .store(w.wrapping_add(1), core::sync::atomic::Ordering::Release);
        true
    }

    /// O(1), lock-free.
    pub fn pop(&self) -> Option<T> {
        let r = self.read_idx.load(core::sync::atomic::Ordering::Relaxed);
        let w = self.write_idx.load(core::sync::atomic::Ordering::Acquire);
        if w == r {
            return None;
        }
        let idx = r & (N - 1);
        let item = unsafe {
            let ptr = self.buffer.as_ptr() as *mut T;
            core::ptr::read(ptr.add(idx))
        };
        self.read_idx
            .store(r.wrapping_add(1), core::sync::atomic::Ordering::Release);
        Some(item)
    }
}

/// PLC bağlantı kopması, ZeroMQ taşması, Sentinel reddi gibi
/// ölümcül hataları taşıyan birim hata tipi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndustrialError {
    OpcUaDisconnect,
    ZmqBackpressure,
    SentinelRejected,
    ClockDrift, // RT döngü 10ms'yi aştı
}

impl fmt::Display for IndustrialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndustrialError::OpcUaDisconnect => write!(f, "OPC UA bağlantı kopması"),
            IndustrialError::ZmqBackpressure => write!(f, "ZeroMQ arabelleği taşması"),
            IndustrialError::SentinelRejected => write!(f, "Sentinel güvenlik reddi"),
            IndustrialError::ClockDrift => write!(f, "RT saat kayması > 10ms"),
        }
    }
}

// Re-exports
pub use fail_safe::{FailSafeCommand, SystemState};
