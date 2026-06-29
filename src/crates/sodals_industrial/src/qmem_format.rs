//! .qmem Formatı: Sentinel Katmanı Bellek Serileştirmesi
//!
//! `.qmem` dosyaları (qmem_crystals/ altında) Euler açıları, frekans imzaları
//! ve holografik projeksiyon verilerini taşır.
//!
//! Format özellikleri:
//! - Sıfır dinamik allocasyon (deserializasyon stack buffer ile yapılır).
//! - `repr(C)` ile Python (PyO3) ve Rust arasında bellek haritalama uyumu.
//! - Endianness: Little-endian (x86_64 hedef).
//! - Versiyonlama: Header'da 2 byte versiyon alanı.

use core::mem;

/// .qmem dosya header'ı (32 byte).
/// Her .qmem kristali bu header ile başlar.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QmemHeader {
    /// Magic: "QMEM" (4 byte)
    pub magic: [u8; 4],
    /// Versiyon major.minor (2 byte)
    pub version: [u8; 2],
    /// Veri tipi: 0=Euler, 1=Frekans İmzası, 2=Holografik Vektör
    pub dtype: u8,
    /// _padding
    pub _pad0: u8,
    /// Boyut (eleman sayısı), u32
    pub len: u32,
    /// Örneklenme frekansı (Hz), f32
    pub sample_rate_hz: f32,
    /// Zaman damgası (Unix nanosaniye), u64
    pub timestamp_ns: u64,
    /// İlgili sensör etiketi (örn: "TMP0"), 8 byte ASCII
    pub label: [u8; 8],
}

impl QmemHeader {
    pub const MAGIC: [u8; 4] = *b"QMEM";
    pub const HEADER_SIZE: usize = mem::size_of::<Self>();

    /// O(1) validasyon: magic kontrolü.
    pub fn is_valid(&self) -> bool {
        self.magic == Self::MAGIC
    }

    /// Stack buffer'a header yazma. Heap allocasyonu yok.
    pub fn write_to(&self, buf: &mut [u8; 32]) {
        buf[..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.version);
        buf[6] = self.dtype;
        buf[7] = self._pad0;
        buf[8..12].copy_from_slice(&self.len.to_le_bytes());
        buf[12..16].copy_from_slice(&self.sample_rate_hz.to_le_bytes());
        buf[16..24].copy_from_slice(&self.timestamp_ns.to_le_bytes());
        buf[24..32].copy_from_slice(&self.label);
    }
}

/// Frekans imzası bloğu: FFT çıkışından alınan 128-bantlı güç spektrumu.
/// Sabit boyut: 128 x f32 = 512 byte veri + 32 byte header = 544 byte/kristal.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FrequencySignatureBlock {
    pub header: QmemHeader,
    /// 128 bantlı güç spektrumu (dB)
    pub bands: [f32; 128],
}

impl FrequencySignatureBlock {
    pub const BLOCK_SIZE: usize = mem::size_of::<Self>();

    /// Sentinel karar fonksiyonu: spektral enerji belirli bir eşiği aşıyorsa
    /// anomali olarak işaretlenir. O(1) - 128 eleman sabit.
    pub fn is_anomalous(&self, threshold_db: f32) -> bool {
        let mut energy: f32 = 0.0;
        // Deterministik döngü: compiler unroll edebilir.
        for i in 0..128 {
            energy += self.bands[i];
        }
        // Basit ortalama enerji kontrolü
        (energy / 128.0) > threshold_db
    }

    /// Euler açı imzası ile frekans imzasını karşılaştırır.
    /// O(128) = O(1) (sabit boyut).
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let mut dot: f32 = 0.0;
        let mut norm_a: f32 = 0.0;
        let mut norm_b: f32 = 0.0;
        for i in 0..128 {
            dot += self.bands[i] * other.bands[i];
            norm_a += self.bands[i] * self.bands[i];
            norm_b += other.bands[i] * other.bands[i];
        }
        dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-8)
    }
}
